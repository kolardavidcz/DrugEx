"""Inspectable sampling budgets and symmetry-aware reference deduplication.

The AQME-style recipe is a baseline heuristic, not a proven DrugEx optimum.
Counts are attempts per stereoisomer; they are not retained ensemble sizes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolAlign
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

from _david.code.chemistry import enumerate_stereoisomers


@dataclass(frozen=True)
class SamplingPlan:
    """Planned conformer generation budget and stereoisomer coverage for one molecule."""

    rotatable_bonds: int
    oh_nh_count: int
    saturated_rings: int
    isomer_upper_bound: int
    reference_isomers: tuple[str, ...]
    conformers_per_isomer: int
    isomer_limit: int

    @property
    def category(self) -> Literal["1", "2", "3-4", "5-8", "9-25"]:
        n = len(self.reference_isomers)
        if n == 1:
            return "1"
        if n == 2:
            return "2"
        if n <= 4:
            return "3-4"
        if n <= 8:
            return "5-8"
        return "9-25"

    @property
    def truncated(self) -> bool:
        return self.isomer_upper_bound > self.isomer_limit

    @property
    def total_conformer_attempts(self) -> int:
        """Requested initial work, not retained reference structures or stereo coverage."""
        return len(self.reference_isomers) * self.conformers_per_isomer

    @property
    def conformer_budget_capped(self) -> bool:
        baseline = (
            30 * (self.rotatable_bonds + self.oh_nh_count + self.saturated_rings) or 10
        )
        return self.conformers_per_isomer < baseline

    def attempts(self, factor: float = 1.0) -> int:
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Sampling factor must be finite and positive")
        return max(1, math.ceil(self.conformers_per_isomer * factor))


def plan_sampling(
    mol: Chem.Mol,
    *,
    isomer_limit: int = 25,
    enumeration_seed: int = 42,
    conformer_limit: int = 250,
) -> SamplingPlan:
    """Compute an AQME mid-style organic-molecule sampling budget with bounded stereo enumeration."""
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("A nonempty prepared molecule is required")
    if not 1 <= isomer_limit <= 25:
        raise ValueError("The selected study supports an isomer ceiling of 1–25")
    if not isinstance(conformer_limit, int) or not 1 <= conformer_limit <= 1000:
        raise ValueError("Conformer ceiling must be an integer from 1 to 1000")
    rotors = int(Lipinski.NumRotatableBonds(mol))
    donors = int(Lipinski.NHOHCount(mol))
    rings = int(Lipinski.NumSaturatedRings(mol))
    complexity = 3 * (rotors + donors + rings)
    attempts = min(conformer_limit, complexity * 10 if complexity else 10)
    isomers = tuple(
        Chem.MolToSmiles(iso, isomericSmiles=True)
        for iso in enumerate_stereoisomers(
            mol, limit=isomer_limit, seed=enumeration_seed
        )
    )
    if not isomers:
        raise ValueError("Stereo enumeration produced no identities")
    return SamplingPlan(
        rotors,
        donors,
        rings,
        int(GetStereoisomerCount(mol)),
        isomers,
        attempts,
        isomer_limit,
    )


def deduplicate_reference(
    records: Sequence[Chem.Mol],
    *,
    threshold: float = 1.25,
) -> list[Chem.Mol]:
    """Keep the given priority order, rejecting RMSD-near-duplicates of one isomer.

    Operates on heavy-atom copies so ``GetBestRMS`` never mutates input coordinates.
    Distinct enumerated stereo identities are never clustered together.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("RMSD threshold must be finite and positive")
    kept: list[Chem.Mol] = []
    geometries: list[Chem.Mol] = []
    identity: str | None = None
    for record in records:
        heavy = Chem.RemoveHs(Chem.Mol(record))
        key = (
            record.GetProp("EnumeratedIsomerSMILES")
            if record.HasProp("EnumeratedIsomerSMILES")
            else Chem.MolToSmiles(heavy, isomericSmiles=True)
        )
        if identity is not None and key != identity:
            raise ValueError("Deduplicate each stereoisomer separately")
        identity = key
        if all(
            rdMolAlign.GetBestRMS(Chem.Mol(heavy), Chem.Mol(other)) >= threshold
            for other in geometries
        ):
            kept.append(record)
            geometries.append(heavy)
    return kept
