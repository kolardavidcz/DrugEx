"""Inspectable sampling budgets and symmetry-aware reference deduplication.

The AQME-style recipe is a baseline heuristic, not a proven DrugEx optimum.
Counts are attempts per stereoisomer; they are not retained ensemble sizes.
"""

from dataclasses import dataclass
import math

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolAlign
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

from _david.code.chemistry import enumerate_stereoisomers


@dataclass(frozen=True)
class SamplingPlan:
    rotatable_bonds: int
    oh_nh_count: int
    saturated_rings: int
    isomer_upper_bound: int
    reference_isomers: tuple[str, ...]
    conformers_per_isomer: int
    isomer_limit: int

    @property
    def category(self):
        n = len(self.reference_isomers)
        return (
            "1"
            if n == 1
            else "2"
            if n == 2
            else "3-4"
            if n <= 4
            else "5-8"
            if n <= 8
            else "9-25"
        )

    @property
    def truncated(self):
        return self.isomer_upper_bound > self.isomer_limit

    @property
    def total_conformer_attempts(self):
        """Requested initial work, not retained reference structures or stereo coverage."""
        return len(self.reference_isomers) * self.conformers_per_isomer

    @property
    def conformer_budget_capped(self):
        baseline = (
            30 * (self.rotatable_bonds + self.oh_nh_count + self.saturated_rings) or 10
        )
        return self.conformers_per_isomer < baseline

    def attempts(self, factor=1.0):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Sampling factor must be finite and positive")
        return max(1, math.ceil(self.conformers_per_isomer * factor))


def plan_sampling(mol, *, isomer_limit=25, enumeration_seed=42, conformer_limit=250):
    """AQME mid-style organic-molecule recipe plus bounded RDKit stereo enumeration.

    AQME's published code counts 3*(rotors + OH/NH + saturated rings), then
    multiplies by 10 and caps at 250; featureless molecules receive 10 attempts.
    We preserve assigned stereochemistry and enumerate only unresolved stereo.
    The 25-isomer ceiling is the user's compute limit, not a literature optimum.
    Call after the same tautomer/protonation preparation as candidate generation.
    """
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


def deduplicate_reference(records, *, threshold=1.25):
    """Keep the given priority order, rejecting RMSD-near-duplicates of one isomer.

    The existing reference preparer supplies energy-ordered Butina representatives.
    Its lowest-energy representatives need not be mutually separated by the
    clustering threshold. This final greedy pass enforces that spacing directly.
    Alignment works on heavy-atom copies because GetBestRMS moves the probe.
    Distinct enumerated stereo identities are never clustered together.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("RMSD threshold must be finite and positive")
    kept, geometries = [], []
    identity = None
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
