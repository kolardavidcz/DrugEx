"""Energy-filtered reference banks with score-blind shape/color diversity ordering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

from rdkit import Chem

from _david.code.chemistry import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_REFERENCE_SEED,
    Optimization,
    enumerate_stereoisomers,
    optimize_conformers,
    prepare_candidate_molecule,
)
from _david.code.conformer.utils import save_sdf
from _david.code.reference import (
    PrepareMolecule,
    balanced_reference_order,
    diverse_order,
)
from _david.code.rocs import NAMED_OPTIMIZATION_METHODS, has_color

METHODS: dict[str, Optimization] = dict(NAMED_OPTIMIZATION_METHODS)
LABELS: dict[str, str] = {
    "etkdg": "ETKDG only",
    "uff": "UFF",
    "mmff_vacuum": "MMFF94s ε=1",
    "mmff_high_dielectric": "MMFF94s ε=78.4",
}

REFERENCE_SEED: int = DEFAULT_REFERENCE_SEED
CANDIDATE_SEED: int = DEFAULT_CANDIDATE_SEED
STEREO_SEED: int = 42
REFERENCE_ISOMER_CAP: int = 8
CANDIDATE_ISOMER_CAP: int = 4
REFERENCE_ATTEMPTS: int = 50
ENERGY_WINDOW_KCAL: float = 8.0

save_records = save_sdf


def prepare_banks(
    smiles: str,
    output: Path,
    methods: Sequence[str] = tuple(METHODS),
) -> tuple[dict[str, list[Chem.Mol]], dict[str, Any]]:
    """Share R50/I8 disjoint-stream embeddings; optimize and diversity-order per method."""
    started = time.perf_counter()
    pools: dict[str, list[Chem.Mol]] = {method: [] for method in methods}
    info: dict[str, dict[str, Any]] = {
        method: dict(optimization_statuses=Counter(), errors=[], energy_survivors=0)
        for method in methods
    }
    try:
        prepared = prepare_candidate_molecule(smiles, protonate=False)
    except (ValueError, RuntimeError) as exc:
        return pools, dict(
            preparation_error=str(exc),
            reference_seconds=time.perf_counter() - started,
            color_excluded=False,
            methods=info,
            enumerated_isomers=0,
        )
    isomers = list(enumerate_stereoisomers(prepared, limit=8, seed=42))
    color_excluded = False
    for iso_index, iso in enumerate(isomers):
        preparer = PrepareMolecule(
            num_conformers=50,
            max_reference_conformers=50,
            random_seed=REFERENCE_SEED,
            num_threads=1,
            rmsd_cutoff=None,
            embedding_rmsd=None,
            embedding_timeout_seconds=60,
            optimization=Optimization("none"),
        )
        try:
            path = preparer.generate_reference_conformers(
                iso, output / f"embedded_stereo_{iso_index}.sdf"
            )
            raw = list(Chem.SDMolSupplier(str(path), removeHs=False))
            if not raw or any(mol is None for mol in raw):
                raise ValueError("Invalid embedded reference records")
            color_excluded |= not has_color(raw[0])
        except (ValueError, RuntimeError) as exc:
            for method in methods:
                info[method]["errors"].append(f"stereo {iso_index}: {exc}")
            continue
        for method in methods:
            if method == "etkdg":
                survivors = [Chem.Mol(mol) for mol in raw]
            else:
                converged, statuses = optimize_conformers(raw, METHODS[method])
                info[method]["optimization_statuses"].update(map(str, statuses))
                lowest = min(
                    (mol.GetDoubleProp("Energy_kcal_mol") for mol in converged),
                    default=0.0,
                )
                survivors = [
                    mol
                    for mol in converged
                    if mol.GetDoubleProp("Energy_kcal_mol") - lowest <= 8.0
                ]
                survivors.sort(key=lambda mol: mol.GetDoubleProp("Energy_kcal_mol"))
            for mol in survivors:
                mol.SetProp(
                    "EnumeratedIsomerSMILES", iso.GetProp("EnumeratedIsomerSMILES")
                )
            pools[method].extend(survivors)
    for method, records in pools.items():
        tick = time.perf_counter()
        order = (
            diverse_order(records) if not color_excluded else list(range(len(records)))
        )
        pools[method] = [records[i] for i in order]
        info[method].update(
            energy_survivors=len(records),
            order=order,
            selection_seconds=time.perf_counter() - tick,
            energy_window_applied=method != "etkdg",
        )
        info[method]["optimization_statuses"] = dict(
            info[method]["optimization_statuses"]
        )
        save_records(output / f"reference_{method}.sdf", pools[method])
    return pools, dict(
        reference_seconds=time.perf_counter() - started,
        enumerated_isomers=len(isomers),
        color_excluded=color_excluded,
        preparation_error="",
        methods=info,
    )


__all__ = [
    "CANDIDATE_ISOMER_CAP",
    "CANDIDATE_SEED",
    "ENERGY_WINDOW_KCAL",
    "LABELS",
    "METHODS",
    "REFERENCE_ATTEMPTS",
    "REFERENCE_ISOMER_CAP",
    "REFERENCE_SEED",
    "STEREO_SEED",
    "balanced_reference_order",
    "diverse_order",
    "prepare_banks",
    "save_records",
]
