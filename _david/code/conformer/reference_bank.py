"""Energy-filtered reference banks with score-blind shape/colour ordering.

No evaluation candidates enter this module. The first pose for each stereo
identity is its lowest-energy survivor. Further poses maximize dissimilarity
to that identity's selected poses, using the native normalized ROCS score.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from rdkit import Chem

from _david.code.chemistry import (
    Optimization,
    enumerate_stereoisomers,
    optimize_conformers,
    prepare_candidate_molecule,
)
from _david.code.reference import PrepareMolecule
from _david.code.rocs import has_color, score_pair

METHODS = {
    "etkdg": Optimization("none"),
    "uff": Optimization("UFF"),
    "mmff_vacuum": Optimization("MMFF94s", 1.0),
    "mmff_high_dielectric": Optimization("MMFF94s", 78.4),
}
LABELS = {
    "etkdg": "ETKDG only",
    "uff": "UFF",
    "mmff_vacuum": "MMFF94s ε=1",
    "mmff_high_dielectric": "MMFF94s ε=78.4",
}


def save_records(path: Path, records: list) -> None:
    with Chem.SDWriter(str(path)) as writer:
        for mol in records:
            writer.write(mol)


def diverse_order(records: list, prefix: int = 12, similarity=score_pair) -> list[int]:
    """Nested caps, stereo round-robin, energy seeds and farthest-first diversity.

    Symmetrize the two directional alignments by their maximum. This is an
    ordering heuristic, not a metric or a physical conformer population.
    Only the first `prefix` poses need diversity ordering; `all` retains every
    survivor, so sorting its remaining tail would waste preparation time.
    """
    groups = {}
    for index, mol in enumerate(records):
        identity = mol.GetProp("EnumeratedIsomerSMILES")
        groups.setdefault(identity, []).append(index)

    def energy(index):
        mol = records[index]
        return (
            mol.GetDoubleProp("Energy_kcal_mol")
            if mol.HasProp("Energy_kcal_mol")
            else 0.0
        )

    for group in groups.values():
        group.sort(key=lambda i: (energy(i), i))
    selected, nearest = [], {}
    while len(selected) < min(prefix, len(records)):
        for group in groups.values():
            remaining = [i for i in group if i not in selected]
            if not remaining:
                continue
            prior = [i for i in group if i in selected]
            if not prior:
                chosen = remaining[0]
            else:
                chosen = min(remaining, key=lambda i: (nearest[i], energy(i), i))
            selected.append(chosen)
            if len(selected) == min(prefix, len(records)):
                break
            for index in remaining:
                if index != chosen:
                    overlap = max(
                        similarity(records[index], records[chosen])[0],
                        similarity(records[chosen], records[index])[0],
                    )
                    nearest[index] = max(nearest.get(index, 0.0), overlap)
    # Continue in stereo round-robin order, retaining every remaining identity.
    tails = [[i for i in group if i not in selected] for group in groups.values()]
    while any(tails):
        for tail in tails:
            if tail:
                selected.append(tail.pop(0))
    return selected


def prepare_banks(smiles: str, output: Path, methods=tuple(METHODS)):
    """Share R50/I8 seed43 embeddings; optimize/filter independently per method."""
    started = time.perf_counter()
    pools = {method: [] for method in methods}
    info = {
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
            random_seed=43,
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
