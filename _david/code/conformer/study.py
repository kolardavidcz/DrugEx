"""MMFF94/TanimotoCombo study using the training candidate generator.

Run from the repository root with ``python -m _david.code.conformer.study``.
Reference counts select energy-ordered basin representatives; 0 means all.
Candidate minimization is a paired diagnostic, not a change to training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from drugex.training.scorers.rocs_rdkit import _score_single_reference
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from _david.code.chemistry import check_identity, minimize_mmff94
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT
from _david.code.environment import PrepareMolecule, create_rocs_scorer
from _david.code.visualization import render_conformer_study


def read_ensemble(path: Path, expected: Chem.Mol) -> list[Chem.Mol]:
    records = list(Chem.SDMolSupplier(str(path), removeHs=False))
    if not records or any(mol is None for mol in records):
        raise ValueError(f"Empty or invalid SDF: {path}")
    for mol in records:
        check_identity(mol, expected)
    return records


def overlay_matrix(candidates, references):
    """Reuse DrugEx's actual alignment; retain pair scores and measured latency."""
    scores = np.zeros((len(candidates), len(references)))
    seconds = np.zeros_like(scores)
    for i, candidate in enumerate(candidates):
        for j, reference in enumerate(references):
            start = time.perf_counter()
            scores[i, j] = _score_single_reference(
                Chem.Mol(candidate), Chem.Mol(reference), "TanimotoCombo", True
            )
            seconds[i, j] = time.perf_counter() - start
    return scores, seconds


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = vars(args) | {"output": str(output)}
    source_paths = [
        "_david/code/chemistry.py",
        "_david/code/conformer/study.py",
        "_david/code/environment.py",
        "_david/code/reference.py",
        "_david/code/config.py",
        "_david/code/visualization.py",
        "_david/code/conformer/visualization.py",
        "drugex/training/scorers/conformer_generators.py",
        "drugex/training/scorers/rocs_rdkit.py",
    ]
    manifest = {
        "state": "running",
        "settings": settings,
        "host": socket.gethostname(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], text=True),
        "code_root": str(PROJECT_ROOT),
        "rdkit": rdBase.rdkitVersion,
        "command": [sys.executable, "-m", "_david.code.conformer.study", *sys.argv[1:]],
        "python": platform.python_version(),
        "cpu_threads": 1,
        "dimorphite_available": importlib.util.find_spec("dimorphite_dl") is not None,
        "candidate_protonate_requested": True,
        "candidate_tautomer_canonicalization": True,
        "embedding_timeout": 0,
        "raw_score": "shape Tanimoto + color Tanimoto [0,2]",
        "normalized_score": "raw / 2 [0,1]",
        "force_field": "MMFF94",
        "max_iterations": 1000,
        "reference_rmsd_cutoff": 1.25,
        "energy_window_kcal": 8.0,
        "candidate_pruning_rmsd": 0.5,
        "source_sha256": {
            p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
            for p in source_paths
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    expected = Chem.MolFromSmiles(PrepareMolecule().standardize(args.smiles))
    if expected is None:
        raise ValueError("Invalid reference")
    # A reference with unresolved stereo needs a selected ensemble policy; do not
    # silently turn its first random embedding into a specified reference.
    if any(
        info.specified == Chem.StereoSpecified.Unspecified
        for info in Chem.FindPotentialStereo(expected)
    ):
        raise ValueError(
            "Specify reference stereochemistry before a self-recovery study"
        )
    candidate_smiles = args.candidate_smiles or args.smiles
    expected_candidate = Chem.MolFromSmiles(
        PrepareMolecule().standardize(candidate_smiles)
    )
    if expected_candidate is None:
        raise ValueError("Invalid candidate")

    preparer = PrepareMolecule(
        num_conformers=args.reference_attempts,
        max_reference_conformers=args.reference_attempts,
        random_seed=args.reference_seed,
        num_threads=1,
    )
    start = time.perf_counter()
    path = preparer.generate_reference_conformers(expected, output / "reference.sdf")
    reference_seconds = time.perf_counter() - start
    references = read_ensemble(path, expected)
    if any(m.GetProp("ForceField") != "MMFF94" for m in references):
        raise ValueError(
            "MMFF94 unavailable: refusing to mix a UFF fallback into this condition"
        )
    controls = []
    for index, reference in enumerate(references):
        score, _ = overlay_matrix([reference], [reference])
        if not np.isclose(score[0, 0], 2.0, atol=1e-5, rtol=0):
            raise ValueError(f"Exact-geometry self-control failed: {score[0, 0]}")
        controls.append(
            {
                "control": "exact_geometry",
                "reference": index,
                "raw_combo": score[0, 0],
                "identity": check_identity(reference, expected),
            }
        )
    centers = Chem.FindMolChiralCenters(expected, includeUnassigned=False)
    if centers:
        wrong = Chem.Mol(expected)
        wrong.GetAtomWithIdx(centers[0][0]).InvertChirality()
        wrong = Chem.AddHs(wrong)
        if AllChem.EmbedMolecule(wrong, randomSeed=args.reference_seed) != 0:
            raise ValueError("Cannot embed wrong-stereo identity control")
        try:
            check_identity(wrong, expected)
        except ValueError:
            wrong_score, _ = overlay_matrix([wrong], references[:1])
            controls.append(
                {
                    "control": "wrong_stereo_rejected",
                    "reference": 0,
                    "raw_combo": wrong_score[0, 0],
                    "identity": Chem.MolToSmiles(Chem.RemoveHs(wrong)),
                }
            )
        else:
            raise ValueError("Wrong-stereo identity control was incorrectly accepted")
    pd.DataFrame(controls).to_csv(output / "controls.csv", index=False)
    manifest.update(
        identity=Chem.MolToSmiles(expected),
        reference_seconds=reference_seconds,
        reference_embedded=references[0].GetIntProp("EmbeddedConformers"),
        reference_converged=references[0].GetIntProp("ConvergedConformers"),
        reference_within_energy_window=references[0].GetIntProp(
            "EnergyWindowConformers"
        ),
        reference_clusters=references[0].GetIntProp("ReferenceClusters"),
        reference_retained=len(references),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    workloads = list(itertools.product(args.isomers, args.conformers, args.seeds))
    random.Random(42).shuffle(workloads)
    with Chem.SDWriter(str(output / "candidates.sdf")) as writer:
        for isomer_cap, conformer_budget, seed in workloads:
            scorer = create_rocs_scorer(
                references,
                max_conformers=conformer_budget,
                max_isomers=isomer_cap,
                timeout=0,
                conformer_threads=1,
                n_jobs=1,
                random_seed=seed,
                minimize_candidates=False,
            )
            start = time.perf_counter()
            with tempfile.TemporaryDirectory() as directory:
                path = scorer.conformer_generator.genConformers(
                    [candidate_smiles], directory
                )
                candidates = read_ensemble(Path(path), expected_candidate)
            generation_seconds = time.perf_counter() - start
            identities = {check_identity(m, expected_candidate) for m in candidates}
            start = time.perf_counter()
            minimized, statuses = minimize_mmff94(candidates)
            minimization_seconds = time.perf_counter() - start
            for mode, pool in (
                ("training_ETKDG", candidates),
                ("MMFF94_minimized", minimized),
            ):
                for mol in pool:
                    mol.SetProp("Mode", mode)
                    mol.SetIntProp("EmbeddingSeed", seed)
                    mol.SetIntProp("ConformerBudget", conformer_budget)
                    mol.SetIntProp("IsomerCap", isomer_cap)
                    writer.write(mol)
                scores, seconds = overlay_matrix(pool, references)
                for requested_references in args.references:
                    available = (
                        requested_references == 0
                        or requested_references <= len(references)
                    )
                    count = min(
                        requested_references or len(references), len(references)
                    )
                    subset = scores[:, :count]
                    raw = float(subset.max()) if subset.size and available else np.nan
                    winner = (
                        check_identity(
                            pool[np.unravel_index(subset.argmax(), subset.shape)[0]],
                            expected_candidate,
                        )
                        if subset.size and available
                        else ""
                    )
                    overlay_seconds = float(seconds[:, :count].sum())
                    extra = minimization_seconds if mode == "MMFF94_minimized" else 0.0
                    rows.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "requested_conformers": conformer_budget,
                            "isomer_cap": isomer_cap,
                            "actual_isomers": len(identities),
                            "embedded_candidates": len(candidates),
                            "scored_candidates": len(pool),
                            "minimization_failures": sum(
                                status != 0 for status in statuses
                            ),
                            "requested_references": requested_references,
                            "actual_references": count if available else 0,
                            "reference_pool_size": len(references),
                            "reference_budget_available": available,
                            "raw_combo": raw,
                            "normalized_combo": raw / 2,
                            "generation_seconds": generation_seconds,
                            "minimization_seconds": extra,
                            "overlay_seconds": overlay_seconds if available else np.nan,
                            "total_seconds": generation_seconds
                            + extra
                            + overlay_seconds
                            if available
                            else np.nan,
                            "winning_isomer": winner,
                        }
                    )
            pd.DataFrame(rows).to_csv(output / "data.csv", index=False)
            print(
                f"C={conformer_budget}, I={isomer_cap}, seed={seed}: {len(candidates)} embedded; "
                f"{len(minimized)} converged",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    render_conformer_study(frame, output / "summary.png")
    manifest.update(
        state="complete",
        measurement_rows=len(rows),
        all_candidate_minimizations_converged=bool(
            (frame.minimization_failures == 0).all()
        ),
    )
    for p, digest in manifest["source_sha256"].items():
        if hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during measurement: {p}")
    manifest["sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file() and p != manifest_path
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles", default=DEFAULT_MOLECULE)
    parser.add_argument(
        "--candidate-smiles",
        help="Optional candidate input; defaults to the reference input",
    )
    parser.add_argument(
        "--conformers", type=int, nargs="+", default=[1, 5, 10, 15, 20, 30, 100, 200]
    )
    parser.add_argument("--isomers", type=int, nargs="+", default=[4])
    parser.add_argument("--references", type=int, nargs="+", default=[1, 2, 3, 0])
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0xC0FFEE, 0xC0FFEF, 0xC0FFF0]
    )
    parser.add_argument("--reference-attempts", type=int, default=1000)
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render_conformer_study(
            pd.read_csv(args.output / "data.csv"), args.output / "summary.png"
        )
    else:
        if any(c < 1 or c > 200 for c in args.conformers):
            parser.error(
                "The current core generator supports 1–200 conformer attempts per isomer"
            )
        if (
            min(args.isomers) < 1
            or min(args.references) < 0
            or args.reference_attempts < 1
        ):
            parser.error("Budgets must be positive; reference cap 0 means all")
        run(args)


if __name__ == "__main__":
    main()
