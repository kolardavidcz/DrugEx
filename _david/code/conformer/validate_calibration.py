"""Verify the saved calibration grid and replay representative public scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.conformer.calibrate import KEYS, save_sdf
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.environment import create_rocs_scorer
from _david.code.reference import select_conformers


def validate(output: Path, replays: int = 6):
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["state"] != "complete":
        raise ValueError("Measurements must finish before validation")
    frame = pd.read_csv(output / "data.csv.gz")
    if len(frame) != manifest["expected_rows"] or frame.duplicated(KEYS).any():
        raise ValueError("Incomplete or duplicated measurement grid")
    if set(frame.groupby("molecule_id").size()) != {36}:
        raise ValueError("Missing per-input comparison cells")
    if not np.allclose(
        frame.score, (frame.shape_tanimoto + frame.color_tanimoto) / 2, equal_nan=True
    ):
        raise ValueError("Component reduction mismatch")
    if not frame.loc[~frame.color_excluded, "score"].between(0, 1.00001).all():
        raise ValueError("Invalid score or silently missing observation")
    panel = pd.read_csv(output / "panel.csv")
    smiles_by_id = panel.set_index("molecule_id").smiles.to_dict()
    for case in panel.to_dict("records"):
        directory = output / "checkpoints" / str(case["molecule_id"])
        detail = json.loads((directory / "complete.json").read_text())
        for filename, expected in detail["files_sha256"].items():
            if digest(directory / filename) != expected:
                raise ValueError(f"Changed checkpoint: {directory / filename}")
    replay_results = []
    valid = frame[
        ~frame.color_excluded & ~frame.reference_failure & ~frame.candidate_failure
    ]
    # Include different pruning levels, both reference budgets and difficult cases.
    ordered = valid.sort_values("score")
    for index in np.linspace(
        0, len(ordered) - 1, min(replays, len(ordered)), dtype=int
    ):
        row = ordered.iloc[index]
        directory = output / "checkpoints" / str(int(row.molecule_id))
        candidates = list(
            Chem.SDMolSupplier(
                str(directory / f"candidate_{int(row.candidate_attempts)}_pool.sdf"),
                removeHs=False,
            )
        )
        references = list(
            Chem.SDMolSupplier(
                str(directory / f"reference_{int(row.reference_attempts)}_pool.sdf"),
                removeHs=False,
            )
        )
        ci = select_conformers(candidates, float(row.candidate_rmsd) or None)
        ri = select_conformers(references, float(row.reference_rmsd))
        if len(ci) != row.actual_candidates or len(ri) != row.actual_references:
            raise ValueError("Saved counts differ from replay pruning")
        with tempfile.TemporaryDirectory() as temporary:
            candidate_path = Path(temporary) / "candidates.sdf"
            reference_path = Path(temporary) / "references.sdf"
            save_sdf(candidate_path, [candidates[i] for i in ci])
            save_sdf(reference_path, [references[i] for i in ri])
            scorer = create_rocs_scorer(reference_path, conformer_threads=1, n_jobs=1)
            scorer.conformer_generator.genConformers = (
                lambda smiles, out, saved=candidate_path: str(saved)
            )
            started = time.perf_counter()
            actual = (
                float(scorer.getScores([smiles_by_id[int(row.molecule_id)]])[0, 0]) / 2
            )
            seconds = time.perf_counter() - started
        difference = abs(actual - row.score)
        if difference > 1e-8:
            raise ValueError(f"Public scorer replay differs by {difference}")
        replay_results.append(
            dict(
                molecule_id=int(row.molecule_id),
                reference_attempts=int(row.reference_attempts),
                candidate_attempts=int(row.candidate_attempts),
                reference_rmsd=float(row.reference_rmsd),
                candidate_rmsd=float(row.candidate_rmsd),
                expected=float(row.score),
                actual=actual,
                difference=difference,
                scorer_only_seconds=seconds,
            )
        )
    manifest["validation"] = dict(
        rows=len(frame),
        unique_complete_grids=True,
        component_reduction=True,
        checkpoint_hashes=True,
        public_scorer_replays=replay_results,
        validation_script_sha256=digest(__file__),
    )
    atomic_json(path, manifest)
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=6)
    args = parser.parse_args()
    validate(args.output, args.replays)
