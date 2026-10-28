"""Verify the saved calibration grid and replay representative public scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.chemistry import check_identity

from _david.code.conformer.calibrate import KEYS, PROFILE, save_sdf
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.environment import create_rocs_scorer
from _david.code.reference import select_conformers


def timing_replay(job):
    """Actually regenerate/minimize/prune/score one input through the RL factory."""
    output, row, smiles = job
    directory = Path(output) / "checkpoints" / str(int(row["molecule_id"]))
    refs = list(
        Chem.SDMolSupplier(str(directory / "reference_50_pool.sdf"), removeHs=False)
    )
    ri = select_conformers(refs, 0.5)
    scorer = create_rocs_scorer(
        [refs[i] for i in ri],
        max_conformers=20,
        max_isomers=4,
        random_seed=42,
        minimize_candidates=True,
        optimization=PROFILE,
        conformer_threads=1,
        timeout=0,
        n_jobs=1,
    )
    if row["candidate_rmsd"]:
        generate = scorer.conformer_generator.genConformers

        def generate_and_prune(smiles_list, out):
            path = generate(smiles_list, out)
            records = list(Chem.SDMolSupplier(path, removeHs=False))
            kept = select_conformers(records, row["candidate_rmsd"])
            save_sdf(path, [records[i] for i in kept])
            return path

        scorer.conformer_generator.genConformers = generate_and_prune
    started = time.perf_counter()
    actual = float(scorer.getScores([smiles])[0, 0]) / 2
    seconds = time.perf_counter() - started
    if abs(actual - row["score"]) > 1e-8:
        raise ValueError(f"End-to-end replay changed score: {row['score']} -> {actual}")
    return dict(
        molecule_id=int(row["molecule_id"]),
        candidate_rmsd=row["candidate_rmsd"],
        measured_seconds=seconds,
        component_estimate=row["seconds"],
        score=actual,
        expected=row["score"],
    )


def validate_end_to_end(output):
    panel = pd.read_csv(output / "panel.csv").head(16)
    frame = pd.read_csv(output / "data.csv.gz")
    selected = frame[
        frame.molecule_id.isin(panel.molecule_id)
        & frame.reference_attempts.eq(50)
        & frame.reference_rmsd.eq(0.5)
        & frame.candidate_attempts.eq(20)
        & frame.candidate_rmsd.isin([0, 0.5])
        & ~frame.color_excluded
        & ~frame.reference_failure
        & ~frame.candidate_failure
    ]
    smiles = panel.set_index("molecule_id").smiles.to_dict()
    jobs = [
        (str(output), row, smiles[row["molecule_id"]])
        for row in selected.to_dict("records")
    ]
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(timing_replay, jobs))
    pd.DataFrame(results).to_csv(output / "end_to_end.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["end_to_end_replay"] = dict(
        rows=len(results),
        selection="First 16 frozen panel IDs, working R50/C20/0.5-A reference preset; candidate post-pruning off vs 0.5 A. Unavailable inputs reported, never substituted.",
        attempted_inputs=len(panel),
        replayed_inputs=selected.molecule_id.nunique(),
        score_tolerance=1e-8,
        script_sha256=digest(__file__),
        scope="Whole getScores wall time, including fresh generation/minimization/pruning/alignment, one input per call; fixed references constructed before timing. Not an RL batch or epoch benchmark.",
        data_sha256=digest(output / "end_to_end.csv"),
    )
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["end_to_end_replay"], indent=2))


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
    identity_checked = 0
    for case in panel.to_dict("records"):
        directory = output / "checkpoints" / str(case["molecule_id"])
        detail = json.loads((directory / "complete.json").read_text())
        for filename, expected in detail["files_sha256"].items():
            if digest(directory / filename) != expected:
                raise ValueError(f"Changed checkpoint: {directory / filename}")
        for pool in directory.glob("*_pool.sdf"):
            if not pool.stat().st_size:
                continue
            for record in Chem.SDMolSupplier(str(pool), removeHs=False):
                if record is None:
                    raise ValueError(f"Invalid saved geometry: {pool}")
                check_identity(
                    record, Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
                )
                identity_checked += 1
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
        accepted_geometry_identity_checks=identity_checked,
        public_scorer_replays=replay_results,
        validation_script_sha256=digest(__file__),
    )
    atomic_json(path, manifest)
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=6)
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args()
    validate(args.output, args.replays)
    if args.end_to_end:
        validate_end_to_end(args.output)
