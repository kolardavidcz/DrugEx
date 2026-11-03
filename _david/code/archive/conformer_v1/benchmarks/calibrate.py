"""Joint drug self-recovery calibration. No RL training or production defaults changed.

Run each requested embedding budget directly, reuse its saved geometries across
RMSD comparisons, and checkpoint one complete 36-cell grid per input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import multiprocessing
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from _david.code.chemistry import (
    IDENTITY_REJECTED,
    CandidateConformerGenerator,
    Optimization,
    check_identity,
    enumerate_stereoisomers,
    optimize_conformers,
    prepare_candidate_molecule,
)
from _david.code.config import PROJECT_ROOT
from _david.code.conformer.benchmarks.self_recovery import SOURCE_PATHS
from _david.code.conformer.core.utils import atomic_json, collect_jobs, digest, save_sdf
from _david.code.reference import PrepareMolecule, select_conformers
from _david.code.rocs import has_color, score_pair

prepared_molecule = prepare_candidate_molecule

PROFILE = Optimization("MMFF94s", 78.4)
REFERENCE_ATTEMPTS = (50, 100)
CANDIDATE_ATTEMPTS = (10, 20, 40)
REFERENCE_RMSD = (0.5, 1.25)
CANDIDATE_RMSD = (None, 0.5, 1.25)
KEYS = [
    "molecule_id",
    "reference_attempts",
    "candidate_attempts",
    "reference_rmsd",
    "candidate_rmsd",
]
SOURCES = (
    *SOURCE_PATHS,
    "_david/code/conformer/calibrate.py",
    "_david/code/conformer/drug_panel.py",
)


def reference_pool(smiles, attempts, directory):
    """The production reference preparer per identity; retain all energy survivors."""
    started = time.perf_counter()
    try:
        mol = prepared_molecule(smiles)
    except (ValueError, RuntimeError) as exc:
        return [], dict(
            seconds=time.perf_counter() - started,
            enumerated_isomers=0,
            embedded_successful_identities=0,
            converged_successful_identities=0,
            failed_identities=1,
            errors=[str(exc)],
        )
    isomers = list(enumerate_stereoisomers(mol, limit=8, seed=42))
    pool, errors, embedded, converged = [], [], 0, 0
    statuses, rejected_identity = Counter(), 0
    for index, iso in enumerate(isomers):
        preparer = PrepareMolecule(
            num_conformers=attempts,
            max_reference_conformers=attempts,
            random_seed=43,
            rmsd_cutoff=None,
            num_threads=1,
            optimization=PROFILE,
        )
        try:
            path = preparer.generate_reference_conformers(
                iso, directory / f"reference_{attempts}_{index}.sdf"
            )
            records = list(Chem.SDMolSupplier(str(path), removeHs=False))
            if not records or any(m is None for m in records):
                raise ValueError("Empty or invalid prepared reference SDF")
            embedded += records[0].GetIntProp("EmbeddedConformers")
            converged += records[0].GetIntProp("ConvergedConformers")
            for record in records:
                check_identity(record, iso)
                record.SetProp(
                    "EnumeratedIsomerSMILES", iso.GetProp("EnumeratedIsomerSMILES")
                )
            pool.extend(records)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"isomer {index}: {type(exc).__name__}: {exc}")
        finally:
            statuses.update(map(str, preparer.last_optimization_statuses))
            rejected_identity += preparer.last_optimization_statuses.count(
                IDENTITY_REJECTED
            ) + len(preparer.last_identity_rejections)
    info = dict(
        seconds=time.perf_counter() - started,
        enumerated_isomers=len(isomers),
        embedded_successful_identities=embedded,
        converged_successful_identities=converged,
        failed_identities=len(errors),
        errors=errors,
        optimization_statuses=dict(statuses),
        identity_rejected=rejected_identity,
    )
    save_sdf(directory / f"reference_{attempts}_pool.sdf", pool)
    return pool, info


def candidate_pool(smiles, attempts, directory):
    started = time.perf_counter()
    generator = CandidateConformerGenerator(
        max_conformers=attempts,
        max_isomers=4,
        random_seed=42,
        stereo_seed=42,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=1,
        timeout=0,
        show_progress=False,
    )
    errors, records, enumerated_isomers = [], [], 0
    try:
        enumerated_isomers = len(
            list(generator._get_isomers(prepared_molecule(smiles)))
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = generator.genConformers([smiles], temporary)
            if Path(path).stat().st_size:
                records = list(Chem.SDMolSupplier(str(path), removeHs=False))
            if any(m is None for m in records):
                raise ValueError("Invalid candidate SDF record")
            # The shared RL generator already rejects invalid embedded identities.
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        records = []
    generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    usable, statuses = optimize_conformers(records, PROFILE)
    info = dict(
        generation_seconds=generation_seconds,
        optimization_seconds=time.perf_counter() - started,
        embedded=len(records),
        usable=len(usable),
        optimization_statuses=dict(Counter(map(str, statuses))),
        errors=errors,
        enumerated_isomers=enumerated_isomers,
        embedded_identity_rejected=len(generator.last_identity_rejections),
        embedded_identity_rejection_reasons=generator.last_identity_rejections,
        minimized_identity_rejected=statuses.count(IDENTITY_REJECTED),
    )
    save_sdf(directory / f"candidate_{attempts}_pool.sdf", usable)
    return usable, info


def selections(records, cutoffs):
    result, timings = {}, {}
    for cutoff in cutoffs:
        started = time.perf_counter()
        result[cutoff] = select_conformers(records, cutoff)
        timings[cutoff] = time.perf_counter() - started if cutoff is not None else 0.0
    return result, timings


def measure_case(case, output):
    directory = Path(output) / "checkpoints" / str(case["molecule_id"])
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    references, ref_info, ref_selections, ref_times = {}, {}, {}, {}
    for attempts in REFERENCE_ATTEMPTS:
        pool, info = reference_pool(case["smiles"], attempts, directory)
        references[attempts], ref_info[attempts] = pool, info
        ref_selections[attempts], ref_times[attempts] = selections(pool, REFERENCE_RMSD)
    all_references = [m for records in references.values() for m in records]
    colorless = bool(all_references) and not all(has_color(m) for m in all_references)
    rows, candidate_info, pair_counts = [], {}, {}
    for attempts in CANDIDATE_ATTEMPTS:
        candidates, info = candidate_pool(case["smiles"], attempts, directory)
        candidate_info[attempts] = info
        candidate_selections, candidate_times = selections(candidates, CANDIDATE_RMSD)
        for ref_attempts in REFERENCE_ATTEMPTS:
            refs = references[ref_attempts]
            scores = np.zeros((len(candidates), len(refs), 3))
            times = np.zeros(scores.shape[:2])
            if not colorless:
                for i, candidate in enumerate(candidates):
                    for j, ref in enumerate(refs):
                        tick = time.perf_counter()
                        scores[i, j] = score_pair(candidate, ref)
                        times[i, j] = time.perf_counter() - tick
            np.savez_compressed(
                directory / f"pairs_r{ref_attempts}_c{attempts}.npz",
                scores=scores,
                seconds=times,
            )
            pair_counts[f"{ref_attempts}/{attempts}"] = scores.shape[:2]
            for ref_cutoff, candidate_cutoff in itertools.product(
                REFERENCE_RMSD, CANDIDATE_RMSD
            ):
                ri, ci = (
                    ref_selections[ref_attempts][ref_cutoff],
                    candidate_selections[candidate_cutoff],
                )
                best, winner = (0.0, 0.0, 0.0), (-1, -1)
                if ci and ri and not colorless:
                    values = scores[np.ix_(ci, ri)]
                    i, j = np.unravel_index(values[:, :, 0].argmax(), values.shape[:2])
                    best, winner = values[i, j], (ci[i], ri[j])
                overlay = float(times[np.ix_(ci, ri)].sum())
                rows.append(
                    dict(
                        molecule_id=case["molecule_id"],
                        name=case["name"],
                        split=case["split"],
                        reference_attempts=ref_attempts,
                        candidate_attempts=attempts,
                        reference_rmsd=ref_cutoff,
                        candidate_rmsd=candidate_cutoff or 0.0,
                        score=float(best[0]) if not colorless else np.nan,
                        shape_tanimoto=float(best[1]) if not colorless else np.nan,
                        color_tanimoto=float(best[2]) if not colorless else np.nan,
                        color_excluded=colorless,
                        reference_failure=not ri,
                        candidate_failure=not ci,
                        actual_references=len(ri),
                        actual_candidates=len(ci),
                        enumerated_reference_isomers=ref_info[ref_attempts][
                            "enumerated_isomers"
                        ],
                        enumerated_candidate_isomers=info["enumerated_isomers"],
                        reference_identity_rejected=ref_info[ref_attempts].get(
                            "identity_rejected", 0
                        ),
                        candidate_embedding_identity_rejected=info[
                            "embedded_identity_rejected"
                        ],
                        candidate_minimization_identity_rejected=info[
                            "minimized_identity_rejected"
                        ],
                        generation_seconds=info["generation_seconds"],
                        optimization_seconds=info["optimization_seconds"],
                        candidate_pruning_seconds=candidate_times[candidate_cutoff],
                        overlay_seconds=overlay,
                        seconds=info["generation_seconds"]
                        + info["optimization_seconds"]
                        + candidate_times[candidate_cutoff]
                        + overlay,
                        reference_seconds=ref_info[ref_attempts]["seconds"]
                        + ref_times[ref_attempts][ref_cutoff],
                        winning_candidate=winner[0],
                        winning_reference=winner[1],
                    )
                )
    frame = pd.DataFrame(rows)
    temporary = directory / "rows.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(directory / "rows.csv")
    try:
        prepared_smiles = Chem.MolToSmiles(prepared_molecule(case["smiles"]))
    except (ValueError, RuntimeError):
        prepared_smiles = ""
    details = dict(
        case=case,
        prepared_smiles=prepared_smiles,
        reference=ref_info,
        candidate=candidate_info,
        pair_matrix_shapes=pair_counts,
        color_excluded=colorless,
        wall_seconds=time.perf_counter() - started,
        reference_selections={
            str(k): {str(c): v for c, v in s.items()} for k, s in ref_selections.items()
        },
        files_sha256={p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    )
    atomic_json(directory / "complete.json", details)
    return case["molecule_id"]


def summarize(frame):
    keys = KEYS[1:]
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        valid = group[~group.color_excluded]
        rows.append(
            dict(zip(keys, values))
            | dict(
                inputs=len(group),
                scored_inputs=len(valid),
                color_excluded=int(group.color_excluded.sum()),
                mean=valid.score.mean(),
                median=valid.score.median(),
                p05=valid.score.quantile(0.05),
                fraction_ge_095=valid.score.ge(0.95).mean(),
                seconds=valid.seconds.mean(),
                p95_seconds=valid.seconds.quantile(0.95),
                mean_references=valid.actual_references.mean(),
                mean_candidates=valid.actual_candidates.mean(),
                reference_seconds=valid.reference_seconds.mean(),
                reference_failures=int(valid.reference_failure.sum()),
                candidate_failures=int(valid.candidate_failure.sum()),
            )
        )
    return pd.DataFrame(rows)


def result_root(output):
    return output.parent if output.name == "raw_data" else output


def save_progress(output, manifest):
    """Keep the full provenance and a small user-facing progress file together."""
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        result_root(output) / "progress.json",
        dict(
            state=manifest["state"],
            completed=len(manifest["completed_molecules"]),
            total=manifest["total_molecules"],
            errors=manifest.get("job_errors", {}),
            updated_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


def run(args):
    output = args.output.resolve()
    hashes = {p: digest(PROJECT_ROOT / p) for p in SOURCES}
    if args.render_only:
        from _david.code.conformer.analysis.visualization import render_joint_calibration

        render_joint_calibration(
            pd.read_csv(output / "summary.csv"), result_root(output)
        )
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "checkpoints").mkdir(exist_ok=True)
    path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(path.read_text())
        if hashes != manifest["source_sha256"]:
            raise ValueError("Resume requires the exact measurement source")
        panel = pd.read_csv(output / "panel.csv")
    else:
        panel = pd.read_csv(args.panel)
        if args.limit:
            panel = panel.head(args.limit)
        if panel.empty or panel.molecule_id.duplicated().any():
            raise ValueError("Empty panel or duplicated molecule ID")
        panel.to_csv(output / "panel.csv", index=False)
        manifest = dict(
            state="measuring",
            created_utc=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            rdkit=rdBase.rdkitVersion,
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_sha256=hashes,
            panel_sha256=digest(output / "panel.csv"),
            profile=asdict(PROFILE),
            reference_attempts=REFERENCE_ATTEMPTS,
            candidate_attempts=CANDIDATE_ATTEMPTS,
            reference_rmsd=REFERENCE_RMSD,
            candidate_post_rmsd=CANDIDATE_RMSD,
            reference_isomer_cap=8,
            candidate_isomer_cap=4,
            stereo_seed=42,
            candidate_geometry_seed=42,
            reference_geometry_seed=43,
            reference_embedding_rmsd=0.35,
            candidate_embedding_rmsd=0.5,
            reference_energy_window_kcal=8,
            candidate_energy_window=None,
            reference_policy="All converged energy-filter survivors, Butina within stereo identity; no global count cap",
            score="Maximum over reference/candidate pairs of native (shapeTanimoto+colorTanimoto)/2; no seed averaging",
            timing="Shared generation/minimization/pruning plus sum of selected pair overlay wall times; not independent end-to-end RL latency. Reference preparation reported separately.",
            failures="Reference/candidate preparation failures score zero; colorless references excluded before alignment and reported separately",
            anchor="Unpruned finite 100-reference-attempt/40-candidate-attempt pair matrix saved separately; not an exhaustive ensemble",
            expected_rows=len(panel) * 36,
            total_molecules=len(panel),
            completed_molecules=[],
            workers=args.workers,
            command=[
                sys.executable,
                "-m",
                "_david.code.conformer.calibrate",
                *sys.argv[1:],
            ],
        )
    save_progress(output, manifest)
    try:
        complete = [
            int(i)
            for i in panel.molecule_id
            if (output / "checkpoints" / str(i) / "complete.json").exists()
        ]
        jobs = [c for c in panel.to_dict("records") if c["molecule_id"] not in complete]
        manifest.update(state="measuring", completed_molecules=complete)
        save_progress(output, manifest)
        with ProcessPoolExecutor(
            max_workers=min(args.workers, max(1, len(jobs))),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            errors = collect_jobs(executor, jobs, output, manifest)
        if errors:
            raise RuntimeError(
                f"Unfinished input jobs: {errors}; see logs/input_*.error.log"
            )
        frame = pd.concat(
            [
                pd.read_csv(output / "checkpoints" / str(i) / "rows.csv")
                for i in panel.molecule_id
            ],
            ignore_index=True,
        )
        for column in [
            "reference_identity_rejected",
            "candidate_embedding_identity_rejected",
            "candidate_minimization_identity_rejected",
        ]:
            # Retained legacy checkpoints had no identity errors; their count is zero.
            if column in frame:
                frame[column] = frame[column].fillna(0).astype(int)
        if len(frame) != manifest["expected_rows"] or frame.duplicated(KEYS).any():
            raise ValueError("Incomplete or duplicated calibration grid")
        if not np.allclose(
            frame.score,
            (frame.shape_tanimoto + frame.color_tanimoto) / 2,
            equal_nan=True,
        ):
            raise ValueError("Score/component mismatch")
        if hashes != {p: digest(PROJECT_ROOT / p) for p in SOURCES}:
            raise ValueError("Measurement source changed")
        frame.to_csv(output / "data.csv.gz", index=False, compression="gzip")
        summary = summarize(frame)
        summary.to_csv(output / "summary.csv", index=False)
        from _david.code.conformer.analysis.visualization import render_joint_calibration

        render_joint_calibration(summary, result_root(output))
        manifest.update(
            state="complete",
            measurement_rows=len(frame),
            sha256={
                p.name: digest(p) for p in output.iterdir() if p.is_file() and p != path
            },
        )
        save_progress(output, manifest)
    except Exception as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        save_progress(output, manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    run(parser.parse_args())
