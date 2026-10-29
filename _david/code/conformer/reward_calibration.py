"""Direct self-recovery score/time grid for a frozen drug panel; no RL updates.

Each cell is a fresh production-factory call. Embeddings are NOT reused across
cells for timing. References are prepared once per input, energy filtered and
ordered without looking at candidates. Whole inputs run in separate workers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from _david.code.config import PROJECT_ROOT
from _david.code.conformer.reference_bank import (
    METHODS,
    REFERENCE_SEED,
    CANDIDATE_SEED,
    prepare_banks,
)
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.environment import create_rocs_scorer

CANDIDATES = (10, 20, 30, 40, 50, 75)
CAPS = (1, 2, 4, 8, 12, 0)  # 0 means all surviving reference structures.
KEYS = ["molecule_id", "method", "reference_cap", "candidate_attempts"]
SOURCES = [
    f"_david/code/{name}.py"
    for name in ("chemistry", "reference", "environment", "rocs")
]
SOURCES += [
    f"_david/code/conformer/{name}.py"
    for name in ("reference_bank", "reward_calibration")
]
SOURCES += [
    "drugex/training/scorers/rocs_rdkit.py",
    "drugex/training/scorers/conformer_generators.py",
]


def conditions(candidates=CANDIDATES, caps=CAPS):
    return [
        dict(method=method, reference_cap=cap, candidate_attempts=count)
        for method, cap, count in itertools.product(METHODS, caps, candidates)
    ]


def measure_cell(case, setting, references, directory):
    """Fresh generation/minimization/I/O/alignment wall time, without ref setup."""
    cap = setting["reference_cap"]
    refs = references[:cap] if cap else references
    method, count = setting["method"], setting["candidate_attempts"]
    scorer = create_rocs_scorer(
        refs,
        max_conformers=count,
        max_isomers=4,
        timeout=10,
        conformer_threads=1,
        n_jobs=1,
        random_seed=CANDIDATE_SEED,
        minimize_candidates=method != "etkdg",
        optimization=METHODS[method],
    )
    generator = scorer.conformer_generator
    generator.embedding_rmsd = None
    generator.protonate = False
    original = generator.genConformers
    capture = directory / "candidate_call.sdf"
    measurement = {}

    def generate(smiles, output):
        tick = time.perf_counter()
        path = original(smiles, output)
        measurement["generation_seconds"] = time.perf_counter() - tick
        # O(1) hard link preserves the file after native TemporaryDirectory cleanup.
        # Count/hash it AFTER the timed call, without a timed checkpoint copy.
        os.link(path, capture)
        return path

    generator.genConformers = generate
    tick = time.perf_counter()
    score = float(scorer.getScores([case["smiles"]])[0, 0]) / 2
    seconds = time.perf_counter() - tick
    records = (
        list(Chem.SDMolSupplier(str(capture), removeHs=False))
        if capture.stat().st_size
        else []
    )
    if any(record is None for record in records):
        raise ValueError("Public scorer produced an invalid saved candidate record")
    fingerprint = digest(capture)
    canonical = directory / f"candidate_{method}_c{count}_{fingerprint[:12]}.sdf"
    if canonical.exists():
        capture.unlink()
    else:
        capture.rename(canonical)
    return dict(
        score=score,
        seconds=seconds,
        generation_seconds=measurement["generation_seconds"],
        alignment_io_seconds=seconds - measurement["generation_seconds"],
        actual_candidates=len(records),
        candidate_failure=not records,
        candidate_attempted=True,
        identity_rejections=len(generator.last_identity_rejections)
        + generator.last_minimization_statuses.count(-2),
        minimization_failures=sum(
            status != 0 for status in generator.last_minimization_statuses
        ),
        candidate_sha256=fingerprint,
        candidate_file=canonical.name,
    )


def measure_input(case, settings, output, hashes):
    directory = Path(output) / "checkpoints" / str(case["molecule_id"])
    directory.mkdir(parents=True, exist_ok=True)
    # A failed earlier call may have left only its temporary capture link.
    (directory / "candidate_call.sdf").unlink(missing_ok=True)
    started = time.perf_counter()
    methods = tuple(dict.fromkeys(setting["method"] for setting in settings))
    references, metadata = prepare_banks(case["smiles"], directory, methods)
    rows = []
    # Reverse half the inputs to reduce a systematic cap/method order effect.
    ordered = settings[:: 1 if case["molecule_id"] % 2 else -1]
    for setting in ordered:
        pool = references[setting["method"]]
        cap = setting["reference_cap"]
        available = min(cap, len(pool)) if cap else len(pool)
        row = dict(
            molecule_id=case["molecule_id"],
            name=case["name"],
            split=case["split"],
            **setting,
            color_excluded=metadata["color_excluded"],
            actual_references=available,
            reference_pool=len(pool),
            reference_failure=not pool,
            reference_seconds=metadata["reference_seconds"],
            enumerated_reference_isomers=metadata["enumerated_isomers"],
        )
        if pool and not metadata["color_excluded"]:
            row.update(measure_cell(case, setting, pool, directory))
        else:
            row.update(
                score=np.nan if metadata["color_excluded"] else 0.0,
                seconds=0.0,
                generation_seconds=0.0,
                alignment_io_seconds=0.0,
                actual_candidates=0,
                candidate_failure=False,
                candidate_attempted=False,
                identity_rejections=0,
                minimization_failures=0,
                candidate_sha256="",
                candidate_file="",
            )
        rows.append(row)
        pd.DataFrame(rows).to_csv(directory / "partial_rows.csv", index=False)
    if hashes != source_hashes():
        raise ValueError("Scientific source changed during measurement")
    pd.DataFrame(rows).to_csv(directory / "rows.csv", index=False)
    (directory / "partial_rows.csv").unlink()
    metadata.update(
        case=case,
        wall_seconds=time.perf_counter() - started,
        source_sha256=hashes,
        files_sha256={p.name: digest(p) for p in directory.iterdir() if p.is_file()},
    )
    atomic_json(directory / "complete.json", metadata)
    return case["molecule_id"]


def source_hashes():
    return {name: digest(PROJECT_ROOT / name) for name in SOURCES}


def summarize(frame):
    rows = []
    for values, group in frame.groupby(KEYS[1:]):
        valid = group[~group.color_excluded]
        rows.append(
            dict(zip(KEYS[1:], values))
            | dict(
                inputs=len(group),
                scored_inputs=len(valid),
                color_excluded=int(group.color_excluded.sum()),
                mean=valid.score.mean(),
                median=valid.score.median(),
                p05=valid.score.quantile(0.05),
                fraction_ge_095=valid.score.ge(0.95).mean(),
                mean_seconds=valid.seconds.mean(),
                median_seconds=valid.seconds.median(),
                p95_seconds=valid.seconds.quantile(0.95),
                mean_references=valid.actual_references.mean(),
                median_references=valid.actual_references.median(),
                max_references=valid.actual_references.max(),
                mean_candidates=valid.actual_candidates.mean(),
                reference_failures=int(valid.reference_failure.sum()),
                candidate_failures=int(valid.candidate_failure.sum()),
            )
        )
    return pd.DataFrame(rows)


def collect(output, settings, panel):
    frames = [
        pd.read_csv(output / "checkpoints" / str(mid) / "rows.csv")
        for mid in panel.molecule_id
    ]
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(panel) * len(settings) or frame.duplicated(KEYS).any():
        raise ValueError("Incomplete or duplicated grid")
    valid = frame[~frame.color_excluded]
    # Native Gaussian overlap can exceed one by a few parts per million.
    # Retain its raw value; this is a validation tolerance, not score clipping.
    if not valid.score.between(0, 1.0001).all():
        raise ValueError("Invalid normalized score")
    for _, group in valid.groupby(["molecule_id", "method", "candidate_attempts"]):
        by_cap = group.assign(order=group.reference_cap.replace(0, 10000)).sort_values(
            "order"
        )
        # Wall-time embedding guards can produce different usable outputs under
        # shared-host load. Compare nested maxima only for identical candidates.
        if (
            group.candidate_sha256.nunique() == 1
            and (by_cap.score.diff().dropna() < -1e-8).any()
        ):
            raise ValueError(
                "A nested larger reference bank decreased the maximum score"
            )
    frame.to_csv(output / "data.csv.gz", index=False)
    summary = summarize(frame)
    summary.to_csv(output / "summary.csv", index=False)
    return frame, summary


def shortlist(summary):
    """Predeclared engineering target: lowest measured mean time at mean≥0.95."""
    chosen = []
    for _method, group in summary.groupby("method"):
        passing = group[group["mean"].ge(0.95)]
        row = (
            passing.sort_values(["mean_seconds", "p05"], ascending=[True, False]).iloc[
                0
            ]
            if len(passing)
            else group.sort_values("mean", ascending=False).iloc[0]
        )
        chosen.append(
            {
                **{
                    key: row[key].item() if hasattr(row[key], "item") else row[key]
                    for key in KEYS[1:]
                },
                "development_target_met": bool(row["mean"] >= 0.95),
            }
        )
    # Also retain highest-quality global cell and an explicit C20/K4 current-profile control.
    row = summary.sort_values("mean", ascending=False).iloc[0]
    chosen.append(
        {
            key: row[key].item() if hasattr(row[key], "item") else row[key]
            for key in KEYS[1:]
        }
    )
    chosen.append(
        dict(method="mmff_high_dielectric", reference_cap=4, candidate_attempts=20)
    )
    unique = {tuple(item[key] for key in KEYS[1:]): item for item in chosen}
    return list(unique.values())


def save_progress(root, raw, manifest):
    atomic_json(raw / "manifest.json", manifest)
    atomic_json(
        root / "progress.json",
        dict(
            phase=manifest["phase"],
            state=manifest["state"],
            completed=len(manifest["completed_molecules"]),
            total=manifest["total_molecules"],
            errors=manifest.get("job_errors", {}),
            updated_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


def run(args):
    root, raw = args.output, args.output / "raw_data" / args.phase
    raw.mkdir(parents=True, exist_ok=True)
    logs = root / "logs" / args.phase
    logs.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel)
    panel = panel[panel.split.eq(args.split)]
    if args.limit:
        panel = panel.head(args.limit)
    settings = (
        json.loads(args.settings.read_text())
        if args.settings
        else conditions(
            tuple(map(int, args.candidates.split(","))),
            tuple(map(int, args.caps.split(","))),
        )
    )
    settings = [{key: item[key] for key in KEYS[1:]} for item in settings]
    hashes = source_hashes()
    path = raw / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["source_sha256"] != hashes or manifest["settings"] != settings:
            raise ValueError("Resume source or protocol mismatch")
        if digest(args.panel) != manifest["source_panel_sha256"]:
            raise ValueError("Frozen panel changed")
        if manifest["input_ids"] != panel.molecule_id.tolist():
            raise ValueError("Resume input subset changed")
    else:
        panel.to_csv(raw / "panel.csv", index=False)
        manifest = dict(
            state="measuring",
            phase=args.phase,
            created_utc=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], text=True),
            rdkit=rdBase.rdkitVersion,
            source_sha256=hashes,
            source_panel_sha256=digest(args.panel),
            settings=settings,
            total_molecules=len(panel),
            input_ids=panel.molecule_id.tolist(),
            expected_rows=len(panel) * len(settings),
            completed_molecules=[],
            workers=args.workers,
            protocol="R50/I8; disjoint geometry seed streams104729/42, stereo42; energy8 within stereo for optimized references; energy-seeded reference-only shape/colour diversity; native max combo /2; C per stereo I4; no RMSD; no pH reassignment; reference embedding60s/stereo, candidate10s/stereo; one convergence1000-iteration solver limit; no FF fallback",
            reference_geometry_seed=REFERENCE_SEED,
            candidate_geometry_seed=CANDIDATE_SEED,
            seed_control=dict(
                effective_seed="base*(conformer_index+1)",
                overlapping_bounded_seeds=sorted(
                    set(REFERENCE_SEED * i for i in range(1, 51))
                    & set(CANDIDATE_SEED * i for i in range(1, 76))
                ),
                max_reference_effective_seed=REFERENCE_SEED * 50,
                max_candidate_effective_seed=CANDIDATE_SEED * 75,
                sequential_random_seeds=False,
            ),
            timing="Fresh getScores per cell including gen/min/I/O/alignment, n_jobs1. Worker handles one complete input. Ref setup excluded. A single hard link captures candidates inside timing; record parsing/hashing and reference selection are outside it.",
            objective="Self-recovery and compute only; 0.95 is an engineering target, not biological accuracy or a statistical significance level",
        )
    jobs = []
    for case in panel.to_dict("records"):
        checkpoint = raw / "checkpoints" / str(case["molecule_id"]) / "complete.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["source_sha256"] != hashes:
                raise ValueError("Checkpoint source mismatch")
            if case["molecule_id"] not in manifest["completed_molecules"]:
                manifest["completed_molecules"].append(case["molecule_id"])
        else:
            jobs.append(case)
    save_progress(root, raw, manifest)
    errors = {}
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(measure_input, case, settings, raw, hashes): case
            for case in jobs
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                manifest["completed_molecules"].append(future.result())
            except Exception as exc:
                errors[str(case["molecule_id"])] = str(exc)
                (logs / f"input_{case['molecule_id']}.error.log").write_text(
                    traceback.format_exc()
                )
            manifest.update(
                job_errors=errors,
                state="measuring_with_errors" if errors else "measuring",
            )
            save_progress(root, raw, manifest)
            print(
                f"{len(manifest['completed_molecules'])}/{len(panel)} complete; {len(errors)} errors",
                flush=True,
            )
    if errors:
        manifest["state"] = "failed"
        save_progress(root, raw, manifest)
        raise RuntimeError(errors)
    try:
        frame, summary = collect(raw, settings, panel)
    except Exception as exc:
        manifest.update(state="failed", postprocessing_error=str(exc))
        save_progress(root, raw, manifest)
        raise
    if not args.settings:
        atomic_json(raw / "shortlist.json", shortlist(summary))
    manifest.update(
        state="complete",
        rows=len(frame),
        output_sha256={
            name: digest(raw / name)
            for name in ("data.csv.gz", "summary.csv", "panel.csv")
        },
    )
    manifest["candidate_geometry_variation_groups"] = int(
        (
            frame.groupby(
                ["molecule_id", "method", "candidate_attempts"]
            ).candidate_sha256.nunique()
            > 1
        ).sum()
    )
    save_progress(root, raw, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--phase",
        choices=["pilot", "development", "confirmation"],
        default="development",
    )
    parser.add_argument(
        "--split", choices=["development", "confirmation"], default="development"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidates", default=",".join(map(str, CANDIDATES)))
    parser.add_argument("--caps", default=",".join(map(str, CAPS)))
    parser.add_argument("--settings", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
