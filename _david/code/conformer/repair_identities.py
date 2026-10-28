"""Recompute only inputs affected by identity rejection, preserving prior evidence.

This can run beside the original job: it replaces only jobs already finished or
failed, never an input still being measured. The original manifest is reconciled
after its coordinator exits; this driver never writes that live manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from _david.code.config import PROJECT_ROOT
from _david.code.conformer.calibrate import SOURCES, measure_case
from _david.code.conformer.self_recovery import atomic_json, digest


def affected_inputs(raw, root):
    ids = set(
        map(int, json.loads((root / "progress.json").read_text()).get("errors", {}))
    )
    for path in (raw / "checkpoints").glob("*/complete.json"):
        item = json.loads(path.read_text())
        errors = [
            error
            for side in ("reference", "candidate")
            for info in item[side].values()
            for error in info["errors"]
        ]
        if any("identity" in error.lower() for error in errors):
            ids.add(int(path.parent.name))
    return ids


def run(root, workers):
    raw = root / "raw_data"
    plan_path = raw / "identity_repair.json"
    hashes = {p: digest(PROJECT_ROOT / p) for p in SOURCES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plan = (
        json.loads(plan_path.read_text())
        if plan_path.exists()
        else dict(
            source_sha256=hashes,
            revision=revision,
            completed=[],
            policy="Reject only invalid conformers; preserve input and every valid conformer",
            source_dirty_state=subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
        )
    )
    if plan["source_sha256"] != hashes:
        raise ValueError("Repair source changed; inspect provenance before resuming")
    targets = sorted(affected_inputs(raw, root) | set(plan.get("input_ids", [])))
    plan["input_ids"] = targets
    plan["state"] = "running"
    atomic_json(plan_path, plan)
    cases = pd.read_csv(raw / "panel.csv").set_index("molecule_id").to_dict("index")
    prior = raw / "provenance/prior_identity_handling"
    prior.mkdir(parents=True, exist_ok=True)
    jobs = []
    for molecule_id in targets:
        if molecule_id in plan["completed"]:
            continue
        directory = raw / "checkpoints" / str(molecule_id)
        archive = prior / str(molecule_id)
        if directory.exists():
            if archive.exists():
                raise ValueError(
                    f"Both current and prior incomplete data exist for {molecule_id}; inspect before overwriting"
                )
            shutil.move(str(directory), str(archive))
        jobs.append(dict(cases[molecule_id], molecule_id=molecule_id))
    with ProcessPoolExecutor(
        max_workers=min(workers, max(1, len(jobs))),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {pool.submit(measure_case, case, str(raw)): case for case in jobs}
        for future in as_completed(futures):
            molecule_id = future.result()
            path = raw / "checkpoints" / str(molecule_id) / "complete.json"
            checkpoint = json.loads(path.read_text())
            checkpoint["identity_repair_source"] = dict(
                revision=revision, source_sha256=hashes
            )
            atomic_json(path, checkpoint)
            plan["completed"].append(molecule_id)
            plan["updated_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(plan_path, plan)
            atomic_json(
                root / "repair_progress.json",
                dict(
                    completed=len(plan["completed"]),
                    total=len(targets),
                    state="running",
                ),
            )
            print(
                f"Identity repair {len(plan['completed'])}/{len(targets)}: {molecule_id}",
                flush=True,
            )
    if hashes != {p: digest(PROJECT_ROOT / p) for p in SOURCES}:
        raise ValueError("Source changed during identity repairs")
    plan["state"] = "complete"
    atomic_json(plan_path, plan)
    atomic_json(
        root / "repair_progress.json",
        dict(completed=len(plan["completed"]), total=len(targets), state="complete"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.workers)
