"""Stratify saved mega-benchmark measurements; never generate or rescore molecules."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from _david.code.config import PROJECT_ROOT
from _david.code.visualization import render_conformer_analysis


STRATA = ("1", "2", "3-4", "5-8", ">=9")
KEYS = ["molecule_id", "method", "conformers", "isomer_cap", "reference_cap"]
MEASURES = [
    "score",
    "seconds",
    "generation_seconds",
    "optimization_seconds",
    "overlay_seconds",
    "candidate_conformers",
    "scored_conformers",
    "actual_isomers",
    "embedding_failures",
    "failed_conformers",
    "failure",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_parent(frame, manifest, molecule_id):
    """Reject incomplete grids and inconsistent scores/counts before aggregation."""
    dimensions = ["method", "seed", "conformers", "isomer_cap", "reference_cap"]
    expected = set(
        itertools.product(
            range(len(manifest["methods"])),
            manifest["seeds"],
            manifest["conformers"],
            manifest["isomer_caps"],
            manifest["reference_caps"],
        )
    )
    observed = list(frame[dimensions].itertuples(index=False, name=None))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(
            f"Incomplete/duplicate configuration grid: molecule {molecule_id}"
        )
    if not frame.molecule_id.eq(molecule_id).all():
        raise ValueError("Checkpoint filename and molecule ID differ")
    pools = frame.method.map(
        {int(k): v["count"] for k, v in manifest["references"].items()}
    )
    available = frame.reference_cap.eq(0) | frame.reference_cap.le(pools)
    assert frame.available.eq(available).all()
    actual = frame.reference_cap.where(frame.reference_cap.ne(0), pools).where(
        available, 0
    )
    assert frame.actual_references.eq(actual).all()
    assert frame.loc[~available, ["score", "seconds"]].isna().all().all()
    valid = frame.loc[available]
    assert valid.score.between(0, 1).all()
    assert np.isfinite(valid[MEASURES]).all().all()
    assert valid[MEASURES[1:]].ge(0).all().all()
    assert valid.failure.eq(valid.scored_conformers.eq(0)).all()
    assert valid.loc[valid.failure, "score"].eq(0).all()
    assert valid.actual_isomers.le(valid.isomer_cap).all()
    assert valid.actual_isomers.le(valid.candidate_conformers).all()
    assert valid.scored_conformers.le(valid.candidate_conformers).all()
    assert valid.failed_conformers.eq(
        valid.candidate_conformers - valid.scored_conformers
    ).all()
    assert np.allclose(
        valid.seconds,
        valid.generation_seconds + valid.optimization_seconds + valid.overlay_seconds,
    )


def aggregate_parents(frame):
    """Seeds are repeated measurements; the parent is the unit of each summary."""
    grouped = frame[frame.available].groupby(KEYS, sort=False)
    result = grouped[MEASURES + ["actual_references"]].mean().reset_index()
    result = result.rename(columns={"failure": "failure_fraction"})
    result["seed_score_sd"] = grouped.score.std().to_numpy()
    return result


def prepare_analysis(parents, panel):
    bounds = panel.isomer_upper_bound
    if not bounds.ge(1).all() or not bounds.eq(bounds.astype(int)).all():
        raise ValueError("Isomer upper bounds must be positive integers")
    panel = panel.assign(
        stratum=np.select(
            [bounds.eq(1), bounds.eq(2), bounds.between(3, 4), bounds.between(5, 8)],
            ["1", "2", "3-4", "5-8"],
            default=">=9",
        )
    )
    parents = parents.merge(
        panel[["molecule_id", "stratum"]], on="molecule_id", validate="many_to_one"
    )
    base_keys = ["molecule_id", "method", "conformers", "reference_cap"]
    base = parents[parents.isomer_cap.eq(1)][base_keys + ["score", "failure_fraction"]]
    base = base.rename(
        columns={"score": "base_score", "failure_fraction": "base_failure"}
    )
    parents = parents.merge(base, on=base_keys, validate="many_to_one")
    parents["isomer_gain"] = parents.score - parents.base_score
    parents["successful_pair_gain"] = parents.isomer_gain.where(
        parents.failure_fraction.eq(0) & parents.base_failure.eq(0)
    )
    minimum = parents[parents.isomer_cap.eq(1) & parents.conformers.eq(1)]
    minimum = minimum[["molecule_id", "method", "reference_cap", "score"]].rename(
        columns={"score": "minimum_score"}
    )
    parents = parents.merge(
        minimum, on=["molecule_id", "method", "reference_cap"], validate="many_to_one"
    )
    parents["budget_gain"] = parents.score - parents.minimum_score
    parents["retained_fraction"] = (
        parents.scored_conformers / parents.candidate_conformers.replace(0, np.nan)
    )
    return parents, panel


def summarize_strata(parents):
    return (
        parents[parents.reference_cap.eq(0)]
        .groupby(["stratum", "method", "conformers", "isomer_cap"], observed=True)
        .agg(
            molecules=("molecule_id", "size"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q90_score=("score", lambda x: x.quantile(0.9)),
            q95_score=("score", lambda x: x.quantile(0.95)),
            mean_isomer_gain=("isomer_gain", "mean"),
            median_isomer_gain=("isomer_gain", "median"),
            successful_pair_gain=("successful_pair_gain", "mean"),
            successful_pairs=("successful_pair_gain", "count"),
            mean_budget_gain=("budget_gain", "mean"),
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            p95_seconds=("seconds", lambda x: x.quantile(0.95)),
            failure_fraction=("failure_fraction", "mean"),
            mean_isomers=("actual_isomers", "mean"),
            median_embedded=("candidate_conformers", "median"),
            retained_fraction=("retained_fraction", "mean"),
            seed_score_sd=("seed_score_sd", "mean"),
        )
        .reset_index()
    )


def run(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    panel = pd.read_csv(source / "panel.csv")
    columns = KEYS + ["seed", "available", "actual_references"] + MEASURES
    paths = sorted((source / "parts").glob("*.csv.gz"))
    if not paths:
        paths = [source / "data.csv.gz"]
    hashes, tables, rows = {}, [], 0
    for path in paths:
        hashes[str(path.relative_to(source))] = sha256(path)
        frame = pd.read_csv(path, usecols=columns)
        for molecule_id, parent in frame.groupby("molecule_id", sort=False):
            if (
                path.parent.name == "parts"
                and int(path.name.split(".")[0]) != molecule_id
            ):
                raise ValueError("Checkpoint filename and contents differ")
            validate_parent(parent, manifest, molecule_id)
        tables.append(aggregate_parents(frame))
        rows += len(frame)
    parents = pd.concat(tables, ignore_index=True)
    if parents.duplicated(KEYS).any():
        raise ValueError("Duplicate parents across checkpoints")
    ids = sorted(parents.molecule_id.unique().tolist())
    if not set(ids).issubset(set(panel.molecule_id)):
        raise ValueError("Unknown molecule IDs in measurements")
    selected = panel[panel.molecule_id.isin(ids)].copy()
    parents, selected = prepare_analysis(parents, selected)
    summary = summarize_strata(parents)
    anchor = parents[
        (parents.method.eq(1))
        & parents.conformers.eq(max(manifest["conformers"]))
        & parents.isomer_cap.eq(max(manifest["isomer_caps"]))
        & parents.reference_cap.eq(0)
    ]
    cohort = anchor.sort_values(["score", "molecule_id"], ascending=[False, True]).head(
        int(np.ceil(0.05 * len(ids)))
    )
    output.mkdir(parents=True, exist_ok=False)
    parents.to_csv(output / "molecules.csv.gz", index=False, compression="gzip")
    selected.to_csv(output / "panel.csv", index=False)
    summary.to_csv(output / "stratified.csv", index=False)
    cohort[["molecule_id", "score"]].to_csv(
        output / "fixed_top_cohort.csv", index=False
    )
    metadata = {
        "state": "provisional" if len(ids) < len(panel) else "complete_panel",
        "created_prague": datetime.now(ZoneInfo("Europe/Prague")).isoformat(),
        "host": socket.gethostname(),
        "source_directory": str(source),
        "measurement_revision": manifest["git_revision"],
        "analysis_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "command": [
            sys.executable,
            "-m",
            "_david.code.conformer.analyze",
            *sys.argv[1:],
        ],
        "planned_molecules": len(panel),
        "completed_molecules": len(ids),
        "missing_ids": sorted(set(panel.molecule_id) - set(ids)),
        "measurement_rows": rows,
        "strata": {s: int(selected.stratum.eq(s).sum()) for s in STRATA},
        "methods": manifest["methods"],
        "references": manifest["references"],
        "conformers": manifest["conformers"],
        "isomer_caps": manifest["isomer_caps"],
        "checkpoint_sha256": hashes,
        "source_panel_sha256": sha256(source / "panel.csv"),
        "source_sha256": {
            p: sha256(PROJECT_ROOT / p)
            for p in (
                "_david/code/conformer/analyze.py",
                "_david/code/visualization.py",
            )
        },
        "definitions": {
            "score": "TanimotoCombo / 2; not biological activity or self-recovery",
            "unit": "Parent molecule after averaging the three embedding seeds",
            "isomer_gain": "Paired score minus cap-1 score at the same method/C/reference budget",
            "budget_gain": "Paired score minus C1/I1 score within the same method/reference budget",
            "successful_pair_gain": "Only parents with no preparation failures across seeds at either cap",
            "top_cohort": "Fixed ceil(5% * completed parents), MMFF94/C100/I30/all; exploratory selection on these measurements",
            "timing": manifest["timing"],
            "uncertainty": "Descriptive summaries; seed SD is not a confidence interval",
            "limitations": "Slow unfinished molecules excluded; time may be biased downward. Isomer groups are upper bounds; all<=30 is not exhaustive.",
        },
        "validation": "Complete unique grids, reference masks/counts, normalized score bounds, failure/count invariants and decomposed time checked before aggregation",
    }
    render_conformer_analysis(
        parents, summary, cohort.molecule_id.tolist(), output, metadata
    )
    metadata["sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "parents": len(ids),
                "rows": rows,
                "strata": metadata["strata"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
