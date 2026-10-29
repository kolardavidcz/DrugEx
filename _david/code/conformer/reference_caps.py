"""Compare retained reference caps using saved geometries and pair measurements.

No embedding, minimization or new ROCS alignment. Reference selection is blind
to candidate scores: cycle over stereo identities, preserving their energy order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.conformer.calibrate import KEYS, summarize
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.reference import select_conformers

REFERENCE_CAPS = (1, 2, 4, 8, 0)  # zero means all available references


def reference_pruning_anchor(raw):
    """Isolate reference pruning loss using the same saved candidate geometries."""
    frame = pd.read_csv(raw / "data.csv.gz")
    cells = frame[
        frame.reference_attempts.eq(100)
        & frame.reference_rmsd.eq(0.5)
        & frame.candidate_attempts.isin([10, 40])
        & frame.candidate_rmsd.eq(0)
    ]
    rows = []
    for cell in cells.to_dict("records"):
        path = (
            raw
            / "checkpoints"
            / str(cell["molecule_id"])
            / f"pairs_r100_c{cell['candidate_attempts']}.npz"
        )
        with np.load(path) as saved:
            scores = saved["scores"]
            best = scores[:, :, 0].max() if scores.size else 0.0
            if cell["color_excluded"]:
                best = np.nan
            elif best < cell["score"] - 1e-8:
                raise ValueError(
                    "Unpruned reference anchor cannot score below its subset"
                )
            rows.append(
                dict(
                    molecule_id=cell["molecule_id"],
                    name=cell["name"],
                    candidate_attempts=cell["candidate_attempts"],
                    score_pruned=cell["score"],
                    score_unpruned=float(best),
                    pruning_loss=float(best - cell["score"]),
                    retained_references=cell["actual_references"],
                    unpruned_references=scores.shape[1],
                    color_excluded=cell["color_excluded"],
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(raw / "reference_pruning_anchor.csv", index=False)
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["reference_pruning_anchor"] = dict(
        rows=len(result),
        data_sha256=digest(raw / "reference_pruning_anchor.csv"),
        script_sha256=digest(__file__),
        scope="Same saved candidates against full reference energy survivors versus 0.5-A post-minimization pruning; embedding pruning and force field unchanged. No new alignments or exhaustive-coverage claim.",
    )
    atomic_json(raw / "manifest.json", manifest)
    return result


def balanced_reference_order(records, indices):
    """Take one energy-ordered representative per stereo identity each round."""
    groups = {}
    for index in indices:
        identity = records[index].GetProp("EnumeratedIsomerSMILES")
        groups.setdefault(identity, []).append(index)
    return [
        index
        for row in itertools.zip_longest(*groups.values())
        for index in row
        if index is not None
    ]


def read_records(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    records = list(Chem.SDMolSupplier(str(path), removeHs=False))
    if any(m is None for m in records):
        raise ValueError(f"Invalid saved geometry: {path}")
    return records


def derive_input(job):
    raw, rows = job
    directory = Path(raw) / "checkpoints" / str(rows[0]["molecule_id"])
    checkpoint = json.loads((directory / "complete.json").read_text())
    refs, masks, pairs, output = {}, {}, {}, []
    for row in rows:
        r, c = int(row["reference_attempts"]), int(row["candidate_attempts"])
        if r not in refs:
            refs[r] = read_records(directory / f"reference_{r}_pool.sdf")
        key = (c, float(row["candidate_rmsd"]))
        if key not in masks:
            candidates = read_records(directory / f"candidate_{c}_pool.sdf")
            masks[key] = select_conformers(candidates, key[1] or None)
        if (r, c) not in pairs:
            with np.load(directory / f"pairs_r{r}_c{c}.npz") as saved:
                pairs[r, c] = saved["scores"], saved["seconds"]
        scores, times = pairs[r, c]
        indices = checkpoint["reference_selections"][str(r)][str(row["reference_rmsd"])]
        order = balanced_reference_order(refs[r], indices)
        ci = masks[key]
        if len(ci) != row["actual_candidates"]:
            raise ValueError("Reconstructed candidate mask differs from saved counts")
        for cap in REFERENCE_CAPS:
            ri = order[:cap] if cap else order
            best, winner = (0.0, 0.0, 0.0), (-1, -1)
            if ri and ci and not row["color_excluded"]:
                values = scores[np.ix_(ci, ri)]
                i, j = np.unravel_index(values[:, :, 0].argmax(), values.shape[:2])
                best, winner = values[i, j], (ci[i], ri[j])
            overlay = float(times[np.ix_(ci, ri)].sum())
            record = dict(
                row,
                reference_cap=cap,
                actual_references=len(ri),
                reference_failure=not ri,
                candidate_failure=not ci,
                winning_candidate=winner[0],
                winning_reference=winner[1],
                score=float(best[0]) if not row["color_excluded"] else np.nan,
                shape_tanimoto=float(best[1]) if not row["color_excluded"] else np.nan,
                color_tanimoto=float(best[2]) if not row["color_excluded"] else np.nan,
                overlay_seconds=overlay,
                seconds=row["seconds"] - row["overlay_seconds"] + overlay,
                covered_reference_identities=len(
                    {refs[r][i].GetProp("EnumeratedIsomerSMILES") for i in ri}
                ),
            )
            if cap == 0 and not np.isclose(
                record["score"], row["score"], atol=1e-8, rtol=0, equal_nan=True
            ):
                raise ValueError(
                    "All-reference replay differs from the original result"
                )
            output.append(record)
    return output


def render(summary, root, name, status=""):
    """Fix the shortlisted preparation budget to isolate retained-count effects."""
    selected = summary[
        summary.reference_attempts.eq(100) & summary.reference_rmsd.eq(0.5)
    ]
    columns = list(itertools.product((10, 20, 40), (0.0, 0.5, 1.25)))
    lookup = selected.set_index(
        ["reference_cap", "candidate_attempts", "candidate_rmsd"]
    )
    fig, axes = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True)
    metrics = [
        ("mean", "Mean normalized ROCS"),
        ("median", "Median normalized ROCS"),
        ("p05", "Fifth percentile ROCS"),
        ("fraction_ge_095", "Fraction reaching 0.95"),
        ("seconds", "Mean generation + minimization + pruning + alignment (s)"),
        ("mean_references", "Mean actual retained references"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = np.array(
            [
                [lookup.loc[(cap, c, pruning), metric] for c, pruning in columns]
                for cap in REFERENCE_CAPS
            ]
        )
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="viridis"
            if metric in {"mean", "median", "p05", "fraction_ge_095"}
            else "magma_r",
        )
        ax.set_yticks(
            range(5), ["at most 1", "at most 2", "at most 4", "at most 8", "all"]
        )
        ax.set_xticks(
            range(9),
            [
                f"C{c}\n{'off' if pruning == 0 else str(pruning) + ' Å'}"
                for c, pruning in columns
            ],
        )
        ax.set_title(title)
        ax.set_xlabel("Candidate requests per stereoisomer / additional RMSD pruning")
        ax.set_ylabel("Retained reference cap")
        for i, j in np.ndindex(values.shape):
            rgba = image.cmap(image.norm(values[i, j]))
            luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
            ax.text(
                j,
                i,
                f"{values[i, j]:.4f}" if metric == "mean" else f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.48 else "white",
                fontsize=9,
            )
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(
        f"{status}Retained references vs candidate search · {int(selected.inputs.iloc[0])} inputs ({int(selected.scored_inputs.iloc[0])} scored)\n"
        "Reference preparation fixed: 100 requests/stereoisomer, cap 8 identities, 0.5 Å pruning · candidate stereo cap 4\n"
        f"MMFF94s ε=78.4 · {int(selected.color_excluded.iloc[0])} color exclusions · {int(selected.reference_failures.iloc[0])} preparation failures score zero · shared component timing, not RL epochs",
        fontsize=14,
    )
    fig.savefig(root / f"{name}.png", dpi=160)
    fig.savefig(root / f"{name}.svg")
    plt.close(fig)


def run(root, workers, data=None, name="reference_caps", partial=False):
    raw = data or root / "raw_data"
    manifest = json.loads((raw / "manifest.json").read_text())
    if not partial and (
        manifest["state"] != "complete" or "validation" not in manifest
    ):
        raise ValueError("Finish and validate the original measurements first")
    pending = []
    if partial:
        panel = pd.read_csv(raw / "panel.csv")
        completed = []
        for molecule_id in panel.molecule_id:
            directory = raw / "checkpoints" / str(molecule_id)
            if not (directory / "complete.json").exists():
                pending.append(int(molecule_id))
                continue
            checkpoint = json.loads((directory / "complete.json").read_text())
            for filename, expected in checkpoint["files_sha256"].items():
                if digest(directory / filename) != expected:
                    raise ValueError(f"Changed checkpoint: {directory / filename}")
            completed.append(pd.read_csv(directory / "rows.csv"))
        original = pd.concat(completed, ignore_index=True)
        original.to_csv(
            raw / "provisional_measurements.csv.gz", index=False, compression="gzip"
        )
    else:
        original = pd.read_csv(raw / "data.csv.gz")
    jobs = [
        (str(raw), group.to_dict("records"))
        for _, group in original.groupby("molecule_id")
    ]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        frame = pd.DataFrame(
            [row for rows in pool.map(derive_input, jobs) for row in rows]
        )
    if (
        len(frame) != len(original) * len(REFERENCE_CAPS)
        or frame.duplicated([*KEYS, "reference_cap"]).any()
    ):
        raise ValueError("Incomplete or duplicated retained-reference grid")
    stem = "provisional_reference_caps" if partial else "reference_caps"
    frame.to_csv(raw / f"{stem}.csv.gz", index=False, compression="gzip")
    summary = pd.concat(
        [
            summarize(group).assign(reference_cap=cap)
            for cap, group in frame.groupby("reference_cap")
        ],
        ignore_index=True,
    )
    if partial:
        denominator = summary.scored_inputs + len(pending)
        summary["full_mean_lower"] = (
            summary["mean"] * summary.scored_inputs / denominator
        )
        summary["full_mean_upper"] = (
            summary["mean"] * summary.scored_inputs + len(pending) * 1.00001
        ) / denominator
    summary.to_csv(raw / f"{stem}_summary.csv", index=False)
    render(
        summary,
        root,
        name,
        f"PROVISIONAL ({len(pending)} pending) · " if partial else "",
    )
    analysis = dict(
        source_data_sha256=digest(
            raw / ("provisional_measurements.csv.gz" if partial else "data.csv.gz")
        ),
        script_sha256=digest(__file__),
        reference_caps=REFERENCE_CAPS,
        rows=len(frame),
        selection="Candidate-score-blind round-robin across stereo identities; preserve each identity's existing energy order. Cap means at most K; do not mask rigid inputs with fewer references.",
        timing="Reuse measured pair times; no new embedding, minimization or alignment. RMSD masks reconstructed from saved geometries.",
        data_sha256=digest(raw / f"{stem}.csv.gz"),
        pending_input_ids=pending,
        partial=partial,
        completed_inputs=len(jobs),
    )
    if partial:
        # The original coordinator may still be writing its own manifest.
        atomic_json(raw / "provisional_analysis.json", analysis)
    else:
        manifest["retained_reference_analysis"] = analysis
        atomic_json(raw / "manifest.json", manifest)
        reference_pruning_anchor(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--data",
        type=Path,
        help="Validated measurement directory; default: OUTPUT/raw_data",
    )
    parser.add_argument("--name", default="reference_caps", help="Figure filename stem")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Explicitly provisional snapshot; retain pending IDs and bound the full-panel mean",
    )
    args = parser.parse_args()
    run(args.output, args.workers, args.data, args.name, args.partial)
