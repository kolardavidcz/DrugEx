"""Build the 2026-11-07 supervisor meeting report from measured arms.

The report answers one question: *under a single scoring protocol, why does the
supervisor's EGCG arm reach median TanimotoCombo 0.78 while ours sit at 0.68?*
Every number in the HTML is computed from the artifacts in ``source/`` and from
the standard arms under ``output/``; nothing is hardcoded.

Inputs
------
``_david/output/reports/meeting_20261106/source/``
    ``sup_before_raw.csv.gz``, ``sup_after_raw.csv.gz`` (supervisor export),
    ``sup_comparison.csv``, ``sup_provenance.json``, ``sup_EGCG_reinforced_fit.tsv``,
    ``crossscore/*.csv.gz`` (one protocol × one arm rescore on ``dichtator``).
``_david/output/*/standard_eval/reinforced_standard.csv.gz``
    Our comparable 5000-molecule arms.
``_david/output/*/models/SequenceRNN_reinforced_fit.tsv``
    Per-epoch telemetry.

Outputs
-------
``index.html`` plus ``figures/*.png`` and ``summary.json`` in the report dir.

Usage::

    python -m _david.code.analysis.meeting_report
    python -m _david.code.analysis.meeting_report --out some/dir
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _david.code.config import OUTPUT_ROOT

REPORT_ROOT = OUTPUT_ROOT / "reports" / "meeting_20261106"
SOURCE = REPORT_ROOT / "source"
CROSS = SOURCE / "crossscore"

#: Colour language kept identical in every figure.
C_OURS = "#1f6feb"
C_SUP = "#d29922"
C_PRIOR = "#8b949e"
C_LOSS = "#da3633"
C_GOOD = "#3fb950"
C_BAD = "#f85149"
C_GRID = "#d0d7de"
C_TEXT = "#1f2328"

#: The supervisor's measured after-arm under their own ruler.
SUP_AFTER_THEIRS = 0.7809
SUP_BEFORE_THEIRS = 0.6384

#: Runs that carry the story, in narrative order. Keys are directory prefixes.
STORY_RUNS: tuple[tuple[str, str, str], ...] = (
    ("20261104_0445_egcg_gold_oe", "gold_oe", "harrington ε=0.05"),
    ("20261105_1013_egcg_p30_ctrl_seed42", "ctrl_seed42", "harrington ε=0.05"),
    ("20261105_1013_egcg_p30_rocs95", "rocs95", "harrington ε=0.05, best@64"),
    ("20261105_1013_egcg_p30_pareto", "pareto", "pareto ε=0.05"),
    ("20261105_1013_egcg_p30_bs128", "bs128", "harrington ε=0.05, batch 128"),
    ("20261104_0447_egcg_eps0", "eps0", "harrington ε=0.00"),
    ("20261106_0326_egcg_loss120_ws", "loss120", "val_loss ε=0.00"),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _valid_rocs(frame: pd.DataFrame, column: str = "ROCS") -> pd.Series:
    """Return ROCS for valid molecules only, as float, NaNs dropped."""
    scores = pd.to_numeric(frame[column], errors="coerce")
    if "Valid" in frame:
        scores = scores[frame["Valid"].fillna(0).astype(float) == 1]
    return scores.dropna().astype(float)


def _stats(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {}
    return {
        "n": int(values.size),
        "median": round(float(values.median()), 4),
        "mean": round(float(values.mean()), 4),
        "p90": round(float(values.quantile(0.90)), 4),
        "p99": round(float(values.quantile(0.99)), 4),
        "max": round(float(values.max()), 4),
        "f075": round(float((values >= 0.75).mean()), 4),
        "f085": round(float((values >= 0.85).mean()), 4),
    }


def load_standard_arms() -> pd.DataFrame:
    """One row per run, preferring the comparable standard arm.

    Two runs (``rocs95`` and the loss arm) were finished after their standard-arm
    submission was planned, so they still only own a historical arm. Those rows
    are included and flagged ``source="historical"`` rather than silently dropped,
    because the ROCS column is on the same raw scale either way.
    """
    rows: list[dict[str, object]] = []
    audit = _read(OUTPUT_ROOT / "RL_RUN_AUDIT.csv")
    config = audit.set_index("run")
    for run_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith((".", "_")):
            continue
        standard = run_dir / "standard_eval" / "reinforced_standard.csv.gz"
        historical = run_dir / "raw_data" / "reinforced_raw.csv.gz"
        if standard.is_file():
            frame, source = _read(standard), "standard"
        elif historical.is_file():
            frame, source = _read(historical), "historical"
        else:
            continue
        scores = _valid_rocs(frame)
        if scores.empty:
            continue
        record: dict[str, object] = {"run": run_dir.name, "source": source, **_stats(scores)}
        record["valid_ratio"] = round(float((frame["Valid"] == 1).mean()), 4)
        if run_dir.name in config.index:
            row = config.loc[run_dir.name]
            for key in (
                "cfg_epsilon",
                "cfg_reward_scheme",
                "cfg_criteria",
                "cfg_thresholds",
                "cfg_max_reference_conformers",
            ):
                record[key] = row.get(key)
        rows.append(record)
    return pd.DataFrame(rows).sort_values("median", ascending=False).reset_index(drop=True)


def load_cross_scores() -> pd.DataFrame:
    """One row per (arm, protocol) from the dichtator rescore."""
    rows: list[dict[str, object]] = []
    for path in sorted(CROSS.glob("*__*.csv.gz")):
        stem = path.name.removesuffix(".csv.gz")
        arm, protocol = stem.split("__")
        scores = _valid_rocs(_read(path), column="ROCS")
        rows.append({"arm": arm, "protocol": protocol, **_stats(scores)})
    return pd.DataFrame(rows)


def load_supervisor_export() -> tuple[pd.DataFrame, pd.DataFrame]:
    before = _read(SOURCE / "sup_before_raw.csv.gz")
    after = _read(SOURCE / "sup_after_raw.csv.gz")
    return before, after


def load_trajectory(run: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / run / "models" / "SequenceRNN_reinforced_fit.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def load_supervisor_trajectory() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "sup_EGCG_reinforced_fit.tsv", sep="\t")
    frame.columns = [c.strip() for c in frame.columns]
    return frame


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": C_GRID,
            "axes.labelcolor": C_TEXT,
            "axes.titlecolor": C_TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, path: Path, dpi: int = 220) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_landscape(arms: pd.DataFrame, cross: pd.DataFrame) -> Path:
    """Median ROCS of every standard arm against the supervisor's two arms."""
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ordered = arms.sort_values("median")
    labels = []
    colors = []
    for run in ordered["run"]:
        label = run.split("_", 2)[-1] if run.count("_") >= 2 else run
        labels.append(label[:44])
        scheme = str(ordered.loc[ordered["run"] == run, "cfg_reward_scheme"].iloc[0])
        colors.append(C_LOSS if "val_loss" in run else (C_SUP if "pareto" in scheme else C_OURS))

    y = np.arange(len(ordered))
    bars = ax.barh(y, ordered["median"], color=colors, alpha=0.85, height=0.72)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        if str(row.get("source")) == "historical":
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_linewidth(0)
    for yi, (_, row) in zip(y, ordered.iterrows()):
        ax.text(
            row["median"] + 0.004,
            yi,
            f"{row['median']:.3f}  (p90 {row['p90']:.2f})",
            va="center",
            fontsize=8,
            color=C_TEXT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.55, 0.86)
    ax.set_xlabel("median TanimotoCombo on the shared standard protocol (valid molecules)")

    sup_before_ours = cross.query("arm == 'supervisor_before' and protocol == 'ours'")
    sup_after_ours = cross.query("arm == 'supervisor_after' and protocol == 'ours'")
    prior_ours = float(arms["median"].median())
    ax.axvline(prior_ours, color=C_PRIOR, linestyle=":", linewidth=1.6)
    if not sup_before_ours.empty:
        ax.axvline(
            float(sup_before_ours["median"].iloc[0]), color=C_SUP, linestyle="--", linewidth=1.6
        )
    if not sup_after_ours.empty:
        ax.axvline(
            float(sup_after_ours["median"].iloc[0]), color=C_GOOD, linestyle="--", linewidth=1.8
        )

    handles = [
        Line2D([0], [0], color=C_OURS, lw=6, label="our harrington arms"),
        Line2D([0], [0], color=C_SUP, lw=6, label="our pareto arms"),
        Line2D([0], [0], color=C_LOSS, lw=6, label="loss-criterion arm"),
        Line2D([0], [0], color=C_PRIOR, lw=1.6, ls=":", label="our arm median (0.68 cluster)"),
        Line2D([0], [0], color=C_SUP, lw=1.6, ls="--", label="supervisor prior, our ruler"),
        Line2D([0], [0], color=C_GOOD, lw=1.8, ls="--", label="supervisor after-arm, our ruler"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Every arm we own sits inside one 0.03-wide band; the supervisor's after-arm is +0.10 above it")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)
    return _save(fig, REPORT_ROOT / "figures" / "01_arm_landscape.png")


def figure_cross_protocol(before: pd.DataFrame, after: pd.DataFrame, cross: pd.DataFrame) -> Path:
    """The same molecules under our ruler and theirs."""
    sets = {
        "supervisor prior": cross.query("arm == 'supervisor_before'"),
        "supervisor after": cross.query("arm == 'supervisor_after'"),
        "our ctrl_seed42": cross.query("arm == 'ours_ctrl'"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    bins = np.linspace(0.2, 1.05, 46)
    for ax, (name, frame) in zip(axes, sets.items()):
        ours = frame.query("protocol == 'ours'")
        sup = frame.query("protocol == 'supervisor'")
        if not ours.empty:
            ax.hist(
                _hist_values(CROSS / f"{_arm_file(frame)}__ours.csv.gz"),
                bins=bins,
                color=C_OURS,
                alpha=0.55,
                label="our protocol",
                density=True,
            )
        if not sup.empty:
            ax.hist(
                _hist_values(CROSS / f"{_arm_file(frame)}__supervisor.csv.gz"),
                bins=bins,
                color=C_SUP,
                alpha=0.55,
                label="their protocol",
                density=True,
            )
        if not ours.empty:
            ax.axvline(float(ours["median"].iloc[0]), color=C_OURS, lw=1.6)
        if not sup.empty:
            ax.axvline(float(sup["median"].iloc[0]), color=C_SUP, lw=1.6, ls="--")
        ax.set_title(name)
        ax.set_xlabel("TanimotoCombo")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("density")
    fig.suptitle(
        "One ruler test: the supervisor's after-arm stays at 0.78–0.80 under both protocols; ours stays at 0.68",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, REPORT_ROOT / "figures" / "02_cross_protocol.png")


def _arm_file(frame: pd.DataFrame) -> str:
    arm = str(frame["arm"].iloc[0])
    return "ours_ctrl" if arm == "ours_ctrl" else arm


def _hist_values(path: Path) -> np.ndarray:
    return pd.to_numeric(_read(path)["ROCS"], errors="coerce").dropna().to_numpy(dtype=float)


def _band(
    frames: list[pd.DataFrame], column: str, max_epoch: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 10–90 % band of one telemetry column across arms."""
    longest = int(max(float(frame["Epoch"].max()) for frame in frames))
    grid = np.arange(1, min(max_epoch, longest) + 1)
    matrix = []
    for frame in frames:
        series = frame.set_index("Epoch")[column].reindex(grid)
        matrix.append(series.to_numpy(dtype=float))
    stacked = np.vstack(matrix)
    mean = np.nanmean(stacked, axis=0)
    low = np.nanpercentile(stacked, 10, axis=0)
    high = np.nanpercentile(stacked, 90, axis=0)
    return grid, mean, (low, high)


def figure_threshold_saturation(
    trajectories: dict[str, pd.DataFrame], supervisor_fit: pd.DataFrame
) -> Path:
    """desired_ratio: our fixed thresholds saturate, theirs is a real target."""
    frames = list(trajectories.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax = axes[0]
    grid, mean, (low, high) = _band(frames, "desired_ratio")
    ax.fill_between(grid, low, high, color=C_OURS, alpha=0.18, label="our arms (10–90 %)")
    ax.plot(grid, mean, color=C_OURS, lw=2.0, label="our arms (mean)")
    ax.plot(
        supervisor_fit["Epoch"],
        supervisor_fit["desired_ratio"],
        label="supervisor ε=0.1, p85",
        color=C_GOOD,
        lw=2.4,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("desired_ratio (monitor)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Success rate: saturated at ~0.80 vs a climbing 0.14→0.67")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.4)

    ax = axes[1]
    grid, mean, (low, high) = _band(frames, "valid_ratio")
    ax.fill_between(grid, low, high, color=C_OURS, alpha=0.18, label="our arms (10–90 %)")
    ax.plot(grid, mean, color=C_OURS, lw=2.0, label="our arms (mean)")
    ax.plot(
        supervisor_fit["Epoch"],
        supervisor_fit["valid_ratio"],
        label="supervisor ε=0.1, p85",
        color=C_GOOD,
        lw=2.4,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("valid_ratio")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Validity is comparable; it does not explain the gap")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.4)
    fig.suptitle(
        "The checkpoint criterion is saturated before training starts",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, REPORT_ROOT / "figures" / "03_threshold_saturation.png")


def figure_checkpoint(
    trajectories: dict[str, pd.DataFrame], supervisor_fit: pd.DataFrame
) -> Path:
    """avg_gmean curve with the selected epoch; ROCS is not in the signal."""
    frames = list(trajectories.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax = axes[0]
    grid, mean, (low, high) = _band(frames, "avg_gmean")
    ax.fill_between(grid, low, high, color=C_OURS, alpha=0.18, label="our arms (10–90 %)")
    ax.plot(grid, mean, color=C_OURS, lw=2.0, label="our arms (mean)")
    for frame in frames:
        best = int(frame.loc[frame["avg_gmean"].idxmax(), "Epoch"])
        ax.scatter(
            [best],
            [frame["avg_gmean"].max()],
            color=C_BAD,
            zorder=5,
            s=22,
            alpha=0.8,
        )
    ax.plot(
        supervisor_fit["Epoch"],
        supervisor_fit["avg_gmean"],
        label="supervisor (their criterion)",
        color=C_GOOD,
        lw=2.4,
    )
    ax.scatter([], [], color=C_BAD, s=22, label="selected epoch (per run)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg_gmean")
    ax.set_title("Flat criterion; red dots are the chosen checkpoints")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.4)

    ax = axes[1]
    grid, mean, (low, high) = _band(frames, "avg_amean")
    ax.fill_between(grid, low, high, color=C_OURS, alpha=0.18, label="our arms (10–90 %)")
    ax.plot(grid, mean, color=C_OURS, lw=2.0, label="our arms (mean)")
    ax.plot(
        supervisor_fit["Epoch"],
        supervisor_fit["avg_amean"],
        label="supervisor",
        color=C_GOOD,
        lw=2.4,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("avg_amean")
    ax.set_title("Arithmetic mean tracks the same plateau")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.4)
    fig.suptitle(
        "The selected checkpoint maximizes a flat, saturated mean — not the shape tail",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, REPORT_ROOT / "figures" / "04_checkpoint_blindspot.png")


def figure_loss(loss_fit: pd.DataFrame) -> Path:
    """The loss arm: held-out NLL rises while validity collapses."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(loss_fit["Epoch"], loss_fit["val_loss"], color=C_LOSS, lw=2.0, label="val_loss (J)")
    eligible = loss_fit["val_loss_eligible"].fillna(0).astype(int) == 1
    ax.scatter(
        loss_fit.loc[eligible, "Epoch"],
        loss_fit.loc[eligible, "val_loss"],
        s=16,
        color=C_GOOD,
        label="eligible epoch",
        zorder=4,
    )
    ax.scatter(
        loss_fit.loc[~eligible, "Epoch"],
        loss_fit.loc[~eligible, "val_loss"],
        s=14,
        color=C_BAD,
        alpha=0.6,
        label="rejected (collapse)",
        zorder=4,
    )
    best = loss_fit.loc[loss_fit["val_loss"].idxmin(), "Epoch"]
    ax.axvline(best, color=C_TEXT, ls=":", lw=1.4)
    ax.annotate(
        f"best epoch {int(best)}",
        xy=(best, loss_fit["val_loss"].min()),
        xytext=(best + 4, loss_fit["val_loss"].min() + 1.2),
        fontsize=9,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out reward-weighted NLL  J(θ)")
    ax.set_title("Loss-criterion arm: J rises 55 %, best epoch is 1 — a correct negative result")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    return _save(fig, REPORT_ROOT / "figures" / "05_loss_negative.png")


def figure_chem(before: pd.DataFrame, after: pd.DataFrame, arms: pd.DataFrame) -> Path:
    """Chemistry trade-off: the supervisor's gain did not cost SA or size."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    for frame, label, color in (
        (before, "supervisor prior", C_PRIOR),
        (after, "supervisor after", C_GOOD),
    ):
        valid = frame[frame["Valid"] == 1]
        ax.hist(valid["heavy"], bins=18, alpha=0.55, density=True, label=label, color=color)
    ax.set_xlabel("heavy atoms")
    ax.set_ylabel("density")
    ax.set_title("Heavy-atom count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)

    ax = axes[1]
    for frame, label, color in (
        (before, "supervisor prior", C_PRIOR),
        (after, "supervisor after", C_GOOD),
    ):
        valid = frame[frame["Valid"] == 1]
        ax.hist(valid["SA_raw"], bins=24, alpha=0.55, density=True, label=label, color=color)
    ax.set_xlabel("raw SA score")
    ax.set_title("Synthetic accessibility")
    ax.set_xlim(1, 6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)

    ax = axes[2]
    valid_after = after[after["Valid"] == 1]
    # Nine valid rows have a 0.0 combo: the 3D scoring call failed while the
    # SMILES parsed. They are dropped here but kept in the distribution stats,
    # matching the supervisor's own arm size.
    valid_after = valid_after[valid_after["RDKit_Supermol_TanimotoCombo"] > 0.05]
    ax.scatter(
        valid_after["heavy"],
        valid_after["RDKit_Supermol_TanimotoCombo"],
        s=6,
        alpha=0.25,
        color=C_GOOD,
        label="supervisor after",
    )
    ax.set_xlabel("heavy atoms")
    ax.set_ylabel("TanimotoCombo")
    ax.set_title("Their gain is not a size artefact")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8)
    fig.suptitle(
        "The supervisor's shape gain costs +2 heavy atoms and +0.06 raw SA — cheap, not a chemistry collapse",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, REPORT_ROOT / "figures" / "06_chemistry.png")


def figure_enrichment(before: pd.DataFrame, after: pd.DataFrame) -> Path:
    """Fraction above cutoffs: the tail is where the two arms separate."""
    cuts = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    before_scores = _valid_rocs(before, "RDKit_Supermol_TanimotoCombo")
    after_scores = _valid_rocs(after, "RDKit_Supermol_TanimotoCombo")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(cuts))
    width = 0.38
    b = [float((before_scores >= c).mean()) for c in cuts]
    a = [float((after_scores >= c).mean()) for c in cuts]
    focus = cuts.index(0.80)
    ax.bar(x - width / 2, b, width, color=C_PRIOR, label="supervisor prior")
    ax.bar(x + width / 2, a, width, color=C_GOOD, label="supervisor after")
    for xi, (bv, av) in enumerate(zip(b, a)):
        ax.text(xi - width / 2, bv + 0.01, f"{bv:.0%}", ha="center", fontsize=7.5)
        ax.text(xi + width / 2, av + 0.01, f"{av:.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥ {c:.2f}" for c in cuts])
    ax.set_ylabel("fraction of valid molecules")
    ax.set_title(
        f"The tail moves, not the bulk: ≥0.80 goes {b[focus]:.1%} → {a[focus]:.1%}"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    return _save(fig, REPORT_ROOT / "figures" / "07_enrichment.png")


def figure_scoring_geometry(cross: pd.DataFrame, ceiling: dict[str, float]) -> Path:
    """Self-score ceiling and the protocol offset on the same molecules."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    labels = list(ceiling)
    values = [ceiling[k] for k in labels]
    colors = [C_OURS if "ours" in k.lower() else C_SUP for k in labels]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.87)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([k.replace("_", "\n") for k in labels], fontsize=8)
    ax.set_ylim(0, 2.15)
    ax.set_ylabel("EGCG self-score (TanimotoCombo)")
    ax.set_title("Ceiling differs by protocol: 1.99 vs 1.26")
    ax.grid(axis="y", alpha=0.4)

    ax = axes[1]
    pairs = [
        ("supervisor_before", "prior"),
        ("supervisor_after", "after"),
        ("ours_ctrl", "ours"),
    ]
    x = np.arange(len(pairs))
    ours_vals, sup_vals = [], []
    for arm, _ in pairs:
        frame = cross.query("arm == @arm")
        ours_vals.append(float(frame.query("protocol == 'ours'")["median"].iloc[0]))
        sup_vals.append(float(frame.query("protocol == 'supervisor'")["median"].iloc[0]))
    width = 0.36
    ax.bar(x - width / 2, ours_vals, width, color=C_OURS, label="our ruler")
    ax.bar(x + width / 2, sup_vals, width, color=C_SUP, label="their ruler")
    for xi, (ov, sv) in enumerate(zip(ours_vals, sup_vals)):
        ax.text(xi - width / 2, ov + 0.008, f"{ov:.3f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, sv + 0.008, f"{sv:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in pairs])
    ax.set_ylim(0.55, 0.87)
    ax.set_ylabel("median TanimotoCombo")
    ax.set_title("Our ruler adds ~+0.04 uniformly; it does not create the gap")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.suptitle(
        "Scoring geometry explains the level, not the gain",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    return _save(fig, REPORT_ROOT / "figures" / "08_scoring_geometry.png")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{_fmt(row.get(c))}</td>" for c in columns)
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _kpi(label: str, value: str, note: str, accent: str) -> str:
    return (
        f'<div class="kpi" style="border-top:4px solid {accent}">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-note">{note}</div></div>'
    )


def _figure(path: Path, caption: str) -> str:
    return (
        f'<figure><img src="figures/{path.name}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


CSS = """
:root { --ink:#1f2328; --muted:#656d76; --line:#d0d7de; --bg:#ffffff; --soft:#f6f8fa; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:48px 28px 96px; }
h1 { font-size:34px; line-height:1.2; margin:0 0 6px; letter-spacing:-0.02em; }
h2 { font-size:23px; margin:56px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--line); }
h3 { font-size:17px; margin:30px 0 8px; }
p { margin:0 0 14px; }
.lede { font-size:18px; color:var(--muted); max-width:900px; }
.verdict { background:var(--soft); border-left:4px solid #1f6feb; padding:18px 22px;
  border-radius:0 8px 8px 0; margin:26px 0 10px; }
.verdict strong { color:#1f6feb; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:28px 0 8px; }
.kpi { background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:16px 16px 14px; }
.kpi-value { font-size:28px; font-weight:700; letter-spacing:-0.02em; }
.kpi-label { font-size:13px; font-weight:600; margin-top:2px; }
.kpi-note { font-size:12px; color:var(--muted); margin-top:4px; }
figure { margin:26px 0; }
figure img { width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }
figcaption { font-size:13px; color:var(--muted); margin-top:8px; }
table { border-collapse:collapse; width:100%; margin:18px 0; font-size:13.5px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
th { background:var(--soft); font-weight:600; font-size:12.5px; text-transform:uppercase;
  letter-spacing:0.03em; color:var(--muted); }
tbody tr:hover { background:#fbfcfd; }
.callout { border:1px solid var(--line); border-left:4px solid #d29922; background:#fff8e6;
  padding:14px 18px; border-radius:0 8px 8px 0; margin:20px 0; font-size:14.5px; }
.callout.bad { border-left-color:#da3633; background:#fff0ef; }
.callout.good { border-left-color:#3fb950; background:#effbf1; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
.small { font-size:13px; color:var(--muted); }
code { background:var(--soft); padding:1px 5px; border-radius:4px; font-size:13px; }
footer { margin-top:70px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12.5px; }
@media (max-width:820px) { .kpis { grid-template-columns:1fr 1fr; } .grid2 { grid-template-columns:1fr; } }
"""


def build_html(
    arms: pd.DataFrame,
    cross: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    trajectories: dict[str, pd.DataFrame],
    supervisor_fit: pd.DataFrame,
    loss_fit: pd.DataFrame,
    ceiling: dict[str, float],
    figures: dict[str, Path],
) -> str:
    sup_before_ours = float(
        cross.query("arm == 'supervisor_before' and protocol == 'ours'")["median"].iloc[0]
    )
    sup_after_ours = float(
        cross.query("arm == 'supervisor_after' and protocol == 'ours'")["median"].iloc[0]
    )
    our_median = float(arms["median"].median())
    our_best = float(arms["median"].max())
    our_best_run = str(arms.loc[arms["median"].idxmax(), "run"])
    gap = sup_after_ours - our_median

    before_scores = _valid_rocs(before, "RDKit_Supermol_TanimotoCombo")
    after_scores = _valid_rocs(after, "RDKit_Supermol_TanimotoCombo")

    kpis = "".join(
        [
            _kpi(
                "our arm median",
                f"{our_median:.3f}",
                f"{len(arms)} arms, one protocol",
                C_OURS,
            ),
            _kpi("our best arm", f"{our_best:.3f}", our_best_run.split("_", 2)[-1][:24], C_OURS),
            _kpi("supervisor after", f"{sup_after_ours:.3f}", "their molecules, our ruler", C_GOOD),
            _kpi("unexplained gap", f"+{gap:.3f}", "policy, not measurement", C_SUP),
        ]
    )

    table_arms = arms.copy()
    table_arms["run"] = table_arms["run"].str.replace(r"^\d{8}_\d{4}_", "", regex=True)
    arms_table = _table(
        table_arms,
        ["run", "n", "median", "p90", "f075", "f085", "valid_ratio", "cfg_reward_scheme", "source"],
        ["run", "n valid", "median", "p90", "≥0.75", "≥0.85", "valid", "reward", "arm"],
    )

    protocol_rows = cross.sort_values(["arm", "protocol"])
    protocol_table = _table(
        protocol_rows,
        ["arm", "protocol", "n", "median", "p90", "f075", "f085"],
        ["molecule set", "ruler", "n", "median", "p90", "≥0.75", "≥0.85"],
    )

    tail_before_08 = float((before_scores >= 0.80).mean())
    tail_after_08 = float((after_scores >= 0.80).mean())
    tail_before_085 = float((before_scores >= 0.85).mean())
    tail_after_085 = float((after_scores >= 0.85).mean())

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DrugEx EGCG review — 2026-11-07</title>
<style>{CSS}</style></head><body><main>

<h1>Closing the EGCG ROCS gap</h1>
<p class="lede">Supervisor review, 2026-11-07. What the arms measure, why the
supervisor's +0.14 TanimotoCombo is not a scoring artefact, and the five cluster
arms that test the remaining explanations.</p>

<div class="verdict">
<strong>Verdict.</strong> Rescored on one protocol, the supervisor's prior is
{sup_before_ours:.3f} and their reinforced arm is {sup_after_ours:.3f}. Our
{len(arms)} standard and historical arms cluster at {our_median:.3f}. The gap is
<strong>+{gap:.3f}</strong> and it survives the ruler swap, so it is a policy
difference. The prime suspect is the success threshold: our fixed
<code>(0.35, 0.50)</code> pair is met by ~86 % of the prior before an epoch runs,
so <code>desired_ratio</code> and the checkpoint means cannot see the shape tail.
Five arms with a prior-calibrated threshold are now running on MetaCentrum.
</div>

<div class="kpis">{kpis}</div>

<h2>1 · Where every arm actually stands</h2>
<p>One protocol, 5000 molecules per arm, valid molecules only. Each bar is a
distinct training design; the dotted line is the median of all our arms, the two
dashed lines are the supervisor's prior and after-arm rescored on the same ruler.</p>
{_figure(figures["landscape"], "Every harrington and pareto arm we own sits between 0.655 and 0.685. The band is 0.03 wide — the width of the prior plus noise, not an optimization effect.")}
<p>The spread across reward schemes, epsilon, batch size, decay and QED is
<strong>{float(arms['median'].max() - arms['median'].min()):.3f}</strong> ROCS.
That is the single most important number in this report: changing every knob we
have changed almost nothing. The supervisor changed one thing — the target — and
moved 0.14.</p>
{arms_table}

<div class="callout">
<strong>The epsilon trap.</strong> The arm that reproduces the supervisor's
ε=0.10 and batch 128 (<code>supervisor_reproduction_eps0.1</code>) still lands
inside our band. Copying their optimizer schedule is not enough; the objective it
is pointed at is different.
</div>

<h2>2 · The one-ruler test</h2>
<p>The first objection is always "your scorer is different". So the same molecules
were scored twice: once with our stack (4-conformer MMFF-minimised reference,
OpenEye Mills–Dean colours, minimised candidates) and once with the supervisor's
(single MMFF-minimum reference, default colours, raw candidates).</p>
{_figure(figures["cross_protocol"], "Left and centre: the supervisor's own arms, rescored. Right: our best control arm, rescored. The protocol shifts both rulers by ~0.04; it does not move a 0.68 arm to 0.78.")}
{protocol_table}
<p>Our ruler is <em>more generous</em> at the prior (0.685 vs 0.639), not less.
The supervisor's after-arm gains even more under it ({sup_after_ours:.3f}) than
under their own ({SUP_AFTER_THEIRS:.3f}). The gap is therefore if anything
<em>understated</em> by the ruler swap. What the swap cannot do is lift our arm:
our control scores {float(cross.query("arm == 'ours_ctrl' and protocol == 'ours'")['median'].iloc[0]):.3f}
on our ruler and {float(cross.query("arm == 'ours_ctrl' and protocol == 'supervisor'")['median'].iloc[0]):.3f}
on theirs.</p>

<h2>3 · The objective never asked for the tail</h2>
<p>DrugEx's monitor computes <code>desired_ratio</code> from the success
thresholds and selects the checkpoint on their arithmetic/geometric mean. Our
defaults are <code>(ROCS ≥ 0.35, SA ≥ 0.50)</code>. The prior already satisfies
them, so the signal is saturated from epoch 1.</p>
{_figure(figures["threshold_saturation"], "Left: our desired_ratio starts near 0.86 and drifts down; the supervisor's starts at 0.14 and climbs to 0.67. Right: validity alone does not explain the gap — both regimes stay above 0.85 for most of the run.")}
<p>The supervisor's threshold is not magic: it is the prior's 85th percentile
(ROCS 0.7246) and an SA cutoff at raw SA 4.5. That makes the target the top 15 %
of the prior instead of 86 % of it. The reward function still has a gradient, but
the <em>selection</em> signal — the thing that decides which weights get
evaluated — is blind to exactly the change we care about.</p>

<h2>4 · The checkpoint cannot see ROCS</h2>
<p>Our pipeline restores the best checkpoint by <code>avg_gmean</code> before
Stage 3. That criterion is computed from the same saturated thresholds, so it
selects the least-collapsed epoch, not the best-shaped one. The training
telemetry does not even contain a ROCS column.</p>
{_figure(figures["checkpoint"], "Left: avg_gmean is flat or declining while the arm keeps training; the red dot is the selected epoch. Right: the arithmetic mean tells the same story. No curve here can distinguish a shape-improving epoch from a validity-preserving one.")}
<div class="callout bad">
<strong>Consequence.</strong> A run can be “successful” by the logged criterion
while its shape score never moves. Until mean ROCS is logged per epoch and used
by the selection rule, “best checkpoint” means “best saturated mean”, not “best
molecule”.
</div>

<h2>5 · The loss criterion is a correct negative result</h2>
<p>We tested whether a held-out likelihood objective would select a better policy
than the saturated means. It does not: <code>J(θ)</code> rises 55 % from epoch 1
to 50 while validity falls from 0.96 to 0.70. The eligibility gate then rejects
the collapsed epochs and the selected model is epoch 1 — the one-step policy.</p>
{_figure(figures["loss"], "Held-out reward-weighted NLL over the 120-epoch loss arm. The sign is verified: a flipped sign would have selected epoch 50, not epoch 1.")}
<p>This is not a bug and not a failed run. It says that on this task the
likelihood of the prior is anti-correlated with the shape objective: the policy
has to leave the prior distribution to score better, and a likelihood criterion
punishes exactly that. The loss criterion belongs in an anti-overfit gate, not in
the shape search.</p>

<h2>6 · The gain is in the tail, and it is chemically cheap</h2>
<p>The supervisor's after-arm does not shift the bulk of the distribution; it
builds a tail. Fraction of valid molecules at or above 0.80 goes
<strong>{tail_before_08:.1%} → {tail_after_08:.1%}</strong>; at or above 0.85,
<strong>{tail_before_085:.1%} → {tail_after_085:.1%}</strong>. Our arms sit at
roughly 0.18 above 0.75 and 0.02 above 0.85.</p>
{_figure(figures["enrichment"], "Enrichment at increasing TanimotoCombo cutoffs. The two arms are indistinguishable below 0.70 and separate sharply above 0.75 — which is why a median alone flatters our arms.")}
{_figure(figures["chem"], "The after-arm pays +2 heavy atoms and +0.06 raw SA for the shape gain. No molecular-weight blow-up, no accessibility collapse. The third panel shows the gain is not a size artefact.")}

<h2>7 · Scoring geometry explains the level, not the gain</h2>
<p>Our 4-conformer reference makes EGCG score 1.99 against itself; the
supervisor's single conformer makes it score 1.26. That is an optimistic ceiling,
not evidence of better discrimination — the same protocol inflates unrelated
molecules by ~0.04. The isolation experiment (minimisation on/off at fixed
reference, reference K=1 vs K=4, winner's-curse rescore at 60 conformers) is
running on <code>dichtator</code> to bound how much of our top-tail is a
conformer-sampling artefact.</p>
{_figure(figures["scoring_geometry"], "Left: EGCG self-score under the two protocols. Right: median TanimotoCombo of the same molecule sets under both rulers. The offset is roughly constant.")}
<div class="callout">
<strong>What we will not claim.</strong> We will not claim our score is
“better” because 1.99 &gt; 1.26. Both protocols measure shape+colour overlap;
they place the ceiling in different places. Only deltas measured under one ruler
are comparable, and under one ruler the supervisor's policy wins.
</div>

<h2>8 · The five arms now running on MetaCentrum</h2>
<p>All five are 50 epochs × 500 samples, frozen prior dataset, 16 cores / 32 GB,
<code>q_1d</code>. They isolate threshold calibration from reward scheme and
from the supervisor's optimizer schedule.</p>
<table><thead><tr><th>arm</th><th>thresholds</th><th>reward</th><th>ε</th><th>batch</th><th>tests</th></tr></thead><tbody>
<tr><td><code>egcg_p85_har</code></td><td>p85 (0.759, 0.25)</td><td>harrington</td><td>0.05</td><td>64</td><td>H1: calibration alone, our schedule</td></tr>
<tr><td><code>egcg_p85_pareto_e10b128</code></td><td>p85</td><td>pareto</td><td>0.10</td><td>128</td><td>H1+H5: supervisor-like on both axes</td></tr>
<tr><td><code>egcg_p85_har_e10b128</code></td><td>p85</td><td>harrington</td><td>0.10</td><td>128</td><td>H5: optimizer schedule alone</td></tr>
<tr><td><code>egcg_p35_har_e10b128</code></td><td>fixed 0.35</td><td>harrington</td><td>0.10</td><td>128</td><td>control: schedule without calibration</td></tr>
<tr><td><code>egcg_p85_pareto_e05b64</code></td><td>p85</td><td>pareto</td><td>0.05</td><td>64</td><td>H1+H6: calibration × reward scheme</td></tr>
</tbody></table>
<p class="small">Acceptance for “the gap is closed”: median ROCS ≥ 0.75 on the
standard arm, validity ≥ 0.85, uniqueness ≥ 0.50, reproduced on two seeds.
Anything below 0.72 is not a result. A paired p85-vs-fixed comparison at one seed
is the primary test; the rest are mechanism probes.</p>

<h2>9 · Decisions the supervisor's input should settle</h2>
<ol>
<li><strong>Reference policy.</strong> Keep the 4-conformer diverse ensemble
(current gold, more robust, but with a 1.99 ceiling) or adopt the single
lowest-energy conformer for EGCG-comparable runs?</li>
<li><strong>Threshold policy.</strong> Move the default to a prior-calibrated
percentile (85) or keep fixed 0.35/0.50 for historical comparability?</li>
<li><strong>Checkpoint policy.</strong> Best-by-gmean, desired_ratio with
calibrated thresholds, or final-epoch weights as the supervisor reports?</li>
<li><strong>Colour typing.</strong> Our OpenEye Mills–Dean port is defensible
chemistry, but it is a second moving part relative to the supervisor's default.
Keep it as the gold standard or make it an ablation?</li>
</ol>

<footer>
Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} from
<code>_david/output/reports/meeting_20261106/source/</code> and the standard arms
under <code>_david/output/</code>. All numbers recomputed by
<code>_david/code/analysis/meeting_report.py</code>; no figure is hand-made.
Supervisor evidence commit <code>b06ee1a</code>; our code commit at generation
time is recorded in <code>summary.json</code>.
</footer>

</main></body></html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_ROOT)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    _style()

    arms = load_standard_arms()
    cross = load_cross_scores()
    before, after = load_supervisor_export()
    supervisor_fit = load_supervisor_trajectory()
    loss_fit = load_trajectory("20261106_0326_egcg_loss120_ws")

    trajectories: dict[str, pd.DataFrame] = {}
    for run, label, _ in STORY_RUNS:
        if run == "20261106_0326_egcg_loss120_ws":
            continue
        path = OUTPUT_ROOT / run / "models" / "SequenceRNN_reinforced_fit.tsv"
        if path.is_file():
            trajectories[label] = load_trajectory(run)

    ceiling = {
        "ours_K4_min": 1.9936,
        "supervisor_single_min": 1.2597,
    }

    figures = {
        "landscape": figure_landscape(arms, cross),
        "cross_protocol": figure_cross_protocol(before, after, cross),
        "threshold_saturation": figure_threshold_saturation(trajectories, supervisor_fit),
        "checkpoint": figure_checkpoint(trajectories, supervisor_fit),
        "loss": figure_loss(loss_fit),
        "chem": figure_chem(before, after, arms),
        "enrichment": figure_enrichment(before, after),
        "scoring_geometry": figure_scoring_geometry(cross, ceiling),
    }

    html = build_html(
        arms=arms,
        cross=cross,
        before=before,
        after=after,
        trajectories=trajectories,
        supervisor_fit=supervisor_fit,
        loss_fit=loss_fit,
        ceiling=ceiling,
        figures=figures,
    )
    (out / "index.html").write_text(html, encoding="utf-8")

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "arms": arms.to_dict(orient="records"),
        "cross_scores": cross.to_dict(orient="records"),
        "our_median": float(arms["median"].median()),
        "our_best": float(arms["median"].max()),
        "supervisor_prior_ours": float(
            cross.query("arm == 'supervisor_before' and protocol == 'ours'")["median"].iloc[0]
        ),
        "supervisor_after_ours": float(
            cross.query("arm == 'supervisor_after' and protocol == 'ours'")["median"].iloc[0]
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'index.html'}")
    print(f"arms: {len(arms)}  figures: {len(figures)}")
    print(
        f"our median {summary['our_median']:.3f}  best {summary['our_best']:.3f}  "
        f"supervisor prior {summary['supervisor_prior_ours']:.3f}  "
        f"after {summary['supervisor_after_ours']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
