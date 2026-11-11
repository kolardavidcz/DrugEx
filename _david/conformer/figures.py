"""Figures for the self-similarity study.

Four sheets, in the order a reader needs them:

- ``01_omnibus.png``   the headline: score against candidate budget at a fixed bank, and
                       against bank size at a fixed candidate budget, one line per method.
- ``02_pareto.png``    the trade-off: score against measured cost, non-dominated cells ringed.
- ``03_grids.png``     every cell, as a score heatmap and a seconds heatmap per method.
- ``04_spread.png``    what the ten seeds disagree about, which is the only honest error bar.

Every panel plots ``self_recovery`` = TanimotoCombo / 2 on [0, 1], the scale the earlier
study's figures used; the raw [0, 2] combo is in ``measurements.tsv``. Cost is CPU
seconds, not wall seconds: cells are measured in parallel, so wall time carries the load
of whatever else was running and CPU time does not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.conformer.config import StudyConfig

logger = logging.getLogger(__name__)

#: One stable colour per method, so a method keeps its colour across all four sheets.
COLOURS = ("#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0891B2", "#DB2777")

#: The cost column every figure reads. Wall seconds are kept in the table for reference.
COST = "cell_cpu_seconds"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "font.size": 10,
        }
    )


def _palette(methods: list[str]) -> dict[str, str]:
    return {name: COLOURS[i % len(COLOURS)] for i, name in enumerate(methods)}


def _subtitle(cfg: StudyConfig, frame: pd.DataFrame) -> str:
    """What produced these numbers, on every sheet, so a loose PNG still describes itself."""
    return (
        f"{cfg.name} · {frame['seed'].nunique()} seeds · colour_ff={cfg.color_ff} · "
        f"shape_weight={cfg.alignment_shape_weight:g} · reference pool {cfg.reference_attempts} "
        f"embedded, RMSD>{cfg.reference_rmsd:g} · isomer cap {cfg.max_isomers} · "
        f"self-recovery, not bound-pose accuracy"
    )


def _agg(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Median over seeds, with the seed spread kept beside it."""
    grouped = frame.groupby(by, as_index=False).agg(
        self_recovery=("self_recovery", "median"),
        low=("self_recovery", lambda s: s.quantile(0.05)),
        high=("self_recovery", lambda s: s.quantile(0.95)),
        worst=("self_recovery", "min"),
        cost=(COST, "median"),
        bank_size=("bank_size", "median"),
        n=("self_recovery", "size"),
    )
    return grouped


def omnibus(frame: pd.DataFrame, cfg: StudyConfig, out: Path) -> Path | None:
    """Score against C at a fixed bank, and against K at a fixed candidate budget."""
    style()
    methods = sorted(frame["method"].unique())
    palette = _palette(methods)
    caps = sorted(frame["reference_cap"].unique())
    counts = sorted(frame["candidate_count"].unique())
    # The slice each panel holds fixed: the bank size nearest 4 and the budget nearest 20,
    # which are the pipeline's own defaults, rather than numbers invented for the figure.
    fixed_cap = min(caps, key=lambda k: abs(k - 4))
    fixed_count = min(counts, key=lambda c: abs(c - 20))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
    fig.suptitle(
        f"Conformer self-recovery of {cfg.name}\n{_subtitle(cfg, frame)}",
        fontsize=11,
        fontweight="bold",
    )

    panels = (
        (
            axes[0],
            "candidate_count",
            frame[frame["reference_cap"] == fixed_cap],
            f"A. Self-recovery vs candidate budget C (K={fixed_cap})",
            "Candidate embedding attempts per stereoisomer (C)",
        ),
        (
            axes[1],
            "reference_cap",
            frame[frame["candidate_count"] == fixed_count],
            f"B. Self-recovery vs reference bank cap K (C={fixed_count})",
            "Reference bank cap (K)",
        ),
    )
    for ax, axis, slice_, title, xlabel in panels:
        for name in methods:
            rows = _agg(slice_[slice_["method"] == name], [axis]).sort_values(axis)
            if rows.empty:
                continue
            ax.plot(
                rows[axis],
                rows["self_recovery"],
                "-o",
                color=palette[name],
                label=name,
                linewidth=1.8,
                markersize=5,
            )
            # The band is the 5th-95th percentile across seeds, not a standard error: with
            # ten seeds the spread is the measurement, and a symmetric bar would hide that
            # the failures are one-sided.
            ax.fill_between(rows[axis], rows["low"], rows["high"], color=palette[name], alpha=0.12)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = out / "01_omnibus.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _nondominated(cost: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Cells no other cell beats on both cost (lower) and score (higher)."""
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        if not keep[i]:
            continue
        beaten = (cost <= cost[i]) & (value >= value[i]) & ((cost < cost[i]) | (value > value[i]))
        if beaten.any():
            keep[i] = False
    return keep


def pareto(frame: pd.DataFrame, cfg: StudyConfig, out: Path) -> Path | None:
    """Score against measured cost, with the non-dominated cells ringed and labelled."""
    style()
    rows = _agg(frame, ["method", "reference_cap", "candidate_count"])
    if rows.empty:
        return None
    methods = sorted(rows["method"].unique())
    palette = _palette(methods)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=200)
    fig.suptitle(
        f"Cost of self-recovery · lower time and higher score preferred\n{_subtitle(cfg, frame)}",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    front = _nondominated(rows["cost"].to_numpy(), rows["self_recovery"].to_numpy())
    for name in methods:
        part = rows[rows["method"] == name]
        ax.scatter(
            part["cost"], part["self_recovery"], color=palette[name], label=name, s=28, alpha=0.8
        )
    ringed = rows[front]
    ax.scatter(
        ringed["cost"],
        ringed["self_recovery"],
        facecolors="none",
        edgecolors="#111827",
        s=130,
        linewidths=1.2,
        label="non-dominated",
    )
    for _, row in ringed.iterrows():
        ax.annotate(
            f"K{int(row['reference_cap'])}/C{int(row['candidate_count'])}",
            (row["cost"], row["self_recovery"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Median CPU seconds per cell (candidates + scoring)")
    ax.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
    ax.set_title("Measured score-cost frontier", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # The right panel is the one that matters for a reward: not the median molecule but
    # the worst seed, because a reward that collapses on one run in ten is not usable.
    ax2 = axes[1]
    for name in methods:
        part = rows[rows["method"] == name]
        ax2.scatter(part["cost"], part["worst"], color=palette[name], label=name, s=28, alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Median CPU seconds per cell (candidates + scoring)")
    ax2.set_ylabel("Worst of the seeds  [0, 1]")
    ax2.set_title(
        "Worst seed, not the median: one bad run in ten is the risk", fontsize=11, fontweight="bold"
    )
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = out / "02_pareto.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def grids(frame: pd.DataFrame, cfg: StudyConfig, out: Path) -> Path | None:
    """Every cell: a score heatmap and a cost heatmap per method, K across, C down."""
    style()
    methods = sorted(frame["method"].unique())
    caps = sorted(frame["reference_cap"].unique())
    counts = sorted(frame["candidate_count"].unique())
    if not methods:
        return None

    fig, axes = plt.subplots(
        len(methods), 2, figsize=(13, 3.1 * len(methods)), dpi=200, squeeze=False
    )
    fig.suptitle(
        f"Median self-recovery and median CPU seconds, every cell\n{_subtitle(cfg, frame)}",
        fontsize=11,
        fontweight="bold",
    )
    rows = _agg(frame, ["method", "reference_cap", "candidate_count"])
    # One score scale across every method, or the panels cannot be compared by eye --
    # which is the entire purpose of putting them on one sheet.
    vmin = float(rows["self_recovery"].min())
    for row_index, name in enumerate(methods):
        part = rows[rows["method"] == name]
        for col, (column, cmap, label) in enumerate(
            ((("self_recovery"), "RdYlGn", "self-recovery"), (("cost"), "magma_r", "CPU seconds"))
        ):
            ax = axes[row_index][col]
            table = part.pivot(index="candidate_count", columns="reference_cap", values=column)
            table = table.reindex(index=counts, columns=caps)
            values = table.to_numpy(dtype=float)
            kwargs = {"vmin": vmin, "vmax": 1.0} if column == "self_recovery" else {}
            image = ax.imshow(values, cmap=cmap, aspect="auto", **kwargs)
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if np.isfinite(values[i, j]):
                        text = (
                            f"{values[i, j]:.4f}"
                            if column == "self_recovery"
                            else f"{values[i, j]:.2f}"
                        )
                        ax.text(j, i, text, ha="center", va="center", fontsize=7.5)
            ax.set_xticks(range(len(caps)), [str(c) for c in caps])
            ax.set_yticks(range(len(counts)), [f"C{c}" for c in counts])
            ax.set_xlabel("Reference bank cap (K)")
            ax.set_ylabel("Candidate attempts")
            ax.set_title(f"{name} · {label}", fontsize=10, fontweight="bold")
            ax.grid(False)
            fig.colorbar(image, ax=ax, fraction=0.046)
    plt.tight_layout(rect=[0, 0, 1, 1 - 0.5 / len(methods)])
    path = out / "03_grids.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def spread(frame: pd.DataFrame, cfg: StudyConfig, out: Path) -> Path | None:
    """What the seeds disagree about. A method whose seeds disagree is not reproducible."""
    style()
    methods = sorted(frame["method"].unique())
    palette = _palette(methods)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=200)
    fig.suptitle(
        f"Seed-to-seed spread · the only honest error bar here\n{_subtitle(cfg, frame)}",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[0]
    data = [frame[frame["method"] == name]["self_recovery"].to_numpy() for name in methods]
    parts = ax.boxplot(data, tick_labels=methods, showfliers=True, patch_artist=True)
    for patch, name in zip(parts["boxes"], methods):
        patch.set_facecolor(palette[name])
        patch.set_alpha(0.45)
    ax.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
    ax.set_title("Every cell of every seed, per method", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    # Range across seeds for one fixed cell, so the comparison is like for like: the box
    # on the left mixes easy cells (large K, large C) with hard ones.
    ax2 = axes[1]
    caps = sorted(frame["reference_cap"].unique())
    counts = sorted(frame["candidate_count"].unique())
    cap = min(caps, key=lambda k: abs(k - 4))
    count = min(counts, key=lambda c: abs(c - 20))
    cell = frame[(frame["reference_cap"] == cap) & (frame["candidate_count"] == count)]
    for name in methods:
        values = cell[cell["method"] == name].sort_values("seed")["self_recovery"].to_numpy()
        ax2.plot(
            range(1, len(values) + 1),
            values,
            "-o",
            color=palette[name],
            label=name,
            linewidth=1.5,
            markersize=4,
        )
    ax2.set_xlabel("Seed (independent embedding stream)")
    ax2.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
    ax2.set_title(f"One cell across seeds (K={cap}, C={count})", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    path = out / "04_spread.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(frame: pd.DataFrame, cfg: StudyConfig, out: Path) -> list[Path]:
    """Every sheet, logging rather than raising when one of them cannot be drawn."""
    written = []
    for name, function in (
        ("01", omnibus),
        ("02", pareto),
        ("03", grids),
        ("04", spread),
    ):
        try:
            path = function(frame, cfg, out)
        except Exception:
            logger.exception("figure %s failed", name)
            continue
        if path is not None:
            written.append(path)
    logger.info("wrote %d figures to %s", len(written), out)
    return written
