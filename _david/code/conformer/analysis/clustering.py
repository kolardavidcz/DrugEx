"""Figures for component 3: clustering and reference-bank benchmark.

``drug128_definitive_synthesis.png`` is the panel that decides the reference
policy.  It combines four views of the same 121-drug grid:

A. Score versus reference cap for Butina ``0.35/0.50/1.25 Å`` and shape+color
   farthest-first.  This is the primary clustering comparison.
B. Speed/accuracy frontier at ``K=4``: is a minimized-but-small candidate pool
   ever beaten by a larger unminimized one?
C. Selectivity audit: self-recovery against cross-drug decoy score, so a metric
   that inflates both (α+β=1 Tversky) cannot masquerade as better.
D. The number of distinct reference clusters per drug, which explains *why*
   ``1.25 Å`` fails: it merges genuinely different basins into one pose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.code.conformer.analysis import loaders
from _david.code.conformer.analysis.style import PALETTE, apply_style, save_figure
from _david.code.conformer.core.paths import component_dir

#: Selection-rule styling shared by panels A and D.
SELECTION_STYLE: dict[str, tuple[str, str, str]] = {
    "diverse": ("Diverse farthest-first (shape+color)", "#d62728", "o"),
    "rmsd_0.35": ("Butina RMSD 0.35 Å", "#1f77b4", "s"),
    "rmsd_0.50": ("Butina RMSD 0.50 Å", "#2ca02c", "^"),
    "rmsd_1.25": ("Butina RMSD 1.25 Å", "#7f7f7f", "d"),
}

#: Metric display order for the selectivity audit.
METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("tanimoto_05_05", "TanimotoCombo\n(0.5S+0.5C, α=β=1)"),
    ("tversky_03_07_sum1", "Tversky\n(0.3, 0.7), α+β=1"),
    ("tversky_06_14_sum2", "Tversky\n(0.6, 1.4), α+β=2"),
    ("tversky_08_12_sum2", "Tversky\n(0.8, 1.2), α+β=2"),
    ("tanimoto_03S_07C", "Color-heavy\n0.3S+0.7C"),
    ("tanimoto_07S_03C", "Shape-heavy\n0.7S+0.3C"),
    ("tversky_03_07_color_heavy", "Tversky (0.6, 1.4)\n+ color-heavy"),
)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing columns {missing}")


def figure_synthesis(
    grid: loaders.Table,
    cross_decoy: loaders.Table | None = None,
    output_dir: Path | str | None = None,
    *,
    filename: str = "drug128_definitive_synthesis.png",
) -> Path:
    """Render the four-panel clustering and reference-bank synthesis."""
    apply_style()
    frame = grid.frame.copy()
    _require(
        frame,
        (
            "mol_id",
            "regime",
            "ref_sel",
            "n_clusters",
            "k_label",
            "k_actual",
            "c_attempts",
            "metric",
            "score",
            "total_sec",
        ),
        "synthesis_grid",
    )
    figure, axes = plt.subplots(2, 2, figsize=(19, 13.5))
    figure.subplots_adjust(hspace=0.42, wspace=0.30)

    # -- A. reference merging ---------------------------------------------
    axis = axes[0, 0]
    caps = ["1", "2", "3", "4", "8", "all"]
    subset = frame[
        (frame["regime"] == "MMFF94s_Ref_vs_ETKDG_Cand")
        & (frame["c_attempts"] == 20)
        & (frame["metric"] == "tanimoto_05_05")
    ]
    x = np.arange(len(caps))
    for key, (label, colour, marker) in SELECTION_STYLE.items():
        data = (
            subset[subset["ref_sel"] == key]
            .groupby("k_label")[["score", "k_actual"]]
            .mean()
            .reindex(caps)
        )
        if data["score"].isna().all():
            continue
        axis.plot(x, data["score"].to_numpy(dtype=float), marker=marker, color=colour, linewidth=2.3, label=label)
        for position, (score, actual) in enumerate(zip(data["score"], data["k_actual"])):
            if not np.isfinite(score):
                continue
            axis.annotate(
                f"{score:.3f} (K̄={actual:.1f})",
                (position, score),
                textcoords="offset points",
                xytext=(0, -16 if key in {"diverse", "rmsd_1.25"} else 8),
                ha="center",
                fontsize=6.8,
                color=colour,
                fontweight="bold",
            )
    axis.set_xticks(x)
    axis.set_xticklabels([f"K={cap}" if cap != "all" else "K=all" for cap in caps])
    axis.set_title("A. Reference merging at C=20\n[0.35 Å and diverse beat legacy 1.25 Å]")
    axis.set_xlabel("Reference cap K")
    axis.set_ylabel("Mean ROCS self-recovery")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7.8, loc="lower right")

    # -- B. speed/accuracy frontier --------------------------------------
    axis = axes[0, 1]
    candidates = [10, 20, 30, 50]
    for key, colour, linestyle in (
        ("diverse", "#d62728", "-"),
        ("rmsd_0.35", "#1f77b4", "--"),
        ("rmsd_0.50", "#2ca02c", ":"),
    ):
        data = (
            frame[
                (frame["regime"] == "MMFF94s_Ref_vs_ETKDG_Cand")
                & (frame["ref_sel"] == key)
                & (frame["k_label"] == "4")
                & (frame["metric"] == "tanimoto_05_05")
            ]
            .groupby("c_attempts")[["score", "total_sec"]]
            .mean()
            .reindex(candidates)
        )
        if data["score"].isna().all():
            continue
        axis.plot(
            data["total_sec"].to_numpy(dtype=float),
            data["score"].to_numpy(dtype=float),
            marker="o",
            linestyle=linestyle,
            color=colour,
            linewidth=2.2,
            label=f"MMFF94s ref ({SELECTION_STYLE[key][0]}) + unminimized ETKDG cand",
        )
        if key == "rmsd_0.35":
            for cap, row in data.iterrows():
                if np.isfinite(row["score"]):
                    axis.annotate(
                        f"C={cap}\n({row['total_sec']:.2f}s)",
                        (row["total_sec"], row["score"]),
                        textcoords="offset points",
                        xytext=(0, 8 if cap == 50 else -20),
                        ha="center",
                        fontsize=7.0,
                        color=colour,
                        fontweight="bold",
                    )
    for regime, colour, marker, label in (
        ("MMFF94s_Ref_vs_MMFF94s_Cand", "#9467bd", "s", "MMFF94s ref + minimized MMFF94s cand (C=20)"),
        ("ETKDG_Ref_vs_ETKDG_Cand", "#8c564b", "^", "unminimized ETKDG ref + ETKDG cand (C=20)"),
    ):
        data = frame[
            (frame["regime"] == regime)
            & (frame["ref_sel"] == "rmsd_0.35")
            & (frame["k_label"] == "4")
            & (frame["metric"] == "tanimoto_05_05")
        ]
        if data.empty:
            continue
        axis.scatter(
            [float(data["total_sec"].mean())],
            [float(data["score"].mean())],
            color=colour,
            marker=marker,
            s=95,
            zorder=5,
            label=f"{label}: {data['score'].mean():.3f} @ {data['total_sec'].mean():.2f}s",
        )
    axis.set_xscale("log")
    axis.set_title("B. Speed/accuracy frontier at K=4\n[why curated MMFF94s references + fast ETKDG candidates win]")
    axis.set_xlabel("Mean candidate evaluation time per drug (s, log scale)")
    axis.set_ylabel("Mean ROCS self-recovery (K=4)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=7.4, loc="center right")

    # -- C. selectivity audit ---------------------------------------------
    axis = axes[1, 0]
    if cross_decoy is not None and not cross_decoy.frame.empty:
        decoy_frame = cross_decoy.frame
        self_scores = (
            frame[
                (frame["regime"] == "MMFF94s_Ref_vs_ETKDG_Cand")
                & (frame["ref_sel"] == "rmsd_0.35")
                & (frame["k_label"] == "4")
                & (frame["c_attempts"] == 20)
            ]
            .groupby("metric")["score"]
            .mean()
        )
        decoy_stats = decoy_frame.groupby("metric")["decoy_score"].agg(
            decoy_mean="mean", decoy_p90=lambda values: float(np.percentile(values, 90))
        )
        metrics = [key for key, _ in METRIC_LABELS if key in self_scores.index]
        labels = [label for key, label in METRIC_LABELS if key in self_scores.index]
        self_values = [float(self_scores.get(key, 0.0)) for key in metrics]
        decoy_values = [float(decoy_stats.loc[key, "decoy_mean"]) for key in metrics]
        margins = [a - b for a, b in zip(self_values, decoy_values)]
        positions = np.arange(len(metrics))
        width = 0.27
        bars = (
            axis.bar(positions - width, self_values, width=width, color="#2ca02c", label="Self-recovery mean"),
            axis.bar(positions, decoy_values, width=width, color="#d62728", label="Cross-drug decoy mean"),
            axis.bar(positions + width, margins, width=width, color="#1f77b4", label="Selectivity margin"),
        )
        axis.axhline(0.35, color="black", linestyle=":", linewidth=1.3, label="Static RL threshold (0.35)")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=24, ha="right", fontsize=7.6)
        axis.set_ylim(0.0, 1.08)
        for group in bars:
            for rectangle in group:
                height = rectangle.get_height()
                axis.text(
                    rectangle.get_x() + rectangle.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.4,
                )
        axis.set_ylabel("ROCS score / selectivity margin")
        axis.set_title("C. Selectivity audit: self vs cross-drug decoys\n[α+β=1 inflates decoys; α+β=2 and color-heavy keep margin]")
        axis.legend(fontsize=7.2, loc="upper right")
        axis.grid(axis="y", alpha=0.3)
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "Cross-decoy table not available", ha="center", va="center")

    # -- D. cluster count distribution ------------------------------------
    axis = axes[1, 1]
    cluster_frame = (
        frame[
            (frame["regime"] == "MMFF94s_Ref_vs_ETKDG_Cand")
            & (frame["k_label"] == "all")
            & (frame["c_attempts"] == 20)
            & (frame["metric"] == "tanimoto_05_05")
        ][["mol_id", "ref_sel", "n_clusters"]]
        .drop_duplicates()
    )
    bins = np.arange(1, 17) - 0.5
    for key, (label, colour, _) in SELECTION_STYLE.items():
        if key == "diverse":
            continue
        values = cluster_frame[cluster_frame["ref_sel"] == key]["n_clusters"].to_numpy()
        if len(values) == 0:
            continue
        axis.hist(
            np.clip(values, 1, 15),
            bins=bins,
            alpha=0.45,
            color=colour,
            edgecolor=colour,
            linewidth=1.2,
            label=f"{label} (mean={values.mean():.1f}, median={np.median(values):.0f})",
        )
    axis.axvline(4.0, color="#d62728", linestyle="--", linewidth=1.8, label="Recommended cap K=4")
    axis.set_xticks(range(1, 16))
    axis.set_xticklabels([str(value) if value < 15 else "15+" for value in range(1, 16)])
    axis.set_title("D. Distinct reference clusters per drug\n[1.25 Å collapses most drugs to 1–2 poses]")
    axis.set_xlabel("Surviving distinct reference clusters (capped at 15+ for display)")
    axis.set_ylabel("Number of drugs")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=7.8, loc="upper left")

    figure.suptitle(
        "121-Drug Definitive Synthesis: reference merging, speed/accuracy frontier, and selectivity",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    target = Path(output_dir) if output_dir else component_dir("clustering")
    inputs = loaders.input_paths(grid, *( [cross_decoy] if cross_decoy is not None else [] ))
    return save_figure(
        figure,
        target / filename,
        inputs=inputs,
        notes="Clustering benchmark synthesis: merging rule, frontier, selectivity, cluster counts.",
    )


def cluster_counts(frame: pd.DataFrame, *, cap_label: str = "all") -> pd.DataFrame:
    """Return per-selection cluster-count statistics at one reference cap."""
    subset = frame[frame["k_label"] == cap_label]
    return (
        subset.groupby("ref_sel")["n_clusters"]
        .agg(molecules="size", mean="mean", median="median", minimum="min", maximum="max")
        .reset_index()
    )


__all__ = ["METRIC_LABELS", "SELECTION_STYLE", "cluster_counts", "figure_synthesis"]
