"""Figures for component 2: color/shape scoring at the production default.

``tversky_color_162grid_selectivity.png`` is the complete
``(alpha, beta) x (w_shape, w_color)`` audit: 18 asymmetric overlap settings
times 9 shape/color weightings, evaluated on every ordered molecule pair.

The figure exists to defend three claims with data rather than assertion:

1. Symmetric TanimotoCombo (``alpha = beta = 1``, ``0.5 S + 0.5 C``) is the
   safest default; moving either dial trades one failure mode for another.
2. Draining ``alpha`` (a sub-shape weighting, ``alpha + beta = 2``) inflates
   small fragments, because the missing reference volume is barely penalized.
   Draining ``beta`` (a super-shape weighting) inflates bloated blobs, because
   the candidate's excess volume is barely penalized.  Both regimes raise the
   corresponding false-positive floor above the RL desirability threshold.
3. The saddle between the traps keeps a positive self-versus-decoy margin on
   curated hard decoys that share the reference pharmacophore.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.code.conformer.analysis import loaders
from _david.code.conformer.analysis.style import PALETTE, apply_style, save_figure
from _david.code.conformer.core.paths import component_dir

#: Historical category columns that may be present in a 162-grid table.
CATEGORY_SERIES: tuple[tuple[str, str, str], ...] = (
    ("small_decoy_p95", "Small-fragment decoy P95", "#d95f02"),
    ("bloated_decoy_p95", "Bloated-monster decoy P95", "#7570b3"),
    ("trojan_decoy_p95", "Hard shape-matched decoy P95", "#1b9e77"),
)

#: Selected configurations shown on the EGCG stress panel.
STRESS_CONFIGS: tuple[tuple[str, str, str, str], ...] = (
    ("sum2_1.0_1.0", "0.5S_0.5C", "Tanimoto (1.0, 1.0) | 0.5S+0.5C [default]", "#666666"),
    ("sum1_0.3_0.7", "0.5S_0.5C", "Tversky (0.3, 0.7), α+β=1 | 0.5S+0.5C", "#e7298a"),
    ("sum2_0.2_1.8", "0.5S_0.5C", "Sub-shape (0.2, 1.8), α+β=2 | 0.5S+0.5C", "#d95f02"),
    ("sum2_1.8_0.2", "0.5S_0.5C", "Super-shape (1.8, 0.2), α+β=2 | 0.5S+0.5C", "#7570b3"),
    ("sum2_0.6_1.4", "0.3S_0.7C", "Tversky (0.6, 1.4), α+β=2 | 0.3S+0.7C", "#1b9e77"),
    ("sum2_1.0_1.0", "0.2S_0.8C", "Tanimoto (1.0, 1.0) | 0.2S+0.8C", "#2b8cbe"),
)


def _order_labels(grid: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return the canonical (tversky order, display, weight order, display)."""
    tversky_order = [
        *(f"sum1_{alpha:.1f}_{1.0 - alpha:.1f}" for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        *(f"sum2_{alpha:.1f}_{2.0 - alpha:.1f}" for alpha in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)),
    ]
    tversky_display = [
        *(f"α+β=1 ({alpha:.1f}, {1.0 - alpha:.1f})" for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        *(
            f"α+β=2 ({alpha:.1f}, {2.0 - alpha:.1f})" + (" [Tan]" if abs(alpha - 1.0) < 1e-6 else "")
            for alpha in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
        ),
    ]
    weight_order = [f"{w:.1f}S_{1.0 - w:.1f}C" for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    weight_display = [f"{w:.1f}S\n{1.0 - w:.1f}C" for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    available = set(grid["tversky_label"])
    available_weights = set(grid["weight_label"])
    keep_t = [index for index, label in enumerate(tversky_order) if label in available]
    keep_w = [index for index, label in enumerate(weight_order) if label in available_weights]
    return (
        [tversky_order[index] for index in keep_t],
        [tversky_display[index] for index in keep_t],
        [weight_order[index] for index in keep_w],
        [weight_display[index] for index in keep_w],
    )


def _stress_groups(decoys: pd.DataFrame) -> list[tuple[str, list[str], bool]]:
    """Group the hard-decoy panel by category, marking decoy groups."""
    groups: list[tuple[str, list[str], bool]] = []
    for category, block in decoys.groupby("category", sort=True):
        names = sorted(block["molecule"].unique())
        positive = category.startswith(("1_", "2_")) or "Self" in category or "Active" in category
        label = str(category).replace("_", " ")
        groups.append((label, names, not positive))
    return groups


def figure_tversky_grid(
    grid: loaders.Table,
    decoys: loaders.Table | None = None,
    output_dir: Path | str | None = None,
    *,
    filename: str = "tversky_color_162grid_selectivity.png",
) -> Path:
    """Render the 162-combination selectivity and decoy-trap synthesis."""
    apply_style()
    frame = grid.frame.copy()
    required = ("tversky_label", "weight_label", "alpha", "beta", "alpha_plus_beta", "w_shape", "self_mean")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"tversky_grid is missing columns {missing}")
    worst_column = (
        "worst_decoy_p95"
        if "worst_decoy_p95" in frame.columns
        else "decoy_p95"
    )
    margin_column = (
        "margin_self_minus_worst_decoy_p95"
        if "margin_self_minus_worst_decoy_p95" in frame.columns
        else "margin_self_minus_decoy_p95"
    )

    figure = plt.figure(figsize=(19.8, 13.8))
    grid_spec = figure.add_gridspec(
        2, 2, hspace=0.33, wspace=0.38, left=0.08, right=0.97, top=0.91, bottom=0.06
    )

    # -- A. asymmetric scissors trap --------------------------------------
    axis = figure.add_subplot(grid_spec[0, 0])
    sum2_half = frame[
        (frame["alpha_plus_beta"].round(6) == 2.0) & (frame["w_shape"].round(6) == 0.5)
    ].sort_values("alpha")
    sum2_shape = frame[
        (frame["alpha_plus_beta"].round(6) == 2.0) & (frame["w_shape"].round(6) == 0.3)
    ].sort_values("alpha")
    sum1_half = frame[
        (frame["alpha_plus_beta"].round(6) == 1.0) & (frame["w_shape"].round(6) == 0.5)
    ].sort_values("alpha")
    x_index = np.arange(len(sum2_half))
    drawn = False
    for column, label, colour, linestyle, linewidth in (
        ("small_decoy_p95", "Small-fragment decoy P95 (0.5S+0.5C, α+β=2)", "#d95f02", "--", 1.6),
        ("bloated_decoy_p95", "Bloated-monster decoy P95 (0.5S+0.5C, α+β=2)", "#7570b3", "--", 1.6),
    ):
        if column in sum2_half.columns and len(sum2_half) == len(x_index):
            axis.plot(x_index, sum2_half[column], "o", linestyle=linestyle, color=colour, linewidth=linewidth, alpha=0.7, label=label)
            drawn = True
    if "worst_decoy_p95" in sum1_half.columns and len(sum1_half) == len(x_index):
        axis.plot(x_index, sum1_half["worst_decoy_p95"], "^:", color="#e7298a", linewidth=1.6, alpha=0.7, label="Worst-decoy P95, α+β=1 (0.5S+0.5C)")
        drawn = True
    for column, label, colour, marker in (
        ("small_decoy_p95", "Small-fragment decoy P95 (0.3S+0.7C, α+β=2)", "#d95f02", "o"),
        ("bloated_decoy_p95", "Bloated-monster decoy P95 (0.3S+0.7C, α+β=2)", "#7570b3", "s"),
        ("trojan_decoy_p95", "Hard shape-matched decoy P95 (0.3S+0.7C, α+β=2)", "#1b9e77", "d"),
    ):
        if column in sum2_shape.columns and len(sum2_shape) == len(x_index):
            axis.plot(x_index, sum2_shape[column], marker=marker, linestyle="-", color=colour, linewidth=2.3, label=label)
            drawn = True
    axis.axvspan(1.8, 5.2, color="#d9f0d3", alpha=0.35, label="Minimax saddle: α ∈ [0.6, 1.2]")
    axis.axhline(0.35, color="#333333", linestyle="-.", linewidth=1.2, label="RL desirability floor (0.35)")
    axis.set_xticks(x_index)
    axis.set_xticklabels(
        [f"({a:.1f}, {2.0 - a:.1f})\n[{a / 2:.1f}:{1 - a / 2:.1f}]" for a in sum2_half["alpha"]],
        fontsize=7.6,
    )
    axis.set_title("A. Asymmetric Tversky traps across cross-drug decoys")
    axis.set_xlabel("Tversky (α_ref, β_cand) on the α+β=2 scale [and α:β ratio]")
    axis.set_ylabel("95th-percentile false-positive score (P95)")
    axis.grid(True, alpha=0.25)
    if drawn:
        axis.legend(loc="upper center", ncol=2, fontsize=7.2)

    # -- B. worst-decoy heatmap -------------------------------------------
    axis = figure.add_subplot(grid_spec[0, 1])
    tversky_order, tversky_display, weight_order, weight_display = _order_labels(frame)
    pivot_p95 = frame.pivot(index="tversky_label", columns="weight_label", values=worst_column).loc[tversky_order, weight_order]
    fp_column = "fp_rate_worst_ge_035" if "fp_rate_worst_ge_035" in frame.columns else "fp_rate_decoy_ge_0.35"
    pivot_fp = (
        frame.pivot(index="tversky_label", columns="weight_label", values=fp_column).loc[tversky_order, weight_order]
        if fp_column in frame.columns
        else pd.DataFrame(0.0, index=tversky_order, columns=weight_order)
    )
    image = axis.imshow(pivot_p95.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    colourbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colourbar.set_label("Worst-regime decoy P95 (lower = safer for RL)", fontsize=8.5)
    for row in range(len(tversky_order)):
        for column in range(len(weight_order)):
            value = float(pivot_p95.iloc[row, column])
            rate = float(pivot_fp.iloc[row, column]) * 100.0
            axis.text(
                column,
                row,
                f"{value:.2f}\n({rate:.0f}%)",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if value >= 0.56 or value <= 0.28 else "#111111",
                fontweight="bold" if value <= 0.35 else "normal",
            )
    axis.axhline(8.5, color="black", linewidth=2.2)
    axis.set_xticks(np.arange(len(weight_order)))
    axis.set_xticklabels(weight_display, fontsize=7.4)
    axis.set_yticks(np.arange(len(tversky_order)))
    axis.set_yticklabels(tversky_display, fontsize=7.2)
    axis.set_xlabel("Shape / color weighting (w_S + w_C = 1)")
    axis.set_title("B. Worst-decoy P95 and (% false positives ≥ 0.35)")

    # -- C. selectivity margin heatmap ------------------------------------
    axis = figure.add_subplot(grid_spec[1, 0])
    pivot_margin = frame.pivot(index="tversky_label", columns="weight_label", values=margin_column).loc[tversky_order, weight_order]
    pivot_self = frame.pivot(index="tversky_label", columns="weight_label", values="self_mean").loc[tversky_order, weight_order]
    image = axis.imshow(pivot_margin.to_numpy(), cmap="YlGnBu", aspect="auto")
    colourbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colourbar.set_label("Selectivity margin = self mean − worst decoy P95", fontsize=8.5)
    for row in range(len(tversky_order)):
        for column in range(len(weight_order)):
            margin = float(pivot_margin.iloc[row, column])
            self_value = float(pivot_self.iloc[row, column])
            axis.text(
                column,
                row,
                f"+{margin:.2f}\n[{self_value:.2f}]",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if margin >= 0.36 else "#111111",
                fontweight="bold" if margin >= 0.42 else "normal",
            )
    if "sum2_0.6_1.4" in tversky_order and "0.2S_0.8C" in weight_order:
        row = tversky_order.index("sum2_0.6_1.4")
        column = weight_order.index("0.2S_0.8C")
        axis.add_patch(
            plt.Rectangle((column - 0.5, row - 0.5), 2.0, 4.0, fill=False, edgecolor="#d95f02", linewidth=2.4, linestyle="--")
        )
    axis.axhline(8.5, color="black", linewidth=2.2)
    axis.set_xticks(np.arange(len(weight_order)))
    axis.set_xticklabels(weight_display, fontsize=7.4)
    axis.set_yticks(np.arange(len(tversky_order)))
    axis.set_yticklabels(tversky_display, fontsize=7.2)
    axis.set_xlabel("Shape / color weighting (w_S + w_C = 1)")
    axis.set_title("C. Selectivity margin (+margin and [self recovery])")

    # -- D. curated hard-decoy stress test --------------------------------
    axis = figure.add_subplot(grid_spec[1, 1])
    if decoys is not None and not decoys.frame.empty:
        decoy_frame = decoys.frame
        groups = _stress_groups(decoy_frame)
        y = np.arange(len(groups))
        bar_height = 0.11
        offsets = np.linspace(-2.5 * bar_height, 2.5 * bar_height, len(STRESS_CONFIGS))
        for index, (tversky_label, weight_label, legend_label, colour) in enumerate(STRESS_CONFIGS):
            subset = decoy_frame[
                (decoy_frame["tversky_label"] == tversky_label)
                & (decoy_frame["weight_label"] == weight_label)
            ]
            values = []
            for _, names, is_decoy in groups:
                scores = subset[subset["molecule"].isin(names)]["score"].to_numpy()
                if len(scores) == 0:
                    values.append(0.0)
                elif is_decoy:
                    values.append(float(np.max(scores)))
                else:
                    values.append(float(np.mean(scores)))
            bars = axis.barh(y + offsets[index], values, height=bar_height, color=colour, alpha=0.9, label=legend_label)
            for bar, value in zip(bars, values):
                axis.text(
                    value + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}",
                    va="center",
                    ha="left",
                    fontsize=6.4,
                    color=colour,
                    fontweight="bold",
                )
        axis.set_yticks(y)
        axis.set_yticklabels([label for label, _, _ in groups], fontsize=7.8, fontweight="bold")
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.05)
        axis.axvline(0.35, color="#333333", linestyle="-.", linewidth=1.2, label="RL threshold (0.35)")
        axis.axvline(0.25, color="#888888", linestyle=":", linewidth=1.1, label="Strict floor (0.25)")
        axis.set_xlabel("3D ROCS score vs EGCG reference (positives: mean | decoys: max)")
        axis.set_title("D. EGCG stress test: active sub-scaffolds vs fragment, steric and bloated decoys")
        axis.grid(True, axis="x", alpha=0.25)
        axis.legend(loc="lower right", fontsize=7.2)
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "Hard-decoy table not available", ha="center", va="center")

    figure.suptitle(
        "Complete 162-combination Tversky (α, β) × shape/color (w_S, w_C) selectivity and decoy-trap synthesis",
        fontsize=13.0,
        fontweight="bold",
        y=0.975,
    )
    target = Path(output_dir) if output_dir else component_dir("color_shape")
    inputs = loaders.input_paths(grid, *( [decoys] if decoys is not None else [] ))
    return save_figure(
        figure,
        target / filename,
        inputs=inputs,
        notes="162-cell asymmetric Tversky grid with curated hard-decoy stress panel.",
    )


#: Settings written to the default-scorer neighbourhood table, as
#: ``(alpha, beta, shape_weight, role)``.  One row is not a table: the point of
#: the neighbourhood is to show how far the numbers move for one step away from
#: the production default.
NEIGHBOURHOOD: tuple[tuple[float, float, float, str], ...] = (
    (1.0, 1.0, 0.5, "default"),
    (1.0, 1.0, 0.4, "lighter-shape"),
    (1.0, 1.0, 0.6, "heavier-shape"),
    (0.8, 1.2, 0.5, "sub-shape"),
    (1.2, 0.8, 0.5, "super-shape"),
    (1.0, 1.0, 0.2, "color-heavy"),
    (1.0, 1.0, 0.8, "shape-heavy"),
)


def default_scorer_summary(grid: loaders.Table) -> pd.DataFrame:
    """Return the production default cell and its nearest neighbours.

    The table exists so the default can be argued from numbers rather than from
    the figure alone: for each neighbour setting it reports self recovery, the
    worst-decoy P95, the margin, and the false-positive rate above the RL
    desirability threshold.
    """
    frame = grid.frame
    columns = [
        column
        for column in (
            "tversky_label",
            "weight_label",
            "alpha",
            "beta",
            "w_shape",
            "w_color",
            "self_mean",
            "self_p10",
            "worst_decoy_p95",
            "margin_self_minus_worst_decoy_p95",
            "fp_rate_worst_ge_035",
        )
        if column in frame.columns
    ]
    wanted = frame[columns]
    rows: list[pd.Series] = []
    for alpha, beta, w_shape, role in NEIGHBOURHOOD:
        match = wanted[
            np.isclose(wanted["alpha"], alpha)
            & np.isclose(wanted["beta"], beta)
            & np.isclose(wanted["w_shape"], w_shape)
        ]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["role"] = role
        rows.append(row)
    if not rows:
        return wanted.head(0)
    summary = pd.DataFrame(rows).reset_index(drop=True)
    return summary[["role", *columns]]


__all__ = [
    "CATEGORY_SERIES",
    "NEIGHBOURHOOD",
    "STRESS_CONFIGS",
    "default_scorer_summary",
    "figure_tversky_grid",
]
