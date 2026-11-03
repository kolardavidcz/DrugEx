"""Figures for component 1: optimization methods and the cost/accuracy frontier.

Every panel answers one falsifiable question:

``egcg_omnibus_diagnostic.png``
    For a single, thoroughly sampled molecule (EGCG): how does self-recovery
    respond to candidate budget ``C``, reference selection rule, reference cap
    ``K``, and the post-minimization energy window?

``drug_panel_stratified_vs_trimmed.png``
    Across a 128-drug panel: is "fewer minimized conformers" actually better
    than "more unminimized conformers", and why does a fixed ``K`` behave
    differently from dropping ``m`` poses from an unbounded bank?

``isomer_stratification.png``
    The same comparison split by the *true* stereoisomer count, with the
    canonical bins ``1–2, 3–4, 5–8, 9–15, 16–32, 33+``.  This is the panel that
    shows whether a physics choice is universal or only works for achiral drugs.

``reference_counts_vs_energy_window.png``
    Side quest: the energy window is a resource dial.  How many of the 100
    embedded conformers survive 3, 5, 8, 12 kcal/mol, and what does the score do?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.code.conformer.analysis import common, loaders
from _david.code.conformer.analysis.style import PALETTE, apply_style, save_figure
from _david.code.conformer.core.paths import component_dir


def _energy_window(value: object) -> float:
    """Parse the historical ``energy_window`` column, including ``inf``."""
    text = str(value).strip().lower()
    if text in {"inf", "infinity", "none", ""}:
        return float("inf")
    return float(text)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing columns {missing}")


# ---------------------------------------------------------------------------
# EGCG omnibus
# ---------------------------------------------------------------------------

#: Arms drawn on the candidate-budget panel, in plotting order.
_ARM_STYLES: tuple[tuple[str, str, str, str, str], ...] = (
    ("sym_mmff94s_eps78.4_const_spec", "MMFF94s ε=78.4 (ref & cand)", "#1b9e77", "-", "o"),
    ("sym_mmff94s_eps6.5_const_spec", "MMFF94s ε=6.5 (ref & cand)", "#d95f02", "--", "s"),
    ("sym_uff_spec", "UFF (ref & cand)", "#7570b3", "-.", "^"),
    ("sym_etkdg_spec", "ETKDG, unminimized (ref & cand)", "#e7298a", "-", "D"),
    ("asym_ref_eps78.4_cand_etkdg_spec", "MMFF94s ref vs ETKDG cand", "#a6761d", "-.", "X"),
    ("rl_spec_ref_masked_cand_mmff78.4", "RL spec ref vs masked MMFF94s cand (I=4)", "#66a61e", "--", "v"),
    ("rl_spec_ref_masked_cand_etkdg", "RL spec ref vs masked ETKDG cand (I=4)", "#e6ab02", ":", "P"),
)


def figure_egcg_omnibus(
    table: loaders.Table,
    output_dir: Path | str | None = None,
    *,
    filename: str = "egcg_omnibus_diagnostic.png",
) -> Path:
    """Render the four-panel EGCG diagnostic."""
    apply_style()
    frame = table.frame.copy()
    _require(
        frame,
        (
            "arm",
            "selection_mode",
            "energy_window",
            "cap_type",
            "cap_label",
            "cap_param",
            "candidate_attempts",
            "combo",
            "color",
            "surviving_ref_pool",
            "total_cand_sec",
        ),
        "egcg_omnibus_grid",
    )
    frame["energy_window_value"] = frame["energy_window"].map(_energy_window)

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -- A. candidate budget curve -----------------------------------------
    axis = axes[0, 0]
    base = frame[
        (frame["cap_label"] == "K=4")
        & (frame["selection_mode"] == "diverse")
        & (frame["energy_window_value"] == 8.0)
        & (frame["cap_type"] == "fixed")
    ]
    drawn = 0
    for key, label, colour, linestyle, marker in _ARM_STYLES:
        subset = base[base["arm"] == key].sort_values("candidate_attempts")
        if subset.empty:
            continue
        axis.plot(
            subset["candidate_attempts"],
            subset["combo"],
            label=label,
            color=colour,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
        drawn += 1
    axis.axhline(0.95, color="gray", linestyle=":", linewidth=1.0, label="0.95 engineering target")
    axis.set_title("A. EGCG self-recovery vs candidate budget C (K=4, diverse)")
    axis.set_xlabel("Candidate conformer attempts per stereoisomer (C)")
    axis.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
    axis.grid(True, alpha=0.25)
    if drawn:
        axis.legend(fontsize=7.5, loc="lower right")

    # -- B. selection rule vs reference cap --------------------------------
    axis = axes[0, 1]
    selection_styles = (
        ("sym_mmff94s_eps78.4_const_spec", "diverse", "MMFF94s ε=78.4 — diverse (shape+color)", "#1b9e77", "-", "o"),
        ("sym_mmff94s_eps78.4_const_spec", "rmsd_0.5", "MMFF94s ε=78.4 — Butina 0.5 Å", "#66a61e", "--", "s"),
        ("sym_etkdg_spec", "diverse", "ETKDG — diverse (shape+color)", "#e7298a", "-", "o"),
        ("sym_etkdg_spec", "energy_only", "ETKDG — energy order", "#7570b3", "-.", "^"),
        ("sym_etkdg_spec", "rmsd_0.5", "ETKDG — Butina 0.5 Å", "#d95f02", "--", "v"),
        ("sym_etkdg_spec", "rmsd_1.25", "ETKDG — Butina 1.25 Å (collapses)", "#a6761d", ":", "X"),
    )
    subset = frame[
        (frame["candidate_attempts"] == 20)
        & (frame["energy_window_value"] == 8.0)
        & (frame["cap_type"] == "fixed")
    ].copy()
    subset["cap_order"] = subset["cap_param"].replace({0: 25})
    drawn = 0
    for arm_key, selection, label, colour, linestyle, marker in selection_styles:
        data = subset[
            (subset["arm"] == arm_key) & (subset["selection_mode"] == selection)
        ].sort_values("cap_order")
        if data.empty:
            continue
        axis.plot(
            data["cap_order"],
            data["combo"],
            label=label,
            color=colour,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )
        drawn += 1
    axis.set_xticks([1, 2, 3, 4, 5, 8, 12, 16, 20, 25])
    axis.set_xticklabels(["1", "2", "3", "4", "5", "8", "12", "16", "20", "all"])
    axis.set_title("B. Reference selection rule and cap (C=20)")
    axis.set_xlabel("Reference cap K")
    axis.set_ylabel("ROCS TanimotoCombo / 2  [0, 1]")
    axis.grid(True, alpha=0.25)
    if drawn:
        axis.legend(fontsize=7.5, loc="lower right")

    # -- C. fixed cap vs tail trimming -------------------------------------
    axis = axes[1, 0]
    twin = axis.twinx()
    trimmed = frame[
        (frame["arm"] == "sym_mmff94s_eps78.4_const_spec")
        & (frame["selection_mode"] == "diverse")
        & (frame["energy_window_value"] == 8.0)
        & (frame["candidate_attempts"] == 20)
    ].sort_values("actual_references")
    trimmed = trimmed.drop_duplicates(subset=["cap_label"])
    positions = np.arange(len(trimmed))
    colours = [
        "#ccebc5" if label.startswith("K=") else "#fed9a6"
        for label in trimmed["cap_label"]
    ]
    twin.bar(
        positions,
        trimmed["total_cand_sec"],
        color=colours,
        alpha=0.55,
        width=0.65,
        label="Scoring latency (s)",
    )
    axis.plot(
        positions,
        trimmed["combo"],
        color="#1b9e77",
        marker="o",
        linewidth=2.2,
        label="ROCS combo (MMFF94s ε=78.4)",
    )
    axis.plot(
        positions,
        trimmed["color"],
        color="#d95f02",
        marker="s",
        linestyle="--",
        linewidth=1.6,
        label="Color Tanimoto",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{label}\n(n={int(count)})"
            for label, count in zip(trimmed["cap_label"], trimmed["actual_references"])
        ],
        rotation=45,
        ha="right",
        fontsize=7.5,
    )
    axis.set_title("C. Fixed cap K vs tail-trimmed (all − m) reference budgets")
    axis.set_ylabel("ROCS score [0, 1]", color="#1b9e77")
    twin.set_ylabel("Candidate scoring time (s)", color="#8c510a")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="lower right", fontsize=7.5)

    # -- D. energy window --------------------------------------------------
    axis = axes[1, 1]
    windows = ["3.0", "5.0", "8.0", "12.0", "inf"]
    window_labels = ["3", "5", "8\n(default)", "12", "∞"]
    positions = np.arange(len(windows))
    width = 0.36
    for offset, arm_key, label, colour in (
        (-0.5, "sym_mmff94s_eps6.5_const_spec", "MMFF94s ε=6.5", "#d95f02"),
        (0.5, "sym_mmff94s_eps78.4_const_spec", "MMFF94s ε=78.4", "#1b9e77"),
    ):
        data = frame[
            (frame["arm"] == arm_key)
            & (frame["selection_mode"] == "diverse")
            & (frame["candidate_attempts"] == 20)
            & (frame["cap_label"] == "K=all")
        ].copy()
        if data.empty:
            continue
        data["window_key"] = data["energy_window"].astype(str)
        ordered = data.set_index("window_key").reindex(windows)
        bars = axis.bar(
            positions + offset * width,
            ordered["surviving_ref_pool"],
            width=width,
            color=colour,
            alpha=0.85,
            label=f"{label} surviving pool (K=all)",
        )
        for bar, score in zip(bars, ordered["combo"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{score:.3f}",
                ha="center",
                fontsize=7.5,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{label}\nkcal/mol" for label in window_labels])
    axis.set_ylabel("Surviving reference conformers (of 100)")
    axis.set_title("D. Energy window vs surviving reference pool and score")
    axis.grid(True, alpha=0.25, axis="y")
    axis.legend(fontsize=7.5, loc="upper left")

    figure.tight_layout()
    target = Path(output_dir) if output_dir else component_dir("optimization_methods")
    return save_figure(
        figure,
        target / filename,
        inputs=loaders.input_paths(table),
        notes="EGCG optimization-method omnibus (candidate budget, selection, cap, energy window).",
    )


# ---------------------------------------------------------------------------
# 128-drug panel
# ---------------------------------------------------------------------------


def _panel_cells(table: loaders.Table) -> pd.DataFrame:
    frame = table.frame.copy()
    _require(
        frame,
        (
            "molecule_id",
            "method",
            "reference_cap",
            "candidate_attempts",
            "color_excluded",
            "actual_references",
            "reference_failure",
            "enumerated_reference_isomers",
            "score",
            "seconds",
        ),
        "drug_panel_cells",
    )
    return frame


def figure_drug_panel(
    cells: loaders.Table,
    summary: loaders.Table,
    output_dir: Path | str | None = None,
    *,
    filename: str = "drug_panel_stratified_vs_trimmed.png",
) -> Path:
    """Render the four-panel 128-drug panel diagnostic."""
    apply_style()
    frame = _panel_cells(cells)
    summary_frame = summary.frame.copy()

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -- A. cost vs accuracy ----------------------------------------------
    axis = axes[0, 0]
    style = {
        "mmff_high_dielectric": ("MMFF94s ε=78.4 (minimized)", "#1b9e77", "o"),
        "mmff_vacuum": ("MMFF94s ε=1.0 (minimized)", "#d95f02", "s"),
        "uff": ("UFF (minimized)", "#7570b3", "^"),
        "etkdg": ("ETKDG only (unminimized)", "#e7298a", "D"),
    }
    for method, (label, colour, marker) in style.items():
        data = summary_frame[summary_frame["method"] == method]
        if data.empty:
            continue
        axis.scatter(
            data["mean_seconds"],
            data["mean"],
            color=colour,
            marker=marker,
            s=36,
            alpha=0.8,
            label=label,
        )
        trajectory = data[data["reference_cap"] == 4].sort_values("candidate_attempts")
        axis.plot(
            trajectory["mean_seconds"],
            trajectory["mean"],
            color=colour,
            linewidth=1.6,
        )
        for _, row in trajectory.iterrows():
            if method in {"mmff_high_dielectric", "etkdg"} and row["candidate_attempts"] in {10, 20, 75}:
                axis.annotate(
                    f"K4/C{int(row['candidate_attempts'])}",
                    (row["mean_seconds"], row["mean"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7.0,
                    color=colour,
                    weight="bold",
                )
    axis.set_xscale("log")
    axis.set_title("A. Fewer minimized conformers beat more unminimized conformers")
    axis.set_xlabel("Mean scoring latency per molecule (s, log scale)")
    axis.set_ylabel("Mean ROCS self-recovery (color-eligible drugs)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7.5, loc="lower right")

    # -- B. stratified by enumerated isomers ------------------------------
    axis = axes[0, 1]
    valid = frame[~frame["color_excluded"].astype(bool) & ~frame["reference_failure"].astype(bool)].copy()
    iso_labels = {
        1.0: "I = 1 (achiral / specified)",
        2.0: "I = 2 (one unassigned centre)",
        4.0: "I = 3–4",
        8.0: "I = 5–8",
    }
    subset = valid[
        (valid["method"] == "mmff_high_dielectric")
        & (valid["candidate_attempts"] == 20)
    ].copy()
    subset["cap_order"] = subset["reference_cap"].replace({0: 16})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for colour, isomers in zip(palette, sorted(iso_labels)):
        group = subset[subset["enumerated_reference_isomers"] == isomers]
        if group.empty:
            continue
        aggregated = group.groupby("cap_order")["score"].mean().reset_index()
        axis.plot(
            aggregated["cap_order"],
            aggregated["score"],
            marker="o",
            linewidth=2.0,
            color=colour,
            label=f"{iso_labels[isomers]} (n={group['molecule_id'].nunique()})",
        )
    axis.set_xticks([1, 2, 4, 8, 12, 16])
    axis.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"])
    axis.set_title("B. Stratified by enumerated reference stereoisomers (C=20)")
    axis.set_xlabel("Reference cap K (balanced across stereoisomers)")
    axis.set_ylabel("Mean ROCS self-recovery")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7.5, loc="lower right")

    # -- C. single-isomer landscape steepness -----------------------------
    axis = axes[1, 0]
    bank_sizes = (
        valid[valid["reference_cap"] == 0]
        .groupby(["method", "molecule_id"])["actual_references"]
        .first()
        .rename("total_bank_refs")
        .reset_index()
    )
    merged = valid.merge(bank_sizes, on=["method", "molecule_id"], how="left")
    single = merged[
        (merged["enumerated_reference_isomers"] == 1)
        & (merged["method"] == "mmff_high_dielectric")
        & (merged["candidate_attempts"] == 20)
    ].copy()
    single["landscape"] = pd.cut(
        single["total_bank_refs"],
        bins=[0, 25, 45, 55],
        labels=[
            "Steep PES / flexible (≤25 survivors)",
            "Moderate PES (26–45 survivors)",
            "Flat PES / rigid (46–50 survivors)",
        ],
    )
    single["cap_order"] = single["reference_cap"].replace({0: 16})
    for colour, label in zip(
        ["#e7298a", "#d95f02", "#1b9e77"],
        [
            "Steep PES / flexible (≤25 survivors)",
            "Moderate PES (26–45 survivors)",
            "Flat PES / rigid (46–50 survivors)",
        ],
    ):
        group = single[single["landscape"] == label]
        if group.empty:
            continue
        aggregated = group.groupby("cap_order")["score"].mean().reset_index()
        axis.plot(
            aggregated["cap_order"],
            aggregated["score"],
            marker="s",
            linewidth=2.0,
            color=colour,
            label=f"{label} (n={group['molecule_id'].nunique()})",
        )
    axis.set_xticks([1, 2, 4, 8, 12, 16])
    axis.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"])
    axis.set_title("C. Single-isomer drugs stratified by surviving reference count")
    axis.set_xlabel("Reference cap K")
    axis.set_ylabel("Mean ROCS self-recovery")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7.0, loc="lower right")

    # -- D. fixed K vs all - m --------------------------------------------
    axis = axes[1, 1]
    sizes = bank_sizes[bank_sizes["method"] == "mmff_high_dielectric"]["total_bank_refs"].to_numpy()
    budgets = [
        ("K=1", np.minimum(sizes, 1)),
        ("K=3", np.minimum(sizes, 3)),
        ("K=5", np.minimum(sizes, 5)),
        ("K=8", np.minimum(sizes, 8)),
        ("K=12", np.minimum(sizes, 12)),
        ("all−10", np.maximum(1, sizes - 10)),
        ("all−5", np.maximum(1, sizes - 5)),
        ("all−3", np.maximum(1, sizes - 3)),
        ("all−1", np.maximum(1, sizes - 1)),
        ("all", sizes),
    ]
    boxplot = axis.boxplot(
        [values for _, values in budgets],
        tick_labels=[label for label, _ in budgets],
        patch_artist=True,
        showfliers=False,
    )
    for patch, (label, _) in zip(boxplot["boxes"], budgets):
        patch.set_facecolor("#ccebc5" if label.startswith("K=") else "#fed9a6")
    axis.set_title("D. Actual references retained per drug: fixed K vs (all − m)")
    axis.set_xlabel("Reference budget policy")
    axis.set_ylabel("Reference conformers scored per drug")
    axis.grid(True, alpha=0.25, axis="y")
    axis.tick_params(axis="x", rotation=30)

    figure.tight_layout()
    target = Path(output_dir) if output_dir else component_dir("optimization_methods")
    return save_figure(
        figure,
        target / filename,
        inputs=loaders.input_paths(cells, summary),
        notes="128-drug panel: cost/accuracy, isomer stratification, landscape steepness, budget policy.",
    )


def figure_isomer_stratification(
    cells: loaders.Table,
    audit: loaders.Table,
    output_dir: Path | str | None = None,
    *,
    filename: str = "isomer_stratification.png",
) -> Path:
    """Minimized vs unminimized scoring, split by true stereoisomer count."""
    apply_style()
    frame = _panel_cells(cells)
    audit_frame = audit.frame
    _require(audit_frame, ("molecule_id", "masked_upper_bound", "split"), "stereo_input_audit")

    bounds = (
        audit_frame[audit_frame["split"] == "development"][
            ["molecule_id", "masked_upper_bound"]
        ]
        .drop_duplicates("molecule_id")
    )
    frame = frame.merge(bounds, on="molecule_id", how="left")
    frame["isomer_bin"] = common.isomer_bin(frame["masked_upper_bound"])

    valid = frame[
        ~frame["color_excluded"].astype(bool)
        & ~frame["reference_failure"].astype(bool)
        & frame["isomer_bin"].notna()
    ]
    fixed_cap = valid[
        (valid["reference_cap"] == 4) & (valid["candidate_attempts"] == 20)
    ]
    methods = {
        "mmff_high_dielectric": ("Minimized (MMFF94s ε=78.4)", "#1b9e77"),
        "etkdg": ("Unminimized (ETKDG)", "#e7298a"),
    }

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -- A. score per bin at K=4, C=20 ------------------------------------
    axis = axes[0, 0]
    labels = list(common.ISOMER_BIN_LABELS)
    positions = np.arange(len(labels))
    width = 0.38
    counts: dict[str, int] = {}
    for offset, (method, (label, colour)) in zip((-0.5, 0.5), methods.items()):
        data = fixed_cap[fixed_cap["method"] == method]
        means = data.groupby("isomer_bin", observed=True)["score"].mean().reindex(labels)
        molecules = data.groupby("isomer_bin", observed=True)["molecule_id"].nunique().reindex(labels)
        counts[method] = int(molecules.fillna(0).sum())
        axis.bar(
            positions + offset * width,
            means.to_numpy(dtype=float),
            width=width,
            color=colour,
            alpha=0.85,
            label=f"{label} (n={int(molecules.fillna(0).sum())})",
        )
        for position, value in zip(positions, means.to_numpy(dtype=float)):
            if np.isfinite(value):
                axis.text(
                    position + offset * width,
                    value + 0.004,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=7.0,
                    rotation=90,
                )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("A. Score at K=4, C=20 by true stereoisomer count")
    axis.set_xlabel("Enumerated stereoisomers before capping")
    axis.set_ylabel("Mean ROCS self-recovery")
    axis.grid(True, alpha=0.25, axis="y")
    axis.legend(fontsize=7.5, loc="lower left")

    # -- B. paired delta per bin ------------------------------------------
    axis = axes[0, 1]
    pivoted = fixed_cap.pivot_table(
        index=["molecule_id", "isomer_bin"],
        columns="method",
        values="score",
        observed=True,
    ).reset_index()
    pivoted = pivoted.dropna(subset=["mmff_high_dielectric", "etkdg"])
    pivoted["delta"] = pivoted["mmff_high_dielectric"] - pivoted["etkdg"]
    deltas = [
        pivoted[pivoted["isomer_bin"] == label]["delta"].to_numpy()
        for label in labels
    ]
    non_empty = [index for index, values in enumerate(deltas) if len(values)]
    if non_empty:
        boxplot = axis.boxplot(
            [deltas[index] for index in non_empty],
            positions=[index for index in non_empty],
            patch_artist=True,
            showfliers=False,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor("#1b9e77")
            patch.set_alpha(0.7)
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_title("B. Paired gain from minimization (minimized − unminimized)")
    axis.set_xlabel("Enumerated stereoisomers before capping")
    axis.set_ylabel("Δ ROCS self-recovery")
    axis.grid(True, alpha=0.25, axis="y")

    # -- C. surviving reference pool per bin ------------------------------
    axis = axes[1, 0]
    all_caps = valid[valid["reference_cap"] == 0]
    for method, (label, colour) in methods.items():
        data = all_caps[all_caps["method"] == method]
        means = (
            data.groupby("isomer_bin", observed=True)["reference_pool"]
            .mean()
            .reindex(labels)
        )
        axis.plot(
            positions,
            means.to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            color=colour,
            label=label,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_title("C. Reference conformers surviving the 8 kcal/mol window")
    axis.set_xlabel("Enumerated stereoisomers before capping")
    axis.set_ylabel("Mean surviving references (K=all)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7.5)

    # -- D. cost per bin ---------------------------------------------------
    axis = axes[1, 1]
    for method, (label, colour) in methods.items():
        data = fixed_cap[fixed_cap["method"] == method]
        means = data.groupby("isomer_bin", observed=True)["seconds"].mean().reindex(labels)
        axis.plot(
            positions,
            means.to_numpy(dtype=float),
            marker="s",
            linewidth=2.0,
            color=colour,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_title("D. Scoring cost per molecule at K=4, C=20")
    axis.set_xlabel("Enumerated stereoisomers before capping")
    axis.set_ylabel("Mean seconds per molecule (log scale)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7.5)

    figure.tight_layout()
    target = Path(output_dir) if output_dir else component_dir("optimization_methods")
    return save_figure(
        figure,
        target / filename,
        inputs=loaders.input_paths(cells, audit),
        notes=(
            "Minimized vs unminimized scoring stratified by the true stereoisomer "
            "upper bound, canonical bins 1-2, 3-4, 5-8, 9-15, 16-32, 33+."
        ),
    )


def figure_reference_counts_vs_energy_window(
    table: loaders.Table,
    output_dir: Path | str | None = None,
    *,
    filename: str = "reference_counts_vs_energy_window.png",
) -> Path:
    """Side quest: how the energy window trades pool size for score."""
    apply_style()
    frame = table.frame.copy()
    _require(
        frame,
        (
            "arm",
            "selection_mode",
            "energy_window",
            "candidate_attempts",
            "cap_label",
            "surviving_ref_pool",
            "combo",
        ),
        "egcg_omnibus_grid",
    )
    windows = ["3.0", "5.0", "8.0", "12.0", "inf"]
    labels = ["3", "5", "8 (default)", "12", "∞"]
    positions = np.arange(len(windows))
    arms = (
        ("sym_mmff94s_eps78.4_const_spec", "MMFF94s ε=78.4", "#1b9e77", "o"),
        ("sym_mmff94s_eps6.5_const_spec", "MMFF94s ε=6.5", "#d95f02", "s"),
        ("sym_etkdg_spec", "ETKDG (unminimized)", "#e7298a", "D"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, column, ylabel, title in (
        (
            axes[0],
            "surviving_ref_pool",
            "Surviving reference conformers (of 100)",
            "A. Pool size after the energy window",
        ),
        (
            axes[1],
            "combo",
            "ROCS TanimotoCombo / 2  [0, 1]",
            "B. Self-recovery after the energy window",
        ),
    ):
        for arm_key, label, colour, marker in arms:
            data = frame[
                (frame["arm"] == arm_key)
                & (frame["selection_mode"] == "diverse")
                & (frame["candidate_attempts"] == 20)
                & (frame["cap_label"] == "K=all")
            ].copy()
            if data.empty:
                continue
            data["window_key"] = data["energy_window"].astype(str)
            ordered = data.set_index("window_key").reindex(windows)
            axis.plot(
                positions,
                ordered[column].to_numpy(dtype=float),
                marker=marker,
                linewidth=2.0,
                color=colour,
                label=label,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{label}\nkcal/mol" for label in labels])
        axis.set_xlabel("Post-minimization energy window")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7.5)

    figure.tight_layout()
    target = Path(output_dir) if output_dir else component_dir("optimization_methods")
    return save_figure(
        figure,
        target / filename,
        inputs=loaders.input_paths(table),
        notes="Side quest: reference conformer survival and score vs the energy window.",
    )


__all__ = [
    "figure_drug_panel",
    "figure_egcg_omnibus",
    "figure_isomer_stratification",
    "figure_reference_counts_vs_energy_window",
]
