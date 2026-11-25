"""Publication figures for the large-sample prior cross-molecule & self-recovery study.

Produces five sheets in the study directory:

- ``01_cohort_flexibility.png``         Rigid / intermediate / flexible taxonomy, 3D pose pool
                                        plasticity vs rotatable bonds, and 2D property spread.
- ``02_cross_bands_and_timing.png``     (K x C) grids per method side-by-side: >=p85 (top 15%)
                                        median +/- std, p50-p85 median +/- std, and CPU/wall cost.
- ``03_band_distributions_and_fractions.png`` Noise floor (<p50) vs signal bands vs self-pairs,
                                        quantile-fraction sensitivity, Shape/Color split, flex tier.
- ``04_self_recovery_large_sample.png`` 50-molecule diverse self-recovery across (K, C), Pareto
                                        frontier, and ceiling rate by molecular flexibility.
- ``05_2d_vs_3d_similarity.png``         2D ECFP4 vs 3D ROCS scaffold-hopping quadrant & paired
                                        force-field deltas (MMFF94s / UFF vs ETKDG).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from _david.code.conformer.cross_study import FLEXIBILITY_ORDER, CrossStudyConfig
from _david.code.conformer.figures import nondominated

logger = logging.getLogger(__name__)

METHOD_COLOURS = {
    "ETKDG": "#4B5563",
    "MMFF94s_e78.4": "#2563EB",
    "UFF": "#DC2626",
    "MMFF94_e78.4": "#16A34A",
    "MMFF94s_e1": "#7C3AED",
}
FALLBACK_COLOURS = ("#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0891B2")
FLEX_COLOURS = {
    "rigid": "#0891B2",
    "intermediate": "#2563EB",
    "flexible": "#DC2626",
}
FLEX_LABELS = {
    "rigid": "Rigid (<=2 RotB)",
    "intermediate": "Intermediate (3-6 RotB)",
    "flexible": "Flexible (>=7 RotB)",
}
_PRES_FLEX_LABELS = {
    "rigid": "Rigid\n<=2 RotB",
    "intermediate": "Intermed.\n3-6 RotB",
    "flexible": "Flexible\n>=7 RotB",
}
_PRESENTATION_CALLOUT_CANDIDATES = (
    (12, 14, "left", "bottom"),
    (-12, 16, "right", "bottom"),
    (14, -22, "left", "top"),
    (-12, -22, "right", "top"),
    (0, 34, "center", "bottom"),
    (-16, 34, "right", "bottom"),
    (16, 30, "left", "bottom"),
    (14, -44, "left", "top"),
    (-14, -44, "right", "top"),
    (0, -54, "center", "top"),
    (-18, 52, "right", "bottom"),
    (18, 48, "left", "bottom"),
)


def _style(*, presentation: bool = False) -> None:
    # Canvas grows ~1.8x in inches while fonts grow ~3.6x in pt -> ~2.0-2.2x visually larger text
    # relative to axes, keeping plot boxes spacious and readable for slides.
    scale = 3.6 if presentation else 1.0
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "font.size": 9.5 * scale,
            "axes.titlesize": 10.2 * scale,
            "axes.labelsize": 9.4 * scale,
            "xtick.labelsize": 8.6 * scale,
            "ytick.labelsize": 8.6 * scale,
            "legend.fontsize": 8.2 * scale,
            "axes.linewidth": 2.0 if presentation else 0.8,
            "xtick.major.width": 2.0 if presentation else 0.8,
            "ytick.major.width": 2.0 if presentation else 0.8,
            "xtick.major.size": 8.0 if presentation else 3.5,
            "ytick.major.size": 8.0 if presentation else 3.5,
        }
    )


def _palette(methods: list[str]) -> dict[str, str]:
    return {
        m: METHOD_COLOURS.get(m, FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)])
        for i, m in enumerate(methods)
    }


def _subtitle(summary: dict[str, Any], cfg: CrossStudyConfig | None) -> str:
    n_mol = sum(summary.get("flexibility_counts_1000", {}).values()) or 1000
    n_ref = sum(summary.get("flexibility_counts_refs", {}).values()) or 50
    rep = summary.get("representative_cell", {"K": 4, "C": 20})
    cff = cfg.color_ff if cfg else "default"
    sw = cfg.alignment_shape_weight if cfg else 0.5
    return (
        f"Papyrus05.5 Prior: {n_mol} molecules x {n_ref} random refs "
        f"({n_ref * (n_mol - 1):,} cross + {n_ref} self pairs) | "
        f"colour_ff={cff}, shape_weight={sw:g} | rep cell K={rep['K']}, C={rep['C']}"
    )


def cohort_flexibility(
    molecules: pd.DataFrame, summary: dict[str, Any], cfg: CrossStudyConfig | None, out: Path
) -> Path:
    """Render flexibility tier breakdown, rotatable bond histogram, pose count, and SA/QED."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=200)
    fig.suptitle(
        f"Prior Cohort Molecular Flexibility & 3D Conformer Plasticity\n{_subtitle(summary, cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )

    # Panel A: Tier counts (1000 prior + 50 refs overlay)
    ax = axes[0][0]
    all_counts = [int((molecules["flexibility_class"] == t).sum()) for t in FLEXIBILITY_ORDER]
    ref_counts = [
        int(((molecules["flexibility_class"] == t) & (molecules["is_reference"] == 1)).sum())
        for t in FLEXIBILITY_ORDER
    ]
    x = np.arange(len(FLEXIBILITY_ORDER))
    bars = ax.bar(
        x - 0.18,
        all_counts,
        width=0.36,
        color=[FLEX_COLOURS[t] for t in FLEXIBILITY_ORDER],
        label="All sampled molecules",
    )
    ref_scale = len(molecules) / max(sum(ref_counts), 1)
    ax.bar(
        x + 0.18,
        [c * ref_scale for c in ref_counts],
        width=0.36,
        color=[FLEX_COLOURS[t] for t in FLEXIBILITY_ORDER],
        alpha=0.40,
        hatch="//",
        edgecolor="#111827",
        label="50 refs (rescaled to N)",
    )
    for bar, count in zip(bars, all_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(all_counts) * 0.02,
            f"{count}\n({count / max(len(molecules), 1):.1%})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for xi, rc in zip(x, ref_counts, strict=True):
        ax.text(
            xi + 0.18,
            rc * ref_scale + max(all_counts) * 0.02,
            f"ref={rc}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )
    ax.set_xticks(x, [FLEX_LABELS[t] for t in FLEXIBILITY_ORDER])
    ax.set_ylim(0, max(all_counts) * 1.22)
    ax.set_ylabel("Molecule count")
    ax.set_title("A. Flexibility Class Breakdown (1,000 Prior & 50 Refs)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Rotatable bonds distribution
    ax = axes[0][1]
    rigid_max = cfg.rigid_max_rot if cfg else 2
    flex_min = cfg.flexible_min_rot if cfg else 7
    rot_vals = np.arange(0, int(molecules["rot_bonds"].max()) + 1)
    hist_all = [int((molecules["rot_bonds"] == r).sum()) for r in rot_vals]
    hist_ref = [
        int(((molecules["rot_bonds"] == r) & (molecules["is_reference"] == 1)).sum())
        for r in rot_vals
    ]
    bar_cols = [
        FLEX_COLOURS["rigid"]
        if r <= rigid_max
        else (FLEX_COLOURS["flexible"] if r >= flex_min else FLEX_COLOURS["intermediate"])
        for r in rot_vals
    ]
    ax.bar(rot_vals, hist_all, color=bar_cols, alpha=0.85, edgecolor="#ffffff", linewidth=0.6)
    ax2 = ax.twinx()
    ax2.plot(rot_vals, hist_ref, "o-", color="#111827", linewidth=1.6, markersize=4, label="50 refs count")
    ax2.set_ylabel("Reference subset count", color="#111827")
    ax2.grid(False)
    ax.set_xlabel("Rotatable bonds (RDKit)")
    ax.set_ylabel("Prior cohort count")
    ax.set_title("B. Rotatable Bond Distribution Across Cohort", fontweight="bold")

    # Panel C: Observed 3D conformers kept vs rotatable bonds
    ax = axes[1][0]
    cand_cols = [c for c in molecules.columns if "_cand_confs_c" in c]
    for method_col in sorted(cand_cols):
        m_name, _, c_tag = method_col.partition("_cand_confs_")
        grouped = molecules.groupby("rot_bonds")[method_col].agg(["median", "std"]).reset_index()
        col = METHOD_COLOURS.get(m_name, "#2563EB")
        ax.plot(
            grouped["rot_bonds"],
            grouped["median"],
            "o-",
            color=col,
            label=f"{m_name} cand ({c_tag})",
            linewidth=1.8,
            markersize=4,
        )
        ax.fill_between(
            grouped["rot_bonds"],
            np.maximum(0, grouped["median"] - grouped["std"].fillna(0)),
            grouped["median"] + grouped["std"].fillna(0),
            color=col,
            alpha=0.12,
        )
    ax.axvline(rigid_max + 0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.axvline(flex_min - 0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_xlabel("Rotatable bonds")
    ax.set_ylabel("Unique ETKDG/FF poses kept (RMSD > 0.5 A)")
    ax.set_title("C. Observed 3D Conformational Plasticity vs Rotatable Bonds", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Panel D: Property spread (QED & SA_raw & HAC) across tiers
    ax = axes[1][1]
    data_sa = [molecules[molecules["flexibility_class"] == t]["sa_raw"].to_numpy() for t in FLEXIBILITY_ORDER]
    data_qed = [molecules[molecules["flexibility_class"] == t]["qed"].to_numpy() for t in FLEXIBILITY_ORDER]
    pos_sa = np.arange(len(FLEXIBILITY_ORDER)) * 2.5
    pos_qed = pos_sa + 0.85
    bp1 = ax.boxplot(data_sa, positions=pos_sa, widths=0.65, patch_artist=True)
    for patch, tier in zip(bp1["boxes"], FLEXIBILITY_ORDER, strict=True):
        patch.set_facecolor(FLEX_COLOURS[tier])
        patch.set_alpha(0.65)
    ax.set_ylabel("SA score (raw, lower = easier synthesis)")
    ax_r = ax.twinx()
    bp2 = ax_r.boxplot(data_qed, positions=pos_qed, widths=0.65, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#F59E0B")
        patch.set_alpha(0.55)
    ax_r.set_ylabel("QED [0, 1]", color="#B45309")
    ax_r.grid(False)
    tick_labels = []
    for tier in FLEXIBILITY_ORDER:
        sub_hac = molecules[molecules["flexibility_class"] == tier]["hac"]
        med_hac = float(sub_hac.median()) if not sub_hac.empty else float("nan")
        suffix = f"HAC med={med_hac:.0f}" if np.isfinite(med_hac) else "n=0"
        tick_labels.append(f"{tier.title()}\n({suffix})")
    ax.set_xticks(pos_sa + 0.425, tick_labels)
    ax.set_title("D. Synthetic Accessibility (left) & QED (right, amber) by Tier", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "01_cohort_flexibility.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_k(sec: float) -> str:
    return f"{sec / 1000:.1f}k" if sec >= 1000 else f"{sec:.0f}"


def _pivot_grid(
    frame: pd.DataFrame, column: str, counts: list[int], caps: list[int]
) -> np.ndarray:
    return (
        frame.pivot(index="candidate_count", columns="reference_cap", values=column)
        .reindex(index=counts, columns=caps)
        .to_numpy(dtype=float)
    )


def _render_cross_bands_sheet(
    grid: pd.DataFrame,
    timing: pd.DataFrame,
    summary: dict[str, Any],
    cfg: CrossStudyConfig | None,
    out_path: Path,
    *,
    top_band: str,
    top_label: str,
    rest_band: str,
    rest_label: str,
) -> Path:
    _style()
    methods = sorted(grid["method"].unique())
    caps = sorted(grid["reference_cap"].unique())
    counts = sorted(grid["candidate_count"].unique())

    g_self = grid[(grid["scope"] == "self_50") & (grid["band"] == "all_self")]
    g_cross = grid[grid["scope"] == "global_cross"]
    b_top = g_cross[g_cross["band"] == top_band]
    b_rest = g_cross[g_cross["band"] == rest_band]
    if b_rest.empty:
        b_rest = g_cross[g_cross["band"] == "p50_to_p85"]

    fig, axes = plt.subplots(
        len(methods), 4, figsize=(24.0, 3.7 * len(methods) + 1.3), dpi=200, squeeze=False
    )
    fig.suptitle(
        f"Self-Similarity vs. Cross-Molecule ROCS Bands ({top_label} vs. {rest_label}) & Runtime Grid\n"
        f"{_subtitle(summary, cfg)} | values show median +/- std (TanimotoCombo [0, 2])",
        fontsize=11.5,
        fontweight="bold",
    )

    vmin_self = float(g_self["median_combo"].min()) if not g_self.empty else 0.0
    vmax_self = float(g_self["median_combo"].max()) if not g_self.empty else 2.0
    vmin_top, vmax_top = float(b_top["median_combo"].min()), float(b_top["median_combo"].max())
    vmin_rest, vmax_rest = float(b_rest["median_combo"].min()), float(b_rest["median_combo"].max())
    vmin_t = float(timing["total_cand_plus_rocs_cpu_s"].min())
    vmax_t = float(timing["total_cand_plus_rocs_cpu_s"].max())

    for row_i, method in enumerate(methods):
        panels = (
            (g_self[g_self["method"] == method], "Identical Molecule (Self, n=50)", "PuBuGn", vmin_self, vmax_self, False),
            (b_top[b_top["method"] == method], top_label, "YlGn", vmin_top, vmax_top, False),
            (b_rest[b_rest["method"] == method], rest_label, "YlGnBu", vmin_rest, vmax_rest, False),
            (timing[timing["method"] == method], "Runtime: CPU s (cand+ROCS) / R:ROCS | Wall", "magma_r", vmin_t, vmax_t, True),
        )
        for col_i, (sub_df, title_tag, cmap, vmin, vmax, is_time) in enumerate(panels):
            ax = axes[row_i][col_i]
            val_col = "total_cand_plus_rocs_cpu_s" if is_time else "median_combo"
            mat = _pivot_grid(sub_df, val_col, counts, caps)

            if is_time:
                piv_wall = _pivot_grid(sub_df, "total_wall_s", counts, caps)
                piv_rocs = _pivot_grid(sub_df, "rocs_cpu_s", counts, caps)
                piv_ms = _pivot_grid(sub_df, "total_cpu_ms_per_pair", counts, caps)
            else:
                piv_std = _pivot_grid(sub_df, "std_combo", counts, caps)

            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            span = (vmax - vmin) or 1.0
            for r in range(len(counts)):
                for c in range(len(caps)):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    dark = (val - vmin) / span > 0.58
                    if is_time:
                        txt = f"{_fmt_k(val)}s CPU"
                        if piv_rocs[r, c] > 0:
                            txt += f"\nR:{_fmt_k(piv_rocs[r, c])}s | {piv_wall[r, c]:.0f}s"
                        else:
                            txt += f"\n{piv_wall[r, c]:.0f}s wall"
                        txt += f"\n{piv_ms[r, c]:.0f} ms/pr"
                    else:
                        txt = f"{val:.3f}\n+/- {piv_std[r, c]:.3f}"
                    ax.text(
                        c,
                        r,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=6.8 if is_time else 7.8,
                        fontweight="bold",
                        color="#ffffff" if dark else "#111827",
                    )
            ax.set_xticks(range(len(caps)), [f"K={k}" for k in caps])
            ax.set_yticks(range(len(counts)), [f"C={c}" for c in counts])
            ax.set_xlabel("Reference bank cap (K)")
            ax.set_ylabel("Candidate attempts (C)")
            ax.set_title(f"{method} | {title_tag}", fontsize=9.5, fontweight="bold")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cross_bands_and_timing(
    grid: pd.DataFrame,
    timing: pd.DataFrame,
    summary: dict[str, Any],
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    """Render side-by-side self/cross band heatmaps and runtime grids (p85 and optional p75)."""
    path_p85 = _render_cross_bands_sheet(
        grid,
        timing,
        summary,
        cfg,
        out / "02_cross_bands_and_timing.png",
        top_band="ge_p85",
        top_label="Top 15% Cross (>= p85)",
        rest_band="lt_p85",
        rest_label="Rest 85% Non-Desired (< p85)",
    )
    if "lt_p75" in set(grid["band"]):
        _render_cross_bands_sheet(
            grid,
            timing,
            summary,
            cfg,
            out / "02_cross_bands_and_timing_p75.png",
            top_band="ge_p75",
            top_label="Top 25% Cross (>= p75)",
            rest_band="lt_p75",
            rest_label="Rest 75% Non-Desired (< p75)",
        )
    return path_p85


def band_distributions_and_fractions(
    grid: pd.DataFrame,
    rep_pairs: pd.DataFrame,
    summary: dict[str, Any],
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    """Render noise vs signal bands, cutoff sensitivity, shape/colour split, and flexibility effect."""
    _ = rep_pairs
    _style()
    methods = sorted(grid["method"].unique())
    palette = _palette(methods)
    rep = summary.get("representative_cell", {"K": 4, "C": 20})
    rep_k, rep_c = int(rep["K"]), int(rep["C"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    fig.suptitle(
        f"Score Bands, Cutoff Fraction Sensitivity, Shape/Color Split & Flexibility Effect "
        f"(K={rep_k}, C={rep_c})\n{_subtitle(summary, cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )

    # Panel A: Signal vs Noise bands + Self-pairs at representative (K, C)
    ax = axes[0][0]
    band_keys = ["lt_p50_noise", "p50_to_p85", "ge_p85", "self_50"]
    band_labels = ["< p50 (Noise)", "p50 - p85", ">= p85 (Top 15%)", "Self (50 refs)"]
    x = np.arange(len(band_keys))
    width = 0.78 / max(len(methods), 1)
    for mi, m in enumerate(methods):
        meds, stds = [], []
        sub_g = grid[
            (grid["method"] == m)
            & (grid["reference_cap"] == rep_k)
            & (grid["candidate_count"] == rep_c)
        ]
        for bk in band_keys:
            row = (
                sub_g[(sub_g["scope"] == "self_50") & (sub_g["band"] == "all_self")]
                if bk == "self_50"
                else sub_g[(sub_g["scope"] == "global_cross") & (sub_g["band"] == bk)]
            )
            meds.append(float(row["median_combo"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std_combo"].iloc[0]) if not row.empty else np.nan)
        offs = (mi - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offs, meds, width=width * 0.92, yerr=stds, capsize=3,
            color=palette[m], alpha=0.88, label=m,
        )
        for bar, med, sd in zip(bars, meds, stds, strict=True):
            if np.isfinite(med):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    med + (sd if np.isfinite(sd) else 0) + 0.025,
                    f"{med:.2f}\n+-{sd:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                )
    ax.set_xticks(x, band_labels)
    ax.set_ylim(0, 2.48)
    ax.set_ylabel("TanimotoCombo [0, 2]  (median +/- std)")
    ax.set_title("A. Noise Floor (<p50) vs Signal Bands vs Self-Recovery", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Cutoff fraction sensitivity
    ax = axes[0][1]
    frac_bands = [
        ("ge_p95", "Top 5%\n(>=p95)"),
        ("ge_p90", "Top 10%\n(>=p90)"),
        ("ge_p85", "Top 15%\n(>=p85)"),
        ("ge_p75", "Top 25%\n(>=p75)"),
        ("p75_to_p95", "75-95%"),
        ("p50_to_p90", "50-90%"),
        ("p50_to_p85", "50-85%"),
    ]
    xf = np.arange(len(frac_bands))
    for mi, m in enumerate(methods):
        sub_g = grid[
            (grid["method"] == m)
            & (grid["reference_cap"] == rep_k)
            & (grid["candidate_count"] == rep_c)
            & (grid["scope"] == "global_cross")
        ]
        meds, stds = [], []
        for bk, _ in frac_bands:
            row = sub_g[sub_g["band"] == bk]
            meds.append(float(row["median_combo"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std_combo"].iloc[0]) if not row.empty else np.nan)
        jitter = (mi - (len(methods) - 1) / 2) * 0.12
        ax.errorbar(
            xf + jitter, meds, yerr=stds, fmt="o-", color=palette[m],
            label=m, capsize=3.5, linewidth=1.7, markersize=5,
        )
    ax.set_xticks(xf, [lbl for _, lbl in frac_bands])
    ax.set_ylabel("TanimotoCombo [0, 2]  (median +/- std)")
    ax.set_title("B. Fraction / Cutoff Sensitivity (Different Tail & Band Slices)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Panel C: Shape vs Color decomposition stacked bar across bands
    ax = axes[1][0]
    comp_bands = [("p50_to_p85", "50-85%"), ("ge_p85", ">= p85"), ("self_50", "Self (50)")]
    xc = np.arange(len(comp_bands))
    for mi, m in enumerate(methods):
        sub_g = grid[
            (grid["method"] == m)
            & (grid["reference_cap"] == rep_k)
            & (grid["candidate_count"] == rep_c)
        ]
        shapes, cols = [], []
        for bk, _ in comp_bands:
            row = (
                sub_g[(sub_g["scope"] == "self_50") & (sub_g["band"] == "all_self")]
                if bk == "self_50"
                else sub_g[(sub_g["scope"] == "global_cross") & (sub_g["band"] == bk)]
            )
            shapes.append(float(row["median_shape"].iloc[0]) if not row.empty else 0.0)
            cols.append(float(row["median_colour"].iloc[0]) if not row.empty else 0.0)
        offs = (mi - (len(methods) - 1) / 2) * width
        ax.bar(xc + offs, shapes, width=width * 0.9, color=palette[m], alpha=0.92)
        ax.bar(
            xc + offs, cols, bottom=shapes, width=width * 0.9,
            color=palette[m], alpha=0.42, hatch="///", edgecolor="#111827",
        )
        for bx, sh, co in zip(xc + offs, shapes, cols, strict=True):
            ax.text(bx, sh / 2, f"{sh:.2f}", ha="center", va="center", fontsize=6.8, color="#ffffff", fontweight="bold")
            ax.text(bx, sh + co / 2, f"{co:.2f}", ha="center", va="center", fontsize=6.8, color="#111827", fontweight="bold")
    ax.set_xticks(xc, [lbl for _, lbl in comp_bands])
    ax.set_ylim(0, 2.25)
    ax.set_ylabel("Shape [0, 1] (solid) + Colour [0, 1] (hatched)")
    ax.set_title("C. Shape vs Colour Contribution Across Bands & Methods", fontweight="bold")
    handles_c = [Patch(facecolor=palette[m], label=m) for m in methods] + [
        Patch(facecolor="#6B7280", alpha=0.92, label="Shape (solid)"),
        Patch(facecolor="#6B7280", alpha=0.42, hatch="///", edgecolor="#111827", label="Colour (hatched)"),
    ]
    ax.legend(handles=handles_c, fontsize=7.5, loc="upper left", ncol=2)

    # Panel D: Cross-band score by candidate flexibility class
    ax = axes[1][1]
    xt = np.arange(len(FLEXIBILITY_ORDER))
    for mi, m in enumerate(methods):
        sub_g = grid[
            (grid["method"] == m)
            & (grid["reference_cap"] == rep_k)
            & (grid["candidate_count"] == rep_c)
        ]
        m85, s85, mmid, smid = [], [], [], []
        for tier in FLEXIBILITY_ORDER:
            r85 = sub_g[(sub_g["scope"] == f"cand_{tier}_cross") & (sub_g["band"] == "ge_p85")]
            rmid = sub_g[(sub_g["scope"] == f"cand_{tier}_cross") & (sub_g["band"] == "p50_to_p85")]
            m85.append(float(r85["median_combo"].iloc[0]) if not r85.empty else np.nan)
            s85.append(float(r85["std_combo"].iloc[0]) if not r85.empty else np.nan)
            mmid.append(float(rmid["median_combo"].iloc[0]) if not rmid.empty else np.nan)
            smid.append(float(rmid["std_combo"].iloc[0]) if not rmid.empty else np.nan)
        jitter = (mi - (len(methods) - 1) / 2) * 0.1
        ax.errorbar(xt + jitter, m85, yerr=s85, fmt="o-", color=palette[m], label=f"{m} (>=p85)", capsize=3, linewidth=1.7)
        ax.errorbar(xt + jitter, mmid, yerr=smid, fmt="s--", color=palette[m], alpha=0.65, label=f"{m} (50-85%)", capsize=3, linewidth=1.3)
    ax.set_xticks(xt, [FLEX_LABELS[t] for t in FLEXIBILITY_ORDER])
    ax.set_ylabel("TanimotoCombo [0, 2]  (median +/- std)")
    ax.set_title("D. Cross-Matching Score by Candidate Flexibility Class", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "03_band_distributions_and_fractions.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_self_curve(
    ax: Axes,
    sliced: pd.DataFrame,
    methods: list[str],
    palette: dict[str, str],
    x_col: str,
    *,
    lw: float,
    ms: float,
    alpha: float,
    xlabel: str,
    title: str,
    presentation: bool,
) -> None:
    for m in methods:
        g = (
            sliced[sliced["method"] == m]
            .groupby(x_col)["_combo"]
            .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        )
        g.columns = ["med", "q25", "q75"]
        ax.plot(g.index, g["med"], "o-", color=palette[m], label=m, linewidth=lw, markersize=ms)
        ax.fill_between(g.index, g["q25"], g["q75"], color=palette[m], alpha=alpha)
    ax.set_xlabel(xlabel, labelpad=12 if presentation else 4)
    ax.set_ylabel(
        "TanimotoCombo [0, 2]" if presentation else "Self-recovery TanimotoCombo  [0, 2]",
        labelpad=10 if presentation else 4,
    )
    ax.set_title(title, fontweight="bold", pad=14 if presentation else 6)
    if not presentation:
        ax.legend(fontsize=8.0, loc="lower right", frameon=True)


def _place_presentation_pareto_labels(
    ax: Axes, front: pd.DataFrame, scale: float
) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    log_xmin = float(np.log10(max(xmin, 1e-9)))
    log_span = float(np.log10(max(xmax, 1e-9)) - log_xmin) or 1.0
    y_span = float(ymax - ymin) or 1.0
    front_pts = [
        (
            float((np.log10(max(r["total_cpu_ms_per_pair"], 1e-9)) - log_xmin) / log_span),
            float((r["_combo"] - ymin) / y_span),
        )
        for _, r in front.iterrows()
    ]
    placed_boxes: list[tuple[float, float]] = []
    for idx, (_, r) in enumerate(front.iterrows()):
        px, py = front_pts[idx]
        lbl = f"{r['method'][:4]} K{int(r['reference_cap'])}/C{int(r['candidate_count'])}"
        best_choice = _PRESENTATION_CALLOUT_CANDIDATES[0]
        best_score = -1e9
        for dx, dy, ha, va in _PRESENTATION_CALLOUT_CANDIDATES:
            off_x = (dx + (-26 if ha == "right" else (26 if ha == "left" else 0))) / 650.0
            off_y = (dy + (10 if va == "bottom" else -10)) / 480.0
            cx, cy = px + off_x, py + off_y
            if not (0.04 <= cx <= 0.96 and 0.06 <= cy <= 0.96):
                continue
            min_box_d = min(
                (max(abs(cx - bx) / 0.145, abs(cy - by) / 0.055) for bx, by in placed_boxes),
                default=5.0,
            )
            min_pt_d = min(
                (
                    max(abs(cx - ox) / 0.08, abs(cy - oy) / 0.04)
                    for j, (ox, oy) in enumerate(front_pts)
                    if j != idx
                ),
                default=5.0,
            )
            score = min(min_box_d, 1.8) * 2.0 + min(min_pt_d, 1.5) - 0.002 * (abs(dx) + abs(dy))
            if score > best_score:
                best_score = score
                best_choice = (dx, dy, ha, va)
        dx, dy, ha, va = best_choice
        off_x = (dx + (-26 if ha == "right" else (26 if ha == "left" else 0))) / 650.0
        off_y = (dy + (10 if va == "bottom" else -10)) / 480.0
        placed_boxes.append((px + off_x, py + off_y))
        arrow = (
            {"arrowstyle": "-", "color": "#6B7280", "lw": 1.6, "shrinkA": 2, "shrinkB": 8}
            if abs(dx) >= 14 or abs(dy) >= 24
            else None
        )
        ax.annotate(
            lbl,
            (r["total_cpu_ms_per_pair"], r["_combo"]),
            fontsize=6.0 * scale,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            arrowprops=arrow,
        )


def _place_standard_pareto_labels(ax: Axes, front: pd.DataFrame) -> None:
    placed: list[tuple[float, float]] = []
    for _, r in front.iterrows():
        lx = float(np.log10(max(r["total_cpu_ms_per_pair"], 1e-9)))
        ly = float(r["_combo"])
        close_cnt = sum(1 for px, py in placed if abs(lx - px) < 0.25 and abs(ly - py) < 0.06)
        if ly >= 1.98:
            dx, dy = [(-6, 6), (6, -9), (6, 5), (-6, -9)][close_cnt % 4]
        else:
            dx, dy = [(5, 3), (5, -9), (-5, 4)][close_cnt % 3]
        ax.annotate(
            f"{r['method'][:4]} K{int(r['reference_cap'])}/C{int(r['candidate_count'])}",
            (r["total_cpu_ms_per_pair"], r["_combo"]),
            fontsize=6.8,
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
        )
        placed.append((lx, ly))


def _plot_self_pareto(
    ax: Axes,
    self_plot: pd.DataFrame,
    timing: pd.DataFrame,
    methods: list[str],
    palette: dict[str, str],
    *,
    presentation: bool,
    scale: float,
) -> None:
    med_self = self_plot.groupby(
        ["method", "reference_cap", "candidate_count"], as_index=False
    )["_combo"].median()
    pareto_df = med_self.merge(
        timing[
            [
                "method",
                "reference_cap",
                "candidate_count",
                "total_cpu_ms_per_pair",
                "cand_embed_cpu_s",
                "cand_relax_cpu_s",
            ]
        ],
        on=["method", "reference_cap", "candidate_count"],
    )
    for m in methods:
        part = pareto_df[pareto_df["method"] == m]
        ax.scatter(
            part["total_cpu_ms_per_pair"],
            part["_combo"],
            color=palette[m],
            label=m,
            s=170 if presentation else 36,
            alpha=0.85,
        )
    nd = nondominated(
        pareto_df["total_cpu_ms_per_pair"].to_numpy(), pareto_df["_combo"].to_numpy()
    )
    front = pareto_df[nd].sort_values("total_cpu_ms_per_pair").reset_index(drop=True)
    ax.scatter(
        front["total_cpu_ms_per_pair"],
        front["_combo"],
        facecolors="none",
        edgecolors="#111827",
        s=580 if presentation else 135,
        linewidths=3.2 if presentation else 1.3,
        label="non-dominated",
    )
    ax.set_xscale("log")
    ax.set_ylim(top=2.16 if presentation else 2.064)
    if presentation:
        _place_presentation_pareto_labels(ax, front, scale)
    else:
        _place_standard_pareto_labels(ax, front)
    ax.set_xlabel(
        "CPU ms per pair (cand + ROCS)"
        if presentation
        else "Measured CPU ms per (ref, cand) pair (cand + ROCS)",
        labelpad=12 if presentation else 4,
    )
    ax.set_ylabel(
        "Median TanimotoCombo [0, 2]"
        if presentation
        else "Median TanimotoCombo across 50 refs  [0, 2]",
        labelpad=10 if presentation else 4,
    )
    ax.set_title(
        "C. Self-Recovery Pareto Frontier"
        if presentation
        else "C. Self-Recovery Pareto Frontier Across 50 Prior Molecules",
        fontweight="bold",
        pad=14 if presentation else 6,
    )
    if presentation:
        handles, labels = ax.get_legend_handles_labels()
        nd_idx = labels.index("non-dominated")
        ax.legend(
            [handles[nd_idx]], ["non-dominated"], fontsize=7.0 * scale, loc="lower right", frameon=True
        )
    else:
        ax.legend(fontsize=7.5, loc="lower right", frameon=True)


def _plot_self_flexibility_bars(
    ax: Axes,
    self_plot: pd.DataFrame,
    methods: list[str],
    palette: dict[str, str],
    fixed_k: int,
    fixed_c: int,
    *,
    presentation: bool,
    scale: float,
) -> None:
    cell = self_plot[
        (self_plot["reference_cap"] == fixed_k) & (self_plot["candidate_count"] == fixed_c)
    ]
    xt = np.arange(len(FLEXIBILITY_ORDER))
    w = 0.78 / max(len(methods), 1)
    for mi, m in enumerate(methods):
        meds, ceils = [], []
        for tier in FLEXIBILITY_ORDER:
            sub = cell[(cell["method"] == m) & (cell["flexibility_class"] == tier)]
            meds.append(float(sub["_combo"].median()) if not sub.empty else np.nan)
            ceils.append(float(sub["at_ceiling"].mean()) if not sub.empty else 0.0)
        offs = (mi - (len(methods) - 1) / 2) * w
        bars = ax.bar(xt + offs, meds, width=w * 0.9, color=palette[m], alpha=0.88, label=m)
        for bar, med, ceil in zip(bars, meds, ceils, strict=True):
            if np.isfinite(med):
                bar_txt = f"{med:.2f}\n{ceil:.0%}" if presentation else f"{med:.3f}\n({ceil:.0%}@2)"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    med + (0.035 if presentation else 0.024),
                    bar_txt,
                    ha="center",
                    va="bottom",
                    fontsize=(6.2 * scale) if presentation else 6.8,
                )
    ax.set_xticks(
        xt,
        [_PRES_FLEX_LABELS[t] if presentation else FLEX_LABELS[t] for t in FLEXIBILITY_ORDER],
    )
    ax.set_ylim(0, 2.52 if presentation else 2.36)
    ax.set_ylabel(
        "Median TanimotoCombo (%@2)"
        if presentation
        else "Median TanimotoCombo [0, 2]  (% at combo=2.0 ceiling)",
        labelpad=10 if presentation else 4,
    )
    ax.set_title(
        f"D. By Flexibility (K={fixed_k}, C={fixed_c})"
        if presentation
        else f"D. Self-Recovery & Pose-Collapse Ceiling by Flexibility (K={fixed_k}, C={fixed_c})",
        fontweight="bold",
        pad=14 if presentation else 6,
    )
    if not presentation:
        ax.legend(fontsize=8.0, loc="upper right", frameon=True)


def self_recovery_large_sample(
    self_df: pd.DataFrame,
    timing: pd.DataFrame,
    summary: dict[str, Any],
    cfg: CrossStudyConfig | None,
    out: Path,
    *,
    presentation: bool = False,
) -> Path:
    """Render the 4-panel 50-molecule self-recovery summary (standard or slide layout)."""
    _style(presentation=presentation)
    scale = 3.6 if presentation else 1.0
    methods = sorted(self_df["method"].unique())
    palette = _palette(methods)
    caps = sorted(self_df["reference_cap"].unique())
    counts = sorted(self_df["candidate_count"].unique())
    fixed_k = min(caps, key=lambda k: abs(k - (cfg.representative_k if cfg else 4)))
    fixed_c = min(counts, key=lambda c: abs(c - (cfg.representative_c if cfg else 20)))

    figsize = (28.0, 21.0) if presentation else (15.5, 10.5)
    dpi = 150 if presentation else 200
    lw = 4.8 if presentation else 2.0
    ms = 12.0 if presentation else 5.0
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

    if presentation:
        n_mol = sum(summary.get("flexibility_counts_1000", {}).values()) or 1000
        n_ref = sum(summary.get("flexibility_counts_refs", {}).values()) or 50
        rep = summary.get("representative_cell", {"K": 4, "C": 20})
        cff = cfg.color_ff if cfg else "default"
        sw = cfg.alignment_shape_weight if cfg else 0.5
        suptitle = (
            "50-Molecule Self-Shape Recovery Across Methods & Flexibility\n"
            f"Papyrus05.5: {n_mol} mols x {n_ref} refs | colour_ff={cff}, w={sw:g} | "
            f"K={rep['K']}, C={rep['C']}"
        )
    else:
        suptitle = (
            "50-Molecule Diverse Self-Shape Recovery Across Methods & Flexibility Tiers\n"
            f"{_subtitle(summary, cfg)}"
        )
    fig.suptitle(
        suptitle,
        fontsize=(10.2 * scale) if presentation else (11.5 * scale),
        fontweight="bold",
        y=0.985 if presentation else 0.98,
    )

    self_scores = (
        self_df["combo"] if "combo" in self_df.columns else self_df["self_recovery"] * 2.0
    )
    self_plot = self_df.assign(_combo=self_scores)

    _plot_self_curve(
        axes[0][0],
        self_plot[self_plot["reference_cap"] == fixed_k],
        methods,
        palette,
        "candidate_count",
        lw=lw,
        ms=ms,
        alpha=0.14,
        xlabel=(
            "Candidate budget per isomer (C)"
            if presentation
            else "Candidate attempts per stereoisomer (C)"
        ),
        title=(
            f"A. Self-Recovery vs C (K={fixed_k})"
            if presentation
            else f"A. Self-Recovery vs Candidate Budget C (K={fixed_k}, median & IQR)"
        ),
        presentation=presentation,
    )
    _plot_self_curve(
        axes[0][1],
        self_plot[self_plot["candidate_count"] == fixed_c],
        methods,
        palette,
        "reference_cap",
        lw=lw,
        ms=ms,
        alpha=0.12,
        xlabel="Reference bank cap (K)",
        title=(
            f"B. Self-Recovery vs K (C={fixed_c})"
            if presentation
            else f"B. Self-Recovery vs Bank Cap K (C={fixed_c}, median & IQR)"
        ),
        presentation=presentation,
    )
    _plot_self_pareto(
        axes[1][0], self_plot, timing, methods, palette, presentation=presentation, scale=scale
    )
    _plot_self_flexibility_bars(
        axes[1][1],
        self_plot,
        methods,
        palette,
        fixed_k,
        fixed_c,
        presentation=presentation,
        scale=scale,
    )

    out.mkdir(parents=True, exist_ok=True)
    if presentation:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=len(methods),
            frameon=True,
            fontsize=8.2 * scale,
        )
        fig.subplots_adjust(
            left=0.075, right=0.98, bottom=0.065, top=0.815, wspace=0.25, hspace=0.36
        )
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "04_self_recovery_large_sample.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def similarity_2d_vs_3d(
    rep_pairs: pd.DataFrame,
    summary: dict[str, Any],
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    """Render 2D ECFP4 vs 3D ROCS scatter and paired force-field relaxation deltas vs ETKDG."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=200)
    rep = summary.get("representative_cell", {"K": 4, "C": 20})
    fig.suptitle(
        f"2D Fingerprint vs 3D ROCS Similarity & Force-Field Pairing Effect "
        f"(K={rep['K']}, C={rep['C']})\n{_subtitle(summary, cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )

    combo_cols = [c for c in rep_pairs.columns if c.endswith("_combo")]
    primary = "MMFF94s_e78.4_combo" if "MMFF94s_e78.4_combo" in combo_cols else combo_cols[0]
    cross = rep_pairs[rep_pairs["is_self"] == 0]
    self_p = rep_pairs[rep_pairs["is_self"] == 1]

    ax = axes[0]
    hb = ax.hexbin(
        cross["ecfp4_tanimoto"], cross[primary], gridsize=55, cmap="viridis", mincnt=1, bins="log"
    )
    fig.colorbar(hb, ax=ax, label="log10(cross pair count)")
    p85 = float(cross[primary].quantile(0.85))
    p50 = float(cross[primary].quantile(0.50))
    ax.axhline(p85, color="#EF4444", linestyle="--", linewidth=1.4, label=f"3D p85 = {p85:.3f}")
    ax.axhline(p50, color="#F59E0B", linestyle=":", linewidth=1.3, label=f"3D p50 = {p50:.3f}")
    ax.axvline(
        0.30, color="#111827", linestyle="--", linewidth=1.1, label="ECFP4 = 0.30 (scaffold hop)"
    )
    hop_share = float(
        ((cross[primary] >= p85) & (cross["ecfp4_tanimoto"] < 0.30)).sum()
        / max((cross[primary] >= p85).sum(), 1)
    )
    ax.text(
        0.03,
        p85 + 0.06,
        f"Scaffold-hop share of >=p85 hits:\n{hop_share:.1%} have ECFP4 < 0.30",
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#E5E7EB"},
    )
    ax.scatter(
        self_p["ecfp4_tanimoto"],
        self_p[primary],
        color="#DC2626",
        edgecolor="#ffffff",
        s=32,
        zorder=5,
        label="50 self-pairs",
    )
    ax.set_xlabel("2D Morgan ECFP4 Tanimoto")
    ax.set_ylabel(f"3D {primary.replace('_combo', '')} TanimotoCombo [0, 2]")
    ax.set_title("A. 2D Fingerprint vs 3D ROCS Similarity", fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower right")

    ax = axes[1]
    if "ETKDG_combo" in combo_cols:
        other_cols = [c for c in combo_cols if c != "ETKDG_combo"]
        p85_etkdg = float(cross["ETKDG_combo"].quantile(0.85))
        top_etkdg_mask = cross["ETKDG_combo"] >= p85_etkdg
        data_boxes, labels = [], []
        for oc in other_cols:
            m_label = oc.replace("_combo", "")
            delta_cross = cross[oc] - cross["ETKDG_combo"]
            delta_top = delta_cross[top_etkdg_mask]
            delta_self = self_p[oc] - self_p["ETKDG_combo"]
            data_boxes.extend(
                [delta_cross.to_numpy(), delta_top.to_numpy(), delta_self.to_numpy()]
            )
            labels.extend(
                [f"{m_label}\nAll Cross", f"{m_label}\n>=p85 ETKDG", f"{m_label}\nSelf (50)"]
            )
        bp = ax.boxplot(data_boxes, tick_labels=labels, showfliers=False, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            col = METHOD_COLOURS.get(other_cols[i // 3].replace("_combo", ""), "#2563EB")
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.1)
        ax.set_ylabel("Paired delta TanimotoCombo (FF - ETKDG)")
        ax.set_title("B. Paired Effect of Force-Field Relaxation vs Raw ETKDG", fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
    else:
        ax.text(0.5, 0.5, "ETKDG baseline column not present", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    path = out / "05_2d_vs_3d_similarity.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(study_dir: Path, cfg: CrossStudyConfig | None = None) -> list[Path]:
    """Render all five cross-study figure sheets into ``study_dir``."""
    molecules = pd.read_csv(study_dir / "molecules.tsv", sep="\t")
    grid = pd.read_csv(study_dir / "cross_grid_summary.tsv", sep="\t")
    timing = pd.read_csv(study_dir / "timing_summary.tsv", sep="\t")
    self_df = pd.read_csv(study_dir / "self_recovery.tsv", sep="\t")
    rep_pairs = pd.read_csv(study_dir / "default_cell_pairs.tsv", sep="\t")
    summary = json.loads((study_dir / "sampling_summary.json").read_text(encoding="utf-8"))

    written: list[Path] = []
    sheets = (
        ("01_cohort_flexibility", cohort_flexibility, (molecules, summary, cfg, study_dir)),
        ("02_cross_bands_and_timing", cross_bands_and_timing, (grid, timing, summary, cfg, study_dir)),
        (
            "03_band_distributions_and_fractions",
            band_distributions_and_fractions,
            (grid, rep_pairs, summary, cfg, study_dir),
        ),
        (
            "04_self_recovery_large_sample",
            self_recovery_large_sample,
            (self_df, timing, summary, cfg, study_dir),
        ),
        ("05_2d_vs_3d_similarity", similarity_2d_vs_3d, (rep_pairs, summary, cfg, study_dir)),
    )
    for name, fn, args in sheets:
        try:
            path = fn(*args)
            if path is not None:
                written.append(path)
        except Exception:
            logger.exception("figure %s failed", name)
    logger.info("wrote %d cross-study figures to %s", len(written), study_dir)
    return written
