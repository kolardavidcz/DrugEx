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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.code.conformer.cross_study import FLEXIBILITY_ORDER, CrossStudyConfig

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


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "font.size": 9.5,
        }
    )


def _palette(methods: list[str]) -> dict[str, str]:
    out = {}
    for i, m in enumerate(methods):
        out[m] = METHOD_COLOURS.get(m, FALLBACK_COLOURS[i % len(FALLBACK_COLOURS)])
    return out


def _subtitle(summary: dict, cfg: CrossStudyConfig | None) -> str:
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
    molecules: pd.DataFrame, summary: dict, cfg: CrossStudyConfig | None, out: Path
) -> Path:
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
    bars = ax.bar(x - 0.18, all_counts, width=0.36, color=[FLEX_COLOURS[t] for t in FLEXIBILITY_ORDER], label="All sampled molecules")
    ax.bar(
        x + 0.18,
        [c * (len(molecules) / max(sum(ref_counts), 1)) for c in ref_counts],
        width=0.36,
        color=[FLEX_COLOURS[t] for t in FLEXIBILITY_ORDER],
        alpha=0.40,
        hatch="//",
        edgecolor="#111827",
        label="50 refs (rescaled to N)",
    )
    for bar, count in zip(bars, all_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + max(all_counts) * 0.02,
            f"{count}\n({count / max(len(molecules), 1):.1%})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    for xi, rc in zip(x, ref_counts):
        scaled = rc * (len(molecules) / max(sum(ref_counts), 1))
        ax.text(
            xi + 0.18,
            scaled + max(all_counts) * 0.02,
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
    rot_vals = np.arange(0, int(molecules["rot_bonds"].max()) + 1)
    hist_all = [int((molecules["rot_bonds"] == r).sum()) for r in rot_vals]
    hist_ref = [int(((molecules["rot_bonds"] == r) & (molecules["is_reference"] == 1)).sum()) for r in rot_vals]
    bar_cols = [
        FLEX_COLOURS["rigid"]
        if r <= (cfg.rigid_max_rot if cfg else 2)
        else (FLEX_COLOURS["flexible"] if r >= (cfg.flexible_min_rot if cfg else 7) else FLEX_COLOURS["intermediate"])
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
    cand_cols = [c for c in molecules.columns if c.endswith("_cand_confs_c30") or "_cand_confs_c" in c]
    if cand_cols:
        for method_col in sorted(cand_cols):
            m_name, _, c_tag = method_col.partition("_cand_confs_")
            grouped = molecules.groupby("rot_bonds")[method_col].agg(["median", "std"]).reset_index()
            col = METHOD_COLOURS.get(m_name, "#2563EB")
            ax.plot(grouped["rot_bonds"], grouped["median"], "o-", color=col, label=f"{m_name} cand ({c_tag})", linewidth=1.8, markersize=4)
            ax.fill_between(
                grouped["rot_bonds"],
                np.maximum(0, grouped["median"] - grouped["std"].fillna(0)),
                grouped["median"] + grouped["std"].fillna(0),
                color=col,
                alpha=0.12,
            )
    ax.axvline((cfg.rigid_max_rot if cfg else 2) + 0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.axvline((cfg.flexible_min_rot if cfg else 7) - 0.5, color="#6B7280", linestyle="--", linewidth=1)
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
    for p, t in zip(bp1["boxes"], FLEXIBILITY_ORDER):
        p.set_facecolor(FLEX_COLOURS[t])
        p.set_alpha(0.65)
    ax.set_ylabel("SA score (raw, lower = easier synthesis)")
    ax_r = ax.twinx()
    bp2 = ax_r.boxplot(data_qed, positions=pos_qed, widths=0.65, patch_artist=True)
    for p in bp2["boxes"]:
        p.set_facecolor("#F59E0B")
        p.set_alpha(0.55)
    ax_r.set_ylabel("QED [0, 1]", color="#B45309")
    ax_r.grid(False)
    tick_labels = []
    for t in FLEXIBILITY_ORDER:
        sub_hac = molecules[molecules["flexibility_class"] == t]["hac"]
        med_hac = float(sub_hac.median()) if not sub_hac.empty else float("nan")
        suffix = f"HAC med={med_hac:.0f}" if np.isfinite(med_hac) else "n=0"
        tick_labels.append(f"{t.title()}\n({suffix})")
    ax.set_xticks(pos_sa + 0.425, tick_labels)
    ax.set_title("D. Synthetic Accessibility (left) & QED (right, amber) by Tier", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "01_cohort_flexibility.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_k(sec: float) -> str:
    if sec >= 1000:
        return f"{sec / 1000:.1f}k"
    return f"{sec:.0f}"


def cross_bands_and_timing(
    grid: pd.DataFrame,
    timing: pd.DataFrame,
    summary: dict,
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    _style()
    methods = sorted(grid["method"].unique())
    caps = sorted(grid["reference_cap"].unique())
    counts = sorted(grid["candidate_count"].unique())

    g_cross = grid[grid["scope"] == "global_cross"]
    b85 = g_cross[g_cross["band"] == "ge_p85"]
    bmid = g_cross[g_cross["band"] == "p50_to_p85"]

    fig, axes = plt.subplots(len(methods), 3, figsize=(19.2, 3.7 * len(methods) + 1.2), dpi=200, squeeze=False)
    fig.suptitle(
        f"Cross-Molecule ROCS Bands (Top 15% >= p85 and Signal 50-85%) & Measured Runtime Grid\n"
        f"{_subtitle(summary, cfg)} | values show median +/- std (TanimotoCombo [0, 2])",
        fontsize=11.5,
        fontweight="bold",
    )

    vmin_85 = float(b85["median_combo"].min())
    vmax_85 = float(b85["median_combo"].max())
    vmin_mid = float(bmid["median_combo"].min())
    vmax_mid = float(bmid["median_combo"].max())
    vmin_t = float(timing["total_cand_plus_rocs_cpu_s"].min())
    vmax_t = float(timing["total_cand_plus_rocs_cpu_s"].max())

    for row_i, method in enumerate(methods):
        # Col 0: >= p85; Col 1: p50-p85; Col 2: timing (CPU s + Wall s)
        for col_i, (sub_df, title_tag, cmap, vmin, vmax, is_time) in enumerate(
            [
                (b85[b85["method"] == method], "Top 15% (>= p85) Combo", "YlGn", vmin_85, vmax_85, False),
                (bmid[bmid["method"] == method], "Band 50-85% (p50-p85) Combo", "YlGnBu", vmin_mid, vmax_mid, False),
                (timing[timing["method"] == method], "Runtime: CPU s (cand+ROCS) / R:ROCS | Wall", "magma_r", vmin_t, vmax_t, True),
            ]
        ):
            ax = axes[row_i][col_i]
            val_col = "total_cand_plus_rocs_cpu_s" if is_time else "median_combo"
            piv = sub_df.pivot(index="candidate_count", columns="reference_cap", values=val_col).reindex(index=counts, columns=caps)
            mat = piv.to_numpy(dtype=float)

            if is_time:
                piv_wall = sub_df.pivot(index="candidate_count", columns="reference_cap", values="total_wall_s").reindex(index=counts, columns=caps).to_numpy(dtype=float)
                piv_rocs = sub_df.pivot(index="candidate_count", columns="reference_cap", values="rocs_cpu_s").reindex(index=counts, columns=caps).to_numpy(dtype=float)
                piv_ms = sub_df.pivot(index="candidate_count", columns="reference_cap", values="total_cpu_ms_per_pair").reindex(index=counts, columns=caps).to_numpy(dtype=float)
            else:
                piv_std = sub_df.pivot(index="candidate_count", columns="reference_cap", values="std_combo").reindex(index=counts, columns=caps).to_numpy(dtype=float)

            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            span = (vmax - vmin) or 1.0
            for r in range(len(counts)):
                for c in range(len(caps)):
                    val = mat[r, c]
                    if not np.isfinite(val):
                        continue
                    pos = (val - vmin) / span
                    dark = pos > 0.58
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
            ax.set_title(f"{method} | {title_tag}", fontsize=9.8, fontweight="bold")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "02_cross_bands_and_timing.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def band_distributions_and_fractions(
    grid: pd.DataFrame,
    rep_pairs: pd.DataFrame,
    summary: dict,
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    _style()
    methods = sorted(grid["method"].unique())
    palette = _palette(methods)
    rep = summary.get("representative_cell", {"K": 4, "C": 20})
    rep_k, rep_c = int(rep["K"]), int(rep["C"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    fig.suptitle(
        f"Score Bands, Cutoff Fraction Sensitivity, Shape/Color Split & Flexibility Effect (K={rep_k}, C={rep_c})\n"
        f"{_subtitle(summary, cfg)}",
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
        sub_g = grid[(grid["method"] == m) & (grid["reference_cap"] == rep_k) & (grid["candidate_count"] == rep_c)]
        for bk in band_keys:
            if bk == "self_50":
                row = sub_g[(sub_g["scope"] == "self_50") & (sub_g["band"] == "all_self")]
            else:
                row = sub_g[(sub_g["scope"] == "global_cross") & (sub_g["band"] == bk)]
            meds.append(float(row["median_combo"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std_combo"].iloc[0]) if not row.empty else np.nan)
        offs = (mi - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offs, meds, width=width * 0.92, yerr=stds, capsize=3, color=palette[m], alpha=0.88, label=m)
        for bar, med, sd in zip(bars, meds, stds):
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
        sub_g = grid[(grid["method"] == m) & (grid["reference_cap"] == rep_k) & (grid["candidate_count"] == rep_c) & (grid["scope"] == "global_cross")]
        meds, stds = [], []
        for bk, _ in frac_bands:
            row = sub_g[sub_g["band"] == bk]
            meds.append(float(row["median_combo"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std_combo"].iloc[0]) if not row.empty else np.nan)
        jitter = (mi - (len(methods) - 1) / 2) * 0.12
        ax.errorbar(xf + jitter, meds, yerr=stds, fmt="o-", color=palette[m], label=m, capsize=3.5, linewidth=1.7, markersize=5)
    ax.set_xticks(xf, [lbl for _, lbl in frac_bands])
    ax.set_ylabel("TanimotoCombo [0, 2]  (median +/- std)")
    ax.set_title("B. Fraction / Cutoff Sensitivity (Different Tail & Band Slices)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Panel C: Shape vs Color decomposition stacked bar across bands
    from matplotlib.patches import Patch

    ax = axes[1][0]
    comp_bands = [("p50_to_p85", "50-85%"), ("ge_p85", ">= p85"), ("self_50", "Self (50)")]
    xc = np.arange(len(comp_bands))
    for mi, m in enumerate(methods):
        sub_g = grid[(grid["method"] == m) & (grid["reference_cap"] == rep_k) & (grid["candidate_count"] == rep_c)]
        shapes, cols = [], []
        for bk, _ in comp_bands:
            row = sub_g[(sub_g["scope"] == "self_50") & (sub_g["band"] == "all_self")] if bk == "self_50" else sub_g[(sub_g["scope"] == "global_cross") & (sub_g["band"] == bk)]
            shapes.append(float(row["median_shape"].iloc[0]) if not row.empty else 0.0)
            cols.append(float(row["median_colour"].iloc[0]) if not row.empty else 0.0)
        offs = (mi - (len(methods) - 1) / 2) * width
        ax.bar(xc + offs, shapes, width=width * 0.9, color=palette[m], alpha=0.92)
        ax.bar(xc + offs, cols, bottom=shapes, width=width * 0.9, color=palette[m], alpha=0.42, hatch="///", edgecolor="#111827")
        for bx, sh, co in zip(xc + offs, shapes, cols):
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

    # Panel D: Cross-band score by candidate flexibility class (representative method MMFF94s_e78.4 or first)
    ax = axes[1][1]
    xt = np.arange(len(FLEXIBILITY_ORDER))
    for mi, m in enumerate(methods):
        sub_g = grid[(grid["method"] == m) & (grid["reference_cap"] == rep_k) & (grid["candidate_count"] == rep_c)]
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


def self_recovery_large_sample(
    self_df: pd.DataFrame,
    timing: pd.DataFrame,
    summary: dict,
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    _style()
    methods = sorted(self_df["method"].unique())
    palette = _palette(methods)
    caps = sorted(self_df["reference_cap"].unique())
    counts = sorted(self_df["candidate_count"].unique())
    fixed_k = min(caps, key=lambda k: abs(k - (cfg.representative_k if cfg else 4)))
    fixed_c = min(counts, key=lambda c: abs(c - (cfg.representative_c if cfg else 20)))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=200)
    fig.suptitle(
        f"50-Molecule Diverse Self-Shape Recovery Across Methods & Flexibility Tiers\n{_subtitle(summary, cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )

    # Panel A: Self-recovery [0, 1] vs C at fixed K
    ax = axes[0][0]
    sl_k = self_df[self_df["reference_cap"] == fixed_k]
    for m in methods:
        g = sl_k[sl_k["method"] == m].groupby("candidate_count")["self_recovery"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        g.columns = ["med", "q25", "q75"]
        ax.plot(g.index, g["med"], "o-", color=palette[m], label=m, linewidth=2, markersize=5)
        ax.fill_between(g.index, g["q25"], g["q75"], color=palette[m], alpha=0.14)
    ax.set_xlabel("Candidate attempts per stereoisomer (C)")
    ax.set_ylabel("Self-recovery (TanimotoCombo / 2)  [0, 1]")
    ax.set_title(f"A. Self-Recovery vs Candidate Budget C (K={fixed_k}, median & IQR)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: Self-recovery vs K at fixed C, PLUS flexibility breakdown for primary FF
    ax = axes[0][1]
    sl_c = self_df[self_df["candidate_count"] == fixed_c]
    for m in methods:
        g = sl_c[sl_c["method"] == m].groupby("reference_cap")["self_recovery"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        g.columns = ["med", "q25", "q75"]
        ax.plot(g.index, g["med"], "o-", color=palette[m], label=m, linewidth=2, markersize=5)
        ax.fill_between(g.index, g["q25"], g["q75"], color=palette[m], alpha=0.12)
    ax.set_xlabel("Reference bank cap (K)")
    ax.set_ylabel("Self-recovery (TanimotoCombo / 2)  [0, 1]")
    ax.set_title(f"B. Self-Recovery vs Bank Cap K (C={fixed_c}, median & IQR)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # Panel C: Pareto cost-quality across (method, K, C), cost = CPU s / 1,000 candidates approx
    ax = axes[1][0]
    med_self = self_df.groupby(["method", "reference_cap", "candidate_count"], as_index=False)["self_recovery"].median()
    pareto_df = med_self.merge(
        timing[["method", "reference_cap", "candidate_count", "total_cpu_ms_per_pair", "cand_embed_cpu_s", "cand_relax_cpu_s"]],
        on=["method", "reference_cap", "candidate_count"],
    )
    for m in methods:
        part = pareto_df[pareto_df["method"] == m]
        ax.scatter(part["total_cpu_ms_per_pair"], part["self_recovery"], color=palette[m], label=m, s=36, alpha=0.85)
    # Ring Pareto front on (total_cpu_ms_per_pair, self_recovery)
    costs = pareto_df["total_cpu_ms_per_pair"].to_numpy()
    vals = pareto_df["self_recovery"].to_numpy()
    nd = np.array([
        not ((costs <= costs[i]) & (vals >= vals[i]) & ((costs < costs[i]) | (vals > vals[i]))).any()
        for i in range(len(costs))
    ])
    front = pareto_df[nd].sort_values("total_cpu_ms_per_pair").reset_index(drop=True)
    ax.scatter(front["total_cpu_ms_per_pair"], front["self_recovery"], facecolors="none", edgecolors="#111827", s=135, linewidths=1.3, label="non-dominated")
    placed: list[tuple[float, float]] = []
    for _, r in front.iterrows():
        lx = float(np.log10(max(r["total_cpu_ms_per_pair"], 1e-9)))
        ly = float(r["self_recovery"])
        close_cnt = sum(1 for px, py in placed if abs(lx - px) < 0.25 and abs(ly - py) < 0.03)
        if ly >= 0.99:
            dx, dy = [(-6, 6), (6, -9), (6, 5), (-6, -9)][close_cnt % 4]
        else:
            dx, dy = [(5, 3), (5, -9), (-5, 4)][close_cnt % 3]
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            f"{r['method'][:4]} K{int(r['reference_cap'])}/C{int(r['candidate_count'])}",
            (r["total_cpu_ms_per_pair"], r["self_recovery"]),
            fontsize=6.8,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
        )
        placed.append((lx, ly))
    ax.set_xscale("log")
    ax.set_ylim(top=1.032)
    ax.set_xlabel("Measured CPU ms per (ref, cand) pair (cand + ROCS)")
    ax.set_ylabel("Median self-recovery across 50 refs  [0, 1]")
    ax.set_title("C. Self-Recovery Pareto Frontier Across 50 Prior Molecules", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")

    # Panel D: Self-recovery & Ceiling Rate split by Flexibility Class (at K=fixed_k, C=fixed_c)
    ax = axes[1][1]
    cell = self_df[(self_df["reference_cap"] == fixed_k) & (self_df["candidate_count"] == fixed_c)]
    xt = np.arange(len(FLEXIBILITY_ORDER))
    w = 0.78 / max(len(methods), 1)
    for mi, m in enumerate(methods):
        meds, ceils = [], []
        for t in FLEXIBILITY_ORDER:
            sub = cell[(cell["method"] == m) & (cell["flexibility_class"] == t)]
            meds.append(float(sub["self_recovery"].median()) if not sub.empty else np.nan)
            ceils.append(float(sub["at_ceiling"].mean()) if not sub.empty else 0.0)
        offs = (mi - (len(methods) - 1) / 2) * w
        bars = ax.bar(xt + offs, meds, width=w * 0.9, color=palette[m], alpha=0.88, label=m)
        for bar, med, ceil in zip(bars, meds, ceils):
            if np.isfinite(med):
                ax.text(bar.get_x() + bar.get_width() / 2, med + 0.012, f"{med:.3f}\n({ceil:.0%}@2)", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(xt, [FLEX_LABELS[t] for t in FLEXIBILITY_ORDER])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Median self-recovery [0, 1]  (% at combo=2.0 ceiling)")
    ax.set_title(f"D. Self-Recovery & Pose-Collapse Ceiling by Flexibility (K={fixed_k}, C={fixed_c})", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "04_self_recovery_large_sample.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def similarity_2d_vs_3d(
    rep_pairs: pd.DataFrame,
    summary: dict,
    cfg: CrossStudyConfig | None,
    out: Path,
) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=200)
    rep = summary.get("representative_cell", {"K": 4, "C": 20})
    fig.suptitle(
        f"2D Fingerprint vs 3D ROCS Similarity & Force-Field Pairing Effect (K={rep['K']}, C={rep['C']})\n"
        f"{_subtitle(summary, cfg)}",
        fontsize=11.5,
        fontweight="bold",
    )

    combo_cols = [c for c in rep_pairs.columns if c.endswith("_combo")]
    primary = "MMFF94s_e78.4_combo" if "MMFF94s_e78.4_combo" in combo_cols else combo_cols[0]
    cross = rep_pairs[rep_pairs["is_self"] == 0]
    self_p = rep_pairs[rep_pairs["is_self"] == 1]

    # Panel A: 2D ECFP4 vs 3D ROCS
    ax = axes[0]
    hb = ax.hexbin(cross["ecfp4_tanimoto"], cross[primary], gridsize=55, cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log10(cross pair count)")
    p85 = float(cross[primary].quantile(0.85))
    p50 = float(cross[primary].quantile(0.50))
    ax.axhline(p85, color="#EF4444", linestyle="--", linewidth=1.4, label=f"3D p85 = {p85:.3f}")
    ax.axhline(p50, color="#F59E0B", linestyle=":", linewidth=1.3, label=f"3D p50 = {p50:.3f}")
    ax.axvline(0.30, color="#111827", linestyle="--", linewidth=1.1, label="ECFP4 = 0.30 (scaffold hop)")
    hop_share = float(((cross[primary] >= p85) & (cross["ecfp4_tanimoto"] < 0.30)).sum() / max((cross[primary] >= p85).sum(), 1))
    ax.text(0.03, p85 + 0.06, f"Scaffold-hop share of >=p85 hits:\n{hop_share:.1%} have ECFP4 < 0.30", fontsize=8.5, fontweight="bold", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#E5E7EB"})
    ax.scatter(self_p["ecfp4_tanimoto"], self_p[primary], color="#DC2626", edgecolor="#ffffff", s=32, zorder=5, label="50 self-pairs")
    ax.set_xlabel("2D Morgan ECFP4 Tanimoto")
    ax.set_ylabel(f"3D {primary.replace('_combo', '')} TanimotoCombo [0, 2]")
    ax.set_title("A. 2D Fingerprint vs 3D ROCS Similarity", fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower right")

    # Panel B: Paired force-field deltas vs ETKDG (stratified on neutral ETKDG baseline >= p85)
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
            data_boxes.extend([delta_cross.to_numpy(), delta_top.to_numpy(), delta_self.to_numpy()])
            labels.extend([f"{m_label}\nAll Cross", f"{m_label}\n>=p85 ETKDG", f"{m_label}\nSelf (50)"])
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
    molecules = pd.read_csv(study_dir / "molecules.tsv", sep="\t")
    grid = pd.read_csv(study_dir / "cross_grid_summary.tsv", sep="\t")
    timing = pd.read_csv(study_dir / "timing_summary.tsv", sep="\t")
    self_df = pd.read_csv(study_dir / "self_recovery.tsv", sep="\t")
    rep_pairs = pd.read_csv(study_dir / "default_cell_pairs.tsv", sep="\t")
    summary = json.loads((study_dir / "sampling_summary.json").read_text())

    written = []
    for name, fn, args in (
        ("01_cohort_flexibility", cohort_flexibility, (molecules, summary, cfg, study_dir)),
        ("02_cross_bands_and_timing", cross_bands_and_timing, (grid, timing, summary, cfg, study_dir)),
        ("03_band_distributions_and_fractions", band_distributions_and_fractions, (grid, rep_pairs, summary, cfg, study_dir)),
        ("04_self_recovery_large_sample", self_recovery_large_sample, (self_df, timing, summary, cfg, study_dir)),
        ("05_2d_vs_3d_similarity", similarity_2d_vs_3d, (rep_pairs, summary, cfg, study_dir)),
    ):
        try:
            path = fn(*args)
            if path is not None:
                written.append(path)
        except Exception:
            logger.exception("figure %s failed", name)
    logger.info("wrote %d cross-study figures to %s", len(written), study_dir)
    return written
