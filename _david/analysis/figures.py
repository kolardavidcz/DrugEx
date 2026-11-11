"""The figure suite: identical visual craft to the reference dashboard suite, on the clean data contract.

Every figure deduplicates on canonical SMILES where a population statistic is shown, keeps
raw SA [1, 10] distinct from ClippedScore SA desirability [0, 1], never heuristic-scales
ROCS, never rescores molecules during analysis, and states the run's provenance
(color_ff, alignment_shape_weight, reference_conformers, conformer relaxation and reward
scheme) directly in the subtitle.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, QED, rdMolDescriptors
from scipy.stats import gaussian_kde

from _david.analysis._helpers import (
    ARMS,
    AlignedDepicter,
    align_lead_to_bank,
    compute_2d_skyline,
    dedup_key,
    dominated_volume,
    enrich_descriptors,
    pareto_mask,
    pareto_rows,
    provenance_tag,
    reference_properties,
    rocs_scale,
    scaffold_counts,
    scaffold_rows,
    scored_points,
    unique,
    unique_per_epoch,
    valid,
)

logger = logging.getLogger(__name__)

TOP_N = 12
TOP_3D = 5

CAPTIONS = {
    "01": "property diagnostic dashboard, prior vs the gmean checkpoint",
    "01b": "property diagnostic dashboard, prior vs the desired_ratio checkpoint",
    "02": "multi-objective trade-offs, step-wise skylines and Pareto leads",
    "03": "reinforcement learning dynamics and convergence diagnostics",
    "04": "longitudinal per-epoch property trajectories (median, IQR, p95, seed reference)",
    "05": "prioritized leads vs best-ROCS hits of the gmean checkpoint: aligned 2D topology",
    "05b": "the same gallery for the desired_ratio checkpoint",
    "06": "objective conflict and property coupling (Spearman rho: prior, reinforced, delta)",
    "07": "Bemis-Murcko scaffold space of the gmean checkpoint: concentration, novelty, dominant cores",
    "07b": "the same scaffold space for the desired_ratio checkpoint",
    "08": "3D superposition: the reference seed against prioritized de novo leads",
    "09": "training diagnostics: hypervolume, Pareto front, population path, sample composition, lead age",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "text.color": "#111827",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "font.size": 9.5,
        }
    )


def _valid(arm: pd.DataFrame | None) -> pd.DataFrame:
    return valid(arm)


def _unique(arm: pd.DataFrame | None) -> pd.DataFrame:
    return unique(arm)


def _short(objective: str) -> str:
    if objective.startswith("RDKit_"):
        return f"{objective.split('_')[-1]} [0, 2]"
    if objective == "SA":
        return "SA desirability [0, 1], higher = easier"
    if objective == "QED":
        return "QED [0, 1]"
    return objective


def caption(path: Path) -> str:
    number = path.name.split("_")[0]
    label = CAPTIONS.get(number)
    if not label:
        return path.stem.replace("_", " ")
    return f"{number} - {label}"


#: The arm every primary figure describes, and the arm its "b" counterpart describes.
#: gmean is the criterion the run trains to the end on, so it is the default; the
#: desired_ratio checkpoint is frozen the first time its patience window expires, which
#: makes it the one comparison worth drawing beside it rather than a third of each figure.
PRIMARY_ARM = "gmean"
SECONDARY_ARM = "desired_ratio"
#: Two lead cards are the same chemotype when they share a Murcko scaffold or their ECFP4
#: Tanimoto reaches this. A collapsed arm ranks hundreds of near-identical analogues at
#: the top, so a gallery that only de-duplicates exact SMILES shows one molecule five
#: times -- which says nothing the scaffold count in the title does not say better. The
#: cards are worth the space only if each one is a different chemotype. 0.7 on ECFP4 is
#: the conventional "same series" cutoff; this is DrugEx's own PRTD idea (rank, then
#: separate by Tanimoto distance) applied to what gets drawn rather than to the reward.
LEAD_SIMILARITY_CEILING = 0.7

#: Rows of progressively more distinct molecules drawn under the lead row in figure 05.
SPREAD_ROWS = 2

#: The policy training actually ended on, as opposed to the best epoch either criterion
#: checkpointed. The lead galleries draw this one: a card of the molecules a *selected*
#: checkpoint produced answers "what is the best this run ever did", which is not the
#: question a lead gallery is read for. The final policy is the one that would be shipped.
FINAL_ARM = "desired_ratio_at_stop"


def shown_arm(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    wanted: str,
    fallback: bool = True,
) -> str | None:
    """``wanted`` when it holds valid unique molecules, else the measured best arm.

    The fallback keeps a primary figure useful when its checkpoint produced nothing
    scorable: the figure describes an arm that exists and names it in the title. The
    ``b`` figures pass ``fallback=False``, because their filename and the report's index
    both say *desired_ratio*: drawing another arm under that name would re-attribute a
    whole checkpoint silently, which is worse than having no 01b at all.
    """
    arm = arms.get(wanted)
    if arm is not None and len(arm) and not _unique(arm).empty:
        return wanted
    return best_arm(arms, objectives) if fallback else None


def best_arm(arms: dict[str, pd.DataFrame], objectives: list[str]) -> str | None:
    """The trained checkpoint with the highest desired ratio, breaking ties by median ROCS."""
    best, ranked = None, (-1.0, -1.0)
    for name in ARMS:
        arm = arms.get(name)
        if name == "prior" or arm is None or not len(arm) or _unique(arm).empty:
            continue
        u = _unique(arm)
        score = (
            float(u["Desired"].sum()) / len(arm) if "Desired" in u else 0.0,
            float(u[objectives[0]].median()) if len(u) and objectives[0] in u else 0.0,
        )
        if score > ranked:
            best, ranked = name, score
    return best


def _bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    a_c = a[np.isfinite(a)]
    b_c = b[np.isfinite(b)]
    if len(a_c) == 0 or len(b_c) == 0:
        return 0.0, 0.0, 0.0
    delta_obs = float(np.median(b_c) - np.median(a_c))
    rng = np.random.default_rng(seed=42)
    boots = np.median(rng.choice(b_c, size=(n_boot, len(b_c)), replace=True), axis=1) - np.median(
        rng.choice(a_c, size=(n_boot, len(a_c)), replace=True), axis=1
    )
    alpha = (1.0 - ci) / 2.0
    return (
        delta_obs,
        float(np.percentile(boots, 100.0 * alpha)),
        float(np.percentile(boots, 100.0 * (1.0 - alpha))),
    )


def _plot_kde(
    ax: plt.Axes,
    data: np.ndarray,
    color: str,
    label: str,
    x_grid: np.ndarray,
    alpha: float = 0.35,
) -> None:
    clean = data[np.isfinite(data)]
    if len(clean) == 0:
        return
    if len(clean) >= 2 and not np.allclose(clean, clean[0]):
        try:
            kde = gaussian_kde(clean)
            density = kde(x_grid)
            ax.plot(x_grid, density, color=color, linewidth=2.0, label=label)
            ax.fill_between(x_grid, 0, density, color=color, alpha=alpha)
            return
        except Exception:
            pass
    ax.hist(
        clean,
        bins=25,
        range=(float(x_grid[0]), float(x_grid[-1])),
        density=True,
        color=color,
        alpha=alpha,
        label=label,
    )


# --------------------------------------------------------------------------------------
# 01 / 01b -- Hierarchical 3x3 Property Diagnostic Dashboard
# --------------------------------------------------------------------------------------


def _render_property_grid(
    b_df: pd.DataFrame,
    a_df: pd.DataFrame,
    target_png: Path,
    ref_vals: dict[str, float],
    title: str,
    note: str,
    a_label: str,
    checkpoints: tuple[str, ...] = (),
    objectives: tuple[str, ...] = (),
) -> Path:
    style()
    target_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5), dpi=200)
    fig.suptitle(
        title if not note else f"{title}\n{note}",
        fontsize=14.0 if not note else 11.8,
        fontweight="bold",
        y=0.988,
    )
    # Grey and below the parameters: which checkpoint is drawn and how large each arm
    # was is provenance, not the subject, so it reads under the settings line rather
    # than competing with the title.
    for index, line in enumerate(checkpoints):
        fig.text(
            0.5,
            0.954 - index * 0.013,
            line,
            ha="center",
            va="top",
            fontsize=9.6,
            color="#64748b",
        )

    metrics_config: list[tuple[str, str, tuple[float, float], str]] = [
        ("ROCS", "3D ROCS Shape/Color Overlap", (0.0, 2.0), "TanimotoCombo Score [0, 2]"),
        (
            "Raw_SA",
            "Synthetic Accessibility (raw SAscore)",
            (1.0, 8.0),
            "Raw SA [1, 10] (lower = easier)",
        ),
        ("QED", "Drug-Likeness (QED)", (0.0, 1.0), "QED Score [0, 1]"),
        ("MW", "Molecular Weight (MW)", (100.0, 700.0), "Molecular Weight (Da)"),
        ("HAC", "Heavy Atom Count (HAC)", (8.0, 48.0), "Heavy Atom Count"),
        ("RotBonds", "Rotatable Bonds Count", (0.0, 15.0), "Rotatable Bonds"),
        ("LogP", "Calculated Lipophilicity (SlogP)", (-2.0, 7.0), "SlogP"),
        ("TPSA", "Polar Surface Area (TPSA)", (0.0, 220.0), "TPSA (Å²)"),
        ("Fsp3", "Carbon sp³ Fraction (Fsp³ / 3D Complexity)", (0.0, 1.0), "Fsp³ Fraction"),
    ]

    ref_key_map = {
        "ROCS": "ROCS",
        "Raw_SA": "SA_raw",
        "QED": "QED",
        "MW": "MW",
        "HAC": "HAC",
        "RotBonds": "RotB",
        "LogP": "cLogP",
        "TPSA": "TPSA",
        "Fsp3": "Fsp3",
    }

    for idx, (col, p_title, x_limits, x_lbl) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]
        b_vals = (
            pd.to_numeric(b_df.get(col, pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        a_vals = (
            pd.to_numeric(a_df.get(col, pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        x_grid = np.linspace(x_limits[0], x_limits[1], 300)
        _plot_kde(ax, b_vals, "#64748B", f"Baseline Prior (n={len(b_vals)})", x_grid, alpha=0.30)
        _plot_kde(ax, a_vals, "#4F46E5", f"{a_label} (n={len(a_vals)})", x_grid, alpha=0.45)

        delta, low, high = _bootstrap_ci(b_vals, a_vals)
        b_med = float(np.median(b_vals)) if len(b_vals) > 0 else float("nan")
        a_med = float(np.median(a_vals)) if len(a_vals) > 0 else float("nan")

        if math.isfinite(b_med):
            ax.axvline(b_med, color="#475569", linestyle="--", linewidth=1.4)
        if math.isfinite(a_med):
            ax.axvline(a_med, color="#3730A3", linestyle="-", linewidth=1.6)

        rk = ref_key_map.get(col, col)
        if rk in ref_vals and math.isfinite(ref_vals[rk]):
            rv = ref_vals[rk]
            ax.axvline(rv, color="#0D9488", linestyle=":", linewidth=1.6, label=f"Seed ({rv:.2f})")

        stat_text = (
            f"Prior Med: {b_med:.2f}\n"
            f"Reinf Med: {a_med:.2f}\n"
            f"Δ = {delta:+.2f} [{low:+.2f}, {high:+.2f}]"
        )
        ax.text(
            0.04,
            0.93,
            stat_text,
            transform=ax.transAxes,
            fontsize=8.2,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.30", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9
            ),
        )

        ax.set_title(p_title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=9.0)
        ax.set_ylabel("Density (valid unique)", fontsize=9.0)
        ax.set_xlim(x_limits)
        ax.grid(True, linestyle=":", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8.0)

    # The top row always draws ROCS, raw SA and QED, but QED is an objective only when
    # the run asked for it -- most do not. Calling it an objective in a run that never
    # optimised it is the kind of caption that gets believed and written up.
    optimised = [
        label
        for label, present in (
            ("ROCS", any(o.startswith("RDKit_") for o in objectives)),
            ("SA", "SA" in objectives),
            ("QED", "QED" in objectives),
        )
        if present
    ]
    spare = [label for label in ("ROCS", "SA", "QED") if label not in optimised]
    row_one = (
        f"PRIMARY RL OBJECTIVES ({', '.join(optimised)}; SA SHOWN RAW)"
        if optimised
        else "PROPERTIES"
    )
    if spare:
        row_one += f" -- {', '.join(spare)}: DIAGNOSTIC ONLY, NOT IN THE REWARD"
    row_titles = [
        row_one,
        "MOLECULAR SIZE & COMPLEXITY DIAGNOSTICS (MW, HAC, ROTB)",
        "PHYSICOCHEMICAL & 3D INTEGRITY (SLOGP, TPSA, FSP³)",
    ]
    for row_idx, r_text in enumerate(row_titles):
        axes[row_idx, 1].annotate(
            r_text,
            xy=(0.5, 1.14),
            xycoords="axes fraction",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#334155",
            bbox=dict(
                boxstyle="square,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8
            ),
        )

    plt.subplots_adjust(top=0.878, bottom=0.05, hspace=0.36, wspace=0.22)
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def before_after_distributions(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    thresholds: list[float],
    out: Path,
    config: dict | None = None,
    reference: Path | None = None,
    arm: str = PRIMARY_ARM,
    name: str = "01_before_after_distributions",
) -> Path | None:
    """The 3x3 property dashboard, prior against one trained arm.

    One function for 01 and 01b: they differ only in which checkpoint the prior is
    compared with, and keeping two copies of the grid is how the two drifted apart.
    """
    win = shown_arm(arms, objectives, arm, fallback=arm == PRIMARY_ARM)
    if win is None:
        return None
    b_u = _unique(arms.get("prior"))
    a_u = _unique(arms.get(win))
    ref_smi = (config or {}).get("reference_smiles")
    b_en = enrich_descriptors(b_u, objectives[0], ref_smi, config)
    a_en = enrich_descriptors(a_u, objectives[0], ref_smi, config)
    ref_vals = reference_properties(config, reference)
    prov = provenance_tag(config, reference)
    counts = [
        f"{n}: {len(_unique(arms[n]))} unique / {len(arms[n])} sampled "
        f"(desired={float(_unique(arms[n])['Desired'].sum()) / max(len(arms[n]), 1):.3f})"
        for n in ARMS
        if n in arms and len(arms[n])
    ]
    # Two lines, split down the middle: four arms on one line ran off the canvas.
    half = (len(counts) + 1) // 2
    checkpoints = (
        f"Drawn: prior vs {win}" + (f"  |  {', '.join(counts[:half])}" if counts else ""),
        ", ".join(counts[half:]),
    )
    return _render_property_grid(
        b_en,
        a_en,
        out / f"{name}.png",
        ref_vals,
        "Hierarchical De Novo Property Diagnostic Dashboard: Baseline Prior vs. Reinforced Policy",
        prov,
        a_label=f"Reinforced ({win})",
        checkpoints=tuple(line for line in checkpoints if line),
        objectives=tuple(objectives),
    )


# --------------------------------------------------------------------------------------
# 02 -- Multi-Objective Optimization Trade-Offs (1x3 scatter + skylines + Pareto stars + contours)
# --------------------------------------------------------------------------------------


def _overlay_prior_contours(ax: plt.Axes, bx: np.ndarray, by: np.ndarray) -> None:
    ok = np.isfinite(bx) & np.isfinite(by)
    bx_v, by_v = bx[ok], by[ok]
    if len(bx_v) < 15 or np.allclose(bx_v, bx_v[0]) or np.allclose(by_v, by_v[0]):
        return
    try:
        kde = gaussian_kde(np.vstack([bx_v, by_v]))
        xi, yi = np.mgrid[bx_v.min() : bx_v.max() : 50j, by_v.min() : by_v.max() : 50j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        v_sorted = np.sort(zi.flatten())[::-1]
        cum = np.cumsum(v_sorted)
        cum /= cum[-1]
        lvl_50 = v_sorted[np.searchsorted(cum, 0.50)]
        lvl_90 = v_sorted[np.searchsorted(cum, 0.90)]
        ax.contour(
            xi,
            yi,
            zi,
            levels=sorted([lvl_90, lvl_50]),
            colors=["#64748b", "#334155"],
            linestyles=["--", "-."],
            linewidths=[1.1, 1.4],
            alpha=0.75,
            zorder=2,
        )
    except Exception:
        pass


def multiobjective_tradeoffs(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    thresholds: list[float],
    out: Path,
    config: dict | None = None,
    reference: Path | None = None,
) -> Path | None:
    if len(objectives) < 2:
        return None
    style()
    win = shown_arm(arms, objectives, PRIMARY_ARM) or "prior"
    a_u = _unique(arms.get(win))
    if a_u.empty:
        return None
    b_u = _unique(arms.get("prior"))
    ref_smi = (config or {}).get("reference_smiles")
    df_clean = enrich_descriptors(a_u, objectives[0], ref_smi, config)
    b_clean = enrich_descriptors(b_u, objectives[0], ref_smi, config) if not b_u.empty else None

    # QED is drawn in two of these three panels but is an objective only when the run
    # asked for it. Where it is not, it is a diagnostic axis: the panels keep it (the
    # correlation is worth seeing) but say so, and no skyline is drawn over it -- a
    # frontier implies a trade-off the reward was never asked to make.
    qed_is_objective = "QED" in objectives
    qed_role = "" if qed_is_objective else ", diagnostic -- not in the reward"
    rocs = pd.to_numeric(df_clean["ROCS"], errors="coerce").to_numpy(dtype=float)
    sa = pd.to_numeric(df_clean["Raw_SA"], errors="coerce").to_numpy(dtype=float)
    qed = pd.to_numeric(df_clean["QED"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(rocs) & np.isfinite(sa) & np.isfinite(qed)
    if not finite.any():
        return None
    rocs, sa, qed = rocs[finite], sa[finite], qed[finite]

    # True Pareto front computed on the run's actual objectives (where higher is always better)
    obj_pts = df_clean.loc[finite, [o for o in objectives if o in df_clean.columns]].to_numpy(
        dtype=float
    )
    p_mask = (
        pareto_mask(obj_pts) if obj_pts.size else pareto_mask(np.column_stack([rocs, -sa, qed]))
    )
    p3_rocs, p3_sa, p3_qed = rocs[p_mask], sa[p_mask], qed[p_mask]

    b_rocs = (
        pd.to_numeric(b_clean["ROCS"], errors="coerce").to_numpy(dtype=float)
        if b_clean is not None
        else np.array([])
    )
    b_sa = (
        pd.to_numeric(b_clean["Raw_SA"], errors="coerce").to_numpy(dtype=float)
        if b_clean is not None
        else np.array([])
    )
    b_qed = (
        pd.to_numeric(b_clean["QED"], errors="coerce").to_numpy(dtype=float)
        if b_clean is not None
        else np.array([])
    )

    thr_map = dict(zip(objectives, thresholds))
    # The manifest's ROCS threshold is in the units that run's reward used; this axis is
    # published TanimotoCombo. Without the factor the line lands at half its position.
    rocs_thr = float(thr_map.get(objectives[0], 0.70)) * rocs_scale(config)
    # Map SA desirability threshold on [0, 1] back to physical raw SA [1, 10]
    sa_mod_thr = float(thr_map.get("SA", 0.63))
    sa_lo = float((config or {}).get("sa_reward_zero", 5.0))
    sa_hi = float((config or {}).get("sa_reward_one", 3.0))
    sa_raw_thr = sa_lo - sa_mod_thr * (sa_lo - sa_hi)
    qed_thr = float(thr_map.get("QED", reference_properties(config, reference).get("QED", 0.21)))
    settings = config or {}
    if "QED" not in objectives:
        qed_thr_label = "QED of the reference"
    elif settings.get("fixed_thresholds"):
        qed_thr_label = "QED threshold (fixed)"
    elif "QED" in (settings.get("threshold_from_reference") or ()):
        qed_thr_label = "QED threshold (reference)"
    else:
        qed_thr_label = f"QED threshold (prior p{settings.get('threshold_percentile', 85):g})"

    prov = provenance_tag(config, reference)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8), dpi=200)
    fig.suptitle(
        f"Multi-Objective Optimization Trade-Offs, Step-Wise Skylines & Pareto Leads ({win}, n={len(rocs)} valid unique)\n{prov}",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )

    # Panel A: ROCS vs Raw SA
    if len(b_rocs) > 0 and len(b_sa) > 0:
        _overlay_prior_contours(ax1, b_rocs, b_sa)
    sc1 = ax1.scatter(
        rocs, sa, c=qed, cmap="viridis", alpha=0.65, edgecolors="none", s=28, zorder=3
    )
    fig.colorbar(sc1, ax=ax1).set_label("QED (Evaluated)", fontsize=9)
    sk1_x, sk1_y = compute_2d_skyline(rocs, sa, maximize_x=True, maximize_y=False)
    if len(sk1_x) > 0:
        ax1.plot(
            sk1_x,
            sk1_y,
            drawstyle="steps-pre",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_rocs) > 0:
        ax1.scatter(
            p3_rocs,
            p3_sa,
            marker="*",
            s=140,
            facecolors="#f59e0b",
            edgecolors="#78350f",
            linewidth=1.0,
            zorder=5,
            label=f"Non-dominated ({len(p3_rocs)})",
        )
    ax1.axvline(
        rocs_thr,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"ROCS Thr ({rocs_thr:.2f})",
    )
    ax1.axhline(
        sa_raw_thr,
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"SA Thr ({sa_raw_thr:.2f})",
    )
    ax1.set_title(
        "(A) 3D Shape (ROCS) vs. Synthesizability (Raw SA)", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("ROCS TanimotoCombo [0, 2] (higher=better)")
    ax1.set_ylabel("Raw SA Score [1, 10] (lower=better)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.8)

    # Panel B: ROCS vs QED
    if len(b_rocs) > 0 and len(b_qed) > 0:
        _overlay_prior_contours(ax2, b_rocs, b_qed)
    sc2 = ax2.scatter(
        rocs, qed, c=sa, cmap="plasma_r", alpha=0.65, edgecolors="none", s=28, zorder=3
    )
    fig.colorbar(sc2, ax=ax2).set_label("Raw SA Score", fontsize=9)
    sk2_x, sk2_y = (
        compute_2d_skyline(rocs, qed, maximize_x=True, maximize_y=True)
        if qed_is_objective
        else (np.array([]), np.array([]))
    )
    if len(sk2_x) > 0:
        ax2.plot(
            sk2_x,
            sk2_y,
            drawstyle="steps-pre",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_rocs) > 0:
        ax2.scatter(
            p3_rocs,
            p3_qed,
            marker="*",
            s=140,
            facecolors="#f59e0b",
            edgecolors="#78350f",
            linewidth=1.0,
            zorder=5,
            label=f"Non-dominated ({len(p3_rocs)})",
        )
    ax2.axvline(
        rocs_thr,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"ROCS Thr ({rocs_thr:.2f})",
    )
    ax2.axhline(
        qed_thr,
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"{qed_thr_label} ({qed_thr:.2f})",
    )
    ax2.set_title(
        f"(B) 3D Shape (ROCS) vs. Drug-Likeness (QED{qed_role})", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("ROCS TanimotoCombo [0, 2] (higher=better)")
    ax2.set_ylabel("QED Score [0, 1] (higher=better)")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=7.8)

    # Panel C: Raw SA vs QED
    if len(b_sa) > 0 and len(b_qed) > 0:
        _overlay_prior_contours(ax3, b_sa, b_qed)
    sc3 = ax3.scatter(sa, qed, c=rocs, cmap="turbo", alpha=0.65, edgecolors="none", s=28, zorder=3)
    fig.colorbar(sc3, ax=ax3).set_label("ROCS TanimotoCombo", fontsize=9)
    sk3_x, sk3_y = (
        compute_2d_skyline(sa, qed, maximize_x=False, maximize_y=True)
        if qed_is_objective
        else (np.array([]), np.array([]))
    )
    if len(sk3_x) > 0:
        ax3.plot(
            sk3_x,
            sk3_y,
            drawstyle="steps-pre",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_sa) > 0:
        ax3.scatter(
            p3_sa,
            p3_qed,
            marker="*",
            s=140,
            facecolors="#f59e0b",
            edgecolors="#78350f",
            linewidth=1.0,
            zorder=5,
            label=f"Non-dominated ({len(p3_sa)})",
        )
    ax3.axvline(
        sa_raw_thr,
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"SA Thr ({sa_raw_thr:.2f})",
    )
    ax3.axhline(
        qed_thr,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"{qed_thr_label} ({qed_thr:.2f})",
    )
    ax3.set_title(
        f"(C) Synthesizability (Raw SA) vs. Drug-Likeness (QED{qed_role})",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Raw SA Score [1, 10] (lower=better)")
    ax3.set_ylabel("QED Score [0, 1] (higher=better)")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=7.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    target_png = out / "02_multiobjective_tradeoffs.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 03 -- SequenceRNN Reinforcement Learning Dynamics & Convergence Diagnostics (2x2)
# --------------------------------------------------------------------------------------


def training_trajectories(
    fit: pd.DataFrame,
    summary: dict,
    out: Path,
    epochs: pd.DataFrame | None = None,
    objectives: list[str] | None = None,
    config: dict | None = None,
    reference: Path | None = None,
) -> Path | None:
    if fit is None or fit.empty:
        return None
    style()
    df = fit.copy()
    rocs_col = (objectives or ["ROCS"])[0]
    prov = provenance_tag(config, reference)

    rocs_median = rocs_q25 = rocs_q75 = rocs_p95 = cum_p95 = None
    sa_grp = None
    if epochs is not None and not epochs.empty:
        vep = enrich_descriptors(
            unique_per_epoch(epochs), rocs_col, (config or {}).get("reference_smiles"), config
        )
        if not vep.empty:
            for col in ("ROCS", "QED"):
                if col in vep.columns:
                    df[col] = df["Epoch"].map(vep.groupby("Epoch")[col].mean())
                    df[f"{col}_median"] = df["Epoch"].map(vep.groupby("Epoch")[col].median())
            if "ROCS" in vep.columns:
                r_grp = vep.groupby("Epoch")["ROCS"]
                rocs_median = r_grp.median()
                rocs_q25 = r_grp.quantile(0.25)
                rocs_q75 = r_grp.quantile(0.75)
                rocs_p95 = r_grp.quantile(0.95)
                if not rocs_p95.dropna().empty:
                    cum_p95 = np.maximum.accumulate(rocs_p95.dropna().values)
            if "Raw_SA" in vep.columns and vep["Raw_SA"].notna().any():
                sa_grp = vep.groupby("Epoch")["Raw_SA"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), dpi=200)
    fig.suptitle(
        f"SequenceRNN Reinforcement Learning Dynamics & Convergence Diagnostics (n_eval={(config or {}).get('n_eval', '?')})\n{prov}",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )

    ep_vals = df["Epoch"]
    best_des_ep = summary.get("best_desired_ratio_epoch")
    freeze_ep = summary.get("desired_ratio_freeze_epoch")
    best_gm_ep = summary.get("best_avg_gmean_epoch")
    froze = bool(summary.get("desired_ratio_frozen"))

    def _mark_events(ax: plt.Axes, is_legend: bool = False) -> None:
        if best_des_ep:
            ax.axvline(
                best_des_ep,
                color="#0D9488",
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label=f"desired_ratio peak (Ep {best_des_ep})" if is_legend else "",
                zorder=2,
            )
        if best_gm_ep:
            ax.axvline(
                best_gm_ep,
                color="#7C3AED",
                linestyle="-.",
                linewidth=1.4,
                alpha=0.85,
                label=f"gmean peak (Ep {best_gm_ep})" if is_legend else "",
                zorder=2,
            )
        if freeze_ep:
            lbl = f"patience freeze (Ep {freeze_ep})" if froze else f"end of run (Ep {freeze_ep})"
            ax.axvline(
                freeze_ep,
                color="#dc2626",
                linestyle=":",
                linewidth=1.4,
                alpha=0.85,
                label=lbl if is_legend else "",
                zorder=2,
            )

    # Panel A: Training Loss & Exploration Rate
    ax1 = axes[0, 0]
    _mark_events(ax1, is_legend=True)
    if "loss_train" in df.columns:
        ax1.plot(
            ep_vals, df["loss_train"], color="#dc2626", linewidth=1.8, label="Train Loss", zorder=3
        )
    ax1.set_title("(A) Training Loss & Exploration Rate", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.4)
    handles, labels = ax1.get_legend_handles_labels()
    if config and "epsilon" in config:
        ax1r = ax1.twinx()
        eps_val = float(config.get("epsilon", 0.0))
        eps_arr = np.full(len(ep_vals), eps_val)
        ax1r.plot(
            ep_vals,
            eps_arr,
            color="#EA580C",
            linewidth=1.8,
            linestyle="--",
            label=f"ε = {eps_val:.2f} (right axis)",
            zorder=4,
        )
        eps_top = max(eps_val, 0.05)
        ax1r.set_ylim(-0.03 * eps_top, eps_top * 1.35)
        ax1r.set_ylabel("Exploration ε", color="#9A3412")
        ax1r.tick_params(axis="y", colors="#9A3412")
        h2, l2 = ax1r.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    if handles:
        ax1.legend(handles, labels, loc="best", fontsize=8.0)

    # Panel B: Selection Criteria & Quality
    ax2 = axes[0, 1]
    _mark_events(ax2)
    series_specs = [
        ("avg_gmean", "Avg Geometric Mean ★", "#7C3AED", "o", True),
        ("avg_amean", "Avg Arithmetic Mean", "#93C5FD", "s", False),
        ("desired_ratio", "Desired Ratio ★", "#0D9488", "^", True),
        ("valid_ratio", "Valid Ratio", "#B45309", "D", False),
        ("unique_ratio", "Unique Ratio", "#BE185D", "v", False),
    ]
    mark_step = max(1, len(ep_vals) // 12)
    for col, label, color, marker, is_crit in series_specs:
        if col in df.columns and not df[col].dropna().empty:
            ax2.plot(
                ep_vals,
                df[col],
                color=color,
                linewidth=2.2 if is_crit else 1.6,
                linestyle="-" if is_crit else "--",
                marker=marker,
                markersize=4.0 if is_crit else 3.2,
                markevery=mark_step,
                label=label,
                zorder=4 if is_crit else 3,
            )
    ax2.set_title(
        "(B) Dual Checkpoint Criteria & Sample Quality (on n_eval)", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Epoch")
    drawn = [
        pd.to_numeric(df[col], errors="coerce").max()
        for col, *_ in series_specs
        if col in df.columns and not df[col].dropna().empty
    ]
    ceiling = max([1.05, *(float(v) * 1.08 for v in drawn if pd.notna(v))])
    ax2.set_ylim(-0.05, ceiling)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="best", fontsize=7.8, ncol=2)

    # Panel C: Objective Mean/Median & Raw Synthesizability
    ax3 = axes[1, 0]
    _mark_events(ax3)
    plotted_objs = False
    # QED only when the run optimised it: enrich_descriptors computes it for every
    # molecule, so the column exists either way and the panel is titled "Objective".
    series = [("ROCS", "#2563EB", "#93C5FD", "ROCS")]
    if "QED" in (objectives or []):
        series.append(("QED", "#F59E0B", "#FDE68A", "QED"))
    for col, mean_c, med_c, lbl in series:
        if col in df.columns and not df[col].dropna().empty:
            ax3.plot(ep_vals, df[col], color=mean_c, linewidth=2.1, label=f"{lbl} Mean", zorder=4)
            if f"{col}_median" in df.columns:
                ax3.plot(
                    ep_vals,
                    df[f"{col}_median"],
                    color=med_c,
                    linewidth=1.6,
                    linestyle="--",
                    label=f"{lbl} Median",
                    zorder=3,
                )
            plotted_objs = True
    if not plotted_objs and "avg_gmean" in df.columns:
        ax3.plot(
            ep_vals,
            df["avg_gmean"],
            color="#7C3AED",
            linewidth=1.8,
            label="Avg Geometric Mean",
            zorder=3,
        )
    ax3.set_title(
        "(C) Objective Mean/Median (left) & Raw Synthesizability (right)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("ROCS TanimotoCombo (0–2) · QED (0–1)")
    ax3.grid(True, linestyle=":", alpha=0.4)
    h3, l3 = ax3.get_legend_handles_labels()
    if sa_grp is not None:
        sai_mean, sai_med, sa_q25, sa_q75 = (
            sa_grp.mean(),
            sa_grp.median(),
            sa_grp.quantile(0.25),
            sa_grp.quantile(0.75),
        )
        ax3r = ax3.twinx()
        ax3r.fill_between(
            sai_med.index,
            sa_q25.values,
            sa_q75.values,
            color="#D1D5DB",
            alpha=0.30,
            linewidth=0,
            label="Raw SA IQR",
            zorder=2,
        )
        ax3r.plot(
            sai_mean.index,
            sai_mean.values,
            color="#4B5563",
            linewidth=2.1,
            label="Raw SA Mean (right)",
            zorder=5,
        )
        ax3r.plot(
            sai_med.index,
            sai_med.values,
            color="#9CA3AF",
            linewidth=1.6,
            linestyle="--",
            label="Raw SA Median (right)",
            zorder=4,
        )
        sa_lo, sa_hi = float(np.nanmin(sa_q25.values)), float(np.nanmax(sa_q75.values))
        pad = max(0.6, 0.15 * (sa_hi - sa_lo))
        ax3r.set_ylim(sa_lo - pad, sa_hi + pad)
        ax3r.set_ylabel("Raw RDKit SA [1, 10] (lower = easier)", color="#4B5563")
        ax3r.tick_params(axis="y", colors="#4B5563")
        hr, lr = ax3r.get_legend_handles_labels()
        h3, l3 = h3 + hr, l3 + lr
    if h3:
        ax3.legend(h3, l3, loc="lower right", fontsize=7.5, ncol=2)

    # Panel D: ROCS Distribution & Stored Shape/Color Split (zero rescoring)
    ax4 = axes[1, 1]
    _mark_events(ax4)
    if rocs_median is not None and not rocs_median.dropna().empty:
        ep_r = rocs_median.index.values
        ax4.fill_between(
            ep_r,
            rocs_q25.values,
            rocs_q75.values,
            color="#93C5FD",
            alpha=0.30,
            linewidth=0,
            label="ROCS IQR (25th–75th)",
            zorder=2,
        )
        ax4.plot(
            ep_r, rocs_median.values, color="#2563eb", linewidth=2.2, label="ROCS Median", zorder=4
        )
        ax4.plot(
            ep_r,
            rocs_p95.values,
            color="#E11D48",
            linewidth=1.4,
            linestyle="-.",
            label="ROCS 95th %ile",
            zorder=3,
        )
        if cum_p95 is not None:
            ax4.plot(
                rocs_p95.dropna().index.values,
                cum_p95,
                color="#4338CA",
                linewidth=1.5,
                linestyle=":",
                label="Cumulative p95 ceiling",
                zorder=3,
            )
    ax4.set_title(
        "(D) ROCS Distribution & Stored Shape/Color Split", fontsize=11, fontweight="bold"
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("ROCS TanimotoCombo (0–2)")
    ax4.grid(True, linestyle=":", alpha=0.4)
    h4, l4 = ax4.get_legend_handles_labels()
    if "avg_ShapeTanimoto" in df.columns and "avg_ColorTanimoto" in df.columns:
        ax4r = ax4.twinx()
        (ls_line,) = ax4r.plot(
            ep_vals,
            df["avg_ShapeTanimoto"],
            color="#0D9488",
            linewidth=1.8,
            label="avg_ShapeTanimoto (right)",
            zorder=5,
        )
        (lc_line,) = ax4r.plot(
            ep_vals,
            df["avg_ColorTanimoto"],
            color="#D97706",
            linewidth=1.8,
            label=f"avg_ColorTanimoto ({(config or {}).get('color_ff', 'fdef')})",
            zorder=5,
        )
        sc_hi = max(float(df[["avg_ShapeTanimoto", "avg_ColorTanimoto"]].to_numpy().max()), 0.1)
        ax4r.set_ylim(0.0, max(sc_hi * 1.28, 0.2))
        ax4r.set_ylabel("Shape / Color Tanimoto [0, 1]", fontsize=8.6)
        h4, l4 = h4 + [ls_line, lc_line], l4 + [ls_line.get_label(), lc_line.get_label()]
    if h4:
        ax4.legend(h4, l4, loc="lower right", fontsize=7.2, ncol=2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    target_png = out / "03_training_trajectories.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 05 / 05b -- Aligned 2D Lead Structure Cards (Target Seed + Row 1 Desired + Row 2 ROCS)
# --------------------------------------------------------------------------------------


def _render_lead_cards(
    arm_df: pd.DataFrame,
    objectives: list[str],
    target_png: Path,
    title: str,
    config: dict | None = None,
    reference: Path | None = None,
    max_leads: int = 5,
) -> Path | None:
    u = _unique(arm_df)
    # Only molecules that cleared every threshold are drawn. A card is a claim that the
    # run produced something worth looking at, and a molecule below the thresholds is
    # not that however well it overlays -- the arm's own `Desired` column is the test.
    if "Desired" in u.columns:
        u = u[pd.to_numeric(u["Desired"], errors="coerce").fillna(0).astype(int) == 1]
    if u.empty:
        return None
    ref_smi = (config or {}).get("reference_smiles", "")
    en = enrich_descriptors(u, objectives[0], ref_smi, config)
    depicter = AlignedDepicter(ref_smi)
    ref_props = reference_properties(config, reference)
    prov = provenance_tag(config, reference)

    top_row: list[dict[str, Any]] = []
    spread_rows: list[list[dict[str, Any]]] = []
    seen_smiles: set[str] = set()

    if depicter.ref_mol is not None:
        ref_clean = Chem.RemoveHs(Chem.Mol(depicter.ref_mol))
        top_row.append(
            {
                "mol": ref_clean,
                "title": "TARGET QUERY SEED",
                "is_seed": True,
                "rocs": None,
                "sa": ref_props.get("SA_raw", 3.74),
                "sa_mod": ref_props.get("SA", 0.63),
                "qed": ref_props.get("QED", 0.21),
                "mw": ref_props.get("MW", 458.4),
                "hac": int(ref_props.get("HAC", 33)),
                "delta_hac": 0,
                "sim": 1.0,
            }
        )
        seen_smiles.add(Chem.MolToSmiles(ref_clean))

    seed_hac = int(ref_props.get("HAC", 33))
    seed_fp = (
        rdMolDescriptors.GetMorganFingerprintAsBitVect(depicter.ref_mol, 2, nBits=2048)
        if depicter.ref_mol is not None
        else None
    )

    def _card(row: pd.Series, card_title: str) -> dict[str, Any] | None:
        smi = str(row.get("SMILES", ""))
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m_clean = Chem.RemoveHs(m)
        hac = int(m_clean.GetNumHeavyAtoms())
        lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m_clean, 2, nBits=2048)
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, lead_fp)) if seed_fp else 0.0
        return {
            "mol": m_clean,
            "title": card_title,
            "is_seed": False,
            "rocs": float(row.get("ROCS", 0.0)),
            "shape": float(row.get("ShapeTanimoto", math.nan)),
            "color": float(row.get("ColorTanimoto", math.nan)),
            "sa": float(row.get("Raw_SA", row.get("SA_raw", 0.0))),
            "sa_mod": float(row.get("SA", 0.0)),
            "qed": float(row.get("QED", 0.0)),
            "mw": float(row.get("MW", Descriptors.MolWt(m_clean))),
            "hac": hac,
            "delta_hac": hac - seed_hac,
            "sim": sim,
        }

    seen_scaffolds: set[str] = set()
    taken_fps: list[Any] = []

    def _is_new_chemotype(row: pd.Series, fp) -> bool:
        scaffold = str(row.get("Scaffold") or "")
        if scaffold and scaffold in seen_scaffolds:
            return False
        return all(
            float(DataStructs.TanimotoSimilarity(fp, other)) < LEAD_SIMILARITY_CEILING
            for other in taken_fps
        )

    def _fill(target: list, ordered: pd.DataFrame, limit: int, title: Any) -> int:
        """Take the best rows that are each a different chemotype, then top up.

        Two passes, both down the same ranking, so the cards stay in score order and the
        figure is never short: the first takes only new chemotypes, the second fills any
        remaining slots from what is left. A card from the second pass says so, because
        at that point the arm genuinely had nothing else to show.
        """
        repeats = 0
        for distinct_only in (True, False):
            for _, row in ordered.iterrows():
                if len(target) >= limit:
                    return repeats
                can = str(row.get("Canonical") or "")
                if not can or can in seen_smiles:
                    continue
                mol = Chem.MolFromSmiles(can)
                if mol is None:
                    continue
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                if distinct_only and not _is_new_chemotype(row, fp):
                    continue
                card = _card(row, title(row, len(target), distinct_only))
                if card is None:
                    continue
                target.append(card)
                seen_smiles.add(can)
                seen_scaffolds.add(str(row.get("Scaffold") or ""))
                taken_fps.append(fp)
                repeats += 0 if distinct_only else 1
        return repeats

    # Row 1: the best of the desired molecules, by the first objective.
    by_score = en.sort_values(by="ROCS", ascending=False)
    repeats = _fill(
        top_row,
        by_score,
        max_leads + (1 if depicter.ref_mol is not None else 0),
        lambda row, taken, distinct: (
            f"LEAD #{taken if depicter.ref_mol is not None else taken + 1}"
            + ("" if distinct else " (analogue)")
        ),
    )
    if not top_row:
        return None

    # The rows below walk away from those leads. A second row of next-best-by-ROCS was
    # five more of the same chemistry, which the ranking guarantees and which taught
    # nothing; distance from the leads is the axis with something left to show. Every
    # card is still a desired molecule, so the grid reads "best" to "most different
    # thing that still cleared the thresholds" rather than "best" to "worst".
    lead_fps = [
        rdMolDescriptors.GetMorganFingerprintAsBitVect(card["mol"], 2, nBits=2048)
        for card in top_row
        if not card["is_seed"]
    ]
    if lead_fps:
        remaining = by_score[~by_score["Canonical"].astype(str).isin(seen_smiles)].copy()
        if not remaining.empty:
            far = []
            for can in remaining["Canonical"].astype(str):
                mol = Chem.MolFromSmiles(can)
                if mol is None:
                    far.append(-1.0)
                    continue
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                far.append(1.0 - max(DataStructs.BulkTanimotoSimilarity(fp, lead_fps)))
            remaining["_from_leads"] = far
            # Ascending: the grid grows more distinct as it is read, row by row.
            ordered = remaining[remaining["_from_leads"] >= 0].sort_values("_from_leads")
            for _ in range(SPREAD_ROWS):
                row_cards: list[dict[str, Any]] = []
                repeats += _fill(
                    row_cards,
                    ordered[~ordered["Canonical"].astype(str).isin(seen_smiles)],
                    max_leads,
                    lambda row, taken, distinct: (
                        f"d={row['_from_leads']:.2f} from leads"
                        + ("" if distinct else " (analogue)")
                    ),
                )
                if not row_cards:
                    break
                spread_rows.append(row_cards)

    flat_spread = [card for row in spread_rows for card in row]
    display_mols = [card["mol"] for card in top_row + flat_spread if card is not None]
    depicter.prepare_template(display_mols)
    for card in top_row + flat_spread:
        card["mol"] = depicter.align_molecule(card["mol"])

    rows: list[list[dict[str, Any] | None]] = [top_row]
    for row_cards in spread_rows:
        rows.append([None] + row_cards if depicter.ref_mol is not None else row_cards)

    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    dpi = 200
    card_w_in = 3.55
    card_h_in = card_w_in * 540 / 760
    pad_x_in = 0.06
    caption_in = 1.25
    title_in = 0.72
    fig_w = ncols * card_w_in + (ncols + 1) * pad_x_in
    fig_h = title_in + nrows * (card_h_in + caption_in)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    axes: list[list[Any]] = []
    for r_idx in range(nrows):
        row_axes = []
        row_top = fig_h - title_in - r_idx * (card_h_in + caption_in)
        for c_idx in range(ncols):
            left = pad_x_in + c_idx * (card_w_in + pad_x_in)
            row_axes.append(
                fig.add_axes(
                    [
                        left / fig_w,
                        (row_top - card_h_in) / fig_h,
                        card_w_in / fig_w,
                        card_h_in / fig_h,
                    ]
                )
            )
        axes.append(row_axes)

    fig.suptitle(
        f"{title}\n{prov}" if prov else title,
        fontsize=11.5,
        fontweight="bold",
        x=0.5,
        y=1 - 0.18 / fig_h,
    )

    for r_idx, row_cards in enumerate(rows):
        for c_idx in range(ncols):
            ax = axes[r_idx][c_idx]
            ax.axis("off")
            card = row_cards[c_idx] if c_idx < len(row_cards) else None
            if card is None:
                if r_idx >= 1 and c_idx == 0 and spread_rows:
                    ax.text(
                        0.5,
                        0.62,
                        f"MORE DISTINCT\n(row {r_idx} of {len(spread_rows)})\n\nstill desired, ordered by\nECFP4 distance from the leads",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9.0,
                        fontweight="bold",
                        color="#475569",
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"
                        ),
                    )
                continue

            img = Draw.MolToImage(
                card["mol"], size=(round(card_w_in * dpi), round(card_h_in * dpi))
            )
            ax.imshow(img)
            header_color = (
                "#1e3a8a"
                if card["is_seed"]
                else ("#0f766e" if str(card["title"]).startswith("ROCS") else "#1e293b")
            )
            ax.text(
                0.5,
                -0.05,
                card["title"],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                color=header_color,
            )

            if card["is_seed"]:
                ax.text(
                    0.5,
                    -0.16,
                    f"HAC: {card['hac']}  |  MW: {card['mw']:.1f} Da",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="#f1f5f9",
                        edgecolor="#cbd5e1",
                        linewidth=0.8,
                    ),
                )
                ax.text(
                    0.5,
                    -0.28,
                    f"Raw SA: {card['sa']:.2f}  |  QED: {card['qed']:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="#f1f5f9",
                        edgecolor="#cbd5e1",
                        linewidth=0.8,
                    ),
                )
                continue

            rocs_c = (
                "#dcfce7"
                if card["rocs"] >= 1.0
                else ("#dbeafe" if card["rocs"] >= 0.70 else "#f1f5f9")
            )
            rocs_e = (
                "#15803d"
                if card["rocs"] >= 1.0
                else ("#2563eb" if card["rocs"] >= 0.70 else "#94a3b8")
            )
            dh = card["delta_hac"]
            dh_c = "#dcfce7" if abs(dh) <= 3 else ("#fee2e2" if dh < -3 else "#dbeafe")
            dh_e = "#15803d" if abs(dh) <= 3 else ("#dc2626" if dh < -3 else "#2563eb")

            ax.text(
                0.5,
                -0.16,
                f"ROCS: {card['rocs']:.3f}   Raw SA: {card['sa']:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25", facecolor=rocs_c, edgecolor=rocs_e, linewidth=1.0
                ),
            )
            ax.text(
                0.5,
                -0.28,
                f"QED: {card['qed']:.2f}   ΔHAC: {dh:+d} (MW: {card['mw']:.0f})",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=dh_c, edgecolor=dh_e, linewidth=0.9),
            )
            sc_str = (
                f"Shape: {card['shape']:.2f} Color: {card['color']:.2f} | "
                if math.isfinite(card.get("shape", math.nan))
                and math.isfinite(card.get("color", math.nan))
                else ""
            )
            ax.text(
                0.5,
                -0.39,
                f"{sc_str}ECFP4 Sim: {card['sim']:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7.8,
                color="#475569",
            )

    fig.savefig(target_png, dpi=dpi)
    plt.close(fig)
    return target_png


def lead_structures_gallery(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    out: Path,
    config: dict | None = None,
    reference: Path | None = None,
    arm: str = FINAL_ARM,
    name: str = "05_lead_structures_gallery",
    fallback: bool = True,
) -> Path | None:
    """The lead cards of one arm. Four galleries of the same layout taught nothing that
    the two the report actually discusses do not."""
    win = shown_arm(arms, objectives, arm, fallback=fallback)
    if win is None:
        return None
    frame = arms[win]
    u = _unique(frame)
    des = float(u["Desired"].sum()) / max(len(frame), 1) if "Desired" in u else 0.0
    # The cards show one molecule per chemotype, so the title has to say how many
    # chemotypes there were: five distinct cards drawn from 40 scaffolds and five drawn
    # from 1900 are the same picture of very different runs. enrich_descriptors caches
    # per molecule, so this re-uses the work _render_lead_cards is about to do.
    drawn = u[pd.to_numeric(u["Desired"], errors="coerce").fillna(0).astype(int) == 1] if "Desired" in u else u
    scaffolds = enrich_descriptors(
        drawn, objectives[0], (config or {}).get("reference_smiles", ""), config
    )
    n_scaffolds = int(scaffolds["Scaffold"].replace("", np.nan).nunique()) if not scaffolds.empty else 0
    return _render_lead_cards(
        frame,
        objectives,
        out / f"{name}.png",
        "Top leads (row 1), then progressively more distinct molecules — desired only, "
        f"one card per chemotype. Arm: {win} (desired={des:.3f}, {len(drawn)} desired of "
        f"{len(u)} unique / {len(frame)} sampled, {n_scaffolds} Murcko scaffolds among them)",
        config=config,
        reference=reference,
    )


# --------------------------------------------------------------------------------------
# 06 -- Objective Conflict & Property Coupling (Spearman rho: Prior, Reinforced, Delta)
# --------------------------------------------------------------------------------------


CONFLICT_PROPERTIES: tuple[tuple[str, str], ...] = (
    ("ROCS", "ROCS"),
    ("ShapeTanimoto", "Shape"),
    ("ColorTanimoto", "Color"),
    ("Raw_SA", "SA (raw)"),
    ("QED", "QED"),
    ("MW", "MW"),
    ("HAC", "HAC"),
    ("LogP", "LogP"),
    ("TPSA", "TPSA"),
    ("RotBonds", "RotB"),
    ("Fsp3", "Fsp³"),
    ("Rings", "Rings"),
)


def _spearman_matrix(df: pd.DataFrame, props: list[tuple[str, str]]) -> np.ndarray:
    n = len(props)
    mat = np.full((n, n), np.nan)
    if df is None or df.empty:
        return mat
    for i, (ca, _) in enumerate(props):
        for j, (cb, _) in enumerate(props):
            if ca not in df.columns or cb not in df.columns:
                continue
            sa = pd.to_numeric(df[ca], errors="coerce")
            sb = pd.to_numeric(df[cb], errors="coerce")
            mask = sa.notna() & sb.notna()
            if (
                int(mask.sum()) > 2
                and float(sa[mask].std()) > 1e-6
                and float(sb[mask].std()) > 1e-6
            ):
                mat[i, j] = 1.0 if i == j else float(sa[mask].corr(sb[mask], method="spearman"))
    return mat


def objective_conflict_correlations(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    out: Path,
    config: dict | None = None,
    reference: Path | None = None,
) -> Path | None:
    if len(objectives) < 2:
        return None
    style()
    win = shown_arm(arms, objectives, PRIMARY_ARM)
    if win is None and "prior" not in arms:
        return None
    target_arm = win or "prior"
    ref_smi = (config or {}).get("reference_smiles")
    a_frame = enrich_descriptors(_unique(arms.get("prior")), objectives[0], ref_smi, config)
    b_frame = enrich_descriptors(_unique(arms.get(target_arm)), objectives[0], ref_smi, config)
    if b_frame.empty and a_frame.empty:
        return None

    sample = b_frame if not b_frame.empty else a_frame
    props = [
        (col, lbl)
        for col, lbl in CONFLICT_PROPERTIES
        if col in sample.columns and sample[col].notna().sum() >= 3
    ]
    if len(props) < 2:
        return None
    labels = [lbl for _, lbl in props]
    mat_a = _spearman_matrix(a_frame, props)
    mat_b = _spearman_matrix(b_frame, props)
    mat_d = mat_b - mat_a

    prov = provenance_tag(config, reference)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=200)
    fig.suptitle(
        f"Objective Conflict & Property Coupling (Spearman ρ, valid unique): Did RL Couple the Objectives?\n{prov}",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )

    for ax, matrix, title in [
        (axes[0], mat_a, f"(A) Baseline Prior (n={len(a_frame)})"),
        (axes[1], mat_b, f"(B) Reinforced Agent: {target_arm} (n={len(b_frame)})"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        color="white" if abs(val) > 0.55 else "#111827",
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("ρ", fontsize=8)

    dmax = max(0.35, float(np.nanmax(np.abs(mat_d))) if np.isfinite(mat_d).any() else 0.35)
    im3 = axes[2].imshow(np.ma.masked_invalid(mat_d), cmap="PuOr_r", vmin=-dmax, vmax=dmax)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    axes[2].set_yticklabels(labels, fontsize=8.5)
    axes[2].set_title("(C) Coupling Change  (B − A)", fontsize=11, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat_d[i, j]
            if np.isfinite(val):
                axes[2].text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if abs(val) > 0.55 * dmax / 0.35 else "#111827",
                )
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.03).set_label("Δρ", fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    target_png = out / "06_objective_conflict_correlations.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 07 -- Bemis-Murcko Scaffold Space: Concentration, De Novo Novelty & Dominant Cores
# --------------------------------------------------------------------------------------


def _spread_scaffolds(counts: pd.Series, k: int) -> list[str]:
    """The ``k`` scaffolds of ``counts`` that are least like each other.

    Panel C ranks cores by prevalence, and a collapsed run's twelve commonest cores are
    twelve decorations of one chemistry -- true and unreadable. This walks the other
    axis: start at the commonest core and repeatedly take whichever remaining scaffold
    is farthest from everything already taken (max-min ECFP4 distance), so the panel
    shows how wide the arm's chemistry actually got. Prevalence stays in each legend, so
    a core that is one molecule in five thousand still reads as one.
    """
    scaffolds = [s for s in counts.index if s]
    if len(scaffolds) <= k:
        return scaffolds
    mols = [Chem.MolFromSmiles(s) for s in scaffolds]
    pairs = [(s, m) for s, m in zip(scaffolds, mols) if m is not None]
    if len(pairs) <= k:
        return [s for s, _ in pairs]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for _, m in pairs]
    chosen = [0]
    nearest = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[0], fps))
    for _ in range(k - 1):
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(
            nearest, 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[nxt], fps))
        )
        nearest[nxt] = -1.0
    return [pairs[i][0] for i in chosen]


def scaffold_diversity(
    arms: dict[str, pd.DataFrame],
    out: Path,
    config: dict | None = None,
    reference: Path | None = None,
    arm: str = PRIMARY_ARM,
    name: str = "07_bemis_murcko_scaffold_diversity",
) -> Path | None:
    rows = scaffold_rows(arms)
    if not rows:
        return None
    style()
    objs = (config or {}).get("objectives", ["ROCS"])
    win = shown_arm(arms, objs, arm, fallback=arm == PRIMARY_ARM)
    if win is None:
        return None
    b_counts = scaffold_counts(arms["prior"]) if "prior" in arms else pd.Series(dtype=int)
    a_counts = scaffold_counts(arms[win]) if win in arms else pd.Series(dtype=int)
    # The graph framework in the legend beside the Murcko count, because the gap between
    # them is how much of a run's apparent diversity is heteroatom decoration of one core.
    b_graphs = scaffold_counts(arms["prior"], generic=True) if "prior" in arms else pd.Series(dtype=int)
    a_graphs = scaffold_counts(arms[win], generic=True) if win in arms else pd.Series(dtype=int)
    if a_counts.empty and b_counts.empty:
        return None

    def _entropy(counts: pd.Series) -> float:
        if counts.empty:
            return 0.0
        p = counts.values / counts.values.sum()
        return float(-np.sum(p * np.log2(p)))

    b_ent, a_ent = _entropy(b_counts), _entropy(a_counts)
    b_set, a_set = set(b_counts.index), set(a_counts.index)
    shared, novel, lost = a_set & b_set, a_set - b_set, b_set - a_set
    jaccard = len(shared) / len(a_set | b_set) if (a_set | b_set) else 0.0
    prov = provenance_tag(config, reference)

    fig = plt.figure(figsize=(20, 17.5), dpi=200)
    # Row 1 is 1 : 1 : 2 -- the two statistics panels are small plots, the scaffold grid
    # is the figure's subject and needs half the width to be legible at 12 cores. Row 2
    # is the whole width, because panel D's job is breadth and it shows its cores larger
    # than C does.
    grid = fig.add_gridspec(2, 4)
    fig.suptitle(
        f"Bemis-Murcko Scaffold Space: Concentration, De Novo Novelty & Dominant Cores ({win} vs prior)\n{prov}",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )

    # Panel A: Scaffold Concentration Curve
    ax1 = fig.add_subplot(grid[0, 0])
    for counts, color, label in [
        (b_counts, "#64748B", f"Baseline Prior ({len(b_counts)} scaffolds, {len(b_graphs)} frameworks)"),
        (a_counts, "#4F46E5", f"Reinforced ({win}: {len(a_counts)} scaffolds, {len(a_graphs)} frameworks)"),
    ]:
        if counts.empty:
            continue
        coverage = np.cumsum(counts.values) / counts.values.sum() * 100.0
        ranks = np.arange(1, len(coverage) + 1)
        ax1.plot(ranks, coverage, color=color, linewidth=2.0, label=label, zorder=3)
        ax1.fill_between(ranks, 0, coverage, color=color, alpha=0.10, linewidth=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Top-N scaffolds (log rank)", fontsize=9.5)
    ax1.set_ylabel("% of Ring-Containing Unique Library Covered", fontsize=9.5)
    ax1.set_ylim(0, 102)
    ax1.axhline(80, color="#CBD5E1", linestyle="--", linewidth=1.0, zorder=2)
    ax1.text(
        0.98,
        82,
        "80% coverage",
        transform=ax1.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#64748B",
    )
    ax1.set_title("(A) Scaffold Concentration Curve", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.text(
        0.03,
        0.96,
        f"Shannon H: prior {b_ent:.2f} → {win} {a_ent:.2f} bits",
        transform=ax1.transAxes,
        va="top",
        fontsize=8.2,
        color="#334155",
    )

    # Panel B: Scaffold Novelty Partition
    ax2 = fig.add_subplot(grid[0, 1])
    parts = [
        ("Shared with prior", len(shared), "#94A3B8"),
        ("De novo (new)", len(novel), "#4F46E5"),
        ("Prior-only (lost)", len(lost), "#E11D48"),
    ]
    x = np.arange(len(parts))
    vals = [v for _, v, _ in parts]
    bars = ax2.bar(
        x, vals, color=[c for _, _, c in parts], alpha=0.9, edgecolor="#1e293b", width=0.6
    )
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + (max(vals) * 0.02 if max(vals) else 0.1),
            str(val),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([n for n, _, _ in parts], fontsize=8.6)
    ax2.set_ylabel("Distinct Scaffolds (valid unique)", fontsize=9.5)
    ax2.set_title(f"(B) Scaffold Novelty Partition ({win})", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.4, axis="y")
    ax2.text(
        0.5,
        0.93,
        f"Jaccard vs prior = {jaccard:.3f}",
        transform=ax2.transAxes,
        ha="center",
        fontsize=8.6,
        color="#334155",
    )

    # Panel C: Top-12 Reinforced Scaffolds
    ax3 = fig.add_subplot(grid[0, 2:])
    top = a_counts.head(12)
    scaf_mols = [Chem.MolFromSmiles(s) for s in top.index]
    total = max(int(a_counts.sum()), 1)
    legends = [
        f"{cnt / total * 100:.1f}% | {'NEW' if s in novel else 'shared'}" for s, cnt in top.items()
    ]
    if scaf_mols and all(m is not None for m in scaf_mols):
        img = Draw.MolsToGridImage(
            scaf_mols, molsPerRow=3, subImgSize=(420, 330), legends=legends, useSVG=False
        )
        ax3.imshow(img)
        ax3.axis("off")
        ax3.set_title(
            f"(C) Top-12 {win} Scaffolds (prevalence | novelty)", fontsize=11, fontweight="bold"
        )
    else:
        ax3.text(0.5, 0.5, "Insufficient scaffold data", ha="center", va="center")
        ax3.axis("off")

    # Panel D: the widest 12 scaffolds this arm reached
    ax4 = fig.add_subplot(grid[1, :])
    spread = _spread_scaffolds(a_counts, 12)
    spread_mols = [Chem.MolFromSmiles(s) for s in spread]
    if spread_mols and all(m is not None for m in spread_mols):
        img = Draw.MolsToGridImage(
            spread_mols,
            molsPerRow=4,
            subImgSize=(420, 330),
            legends=[
                f"{a_counts[s] / total * 100:.1f}% | {'NEW' if s in novel else 'shared'}"
                for s in spread
            ],
            useSVG=False,
        )
        ax4.imshow(img)
        ax4.axis("off")
        ax4.set_title(
            f"(D) The 12 most dissimilar {win} scaffolds, farthest-point on ECFP4 from the "
            "most common one (prevalence | novelty)",
            fontsize=11,
            fontweight="bold",
        )
    else:
        ax4.text(0.5, 0.5, "Insufficient scaffold data", ha="center", va="center")
        ax4.axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    target_png = out / f"{name}.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 10 -- 3D Molecular Superposition: Target Query Seed vs. Prioritized De Novo Leads
# --------------------------------------------------------------------------------------


def _leads_for_3d(
    arms: dict[str, pd.DataFrame], objectives: list[str]
) -> tuple[str, pd.DataFrame] | None:
    name = shown_arm(arms, objectives, PRIMARY_ARM) or "prior"
    frame = arms.get(name)
    if frame is None or _unique(frame).empty:
        return None
    # Same rule as the lead gallery: a molecule below the thresholds is not a lead, so
    # it does not get a pose drawn against the reference either.
    pool = _unique(frame)
    if "Desired" in pool.columns:
        desired = pool[pd.to_numeric(pool["Desired"], errors="coerce").fillna(0).astype(int) == 1]
        if not desired.empty:
            pool = desired
    return name, pool.nlargest(TOP_3D, objectives[0])


def lead_to_seed_superposition(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    reference: Path,
    out: Path,
    generator=None,
    config: dict | None = None,
) -> Path | None:
    if generator is None or not reference.is_file():
        return None
    refs = [m for m in Chem.SDMolSupplier(str(reference), removeHs=False) if m is not None]
    if not refs:
        return None
    selection = _leads_for_3d(arms, objectives)
    if selection is None:
        return None
    arm, best = selection
    color_ff = (config or {}).get("color_ff", "fdef")
    shape_wt = float((config or {}).get("alignment_shape_weight", 0.5))
    prov = provenance_tag(config, reference)
    ref_smi = (config or {}).get("reference_smiles")
    best_en = enrich_descriptors(best, objectives[0], ref_smi, config)

    aligned_leads: list[tuple[Chem.Mol, Chem.Mol, dict[str, Any]]] = []
    for rank, (_, row) in enumerate(best_en.iterrows(), start=1):
        smiles = str(row["SMILES"])
        # The enriched column, not the stored one: `best_en` already carries published
        # TanimotoCombo, while row[objectives[0]] is whatever units the run stored. Panel
        # B prints this beside the pose's own shape + colour, so the two must match.
        pipe_score = float(row["ROCS"]) if "ROCS" in row else float(row[objectives[0]])
        try:
            res = align_lead_to_bank(
                smiles, refs, generator, color_ff=color_ff, alignment_shape_weight=shape_wt
            )
            if res is None:
                continue
            ref_pose, aligned_probe, combo, sh, col, ref_idx = res
            ref_heavy = Chem.RemoveHs(ref_pose)
            lead_heavy = Chem.RemoveHs(aligned_probe)
            seed_hac = ref_heavy.GetNumHeavyAtoms()
            lead_hac = lead_heavy.GetNumHeavyAtoms()
            props = {
                "Rank": rank,
                "SMILES": smiles,
                "source_arm": arm,
                f"pipeline_{objectives[0]}": pipe_score,
                "ROCS": pipe_score,
                "SA": float(row.get("Raw_SA", row.get("SA_raw", 0.0))),
                "QED": float(row.get("QED", 0.0)),
                "MW": float(row.get("MW", Descriptors.MolWt(lead_heavy))),
                "HAC": lead_hac,
                "Delta_HAC": lead_hac - seed_hac,
                "TanimotoCombo": combo,
                "ShapeScore": sh,
                "ColorScore": col,
                "matched_reference_conformer": ref_idx,
            }
            aligned_leads.append((ref_heavy, lead_heavy, props))
        except Exception as error:
            logger.warning("3D overlay failed for %s: %s", smiles, error)

    if not aligned_leads:
        return None

    ref_heavy0 = aligned_leads[0][0]
    seed_hac = ref_heavy0.GetNumHeavyAtoms()

    writer = Chem.SDWriter(str(out / "top_leads_aligned_to_seed.sdf"))
    for _, lead_mol, props in aligned_leads:
        m_out = Chem.Mol(lead_mol)
        m_out.SetProp("_Name", props["SMILES"])
        for k, v in props.items():
            # "SA" in this project means the ClippedScore desirability on [0, 1]; what
            # the lead carries is the raw 1-10 score, so it is written under that name.
            key = "SA_raw" if k == "SA" else str(k)
            if isinstance(v, float):
                m_out.SetDoubleProp(key, v)
            else:
                m_out.SetProp(key, str(v))
        m_out.SetProp(
            "pose_note",
            "illustrative re-alignment using pipeline color_ff and alignment_shape_weight",
        )
        writer.write(m_out)
    writer.close()

    style()
    fig = plt.figure(figsize=(16, 6.1), dpi=200)
    fig.suptitle(
        f"3D Molecular Superposition: Target Query Seed vs. Top Leads of {arm}\n{prov}",
        fontsize=12.2,
        fontweight="bold",
        y=0.99,
    )

    def _draw_3d(
        ax: Any, mol: Chem.Mol, color: str, alpha: float = 0.8, lw: float = 1.3, s: float = 14
    ) -> None:
        conf = mol.GetConformer()
        pts = conf.GetPositions()
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [pts[i, 2], pts[j, 2]],
                color=color,
                linewidth=lw,
                alpha=alpha,
            )
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=s, alpha=alpha)

    frames = [ref_heavy0.GetConformer().GetPositions()] + [
        m.GetConformer().GetPositions() for _, m, _ in aligned_leads
    ]
    stack = np.vstack(frames)
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    pad = 0.06 * max(float((hi - lo).max()), 1e-3)
    lo, hi = lo - pad, hi + pad

    def _share_frame(ax: Any) -> None:
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_zlim(float(lo[2]), float(hi[2]))
        ax.set_box_aspect(tuple(float(v) for v in (hi - lo)))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_3d(ax1, ref_heavy0, color="#2563eb", alpha=0.9, lw=1.6, s=18)
    ax1.set_title(
        f"(A) Target Query Seed (Bank Pose #{aligned_leads[0][2]['matched_reference_conformer'] + 1}/{len(refs)})\nHAC = {seed_hac} | MW = {Descriptors.MolWt(ref_heavy0):.1f} Da",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax1.view_init(elev=20, azim=40)
    ax1.set_axis_off()
    _share_frame(ax1)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_3d(ax2, ref_heavy0, color="#94a3b8", alpha=0.45, lw=1.2, s=10)
    _, lead1, p1 = aligned_leads[0]
    _draw_3d(ax2, lead1, color="#16a34a", alpha=0.9, lw=1.6, s=18)
    ax2.set_title(
        f"(B) Top Lead #1 Aligned to Seed\nPose combo = {p1['TanimotoCombo']:.2f} (shape {p1['ShapeScore']:.2f}, color {p1['ColorScore']:.2f}) | pipeline = {p1['ROCS']:.3f}\nΔHAC = {p1['Delta_HAC']:+d} | Raw SA = {p1['SA']:.2f}",
        fontsize=9.6,
        fontweight="bold",
        pad=10,
    )
    ax2.view_init(elev=20, azim=40)
    ax2.set_axis_off()
    _share_frame(ax2)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_3d(ax3, ref_heavy0, color="#1e293b", alpha=0.6, lw=1.5, s=14)
    lead_colors = ["#16a34a", "#2563eb", "#ea580c", "#8b5cf6", "#d97706"]
    # Only the leads that matched the conformer this panel draws. Each bank conformer is
    # canonicalised into its own inertial frame, so a lead aligned to another one would
    # be drawn in a frame it was never fitted to -- it would look mis-overlaid, or fit,
    # for reasons that have nothing to do with its score.
    shown_pose = aligned_leads[0][2]["matched_reference_conformer"]
    same_pose = [
        entry for entry in aligned_leads if entry[2]["matched_reference_conformer"] == shown_pose
    ]
    for i, (_, l_mol, _p) in enumerate(same_pose):
        _draw_3d(ax3, l_mol, color=lead_colors[i % len(lead_colors)], alpha=0.75, lw=1.3, s=12)
    ax3.set_title(
        f"(C) Superposition Ensemble ({len(same_pose)} of {len(aligned_leads)} leads of "
        f"{arm} that matched bank pose #{shown_pose + 1})\n"
        f"Aligned via color_ff={color_ff}, shape_wt={shape_wt:g}",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax3.view_init(elev=20, azim=40)
    ax3.set_axis_off()
    _share_frame(ax3)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    target_png = out / "08_lead_to_seed_3d_superposition.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 11 -- Longitudinal Per-Epoch Property Trajectories (3x3: Median, IQR, p95, Prior & Seed)
# --------------------------------------------------------------------------------------


def epoch_property_trajectories(
    epochs: pd.DataFrame,
    prior: pd.DataFrame,
    out: Path,
    objectives: list[str] | None = None,
    config: dict | None = None,
    reference: Path | None = None,
) -> Path | None:
    if epochs is None or epochs.empty:
        return None
    vep = unique_per_epoch(epochs)
    if vep.empty:
        return None
    style()
    rocs_col = (objectives or ["ROCS"])[0]
    ref_smi = (config or {}).get("reference_smiles")
    valid_df = enrich_descriptors(vep, rocs_col, ref_smi, config)
    prior_df = (
        enrich_descriptors(_unique(prior), rocs_col, ref_smi, config)
        if prior is not None and not prior.empty
        else pd.DataFrame()
    )
    ref_vals = reference_properties(config, reference)
    prov = provenance_tag(config, reference)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), dpi=200)
    fig.suptitle(
        f"Longitudinal Per-Epoch Property Trajectories (Median, IQR, p95, Prior Baseline & Seed Reference)\n{prov}",
        fontsize=12.5,
        fontweight="bold",
        y=0.992,
    )

    panels_config = [
        ("ROCS", "ROCS TanimotoCombo (3D Shape+Color)", "TanimotoCombo (0–2)", "ROCS"),
        ("QED", "QED Drug-Likeness", "QED (0–1)", "QED"),
        (
            "Raw_SA",
            "Synthetic Accessibility (raw RDKit SA)",
            "SA [1, 10] (lower = easier)",
            "SA_raw",
        ),
        ("MW", "Molecular Weight Drift", "MW (Da)", "MW"),
        ("HAC", "Heavy Atom Count (Shrinkage Guard)", "Atoms", "HAC"),
        ("LogP", "Lipophilicity", "cLogP", "cLogP"),
        ("TPSA", "Polar Surface Area", "TPSA (Å²)", "TPSA"),
        ("Fsp3", "Carbon sp³ Fraction", "Fraction", "Fsp3"),
        ("RotBonds", "Rotatable Bonds", "Bonds", "RotB"),
    ]

    for idx, (col, p_title, y_lbl, r_key) in enumerate(panels_config):
        ax = axes[idx // 3, idx % 3]
        if col not in valid_df.columns or valid_df[col].notna().sum() == 0:
            ax.axis("off")
            continue

        grp = valid_df.groupby("Epoch")[col]
        med, p25, p75, p95 = (
            grp.median(),
            grp.quantile(0.25),
            grp.quantile(0.75),
            grp.quantile(0.95),
        )
        if med.dropna().empty:
            ax.axis("off")
            continue
        ep = med.index.values

        ax.plot(ep, med.values, color="#4F46E5", linewidth=2.0, label="Agent Median", zorder=4)
        ax.fill_between(
            ep,
            p25.values,
            p75.values,
            color="#4F46E5",
            alpha=0.20,
            linewidth=0,
            label="IQR (25th–75th %ile)",
            zorder=2,
        )
        ax.plot(
            ep,
            p95.values,
            color="#E11D48",
            linestyle="-.",
            linewidth=1.4,
            alpha=0.85,
            label="Upper 95th %ile",
            zorder=3,
        )

        if not prior_df.empty and col in prior_df.columns:
            p_vals = pd.to_numeric(prior_df[col], errors="coerce").dropna()
            if not p_vals.empty:
                p_med = float(p_vals.median())
                ax.axhline(
                    p_med,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Prior Med ({p_med:.2f})",
                    zorder=2,
                )

        ref_v = ref_vals.get(r_key)
        if ref_v is not None and math.isfinite(ref_v):
            ax.axhline(
                ref_v,
                color="#0D9488",
                linestyle=":",
                linewidth=1.6,
                label=f"Seed ({ref_v:.2f})",
                zorder=2,
            )

        lo, hi = float(np.nanmin(p25.values)), float(np.nanmax(p95.values))
        if ref_v is not None and math.isfinite(ref_v) and col != "ROCS":
            lo, hi = min(lo, ref_v), max(hi, ref_v)
        span = hi - lo if hi > lo else 1.0
        ax.set_ylim(lo - 0.10 * span, hi + 0.10 * span)
        ax.set_title(f"({chr(65 + idx)}) {p_title}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9.0)
        ax.set_ylabel(y_lbl, fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=7.2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.955])
    target_png = out / "04_epoch_property_trajectories.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------
# 12 -- Six-Panel Longitudinal Training Diagnostics
# --------------------------------------------------------------------------------------


def training_diagnostics(
    fit: pd.DataFrame,
    epochs: pd.DataFrame,
    prior: pd.DataFrame | None = None,
    objectives: list[str] | None = None,
    out: Path | None = None,
    config: dict | None = None,
    reference: Path | None = None,
    arms: dict[str, pd.DataFrame] | None = None,
    summary: dict | None = None,
) -> Path | None:
    if isinstance(prior, Path) and out is None:
        out = prior
        prior = None
    if fit is None or fit.empty or out is None:
        return None
    if epochs is None or epochs.empty:
        return None

    style()
    if not objectives:
        objectives = [c for c in epochs.columns if c.startswith("RDKit_")] + (
            ["SA"] if "SA" in epochs.columns else []
        )
    objectives = objectives or ["ROCS", "SA"]
    rocs_col = objectives[0]
    ref_smi = (config or {}).get("reference_smiles", "")
    prov = provenance_tag(config, reference)
    summary = summary or (config or {}).get("summary", {})

    table = enrich_descriptors(epochs, rocs_col, ref_smi, config)
    val_df = _valid(table)
    if val_df.empty:
        return None

    ep_nums = np.array(sorted(val_df["Epoch"].unique()), dtype=int)
    # The arm this figure delivers, resolved here because the "best checkpoint" marker
    # below has to agree with it.
    win = shown_arm(arms or {}, objectives, PRIMARY_ARM)
    # The epoch of that arm, not whichever criterion peaked first: marking the
    # desired_ratio epoch on a figure whose delivered arm is gmean points at a
    # checkpoint that produced none of the molecules shown.
    best_epoch = (
        summary.get("best_avg_gmean_epoch")
        if win == "gmean"
        else summary.get("best_desired_ratio_epoch")
    ) or summary.get("best_avg_gmean_epoch")

    # (A) Hypervolume per epoch and cumulative in actual objective space
    hv_epoch: list[float] = []
    hv_cumulative: list[float] = []
    cum_front = np.empty((0, len(objectives)))
    for ep in ep_nums:
        grp = val_df[val_df["Epoch"] == ep]
        pts = scored_points(grp, objectives)
        hv_epoch.append(dominated_volume(pts))
        if len(pts):
            cum_front = np.vstack([cum_front, pts])
            cum_front = cum_front[pareto_mask(cum_front)]
        hv_cumulative.append(dominated_volume(cum_front))

    # (C) Centroid trajectory in ECFP4 seed similarity vs ROCS space
    centroid_sim, centroid_rocs = [], []
    for ep in ep_nums:
        grp = val_df[val_df["Epoch"] == ep]
        centroid_sim.append(float(grp["SimToSeed"].mean()) if "SimToSeed" in grp else 0.0)
        centroid_rocs.append(float(grp["ROCS"].mean()) if "ROCS" in grp else 0.0)
    centroid_sim_arr = np.asarray(centroid_sim, dtype=float)
    centroid_rocs_arr = np.asarray(centroid_rocs, dtype=float)

    prior_en = (
        enrich_descriptors(_unique(prior), rocs_col, ref_smi, config)
        if prior is not None and not prior.empty
        else pd.DataFrame()
    )

    # (D) Per-epoch sample outcome composition (deduplicating on Canonical SMILES)
    outcomes: list[str] = []
    seen_canonical: set[str] = set()
    for _, row in table.iterrows():
        is_val = int(row.get("Valid", 0)) == 1
        is_des = float(row.get("Desired", 0.0) or 0.0) > 0.5
        canon = str(row.get("Canonical") or "")
        if not is_val:
            outcomes.append("invalid")
        elif not is_des:
            outcomes.append("valid_undesired")
        elif canon and canon in seen_canonical:
            outcomes.append("desired_duplicate")
        else:
            outcomes.append("desired_novel")
            if canon:
                seen_canonical.add(canon)
    outcome_frame = table.copy()
    outcome_frame["Outcome"] = outcomes
    outcome_order = ["desired_novel", "desired_duplicate", "valid_undesired", "invalid"]
    outcome_labels = {
        "desired_novel": "Desired & new",
        "desired_duplicate": "Desired but duplicate",
        "valid_undesired": "Valid but undesired",
        "invalid": "Unparsable",
    }
    outcome_colors = {
        "desired_novel": "#34D399",
        "desired_duplicate": "#93C5FD",
        "valid_undesired": "#FDE68A",
        "invalid": "#FCA5A5",
    }
    outcome_counts = pd.crosstab(outcome_frame["Epoch"], outcome_frame["Outcome"]).reindex(
        columns=outcome_order, fill_value=0
    )
    outcome_fractions = outcome_counts.div(outcome_counts.sum(axis=1), axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.4), dpi=200)
    fig.suptitle(
        f"Training Diagnostics: Weight-Free Progress, Population Path and Lead Discovery\n{prov}",
        fontsize=12.8,
        fontweight="bold",
        y=0.99,
    )

    def _mark_best(ax: plt.Axes) -> None:
        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color="#7C3AED",
                linewidth=1.4,
                linestyle="-.",
                label=f"Best checkpoint (ep {best_epoch})",
                zorder=1,
            )

    # Panel A: Dominated Hypervolume per Epoch
    ax = axes[0, 0]
    _mark_best(ax)
    ax.plot(
        ep_nums,
        hv_epoch,
        color="#93C5FD",
        linewidth=1.1,
        alpha=0.9,
        label="Per-epoch hypervolume",
        zorder=2,
    )
    ax.plot(
        ep_nums,
        pd.Series(hv_epoch).rolling(5, center=True, min_periods=1).median().to_numpy(),
        color="#1D4ED8",
        linewidth=2.2,
        label="Per-epoch (rolling median)",
        zorder=3,
    )
    ax.plot(
        ep_nums,
        hv_cumulative,
        color="#0F172A",
        linewidth=1.8,
        linestyle="--",
        label="Cumulative hypervolume",
        zorder=3,
    )
    if not prior_en.empty:
        p_pts = scored_points(prior_en, objectives)
        if len(p_pts):
            p_hv = dominated_volume(p_pts)
            ax.axhline(
                p_hv,
                color="#64748B",
                linewidth=1.3,
                linestyle=":",
                label=f"Prior arm ({p_hv:.3f}, n={len(p_pts)})",
                zorder=1,
            )
    ax.set_title(
        "(A) Dominated Hypervolume per Epoch (in objective space)", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dominated objective volume")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.4)

    # Panel B: Pareto Front at the Best Delivered Arm (colour = QED)
    ax = axes[0, 1]
    if not prior_en.empty and {"ROCS", "Raw_SA"}.issubset(prior_en.columns):
        p_sub = prior_en[["ROCS", "Raw_SA"]].dropna()
        ax.scatter(
            p_sub["ROCS"],
            p_sub["Raw_SA"],
            s=10,
            color="#CBD5E1",
            alpha=0.55,
            linewidth=0,
            label="Baseline prior",
            zorder=1,
        )
    deliv_df = (
        enrich_descriptors(_unique(arms[win]), rocs_col, ref_smi, config)
        if (arms and win and win in arms)
        else val_df[val_df["Epoch"] == ep_nums[-1]]
    )
    stage_short = f"Arm {win}" if win else f"Epoch {ep_nums[-1]}"
    sub_d = deliv_df.dropna(
        subset=[c for c in ("ROCS", "Raw_SA", "QED", *objectives) if c in deliv_df.columns]
    )
    if not sub_d.empty:
        d_pts = sub_d[[o for o in objectives if o in sub_d.columns]].to_numpy(dtype=float)
        mask_f = pareto_mask(d_pts) if d_pts.size else np.zeros(len(sub_d), dtype=bool)
        ax.scatter(
            sub_d["ROCS"],
            sub_d["Raw_SA"],
            c=sub_d["QED"],
            cmap="viridis",
            s=42,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.4,
            label=stage_short,
            zorder=2,
        )
        front_sc = ax.scatter(
            sub_d.loc[sub_d.index[mask_f], "ROCS"],
            sub_d.loc[sub_d.index[mask_f], "Raw_SA"],
            c=sub_d.loc[sub_d.index[mask_f], "QED"],
            cmap="viridis",
            s=95,
            edgecolor="#111827",
            linewidth=1.0,
            label="Pareto front",
            zorder=3,
        )
        plt.colorbar(front_sc, ax=ax, label="QED", fraction=0.046, pad=0.02)
        y_lo, y_hi = float(sub_d["Raw_SA"].min()), float(sub_d["Raw_SA"].max())
        y_pad = 0.08 * max(y_hi - y_lo, 0.5)
        ax.set_ylim(y_hi + y_pad, max(1.0, y_lo - y_pad))
    ax.set_title(
        f"(B) Pareto Front at {stage_short} (colour = QED)", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("ROCS TanimotoCombo (0–2)")
    ax.set_ylabel("Raw RDKit SA [1, 10] (lower = easier)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=7.2)

    # Panel C: Fidelity to Seed vs Shape Fit
    ax = axes[0, 2]
    if not prior_en.empty and {"SimToSeed", "ROCS"}.issubset(prior_en.columns):
        ps = prior_en[["SimToSeed", "ROCS"]].dropna()
        ax.scatter(
            ps["SimToSeed"],
            ps["ROCS"],
            s=8,
            color="#CBD5E1",
            alpha=0.45,
            linewidth=0,
            label="Baseline prior",
            zorder=1,
        )
    ax.text(
        0.02,
        0.97,
        "seed: sim 1.00 · ROCS 2.00 (off-scale)",
        transform=ax.transAxes,
        fontsize=7.6,
        color="#7F1D1D",
        va="top",
        ha="left",
    )
    if np.isfinite(centroid_sim_arr).any():
        ax.plot(centroid_sim_arr, centroid_rocs_arr, color="#94A3B8", linewidth=1.2, zorder=2)
        norm = plt.Normalize(vmin=float(ep_nums.min()), vmax=float(ep_nums.max()))
        ep_sc = ax.scatter(
            centroid_sim_arr,
            centroid_rocs_arr,
            c=ep_nums,
            cmap="viridis",
            norm=norm,
            s=34,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.scatter(
            [centroid_sim_arr[0]],
            [centroid_rocs_arr[0]],
            marker="o",
            s=110,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.4,
            label=f"Epoch {ep_nums[0]}",
            zorder=4,
        )
        ax.scatter(
            [centroid_sim_arr[-1]],
            [centroid_rocs_arr[-1]],
            marker="s",
            s=95,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.4,
            label=f"Epoch {ep_nums[-1]}",
            zorder=4,
        )
        if best_epoch is not None and best_epoch in set(ep_nums.tolist()):
            idx_b = int(np.flatnonzero(ep_nums == best_epoch)[0])
            ax.scatter(
                [centroid_sim_arr[idx_b]],
                [centroid_rocs_arr[idx_b]],
                marker="*",
                s=240,
                color="#7C3AED",
                edgecolor="white",
                linewidth=0.8,
                label=f"Best checkpoint (ep {best_epoch})",
                zorder=5,
            )
        plt.colorbar(ep_sc, ax=ax, label="Epoch", fraction=0.046, pad=0.02)
    ax.set_title("(C) Fidelity to the Seed vs. Shape Fit", fontsize=11, fontweight="bold")
    ax.set_xlabel("ECFP4 Tanimoto similarity to seed (centroid)")
    ax.set_ylabel("ROCS TanimotoCombo (centroid)")
    ax.set_xlim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.0)

    # Panel D: Composition of Each Epoch's Samples
    ax = axes[1, 0]
    ax.stackplot(
        outcome_fractions.index,
        *[outcome_fractions[c] for c in outcome_order],
        labels=[outcome_labels[c] for c in outcome_order],
        colors=[outcome_colors[c] for c in outcome_order],
        alpha=0.85,
        linewidth=0,
    )
    ax.plot(
        outcome_fractions.index,
        1.0 - outcome_fractions["invalid"],
        color="#111827",
        linewidth=1.1,
        linestyle="--",
        label="Monitor valid ratio",
    )
    ax.plot(
        outcome_fractions.index,
        outcome_fractions["desired_novel"],
        color="#064E3B",
        linewidth=1.3,
        linestyle="-",
        label="Desired and first seen in this epoch",
    )
    _mark_best(ax)
    # "first seen" is cumulative across the run, so this green line sits below the
    # desired_ratio of figure 03, which deduplicates inside one epoch only. Saying so
    # here stops the two being read as the same quantity disagreeing.
    ax.set_title(
        "(D) Composition of Each Epoch's Samples (novelty is cumulative over the run, "
        "so it sits below figure 03's within-epoch desired_ratio)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Share of samples (on n_eval)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower left", fontsize=6.8, ncol=2)

    # Panel E: Age of the Best Molecule (ROCS & gmean)
    ax = axes[1, 1]
    incumbent = -math.inf
    inc_ep: int | None = None
    ages: list[int] = []
    for ep in ep_nums:
        grp = val_df[val_df["Epoch"] == ep]["ROCS"].dropna()
        cur = float(grp.max()) if not grp.empty else -math.inf
        if cur > incumbent or inc_ep is None:
            incumbent, inc_ep = cur, int(ep)
        ages.append(int(ep) - int(inc_ep))
    ax.step(
        ep_nums, ages, where="post", color="#0D9488", linewidth=2.2, label="Best ROCS molecule age"
    )
    if "avg_gmean" in fit.columns:
        inc_g = -math.inf
        inc_gep: int | None = None
        ages_g: list[int] = []
        for _, r in fit.iterrows():
            ep_i = int(r["Epoch"])
            gv = float(r.get("avg_gmean", -math.inf))
            if gv > inc_g or inc_gep is None:
                inc_g, inc_gep = gv, ep_i
            ages_g.append(ep_i - int(inc_gep))
        ax.step(
            fit["Epoch"],
            ages_g,
            where="post",
            color="#2563EB",
            linewidth=1.8,
            linestyle="--",
            label="Best avg_gmean epoch age",
        )
    _mark_best(ax)
    ax.set_title("(E) Age of the Incumbent Best Molecule / Epoch", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epochs since best was found")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.4)

    # Panel F: When the Global Top-k Leads First Appeared (ranked by primary objective ROCS)
    ax = axes[1, 2]
    k = max(5, int(round(0.15 * float(val_df.groupby("Epoch").size().median()))))
    ranked = (
        val_df.dropna(subset=["ROCS"])
        .sort_values("ROCS", ascending=False)
        .drop_duplicates(subset=["Canonical"], keep="first")
    )
    global_top = ranked.head(k)
    first_seen: dict[str, int] = {}
    for ep in ep_nums:
        for s in val_df[val_df["Epoch"] == ep]["Canonical"]:
            if s:
                first_seen.setdefault(str(s), int(ep))
    found = [first_seen[str(s)] for s in global_top["Canonical"].tolist() if str(s) in first_seen]
    if found:
        discovery = [sum(1 for e in found if e <= cur) / len(found) for cur in ep_nums]
        ax.fill_between(ep_nums, 0, discovery, color="#DBEAFE", alpha=0.7, linewidth=0)
        ax.plot(
            ep_nums,
            discovery,
            color="#1D4ED8",
            linewidth=2.4,
            label=f"Fraction of global top-{len(found)} discovered",
        )
        median_ep = int(np.median(found))
        ax.axvline(
            median_ep,
            color="#7C3AED",
            linewidth=1.2,
            linestyle="--",
            label=f"Median discovery epoch ({median_ep})",
        )
        if best_epoch is not None:
            share = sum(1 for e in found if e <= best_epoch) / len(found)
            ax.scatter(
                [best_epoch],
                [share],
                marker="*",
                s=230,
                color="#7C3AED",
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
                label=f"Best checkpoint (ep {best_epoch})",
            )
            ax.text(
                0.02,
                0.96,
                f"{share:.0%} of top-{len(found)} existed by ep {best_epoch};\n{sum(1 for e in found if e > best_epoch)} appeared after it",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.4,
                color="#1E3A8A",
            )
    ax.set_title(
        f"(F) When the Global Top-{len(found) or k} ROCS Leads First Appeared",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative fraction of leads")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.4)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    target_png = out / "09_training_diagnostics.png"
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


# --------------------------------------------------------------------------------------


def render_all(
    *,
    arms: dict[str, pd.DataFrame],
    epochs: pd.DataFrame | None,
    fit: pd.DataFrame | None,
    objectives: list[str],
    thresholds: list[float],
    summary: dict,
    reference: Path,
    out: Path,
    generator=None,
    config: dict | None = None,
) -> list[Path]:
    """Render every figure in the suite. Removes stale PNGs first and logs any individual error."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        try:
            stale.unlink()
        except Exception:
            pass

    if config is None:
        cfg_path = out.parent / "config.json"
        if cfg_path.is_file():
            try:
                import json

                config = json.loads(cfg_path.read_text())
            except Exception:
                config = {}

    prior = arms.get("prior")
    # Numbered contiguously, in reading order: the two distribution dashboards, the
    # trade-offs, the two training views, the two lead galleries, the coupling matrix,
    # the two scaffold views, the 3D overlay and the diagnostics. A "b" figure is its
    # primary repeated on the desired_ratio checkpoint, never a different drawing.
    jobs = [
        (
            "01",
            lambda: before_after_distributions(
                arms,
                objectives,
                thresholds,
                out,
                config=config,
                reference=reference,
                arm=PRIMARY_ARM,
                name="01_before_after_distributions",
            ),
        ),
        (
            "01b",
            lambda: before_after_distributions(
                arms,
                objectives,
                thresholds,
                out,
                config=config,
                reference=reference,
                arm=SECONDARY_ARM,
                name="01b_before_after_distributions_desired_ratio",
            ),
        ),
        (
            "02",
            lambda: multiobjective_tradeoffs(
                arms, objectives, thresholds, out, config=config, reference=reference
            ),
        ),
        (
            "03",
            lambda: training_trajectories(
                fit,
                summary,
                out,
                epochs=epochs,
                objectives=objectives,
                config=config,
                reference=reference,
            ),
        ),
        (
            "04",
            lambda: epoch_property_trajectories(
                epochs, prior, out, objectives=objectives, config=config, reference=reference
            ),
        ),
        (
            "05",
            lambda: lead_structures_gallery(
                arms,
                objectives,
                out,
                config=config,
                reference=reference,
                arm=FINAL_ARM,
                name="05_lead_structures_gallery",
            ),
        ),
        (
            "05b",
            lambda: lead_structures_gallery(
                arms,
                objectives,
                out,
                config=config,
                reference=reference,
                arm=SECONDARY_ARM,
                name="05b_lead_structures_gallery_desired_ratio",
                fallback=False,
            ),
        ),
        (
            "06",
            lambda: objective_conflict_correlations(
                arms, objectives, out, config=config, reference=reference
            ),
        ),
        (
            "07",
            lambda: scaffold_diversity(
                arms,
                out,
                config=config,
                reference=reference,
                arm=PRIMARY_ARM,
                name="07_bemis_murcko_scaffold_diversity",
            ),
        ),
        (
            "07b",
            lambda: scaffold_diversity(
                arms,
                out,
                config=config,
                reference=reference,
                arm=SECONDARY_ARM,
                name="07b_bemis_murcko_scaffold_diversity_desired_ratio",
            ),
        ),
        (
            "08",
            lambda: lead_to_seed_superposition(
                arms, objectives, reference, out, generator=generator, config=config
            ),
        ),
        (
            "09",
            lambda: training_diagnostics(
                fit,
                epochs,
                prior,
                objectives,
                out,
                config=config,
                reference=reference,
                arms=arms,
                summary=summary,
            ),
        ),
    ]
    written: list[Path] = []
    for number, job in jobs:
        # Timed per figure: this suite is re-rendered over a whole campaign, so a panel
        # that costs a minute at n_final=5000 has to be visible rather than guessed at.
        started = time.monotonic()
        try:
            result = job()
        except Exception:
            logger.exception("figure %s failed", number)
            continue
        finally:
            logger.debug("figure %s took %.1fs", number, time.monotonic() - started)
        if isinstance(result, list):
            written.extend(result)
        elif result is not None:
            written.append(result)
    return written
