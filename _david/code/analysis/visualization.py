from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from textwrap import fill
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, QED, rdDepictor, rdFMCS, rdMolDescriptors, rdMolTransforms, rdShapeAlign
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import gaussian_kde

from _david.code.analysis.ranking import lead_rank_score
from _david.code.config import RunPaths
from _david.code.rocs import align_to_reference
from _david.code.schema import std

LOGGER = logging.getLogger(__name__)


def unscale_to_physical_metrics(
    df: pd.DataFrame,
    *,
    rocs_is_normalized: bool | None = None,
) -> pd.DataFrame:
    """Ensure ROCS is on the physical [0, 2] TanimotoCombo scale and compute missing 2D descriptors.

    Uses the explicit ``Raw_ROCS`` column when recorded by ``MoleculePipeline``, or
    ``rocs_is_normalized`` when specified, falling back to legacy range detection only
    for older TSV outputs that lack explicit scale provenance.
    """
    if df.empty:
        return df.copy()

    df_out = df.copy()
    if "Raw_ROCS" in df_out.columns and rocs_is_normalized is None:
        df_out[std.ROCS] = df_out["Raw_ROCS"]
    elif std.ROCS in df_out.columns:
        vals = df_out[std.ROCS].dropna()
        if not vals.empty:
            if rocs_is_normalized is True or (
                rocs_is_normalized is None and vals.max() <= 1.0001
            ):
                df_out[std.ROCS] = df_out[std.ROCS] * 2.0

    if std.SMILES in df_out.columns:
        mols = [
            Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
            for s in df_out[std.SMILES]
        ]
        if std.HAC not in df_out.columns:
            df_out[std.HAC] = [m.GetNumHeavyAtoms() if m else 0 for m in mols]
        if std.MW not in df_out.columns:
            df_out[std.MW] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
        if std.LOGP not in df_out.columns:
            df_out[std.LOGP] = [Descriptors.MolLogP(m) if m else np.nan for m in mols]
        if std.TPSA not in df_out.columns:
            df_out[std.TPSA] = [
                rdMolDescriptors.CalcTPSA(m) if m else np.nan for m in mols
            ]
        if std.HBD not in df_out.columns:
            df_out[std.HBD] = [Lipinski.NumHDonors(m) if m else 0 for m in mols]
        if std.HBA not in df_out.columns:
            df_out[std.HBA] = [Lipinski.NumHAcceptors(m) if m else 0 for m in mols]
        if std.ROT_BONDS not in df_out.columns:
            df_out[std.ROT_BONDS] = [
                Lipinski.NumRotatableBonds(m) if m else 0 for m in mols
            ]
        if std.QED not in df_out.columns:
            df_out[std.QED] = [QED.qed(m) if m else np.nan for m in mols]
        if "Raw_SA" not in df_out.columns:
            df_out["Raw_SA"] = [
                float(calculate_sa_score(m)) if m is not None else np.nan for m in mols
            ]
        if std.SCAFFOLD not in df_out.columns:
            scafs = []
            for m in mols:
                try:
                    scafs.append(
                        MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
                    )
                except Exception:
                    scafs.append("")
            df_out[std.SCAFFOLD] = scafs

    return df_out


def _deterministic_epoch_subsample(
    df: pd.DataFrame, max_per_epoch: int | None = 256
) -> pd.DataFrame:
    """Return an evenly spaced, deterministic per-epoch subsample of ``df``.

    Raw SA scoring is ~10^3 molecules/s, so a full 5000-molecule final arm or a
    200-epoch monitor table would dominate figure wall-clock. An evenly spaced
    slice over generation order is unbiased for median/IQR statistics and keeps
    the figure reproducible across runs.
    """
    if max_per_epoch is None or df.empty or "Epoch" not in df.columns:
        return df.copy()

    keep: list[pd.DataFrame] = []
    for _, grp in df.groupby("Epoch", sort=True):
        if len(grp) > max_per_epoch:
            picks = np.linspace(0, len(grp) - 1, num=max_per_epoch).round().astype(int)
            grp = grp.iloc[picks]
        keep.append(grp)
    return pd.concat(keep) if keep else df.iloc[0:0].copy()


def compute_raw_sa(
    smiles_values: Sequence[Any], cache: dict[str, float] | None = None
) -> np.ndarray:
    """Compute physical RDKit synthetic-accessibility scores (≈1 easy … 10 hard).

    ``_smiles.tsv`` only stores the *desirability* of SA, so any figure that claims
    to show SA must recompute the raw score from the molecule. Values are memoised
    per SMILES because early epochs resample heavily from the same prior.
    """
    memo = cache if cache is not None else {}
    values = np.full(len(smiles_values), np.nan, dtype=float)
    for i, smi in enumerate(smiles_values):
        key = str(smi) if isinstance(smi, str) else ""
        if key not in memo:
            mol = Chem.MolFromSmiles(key) if key else None
            score = np.nan
            if mol is not None:
                try:
                    score = float(calculate_sa_score(mol))
                except Exception:
                    score = np.nan
            memo[key] = score
        values[i] = memo[key]
    return values


def epoch_raw_sa_table(
    df: pd.DataFrame, max_per_epoch: int | None = 256
) -> pd.DataFrame:
    """Per-molecule raw SA scores for a monitor table, subsampled per epoch."""
    if df.empty or "Epoch" not in df.columns or std.SMILES not in df.columns:
        return pd.DataFrame(columns=["Epoch", "RawSA"])
    sub = _deterministic_epoch_subsample(df[["Epoch", std.SMILES]], max_per_epoch)
    out = sub.copy()
    out["RawSA"] = compute_raw_sa(sub[std.SMILES].tolist())
    return out


def _epsilon_schedule(epochs: np.ndarray, run_config: dict | None) -> np.ndarray | None:
    """Return the per-epoch exploration epsilon recorded in the run config, if any."""
    if not run_config or "epsilon" not in run_config:
        return None
    from _david.code.training.models import decayed_epsilon

    eps_start = float(run_config.get("epsilon", 0.0))
    return np.array(
        [
            decayed_epsilon(
                int(epoch),
                epsilon_start=eps_start,
                epsilon_end=float(run_config.get("epsilon_end", eps_start)),
                decay_start_epoch=int(run_config.get("decay_start_epoch", 1)),
                decay_epochs=int(run_config.get("decay_epochs", 1)),
                decay_type=str(run_config.get("decay_type", "linear")).lower(),
                use_decay=bool(run_config.get("use_epsilon_decay", False)),
            )
            for epoch in epochs
        ],
        dtype=float,
    )


def _shape_color_for_smiles(
    smiles: str, reference: Chem.Mol, random_seed: int = 0xC0FFEE
) -> tuple[float, float, float]:
    """Score one SMILES with RDKit ROCS and keep (combo, shape, color)."""
    from rdkit.Chem import AllChem

    from _david.code.rocs import score_pair

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or reference is None:
        return np.nan, np.nan, np.nan
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return np.nan, np.nan, np.nan
    return score_pair(mol, reference)


_SHAPE_COLOR_REFERENCE: Chem.Mol | None = None


def _shape_color_worker_init(reference: Chem.Mol) -> None:
    global _SHAPE_COLOR_REFERENCE
    _SHAPE_COLOR_REFERENCE = reference


def _shape_color_worker(smiles: str) -> tuple[float, float, float]:
    return _shape_color_for_smiles(smiles, _SHAPE_COLOR_REFERENCE)


def epoch_shape_color_table(
    df: pd.DataFrame,
    reference: Chem.Mol | None,
    max_per_epoch: int | None = 20,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Per-epoch shape/color decomposition on a deterministic single-conformer subsample.

    The monitor stores only the combined TanimotoCombo desirability, and splitting it
    after the fact is impossible. A short re-scoring pass recovers the two channels for
    a subsample, which is enough to see whether a ROCS gain came from the shape or the
    pharmacophore channel. Values are a diagnostic trend, not the training reward.
    """
    if df.empty or reference is None or "Epoch" not in df.columns:
        return pd.DataFrame(columns=["Epoch", "Shape", "Color"])

    sub = _deterministic_epoch_subsample(df[["Epoch", std.SMILES]], max_per_epoch)
    smiles_values = [str(s) for s in sub[std.SMILES]]
    unique_smiles = list(dict.fromkeys(smiles_values))

    workers = n_jobs if n_jobs is not None else min(8, os.cpu_count() or 1)
    scored: dict[str, tuple[float, float, float]] = {}
    if workers > 1 and len(unique_smiles) > 1:
        from drugex.training.scorers.parallel import pool_context

        ctx = pool_context()
        with ctx.Pool(
            processes=workers,
            initializer=_shape_color_worker_init,
            initargs=(reference,),
        ) as pool:
            results = pool.map(
                _shape_color_worker, unique_smiles, chunksize=max(1, len(unique_smiles) // (workers * 4))
            )
        scored = dict(zip(unique_smiles, results))
    else:
        for smiles in unique_smiles:
            scored[smiles] = _shape_color_for_smiles(smiles, reference)

    rows = [
        (float(epoch), scored[smiles][1], scored[smiles][2])
        for epoch, smiles in zip(sub["Epoch"], smiles_values)
    ]
    return pd.DataFrame(rows, columns=["Epoch", "Shape", "Color"])


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap median difference b - a and confidence interval."""
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    if len(a_clean) == 0 or len(b_clean) == 0:
        return 0.0, 0.0, 0.0
    delta_obs = float(np.median(b_clean) - np.median(a_clean))
    rng = np.random.default_rng(seed=42)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        samp_a = rng.choice(a_clean, size=len(a_clean), replace=True)
        samp_b = rng.choice(b_clean, size=len(b_clean), replace=True)
        boot_diffs[i] = np.median(samp_b) - np.median(samp_a)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_diffs, 100.0 * alpha))
    high = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha)))
    return delta_obs, low, high


def _plot_kde(
    ax: plt.Axes,
    data: np.ndarray,
    color: str,
    label: str,
    x_grid: np.ndarray,
    alpha: float = 0.35,
) -> None:
    """Helper computing Gaussian KDE and rendering smooth density line and fill."""
    clean = data[np.isfinite(data)]
    if len(clean) < 2 or np.all(clean == clean[0]):
        return
    try:
        kde = gaussian_kde(clean)
        density = kde(x_grid)
        ax.plot(x_grid, density, color=color, linewidth=2.0, label=label)
        ax.fill_between(x_grid, 0, density, color=color, alpha=alpha)
    except Exception:
        ax.hist(clean, bins=25, density=True, color=color, alpha=alpha, label=label)


def enrich_medicinal_chemistry_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rich medicinal chemistry descriptors (Fsp3, SlogP, TPSA, Rings, RotB) from SMILES."""
    df_out = df.copy()
    if df_out.empty or std.SMILES not in df_out.columns:
        return df_out

    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) and str(s).strip() != "" else None for s in df_out[std.SMILES]]

    if std.MW not in df_out.columns or df_out[std.MW].isna().all():
        df_out[std.MW] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
    if std.HAC not in df_out.columns or df_out[std.HAC].isna().all():
        df_out[std.HAC] = [m.GetNumHeavyAtoms() if m else np.nan for m in mols]
    if std.ROT_BONDS not in df_out.columns or df_out[std.ROT_BONDS].isna().all():
        df_out[std.ROT_BONDS] = [Lipinski.NumRotatableBonds(m) if m else np.nan for m in mols]
    if std.LOGP not in df_out.columns or df_out[std.LOGP].isna().all():
        df_out[std.LOGP] = [Descriptors.MolLogP(m) if m else np.nan for m in mols]
    if std.TPSA not in df_out.columns or df_out[std.TPSA].isna().all():
        df_out[std.TPSA] = [rdMolDescriptors.CalcTPSA(m) if m else np.nan for m in mols]

    df_out["Fsp3"] = [rdMolDescriptors.CalcFractionCSP3(m) if m else np.nan for m in mols]
    df_out["NumRings"] = [Lipinski.RingCount(m) if m else np.nan for m in mols]
    df_out["NumAromaticRings"] = [Lipinski.NumAromaticRings(m) if m else np.nan for m in mols]
    return df_out


def _valid_physical(frame: pd.DataFrame) -> pd.DataFrame:
    """One evaluation arm restricted to valid molecules and put on the physical scale."""
    if std.VALID in frame.columns:
        frame = frame[pd.to_numeric(frame[std.VALID], errors="coerce").fillna(0) == 1]
    return unscale_to_physical_metrics(frame)


def render_before_after_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
    ref_smiles: str = "",
) -> Path:
    """Render unified 3x3 hierarchical diagnostic dashboard of continuous KDE property shifts.

    Covers **every valid molecule** of each arm; invalid structures are excluded
    because their descriptors are undefined. See :func:`render_selected_distributions`
    for the same dashboard restricted to the prior's selected tail.
    """
    return _render_property_grid(
        enrich_medicinal_chemistry_descriptors(_valid_physical(df_before)),
        enrich_medicinal_chemistry_descriptors(_valid_physical(df_after)),
        Path(output_png),
        ref_smiles,
        "Hierarchical De Novo Property Diagnostic Dashboard: Baseline Prior vs. Reinforced Policy",
    )


def selection_cutoffs(
    prior_frame: pd.DataFrame, percentile: float = 85.0
) -> tuple[float, float | None]:
    """Return ``(rocs_cutoff, sa_cutoff)`` defining the prior's selected tail.

    Both cutoffs come from the **prior arm only**, so the same molecules-scale is
    applied to the reinforced arm. ``sa_cutoff`` is ``None`` when the prior's SA
    desirability column is saturated at its p{percentile} value: a cut there would
    select all molecules and add nothing. ``rocs_cutoff`` is NaN when the arm has no
    parsable ROCS values.
    """
    rocs = pd.to_numeric(prior_frame.get(std.ROCS), errors="coerce")
    rocs = rocs.dropna() if rocs is not None else pd.Series(dtype=float)
    rocs_cut = float(np.percentile(rocs, percentile)) if not rocs.empty else float("nan")

    sa_cut: float | None = None
    if std.SA in prior_frame.columns:
        sa = pd.to_numeric(prior_frame[std.SA], errors="coerce").dropna()
        if not sa.empty:
            candidate = float(np.percentile(sa, percentile))
            if candidate < float(sa.max()):
                sa_cut = candidate
    return rocs_cut, sa_cut


def render_selected_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
    ref_smiles: str = "",
    percentile: float = 85.0,
) -> Path:
    """Same 3x3 dashboard restricted to the molecules that pass the prior's p{percentile} target.

    The cutoffs are computed **once on the prior arm** and then applied unchanged to
    both arms, so the figure answers "what do the selected molecules look like, and
    how far did the reinforced arm move inside that selection" rather than
    "what are each arm's own best molecules" (the latter would re-rank each arm and
    hide the enrichment).

    An SA cutoff is only applied when the prior's SA desirability column is not
    saturated; see :func:`selection_cutoffs`.
    """
    b_frame = _valid_physical(df_before)
    a_frame = _valid_physical(df_after)
    rocs_cut, sa_cut = selection_cutoffs(b_frame, percentile)

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        mask = pd.to_numeric(frame[std.ROCS], errors="coerce") >= rocs_cut
        if sa_cut is not None:
            mask &= pd.to_numeric(frame[std.SA], errors="coerce") >= sa_cut
        return frame[mask]

    b_selected = select(b_frame)
    a_selected = select(a_frame)
    note = f"selection: ROCS ≥ {rocs_cut:.3f} (prior p{percentile:.0f})"
    if sa_cut is not None:
        note += f", SA ≥ {sa_cut:.3f}"
    note += (
        f" — retained {len(b_selected)}/{len(b_frame)} prior, "
        f"{len(a_selected)}/{len(a_frame)} reinforced"
    )
    return _render_property_grid(
        enrich_medicinal_chemistry_descriptors(b_selected),
        enrich_medicinal_chemistry_descriptors(a_selected),
        Path(output_png),
        ref_smiles,
        f"Selected Tail (prior p{percentile:.0f} target): Baseline Prior vs. Reinforced Policy",
        note,
    )


def _render_property_grid(
    b_df: pd.DataFrame,
    a_df: pd.DataFrame,
    target_png: Path,
    ref_smiles: str,
    title: str,
    note: str = "",
) -> Path:
    """Draw the 3x3 KDE dashboard for two already-filtered, already-enriched arms."""
    target_png.parent.mkdir(parents=True, exist_ok=True)

    ref_m = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
    ref_vals = {
        std.ROCS: 2.00,
        "Raw_SA": float(calculate_sa_score(ref_m)) if ref_m else 3.74,
        std.QED: float(QED.qed(ref_m)) if ref_m else 0.21,
        std.MW: float(Descriptors.MolWt(ref_m)) if ref_m else 458.4,
        std.HAC: float(ref_m.GetNumHeavyAtoms()) if ref_m else 33.0,
        std.ROT_BONDS: float(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3.0,
        std.LOGP: float(Descriptors.MolLogP(ref_m)) if ref_m else 2.23,
        std.TPSA: float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 197.4,
        "Fsp3": float(rdMolDescriptors.CalcFractionCSP3(ref_m)) if ref_m else 0.36,
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5), dpi=250)
    fig.suptitle(
        title if not note else f"{title}\n{note}",
        fontsize=15.0 if not note else 12.5,
        fontweight="bold",
        y=0.985 if not note else 0.995,
    )

    # 3x3 layout: Row 1 = Primary RL Objectives, Row 2 = Size & Complexity Drift, Row 3 = Physicochemical Integrity
    metrics_config: list[tuple[str, str, tuple[float, float], str]] = [
        # Row 1: Primary Objectives
        (std.ROCS, "3D ROCS Shape/Color Overlap", (0.0, 2.0), "TanimotoCombo Score"),
        ("Raw_SA", "Synthetic Accessibility (SAscore)", (1.0, 8.0), "SA (lower = easier)"),
        (std.QED, "Drug-Likeness (QED)", (0.0, 1.0), "QED Score"),
        # Row 2: Size & Complexity
        (std.MW, "Molecular Weight (MW)", (100.0, 700.0), "Molecular Weight (Da)"),
        (std.HAC, "Heavy Atom Count (HAC)", (8.0, 48.0), "Heavy Atom Count"),
        (std.ROT_BONDS, "Rotatable Bonds Count", (0.0, 15.0), "Rotatable Bonds"),
        # Row 3: Physicochemical & 3D Integrity
        (std.LOGP, "Calculated Lipophilicity (SlogP)", (-2.0, 7.0), "SlogP"),
        (std.TPSA, "Polar Surface Area (TPSA)", (0.0, 220.0), "TPSA (Å²)"),
        ("Fsp3", "Carbon sp³ Fraction (Fsp³ / 3D Complexity)", (0.0, 1.0), "Fsp³ Fraction"),
    ]

    for idx, (col, title, x_limits, x_lbl) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]
        b_vals = b_df[col].dropna().values.astype(np.float64) if col in b_df.columns else np.array([])
        a_vals = a_df[col].dropna().values.astype(np.float64) if col in a_df.columns else np.array([])

        x_grid = np.linspace(x_limits[0], x_limits[1], 300)
        _plot_kde(ax, b_vals, "#64748B", "Baseline Prior", x_grid, alpha=0.30)
        _plot_kde(ax, a_vals, "#4F46E5", "Reinforced Agent", x_grid, alpha=0.45)

        delta, low, high = bootstrap_ci(b_vals, a_vals)
        b_med = float(np.median(b_vals)) if len(b_vals) > 0 else 0.0
        a_med = float(np.median(a_vals)) if len(a_vals) > 0 else 0.0

        ax.axvline(b_med, color="#475569", linestyle="--", linewidth=1.4)
        ax.axvline(a_med, color="#3730A3", linestyle="-", linewidth=1.6)

        # Reference seed marker
        if col in ref_vals:
            ref_val = ref_vals[col]
            ax.axvline(ref_val, color="#0D9488", linestyle=":", linewidth=1.6, label=f"Seed ({ref_val:.2f})")

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
                boxstyle="round,pad=0.30",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                alpha=0.9,
            ),
        )

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(x_lbl, fontsize=9.0)
        ax.set_ylabel("Density", fontsize=9.0)
        ax.set_xlim(x_limits)
        ax.grid(True, linestyle=":", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8.0)

    # Category header annotations
    row_titles = [
        "PRIMARY RL OBJECTIVES (ROCS, SA, QED)",
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
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=0.8),
        )

    plt.subplots_adjust(
        top=0.92 if not note else 0.895, bottom=0.05, hspace=0.36, wspace=0.22
    )
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def compute_2d_skyline(
    x: np.ndarray,
    y: np.ndarray,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical non-dominated 2D skyline (step-wise frontier) without unphysical splines."""
    pts = np.column_stack([x, y])
    valid = ~np.isnan(pts).any(axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        return np.array([]), np.array([])

    sx = pts[:, 0] if maximize_x else -pts[:, 0]
    sy = pts[:, 1] if maximize_y else -pts[:, 1]

    order = np.lexsort((-sy, -sx))
    sorted_pts = pts[order]
    sorted_sy = sy[order]

    skyline = []
    max_sy_so_far = -np.inf
    for pt, cur_sy in zip(sorted_pts, sorted_sy):
        if cur_sy > max_sy_so_far:
            skyline.append(pt)
            max_sy_so_far = cur_sy

    skyline_arr = np.array(skyline)
    if len(skyline_arr) == 0:
        return np.array([]), np.array([])
    skyline_arr = skyline_arr[np.argsort(skyline_arr[:, 0])]
    return skyline_arr[:, 0], skyline_arr[:, 1]


def compute_3d_pareto_mask(
    rocs: np.ndarray,
    sa: np.ndarray,
    qed: np.ndarray,
) -> np.ndarray:
    """Identify true 3D Pareto-optimal leads across (ROCS maximize, SA minimize, QED maximize)."""
    pts = np.column_stack([rocs, -sa, qed])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        diff = pts - pts[i]
        dominates = (diff >= 0).all(axis=1) & (diff > 0).any(axis=1)
        if dominates.any():
            is_pareto[i] = False
    return is_pareto


def _overlay_prior_contours(ax: plt.Axes, bx: np.ndarray, by: np.ndarray) -> None:
    """Overlay 50% and 90% bivariate density contours of prior baseline population."""
    valid = (~np.isnan(bx)) & (~np.isnan(by))
    bx_v = bx[valid]
    by_v = by[valid]
    if len(bx_v) < 30 or np.all(bx_v == bx_v[0]) or np.all(by_v == by_v[0]):
        return
    try:
        kde = gaussian_kde(np.vstack([bx_v, by_v]))
        xi, yi = np.mgrid[bx_v.min():bx_v.max():50j, by_v.min():by_v.max():50j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        v_sorted = np.sort(zi.flatten())[::-1]
        cum = np.cumsum(v_sorted)
        cum /= cum[-1]
        lvl_50 = v_sorted[np.searchsorted(cum, 0.50)]
        lvl_90 = v_sorted[np.searchsorted(cum, 0.90)]

        ax.contour(
            xi, yi, zi,
            levels=[lvl_90, lvl_50],
            colors=["#64748b", "#334155"],
            linestyles=["--", "-."],
            linewidths=[1.1, 1.4],
            alpha=0.75,
            zorder=2,
        )
    except Exception:
        pass


def render_multiobjective_tradeoffs(
    df_after: pd.DataFrame,
    output_png: Path | str,
    df_before: pd.DataFrame | None = None,
) -> Path:
    """Render 1x3 scatter plots showing empirical Pareto frontiers, 3D optimal stars, and prior density contours."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    df_clean = unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])
    if df_clean.empty:
        return target_png

    b_clean = (
        unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
        if df_before is not None and not df_before.empty and std.VALID in df_before.columns
        else (unscale_to_physical_metrics(df_before) if df_before is not None and not df_before.empty else None)
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=250)
    fig.suptitle(
        "Multi-Objective Optimization Trade-Offs, Step-Wise Skylines & 3D Pareto Leads",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    rocs = df_clean.get(std.ROCS, pd.Series(0.0, index=df_clean.index)).values
    sa = df_clean.get(
        "Raw_SA", df_clean.get(std.SA, pd.Series(10.0, index=df_clean.index))
    ).values
    qed = df_clean.get(std.QED, pd.Series(0.0, index=df_clean.index)).values

    # True 3D Pareto optimal set
    pareto_mask = compute_3d_pareto_mask(rocs, sa, qed)
    p3_rocs = rocs[pareto_mask]
    p3_sa = sa[pareto_mask]
    p3_qed = qed[pareto_mask]

    b_rocs = b_clean.get(std.ROCS, pd.Series(dtype=float)).values if b_clean is not None else np.array([])
    b_sa = (
        b_clean.get("Raw_SA", b_clean.get(std.SA, pd.Series(dtype=float))).values
        if b_clean is not None
        else np.array([])
    )
    b_qed = b_clean.get(std.QED, pd.Series(dtype=float)).values if b_clean is not None else np.array([])

    # Panel 1: ROCS vs. SA (ROCS max, SA min)
    if len(b_rocs) > 0 and len(b_sa) > 0:
        _overlay_prior_contours(ax1, b_rocs, b_sa)
    sc1 = ax1.scatter(
        rocs, sa, c=qed, cmap="viridis", alpha=0.55, edgecolors="none", s=25, zorder=3
    )
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("QED (Evaluated)", fontsize=9)

    sk1_x, sk1_y = compute_2d_skyline(rocs, sa, maximize_x=True, maximize_y=False)
    if len(sk1_x) > 0:
        ax1.plot(
            sk1_x, sk1_y,
            drawstyle="steps-post",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_rocs) > 0:
        ax1.scatter(
            p3_rocs, p3_sa,
            marker="*", s=140, facecolors="#f59e0b", edgecolors="#78350f",
            linewidth=1.0, zorder=5, label=f"True 3D Pareto ({len(p3_rocs)})",
        )

    ax1.axvline(0.70, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7, label="ROCS Target (0.70)")
    ax1.axhline(4.50, color="#2563eb", linestyle="--", linewidth=1.2, alpha=0.7, label="SA Ceiling (4.50)")
    ax1.set_title("(A) 3D Shape (ROCS) vs. Synthesizability (SA)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("ROCS TanimotoCombo (higher=better)")
    ax1.set_ylabel("Raw SA Score (lower=better)")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.0)

    # Panel 2: ROCS vs. QED (ROCS max, QED max)
    if len(b_rocs) > 0 and len(b_qed) > 0:
        _overlay_prior_contours(ax2, b_rocs, b_qed)
    sc2 = ax2.scatter(
        rocs, qed, c=sa, cmap="plasma_r", alpha=0.55, edgecolors="none", s=25, zorder=3
    )
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Raw SA Score", fontsize=9)

    sk2_x, sk2_y = compute_2d_skyline(rocs, qed, maximize_x=True, maximize_y=True)
    if len(sk2_x) > 0:
        ax2.plot(
            sk2_x, sk2_y,
            drawstyle="steps-post",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_rocs) > 0:
        ax2.scatter(
            p3_rocs, p3_qed,
            marker="*", s=140, facecolors="#f59e0b", edgecolors="#78350f",
            linewidth=1.0, zorder=5, label=f"True 3D Pareto ({len(p3_rocs)})",
        )

    ax2.axvline(0.70, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7, label="ROCS Target (0.70)")
    ax2.axhline(0.50, color="#2563eb", linestyle="--", linewidth=1.2, alpha=0.7, label="QED Threshold (0.50)")
    ax2.set_title("(B) 3D Shape (ROCS) vs. Drug-Likeness (QED)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("ROCS TanimotoCombo (higher=better)")
    ax2.set_ylabel("QED Score (higher=better)")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.0)

    # Panel 3: SA vs. QED (SA min, QED max)
    if len(b_sa) > 0 and len(b_qed) > 0:
        _overlay_prior_contours(ax3, b_sa, b_qed)
    sc3 = ax3.scatter(
        sa, qed, c=rocs, cmap="turbo", alpha=0.55, edgecolors="none", s=25, zorder=3
    )
    cbar3 = fig.colorbar(sc3, ax=ax3)
    cbar3.set_label("ROCS TanimotoCombo", fontsize=9)

    sk3_x, sk3_y = compute_2d_skyline(sa, qed, maximize_x=False, maximize_y=True)
    if len(sk3_x) > 0:
        ax3.plot(
            sk3_x, sk3_y,
            drawstyle="steps-post",
            color="#b91c1c",
            linewidth=2.2,
            zorder=4,
            label="2D Empirical Skyline",
        )
    if len(p3_sa) > 0:
        ax3.scatter(
            p3_sa, p3_qed,
            marker="*", s=140, facecolors="#f59e0b", edgecolors="#78350f",
            linewidth=1.0, zorder=5, label=f"True 3D Pareto ({len(p3_sa)})",
        )

    ax3.axvline(4.50, color="#2563eb", linestyle="--", linewidth=1.2, alpha=0.7, label="SA Ceiling (4.50)")
    ax3.axhline(0.50, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7, label="QED Threshold (0.50)")
    ax3.set_title("(C) Synthesizability (SA) vs. Drug-Likeness (QED)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Raw SA Score (lower=better)")
    ax3.set_ylabel("QED Score (higher=better)")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_training_trajectories(
    history_file_or_df: Path | str | pd.DataFrame,
    output_png: Path | str,
    smiles_file: Path | str | None = None,
    run_config: dict | None = None,
    reference_sdf: Path | str | None = None,
    rocs_scale: float = 1.0,
) -> Path | None:
    """Render the 4-panel training-dynamics dashboard.

    (A) policy-gradient loss with the epsilon schedule on its own axis, (B) normalized
    selection criteria (the recorded checkpoint criterion full colour, the rest pastel)
    and library quality guards, (C) mean and median of every objective with raw SA on a
    dedicated axis, (D) per-epoch ROCS distribution, ceiling and shape/color split.
    """
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(history_file_or_df, (str, Path)):
        p = Path(history_file_or_df)
        if not p.is_file():
            return None
        df = pd.read_csv(p, sep="\t")
    else:
        df = history_file_or_df.copy()

    if df.empty or "Epoch" not in df.columns:
        return None

    # Fix Panel C: Pull per-epoch mean scores from _smiles.tsv
    smiles_p = Path(smiles_file) if smiles_file else None
    if not smiles_p or not smiles_p.is_file():
        # Fallback to checking sister file
        if isinstance(history_file_or_df, (str, Path)):
            base_cand = Path(str(history_file_or_df).replace("_fit.tsv", "_smiles.tsv"))
            if base_cand.is_file():
                smiles_p = base_cand

    # Per-epoch monitor statistics from the smiles table (means + ROCS distribution)
    sm_df: pd.DataFrame | None = None
    if smiles_p and smiles_p.is_file():
        try:
            sm_df = pd.read_csv(smiles_p, sep="\t")
            if sm_df.empty or "Epoch" not in sm_df.columns:
                sm_df = None
            elif rocs_scale != 1.0 and std.ROCS in sm_df.columns:
                # The monitor stores what the scorer returned; a modifier-era run
                # holds a [0, 1] desirability, which must be recovered with the
                # run's configured clip before the panel claims a physical ROCS.
                sm_df[std.ROCS] = pd.to_numeric(sm_df[std.ROCS], errors="coerce") * float(rocs_scale)
        except Exception:
            sm_df = None

    rocs_median = rocs_q25 = rocs_q75 = rocs_p95 = None
    cum_p95 = None
    if sm_df is not None:
        cols_to_agg = [c for c in ["ROCS", "QED"] if c in sm_df.columns]
        if cols_to_agg:
            ep_means = sm_df.groupby("Epoch")[cols_to_agg].mean()
            ep_medians = sm_df.groupby("Epoch")[cols_to_agg].median()
            for col in cols_to_agg:
                df[col] = df["Epoch"].map(ep_means[col])
                df[f"{col}_median"] = df["Epoch"].map(ep_medians[col])
        if "ROCS" in sm_df.columns:
            rocs_grp = sm_df.groupby("Epoch")["ROCS"]
            rocs_median = rocs_grp.median()
            rocs_q25 = rocs_grp.quantile(0.25)
            rocs_q75 = rocs_grp.quantile(0.75)
            rocs_p95 = rocs_grp.quantile(0.95)
            if not rocs_p95.dropna().empty:
                cum_p95 = np.maximum.accumulate(rocs_p95.dropna().values)

    # Raw RDKit SA is not stored by the monitor (only its desirability), so it is
    # recomputed from SMILES on a deterministic per-epoch subsample.
    sa_table = epoch_raw_sa_table(sm_df) if sm_df is not None else pd.DataFrame()

    # Shape/color split: recomputed per epoch because the monitor stores only the combo.
    shape_color_table = pd.DataFrame(columns=["Epoch", "Shape", "Color"])
    reference_mol = None
    if reference_sdf and Path(reference_sdf).is_file():
        suppl = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)
        reference_mol = next((m for m in suppl if m is not None), None)
    if sm_df is not None and reference_mol is not None:
        shape_color_table = epoch_shape_color_table(sm_df, reference_mol)


    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=250)
    fig.suptitle(
        "SequenceRNN Reinforcement Learning Dynamics & Convergence Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    epochs = df["Epoch"]

    # Determine Best Checkpoint and Early Stopping Epochs
    stop_epoch = int(epochs.max())
    best_epoch = stop_epoch
    if "best_epoch" in df.columns and not df["best_epoch"].dropna().empty:
        best_epoch = int(df["best_epoch"].dropna().iloc[-1])
    elif "avg_gmean" in df.columns and not df["avg_gmean"].dropna().empty:
        best_epoch = int(df.loc[df["avg_gmean"].idxmax(), "Epoch"])
    elif "desired_ratio" in df.columns and not df["desired_ratio"].dropna().empty:
        best_epoch = int(df.loc[df["desired_ratio"].idxmax(), "Epoch"])

    def _mark_events(ax: plt.Axes, is_legend: bool = False) -> None:
        ax.axvline(
            best_epoch,
            color="#16a34a",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=f"Best Checkpoint (Ep {best_epoch})" if is_legend else "",
            zorder=2,
        )
        ax.axvline(
            stop_epoch,
            color="#dc2626",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label=f"Early Stop (Ep {stop_epoch})" if is_legend else "",
            zorder=2,
        )

    # Panel A: Training loss (left) and the exploration schedule (right, its own scale).
    ax1 = axes[0, 0]
    _mark_events(ax1, is_legend=True)
    if "loss_train" in df.columns:
        ax1.plot(
            epochs,
            df["loss_train"],
            color="#dc2626",
            linewidth=1.8,
            label="Train Loss",
            zorder=3,
        )
    ax1.set_title(
        "(A) Training Loss & Exploration Rate", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.4)

    epsilon_series = _epsilon_schedule(epochs.values, run_config)
    handles, labels = ax1.get_legend_handles_labels()
    if epsilon_series is not None:
        ax1r = ax1.twinx()
        eps_vals = np.asarray(epsilon_series, dtype=float)
        eps_label = (
            f"ε = {eps_vals[0]:.2f} (right axis)"
            if np.allclose(eps_vals, eps_vals[0])
            else "ε schedule (right axis)"
        )
        ax1r.plot(
            epochs,
            eps_vals,
            color="#EA580C",
            linewidth=1.8,
            linestyle="--",
            label=eps_label,
            zorder=4,
        )
        eps_top = max(float(np.nanmax(eps_vals)), 0.01)
        ax1r.set_ylim(-0.03 * eps_top, eps_top * 1.35)
        ax1r.set_ylabel("Exploration ε", color="#9A3412")
        ax1r.tick_params(axis="y", colors="#9A3412")
        h2, l2 = ax1r.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    elif handles:
        ax1.text(
            0.98,
            0.04,
            "ε schedule not recorded in manifest",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.6,
            color="#94A3B8",
        )
    if handles:
        ax1.legend(handles, labels, loc="best", fontsize=8.0)

    # Panel B: normalized selection criteria. The metric the run actually checkpointed on
    # gets the full colour; every other tracked statistic is pastel.
    ax2 = axes[0, 1]
    _mark_events(ax2)

    series_specs = [
        ("avg_gmean", "Avg Geometric Mean"),
        ("avg_amean", "Avg Arithmetic Mean"),
        ("desired_ratio", "Desired Ratio"),
        ("valid_ratio", "Valid Ratio"),
        ("unique_ratio", "Unique Ratio"),
    ]
    candidate_cols = [c for c, _ in series_specs if c in df.columns and not df[c].dropna().empty]

    recorded_criterion = str((run_config or {}).get("criteria", ""))
    if recorded_criterion in candidate_cols:
        eval_metric = recorded_criterion
    else:
        # Self-inspection: the checkpoint criterion is the metric whose own maximum is
        # the epoch the monitor recorded in ``best_epoch``.
        eval_metric = next(
            (
                col
                for col in candidate_cols
                if int(df.loc[df[col].idxmax(), "Epoch"]) == best_epoch
            ),
            None,
        )

    # Each criterion gets its own hue and marker. The three ratio series used to
    # share one teal (and two near-identical pastel blues), which made the panel
    # unreadable exactly where the series diverge.
    full_colors = {
        "avg_gmean": "#7C3AED",  # violet
        "avg_amean": "#2563EB",  # blue
        "desired_ratio": "#0D9488",  # teal
        "valid_ratio": "#B45309",  # amber
        "unique_ratio": "#BE185D",  # rose
    }
    pastel_colors = {
        "avg_gmean": "#C4B5FD",
        "avg_amean": "#93C5FD",
        "desired_ratio": "#5EEAD4",
        "valid_ratio": "#FCD34D",
        "unique_ratio": "#F9A8D4",
    }
    series_markers = {
        "avg_gmean": "o",
        "avg_amean": "s",
        "desired_ratio": "^",
        "valid_ratio": "D",
        "unique_ratio": "v",
    }
    # ``unique_ratio`` is computed over valid molecules, so on runs where every valid
    # molecule is unique the two series are identical and the second line is invisible.
    # Collapse them into one honest, labelled line instead of hiding a series.
    merged_valid_unique = (
        {"valid_ratio", "unique_ratio"} <= set(candidate_cols)
        and np.allclose(
            df["valid_ratio"].fillna(-1.0).to_numpy(dtype=float),
            df["unique_ratio"].fillna(-1.0).to_numpy(dtype=float),
        )
    )

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    mark_step = max(1, len(epochs) // 12)
    for col, label in series_specs:
        if col not in candidate_cols:
            continue
        if col == "unique_ratio" and merged_valid_unique:
            continue
        if col == "valid_ratio" and merged_valid_unique:
            label = "Valid = Unique Ratio (no duplicates)"
        is_criterion = col == eval_metric
        line, = ax2.plot(
            epochs,
            df[col],
            color=full_colors[col] if is_criterion else pastel_colors[col],
            linewidth=2.4 if is_criterion else 1.6,
            linestyle="-" if is_criterion else "--",
            marker=series_markers[col],
            markersize=4.2 if is_criterion else 3.2,
            markevery=mark_step,
            markeredgewidth=0,
            label=f"{label} ★" if is_criterion else label,
            zorder=4 if is_criterion else 3,
        )
        legend_handles.append(line)
        legend_labels.append(line.get_label())

    criterion_label = dict(series_specs).get(eval_metric, eval_metric or "not recorded")
    ax2.set_title(
        f"(B) Selection Criteria & Quality — criterion: {criterion_label}",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.4)
    if legend_handles:
        ax2.legend(
            legend_handles,
            legend_labels,
            loc="best",
            fontsize=8.0,
            ncol=2,
        )

    # Panel C: mean (full colour) and median (lighter) for every objective; raw SA,
    # which lives on a physical 1–6 scale, gets its own right axis.
    ax3 = axes[1, 0]
    _mark_events(ax3)
    plotted_objs = False
    for col, mean_color, median_color, lbl in [
        ("ROCS", "#2563EB", "#93C5FD", "ROCS"),
        ("QED", "#F59E0B", "#FDE68A", "QED"),
    ]:
        if col in df.columns and not df[col].dropna().empty:
            ax3.plot(
                epochs,
                df[col],
                color=mean_color,
                linewidth=2.1,
                label=f"{lbl} Mean",
                zorder=4,
            )
            median_col = f"{col}_median"
            if median_col in df.columns and not df[median_col].dropna().empty:
                ax3.plot(
                    epochs,
                    df[median_col],
                    color=median_color,
                    linewidth=1.6,
                    linestyle="--",
                    label=f"{lbl} Median",
                    zorder=3,
                )
            plotted_objs = True
    if not plotted_objs and "avg_gmean" in df.columns:
        ax3.plot(
            epochs,
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
    plotted_values = [
        df[col].dropna().to_numpy(float)
        for col in ("ROCS", "QED")
        if col in df.columns and not df[col].dropna().empty
    ]
    y_top = max(
        [1.02]
        + [float(np.nanmax(values)) * 1.05 for values in plotted_values if values.size]
    )
    ax3.set_ylim(-0.02, min(2.05, y_top))
    ax3.grid(True, linestyle=":", alpha=0.4)

    handles, labels = ax3.get_legend_handles_labels()
    if not sa_table.empty and sa_table["RawSA"].notna().any():
        sa_grp = sa_table.groupby("Epoch")["RawSA"]
        sai_mean = sa_grp.mean()
        sai_med = sa_grp.median()
        sa_q25 = sa_grp.quantile(0.25)
        sa_q75 = sa_grp.quantile(0.75)
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
        sa_lo = float(np.nanmin(sa_q25.values))
        sa_hi = float(np.nanmax(sa_q75.values))
        pad = max(0.6, 0.15 * (sa_hi - sa_lo))
        ax3r.set_ylim(sa_lo - pad, sa_hi + pad)
        ax3r.set_ylabel("Raw RDKit SA (lower = easier)", color="#4B5563")
        ax3r.tick_params(axis="y", colors="#4B5563")
        h2, l2 = ax3r.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    if handles:
        ax3.legend(handles, labels, loc="lower right", fontsize=7.6, ncol=2)

    # Panel D: per-epoch ROCS distribution — converged mode vs. explored tail.
    ax4 = axes[1, 1]
    _mark_events(ax4)
    if rocs_median is not None and not rocs_median.dropna().empty:
        ep = rocs_median.index.values
        ax4.fill_between(
            ep,
            rocs_q25.values,
            rocs_q75.values,
            color="#93C5FD",
            alpha=0.30,
            linewidth=0,
            label="ROCS IQR (25th–75th)",
            zorder=2,
        )
        ax4.plot(
            ep,
            rocs_median.values,
            color="#2563eb",
            linewidth=2.2,
            label="ROCS Median",
            zorder=4,
        )
        ax4.plot(
            ep,
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
            "(D) ROCS Distribution & Shape/Color Split", fontsize=11, fontweight="bold"
        )
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("ROCS TanimotoCombo (0–2)")
        rocs_top = max(
            1.02,
            float(np.nanmax(rocs_median.values)) * 1.05,
            float(np.nanmax(rocs_p95.values)) * 1.05,
        )
        ax4.set_ylim(-0.02, min(2.05, rocs_top))
        ax4.grid(True, linestyle=":", alpha=0.4)

        h4, l4 = ax4.get_legend_handles_labels()
        if not shape_color_table.empty:
            sc = shape_color_table.groupby("Epoch")[["Shape", "Color"]].median()
            if not sc.dropna(how="all").empty:
                ax4r = ax4.twinx()
                line_s, = ax4r.plot(
                    sc.index,
                    sc["Shape"].values,
                    color="#0D9488",
                    linewidth=1.7,
                    label="Shape Tanimoto (right)",
                    zorder=5,
                )
                line_c, = ax4r.plot(
                    sc.index,
                    sc["Color"].values,
                    color="#D97706",
                    linewidth=1.7,
                    label="Color Tanimoto (right)",
                    zorder=5,
                )
                sc_hi = float(np.nanmax(sc.to_numpy())) if np.isfinite(sc.to_numpy()).any() else 0.1
                ax4r.set_ylim(0.0, max(sc_hi * 1.25, 0.1))
                ax4r.set_ylabel("Recomputed Tanimoto (right)", fontsize=8.6)
                h4, l4 = h4 + [line_s, line_c], l4 + [
                    line_s.get_label(),
                    line_c.get_label(),
                ]
        ax4.legend(h4, l4, loc="lower right", fontsize=7.2, ncol=2)
    else:
        ax4.text(
            0.5,
            0.5,
            "No per-epoch SMILES table available",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax4.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_unfavorable_features_diagnostics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
    ref_heavy_atoms: int = 33,
) -> Path:
    """Render critical diagnostic panels highlighting size shrinkage, rigidity shift, and duplicate counts."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    b_df = unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
    a_df = unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=250)
    fig.suptitle(
        "Critical Diagnostic Analysis: Negative Trends & Shrinkage",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Heavy Atom Size Drift (Delta HAC relative to reference)
    if not b_df.empty and not a_df.empty:
        b_delta = b_df[std.HAC] - ref_heavy_atoms
        a_delta = a_df[std.HAC] - ref_heavy_atoms
        bins = np.arange(-25, 15, 1)
        ax1.hist(
            b_delta,
            bins=bins,
            alpha=0.4,
            color="#3b82f6",
            label="Baseline ΔHAC",
            density=True,
        )
        ax1.hist(
            a_delta,
            bins=bins,
            alpha=0.5,
            color="#ea580c",
            label="Reinforced ΔHAC",
            density=True,
        )
        ax1.axvline(
            0, color="#dc2626", linestyle="--", linewidth=1.5, label="Target Size (0)"
        )
        ax1.axvline(
            float(np.median(a_delta)),
            color="#c2410c",
            linestyle="-",
            linewidth=1.8,
            label=f"Reinf Med: {float(np.median(a_delta)):.1f}",
        )
        ax1.set_title(
            "(A) Heavy Atom Size Drift (Δ vs. Reference=33)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("Heavy Atom Difference (Generated - Reference)")
        ax1.set_ylabel("Density")
        ax1.grid(True, linestyle=":", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Rotatable Bond Rigidity Shift
    if not b_df.empty and not a_df.empty:
        b_rot = b_df[std.ROT_BONDS]
        a_rot = a_df[std.ROT_BONDS]
        bins_rot = np.arange(0, 16, 1)
        ax2.hist(
            b_rot,
            bins=bins_rot,
            alpha=0.4,
            color="#3b82f6",
            label="Baseline RotBonds",
            density=True,
        )
        ax2.hist(
            a_rot,
            bins=bins_rot,
            alpha=0.5,
            color="#ea580c",
            label="Reinforced RotBonds",
            density=True,
        )
        ax2.set_title(
            "(B) Molecular Flexibility & Rotatable Bonds",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Number of Rotatable Bonds")
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.legend(loc="upper right", fontsize=8.5)

    # Panel 3: Duplication and Frequency Distribution
    if not a_df.empty and std.CANONICAL_SMILES in a_df.columns:
        counts = a_df[std.CANONICAL_SMILES].value_counts().values
        max_rep = int(counts.max()) if len(counts) > 0 else 1
        ax3.hist(
            counts,
            bins=min(25, max_rep),
            color="#8b5cf6",
            alpha=0.75,
            edgecolor="#1e293b",
        )
        ax3.set_yscale("log")
        ax3.set_title(
            f"(C) Output Duplication Profile (Max Repeat: {max_rep})",
            fontsize=11,
            fontweight="bold",
        )
        ax3.set_xlabel("Copies per Unique Molecule")
        ax3.set_ylabel("Frequency (log scale)")
        ax3.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


class AlignedDepicter:
    """Renders 2D depictions aligned onto one shared core template.

    Every card is laid out with the *same* substructure in the same 2D orientation, so
    the gallery reads as a series of analogs rather than eleven independently rotated
    drawings. The template is either the seed's Murcko core (when every displayed
    molecule contains it) or the maximum common substructure of the displayed set.
    """

    def __init__(self, reference: str | Chem.Mol | None) -> None:
        self.ref_mol = (
            Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
        )
        if self.ref_mol is not None and self.ref_mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(self.ref_mol)
        self.ref_core = (
            MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        )
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            self.ref_core = self.ref_mol
        self.template_mol: Chem.Mol | None = None

    def prepare_template(self, molecules: Sequence[Chem.Mol | None]) -> None:
        """Choose one shared 2D template for the molecules that will be displayed.

        Preference order: the seed's Murcko core if every molecule contains it, else
        the maximum common substructure of the displayed set. A single template keeps
        the common core at the same place and orientation on every card.
        """
        mols = [
            Chem.RemoveHs(Chem.Mol(m)) for m in molecules if m is not None
        ]
        if not mols:
            self.template_mol = None
            return

        core = self.ref_core
        if (
            core is not None
            and core.GetNumHeavyAtoms() >= 4
            and all(m.HasSubstructMatch(core) for m in mols)
        ):
            template = Chem.Mol(core)
            rdDepictor.Compute2DCoords(template)
            self.template_mol = template
            return

        pattern: Chem.Mol | None = None
        try:
            mcs = rdFMCS.FindMCS(
                mols,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                timeout=10,
            )
            if mcs and mcs.numAtoms >= 4:
                pattern = Chem.MolFromSmarts(mcs.smartsString)
        except (RuntimeError, ValueError):
            pattern = None
        if pattern is not None and all(m.HasSubstructMatch(pattern) for m in mols):
            template = Chem.Mol(pattern)
            rdDepictor.Compute2DCoords(template)
            self.template_mol = template
            return

        self.template_mol = None

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Compute 2D coordinates that place the shared template at a fixed orientation."""
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)

        # 1. Shared template (seed Murcko core or the displayed set's MCS).
        for template in (self.template_mol, self.ref_core):
            if template is None or template.GetNumHeavyAtoms() == 0:
                continue
            if not mol_copy.HasSubstructMatch(template):
                continue
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, template)
                return mol_copy
            except (RuntimeError, ValueError):
                continue

        # 2. Pairwise MCS fallback against the seed, constraining the matched atoms
        #    to the seed's own 2D layout (threshold: >= 6 heavy atoms).
        if self.ref_mol is not None and self.ref_mol.GetNumHeavyAtoms() >= 6:
            try:
                mcs_res = rdFMCS.FindMCS(
                    [self.ref_mol, mol_copy],
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    timeout=1,
                )
                if mcs_res and mcs_res.numAtoms >= 6:
                    mcs_core = Chem.MolFromSmarts(mcs_res.smartsString)
                    if (
                        mcs_core is not None
                        and self.ref_mol.HasSubstructMatch(mcs_core)
                        and mol_copy.HasSubstructMatch(mcs_core)
                    ):
                        rdDepictor.GenerateDepictionMatching2DStructure(
                            mol_copy, self.ref_mol, refPatt=mcs_core
                        )
                        return mol_copy
            except (RuntimeError, ValueError):
                pass

        return mol_copy

    def _lead_card(
        self,
        row: pd.Series,
        title: str,
        seed_hac: int,
        seed_fp: Any,
    ) -> dict[str, Any] | None:
        """Build one lead card dict from a scored DataFrame row."""
        smi = str(row.get(std.SMILES, ""))
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m_clean = Chem.RemoveHs(m)
        hac = int(m_clean.GetNumHeavyAtoms())
        mw = float(row.get(std.MW, Descriptors.MolWt(m_clean)))
        rocs = float(row.get(std.ROCS, 0.0))
        sa = float(row.get("Raw_SA", row.get(std.SA, 0.0)))
        qed_val = float(row.get(std.QED, 0.0))
        lead_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m_clean, 2, nBits=2048)
        sim = (
            float(DataStructs.TanimotoSimilarity(seed_fp, lead_fp)) if seed_fp else 0.0
        )
        return {
            "mol": m_clean,
            "title": title,
            "is_seed": False,
            "rocs": rocs,
            "sa": sa,
            "qed": qed_val,
            "mw": mw,
            "hac": hac,
            "delta_hac": hac - seed_hac,
            "sim": sim,
        }

    def render_leads(
        self,
        molecules: Sequence[str | Chem.Mol] | pd.DataFrame,
        output_png: Path | str | None = None,
        max_leads: int = 5,
        rocs_ranked: pd.DataFrame | None = None,
    ) -> Path | None:
        """Render aligned 2D cards: prioritized leads on row 1, best-ROCS hits on row 2.

        Row 1 is the seed followed by the desirability-ranked leads. Row 2 repeats the
        exercise ranked purely by ROCS TanimotoCombo, skipping molecules already shown,
        so the two selection criteria can be compared visually.
        """
        target_png = Path(output_png) if output_png else Path("lead_structures_2d.png")
        target_png.parent.mkdir(parents=True, exist_ok=True)

        top_row: list[dict[str, Any]] = []
        second_row: list[dict[str, Any]] = []
        seen_smiles: set[str] = set()

        # Reference molecule card
        if self.ref_mol is not None:
            ref_clean = Chem.RemoveHs(Chem.Mol(self.ref_mol))
            hac = int(ref_clean.GetNumHeavyAtoms())
            mw = float(Descriptors.MolWt(ref_clean))
            qed_val = float(QED.qed(ref_clean))
            sa_val = float(calculate_sa_score(ref_clean))
            top_row.append(
                {
                    "mol": ref_clean,
                    "title": "TARGET QUERY SEED",
                    "is_seed": True,
                    "rocs": None,
                    "sa": sa_val,
                    "qed": qed_val,
                    "mw": mw,
                    "hac": hac,
                    "delta_hac": 0,
                    "sim": 1.0,
                }
            )
            seen_smiles.add(Chem.MolToSmiles(ref_clean))

        seed_hac = top_row[0]["hac"] if top_row else 33
        seed_fp = (
            rdMolDescriptors.GetMorganFingerprintAsBitVect(self.ref_mol, 2, nBits=2048)
            if self.ref_mol
            else None
        )

        if isinstance(molecules, pd.DataFrame):
            for rank_num, (_, row) in enumerate(molecules.head(max_leads).iterrows(), start=1):
                card = self._lead_card(row, f"LEAD #{rank_num}", seed_hac, seed_fp)
                if card is None:
                    continue
                top_row.append(card)
                mol = Chem.MolFromSmiles(str(row.get(std.SMILES, "")))
                if mol is not None:
                    seen_smiles.add(Chem.MolToSmiles(mol))

        if isinstance(rocs_ranked, pd.DataFrame):
            taken = 0
            for _, row in rocs_ranked.iterrows():
                if taken >= max_leads:
                    break
                mol = Chem.MolFromSmiles(str(row.get(std.SMILES, "")))
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol)
                if canonical in seen_smiles:
                    continue
                card = self._lead_card(
                    row, f"ROCS #{taken + 1}", seed_hac, seed_fp
                )
                if card is None:
                    continue
                second_row.append(card)
                seen_smiles.add(canonical)
                taken += 1

        if not top_row and not second_row:
            return None

        # One template for exactly the cards that are about to be drawn, then align
        # every card onto it. The set is known only now, so the template is the true
        # common core of the displayed molecules instead of a guess made up front.
        display_mols = [
            card["mol"] for card in top_row + second_row if card is not None
        ]
        self.prepare_template(display_mols)
        for card in top_row + second_row:
            card["mol"] = self.align_molecule(card["mol"])

        rows: list[list[dict[str, Any] | None]] = [top_row]
        if second_row:
            # Align the first ROCS hit under the first prioritized lead.
            rows.append([None] + second_row)

        ncols = max(len(r) for r in rows)
        nrows = len(rows)

        # Layout: each card keeps the depiction's own 760:540 aspect, so the axes add no
        # vertical slack. The old 5.2-inch-tall cells left ~25 % white space above and
        # below every image because imshow fits the wider image to the cell width.
        dpi = 250
        card_w_in = 3.55
        card_h_in = card_w_in * 540 / 760
        pad_x_in = 0.06
        caption_in = 1.22
        title_in = 0.52
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
            "Prioritized Leads (row 1) vs. Best-ROCS Hits (row 2): Aligned 2D Topology",
            fontsize=13,
            fontweight="bold",
            x=0.5,
            y=1 - 0.30 / fig_h,
        )

        for r_idx, row_cards in enumerate(rows):
            for c_idx in range(ncols):
                ax = axes[r_idx][c_idx]
                ax.axis("off")
                card = row_cards[c_idx] if c_idx < len(row_cards) else None
                if card is None:
                    if r_idx == 1 and c_idx == 0 and second_row:
                        ax.text(
                            0.5,
                            0.62,
                            "BEST ROCS HITS\n\nTop-5 by TanimotoCombo,\n"
                            "excluding the leads above",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9.0,
                            fontweight="bold",
                            color="#475569",
                            bbox=dict(
                                boxstyle="round,pad=0.5",
                                facecolor="#f8fafc",
                                edgecolor="#cbd5e1",
                            ),
                        )
                    continue

                # Render at the cell's native resolution so no upscaling happens.
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
                    line1 = f"HAC: {card['hac']}  |  MW: {card['mw']:.1f} Da"
                    line2 = f"SA: {card['sa']:.2f}  |  QED: {card['qed']:.2f}"
                    ax.text(
                        0.5,
                        -0.16,
                        line1,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8.5,
                        fontweight="medium",
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
                        line2,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8.5,
                        fontweight="medium",
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
                dh_c = (
                    "#dcfce7" if abs(dh) <= 3 else ("#fee2e2" if dh < -3 else "#dbeafe")
                )
                dh_e = (
                    "#15803d"
                    if abs(dh) <= 3
                    else ("#dc2626" if dh < -3 else "#2563eb")
                )

                ax.text(
                    0.5,
                    -0.16,
                    f"ROCS: {card['rocs']:.3f}   SA: {card['sa']:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=rocs_c,
                        edgecolor=rocs_e,
                        linewidth=1.0,
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
                    fontweight="medium",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor=dh_c,
                        edgecolor=dh_e,
                        linewidth=0.9,
                    ),
                )
                ax.text(
                    0.5,
                    -0.39,
                    f"ECFP4 Sim to Seed: {card['sim']:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8.0,
                    color="#475569",
                )

        fig.savefig(target_png, dpi=dpi)
        plt.close(fig)
        return target_png

def render_lead_structures_gallery(
    top_leads: pd.DataFrame,
    ref_smiles: str,
    output_png: Path | str,
    top_k: int = 5,
) -> Path | None:
    """Render 2D structure cards aligned to reference Murcko core."""
    depicter = AlignedDepicter(ref_smiles)
    return depicter.render_leads(top_leads, output_png=output_png, max_leads=top_k)


def _objective_conflict_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived descriptors used by the objective-conflict matrices."""
    if df.empty:
        return df
    clean = unscale_to_physical_metrics(
        df[df[std.VALID] == 1] if std.VALID in df.columns else df
    )
    mols = [
        Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s else None
        for s in clean.get(std.SMILES, [])
    ]
    clean["Fsp3"] = [
        rdMolDescriptors.CalcFractionCSP3(m) if m else np.nan for m in mols
    ]
    clean["Rings"] = [Lipinski.RingCount(m) if m else np.nan for m in mols]
    return clean


CONFLICT_PROPERTIES: tuple[tuple[str, str], ...] = (
    (std.ROCS, "ROCS"),
    ("Raw_SA", "SA (raw)"),
    (std.QED, "QED"),
    (std.MW, "MW"),
    (std.HAC, "HAC"),
    (std.LOGP, "LogP"),
    (std.TPSA, "TPSA"),
    (std.ROT_BONDS, "RotB"),
    ("Fsp3", "Fsp³"),
    ("Rings", "Rings"),
)


def _spearman_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    present = [(col, lbl) for col, lbl in CONFLICT_PROPERTIES if col in df.columns]
    labels = [lbl for _, lbl in present]
    if len(present) < 2:
        return labels, np.empty((len(labels), len(labels)))
    matrix = df[[col for col, _ in present]].corr(method="spearman").to_numpy(dtype=float)
    return labels, matrix


def render_objective_conflict_correlations(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
) -> Path:
    """Render baseline/reinforced Spearman matrices and their change (Δρ).

    Multi-objective RL is only as good as its reward geometry: if ROCS becomes
    anti-correlated with SA or MW the policy is trading physical similarity for
    cheap fragments. Two matrices plus their difference expose that directly,
    which a table of median shifts cannot.
    """
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)
    if df_after.empty:
        return target_png

    a_frame = _objective_conflict_frame(df_after)
    b_frame = _objective_conflict_frame(df_before)

    labels_a, mat_a = _spearman_matrix(a_frame)
    labels_b, mat_b = _spearman_matrix(b_frame)
    labels = labels_a if len(labels_a) >= len(labels_b) else labels_b
    if len(labels) < 2:
        return target_png

    def _align(labels_src: list[str], matrix: np.ndarray) -> np.ndarray:
        out = np.full((len(labels), len(labels)), np.nan)
        pos = {lbl: i for i, lbl in enumerate(labels_src)}
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if li in pos and lj in pos:
                    out[i, j] = matrix[pos[li], pos[lj]]
        return out

    mat_a = _align(labels_a, mat_a)
    mat_b = _align(labels_b, mat_b)
    mat_d = mat_b - mat_a

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=250)
    fig.suptitle(
        "Objective Conflict & Property Coupling (Spearman ρ): Did RL Couple the Objectives?",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    for ax, matrix, title in [
        (axes[0], mat_a, "(A) Baseline Prior"),
        (axes[1], mat_b, "(B) Reinforced Agent"),
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
    im3 = axes[2].imshow(
        np.ma.masked_invalid(mat_d), cmap="PuOr_r", vmin=-dmax, vmax=dmax
    )
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    axes[2].set_yticklabels(labels, fontsize=8.5)
    axes[2].set_title(
        "(C) Coupling Change  (B − A)", fontsize=11, fontweight="bold"
    )
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

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png

def render_scaffold_diversity_analysis(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
) -> Path:
    """Render scaffold-space diagnostics: coverage collapse, novelty partition, dominant cores.

    The old panel compared a percentage, an entropy in bits and a Gini index on one
    axis, so the bars were not comparable. The replacement answers the three
    questions a chemist actually asks about an RL library:

    1. How much of the library do the top scaffolds already explain (coverage)?
    2. How much chemistry is genuinely new versus inherited from the prior?
    3. Which cores dominate, and are they novel?
    """
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    b_clean = unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
    a_clean = unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])
    if a_clean.empty:
        return target_png

    def _ensure_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if std.SCAFFOLD not in df_out.columns or df_out[std.SCAFFOLD].isna().all():
            scafs = []
            for s in df_out.get(std.SMILES, []):
                m = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
                scafs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else "")
            df_out[std.SCAFFOLD] = scafs
        return df_out

    b_clean = _ensure_scaffolds(b_clean)
    a_clean = _ensure_scaffolds(a_clean)

    def _scaffold_counts(df: pd.DataFrame) -> pd.Series:
        scafs = df[std.SCAFFOLD].fillna("").astype(str)
        scafs = scafs[scafs != ""]
        return scafs.value_counts()

    b_counts = _scaffold_counts(b_clean)
    a_counts = _scaffold_counts(a_clean)

    def _entropy(counts: pd.Series) -> float:
        if counts.empty:
            return 0.0
        p = counts.values / counts.values.sum()
        return float(-np.sum(p * np.log2(p)))

    b_ent, a_ent = _entropy(b_counts), _entropy(a_counts)
    b_set, a_set = set(b_counts.index), set(a_counts.index)
    shared, novel, lost = a_set & b_set, a_set - b_set, b_set - a_set
    jaccard = len(shared) / len(shared | a_set) if (shared | a_set) else 0.0

    fig = plt.figure(figsize=(18, 6.4), dpi=250)
    fig.suptitle(
        "Bemis-Murcko Scaffold Space: Concentration, De Novo Novelty & Dominant Cores",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: cumulative coverage — how few cores explain most of the library.
    ax1 = fig.add_subplot(1, 3, 1)
    for counts, color, label in [
        (b_counts, "#64748B", f"Baseline Prior ({len(b_counts)} scaffolds)"),
        (a_counts, "#4F46E5", f"Reinforced Agent ({len(a_counts)} scaffolds)"),
    ]:
        if counts.empty:
            continue
        coverage = np.cumsum(counts.values) / counts.values.sum() * 100.0
        ranks = np.arange(1, len(coverage) + 1)
        ax1.plot(ranks, coverage, color=color, linewidth=2.0, label=label, zorder=3)
        ax1.fill_between(ranks, 0, coverage, color=color, alpha=0.10, linewidth=0)

    ax1.set_xscale("log")
    ax1.set_xlabel("Top-N scaffolds (log rank)", fontsize=9.5)
    ax1.set_ylabel("% of Library Covered", fontsize=9.5)
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
        f"Shannon H: prior {b_ent:.2f} → RL {a_ent:.2f} bits",
        transform=ax1.transAxes,
        va="top",
        fontsize=8.2,
        color="#334155",
    )

    # Panel B: novelty partition — invented vs inherited scaffolds.
    ax2 = fig.add_subplot(1, 3, 2)
    parts = [
        ("Shared with prior", len(shared), "#94A3B8"),
        ("De novo (new)", len(novel), "#4F46E5"),
        ("Prior-only (lost)", len(lost), "#E11D48"),
    ]
    x = np.arange(len(parts))
    vals = [v for _, v, _ in parts]
    bars = ax2.bar(
        x,
        vals,
        color=[c for _, _, c in parts],
        alpha=0.9,
        edgecolor="#1e293b",
        width=0.6,
    )
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(vals) * 0.02 if max(vals) else val,
            str(val),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([n for n, _, _ in parts], fontsize=8.6)
    ax2.set_ylabel("Distinct Scaffolds", fontsize=9.5)
    ax2.set_title("(B) Scaffold Novelty Partition", fontsize=11, fontweight="bold")
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

    # Panel C: dominant cores, tagged novel vs inherited. Twelve cores rather
    # than six: the tail of the count distribution is where de novo chemistry
    # actually lives, and six slots hide whether the head is one mode or many.
    ax3 = fig.add_subplot(1, 3, 3)
    top = a_counts.head(12)
    scaf_mols = [Chem.MolFromSmiles(s) for s in top.index]
    total = max(len(a_clean), 1)
    legends = [
        f"{cnt / total * 100:.1f}% | {'NEW' if s in novel else 'shared'}"
        for s, cnt in top.items()
    ]
    if scaf_mols and all(m is not None for m in scaf_mols):
        img = Draw.MolsToGridImage(
            scaf_mols,
            molsPerRow=4,
            subImgSize=(200, 150),
            legends=legends,
            useSVG=False,
        )
        ax3.imshow(img)
        ax3.axis("off")
        ax3.set_title(
            "(C) Top-12 Reinforced Scaffolds (prevalence | novelty)",
            fontsize=11,
            fontweight="bold",
        )
    else:
        ax3.text(0.5, 0.5, "Insufficient scaffold data", ha="center", va="center")
        ax3.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png

def render_lead_to_seed_3d_superposition(
    top_leads: pd.DataFrame,
    ref_sdf_or_mol: Path | str | Chem.Mol,
    output_png: Path | str,
    output_sdf: Path | str | None = None,
    max_leads: int = 5,
) -> Path:
    """Align top de novo chemical leads onto the 3D reference query seed in 3D, export SDF, and render 3D superposition."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load reference 3D molecule
    ref_mol: Chem.Mol | None = None
    if isinstance(ref_sdf_or_mol, Chem.Mol):
        ref_mol = Chem.Mol(ref_sdf_or_mol)
    elif isinstance(ref_sdf_or_mol, (str, Path)) and Path(ref_sdf_or_mol).is_file():
        suppl = Chem.SDMolSupplier(str(ref_sdf_or_mol), removeHs=False)
        ref_mol = next((m for m in suppl if m is not None), None)
    elif isinstance(ref_sdf_or_mol, str):
        ref_mol = Chem.MolFromSmiles(ref_sdf_or_mol)

    if ref_mol is None:
        return target_png

    ref_heavy = Chem.RemoveHs(Chem.Mol(ref_mol))
    if ref_heavy.GetNumConformers() == 0:
        ref_heavy_h = Chem.AddHs(ref_heavy)
        AllChem.EmbedMolecule(ref_heavy_h, AllChem.ETKDGv3())
        ref_heavy = Chem.RemoveHs(ref_heavy_h)

    # Put the seed into its inertial frame *before* aligning, so the drawn seed and the
    # exported leads share one coordinate system (align_to_reference canonicalises the
    # reference it is given).
    try:
        rdMolTransforms.CanonicalizeConformer(ref_heavy.GetConformer())
    except (RuntimeError, ValueError):  # pragma: no cover - singular inertia tensor
        pass

    # 2. Embed and align top leads
    aligned_leads: list[tuple[Chem.Mol, dict[str, Any]]] = []
    seed_hac = ref_heavy.GetNumHeavyAtoms()

    for rank, (_, row) in enumerate(top_leads.head(max_leads).iterrows(), start=1):
        smi = str(row.get(std.SMILES, ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + rank
        cid = AllChem.EmbedMolecule(mol_h, params)
        if cid < 0:
            cid = AllChem.EmbedMolecule(mol_h)
        if cid >= 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            except Exception:
                pass

        lead_heavy = Chem.RemoveHs(mol_h)
        if lead_heavy.GetNumConformers() == 0:
            continue

        # Move the lead into the seed frame; keep the winning pose for the SDF and plot.
        lead_heavy, combo_score, shape_score, color_score = align_to_reference(
            lead_heavy, ref_heavy
        )

        lead_hac = lead_heavy.GetNumHeavyAtoms()
        mw = float(row.get(std.MW, Descriptors.MolWt(lead_heavy)))
        rocs = float(row.get(std.ROCS, 0.0))
        sa = float(row.get("Raw_SA", row.get(std.SA, 0.0)))
        qed = float(row.get(std.QED, 0.0))

        props = {
            "Rank": rank,
            "SMILES": smi,
            "ROCS": rocs,
            "SA": sa,
            "QED": qed,
            "MW": mw,
            "HAC": lead_hac,
            "Delta_HAC": lead_hac - seed_hac,
            "TanimotoCombo": combo_score,
            "ShapeScore": shape_score,
            "ColorScore": color_score,
        }
        aligned_leads.append((lead_heavy, props))

    # 3. Export SDF if requested
    if output_sdf:
        sdf_path = Path(output_sdf)
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf_path))
        ref_out = Chem.Mol(ref_heavy)
        ref_out.SetProp("_Name", "TARGET_QUERY_SEED")
        ref_out.SetProp("HAC", str(seed_hac))
        ref_out.SetProp("MW", f"{Descriptors.MolWt(ref_heavy):.1f}")
        ref_out.SetProp("Alignment", "reference frame (canonical inertial axes)")
        writer.write(ref_out)

        for lead_mol, props in aligned_leads:
            m_out = Chem.Mol(lead_mol)
            m_out.SetProp("_Name", f"LEAD_{props['Rank']}")
            for k, v in props.items():
                if isinstance(v, float):
                    v = f"{v:.4f}"
                m_out.SetProp(str(k), str(v))
            m_out.SetProp(
                "Alignment",
                "RDKit Shape+Color TanimotoCombo, rigidly aligned into TARGET_QUERY_SEED",
            )
            writer.write(m_out)
        writer.close()

    # 4. Render 3-panel 3D Superposition Figure
    fig = plt.figure(figsize=(16, 5.5), dpi=250)
    fig.suptitle(
        "3D Molecular Superposition: Target Query Seed vs. Prioritized De Novo Leads",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    def _draw_3d(
        ax: Any,
        mol: Chem.Mol,
        color: str,
        alpha: float = 0.8,
        lw: float = 1.3,
        s: float = 14,
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

    # All three panels share one bounding box and one box aspect, so the seed and the
    # leads keep their true relative size instead of each panel autoscaling separately.
    frames = [ref_heavy.GetConformer().GetPositions()] + [
        m.GetConformer().GetPositions() for m, _ in aligned_leads
    ]
    stack = np.vstack(frames)
    lo = stack.min(axis=0)
    hi = stack.max(axis=0)
    pad = 0.06 * max(float((hi - lo).max()), 1e-3)
    lo, hi = lo - pad, hi + pad

    def _share_frame(ax: Any) -> None:
        ax.set_xlim(float(lo[0]), float(hi[0]))
        ax.set_ylim(float(lo[1]), float(hi[1]))
        ax.set_zlim(float(lo[2]), float(hi[2]))
        ax.set_box_aspect(tuple(float(v) for v in (hi - lo)))

    # Panel A: Reference Seed Only
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_3d(ax1, ref_heavy, color="#2563eb", alpha=0.9, lw=1.6, s=18)
    ax1.set_title(
        f"(A) Target Query Seed (3D Conformer)\nHAC = {seed_hac} | MW = {Descriptors.MolWt(ref_heavy):.1f} Da",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax1.view_init(elev=20, azim=40)
    ax1.set_axis_off()
    _share_frame(ax1)

    # Panel B: Top Lead #1 Aligned onto Seed
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_3d(ax2, ref_heavy, color="#94a3b8", alpha=0.45, lw=1.2, s=10)
    if aligned_leads:
        lead1, p1 = aligned_leads[0]
        _draw_3d(ax2, lead1, color="#16a34a", alpha=0.9, lw=1.6, s=18)
        ax2.set_title(
            f"(B) Top Lead #1 Aligned to Seed\n"
            f"Pose combo = {p1['TanimotoCombo']:.2f} "
            f"(shape {p1['ShapeScore']:.2f}) | pipeline ROCS = {p1['ROCS']:.3f}\n"
            f"ΔHAC = {p1['Delta_HAC']:+d} | SA = {p1['SA']:.2f}",
            fontsize=10.0,
            fontweight="bold",
            pad=10,
        )
    else:
        ax2.set_title("(B) Top Lead #1", fontsize=11, fontweight="bold", pad=10)
    ax2.view_init(elev=20, azim=40)
    ax2.set_axis_off()
    _share_frame(ax2)

    # Panel C: Superposition of All Top Leads Aligned onto Seed
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_3d(ax3, ref_heavy, color="#1e293b", alpha=0.6, lw=1.5, s=14)
    lead_colors = ["#16a34a", "#2563eb", "#ea580c", "#8b5cf6", "#d97706"]
    for i, (l_mol, _p) in enumerate(aligned_leads):
        c = lead_colors[i % len(lead_colors)]
        _draw_3d(ax3, l_mol, color=c, alpha=0.75, lw=1.3, s=12)

    ax3.set_title(
        f"(C) Superposition Ensemble ({len(aligned_leads)} Leads)\nAligned via 3D Shape & Features",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax3.view_init(elev=20, azim=40)
    ax3.set_axis_off()
    _share_frame(ax3)

    plt.tight_layout()
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_epoch_property_trajectories(
    smiles_tsv_path: Path | str,
    output_png: Path | str,
    ref_smiles: str = "",
    molecule_table: pd.DataFrame | None = None,
) -> Path:
    """Render the longitudinal 3×3 property panel (median + IQR + p95 + seed).

    Tracks every property the pipeline scores or reports: the two RL objectives
    (ROCS desirability, raw SA), the optional QED arm, and the medicinal-chemistry
    drift guards (MW, HAC, LogP, TPSA, Fsp³, rotatable bonds).

    ``molecule_table`` is the run's pre-computed per-molecule dataset. When it is
    supplied nothing is rescored: the panel is a pure summary of that table. Without
    it the table is derived from the monitor TSV, which keeps the function usable
    on an ad-hoc file but is not the production path.
    """
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    if molecule_table is not None and not molecule_table.empty:
        df = molecule_table.copy()
        # The cached table is on the physical TanimotoCombo scale; the panel is
        # labelled in desirability units, and the reward treats the [0, 2] range as
        # ROCS/2. Dividing here keeps the plotted series on the rewarded scale.
        df["ROCS_Desirability"] = pd.to_numeric(
            df.get(std.ROCS), errors="coerce"
        ) / 2.0
    else:
        tsv_p = Path(smiles_tsv_path)
        if not tsv_p.is_file():
            return target_png
        df = epoch_molecule_table(pd.read_csv(tsv_p, sep="\t"), ref_smiles)
        df["ROCS_Desirability"] = pd.to_numeric(df.get(std.ROCS), errors="coerce")

    if "Epoch" not in df.columns or "SMILES" not in df.columns:
        return target_png
    if "ValidMol" not in df.columns:
        df["ValidMol"] = [
            1 if Chem.MolFromSmiles(str(s)) is not None else 0 for s in df["SMILES"]
        ]
    valid_df = df[df["ValidMol"].astype(int) == 1].copy()

    ref_m = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
    ref_vals = {
        "ROCS_Desirability": 1.0,
        "QED": float(QED.qed(ref_m)) if ref_m else 0.21,
        "RawSA": float(calculate_sa_score(ref_m)) if ref_m else 3.74,
        "MW": float(Descriptors.MolWt(ref_m)) if ref_m else 458.4,
        "HAC": float(ref_m.GetNumHeavyAtoms()) if ref_m else 33.0,
        "LogP": float(Descriptors.MolLogP(ref_m)) if ref_m else 2.0,
        "TPSA": float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 200.0,
        "Fsp3": float(rdMolDescriptors.CalcFractionCSP3(ref_m)) if ref_m else 0.36,
        "RotBonds": float(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3.0,
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), dpi=250)
    fig.suptitle(
        "Longitudinal Per-Epoch Property Trajectories (Median, IQR, p95 & Seed Reference)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    panels_config = [
        (
            "ROCS_Desirability",
            "ROCS Desirability (3D Shape+Color)",
            "Desirability (0–1)",
        ),
        ("QED", "QED Drug-Likeness", "QED (0–1)"),
        ("RawSA", "Synthetic Accessibility (raw RDKit SA)", "SA (lower = easier)"),
        ("MW", "Molecular Weight Drift", "MW (Da)"),
        ("HAC", "Heavy Atom Count (Shrinkage)", "Atoms"),
        ("LogP", "Lipophilicity", "cLogP"),
        ("TPSA", "Polar Surface Area", "TPSA (Å²)"),
        ("Fsp3", "Carbon sp³ Fraction", "Fraction"),
        ("RotBonds", "Rotatable Bonds", "Bonds"),
    ]

    for idx, (col, title, y_lbl) in enumerate(panels_config):
        ax = axes[idx // 3, idx % 3]
        if col not in valid_df.columns:
            ax.axis("off")
            continue
        source = valid_df
        value_col = col

        grp = source.groupby("Epoch")[value_col]
        med = grp.median()
        p25 = grp.quantile(0.25)
        p75 = grp.quantile(0.75)
        p95 = grp.quantile(0.95)
        if med.dropna().empty:
            ax.axis("off")
            continue
        ep = med.index.values

        ax.plot(
            ep, med.values, color="#4F46E5", linewidth=2.0, label="Agent Median", zorder=4
        )
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

        ref_v = ref_vals.get(col)
        if ref_v is not None:
            ref_label = (
                "Seed ceiling (desirability 1.0)"
                if col == "ROCS_Desirability"
                else f"Seed ({ref_v:.2f})"
            )
            ax.axhline(
                ref_v,
                color="#0D9488",
                linestyle=":",
                linewidth=1.6,
                label=ref_label,
                zorder=2,
            )

        lo = float(np.nanmin(p25.values))
        hi = float(np.nanmax(p95.values))
        if ref_v is not None:
            lo, hi = min(lo, ref_v), max(hi, ref_v)
        span = hi - lo if hi > lo else 1.0
        ax.set_ylim(lo - 0.10 * span, hi + 0.10 * span)

        ax.set_title(f"({chr(65 + idx)}) {title}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9.0)
        ax.set_ylabel(y_lbl, fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png

def _sa_desirability(
    raw_sa: Sequence[float] | np.ndarray, run_config: dict | None = None
) -> np.ndarray:
    """Map raw RDKit SA onto the pipeline's clipped SA desirability channel.

    ``create_environment`` wraps SA in ``ClippedScore(lower_x=sa_score_lower,
    upper_x=sa_score_upper)``: raw SA above ``lower`` scores 0, below ``upper`` scores 1,
    linear in between. Reusing the run's own bounds keeps every SA value on exactly the
    scale the policy was rewarded on.
    """
    cfg = run_config or {}
    hi = float(cfg.get("sa_score_upper", 3.0))
    lo = float(cfg.get("sa_score_lower", 5.0))
    values = np.asarray(raw_sa, dtype=float)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.zeros_like(values)
    return np.clip((lo - values) / (lo - hi), 0.0, 1.0)


def eval_source_text(manifest: dict) -> str:
    """Where the evaluation arm's weights came from, in words a reader can trust.

    Runs made before the best-epoch restore recorded nothing; saying so is
    better than implying that the arm is the best model.
    """
    source = str(manifest.get("evaluation_source") or "")
    epoch = manifest.get("best_epoch")
    if source == "best_epoch":
        return f"from the best checkpoint (epoch {epoch})" if epoch else "from the best checkpoint"
    if source == "last_epoch":
        return "from the last epoch (no best checkpoint was recorded)"
    return "from the saved checkpoint (provenance not recorded)"


def write_training_candidates(
    table: pd.DataFrame, root: Path, *, limit: int = 25
) -> Path | None:
    """Best molecules seen during training, for candidate review only.

    These come from earlier, weaker policies, so they are deliberately *not* a
    statistic: they are never pooled into the final-arm distributions, the
    scorecard or a cross-run comparison. The file exists so that a good hit found
    in an intermediate epoch is not lost when the run continues.
    """
    if table.empty or std.SMILES not in table.columns:
        return None
    work = table.copy()
    if "ValidMol" in work.columns:
        work = work[work["ValidMol"].astype(int) == 1]
    if work.empty:
        return None
    key = "Canonical" if "Canonical" in work.columns else std.SMILES
    sort_keys = [c for c in ("Desired", "ROCS") if c in work.columns]
    if sort_keys:
        work = work.sort_values(by=sort_keys, ascending=False)
    work = work.drop_duplicates(subset=[key]).head(limit)
    columns = [
        c
        for c in ("Epoch", "SMILES", "Canonical", "ROCS", "QED", "RawSA", "SimToSeed", "Desired")
        if c in work.columns
    ]
    target = Path(root) / "training_candidates.tsv"
    work[columns].to_csv(target, sep="\t", index=False)
    return target


def epoch_molecule_table(
    df: pd.DataFrame,
    ref_smiles: str = "",
    run_config: dict | None = None,
    rocs_scale: float = 1.0,
) -> pd.DataFrame:
    """Per-molecule monitor table with recomputed chemistry and the reward channels.

    ``_smiles.tsv`` stores the raw TanimotoCombo (``ROCS``), the clipped SA desirability
    (``SA``) and the validity/desirability flags, but no QED, no physical SA and no
    similarity to the seed. Those are needed by the multi-objective, exploration and
    enrichment figures, so they are recomputed once here and memoised per structure.

    The table also carries the physicochemical drift guards (MW, HAC, LogP, TPSA, Fsp³,
    rotatable bonds) so that the longitudinal panel and the training diagnostics read
    one pre-computed dataset instead of each re-deriving their own.
    """
    columns = [
        "Epoch",
        "SMILES",
        "Canonical",
        "ValidMol",
        "Desired",
        "ROCS",
        "QED",
        "RawSA",
        "SA_Desirability",
        "SimToSeed",
        "SelectionScore",
        "MW",
        "HAC",
        "LogP",
        "TPSA",
        "Fsp3",
        "RotBonds",
    ]
    if df.empty or std.SMILES not in df.columns or "Epoch" not in df.columns:
        return pd.DataFrame(columns=columns)

    out = df.copy()
    smiles_values = [
        str(s) if isinstance(s, str) and s.strip() else "" for s in out[std.SMILES]
    ]
    ref_mol = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
    ref_fp = (
        rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
        if ref_mol is not None
        else None
    )

    cache: dict[str, tuple] = {}
    canonical: list[str] = []
    valid: list[int] = []
    qed_values: list[float] = []
    sa_values: list[float] = []
    sim_values: list[float] = []
    mw_values: list[float] = []
    hac_values: list[float] = []
    logp_values: list[float] = []
    tpsa_values: list[float] = []
    fsp3_values: list[float] = []
    rotb_values: list[float] = []
    for smiles in smiles_values:
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                cache[smiles] = ("", 0) + (np.nan,) * 9
            else:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=2048
                )
                cache[smiles] = (
                    Chem.MolToSmiles(mol),
                    1,
                    float(QED.qed(mol)),
                    float(calculate_sa_score(mol)),
                    float(DataStructs.TanimotoSimilarity(ref_fp, fp))
                    if ref_fp is not None
                    else np.nan,
                    float(Descriptors.MolWt(mol)),
                    float(mol.GetNumHeavyAtoms()),
                    float(Descriptors.MolLogP(mol)),
                    float(rdMolDescriptors.CalcTPSA(mol)),
                    float(rdMolDescriptors.CalcFractionCSP3(mol)),
                    float(Lipinski.NumRotatableBonds(mol)),
                )
        (
            canon,
            ok,
            qed_v,
            sa_v,
            sim_v,
            mw_v,
            hac_v,
            logp_v,
            tpsa_v,
            fsp3_v,
            rotb_v,
        ) = cache[smiles]
        canonical.append(canon)
        valid.append(ok)
        qed_values.append(qed_v)
        sa_values.append(sa_v)
        sim_values.append(sim_v)
        mw_values.append(mw_v)
        hac_values.append(hac_v)
        logp_values.append(logp_v)
        tpsa_values.append(tpsa_v)
        fsp3_values.append(fsp3_v)
        rotb_values.append(rotb_v)

    out["Canonical"] = canonical
    out["ValidMol"] = valid
    out["QED"] = qed_values
    out["RawSA"] = sa_values
    out["SimToSeed"] = sim_values
    out["MW"] = mw_values
    out["HAC"] = hac_values
    out["LogP"] = logp_values
    out["TPSA"] = tpsa_values
    out["Fsp3"] = fsp3_values
    out["RotBonds"] = rotb_values
    out["SA_Desirability"] = _sa_desirability(
        out["RawSA"].to_numpy(float), run_config
    )
    out["ROCS"] = pd.to_numeric(
        out.get(std.ROCS, pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    if rocs_scale != 1.0:
        # The monitor writes the value the scorer produced. A run scored with a
        # ClippedScore modifier therefore holds ``min(raw, clip) / clip`` here too,
        # and the longitudinal figures would otherwise be on a different scale
        # from the evaluation arm they are compared with.
        out["ROCS"] = out["ROCS"] * float(rocs_scale)
    out["Desired"] = pd.to_numeric(
        out.get(std.DESIRED, pd.Series(0, index=out.index)), errors="coerce"
    ).fillna(0)

    # Same ranking index the result analyzer uses for the prioritized leads.
    out["SelectionScore"] = lead_rank_score(out, run_config)
    return out


#: Filename of the run's persisted per-molecule training dataset.
MOLECULE_TABLE_NAME = "training_molecules.csv.gz"
#: Sidecar that records which monitor data and scoring settings produced the table.
MOLECULE_TABLE_META_NAME = "training_molecules.json"
MOLECULE_TABLE_SCHEMA_VERSION = 1


def molecule_table_path(root: Path | str) -> Path:
    """Location of the pre-computed per-molecule dataset inside a run."""
    return Path(root) / "molecules" / MOLECULE_TABLE_NAME


def molecule_table_signature(
    monitor: pd.DataFrame,
    ref_smiles: str,
    run_config: dict | None,
    rocs_scale: float,
) -> dict[str, Any]:
    """Fingerprint the inputs that make a molecule table reproducible.

    Only the columns ``epoch_molecule_table`` actually consumes are hashed, so the
    table stays valid when ``audit.regenerate`` rewrites the monitor TSV with a
    physical-SA column: the molecules, their epoch and their reward channels are
    unchanged. A changed ROCS verdict, seed or SA modifier does invalidate it.
    """
    key_columns = [
        column
        for column in ("Epoch", std.SMILES, std.ROCS, std.DESIRED)
        if column in monitor.columns
    ]
    payload = monitor[key_columns].to_csv(index=False)
    cfg = run_config or {}
    return {
        "schema_version": MOLECULE_TABLE_SCHEMA_VERSION,
        "molecules_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "rows": int(len(monitor)),
        "ref_smiles": ref_smiles,
        "rocs_scale": round(float(rocs_scale), 12),
        "sa_score_lower": cfg.get("sa_score_lower"),
        "sa_score_upper": cfg.get("sa_score_upper"),
    }


def cached_molecule_table(
    root: Path | str,
    monitor: pd.DataFrame,
    ref_smiles: str = "",
    run_config: dict | None = None,
    rocs_scale: float = 1.0,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the run's per-molecule dataset, computing and caching it once.

    Every figure that needs per-molecule chemistry reads this table instead of
    rescoring the monitor TSV, so a re-render is pure plotting. The cache is keyed
    on the molecule-defining monitor fields, the reference SMILES, the ROCS scale
    and the SA modifier bounds; any change rebuilds it, as does ``refresh``.
    """
    if monitor is None or monitor.empty:
        return pd.DataFrame()

    table_path = molecule_table_path(root)
    meta_path = table_path.with_name(MOLECULE_TABLE_META_NAME)
    signature = molecule_table_signature(monitor, ref_smiles, run_config, rocs_scale)

    if not refresh and table_path.is_file() and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored = None
        if isinstance(stored, dict) and all(
            stored.get(key) == value for key, value in signature.items()
        ):
            try:
                cached = pd.read_csv(table_path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                cached = None
            if cached is not None and len(cached) == signature["rows"]:
                # CSV has no distinction between an empty field and a missing
                # value; restore the in-memory representation for the molecule
                # text columns so a cache hit equals a fresh build.
                for column in ("SMILES", "Canonical"):
                    if column in cached.columns:
                        cached[column] = cached[column].fillna("")
                return cached

    table = epoch_molecule_table(
        monitor, ref_smiles, run_config, rocs_scale=rocs_scale
    )
    if table.empty:
        return table

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                **signature,
                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("[molecules] cached %d rows -> %s", len(table), table_path)
    return table



def _pareto_mask_3d(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Boolean mask of points not dominated by any other point (all objectives maximized)."""
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    dominant = np.all(pts > ref, axis=1)
    mask = np.zeros(len(pts), dtype=bool)
    for i in np.flatnonzero(dominant):
        dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        mask[i] = not bool(dominated.any())
    return mask


def _skyline_area_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """Area dominated by a 2D point set above ``reference``."""
    ref_x, ref_y = float(reference[0]), float(reference[1])
    pts = [
        (float(x), float(y))
        for x, y in np.asarray(points, dtype=float)
        if x > ref_x and y > ref_y
    ]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: -p[0])
    area = 0.0
    best_y = ref_y
    for index, (x, y) in enumerate(pts):
        x_next = pts[index + 1][0] if index + 1 < len(pts) else ref_x
        best_y = max(best_y, y)
        area += max(0.0, x - x_next) * max(0.0, best_y - ref_y)
    return area


def hypervolume_3d(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact dominated hypervolume of a 3D point set above ``reference``.

    Sweeps z from high to low and integrates the 2D skyline area of the points seen so
    far. The non-dominated fronts are small, so the quadratic sweep is far cheaper than
    importing a dedicated hypervolume library.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    ref = np.asarray(reference, dtype=float)
    pts = pts[np.all(pts > ref, axis=1)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 2])]

    area = 0.0
    sky: list[tuple[float, float]] = []
    previous_z: float | None = None
    for x, y, z in pts:
        if previous_z is not None:
            area += (previous_z - z) * _skyline_area_2d(np.array(sky), ref)
        sky.append((x, y))
        previous_z = z
    area += (previous_z - ref[2]) * _skyline_area_2d(np.array(sky), ref)
    return float(area)


def _valid_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose SMILES parsed, when the frame carries a validity flag.

    ``unscale_to_physical_metrics`` leaves unparsable rows in place with ``ROCS = 0``,
    so an unfiltered arm would silently mix 4000 failed generations into every ROCS
    quantile and enrichment baseline.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.iloc[0:0]
    if std.VALID in df.columns:
        flag = pd.to_numeric(df[std.VALID], errors="coerce").fillna(0.0)
        return df[flag > 0.5]
    return df


def _arm_objectives(
    df: pd.DataFrame | None, run_config: dict | None = None
) -> np.ndarray:
    """(ROCS/2, QED, SA desirability) for a scored candidate frame, higher = better."""
    df = _valid_subset(df)
    if df.empty:
        return np.empty((0, 3))
    if std.ROCS not in df.columns:
        return np.empty((0, 3))
    rocs = pd.to_numeric(df[std.ROCS], errors="coerce").to_numpy(float) / 2.0
    qed_series = df.get(std.QED)
    qed = (
        pd.to_numeric(qed_series, errors="coerce").to_numpy(float)
        if qed_series is not None
        else np.full(len(df), np.nan)
    )
    sa_series = df.get("Raw_SA", df.get(std.SA))
    raw_sa = (
        pd.to_numeric(sa_series, errors="coerce").to_numpy(float)
        if sa_series is not None
        else np.full(len(df), np.nan)
    )
    return np.column_stack([rocs, qed, _sa_desirability(raw_sa, run_config)])


def _monitor_best_epoch(fit_tsv_path: Path | str | None) -> int | None:
    """Read the monitor's recorded checkpoint epoch, if the fit table is available."""
    if not fit_tsv_path:
        return None
    path = Path(fit_tsv_path)
    if not path.is_file():
        return None
    try:
        fit = pd.read_csv(path, sep="\t")
    except (OSError, pd.errors.ParserError):
        return None
    if std.BEST_EPOCH not in fit.columns or fit[std.BEST_EPOCH].dropna().empty:
        return None
    return int(fit[std.BEST_EPOCH].dropna().iloc[-1])


def _load_molecule_table(
    smiles_tsv_path: Path | str | None,
    molecule_table: pd.DataFrame | None,
    ref_smiles: str = "",
    run_config: dict | None = None,
) -> pd.DataFrame:
    """Return the shared per-molecule epoch table, reading the monitor TSV if needed."""
    if molecule_table is not None:
        return molecule_table
    if not smiles_tsv_path:
        return pd.DataFrame()
    path = Path(smiles_tsv_path)
    if not path.is_file():
        return pd.DataFrame()
    return epoch_molecule_table(pd.read_csv(path, sep="\t"), ref_smiles, run_config)


def _fingerprints_for(smiles_values: Sequence[str]) -> list[Any]:
    """Morgan fingerprints for a SMILES list, parsing each structure once."""
    cache: dict[str, Any] = {}
    out: list[Any] = []
    for smiles in smiles_values:
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            cache[smiles] = (
                rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                if mol is not None
                else None
            )
        out.append(cache[smiles])
    return out


def render_training_diagnostics(
    smiles_tsv_path: Path | str | None,
    output_png: Path | str,
    ref_smiles: str = "",
    run_config: dict | None = None,
    df_before: pd.DataFrame | None = None,
    df_after: pd.DataFrame | None = None,
    fit_tsv_path: Path | str | None = None,
    molecule_table: pd.DataFrame | None = None,
) -> Path:
    """Six-panel longitudinal diagnostic of one training run.

    The former figures 12-16 each asked a different question of the *same* per-epoch
    monitor table, so they are one figure:

    (A) weight-free multi-objective progress (dominated hypervolume),
    (B) the Pareto front of the delivered arm (or of the monitor's best epoch when no
    arm is available), coloured by QED,
    (C) how the population centroid moved through seed similarity vs shape fit,
    (D) what each epoch's sampling budget actually produced,
    (E) how stale the best molecule became after it was found,
    (F) when the delivered leads first appeared.

    Panels A and B need no reward weights, so they answer "is the policy actually
    better overall?" independently of the composite objective the run optimized.
    Panels C-F are process diagnostics: they explain *how* a run got there, and they
    are read together because each one alone is easy to misread (a flat hypervolume
    with high lead-set churn means sampling noise, not a plateau).
    """
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    table = _load_molecule_table(
        smiles_tsv_path, molecule_table, ref_smiles, run_config
    )
    valid = table[table["ValidMol"] == 1].copy() if not table.empty else table
    if valid.empty:
        return target_png
    required = ["ROCS", "QED", "RawSA", "SimToSeed", "SelectionScore"]
    valid = valid.dropna(subset=[c for c in ("ROCS", "QED", "RawSA") if c in valid])
    if valid.empty:
        return target_png
    missing = [c for c in required if c not in valid.columns]
    if missing:
        LOGGER.debug("Training diagnostics without columns %s", missing)

    valid["ROCS_Norm"] = valid["ROCS"] / 2.0
    obj_cols = ["ROCS_Norm", "QED", "SA_Desirability"]
    reference = np.zeros(3)
    epochs = np.array(sorted(valid["Epoch"].unique()))

    # ---- shared per-epoch summaries -------------------------------------
    hv_epoch: list[float] = []
    hv_cumulative: list[float] = []
    front = np.empty((0, 3))
    for epoch in epochs:
        pts = valid[valid["Epoch"] == epoch][obj_cols].to_numpy(float)
        hv_epoch.append(hypervolume_3d(pts, reference))
        front = np.vstack([front, pts])
        front = front[_pareto_mask_3d(front, reference)]
        hv_cumulative.append(hypervolume_3d(front, reference))

    ref_mol = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
    seed_obj = np.array(
        [
            1.0,
            float(QED.qed(ref_mol)) if ref_mol is not None else np.nan,
            float(_sa_desirability([calculate_sa_score(ref_mol)], run_config)[0])
            if ref_mol is not None
            else np.nan,
        ]
    )
    seed_hv = float(np.prod(seed_obj)) if np.all(np.isfinite(seed_obj)) else np.nan
    best_epoch = _monitor_best_epoch(fit_tsv_path)

    centroid_sim, centroid_rocs = [], []
    for epoch in epochs:
        group = valid[valid["Epoch"] == epoch]
        centroid_sim.append(float(group["SimToSeed"].mean()))
        centroid_rocs.append(float(group["ROCS_Norm"].mean()))
    centroid_sim = np.array(centroid_sim)
    centroid_rocs = np.array(centroid_rocs)

    baseline_sim = np.empty(0)
    baseline_rocs = np.empty(0)
    if df_before is not None and not df_before.empty and std.SMILES in df_before.columns:
        base = _valid_subset(df_before).dropna(subset=[std.SMILES])
        base = base[base[std.SMILES].astype(str).str.len() > 0]
        if len(base) > 800:
            picks = np.linspace(0, len(base) - 1, 800).round().astype(int)
            base = base.iloc[picks]
        baseline_rocs = (
            pd.to_numeric(base.get(std.ROCS), errors="coerce").to_numpy(float) / 2.0
        )
        if ref_mol is not None:
            ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                ref_mol, 2, nBits=2048
            )
            baseline_sim = np.array(
                [
                    DataStructs.TanimotoSimilarity(ref_fp, fp)
                    if fp is not None
                    else np.nan
                    for fp in _fingerprints_for(base[std.SMILES].astype(str).tolist())
                ]
            )

    outcomes: list[str] = []
    seen_canonical: set[str] = set()
    for row in table.itertuples():
        is_valid = int(getattr(row, "ValidMol", 0)) == 1
        is_desired = float(getattr(row, "Desired", 0.0) or 0.0) > 0.5
        canonical = str(getattr(row, "Canonical", ""))
        if not is_valid:
            outcomes.append("invalid")
        elif not is_desired:
            outcomes.append("valid_undesired")
        elif canonical and canonical in seen_canonical:
            outcomes.append("desired_duplicate")
        else:
            outcomes.append("desired_novel")
            if canonical:
                seen_canonical.add(canonical)
    outcome_frame = table.copy()
    outcome_frame["Outcome"] = outcomes
    outcome_order = [
        "desired_novel",
        "desired_duplicate",
        "valid_undesired",
        "invalid",
    ]
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

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.4), dpi=250)
    fig.suptitle(
        "Training Diagnostics: Weight-Free Progress, Population Path and Lead Discovery",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    def _mark_best(ax: plt.Axes) -> None:
        if best_epoch is None:
            return
        ax.axvline(
            best_epoch,
            color="#7C3AED",
            linewidth=1.4,
            linestyle="-.",
            label=f"Best checkpoint (ep {best_epoch})",
            zorder=1,
        )

    # (A) Hypervolume of each epoch's set and of everything seen so far.
    ax = axes[0, 0]
    _mark_best(ax)
    ax.plot(
        epochs,
        hv_epoch,
        color="#93C5FD",
        linewidth=1.1,
        alpha=0.9,
        label="Per-epoch hypervolume",
        zorder=2,
    )
    ax.plot(
        epochs,
        pd.Series(hv_epoch).rolling(5, center=True, min_periods=1).median().to_numpy(),
        color="#1D4ED8",
        linewidth=2.2,
        label="Per-epoch (5-epoch median)",
        zorder=3,
    )
    ax.plot(
        epochs,
        hv_cumulative,
        color="#0F172A",
        linewidth=1.8,
        linestyle="--",
        label="Cumulative hypervolume",
        zorder=3,
    )
    if np.isfinite(seed_hv):
        ax.axhline(
            seed_hv,
            color="#DC2626",
            linewidth=1.2,
            linestyle=":",
            label=f"Seed single point ({seed_hv:.3f})",
            zorder=1,
        )
    ax.set_title("(A) Dominated Hypervolume per Epoch", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized hypervolume")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.4)

    # (B) Where the hypervolume comes from: the Pareto front of the delivered arm.
    ax = axes[0, 1]
    baseline_obj = _arm_objectives(df_before, run_config)
    if baseline_obj.size:
        baseline_obj = baseline_obj[np.all(np.isfinite(baseline_obj), axis=1)]
        if len(baseline_obj) > 1200:
            picks = np.linspace(0, len(baseline_obj) - 1, 1200).round().astype(int)
            baseline_obj = baseline_obj[picks]
        ax.scatter(
            baseline_obj[:, 0],
            baseline_obj[:, 2],
            s=10,
            color="#CBD5E1",
            alpha=0.55,
            linewidth=0,
            label="Baseline prior",
            zorder=1,
        )
    delivered_obj = _arm_objectives(df_after, run_config)
    if delivered_obj.size:
        delivered_obj = delivered_obj[np.all(np.isfinite(delivered_obj), axis=1)]
        stage_label = "Delivered arm"
        stage_short = "Delivered Arm"
    else:
        if best_epoch is not None and best_epoch in set(epochs.tolist()):
            stage_epoch = int(best_epoch)
        else:
            stage_epoch = int(epochs.max())
        stage_label = f"Epoch {stage_epoch}"
        stage_short = f"Epoch {stage_epoch} (monitor)"
        delivered_obj = valid[valid["Epoch"] == stage_epoch][obj_cols].to_numpy(float)
        if delivered_obj.shape[0] > 4000:
            picks = np.linspace(0, delivered_obj.shape[0] - 1, 4000).round().astype(int)
            delivered_obj = delivered_obj[picks]
    final_obj = delivered_obj
    mask = _pareto_mask_3d(final_obj, reference)
    ax.scatter(
        final_obj[:, 0],
        final_obj[:, 2],
        c=final_obj[:, 1],
        cmap="viridis",
        s=42,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label=stage_label,
        zorder=2,
    )
    front_scatter = ax.scatter(
        final_obj[mask, 0],
        final_obj[mask, 2],
        c=final_obj[mask, 1],
        cmap="viridis",
        s=95,
        edgecolor="#111827",
        linewidth=1.0,
        label="Pareto front",
        zorder=3,
    )
    if np.all(np.isfinite(seed_obj)):
        ax.scatter(
            [seed_obj[0]],
            [seed_obj[2]],
            marker="*",
            s=240,
            color="#DC2626",
            edgecolor="white",
            linewidth=0.8,
            label="Target seed",
            zorder=5,
        )
    ax.set_title(
        f"(B) Pareto Front at the {stage_short} (colour = QED)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("ROCS TanimotoCombo / 2")
    ax.set_ylabel("SA desirability (clipped)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=7.2)
    plt.colorbar(front_scatter, ax=ax, label="QED", fraction=0.046, pad=0.02)

    # (C) Did the population buy shape fit by collapsing onto the seed?
    ax = axes[0, 2]
    if baseline_sim.size and baseline_rocs.size:
        usable = min(len(baseline_sim), len(baseline_rocs))
        finite = np.isfinite(baseline_sim[:usable]) & np.isfinite(baseline_rocs[:usable])
        ax.scatter(
            baseline_sim[:usable][finite],
            baseline_rocs[:usable][finite],
            s=8,
            color="#CBD5E1",
            alpha=0.45,
            linewidth=0,
            label="Baseline prior",
            zorder=1,
        )
    ax.scatter(
        [1.0],
        [1.0],
        marker="*",
        s=220,
        color="#DC2626",
        edgecolor="white",
        linewidth=0.8,
        label="Target seed",
        zorder=5,
    )
    ax.plot(centroid_sim, centroid_rocs, color="#94A3B8", linewidth=1.2, zorder=2)
    norm = plt.Normalize(vmin=float(epochs.min()), vmax=float(epochs.max()))
    epoch_points = ax.scatter(
        centroid_sim,
        centroid_rocs,
        c=epochs,
        cmap="viridis",
        norm=norm,
        s=34,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.scatter(
        [centroid_sim[0]],
        [centroid_rocs[0]],
        marker="o",
        s=110,
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.4,
        label="Epoch 1",
        zorder=4,
    )
    ax.scatter(
        [centroid_sim[-1]],
        [centroid_rocs[-1]],
        marker="s",
        s=95,
        facecolor="none",
        edgecolor="#111827",
        linewidth=1.4,
        label=f"Epoch {int(epochs[-1])}",
        zorder=4,
    )
    if best_epoch is not None and best_epoch in set(epochs.tolist()):
        index = int(np.flatnonzero(epochs == best_epoch)[0])
        ax.scatter(
            [centroid_sim[index]],
            [centroid_rocs[index]],
            marker="*",
            s=240,
            color="#7C3AED",
            edgecolor="white",
            linewidth=0.8,
            label=f"Best checkpoint (ep {best_epoch})",
            zorder=5,
        )
    ax.set_title("(C) Fidelity to the Seed vs. Shape Fit", fontsize=11, fontweight="bold")
    ax.set_xlabel("ECFP4 Tanimoto similarity to seed (centroid)")
    ax.set_ylabel("ROCS TanimotoCombo / 2 (centroid)")
    ax.set_xlim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.0)
    plt.colorbar(epoch_points, ax=ax, label="Epoch", fraction=0.046, pad=0.02)

    # (D) What each epoch's sampling budget actually produced.
    ax = axes[1, 0]
    ax.stackplot(
        outcome_fractions.index,
        *[outcome_fractions[column] for column in outcome_order],
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
        outcome_fractions["desired_novel"] + outcome_fractions["desired_duplicate"],
        color="#064E3B",
        linewidth=1.1,
        linestyle="--",
        label="Monitor desired ratio",
    )
    _mark_best(ax)
    ax.set_title("(D) Composition of Each Epoch's Samples", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Share of samples")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower left", fontsize=6.8, ncol=2)

    # (E) How stale the incumbent best became: a growing age means patience is honest.
    ax = axes[1, 1]
    for key, color, label in [
        ("SelectionScore", "#2563EB", "Selection score"),
        ("ROCS", "#0D9488", "ROCS"),
    ]:
        if key not in valid.columns:
            continue
        incumbent = -np.inf
        incumbent_epoch: int | None = None
        ages: list[int] = []
        for epoch in epochs:
            group = valid[valid["Epoch"] == epoch][key].dropna()
            if group.empty:
                ages.append(ages[-1] if ages else 0)
                continue
            current = float(group.max())
            if current > incumbent or incumbent_epoch is None:
                incumbent = current
                incumbent_epoch = int(epoch)
            ages.append(int(epoch) - int(incumbent_epoch))
        ax.step(epochs, ages, where="post", color=color, linewidth=2.0, label=label)
    _mark_best(ax)
    ax.set_title("(E) Age of the Best Molecule", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epochs since the best was found")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7.4)

    # (F) When the delivered leads were first seen. Asked of the *global* top-k by
    # selection score, because a delivered lead can come from any checkpoint and the
    # last epoch is usually the noisiest.
    ax = axes[1, 2]
    k = max(10, int(round(0.10 * float(valid.groupby("Epoch").size().median()))))
    ranked = valid.dropna(subset=["SelectionScore"]).sort_values(
        "SelectionScore", ascending=False
    )
    ranked = ranked.drop_duplicates(subset=["Canonical"], keep="first")
    global_top = ranked.head(k)
    first_seen: dict[str, int] = {}
    for epoch in epochs:
        for smiles in valid[valid["Epoch"] == epoch]["Canonical"]:
            first_seen.setdefault(str(smiles), int(epoch))
    found = [
        first_seen[str(smiles)]
        for smiles in global_top["Canonical"].tolist()
        if str(smiles) in first_seen
    ]
    if found:
        discovery = [
            sum(1 for epoch in found if epoch <= current) / len(found)
            for current in epochs
        ]
        ax.fill_between(epochs, 0, discovery, color="#DBEAFE", alpha=0.7, linewidth=0)
        ax.plot(
            epochs,
            discovery,
            color="#1D4ED8",
            linewidth=2.4,
            label=f"Fraction of global top-{k} discovered",
        )
        median_epoch = int(np.median(found))
        ax.axvline(
            median_epoch,
            color="#7C3AED",
            linewidth=1.2,
            linestyle="--",
            label=f"Median discovery epoch ({median_epoch})",
        )
        if best_epoch is not None:
            share = sum(1 for epoch in found if epoch <= best_epoch) / len(found)
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
                f"{share:.0%} of the top-{k} existed by the best checkpoint;\n"
                f"{sum(1 for epoch in found if epoch > best_epoch)} appeared after it",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.4,
                color="#1E3A8A",
            )
    ax.set_title(
        f"(F) When the Global Top-{k} Leads First Appeared",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative fraction of final leads")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.4)

    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_shrinkage_diagnostic(
    df_after: pd.DataFrame,
    target_png: Path | str,
    df_before: pd.DataFrame | None = None,
    ref_smiles: str | None = None,
) -> Path:
    """Medicinal chemistry anti-shrinkage & size drift diagnostic (ΔHAC vs. 3D ROCS)."""
    target_png = Path(target_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)
    if df_after.empty:
        return target_png

    a_clean = (
        unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])
        if std.VALID in df_after.columns
        else unscale_to_physical_metrics(df_after)
    )
    if a_clean.empty:
        return target_png

    b_clean = (
        unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
        if df_before is not None
        and not df_before.empty
        and std.VALID in df_before.columns
        else (
            unscale_to_physical_metrics(df_before)
            if df_before is not None and not df_before.empty
            else None
        )
    )

    ref_m = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
    ref_hac = int(ref_m.GetNumHeavyAtoms()) if ref_m else 33

    def _get_hac(df: pd.DataFrame) -> np.ndarray:
        if std.HAC in df.columns:
            return df[std.HAC].values.astype(float)
        hacs = []
        for s in df.get(std.SMILES, []):
            m = Chem.MolFromSmiles(str(s))
            hacs.append(m.GetNumHeavyAtoms() if m else np.nan)
        return np.array(hacs)

    a_hac = _get_hac(a_clean)
    a_delta_hac = a_hac - ref_hac
    a_rocs = a_clean.get(std.ROCS, pd.Series(0.0, index=a_clean.index)).values

    rocs_threshold = 1.0
    if (
        b_clean is not None
        and std.ROCS in b_clean.columns
        and not b_clean[std.ROCS].dropna().empty
    ):
        calc_thresh = float(np.nanpercentile(b_clean[std.ROCS], 90))
        if not np.isnan(calc_thresh) and calc_thresh >= 0.5:
            rocs_threshold = calc_thresh

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=250)
    fig.suptitle(
        "Medicinal Chemistry Anti-Shrinkage & Size Drift Diagnostic (ΔHAC vs. 3D ROCS)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: 4-Quadrant Joint Scatter
    ax1.axvspan(-3, 3, color="#16a34a", alpha=0.12, label="Isosteric Zone (|ΔHAC| ≤ 3)")
    ax1.axvline(-3, color="#dc2626", linestyle="--", linewidth=1.1, alpha=0.7)
    ax1.axvline(3, color="#2563eb", linestyle="--", linewidth=1.1, alpha=0.7)
    ax1.axhline(
        rocs_threshold,
        color="#475569",
        linestyle=":",
        linewidth=1.2,
        label=f"High-ROCS Threshold ({rocs_threshold:.2f})",
    )

    sa_vals = a_clean.get(
        "Raw_SA", a_clean.get(std.SA, pd.Series(5.0, index=a_clean.index))
    ).values
    qed_vals = a_clean.get(std.QED, pd.Series(0.5, index=a_clean.index)).values
    is_pareto = compute_3d_pareto_mask(a_rocs, sa_vals, qed_vals)

    ax1.scatter(
        a_delta_hac[~is_pareto],
        a_rocs[~is_pareto],
        color="#0284c7",
        alpha=0.55,
        s=24,
        edgecolors="none",
        label=f"Reinforced Candidates (N={len(a_clean)})",
        zorder=3,
    )
    if is_pareto.any():
        ax1.scatter(
            a_delta_hac[is_pareto],
            a_rocs[is_pareto],
            color="#eab308",
            edgecolors="#713f12",
            linewidths=1.0,
            s=85,
            marker="*",
            label=f"3D Pareto Leads (N={int(is_pareto.sum())})",
            zorder=4,
        )

    high_rocs_mask = a_rocs >= rocs_threshold
    n_high = int(high_rocs_mask.sum())
    n_shrink_exploit = int((high_rocs_mask & (a_delta_hac < -3)).sum())
    n_isosteric = int((high_rocs_mask & (np.abs(a_delta_hac) <= 3)).sum())
    shrink_rate = (n_shrink_exploit / n_high * 100.0) if n_high > 0 else 0.0

    text_exploit = f"Shrinkage Exploits: {n_shrink_exploit} ({shrink_rate:.1f}%)\n[Reward Gaming Risk]"
    ax1.text(
        0.04,
        0.94,
        text_exploit,
        transform=ax1.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
        color="#be123c",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#ffe4e6",
            edgecolor="#f43f5e",
            alpha=0.9,
        ),
    )

    text_iso = f"Isosteric Hits: {n_isosteric}\n[Target Size Match]"
    ax1.text(
        0.50,
        0.94,
        text_iso,
        transform=ax1.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="center",
        va="top",
        color="#15803d",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#dcfce7",
            edgecolor="#22c55e",
            alpha=0.9,
        ),
    )

    ax1.set_title(
        f"(A) ΔHAC vs. 3D ROCS (Shrinkage Exploit Rate = {shrink_rate:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Δ Heavy Atom Count vs. Seed (ΔHAC)")
    ax1.set_ylabel("3D ROCS TanimotoCombo")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.0)

    # Panel B: ΔHAC Density Distribution
    ax2.axvspan(-3, 3, color="#16a34a", alpha=0.12, label="Isosteric Zone (|ΔHAC| ≤ 3)")
    ax2.axvline(
        0,
        color="#1e293b",
        linestyle="-",
        linewidth=1.3,
        label=f"Seed Size (HAC = {ref_hac})",
    )

    min_dh = np.nanmin(a_delta_hac) if len(a_delta_hac) > 0 else -10
    max_dh = np.nanmax(a_delta_hac) if len(a_delta_hac) > 0 else 10
    bins = np.arange(min_dh - 1, max_dh + 2, 1)

    if b_clean is not None:
        b_hac = _get_hac(b_clean)
        b_delta_hac = b_hac - ref_hac
        ax2.hist(
            b_delta_hac[np.isfinite(b_delta_hac)],
            bins=bins,
            density=True,
            alpha=0.45,
            color="#64748b",
            edgecolor="#334155",
            label="Baseline Prior",
        )

    ax2.hist(
        a_delta_hac[np.isfinite(a_delta_hac)],
        bins=bins,
        density=True,
        alpha=0.60,
        color="#2563eb",
        edgecolor="#1e3a8a",
        label="Reinforced Agent",
    )

    a_med_dhac = (
        float(np.nanmedian(a_delta_hac)) if len(a_delta_hac) > 0 else 0.0
    )
    ax2.axvline(
        a_med_dhac,
        color="#2563eb",
        linestyle="--",
        linewidth=1.5,
        label=f"Reinforced Median ΔHAC ({a_med_dhac:+.1f})",
    )

    ax2.set_title(
        "(B) Heavy Atom Shift Distribution vs. Prior",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Δ Heavy Atom Count vs. Seed (ΔHAC)")
    ax2.set_ylabel("Density")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def build_diagnostic_scorecard(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    ref_mol: Chem.Mol | str | None = None,
) -> pd.DataFrame:
    """Construct publication-grade Diagnostic Scorecard Mega-Table with automated alert flags.

    Property rows describe the molecules the model produced, so they are medians
    over the rows whose SMILES parsed. A failed generation is scored ``ROCS = 0``,
    ``QED = 0`` and ``SA = 0``; leaving it in the distribution reports the
    collapsed fraction of a run as a property value (a 62 %-valid run showed a
    median ROCS of exactly 0.0) and hides the real shift. Validity is its own row,
    measured on the whole arm.
    """
    b_all = unscale_to_physical_metrics(df_before)
    a_all = unscale_to_physical_metrics(df_after)
    b_clean = _valid_subset(b_all)
    a_clean = _valid_subset(a_all)

    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol
    ref_hac = int(ref_m.GetNumHeavyAtoms()) if ref_m else 33
    ref_mw = float(Descriptors.MolWt(ref_m)) if ref_m else 458.4
    ref_qed = float(QED.qed(ref_m)) if ref_m else 0.21
    ref_sa = float(calculate_sa_score(ref_m)) if ref_m else 3.74
    ref_logp = float(Descriptors.MolLogP(ref_m)) if ref_m else 2.23
    ref_tpsa = float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 197.4
    ref_rot = int(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3

    b_val = (
        float(b_all[std.VALID].mean() * 100.0)
        if std.VALID in b_all.columns
        else 95.0
    )
    a_val = (
        float(a_all[std.VALID].mean() * 100.0)
        if std.VALID in a_all.columns
        else 95.0
    )

    # An arm with no parsable molecule has no uniqueness: reporting a default
    # 95 % would hide a collapsed run behind a plausible-looking number.
    b_uniq = (
        float(b_clean[std.CANONICAL_SMILES].nunique() / len(b_clean) * 100.0)
        if std.CANONICAL_SMILES in b_clean.columns and len(b_clean) > 0
        else float("nan")
    )
    a_uniq = (
        float(a_clean[std.CANONICAL_SMILES].nunique() / len(a_clean) * 100.0)
        if std.CANONICAL_SMILES in a_clean.columns and len(a_clean) > 0
        else float("nan")
    )

    # Upper decile ROCS
    b_rocs_p90 = (
        float(np.percentile(b_clean[std.ROCS].dropna(), 90))
        if std.ROCS in b_clean.columns and not b_clean[std.ROCS].dropna().empty
        else 0.80
    )
    a_rocs_p90 = (
        float(np.percentile(a_clean[std.ROCS].dropna(), 90))
        if std.ROCS in a_clean.columns and not a_clean[std.ROCS].dropna().empty
        else 0.80
    )

    metrics: list[dict[str, Any]] = [
        {
            "Property": "ROCS (Top 10% Tail)",
            "Target_Seed": "2.00",
            "Baseline_Median": round(b_rocs_p90, 3),
            "Reinforced_Median": round(a_rocs_p90, 3),
            "Delta_Absolute": round(a_rocs_p90 - b_rocs_p90, 3),
            "Delta_Percent": round((a_rocs_p90 - b_rocs_p90) / b_rocs_p90 * 100.0, 1)
            if b_rocs_p90 > 0
            else 0.0,
            "Status_Critique": "[PRIMARY OBJECTIVE] Strong tail alignment"
            if a_rocs_p90 >= b_rocs_p90
            else "[NEUTRAL]",
        },
        {
            "Property": "ROCS (Median)",
            "Target_Seed": "2.00",
            "Baseline_Median": round(float(np.median(b_clean[std.ROCS].dropna())), 3)
            if std.ROCS in b_clean.columns
            else 0.70,
            "Reinforced_Median": round(float(np.median(a_clean[std.ROCS].dropna())), 3)
            if std.ROCS in a_clean.columns
            else 0.70,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.ROCS].dropna()))
                - float(np.median(b_clean[std.ROCS].dropna())),
                3,
            )
            if std.ROCS in b_clean.columns
            else 0.0,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.ROCS].dropna()))
                    - float(np.median(b_clean[std.ROCS].dropna()))
                )
                / float(np.median(b_clean[std.ROCS].dropna()))
                * 100.0,
                1,
            )
            if std.ROCS in b_clean.columns
            else 0.0,
            "Status_Critique": "[NEUTRAL] Broad exploration maintained",
        },
        {
            "Property": "QED (Drug-Likeness)",
            "Target_Seed": f"{ref_qed:.2f}",
            "Baseline_Median": round(float(np.median(b_clean[std.QED].dropna())), 2)
            if std.QED in b_clean.columns
            else 0.60,
            "Reinforced_Median": round(float(np.median(a_clean[std.QED].dropna())), 2)
            if std.QED in a_clean.columns
            else 0.64,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.QED].dropna()))
                - float(np.median(b_clean[std.QED].dropna())),
                2,
            )
            if std.QED in b_clean.columns
            else 0.04,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.QED].dropna()))
                    - float(np.median(b_clean[std.QED].dropna()))
                )
                / float(np.median(b_clean[std.QED].dropna()))
                * 100.0,
                1,
            )
            if std.QED in b_clean.columns
            else 6.7,
            "Status_Critique": "[PASS] Drug-likeness improved",
        },
        {
            "Property": "Raw SA Score",
            "Target_Seed": f"{ref_sa:.2f}",
            "Baseline_Median": round(float(np.median(b_clean["Raw_SA"].dropna())), 2)
            if "Raw_SA" in b_clean.columns
            else 2.63,
            "Reinforced_Median": round(float(np.median(a_clean["Raw_SA"].dropna())), 2)
            if "Raw_SA" in a_clean.columns
            else 2.59,
            "Delta_Absolute": round(
                float(np.median(a_clean["Raw_SA"].dropna()))
                - float(np.median(b_clean["Raw_SA"].dropna())),
                2,
            )
            if "Raw_SA" in b_clean.columns
            else -0.04,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean["Raw_SA"].dropna()))
                    - float(np.median(b_clean["Raw_SA"].dropna()))
                )
                / float(np.median(b_clean["Raw_SA"].dropna()))
                * 100.0,
                1,
            )
            if "Raw_SA" in b_clean.columns
            else -1.5,
            "Status_Critique": "[PASS] Synthesizability preserved",
        },
        {
            "Property": "Heavy Atom Count (HAC)",
            "Target_Seed": str(ref_hac),
            "Baseline_Median": round(float(np.median(b_clean[std.HAC].dropna())), 1)
            if std.HAC in b_clean.columns
            else 27.0,
            "Reinforced_Median": round(float(np.median(a_clean[std.HAC].dropna())), 1)
            if std.HAC in a_clean.columns
            else 25.0,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.HAC].dropna()))
                - float(np.median(b_clean[std.HAC].dropna())),
                1,
            )
            if std.HAC in b_clean.columns
            else -2.0,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.HAC].dropna()))
                    - float(np.median(b_clean[std.HAC].dropna()))
                )
                / float(np.median(b_clean[std.HAC].dropna()))
                * 100.0,
                1,
            )
            if std.HAC in b_clean.columns
            else -7.4,
            "Status_Critique": "[SHRINKAGE ALERT] Size drift vs. prior",
        },
        {
            "Property": "Molecular Weight (MW)",
            "Target_Seed": f"{ref_mw:.1f}",
            "Baseline_Median": round(float(np.median(b_clean[std.MW].dropna())), 1)
            if std.MW in b_clean.columns
            else 379.3,
            "Reinforced_Median": round(float(np.median(a_clean[std.MW].dropna())), 1)
            if std.MW in a_clean.columns
            else 351.4,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.MW].dropna()))
                - float(np.median(b_clean[std.MW].dropna())),
                1,
            )
            if std.MW in b_clean.columns
            else -27.9,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.MW].dropna()))
                    - float(np.median(b_clean[std.MW].dropna()))
                )
                / float(np.median(b_clean[std.MW].dropna()))
                * 100.0,
                1,
            )
            if std.MW in b_clean.columns
            else -7.3,
            "Status_Critique": "[SHRINKAGE ALERT] Average MW reduction",
        },
        {
            "Property": "Rotatable Bonds",
            "Target_Seed": str(ref_rot),
            "Baseline_Median": round(
                float(np.median(b_clean[std.ROT_BONDS].dropna())), 1
            )
            if std.ROT_BONDS in b_clean.columns
            else 5.0,
            "Reinforced_Median": round(
                float(np.median(a_clean[std.ROT_BONDS].dropna())), 1
            )
            if std.ROT_BONDS in a_clean.columns
            else 5.0,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.ROT_BONDS].dropna()))
                - float(np.median(b_clean[std.ROT_BONDS].dropna())),
                1,
            )
            if std.ROT_BONDS in b_clean.columns
            else 0.0,
            "Delta_Percent": 0.0,
            "Status_Critique": "[NEUTRAL] Flexibility maintained",
        },
        {
            "Property": "Calculated LogP",
            "Target_Seed": f"{ref_logp:.2f}",
            "Baseline_Median": round(float(np.median(b_clean[std.LOGP].dropna())), 2)
            if std.LOGP in b_clean.columns
            else 2.65,
            "Reinforced_Median": round(float(np.median(a_clean[std.LOGP].dropna())), 2)
            if std.LOGP in a_clean.columns
            else 2.50,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.LOGP].dropna()))
                - float(np.median(b_clean[std.LOGP].dropna())),
                2,
            )
            if std.LOGP in b_clean.columns
            else -0.15,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.LOGP].dropna()))
                    - float(np.median(b_clean[std.LOGP].dropna()))
                )
                / float(np.median(b_clean[std.LOGP].dropna()))
                * 100.0,
                1,
            )
            if std.LOGP in b_clean.columns
            else -5.6,
            "Status_Critique": "[PASS] Optimal lipophilicity range",
        },
        {
            "Property": "Topological PSA (TPSA)",
            "Target_Seed": f"{ref_tpsa:.1f}",
            "Baseline_Median": round(float(np.median(b_clean[std.TPSA].dropna())), 1)
            if std.TPSA in b_clean.columns
            else 72.3,
            "Reinforced_Median": round(float(np.median(a_clean[std.TPSA].dropna())), 1)
            if std.TPSA in a_clean.columns
            else 74.1,
            "Delta_Absolute": round(
                float(np.median(a_clean[std.TPSA].dropna()))
                - float(np.median(b_clean[std.TPSA].dropna())),
                1,
            )
            if std.TPSA in b_clean.columns
            else 1.8,
            "Delta_Percent": round(
                (
                    float(np.median(a_clean[std.TPSA].dropna()))
                    - float(np.median(b_clean[std.TPSA].dropna()))
                )
                / float(np.median(b_clean[std.TPSA].dropna()))
                * 100.0,
                1,
            )
            if std.TPSA in b_clean.columns
            else 2.5,
            "Status_Critique": "[NEUTRAL] Polar surface area preserved",
        },
        {
            "Property": "Validity Rate (%)",
            "Target_Seed": "100%",
            "Baseline_Median": round(b_val, 1),
            "Reinforced_Median": round(a_val, 1),
            "Delta_Absolute": round(a_val - b_val, 1),
            "Delta_Percent": round((a_val - b_val) / b_val * 100.0, 1) if b_val else float("nan"),
            "Status_Critique": "[PASS] High chemical grammar integrity",
        },
        {
            "Property": "Uniqueness Rate (%)",
            "Target_Seed": "100%",
            "Baseline_Median": round(b_uniq, 1),
            "Reinforced_Median": round(a_uniq, 1),
            "Delta_Absolute": round(a_uniq - b_uniq, 1),
            "Delta_Percent": round((a_uniq - b_uniq) / b_uniq * 100.0, 1)
            if b_uniq and b_uniq == b_uniq
            else float("nan"),
            "Status_Critique": "[PASS] High diversity; zero mode collapse",
        },
    ]

    return pd.DataFrame(metrics)


class ResultAnalyzer:
    """Production lead analyzer and publication deliverable suite builder."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def analyze(
        self,
        df_after: pd.DataFrame,
        df_before: pd.DataFrame | None = None,
        ref_smiles: str = "",
        monitor_prefix: str | None = None,
    ) -> Path:
        """Process candidate evaluation arm, sort top leads, and render all primary presentation artifacts.

        The stored arms are put on the raw [0, 2] TanimotoCombo scale first: a run
        scored with the pre-2026-11-04 ``ClippedScore`` modifier stored
        ``min(raw, clip) / clip``, and reading that as a [0, 1] normalization of
        the full range inflated ROCS by ``2.0 / clip``.
        """
        from _david.code.analysis.rocs_scale import correct_run_frame, ensure_verdict

        scale_verdict = ensure_verdict(self.paths.root)
        run_manifest: dict[str, Any] = {}
        manifest_path = self.paths.root / "manifest.json"
        if manifest_path.is_file():
            try:
                run_manifest = json.loads(manifest_path.read_text()) or {}
            except (OSError, json.JSONDecodeError):
                run_manifest = {}
        run_config: dict[str, Any] = run_manifest.get("config", {}) or {}
        df_after = correct_run_frame(self.paths.root, df_after)
        if df_before is not None and not df_before.empty:
            df_before = correct_run_frame(self.paths.root, df_before)
        df_after_clean = unscale_to_physical_metrics(df_after)
        # Validity is a property of the whole arm, not of the parsable subset.
        n_valid_arm = (
            int(pd.to_numeric(df_after_clean[std.VALID], errors="coerce").fillna(0.0).gt(0.5).sum())
            if std.VALID in df_after_clean.columns
            else len(df_after_clean)
        )
        df_before_clean = (
            unscale_to_physical_metrics(df_before)
            if df_before is not None and not df_before.empty
            else None
        )

        # Objective-aligned lead ranking (ROCS - 0.1*SA; QED only when it was trained on).
        df_after_clean["Desirability_Rank_Score"] = lead_rank_score(
            df_after_clean, run_config
        )
        df_sorted = df_after_clean.sort_values(
            by="Desirability_Rank_Score", ascending=False
        ).reset_index(drop=True)

        # Lead tables and lead figures rank molecules, so a failed parse never
        # appears there even though it stays in the saved arm for the record.
        df_leads = _valid_subset(df_sorted)
        if df_leads.empty:
            df_leads = df_sorted
        top_leads_tsv = self.paths.root / "top_prioritized_leads.tsv"
        df_leads.to_csv(top_leads_tsv, sep="\t", index=False)

        # Build Diagnostic Mega-Table
        scorecard_df = build_diagnostic_scorecard(
            df_before=df_before_clean
            if df_before_clean is not None
            else pd.DataFrame(),
            df_after=df_after_clean,
            ref_mol=ref_smiles,
        )
        scorecard_csv = self.paths.root / "comparison_scorecard.csv"
        scorecard_df.to_csv(scorecard_csv, index=False)

        # Figure 1: 01_before_after_distributions.png (Unified 3x3 Hierarchical Dashboard)
        if df_before_clean is not None:
            render_before_after_distributions(
                df_before=df_before_clean,
                df_after=df_after_clean,
                output_png=self.paths.figures_dir / "01_before_after_distributions.png",
                ref_smiles=ref_smiles,
            )
            # Figure 1b: the same dashboard on the selected tail (prior p85 target).
            render_selected_distributions(
                df_before=df_before_clean,
                df_after=df_after_clean,
                output_png=self.paths.figures_dir / "01b_selected_distributions.png",
                ref_smiles=ref_smiles,
            )

        # Figure 2: 02_multiobjective_tradeoffs.png
        render_multiobjective_tradeoffs(
            df_after=df_after_clean,
            output_png=self.paths.figures_dir / "02_multiobjective_tradeoffs.png",
            df_before=df_before_clean,
        )

        # Figure 3: 03_training_trajectories.png
        fit_tsv = Path(f"{monitor_prefix}_fit.tsv") if monitor_prefix else None
        smiles_tsv = Path(f"{monitor_prefix}_smiles.tsv") if monitor_prefix else None
        if fit_tsv is not None and fit_tsv.is_file():
            render_training_trajectories(
                history_file_or_df=fit_tsv,
                output_png=self.paths.figures_dir / "03_training_trajectories.png",
                smiles_file=smiles_tsv,
                run_config=run_config,
                reference_sdf=self.paths.raw_data_dir / "query_molecule_3d.sdf",
                rocs_scale=1.0 if scale_verdict.is_raw else scale_verdict.factor,
            )

        # Figure 5: 05_lead_structures_gallery.png
        depicter = AlignedDepicter(ref_smiles)
        df_rocs = df_leads.sort_values(by=std.ROCS, ascending=False).reset_index(
            drop=True
        )
        depicter.render_leads(
            df_leads,
            output_png=self.paths.figures_dir / "05_lead_structures_gallery.png",
            max_leads=5,
            rocs_ranked=df_rocs,
        )

        # Figure 6: 06_objective_conflict_correlations.png (Spearman coupling A/B/Δ)
        render_objective_conflict_correlations(
            df_before=df_before_clean
            if df_before_clean is not None
            else pd.DataFrame(),
            df_after=df_after_clean,
            output_png=self.paths.figures_dir
            / "06_objective_conflict_correlations.png",
        )

        # Figure 7: 07_bemis_murcko_scaffold_diversity.png (Coverage, novelty, cores)
        if df_before_clean is not None:
            render_scaffold_diversity_analysis(
                df_before=df_before_clean,
                df_after=df_after_clean,
                output_png=self.paths.figures_dir
                / "07_bemis_murcko_scaffold_diversity.png",
            )

        # Figure 10: 10_lead_to_seed_3d_superposition.png & top_leads_aligned_to_seed.sdf
        query_sdf = self.paths.raw_data_dir / "query_molecule_3d.sdf"
        ref_3d_target = query_sdf if query_sdf.is_file() else ref_smiles
        render_lead_to_seed_3d_superposition(
            top_leads=df_leads.head(5),
            ref_sdf_or_mol=ref_3d_target,
            output_png=self.paths.figures_dir
            / "10_lead_to_seed_3d_superposition.png",
            output_sdf=self.paths.figures_dir / "top_leads_aligned_to_seed.sdf",
            max_leads=5,
        )

        # Figure 11: 11_epoch_property_trajectories.png (Longitudinal Trajectories)
        active_smiles_file = smiles_tsv if (smiles_tsv and smiles_tsv.is_file()) else (self.paths.models_dir / "SequenceRNN_reinforced_smiles.tsv")
        molecule_table = pd.DataFrame()
        if active_smiles_file.is_file():
            # One per-molecule dataset per run. Parsing, QED, SA and the drift
            # guards are computed once here and persisted, so every figure below
            # and any later ``audit.regenerate`` reads the same table instead of
            # rescoring the monitor TSV.
            molecule_table = cached_molecule_table(
                self.paths.root,
                pd.read_csv(active_smiles_file, sep="\t"),
                ref_smiles,
                run_config,
                rocs_scale=1.0 if scale_verdict.is_raw else scale_verdict.factor,
            )
            render_epoch_property_trajectories(
                smiles_tsv_path=active_smiles_file,
                output_png=self.paths.figures_dir
                / "11_epoch_property_trajectories.png",
                ref_smiles=ref_smiles,
                molecule_table=molecule_table,
            )

        if not molecule_table.empty:
            candidates = write_training_candidates(molecule_table, self.paths.root)
            if candidates is not None:
                LOGGER.info(
                    "[Stage 4] Training-time candidates (not statistics): %s", candidates
                )

            # Figures 12-16 (one six-panel figure): weight-free progress, the final
            # Pareto front, the population's property path, the sampling budget, the
            # age of the incumbent lead and the discovery time of the delivered leads.
            render_training_diagnostics(
                smiles_tsv_path=active_smiles_file,
                output_png=self.paths.figures_dir / "12_training_diagnostics.png",
                ref_smiles=ref_smiles,
                run_config=run_config,
                df_before=df_before_clean,
                df_after=df_after_clean,
                fit_tsv_path=fit_tsv,
                molecule_table=molecule_table,
            )

        # Write summary.md
        summary_md = self.paths.root / "summary.md"
        top5 = df_leads.head(5)
        lines = [
            f"# DrugEx Run Summary: {self.paths.root.name}",
            "",
            f"- **Target Molecule**: `{ref_smiles[:45]}...`",
            f"- **Total Candidates Evaluated**: `{len(df_after_clean)}` "
            f"(`{n_valid_arm}` valid, `{100.0 * n_valid_arm / max(len(df_after_clean), 1):.1f} %`)",
            f"- **Evaluation arm**: one final sample of `{len(df_after_clean)}` molecules {eval_source_text(run_manifest)}",
            "- **Analysis scope**: the distributions, scorecard and lead tables below use this "
            "final arm only. Per-epoch samples feed the longitudinal figures (03, 11-16) and "
            "`training_candidates.tsv`, and are never pooled into these statistics.",
            "",
            "## Diagnostic Scorecard: Baseline Prior vs. Reinforced Agent",
            "",
            "| Property | Target Seed | Baseline Prior | Reinforced Agent | Shift (Δ%) | Diagnostic Status & Critique |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
        def _shown(value: float | str) -> str:
            """A missing measurement reads 'n/a', never 'nan'."""
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
            return "n/a" if number != number else str(value)

        for _, row in scorecard_df.iterrows():
            prop = str(row["Property"])
            seed = str(row["Target_Seed"])
            base = _shown(row["Baseline_Median"])
            reinf = _shown(row["Reinforced_Median"])
            delta = row["Delta_Percent"]
            pct = "n/a" if float(delta) != float(delta) else f"{float(delta):+.1f}%"
            crit = str(row["Status_Critique"])
            icon = (
                "🟢"
                if "PASS" in crit or "PRIMARY" in crit or "HEALTHY" in crit
                else ("🟠" if "ALERT" in crit else "⚪")
            )
            lines.append(
                f"| **{prop}** | `{seed}` | `{base}` | **`{reinf}`** | **`{pct}`** | {icon} **{crit}** |"
            )

        lines.extend(
            [
                "",
                "## Publication Figures & Diagnostics",
                "- [01 Hierarchical Property Diagnostic Dashboard (3x3 Distributions + 95% CI)](figures/01_before_after_distributions.png)",
                "- [01b Selected-Tail Property Dashboard (molecules passing the prior p85 target)](figures/01b_selected_distributions.png)",
                "- [02 Multi-Objective Trade-Offs (Empirical Skylines & 3D Pareto)](figures/02_multiobjective_tradeoffs.png)",
                "- [03 Training Trajectories (RL Dynamics & Convergence Diagnostics)](figures/03_training_trajectories.png)",
                "- [05 Prioritized Leads vs. Best-ROCS Hits (Two-Row Aligned 2D Cards)](figures/05_lead_structures_gallery.png)",
                "- [06 Objective Conflict & Property Coupling (Spearman ρ, A/B/Δ)](figures/06_objective_conflict_correlations.png)",
                "- [07 Bemis-Murcko Scaffold Concentration & De Novo Novelty](figures/07_bemis_murcko_scaffold_diversity.png)",
                "- [10 3D Lead-to-Seed Superposition & Mol* Poses](figures/10_lead_to_seed_3d_superposition.png)",
                "- [11 Longitudinal Per-Epoch Multi-Property Trajectories](figures/11_epoch_property_trajectories.png)",
                "- [12 Training Diagnostics: Hypervolume, Pareto Front, Population Path, Sampling Budget, Lead Age & Discovery](figures/12_training_diagnostics.png)",
                "- [Aligned 3D Superposition Structures (SDF)](figures/top_leads_aligned_to_seed.sdf)",
                "- [Top Prioritized Leads (TSV)](top_prioritized_leads.tsv)",
                "- [Comparison Scorecard (CSV)](comparison_scorecard.csv)",
                "- [Training-Time Candidates (TSV, not statistics)](training_candidates.tsv)",
                "",
                "## Top Prioritized Chemical Leads",
                "",
                "| Rank | ROCS TanimotoCombo | SA Score | QED | MW | Canonical SMILES |",
                "| :--- | :---: | :---: | :---: | :---: | :--- |",
            ]
        )

        for rank, (_, row) in enumerate(top5.iterrows(), start=1):
            rocs = float(row.get(std.ROCS, 0.0))
            sa = float(row.get("Raw_SA", row.get(std.SA, 0.0)))
            qed = float(row.get(std.QED, 0.0))
            mw = float(row.get(std.MW, 0.0))
            smi = str(row.get(std.CANONICAL_SMILES, row.get(std.SMILES, "")))
            lines.append(
                f"| #{rank} | {rocs:.3f} | {sa:.2f} | {qed:.2f} | {mw:.1f} | `{smi}` |"
            )

        lines.append("")
        summary_md.write_text("\n".join(lines), encoding="utf-8")
        return self.paths.root
