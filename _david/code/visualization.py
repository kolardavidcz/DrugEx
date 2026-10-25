from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, Lipinski, QED, rdDepictor, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import gaussian_kde

from _david.code.config import RunPaths
from _david.code.schema import std


def unscale_to_physical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ROCS is [0, 2] TanimotoCombo and calculate raw physical properties if missing."""
    if df.empty:
        return df.copy()

    df_out = df.copy()
    if std.ROCS in df_out.columns:
        vals = df_out[std.ROCS].dropna()
        if not vals.empty and vals.max() <= 1.0001:
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
        if std.QED not in df_out.columns:
            df_out[std.QED] = [QED.qed(m) if m else np.nan for m in mols]
        if std.ROT_BONDS not in df_out.columns:
            df_out[std.ROT_BONDS] = [
                Lipinski.NumRotatableBonds(m) if m else 0 for m in mols
            ]
        if "Raw_SA" not in df_out.columns:
            df_out["Raw_SA"] = [
                float(calculate_sa_score(m)) if m is not None else np.nan for m in mols
            ]

    return df_out


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap median difference b - a and (1 - ci) confidence interval."""
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


def get_universal_rigid_anchor(mol: Chem.Mol) -> list[int]:
    """Identify indices of the largest rigid ring block for visual 3D coordinate alignment."""
    rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(rot_smarts)
    rot_bonds = [
        b.GetIdx()
        for a1, a2 in matches
        if (b := mol.GetBondBetweenAtoms(a1, a2)) is not None and not b.IsInRing()
    ]
    if not rot_bonds:
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        return heavy[: min(10, len(heavy))]

    frag_mol = Chem.FragmentOnBonds(mol, rot_bonds, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    heavy_frags = [
        [idx for idx in f if mol.GetAtomWithIdx(idx).GetSymbol() != "H"] for f in frags
    ]
    largest_frag = max(heavy_frags, key=len, default=[])
    return sorted(largest_frag)


def render_reference_ensemble(
    sdf_path: Path | str,
    output_png: Path | str | None = None,
) -> Path:
    """Render 3D reference conformer ensemble superposition figure using pre-computed conformer properties."""
    from rdkit.Chem import AllChem

    sdf_file = Path(sdf_path)
    target_png = (
        Path(output_png)
        if output_png
        else sdf_file.parent / "reference_3d_superposition.png"
    )
    target_png.parent.mkdir(parents=True, exist_ok=True)

    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return target_png

    mol_combined = Chem.Mol(mols[0])
    mol_combined.RemoveAllConformers()
    for m in mols:
        mol_combined.AddConformer(m.GetConformer(), assignId=True)

    de_vals: list[float] = []
    for m in mols:
        if m.HasProp("DeltaE_kcal_mol"):
            de_vals.append(m.GetDoubleProp("DeltaE_kcal_mol"))
        elif m.HasProp("Energy_MMFF_kcal_mol"):
            de_vals.append(m.GetDoubleProp("Energy_MMFF_kcal_mol"))
        else:
            de_vals.append(0.0)
    if not any(m.HasProp("DeltaE_kcal_mol") for m in mols):
        min_e = min(de_vals) if de_vals else 0.0
        de_vals = [e - min_e for e in de_vals]

    min_idx = int(np.argmin(de_vals))
    max_idx = int(np.argmax(de_vals))

    anchor_atoms = get_universal_rigid_anchor(mol_combined)
    try:
        AllChem.AlignMolConformers(mol_combined, atomIds=anchor_atoms, refId=min_idx)
    except Exception:
        AllChem.AlignMolConformers(mol_combined)

    for i, m in enumerate(mols):
        m.RemoveAllConformers()
        m.AddConformer(mol_combined.GetConformer(i))

    core_atoms = set(anchor_atoms)
    fig = plt.figure(figsize=(18, 5.8), dpi=200)

    def _draw_mol(
        ax: Any,
        mol: Chem.Mol,
        bond_c: Any,
        atom_c: Any,
        alpha: float = 1.0,
        lw: float = 2.0,
        pt_size: float = 30,
    ) -> None:
        conf = mol.GetConformer()
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetSymbol() == "H" or a2.GetSymbol() == "H":
                continue
            p1, p2 = (
                conf.GetAtomPosition(a1.GetIdx()),
                conf.GetAtomPosition(a2.GetIdx()),
            )
            is_core = (a1.GetIdx() in core_atoms) and (a2.GetIdx() in core_atoms)
            c = "#1e293b" if is_core else bond_c
            w = lw * 1.3 if is_core else lw
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                [p1.z, p2.z],
                color=c,
                alpha=alpha,
                linewidth=w,
            )
        heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        for a in heavy:
            p = conf.GetAtomPosition(a.GetIdx())
            is_core = a.GetIdx() in core_atoms
            c = "#0f172a" if is_core else atom_c
            s = pt_size * 1.4 if is_core else pt_size
            ax.scatter(
                [p.x], [p.y], [p.z], color=c, s=s, alpha=alpha, edgecolors="none"
            )

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_mol(ax1, mols[min_idx], "#0284c7", "#0284c7", lw=2.4, pt_size=40)
    ax1.set_title(
        f"(A) Ground State Basin (ΔE = {de_vals[min_idx]:.2f} kcal/mol)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.view_init(elev=20, azim=40)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_mol(ax2, mols[max_idx], "#f59e0b", "#f59e0b", lw=2.4, pt_size=40)
    ax2.set_title(
        f"(B) Elevated Basin (ΔE = {de_vals[max_idx]:.2f} kcal/mol)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax2.view_init(elev=20, azim=40)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(de_vals) if max(de_vals) > 0 else 1.0)
    for i, m in enumerate(mols):
        color = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, color, color, alpha=0.75, lw=1.5, pt_size=20)
    ax3.set_title(
        f"(C) Ensemble Superposition ({len(mols)} Basins)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax3.view_init(elev=20, azim=40)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def _plot_kde(
    ax: Any,
    vals: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.30,
    linewidth: float = 1.8,
) -> None:
    """Plot smoothed KDE density curve using scipy.stats.gaussian_kde with histogram fallback."""
    if len(vals) < 2:
        return
    try:
        kde = gaussian_kde(vals)
        xs = np.linspace(float(vals.min()), float(vals.max()), 200)
        ys = kde(xs)
        ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)
        ax.fill_between(xs, ys, color=color, alpha=alpha)
    except Exception:
        ax.hist(vals, bins=20, color=color, alpha=alpha, density=True, label=label)


def render_before_after_distributions(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
    ref_mol: Chem.Mol | str | None = None,
) -> Path:
    """Render publication-grade 2x3 KDE distribution panel comparing baseline vs. reinforced molecules."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    b_df = unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
    a_df = unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])

    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol
    ref_props: dict[str, float | None] = {
        std.ROCS: 2.0 if ref_m else None,
        "Raw_SA": float(calculate_sa_score(ref_m)) if ref_m else 3.74,
        std.QED: float(QED.qed(ref_m)) if ref_m else 0.21,
        std.MW: float(Descriptors.MolWt(ref_m)) if ref_m else 458.4,
        std.HAC: float(ref_m.GetNumHeavyAtoms()) if ref_m else 33.0,
        std.ROT_BONDS: float(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3.0,
    }

    panels = [
        (std.ROCS, "ROCS TanimotoCombo [0, 2]", (0.0, 1.5)),
        ("Raw_SA", "Synthetic Accessibility (SAscore [1, 10])", (1.0, 7.0)),
        (std.QED, "Drug-likeness (QED [0, 1])", (0.0, 1.0)),
        (std.MW, "Molecular Weight (MW [Da])", (100.0, 600.0)),
        (std.HAC, "Heavy Atom Count (HAC)", (10.0, 45.0)),
        (std.ROT_BONDS, "Rotatable Bonds Count", (0.0, 15.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=250)
    fig.suptitle(
        "De Novo SequenceRNN: Pre-training Baseline vs. Reinforced Distributions",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for idx, (col, title, xlim) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        b_vals = b_df[col].dropna().values
        a_vals = a_df[col].dropna().values

        if len(b_vals) > 0 and len(a_vals) > 0:
            _plot_kde(
                ax,
                b_vals,
                color="#3b82f6",
                label="Baseline (Papyrus prior)",
                alpha=0.30,
                linewidth=1.8,
            )
            _plot_kde(
                ax,
                a_vals,
                color="#ea580c",
                label="Reinforced (Active agent)",
                alpha=0.35,
                linewidth=2.0,
            )

            b_med = float(np.median(b_vals))
            a_med = float(np.median(a_vals))
            delta, ci_l, ci_h = bootstrap_ci(b_vals, a_vals)

            ax.axvline(
                b_med,
                color="#2563eb",
                linestyle="--",
                linewidth=1.2,
                label=f"Base Med: {b_med:.2f}",
            )
            ax.axvline(
                a_med,
                color="#c2410c",
                linestyle="-",
                linewidth=1.5,
                label=f"Reinf Med: {a_med:.2f}",
            )

            sign = "+" if delta >= 0 else ""
            ax.text(
                0.04,
                0.92,
                f"ΔMed: {sign}{delta:.2f}\n95% CI: [{ci_l:.2f}, {ci_h:.2f}]",
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#cbd5e1",
                    alpha=0.9,
                ),
                verticalalignment="top",
            )

        ref_v = ref_props.get(col)
        if ref_v is not None and np.isfinite(ref_v):
            ax.axvline(
                ref_v,
                color="#dc2626",
                linestyle=":",
                linewidth=1.5,
                label=f"Ref EGCG: {ref_v:.1f}",
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(xlim)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_multiobjective_tradeoffs(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_png: Path | str,
) -> Path:
    """Render 2-panel trade-off scatter plots: ROCS vs SA (colored by QED) and ROCS vs Heavy Atoms."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    b_df = unscale_to_physical_metrics(df_before[df_before[std.VALID] == 1])
    a_df = unscale_to_physical_metrics(df_after[df_after[std.VALID] == 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=250)
    fig.suptitle(
        "Multi-Objective Trade-Offs & Size Evolution",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: ROCS vs. SA (colored by QED)
    if not b_df.empty and not a_df.empty:
        ax1.scatter(
            b_df[std.ROCS],
            b_df["Raw_SA"],
            c="#94a3b8",
            alpha=0.25,
            s=18,
            label="Baseline",
            edgecolors="none",
        )
        sc = ax1.scatter(
            a_df[std.ROCS],
            a_df["Raw_SA"],
            c=a_df[std.QED],
            cmap="viridis",
            alpha=0.75,
            s=28,
            edgecolors="none",
        )
        cb = plt.colorbar(sc, ax=ax1, pad=0.02)
        cb.set_label("QED (Drug-likeness)", fontsize=9)
        ax1.set_xlabel(
            "3D ROCS TanimotoCombo (higher = better)", fontsize=10, fontweight="bold"
        )
        ax1.set_ylabel(
            "Raw SAscore (lower = easier synthesis)", fontsize=10, fontweight="bold"
        )
        ax1.set_title(
            "(A) Shape Alignment vs. Synthetic Accessibility",
            fontsize=11,
            fontweight="bold",
        )
        ax1.grid(True, linestyle=":", alpha=0.4)
        ax1.legend(loc="upper right", fontsize=8.5)

    # Panel 2: ROCS vs. Heavy Atom Count (identifying molecule shrinkage)
    if not b_df.empty and not a_df.empty:
        ax2.scatter(
            b_df[std.HAC],
            b_df[std.ROCS],
            color="#3b82f6",
            alpha=0.30,
            s=20,
            label="Baseline Prior",
        )
        ax2.scatter(
            a_df[std.HAC],
            a_df[std.ROCS],
            color="#ea580c",
            alpha=0.60,
            s=24,
            label="Reinforced Agent",
        )
        ax2.axvline(
            33,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label="EGCG Size (33 atoms)",
        )
        ax2.axhline(
            0.70,
            color="#10b981",
            linestyle=":",
            linewidth=1.5,
            label="ROCS Threshold (0.70)",
        )
        ax2.set_xlabel("Heavy Atom Count (HAC)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("3D ROCS TanimotoCombo", fontsize=10, fontweight="bold")
        ax2.set_title(
            "(B) Shape Matching vs. Molecular Size", fontsize=11, fontweight="bold"
        )
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_training_trajectories(
    history_file_or_df: Path | str | pd.DataFrame,
    output_png: Path | str,
) -> Path | None:
    """Render 4-panel training trajectory: loss, desired/valid ratio, objective scores, and epsilon."""
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), dpi=250)
    fig.suptitle(
        "SequenceRNN Reinforcement Learning Training Dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    epochs = df["Epoch"]

    # Panel 1: Loss & Mean Reward
    ax1 = axes[0, 0]
    if "loss_train" in df.columns:
        ax1.plot(
            epochs, df["loss_train"], color="#dc2626", linewidth=1.8, label="Train Loss"
        )
    if "avg_amean" in df.columns:
        ax1.plot(
            epochs,
            df["avg_amean"],
            color="#2563eb",
            linewidth=1.8,
            label="Avg Arithmetic Mean",
        )
    ax1.set_title(
        "(A) Training Loss & Mean Objective Score", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Epoch")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.5)

    # Panel 2: Desired & Valid Ratio
    ax2 = axes[0, 1]
    if "valid_ratio" in df.columns:
        ax2.plot(
            epochs,
            df["valid_ratio"],
            color="#10b981",
            linewidth=1.8,
            label="Valid Ratio",
        )
    if "desired_ratio" in df.columns:
        ax2.plot(
            epochs,
            df["desired_ratio"],
            color="#8b5cf6",
            linewidth=2.0,
            label="Desired Ratio",
        )
    if "unique_ratio" in df.columns:
        ax2.plot(
            epochs,
            df["unique_ratio"],
            color="#06b6d4",
            linewidth=1.5,
            linestyle="--",
            label="Unique Ratio",
        )
    ax2.set_title(
        "(B) Validity, Uniqueness & Desired Ratio", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="best", fontsize=8.5)

    # Panel 3: Individual Objective Dynamics
    ax3 = axes[1, 0]
    for col, color, lbl in [
        ("ROCS", "#2563eb", "ROCS"),
        ("SA", "#10b981", "SA Desirability"),
        ("QED", "#f59e0b", "QED"),
    ]:
        if col in df.columns:
            ax3.plot(epochs, df[col], color=color, linewidth=1.8, label=lbl)
    ax3.set_title(
        "(C) Multi-Objective Desirability Evolution", fontsize=11, fontweight="bold"
    )
    ax3.set_xlabel("Epoch")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="best", fontsize=8.5)

    # Panel 4: Exploration Noise (Epsilon) Schedule
    ax4 = axes[1, 1]
    if "epsilon" in df.columns:
        ax4.plot(
            epochs,
            df["epsilon"],
            color="#ea580c",
            linewidth=2.0,
            label="Epsilon (Noise)",
        )
    ax4.set_title("(D) Scheduled Exploration Rate (ε)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylim(0.0, 0.30)
    ax4.grid(True, linestyle=":", alpha=0.4)
    ax4.legend(loc="best", fontsize=8.5)

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
    """Render diagnostic panels highlighting flaws, shrinkage, and distribution shifts.

    Directly implements academic critique requirement:
    Visualizes negative / unpromising features (molecular shrinkage, repeat counts)
    to spot inaccuracies and limitations in the methods.
    """
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

    # Panel 1: Heavy Atom Shrinkage Distribution (Delta HAC relative to reference)
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
            "(A) Heavy Atom Size Drift (Δ vs. EGCG=33)", fontsize=11, fontweight="bold"
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

    # Panel 3: Duplication and Frequency Distribution (Soft Mode Collapse Diagnosis)
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


def render_lead_structures_gallery(
    top_leads: pd.DataFrame,
    ref_smiles: str,
    output_png: Path | str,
    top_k: int = 5,
) -> Path | None:
    """Render 2D grid image showing top prioritized leads aligned alongside the reference molecule."""
    target_png = Path(output_png)
    target_png.parent.mkdir(parents=True, exist_ok=True)

    if top_leads.empty:
        return None

    clean_leads = (
        top_leads.drop_duplicates(subset=[std.CANONICAL_SMILES]).head(top_k).copy()
    )
    ref_m = Chem.MolFromSmiles(ref_smiles)

    mols: list[Chem.Mol] = [ref_m] if ref_m else []
    legends: list[str] = ["EGCG (Reference)\nCID 65064 | 33 atoms"] if ref_m else []

    rocs_col = std.ROCS if std.ROCS in clean_leads.columns else clean_leads.columns[0]
    for _, row in clean_leads.iterrows():
        s = row.get(std.CANONICAL_SMILES) or row.get(std.SMILES)
        m = Chem.MolFromSmiles(str(s))
        if m:
            mols.append(m)
            rocs_v = float(row.get(rocs_col, 0.0))
            sa_v = float(row.get("Raw_SA", calculate_sa_score(m)))
            qed_v = float(row.get(std.QED, QED.qed(m)))
            mw_v = float(row.get(std.MW, Descriptors.MolWt(m)))
            hac_v = int(row.get(std.HAC, m.GetNumHeavyAtoms()))
            legends.append(
                f"ROCS={rocs_v:.3f} | SA={sa_v:.2f}\nQED={qed_v:.2f} | MW={mw_v:.0f} ({hac_v}at)"
            )

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(len(mols), 3),
        subImgSize=(300, 280),
        legends=legends,
        useSVG=False,
    )
    img.save(str(target_png))
    return target_png


class ResultAnalyzer:
    """Comprehensive post-run analyzer computing statistics, CI bounds, and delivering visualization artifacts."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def generate_full_report(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        ref_smiles: str,
        monitor_prefix: str | Path | None = None,
    ) -> pd.DataFrame:
        """Produce authoritative comparative analysis, bootstrap CIs, and full figure suite."""
        b_df = unscale_to_physical_metrics(df_before)
        a_df = unscale_to_physical_metrics(df_after)

        b_valid = b_df[b_df[std.VALID] == 1]
        a_valid = a_df[a_df[std.VALID] == 1]

        # Summarize key properties
        metrics_before = {
            "validity": len(b_valid) / max(1, len(b_df)),
            "uniqueness": b_valid[std.CANONICAL_SMILES].nunique() / max(1, len(b_df)),
            "desired_ratio": float(b_df[std.DESIRED].mean())
            if std.DESIRED in b_df.columns
            else 0.0,
            "median_rocs": float(b_valid[std.ROCS].median())
            if not b_valid.empty
            else 0.0,
            "p90_rocs": float(b_valid[std.ROCS].quantile(0.90))
            if not b_valid.empty
            else 0.0,
            "median_raw_sa": float(b_valid["Raw_SA"].median())
            if not b_valid.empty
            else 0.0,
            "median_qed": float(b_valid[std.QED].median())
            if not b_valid.empty
            else 0.0,
            "median_hac": float(b_valid[std.HAC].median())
            if not b_valid.empty
            else 0.0,
            "median_mw": float(b_valid[std.MW].median()) if not b_valid.empty else 0.0,
        }

        metrics_after = {
            "validity": len(a_valid) / max(1, len(a_df)),
            "uniqueness": a_valid[std.CANONICAL_SMILES].nunique() / max(1, len(a_df)),
            "desired_ratio": float(a_df[std.DESIRED].mean())
            if std.DESIRED in a_df.columns
            else 0.0,
            "median_rocs": float(a_valid[std.ROCS].median())
            if not a_valid.empty
            else 0.0,
            "p90_rocs": float(a_valid[std.ROCS].quantile(0.90))
            if not a_valid.empty
            else 0.0,
            "median_raw_sa": float(a_valid["Raw_SA"].median())
            if not a_valid.empty
            else 0.0,
            "median_qed": float(a_valid[std.QED].median())
            if not a_valid.empty
            else 0.0,
            "median_hac": float(a_valid[std.HAC].median())
            if not a_valid.empty
            else 0.0,
            "median_mw": float(a_valid[std.MW].median()) if not a_valid.empty else 0.0,
        }

        comparison = pd.DataFrame({"Before": metrics_before, "After": metrics_after})
        comparison["Delta"] = comparison["After"] - comparison["Before"]

        # Save comparison table
        comp_csv = self.paths.root / "comparison.csv"
        comparison.to_csv(comp_csv)

        # Prioritize top unique leads: ROCS descending, SA ascending, QED descending
        top_leads = a_valid.sort_values(
            by=[std.ROCS, "Raw_SA", std.QED], ascending=[False, True, False]
        )
        top_leads_unique = top_leads.drop_duplicates(
            subset=[std.CANONICAL_SMILES]
        ).head(20)
        leads_tsv = self.paths.root / "top_prioritized_leads.tsv"
        top_leads_unique.to_csv(leads_tsv, sep="\t", index=False)

        # Generate complete publication visualization suite
        render_before_after_distributions(
            df_before=b_df,
            df_after=a_df,
            output_png=self.paths.figures_dir / "01_before_after_distributions.png",
            ref_mol=ref_smiles,
        )
        render_multiobjective_tradeoffs(
            df_before=b_df,
            df_after=a_df,
            output_png=self.paths.figures_dir / "02_multiobjective_tradeoffs.png",
        )
        if monitor_prefix:
            fit_tsv = Path(f"{monitor_prefix}_fit.tsv")
            if fit_tsv.is_file():
                render_training_trajectories(
                    history_file_or_df=fit_tsv,
                    output_png=self.paths.figures_dir / "03_training_trajectories.png",
                )
        render_unfavorable_features_diagnostics(
            df_before=b_df,
            df_after=a_df,
            output_png=self.paths.figures_dir
            / "04_diagnostic_unfavorable_features.png",
        )
        render_lead_structures_gallery(
            top_leads=top_leads_unique,
            ref_smiles=ref_smiles,
            output_png=self.paths.figures_dir / "05_lead_structures_gallery.png",
            top_k=5,
        )

        return comparison
