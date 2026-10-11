"""Comprehensive academic visualization suite for Ligand-Based De Novo Drug Design.

Includes 2D scaffold-aligned depictions, 3D conformer superposition ensembles,
RL training convergence dynamics, multi-objective Pareto frontiers, and
physicochemical descriptor distribution benchmarking.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_universal_rigid_anchor(mol: Chem.Mol) -> list[int]:
    """Universal First-Principles Rigid Anchor Extraction for Visualization.

    Decomposes any small molecule into internally rigid blocks by cutting
    all non-ring rotatable single bonds. Returns the heavy atom indices
    of the largest rigid block (guaranteed zero internal degrees of rotational freedom).
    """
    rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(rot_smarts)
    rot_bonds = [
        b.GetIdx() for a1, a2 in matches
        if (b := mol.GetBondBetweenAtoms(a1, a2)) is not None and not b.IsInRing()
    ]
    if not rot_bonds:
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        return heavy[: min(10, len(heavy))]

    frag_mol = Chem.FragmentOnBonds(mol, rot_bonds, addDummies=False)
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    heavy_frags = [
        [idx for idx in f if mol.GetAtomWithIdx(idx).GetSymbol() != "H"]
        for f in frags
    ]
    largest_frag = max(heavy_frags, key=len, default=[])
    return sorted(largest_frag)


def render_reference_ensemble(
    sdf_path: Path | str,
    output_png: Path | str | None = None,
) -> Path:
    """Render publication-quality 3-panel 3D superposition figure of reference conformers."""
    from rdkit.Chem import AllChem

    sdf_file = Path(sdf_path)
    target_png = Path(output_png) if output_png else sdf_file.parent / "reference_3d_superposition.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return target_png

    mol_combined = Chem.Mol(mols[0])
    mol_combined.RemoveAllConformers()
    for m in mols:
        mol_combined.AddConformer(m.GetConformer(), assignId=True)

    props = AllChem.MMFFGetMoleculeProperties(mol_combined)
    energies: list[float] = []
    for i in range(mol_combined.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol_combined, props, confId=i) if props else None
        if not ff:
            ff = AllChem.UFFGetMoleculeForceField(mol_combined, confId=i)
        energies.append(float(ff.CalcEnergy()) if ff else 0.0)

    min_e = min(energies)
    de_vals = [e - min_e for e in energies]
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
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(de_vals) if max(de_vals) > 0 else 1.0)

    def _draw_mol(ax: Any, mol: Chem.Mol, bond_c: Any, atom_c: Any, alpha: float = 1.0, lw: float = 2.0, pt_size: float = 30, label_core: bool = False) -> None:
        conf = mol.GetConformer()
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            if a1.GetSymbol() == "H" or a2.GetSymbol() == "H":
                continue
            p1, p2 = conf.GetAtomPosition(a1.GetIdx()), conf.GetAtomPosition(a2.GetIdx())
            is_core = (a1.GetIdx() in core_atoms) and (a2.GetIdx() in core_atoms)
            c = "#1e293b" if (label_core and is_core) else bond_c
            w = lw * 1.3 if (label_core and is_core) else lw
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color=c, alpha=alpha, linewidth=w)

        heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        for a in heavy:
            p = conf.GetAtomPosition(a.GetIdx())
            is_core = a.GetIdx() in core_atoms
            c = "#0f172a" if (label_core and is_core) else atom_c
            s = pt_size * 1.4 if (label_core and is_core) else pt_size
            ax.scatter([p.x], [p.y], [p.z], color=c, s=s, alpha=alpha, edgecolors="none")

    # Panel A: Ground state
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_mol(ax1, mols[min_idx], "#0284c7", "#0284c7", lw=2.4, pt_size=40, label_core=True)
    ax1.set_title(f"(A) Ground State Basin (ΔE = {de_vals[min_idx]:.2f} kcal/mol)\nCompact Folded Conformation", fontsize=11, fontweight="bold", pad=12)
    ax1.view_init(elev=20, azim=40)
    ax1.set_xlabel("X (Å)", fontsize=8)
    ax1.set_ylabel("Y (Å)", fontsize=8)
    ax1.set_zlabel("Z (Å)", fontsize=8)

    # Panel B: Most extended state
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_mol(ax2, mols[max_idx], "#f59e0b", "#f59e0b", lw=2.4, pt_size=40, label_core=True)
    ax2.set_title(f"(B) Pocket-Adapted Basin (ΔE = {de_vals[max_idx]:.2f} kcal/mol)\nExtended Conformation", fontsize=11, fontweight="bold", pad=12)
    ax2.view_init(elev=20, azim=40)
    ax2.set_xlabel("X (Å)", fontsize=8)
    ax2.set_ylabel("Y (Å)", fontsize=8)
    ax2.set_zlabel("Z (Å)", fontsize=8)

    # Panel C: Core-aligned superposition
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for i, m in enumerate(mols):
        c = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, c, c, alpha=0.75, lw=1.6, pt_size=20, label_core=True)
    _draw_mol(ax3, mols[min_idx], "#0f172a", "#0f172a", alpha=0.95, lw=2.8, pt_size=38, label_core=True)
    ax3.set_title(f"(C) Core-Aligned Ensemble ({len(mols)} Centroids)\nPharmacophore Envelope & Rotameric Fanning", fontsize=11, fontweight="bold", pad=12)
    ax3.view_init(elev=20, azim=40)
    ax3.set_xlabel("X (Å)", fontsize=8)
    ax3.set_ylabel("Y (Å)", fontsize=8)
    ax3.set_zlabel("Z (Å)", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label("Relative Energy ΔE (kcal/mol)", fontsize=8, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)

    fig.text(
        0.5,
        0.015,
        "Note: Superposition aligned on the principal rigid block solely for visual clarity. "
        "Downstream scoring (RDKitROCSScorer Open3DAlign) evaluates each conformer independently and is fully rotation- and translation-invariant.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontstyle="italic",
        color="#334155",
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


class AlignedDepicter:
    """Generates 2D molecular structure depictions aligned to a reference scaffold."""

    def __init__(self, reference: str | Chem.Mol) -> None:
        self.ref_mol = Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
        self.ref_core = MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            rdDepictor.Compute2DCoords(self.ref_mol)
            self.ref_core = self.ref_mol

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Aligns 2D layout of molecule matching reference core."""
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)
        if self.ref_core and mol_copy.HasSubstructMatch(self.ref_core):
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, self.ref_core)
            except Exception:
                pass
        return mol_copy

    def create_aligned_grid(
        self,
        molecules: Sequence[str | Chem.Mol],
        legends: Sequence[str] | None = None,
        output_path: Path | str | None = None,
        mols_per_row: int = 4,
        sub_img_size: tuple[int, int] = (350, 300),
    ) -> Any:
        """Renders grid image of aligned molecules."""
        mol_objs = []
        for m in molecules:
            mol_obj = Chem.MolFromSmiles(m) if isinstance(m, str) else m
            if mol_obj:
                mol_objs.append(self.align_molecule(mol_obj))

        img = Draw.MolsToGridImage(
            mol_objs,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=list(legends) if legends else None,
            useSVG=False,
        )
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img


def render_lead_structures_2d(
    molecules: Sequence[str | Chem.Mol],
    legends: Sequence[str] | None = None,
    output_png: Path | str | None = None,
    reference: str | Chem.Mol | None = None,
) -> Path | None:
    """Render 2D depiction grid of prioritized lead structures aligned to reference core."""
    if not molecules:
        return None
    target_png = Path(output_png) if output_png else Path("lead_structures_2d.png")
    depicter = AlignedDepicter(reference=reference or molecules[0])
    depicter.create_aligned_grid(molecules, legends=legends, output_path=target_png)
    return target_png


def render_training_curves(
    fit_tsv_path: Path | str,
    output_png: Path | str | None = None,
) -> Path | None:
    """Render RL training dynamics (loss, validity, uniqueness, desired ratio) across epochs."""
    tsv_file = Path(fit_tsv_path)
    if not tsv_file.exists():
        return None

    df_fit = pd.read_csv(tsv_file, sep="\t")
    if df_fit.empty or "Epoch" not in df_fit.columns:
        return None

    target_png = Path(output_png) if output_png else tsv_file.parent / "rl_training_curves.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=200)

    # Left Y-axis: Loss
    loss_col = [c for c in df_fit.columns if "loss" in c.lower()]
    if loss_col:
        color = "#e11d48"
        ax1.set_xlabel("Training Epoch", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Training Loss", color=color, fontsize=10, fontweight="bold")
        l1 = ax1.plot(df_fit["Epoch"], df_fit[loss_col[0]], color=color, linewidth=2.2, label=f"Loss ({loss_col[0]})")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.4)
    else:
        l1 = []

    # Right Y-axis: Success ratios [0, 1]
    ax2 = ax1.twinx()
    ratio_colors = {
        "valid_ratio": ("#0284c7", "-", "Validity"),
        "unique_ratio": ("#10b981", "--", "Uniqueness"),
        "desired_ratio": ("#8b5cf6", "-.", "Multi-Objective Desired"),
    }
    ratio_lines = []
    for col, (c, ls, lbl) in ratio_colors.items():
        if col in df_fit.columns:
            line = ax2.plot(df_fit["Epoch"], df_fit[col], color=c, linestyle=ls, linewidth=2.0, label=lbl)
            ratio_lines.extend(line)

    ax2.set_ylabel("Population Ratio (0.0 – 1.0)", color="#1e293b", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#1e293b")

    # Combine legends
    all_lines = l1 + ratio_lines
    labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    plt.title("Reinforcement Learning Training Dynamics & Convergence", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def find_sa_column(columns: Sequence[str]) -> str | None:
    """Safely identify synthesizability (SA) column without matching TPSA or unrelated descriptors."""
    for col in columns:
        c_up = col.upper()
        if c_up in ("SA", "SASCORE", "SA_SCORE", "SYNTHETIC_ACCESSIBILITY"):
            return col
    for col in columns:
        c_up = col.upper()
        if c_up == "TPSA" or "TPSA" in c_up:
            continue
        if c_up.startswith("SA_") or c_up.endswith("_SA") or "SASCORE" in c_up:
            return col
    return None


def find_rocs_column(columns: Sequence[str]) -> str | None:
    """Safely identify 3D ROCS / shape similarity column."""
    for col in columns:
        c_up = col.upper()
        if any(k in c_up for k in ("ROCS", "TANIMOTOCOMBO", "SHAPE_SIMILARITY", "SIMILARITY_3D")):
            return col
    return None


def render_epoch_pareto_evolution(
    smiles_tsv_path: Path | str,
    output_png: Path | str | None = None,
    max_epochs_to_plot: int = 5,
) -> Path | None:
    """Micro-visualize Pareto frontier progression across RL epochs.

    Shows how reinforcement learning pushes the frontier toward optimal
    multi-objective space (Shape Complementarity vs Synthesizability).
    """
    tsv_file = Path(smiles_tsv_path)
    if not tsv_file.exists():
        return None

    df = pd.read_csv(tsv_file, sep="\t")
    if df.empty or "Epoch" not in df.columns:
        return None

    # Resolve objective columns safely
    rocs_name = find_rocs_column(df.columns)
    sa_name = find_sa_column(df.columns)
    if not rocs_name or not sa_name:
        return None

    target_png = Path(output_png) if output_png else tsv_file.parent / "pareto_epoch_evolution.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    epochs = sorted(df["Epoch"].unique())
    if len(epochs) > max_epochs_to_plot:
        # Pick uniformly distributed sample epochs (e.g. initial, early, mid, late, final)
        indices = np.linspace(0, len(epochs) - 1, max_epochs_to_plot, dtype=int)
        selected_epochs = [epochs[i] for i in indices]
    else:
        selected_epochs = epochs

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    # Plot all points in faint background
    ax.scatter(df[sa_name], df[rocs_name], c="#cbd5e1", alpha=0.25, s=15, label="Exploration Pool")

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(selected_epochs), vmax=max(selected_epochs))

    for ep in selected_epochs:
        sub = df[df["Epoch"] == ep].copy()
        if sub.empty:
            continue
        scores = sub[[rocs_name, sa_name]].to_numpy(dtype=float)
        # Compute non-dominated points for this epoch
        is_pareto = np.ones(len(scores), dtype=bool)
        for i in range(len(scores)):
            dominated = np.any((scores >= scores[i]).all(axis=1) & (scores > scores[i]).any(axis=1))
            if dominated:
                is_pareto[i] = False

        front = sub[is_pareto].sort_values(by=sa_name)
        c = cmap(norm(ep))
        ax.scatter(front[sa_name], front[rocs_name], color=c, s=55, edgecolors="#0f172a", linewidth=0.8, zorder=5)
        if len(front) > 1:
            ax.plot(front[sa_name], front[rocs_name], color=c, linestyle="--", linewidth=1.8, label=f"Epoch {ep} Front", zorder=4)
        else:
            ax.scatter([], [], color=c, label=f"Epoch {ep} Front")

    # Annotate evolution direction arrow safely within axes bounds
    x_end = float(df[sa_name].quantile(0.85))
    y_end = float(df[rocs_name].quantile(0.85))
    x_start = float(df[sa_name].quantile(0.35))
    y_start = float(df[rocs_name].quantile(0.35))

    if abs(x_end - x_start) < 0.15:
        x_start = max(0.0, x_end - 0.25)
        y_start = max(0.0, y_end - 0.12)

    ax.annotate(
        "Pareto Vector",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(facecolor="#e11d48", edgecolor="#e11d48", width=2, headwidth=7, shrink=0.08),
        fontsize=9,
        fontweight="bold",
        color="#be123c",
        ha="center",
    )

    ax.set_title("Micro-Evolution of Pareto Frontier Across RL Training Epochs", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"{sa_name} (Synthesizability / Higher is better)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{rocs_name} (3D Shape Overlap)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_property_distributions(
    df: pd.DataFrame,
    output_png: Path | str | None = None,
    ref_mol: Chem.Mol | str | None = None,
) -> Path | None:
    """Render multi-panel property distribution histograms with query reference benchmarks."""
    if df.empty or "SMILES" not in df.columns:
        return None

    target_png = Path(output_png) if output_png else Path("property_distributions.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    # Compute physicochemical descriptors for generated molecules if missing
    from rdkit.Chem import rdMolDescriptors
    calc_df = df.copy()
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in calc_df["SMILES"]]

    if "MW" not in calc_df.columns:
        calc_df["MW"] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
    if "LogP" not in calc_df.columns and "logP" not in calc_df.columns:
        calc_df["LogP"] = [Descriptors.MolLogP(m) if m else np.nan for m in mols]
    if "TPSA" not in calc_df.columns:
        calc_df["TPSA"] = [rdMolDescriptors.CalcTPSA(m) if m else np.nan for m in mols]
    if "QED" not in calc_df.columns:
        calc_df["QED"] = [Descriptors.qed(m) if m else np.nan for m in mols]
    if "RotBonds" not in calc_df.columns:
        calc_df["RotBonds"] = [Lipinski.NumRotatableBonds(m) if m else np.nan for m in mols]
    if "HBD" not in calc_df.columns:
        calc_df["HBD"] = [Lipinski.NumHDonors(m) if m else np.nan for m in mols]
    if "HBA" not in calc_df.columns:
        calc_df["HBA"] = [Lipinski.NumHAcceptors(m) if m else np.nan for m in mols]

    # Reference values
    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol
    ref_vals: dict[str, float] = {}
    if ref_m:
        ref_vals = {
            "MW": float(Descriptors.MolWt(ref_m)),
            "LogP": float(Descriptors.MolLogP(ref_m)),
            "TPSA": float(rdMolDescriptors.CalcTPSA(ref_m)),
            "QED": float(Descriptors.qed(ref_m)),
            "RotBonds": float(Lipinski.NumRotatableBonds(ref_m)),
            "HBD": float(Lipinski.NumHDonors(ref_m)),
            "HBA": float(Lipinski.NumHAcceptors(ref_m)),
        }

    # Descriptor metadata map
    descriptor_map: dict[str, dict[str, Any]] = {
        "MW": {"title": "Molecular Weight (Da)", "color": "#0284c7"},
        "LogP": {"title": "Lipophilicity (CLogP)", "color": "#059669"},
        "TPSA": {"title": "Polar Surface Area (TPSA, Å²)", "color": "#0d9488"},
        "QED": {"title": "Drug-Likeness (QED)", "color": "#8b5cf6"},
        "RotBonds": {"title": "Rotatable Bonds (Flexibility)", "color": "#475569"},
        "HBD": {"title": "H-Bond Donors (HBD)", "color": "#d97706"},
        "HBA": {"title": "H-Bond Acceptors (HBA)", "color": "#0891b2"},
    }

    # Add scoring objectives if present
    rocs_col = find_rocs_column(calc_df.columns)
    if rocs_col:
        descriptor_map[rocs_col] = {"title": "3D Shape Similarity (ROCS)", "color": "#e11d48"}
    sa_col = find_sa_column(calc_df.columns)
    if sa_col:
        descriptor_map[sa_col] = {"title": "Synthesizability (SAscore)", "color": "#2563eb"}

    active_props = [p for p in descriptor_map if p in calc_df.columns]
    n_plots = len(active_props)
    if n_plots == 0:
        return None

    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), dpi=200)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    fig.suptitle("De Novo Generated Leads: Multi-Property Distribution Profiles", fontsize=13, fontweight="bold", y=0.99)

    for idx, prop in enumerate(active_props):
        ax = axes_flat[idx]
        data = pd.to_numeric(calc_df[prop], errors="coerce").dropna()
        meta = descriptor_map[prop]
        color = meta["color"]

        if len(data) > 0:
            ax.hist(data, bins=16, color=color, edgecolor="#0f172a", alpha=0.75, linewidth=0.8)
            mean_v = float(data.mean())
            ax.axvline(mean_v, color="#0f172a", linestyle=":", linewidth=1.8, label=f"Mean: {mean_v:.2f}")

        if prop in ref_vals:
            rv = ref_vals[prop]
            ax.axvline(rv, color="#e11d48", linestyle="--", linewidth=2.0, label=f"Query: {rv:.1f}")

        ax.set_title(meta["title"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=7.5)

    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_pareto_frontier(
    df: pd.DataFrame,
    output_png: Path | str | None = None,
    rocs_col: str | None = None,
    sa_col: str | None = None,
) -> Path | None:
    """Render 2D scatter of multi-objective Pareto non-dominated front."""
    if df.empty or "Pareto_Rank" not in df.columns:
        return None

    target_png = Path(output_png) if output_png else Path("pareto_frontier.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    r_col = rocs_col or find_rocs_column(df.columns)
    s_col = sa_col or find_sa_column(df.columns)
    if not r_col or not s_col:
        return None

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    front_0 = df[df["Pareto_Rank"] == 0]
    others = df[df["Pareto_Rank"] > 0]

    ax.scatter(others[s_col], others[r_col], c="#94a3b8", alpha=0.45, s=25, label="Dominated Candidates")
    ax.scatter(front_0[s_col], front_0[r_col], c="#e11d48", alpha=0.95, s=60, edgecolors="#0f172a", linewidth=1.0, label="Pareto Front (Rank 0)")

    ax.set_xlabel(f"{s_col} (Synthesizability)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{r_col} (3D Shape Overlap)", fontsize=10, fontweight="bold")
    ax.set_title("Multi-Objective Pareto Trade-off Frontier", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png
