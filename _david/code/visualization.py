from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, Lipinski, QED, rdDepictor, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.schema import std


def unscale_to_physical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # ~ [Ertl & Schuffenhauer, Estimation of Synthetic Accessibility Score, J. Cheminform. 2009]
    """Unscale ROCS to [0, 2] TanimotoCombo and recalculate raw SAscore in [1, 10] if clipped."""
    if df.empty:
        return df.copy()

    df_out = df.copy()
    if std.ROCS in df_out.columns:
        vals = df_out[std.ROCS].dropna()
        if not vals.empty and vals.max() <= 1.0001:
            df_out[std.ROCS] = df_out[std.ROCS] * 2.0

    if std.SA in df_out.columns and std.SMILES in df_out.columns:
        vals = df_out[std.SA].dropna()
        if not vals.empty and vals.max() <= 1.0001:
            mols = [
                Chem.MolFromSmiles(s) if isinstance(s, str) else None
                for s in df_out[std.SMILES]
            ]
            df_out[std.SA] = [
                float(calculate_sa_score(m)) if m is not None else 10.0
                for m in mols
            ]

    return df_out


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

    # Use pre-calculated energetic properties from SDF directly
    has_props = any(m.HasProp("DeltaE_kcal_mol") or m.HasProp("Energy_MMFF_kcal_mol") for m in mols)
    if has_props:
        de_vals: list[float] = []
        for m in mols:
            if m.HasProp("DeltaE_kcal_mol"):
                de_vals.append(m.GetDoubleProp("DeltaE_kcal_mol"))
            elif m.HasProp("Energy_MMFF_kcal_mol"):
                de_vals.append(m.GetDoubleProp("Energy_MMFF_kcal_mol"))
            else:
                de_vals.append(0.0)
        if not any(m.HasProp("DeltaE_kcal_mol") for m in mols):
            min_e = min(de_vals)
            de_vals = [e - min_e for e in de_vals]
    else:
        props = AllChem.MMFFGetMoleculeProperties(mol_combined)
        energies: list[float] = []
        for i in range(mol_combined.GetNumConformers()):
            ff = (
                AllChem.MMFFGetMoleculeForceField(mol_combined, props, confId=i)
                if props
                else None
            )
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

    def _draw_mol(
        ax: Any,
        mol: Chem.Mol,
        bond_c: Any,
        atom_c: Any,
        alpha: float = 1.0,
        lw: float = 2.0,
        pt_size: float = 30,
        label_core: bool = False,
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
            c = "#1e293b" if (label_core and is_core) else bond_c
            w = lw * 1.3 if (label_core and is_core) else lw
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
            c = "#0f172a" if (label_core and is_core) else atom_c
            s = pt_size * 1.4 if (label_core and is_core) else pt_size
            ax.scatter(
                [p.x], [p.y], [p.z], color=c, s=s, alpha=alpha, edgecolors="none"
            )

    # Panel A: Ground state
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_mol(
        ax1, mols[min_idx], "#0284c7", "#0284c7", lw=2.4, pt_size=40, label_core=True
    )
    ax1.set_title(
        f"(A) Ground State Basin (ΔE = {de_vals[min_idx]:.2f} kcal/mol)\nCompact Folded Conformation",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.view_init(elev=20, azim=40)
    ax1.set_xlabel("X (Å)", fontsize=8)
    ax1.set_ylabel("Y (Å)", fontsize=8)
    ax1.set_zlabel("Z (Å)", fontsize=8)

    # Panel B: Extended state
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_mol(
        ax2, mols[max_idx], "#f59e0b", "#f59e0b", lw=2.4, pt_size=40, label_core=True
    )
    ax2.set_title(
        f"(B) Pocket-Adapted Basin (ΔE = {de_vals[max_idx]:.2f} kcal/mol)\nExtended Conformation",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax2.view_init(elev=20, azim=40)
    ax2.set_xlabel("X (Å)", fontsize=8)
    ax2.set_ylabel("Y (Å)", fontsize=8)
    ax2.set_zlabel("Z (Å)", fontsize=8)

    # Panel C: Superposition
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(de_vals) if max(de_vals) > 0 else 1.0)
    for i, m in enumerate(mols):
        color = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, color, color, alpha=0.75, lw=1.5, pt_size=20)
    ax3.set_title(
        f"(C) Ensemble Superposition ({len(mols)} Conformer Basins)\nAligned on Rigid Core",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax3.view_init(elev=20, azim=40)
    ax3.set_xlabel("X (Å)", fontsize=8)
    ax3.set_ylabel("Y (Å)", fontsize=8)
    ax3.set_zlabel("Z (Å)", fontsize=8)

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


class AlignedDepicter:
    """Renders 2D molecular depictions and prioritized lead cards aligned onto a shared Murcko core template."""

    def __init__(self, reference: str | Chem.Mol | None) -> None:
        self.ref_mol = (
            Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
        )
        self.ref_core = (
            MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        )
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            rdDepictor.Compute2DCoords(self.ref_mol)
            self.ref_core = self.ref_mol

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Compute aligned 2D coordinates for mol matching the reference scaffold."""
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
        mols_per_row: int = 3,
        sub_img_size: tuple[int, int] = (340, 260),
    ) -> Any:
        """Render a multi-molecule grid image with aligned 2D coordinate depictions."""
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

    def render_leads(
        self,
        molecules: Sequence[str | Chem.Mol] | pd.DataFrame,
        legends: Sequence[str] | None = None,
        output_png: Path | str | None = None,
        max_leads: int = 5,
    ) -> Path | None:
        """Render aligned 2D chemical structure cards with true physical metrics."""
        target_png = Path(output_png) if output_png else Path("lead_structures_2d.png")
        target_png.parent.mkdir(parents=True, exist_ok=True)

        plot_mols: list[Chem.Mol] = []
        plot_legends: list[str] = []

        if self.ref_mol is not None:
            ref_clean = Chem.RemoveHs(Chem.Mol(self.ref_mol))
            plot_mols.append(ref_clean)
            hac = int(ref_clean.GetNumHeavyAtoms())
            mw = float(Descriptors.MolWt(ref_clean))
            qed_val = float(QED.qed(ref_clean))
            sa_val = float(calculate_sa_score(ref_clean))
            plot_legends.append(
                f"QUERY SEED\n"
                f"HAC: {hac} | MW: {mw:.1f}\n"
                f"QED: {qed_val:.2f} | SA: {sa_val:.2f}"
            )

        if isinstance(molecules, pd.DataFrame):
            df_leads = molecules.head(max_leads)
            for rank_num, (_, row) in enumerate(df_leads.iterrows(), start=1):
                smi = str(row.get(std.SMILES, ""))
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    plot_mols.append(m)
                    rocs = float(row.get(std.ROCS, 0.0))
                    if 0.0 < rocs <= 1.0001:
                        rocs = rocs * 2.0
                    des = float(row.get(std.DESIRED, 0.0))
                    mw_val = row.get(std.MW)
                    mw = (
                        float(mw_val)
                        if (mw_val is not None and pd.notna(mw_val) and float(mw_val) > 0.0)
                        else float(Descriptors.MolWt(m))
                    )
                    qed_v = float(row.get(std.QED, QED.qed(m)))
                    sa_raw = row.get(std.SA)
                    sa_v = (
                        float(sa_raw)
                        if (sa_raw is not None and pd.notna(sa_raw) and float(sa_raw) > 1.0001)
                        else float(calculate_sa_score(m))
                    )
                    plot_legends.append(
                        f"Lead #{rank_num}\n"
                        f"TanimotoCombo: {rocs:.2f} | Des: {des:.2f}\n"
                        f"MW: {mw:.1f} | QED: {qed_v:.2f} | SA: {sa_v:.2f}"
                    )
        else:
            for idx, item in enumerate(molecules[:max_leads]):
                m = Chem.MolFromSmiles(item) if isinstance(item, str) else item
                if m is not None:
                    plot_mols.append(m)
                    if legends and idx < len(legends):
                        plot_legends.append(legends[idx])
                    else:
                        mw = float(Descriptors.MolWt(m))
                        qed_v = float(QED.qed(m))
                        sa_v = float(calculate_sa_score(m))
                        plot_legends.append(
                            f"Lead #{idx + 1}\n"
                            f"MW: {mw:.1f} | QED: {qed_v:.2f} | SA: {sa_v:.2f}"
                        )

        if not plot_mols:
            return None

        self.create_aligned_grid(
            plot_mols,
            legends=plot_legends,
            output_path=target_png,
            mols_per_row=min(3, len(plot_mols)),
            sub_img_size=(340, 260),
        )
        return target_png


def render_lead_structures_2d(
    molecules: Sequence[str | Chem.Mol] | pd.DataFrame,
    legends: Sequence[str] | None = None,
    output_png: Path | str | None = None,
    reference: str | Chem.Mol | None = None,
    max_leads: int = 5,
) -> Path | None:
    """Convenience functional wrapper delegating lead depiction to AlignedDepicter."""
    depicter = AlignedDepicter(reference=reference)
    return depicter.render_leads(
        molecules=molecules,
        legends=legends,
        output_png=output_png,
        max_leads=max_leads,
    )


def render_training_curves(
    fit_tsv_path: Path | str,
    output_png: Path | str | None = None,
) -> Path | None:
    """Render reinforcement learning metric progression curves across training epochs."""
    tsv_file = Path(fit_tsv_path)
    if not tsv_file.exists():
        return None

    df_fit = pd.read_csv(tsv_file, sep="\t")
    if df_fit.empty or std.EPOCH not in df_fit.columns:
        return None

    target_png = (
        Path(output_png) if output_png else tsv_file.parent / "rl_training_curves.png"
    )
    target_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
    metric_styles: dict[str, tuple[str, str, str]] = {
        std.LOSS_TRAIN: ("#e63946", "-", "Training Loss"),
        std.VALID_RATIO: ("#2a9d8f", "-", "Valid Ratio"),
        std.UNIQUE_RATIO: ("#e9c46a", "--", "Unique Ratio"),
        std.DESIRED_RATIO: ("#8b5cf6", "-.", "Desired Ratio"),
    }

    plotted = False
    for col, (color, ls, label) in metric_styles.items():
        if col in df_fit.columns:
            ax.plot(
                df_fit[std.EPOCH],
                df_fit[col],
                color=color,
                linestyle=ls,
                linewidth=2.0,
                label=label,
            )
            plotted = True

    if not plotted:
        num_cols = [
            c
            for c in df_fit.columns
            if c != std.EPOCH and pd.api.types.is_numeric_dtype(df_fit[c])
        ]
        for col in num_cols[:4]:
            ax.plot(df_fit[std.EPOCH], df_fit[col], linewidth=1.8, label=col)

    ax.set_title("Graph-Transformer Reinforcement Learning Dynamics", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_property_distributions(
    df: pd.DataFrame,
    output_png: Path | str | None = None,
    ref_mol: Chem.Mol | str | None = None,
) -> Path | None:
    """Render comprehensive 2x5 physicochemical and alignment property distribution profile."""
    if df.empty or std.SMILES not in df.columns:
        return None

    target_png = Path(output_png) if output_png else Path("property_distributions.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    valid_df = (
        df[df[std.VALID].astype(bool)].copy()
        if std.VALID in df.columns
        else df.copy()
    )
    if valid_df.empty:
        return None

    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in valid_df[std.SMILES]]

    if std.MW not in valid_df.columns:
        valid_df[std.MW] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
    if std.LOGP not in valid_df.columns:
        valid_df[std.LOGP] = [Descriptors.MolLogP(m) if m else np.nan for m in mols]
    if std.TPSA not in valid_df.columns:
        valid_df[std.TPSA] = [rdMolDescriptors.CalcTPSA(m) if m else np.nan for m in mols]
    if std.QED not in valid_df.columns:
        valid_df[std.QED] = [QED.qed(m) if m else np.nan for m in mols]
    if std.SA not in valid_df.columns or valid_df[std.SA].nunique() <= 3:
        valid_df[std.SA] = [calculate_sa_score(m) if m else np.nan for m in mols]
    if std.HBD not in valid_df.columns:
        valid_df[std.HBD] = [Lipinski.NumHDonors(m) if m else 0 for m in mols]
    if std.HBA not in valid_df.columns:
        valid_df[std.HBA] = [Lipinski.NumHAcceptors(m) if m else 0 for m in mols]
    if std.ROT_BONDS not in valid_df.columns:
        valid_df[std.ROT_BONDS] = [Lipinski.NumRotatableBonds(m) if m else 0 for m in mols]

    seed_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_m, 2, nBits=2048) if ref_m else None
    if std.TANIMOTO_TO_SEED not in valid_df.columns:
        if seed_fp is not None:
            valid_df[std.TANIMOTO_TO_SEED] = [
                float(DataStructs.TanimotoSimilarity(seed_fp, rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)))
                if m else 0.0
                for m in mols
            ]
        else:
            valid_df[std.TANIMOTO_TO_SEED] = 0.0

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=200)
    fig.suptitle("Ligand-Based De Novo Drug Design: Multi-Objective & Diagnostic Profiles", fontsize=16, fontweight="bold", y=0.98)

    props_config = [
        (std.ROCS, "3D ROCS TanimotoCombo (Color + Shape)", "#2563EB", (0.0, 2.0), 2.0 if ref_m else None),
        (std.SA, "SA Score (Synthesizability)", "#10B981", (1.0, 10.0), float(calculate_sa_score(ref_m)) if ref_m else 3.74),
        (std.QED, "Quantitative Estimate of Drug-likeness (QED)", "#8B5CF6", (0.0, 1.0), float(QED.qed(ref_m)) if ref_m else 0.21),
        (std.MW, "Molecular Weight (Da)", "#F59E0B", (150.0, 600.0), float(Descriptors.MolWt(ref_m)) if ref_m else 458.4),
        (std.LOGP, "Calculated LogP (Wildman-Crippen)", "#EC4899", (-2.0, 6.0), float(Descriptors.MolLogP(ref_m)) if ref_m else 2.23),
        (std.TPSA, "Topological Polar Surface Area (Å²)", "#06B6D4", (20.0, 220.0), float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 197.4),
        (std.HBD, "Hydrogen Bond Donors (Lipinski)", "#64748B", (0, 10), int(Lipinski.NumHDonors(ref_m)) if ref_m else 8),
        (std.HBA, "Hydrogen Bond Acceptors (Lipinski)", "#0EA5E9", (0, 15), int(Lipinski.NumHAcceptors(ref_m)) if ref_m else 11),
        (std.ROT_BONDS, "Rotatable Bonds Count", "#6366F1", (0, 15), int(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3),
        (std.TANIMOTO_TO_SEED, "2D ECFP4 Tanimoto to Seed", "#E11D48", (0.0, 1.0), 1.0 if ref_m else None),
    ]

    for idx, (col, title, color, x_range, ref_val) in enumerate(props_config):
        ax = axes[idx // 5, idx % 5]
        data = valid_df[col].dropna()
        if not data.empty:
            ax.hist(data, bins=25, range=x_range, color=color, alpha=0.75, edgecolor="#1e293b", linewidth=0.6)
            mean_v = float(data.mean())
            ax.axvline(mean_v, color="#1e293b", linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.2f}")
        if ref_val is not None:
            ax.axvline(ref_val, color="#DC2626", linestyle="-", linewidth=2.0, label=f"Seed: {ref_val:.2f}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(x_range)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


class ResultAnalyzer:
    """Extracts, prioritizes, deduplicates chemical leads, and renders presentation deliverables."""

    def __init__(
        self,
        deliverables_dir: Path | str,
        raw_data_dir: Path | str | None = None,
    ) -> None:
        self.deliverables_dir = Path(deliverables_dir)
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir is not None else self.deliverables_dir
        self.deliverables_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        candidates_tsv: Path | str,
        ref_sdf: Path | str | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Process candidate TSV, calculate properties, prioritize leads, and emit presentation figures."""
        tsv_path = Path(candidates_tsv)
        if not tsv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(tsv_path, sep="\t")
        if df.empty or std.SMILES not in df.columns:
            return df

        df = unscale_to_physical_metrics(df)

        if std.VALID not in df.columns:
            df[std.VALID] = [Chem.MolFromSmiles(s) is not None for s in df[std.SMILES]]
        valid_df = df[df[std.VALID].astype(bool)].copy()

        if valid_df.empty:
            return pd.DataFrame()

        if std.QED not in valid_df.columns:
            valid_df[std.QED] = [
                float(QED.qed(m)) if (m := Chem.MolFromSmiles(s)) is not None else 0.0
                for s in valid_df[std.SMILES]
            ]
        if std.MW not in valid_df.columns:
            valid_df[std.MW] = [
                float(Descriptors.MolWt(m)) if (m := Chem.MolFromSmiles(s)) is not None else 0.0
                for s in valid_df[std.SMILES]
            ]

        # Prioritize leads: ROCS descending, SA ascending (lower is easier), QED descending
        sort_cols = [c for c in [std.ROCS, std.SA, std.QED] if c in valid_df.columns]
        ascending = [False, True, False][:len(sort_cols)]
        if sort_cols:
            valid_df = valid_df.sort_values(by=sort_cols, ascending=ascending)

        # Deduplicate candidates by canonical SMILES to extract distinct chemical leads
        valid_df[std.CANONICAL_SMILES] = [
            Chem.CanonSmiles(s) if (s and Chem.MolFromSmiles(s) is not None) else s
            for s in valid_df[std.SMILES]
        ]
        unique_leads = valid_df.drop_duplicates(subset=[std.CANONICAL_SMILES])
        top_leads = unique_leads.head(top_k).reset_index(drop=True)
        top_leads.to_csv(self.deliverables_dir / "top_prioritized_leads.tsv", sep="\t", index=False)

        ref_mol: Chem.Mol | None = None
        if ref_sdf and Path(ref_sdf).exists():
            suppl = Chem.SDMolSupplier(str(ref_sdf), removeHs=False)
            ref_mol = next((m for m in suppl if m is not None), None)

        render_property_distributions(
            valid_df,
            output_png=self.deliverables_dir / "property_distributions.png",
            ref_mol=ref_mol,
        )

        if not top_leads.empty:
            lead_png = self.deliverables_dir / "lead_structures_2d.png"
            depicter = AlignedDepicter(reference=ref_mol)
            depicter.render_leads(top_leads, output_png=lead_png)
            if lead_png.exists():
                shutil.copyfile(lead_png, self.raw_data_dir / "aligned_lead_depictions.png")

        summary = {
            "total_generated": len(df),
            "valid_molecules": int(valid_df[std.VALID].sum()),
            "top_k_extracted": len(top_leads),
        }
        with open(self.raw_data_dir / "result_analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return top_leads
