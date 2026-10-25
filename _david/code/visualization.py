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

from _david.code.config import RunPaths
from _david.code.schema import std


def unscale_to_physical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ROCS is [0, 2] TanimotoCombo and calculate physical properties if missing."""
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
        raise ValueError(f"No valid molecules found in reference SDF: {sdf_file}")

    mol_combined = Chem.Mol(mols[0])
    for m in mols[1:]:
        mol_combined.AddConformer(m.GetConformer(), assignId=True)

    de_vals = [
        float(m.GetProp("DeltaE_kcal_mol"))
        if m.HasProp("DeltaE_kcal_mol")
        else float(i) * 1.5
        for i, m in enumerate(mols)
    ]

    anchor_atoms = get_universal_rigid_anchor(mol_combined)
    for i in range(1, len(mols)):
        try:
            AllChem.AlignMol(mols[i], mols[0], atomMap=[(a, a) for a in anchor_atoms])
        except Exception:
            pass

    fig = plt.figure(figsize=(15, 5), dpi=200)

    def _draw_mol(
        ax: Any,
        m: Chem.Mol,
        c_atom: Any,
        c_bond: Any,
        alpha: float = 0.8,
        lw: float = 1.2,
    ) -> None:
        conf = m.GetConformer()
        pts = conf.GetPositions()
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [pts[i, 2], pts[j, 2]],
                color=c_bond,
                linewidth=lw,
                alpha=alpha,
            )
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c_atom, s=15, alpha=alpha)

    # Panel A: Lowest energy conformer
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_mol(ax1, mols[0], "#2563eb", "#3b82f6", alpha=0.9, lw=1.5)
    ax1.set_title(
        f"(A) Lowest-Energy Basin\nΔE = {de_vals[0]:.2f} kcal/mol",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.view_init(elev=20, azim=40)

    # Panel B: Second basin or profile
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    if len(mols) > 1:
        _draw_mol(ax2, mols[1], "#059669", "#10b981", alpha=0.9, lw=1.5)
        ax2.set_title(
            f"(B) Secondary Basin\nΔE = {de_vals[1]:.2f} kcal/mol",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
    else:
        _draw_mol(ax2, mols[0], "#2563eb", "#3b82f6", alpha=0.9, lw=1.5)
        ax2.set_title("(B) Primary Basin", fontsize=11, fontweight="bold", pad=12)
    ax2.view_init(elev=20, azim=40)

    # Panel C: Superposition
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(de_vals) if max(de_vals) > 0 else 1.0)
    for i, m in enumerate(mols):
        color = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, color, color, alpha=0.75, lw=1.5)
    ax3.set_title(
        f"(C) Ensemble Superposition ({len(mols)} Basins)\nAligned on Rigid Core",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax3.view_init(elev=20, azim=40)

    plt.tight_layout()
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

    def render_leads(
        self,
        molecules: Sequence[str | Chem.Mol] | pd.DataFrame,
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
            plot_mols.append(self.align_molecule(ref_clean))
            hac = int(ref_clean.GetNumHeavyAtoms())
            mw = float(Descriptors.MolWt(ref_clean))
            qed_val = float(QED.qed(ref_clean))
            sa_val = float(calculate_sa_score(ref_clean))
            plot_legends.append(
                f"TARGET QUERY SEED\nHAC: {hac} | MW: {mw:.1f}\nQED: {qed_val:.2f} | SA: {sa_val:.2f}"
            )

        if isinstance(molecules, pd.DataFrame):
            df_leads = molecules.head(max_leads)
            seed_fp = (
                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    self.ref_mol, 2, nBits=2048
                )
                if self.ref_mol
                else None
            )
            for rank_num, (_, row) in enumerate(df_leads.iterrows(), start=1):
                smi = str(row.get(std.SMILES, ""))
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    plot_mols.append(self.align_molecule(m))
                    rocs = float(row.get(std.ROCS, 0.0))
                    sa = float(row.get("Raw_SA", row.get(std.SA, 0.0)))
                    qed_val = float(row.get(std.QED, 0.0))
                    mw = float(row.get(std.MW, Descriptors.MolWt(m)))
                    sim = (
                        float(
                            DataStructs.TanimotoSimilarity(
                                seed_fp,
                                rdMolDescriptors.GetMorganFingerprintAsBitVect(
                                    m, 2, nBits=2048
                                ),
                            )
                        )
                        if seed_fp
                        else 0.0
                    )
                    plot_legends.append(
                        f"LEAD #{rank_num}\nROCS: {rocs:.3f} | SA: {sa:.2f}\nQED: {qed_val:.2f} | MW: {mw:.1f}\nECFP4 Sim: {sim:.2f}"
                    )

        if not plot_mols:
            return None

        img = Draw.MolsToGridImage(
            plot_mols,
            molsPerRow=min(6, len(plot_mols)),
            subImgSize=(340, 260),
            legends=plot_legends,
            useSVG=False,
        )
        img.save(str(target_png))
        return target_png


def render_training_curves(
    tsv_path: Path | str,
    output_png: Path | str | None = None,
) -> Path | None:
    """Render reinforcement learning training dynamics over epochs."""
    tsv_file = Path(tsv_path)
    if not tsv_file.is_file():
        return None

    try:
        df_fit = pd.read_csv(tsv_file, sep="\t")
    except Exception:
        return None

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
        "epsilon": ("#64748b", ":", "Exploration Noise (ε)"),
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

    ax.set_title(
        "SequenceRNN Reinforcement Learning Dynamics",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_property_distributions(
    df_after: pd.DataFrame,
    df_before: pd.DataFrame | None = None,
    output_png: Path | str | None = None,
    ref_mol: Chem.Mol | str | None = None,
) -> Path | None:
    """Render comprehensive 2x5 profile comparing Baseline vs Reinforced distributions."""
    if df_after.empty or std.SMILES not in df_after.columns:
        return None

    target_png = Path(output_png) if output_png else Path("property_distributions.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    after_clean = unscale_to_physical_metrics(df_after)
    before_clean = (
        unscale_to_physical_metrics(df_before)
        if df_before is not None and not df_before.empty
        else None
    )
    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), dpi=200)
    fig.suptitle(
        "Ligand-Based De Novo Design: Multi-Objective & Physicochemical Distributions",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    sa_col = "Raw_SA" if "Raw_SA" in after_clean.columns else std.SA
    props_config = [
        (
            std.ROCS,
            "3D ROCS TanimotoCombo",
            "#2563EB",
            (0.0, 2.0),
            2.0 if ref_m else None,
        ),
        (
            sa_col,
            "SA Score (Synthesizability)",
            "#10B981",
            (1.0, 10.0),
            float(calculate_sa_score(ref_m)) if ref_m else 3.74,
        ),
        (
            std.QED,
            "Drug-likeness (QED)",
            "#8B5CF6",
            (0.0, 1.0),
            float(QED.qed(ref_m)) if ref_m else 0.21,
        ),
        (
            std.MW,
            "Molecular Weight (Da)",
            "#F59E0B",
            (150.0, 600.0),
            float(Descriptors.MolWt(ref_m)) if ref_m else 458.4,
        ),
        (
            std.LOGP,
            "Calculated LogP",
            "#EC4899",
            (-2.0, 6.0),
            float(Descriptors.MolLogP(ref_m)) if ref_m else 2.23,
        ),
        (
            std.TPSA,
            "Topological Polar Surface Area (Å²)",
            "#06B6D4",
            (20.0, 220.0),
            float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 197.4,
        ),
        (
            std.HBD,
            "H-Bond Donors",
            "#64748B",
            (0, 10),
            int(Lipinski.NumHDonors(ref_m)) if ref_m else 8,
        ),
        (
            std.HBA,
            "H-Bond Acceptors",
            "#0EA5E9",
            (0, 15),
            int(Lipinski.NumHAcceptors(ref_m)) if ref_m else 11,
        ),
        (
            std.ROT_BONDS,
            "Rotatable Bonds",
            "#6366F1",
            (0, 15),
            int(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3,
        ),
        (
            std.HAC,
            "Heavy Atom Count",
            "#E11D48",
            (10, 45),
            int(ref_m.GetNumHeavyAtoms()) if ref_m else 33,
        ),
    ]

    for idx, (col, title, color, x_range, ref_val) in enumerate(props_config):
        ax = axes[idx // 5, idx % 5]

        # Baseline distribution (gray hatched)
        if before_clean is not None and col in before_clean.columns:
            data_before = before_clean[col].dropna()
            if not data_before.empty:
                ax.hist(
                    data_before,
                    bins=25,
                    range=x_range,
                    color="#94a3b8",
                    alpha=0.45,
                    density=True,
                    edgecolor="#475569",
                    linewidth=0.5,
                    label="Baseline Prior",
                )

        # Reinforced distribution (colored solid)
        if col in after_clean.columns:
            data_after = after_clean[col].dropna()
            if not data_after.empty:
                ax.hist(
                    data_after,
                    bins=25,
                    range=x_range,
                    color=color,
                    alpha=0.65,
                    density=True,
                    edgecolor="#1e293b",
                    linewidth=0.6,
                    label="Reinforced Agent",
                )

        if ref_val is not None:
            ax.axvline(
                ref_val,
                color="#dc2626",
                linestyle="--",
                linewidth=1.8,
                label=f"Target ({ref_val:.1f})",
            )

        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlim(x_range)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


class ResultAnalyzer:
    """Production lead analyzer and deliverable builder."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def analyze(
        self,
        df_after: pd.DataFrame,
        df_before: pd.DataFrame | None = None,
        ref_smiles: str = "",
        monitor_prefix: str | None = None,
    ) -> Path:
        """Process candidate evaluation arm, sort top leads, and render all primary presentation artifacts."""
        df_after_clean = unscale_to_physical_metrics(df_after)

        # Compute simple Pareto non-dominated rank on ROCS (max), SA (min), QED (max)
        rocs_vals = df_after_clean.get(
            std.ROCS, pd.Series(0.0, index=df_after_clean.index)
        ).values
        sa_vals = df_after_clean.get(
            "Raw_SA",
            df_after_clean.get(std.SA, pd.Series(10.0, index=df_after_clean.index)),
        ).values
        qed_vals = df_after_clean.get(
            std.QED, pd.Series(0.0, index=df_after_clean.index)
        ).values

        # Desirability scoring: ROCS + QED - 0.1 * SA
        desirability = rocs_vals + qed_vals - 0.10 * sa_vals
        df_after_clean["Desirability_Rank_Score"] = desirability
        df_sorted = df_after_clean.sort_values(
            by="Desirability_Rank_Score", ascending=False
        ).reset_index(drop=True)

        # Save top leads
        top_leads_tsv = self.paths.root / "top_prioritized_leads.tsv"
        df_sorted.to_csv(top_leads_tsv, sep="\t", index=False)

        # Render 2D leads
        depicter = AlignedDepicter(ref_smiles)
        depicter.render_leads(
            df_sorted,
            output_png=self.paths.root / "lead_structures_2d.png",
            max_leads=5,
        )

        # Render comparative distributions
        render_property_distributions(
            df_after=df_after_clean,
            df_before=df_before,
            output_png=self.paths.root / "property_distributions.png",
            ref_mol=ref_smiles,
        )

        # Render training curves
        if monitor_prefix:
            fit_tsv = Path(f"{monitor_prefix}_fit.tsv")
            if fit_tsv.is_file():
                render_training_curves(
                    fit_tsv, output_png=self.paths.root / "rl_training_curves.png"
                )

        # Write summary.md
        summary_md = self.paths.root / "summary.md"
        top5 = df_sorted.head(5)
        lines = [
            f"# DrugEx Run Summary: {self.paths.root.name}",
            "",
            f"- **Target Molecule**: `{ref_smiles[:45]}...`",
            f"- **Total Candidates Evaluated**: `{len(df_after_clean)}`",
            "",
            "## Presentation Deliverables",
            "- [Lead Structures 2D](lead_structures_2d.png)",
            "- [Property Distributions (2x5 Profile)](property_distributions.png)",
            "- [RL Training Curves](rl_training_curves.png)",
            "- [Top Prioritized Leads (TSV)](top_prioritized_leads.tsv)",
            "",
            "## Top Prioritized Chemical Leads",
            "",
            "| Rank | ROCS TanimotoCombo | SA Score | QED | MW | Canonical SMILES |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
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
