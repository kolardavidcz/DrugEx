from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, Lipinski, QED, rdDepictor, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score


def get_universal_rigid_anchor(mol: Chem.Mol) -> list[int]:
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
        [idx for idx in f if mol.GetAtomWithIdx(idx).GetSymbol() != "H"]
        for f in frags
    ]
    largest_frag = max(heavy_frags, key=len, default=[])
    return sorted(largest_frag)


def render_reference_ensemble(
    sdf_path: Path | str,
    output_png: Path | str | None = None,
) -> Path:
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

    # Panel B: Most extended state
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_mol(ax2, mols[max_idx], "#f59e0b", "#f59e0b", lw=2.4, pt_size=40, label_core=True)
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

    # Panel C: Core-aligned superposition
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    for i, m in enumerate(mols):
        c = cmap(norm(de_vals[i]))
        _draw_mol(ax3, m, c, c, alpha=0.75, lw=1.6, pt_size=20, label_core=True)
    _draw_mol(ax3, mols[min_idx], "#0f172a", "#0f172a", alpha=0.95, lw=2.8, pt_size=38, label_core=True)
    ax3.set_title(
        f"(C) Core-Aligned Ensemble ({len(mols)} Centroids)\nPharmacophore Envelope & Rotameric Fanning",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
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

    def __init__(self, reference: str | Chem.Mol) -> None:
        self.ref_mol = Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
        self.ref_core = MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            rdDepictor.Compute2DCoords(self.ref_mol)
            self.ref_core = self.ref_mol

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
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
    molecules: Sequence[str | Chem.Mol] | pd.DataFrame,
    legends: Sequence[str] | None = None,
    output_png: Path | str | None = None,
    reference: str | Chem.Mol | None = None,
    max_leads: int = 5,
) -> Path | None:
    target_png = Path(output_png) if output_png else Path("lead_structures_2d.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    plot_mols: list[Chem.Mol] = []
    plot_legends: list[str] = []

    # 1. Primary Query Seed Benchmark
    ref_m = Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
    if ref_m is not None:
        ref_clean = Chem.RemoveHs(Chem.Mol(ref_m))
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

    # 2. Prioritized Elite Leads
    if isinstance(molecules, pd.DataFrame):
        df_leads = molecules.head(max_leads)
        for idx, row in df_leads.iterrows():
            smi = str(row.get("SMILES", ""))
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                plot_mols.append(m)
                rocs = float(row.get("ROCS", 0.0))
                des = float(row.get("Desirability", row.get("Des", 0.0)))
                mw = float(row.get("MW", Descriptors.MolWt(m)))
                qed_v = float(row.get("QED", QED.qed(m)))
                sa_v = float(row.get("SA", calculate_sa_score(m)))
                lead_num = idx + 1 if isinstance(idx, int) else len(plot_mols) - 1
                plot_legends.append(
                    f"Lead #{lead_num}\n"
                    f"ROCS: {rocs:.2f} | Des: {des:.2f}\n"
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

    depicter = AlignedDepicter(reference=ref_m or plot_mols[0])
    depicter.create_aligned_grid(
        plot_mols,
        legends=plot_legends,
        output_path=target_png,
        mols_per_row=min(3, len(plot_mols)),
        sub_img_size=(340, 260),
    )
    return target_png


def render_training_curves(
    fit_tsv_path: Path | str,
    output_png: Path | str | None = None,
) -> Path | None:
    tsv_file = Path(fit_tsv_path)
    if not tsv_file.exists():
        return None

    df_fit = pd.read_csv(tsv_file, sep="\t")
    if df_fit.empty or "Epoch" not in df_fit.columns:
        return None

    target_png = Path(output_png) if output_png else tsv_file.parent / "rl_training_curves.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)

    metric_styles: dict[str, tuple[str, str, str]] = {
        "loss_train": ("#e63946", "-", "Training Loss"),
        "valid_ratio": ("#2a9d8f", "-", "Valid Ratio"),
        "unique_ratio": ("#e9c46a", "--", "Unique Ratio"),
        "desired_ratio": ("#8b5cf6", "-.", "Desired Ratio"),
    }

    plotted = False
    for col, (color, ls, label) in metric_styles.items():
        if col in df_fit.columns:
            ax.plot(
                df_fit["Epoch"],
                df_fit[col],
                color=color,
                linestyle=ls,
                linewidth=2.0,
                label=label,
            )
            plotted = True

    if not plotted:
        num_cols = [c for c in df_fit.columns if c != "Epoch" and pd.api.types.is_numeric_dtype(df_fit[c])]
        for col in num_cols[:4]:
            ax.plot(df_fit["Epoch"], df_fit[col], linewidth=1.8, label=col)

    ax.set_title("Graph-Transformer Reinforcement Learning Dynamics", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png


def find_sa_column(columns: Sequence[str]) -> str | None:
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
    for col in columns:
        c_up = col.upper()
        if any(k in c_up for k in ("ROCS", "TANIMOTOCOMBO", "SHAPE_SIMILARITY", "SIMILARITY_3D")):
            return col
    return None


def render_epoch_pareto_evolution(
    smiles_tsv_path: Path | str,
    output_png: Path | str | None = None,
    max_grid_epochs: int = 6,
    maximize: Sequence[bool] = (True, True),
) -> Path | None:
    tsv_file = Path(smiles_tsv_path)
    if not tsv_file.exists():
        return None

    df = pd.read_csv(tsv_file, sep="\t")
    if df.empty or "Epoch" not in df.columns or "SMILES" not in df.columns:
        return None

    rocs_name = find_rocs_column(df.columns)
    if not rocs_name:
        return None

    out_dir = Path(output_png).parent if output_png else tsv_file.parent
    target_png = Path(output_png) if output_png else out_dir / "pareto_epoch_evolution.png"
    target_png.parent.mkdir(parents=True, exist_ok=True)

    # Detect SA desirability ("SA" column in [0, 1]) or fallback to raw SAscore
    if "SA" in df.columns:
        sa_name = "SA"
    elif "SAscore" in df.columns:
        sa_name = "SAscore"
    else:
        sa_name = "SAscore"
        mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in df["SMILES"]]
        sa_scores = []
        for m in mols:
            try:
                sa_scores.append(float(calculate_sa_score(m)) if m else np.nan)
            except Exception:
                sa_scores.append(np.nan)
        df[sa_name] = sa_scores

    valid_df = df.dropna(subset=[rocs_name, sa_name]).copy()
    if valid_df.empty:
        return None

    epochs = sorted(int(e) for e in valid_df["Epoch"].unique())
    if not epochs:
        return None

    # Global coordinate boundaries for constant scale across all epoch plots
    x_min, x_max = float(valid_df[sa_name].min()), float(valid_df[sa_name].max())
    y_min, y_max = float(valid_df[rocs_name].min()), float(valid_df[rocs_name].max())
    x_pad = 0.05 * (x_max - x_min) if (x_max > x_min) else 0.5
    y_pad = 0.05 * (y_max - y_min) if (y_max > y_min) else 0.05
    x_limits = (max(0.0 if maximize[1] else 1.0, x_min - x_pad), x_max + x_pad)
    y_limits = (max(0.0, y_min - y_pad), min(2.0, y_max + y_pad))

    # 1. Generate Individual Pareto Plot for EVERY Epoch (e.g. 30 plots for 30 epochs)
    by_epoch_dir = out_dir / "pareto_by_epoch"
    by_epoch_dir.mkdir(parents=True, exist_ok=True)
    generated_epoch_pngs: list[Path] = []

    mult = np.array([1.0 if m else -1.0 for m in maximize], dtype=np.float64)

    for ep in epochs:
        sub = valid_df[valid_df["Epoch"] == ep].copy()
        if len(sub) < 2:
            continue

        fig_ep, ax_ep = plt.subplots(figsize=(8.5, 5.8), dpi=180)
        scores = sub[[rocs_name, sa_name]].to_numpy(dtype=float)

        # Non-dominated Front 0: Maximize ROCS, Maximize SA Desirability (maximize=[True, True])
        scaled = scores * mult
        is_front0 = np.ones(len(scores), dtype=bool)
        for i in range(len(scores)):
            dominated = np.any(
                np.all(scaled >= scaled[i], axis=1) & np.any(scaled > scaled[i], axis=1)
            )
            if dominated:
                is_front0[i] = False

        front_0 = sub[is_front0].sort_values(by=sa_name, ascending=not maximize[1]).drop_duplicates(subset=[sa_name, rocs_name])

        # Second-tier Front 1 (Dominated Tier)
        sub_rem = sub[~is_front0]
        if len(sub_rem) >= 2:
            scores_rem = sub_rem[[rocs_name, sa_name]].to_numpy(dtype=float)
            scaled_rem = scores_rem * mult
            is_front1 = np.ones(len(scores_rem), dtype=bool)
            for i in range(len(scores_rem)):
                dom1 = np.any(
                    np.all(scaled_rem >= scaled_rem[i], axis=1) & np.any(scaled_rem > scaled_rem[i], axis=1)
                )
                if dom1:
                    is_front1[i] = False
            front_1 = sub_rem[is_front1].sort_values(by=sa_name, ascending=not maximize[1]).drop_duplicates(subset=[sa_name, rocs_name])
            others = sub_rem[~is_front1]
        else:
            front_1 = pd.DataFrame(columns=sub.columns)
            others = sub_rem

        # Dominated points
        if not others.empty:
            ax_ep.scatter(others[sa_name], others[rocs_name], c="#94a3b8", alpha=0.35, s=30, label="Dominated Candidates", zorder=2)

        # Front 1 (Dominated Tier) - Dashed curve
        if not front_1.empty:
            ax_ep.scatter(front_1[sa_name], front_1[rocs_name], color="#f43f5e", s=45, alpha=0.85, zorder=3)
            if len(front_1) > 1:
                ax_ep.plot(front_1[sa_name], front_1[rocs_name], color="#f43f5e", linestyle="--", linewidth=1.8, label="Front 1 (Dominated Tier)", zorder=3)

        # Front 0 (Optimal Frontier) - Solid curve connecting points
        if not front_0.empty:
            ax_ep.scatter(front_0[sa_name], front_0[rocs_name], color="#0284c7", s=75, edgecolors="#0f172a", linewidth=1.1, zorder=5)
            if len(front_0) > 1:
                ax_ep.plot(front_0[sa_name], front_0[rocs_name], color="#0284c7", linestyle="-", linewidth=2.4, label="Front 0 (Optimal Frontier)", zorder=4)
            else:
                ax_ep.scatter([], [], color="#0284c7", label="Front 0 (Optimal Frontier)")

        ax_ep.set_xlim(x_limits)
        ax_ep.set_ylim(y_limits)
        ax_ep.set_title(f"Epoch {ep} Pareto Frontier (Optimal Trade-off & Dominated Tiers)", fontsize=11, fontweight="bold", pad=12)
        xlabel = (
            "Synthetic Accessibility Desirability (Higher is Better)"
            if maximize[1]
            else "Synthetic Accessibility (SAscore / Lower is Easier)"
        )
        ax_ep.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax_ep.set_ylabel("3D Gaussian ROCS Overlap (TanimotoCombo)", fontsize=10, fontweight="bold")
        ax_ep.grid(True, linestyle=":", alpha=0.6)
        ax_ep.legend(loc="lower left", framealpha=0.9, fontsize=8.5)

        # Metric summary callout box
        max_r = front_0[rocs_name].max() if not front_0.empty else 0.0
        best_s = (
            front_0[sa_name].max() if maximize[1] else front_0[sa_name].min()
        ) if not front_0.empty else 0.0
        sa_metric = "Peak SA Desirability" if maximize[1] else "Best SA"
        stats_text = (
            f"Epoch {ep}/{len(epochs)}\n"
            f"Front 0 Size: {len(front_0)}\n"
            f"Peak ROCS: {max_r:.2f}\n"
            f"{sa_metric}: {best_s:.2f}"
        )
        ax_ep.text(0.97, 0.95, stats_text, transform=ax_ep.transAxes, fontsize=8.5, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9))

        plt.tight_layout()
        ep_png = by_epoch_dir / f"epoch_{ep:02d}.png"
        plt.savefig(ep_png, dpi=180, bbox_inches="tight")
        plt.close(fig_ep)
        generated_epoch_pngs.append(ep_png)

    # 2. Interactive HTML Epoch Scrubber Viewer
    html_path = out_dir / "pareto_epoch_viewer.html"
    epoch_files_js = [f"pareto_by_epoch/epoch_{ep:02d}.png" for ep in epochs]
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Reinforcement Learning Epoch Pareto Evolution</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; text-align: center; }}
        h1 {{ font-size: 1.4rem; margin-bottom: 8px; color: #38bdf8; }}
        .controls {{ display: flex; align-items: center; justify-content: center; gap: 16px; margin: 16px auto; max-width: 600px; }}
        button {{ background: #0284c7; color: white; border: none; padding: 8px 18px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 0.9rem; }}
        button:hover {{ background: #0369a1; }}
        input[type=range] {{ flex-grow: 1; accent-color: #38bdf8; cursor: pointer; }}
        .counter {{ font-size: 1.1rem; font-weight: bold; min-width: 100px; color: #f59e0b; }}
        .img-container {{ max-width: 900px; margin: 12px auto; background: white; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); overflow: hidden; }}
        img {{ width: 100%; height: auto; display: block; }}
        .hint {{ font-size: 0.85rem; color: #94a3b8; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>Micro-Evolution of Pareto Frontier Across RL Training Epochs</h1>
    <div class="controls">
        <button id="prevBtn">&larr; Prev</button>
        <button id="playBtn">Play &#9654;</button>
        <button id="nextBtn">Next &rarr;</button>
        <input type="range" id="slider" min="1" max="{len(epochs)}" value="1">
        <span class="counter" id="epochLabel">Epoch 1/{len(epochs)}</span>
    </div>
    <div class="img-container">
        <img id="paretoImg" src="{epoch_files_js[0]}" alt="Epoch Pareto Frontier">
    </div>
    <div class="hint">Use slider or left/right arrow keys to inspect the frontier shifting over epochs.</div>
    <script>
        const images = {epoch_files_js};
        let currentIdx = 0;
        let isPlaying = false;
        let playInterval = null;

        const imgEl = document.getElementById("paretoImg");
        const sliderEl = document.getElementById("slider");
        const labelEl = document.getElementById("epochLabel");
        const playBtn = document.getElementById("playBtn");

        function showEpoch(idx) {{
            currentIdx = Math.max(0, Math.min(images.length - 1, idx));
            imgEl.src = images[currentIdx];
            sliderEl.value = currentIdx + 1;
            labelEl.textContent = `Epoch ${{currentIdx + 1}}/${{images.length}}`;
        }}

        document.getElementById("prevBtn").addEventListener("click", () => showEpoch(currentIdx - 1));
        document.getElementById("nextBtn").addEventListener("click", () => showEpoch(currentIdx + 1));
        sliderEl.addEventListener("input", (e) => showEpoch(parseInt(e.target.value) - 1));

        function togglePlay() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playBtn.innerHTML = "Play &#9654;";
                isPlaying = false;
            }} else {{
                playInterval = setInterval(() => {{
                    currentIdx = (currentIdx + 1) % images.length;
                    showEpoch(currentIdx);
                }}, 350);
                playBtn.innerHTML = "Pause &#10074;&#10074;";
                isPlaying = true;
            }}
        }}

        playBtn.addEventListener("click", togglePlay);
        document.addEventListener("keydown", (e) => {{
            if (e.key === "ArrowLeft") showEpoch(currentIdx - 1);
            if (e.key === "ArrowRight") showEpoch(currentIdx + 1);
            if (e.key === " ") {{ e.preventDefault(); togglePlay(); }}
        }});
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 3. Clean Multi-Panel Progression Grid (2x3 or 2x2 side-by-side epochs)
    if len(epochs) > max_grid_epochs:
        grid_indices = np.linspace(0, len(epochs) - 1, max_grid_epochs, dtype=int)
        grid_epochs = [epochs[i] for i in grid_indices]
    else:
        grid_epochs = epochs

    n_panels = len(grid_epochs)
    ncols = 3 if n_panels >= 3 else n_panels
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.2 * nrows), dpi=200)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle("Reinforcement Learning Pareto Frontier Progression Across Epochs", fontsize=14, fontweight="bold", y=0.98)

    for idx, ep in enumerate(grid_epochs):
        ax = axes_flat[idx]
        sub = valid_df[valid_df["Epoch"] == ep].copy()
        if len(sub) < 2:
            continue

        scores = sub[[rocs_name, sa_name]].to_numpy(dtype=float)
        is_front0 = np.ones(len(scores), dtype=bool)
        for i in range(len(scores)):
            dominated = np.any(
                (scores[:, 0] >= scores[i, 0]) & (scores[:, 1] <= scores[i, 1]) &
                ((scores[:, 0] > scores[i, 0]) | (scores[:, 1] < scores[i, 1]))
            )
            if dominated:
                is_front0[i] = False

        front_0 = sub[is_front0].sort_values(by=sa_name).drop_duplicates(subset=[sa_name, rocs_name])
        sub_rem = sub[~is_front0]

        # Dominated points
        if not sub_rem.empty:
            ax.scatter(sub_rem[sa_name], sub_rem[rocs_name], c="#94a3b8", alpha=0.35, s=20, label="Dominated")

        # Front 0 (Optimal Frontier)
        if not front_0.empty:
            ax.scatter(front_0[sa_name], front_0[rocs_name], color="#0284c7", s=55, edgecolors="#0f172a", linewidth=0.9, zorder=5)
            if len(front_0) > 1:
                ax.plot(front_0[sa_name], front_0[rocs_name], color="#0284c7", linestyle="-", linewidth=2.0, label="Front 0", zorder=4)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_title(f"Epoch {ep} Frontier (Size: {len(front_0)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("SAscore (Lower is Easier)", fontsize=8.5)
        ax.set_ylabel("ROCS (TanimotoCombo)", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return target_png


def render_property_distributions(
    df: pd.DataFrame,
    output_png: Path | str | None = None,
    ref_mol: Chem.Mol | str | None = None,
) -> Path | None:
    if df.empty or "SMILES" not in df.columns:
        return None

    target_png = Path(output_png) if output_png else Path("property_distributions.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    calc_df = df.copy()
    if "Valid" in calc_df.columns:
        valid_df = calc_df[calc_df["Valid"].astype(bool)].copy()
    else:
        valid_df = calc_df.copy()

    if valid_df.empty:
        return None

    # Reference seed benchmarks
    ref_m = Chem.MolFromSmiles(ref_mol) if isinstance(ref_mol, str) else ref_mol
    seed_mw = float(Descriptors.MolWt(ref_m)) if ref_m else 458.4
    seed_logp = float(Descriptors.MolLogP(ref_m)) if ref_m else 2.23
    seed_tpsa = float(rdMolDescriptors.CalcTPSA(ref_m)) if ref_m else 197.4
    seed_qed = float(QED.qed(ref_m)) if ref_m else 0.21
    seed_sa = float(calculate_sa_score(ref_m)) if ref_m else 3.74
    seed_hbd = int(Lipinski.NumHDonors(ref_m)) if ref_m else 8
    seed_rot = int(Lipinski.NumRotatableBonds(ref_m)) if ref_m else 3
    seed_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_m, 2, nBits=2048) if ref_m else None

    # Dynamically compute any missing property columns
    mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in valid_df["SMILES"]]

    if "MW" not in valid_df.columns:
        valid_df["MW"] = [Descriptors.MolWt(m) if m else np.nan for m in mols]
    if "LogP" not in valid_df.columns and "logP" not in valid_df.columns:
        valid_df["LogP"] = [Descriptors.MolLogP(m) if m else np.nan for m in mols]
    elif "logP" in valid_df.columns and "LogP" not in valid_df.columns:
        valid_df["LogP"] = valid_df["logP"]

    if "TPSA" not in valid_df.columns:
        valid_df["TPSA"] = [rdMolDescriptors.CalcTPSA(m) if m else np.nan for m in mols]
    if "QED" not in valid_df.columns:
        valid_df["QED"] = [QED.qed(m) if m else np.nan for m in mols]
    if "SA" not in valid_df.columns or valid_df["SA"].nunique() <= 3:
        valid_df["SA"] = [calculate_sa_score(m) if m else np.nan for m in mols]
    if "HBD" not in valid_df.columns:
        valid_df["HBD"] = [Lipinski.NumHDonors(m) if m else 0 for m in mols]
    if "RotBonds" not in valid_df.columns:
        valid_df["RotBonds"] = [Lipinski.NumRotatableBonds(m) if m else 0 for m in mols]

    if "Tanimoto_to_Seed" not in valid_df.columns:
        if seed_fp is not None:
            sims = []
            for m in mols:
                if m is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                    sims.append(float(DataStructs.TanimotoSimilarity(seed_fp, fp)))
                else:
                    sims.append(0.0)
            valid_df["Tanimoto_to_Seed"] = sims
        else:
            valid_df["Tanimoto_to_Seed"] = 0.0

    rocs_col = find_rocs_column(valid_df.columns)
    rocs_data = valid_df[rocs_col] if rocs_col else pd.Series([0.0] * len(valid_df))

    # Construct the 3x3 diagnostic & multi-objective grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), dpi=200)
    fig.suptitle(
        "Ligand-Based De Novo Drug Design: Multi-Objective & Diagnostic Profiles",
        fontsize=15,
        weight="bold",
        y=0.98,
    )

    # Subplot 1: 3D ROCS Distribution (Row 0, Col 0 - Active RL Scorer)
    ax = axes[0, 0]
    ax.hist(rocs_data.dropna(), bins=15, color="#2A9D8F", edgecolor="black", alpha=0.80)
    ax.axvline(1.00, color="#E76F51", linestyle="--", linewidth=1.5, label="High Overlap (1.00)")
    ax.set_title("3D Gaussian ROCS (TanimotoCombo)", fontsize=11, weight="bold")
    ax.set_xlabel("ROCS Score (0 to 2.0)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: QED Drug-likeness Distribution (Row 0, Col 1 - Secondary Objective)
    ax = axes[0, 1]
    ax.hist(valid_df["QED"].dropna(), bins=15, color="#457B9D", edgecolor="black", alpha=0.80)
    ax.axvline(seed_qed, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({seed_qed:.2f})")
    ax.set_title("Drug-likeness Estimate (QED)", fontsize=11, weight="bold")
    ax.set_xlabel("QED Score (0 to 1.0)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Synthetic Accessibility Floor (Row 0, Col 2 - Active Floor)
    ax = axes[0, 2]
    ax.hist(valid_df["SA"].dropna(), bins=15, color="#E9C46A", edgecolor="black", alpha=0.80)
    ax.axvspan(1.0, 5.5, color="#2A9D8F", alpha=0.15, label="Feasible (<= 5.5)")
    ax.axvline(5.5, color="#2A9D8F", linestyle="--", linewidth=1.5)
    ax.axvline(6.0, color="#E76F51", linestyle=":", linewidth=1.5, label="Penalized (> 6.0)")
    ax.axvline(seed_sa, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({seed_sa:.2f})")
    ax.set_title("Synthetic Accessibility (SAscore)", fontsize=11, weight="bold")
    ax.set_xlabel("SA Score (1 = Easy, 10 = Hard)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 4: Molecular Weight & Dynamic Envelope (Row 1, Col 0 - Active Envelope)
    ax = axes[1, 0]
    ax.hist(valid_df["MW"].dropna(), bins=15, color="#264653", edgecolor="black", alpha=0.80)
    mw_lo, mw_hi = round(0.70 * seed_mw, 1), round(1.30 * seed_mw, 1)
    ax.axvspan(mw_lo, mw_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{mw_lo}, {mw_hi}]")
    ax.axvline(seed_mw, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({seed_mw:.1f})")
    ax.set_title("Molecular Weight (MW)", fontsize=11, weight="bold")
    ax.set_xlabel("MW (Da)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 5: CLogP Lipophilicity & Dynamic Envelope (Row 1, Col 1 - Active Envelope)
    ax = axes[1, 1]
    ax.hist(valid_df["LogP"].dropna(), bins=15, color="#1D3557", edgecolor="black", alpha=0.80)
    logp_lo, logp_hi = round(seed_logp - 1.5, 2), round(seed_logp + 1.5, 2)
    ax.axvspan(logp_lo, logp_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{logp_lo}, {logp_hi}]")
    ax.axvline(seed_logp, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({seed_logp:.2f})")
    ax.set_title("Calculated Lipophilicity (CLogP)", fontsize=11, weight="bold")
    ax.set_xlabel("CLogP")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Subplot 6: Polar Surface Area & Dynamic Envelope (Row 1, Col 2 - Active Envelope)
    ax = axes[1, 2]
    ax.hist(valid_df["TPSA"].dropna(), bins=15, color="#6D597A", edgecolor="black", alpha=0.80)
    tpsa_lo, tpsa_hi = round(0.70 * seed_tpsa, 1), round(1.30 * seed_tpsa, 1)
    ax.axvspan(tpsa_lo, tpsa_hi, color="#457B9D", alpha=0.15, label=f"Envelope [{tpsa_lo}, {tpsa_hi}]")
    ax.axvline(seed_tpsa, color="#E63946", linestyle="--", linewidth=1.5, label=f"Query ({seed_tpsa:.1f})")
    ax.set_title("Topological Polar Surface Area (TPSA)", fontsize=11, weight="bold")
    ax.set_xlabel("TPSA (Å²)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    # =========================================================================
    # ROW 2: UNOPTIMIZED / ORTHOGONAL DIAGNOSTIC PROPERTIES (MUTED GRAY BACKGROUND)
    # =========================================================================
    diagnostic_bg = "#F0F2F5"

    # Subplot 7: Hydrogen Bond Donors (Row 2, Col 0 - Diagnostic Epitope Loss)
    ax = axes[2, 0]
    ax.set_facecolor(diagnostic_bg)
    max_hbd = max(int(valid_df["HBD"].max()) if not valid_df.empty else 0, seed_hbd)
    bins_hbd = np.arange(0, max_hbd + 2) - 0.5
    ax.hist(valid_df["HBD"], bins=bins_hbd, color="#6C757D", edgecolor="black", alpha=0.85, rwidth=0.8)
    ax.axvline(seed_hbd, color="#E63946", linestyle="--", linewidth=1.8, label=f"Query ({seed_hbd})")
    mean_hbd = float(valid_df["HBD"].mean())
    ax.axvline(mean_hbd, color="#212529", linestyle=":", linewidth=1.8, label=f"Mean ({mean_hbd:.2f})")
    ax.set_title("[UNOPTIMIZED] H-Bond Donors (Epitope Loss)", fontsize=11, weight="bold", color="#212529")
    ax.set_xlabel("HBD Count (Loss vs. Bioactive Query)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0")

    # Subplot 8: Rotatable Bonds (Row 2, Col 1 - Conformational Flexibility)
    ax = axes[2, 1]
    ax.set_facecolor(diagnostic_bg)
    max_rot = max(int(valid_df["RotBonds"].max()) if not valid_df.empty else 0, seed_rot)
    bins_rot = np.arange(0, max_rot + 2) - 0.5
    ax.hist(valid_df["RotBonds"], bins=bins_rot, color="#495057", edgecolor="black", alpha=0.85, rwidth=0.8)
    ax.axvline(seed_rot, color="#E63946", linestyle="--", linewidth=1.8, label=f"Query ({seed_rot})")
    mean_rot = float(valid_df["RotBonds"].mean())
    ax.axvline(mean_rot, color="#212529", linestyle=":", linewidth=1.8, label=f"Mean ({mean_rot:.2f})")
    ax.set_title("[UNOPTIMIZED] Rotatable Bonds (Flexibility)", fontsize=11, weight="bold", color="#212529")
    ax.set_xlabel("Rotatable Bonds Count (Max Allowed: 25)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0")

    # Subplot 9: 2D Morgan Tanimoto to Query (Row 2, Col 2 - Scaffold Hopping)
    ax = axes[2, 2]
    ax.set_facecolor(diagnostic_bg)
    sim_col = valid_df["Tanimoto_to_Seed"]
    ax.hist(sim_col.dropna(), bins=15, color="#343A40", edgecolor="black", alpha=0.85)
    ax.axvline(0.35, color="#0077B6", linestyle=":", linewidth=1.8, label="Hop Threshold (<= 0.35)")
    mean_sim = float(sim_col.mean())
    ax.axvline(mean_sim, color="#E63946", linestyle="--", linewidth=1.8, label=f"Mean Sim ({mean_sim:.3f})")
    ax.set_title("[UNOPTIMIZED] 2D ECFP4 Tanimoto to Seed", fontsize=11, weight="bold", color="#212529")
    ax.set_xlabel("2D Fingerprint Similarity (Scaffold Divergence)")
    ax.set_ylabel("Molecule Count")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#CBD5E0")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(target_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target_png


def render_pareto_frontier(
    df: pd.DataFrame,
    output_png: Path | str | None = None,
    rocs_col: str | None = None,
    sa_col: str | None = None,
) -> Path | None:
    if df.empty or "Pareto_Rank" not in df.columns:
        return None

    target_png = Path(output_png) if output_png else Path("pareto_frontier.png")
    target_png.parent.mkdir(parents=True, exist_ok=True)

    r_col = rocs_col or find_rocs_column(df.columns)
    s_col = sa_col or find_sa_column(df.columns)
    if not r_col or not s_col:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=200)

    front_0 = df[df["Pareto_Rank"] == 0].sort_values(by=s_col).drop_duplicates(subset=[s_col, r_col])
    front_1 = df[df["Pareto_Rank"] == 1].sort_values(by=s_col).drop_duplicates(subset=[s_col, r_col])
    others = df[df["Pareto_Rank"] > 1]

    # Dominated background points
    if not others.empty:
        ax.scatter(others[s_col], others[r_col], c="#94a3b8", alpha=0.40, s=25, label="Dominated Candidates")

    # Front 1 (Dominated Tier) - dashed line
    if not front_1.empty:
        ax.scatter(front_1[s_col], front_1[r_col], c="#f43f5e", alpha=0.85, s=45, label="Front 1 (Dominated Tier)")
        if len(front_1) > 1:
            ax.plot(front_1[s_col], front_1[r_col], color="#f43f5e", linestyle="--", linewidth=1.8)

    # Front 0 (Optimal Frontier) - solid line connecting non-dominated frontier
    if not front_0.empty:
        ax.scatter(front_0[s_col], front_0[r_col], c="#0284c7", alpha=0.95, s=70, edgecolors="#0f172a", linewidth=1.0, zorder=5)
        if len(front_0) > 1:
            ax.plot(front_0[s_col], front_0[r_col], color="#0284c7", linestyle="-", linewidth=2.4, label="Front 0 (Optimal Frontier)", zorder=4)
        else:
            ax.scatter([], [], color="#0284c7", label="Front 0 (Optimal Frontier)")

    ax.set_xlabel(f"{s_col} (Synthesizability)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{r_col} (3D Shape Overlap)", fontsize=10, fontweight="bold")
    ax.set_title("Multi-Objective Pareto Trade-off Frontier", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(target_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_png
