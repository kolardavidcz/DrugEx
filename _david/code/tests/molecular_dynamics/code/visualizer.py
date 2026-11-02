"""Publication visualizer for multi-target validation and HBD collapse analysis.

Renders:
1. Cross-target binding affinity heatmap (6 ligands x 3 distinct target classes).
2. Hydrogen-bond donor/acceptor profile highlighting the empirical HBD collapse.
3. Multi-target selectivity and interaction radar/scatter comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _david.validation.code.docking_engine import DockingResult
from _david.validation.code.prepare_ligands import LigandRecord
from _david.validation.code.prepare_targets import TargetMetadata


class ValidationVisualizer:
    """Generates publication-quality charts for cross-target docking and interaction profiles."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialize ValidationVisualizer.

        Args:
            output_dir: Directory where figures will be saved.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configurations for publication elegance
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        })

    def plot_affinity_heatmap(
        self, results: List[DockingResult], targets: Dict[str, TargetMetadata]
    ) -> Path:
        """Plot binding affinity matrix (kcal/mol) with target class annotations."""
        # [PACKAGE: matplotlib] WHY: Publication-grade comparative heatmaps. WHAT: imshow with explicit cell value annotations. HOW: ax.imshow + text annotations.
        ligand_ids = sorted(list(set(r.ligand_id for r in results)))
        target_ids = list(targets.keys())

        # Build affinity matrix
        data = np.zeros((len(ligand_ids), len(target_ids)))
        lookup = {(r.ligand_id, r.target_id): r.affinity_kcal_mol for r in results}

        for i, lid in enumerate(ligand_ids):
            for j, tid in enumerate(target_ids):
                data[i, j] = lookup.get((lid, tid), 0.0)

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(data, cmap="YlGnBu_r", aspect="auto")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Binding Affinity ΔG (kcal/mol, more negative = tighter)", fontsize=10)

        # Labels
        target_labels = [
            f"{targets[tid].pdb_id}: {targets[tid].common_name.split('(')[0].strip()}\n[{targets[tid].biological_class}]"
            for tid in target_ids
        ]
        ax.set_xticks(np.arange(len(target_ids)))
        ax.set_yticks(np.arange(len(ligand_ids)))
        ax.set_xticklabels(target_labels, rotation=15, ha="right", fontweight="bold")
        ax.set_yticklabels(ligand_ids, fontweight="bold")

        # Cell annotations
        for i in range(len(ligand_ids)):
            for j in range(len(target_ids)):
                val = data[i, j]
                color = "white" if val < -7.0 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

        ax.set_title("Multi-Target Binding Affinity Matrix across Distinct Biological Classes", pad=15)
        plt.tight_layout()

        out_path = self.output_dir / "affinity_matrix_heatmap.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_hbond_collapse_profile(self, results: List[DockingResult]) -> Path:
        """Plot Protein-Donor vs. Ligand-Donor H-bonds demonstrating the HBD loss."""
        # Group by ligand
        ligand_ids = sorted(list(set(r.ligand_id for r in results)))
        target_ids = sorted(list(set(r.target_id for r in results)))

        fig, axes = plt.subplots(1, len(target_ids), figsize=(14, 5), sharey=True)
        if len(target_ids) == 1:
            axes = [axes]

        for ax, tid in zip(axes, target_ids):
            target_res = [r for r in results if r.target_id == tid]
            lookup = {r.ligand_id: r for r in target_res}

            prot_donors = [lookup[lid].interactions.num_hbonds_prot_donor for lid in ligand_ids]
            lig_donors = [lookup[lid].interactions.num_hbonds_lig_donor for lid in ligand_ids]

            x = np.arange(len(ligand_ids))
            width = 0.35

            ax.bar(x - width / 2, prot_donors, width, label="Protein Donor (H-bond Acceptor Ligand)", color="#2b5c8f")
            ax.bar(x + width / 2, lig_donors, width, label="Ligand Donor (H-bond Donor Ligand)", color="#d95f02")

            ax.set_xticks(x)
            ax.set_xticklabels(ligand_ids, rotation=45, ha="right")
            ax.set_title(f"Target: {tid.split('_')[0]}", fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        axes[0].set_ylabel("Number of Hydrogen Bonds", fontsize=11)
        axes[0].legend(loc="upper right", fontsize=8)

        fig.suptitle(
            "Empirical HBD Collapse Analysis: Disappearance of Ligand Proton Donors in De Novo Leads",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        out_path = self.output_dir / "hbond_collapse_profile.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_interaction_radar_and_breakdown(self, results: List[DockingResult]) -> Path:
        """Plot interaction breakdown (Hbonds, Hydrophobic, Pi-stacking) per molecule."""
        df_rows = []
        for r in results:
            df_rows.append({
                "Ligand": r.ligand_id,
                "Target": r.target_id.split("_")[0],
                "Affinity": r.affinity_kcal_mol,
                "Prot_HB": r.interactions.num_hbonds_prot_donor,
                "Lig_HB": r.interactions.num_hbonds_lig_donor,
                "Hydrophobic": r.interactions.num_hydrophobic,
                "Pi_Stacking": r.interactions.num_pi_stacking,
                "Is_Ref": r.is_reference,
            })
        df = pd.DataFrame(df_rows)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: Affinity comparison across targets
        for tid in df["Target"].unique():
            sub = df[df["Target"] == tid]
            ax1.plot(sub["Ligand"], sub["Affinity"], marker="o", label=f"Target {tid}", linewidth=2)

        ax1.set_ylabel("Docking Affinity (kcal/mol)", fontsize=11)
        ax1.set_title("Cross-Target Affinity Profiles (Polypharmacology Spectrum)", fontweight="bold")
        ax1.set_xticklabels(df["Ligand"].unique(), rotation=35, ha="right")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        # Panel 2: Hydrophobic packing vs. total Hbonds
        total_hb = df["Prot_HB"] + df["Lig_HB"]
        ax2.scatter(
            total_hb[df["Is_Ref"]],
            df["Hydrophobic"][df["Is_Ref"]],
            color="#e41a1c",
            s=120,
            label="EGCG Seed (Reference)",
            marker="*",
            zorder=5,
        )
        ax2.scatter(
            total_hb[~df["Is_Ref"]],
            df["Hydrophobic"][~df["Is_Ref"]],
            color="#377eb8",
            s=70,
            alpha=0.8,
            label="De Novo Leads (M4)",
            marker="o",
            zorder=4,
        )

        ax2.set_xlabel("Total Hydrogen Bonds Detected (PLIP)", fontsize=11)
        ax2.set_ylabel("Hydrophobic Contacts (PLIP)", fontsize=11)
        ax2.set_title("Binding Mechanism Shift: Polar H-bonding vs Hydrophobic Packing", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

        plt.tight_layout()
        out_path = self.output_dir / "interaction_mechanism_shift.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
