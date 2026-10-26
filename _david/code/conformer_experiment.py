from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdForceFieldHelpers,
    rdMolAlign,
    rdMolTransforms,
    rdShapeAlign,
)
from rdkit.ML.Cluster import Butina

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.config import DEFAULT_MOLECULE


@dataclass(frozen=True)
class ForceFieldVariant:
    """Specification for conformer optimization force field and solvent dielectric environment."""

    name: str
    display_title: str
    ff_type: str  # "mmff" or "uff"
    dielectric_model: int = 1  # 1 = constant, 2 = distance-dependent
    dielectric_constant: float = 1.0
    energy_window_kcal: float | None = 8.0  # None = no energy cutoff
    rmsd_cutoff: float = 1.25  # Angstroms


class ConformerEnsembleBuilder:
    """Generates and curates non-degenerate conformer ensembles across force field environments."""

    def __init__(self, molecule_smiles: str = DEFAULT_MOLECULE) -> None:
        self.smiles = molecule_smiles
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {molecule_smiles}")
        self.mol_h = Chem.AddHs(mol)

    def generate_raw_conformers(
        self, n_conformers: int = 250, seed: int = 42
    ) -> Chem.Mol:
        """Embed unminimized 3D conformers using RDKit ETKDGv3."""
        mol = Chem.Mol(self.mol_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = 0.50  # Early pruning during embedding
        AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        return mol

    def minimize_and_score(
        self, mol: Chem.Mol, variant: ForceFieldVariant
    ) -> tuple[Chem.Mol, dict[int, float]]:
        """Minimize each conformer in place according to the variant's force field and dielectric."""
        energies: dict[int, float] = {}
        for cid in range(mol.GetNumConformers()):
            if variant.ff_type.lower() == "uff":
                ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            else:
                mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(
                    mol, mmffVariant="MMFF94s"
                )
                mp.SetMMFFDielectricModel(variant.dielectric_model)
                mp.SetMMFFDielectricConstant(variant.dielectric_constant)
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=cid)

            if ff is not None:
                ff.Initialize()
                ff.Minimize(maxIts=500)
                energies[cid] = ff.CalcEnergy()
            else:
                energies[cid] = 1e6
        return mol, energies

    def cluster_and_filter(
        self,
        mol: Chem.Mol,
        energies: dict[int, float],
        variant: ForceFieldVariant,
    ) -> list[int]:
        """Apply energy window and Butina RMSD clustering at the variant's cutoff."""
        min_e = min(energies.values()) if energies else 0.0

        # Step 1: Energy filtering
        if variant.energy_window_kcal is not None:
            valid_cids = [
                cid
                for cid, e in energies.items()
                if (e - min_e) <= variant.energy_window_kcal
            ]
        else:
            valid_cids = list(energies.keys())

        if not valid_cids:
            return [min(energies, key=energies.get)] if energies else []

        valid_cids.sort(key=energies.get)

        # Step 2: Butina RMSD clustering
        n_cids = len(valid_cids)
        if n_cids <= 1:
            return valid_cids

        heavy_mol = Chem.RemoveHs(mol)
        dists: list[float] = []
        for i in range(1, n_cids):
            for j in range(i):
                rmsd = rdMolAlign.GetBestRMS(
                    heavy_mol, heavy_mol, valid_cids[i], valid_cids[j]
                )
                dists.append(rmsd)

        clusters = Butina.ClusterData(
            dists, n_cids, distThresh=variant.rmsd_cutoff, isDistData=True
        )

        # Centroids sorted by energy
        centroids = [valid_cids[c[0]] for c in clusters]
        centroids.sort(key=energies.get)
        return centroids

    def build_ensemble(
        self, variant: ForceFieldVariant, n_raw: int = 250, seed: int = 42
    ) -> tuple[Chem.Mol, list[int], dict[int, float]]:
        """Complete workflow: embed -> minimize under dielectric -> filter & cluster."""
        raw_mol = self.generate_raw_conformers(n_conformers=n_raw, seed=seed)
        min_mol, energies = self.minimize_and_score(raw_mol, variant)
        curated_cids = self.cluster_and_filter(min_mol, energies, variant)
        return min_mol, curated_cids, energies


class ConformerConvergenceAnalyzer:
    """Computes the ROCS TanimotoCombo self-similarity response surface across ensemble sizes."""

    @staticmethod
    def compute_radius_of_gyration(mol: Chem.Mol, conf_id: int) -> float:
        """Compute radius of gyration (Rg in Angstroms) as a measure of conformational expansion."""
        conf = mol.GetConformer(conf_id)
        pts = [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumHeavyAtoms())]
        centroid = np.mean(pts, axis=0)
        sq_dists = [np.sum((p - centroid) ** 2) for p in pts]
        return float(math.sqrt(np.mean(sq_dists)))

    @staticmethod
    def evaluate_pairwise_rocs(
        ref_mol: Chem.Mol,
        ref_cids: Sequence[int],
        query_mol: Chem.Mol,
        query_cids: Sequence[int],
    ) -> tuple[float, float, float]:
        """Align top query conformers against top reference conformers and return max TanimotoCombo."""
        best_tc = 0.0
        best_shape = 0.0
        best_color = 0.0

        for q_cid in query_cids:
            for r_cid in ref_cids:
                try:
                    shape_t, color_t = rdShapeAlign.AlignMol(
                        ref_mol, query_mol, probeConfId=q_cid, refConfId=r_cid
                    )
                    tc = float(shape_t + color_t)
                    if tc > best_tc:
                        best_tc = tc
                        best_shape = float(shape_t)
                        best_color = float(color_t)
                except Exception:
                    continue

        return best_tc, best_shape, best_color

    def run_surface_scan(
        self,
        ref_mol: Chem.Mol,
        ref_cids: list[int],
        query_mol: Chem.Mol,
        query_cids: list[int],
        k_ref_grid: Sequence[int],
        k_gen_grid: Sequence[int],
    ) -> pd.DataFrame:
        """Evaluate the grid of (k_ref, k_gen) combinations and construct the response surface."""
        rows: list[dict[str, object]] = []

        max_r = len(ref_cids)
        max_q = len(query_cids)

        for kr in k_ref_grid:
            actual_kr = min(kr, max_r)
            active_r_cids = ref_cids[:actual_kr]

            for kq in k_gen_grid:
                actual_kq = min(kq, max_q)
                active_q_cids = query_cids[:actual_kq]

                tc, shape_t, color_t = self.evaluate_pairwise_rocs(
                    ref_mol=ref_mol,
                    ref_cids=active_r_cids,
                    query_mol=query_mol,
                    query_cids=active_q_cids,
                )

                rows.append(
                    {
                        "k_ref_requested": kr,
                        "k_gen_requested": kq,
                        "k_ref_actual": actual_kr,
                        "k_gen_actual": actual_kq,
                        "tanimoto_combo": tc,
                        "shape_tanimoto": shape_t,
                        "color_tanimoto": color_t,
                    }
                )

        return pd.DataFrame(rows)


class ConvergenceVisualizer:
    """Renders authoritative 3D surfaces, 2D cross-sections, and conformational property profiles."""

    @staticmethod
    def render_3d_surfaces(
        results_by_variant: dict[str, pd.DataFrame], output_path: Path
    ) -> None:
        """Generate side-by-side 3D surface plots showing ROCS self-similarity convergence."""
        n_vars = len(results_by_variant)
        fig = plt.figure(figsize=(6 * n_vars, 6), dpi=300)

        for idx, (title, df) in enumerate(results_by_variant.items(), start=1):
            ax = fig.add_subplot(1, n_vars, idx, projection="3d")

            # Pivot into meshgrid
            piv = df.pivot(
                index="k_gen_requested",
                columns="k_ref_requested",
                values="tanimoto_combo",
            )
            X = piv.columns.to_numpy()
            Y = piv.index.to_numpy()
            X_mesh, Y_mesh = np.meshgrid(X, Y)
            Z = piv.to_numpy()

            surf = ax.plot_surface(
                X_mesh,
                Y_mesh,
                Z,
                cmap="viridis",
                edgecolor="none",
                alpha=0.88,
                antialiased=True,
            )
            ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel(
                r"$K_{\mathrm{ref}}$ (Reference Basins)", fontsize=9, labelpad=8
            )
            ax.set_ylabel(
                r"$K_{\mathrm{gen}}$ (Candidate Pool)", fontsize=9, labelpad=8
            )
            ax.set_zlabel("ROCS TanimotoCombo", fontsize=9, labelpad=8)
            ax.set_zlim(0.8, 2.05)
            ax.view_init(elev=28, azim=-125)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def render_cross_sectional_recovery(
        results_by_variant: dict[str, pd.DataFrame], output_path: Path
    ) -> None:
        """Generate 2D cross-sectional recovery curves comparing fixed K_ref slices."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # Panel A: Fixed K_ref = 3 (our DrugEx default) across all variants
        ax1 = axes[0]
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

        for (title, df), col in zip(results_by_variant.items(), colors):
            slice_df = df[df["k_ref_requested"] == 3].sort_values("k_gen_requested")
            ax1.plot(
                slice_df["k_gen_requested"],
                slice_df["tanimoto_combo"],
                marker="o",
                linewidth=2.0,
                color=col,
                label=title,
            )

        ax1.axhline(
            2.0, color="gray", linestyle="--", alpha=0.6, label="Ideal Self-Match (2.0)"
        )
        ax1.set_title(
            r"A: Self-Similarity Scaling at Fixed $K_{\mathrm{ref}}=3$",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel(
            r"Number of Generated Candidate Conformers ($K_{\mathrm{gen}}$)",
            fontsize=10,
        )
        ax1.set_ylabel("Max ROCS TanimotoCombo", fontsize=10)
        ax1.set_ylim(0.8, 2.05)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(fontsize=8, loc="lower right")

        # Panel B: Fixed K_gen = 30 across all variants vs K_ref
        ax2 = axes[1]
        for (title, df), col in zip(results_by_variant.items(), colors):
            slice_df = df[df["k_gen_requested"] == 30].sort_values("k_ref_requested")
            ax2.plot(
                slice_df["k_ref_requested"],
                slice_df["tanimoto_combo"],
                marker="s",
                linewidth=2.0,
                color=col,
                label=title,
            )

        ax2.axhline(
            2.0, color="gray", linestyle="--", alpha=0.6, label="Ideal Self-Match (2.0)"
        )
        ax2.set_title(
            r"B: Reference Basin Density Impact at Fixed $K_{\mathrm{gen}}=30$",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel(r"Number of Reference Basins ($K_{\mathrm{ref}}$)", fontsize=10)
        ax2.set_ylabel("Max ROCS TanimotoCombo", fontsize=10)
        ax2.set_ylim(0.8, 2.05)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def render_expansion_profiles(
        expansion_data: list[dict[str, object]], output_path: Path
    ) -> None:
        """Render distribution of Radius of Gyration (Rg) illustrating vacuum collapse vs solvent expansion."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_exp = pd.DataFrame(expansion_data)

        variants = df_exp["variant"].unique()
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

        data_to_plot = [
            df_exp[df_exp["variant"] == v]["radius_of_gyration"].to_numpy()
            for v in variants
        ]

        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=variants,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, col in zip(bplot["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)

        ax.set_title(
            "Conformational Expansion Profile: Vacuum Collapse vs. Solvent Shielding",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(
            r"Heavy Atom Radius of Gyration ($R_g$, $\mathrm{\AA}$)", fontsize=10
        )
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        plt.xticks(rotation=20, ha="right", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_full_study() -> None:
    """Execute the complete conformer self-similarity and dielectric convergence experiment."""
    output_dir = PROJECT_ROOT / "_david" / "output" / "conformer_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING CONFORMER SELF-SIMILARITY & DIELECTRIC CONVERGENCE STUDY")
    print(f"Target: EGCG | Output: {output_dir}")
    print("=" * 80)

    # 5 Rigorous Variants
    variants: list[ForceFieldVariant] = [
        ForceFieldVariant(
            name="MMFF94s_Vacuum",
            display_title="1. MMFF94s (Vacuum, eps=1.0)",
            ff_type="mmff",
            dielectric_model=1,
            dielectric_constant=1.0,
            energy_window_kcal=8.0,
            rmsd_cutoff=1.25,
        ),
        ForceFieldVariant(
            name="UFF_Vacuum",
            display_title="2. UFF (Vacuum)",
            ff_type="uff",
            dielectric_model=1,
            dielectric_constant=1.0,
            energy_window_kcal=8.0,
            rmsd_cutoff=1.25,
        ),
        ForceFieldVariant(
            name="MMFF94s_Pocket",
            display_title="3. MMFF94s (Pocket, eps=6.5)",
            ff_type="mmff",
            dielectric_model=1,
            dielectric_constant=6.5,  # Warshel / Simonson / NMR CSP pocket consensus
            energy_window_kcal=None,  # Relaxed window for physiological environment
            rmsd_cutoff=1.25,
        ),
        ForceFieldVariant(
            name="MMFF94s_Water",
            display_title="4. MMFF94s (Aqueous, eps=78.4)",
            ff_type="mmff",
            dielectric_model=1,
            dielectric_constant=78.4,
            energy_window_kcal=None,  # Relaxed window
            rmsd_cutoff=1.25,
        ),
        ForceFieldVariant(
            name="MMFF94s_DistanceDep",
            display_title="5. MMFF94s (Distance-Dep eps(r)=r)",
            ff_type="mmff",
            dielectric_model=2,  # Model 2 = distance-dependent screening
            dielectric_constant=1.0,
            energy_window_kcal=None,
            rmsd_cutoff=1.25,
        ),
    ]

    builder = ConformerEnsembleBuilder(DEFAULT_MOLECULE)
    analyzer = ConformerConvergenceAnalyzer()

    # Evaluation Grids
    k_ref_grid = [1, 2, 3, 5, 8, 12]
    k_gen_grid = [1, 2, 3, 5, 8, 12, 16, 20, 30]

    results_by_variant: dict[str, pd.DataFrame] = {}
    expansion_data: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for v in variants:
        t0 = time.time()
        print(f"\nEvaluating Variant: {v.name} ({v.display_title})...")

        # Build reference ensemble (seed 42)
        ref_mol, ref_cids, ref_energies = builder.build_ensemble(v, n_raw=200, seed=42)
        # Build independent probe candidate ensemble (seed 9999)
        probe_mol, probe_cids, probe_energies = builder.build_ensemble(
            v, n_raw=200, seed=9999
        )

        print(
            f"  -> Generated {len(ref_cids)} distinct reference conformers (cutoff={v.rmsd_cutoff} A)"
        )
        print(f"  -> Generated {len(probe_cids)} distinct probe candidate conformers")

        # Record radius of gyration
        for cid in ref_cids:
            rg = analyzer.compute_radius_of_gyration(ref_mol, cid)
            expansion_data.append(
                {"variant": v.name, "cid": cid, "radius_of_gyration": rg}
            )

        # Run grid scan
        df_res = analyzer.run_surface_scan(
            ref_mol=ref_mol,
            ref_cids=ref_cids,
            query_mol=probe_mol,
            query_cids=probe_cids,
            k_ref_grid=k_ref_grid,
            k_gen_grid=k_gen_grid,
        )
        df_res["variant"] = v.name
        results_by_variant[v.display_title] = df_res

        # Key stats
        score_k1_1 = df_res[
            (df_res["k_ref_requested"] == 1) & (df_res["k_gen_requested"] == 1)
        ]["tanimoto_combo"].values[0]
        score_k3_30 = df_res[
            (df_res["k_ref_requested"] == 3) & (df_res["k_gen_requested"] == 30)
        ]["tanimoto_combo"].values[0]
        score_kmax = df_res["tanimoto_combo"].max()

        summary_rows.append(
            {
                "Variant": v.name,
                "Environment": v.display_title,
                "Unique_Basins": len(ref_cids),
                "Mean_Rg_A": np.mean(
                    [
                        d["radius_of_gyration"]
                        for d in expansion_data
                        if d["variant"] == v.name
                    ]
                ),
                "ROCS_K1_Q1": score_k1_1,
                "ROCS_K3_Q30": score_k3_30,
                "ROCS_Max_Recovery": score_kmax,
                "Duration_sec": time.time() - t0,
            }
        )

    # Save summary dataframe
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "conformer_convergence_scorecard.csv", index=False)
    print("\n" + "=" * 80)
    print("CONFORMER STUDY SUMMARY SCORECARD:")
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # Render publication figures
    print("\nRendering Publication Figures...")
    vis = ConvergenceVisualizer()
    vis.render_3d_surfaces(
        results_by_variant, output_dir / "01_conformer_convergence_3d_surfaces.png"
    )
    vis.render_cross_sectional_recovery(
        results_by_variant, output_dir / "02_cross_sectional_recovery_curves.png"
    )
    vis.render_expansion_profiles(
        expansion_data,
        output_dir / "03_conformational_volume_and_energy_distributions.png",
    )

    print(f"\nAll deliverables successfully generated at: {output_dir}")


if __name__ == "__main__":
    run_full_study()
