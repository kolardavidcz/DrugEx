from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from _david.code.conformer.analyzer import ConformerConvergenceAnalyzer
from _david.code.conformer.generator import ConformerGenerator
from _david.code.conformer.io import ConformerIOManager
from _david.code.conformer.physics import ForceFieldVariant, PhysicalVariantFactory
from _david.code.conformer.visualization import ConformerVisualizer


class ConformerStudyOrchestrator:
    """Class-first orchestrator coordinating physical conformer simulations, scoring, and deliverables.

    Integrates:
    - ConformerGenerator (ETKDGv3 embedding, MMFF94s/UFF physics, Butina clustering)
    - ConformerConvergenceAnalyzer (pairwise ROCS alignment, Rg metric extraction)
    - ConformerIOManager (multi-model SDF/PDB export for VS Code 3D viewers)
    - ConformerVisualizer (3D surfaces, 2D heatmaps, Plotly interactive widget)
    """

    DEFAULT_K_REF_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 12)
    DEFAULT_K_GEN_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 16, 20, 30)

    def __init__(
        self,
        molecule_smiles: str,
        output_dir: Path | str,
        variants: Sequence[ForceFieldVariant] | None = None,
        k_ref_grid: Sequence[int] | None = None,
        k_gen_grid: Sequence[int] | None = None,
        n_raw_conformers: int = 200,
    ) -> None:
        """Initialize the study orchestrator.

        Args:
            molecule_smiles: SMILES string of the target ligand.
            output_dir: Root directory for generated deliverables.
            variants: Sequence of ForceFieldVariant definitions (defaults to authoritative 5-variant suite).
            k_ref_grid: Sequence of reference ensemble sizes to evaluate.
            k_gen_grid: Sequence of candidate ensemble sizes to evaluate.
            n_raw_conformers: Number of raw ETKDGv3 conformers to generate before filtering.
        """
        self.smiles = molecule_smiles
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.variants = (
            variants
            if variants is not None
            else PhysicalVariantFactory.get_benchmark_suite()
        )
        self.k_ref_grid = (
            k_ref_grid if k_ref_grid is not None else self.DEFAULT_K_REF_GRID
        )
        self.k_gen_grid = (
            k_gen_grid if k_gen_grid is not None else self.DEFAULT_K_GEN_GRID
        )
        self.n_raw_conformers = n_raw_conformers

        self.generator = ConformerGenerator(self.smiles)
        self.analyzer = ConformerConvergenceAnalyzer()
        self.io_manager = ConformerIOManager(self.output_dir)
        self.visualizer = ConformerVisualizer()

    def run_study(self) -> tuple[pd.DataFrame, dict[str, Path]]:
        """Execute the end-to-end conformer self-similarity and dielectric convergence experiment.

        Returns:
            Tuple of (summary_scorecard_dataframe, dictionary_of_deliverable_file_paths).
        """
        print("=" * 80)
        print("STARTING CONFORMER SELF-SIMILARITY & DIELECTRIC CONVERGENCE STUDY")
        print(f"Target SMILES: {self.smiles[:40]}... | Output Dir: {self.output_dir}")
        print("=" * 80)

        results_by_variant: dict[str, pd.DataFrame] = {}
        expansion_data: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        deliverable_paths: dict[str, Path] = {}

        for v in self.variants:
            t0 = time.time()
            print(f"\nEvaluating Variant: {v.name} ({v.display_title})...")

            # 1. Build reference ensemble (seed 42)
            ref_mol, ref_cids, ref_energies = self.generator.build_ensemble(
                variant=v, n_raw=self.n_raw_conformers, seed=42
            )

            # 2. Build independent probe candidate ensemble (seed 9999)
            probe_mol, probe_cids, _probe_energies = self.generator.build_ensemble(
                variant=v, n_raw=self.n_raw_conformers, seed=9999
            )

            print(
                f"  -> Generated {len(ref_cids)} distinct reference conformers (cutoff={v.rmsd_cutoff} A)"
            )
            print(
                f"  -> Generated {len(probe_cids)} distinct probe candidate conformers"
            )

            # 3. Compute Radius of Gyration and cache per conformer
            rg_dict: dict[int, float] = {}
            for cid in ref_cids:
                rg = self.analyzer.compute_radius_of_gyration(ref_mol, cid)
                rg_dict[cid] = rg
                expansion_data.append(
                    {"variant": v.name, "cid": cid, "radius_of_gyration": rg}
                )

            # 4. Export 3D multi-model SDF and PDB for VS Code viewer extension
            base_filename = f"egcg_conformers_{v.name.lower()}"
            sdf_path = self.io_manager.export_sdf(
                mol=ref_mol,
                conf_ids=ref_cids,
                energies=ref_energies,
                rg_values=rg_dict,
                filename=f"{base_filename}.sdf",
            )
            pdb_path = self.io_manager.export_pdb(
                mol=ref_mol,
                conf_ids=ref_cids,
                filename=f"{base_filename}.pdb",
            )
            deliverable_paths[f"sdf_{v.name}"] = sdf_path
            deliverable_paths[f"pdb_{v.name}"] = pdb_path
            print(
                f"  -> Exported 3D Models for VS Code: {sdf_path.name}, {pdb_path.name}"
            )

            # 5. Execute 2D grid scan
            df_res = self.analyzer.run_surface_scan(
                ref_mol=ref_mol,
                ref_cids=ref_cids,
                query_mol=probe_mol,
                query_cids=probe_cids,
                k_ref_grid=self.k_ref_grid,
                k_gen_grid=self.k_gen_grid,
            )
            df_res["variant"] = v.name
            results_by_variant[v.display_title] = df_res

            # 6. Extract key quantitative metrics
            sub_k1 = df_res[
                (df_res["k_ref_requested"] == 1) & (df_res["k_gen_requested"] == 1)
            ]["tanimoto_combo"]
            score_k1_1 = float(np.asarray(sub_k1)[0])

            sub_k3 = df_res[
                (df_res["k_ref_requested"] == 3) & (df_res["k_gen_requested"] == 30)
            ]["tanimoto_combo"]
            score_k3_30 = float(np.asarray(sub_k3)[0])
            score_kmax = float(np.max(np.asarray(df_res["tanimoto_combo"])))

            summary_rows.append(
                {
                    "Variant": v.name,
                    "Environment": v.display_title,
                    "Unique_Basins": len(ref_cids),
                    "Mean_Rg_A": float(
                        np.mean(
                            [
                                float(cast(Any, d["radius_of_gyration"]))
                                for d in expansion_data
                                if d["variant"] == v.name
                            ]
                        )
                    ),
                    "ROCS_K1_Q1": score_k1_1,
                    "ROCS_K3_Q30": score_k3_30,
                    "ROCS_Max_Recovery": score_kmax,
                    "Duration_sec": time.time() - t0,
                }
            )

        # 7. Export summary scorecard
        scorecard_path = self.io_manager.export_scorecard(summary_rows)
        deliverable_paths["scorecard"] = scorecard_path
        df_summary = pd.DataFrame(summary_rows)

        print("\n" + "=" * 80)
        print("CONFORMER STUDY SUMMARY SCORECARD:")
        print(df_summary.to_string(index=False))
        print("=" * 80)

        # 8. Render visualization deliverables
        print("\nRendering Publication Figures and Interactive Widgets...")
        p_3d = self.visualizer.render_3d_surfaces(
            results_by_variant,
            self.output_dir / "01_conformer_convergence_3d_surfaces.png",
        )
        p_heat = self.visualizer.render_heatmaps(
            results_by_variant,
            self.output_dir / "01b_conformer_convergence_heatmaps.png",
        )
        p_plotly = self.visualizer.render_interactive_plotly(
            results_by_variant,
            self.output_dir / "01_conformer_convergence_interactive.html",
        )
        p_cross = self.visualizer.render_cross_sectional_recovery(
            results_by_variant,
            self.output_dir / "02_cross_sectional_recovery_curves.png",
        )
        p_exp = self.visualizer.render_expansion_profiles(
            expansion_data,
            self.output_dir / "03_conformational_volume_and_energy_distributions.png",
        )

        deliverable_paths["figure_3d_surfaces"] = p_3d
        deliverable_paths["figure_heatmaps"] = p_heat
        deliverable_paths["figure_interactive_html"] = p_plotly
        deliverable_paths["figure_cross_sections"] = p_cross
        deliverable_paths["figure_expansion"] = p_exp

        print(f"\nAll deliverables successfully generated at: {self.output_dir}")
        return df_summary, deliverable_paths
