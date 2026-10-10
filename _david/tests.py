"""Unit tests for active Graph-Transformer LBDD pipeline, rewards, and fallback monitoring.

These tests empirically verify:
1. AugmentedChebyshevReward and AntiCollapseSoftGateReward penalize cheating candidates.
2. GeometricMeanReward calculates mathematically correct weighted geometric means.
3. SmoothHump modifier creates smooth bell curves for molecular properties.
4. Scaffold modifier correctly flags arbitrary substructure patterns.
5. dummyMolsFromFragments successfully constructs graph fragment datasets from intact Murcko cores.
6. FallbackMonitor tracks, alerts, counts, and exports fallback audits.
7. DynamicPropertyEnvelope dynamically scales HAC, CLogP, and TPSA bounds to query ligands.
8. GraphTransformerOptimizer environment configurations and seed fragment loader.
9. ResultAnalyzer candidate profiling, diverse lead selection, and summary export.
10. MoleculePipeline execution in benchmark_only mode.
11. 1-click runner module import and interface compliance.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure repository root is on sys.path for direct execution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.scorers.modifiers import SmoothHump
from drugex.training.scorers.properties import Scaffold
from rdkit import Chem
from rdkit.Chem import AllChem

import _david.run as david_run
from _david.code.analysis import AlignedDepicter, ParetoFrontier, ResultAnalyzer
from _david.code.experiments import ExperimentRunner, ThresholdAblationRunner
from _david.code.models import GraphTransformerOptimizer, RL_EPSILON
from _david.code.objectives import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    DynamicPropertyEnvelope,
    GeometricMeanReward,
    MY_MOLECULE,
    ObjectiveBuilder,
    ObjectivePreset,
    QueryConformerPreparer,
    create_dynamic_property_scorers,
    get_seed_hash,
)
from _david.code.pipeline import MoleculePipeline
from _david.code.utils import FallbackMonitor, display_dashboard


class TestActivePipeline(unittest.TestCase):
    """Test suite for the active Graph-Transformer LBDD architecture and rewards."""

    def setUp(self) -> None:
        """Reset FallbackMonitor before each test to ensure clean isolated test state."""
        FallbackMonitor.reset()

    def test_smooth_hump_modifier(self) -> None:
        """Verify SmoothHump yields near 1.0 within optimal range and 0.0 outside."""
        hump = SmoothHump(lower_x=28, upper_x=38, sigma=4)
        # Inside target plateau [28, 38]
        score_mid = hump(33.0)
        self.assertAlmostEqual(score_mid, 1.0, places=2)
        # Outside target range
        score_low = hump(10.0)
        self.assertLess(score_low, 0.05)
        score_high = hump(60.0)
        self.assertLess(score_high, 0.05)

    def test_scaffold_substructure_penalty(self) -> None:
        """Verify Scaffold correctly flags substructures using arbitrary query SMARTS."""
        nitro_smarts = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
        scaffold_filter = Scaffold(smart=nitro_smarts, is_match=False)
        nitrobenzene_mol = Chem.MolFromSmiles("c1ccccc1[N+](=O)[O-]")
        scores_nitro = scaffold_filter([nitrobenzene_mol])
        self.assertEqual(scores_nitro[0], 0.0)

        benzene_mol = Chem.MolFromSmiles("c1ccccc1")
        scores_benzene = scaffold_filter([benzene_mol])
        self.assertEqual(scores_benzene[0], 1.0)

    def test_augmented_chebyshev_anti_cheating(self) -> None:
        """Verify Chebyshev penalizes a candidate that collapses in 3D shape."""
        reward_fn = AugmentedChebyshevReward()
        # Candidate A: balanced [0.70, 0.70, 0.70, 0.70, 0.70, 0.70]
        # Candidate B: cheating fragment [0.98, 0.98, 0.98, 0.98, 0.98, 0.02] (fails 3D shape)
        scores = np.array([
            [0.70, 0.70, 0.70, 0.70, 0.70, 0.70],
            [0.98, 0.98, 0.98, 0.98, 0.98, 0.02],
        ], dtype=np.float64)
        smiles = ["C1", "C2"]
        thresholds = [0.5] * 6

        r = reward_fn(smiles, scores, thresholds)
        self.assertEqual(r.shape, (2, 1))
        # Balanced molecule must receive higher reward than 3D-collapsed cheater
        self.assertGreater(r[0, 0], r[1, 0])
        # Candidate B must be severely penalized (cheater reward < 0.20)
        self.assertLess(r[1, 0], 0.20)

    def test_geometric_mean_reward(self) -> None:
        """Verify GeometricMeanReward calculates accurate log-sum-exp geometric mean."""
        reward_fn = GeometricMeanReward()
        scores = np.array([[0.8, 0.8], [0.1, 0.9]], dtype=np.float64)
        smiles = ["C1", "C2"]
        thresholds = [0.5, 0.5]

        r = reward_fn(smiles, scores, thresholds)
        self.assertAlmostEqual(r[0, 0], 0.8, places=4)
        self.assertAlmostEqual(r[1, 0], np.sqrt(0.1 * 0.9), places=4)

    def test_anti_collapse_soft_gate_reward(self) -> None:
        """Verify AntiCollapseSoftGateReward eliminates Pareto cheating via continuous gating."""
        reward_fn = AntiCollapseSoftGateReward(
            tau_3d=0.60,
            w_3d=0.70,
            w_props=0.30,
            rocs_col_idx=0,
            sa_col_idx=4,
        )

        # Candidate A: high 3D ROCS (1.2 / 2.0 = 0.60 normalized), good properties
        # Candidate B: collapsed 3D ROCS (0.10 / 2.0 = 0.05 normalized), near-perfect properties (Pareto cheater)
        scores = np.array([
            [1.20, 0.80, 0.80, 0.80, 1.0],  # ROCS=1.2, Props=0.8, SA=1.0
            [0.10, 0.99, 0.99, 0.99, 1.0],  # ROCS=0.1, Props=0.99, SA=1.0 (cheater)
        ], dtype=np.float64)
        smiles = ["C1", "C2"]
        thresholds = [0.5] * 5

        r = reward_fn(smiles, scores, thresholds)
        self.assertEqual(r.shape, (2, 1))
        # Candidate A must decisively beat Candidate B
        self.assertGreater(r[0, 0], r[1, 0])

    def test_reward_extreme_edge_cases(self) -> None:
        """Verify reward functions handle empty inputs, zero scores, and NaN inputs gracefully."""
        cheb = AugmentedChebyshevReward()
        geom = GeometricMeanReward()

        # Empty inputs
        empty_scores = np.empty((0, 6), dtype=np.float64)
        r_cheb_empty = cheb([], empty_scores, [0.5] * 6)
        r_geom_empty = geom([], empty_scores, [0.5] * 6)
        self.assertEqual(r_cheb_empty.shape, (0, 1))
        self.assertEqual(r_geom_empty.shape, (0, 1))

        # NaN score handling
        scores_nan = np.array([[np.nan, 0.8], [0.5, np.nan]])
        r_cheb_nan = cheb(["C1", "C2"], scores_nan, [0.5, 0.5])
        r_geom_nan = geom(["C1", "C2"], scores_nan, [0.5, 0.5])
        self.assertFalse(np.isnan(r_cheb_nan).any())
        self.assertFalse(np.isnan(r_geom_nan).any())

    def test_dynamic_property_envelope(self) -> None:
        """Verify DynamicPropertyEnvelope dynamically scales HAC, CLogP, and TPSA bounds to query ligands."""
        # EGCG seed
        envelope_egcg = DynamicPropertyEnvelope.from_query(MY_MOLECULE)
        self.assertEqual(envelope_egcg.target_hac, 33.0)
        self.assertLess(envelope_egcg.hac_lower, 33.0)
        self.assertGreater(envelope_egcg.hac_upper, 33.0)

        # Caffeine seed
        caffeine_smiles = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
        envelope_caffeine = DynamicPropertyEnvelope.from_query(caffeine_smiles)
        self.assertEqual(envelope_caffeine.target_hac, 14.0)
        self.assertLess(envelope_caffeine.hac_lower, 14.0)
        self.assertGreater(envelope_caffeine.hac_upper, 14.0)

        # Verify scorers can be instantiated directly (Heavy and logP)
        scorers = create_dynamic_property_scorers(caffeine_smiles)
        self.assertEqual(len(scorers), 2)

    def test_graph_transformer_prepare_seed_fragments(self) -> None:
        """Verify GraphTransformerOptimizer._prepare_seed_fragments successfully creates DataLoader and dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(output_dir=Path(tmp_dir))
            scaffolds_file = Path(tmp_dir) / "test_scaffolds.tsv"
            pd.DataFrame({"Scaffold": ["c1ccccc1", "c1ccncc1"]}).to_csv(scaffolds_file, sep="\t", index=False)
            loader, dataset = optimizer._prepare_seed_fragments(
                MY_MOLECULE, n_prompts=10, scaffolds_path=scaffolds_file
            )
            self.assertIsNotNone(loader)
            self.assertIsNotNone(dataset)
            self.assertGreater(len(dataset), 0)
            batch = next(iter(loader))
            self.assertEqual(batch.ndim, 3)

    def test_universal_environment_defaults(self) -> None:
        """Verify GraphTransformerOptimizer builds 4 universal objectives with zero approximations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(output_dir=Path(tmp_dir))
            ref_sdf = Path(__file__).parent / "references" / "my_molecule_3d.sdf"
            if not ref_sdf.exists():
                QueryConformerPreparer(ref_sdf.parent).generate_reference_sdf(output_path=ref_sdf)

            env_default = optimizer.build_environment(ref_sdf)
            self.assertEqual(len(env_default.scorers), 4)

            # Verify RDKit ROCS scorer
            rocs_scorer = env_default.scorers[0]
            self.assertEqual(rocs_scorer.score_type, "TanimotoCombo")
            self.assertEqual(rocs_scorer.color_ff, "oe_mills_dean")

            # Verify property keys
            scorer_keys: list[str] = []
            for s in env_default.scorers:
                k = s.getKey()
                if isinstance(k, list):
                    scorer_keys.extend(k)
                else:
                    scorer_keys.append(k)
            self.assertIn("Heavy", scorer_keys)
            self.assertIn("logP", scorer_keys)
            self.assertNotIn("TPSA", scorer_keys)
            self.assertIn("SA", scorer_keys)
            self.assertNotIn("ElectroShape", scorer_keys)
            self.assertNotIn("USRCAT", scorer_keys)
            self.assertNotIn("RAscore", scorer_keys)

    def test_fallback_monitor_recording_and_audit(self) -> None:
        """Verify FallbackMonitor records alerts, accumulates counts, and exports audit JSON."""
        self.assertFalse(FallbackMonitor.has_fallbacks())
        self.assertEqual(len(FallbackMonitor.get_events()), 0)

        ev = FallbackMonitor.record_fallback(
            fallback_type="test_conformer_random_coords",
            message="ETKDGv3 embedding failed; relaxed bounds",
            context={"smiles": "CC1CC1", "num_confs": 10},
        )
        self.assertTrue(FallbackMonitor.has_fallbacks())
        self.assertEqual(len(FallbackMonitor.get_events()), 1)
        self.assertEqual(ev.fallback_type, "test_conformer_random_coords")

        FallbackMonitor.record_fallback(
            fallback_type="test_rocs_fallback",
            message="CDPKit ROCS failed; fell back to RDKit",
        )
        counts = FallbackMonitor.get_counts()
        self.assertEqual(counts["test_conformer_random_coords"], 1)
        self.assertEqual(counts["test_rocs_fallback"], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit.json"
            saved = FallbackMonitor.save_audit_json(audit_path)
            self.assertTrue(saved.exists())
            with open(saved, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total_fallbacks"], 2)
            self.assertIn("test_conformer_random_coords", data["counts_by_type"])

    def test_one_click_runner_interface(self) -> None:
        """Verify 1-click runner _david/run.py exists and exports main entrypoint."""
        self.assertTrue(hasattr(david_run, "main"))
        self.assertTrue(callable(david_run.main))

    def test_seed_hash_and_core_scaffold_extraction(self) -> None:
        """Verify deterministic seed hashing and Murcko core scaffold extraction for universal seeds."""
        caffeine_smiles = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
        caffeine_hash = get_seed_hash(caffeine_smiles)
        self.assertIsInstance(caffeine_hash, str)
        self.assertEqual(len(caffeine_hash), 8)
        self.assertEqual(caffeine_hash, get_seed_hash(caffeine_smiles))

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tsv = Path(tmp_dir) / "caffeine_core.tsv"
            preparer = QueryConformerPreparer(Path(tmp_dir))
            extracted_smi = preparer.extract_seed_core_scaffold(caffeine_smiles, out_tsv)
            self.assertTrue(out_tsv.exists())
            self.assertGreater(len(extracted_smi), 0)

    def test_monitor_dashboard_display(self) -> None:
        """Verify monitor tool display_dashboard runs without exception on pipeline run directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            with open(run_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": "test_run",
                    "mode": "test",
                    "current_stage": "COMPLETED",
                    "overall_status": "SUCCESS",
                    "elapsed_seconds": 12.5,
                    "stages": {"Stage 0": "SUCCESS"},
                    "gpu_telemetry": {"cuda_available": False, "devices": []},
                }, f)
            with open(run_dir / "pipeline.log", "w", encoding="utf-8") as f:
                f.write("[2026-10-02 00:00:00] Pipeline test run\n")
            display_dashboard(run_dir)

    def test_result_analyzer_candidate_profiling(self) -> None:
        """Verify ResultAnalyzer accurately profiles candidate molecules across 2D, 3D, and properties."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analyzer = ResultAnalyzer(output_dir=Path(tmp_dir), seed_smiles="c1ccccc1")
            candidates = [
                "c1ccccc1",  # Benzene (valid, clean)
                "c1ccccc1O",  # Phenol (valid)
                "INVALID_SMILES_STRING",  # Invalid SMILES
            ]
            df = analyzer.analyze_candidates(candidates)
            self.assertEqual(len(df), 3)
            self.assertTrue(df.loc[df["SMILES"] == "c1ccccc1", "Valid"].values[0])
            self.assertFalse(df.loc[df["SMILES"] == "INVALID_SMILES_STRING", "Valid"].values[0])
            self.assertIn("QED", df.columns)
            self.assertIn("MW", df.columns)
            self.assertIn("SA", df.columns)
            self.assertIn("SAscore", df.columns)
            self.assertIn("Desirability", df.columns)

    def test_result_analyzer_lead_prioritization_and_export(self) -> None:
        """Verify ResultAnalyzer prioritizes diverse leads and exports visualizations and JSON summary."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            analyzer = ResultAnalyzer(output_dir=out_path, seed_smiles="c1ccccc1")
            candidates = [
                "c1ccccc1CC",
                "c1ccccc1O",
                "c1ccc(CC)cc1",
                "c1ccncc1",
            ]
            df_all = analyzer.analyze_candidates(candidates)
            df_leads = analyzer.prioritize_leads(df_all, top_k=2)

            self.assertLessEqual(len(df_leads), 2)
            des_vals = df_leads["Desirability"].tolist()
            self.assertEqual(des_vals, sorted(des_vals, reverse=True))

            figs = analyzer.visualize_leads(df_leads, df_all)
            self.assertIn("structures_grid", figs)
            self.assertTrue(figs["structures_grid"].exists())

            summary = analyzer.export_summary(df_all, df_leads)
            self.assertIn("total_candidates_generated", summary)
            self.assertEqual(summary["total_candidates_generated"], 4)
            self.assertTrue((out_path / "result_analysis_summary.json").exists())

    def test_pipeline_benchmark_only_creates_summaries(self) -> None:
        """Verify MoleculePipeline in benchmark_only mode produces summary and audit JSON files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "test_run"
            pipe = MoleculePipeline(output_dir=pipe_dir, mode="benchmark_only")

            with patch("drugex.training.scorers.sascorer.calculateScore", return_value=3.0), \
                 patch("_david.code.result_analyzer.ResultAnalyzer.compute_rocs_scores", side_effect=lambda smi_list: np.full(len(smi_list), 0.75, dtype=np.float32)):
                summary = pipe.run(molecules="c1ccccc1O")
                self.assertTrue((pipe_dir / "pipeline_summary.json").exists())
                self.assertTrue((pipe_dir / "fallback_audit.json").exists())
                self.assertTrue((pipe_dir / "pipeline_status.json").exists())
                self.assertTrue((pipe_dir / "pipeline.log").exists())
                self.assertEqual(summary["mode"], "benchmark_only")

    def test_objective_builder_presets(self) -> None:
        """Verify ObjectiveBuilder correctly constructs environments for pure_3d_sa and gated_envelope presets."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "ref.sdf"
            # Write minimal valid dummy SDF
            m = Chem.MolFromSmiles("c1ccccc1O")
            AllChem.EmbedMolecule(m)
            w = Chem.SDWriter(str(ref_sdf))
            w.write(m)
            w.close()

            # 1. Test PURE_3D_SA preset
            builder_pure = ObjectiveBuilder(preset=ObjectivePreset.PURE_3D_SA, rocs_threshold=0.25, sa_threshold=0.25)
            env_pure = builder_pure.build_environment(ref_sdf, seed_smiles="c1ccccc1O")
            self.assertEqual(len(env_pure.scorers), 2)  # ROCS + SA
            self.assertEqual(len(env_pure.thresholds), 2)
            self.assertEqual(env_pure.thresholds[0], 0.25)
            self.assertEqual(env_pure.thresholds[1], 0.25)
            self.assertIsInstance(env_pure.rewardScheme, AntiCollapseSoftGateReward)
            self.assertEqual(env_pure.rewardScheme.rocs_col_idx, 0)
            self.assertEqual(env_pure.rewardScheme.sa_col_idx, 1)
            self.assertEqual(env_pure.rewardScheme.w_props, 0.0)

            # 2. Test GATED_ENVELOPE preset
            builder_env = ObjectiveBuilder(preset=ObjectivePreset.GATED_ENVELOPE, rocs_threshold=0.25, prop_threshold=0.25, sa_threshold=0.25)
            env_gated = builder_env.build_environment(ref_sdf, seed_smiles="c1ccccc1O")
            self.assertGreater(len(env_gated.scorers), 2)  # ROCS + properties + SA
            self.assertEqual(len(env_gated.thresholds), len(env_gated.scorers))
            self.assertEqual(env_gated.rewardScheme.sa_col_idx, len(env_gated.scorers) - 1)
            self.assertEqual(env_gated.rewardScheme.w_props, 0.30)

    def test_aligned_depicter(self) -> None:
        """Verify AlignedDepicter aligns 2D coordinates and generates grid images."""
        depicter = AlignedDepicter(reference="c1ccccc1O")
        m1 = Chem.MolFromSmiles("c1ccccc1CC")
        aligned = depicter.align_molecule(m1)
        self.assertIsNotNone(aligned)
        self.assertGreater(aligned.GetNumConformers(), 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "aligned_grid.png"
            img = depicter.create_aligned_grid([m1], legends=["Lead 1"], output_path=img_path)
            self.assertIsNotNone(img)
            self.assertTrue(img_path.exists())

    def test_pareto_frontier(self) -> None:
        """Verify ParetoFrontier identifies non-dominated candidates and computes Pareto ranks."""
        pareto = ParetoFrontier()
        # Candidate 0: ROCS=1.5, SA=2.0 (Dominates candidate 1: ROCS=1.0, SA=4.0)
        # Candidate 2: ROCS=0.8, SA=1.5 (Non-dominated tradeoff with Candidate 0)
        df_candidates = pd.DataFrame({
            "SMILES": ["mol0", "mol1", "mol2"],
            "ROCS": [1.5, 1.0, 0.8],
            "SA": [2.0, 4.0, 1.5],
            "Valid": [True, True, True],
        })
        ranked = pareto.rank_pareto_fronts(df_candidates, objectives=("ROCS", "SA"), maximize=(True, False))
        self.assertTrue(ranked.loc[0, "Pareto_Optimal"])
        self.assertFalse(ranked.loc[1, "Pareto_Optimal"])
        self.assertTrue(ranked.loc[2, "Pareto_Optimal"])
        self.assertEqual(ranked.loc[0, "Pareto_Rank"], 0)
        self.assertEqual(ranked.loc[1, "Pareto_Rank"], 1)
        self.assertEqual(ranked.loc[2, "Pareto_Rank"], 0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = Path(tmp_dir) / "pareto.png"
            p = pareto.plot_pareto_frontier(ranked, x_col="ROCS", y_col="SA", output_path=plot_path)
            self.assertIsNotNone(p)
            self.assertTrue(plot_path.exists())


if __name__ == "__main__":
    unittest.main()
