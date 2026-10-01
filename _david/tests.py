"""Unit tests for anti-cheating multi-objective rewards, fallback monitoring, and DrugEx API parity.

These tests empirically verify:
1. AugmentedChebyshevReward penalizes unbalanced candidates exploiting QED/SA at the expense of 3D shape.
2. GeometricMeanReward calculates mathematically correct weighted geometric means.
3. SmoothHump modifier creates smooth bell curves for molecular properties.
4. Scaffold modifier correctly detects and penalizes PAINS catechol toxicophores.
5. dummyMolsFromFragments successfully constructs graph fragment datasets from intact Murcko cores.
6. Edge case handling (empty inputs, zero scores, 1.0 scores).
7. FallbackMonitor tracks, alerts, counts, and exports fallback audits.
8. Dynamic size scaling derived from seed heavy atom counts.
9. ScaffoldFunnel gated two-tier pre-filtering and chemotype extraction.
10. SequenceRNNOptimizer and GraphTransformerOptimizer environment configurations.
11. 1-click runner module import and interface compliance.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#? [NumPy] Vectorized array test comparisons and assertions.
import numpy as np
#? [RDKit] MolFromSmiles validation and Murcko scaffold extraction.
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugex.training.scorers.modifiers import SmoothHump
from drugex.training.scorers.properties import Scaffold
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments

# Import from modular code package
from _david.code.fallback_monitor import FallbackEvent, FallbackMonitor
from _david.code.anti_cheating_rewards import AugmentedChebyshevReward, GeometricMeanReward
from _david.code.usrcat_scorer import USRCATScorer
from _david.code.scaffold_funnel import ScaffoldFunnel
from _david.code.SequenceRNN import SequenceRNNOptimizer
from _david.code.GraphTransformer import GraphTransformerOptimizer
from _david.code.prepare_query_conformer import QueryConformerPreparer, MY_MOLECULE
import _david.run as david_run


class TestAntiCheatingRewards(unittest.TestCase):
    """Test suite for anti-cheating reward schemes, fallback monitors, and API parity."""

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

    def test_scaffold_catechol_penalty(self) -> None:
        """Verify Scaffold correctly flags catechol toxicophore in EGCG."""
        catechol_smarts = "c1c([OH])c([OH])ccc1"
        scaffold_filter = Scaffold(smart=catechol_smarts, is_match=False)
        egcg_smiles = "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
        egcg_mol = Chem.MolFromSmiles(egcg_smiles)
        # EGCG contains catechol/gallate rings, is_match=False -> should be penalized (score 0.0)
        scores_egcg = scaffold_filter([egcg_mol])
        self.assertEqual(scores_egcg[0], 0.0)

        # Benzene has no catechol -> should pass (score 1.0)
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
        scores = np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.9, 0.0001],
        ], dtype=np.float64)
        smiles = ["C1", "C2"]
        r = reward_fn(smiles, scores, [0.5] * 4)

        self.assertAlmostEqual(r[0, 0], 0.5, places=3)
        self.assertLess(r[1, 0], 0.15)

    def test_dummy_mols_from_fragments(self) -> None:
        """Verify dummyMolsFromFragments generates valid graph inputs from Murcko core."""
        from drugex.data.corpus.vocabulary import VocGraph
        from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
        from drugex.data.datasets import GraphFragDataSet

        egcg_smiles = "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
        mol = Chem.MolFromSmiles(egcg_smiles)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        core_smiles = Chem.MolToSmiles(core, isomericSmiles=False)
        self.assertTrue(len(core_smiles) > 0)

        # dummyMolsFromFragments is a MolConverter instantiated with 0 arguments
        converter = dummyMolsFromFragments()
        pairs = converter(core_smiles)
        self.assertIsNotNone(pairs)
        self.assertEqual(len(pairs), 1)
        frag, dummy_mol = pairs[0]
        self.assertEqual(frag, core_smiles)
        self.assertIsNotNone(dummy_mol)

        # End-to-end verification: FragmentCorpusEncoder successfully encodes Murcko core into GraphFragDataSet
        encoder = FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=GraphFragmentEncoder(VocGraph(n_frags=4)),
            pairs_splitter=None,
            n_proc=1,
        )
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            tmp_tsv = tmp.name

        dataset = GraphFragDataSet(tmp_tsv, rewrite=True)
        encoder.apply([core_smiles], encodingCollectors=[dataset])
        data = dataset.getData()
        self.assertGreater(len(data), 0)

    def test_edge_cases(self) -> None:
        """Verify empty, NaN, and 1D edge cases do not throw uncaught exceptions."""
        cheb = AugmentedChebyshevReward()
        geom = GeometricMeanReward()

        # Empty inputs
        empty_scores = np.zeros((0, 4))
        r_cheb_empty = cheb([], empty_scores, [])
        r_geom_empty = geom([], empty_scores, [])
        self.assertEqual(r_cheb_empty.shape, (0, 1))
        self.assertEqual(r_geom_empty.shape, (0, 1))

        # All ones (ideal point)
        ones_scores = np.ones((1, 4))
        r_cheb_ones = cheb(["C"], ones_scores, [0.5] * 4)
        r_geom_ones = geom(["C"], ones_scores, [0.5] * 4)
        self.assertAlmostEqual(r_cheb_ones[0, 0], 1.0, places=4)
        self.assertAlmostEqual(r_geom_ones[0, 0], 1.0, places=4)

        # 1D array handling
        scores_1d = np.array([0.9, 0.2], dtype=np.float64)
        r_cheb_1d = cheb(["C1", "C2"], scores_1d, [0.5])
        r_geom_1d = geom(["C1", "C2"], scores_1d, [0.5])
        self.assertEqual(r_cheb_1d.shape, (2, 1))
        self.assertEqual(r_geom_1d.shape, (2, 1))

        # NaN score handling
        scores_nan = np.array([[np.nan, 0.8], [0.5, np.nan]])
        r_cheb_nan = cheb(["C1", "C2"], scores_nan, [0.5, 0.5])
        r_geom_nan = geom(["C1", "C2"], scores_nan, [0.5, 0.5])
        self.assertFalse(np.isnan(r_cheb_nan).any())
        self.assertFalse(np.isnan(r_geom_nan).any())

    def test_rocs_flattened_output(self) -> None:
        """Verify compute_rocs_scores outputs a flat 1D array."""
        funnel = ScaffoldFunnel()
        rocs = funnel.compute_rocs_scores(["c1ccccc1", "c1ccncc1"])
        self.assertEqual(rocs.ndim, 1)
        self.assertEqual(len(rocs), 2)

    def test_graph_transformer_prepare_seed_fragments(self) -> None:
        """Verify GraphTransformerOptimizer._prepare_seed_fragments successfully creates DataLoader and dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(output_dir=Path(tmp_dir))
            scaffolds_file = Path(__file__).parent / "funnel_scaffolds.tsv"
            loader, dataset = optimizer._prepare_seed_fragments(
                MY_MOLECULE, n_prompts=10, scaffolds_path=scaffolds_file
            )
            self.assertIsNotNone(loader)
            self.assertIsNotNone(dataset)
            self.assertGreater(len(dataset), 0)
            batch = next(iter(loader))
            self.assertEqual(batch.ndim, 3)

    def test_dynamic_size_scaling(self) -> None:
        """Verify ScaffoldFunnel dynamically adapts size envelopes to query seed molecules."""
        # EGCG seed (33 heavy atoms)
        funnel_egcg = ScaffoldFunnel(seed_smiles="C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O")
        self.assertEqual(funnel_egcg.SEED_HAC, 33)
        self.assertEqual(funnel_egcg.MIN_CANDIDATE_HEAVY_ATOMS, 19)
        self.assertEqual(funnel_egcg.MAX_CANDIDATE_HEAVY_ATOMS, 46)

        # Caffeine seed (14 heavy atoms)
        caffeine_smiles = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
        funnel_caffeine = ScaffoldFunnel(seed_smiles=caffeine_smiles)
        self.assertEqual(funnel_caffeine.SEED_HAC, 14)
        self.assertEqual(funnel_caffeine.MIN_CANDIDATE_HEAVY_ATOMS, 8)
        self.assertEqual(funnel_caffeine.MAX_CANDIDATE_HEAVY_ATOMS, 19)

    def test_universal_environment_defaults(self) -> None:
        """Verify SequenceRNNOptimizer builds 5 universal objectives by default and 6 with custom toxicophore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = SequenceRNNOptimizer(output_dir=Path(tmp_dir))
            ref_sdf = Path(__file__).parent / "references" / "my_molecule_3d.sdf"
            if not ref_sdf.exists():
                QueryConformerPreparer(ref_sdf.parent).generate_reference_sdf(output_path=ref_sdf)

            # Default environment: 5 universal objectives (ElectroShape, Heavy, RAscore, QED, SA)
            env_default = optimizer.build_environment(ref_sdf)
            self.assertEqual(len(env_default.scorers), 5)
            scorer_keys = [s.getKey() for s in env_default.scorers]
            self.assertIn("ElectroShape", scorer_keys)
            self.assertIn("Heavy", scorer_keys)
            self.assertIn("QED", scorer_keys)
            self.assertIn("SA", scorer_keys)

            # Environment with optional toxicophore: 6 objectives
            env_tox = optimizer.build_environment(ref_sdf, toxicophore_smarts="c1c([OH])c([OH])ccc1")
            self.assertEqual(len(env_tox.scorers), 6)

    def test_scaffold_funnel_gated_execution(self) -> None:
        """Verify ScaffoldFunnel correctly pre-filters candidates before Stage 2 clustering."""
        import pandas as pd

        funnel = ScaffoldFunnel()
        dummy_df = pd.DataFrame({
            "SMILES": [
                "C",  # Methane: tiny fragment, should fail size envelope
                "invalid_smiles_string",  # Invalid chemical graph
                "c1ccccc1NC(=O)c1ccc(cn1)C1CC1",  # Valid drug-like molecule
                "Cc1ccc(cc1)N1CCN(CC1)c1nc2ccccc2n1C",  # Valid drug-like molecule
            ],
            "Valid": [1, 0, 1, 1],
            "ElectroShape": [0.10, 0.00, 0.65, 0.72],
            "RAscore": [0.10, 0.00, 0.88, 0.78],
            "QED": [0.10, 0.00, 0.75, 0.65],
        })
        res = funnel.process_candidates_dataframe(dummy_df)
        self.assertGreater(len(res), 0)
        self.assertIn("Scaffold", res.columns)
        self.assertIn("ElectroShape", res.columns)
        self.assertIn("ROCS", res.columns)

    def test_fallback_monitor_recording_and_audit(self) -> None:
        """Verify FallbackMonitor records alerts, accumulates counts, and exports audit JSON."""
        self.assertFalse(FallbackMonitor.has_fallbacks())
        self.assertEqual(len(FallbackMonitor.get_events()), 0)

        # Record a fallback event
        ev = FallbackMonitor.record_fallback(
            fallback_type="test_conformer_random_coords",
            message="ETKDGv3 embedding failed; relaxed bounds",
            context={"smiles": "CC1CC1", "num_confs": 10},
        )
        self.assertTrue(FallbackMonitor.has_fallbacks())
        self.assertEqual(len(FallbackMonitor.get_events()), 1)
        self.assertEqual(ev.fallback_type, "test_conformer_random_coords")

        # Record a second event of different type
        FallbackMonitor.record_fallback(
            fallback_type="test_rocs_fallback",
            message="CDPKit ROCS failed; fell back to RDKit",
        )
        counts = FallbackMonitor.get_counts()
        self.assertEqual(counts["test_conformer_random_coords"], 1)
        self.assertEqual(counts["test_rocs_fallback"], 1)

        # Test audit JSON export
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_path = Path(tmp_dir) / "audit.json"
            saved = FallbackMonitor.save_audit_json(audit_path)
            self.assertTrue(saved.exists())
            import json
            with open(saved, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["total_fallbacks"], 2)
            self.assertIn("test_conformer_random_coords", data["counts_by_type"])

    def test_backward_compatibility_imports(self) -> None:
        """Verify legacy imports from _david root continue to function via aliasing."""
        import _david
        from _david.anti_cheating_rewards import AugmentedChebyshevReward as LegacyCheb
        from _david.scaffold_funnel import ScaffoldFunnel as LegacyFunnel
        from _david.fallback_monitor import FallbackMonitor as LegacyMonitor

        self.assertIsNotNone(LegacyCheb)
        self.assertIsNotNone(LegacyFunnel)
        self.assertIsNotNone(LegacyMonitor)

    def test_one_click_runner_interface(self) -> None:
        """Verify 1-click runner _david/run.py exists and exports main entrypoint."""
        self.assertTrue(hasattr(david_run, "main"))
        self.assertTrue(callable(david_run.main))

    def test_prepare_regime_dataframe_with_frags(self) -> None:
        """Verify ThesisBenchmark.prepare_regime_dataframe does not treat FRAGS string column as RAscore."""
        from _david.code.thesis_benchmark import ThesisBenchmark
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp_dir:
            bench = ThesisBenchmark(output_dir=Path(tmp_dir))
            raw_df = pd.DataFrame({
                "SMILES": ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"],
                "FRAGS": ["c1ccccc1", "c1ccccc1C(=O)O"],
                "RAscore": [0.85, 0.92],
                "ROCS": [0.65, 0.70],
                "Desired": [1, 1],
            })
            clean = bench.prepare_regime_dataframe(raw_df)
            self.assertEqual(len(clean), 2)
            self.assertAlmostEqual(clean["RAscore"].iloc[0], 0.85, places=2)
            self.assertAlmostEqual(clean["RAscore"].iloc[1], 0.92, places=2)

            # Test DataFrame with FRAGS but NO RAscore column (should default safely without crash)
            raw_no_ra = pd.DataFrame({
                "SMILES": ["c1ccccc1O"],
                "FRAGS": ["c1ccccc1"],
            })
            clean_no_ra = bench.prepare_regime_dataframe(raw_no_ra)
            self.assertEqual(len(clean_no_ra), 1)
            self.assertAlmostEqual(clean_no_ra["RAscore"].iloc[0], 0.50, places=2)

    def test_ligand_efficiency_scorer_handles_strings_and_mols(self) -> None:
        """Verify drugex LigandEfficiency computes correct scores and handles strings, Mols, and None."""
        from drugex.training.scorers.properties import LigandEfficiency
        from drugex.training.scorers.interfaces import Scorer

        class DummyQSAR(Scorer):
            def getScores(self, mols, frags=None):
                return np.array([7.0] * len(mols), dtype=np.float64)
            def getKey(self):
                return "DummyQSAR"

        dummy_qsar = DummyQSAR()
        le_scorer = LigandEfficiency(qsar_scorer=dummy_qsar)

        # Test with SMILES strings (benzene has 6 heavy atoms: 1.4 * 7.0 / 6 = 1.6333...)
        scores = le_scorer.getScores(["c1ccccc1", "c1ccccc1O", None, "INVALID_SMILES"])
        self.assertEqual(len(scores), 4)
        self.assertAlmostEqual(scores[0], 1.4 * 7.0 / 6.0, places=3)
        self.assertAlmostEqual(scores[1], 1.4 * 7.0 / 7.0, places=3)
        self.assertEqual(scores[2], 0.0)
        self.assertEqual(scores[3], 0.0)

        # Test with Chem.Mol
        mol = Chem.MolFromSmiles("c1ccccc1")
        scores_mol = le_scorer.getScores([mol])
        self.assertAlmostEqual(scores_mol[0], 1.4 * 7.0 / 6.0, places=3)

    def test_scaffold_funnel_column_matching_isolation(self) -> None:
        """Verify ScaffoldFunnel does not confuse ElectroShape with ROCS or FRAGS with RAscore."""
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp_dir:
            funnel = ScaffoldFunnel(output_dir=Path(tmp_dir), seed_smiles="c1ccccc1")
            called_rocs = []
            def mock_rocs(smiles_list):
                called_rocs.extend(smiles_list)
                return np.full(len(smiles_list), 0.85, dtype=np.float32)
            funnel.compute_rocs_scores = mock_rocs

            cand_df = pd.DataFrame({
                "SMILES": ["c1ccccc1", "c1ccccc1CC"],
                "ElectroShape": [0.90, 0.85],
                "FRAGS": ["c1ccccc1", "c1ccccc1"],
                "RAscore": [0.80, 0.75],
                "QED": [0.60, 0.65],
                "Valid": [1, 1],
            })
            res = funnel.process_candidates_dataframe(cand_df)
            self.assertGreater(len(called_rocs), 0)
            self.assertTrue("ROCS" in res.columns)

    def test_usrcat_scorer_companion_npy_loading(self) -> None:
        """Verify USRCATScorer loads companion pre-computed .npy moments alongside .sdf."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sdf_path = Path(tmp_dir) / "query_3d.sdf"
            m = Chem.MolFromSmiles("c1ccccc1")
            AllChem.EmbedMolecule(m, randomSeed=42)
            w = Chem.SDWriter(str(sdf_path))
            w.write(m)
            w.close()

            comp_npy = Path(tmp_dir) / "query_3d_electroshape.npy"
            mock_moments = np.ones((1, 75), dtype=np.float64)
            np.save(comp_npy, mock_moments)

            scorer = USRCATScorer(references=sdf_path, score_type="electroshape")
            np.testing.assert_array_equal(scorer.ref_moments, mock_moments)

    def test_thesis_benchmark_ref_sdf_and_seed_isolation(self) -> None:
        """Verify ThesisBenchmark correctly accepts custom ref_sdf and seed_smiles."""
        from _david.code.thesis_benchmark import ThesisBenchmark
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_sdf = Path(tmp_dir) / "custom_seed.sdf"
            bench = ThesisBenchmark(
                output_dir=Path(tmp_dir),
                ref_sdf=custom_sdf,
                seed_smiles="c1ccccc1O",
            )
            self.assertEqual(bench.REF_SDF, custom_sdf)
            self.assertEqual(bench.seed_smiles, "c1ccccc1O")
            self.assertIsNotNone(bench.SEED_MOL)
            self.assertEqual(bench.SEED_MOL.GetNumHeavyAtoms(), 7)

    def test_pipeline_benchmark_only_creates_summaries(self) -> None:
        """Verify MoleculePipeline in benchmark_only mode produces summary and audit JSON files."""
        from _david.code.pipeline import MoleculePipeline
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "test_run"
            bench_dir = pipe_dir / "benchmark"
            bench_dir.mkdir(parents=True, exist_ok=True)
            pipe = MoleculePipeline(output_dir=pipe_dir, mode="benchmark_only")
            import pandas as pd
            from unittest.mock import patch
            with patch("drugex.training.scorers.sascorer.calculateScore", return_value=3.0), \
                 patch("_david.code.thesis_benchmark.ThesisBenchmark.run_benchmark", return_value=pd.DataFrame()):
                summary = pipe.run(molecules="c1ccccc1O")
                self.assertTrue((pipe_dir / "pipeline_summary.json").exists())
                self.assertTrue((pipe_dir / "fallback_audit.json").exists())
                self.assertEqual(summary["mode"], "benchmark_only")


if __name__ == "__main__":
    unittest.main()
