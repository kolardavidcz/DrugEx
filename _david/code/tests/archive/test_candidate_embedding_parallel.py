"""Comprehensive verification for parallel candidate conformer embedding and batch probe."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem

from _david.code.chemistry import (
    CandidateConformerGenerator,
    Optimization,
    check_identity,
    filter_conformer_identities,
    optimize_conformers,
)
from _david.code.conformer.batch_probe import (
    SAMPLE_CANDIDATE_SMILES,
    measure_batch_scoring,
    prepare_stage0_reference,
    run_batch_probe,
)
from _david.code.environment import create_rocs_scorer
from _david.code.rocs import ConfigurableRDKitROCSScorer


class TestCandidateEmbeddingParallel(unittest.TestCase):
    """Test parallel candidate conformer generation, invariants, error handling, and batch probe."""

    def test_parameter_validation(self) -> None:
        """Verify strict type and boundary validation on candidate conformer generator settings."""
        # Valid cases
        gen = CandidateConformerGenerator(
            max_conformers=20,
            max_isomers=4,
            timeout=15,
            n_jobs=4,
        )
        self.assertEqual(gen.max_conformers, 20)
        self.assertEqual(gen.max_isomers, 4)
        self.assertEqual(gen.timeout, 15)
        self.assertEqual(gen.n_jobs, 4)

        # n_jobs = -1 treated as None (auto-detect)
        gen_auto = CandidateConformerGenerator(n_jobs=-1)
        self.assertIsNone(gen_auto.n_jobs)

        # Dynamic setter assignment of -1
        gen.n_jobs = -1
        self.assertIsNone(gen.n_jobs)
        gen.n_jobs = 2
        self.assertEqual(gen.n_jobs, 2)

        # Invalid bounds and boolean type rejection
        with self.assertRaisesRegex(ValueError, "Conformer attempts must be an integer"):
            CandidateConformerGenerator(max_conformers=0)
        with self.assertRaisesRegex(ValueError, "Conformer attempts must be an integer"):
            CandidateConformerGenerator(max_conformers=True)
        with self.assertRaisesRegex(ValueError, "max_isomers must be a positive integer"):
            CandidateConformerGenerator(max_isomers=0)
        with self.assertRaisesRegex(ValueError, "max_isomers must be a positive integer"):
            CandidateConformerGenerator(max_isomers=True)
        with self.assertRaisesRegex(ValueError, "timeout must be a non-negative integer"):
            CandidateConformerGenerator(timeout=-5)
        with self.assertRaisesRegex(ValueError, "timeout must be a non-negative integer"):
            CandidateConformerGenerator(timeout=True)
        with self.assertRaisesRegex(ValueError, "n_jobs must be a positive integer"):
            CandidateConformerGenerator(n_jobs=-2)
        with self.assertRaisesRegex(ValueError, "n_jobs must be a positive integer"):
            CandidateConformerGenerator(n_jobs=True)

    def test_serial_vs_parallel_conformer_identity(self) -> None:
        """Verify that parallel embedding (n_jobs=4) produces bit-identical conformers to serial (n_jobs=1)."""
        smiles_batch = list(SAMPLE_CANDIDATE_SMILES[:8])  # 8 diverse molecules

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)

            # Serial generation (n_jobs=1)
            gen_serial = CandidateConformerGenerator(
                max_conformers=5,
                max_isomers=2,
                random_seed=42,
                stereo_seed=42,
                n_jobs=1,
            )
            serial_sdf = gen_serial.genConformers(smiles_batch, dir_path / "serial")
            serial_records = [
                m for m in Chem.SDMolSupplier(serial_sdf, removeHs=False) if m is not None
            ]

            # Parallel generation (n_jobs=4)
            gen_parallel = CandidateConformerGenerator(
                max_conformers=5,
                max_isomers=2,
                random_seed=42,
                stereo_seed=42,
                n_jobs=4,
            )
            parallel_sdf = gen_parallel.genConformers(
                smiles_batch, dir_path / "parallel"
            )
            parallel_records = [
                m
                for m in Chem.SDMolSupplier(parallel_sdf, removeHs=False)
                if m is not None
            ]

            self.assertGreater(len(serial_records), 0)
            self.assertEqual(len(serial_records), len(parallel_records))

            for idx, (s_mol, p_mol) in enumerate(zip(serial_records, parallel_records)):
                self.assertEqual(
                    s_mol.GetProp("_Name"),
                    p_mol.GetProp("_Name"),
                    f"Conformer {idx} name mismatch",
                )
                self.assertEqual(
                    s_mol.GetProp("EnumeratedIsomerSMILES"),
                    p_mol.GetProp("EnumeratedIsomerSMILES"),
                    f"Conformer {idx} stereo SMILES mismatch",
                )
                pos_s = s_mol.GetConformer().GetPositions()
                pos_p = p_mol.GetConformer().GetPositions()
                max_coord_diff = np.abs(pos_s - pos_p).max()
                self.assertAlmostEqual(
                    max_coord_diff,
                    0.0,
                    places=6,
                    msg=f"Conformer {idx} coordinate divergence: {max_coord_diff}",
                )

    def test_heavy_atom_and_rotor_filtering_in_parallel(self) -> None:
        """Verify that size and rotatable bond filters apply accurately in worker chunks."""
        test_smiles = [
            "CCO",  # 3 heavy atoms, 0 rotatable bonds -> kept
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",  # 33 heavy atoms, 30 rotatable bonds -> filtered by rotors
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin -> kept
            "c1ccccc1",  # Benzene -> kept
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = CandidateConformerGenerator(
                max_heavy_atoms=30,
                max_rotatable_bonds=10,
                n_jobs=2,
            )
            sdf = generator.genConformers(test_smiles, tmp_dir)
            records = [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]

            # The 33-carbon chain must be filtered out
            self.assertIn(test_smiles[1], generator.last_filtered_molecules)
            # The remaining molecules must have conformers
            names = {m.GetProp("_Name") for m in records}
            self.assertTrue(any(n.startswith("mol_0+") for n in names))
            self.assertFalse(any(n.startswith("mol_1+") for n in names))
            self.assertTrue(any(n.startswith("mol_2+") for n in names))
            self.assertTrue(any(n.startswith("mol_3+") for n in names))

    def test_empty_and_corrupt_inputs_handling(self) -> None:
        """Verify robust handling of empty lists, corrupt SMILES, and missing atoms."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            gen = CandidateConformerGenerator(n_jobs=2)

            # Empty list
            empty_sdf = gen.genConformers([], tmp_dir)
            self.assertTrue(Path(empty_sdf).is_file())
            self.assertEqual(Path(empty_sdf).stat().st_size, 0)

            # Completely invalid SMILES
            invalid_sdf = gen.genConformers(
                ["INVALID_SYNTAX_@@@", "", "NOT_A_MOLECULE"], tmp_dir
            )
            self.assertTrue(Path(invalid_sdf).is_file())
            self.assertEqual(Path(invalid_sdf).stat().st_size, 0)

            # Mixed valid and invalid
            mixed_sdf = gen.genConformers(["INVALID", "CCO", ""], tmp_dir)
            records = [
                m for m in Chem.SDMolSupplier(mixed_sdf, removeHs=False) if m is not None
            ]
            self.assertGreater(len(records), 0)
            # Conformer name should be for mol_1
            self.assertTrue(all(m.GetProp("_Name").startswith("mol_1+") for m in records))

    def test_daemon_process_fallback_to_serial(self) -> None:
        """Verify that CandidateConformerGenerator detects daemon process and falls back to serial without error."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("multiprocessing.current_process") as mock_proc,
        ):
            mock_proc.return_value.daemon = True
            gen = CandidateConformerGenerator(n_jobs=4, max_conformers=2)
            # Batch size >= 4 would normally trigger parallel pool, but daemon=True forces serial fallback
            sdf = gen.genConformers(["CCO", "CCN", "CCC", "CCF"], tmp_dir)
            records = [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]
            self.assertEqual(len(records), 4)

    def test_pool_exception_gracefully_falls_back_to_serial(self) -> None:
        """Verify that an unexpected error during pool creation triggers clean serial fallback."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "drugex.training.scorers.parallel.pool_context",
                side_effect=RuntimeError("Simulated pool initialization error"),
            ),
        ):
            gen = CandidateConformerGenerator(n_jobs=4, max_conformers=2)
            sdf = gen.genConformers(["CCO", "CCN", "CCC", "CCF"], tmp_dir)
            records = [m for m in Chem.SDMolSupplier(sdf, removeHs=False) if m is not None]
            self.assertEqual(len(records), 4)

    def test_scorer_n_jobs_synchronization(self) -> None:
        """Verify that ConfigurableRDKitROCSScorer synchronizes n_jobs to its conformer generator."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        Chem.AllChem.EmbedMolecule(ref_mol, randomSeed=42)

        scorer = create_rocs_scorer(
            [ref_mol],
            max_conformers=5,
            n_jobs=4,
        )
        self.assertEqual(scorer.n_jobs, 4)
        self.assertEqual(scorer.conformer_generator.n_jobs, 4)

        # Updating n_jobs property on scorer must propagate to conformer_generator
        scorer.n_jobs = 8
        self.assertEqual(scorer.n_jobs, 8)
        self.assertEqual(scorer.conformer_generator.n_jobs, 8)

        scorer.n_jobs = 1
        self.assertEqual(scorer.n_jobs, 1)
        self.assertEqual(scorer.conformer_generator.n_jobs, 1)

        # ConfigurableRDKitROCSScorer with n_jobs=-1 must resolve to a valid positive integer
        scorer_neg = ConfigurableRDKitROCSScorer(
            conformer_generator=CandidateConformerGenerator(n_jobs=-1),
            references=ref_mol,
            score_variant="selectivity_color_heavy",
            n_jobs=-1,
        )
        self.assertGreaterEqual(scorer_neg.n_jobs, 1)
        self.assertEqual(scorer_neg.conformer_generator.n_jobs, scorer_neg.n_jobs)
        # Running getScores with custom preset must execute without crashing or pool error
        scores = scorer_neg.getScores(["CCO", "CCN"])
        self.assertEqual(scores.shape, (2, 1))

        # Dynamic setting of -1 on scorer propagates synchronized positive integer to conformer_generator
        scorer.n_jobs = -1
        self.assertGreaterEqual(scorer.n_jobs, 1)
        self.assertEqual(scorer.conformer_generator.n_jobs, scorer.n_jobs)

    def test_filter_conformer_identities_without_isomer_smiles(self) -> None:
        """Verify that conformers without EnumeratedIsomerSMILES property do not raise KeyError."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
        # mol has no EnumeratedIsomerSMILES property
        self.assertFalse(mol.HasProp("EnumeratedIsomerSMILES"))
        accepted, rejected = filter_conformer_identities([mol])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)

    def test_batch_probe_reports_minimization_seconds(self) -> None:
        """Verify that batch probe separately tracks embedding and minimization durations."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
        Chem.AllChem.EmbedMolecule(ref_mol, randomSeed=104729)
        scores, timings = measure_batch_scoring(
            ["CCO", "CCN", "CCC", "CCF"],
            [ref_mol],
            workers=1,
            max_conformers=2,
            minimize_candidates=True,
        )
        self.assertGreater(timings["embedding_seconds"], 0.0)
        self.assertGreater(timings["minimization_seconds"], 0.0)

    def test_lazy_conformer_init_no_runtime_warning(self) -> None:
        """Verify that _david.code.conformer exports measure_batch_scoring and run_batch_probe lazily."""
        import _david.code.conformer as conf_pkg

        self.assertTrue(callable(conf_pkg.measure_batch_scoring))
        self.assertTrue(callable(conf_pkg.run_batch_probe))

    def test_batch_probe_measure_and_parity(self) -> None:
        """Test the realistic batch scoring benchmark probe and verify serial/parallel score agreement."""
        ref_mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccc(cc1)C(=O)O"))  # 4-hydroxybenzoic acid
        Chem.AllChem.EmbedMolecule(ref_mol, randomSeed=104729)

        test_batch = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CCO", "CCN"]

        # Run serial (workers=1)
        scores_s, timings_s = measure_batch_scoring(
            test_batch,
            [ref_mol],
            workers=1,
            max_conformers=5,
            minimize_candidates=False,
        )

        # Run parallel (workers=2)
        scores_p, timings_p = measure_batch_scoring(
            test_batch,
            [ref_mol],
            workers=2,
            max_conformers=5,
            minimize_candidates=False,
        )

        self.assertEqual(len(scores_s), len(test_batch))
        self.assertEqual(len(scores_p), len(test_batch))
        self.assertGreater(timings_s["total_batch_seconds"], 0)
        self.assertGreater(timings_p["total_batch_seconds"], 0)

        # Parity assertion: scores must match within 1e-8
        score_diff = np.abs(scores_s - scores_p).max()
        self.assertLessEqual(score_diff, 1e-8)

    def test_run_batch_probe_small(self) -> None:
        """Test run_batch_probe helper across 1 repeat and verify outputs."""
        df, max_diff = run_batch_probe(
            batch_sizes=[4],
            worker_counts=[1, 2],
            minimize_candidates=False,
            repeats=1,
            reference_smiles="CC(=O)Oc1ccccc1C(=O)O",
        )
        self.assertEqual(len(df), 2)
        self.assertIn("speedup", df.columns)
        self.assertIn("embedding_seconds", df.columns)
        self.assertIn("minimization_seconds", df.columns)
        self.assertIn("alignment_seconds", df.columns)
        self.assertLessEqual(max_diff, 1e-8)

    def test_optimize_conformers_parameter_validation(self) -> None:
        """Verify strict validation on n_jobs for optimize_conformers."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
        opt = Optimization("MMFF94s", 78.4)

        with self.assertRaisesRegex(ValueError, "n_jobs must be a positive integer"):
            optimize_conformers([mol], opt, n_jobs=-2)
        with self.assertRaisesRegex(ValueError, "n_jobs must be a positive integer"):
            optimize_conformers([mol], opt, n_jobs=True)

    def test_minimized_candidates_empty_on_failed_embedding_or_optimization(self) -> None:
        """Verify that when minimize_candidates=True, failed embedding or optimization produces size 0 SDF."""
        with tempfile.TemporaryDirectory() as d:
            # 1. Failed embedding
            gen = CandidateConformerGenerator(minimize_candidates=True, n_jobs=1)
            with patch("_david.code.chemistry.AllChem.EmbedMultipleConfs", return_value=[-1]):
                path_fail_embed = gen.genConformers(["CCO"], Path(d) / "fail_embed")
                self.assertEqual(Path(path_fail_embed).stat().st_size, 0)
                self.assertEqual(gen.last_embedding_failures, 1)

            # 2. Failed optimization (no converged conformers)
            with patch("_david.code.chemistry.optimize_conformers", return_value=([], [1])):
                path_fail_opt = gen.genConformers(["CCO"], Path(d) / "fail_opt")
                self.assertEqual(Path(path_fail_opt).stat().st_size, 0)

    def test_serial_vs_parallel_conformer_identity_with_minimization(self) -> None:
        """Verify that parallel candidate generation with minimization matches serial bit-for-bit."""
        smiles_batch = list(SAMPLE_CANDIDATE_SMILES[:6])
        with tempfile.TemporaryDirectory() as tmp_dir:
            d = Path(tmp_dir)
            gen_s = CandidateConformerGenerator(
                max_conformers=3,
                max_isomers=2,
                random_seed=42,
                stereo_seed=42,
                n_jobs=1,
                minimize_candidates=True,
                optimization=Optimization("MMFF94s", 78.4),
            )
            sdf_s = gen_s.genConformers(smiles_batch, d / "s")
            recs_s = [m for m in Chem.SDMolSupplier(sdf_s, removeHs=False) if m is not None]

            gen_p = CandidateConformerGenerator(
                max_conformers=3,
                max_isomers=2,
                random_seed=42,
                stereo_seed=42,
                n_jobs=4,
                minimize_candidates=True,
                optimization=Optimization("MMFF94s", 78.4),
            )
            sdf_p = gen_p.genConformers(smiles_batch, d / "p")
            recs_p = [m for m in Chem.SDMolSupplier(sdf_p, removeHs=False) if m is not None]

            self.assertGreater(len(recs_s), 0)
            self.assertEqual(len(recs_s), len(recs_p))
            for s, p in zip(recs_s, recs_p):
                diff = np.abs(s.GetConformer().GetPositions() - p.GetConformer().GetPositions()).max()
                self.assertLessEqual(diff, 1e-8)


if __name__ == "__main__":
    unittest.main()
