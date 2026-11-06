"""Scientific controls for the joint calibration and shared production pruner."""

import unittest
import tempfile
from pathlib import Path
from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.reference import (
    PrepareMolecule,
    balanced_reference_order,
    select_conformers,
)
from _david.code.chemistry import (
    CandidateConformerGenerator,
    IDENTITY_REJECTED,
    Optimization,
    check_identity,
    optimize_conformers,
    prepare_candidate_molecule,
    verify_disjoint_conformer_seeds,
)
from _david.code.conformer.calibrate import summarize, collect_jobs
from _david.code.conformer.reward_calibration import (
    budget_order,
    select_replay_cells,
)
from _david.code.rocs import WholeCallScorer
from _david.code.schema import std
from _david.code.analysis.visualization import unscale_to_physical_metrics
import pandas as pd


class TestJointCalibration(unittest.TestCase):
    def test_confirmation_replays_cover_methods_without_development_grid_cells(self):

        rows = [
            dict(
                molecule_id=mid,
                method=method,
                reference_cap=cap,
                candidate_attempts=75,
                actual_references=refs,
                actual_candidates=75,
                score=0.9,
            )
            for method in ("etkdg", "uff")
            for mid in (1, 2, 3)
            for cap, refs in ((0, 50), (4, 4))
        ]
        rows.append(rows[0] | dict(molecule_id=99, score=1.00002))
        selected = select_replay_cells(pd.DataFrame(rows), 4)
        ordinary = selected[selected.molecule_id.ne(99)]
        self.assertEqual(
            ordinary.groupby("method").size().to_dict(), {"etkdg": 2, "uff": 2}
        )
        self.assertFalse(ordinary.duplicated(["method", "molecule_id"]).any())
        self.assertTrue(ordinary.reference_cap.eq(4).all())
        self.assertEqual(len(selected), 5)
        self.assertIn(99, selected.molecule_id.tolist())

    def test_development_replays_prefer_small_caps_before_cheaper_other_settings(self):
        row = dict(molecule_id=1, method="uff", score=0.9)
        frame = pd.DataFrame(
            [
                row
                | dict(
                    reference_cap=1,
                    candidate_attempts=10,
                    actual_references=1,
                    actual_candidates=10,
                ),
                row
                | dict(
                    reference_cap=0,
                    candidate_attempts=75,
                    actual_references=1,
                    actual_candidates=1,
                ),
            ]
        )
        selected = select_replay_cells(frame, 1)
        self.assertEqual(selected.reference_cap.tolist(), [1])
        self.assertEqual(selected.candidate_attempts.tolist(), [10])

    def test_whole_call_dispatch_matches_singletons_and_restores_duplicate_rows(self):
        from _david.code.training.environment import create_rocs_scorer

        reference = self.geometry("CCO")
        inputs = [
            Chem.MolFromSmiles("CCO"),
            None,
            Chem.MolFromSmiles("CCN"),
            Chem.MolFromSmiles("CCO"),
        ]
        setting = dict(method="etkdg", candidate_attempts=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.sdf"
            with Chem.SDWriter(str(path)) as writer:
                writer.write(reference)
            original = create_rocs_scorer(
                path, max_conformers=10, n_jobs=1, conformer_threads=1
            )
            original.conformer_generator.embedding_rmsd = None
            expected = np.vstack([original.getScores([mol]) for mol in inputs])
            parallel = WholeCallScorer(original, path, setting, workers=2)
            try:
                actual = parallel.getScores(inputs)
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                self.assertEqual(actual[1, 0], 0)
                self.assertEqual(actual[0, 0], actual[3, 0])
            finally:
                parallel.close()

    def test_query_budget_order_uses_actual_reference_counts_without_redundancy(self):
        frame = pd.DataFrame(
            [
                dict(
                    method="uff",
                    reference_cap=1,
                    candidate_attempts=10,
                    mean_seconds=0.1,
                ),
                dict(
                    method="uff",
                    reference_cap=4,
                    candidate_attempts=10,
                    mean_seconds=0.2,
                ),
                dict(
                    method="uff",
                    reference_cap=0,
                    candidate_attempts=10,
                    mean_seconds=0.3,
                ),
                dict(
                    method="uff",
                    reference_cap=2,
                    candidate_attempts=20,
                    mean_seconds=0.4,
                ),
                dict(
                    method="etkdg",
                    reference_cap=1,
                    candidate_attempts=10,
                    mean_seconds=0.01,
                ),
            ]
        )
        budgets = list(budget_order(frame, "uff", available=1))
        self.assertEqual(
            [(row["reference_cap"], row["candidate_attempts"]) for row in budgets],
            [(1, 10), (1, 20)],
        )

    def test_reference_and_candidate_embedding_streams_do_not_overlap(self):
        from _david.code.conformer.reference_bank import REFERENCE_SEED, CANDIDATE_SEED

        self.assertEqual(
            set(43 * i for i in range(1, 51)) & set(42 * i for i in range(1, 76)),
            {1806},
        )
        self.assertFalse(
            set(REFERENCE_SEED * i for i in range(1, 51))
            & set(CANDIDATE_SEED * i for i in range(1, 76))
        )
        self.assertLess(REFERENCE_SEED * 50, 2**31)
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
        positions = []
        for seed, count in [(REFERENCE_SEED, 50), (CANDIDATE_SEED, 75)]:
            params = AllChem.ETKDGv3()
            params.randomSeed, params.pruneRmsThresh, params.numThreads = seed, -1, 1
            params.enableSequentialRandomSeeds = False
            copy = Chem.Mol(mol)
            AllChem.EmbedMultipleConfs(copy, numConfs=count, params=params)
            positions.append(
                {conf.GetPositions().tobytes() for conf in copy.GetConformers()}
            )
        self.assertFalse(positions[0] & positions[1])

    def test_energy_seeded_diversity_protects_stereo_identities(self):
        from _david.code.conformer.reference_bank import diverse_order

        records = []
        for index, (identity, energy) in enumerate(
            [("A", 2.0), ("A", 1.0), ("A", 3.0), ("B", 10.0)]
        ):
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("EnumeratedIsomerSMILES", identity)
            mol.SetDoubleProp("Energy_kcal_mol", energy)
            mol.SetIntProp("FixtureIndex", index)
            records.append(mol)

        def similarity(left, right):
            pair = {left.GetIntProp("FixtureIndex"), right.GetIntProp("FixtureIndex")}
            return (0.2 if 2 in pair else 0.99, 0.0, 0.0)

        self.assertEqual(
            diverse_order(records, prefix=3, similarity=similarity), [1, 3, 2, 0]
        )
        self.assertEqual(diverse_order([], similarity=similarity), [])

    @staticmethod
    def geometry(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise RuntimeError("Fixture embedding failed")
        return mol

    def test_identity_normalizes_ordinary_hydrogen_but_preserves_stereo_and_isotopes(
        self,
    ):
        expected = Chem.MolFromSmiles("CN(CC(=O)O)C(=N)N")
        check_identity(Chem.MolFromSmiles("[H]/N=C(/N)N(C)CC(=O)O"), expected)
        with self.assertRaises(ValueError):
            check_identity(
                Chem.MolFromSmiles("F/C=C\\F"), Chem.MolFromSmiles("F/C=C/F")
            )
        with self.assertRaises(ValueError):
            check_identity(Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("[2H]CO"))

    def test_one_minimization_stereo_failure_keeps_the_other_conformer(self):
        valid = self.geometry("F/C=C/F")
        invalid = Chem.Mol(valid)
        invalid.SetBoolProp("fixture_flip", True)
        wrong_positions = self.geometry("F/C=C\\F").GetConformer().GetPositions()

        def field(mol, props):
            result = Mock()

            def minimize(**kwargs):
                if mol.HasProp("fixture_flip"):
                    for index, point in enumerate(wrong_positions):
                        mol.GetConformer().SetAtomPosition(index, point)
                return 0

            result.Minimize.side_effect = minimize
            result.CalcEnergy.return_value = 1.0
            return result

        with patch(
            "_david.code.chemistry.AllChem.MMFFGetMoleculeForceField", side_effect=field
        ):
            accepted, statuses = optimize_conformers([invalid, valid], Optimization())
        self.assertEqual(statuses, [IDENTITY_REJECTED, 0])
        self.assertEqual(len(accepted), 1)
        check_identity(accepted[0], valid)

    def test_reference_filters_each_bad_embedding_instead_of_its_stereoisomer(self):
        valid, invalid = self.geometry("F/C=C/F"), self.geometry("F/C=C\\F")

        def embed(mol, **kwargs):
            for fixture in [valid, invalid]:
                mol.AddConformer(fixture.GetConformer(), assignId=True)
            return [0, 1]

        preparer = PrepareMolecule(
            num_conformers=2, rmsd_cutoff=None, optimization=Optimization("none")
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "_david.code.reference.AllChem.EmbedMultipleConfs", side_effect=embed
            ),
        ):
            path = preparer.generate_reference_conformers(
                Chem.MolFromSmiles("F/C=C/F"), Path(directory) / "reference.sdf"
            )
            records = list(Chem.SDMolSupplier(str(path), removeHs=False))
        self.assertEqual(len(records), 1)
        self.assertEqual(len(preparer.last_identity_rejections), 1)
        self.assertEqual(records[0].GetIntProp("IdentityRejectedConformers"), 1)

    def test_reference_cap_cycles_identities_without_looking_at_candidate_scores(self):
        records = [Chem.MolFromSmiles("CCO") for _ in range(4)]
        for mol, identity in zip(records, ["first", "first", "second", "second"]):
            mol.SetProp("EnumeratedIsomerSMILES", identity)
        self.assertEqual(balanced_reference_order(records, [0, 1, 2, 3]), [0, 2, 1, 3])
        self.assertEqual(balanced_reference_order(records, []), [])

    def test_worker_error_does_not_hide_later_completed_inputs(self):
        failed, completed = Future(), Future()
        failed.set_exception(ValueError("inspect this input"))
        completed.set_result(2)
        executor = Mock()
        executor.submit.side_effect = [failed, completed]
        manifest = dict(state="measuring", completed_molecules=[], total_molecules=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw_data"
            output.mkdir()
            errors = collect_jobs(
                executor, [{"molecule_id": 1}, {"molecule_id": 2}], output, manifest
            )
            self.assertEqual(manifest["completed_molecules"], [2])
            self.assertIn("1", errors)
            self.assertTrue((Path(directory) / "progress.json").exists())
            self.assertIn(
                "inspect this input",
                (Path(directory) / "logs/input_1.error.log").read_text(),
            )

    def test_incompatible_tautomer_stereo_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "loses assigned stereochemistry"):
            prepare_candidate_molecule("C/C(O)=C/C")

    def test_preparation_does_not_erase_assigned_amino_acid_stereo(self):
        smiles = "CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        original = Chem.MolFromSmiles(smiles)
        prepared = prepare_candidate_molecule(smiles, protonate=False)
        self.assertTrue(prepared.HasSubstructMatch(original, useChirality=True))
        generator = CandidateConformerGenerator(max_isomers=4)
        self.assertEqual(len(list(generator._get_isomers(prepared))), 1)
        unresolved = prepare_candidate_molecule(
            "CC(=O)NC(Cc1ccc(O)cc1)C(=O)O", protonate=False
        )
        self.assertEqual(len(list(generator._get_isomers(unresolved))), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = generator.genConformers([smiles], directory)
            records = list(Chem.SDMolSupplier(path, removeHs=False))
        self.assertTrue(records)
        for record in records:
            assigned = Chem.Mol(record)
            Chem.AssignStereochemistryFrom3D(assigned, replaceExistingTags=True)
            check_identity(assigned, prepare_candidate_molecule(smiles))

    def test_ph_preparation_cannot_silently_skip_missing_or_failed_dependency(self):
        with patch(
            "_david.code.chemistry.core_protonation._DIMORPHITE_AVAILABLE", False
        ):
            with self.assertRaisesRegex(ImportError, "requires dimorphite_dl"):
                prepare_candidate_molecule("CCN", protonate=True)
        with patch(
            "_david.code.chemistry.core_protonation.dimorphite_dl.protonate_smiles",
            return_value=[],
        ):
            with self.assertRaisesRegex(ValueError, "no valid chemical state"):
                prepare_candidate_molecule("CCN", protonate=True)
        prepared = prepare_candidate_molecule("CCN", protonate=True)
        self.assertEqual(Chem.GetFormalCharge(prepared), 1)
        self.assertEqual(Chem.GetFormalCharge(prepare_candidate_molecule("CCN")), 0)

    def test_embedding_pruning_is_explicit_and_optional(self):
        default = CandidateConformerGenerator()._create_fresh_etkdg()
        unpruned = CandidateConformerGenerator(
            embedding_rmsd=None
        )._create_fresh_etkdg()
        self.assertEqual(default.pruneRmsThresh, 0.5)
        self.assertEqual(unpruned.pruneRmsThresh, -1)

    def test_pruning_preserves_stereo_lowest_energy_and_coordinates(self):
        first = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F"))
        other = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)F"))
        for mol in (first, other):
            self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        duplicate = Chem.Mol(first)
        for mol, energy in zip((first, duplicate, other), (2.0, 1.0, 3.0)):
            mol.SetDoubleProp("Energy_kcal_mol", energy)
        before = [
            m.GetConformer().GetPositions().copy() for m in (first, duplicate, other)
        ]
        self.assertEqual(select_conformers([first, duplicate, other], 1.25), [1, 2])
        self.assertEqual(select_conformers([first, duplicate, other], None), [1, 0, 2])
        for mol, positions in zip((first, duplicate, other), before):
            np.testing.assert_array_equal(mol.GetConformer().GetPositions(), positions)

    def test_summary_keeps_failures_and_does_not_weight_by_conformer_count(self):
        frame = pd.DataFrame(
            [
                dict(
                    molecule_id=i,
                    reference_attempts=50,
                    candidate_attempts=20,
                    reference_rmsd=0.5,
                    candidate_rmsd=0.0,
                    score=score,
                    color_excluded=False,
                    seconds=1.0,
                    actual_references=references,
                    actual_candidates=1,
                    reference_seconds=1.0,
                    reference_failure=score == 0,
                    candidate_failure=False,
                )
                for i, score, references in [(1, 1.0, 10), (2, 0.0, 0)]
            ]
        )
        summary = summarize(frame).iloc[0]
        self.assertEqual(summary["mean"], 0.5)
        self.assertEqual(summary.scored_inputs, 2)
        self.assertEqual(summary.reference_failures, 1)
        self.assertEqual(summary.fraction_ge_095, 0.5)

    def test_batch_conformer_generation_does_not_leak_max_iterations_across_molecules(
        self,
    ):
        generator = CandidateConformerGenerator(max_conformers=2, max_isomers=1)
        seen_max_iterations: list[int] = []
        original_embed = AllChem.EmbedMultipleConfs

        def spying_embed(mol, **kwargs):
            params = kwargs["params"]
            seen_max_iterations.append(int(params.maxIterations))
            return original_embed(mol, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "_david.code.chemistry.AllChem.EmbedMultipleConfs",
                side_effect=spying_embed,
            ),
        ):
            generator.genConformers(["CCO", "CCCCCCCCCC"], directory)
        self.assertEqual(seen_max_iterations, [0, 0])

    def test_verify_disjoint_conformer_seeds_detects_collisions_and_overflow(self):
        verify_disjoint_conformer_seeds(104729, 50, 42, 75)
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_disjoint_conformer_seeds(43, 50, 42, 75)
        with self.assertRaisesRegex(ValueError, "32-bit"):
            verify_disjoint_conformer_seeds(2**30, 4, 42, 10)

    def test_diverse_reference_preparation_and_multi_isomer_support(self):
        preparer = PrepareMolecule(
            num_conformers=4,
            max_reference_conformers=2,
            max_reference_isomers=2,
            selection_mode="diverse",
            optimization=Optimization("none"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = preparer.generate_reference_conformers(
                "CC(O)F", Path(directory) / "diverse_ref.sdf"
            )
            records = list(Chem.SDMolSupplier(str(path), removeHs=False))
            loaded_mol, is_3d = preparer.load_molecule(path)
        self.assertTrue(is_3d)
        self.assertEqual(loaded_mol.GetNumConformers(), 2)
        self.assertEqual(len(records), 2)
        identities = {
            rec.GetProp("EnumeratedIsomerSMILES")
            for rec in records
            if rec.HasProp("EnumeratedIsomerSMILES")
        }
        self.assertEqual(len(identities), 2)

    def test_unscale_to_physical_metrics_prefers_explicit_raw_rocs(self):
        df_with_raw = pd.DataFrame(
            {std.SMILES: ["CCO"], std.ROCS: [0.4], "Raw_ROCS": [0.8]}
        )
        out = unscale_to_physical_metrics(df_with_raw)
        self.assertAlmostEqual(float(out[std.ROCS].iloc[0]), 0.8)

        df_already_raw = pd.DataFrame({std.SMILES: ["CCO"], std.ROCS: [0.85]})
        out_raw = unscale_to_physical_metrics(
            df_already_raw, rocs_is_normalized=False
        )
        self.assertAlmostEqual(float(out_raw[std.ROCS].iloc[0]), 0.85)


if __name__ == "__main__":
    unittest.main()
