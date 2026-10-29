"""Scientific controls for the joint calibration and shared production pruner."""

import unittest
import tempfile
from pathlib import Path
from concurrent.futures import Future
from unittest.mock import Mock, patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.reference import select_conformers
from _david.code.chemistry import (
    CandidateConformerGenerator,
    IDENTITY_REJECTED,
    Optimization,
    check_identity,
    optimize_conformers,
    prepare_candidate_molecule,
)
from _david.code.reference import PrepareMolecule
from _david.code.conformer.calibrate import summarize, collect_jobs
from _david.code.conformer.reference_caps import balanced_reference_order
import pandas as pd


class TestJointCalibration(unittest.TestCase):
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
        unresolved = prepare_candidate_molecule("CC(=O)NC(Cc1ccc(O)cc1)C(=O)O", protonate=False)
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
        with patch("_david.code.chemistry.core_protonation._DIMORPHITE_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "requires dimorphite_dl"):
                prepare_candidate_molecule("CCN", protonate=True)
        with patch("_david.code.chemistry.core_protonation.dimorphite_dl.protonate_smiles", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no valid chemical state"):
                prepare_candidate_molecule("CCN", protonate=True)
        prepared = prepare_candidate_molecule("CCN", protonate=True)
        self.assertEqual(Chem.GetFormalCharge(prepared), 1)
        self.assertEqual(Chem.GetFormalCharge(prepare_candidate_molecule("CCN")), 0)

    def test_embedding_pruning_is_explicit_and_optional(self):
        default = CandidateConformerGenerator()._create_fresh_etkdg()
        unpruned = CandidateConformerGenerator(embedding_rmsd=None)._create_fresh_etkdg()
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


if __name__ == "__main__":
    unittest.main()
