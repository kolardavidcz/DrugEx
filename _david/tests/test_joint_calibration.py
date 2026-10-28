"""Scientific controls for the joint calibration and shared production pruner."""

import unittest
import tempfile
from pathlib import Path
from concurrent.futures import Future
from unittest.mock import Mock

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.reference import select_conformers
from _david.code.chemistry import (
    CandidateConformerGenerator,
    prepare_candidate_molecule,
)
from _david.code.conformer.calibrate import summarize, collect_jobs
from _david.code.conformer.reference_caps import balanced_reference_order
import pandas as pd


class TestJointCalibration(unittest.TestCase):
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
        prepared = prepare_candidate_molecule(smiles)
        self.assertTrue(prepared.HasSubstructMatch(original, useChirality=True))
        generator = CandidateConformerGenerator(max_isomers=4)
        self.assertEqual(len(list(generator._get_isomers(prepared))), 1)
        unresolved = prepare_candidate_molecule("CC(=O)NC(Cc1ccc(O)cc1)C(=O)O")
        self.assertEqual(len(list(generator._get_isomers(unresolved))), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = generator.genConformers([smiles], directory)
            records = list(Chem.SDMolSupplier(path, removeHs=False))
        self.assertTrue(records)
        for record in records:
            assigned = Chem.Mol(record)
            Chem.AssignStereochemistryFrom3D(assigned, replaceExistingTags=True)
            self.assertTrue(assigned.HasSubstructMatch(original, useChirality=True))

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
