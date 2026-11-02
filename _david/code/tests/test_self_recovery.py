"""Scientific invariants for the corrected self-recovery study."""

import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from drugex.training.scorers.rocs_rdkit import _score_single_reference
from _david.code.chemistry import CandidateConformerGenerator
from _david.code.config import DEFAULT_MOLECULE
from _david.code.conformer.sampling import plan_sampling, deduplicate_reference
from _david.code.conformer.self_recovery import score_pair


def embedded(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("Test fixture could not embed")
    return mol


class TestSelfRecovery(unittest.TestCase):
    def test_policy_preserves_stereo_and_reports_attempts_not_retained_counts(self):
        egcg = plan_sampling(Chem.MolFromSmiles(DEFAULT_MOLECULE))
        self.assertEqual(
            (egcg.rotatable_bonds, egcg.oh_nh_count, egcg.saturated_rings), (3, 8, 0)
        )
        self.assertEqual(egcg.conformers_per_isomer, 250)
        self.assertEqual([egcg.attempts(s) for s in [0.5, 1, 2]], [125, 250, 500])
        self.assertEqual(len(egcg.reference_isomers), 1)
        self.assertEqual(
            len(plan_sampling(Chem.MolFromSmiles("CC(O)F")).reference_isomers), 2
        )
        self.assertEqual(
            len(plan_sampling(Chem.MolFromSmiles("C[C@H](O)F")).reference_isomers), 1
        )
        large = plan_sampling(Chem.MolFromSmiles("Br" + "[CH](Cl)" * 8 + "F"))
        self.assertEqual(len(large.reference_isomers), 25)
        self.assertEqual(large.category, "9-25")
        self.assertTrue(large.truncated)

    def test_symmetry_duplicates_are_removed_without_mutating_the_original(self):
        mol = embedded("Cc1ccccc1", 42)
        before = mol.GetConformer().GetPositions().copy()
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        conf = reordered.GetConformer()
        for i, p in enumerate(conf.GetPositions()):
            conf.SetAtomPosition(i, (-p[1] + 5, p[0] - 3, p[2] + 7))
        kept = deduplicate_reference([mol, reordered])
        self.assertEqual(len(kept), 1)
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(), before)
        with self.assertRaises(ValueError):
            deduplicate_reference(
                [embedded("F[C@](Cl)(Br)I", 1), embedded("F[C@@](Cl)(Br)I", 2)]
            )

    def test_large_explicit_budget_is_not_silently_clamped(self):
        generator = CandidateConformerGenerator(max_conformers=500)
        self.assertEqual(generator.max_conformers, 500)
        variants = list(generator._get_isomers(Chem.MolFromSmiles("CC(O)F")))
        self.assertEqual(len(variants), 2)
        self.assertTrue(all(m.HasProp("EnumeratedIsomerSMILES") for m in variants))
        self.assertEqual(CandidateConformerGenerator().max_conformers, 10)

    def test_components_match_the_public_combo_on_the_same_winning_pose(self):
        reference = embedded("CCOc1ccccc1O", 42)
        for candidate in [reference, embedded("CCOc1ccccc1O", 123), embedded("CCN", 4)]:
            score, shape, color = score_pair(candidate, reference)
            expected = (
                _score_single_reference(
                    Chem.Mol(candidate), Chem.Mol(reference), "TanimotoCombo", True
                )
                / 2
            )
            self.assertAlmostEqual(score, expected, places=12)
            self.assertAlmostEqual(score, (shape + color) / 2, places=12)


if __name__ == "__main__":
    unittest.main()
