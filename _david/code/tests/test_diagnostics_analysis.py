from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdShapeAlign

from _david.code.rocs import align_to_reference, score_pair
from _david.code.schema import std
from _david.code.analysis.visualization import (
    AlignedDepicter,
    _sa_desirability,
    _valid_subset,
    hypervolume_3d,
)

SEED_SMILES = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
LEAD_SMILES = "COc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2"


def _embedded(smiles: str, seed: int) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return Chem.RemoveHs(mol)


class TestRocsPoseAlignment(unittest.TestCase):
    """The exported 3D pose must be the pose that produced the reported score."""

    def test_returned_pose_reproduces_the_reported_combo(self) -> None:
        reference = _embedded(SEED_SMILES, 7)
        candidate = _embedded(LEAD_SMILES, 11)

        pose, combo, shape, color = align_to_reference(candidate, reference)

        # align_to_reference works in the reference's canonical inertial frame, which is
        # also where rdShapeAlign scores a stored pose without moving it.
        reference_frame = Chem.Mol(reference)
        rdMolTransforms.CanonicalizeConformer(reference_frame.GetConformer())
        options = rdShapeAlign.ShapeInputOptions()
        options.useColors = True
        pose_shape, pose_color = rdShapeAlign.ScoreMol(
            reference_frame, pose, options, options, -1, -1
        )

        self.assertAlmostEqual(pose_shape, shape, places=3)
        self.assertAlmostEqual(pose_color, color, places=3)
        self.assertAlmostEqual(pose_shape + pose_color, combo, places=3)

    def test_alignment_agrees_with_the_pairwise_scorer(self) -> None:
        reference = _embedded(SEED_SMILES, 7)
        candidate = _embedded(LEAD_SMILES, 11)

        _, combo, _, _ = align_to_reference(candidate, reference)
        pair_combo, _, _ = score_pair(candidate, reference)

        self.assertAlmostEqual(combo, pair_combo, places=3)

    def test_alignment_does_not_mutate_the_candidate(self) -> None:
        reference = _embedded(SEED_SMILES, 7)
        candidate = _embedded(LEAD_SMILES, 11)
        before = candidate.GetConformer().GetPositions()

        align_to_reference(candidate, reference)

        np.testing.assert_allclose(candidate.GetConformer().GetPositions(), before)


class TestHypervolume(unittest.TestCase):
    """Hypervolume is the exact volume dominated by a point set above a reference."""

    def test_single_corner_point_fills_the_unit_cube(self) -> None:
        points = np.array([[1.0, 1.0, 1.0]])
        self.assertAlmostEqual(hypervolume_3d(points, np.zeros(3)), 1.0, places=9)

    def test_hand_checked_three_point_union(self) -> None:
        points = np.array(
            [[1.0, 1.0, 0.5], [1.0, 0.5, 1.0], [0.5, 1.0, 1.0]]
        )
        # |A u B u C| = 3 * 0.5 - 3 * 0.25 + 0.125
        self.assertAlmostEqual(hypervolume_3d(points, np.zeros(3)), 0.875, places=9)

    def test_points_below_the_reference_contribute_nothing(self) -> None:
        points = np.array([[0.5, 0.5, 0.5], [-0.1, 0.2, 0.3]])
        self.assertAlmostEqual(hypervolume_3d(points, np.ones(3) * 0.6), 0.0, places=9)

    def test_nan_rows_are_ignored(self) -> None:
        clean = np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0]])
        noisy = np.vstack([clean, [np.nan, 0.5, 0.5]])
        self.assertAlmostEqual(
            hypervolume_3d(noisy, np.zeros(3)),
            hypervolume_3d(clean, np.zeros(3)),
            places=12,
        )


class TestSaDesirability(unittest.TestCase):
    """SA must be mapped through the run's own clipped desirability bounds."""

    def test_clips_between_the_run_bounds(self) -> None:
        values = _sa_desirability([2.0, 3.0, 4.0, 5.0, 6.0], {})
        np.testing.assert_allclose(values, [1.0, 1.0, 0.5, 0.0, 0.0])

    def test_honours_non_default_bounds(self) -> None:
        values = _sa_desirability([4.0], {"sa_score_upper": 4.0, "sa_score_lower": 6.0})
        np.testing.assert_allclose(values, [1.0])
        values = _sa_desirability([5.0], {"sa_score_upper": 4.0, "sa_score_lower": 6.0})
        np.testing.assert_allclose(values, [0.5])

    def test_degenerate_bounds_do_not_divide_by_zero(self) -> None:
        values = _sa_desirability([1.0, 4.0], {"sa_score_upper": 3.0, "sa_score_lower": 3.0})
        np.testing.assert_allclose(values, [0.0, 0.0])


class TestValidSubset(unittest.TestCase):
    """Unparsable generations carry ROCS = 0 and must never enter a ROCS statistic."""

    def test_invalid_rows_are_dropped(self) -> None:
        frame = pd.DataFrame(
            {
                std.SMILES: ["CCO", "not_a_smiles"],
                std.VALID: [1.0, 0.0],
                std.ROCS: [0.8, 0.0],
            }
        )
        kept = _valid_subset(frame)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[std.SMILES].iloc[0], "CCO")

    def test_frame_without_a_validity_column_is_returned_unchanged(self) -> None:
        frame = pd.DataFrame({std.SMILES: ["CCO", "CCC"]})
        self.assertEqual(len(_valid_subset(frame)), 2)


class TestDepictionAlignment(unittest.TestCase):
    """Every card must place the shared core at the same 2D coordinates."""

    @staticmethod
    def _matched_rmsd(mol: Chem.Mol, template: Chem.Mol) -> float:
        template_xy = template.GetConformer().GetPositions()
        positions = mol.GetConformer().GetPositions()
        rmsds = [
            float(
                np.sqrt(
                    ((positions[list(match)] - template_xy) ** 2).sum(axis=1).mean()
                )
            )
            for match in mol.GetSubstructMatches(template)
        ]
        return min(rmsds) if rmsds else float("inf")

    def test_seed_core_is_used_when_every_molecule_contains_it(self) -> None:
        depicter = AlignedDepicter("Oc1ccc(cc1)C(=O)O")
        molecules = [
            Chem.MolFromSmiles("Oc1ccc(cc1)C(=O)O"),
            Chem.MolFromSmiles("Oc1ccc(cc1)CC(=O)O"),
            Chem.MolFromSmiles("Nc1ccc(cc1)C(=O)OC"),
        ]
        depicter.prepare_template(molecules)

        self.assertIsNotNone(depicter.template_mol)
        assert depicter.template_mol is not None
        self.assertTrue(all(mol.HasSubstructMatch(depicter.template_mol) for mol in molecules))
        for mol in molecules:
            aligned = depicter.align_molecule(mol)
            self.assertLess(self._matched_rmsd(aligned, depicter.template_mol), 1e-6)

    def test_common_substructure_is_used_when_the_seed_core_is_absent(self) -> None:
        depicter = AlignedDepicter("c1ccc2ccccc2c1")
        molecules = [
            Chem.MolFromSmiles("c1ccc2ccccc2c1"),
            Chem.MolFromSmiles("Oc1ccccc1"),
            Chem.MolFromSmiles("Nc1ccccc1"),
        ]
        depicter.prepare_template(molecules)

        self.assertIsNotNone(depicter.template_mol)
        assert depicter.template_mol is not None
        self.assertGreaterEqual(depicter.template_mol.GetNumHeavyAtoms(), 4)
        self.assertTrue(all(mol.HasSubstructMatch(depicter.template_mol) for mol in molecules))
        for mol in molecules:
            aligned = depicter.align_molecule(mol)
            self.assertLess(self._matched_rmsd(aligned, depicter.template_mol), 1e-6)

    def test_alignment_leaves_the_input_molecule_untouched(self) -> None:
        depicter = AlignedDepicter("Oc1ccc(cc1)C(=O)O")
        molecule = Chem.MolFromSmiles("Oc1ccc(cc1)CC(=O)O")
        depicter.prepare_template([molecule])
        before = Chem.Mol(molecule)

        depicter.align_molecule(molecule)

        self.assertEqual(Chem.MolToSmiles(molecule), Chem.MolToSmiles(before))


if __name__ == "__main__":
    unittest.main()
