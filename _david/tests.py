"""Unit tests for anti-cheating multi-objective rewards and DrugEx v3 API parity in _david.

These tests empirically verify:
1. AugmentedChebyshevReward penalizes unbalanced candidates exploiting QED/SA at the expense of 3D shape.
2. GeometricMeanReward calculates mathematically correct weighted geometric means.
3. SmoothHump modifier creates smooth bell curves for molecular properties.
4. Scaffold modifier correctly detects and penalizes PAINS catechol toxicophores.
5. dummyMolsFromFragments successfully constructs graph fragment datasets from intact Murcko cores.
6. Edge case handling (empty inputs, zero scores, 1.0 scores).
"""
from __future__ import annotations

import unittest
#? [NumPy] Vectorized array test comparisons and assertions.
import numpy as np
#? [RDKit] MolFromSmiles validation and Murcko scaffold extraction.
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugex.training.scorers.modifiers import SmoothHump
from drugex.training.scorers.properties import Scaffold
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from _david.anti_cheating_rewards import AugmentedChebyshevReward, GeometricMeanReward


class TestAntiCheatingRewards(unittest.TestCase):
    """Test suite for anti-cheating reward schemes and API parity."""

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
        score_egcg = scaffold_filter(egcg_mol)
        self.assertEqual(score_egcg, 0.0)

        # Benzene has no catechol -> should pass (score 1.0)
        benzene_mol = Chem.MolFromSmiles("c1ccccc1")
        score_benzene = scaffold_filter(benzene_mol)
        self.assertEqual(score_benzene, 1.0)

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
        egcg_smiles = "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
        mol = Chem.MolFromSmiles(egcg_smiles)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        core_smiles = Chem.MolToSmiles(core)
        self.assertTrue(len(core_smiles) > 0)

        # dummyMolsFromFragments appends attachment vectors directly
        dummy_mols = dummyMolsFromFragments([core_smiles])
        self.assertEqual(len(dummy_mols), 1)
        self.assertIsNotNone(dummy_mols[0])
        self.assertTrue(Chem.MolToSmiles(dummy_mols[0]).startswith("[*]") or "*" in Chem.MolToSmiles(dummy_mols[0]))

    def test_edge_cases(self) -> None:
        """Verify empty and extreme edge cases do not throw uncaught exceptions."""
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


if __name__ == "__main__":
    unittest.main()
