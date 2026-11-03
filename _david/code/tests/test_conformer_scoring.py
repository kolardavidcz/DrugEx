"""Tests for shape/color scoring mathematics and reference-bank selection."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from rdkit import Chem

from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import ScoreSpec, SelectSpec, SelectionMode
from _david.code.conformer.ensemble.select import (
    cluster_statistics,
    select_ensemble,
    selection_histogram,
    selection_overlap,
)
from _david.code.conformer.scoring.tversky import (
    ALPHAS,
    BETAS,
    TVERSKY_GRID,
    WEIGHT_GRID,
    PairwiseTversky,
    grid_scores,
    pairwise_tversky,
    score_tversky,
    tversky_from_tanimoto,
)

ENSEMBLE = Path(__file__).parent / "data" / "EGCG_ensemble_4conf.sdf"


def load_ensemble() -> tuple[Chem.Mol, ...]:
    records = tuple(
        mol
        for mol in Chem.SDMolSupplier(str(ENSEMBLE), removeHs=False)
        if mol is not None
    )
    if not records:
        raise RuntimeError(f"Could not load test ensemble {ENSEMBLE}")
    return records


class TestTverskyMathematics(unittest.TestCase):
    def test_symmetric_tversky_reduces_to_tanimoto(self) -> None:
        tanimoto = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        reference_self = np.full_like(tanimoto, 120.0)
        candidate_self = np.full_like(tanimoto, 90.0)
        result = tversky_from_tanimoto(
            tanimoto, reference_self, candidate_self, alpha=1.0, beta=1.0
        )
        np.testing.assert_allclose(result, tanimoto, atol=1e-12)

    def test_zero_overlap_stays_zero(self) -> None:
        result = tversky_from_tanimoto(
            np.asarray([0.0]),
            np.asarray([100.0]),
            np.asarray([50.0]),
            alpha=0.3,
            beta=0.7,
        )
        self.assertEqual(float(result[0]), 0.0)

    def test_asymmetry_moves_the_false_positive_floor(self) -> None:
        """A small candidate is inflated by a large beta, a bloated one by a large alpha."""
        tanimoto = np.asarray([0.4])
        reference_self = np.asarray([200.0])
        small_candidate = np.asarray([20.0])
        bloated_candidate = np.asarray([400.0])

        def score(alpha: float, beta: float, candidate: np.ndarray) -> float:
            return float(
                tversky_from_tanimoto(
                    tanimoto, reference_self, candidate, alpha=alpha, beta=beta
                )[0]
            )

        # Sub-shape weighting (small alpha) barely penalizes a missing candidate
        # volume, so small fragments score high.
        self.assertGreater(
            score(0.2, 1.8, small_candidate), score(1.0, 1.0, small_candidate)
        )
        # Super-shape weighting (small beta) barely penalizes excess candidate
        # volume, so bloated decoys score high.
        self.assertGreater(
            score(1.8, 0.2, bloated_candidate), score(1.0, 1.0, bloated_candidate)
        )
        # And the converse dials suppress them again.
        self.assertLess(
            score(1.8, 0.2, small_candidate), score(1.0, 1.0, small_candidate)
        )
        self.assertLess(
            score(0.2, 1.8, bloated_candidate), score(1.0, 1.0, bloated_candidate)
        )

    def test_grid_is_eighteen_by_nine(self) -> None:
        self.assertEqual(len(TVERSKY_GRID), 18)
        self.assertEqual(len(WEIGHT_GRID), 9)
        self.assertEqual(len({entry[0] for entry in TVERSKY_GRID}), 18)
        self.assertEqual(len({entry[0] for entry in WEIGHT_GRID}), 9)

    def test_grid_scores_shape_and_symmetry(self) -> None:
        shape = np.asarray([[0.8, 0.6], [0.5, 0.95]])
        color = np.asarray([[0.4, 0.3], [0.2, 0.5]])
        volumes = np.asarray([[100.0, 20.0], [100.0, 20.0]])
        pairwise = PairwiseTversky(
            shape=shape,
            color=color,
            reference_volumes=volumes,
            candidate_volumes=volumes,
        )
        tensor = grid_scores(pairwise)
        self.assertEqual(tensor.shape, (18, 9))
        self.assertTrue(np.all(tensor >= 0.0))
        self.assertTrue(np.all(tensor <= 1.0))

        index = int(np.argmin(np.abs(ALPHAS - 1.0) + np.abs(BETAS - 1.0)))
        weight = [entry[1] for entry in WEIGHT_GRID].index(0.5)
        expected = float(0.5 * (shape + color).max())
        self.assertAlmostEqual(tensor[index, weight], expected, places=12)


class TestTverskyOnRealConformers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_ensemble()
        cls.pairwise = pairwise_tversky(cls.records[:2], cls.records[2:])

    def test_pairwise_matrices_are_tanimoto_scaled(self) -> None:
        self.assertEqual(self.pairwise.shape.shape, (2, 2))
        self.assertEqual(self.pairwise.color.shape, (2, 2))
        self.assertTrue(np.all(self.pairwise.shape >= 0.0))
        self.assertTrue(np.all(self.pairwise.shape <= 1.0))
        self.assertTrue(np.all(self.pairwise.reference_volumes > 0.0))

    def test_asymmetric_settings_are_distinguishable(self) -> None:
        dice, _, _ = score_tversky(
            self.records[:2], self.records[2:], 0.4, 0.6, 0.5, 0.5
        )
        tanimoto_scale, _, _ = score_tversky(
            self.records[:2], self.records[2:], 0.4, 1.6, 0.5, 0.5
        )
        self.assertNotAlmostEqual(dice, tanimoto_scale, places=6)

    def test_off_grid_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_tversky(self.records[:2], self.records[2:], 0.35, 0.65, 0.5, 0.5)

    def test_default_colour_shape_setting_scores_self_recovery(self) -> None:
        combo, shape, color = score_tversky(
            self.records, self.records, 1.0, 1.0, 0.5, 0.5
        )
        self.assertAlmostEqual(combo, 1.0, places=6)
        self.assertAlmostEqual(shape, 1.0, places=6)
        self.assertAlmostEqual(color, 1.0, places=6)


class TestSelectionModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_ensemble()
        for index, record in enumerate(cls.records):
            record.SetDoubleProp("Energy_kcal_mol", float(index))
            record.SetIntProp("ConformerIndex", index)

    def test_energy_mode_picks_lowest_energies(self) -> None:
        result = select_ensemble(
            self.records, SelectSpec(mode=SelectionMode.ENERGY, cap=2)
        )
        self.assertEqual(result.retained, 2)
        self.assertEqual(result.indices, (0, 1))

    def test_cap_zero_retains_every_conformer(self) -> None:
        result = select_ensemble(
            self.records, SelectSpec(mode=SelectionMode.ENERGY, cap=0)
        )
        self.assertEqual(result.retained, len(self.records))
        self.assertEqual(result.pool_size, len(self.records))

    def test_diverse_mode_starts_from_the_lowest_energy_pose(self) -> None:
        result = select_ensemble(
            self.records, SelectSpec(mode=SelectionMode.DIVERSE, cap=2)
        )
        self.assertEqual(result.retained, 2)
        self.assertEqual(result.indices[0], 0)

    def test_butina_mode_reports_clusters(self) -> None:
        result = select_ensemble(
            self.records,
            SelectSpec(mode=SelectionMode.BUTINA, cap=0, rmsd_cutoff=0.35),
        )
        self.assertGreaterEqual(result.cluster_count, 1)
        self.assertEqual(sum(result.per_isomer_clusters), result.cluster_count)

    def test_balanced_mode_is_a_permutation_of_butina(self) -> None:
        spec = SelectSpec(mode=SelectionMode.BUTINA, cap=0)
        butina = select_ensemble(self.records, spec)
        balanced = select_ensemble(
            self.records, SelectSpec(mode=SelectionMode.BALANCED, cap=0)
        )
        self.assertEqual(set(butina.indices), set(balanced.indices))

    def test_cluster_statistics_groups_by_identity(self) -> None:
        total, per_isomer = cluster_statistics(self.records, 0.35)
        self.assertEqual(total, sum(per_isomer))
        self.assertEqual(len(per_isomer), 1)

    def test_overlap_detects_identical_and_different_banks(self) -> None:
        left = select_ensemble(self.records, SelectSpec(cap=2))
        right = select_ensemble(self.records, SelectSpec(cap=2))
        other = select_ensemble(self.records, SelectSpec(mode=SelectionMode.ENERGY, cap=2))
        self.assertAlmostEqual(selection_overlap(left, right), 1.0)
        self.assertLess(selection_overlap(left, other), 1.0)

    def test_histogram_counts_retained_per_isomer(self) -> None:
        result = select_ensemble(self.records, SelectSpec(cap=2))
        histogram = selection_histogram(self.records, result.indices)
        self.assertEqual(sum(histogram.values()), result.retained)

    def test_cache_round_trip_and_hit_reports_zero_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArtifactCache(tmp)
            spec = SelectSpec(mode=SelectionMode.DIVERSE, cap=2)
            first = select_ensemble(self.records, spec, cache=cache)
            second = select_ensemble(self.records, spec, cache=cache)
            self.assertGreater(first.seconds, 0.0)
            self.assertEqual(second.seconds, 0.0)
            self.assertEqual(second.indices, first.indices)
            self.assertEqual(cache.stats.hits, 1)

    def test_pool_identity_participates_in_the_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArtifactCache(tmp)
            spec = SelectSpec(cap=2)
            select_ensemble(self.records, spec, cache=cache)
            shorter = select_ensemble(self.records[:3], spec, cache=cache)
            # A different pool must not be served the four-conformer bank.
            self.assertTrue(all(index < 3 for index in shorter.indices))
            self.assertEqual(cache.stats.hits, 0)


class TestScoreSpecScales(unittest.TestCase):
    def test_tanimoto_spec_default_is_symmetric(self) -> None:
        spec = ScoreSpec()
        self.assertEqual((spec.alpha, spec.beta), (1.0, 1.0))
        self.assertEqual(spec.label, "tanimoto_combo")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
