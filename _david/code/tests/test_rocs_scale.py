"""Behaviour tests for the ROCS scale audit."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _david.code import rocs_scale as rs
from _david.code.schema import std


def _frame(rocs: list[float], raw: list[float] | None = None) -> pd.DataFrame:
    data = {std.SMILES: [f"C{'C' * i}" for i in range(len(rocs))], std.VALID: [1] * len(rocs)}
    data[std.ROCS] = rocs
    if raw is not None:
        data["Raw_ROCS"] = raw
    return pd.DataFrame(data)


class TestClassification(unittest.TestCase):
    def test_raw_rocs_equal_to_rocs_means_the_arm_is_raw(self) -> None:
        verdict = rs.classify(_frame([0.4, 1.2], [0.4, 1.2]), {"rocs_use_modifier": False})
        self.assertEqual(verdict.verdict, rs.RAW)
        self.assertEqual(verdict.factor, 1.0)

    def test_raw_rocs_at_the_legacy_clip_recovers_that_clip(self) -> None:
        verdict = rs.classify(
            _frame([0.5, 1.0], [0.9, 1.8]),
            {"rocs_use_modifier": True, "rocs_clip_upper": 1.8},
        )
        self.assertEqual(verdict.verdict, rs.NORMALIZED)
        self.assertAlmostEqual(verdict.factor, 1.8)
        self.assertAlmostEqual(verdict.clip, 1.8)
        self.assertEqual(verdict.n_saturated, 1)
        self.assertTrue(verdict.is_inflated)

    def test_clip_two_normalization_is_not_inflated(self) -> None:
        verdict = rs.classify(
            _frame([0.5, 1.0], [1.0, 2.0]),
            {"rocs_use_modifier": True, "rocs_clip_upper": 2.0},
        )
        self.assertEqual(verdict.verdict, rs.NORMALIZED)
        self.assertAlmostEqual(verdict.factor, 2.0)
        self.assertFalse(verdict.is_inflated)

    def test_a_ratio_that_matches_no_clip_is_unknown(self) -> None:
        verdict = rs.classify(
            _frame([0.9, 0.4], [0.9, 0.4 * 3.0]),
            {"rocs_use_modifier": True, "rocs_clip_upper": 1.8},
        )
        self.assertEqual(verdict.verdict, rs.UNKNOWN)

    def test_a_legacy_manifest_without_the_field_implies_the_modifier(self) -> None:
        modifier, clip = rs.era_scoring({"rocs_clip_upper": 1.8})
        self.assertTrue(modifier)
        self.assertAlmostEqual(clip, 1.8)

    def test_a_manifest_with_the_field_is_taken_at_its_word(self) -> None:
        modifier, clip = rs.era_scoring({"rocs_use_modifier": False, "rocs_clip_upper": 1.8})
        self.assertFalse(modifier)
        self.assertAlmostEqual(clip, 1.8)

    def test_without_a_config_the_run_date_decides(self) -> None:
        early = datetime(2026, 10, 27, 20, 35, tzinfo=timezone.utc)
        late = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(rs.era_scoring({}, early), (True, rs.LEGACY_CLIP))
        self.assertEqual(rs.era_scoring({}, late), (False, rs.CURRENT_CLIP))

    def test_a_pure_unit_range_arm_with_the_modifier_on_is_normalized(self) -> None:
        verdict = rs.classify(_frame([0.2, 0.46]), {}, datetime(2026, 10, 27, tzinfo=timezone.utc))
        self.assertEqual(verdict.verdict, rs.NORMALIZED)
        self.assertAlmostEqual(verdict.factor, 1.8)

    def test_a_pure_unit_range_arm_with_out_the_modifier_is_unknown(self) -> None:
        verdict = rs.classify(_frame([0.2, 0.46]), {}, datetime(2026, 11, 2, tzinfo=timezone.utc))
        self.assertEqual(verdict.verdict, rs.UNKNOWN)

    def test_an_all_zero_arm_with_matching_raw_rocs_is_raw_not_unknown(self) -> None:
        verdict = rs.classify(_frame([0.0, 0.0], [0.0, 0.0]), {"rocs_use_modifier": False})
        self.assertEqual(verdict.verdict, rs.RAW)

    def test_values_above_one_prove_the_arm_is_raw(self) -> None:
        verdict = rs.classify(_frame([0.2, 1.4]), {}, datetime(2026, 11, 2, tzinfo=timezone.utc))
        self.assertEqual(verdict.verdict, rs.RAW)


class TestCorrection(unittest.TestCase):
    def test_a_normalized_arm_without_raw_rocs_is_scaled_by_the_clip(self) -> None:
        verdict = rs.ScaleVerdict(run="r", verdict=rs.NORMALIZED, clip=1.8, factor=1.8, n_rows=2)
        corrected = rs.correct_frame(_frame([0.5, 1.0]), verdict)
        self.assertAlmostEqual(corrected[std.ROCS].iloc[0], 0.9)
        self.assertAlmostEqual(corrected["Raw_ROCS"].iloc[1], 1.8)
        self.assertEqual(corrected["ROCS_Saturated"].tolist(), [0, 1])

    def test_an_arm_with_raw_rocs_prefers_that_column(self) -> None:
        verdict = rs.ScaleVerdict(run="r", verdict=rs.NORMALIZED, clip=1.8, factor=1.8, n_rows=2)
        corrected = rs.correct_frame(_frame([0.5, 1.0], [0.9, 1.8]), verdict)
        self.assertAlmostEqual(corrected[std.ROCS].iloc[0], 0.9)
        self.assertAlmostEqual(corrected[std.ROCS].iloc[1], 1.8)
        self.assertEqual(corrected["ROCS_Saturated"].tolist(), [0, 1])

    def test_a_raw_arm_is_left_alone(self) -> None:
        verdict = rs.ScaleVerdict(run="r", verdict=rs.RAW, clip=None, factor=1.0, n_rows=2)
        corrected = rs.correct_frame(_frame([0.4, 1.2]), verdict)
        self.assertEqual(corrected[std.ROCS].tolist(), [0.4, 1.2])
        self.assertEqual(corrected["ROCS_Saturated"].tolist(), [0, 0])

    def test_a_run_without_an_arm_is_unknown_and_uncorrected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "20261101_1200_empty"
            run.mkdir()
            verdict = rs.inspect(run)
            self.assertEqual(verdict.verdict, rs.UNKNOWN)
            self.assertFalse(run.joinpath("raw_data", rs.SCALE_NAME).exists())


class TestReport(unittest.TestCase):
    def test_the_report_names_the_inflated_arm_and_its_factor(self) -> None:
        verdicts = [
            rs.ScaleVerdict(
                run="20261027_2035_old",
                verdict=rs.NORMALIZED,
                clip=1.8,
                factor=1.8,
                n_rows=100,
                n_saturated=3,
                evidence="era",
            ),
            rs.ScaleVerdict(run="20261104_0445_new", verdict=rs.RAW, clip=None, factor=1.0, n_rows=100),
        ]
        text = rs.render_report(verdicts)
        self.assertIn("inflated by the 2.0-vs-1.8 clip mismatch: **1**", text)
        self.assertIn("20261027_2035_old", text)
        self.assertIn("3/100", text)


class TestConsumers(unittest.TestCase):
    """The scale is applied once, where the value is consumed."""

    def test_the_monitor_table_recovers_raw_rocs_with_the_audited_factor(self) -> None:
        from _david.code.visualization import epoch_molecule_table

        monitor = pd.DataFrame(
            {
                "Epoch": [1, 1],
                "SMILES": ["CCO", "CCN"],
                "Valid": [1, 1],
                "Desired": [0, 0],
                "ROCS": [0.5, 1.0],
                "SA": [1.0, 1.0],
            }
        )
        raw = epoch_molecule_table(monitor, rocs_scale=1.8)
        self.assertAlmostEqual(float(raw["ROCS"].iloc[0]), 0.9)
        self.assertAlmostEqual(float(raw["ROCS"].iloc[1]), 1.8)

    def test_training_candidates_are_written_best_first_and_invalid_dropped(self) -> None:
        from _david.code.visualization import write_training_candidates

        table = pd.DataFrame(
            {
                "Epoch": [1, 2, 3],
                "SMILES": ["CCO", "CCN", "CCC"],
                "Canonical": ["CCO", "CCN", "CCC"],
                "ValidMol": [1, 1, 0],
                "Desired": [0, 1, 1],
                "ROCS": [0.4, 0.9, 0.99],
                "QED": [0.5, 0.6, 0.7],
                "RawSA": [2.0, 2.0, 2.0],
                "SimToSeed": [0.1, 0.2, 0.3],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_training_candidates(table, Path(tmp))
            self.assertIsNotNone(path)
            written = pd.read_csv(path, sep="\t")
            self.assertEqual(written["SMILES"].tolist(), ["CCN", "CCO"])


if __name__ == "__main__":
    unittest.main()
