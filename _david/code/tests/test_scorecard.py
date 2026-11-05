"""Behaviour tests for the diagnostic scorecard's statistics contract."""

from __future__ import annotations

import unittest

import pandas as pd

from _david.code import visualization as viz


def _arm(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _scored(smiles: str, valid: int, rocs: float, sa: float, qed: float) -> dict:
    return {
        "SMILES": smiles,
        "Valid": valid,
        "ROCS": rocs,
        "SA": sa,
        "QED": qed,
        "Raw_ROCS": rocs,
        "Raw_SA": 1.0,
        "HAC": 20,
        "MW": 300.0,
        "LogP": 1.5,
        "TPSA": 60.0,
        "RotBonds": 3,
        "Canonical_SMILES": smiles,
    }


class TestScorecardStatistics(unittest.TestCase):
    """Failed generations must not be read as property values."""

    def test_unparsable_rows_do_not_drag_the_property_medians_down(self) -> None:
        before = _arm(
            [
                _scored("CCO", 1, 0.5, 0.5, 0.5),
                _scored("CCN", 1, 0.7, 0.5, 0.5),
                _scored("bad", 0, 0.0, 0.0, 0.0),
                _scored("bad2", 0, 0.0, 0.0, 0.0),
            ]
        )
        after = _arm(
            [
                _scored("CCC", 1, 0.9, 0.5, 0.8),
                _scored("bad", 0, 0.0, 0.0, 0.0),
                _scored("bad2", 0, 0.0, 0.0, 0.0),
            ]
        )
        card = viz.build_diagnostic_scorecard(before, after)
        rows = card.set_index("Property")
        self.assertAlmostEqual(float(rows.loc["ROCS (Median)", "Reinforced_Median"]), 0.9)
        self.assertAlmostEqual(float(rows.loc["QED (Drug-Likeness)", "Reinforced_Median"]), 0.8)
        self.assertAlmostEqual(float(rows.loc["ROCS (Median)", "Baseline_Median"]), 0.6)

    def test_validity_is_still_measured_on_the_whole_arm(self) -> None:
        before = _arm([_scored("CCO", 1, 0.5, 0.5, 0.5), _scored("CCN", 1, 0.7, 0.5, 0.5)])
        after = _arm(
            [
                _scored("CCC", 1, 0.9, 0.5, 0.8),
                _scored("bad", 0, 0.0, 0.0, 0.0),
                _scored("bad2", 0, 0.0, 0.0, 0.0),
            ]
        )
        rows = viz.build_diagnostic_scorecard(before, after).set_index("Property")
        self.assertAlmostEqual(float(rows.loc["Validity Rate (%)", "Reinforced_Median"]), 33.3)
        self.assertAlmostEqual(float(rows.loc["Validity Rate (%)", "Baseline_Median"]), 100.0)

    def test_an_all_invalid_arm_reports_its_validity_and_leaves_properties_empty(self) -> None:
        before = _arm([_scored("CCO", 1, 0.5, 0.5, 0.5)])
        after = _arm([_scored("bad", 0, 0.0, 0.0, 0.0), _scored("bad2", 0, 0.0, 0.0, 0.0)])
        rows = viz.build_diagnostic_scorecard(before, after).set_index("Property")
        self.assertAlmostEqual(float(rows.loc["Validity Rate (%)", "Reinforced_Median"]), 0.0)
        self.assertTrue(pd.isna(rows.loc["ROCS (Median)", "Reinforced_Median"]))


if __name__ == "__main__":
    unittest.main()
