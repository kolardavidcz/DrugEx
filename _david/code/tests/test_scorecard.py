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


class TestFigureEvaluationScope(unittest.TestCase):
    """Distributional figures must describe the evaluated arm, not the last epoch."""

    def _run(self, tmp: str) -> tuple[object, object]:
        from pathlib import Path

        from _david.code.run_audit.audit_rl_runs import RunAudit

        root = Path(tmp) / "20261101_1200_scope"
        (root / "raw_data").mkdir(parents=True)
        pd.DataFrame(
            [{"SMILES": "CCO", "ROCS": 0.2, "SA": 0.5, "QED": 0.5, "Valid": 1,
              "Raw_ROCS": 0.2, "Raw_SA": 1.0}]
        ).to_csv(root / "raw_data" / "reinforced_raw.csv.gz", index=False)
        pd.DataFrame(
            [{"SMILES": "CCC", "ROCS": 0.3, "SA": 0.5, "QED": 0.5, "Valid": 1,
              "Raw_ROCS": 0.3, "Raw_SA": 1.0}]
        ).to_csv(root / "raw_data" / "baseline_raw.csv.gz", index=False)
        run = RunAudit(
            path=root,
            rel=root.name,
            manifest=None,
            provenance=None,
            state="complete",
            host="",
            pbs_job_id="",
            source_revision="",
            elapsed_seconds=None,
            query="CCO",
            config={},
            fingerprint=None,
        )
        return run, root

    def test_the_standard_arm_wins_when_it_exists(self) -> None:
        import tempfile
        import json

        from _david.code.run_audit.audit_rl_runs import evaluation_frames
        from _david.code.standard_eval import CSV_NAME, MANIFEST_NAME, STANDARD_DIR

        with tempfile.TemporaryDirectory() as tmp:
            run, root = self._run(tmp)
            target = root / STANDARD_DIR
            target.mkdir()
            pd.DataFrame(
                [
                    {"SMILES": "CCN", "ROCS": 0.9, "SA": 0.5, "QED": 0.5,
                     "Valid": 1, "Raw_ROCS": 0.9, "Raw_SA": 1.0}
                ]
            ).to_csv(target / CSV_NAME, index=False)
            (target / MANIFEST_NAME).write_text(
                json.dumps({"n_requested": 5000, "resampled": True, "dataset": {}})
            )
            after, _before, provenance = evaluation_frames(run)

        self.assertEqual(provenance["eval_source"], "standard_arm")
        self.assertEqual(list(after["SMILES"]), ["CCN"])

    def test_the_historical_arm_is_the_documented_fallback(self) -> None:
        import tempfile

        from _david.code.run_audit.audit_rl_runs import evaluation_frames

        with tempfile.TemporaryDirectory() as tmp:
            run, _root = self._run(tmp)
            after, _before, provenance = evaluation_frames(run)

        self.assertEqual(provenance["eval_source"], "historical_arm")
        self.assertEqual(list(after["SMILES"]), ["CCO"])


if __name__ == "__main__":
    unittest.main()
