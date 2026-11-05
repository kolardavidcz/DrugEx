"""Behaviour tests for the standard re-evaluation planner and reward column."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from _david.code import standard_eval as seval
from _david.code.config import PipelineConfig
from _david.code.schema import std


def _run(path: Path, *, checkpoint: bool = False, arm: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({"state": "complete", "config": {}}))
    if checkpoint:
        (path / "models").mkdir()
        (path / "models" / "SequenceRNN_reinforced.pkg").write_bytes(b"weights")
    if arm:
        (path / "raw_data").mkdir()
        pd.DataFrame({std.SMILES: ["CCO", "c1ccccc1"]}).to_csv(
            path / "raw_data" / "reinforced_raw.csv.gz", index=False
        )
    return path


class TestPlanning(unittest.TestCase):
    def test_checkpoint_wins_over_a_stored_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", checkpoint=True, arm=True)
            self.assertEqual(seval.plan_run(run).action, "resample")

    def test_arm_only_is_rescored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", arm=True)
            plan = seval.plan_run(run)
            self.assertEqual(plan.action, "rescore")
            self.assertIn("2 rows", plan.reason)

    def test_missing_artifacts_are_skipped_with_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = seval.plan_run(_run(Path(tmp) / "20261101_1200_a"))
            self.assertEqual(plan.action, "skip")
            self.assertIn("no checkpoint", plan.reason)

    def test_an_existing_standard_arm_is_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", checkpoint=True)
            target = run / seval.STANDARD_DIR
            target.mkdir()
            (target / seval.CSV_NAME).write_text("x")
            self.assertEqual(seval.plan_run(run).action, "done")

    def test_the_selected_checkpoint_is_preferred_to_the_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", checkpoint=True)
            (run / "models" / "SequenceRNN_reinforced_final_epoch.pkg").write_bytes(b"x")
            self.assertEqual(
                seval._checkpoint(run).name, "SequenceRNN_reinforced.pkg"
            )


class TestStandardReward(unittest.TestCase):
    def test_higher_rocs_yields_a_higher_standard_reward(self) -> None:
        frame = pd.DataFrame(
            {
                std.SMILES: ["CCO", "c1ccccc1"],
                std.ROCS: [0.2, 1.8],
                std.SA: [1.0, 1.0],
                std.QED: [0.7, 0.7],
            }
        )
        manifest = {
            "scoring": {
                "reward_scheme": "harrington",
                "weights": [0.5, 0.25, 0.25],
                "thresholds": [0.35, 0.5, 0.5],
            }
        }
        scored = seval._add_standard_reward(frame, manifest, PipelineConfig())
        self.assertGreater(scored["StandardReward"].iloc[1], scored["StandardReward"].iloc[0])


class TestVerify(unittest.TestCase):
    def test_a_modified_standard_arm_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root / "20261101_1200_a", checkpoint=True)
            target = run / seval.STANDARD_DIR
            target.mkdir()
            (target / seval.CSV_NAME).write_text("content")
            manifest = {
                "files": {seval.CSV_NAME: seval.sha256_file(target / seval.CSV_NAME)}
            }
            (target / seval.MANIFEST_NAME).write_text(json.dumps(manifest))
            self.assertEqual(seval.verify(root), [])
            (target / seval.CSV_NAME).write_text("changed")
            self.assertTrue(seval.verify(root))


if __name__ == "__main__":
    unittest.main()
