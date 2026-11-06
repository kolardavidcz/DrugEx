"""Behaviour tests for the standard re-evaluation planner and reward column."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from _david.code.analysis import standard_eval as seval
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
            (target / seval.MANIFEST_NAME).write_text("{}")
            self.assertEqual(seval.plan_run(run).action, "done")

    def test_a_half_written_arm_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", checkpoint=True)
            target = run / seval.STANDARD_DIR
            target.mkdir()
            (target / seval.CSV_NAME).write_text("truncated")
            self.assertEqual(seval.plan_run(run).action, "resample")

    def test_the_selected_checkpoint_is_preferred_to_the_final_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp) / "20261101_1200_a", checkpoint=True)
            (run / "models" / "SequenceRNN_reinforced_final_epoch.pkg").write_bytes(b"x")
            self.assertEqual(
                seval._checkpoint(run).name, "SequenceRNN_reinforced.pkg"
            )


class TestStandardReward(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                std.SMILES: ["CCO", "c1ccccc1"],
                std.ROCS: [0.2, 1.8],
                std.SA: [1.0, 1.0],
                std.QED: [0.7, 0.7],
            }
        )

    def test_higher_rocs_yields_a_higher_standard_reward(self) -> None:
        manifest = {
            "scoring": {
                "reward_scheme": "harrington",
                "weights": [0.5, 0.25, 0.25],
                "thresholds": [0.35, 0.5, 0.5],
            }
        }
        scored = seval._add_standard_reward(self._frame(), manifest, PipelineConfig())
        self.assertGreater(scored["StandardReward"].iloc[1], scored["StandardReward"].iloc[0])

    def test_the_reward_does_not_depend_on_the_dataset_build_config(self) -> None:
        two_objective = {"scoring": {"weights": [0.75, 0.25], "thresholds": [0.35, 0.5]}}
        three_objective = {
            "scoring": {"weights": [0.5, 0.25, 0.25], "thresholds": [0.35, 0.5, 0.5]}
        }
        first = seval._add_standard_reward(self._frame(), two_objective, PipelineConfig())
        second = seval._add_standard_reward(self._frame(), three_objective, PipelineConfig())
        np.testing.assert_allclose(
            first["StandardReward"], second["StandardReward"]
        )


class TestPush(unittest.TestCase):
    def test_dry_run_copies_inputs_but_not_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root / "20261101_1200_a", checkpoint=True, arm=True)
            (run / "figures").mkdir()
            (run / "figures" / "01.png").write_text("png")
            commands = seval.push_runs(
                ["20261101_1200_a"], root=root, dry_run=True
            )
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertIn("20261101_1200_a/models/SequenceRNN_reinforced.pkg", command)
            self.assertIn("20261101_1200_a/manifest.json", command)
            self.assertNotIn("figures", command)
            self.assertTrue(command.startswith("rsync -aR"))


class TestPull(unittest.TestCase):
    def test_an_archived_run_is_not_resurrected_as_a_stub(self) -> None:
        """Pulling every remote arm recreated archived runs as bare ``standard_eval`` dirs.

        Only arms of runs that exist locally may land locally; the archived copies
        already hold their own arm, and a manifest-less directory is not a run.
        """
        storage = "/storage/test/DrugEx"
        listing = (
            f"{storage}/_david/output/20261101_1200_present/standard_eval\n"
            f"{storage}/_david/output/20261101_1200_gone/standard_eval\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20261101_1200_present").mkdir()
            completed = mock.Mock(stdout=listing)
            with (
                mock.patch(
                    "_david.code.metacentrum.submit.remote", return_value=completed
                ),
                mock.patch(
                    "_david.code.metacentrum.submit.storage_root", return_value=storage
                ),
                mock.patch("subprocess.run") as run,
            ):
                copied = seval.pull_standard_arms(root)
            self.assertEqual(
                copied, [(root / "20261101_1200_present" / "standard_eval").as_posix()]
            )
            self.assertFalse((root / "20261101_1200_gone").exists())
            self.assertEqual(run.call_count, 1)


DATASET_NAME = "prior_test"
DATASET_RECORD = {
    "name": DATASET_NAME,
    "manifest_sha256": "a" * 64,
    "csv_sha256": "b" * 64,
    "reference_sha256": "c" * 64,
}


def _dataset(root: Path) -> Path:
    path = root / DATASET_NAME
    path.mkdir(parents=True, exist_ok=True)
    (path / seval.MANIFEST_NAME).write_text(
        json.dumps({"scoring": {"sa_score_lower": 5.0, "sa_score_upper": 3.0}})
    )
    return path


def _arm(root: Path, name: str, **overrides) -> Path:
    run = _run(root / name, checkpoint=True)
    target = run / seval.STANDARD_DIR
    target.mkdir()
    (target / seval.CSV_NAME).write_text(name)
    manifest = {
        "run": name,
        "resampled": True,
        "n_requested": 5000,
        "dataset": dict(DATASET_RECORD),
        "files": {seval.CSV_NAME: seval.sha256_file(target / seval.CSV_NAME)},
    }
    manifest.update(overrides)
    (target / seval.MANIFEST_NAME).write_text(json.dumps(manifest))
    return run


class TestProtocol(unittest.TestCase):
    """Two arms are comparable only when their scoring protocol hash agrees."""

    def setUp(self) -> None:
        self.cfg = seval._standard_config({"scoring": {}})

    def test_execution_only_fields_do_not_change_the_hash(self) -> None:
        other = PipelineConfig(
            **{
                **seval.asdict(self.cfg),
                "gpu": "cpu",
                "conformer_threads": 8,
                "batch_size": 8,
                "epochs": 7,
            }
        )
        first = seval.scoring_protocol(self.cfg, DATASET_RECORD)
        second = seval.scoring_protocol(other, DATASET_RECORD)
        self.assertEqual(first["sha256"], second["sha256"])

    def test_validation_holdout_knobs_do_not_change_the_hash(self) -> None:
        """Training-only knobs must never invalidate a finished arm's comparability.

        The ``val_loss`` criterion added five ``PipelineConfig`` fields that a
        re-evaluation never reads; when they entered the hash, every arm scored
        before that commit mismatched the arms scored after it.
        """
        fields = {
            "val_holdout_dataset": "some_other_dataset",
            "val_holdout_size": 64,
            "val_holdout_seed": 7,
            "val_min_valid_ratio": 0.1,
            "val_min_unique_ratio": 0.9,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                other = PipelineConfig(**{**seval.asdict(self.cfg), field: value})
                self.assertNotEqual(getattr(other, field), getattr(self.cfg, field))
                self.assertEqual(
                    seval.scoring_protocol(self.cfg, DATASET_RECORD)["sha256"],
                    seval.scoring_protocol(other, DATASET_RECORD)["sha256"],
                )

    def test_a_scoring_field_changes_the_hash(self) -> None:
        other = PipelineConfig(**{**seval.asdict(self.cfg), "max_conformers": 99})
        self.assertNotEqual(
            seval.scoring_protocol(self.cfg, DATASET_RECORD)["sha256"],
            seval.scoring_protocol(other, DATASET_RECORD)["sha256"],
        )

    def test_the_dataset_identity_changes_the_hash(self) -> None:
        other = {**DATASET_RECORD, "csv_sha256": "d" * 64}
        self.assertNotEqual(
            seval.scoring_protocol(self.cfg, DATASET_RECORD)["sha256"],
            seval.scoring_protocol(self.cfg, other)["sha256"],
        )

    def test_arms_on_one_protocol_verify_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)
            _arm(root, "20261101_1200_a")
            _arm(root, "20261101_1200_b")
            self.assertEqual(seval.verify(root, dataset_root=root), [])

    def test_a_recorded_protocol_that_no_longer_matches_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)
            _arm(root, "20261101_1200_a", protocol={"sha256": "0" * 64})
            problems = seval.verify(root, dataset_root=root)
            self.assertTrue(any("does not match" in problem for problem in problems))

    def test_arms_on_different_arm_sizes_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)
            _arm(root, "20261101_1200_a")
            _arm(root, "20261101_1200_b", n_requested=1000)
            problems = seval.verify(root, dataset_root=root)
            self.assertTrue(any("arm size" in problem for problem in problems))


class TestVerify(unittest.TestCase):
    def test_a_modified_standard_arm_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)
            run = _arm(root, "20261101_1200_a")
            target = run / seval.STANDARD_DIR
            self.assertEqual(seval.verify(root, dataset_root=root), [])
            (target / seval.CSV_NAME).write_text("changed")
            self.assertTrue(seval.verify(root, dataset_root=root))


class TestPbsScript(unittest.TestCase):
    def test_the_pretrained_checkpoint_is_exported_without_a_dataset_build(self) -> None:
        script = seval.pbs_script(
            "/storage/x", "/storage/x/current", dataset="prior"
        )
        self.assertIn('export DRUGEX_PRETRAINED_DIR="$STORAGE_DIR/models"', script)
        self.assertIn("standard_eval run", script)

    def test_a_dataset_build_keeps_the_same_pretrained_location(self) -> None:
        script = seval.pbs_script(
            "/storage/x",
            "/storage/x/current",
            dataset="prior",
            build_dataset=True,
        )
        self.assertEqual(script.count("DRUGEX_PRETRAINED_DIR"), 1)
        self.assertIn("prior_dataset build", script)


if __name__ == "__main__":
    unittest.main()
