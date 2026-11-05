"""Behaviour tests for the output layout normalizer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from _david.code import output_organization as org


def _make_run(path: Path, *, state: str = "complete", arm: str = "") -> Path:
    """Create a minimal RL arm on disk (manifest with config + reinforced arm)."""
    (path / "raw_data").mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"state": state, "config": {"epochs": 200}})
    )
    (path / "raw_data" / "reinforced_raw.csv.gz").write_text(arm or f"arm:{path.name}")
    stamp = datetime(2026, 11, 1, 14, 41, 0).timestamp()
    os.utime(path / "manifest.json", (stamp, stamp))
    return path


class TestNaming(unittest.TestCase):
    def test_existing_stamp_is_preserved(self) -> None:
        self.assertEqual(
            org.standard_name("20261104_0445_egcg_gold_oe", datetime(2030, 1, 1)),
            "20261104_0445_egcg_gold_oe",
        )

    def test_missing_stamp_uses_the_start_time(self) -> None:
        self.assertEqual(
            org.standard_name("sprint_h1_pareto", datetime(2026, 11, 1, 14, 41)),
            "20261101_1441_sprint_h1_pareto",
        )

    def test_unknown_start_leaves_the_name_alone(self) -> None:
        self.assertEqual(org.standard_name("mystery", None), "mystery")


class TestClassification(unittest.TestCase):
    def test_manifest_with_config_is_an_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp) / "run")
            self.assertTrue(org.is_rl_run(run))

    def test_arm_without_manifest_is_still_an_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "legacy"
            (run / "raw_data").mkdir(parents=True)
            (run / "raw_data" / "reinforced_raw.csv.gz").write_text("x")
            self.assertTrue(org.is_rl_run(run))

    def test_plain_directory_is_not_an_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "conformer_study"
            plain.mkdir()
            (plain / "data.csv").write_text("x")
            self.assertFalse(org.is_rl_run(plain))


class TestPlan(unittest.TestCase):
    def _tree(self, tmp: str) -> tuple[Path, Path]:
        root = Path(tmp) / "output"
        root.mkdir()
        (root / "archive").mkdir()
        # A group folder holding one already-stamped arm and a loose README.
        group = root / "20261027_0130_rl_suite"
        group.mkdir()
        _make_run(group / "20261030_1700_rl_option_one")
        suite = group / "20261031_0100_rl_suite"
        _make_run(suite / "rl01_arm")
        aborted = group / "rl01_arm.incomplete_20261030_032504"
        aborted.mkdir()
        (aborted / "logs").mkdir()
        (aborted / "logs" / "run.log").write_text("stopped")
        (group / "README.md").write_text("group notes")
        # An unstamped arm at the root.
        _make_run(root / "sprint_h1_pareto")
        # Two byte-identical copies, one stamped and one not.
        _make_run(root / "20261102_0630_sprint_h2", arm="same-arm")
        _make_run(root / "sprint_h2_copy", arm="same-arm")
        # A non-RL study tree and an empty leftover.
        study = root / "20261028_conformer_self_similarity"
        study.mkdir()
        (study / "data.csv").write_text("x")
        (root / "empty_run").mkdir()
        return root, group

    def test_plan_flattens_categories_renames_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._tree(tmp)
            result = org.plan(root)
            kinds = {action.kind for action in result.actions}
            self.assertEqual(
                kinds,
                {
                    "flatten",
                    "group_file",
                    "category",
                    "duplicate",
                    "rename",
                    "empty",
                    "variant",
                },
            )
            # Every arm that reaches the root without a stamp is stamped, including
            # an arm that was flattened out of a nested suite (the layout contract is
            # one stamped arm per directory, not one stamped container of arms).
            renames = {a.target.name for a in result.actions if a.kind == "rename"}
            self.assertIn("20261101_1441_sprint_h1_pareto", renames)
            self.assertIn("20261101_1441_rl01_arm", renames)
            self.assertTrue(all(org.has_stamp(name) for name in renames))
            duplicates = [a for a in result.actions if a.kind == "duplicate"]
            self.assertEqual([a.source.name for a in duplicates], ["sprint_h2_copy"])
            self.assertEqual(
                [a.target.parts[-2] for a in result.actions if a.kind == "category"],
                ["conformer"],
            )

    def test_plan_does_not_touch_the_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, group = self._tree(tmp)
            org.plan(root)
            self.assertTrue((group / "20261030_1700_rl_option_one").is_dir())
            self.assertTrue((root / "sprint_h1_pareto").is_dir())

    def test_apply_is_idempotent_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, group = self._tree(tmp)
            executed = org.apply(root)
            self.assertTrue(executed)
            self.assertTrue((root / "20261101_1441_sprint_h1_pareto").is_dir())
            self.assertTrue((root / "20261030_1700_rl_option_one").is_dir())
            self.assertTrue(list(root.glob("*_rl01_arm")), "nested arm must be flattened")
            self.assertFalse(group.exists(), "dissolved group folder must not linger")
            self.assertTrue((root / "conformer" / "self_similarity").is_dir())
            self.assertTrue(
                (
                    root
                    / "archive"
                    / "rl_suite_variants"
                    / "rl01_arm.incomplete_20261030_032504"
                ).is_dir()
            )
            self.assertTrue(
                (root / "archive" / "duplicate_runs" / "sprint_h2_copy").is_dir()
            )
            log = (root / org.LOG_NAME).read_text()
            self.assertIn("normalized to", log)
            self.assertFalse(org.plan(root), "a normalized tree must plan no actions")


if __name__ == "__main__":
    unittest.main()
