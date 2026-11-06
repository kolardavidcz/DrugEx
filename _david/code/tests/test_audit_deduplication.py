"""Behaviour tests for the audit's copy folding rules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _david.code.run_audit import audit_rl_runs as audit


def _entry(path: Path, rel: str, *, archived: bool, fingerprint: str = "same") -> audit.RunAudit:
    return audit.RunAudit(
        path=path,
        rel=rel,
        manifest={"state": "complete", "config": {}},
        provenance=None,
        state="complete",
        host="",
        pbs_job_id="",
        source_revision="",
        elapsed_seconds=None,
        query="",
        config={},
        fingerprint=fingerprint,
        archived=archived,
    )


class TestDeduplication(unittest.TestCase):
    def test_a_live_run_represents_its_archived_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "20261030_1700_live"
            archived = root / "archive" / "duplicate_runs" / "live_copy"
            for path in (live, archived):
                path.mkdir(parents=True)
            # The archived copy was rendered more often; that must not promote it.
            (archived / "figures").mkdir()
            (archived / "figures" / audit.CURRENT_FIGURES[0]).write_text("png")
            runs = audit._deduplicate(
                [
                    _entry(live, "20261030_1700_live", archived=False),
                    _entry(archived, "archive/duplicate_runs/live_copy", archived=True),
                ]
            )
            self.assertEqual([run.rel for run in runs], ["20261030_1700_live"])
            self.assertEqual(runs[0].copies, ["archive/duplicate_runs/live_copy"])

    def test_distinct_fingerprints_are_never_folded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _entry(root / "a", "a", archived=False, fingerprint="one")
            second = _entry(root / "b", "b", archived=False, fingerprint="two")
            runs = audit._deduplicate([first, second])
            self.assertEqual({run.rel for run in runs}, {"a", "b"})


def _rl_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "state": "complete",
                "config": {"reward_scheme": "harrington", "n_samples": 500},
            }
        )
    )


class TestDiscovery(unittest.TestCase):
    def test_a_standard_eval_subdirectory_is_not_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _rl_manifest(root / "20261101_1200_run" / "manifest.json")
            # The re-evaluation manifest must be ignored even when its config
            # carries the RL keys the discovery rule looks for.
            _rl_manifest(root / "20261101_1200_run" / "standard_eval" / "manifest.json")
            runs = audit.discover(root)
            self.assertEqual([run.rel for run in runs], ["20261101_1200_run"])

    def test_relative_paths_are_computed_against_the_given_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _rl_manifest(root / "20261101_1200_run" / "manifest.json")
            [run] = audit.discover(root)
            self.assertEqual(run.rel, "20261101_1200_run")


class TestProvenanceRecord(unittest.TestCase):
    def test_a_run_without_an_evaluation_arm_is_reported_not_hashed(self) -> None:
        # An engineering smoke test or a calibration study keeps the run layout
        # (manifest + telemetry) but never samples an evaluation arm. The CSV
        # export used to crash on it while simply hashing the missing file.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "20261029_1155_smoke"
            (path / "models").mkdir(parents=True)
            run = audit.RunAudit(
                path=path,
                rel=path.name,
                manifest={"state": "complete", "config": {}},
                provenance=None,
                state="complete",
                host="",
                pbs_job_id="",
                source_revision="",
                elapsed_seconds=None,
                query="",
                config={"n_eval_samples": 1000},
                fingerprint="solo",
                archived=False,
            )
            record = audit._provenance_record(run)
            self.assertEqual(record["eval_source"], "missing_arm")
            self.assertIsNone(record["eval_arm_sha256"])
            self.assertIsNone(record["eval_n"])


if __name__ == "__main__":
    unittest.main()
