"""Behaviour tests for the audit's copy folding rules."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
