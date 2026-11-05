"""Behaviour tests for the self-contained run dashboard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from _david.code import dashboard
from _david.code.schema import std


def _make_run(
    path: Path,
    *,
    config: dict | None = None,
    state: str = "complete",
    arm: bool = True,
) -> Path:
    (path / "raw_data").mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "state": state,
                "config": config if config is not None else {"epochs": 200},
                "source_revision": "abc123",
            }
        )
    )
    if arm:
        pd.DataFrame(
            {
                std.SMILES: ["CCO", "c1ccccc1"],
                std.VALID: [1, 1],
                std.ROCS: [0.6, 0.8],
                "Raw_ROCS": [0.6, 0.8],
                std.SA: [0.9, 0.95],
            }
        ).to_csv(path / "raw_data" / "reinforced_raw.csv.gz", index=False)
    return path


class TestDiscovery(unittest.TestCase):
    def test_only_stamped_arms_are_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "20261101_1200_first")
            (root / "20261029_unstamped").mkdir()
            _make_run(root / "archive" / "20260101_0000_old")
            _make_run(root / "conformer" / "20260101_0000_study")
            suite = root / "20261102_0000_suite"
            _make_run(suite / "arm_a")
            names = [path.relative_to(root).as_posix() for path in dashboard.discover_runs(root)]
            self.assertEqual(names, ["20261101_1200_first", "20261102_0000_suite/arm_a"])

    def test_a_container_without_manifest_children_is_not_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20261102_0000_suite").mkdir()
            self.assertEqual(dashboard.discover_runs(root), [])


class TestConfigNormalisation(unittest.TestCase):
    def test_canonical_fields_are_projected_and_legacy_keys_reported(self) -> None:
        config = dashboard.normalize_config(
            {
                "epochs": 50,
                "rocs_color_weight": 0.7,
                "energy_window_kcal": 8.0,
                "n_eval_samples": 1000,
            }
        )
        self.assertEqual(config["epochs"], 50)
        self.assertIsNone(config["reward_scheme"])
        self.assertEqual(
            config["retired"],
            {"rocs_color_weight": 0.7, "energy_window_kcal": 8.0},
        )


class TestClustering(unittest.TestCase):
    def test_identical_configs_collapse_into_one_leaf(self) -> None:
        runs = [
            {"name": "a", "config": dashboard.normalize_config({"epochs": 200, "beta": 0.0})},
            {"name": "b", "config": dashboard.normalize_config({"epochs": 200, "beta": 0.0})},
        ]
        node = dashboard.cluster_runs(runs)
        self.assertEqual(node["kind"], "leaf")
        self.assertEqual(node["runs"], ["a", "b"])

    def test_the_most_discriminating_objective_field_splits_first(self) -> None:
        runs = [
            {"name": "a", "config": dashboard.normalize_config({"reward_scheme": "harrington", "beta": 0.0})},
            {"name": "b", "config": dashboard.normalize_config({"reward_scheme": "pareto_crowding", "beta": 0.0})},
        ]
        node = dashboard.cluster_runs(runs)
        self.assertEqual(node["kind"], "split")
        self.assertEqual(node["field"], "reward_scheme")
        self.assertEqual(len(node["children"]), 2)


class TestDesignGrid(unittest.TestCase):
    def test_an_unrun_combination_is_an_empty_cell(self) -> None:
        runs = [
            dashboard.RunSummary(
                name="a",
                path=Path("a"),
                started="",
                state="complete",
                elapsed_hours=None,
                host="",
                pbs_job_id="",
                source_revision="",
                config={"reward_scheme": "harrington", "criteria": "avg_gmean"},
            ),
            dashboard.RunSummary(
                name="b",
                path=Path("b"),
                started="",
                state="complete",
                elapsed_hours=None,
                host="",
                pbs_job_id="",
                source_revision="",
                config={"reward_scheme": "pareto_crowding", "criteria": "avg_gmean"},
            ),
        ]
        grid = dashboard.design_grids(runs)[0]
        self.assertEqual(grid["rows"], ["harrington", "pareto_crowding"])
        self.assertEqual(grid["cols"], ["avg_gmean"])
        self.assertNotIn("avg_amean", grid["cols"])


class TestStandardArm(unittest.TestCase):
    def test_a_standard_arm_is_preferred_for_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root / "20261101_1200_first")
            standard = run / "standard_eval"
            standard.mkdir()
            pd.DataFrame(
                {
                    std.SMILES: ["CCO", "c1ccccc1"],
                    std.VALID: [1, 1],
                    std.ROCS: [1.5, 1.7],
                    "Raw_ROCS": [1.5, 1.7],
                    std.SA: [0.9, 0.95],
                    std.QED: [0.6, 0.7],
                }
            ).to_csv(standard / "reinforced_standard.csv.gz", index=False)
            payload = dashboard.collect(root, include_molecules=True)
            summary = payload["runs"][0]
            self.assertTrue(summary["artifacts"]["standard_arm"])
            self.assertAlmostEqual(summary["standard_metrics"]["rocs_median"], 1.6)
            self.assertEqual(payload["best_molecules"]["global"][0]["arm"], "standard")
            self.assertEqual(payload["totals"]["with_standard_arm"], 1)


    def test_the_standard_arm_names_its_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _make_run(root / "20261105_1013_best")
            standard = run / "standard_eval"
            standard.mkdir()
            (standard / "manifest.json").write_text(json.dumps({"resampled": True}))
            (run / "models").mkdir(exist_ok=True)
            pd.DataFrame({"Epoch": [1, 2, 3], "best_epoch": [1, 2, 2]}).to_csv(
                run / "models" / "SequenceRNN_reinforced_fit.tsv", sep="\t", index=False
            )
            self.assertEqual(dashboard.standard_model(run), "best@2")

            (standard / "manifest.json").write_text(json.dumps({"resampled": False}))
            self.assertEqual(dashboard.standard_model(run), "rescore")


class TestRendering(unittest.TestCase):
    def test_payload_cannot_close_the_script_tag(self) -> None:
        payload = {
            "runs": [{"name": "</script><script>alert(1)</script>"}],
            "totals": {},
            "design_grids": [],
            "cluster_tree": {},
            "best_molecules": {"per_run": {}, "global": []},
        }
        page = dashboard.render_html(payload)
        self.assertNotIn("</script><script>alert(1)", page)
        self.assertIn("\\u003c/script", page)

    def test_page_has_no_external_asset(self) -> None:
        page = dashboard.render_html(
            {
                "runs": [],
                "totals": {},
                "design_grids": [],
                "cluster_tree": {},
                "best_molecules": {"per_run": {}, "global": []},
            }
        )
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertNotIn("<link", page)

    def test_generate_writes_the_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root / "20261101_1200_first")
            target = dashboard.generate(root, include_molecules=False)
            self.assertTrue(target.is_file())
            self.assertEqual(target.relative_to(root).as_posix(), "dashboard/index.html")
            self.assertIn("DrugEx RL run overview", target.read_text())


if __name__ == "__main__":
    unittest.main()
