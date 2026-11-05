"""Check the reward-Pareto renderer and the promoted drug-panel figure module."""

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from _david.code.conformer.panel_figures import (
    FIGURE_NAME,
    MANIFEST_NAME,
    render_drug_panel_figure,
    write_figure_manifest,
)
from _david.code.conformer.visualization import (
    render_reward_pareto,
    rl_reference_measurement,
)

LABELS = {
    "etkdg": "ETKDG only",
    "mmff_high_dielectric": "MMFF94s ε=78.4",
}
KEYS = ("molecule_id", "method", "reference_cap", "candidate_attempts")


def calibration_frame() -> pd.DataFrame:
    """Two methods x (K1/C10, all/C20) x two molecules with component timings."""
    rows = []
    for molecule in range(2):
        for method in LABELS:
            for cap, attempts, seconds, gen, align, score in (
                (1, 10, 1.0, 0.6, 0.4, 0.80),
                (0, 20, 4.0, 1.0, 3.0, 0.95),
            ):
                rows.append(
                    dict(
                        molecule_id=molecule,
                        method=method,
                        reference_cap=cap,
                        candidate_attempts=attempts,
                        color_excluded=False,
                        score=score,
                        seconds=seconds,
                        generation_seconds=gen,
                        alignment_io_seconds=align,
                    )
                )
    return pd.DataFrame(rows)


def calibration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = list(KEYS[1:])
    grouped = frame.groupby(keys, sort=True)
    summary = grouped.agg(
        mean=("score", "mean"),
        mean_seconds=("seconds", "mean"),
        median_seconds=("seconds", "median"),
        p95_seconds=("seconds", lambda s: s.quantile(0.95)),
    ).reset_index()
    return summary


class TestRewardPareto(unittest.TestCase):
    def test_pareto_csv_and_component_columns_from_legacy_frame(self):
        frame = calibration_frame()
        summary = calibration_summary(frame)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "raw_data/development").mkdir(parents=True)
            render_reward_pareto(
                frame, summary, output, labels=LABELS, keys=KEYS
            )
            self.assertTrue((output / "pareto.png").exists())
            self.assertTrue((output / "pareto.svg").exists())
            saved = pd.read_csv(output / "raw_data/development/pareto.csv")
            for column in (
                "mean_generation_minimization_seconds",
                "mean_alignment_io_seconds",
                "pareto",
            ):
                self.assertIn(column, saved.columns)
            # The K1/C10 cell is dominated by nothing; all/C20 dominates on
            # score but not time, so both cells must be nondominated.
            self.assertTrue(saved.pareto.all())
            small = saved[
                saved.reference_cap.eq(1) & saved.candidate_attempts.eq(10)
            ].iloc[0]
            self.assertAlmostEqual(small.mean_generation_minimization_seconds, 0.6)
            self.assertAlmostEqual(small.mean_alignment_io_seconds, 0.4)

    def test_rl_reference_measurement_reads_live_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            raw = output / "raw_data" / "rl_smoke"
            raw.mkdir(parents=True)
            (raw / "manifest.json").write_text(
                json.dumps(
                    dict(
                        state="complete",
                        host="test-host",
                        actual_alignment_workers=2,
                        baseline=dict(
                            method="mmff_high_dielectric",
                            reference_cap=4,
                            candidate_attempts=20,
                        ),
                    )
                )
            )
            pd.DataFrame(
                [dict(rocs_seconds=20.0, environment_rows=10)]
            ).to_csv(raw / "epoch_timings.csv", index=False)
            reference = rl_reference_measurement(output)
            self.assertIsNotNone(reference)
            self.assertEqual(reference["wall_seconds_per_row"], 2.0)
            self.assertEqual(reference["serial_equivalent_seconds_per_row"], 4.0)
            self.assertEqual(
                rl_reference_measurement(output.parent / "nowhere"), None
            )


class TestDrugPanelFigure(unittest.TestCase):
    def test_renders_three_panels_with_manifest(self):
        frame = calibration_frame()
        frame["reference_failure"] = False
        # Per-molecule row order is etkdg/K1, etkdg/all, mmff/K1, mmff/all:
        # give the mmff C20 rows a single isomer so panel C has a non-empty
        # single-isomer stratum, and etkdg rows varied isomer counts.
        frame["enumerated_reference_isomers"] = [2, 3, 1, 1] * 2
        frame["actual_references"] = 4
        summary = calibration_summary(frame)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calibration = output / "calibration"
            raw = calibration / "raw_data" / "development"
            raw.mkdir(parents=True)
            frame.to_csv(raw / "data.csv.gz", index=False)
            summary.to_csv(raw / "summary.csv", index=False)
            target = render_drug_panel_figure(
                frame, summary, output, rl_note="unit-test RL note"
            )
            self.assertEqual(target.name, FIGURE_NAME)
            self.assertTrue(target.exists())
            manifest_path = write_figure_manifest(
                output, calibration, "unit-test RL note"
            )
            self.assertEqual(manifest_path.name, MANIFEST_NAME)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["panels"]), 3)
            self.assertTrue(manifest["removed_panels"])
            self.assertIn("unit-test RL note", manifest["rl_note"])
            self.assertIn("data.csv.gz", manifest["inputs"])


if __name__ == "__main__":
    unittest.main()
