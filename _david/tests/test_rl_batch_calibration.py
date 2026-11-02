"""Check the 1:1 training-interface cost calibration module.

Covers arm parsing/labels, the sprint-parity pipeline configuration, the
per-epoch stage breakdown arithmetic (including negative-residual clipping),
cold/warm timing aggregation, and an end-to-end render over synthetic arm
directories.
"""

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from _david.code.conformer.rl_batch_calibration import (
    Arm,
    _sprint_config,
    arm_label,
    epoch_stage_breakdown,
    parse_arms,
    rank_agreement,
    render_cost_summary,
    score_row,
    summarize_timing,
)


def timing_frame() -> pd.DataFrame:
    """Two arms x (cold, warm) repeats with known wall and stage seconds."""
    rows = []
    for arm, method, cap, attempts, refs, walls in (
        ("etkdg_k1_c30", "etkdg", 1, 30, 1, (12.0, 10.0, 10.0)),
        ("mmff_high_dielectric_k4_c20", "mmff_high_dielectric", 4, 20, 4, (6.0, 5.0, 5.0)),
    ):
        for repeat, wall in enumerate(walls):
            rows.append(
                dict(
                    arm=arm,
                    method=method,
                    reference_cap=cap,
                    candidate_attempts=attempts,
                    repeat=repeat,
                    cold=repeat == 0,
                    inputs=100,
                    references=refs,
                    valid=90,
                    zero_score_valid=5,
                    embedding_failures=2,
                    identity_rejections=1,
                    filtered_molecules=0,
                    minimization_attempts=80 if method != "etkdg" else 0,
                    minimization_failures=3 if method != "etkdg" else 0,
                    wall_seconds=wall,
                    cpu_seconds=wall,
                    generate_seconds=wall / 2,
                    embedding_seconds=wall / 4,
                    minimization_seconds=(wall / 4 if method != "etkdg" else 0.0),
                    scorer_seconds=wall * 0.8,
                    environment_seconds=wall * 0.9,
                    mean_rocs=0.9,
                    median_rocs=0.9,
                )
            )
    return pd.DataFrame(rows)


def epochs_frame() -> pd.DataFrame:
    """One epoch per arm with strictly nested stage clocks."""
    return pd.DataFrame(
        [
            dict(
                arm="etkdg_k1_c30",
                epoch=1,
                epoch_seconds=100.0,
                rocs_seconds=50.0,
                generation_minimization_seconds=25.0,
                environment_seconds=60.0,
                environment_rows=500,
                scorer_calls=2,
            ),
            dict(
                arm="mmff_high_dielectric_k4_c20",
                epoch=1,
                epoch_seconds=80.0,
                rocs_seconds=40.0,
                generation_minimization_seconds=30.0,
                environment_seconds=45.0,
                scorer_calls=2,
                environment_rows=500,
            ),
        ]
    )


class ArmParsingTest(unittest.TestCase):
    """Arm labels and the method:cap:attempts parser."""

    def test_label_formats_cap_and_attempts(self) -> None:
        self.assertEqual(arm_label("etkdg", 1, 30), "etkdg_k1_c30")
        self.assertEqual(arm_label("mmff_vacuum", 0, 75), "mmff_vacuum_kall_c75")
        self.assertEqual(Arm("uff", 4, 20).label, "uff_k4_c20")

    def test_parse_arms_deduplicates_and_validates(self) -> None:
        arms = parse_arms("etkdg:1:30,mmff_vacuum:0:75,etkdg:1:30")
        self.assertEqual(
            arms,
            [Arm("etkdg", 1, 30), Arm("mmff_vacuum", 0, 75)],
        )
        with self.assertRaises(ValueError):
            parse_arms("etkdg:1")
        with self.assertRaises(ValueError):
            parse_arms("not_a_method:1:30")
        with self.assertRaises(ValueError):
            parse_arms("")

    def test_minimize_token_parses_and_suffixes_label(self) -> None:
        arms = parse_arms("mmff_high_dielectric:4:20:0,mmff_high_dielectric:4:20:m1")
        self.assertEqual(
            [arm.minimize for arm in arms],
            [False, True],
        )
        self.assertEqual(
            [arm.label for arm in arms],
            [
                "mmff_high_dielectric_k4_c20_m0",
                "mmff_high_dielectric_k4_c20_m1",
            ],
        )
        # Token-less arms keep the historical (method-default) label so
        # earlier run directories remain resumable.
        plain = parse_arms("mmff_high_dielectric:4:20")[0]
        self.assertIsNone(plain.minimize)
        self.assertTrue(plain.effective_minimize)
        self.assertEqual(plain.label, "mmff_high_dielectric_k4_c20")
        self.assertFalse(parse_arms("etkdg:1:30")[0].effective_minimize)

    def test_etkdg_cannot_minimize(self) -> None:
        with self.assertRaises(ValueError):
            parse_arms("etkdg:1:30:1")

    def test_invalid_minimize_token_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_arms("mmff_high_dielectric:4:20:maybe")


class SprintConfigTest(unittest.TestCase):
    """The generated PipelineConfig must mirror the sprint H1 training run."""

    def test_h1_parity_fields(self) -> None:
        config = _sprint_config(
            Arm("etkdg", 1, 30), samples=500, epochs=5, gpu=0
        )
        self.assertEqual(config.max_conformers, 30)
        self.assertEqual(config.max_isomers, 4)
        self.assertFalse(config.minimize_candidates)
        self.assertEqual(config.epsilon, 0.05)
        self.assertFalse(config.use_epsilon_decay)
        self.assertEqual(config.score_variant, "tanimoto_combo")
        self.assertEqual(config.reward_scheme, "pareto_crowding")
        self.assertFalse(config.include_qed)
        self.assertEqual(config.cpu_ratio, 0.90)
        self.assertEqual(config.n_samples, 500)
        self.assertEqual(config.timeout_seconds, 30)

    def test_minimized_arm_switches_optimization(self) -> None:
        config = _sprint_config(
            Arm("mmff_high_dielectric", 4, 20), samples=100, epochs=5, gpu=0
        )
        self.assertTrue(config.minimize_candidates)
        self.assertEqual(config.optimization.dielectric, 78.4)
        self.assertEqual(config.max_conformers, 20)

    def test_explicit_minimize_overrides_method_default(self) -> None:
        # Regime-1 control: MMFF reference bank, unminimized ETKDG candidates.
        config = _sprint_config(
            Arm("mmff_high_dielectric", 4, 20, False), samples=100, epochs=5, gpu=0
        )
        self.assertFalse(config.minimize_candidates)
        self.assertEqual(config.optimization.dielectric, 78.4)
        self.assertEqual(config.max_conformers, 20)


class ScoreRowTest(unittest.TestCase):
    """Epoch validation-score summarization."""

    def test_score_row_aggregates_valid_molecules(self) -> None:
        frame = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCC", "invalid", "CCN"],
                "Valid": [1, 1, 0, 1],
                "Desired": [0, 1, 0, 0],
                "ROCS": [0.4, 0.8, 0.0, 1.2],
            }
        )
        row = score_row(frame, epoch=3)
        self.assertEqual(row["epoch"], 3)
        self.assertEqual(row["rows"], 4)
        self.assertEqual(row["valid"], 3)
        self.assertEqual(row["unique"], 4)
        self.assertAlmostEqual(row["mean_rocs"], (0.4 + 0.8 + 1.2) / 3)
        self.assertAlmostEqual(row["desired_ratio"], 0.25)

    def test_score_row_handles_missing_columns(self) -> None:
        row = score_row(pd.DataFrame({"SMILES": ["CCO"]}), epoch=1)
        self.assertTrue(np.isnan(row["mean_rocs"]))
        self.assertEqual(row["rows"], 1)


class StageBreakdownTest(unittest.TestCase):
    """Per-epoch cost stages must partition the epoch without overlap."""

    def test_stages_sum_to_epoch(self) -> None:
        breakdown = epoch_stage_breakdown(epochs_frame())
        self.assertEqual(len(breakdown), 2)
        for row in breakdown.itertuples():
            total = (
                row.generation_minimization_seconds
                + row.align_io_seconds
                + row.env_other_seconds
                + row.rnn_residual_seconds
            )
            self.assertAlmostEqual(total, row.epoch_seconds)
        etkdg = breakdown[breakdown.arm.eq("etkdg_k1_c30")].iloc[0]
        self.assertAlmostEqual(etkdg.align_io_seconds, 25.0)
        self.assertAlmostEqual(etkdg.env_other_seconds, 10.0)
        self.assertAlmostEqual(etkdg.rnn_residual_seconds, 40.0)

    def test_negative_residuals_are_clipped(self) -> None:
        frame = pd.DataFrame(
            [
                dict(
                    arm="a",
                    epoch=1,
                    epoch_seconds=10.0,
                    rocs_seconds=30.0,
                    generation_minimization_seconds=5.0,
                    environment_seconds=20.0,
                    environment_rows=1,
                    scorer_calls=1,
                )
            ]
        )
        row = epoch_stage_breakdown(frame).iloc[0]
        self.assertEqual(row.align_io_seconds, 25.0)
        self.assertEqual(row.env_other_seconds, 0.0)
        self.assertEqual(row.rnn_residual_seconds, 0.0)

    def test_stage_breakdown_reports_embed_minimize_split(self) -> None:
        frame = pd.DataFrame(
            [
                dict(
                    arm="a",
                    epoch=1,
                    epoch_seconds=100.0,
                    rocs_seconds=50.0,
                    generation_minimization_seconds=30.0,
                    embedding_seconds=20.0,
                    minimization_seconds=10.0,
                    environment_seconds=60.0,
                    environment_rows=10,
                    scorer_calls=2,
                )
            ]
        )
        row = epoch_stage_breakdown(frame).iloc[0]
        self.assertAlmostEqual(row.embedding_seconds, 20.0)
        self.assertAlmostEqual(row.minimization_seconds, 10.0)
        self.assertAlmostEqual(row.generation_minimization_seconds, 30.0)


class SummarizeTimingTest(unittest.TestCase):
    """Cold/warm aggregation of the batch-timing table."""

    def test_warm_excludes_cold_repeat(self) -> None:
        summary = summarize_timing(timing_frame())
        self.assertEqual(len(summary), 2)
        etkdg = summary[summary.arm.eq("etkdg_k1_c30")].iloc[0]
        self.assertAlmostEqual(etkdg.cold_per_molecule_seconds, 0.12)
        self.assertAlmostEqual(etkdg.warm_per_molecule_seconds, 0.10)
        self.assertEqual(etkdg.references, 1)
        self.assertAlmostEqual(etkdg.gen_per_molecule, 0.05)
        self.assertAlmostEqual(etkdg.align_io_per_molecule, 0.03)
        self.assertAlmostEqual(etkdg.env_other_per_molecule, 0.02)
        self.assertAlmostEqual(etkdg.valid_rate, 0.9)
        self.assertAlmostEqual(etkdg.zero_score_valid_rate, 0.05)

    def test_per_molecule_stages_sum_to_warm_total(self) -> None:
        summary = summarize_timing(timing_frame())
        for row in summary.itertuples():
            total = (
                row.gen_per_molecule
                + row.align_io_per_molecule
                + row.env_other_per_molecule
            )
            self.assertAlmostEqual(total, row.warm_per_molecule_seconds)

    def test_zero_scorer_seconds_yields_nan_not_error(self) -> None:
        frame = timing_frame()
        frame.loc[frame.repeat.ne(0), "scorer_seconds"] = 0.0
        summary = summarize_timing(frame)
        etkdg = summary[summary.arm.eq("etkdg_k1_c30")].iloc[0]
        self.assertTrue(np.isnan(etkdg.align_io_per_molecule))
        self.assertAlmostEqual(etkdg.gen_per_molecule, 0.05)

    def test_single_cold_repeat_falls_back_to_all_rows(self) -> None:
        frame = timing_frame()
        summary = summarize_timing(frame[frame.repeat.eq(0)])
        etkdg = summary[summary.arm.eq("etkdg_k1_c30")].iloc[0]
        self.assertAlmostEqual(etkdg.warm_per_molecule_seconds, 0.12)

    def test_embed_and_minimize_split_per_molecule(self) -> None:
        summary = summarize_timing(timing_frame())
        etkdg = summary[summary.arm.eq("etkdg_k1_c30")].iloc[0]
        mmff = summary[summary.arm.eq("mmff_high_dielectric_k4_c20")].iloc[0]
        # etkdg arm: embedding only (0.10 / 4 per warm molecule, 2 warm rows).
        self.assertAlmostEqual(etkdg.embed_per_molecule, 0.025)
        self.assertAlmostEqual(etkdg.minimize_per_molecule, 0.0)
        # minimized arm: embedding + minimization = gen split.
        self.assertAlmostEqual(mmff.embed_per_molecule, 0.0125)
        self.assertAlmostEqual(mmff.minimize_per_molecule, 0.0125)
        self.assertAlmostEqual(
            mmff.embed_per_molecule + mmff.minimize_per_molecule,
            mmff.gen_per_molecule,
        )


class RankAgreementTest(unittest.TestCase):
    """Pre-registered rank-agreement metric against the gold arm."""

    @staticmethod
    def _write_scores(
        output: Path, arm: str, rocs: list[float], repeat: int = 0
    ) -> None:
        raw = output / "raw_data" / arm
        raw.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Valid": [1] * len(rocs), "ROCS": rocs}).to_csv(
            raw / f"batch_scores_repeat{repeat}.csv", index=False
        )

    def test_perfect_and_reversed_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            # Ascending distinct values: the "reversed" arm (descending) has
            # exactly inverted ranks -> rho = -1 and disjoint top-1 sets.
            gold_rocs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self._write_scores(output, "gold", gold_rocs)
            self._write_scores(output, "twin", list(gold_rocs))
            self._write_scores(output, "reversed", list(reversed(gold_rocs)))
            agreement = rank_agreement(output, ["twin", "reversed"], "gold")
            twin = agreement[agreement.arm.eq("twin")].iloc[0]
            reversed_row = agreement[agreement.arm.eq("reversed")].iloc[0]
            self.assertAlmostEqual(twin.spearman_valid, 1.0)
            self.assertAlmostEqual(twin.top10_overlap_valid, 1.0)
            self.assertAlmostEqual(twin.mean_delta_rocs_valid, 0.0)
            self.assertAlmostEqual(reversed_row.spearman_valid, -1.0)
            self.assertAlmostEqual(reversed_row.top10_overlap_valid, 0.0)
            self.assertLessEqual(
                reversed_row.delta_rocs_ci95_low,
                reversed_row.mean_delta_rocs_valid + 1e-9,
            )
            self.assertGreaterEqual(
                reversed_row.delta_rocs_ci95_high,
                reversed_row.mean_delta_rocs_valid - 1e-9,
            )

    def test_timeout_zeros_excluded_from_valid_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            gold_rocs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.55]
            arm_rocs = list(gold_rocs)
            arm_rocs[0] = 0.0  # alignment timeout in this arm only
            self._write_scores(output, "gold", gold_rocs)
            self._write_scores(output, "arm", arm_rocs)
            row = rank_agreement(output, ["arm"], "gold").iloc[0]
            self.assertAlmostEqual(row.spearman_valid, 1.0)
            self.assertAlmostEqual(row.spearman_all, 0.8787878787878788)
            self.assertAlmostEqual(row.mean_delta_rocs_valid, 0.0)

    def test_unpaired_arm_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self._write_scores(output, "gold", [0.1, 0.5, 0.9])
            self._write_scores(output, "late", [0.2, 0.4, 0.8], repeat=1)
            row = rank_agreement(output, ["late"], "gold").iloc[0]
            self.assertIn("unpaired", row.note)
            self.assertNotIn("spearman_all", row.index)

    def test_missing_gold_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._write_scores(Path(tmp), "a", [0.1, 0.5])
            with self.assertRaises(ValueError):
                rank_agreement(Path(tmp), ["a"], "gold")


class RenderSummaryTest(unittest.TestCase):
    """End-to-end render over synthetic completed arm directories."""

    def test_render_cost_summary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for arm in ("etkdg_k1_c30", "mmff_high_dielectric_k4_c20"):
                raw = output / "raw_data" / arm
                raw.mkdir(parents=True)
                (raw / "complete.json").write_text(json.dumps({"state": "complete"}))
            timing_frame().to_csv(
                output / "raw_data" / "etkdg_k1_c30" / "batch_timing.csv", index=False
            )
            timing_frame().to_csv(
                output
                / "raw_data"
                / "mmff_high_dielectric_k4_c20"
                / "batch_timing.csv",
                index=False,
            )
            epochs_frame().to_csv(
                output / "raw_data" / "etkdg_k1_c30" / "epoch_timings.csv",
                index=False,
            )
            scores = pd.DataFrame(
                [
                    dict(arm="etkdg_k1_c30", epoch=1, rows=50, valid=45,
                         unique=50, mean_rocs=0.5, median_rocs=0.5,
                         p90_rocs=0.7, desired_ratio=0.1),
                    dict(arm="etkdg_k1_c30", epoch=2, rows=50, valid=46,
                         unique=50, mean_rocs=0.6, median_rocs=0.6,
                         p90_rocs=0.8, desired_ratio=0.2),
                ]
            )
            scores.to_csv(
                output / "raw_data" / "etkdg_k1_c30" / "epoch_scores.csv",
                index=False,
            )
            target = render_cost_summary(output)
            self.assertTrue(target.exists())
            self.assertTrue((output / "SUMMARY.md").exists())
            summary = pd.read_csv(output / "cost_summary.csv")
            self.assertEqual(len(summary), 2)
            self.assertIn("warm_per_molecule_seconds", summary.columns)
            text = (output / "SUMMARY.md").read_text()
            self.assertIn("etkdg_k1_c30", text)
            self.assertIn("mmff_high_dielectric_k4_c20", text)

    def test_render_requires_completed_arms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render_cost_summary(Path(tmp))

    def test_render_includes_rank_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for arm, rocs in (
                ("cheap", [0.3, 0.4, 0.8]),
                ("gold", [0.2, 0.5, 0.9]),
            ):
                raw = output / "raw_data" / arm
                raw.mkdir(parents=True)
                (raw / "complete.json").write_text(
                    json.dumps({"state": "complete"})
                )
                pd.DataFrame({"Valid": [1, 1, 1], "ROCS": rocs}).to_csv(
                    raw / "batch_scores_repeat0.csv", index=False
                )
                pd.DataFrame(
                    [
                        dict(
                            arm=arm,
                            method="etkdg",
                            reference_cap=1,
                            candidate_attempts=30,
                            repeat=0,
                            cold=True,
                            inputs=3,
                            references=1,
                            valid=3,
                            zero_score_valid=0,
                            embedding_failures=0,
                            identity_rejections=0,
                            filtered_molecules=0,
                            minimization_attempts=0,
                            minimization_failures=0,
                            wall_seconds=3.0,
                            cpu_seconds=3.0,
                            generate_seconds=1.5,
                            embedding_seconds=1.5,
                            minimization_seconds=0.0,
                            scorer_seconds=2.5,
                            environment_seconds=2.8,
                            mean_rocs=0.5,
                            median_rocs=0.5,
                        )
                    ]
                ).to_csv(raw / "batch_timing.csv", index=False)
            target = render_cost_summary(output, gold="gold")
            self.assertTrue(target.exists())
            self.assertTrue((output / "rank_agreement.csv").exists())
            text = (output / "SUMMARY.md").read_text()
            self.assertIn("Rank agreement vs gold (gold)", text)
            agreement = pd.read_csv(output / "rank_agreement.csv")
            self.assertAlmostEqual(agreement.spearman_valid.iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
