"""Behaviour tests for the held-out reward-weighted NLL selection criterion.

The contract under test is the one recorded in ``tests/plan.md`` §7: a frozen,
seeded draw from a prior dataset, weights equal to the run's composite desirability
on the *scaled* score columns, a masked per-sequence likelihood, and an eligibility
gate that can veto a checkpoint without hiding the loss it saw.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from drugex.training.explorers import SequenceExplorer

from _david.code import loss_criterion as lc
from _david.code.config import PipelineConfig
from _david.code.models import DecayingSequenceExplorer, LossSelectedSequenceExplorer
from _david.code.prior_dataset import CSV_NAME, MANIFEST_NAME, sha256_file
from _david.code.schema import std

ALPHABET = "CNOc()=1234567890"


class _CharVocab:
    """Character-level stand-in for ``VocSmiles`` with the same token contract."""

    def __init__(self, max_len: int = 64) -> None:
        self.max_len = max_len
        self.tk2ix: dict[str, int] = {"EOS": 0, "GO": 1}
        for char in ALPHABET:
            self.tk2ix.setdefault(char, len(self.tk2ix))

    def splitSequence(self, smiles: str) -> list[str]:
        return list(smiles) + ["EOS"]

    def encode(self, sequences: list[list[str]]) -> torch.Tensor:
        output = torch.zeros(len(sequences), self.max_len, dtype=torch.long)
        for row, tokens in enumerate(sequences):
            for column, token in enumerate(tokens):
                if column < self.max_len and token in self.tk2ix:
                    output[row, column] = self.tk2ix[token]
        return output


class _ConstantLogitAgent:
    """Agent whose log-probabilities are identical at every step and batch row."""

    def __init__(self, vocab: _CharVocab, logits: list[float]) -> None:
        self.voc = vocab
        self.device = torch.device("cpu")
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = bool(mode)

    def init_h(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, batch_size, 1)

    def __call__(
        self, token: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits.expand(token.shape[0], -1), hidden


class _StubHoldout:
    """Holdout stand-in that reports a fixed loss and counts evaluations."""

    def __init__(self, value: float = 1.25) -> None:
        self.value = value
        self.calls = 0

    def evaluate(self, agent: object, batch_size: int = 256) -> float:
        self.calls += 1
        return self.value


class _RecordingReward:
    """Reward callable that records the matrix it was asked to score."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], np.ndarray]] = []

    def __call__(self, smiles: list[str], scores: np.ndarray) -> np.ndarray:
        self.calls.append((list(smiles), np.array(scores, dtype=np.float64)))
        return np.asarray(scores, dtype=np.float64).mean(axis=1) + 0.1


def _row(
    smiles: str,
    *,
    valid: int = 1,
    rocs: float = 0.80,
    sa: float = 4.0,
    qed: float = 0.50,
    canonical: str | None = None,
) -> dict[str, object]:
    return {
        std.SMILES: smiles,
        std.CANONICAL_SMILES: canonical if canonical is not None else smiles,
        "Raw_ROCS": rocs,
        "Raw_SA": sa,
        std.QED: qed,
        std.VALID: valid,
    }


def _toy_rows(n: int = 40) -> list[dict[str, object]]:
    """``n`` distinct, encodable rows with a spread of score values."""
    return [
        _row(f"{'C' * (index + 1)}O", rocs=0.40 + 0.005 * index, sa=2.0 + 0.02 * index)
        for index in range(n)
    ]


def _write_dataset(root: Path, rows: list[dict[str, object]]) -> Path:
    target = root / "prior_toy"
    target.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(target / CSV_NAME, index=False, compression="gzip")
    (target / MANIFEST_NAME).write_text(json.dumps({"name": target.name}))
    return target


class TestHoldoutDraw(unittest.TestCase):
    def test_draw_is_reproducible_and_seed_dependent(self) -> None:
        vocab = _CharVocab()
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(
                Path(tmp),
                _toy_rows(40)
                + [
                    _row("CCO"),  # duplicate canonical SMILES of row 1
                    _row("not-a-smiles", valid=0, canonical=""),
                ],
            )
            digest = sha256_file(target / CSV_NAME)
            reward = _RecordingReward()
            base = dict(prior_dataset=str(target), val_holdout_size=8)
            first = lc.build_holdout(
                PipelineConfig(**base, val_holdout_seed=7), vocab, reward
            )
            second = lc.build_holdout(
                PipelineConfig(**base, val_holdout_seed=7), vocab, reward
            )
            other = lc.build_holdout(
                PipelineConfig(**base, val_holdout_seed=8), vocab, reward
            )

        self.assertEqual(len(first), 8)
        self.assertEqual(first.available, 40)
        self.assertEqual(first.excluded, 2)
        self.assertEqual(first.smiles, second.smiles)
        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertTrue(torch.equal(first.weights, second.weights))
        self.assertTrue(torch.equal(first.lengths, second.lengths))
        self.assertNotEqual(set(first.smiles), set(other.smiles))
        self.assertEqual(first.dataset, target.name)
        self.assertEqual(first.dataset_sha256, digest)

    def test_weights_come_from_the_scaled_score_columns(self) -> None:
        vocab = _CharVocab()
        rows = [
            _row("CO", rocs=1.0, sa=4.0, qed=0.2),
            _row("CCO", rocs=1.6, sa=3.0, qed=0.6),
            _row("CCCO", rocs=2.0, sa=5.0, qed=1.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), rows)
            config = PipelineConfig(
                prior_dataset=str(target),
                val_holdout_size=3,
                include_qed=True,
                rocs_use_modifier=True,
                rocs_clip_upper=2.0,
            )
            reward = _RecordingReward()
            sample = lc.build_holdout(config, vocab, reward)

        expected = np.array(
            [
                [0.5, 0.5, 0.2],
                [0.8, 1.0, 0.6],
                [1.0, 0.0, 1.0],
            ]
        )
        self.assertEqual(sample.columns, (std.ROCS, std.SA, std.QED))
        np.testing.assert_allclose(sample.scores, expected)
        self.assertEqual(len(reward.calls), 1)
        _, seen = reward.calls[0]
        np.testing.assert_allclose(seen, expected)
        np.testing.assert_allclose(
            sample.weights.numpy(), expected.mean(axis=1) + 0.1
        )

    def test_draw_larger_than_the_usable_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(5))
            config = PipelineConfig(
                prior_dataset=str(target), val_holdout_size=6, val_holdout_seed=1
            )
            with self.assertRaises(ValueError):
                lc.build_holdout(config, _CharVocab(), _RecordingReward())

    def test_dataset_without_score_columns_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(4))
            frame = pd.read_csv(target / CSV_NAME)
            frame.drop(columns=["Raw_SA"]).to_csv(
                target / CSV_NAME, index=False, compression="gzip"
            )
            config = PipelineConfig(prior_dataset=str(target), val_holdout_size=2)
            with self.assertRaises(ValueError):
                lc.build_holdout(config, _CharVocab(), _RecordingReward())

    def test_unencodable_rows_are_excluded_before_the_draw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(4) + [_row("S" * 80)])
            sample = lc.build_holdout(
                PipelineConfig(
                    prior_dataset=str(target), val_holdout_size=4, val_holdout_seed=2
                ),
                _CharVocab(),
                _RecordingReward(),
            )
        self.assertEqual(sample.available, 4)
        self.assertEqual(sample.excluded, 1)
        self.assertTrue(all("S" not in smiles for smiles in sample.smiles))

    def test_qed_is_weighted_only_when_the_run_tracks_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(5))
            without = lc.build_holdout(
                PipelineConfig(
                    prior_dataset=str(target), val_holdout_size=5, val_holdout_seed=2
                ),
                _CharVocab(),
                _RecordingReward(),
            )
            with_qed = lc.build_holdout(
                PipelineConfig(
                    prior_dataset=str(target),
                    val_holdout_size=5,
                    val_holdout_seed=2,
                    include_qed=True,
                ),
                _CharVocab(),
                _RecordingReward(),
            )
        self.assertEqual(without.columns, (std.ROCS, std.SA))
        self.assertEqual(without.scores.shape[1], 2)
        self.assertEqual(with_qed.columns, (std.ROCS, std.SA, std.QED))
        self.assertEqual(with_qed.scores.shape[1], 3)

    def test_a_reward_with_the_wrong_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(5))
            config = PipelineConfig(
                prior_dataset=str(target), val_holdout_size=2, val_holdout_seed=1
            )
            with self.assertRaisesRegex(ValueError, "weights"):
                lc.build_holdout(
                    config, _CharVocab(), lambda smiles, scores: np.zeros(1)
                )


class TestEvaluate(unittest.TestCase):
    def _sample(self) -> lc.HoldoutSample:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(12))
            config = PipelineConfig(
                prior_dataset=str(target), val_holdout_size=12, val_holdout_seed=5
            )
            return lc.build_holdout(config, _CharVocab(), _RecordingReward())

    def test_loss_is_the_weighted_mean_of_masked_log_likelihoods(self) -> None:
        sample = self._sample()
        vocab = _CharVocab()
        logits = np.linspace(-1.0, 1.0, num=len(vocab.tk2ix))
        agent = _ConstantLogitAgent(vocab, logits.tolist())

        value = sample.evaluate(agent, batch_size=5)

        log_probs = torch.log_softmax(agent.logits, dim=-1)
        total = 0.0
        weight_sum = 0.0
        for row in range(len(sample)):
            length = int(sample.lengths[row])
            score = sum(
                float(log_probs[int(sample.tokens[row, step])]) for step in range(length)
            )
            total += float(sample.weights[row]) * score
            weight_sum += float(sample.weights[row])
        self.assertAlmostEqual(value, -total / weight_sum, places=5)
        # The toy rows have different lengths, so padding exists and would change
        # the value if it were charged.
        self.assertGreater(int(sample.tokens.size(1)), int(sample.lengths.min()))

    def test_evaluate_restores_the_agents_training_flag(self) -> None:
        sample = self._sample()
        vocab = _CharVocab()
        agent = _ConstantLogitAgent(vocab, [0.0] * len(vocab.tk2ix))

        agent.train(True)
        sample.evaluate(agent)
        self.assertTrue(agent.training)

        agent.eval()
        sample.evaluate(agent)
        self.assertFalse(agent.training)


class TestRecord(unittest.TestCase):
    def test_record_persists_the_sample_and_its_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _write_dataset(Path(tmp), _toy_rows(20))
            digest = sha256_file(target / CSV_NAME)
            config = PipelineConfig(
                prior_dataset=str(target), val_holdout_size=6, val_holdout_seed=11
            )
            sample = lc.build_holdout(config, _CharVocab(), _RecordingReward())
            csv_path, json_path = lc.write_holdout_record(Path(tmp) / "models", sample)

            table = pd.read_csv(csv_path)
            meta = json.loads(json_path.read_text())
            expected = sample.provenance()

        self.assertEqual(len(table), len(sample))
        for column in (std.SMILES, "Weight", "Tokens", std.ROCS, std.SA):
            self.assertIn(column, table.columns)
        self.assertEqual(meta["criterion"], std.VAL_LOSS)
        self.assertEqual(meta["schema_version"], lc.HOLDOUT_SCHEMA_VERSION)
        self.assertEqual(meta["rows"], len(sample))
        self.assertEqual(meta["seed"], 11)
        self.assertEqual(meta["dataset_sha256"], digest)
        self.assertEqual(meta["columns"], list(sample.columns))
        self.assertEqual(meta["holdout_sha256"], expected["holdout_sha256"])
        self.assertAlmostEqual(
            meta["weight_mean"], float(sample.weights.mean()), places=6
        )


class TestEligibilityGate(unittest.TestCase):
    def _explorer(self, value: float = 1.25) -> LossSelectedSequenceExplorer:
        explorer = object.__new__(LossSelectedSequenceExplorer)
        explorer.holdout = _StubHoldout(value)
        explorer.min_valid_ratio = 0.85
        explorer.min_unique_ratio = 0.50
        explorer.agent = mock.Mock()
        return explorer

    def _metrics(
        self,
        explorer: LossSelectedSequenceExplorer,
        valid_ratio: float,
        unique_ratio: float,
    ) -> dict[str, float]:
        base = {std.VALID_RATIO: valid_ratio, std.UNIQUE_RATIO: unique_ratio}
        with mock.patch.object(
            SequenceExplorer, "getNovelMoleculeMetrics", return_value=dict(base)
        ):
            return explorer.getNovelMoleculeMetrics(pd.DataFrame())

    def test_eligible_epoch_exposes_the_negated_loss(self) -> None:
        explorer = self._explorer(1.25)
        metrics = self._metrics(explorer, 0.90, 0.60)

        self.assertAlmostEqual(metrics[std.VAL_LOSS], 1.25)
        self.assertAlmostEqual(metrics[std.VAL_LOSS_NEG], -1.25)
        self.assertEqual(metrics[std.VAL_LOSS_ELIGIBLE], 1.0)
        self.assertEqual(explorer.holdout.calls, 1)

    def test_ineligible_epoch_keeps_the_loss_but_cannot_win(self) -> None:
        explorer = self._explorer(1.25)
        metrics = self._metrics(explorer, 0.50, 0.60)

        self.assertAlmostEqual(metrics[std.VAL_LOSS], 1.25)
        self.assertTrue(np.isnan(metrics[std.VAL_LOSS_NEG]))
        self.assertEqual(metrics[std.VAL_LOSS_ELIGIBLE], 0.0)
        # The core loop compares ``metrics[criterion] > best_value``; NaN is False.
        self.assertFalse(metrics[std.VAL_LOSS_NEG] > float("-inf"))
        self.assertEqual(explorer.holdout.calls, 1)

    def test_uniqueness_floor_alone_rejects_an_epoch(self) -> None:
        explorer = self._explorer()
        metrics = self._metrics(explorer, 0.99, 0.10)
        self.assertTrue(np.isnan(metrics[std.VAL_LOSS_NEG]))
        self.assertEqual(metrics[std.VAL_LOSS_ELIGIBLE], 0.0)

    def test_fit_forces_the_loss_criterion(self) -> None:
        explorer = self._explorer()
        with mock.patch.object(DecayingSequenceExplorer, "fit") as parent:
            explorer.fit(epochs=2, criteria=None)
            parent.assert_called_once_with(epochs=2, criteria=std.VAL_LOSS_NEG)
            with self.assertRaises(ValueError):
                explorer.fit(epochs=2, criteria=std.AVG_GMEAN)


class TestLossConfig(unittest.TestCase):
    def test_loss_criterion_requires_a_dataset(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(criteria=std.VAL_LOSS)
        PipelineConfig(criteria=std.VAL_LOSS, prior_dataset="toy")

    def test_holdout_dataset_takes_precedence(self) -> None:
        config = PipelineConfig(prior_dataset="base", val_holdout_dataset="override")
        self.assertEqual(lc.resolve_holdout_dataset(config), "override")

    def test_floors_must_be_probabilities(self) -> None:
        for kwargs in (
            {"val_min_valid_ratio": 1.5},
            {"val_min_unique_ratio": -0.1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    PipelineConfig(**kwargs)

    def test_holdout_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(val_holdout_size=0)


if __name__ == "__main__":
    unittest.main()
