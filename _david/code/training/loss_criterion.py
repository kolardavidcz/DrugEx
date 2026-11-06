"""Held-out reward-weighted NLL: the checkpoint criterion of the loss-selected run.

Design record: ``_david/code/tests/plan.md`` §7.  The training loss in
``SequenceExplorer.policy_gradient`` is not a usable selection criterion — with
``beta = 0`` it is minimized by policy sharpening rather than by reward, and with
``beta > 0`` it can go negative while the generator is fully collapsed.  This
module therefore scores a **fixed, never-trained-on** sample of prior molecules:

    J(theta) = - sum_i w_i * log pi_theta(x_i) / sum_i w_i        (lower is better)

with ``w_i`` the composite Harrington desirability of molecule ``x_i`` under the
run's own reward scheme.  ``J`` falls when the policy puts mass on high-desirability
prior molecules and rises when it loses mass elsewhere, including when it
collapses, because ``-log pi(x_i)`` explodes for every ``x_i`` the collapsed policy
no longer generates.

The sample is drawn deterministically from a frozen prior dataset, so the criterion
is reproducible from the dataset plus the seed and never depends on the policy's
own samples.  The draw is recorded in the run's ``models/loss_criterion.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

from _david.code.config import PipelineConfig
from _david.code.prior_dataset import (
    CSV_NAME,
    load_arm,
    resolve_dataset,
    sha256_file,
)
from _david.code.schema import std

#: Bump when the meaning of the sample, the weights or the loss changes.
HOLDOUT_SCHEMA_VERSION: int = 1

RewardCallable = Callable[[Sequence[str], np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HoldoutSample:
    """A frozen, weighted set of tokenized prior molecules.

    Attributes:
        smiles: Canonical SMILES of the sampled molecules, in draw order.
        tokens: ``LongTensor`` of shape ``(n, max_len)`` with right-zero padding.
        lengths: Token count per row; only the first ``length`` entries are scored.
        weights: Composite desirability per row, in ``[0, 1]``.
        scores: The scaled score matrix the weights were computed from, one row
            per molecule and one column per objective (ROCS, SA[, QED]).
        dataset: Dataset directory name.
        dataset_path: Resolved dataset directory.
        dataset_sha256: SHA-256 of the dataset CSV the sample was drawn from.
        seed: RNG seed of the draw.
        available: Valid, unique, encodable rows the draw selected from.
        excluded: Dataset rows rejected before the draw (invalid, duplicate or
            unencodable), so the sampling pool is auditable.
        columns: Objective columns in ``scores``.
    """

    smiles: tuple[str, ...]
    tokens: torch.Tensor
    lengths: torch.Tensor
    weights: torch.Tensor
    scores: np.ndarray
    dataset: str
    dataset_path: str
    dataset_sha256: str
    seed: int
    available: int
    excluded: int
    columns: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.smiles)

    def provenance(self) -> dict[str, Any]:
        """Machine-readable record of how this sample was produced."""
        payload = "\n".join(
            f"{smiles}\t{weight:.9f}"
            for smiles, weight in zip(self.smiles, self.weights.tolist())
        )
        weight_values = self.weights.detach().cpu().numpy()
        return {
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "criterion": std.VAL_LOSS,
            "formula": "-sum_i w_i log pi(x_i) / sum_i w_i (lower is better)",
            "dataset": self.dataset,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "columns": list(self.columns),
            "available_rows": self.available,
            "excluded_rows": self.excluded,
            "rows": len(self),
            "seed": self.seed,
            "sampling": (
                "numpy.random.default_rng(seed).choice over valid, unique, "
                "encodable rows without replacement; indices sorted ascending"
            ),
            "holdout_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "max_tokens": int(self.lengths.max().item()),
            "weight_min": float(weight_values.min()),
            "weight_median": float(np.median(weight_values)),
            "weight_max": float(weight_values.max()),
            "weight_mean": float(weight_values.mean()),
        }

    def to_frame(self) -> pd.DataFrame:
        """The scored sample as a table, for the run record and diagnostics."""
        table = {
            std.SMILES: list(self.smiles),
            "Weight": self.weights.detach().cpu().numpy(),
            "Tokens": self.lengths.detach().cpu().numpy(),
        }
        for index, column in enumerate(self.columns):
            table[column] = self.scores[:, index]
        return pd.DataFrame(table)

    def write_csv(self, path: Path) -> None:
        """Persist the sample beside the run, so the criterion is auditable."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False, compression="gzip")

    def evaluate(self, agent: Any, batch_size: int = 256) -> float:
        """Return ``J`` for ``agent`` on this sample.

        The agent is temporarily switched to evaluation mode; gradients are never
        built, so the criterion cannot perturb the policy.
        """
        was_training = bool(getattr(agent, "training", False))
        agent.eval()
        try:
            total = 0.0
            weight_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(self), batch_size):
                    stop = min(start + batch_size, len(self))
                    tokens = self.tokens[start:stop].to(_agent_device(agent))
                    weights = self.weights[start:stop]
                    log_likelihood = _sequence_log_likelihood(
                        agent, tokens, self.lengths[start:stop]
                    )
                    total += float(
                        (log_likelihood.detach().cpu() * weights).sum().item()
                    )
                    weight_sum += float(weights.sum().item())
            if weight_sum <= 0.0:
                raise ValueError("holdout weights sum to zero; re-draw the sample")
            return -total / weight_sum
        finally:
            agent.train(was_training)


def _agent_device(agent: Any) -> torch.device:
    device = getattr(agent, "device", "cpu")
    return device if isinstance(device, torch.device) else torch.device(str(device))


def _sequence_log_likelihood(
    agent: Any, tokens: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Per-sequence ``log pi(x)`` over the real tokens only.

    Mirrors ``SequenceRNN.likelihood`` (the thesis keeps DrugEx core read-only) but
    masks the zero padding, so a 30-token molecule is not charged for 70 pad
    tokens.  The forward pass starts from ``GO`` exactly as generation does.
    """
    batch_size, max_len = tokens.size()
    device = tokens.device
    token = torch.full(
        (batch_size,),
        int(agent.voc.tk2ix["GO"]),
        dtype=torch.long,
        device=device,
    )
    hidden = agent.init_h(batch_size)
    scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    steps = min(max_len, int(lengths.max().item()))
    for step in range(steps):
        logits, hidden = agent(token, hidden)
        log_probs = logits.log_softmax(dim=-1)
        next_token = tokens[:, step]
        step_scores = log_probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
        scores = scores + torch.where(lengths > step, step_scores, zero)
        token = next_token
    return scores


def resolve_holdout_dataset(config: PipelineConfig) -> str:
    """Dataset name or path the loss sample is drawn from."""
    source = config.val_holdout_dataset or config.prior_dataset
    if not source:
        raise ValueError(
            "The val_loss criterion needs a frozen prior dataset: pass "
            "--val-holdout-dataset or --prior-dataset (the standard prior arm is "
            "the intended source)."
        )
    return str(source)


def _tokenize(vocab: Any, smiles: str) -> list[str] | None:
    """Tokenize ``smiles`` including EOS, or return ``None`` if unencodable."""
    tokens = vocab.splitSequence(smiles)
    if len(tokens) > int(vocab.max_len):
        return None
    if set(tokens) - set(vocab.tk2ix):
        return None
    return tokens


def build_holdout(
    config: PipelineConfig,
    vocab: Any,
    reward: RewardCallable,
) -> HoldoutSample:
    """Draw and weight the frozen held-out sample for ``config``.

    Args:
        config: The run's configuration; supplies the dataset, size, seed and the
            ROCS/SA modifier windows used to derive the scaled score columns.
        vocab: The agent vocabulary; decides which molecules are encodable.
        reward: The environment's reward scheme (composite desirability), called
            as ``reward(smiles_list, score_matrix)``.

    Raises:
        ValueError: When no dataset is configured, it lacks the score columns, or
            the requested draw is larger than the usable rows.
    """
    source = resolve_holdout_dataset(config)
    target = resolve_dataset(source)
    frame, _ = load_arm(source, config)
    required = {std.SMILES, std.VALID, std.ROCS, std.SA}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prior dataset {source!r} lacks columns: {missing}")

    valid = frame[
        (pd.to_numeric(frame[std.VALID], errors="coerce") == 1)
        & frame[std.SMILES].astype(str).str.len().gt(0)
    ].drop_duplicates(subset=std.CANONICAL_SMILES)
    # Tokenize once: an unencodable molecule cannot enter the pool, so the draw
    # is exactly ``val_holdout_size`` rows and no post-draw drop can bias it.
    valid = valid.assign(
        _tokens=valid[std.SMILES]
        .astype(str)
        .map(lambda smiles: _tokenize(vocab, smiles))
    )
    valid = valid[valid["_tokens"].notna()].reset_index(drop=True)
    if valid.empty:
        raise ValueError(
            f"prior dataset {source!r} has no valid molecules encodable with this "
            f"vocabulary (max_len={vocab.max_len})"
        )
    excluded = int(len(frame) - len(valid))

    rng = np.random.default_rng(config.val_holdout_seed)
    if config.val_holdout_size > len(valid):
        raise ValueError(
            f"holdout size {config.val_holdout_size} exceeds the {len(valid)} "
            f"usable rows of {source!r}"
        )
    picked = np.sort(
        rng.choice(len(valid), size=config.val_holdout_size, replace=False)
    )
    sample = valid.iloc[picked].reset_index(drop=True)
    token_rows = [list(tokens) for tokens in sample["_tokens"]]
    smiles = tuple(sample[std.SMILES].astype(str))

    score_columns = [std.ROCS, std.SA]
    if config.include_qed:
        score_columns.append(std.QED)
    scores = (
        sample.loc[:, score_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    weights = np.asarray(reward(list(smiles), scores), dtype=np.float64).reshape(-1)
    if len(weights) != len(smiles):
        raise ValueError(
            f"reward scheme returned {len(weights)} weights for {len(smiles)} molecules"
        )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError("holdout weights are not finite and positive")

    lengths = np.asarray([len(tokens) for tokens in token_rows], dtype=np.int64)
    token_matrix = np.zeros((len(token_rows), int(lengths.max())), dtype=np.int64)
    for row, tokens in enumerate(token_rows):
        encoded = vocab.encode([tokens]).numpy().reshape(-1)
        token_matrix[row, : len(tokens)] = encoded[: len(tokens)]

    return HoldoutSample(
        smiles=smiles,
        tokens=torch.from_numpy(token_matrix),
        lengths=torch.from_numpy(lengths),
        weights=torch.from_numpy(weights.astype(np.float32)),
        scores=scores,
        dataset=target.name,
        dataset_path=str(target),
        dataset_sha256=sha256_file(target / CSV_NAME),
        seed=int(config.val_holdout_seed),
        available=int(len(valid)),
        excluded=excluded,
        columns=tuple(score_columns),
    )


def write_holdout_record(directory: Path | str, sample: HoldoutSample) -> tuple[Path, Path]:
    """Write the scored sample and its provenance under ``directory``."""
    target = Path(directory)
    csv_path = target / "loss_holdout.csv.gz"
    json_path = target / "loss_criterion.json"
    sample.write_csv(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(sample.provenance(), indent=2) + "\n")
    return csv_path, json_path
