from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, RewardScheme, WeightedSum
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from rdkit import Chem

from _david.code.config import PipelineConfig
from _david.code.reference import PrepareMolecule
from _david.code.rocs import create_rocs_scorer
from _david.code.schema import std


class HarringtonDesirabilityReward(RewardScheme):
    """Continuous Multi-Attribute Desirability (Harrington 1965) for Molecular RL.

    Maps each objective score s_j through a two-parameter Gompertz desirability function:
        d_j(s_j) = exp(-exp(-(b_{0,j} + b_{1,j} * s_j)))
    The composite desirability D(x) is the weighted geometric mean across all M objectives:
        D(x) = exp(sum_j w_j * ln(d_j(s_j) + eps) / sum_j w_j)
    """

    def __init__(
        self,
        weights: Sequence[float] = (0.75, 0.25),
        rocs_anchors: tuple[float, float, float, float] | None = None,
        sa_anchors: tuple[float, float, float, float] = (0.20, 0.05, 0.80, 0.90),
        qed_anchors: tuple[float, float, float, float] = (0.30, 0.05, 0.70, 0.90),
        eps: float = 1e-6,
        rocs_raw: bool = True,
    ) -> None:
        super().__init__()
        self.weights = np.array(weights, dtype=np.float64)
        self.eps = float(eps)
        self.rocs_raw = bool(rocs_raw)
        if rocs_anchors is None:
            rocs_anchors = (
                (0.40, 0.05, 1.20, 0.80)
                if self.rocs_raw
                else (0.20, 0.05, 0.60, 0.80)
            )
        self.params: list[tuple[float, float]] = [
            self._fit_anchors(*rocs_anchors),
            self._fit_anchors(*sa_anchors),
            self._fit_anchors(*qed_anchors),
        ]

    @staticmethod
    def _fit_anchors(y0: float, d0: float, y1: float, d1: float) -> tuple[float, float]:
        """Fit Gompertz parameters (b0, b1) from two anchor points (y0, d0) and (y1, d1)."""
        y0_prime = -np.log(-np.log(d0))
        y1_prime = -np.log(-np.log(d1))
        b1 = (y1_prime - y0_prime) / (y1 - y0)
        b0 = y0_prime - b1 * y0
        return float(b0), float(b1)

    @staticmethod
    def _eval_desirability(s: np.ndarray, b0: float, b1: float) -> np.ndarray:
        """Evaluate Gompertz desirability function on score array s."""
        z = np.clip(b0 + b1 * s, -30.0, 30.0)
        return np.exp(-np.exp(-z))

    @staticmethod
    def _coerce_score_matrix(
        smiles: Sequence[str] | np.ndarray,
        scores: np.ndarray | None,
    ) -> np.ndarray:
        """Normalize caller arguments (either (smiles, scores) or raw score matrix) to 2D float64."""
        is_direct_matrix = isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        )
        raw = (
            np.asarray(smiles, dtype=np.float64)
            if is_direct_matrix
            else (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )
        )
        if raw.ndim == 1:
            return raw.reshape(1, -1) if len(raw) > 0 else raw.reshape(0, 1)
        return raw

    def __call__(
        self,
        smiles: Sequence[str] | np.ndarray,
        scores: np.ndarray | None = None,
        thresholds: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate continuous composite Harrington desirability rewards."""
        scores_arr = self._coerce_score_matrix(smiles, scores)
        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        clean = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)
        scores_clean = np.zeros_like(clean, dtype=np.float64)
        n_objs = clean.shape[1]
        for j in range(n_objs):
            max_val = 2.0 if (j == 0 and self.rocs_raw) else 1.0
            scores_clean[:, j] = np.clip(clean[:, j], 0.0, max_val)

        if len(self.weights) >= n_objs:
            w = np.array(self.weights[:n_objs], dtype=np.float64)
        else:
            padding = np.full(n_objs - len(self.weights), 0.25, dtype=np.float64)
            w = np.concatenate([self.weights, padding])
        w_norm = w / np.sum(w)

        log_d_sum = np.zeros(n_samples, dtype=np.float64)
        for j in range(n_objs):
            b0, b1 = self.params[j] if j < len(self.params) else (0.0, 5.0)
            d_j = self._eval_desirability(scores_clean[:, j], b0, b1)
            log_d_sum += w_norm[j] * np.log(d_j + self.eps)

        rewards = np.clip(np.exp(log_d_sum), 0.0, 1.0).reshape(-1, 1)
        if isinstance(smiles, (list, tuple)):
            for idx, s in enumerate(smiles):
                if isinstance(s, str) and (
                    re.search(r"([A-Za-z])\1{7,}", s)
                    or (len(s) > 20 and s.count("N") / len(s) > 0.50)
                ):
                    rewards[idx] = 0.0
        return rewards


def resolve_reward_scheme(
    scheme: str | RewardScheme,
    weights: tuple[float, ...] = (0.75, 0.25),
    rocs_raw: bool = True,
) -> RewardScheme:
    """Resolve RewardScheme instance strictly from canonical std.REWARD_* constant or pass through."""
    if isinstance(scheme, RewardScheme):
        return scheme
    if scheme == std.REWARD_WEIGHTED_SUM:
        return WeightedSum()
    if scheme == std.REWARD_PARETO:
        return ParetoCrowdingDistance()
    if scheme == std.REWARD_HARRINGTON:
        return HarringtonDesirabilityReward(weights=weights, rocs_raw=rocs_raw)
    raise ValueError(
        f"Unknown reward scheme: '{scheme}'. Canonical schema options: "
        f"[{std.REWARD_WEIGHTED_SUM!r}, {std.REWARD_PARETO!r}, {std.REWARD_HARRINGTON!r}]"
    )


def create_environment(
    ref_sdf: str | Path,
    config: PipelineConfig | None = None,
) -> DrugExEnvironment:
    """Construct multi-objective DrugEx reinforcement learning environment from PipelineConfig."""
    cfg = config or PipelineConfig()
    rocs_scorer = create_rocs_scorer(ref_sdf, cfg=cfg)

    if cfg.rocs_use_modifier:
        rocs_scorer.setModifier(
            ClippedScore(
                lower_x=0.0, upper_x=cfg.rocs_clip_upper, low_score=0.0, high_score=1.0
                # lower_x=0.3, upper_x=cfg.rocs_clip_upper, low_score=0.0, high_score=1.0
            )
        )
    else:
        rocs_scorer.setModifier(None)

    sa_scorer = Property(std.SA)
    sa_scorer.setModifier(
        ClippedScore(
            lower_x=cfg.sa_score_lower,
            upper_x=cfg.sa_score_upper,
            low_score=0.0,
            high_score=1.0,
        )
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer]
    if cfg.include_qed:
        qed_scorer = Property(std.QED)
        qed_scorer.setModifier(
            ClippedScore(lower_x=0.0, upper_x=1.0, low_score=0.0, high_score=1.0)
        )
        scorers.append(qed_scorer)

    scheme = resolve_reward_scheme(
        cfg.reward_scheme,
        weights=cfg.weights or ((0.50, 0.25, 0.25) if cfg.include_qed else (0.75, 0.25)),
        rocs_raw=(not cfg.rocs_use_modifier),
    )

    active_thresholds = list(
        cfg.thresholds
        if cfg.thresholds is not None
        else ((0.35, 0.50, 0.50) if cfg.include_qed else (0.35, 0.50))
    )
    if len(active_thresholds) != len(scorers):
        if len(active_thresholds) > len(scorers):
            active_thresholds = active_thresholds[: len(scorers)]
        else:
            active_thresholds.extend([0.50] * (len(scorers) - len(active_thresholds)))

    return DrugExEnvironment(
        scorers=scorers,
        thresholds=active_thresholds,
        reward_scheme=scheme,
    )
