"""Augmented Chebyshev, Geometric Mean, and Anti-Collapse Soft-Gated reward schemes."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

# [PACKAGE: numpy] WHY: Array conversions, nan sanitization, and matrix scalarization. WHAT: Numerical computing library.
import numpy as np
from drugex.training.interfaces import RewardScheme
from drugex.training.rewards import AntiCollapseSoftGateReward

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
]


class AugmentedChebyshevReward(RewardScheme):
    """Augmented Chebyshev (Tchebycheff) scalarization reward scheme."""

    def __init__(self, weights: Sequence[float] | None = None, rho: float = 0.02) -> None:
        """Initialize AugmentedChebyshevReward with optional per-objective weights and rho."""
        super().__init__()
        self.weights: np.ndarray | None = (
            np.array(weights, dtype=np.float64) if weights is not None else None
        )
        self.rho: float = float(rho)

    def __call__(
        self,
        smiles: Sequence[str],
        scores: np.ndarray,
        thresholds: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate scalarized anti-cheating rewards for generated molecules."""
        n_samples = len(smiles)
        if n_samples == 0 or scores.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float64)

        #? [NumPy] Clean NaN/Inf, clip scores into [0.0, 1.0], and ensure 2D array shape.
        scores_raw = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        scores_arr = np.clip(scores_raw, 0.0, 1.0)
        if scores_arr.ndim == 1:
            scores_arr = scores_arr.reshape(-1, 1)
        n_objs = scores_arr.shape[1]

        if self.weights is None or len(self.weights) != n_objs or np.sum(self.weights) == 0:
            # Default to equal weighting across all objectives
            w = np.ones(n_objs, dtype=np.float64) / float(n_objs)
        else:
            w = self.weights / np.sum(self.weights)

        # Deviations from ideal point z* = 1.0: d_ij = w_j * (1.0 - s_ij)
        deviations = w * (1.0 - scores_arr)  # Shape (N, M)

        # Max deviation per candidate: penalizes the weakest dimension
        max_dev = np.max(deviations, axis=1)  # Shape (N,)

        # Sum deviation: augmentation term ensuring strict Pareto efficiency
        sum_dev = np.sum(deviations, axis=1)  # Shape (N,)

        raw_cheb = max_dev + self.rho * sum_dev  # Shape (N,)
        max_possible = np.max(w) + self.rho * np.sum(w)

        # Invert distance so higher scores yield higher rewards in [0.0, 1.0]
        rewards = 1.0 - (raw_cheb / max_possible) if max_possible > 0 else 1.0 - raw_cheb
        rewards = np.clip(rewards, 0.0, 1.0)

        return rewards.reshape(-1, 1)


class GeometricMeanReward(RewardScheme):
    """Weighted Geometric Mean scalarization reward scheme."""

    def __init__(self, weights: Sequence[float] | None = None, eps: float = 1e-4) -> None:
        """Initialize GeometricMeanReward with optional weights and epsilon floor."""
        super().__init__()
        self.weights: np.ndarray | None = (
            np.array(weights, dtype=np.float64) if weights is not None else None
        )
        self.eps: float = float(eps)

    def __call__(
        self,
        smiles: Sequence[str],
        scores: np.ndarray,
        thresholds: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate geometric mean reward across all objectives."""
        n_samples = len(smiles)
        if n_samples == 0 or scores.shape[0] == 0:
            return np.zeros((0, 1), dtype=np.float64)

        #? [NumPy] Clean NaN/Inf, floor scores at eps to protect log evaluation from underflow, and ensure 2D shape.
        scores_raw = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=self.eps, posinf=1.0, neginf=self.eps)
        scores_arr = np.clip(scores_raw, self.eps, 1.0)
        if scores_arr.ndim == 1:
            scores_arr = scores_arr.reshape(-1, 1)
        n_objs = scores_arr.shape[1]

        if self.weights is None or len(self.weights) != n_objs or np.sum(self.weights) == 0:
            w = np.ones(n_objs, dtype=np.float64) / float(n_objs)
        else:
            w = self.weights / np.sum(self.weights)

        #? [NumPy] Vectorized log-sum-exp computation for geometric mean.
        log_scores = np.log(scores_arr)
        weighted_log_mean = np.sum(log_scores * w, axis=1) / np.sum(w)
        rewards = np.exp(weighted_log_mean)

        return rewards.reshape(-1, 1)
