"""Augmented Chebyshev and Geometric Mean anti-cheating reward schemes.

In multi-objective optimization (MORL) with 5+ objectives, standard Pareto-dominance
ranking suffers from extreme boundary exploitation: a candidate with high QED and synthesizability
but near-zero 3D shape similarity can sit on the first Pareto frontier.
These scalarization reward schemes strictly penalize candidates that collapse in any objective dimension.
"""
from __future__ import annotations

from typing import Sequence
#? [NumPy] Vectorized array operations, clip bounds, and multi-objective norm reductions.
import numpy as np

from drugex.training.interfaces import RewardScheme


class AugmentedChebyshevReward(RewardScheme):
    """Augmented Chebyshev (Tchebycheff) scalarization reward scheme.

    Skeptical Academic Rationale:
    In multi-objective optimization (MORL) with 5+ objectives, Pareto-dominance
    ranking suffers from extreme boundary exploitation: a candidate with high QED (0.95)
    and high synthesizability (0.95) but near-zero 3D shape similarity (0.05) sits on the
    first Pareto frontier because no other candidate dominates its extreme scores.

    Augmented Chebyshev scalarization measures distance to the ideal target point
    z* = (1.0, 1.0, ..., 1.0) along the worst-performing objective dimension:
        f_cheb(s) = max_j (w_j * (1 - s_j)) + rho * sum_j (w_j * (1 - s_j))
    The resulting scalar reward is:
        R(s) = 1.0 - f_cheb(s) / normalization
    Any candidate with a disastrous score in even one objective (e.g. 3D shape = 0.05
    or heavy atom size = 0.05) has (1 - s_j) = 0.95, which directly collapses R(s)
    to near zero. This physically forces the generator to balance all objectives simultaneously.

    Attributes:
        weights: Normalized per-objective weighting array.
        rho: Augmentation coefficient balancing Chebyshev minimax with linear sum.
    """

    def __init__(self, weights: Sequence[float] | None = None, rho: float = 0.02) -> None:
        """Initialize the AugmentedChebyshevReward scheme.

        Args:
            weights: Optional per-objective weighting factors. Defaults to equal weights.
            rho: Augmentation parameter balancing Chebyshev minimax with linear sum (default: 0.02).
        """
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
        """Calculate scalarized anti-cheating rewards for generated molecules.

        Args:
            smiles: List of valid SMILES strings to score.
            scores: Matrix of objective scores with shape (N, M), values in [0.0, 1.0].
            thresholds: Desirability thresholds for objectives (maintained for DrugEx API parity).

        Returns:
            Calculated scalar rewards with shape (N, 1), normalized into [0.0, 1.0].
        """
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
        if max_possible > 0:
            rewards = 1.0 - (raw_cheb / max_possible)
        else:
            rewards = 1.0 - raw_cheb
        rewards = np.clip(rewards, 0.0, 1.0)

        return rewards.reshape(-1, 1)


class GeometricMeanReward(RewardScheme):
    """Weighted Geometric Mean scalarization reward scheme.

    Skeptical Academic Rationale:
    In geometric mean scalarization, the total reward is the geometric product
    of all normalized objective scores:
        R(s) = exp( sum_j (w_j * ln(max(s_j, eps))) / sum_j w_j )
    If any single objective score approaches zero (e.g., candidate fails 3D shape
    or heavy atom volume), the geometric mean precipitously drops towards zero.
    Consequently, fragments that maximize QED while neglecting 3D shape receive
    negligible policy gradient updates.

    Attributes:
        weights: Normalized per-objective weighting array.
        eps: Small floor value preventing log(0) numerical singularities.
    """

    def __init__(self, weights: Sequence[float] | None = None, eps: float = 1e-4) -> None:
        """Initialize the GeometricMeanReward scheme.

        Args:
            weights: Weights per objective. Defaults to equal weighting.
            eps: Epsilon floor to avoid undefined logarithms (default: 1e-4).
        """
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
        """Calculate geometric mean reward across all objectives.

        Args:
            smiles: List of valid SMILES strings.
            scores: Matrix of objective scores with shape (N, M).
            thresholds: Desirability thresholds (unused, maintained for API parity).

        Returns:
            Scalar geometric rewards with shape (N, 1).
        """
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
