"""Archived Multi-Objective Reinforcement Learning (MORL) Anti-Collapse Soft Gate Reward.

====================================================================================================
HISTORICAL ARCHIVE NOTE:
This module contains AntiCollapseSoftGateReward, developed during early exploratory phases.
It is archived under _david/legacy/ because the synthetic collapse it addressed was primarily an
artifact of misconfigured hyperparameters (e.g. exploration rate epsilon = 0.0, improper patience).
Once hyperparameters are properly calibrated, native DrugEx reward schemes (WeightedSum,
ParetoCrowdingDistance) operate stably without this ad-hoc non-linear gating heuristic.
See _david/legacy/README.md for full mathematical and experimental rationale.
====================================================================================================
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

# [PACKAGE: numpy] WHY: Matrix calculations, array sorting, clipping, and vector operations. WHAT: Numerical computing library. HOW: np.asarray, np.nan_to_num, np.clip, np.where.
import numpy as np
from drugex.training.interfaces import RewardScheme

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
]


class AntiCollapseSoftGateReward(RewardScheme):
    """Continuous soft-gated multi-objective reward scheme eliminating Pareto cheating.

    Theoretical Architecture & Mathematical Derivation:
    --------------------------------------------------
    Standard weighted sum scalarization permits 'Pareto Cheating' in molecular RL: the agent
    discovers small, trivial fragments that satisfy simple physicochemical constraints (LogP, TPSA,
    MW) while completely failing the primary 3D template alignment (S_3D = 0), yet receiving high
    aggregate rewards. Conversely, hard step gating (R = 0 if S_3D < tau) creates a zero-gradient
    valley in early epochs where naive priors sample low S_3D candidates, arresting exploration.

    AntiCollapseSoftGateReward resolves both pathologies through continuous soft gating:
        R(x) = w_3D * S_3D(x) + w_props * G(S_3D(x)) * sum_i(w_tilde_i * S_prop,i(x)) + S_SA_floor(x)

    Where:
        G(S_3D) = min(1.0, max(0.0, S_3D / tau_3D))  (tau_3D = 0.60 saturation threshold)
        w_3D = 0.70 (primary 3D shape/color alignment weight)
        w_props = 0.30 (gated secondary property weight)
        w_tilde_i = Normalized weights for secondary physicochemical properties
    """

    def __init__(
        self,
        rocs_col_idx: int = 0,
        sa_col_idx: Optional[int] = None,
        property_weights: Optional[Sequence[float]] = None,
        w_3d: float = 0.70,
        w_props: float = 0.30,
        tau_3d: float = 0.60,
        normalize_rocs: bool = False,
        sa_floor_val: float = 5.5,
        sa_ceil_val: float = 6.0,
    ) -> None:
        """Initialize AntiCollapseSoftGateReward with objective weights and gating thresholds.

        Args:
            rocs_col_idx: Column index of 3D ROCS alignment score in score matrix (default 0).
            sa_col_idx: Column index of synthesizability score (if present in score matrix).
            property_weights: Optional weights for secondary physicochemical property objectives.
            w_3d: Weight coefficient for primary 3D shape/color alignment objective (default 0.70).
            w_props: Weight coefficient for gated secondary property objectives (default 0.30).
            tau_3d: Soft-gating saturation threshold for 3D alignment score (default 0.60).
            normalize_rocs: Whether to divide raw TanimotoCombo scores in [0, 2] by 2.0 (default False).
            sa_floor_val: Upper bound of unpenalized synthesizability complexity (default 5.5).
            sa_ceil_val: Lower bound of penalized synthesizability inaccessibility (default 6.0).
        """
        super().__init__()
        self.rocs_col_idx: int = int(rocs_col_idx)
        self.sa_col_idx: Optional[int] = int(sa_col_idx) if sa_col_idx is not None else None
        self.w_3d: float = float(w_3d)
        self.w_props: float = float(w_props)
        self.tau_3d: float = float(tau_3d)
        self.normalize_rocs: bool = bool(normalize_rocs)
        self.sa_floor_val: float = float(sa_floor_val)
        self.sa_ceil_val: float = float(sa_ceil_val)
        self.property_weights: Optional[np.ndarray] = (
            np.asarray(property_weights, dtype=np.float64) if property_weights is not None else None
        )

    def gate(self, s_3d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute continuous soft-gate multiplier G(S_3D) = min(1.0, max(0.0, S_3D / tau_3D))."""
        s_arr = np.asarray(s_3d, dtype=np.float64)
        g_val = np.minimum(1.0, np.maximum(0.0, s_arr / self.tau_3d))
        return float(g_val) if np.isscalar(s_3d) else g_val

    def sa_floor(self, sa_raw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute single synthesizability feasibility floor S_SA_floor(x)."""
        sa_arr = np.asarray(sa_raw, dtype=np.float64)
        if np.all(sa_arr <= 1.0) and np.all(sa_arr >= 0.0):
            return float(sa_arr) if np.isscalar(sa_raw) else sa_arr
        denom = max(1e-9, self.sa_ceil_val - self.sa_floor_val)
        floor_score = np.where(
            sa_arr <= self.sa_floor_val,
            1.0,
            np.where(
                sa_arr >= self.sa_ceil_val,
                0.0,
                (self.sa_ceil_val - sa_arr) / denom,
            ),
        )
        return float(floor_score) if np.isscalar(sa_raw) else floor_score

    def __call__(
        self,
        smiles: Union[Sequence[str], np.ndarray],
        scores: Optional[np.ndarray] = None,
        thresholds: Optional[Union[Sequence[float], np.ndarray]] = None,
        sa_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate scalarized anti-collapse soft-gated rewards for candidate molecules."""
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            # Standalone call: scheme(scores, sa_scores=...)
            scores_arr = np.asarray(smiles, dtype=np.float64)
            if scores is not None and sa_scores is None:
                if isinstance(scores, (np.ndarray, list, tuple)):
                    sa_scores = np.asarray(scores, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = scores_arr.reshape(1, -1) if len(scores_arr) > 0 else scores_arr.reshape(0, 1)

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        scores_clean = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)

        # 1. Primary 3D ROCS alignment score extraction & clamping
        col_3d = min(self.rocs_col_idx, scores_clean.shape[1] - 1)
        s_3d = scores_clean[:, col_3d].copy()
        if self.normalize_rocs:
            s_3d = np.clip(s_3d / 2.0, 0.0, 1.0)
        else:
            s_3d = np.clip(s_3d, 0.0, 1.0)

        # 2. Continuous soft-gate multiplier computation G(S_3D)
        g_3d = self.gate(s_3d)

        # 3. Secondary property objectives weighted aggregation
        prop_indices = [
            i for i in range(scores_clean.shape[1])
            if i != col_3d and (self.sa_col_idx is None or i != self.sa_col_idx)
        ]
        if prop_indices:
            props = scores_clean[:, prop_indices]
            n_props = props.shape[1]
            if self.property_weights is not None and len(self.property_weights) == n_props:
                w_norm = self.property_weights / np.sum(self.property_weights)
            else:
                w_norm = np.ones(n_props, dtype=np.float64) / float(n_props)
            props_weighted_sum = np.sum(props * w_norm, axis=1)
            effective_w_3d = self.w_3d
            effective_w_props = self.w_props
        else:
            props_weighted_sum = np.zeros(n_samples, dtype=np.float64)
            effective_w_3d = self.w_3d + self.w_props
            effective_w_props = 0.0

        # Strict Pareto anti-cheating guard: candidates with S_3D <= 1e-6 receive strictly 0.0
        zero_mask = s_3d <= 1e-6
        base_reward = effective_w_3d * s_3d + effective_w_props * g_3d * props_weighted_sum
        base_reward[zero_mask] = 0.0

        # 4. Synthesizability Feasibility Floor integration
        if sa_scores is not None:
            sa_vals = np.asarray(sa_scores, dtype=np.float64).flatten()
            if np.any(sa_vals > 1.0):
                sa_term = self.sa_floor(sa_vals)
            else:
                sa_term = sa_vals
            sa_term_copy = np.asarray(sa_term, dtype=np.float64).copy()
            sa_term_copy[zero_mask] = 0.0
            total_reward = base_reward + sa_term_copy
        elif self.sa_col_idx is not None and self.sa_col_idx < scores_clean.shape[1]:
            sa_vals = scores_clean[:, self.sa_col_idx]
            if np.any(sa_vals > 1.0):
                sa_term = self.sa_floor(sa_vals)
            else:
                sa_term = sa_vals
            sa_term_copy = np.asarray(sa_term, dtype=np.float64).copy()
            sa_term_copy[zero_mask] = 0.0
            total_reward = base_reward + sa_term_copy
        else:
            total_reward = base_reward

        total_reward[zero_mask] = 0.0
        return total_reward.reshape(-1, 1)
