from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance, RewardScheme, WeightedSum
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore, SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from rdkit import Chem

from _david.code.config import PipelineConfig
from _david.code.chemistry import (
    CandidateConformerGenerator,
    Optimization,
)
from _david.code.reference import PrepareMolecule
from _david.code.schema import std
from _david.code.utils import HardwareManager


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    max_conformers: int = 30,
    timeout: int = 60,
    score_variant: str = "tanimoto_combo",
    conformer_threads: int = 0,
    n_jobs: int | None = None,
    max_isomers: int = 4,
    random_seed: int = 42,
    minimize_candidates: bool = False,
    optimization: Optimization | None = None,
) -> RDKitROCSScorer:
    """Build multi-threaded RDKitROCSScorer configured to produce canonical std.ROCS key."""
    ref_input = str(references) if isinstance(references, Path) else references
    conf_gen = CandidateConformerGenerator(
        max_conformers=max_conformers,
        max_isomers=max_isomers,
        max_heavy_atoms=45,
        max_rotatable_bonds=15,
        num_threads=conformer_threads,
        show_progress=False,
        timeout=timeout,
        random_seed=random_seed,
        minimize_candidates=minimize_candidates,
        optimization=optimization,
    )
    safe_jobs = (
        n_jobs
        if n_jobs is not None
        else HardwareManager.get_safe_parallel_jobs(ratio=0.90)
    )
    variant = (
        "tanimoto" if score_variant in ("tanimoto", "tanimoto_combo") else score_variant
    )
    scorer = RDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type="TanimotoCombo",
        score_variant=variant,
        use_colors=True,
        show_progress=False,
        n_jobs=safe_jobs,
    )
    scorer.getKey = lambda: [std.ROCS]
    return scorer


class HarringtonDesirabilityReward(RewardScheme):
    """Continuous Multi-Attribute Desirability (Harrington 1965) for Molecular RL.

    Maps each objective score s_j through a two-parameter Gompertz desirability function:
        d_j(s_j) = exp(-exp(-(b_{0,j} + b_{1,j} * s_j)))
    The composite desirability D(x) is the weighted geometric mean across all M objectives:
        D(x) = exp(sum_j w_j * ln(d_j(s_j) + eps) / sum_j w_j)
    """

    def __init__(
        self,
        weights: Sequence[float] = (0.50, 0.25, 0.25),
        rocs_anchors: tuple[float, float, float, float] = (0.50, 0.05, 0.70, 0.70),
        sa_anchors: tuple[float, float, float, float] = (0.20, 0.05, 0.80, 0.90),
        qed_anchors: tuple[float, float, float, float] = (0.30, 0.05, 0.70, 0.90),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = np.array(weights, dtype=np.float64)
        self.eps = float(eps)
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
        z = b0 + b1 * s
        z_clipped = np.clip(z, -30.0, 30.0)
        return np.exp(-np.exp(-z_clipped))

    def __call__(
        self,
        smiles: Sequence[str] | np.ndarray,
        scores: np.ndarray | None = None,
        thresholds: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate continuous composite Harrington desirability rewards."""
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            scores_arr = np.asarray(smiles, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = (
                scores_arr.reshape(1, -1)
                if len(scores_arr) > 0
                else scores_arr.reshape(0, 1)
            )

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        scores_clean = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)
        scores_clean = np.clip(scores_clean, 0.0, 1.0)
        n_objs = scores_clean.shape[1]

        w = (
            self.weights[:n_objs]
            if len(self.weights) >= n_objs
            else np.ones(n_objs, dtype=np.float64)
        )
        w_norm = w / np.sum(w)

        log_d_sum = np.zeros(n_samples, dtype=np.float64)
        for j in range(n_objs):
            b0, b1 = self.params[j] if j < len(self.params) else (0.0, 5.0)
            d_j = self._eval_desirability(scores_clean[:, j], b0, b1)
            log_d_sum += w_norm[j] * np.log(d_j + self.eps)

        rewards = np.exp(log_d_sum)
        rewards = np.clip(rewards, 0.0, 1.0)
        return rewards.reshape(-1, 1)


def resolve_reward_scheme(
    scheme: str | RewardScheme | None = None,
    weights: tuple[float, ...] = (0.50, 0.25, 0.25),
) -> RewardScheme:
    """Resolve RewardScheme instance from canonical name or pass through existing instance."""
    if isinstance(scheme, RewardScheme):
        return scheme
    name = (scheme or "weighted_sum").lower().strip()
    if name in ("weighted_sum", "ws"):
        return WeightedSum()
    elif name in ("pareto", "pareto_crowding"):
        return ParetoCrowdingDistance()
    elif name in ("harrington", "harrington_desirability", "desirability"):
        return HarringtonDesirabilityReward(weights=weights)
    else:
        raise ValueError(
            f"Unknown reward scheme: '{scheme}'. Valid options: ['weighted_sum', 'pareto_crowding', 'harrington']"
        )


def create_environment(
    ref_sdf: str | Path,
    config: PipelineConfig | None = None,
    thresholds: tuple[float, ...] | None = None,
    max_conformers: int | None = None,
    timeout: int | None = None,
    reward_scheme: str | RewardScheme | None = None,
    score_variant: str | None = None,
    conformer_threads: int | None = None,
) -> DrugExEnvironment:
    """Construct multi-objective DrugEx reinforcement learning environment for ROCS + SA + QED."""
    cfg = config or PipelineConfig()
    active_thresholds = thresholds if thresholds is not None else cfg.thresholds
    active_scheme_str = (
        reward_scheme if reward_scheme is not None else cfg.reward_scheme
    )
    active_variant = score_variant or cfg.score_variant
    active_max_confs = cfg.max_conformers if max_conformers is None else max_conformers
    active_threads = (
        conformer_threads if conformer_threads is not None else cfg.conformer_threads
    )

    rocs_scorer = create_rocs_scorer(
        ref_sdf,
        max_conformers=active_max_confs,
        timeout=cfg.timeout_seconds if timeout is None else timeout,
        score_variant=active_variant,
        conformer_threads=active_threads,
        max_isomers=cfg.max_isomers,
        minimize_candidates=cfg.minimize_candidates,
        optimization=cfg.optimization,
        n_jobs=HardwareManager.get_safe_parallel_jobs(ratio=cfg.cpu_ratio),
        random_seed=cfg.random_seed,
    )
    # Map raw TanimotoCombo [0, 2] linearly into [0, 1] desirability
    rocs_scorer.setModifier(
        ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)
    )

    # SA scorer: Smooth clipped score (raw 3.0 -> 0.88, raw 5.0 -> 0.12)
    sa_scorer = Property("SA")
    sa_scorer.setModifier(
        SmoothClippedScore(lower_x=5.0, upper_x=3.0, low_score=0.0, high_score=1.0)
    )

    # QED scorer: Drug-likeness score [0, 1]
    qed_scorer = Property("QED")
    qed_scorer.setModifier(
        ClippedScore(lower_x=0.0, upper_x=1.0, low_score=0.0, high_score=1.0)
    )

    scorers: list[Scorer] = [rocs_scorer, sa_scorer, qed_scorer]
    active_reward_scheme = resolve_reward_scheme(active_scheme_str, weights=cfg.weights)

    return DrugExEnvironment(
        scorers=scorers,
        thresholds=list(active_thresholds),
        reward_scheme=active_reward_scheme,
    )
