from __future__ import annotations

import re
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
from _david.code.rocs import ConfigurableRDKitROCSScorer
from _david.code.schema import std
from _david.code.utils import HardwareManager


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    config: PipelineConfig | None = None,
    max_conformers: int | None = None,
    timeout: int | None = None,
    score_variant: str | None = None,
    conformer_threads: int | None = None,
    n_jobs: int | None = None,
    max_isomers: int | None = None,
    random_seed: int | None = None,
    minimize_candidates: bool | None = None,
    optimization: Optimization | None = None,
    max_heavy_atoms: int | None = 45,
    max_rotatable_bonds: int | None = 15,
    tversky_alpha: float | None = None,
    tversky_beta: float | None = None,
    shape_weight: float | None = None,
    color_weight: float | None = None,
) -> RDKitROCSScorer:
    """Build multi-threaded RDKitROCSScorer configured to produce canonical std.ROCS key."""
    cfg = config or PipelineConfig()
    ref_input = str(references) if isinstance(references, Path) else references
    active_variant = (
        (std.VARIANT_TANIMOTO_COMBO if config is None else cfg.score_variant)
        if score_variant is None
        else score_variant
    )
    active_minimize = (
        (False if config is None else cfg.minimize_candidates)
        if minimize_candidates is None
        else minimize_candidates
    )
    active_opt = (
        (None if config is None else cfg.optimization)
        if optimization is None
        else optimization
    )
    safe_jobs = (
        n_jobs
        if n_jobs is not None
        else HardwareManager.get_safe_parallel_jobs(ratio=cfg.cpu_ratio)
    )
    conf_gen = CandidateConformerGenerator(
        max_conformers=cfg.max_conformers if max_conformers is None else max_conformers,
        max_isomers=cfg.max_isomers if max_isomers is None else max_isomers,
        max_heavy_atoms=cfg.max_heavy_atoms if max_heavy_atoms == 45 and config is not None else max_heavy_atoms,
        max_rotatable_bonds=cfg.max_rotatable_bonds if max_rotatable_bonds == 15 and config is not None else max_rotatable_bonds,
        num_threads=cfg.conformer_threads if conformer_threads is None else conformer_threads,
        show_progress=False,
        timeout=60 if (timeout is None and config is None) else (cfg.timeout_seconds if timeout is None else timeout),
        random_seed=cfg.random_seed if random_seed is None else random_seed,
        minimize_candidates=active_minimize,
        optimization=active_opt,
        n_jobs=safe_jobs,
    )
    variant = (
        std.VARIANT_TANIMOTO
        if active_variant in (std.VARIANT_TANIMOTO, std.VARIANT_TANIMOTO_COMBO)
        else active_variant
    )
    scorer = ConfigurableRDKitROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type=std.ROCS_SCORE_TYPE,
        score_variant=variant,
        use_colors=True,
        show_progress=False,
        n_jobs=safe_jobs,
        tversky_alpha=cfg.rocs_tversky_alpha if tversky_alpha is None else tversky_alpha,
        tversky_beta=cfg.rocs_tversky_beta if tversky_beta is None else tversky_beta,
        shape_weight=cfg.rocs_shape_weight if shape_weight is None else shape_weight,
        color_weight=cfg.rocs_color_weight if color_weight is None else color_weight,
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
            # Calibrate Gompertz anchors to physical [0, 2] scale when raw (0.4 raw ~ 0.2 norm, 1.2 raw ~ 0.6 norm)
            # or [0, 1] when normalized.
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
    scheme: str | RewardScheme | None = None,
    weights: tuple[float, ...] = (0.75, 0.25),
    rocs_raw: bool = True,
) -> RewardScheme:
    """Resolve RewardScheme instance from canonical std.REWARD_* constant or pass through."""
    if isinstance(scheme, RewardScheme):
        return scheme
    name = (scheme or std.REWARD_WEIGHTED_SUM).lower().strip()
    if name in (std.REWARD_WEIGHTED_SUM, "ws"):
        return WeightedSum()
    if name in (std.REWARD_PARETO, "pareto", "pareto_crowding"):
        return ParetoCrowdingDistance()
    if name in (std.REWARD_HARRINGTON, "harrington_desirability", "desirability"):
        return HarringtonDesirabilityReward(weights=weights, rocs_raw=rocs_raw)
    raise ValueError(
        f"Unknown reward scheme: '{scheme}'. Valid options: "
        f"[{std.REWARD_WEIGHTED_SUM!r}, {std.REWARD_PARETO!r}, {std.REWARD_HARRINGTON!r}]"
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
    include_qed: bool | None = None,
    rocs_use_modifier: bool | None = None,
) -> DrugExEnvironment:
    """Construct multi-objective DrugEx reinforcement learning environment."""
    cfg = config or PipelineConfig()
    use_qed = cfg.include_qed if include_qed is None else bool(include_qed)
    use_rocs_mod = (
        cfg.rocs_use_modifier if rocs_use_modifier is None else bool(rocs_use_modifier)
    )

    scorer_kwargs = cfg.rocs_scorer_kwargs(
        n_jobs=HardwareManager.get_safe_parallel_jobs(ratio=cfg.cpu_ratio)
    )
    if max_conformers is not None:
        scorer_kwargs["max_conformers"] = max_conformers
    if timeout is not None:
        scorer_kwargs["timeout"] = timeout
    if score_variant is not None:
        scorer_kwargs["score_variant"] = score_variant
    if conformer_threads is not None:
        scorer_kwargs["conformer_threads"] = conformer_threads

    rocs_scorer = create_rocs_scorer(ref_sdf, **scorer_kwargs)
    if use_rocs_mod:
        rocs_scorer.setModifier(
            ClippedScore(
                lower_x=0.0, upper_x=cfg.rocs_clip_upper, low_score=0.0, high_score=1.0
            )
        )
    else:
        # None / raw [0, 2] ROCS modifier
        rocs_scorer.setModifier(None)

    # SA ClippedScore(5, 3) so reward is flat (1.0) below raw SA 3.0, stopping artificial shrinkage
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

    if use_qed:
        qed_scorer = Property(std.QED)
        qed_scorer.setModifier(
            ClippedScore(lower_x=0.0, upper_x=1.0, low_score=0.0, high_score=1.0)
        )
        scorers.append(qed_scorer)

    if thresholds is not None:
        active_thresholds = thresholds
    elif cfg.thresholds is not None:
        active_thresholds = cfg.thresholds
    else:
        active_thresholds = (0.35, 0.50, 0.50) if use_qed else (0.35, 0.50)

    # Validate threshold dimensions against scorers
    if len(active_thresholds) != len(scorers):
        if len(active_thresholds) > len(scorers):
            active_thresholds = tuple(active_thresholds[: len(scorers)])
        else:
            padding = [0.50] * (len(scorers) - len(active_thresholds))
            active_thresholds = tuple(list(active_thresholds) + padding)

    active_scheme = reward_scheme if reward_scheme is not None else cfg.reward_scheme
    weights = (
        cfg.weights
        if cfg.weights is not None
        else ((0.50, 0.25, 0.25) if use_qed else (0.75, 0.25))
    )
    if len(weights) != len(scorers):
        if len(weights) > len(scorers):
            weights = tuple(weights[: len(scorers)])
        else:
            padding = [0.25] * (len(scorers) - len(weights))
            weights = tuple(list(weights) + padding)

    return DrugExEnvironment(
        scorers=scorers,
        thresholds=list(active_thresholds),
        reward_scheme=resolve_reward_scheme(
            active_scheme,
            weights=weights,
            rocs_raw=(not use_rocs_mod),
        ),
    )
