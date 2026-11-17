from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.config import MOLECULE_LEVEL, ROCS_PUBLISHED_MAX, SCHEMES, Config
from _david.code.scorers.conformers import RelaxingConformerGenerator
from _david.code.scorers.rocs import ComboROCSScorer
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property

logger = logging.getLogger(__name__)


#! could we just have assert to use fappropirate funcison names + in config.py use acctuall funl names isnted of our short version
def reward_scheme(name: str):
    try:
        return SCHEMES[name]()
    except KeyError:
        raise ValueError(
            f"unknown reward scheme {name!r}; DrugEx's are {', '.join(sorted(SCHEMES))}"
        ) from None


def build_scorers(
    cfg: Config,
    references: list[Chem.Mol],
    *,
    basin_log_dir: Path | None = None,
    basin_stage: str = "init",
) -> list:
    if cfg.track_conformer_basins:
        from _david.code.scorers.rocs_basin_tmp import BasinTrackingROCSScorer

        rocs = BasinTrackingROCSScorer(
            conformer_generator=RelaxingConformerGenerator(cfg),
            references=references,
            score_type="TanimotoCombo",
            color_ff=cfg.color_ff,
            n_jobs=cfg.n_jobs,
            show_progress=False,
            alignment_shape_weight=cfg.alignment_shape_weight,
            cfg=cfg,
            basin_log_dir=basin_log_dir,
            stage=basin_stage,
        )
    else:
        rocs = ComboROCSScorer(
            conformer_generator=RelaxingConformerGenerator(cfg),
            references=references,
            score_type="TanimotoCombo",
            color_ff=cfg.color_ff,
            n_jobs=cfg.n_jobs,
            show_progress=False,
            alignment_shape_weight=cfg.alignment_shape_weight,
        )
    # Onto [0, 1]: WS sums the columns, so raw [0, 2] gave ROCS twice the leverage.
    rocs.setModifier(ClippedScore(lower_x=0.0, upper_x=ROCS_PUBLISHED_MAX))
    scorers = [
        rocs,
        # Descending: native's unscorable SA of 10.0 must map to 0.0, not to the best.
        Property(
            "SA", modifier=ClippedScore(lower_x=cfg.sa_reward_zero, upper_x=cfg.sa_reward_one)
        ),
    ]
    if cfg.include_qed:
        scorers.append(Property("QED"))
    return scorers


def set_basin_stage(scorers_or_env, stage: str) -> None:
    """Update the stage label on any ``BasinTrackingROCSScorer`` in ``scorers_or_env``."""
    scorers = getattr(scorers_or_env, "scorers", scorers_or_env)
    if not isinstance(scorers, (list, tuple)):
        return
    for scorer in scorers:
        if hasattr(scorer, "set_stage"):
            scorer.set_stage(stage)


def _reference_thresholds(cfg: Config, scorers: list, keys: list[str]) -> dict[str, float]:
    named = set(cfg.threshold_from_reference)
    # Before filtering to this run's objectives, or a typo falls through to the percentile.
    unknown = named - set(MOLECULE_LEVEL)
    if unknown:
        raise ValueError(
            f"threshold_from_reference names {sorted(unknown)}, which is not a "
            f"molecule-level objective; choose from {sorted(MOLECULE_LEVEL)}"
        )
    wanted = named & set(keys)
    if not wanted:
        return {}
    mol = Chem.MolFromSmiles(cfg.reference_smiles)
    if mol is None:
        raise ValueError(f"reference_smiles does not parse: {cfg.reference_smiles!r}")

    by_key = {}
    for scorer in scorers:
        key = scorer.getKey()
        by_key[key[0] if isinstance(key, list) else key] = scorer
    missing = wanted - set(by_key)
    if missing:
        raise ValueError(f"no scorer provides {sorted(missing)}; objectives are {keys}")
    return {name: float(np.asarray(by_key[name]([mol])).ravel()[0]) for name in sorted(wanted)}


def _percentile_threshold(column: pd.Series, percentile: float, key: str) -> float:
    threshold = float(np.percentile(column, percentile))
    if threshold >= column.max():
        logger.warning(
            "%s threshold %.4f is the maximum observed value (%d/%d prior molecules tied "
            "there): Desired is a hard gate on this objective, not a percentile",
            key,
            threshold,
            int((column >= threshold).sum()),
            len(column),
        )
    return threshold


def thresholds_from_scores(
    cfg: Config, scorers: list, scores: pd.DataFrame, keys: list[str]
) -> list[float]:
    """Compute per-objective thresholds from a scored prior DataFrame without re-scoring."""
    valid = scores[scores["Valid"] == 1]
    if valid.empty:
        raise RuntimeError(
            f"the prior produced no valid molecules out of {len(scores)}; nothing to "
            "calibrate against"
        )

    if cfg.fixed_thresholds:
        if len(cfg.fixed_thresholds) != len(keys):
            raise ValueError(
                f"fixed_thresholds has {len(cfg.fixed_thresholds)} values for {len(keys)} "
                f"objectives {keys}"
            )
        thresholds = [float(value) for value in cfg.fixed_thresholds]
        logger.info(
            "fixed thresholds (percentile calibration skipped): %s",
            dict(zip(keys, [round(value, 4) for value in thresholds])),
        )
        return thresholds

    # SA saturates: p85 of that column is its maximum, so the reference's own value is used.
    from_reference = _reference_thresholds(cfg, scorers, keys)

    thresholds = []
    for key in keys:
        if key in from_reference:
            logger.info("%s threshold %.4f taken from the reference", key, from_reference[key])
            thresholds.append(from_reference[key])
        else:
            thresholds.append(_percentile_threshold(valid[key], cfg.threshold_percentile, key))
    logger.info(
        "thresholds at p%g of %d valid prior molecules: %s",
        cfg.threshold_percentile,
        len(valid),
        dict(zip(keys, [round(value, 4) for value in thresholds])),
    )
    return thresholds


def calibrate_thresholds(
    cfg: Config, scorers: list, prior_smiles: list[str]
) -> tuple[list[float], pd.DataFrame]:
    """A percentile of the prior's own output, plus the scored prior arm as the baseline."""
    probe = DrugExEnvironment(scorers, [0.0] * len(scorers), ParetoCrowdingDistance())
    scores = probe.getScores(prior_smiles)
    keys = probe.getScorerKeys()
    thresholds = thresholds_from_scores(cfg, scorers, scores, keys)
    return thresholds, scores


def mark_desired(
    scores: pd.DataFrame, objectives: list[str], thresholds: list[float]
) -> pd.DataFrame:
    marked = scores.copy()
    meets = (marked[objectives] >= thresholds).all(axis=1)
    marked["Desired"] = (meets & marked["Valid"].eq(1)).astype(int)
    return marked


def build_environment(scorers: list, thresholds: list[float], scheme) -> DrugExEnvironment:
    return DrugExEnvironment(scorers, thresholds, scheme)
