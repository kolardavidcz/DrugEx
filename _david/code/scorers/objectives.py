from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem

from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import (
    ParetoCrowdingDistance,
    ParetoTanimotoDistance,
    WeightedSum,
)
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property

from _david.code.config import Config
from _david.code.scorers.conformers import RelaxingConformerGenerator
from _david.code.scorers.rocs import ComboROCSScorer

logger = logging.getLogger(__name__)

SCHEMES = {
    "PRCD": ParetoCrowdingDistance,
    "PRTD": ParetoTanimotoDistance,
    "WS": WeightedSum,
}
ROCS_PUBLISHED_MAX = 2.0
MOLECULE_LEVEL = ("SA", "QED")


def reward_scheme(name: str):
    try:
        return SCHEMES[name]()
    except KeyError:
        raise ValueError(
            f"unknown reward scheme {name!r}; DrugEx's are {', '.join(sorted(SCHEMES))}"
        ) from None


def build_scorers(cfg: Config, references: list[Chem.Mol]) -> list:
    rocs = ComboROCSScorer(
        conformer_generator=RelaxingConformerGenerator(cfg),
        references=references,
        score_type="TanimotoCombo",
        color_ff=cfg.color_ff,
        n_jobs=cfg.n_jobs,
        show_progress=False,
        alignment_shape_weight=cfg.alignment_shape_weight,
    )
    rocs.setModifier(ClippedScore(lower_x=0.0, upper_x=ROCS_PUBLISHED_MAX))
    scorers = [
        rocs,
        Property(
            "SA", modifier=ClippedScore(lower_x=cfg.sa_reward_zero, upper_x=cfg.sa_reward_one)
        ),
    ]
    if cfg.include_qed:
        scorers.append(Property("QED"))
    return scorers


def _reference_thresholds(cfg: Config, scorers: list, keys: list[str]) -> dict[str, float]:
    named = set(cfg.threshold_from_reference)
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


def calibrate_thresholds(
    cfg: Config, scorers: list, prior_smiles: list[str]
) -> tuple[list[float], pd.DataFrame]:
    probe = DrugExEnvironment(scorers, [0.0] * len(scorers), ParetoCrowdingDistance())
    scores = probe.getScores(prior_smiles)
    valid = scores[scores["Valid"] == 1]
    if valid.empty:
        raise RuntimeError(
            f"the prior produced no valid molecules out of {len(prior_smiles)}; nothing to "
            "calibrate against"
        )

    keys = probe.getScorerKeys()
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
        return thresholds, scores

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
