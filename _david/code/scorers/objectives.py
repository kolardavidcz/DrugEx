"""Objective scorers, reward schemes, prior threshold calibration, and DrugEx environment."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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


def reward_scheme(name: str) -> Any:
    """Instantiate a DrugEx reward scheme from its short name (``PRCD``, ``PRTD``, ``WS``)."""
    if name not in SCHEMES:
        raise ValueError(f"unknown reward scheme {name!r}; expected one of {', '.join(sorted(SCHEMES))}")
    return SCHEMES[name]()


def scorer_key(scorer: Any) -> str:
    """Primary column key emitted by a single DrugEx scorer."""
    key = scorer.getKey()
    return str(key[0] if isinstance(key, list) else key)


def scorer_keys(scorers: Sequence[Any]) -> list[str]:
    """Flatten every column key emitted by ``scorers`` in order."""
    return [
        str(item)
        for scorer in scorers
        for item in (scorer.getKey() if isinstance(scorer.getKey(), list) else [scorer.getKey()])
    ]


def build_scorers(
    cfg: Config,
    references: list[Chem.Mol] | list[str],
    *,
    basin_log_dir: Path | None = None,
    basin_stage: str = "init",
) -> list[Any]:
    """Build ordered objective scorers on [0, 1]: ROCS, SA desirability, and optional QED."""
    rocs_kwargs = {
        "conformer_generator": RelaxingConformerGenerator(cfg),
        "references": references,
        "score_type": "TanimotoCombo",
        "color_ff": cfg.color_ff,
        "n_jobs": cfg.n_jobs,
        "show_progress": False,
        "alignment_shape_weight": cfg.alignment_shape_weight,
    }
    if cfg.track_conformer_basins:
        from _david.code.scorers.rocs_basin_tmp import BasinTrackingROCSScorer

        rocs: ComboROCSScorer = BasinTrackingROCSScorer(
            **rocs_kwargs, cfg=cfg, basin_log_dir=basin_log_dir, stage=basin_stage
        )
    else:
        rocs = ComboROCSScorer(**rocs_kwargs)

    # Scale ROCS [0, 2] -> [0, 1] so WeightedSum weights every objective on the same range.
    rocs.setModifier(ClippedScore(lower_x=0.0, upper_x=ROCS_PUBLISHED_MAX))
    # Descending SA modifier: DrugEx's unscorable SA=10.0 maps to 0.0, not 1.0.
    scorers: list[Any] = [
        rocs,
        Property("SA", modifier=ClippedScore(lower_x=cfg.sa_reward_zero, upper_x=cfg.sa_reward_one)),
    ]
    if cfg.include_qed:
        scorers.append(Property("QED"))
    return scorers


def set_basin_stage(target: Any, stage: str) -> None:
    """Update the stage label on any ``BasinTrackingROCSScorer`` in ``target``."""
    for scorer in getattr(target, "scorers", target if isinstance(target, (list, tuple)) else ()):
        if hasattr(scorer, "set_stage"):
            scorer.set_stage(stage)


def _reference_thresholds(
    cfg: Config, scorers: Sequence[Any], keys: Sequence[str]
) -> dict[str, float]:
    named = set(cfg.threshold_from_reference)
    unknown = named - set(MOLECULE_LEVEL)
    if unknown:
        raise ValueError(
            f"threshold_from_reference names {sorted(unknown)}; choose from {sorted(MOLECULE_LEVEL)}"
        )
    wanted = named & set(keys)
    if not wanted:
        return {}
    mol = Chem.MolFromSmiles(cfg.reference_smiles)
    if mol is None:
        raise ValueError(f"reference_smiles does not parse: {cfg.reference_smiles!r}")
    by_key = {scorer_key(scorer): scorer for scorer in scorers}
    return {name: float(np.asarray(by_key[name]([mol])).ravel()[0]) for name in sorted(wanted)}


def thresholds_from_scores(
    cfg: Config, scorers: Sequence[Any], scores: pd.DataFrame, keys: Sequence[str]
) -> list[float]:
    """Derive per-objective thresholds from a scored prior DataFrame."""
    valid = scores[scores["Valid"] == 1]
    if valid.empty:
        raise RuntimeError(f"prior produced 0 valid molecules out of {len(scores)}")

    if cfg.fixed_thresholds:
        if len(cfg.fixed_thresholds) != len(keys):
            raise ValueError(
                f"fixed_thresholds has {len(cfg.fixed_thresholds)} values for {len(keys)} objectives"
            )
        thresholds = [float(v) for v in cfg.fixed_thresholds]
        logger.info("fixed thresholds: %s", dict(zip(keys, [round(v, 4) for v in thresholds])))
        return thresholds

    from_ref = _reference_thresholds(cfg, scorers, keys)
    thresholds: list[float] = []
    for key in keys:
        if key in from_ref:
            thresholds.append(from_ref[key])
            logger.info("%s threshold %.4f taken from reference", key, from_ref[key])
        else:
            col = valid[key]
            cut = float(np.percentile(col, cfg.threshold_percentile))
            if cut >= col.max():
                logger.warning(
                    "%s threshold %.4f hits prior maximum (%d/%d tied)",
                    key,
                    cut,
                    int((col >= cut).sum()),
                    len(col),
                )
            thresholds.append(cut)

    logger.info(
        "thresholds at p%g of %d valid prior molecules: %s",
        cfg.threshold_percentile,
        len(valid),
        dict(zip(keys, [round(v, 4) for v in thresholds])),
    )
    return thresholds


def calibrate_thresholds(
    cfg: Config, scorers: Sequence[Any], prior_smiles: list[str]
) -> tuple[list[float], pd.DataFrame]:
    """Score the prior sample and calibrate per-objective thresholds from its valid rows."""
    probe = DrugExEnvironment(list(scorers), [0.0] * len(scorers), ParetoCrowdingDistance())
    scores = probe.getScores(prior_smiles)
    return thresholds_from_scores(cfg, scorers, scores, probe.getScorerKeys()), scores


def mark_desired(
    scores: pd.DataFrame, objectives: Sequence[str], thresholds: Sequence[float]
) -> pd.DataFrame:
    """Return a copy of ``scores`` with binary ``Desired`` set where valid rows clear all cuts."""
    marked = scores.copy()
    meets = (marked[list(objectives)] >= thresholds).all(axis=1)
    marked["Desired"] = (meets & marked["Valid"].eq(1)).astype(int)
    return marked


def build_environment(
    scorers: Sequence[Any], thresholds: Sequence[float], scheme: Any
) -> DrugExEnvironment:
    """Assemble the :class:`DrugExEnvironment` pairing scorers, thresholds, and reward scheme."""
    return DrugExEnvironment(list(scorers), list(thresholds), scheme)
