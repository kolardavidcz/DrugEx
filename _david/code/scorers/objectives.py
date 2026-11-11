"""The objectives, their desirability thresholds, and the environment that scores them."""

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

#: DrugEx's own schemes under DrugEx's own ``--scheme`` names. Nothing of ours is in any
#: of them: the experiment compares DrugEx against DrugEx.
SCHEMES = {
    "PRCD": ParetoCrowdingDistance,
    "PRTD": ParetoTanimotoDistance,
    "WS": WeightedSum,
}

#: Upper bound of raw TanimotoCombo, so also the factor between a ROCS value in reward
#: units and the published one. Recorded per run as ``rocs_reward_scale``; a run made
#: before the modifier has no such field, which is what its 1.0 default means.
ROCS_PUBLISHED_MAX = 2.0


def reward_scheme(name: str):
    """One of DrugEx's reward schemes, by name. Resolve it before anything expensive.

    Both Pareto schemes ignore the thresholds they are handed, which is what lets this
    project say thresholds reach the stop condition and nothing else. ``WS`` builds its
    weights *from* them, so a WS run and a PRCD run answer different questions.
    """
    try:
        return SCHEMES[name]()
    except KeyError:
        raise ValueError(
            f"unknown reward scheme {name!r}; DrugEx's are {', '.join(sorted(SCHEMES))}"
        ) from None


def build_scorers(cfg: Config, references: list[Chem.Mol]) -> list:
    """The objectives, in a fixed order: shape/colour, synthesizability, optionally QED."""
    rocs = ComboROCSScorer(
        conformer_generator=RelaxingConformerGenerator(cfg),
        references=references,
        score_type="TanimotoCombo",
        color_ff=cfg.color_ff,
        n_jobs=cfg.n_jobs,
        show_progress=False,
        alignment_shape_weight=cfg.alignment_shape_weight,
    )
    # Every objective on [0, 1]. Load-bearing under WS, which sums the columns, so raw
    # [0, 2] gave ROCS twice the leverage and pushed the reward to 1.38; a provable
    # no-op under Pareto, which is ordinal. Reporting is unaffected: ShapeTanimoto and
    # ColorTanimoto come from `components` and are never modified.
    rocs.setModifier(ClippedScore(lower_x=0.0, upper_x=ROCS_PUBLISHED_MAX))
    # Native Property: DrugEx leaves an unscorable SA at 10.0, which this descending
    # modifier maps to 0.0. A failure used to sit at 0.0 and rank as the best molecule.
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
    """Each requested objective's score for the reference molecule, on the modified scale.

    Only cheap molecule-level properties: the scorer is called on one molecule, so an
    objective needing conformers would re-embed the reference.
    """
    named = set(cfg.threshold_from_reference)
    # Against every name a scorer could provide, not the objectives this run enables:
    # filtering first made this guard unreachable, so a typo fell through to the
    # percentile -- for SA, back into the saturated gate the setting exists to avoid.
    known = {"SA", "QED"}
    unknown = named - known
    if unknown:
        raise ValueError(
            f"threshold_from_reference names {sorted(unknown)}, which is not a "
            f"molecule-level objective; choose from {sorted(known)}"
        )
    wanted = named & set(keys)
    if not wanted:
        return {}
    mol = Chem.MolFromSmiles(cfg.reference_smiles)
    if mol is None:
        raise ValueError(f"reference_smiles does not parse: {cfg.reference_smiles!r}")

    out = {}
    for scorer in scorers:
        key = scorer.getKey()
        name = key[0] if isinstance(key, list) else key
        if name in wanted:
            out[name] = float(np.asarray(scorer([mol])).ravel()[0])
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"no scorer provides {sorted(missing)}; objectives are {keys}")
    return out


def calibrate_thresholds(
    cfg: Config, scorers: list, prior_smiles: list[str]
) -> tuple[list[float], pd.DataFrame]:
    """Thresholds as a percentile of the prior's own output, plus the scored prior arm.

    Under either Pareto scheme these reach ``Desired`` -> ``desired_ratio`` ->
    checkpointing and early stopping, and nothing else. Do not add a reward term that
    reads them: that is a deliberate property of this pipeline. ``WS`` is the exception
    and builds its weights from these very numbers -- see :func:`reward_scheme`.

    The scored frame is returned because it *is* the pre-training baseline every later
    comparison is made against, and rescoring it would cost another full conformer pass.
    """
    # getScores only; no reward is ranked through this, so its scheme is immaterial.
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
        # A hand-set bar, so desired_ratio means "clears my bar", not "beats pXY of the
        # prior". Prefer the percentile once SA stops saturating -- see the warning below.
        logger.info(
            "fixed thresholds (percentile calibration skipped): %s",
            dict(zip(keys, [round(value, 4) for value in thresholds])),
        )
        return thresholds, scores

    # Measured against the reference instead of a percentile. For SA this rescues the
    # criterion: ClippedScore(5->3) pins ~68% of a drug-like prior at exactly 1.0, so p85
    # of that column IS its maximum. EGCG's own 0.63 is a real bar, and "at least as
    # synthesizable as the reference" is defensible where "top 15% of a tied column" is not.
    from_reference = _reference_thresholds(cfg, scorers, keys)

    thresholds = []
    for key in keys:
        if key in from_reference:
            thresholds.append(from_reference[key])
            logger.info("%s threshold %.4f taken from the reference", key, from_reference[key])
            continue
        column = valid[key]
        threshold = float(np.percentile(column, cfg.threshold_percentile))
        thresholds.append(threshold)
        # A percentile landing on the maximum means the objective became a hard gate at
        # the ceiling. Defensible, but not something a reader should infer from a flat column.
        if threshold >= column.max():
            logger.warning(
                "%s threshold %.4f is the maximum observed value (%d/%d prior molecules tied "
                "there): Desired is a hard gate on this objective, not a percentile",
                key,
                threshold,
                int((column >= threshold).sum()),
                len(column),
            )
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
    """Recompute ``Desired`` against ``thresholds``, by DrugEx's rule, without rescoring.

    The prior arm is scored before its own thresholds exist, so it comes back marked
    against placeholders. Rewriting it is what makes the prior comparable with the
    trained arms.
    """
    marked = scores.copy()
    meets = (marked[objectives] >= thresholds).all(axis=1)
    # An invalid molecule has every objective zeroed, so it clears an all-zero threshold
    # set. Native getScores does not guard this; run.py applies this rule to every arm.
    marked["Desired"] = (meets & marked["Valid"].eq(1)).astype(int)
    return marked


def build_environment(scorers: list, thresholds: list[float], scheme) -> DrugExEnvironment:
    """The multi-objective environment. ``scheme`` is an instance from :func:`reward_scheme`."""
    return DrugExEnvironment(scorers, thresholds, scheme)
