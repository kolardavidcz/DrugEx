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

from _david.config import Config
from _david.scorers.conformers import RelaxingConformerGenerator
from _david.scorers.rocs import ComboROCSScorer

logger = logging.getLogger(__name__)

#: DrugEx's own reward schemes under DrugEx's own ``--scheme`` names (``drugex/train.py``).
#: Nothing of ours is in any of them; the experiment compares DrugEx against DrugEx.
SCHEMES = {
    "PRCD": ParetoCrowdingDistance,
    "PRTD": ParetoTanimotoDistance,
    "WS": WeightedSum,
}


def reward_scheme(name: str):
    """One of DrugEx's reward schemes, by name. Resolve it *before* anything expensive.

    **The two families differ in what they read.** Both Pareto schemes accept the
    thresholds and ignore them, which is what lets this project state that thresholds
    reach the stop condition and nothing else. ``WS`` does not: ``WeightedSum`` builds
    each objective's weight from the fraction of the batch below versus above its own
    threshold, so under ``WS`` the thresholds *are* part of the reward. That is DrugEx's
    design, not an addition here, but it means a ``WS`` run and a ``PRCD`` run are not
    answering the same question, and `report.md` says so in the run's own words.

    ``WS`` also sums the raw columns, so an objective's *range* is leverage:
    TanimotoCombo on [0, 2] moves the sum twice as far as SA or QED on [0, 1] at equal
    weight. Pareto ranking is scale-free and has no such asymmetry.
    """
    try:
        return SCHEMES[name]()
    except KeyError:
        raise ValueError(
            f"unknown reward scheme {name!r}; DrugEx's are {', '.join(sorted(SCHEMES))}"
        ) from None


def build_scorers(cfg: Config, references: list[Chem.Mol]) -> list:
    """The objectives, in a fixed order: shape/colour, synthesizability, optionally QED.

    The ROCS scorer carries no modifier, so its raw TanimotoCombo on [0, 2] is what enters
    the reward and the metrics. Under Pareto ranking the scale does not affect which
    molecule wins, and ``avg_gmean`` is scale-free in the same way: it responds to a
    relative change in either objective equally, whatever their units.
    """
    rocs = ComboROCSScorer(
        conformer_generator=RelaxingConformerGenerator(cfg),
        references=references,
        score_type="TanimotoCombo",
        color_ff=cfg.color_ff,
        n_jobs=cfg.n_jobs,
        show_progress=False,
        alignment_shape_weight=cfg.alignment_shape_weight,
    )
    # Native Property, not a subclass: drugex's FAILURE_VALUES leaves an unscorable SA at
    # 10.0, the worst value on its own scale, which this descending modifier maps to 0.0.
    # Before that, a failure sat at 0.0 and the same modifier called it the best molecule
    # in the batch.
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

    Only cheap molecule-level properties belong here: the scorer is called on one
    molecule, so an objective needing conformers would re-embed the reference.
    """
    named = set(cfg.threshold_from_reference)
    # Validate against the names any scorer could provide, not against the objectives this
    # run happens to enable: filtering first made the guard below unreachable, so a typo
    # ("Sa", "SAscore") was silently dropped and the run fell back to the percentile --
    # for SA, straight back into the saturated hard gate the setting exists to avoid.
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
        if name not in wanted:
            continue
        value = float(np.asarray(scorer([mol])).ravel()[0])
        out[name] = value
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"no scorer provides {sorted(missing)}; objectives are {keys}")
    return out


def calibrate_thresholds(
    cfg: Config, scorers: list, prior_smiles: list[str]
) -> tuple[list[float], pd.DataFrame]:
    """Thresholds as a percentile of the prior's own output, plus the scored prior arm.

    Under either Pareto scheme the thresholds reach ``Desired`` -> ``desired_ratio`` ->
    checkpointing and early stopping, and nothing else: both ignore the thresholds they
    are handed, so no part of this touches the policy gradient. Do not add a reward term
    that reads them -- that is a deliberate property of this pipeline, not an accident.

    The one exception is DrugEx's own ``WS`` (``scheme = "WS"``), where ``WeightedSum``
    builds the per-objective weights from these very numbers. Nothing here changes for
    it; it is named so that the exception is visible where the thresholds are made, and
    ``report.md`` states it in any run that uses it. See :func:`reward_scheme`.

    The scored frame is returned rather than discarded because it *is* the pre-training
    baseline every later comparison is made against, and rescoring it later would cost
    another full pass over the conformer generator.
    """
    # The probe only calls getScores, which computes the objective columns and `Desired`;
    # no reward is ever ranked through it, so its scheme is immaterial and stays native's.
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
        # NOTE: a fixed threshold is a hand-set bar, so `desired_ratio` means "clears my bar",
        # not "beats pXY of the prior". Prefer the percentile once SA no longer saturates:
        # ClippedScore(5->3) pins ~68% of prior molecules at exactly 1.0, so p85 of that column
        # IS the column maximum and the percentile degenerates into a gate at raw SA <= 3.
        # Widen the clip (sa_reward_zero/sa_reward_one) or take the percentile on the raw SA
        # scale, then drop fixed_thresholds and go back to pXY -- it is the defensible baseline
        # for a thesis because it is measured against this run's own prior, not chosen.
        logger.info(
            "fixed thresholds (percentile calibration skipped): %s",
            dict(zip(keys, [round(value, 4) for value in thresholds])),
        )
        return thresholds, scores

    # Objectives measured against the reference molecule itself rather than against a
    # percentile of the prior. For SA this is what rescues the criterion: ClippedScore
    # saturates roughly two thirds of a drug-like prior at exactly 1.0, so p85 of that
    # column IS the column maximum and "desired" degenerates into "equal to the best".
    # EGCG's own modified SA is 0.63, which is a real bar, and "at least as synthesizable
    # as the reference" is a claim a thesis can defend -- unlike "in the top 15 % of a
    # column where 68 % are tied".
    from_reference = _reference_thresholds(cfg, scorers, keys)

    thresholds = []
    for key in keys:
        if key in from_reference:
            thresholds.append(from_reference[key])
            logger.info(
                "%s threshold %.4f taken from the reference molecule itself",
                key,
                from_reference[key],
            )
            continue
        column = valid[key]
        threshold = float(np.percentile(column, cfg.threshold_percentile))
        thresholds.append(threshold)
        # A percentile landing on the maximum means at least (100 - p)% of the prior is
        # tied at the top, which happens whenever a modifier saturates: ClippedScore(5->3)
        # gives every molecule with raw SA <= 3 exactly 1.0. The objective then stops being
        # "better than p% of the prior" and becomes a hard gate at the ceiling. That is
        # defensible, but it must not be something a reader has to infer from a flat column.
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

    The prior arm is scored before its own thresholds exist, so the ``Desired`` column it
    comes back with was computed against placeholders. Rewriting it here is what makes the
    prior's desired_ratio comparable with the trained arms', which is the whole point of
    having a baseline.
    """
    marked = scores.copy()
    meets = (marked[objectives] >= thresholds).all(axis=1)
    # An invalid molecule has every objective zeroed, so it clears an all-zero threshold
    # set. Native DrugExEnvironment.getScores does NOT guard this -- it computes Desired
    # from the thresholds alone -- so with any threshold at 0.0 it marks invalid rows
    # desirable. This does guard it, and run.py applies the same rule to every arm.
    marked["Desired"] = (meets & marked["Valid"].eq(1)).astype(int)
    return marked


def build_environment(scorers: list, thresholds: list[float], scheme) -> DrugExEnvironment:
    """The multi-objective environment, under one of DrugEx's reward schemes.

    The default is ``ParetoCrowdingDistance`` (``--scheme PRCD``), DrugEx's own: it ranks
    by domination and breaks ties inside a front by NSGA-II crowding distance, so no
    objective is given a weight and the thresholds stay out of the reward. ``scheme`` is
    an instance from :func:`reward_scheme`, which documents what ``WS`` changes about
    that sentence.
    """
    return DrugExEnvironment(scorers, thresholds, scheme)
