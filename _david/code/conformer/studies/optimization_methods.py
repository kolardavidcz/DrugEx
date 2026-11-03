"""Optimization-method and Pareto studies (component 1).

Question this study answers
---------------------------
Given a fixed 3D similarity metric (TanimotoCombo), which combination of
reference physics, candidate physics, reference bank size ``K``, and candidate
budget ``C`` buys the most self-recovery per second?  The answer is a Pareto
front, not a single winner: minimized references are expensive but improve the
ceiling, unminimized candidates are cheap but lose tail accuracy.

Arm naming
----------
``ref-<method>__cand-<method>__<selection>-K<cap>__C<attempts>``

Methods are ``mmff94s_water``, ``mmff94s_vacuum``, ``uff``, and ``etkdg``.
Every arm uses the same preparation, stereoisomer cap, and disjoint seed streams
as the RL reward pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _david.code.chemistry import Optimization
from _david.code.conformer.core.contracts import (
    ArmSpec,
    EmbedSpec,
    MinimizeSpec,
    MoleculeSpec,
    ReferenceSpec,
    ScoreScale,
    ScoreSpec,
    SeedPlan,
    SelectSpec,
    SelectionMode,
    StudySpec,
    default_candidate_spec,
)
from _david.code.conformer.core.errors import StudyError
from _david.code.conformer.core.taxonomy import isomer_bin

#: Named force-field conditions used by the optimization-method study.
METHODS: dict[str, Optimization | None] = {
    "mmff94s_water": Optimization("MMFF94s", 78.4),
    "mmff94s_vacuum": Optimization("MMFF94s", 1.0),
    "uff": Optimization("UFF", 1.0),
    "etkdg": None,
}

#: Human-readable physics label for tables and figures.
METHOD_LABELS: dict[str, str] = {
    "mmff94s_water": "MMFF94s ε=78.4",
    "mmff94s_vacuum": "MMFF94s ε=1.0",
    "uff": "UFF",
    "etkdg": "ETKDG (unminimized)",
}

#: Selection modes available for the reference bank axis.
SELECTION_MODES: dict[str, SelectionMode] = {
    "diverse": SelectionMode.DIVERSE,
    "rmsd_0.35": SelectionMode.BUTINA,
    "rmsd_0.50": SelectionMode.BUTINA,
    "rmsd_1.25": SelectionMode.BUTINA,
}

SELECTION_CUTOFFS: dict[str, float] = {
    "diverse": 0.35,
    "rmsd_0.35": 0.35,
    "rmsd_0.50": 0.50,
    "rmsd_1.25": 1.25,
}

#: Score specification shared by every optimization-method arm.
STUDY_SCORE = ScoreSpec(scale=ScoreScale.NORMALIZED)


@dataclass(frozen=True)
class MethodPair:
    """One reference/candidate physics combination."""

    reference: str
    candidate: str

    @property
    def label(self) -> str:
        return f"ref-{self.reference}__cand-{self.candidate}"


def symmetric_pairs(methods: Sequence[str] = tuple(METHODS)) -> list[MethodPair]:
    """Every method scored against its own physics (the symmetric matrix)."""
    for name in methods:
        if name not in METHODS:
            raise StudyError(f"Unknown optimization method {name!r}")
    return [MethodPair(name, name) for name in methods]


def deployable_pairs(reference: str = "mmff94s_water") -> list[MethodPair]:
    """Asymmetric arms that isolate the reward-pipeline trade-off."""
    if reference not in METHODS:
        raise StudyError(f"Unknown optimization method {reference!r}")
    return [MethodPair(reference, candidate) for candidate in METHODS]


def _reference_spec(
    method: str,
    *,
    selection: str,
    cap: int,
    attempts: int,
) -> ReferenceSpec:
    if method not in METHODS:
        raise StudyError(f"Unknown optimization method {method!r}")
    if selection not in SELECTION_MODES:
        raise StudyError(f"Unknown selection mode {selection!r}")
    mode = SELECTION_MODES[selection]
    return ReferenceSpec(
        embed=EmbedSpec(
            attempts=attempts,
            isomer_cap=8,
            seed=104729,
            stereo_seed=42,
            rmsd=0.35,
        ),
        minimize=MinimizeSpec(profile=METHODS[method], energy_window=8.0),
        select=SelectSpec(mode=mode, cap=cap, rmsd_cutoff=SELECTION_CUTOFFS[selection]),
    )


def build_arms(
    *,
    pairs: Iterable[MethodPair] = (),
    selections: Sequence[str] = ("diverse",),
    caps: Sequence[int] = (1, 2, 3, 4, 8, 12, 0),
    candidate_attempts: Sequence[int] = (10, 20, 30, 50),
    reference_attempts: int = 50,
) -> list[ArmSpec]:
    """Expand the study axes into a de-duplicated arm list."""
    pairs = list(pairs) or symmetric_pairs()
    arms: list[ArmSpec] = []
    seen: set[str] = set()
    for pair in pairs:
        for selection in selections:
            for cap in caps:
                for attempts in candidate_attempts:
                    name = (
                        f"{pair.label}__{selection}-K{cap}__C{attempts}"
                    )
                    if name in seen:
                        continue
                    seen.add(name)
                    arms.append(
                        ArmSpec(
                            name=name,
                            reference=_reference_spec(
                                pair.reference,
                                selection=selection,
                                cap=cap,
                                attempts=reference_attempts,
                            ),
                            candidate=default_candidate_spec(
                                method=METHODS[pair.candidate],
                                attempts=attempts,
                                isomer_cap=4,
                                seed=42,
                                rmsd=0.5,
                            ),
                            score=STUDY_SCORE,
                            description=(
                                f"reference={METHOD_LABELS[pair.reference]}, "
                                f"candidate={METHOD_LABELS[pair.candidate]}, "
                                f"selection={selection}, K={cap}, C={attempts}"
                            ),
                            tags=(pair.reference, pair.candidate, selection),
                        )
                    )
    if not arms:
        raise StudyError("Arm matrix expanded to zero arms")
    return arms


def build_study(
    molecules: Sequence[MoleculeSpec],
    output_dir: Path | str,
    *,
    pairs: Iterable[MethodPair] = (),
    selections: Sequence[str] = ("diverse",),
    caps: Sequence[int] = (1, 2, 3, 4, 8, 12, 0),
    candidate_attempts: Sequence[int] = (10, 20, 30, 50),
    reference_attempts: int = 50,
    name: str = "optimization_methods",
    notes: str = "",
) -> StudySpec:
    """Build the optimization-method study matrix."""
    return StudySpec(
        name=name,
        molecules=tuple(molecules),
        arms=tuple(
            build_arms(
                pairs=pairs,
                selections=selections,
                caps=caps,
                candidate_attempts=candidate_attempts,
                reference_attempts=reference_attempts,
            )
        ),
        output_dir=Path(output_dir),
        seeds=SeedPlan(reference_base=104729, candidate_base=42, stereo=42, repeats=(0,)),
        notes=notes,
    )


def pareto_front(
    frame: pd.DataFrame,
    *,
    score: str = "combo",
    cost: str = "total_seconds",
    group: str = "arm",
) -> pd.DataFrame:
    """Return the non-dominated arms: no other arm is both cheaper and better.

    Ties are resolved by keeping the cheaper arm.  The result is sorted by cost
    so it reads left-to-right as a cost-accuracy frontier.
    """
    if frame.empty:
        return frame.copy()
    summary = (
        frame.groupby(group, as_index=False)
        .agg(score=(score, "mean"), cost=(cost, "mean"), n=(score, "size"))
        .sort_values(["cost", "score"], ascending=[True, False])
    )
    keep: list[int] = []
    best = float("-inf")
    for index, row in summary.iterrows():
        if row["score"] > best + 1e-12:
            keep.append(index)
            best = float(row["score"])
    return summary.loc[keep].sort_values("cost").reset_index(drop=True)


def stratified_summary(
    frame: pd.DataFrame,
    *,
    score: str = "combo",
) -> pd.DataFrame:
    """Mean score per reference cap, split by true stereoisomer count.

    Uses the canonical isomer bins so this table and the component-1
    stratification figure describe exactly the same groups.
    """
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    column = (
        "isomer_upper_bound"
        if "isomer_upper_bound" in working.columns
        else "isomers"
    )
    working["isomer_bin"] = isomer_bin(working[column])
    return (
        working.groupby(["isomer_bin", "reference_cap"], observed=True, as_index=False)
        .agg(score=(score, "mean"), molecules=("molecule", "nunique"))
    )


__all__ = [
    "METHODS",
    "METHOD_LABELS",
    "MethodPair",
    "SELECTION_CUTOFFS",
    "SELECTION_MODES",
    "STUDY_SCORE",
    "build_arms",
    "build_study",
    "deployable_pairs",
    "pareto_front",
    "stratified_summary",
    "symmetric_pairs",
]
