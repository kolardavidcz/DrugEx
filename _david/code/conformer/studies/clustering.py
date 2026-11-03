"""Clustering-method benchmark (component 3).

Question this study answers
---------------------------
Does Butina RMSD merging at the ETKDG default ``0.35 Å`` preserve the basins a
shape+color farthest-first bank keeps, and how many conformers does each rule
retain?  Legacy ``1.25 Å`` merging collapses most drugs to one or two poses,
which destroys score; ``0.35 Å`` keeps true basins at a modest bank size.

The table carries both the *cost* of a rule (``cluster_count``, i.e. how many
distinct conformers survive merging) and its *value* (score at a fixed cap), so
"more clusters" is never mistaken for "better" without the score column beside
it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from _david.code.conformer.core.contracts import MoleculeSpec, StudySpec
from _david.code.conformer.studies.optimization_methods import (
    MethodPair,
    build_study as build_method_study,
)

#: Selection rules compared in the benchmark, cheapest to strictest merging.
SELECTIONS: tuple[str, ...] = ("diverse", "rmsd_0.35", "rmsd_0.50", "rmsd_1.25")

#: Short labels used on figure axes.
SELECTION_LABELS: dict[str, str] = {
    "diverse": "Diverse farthest-first (shape+color)",
    "rmsd_0.35": "Butina RMSD 0.35 Å (ETKDG default)",
    "rmsd_0.50": "Butina RMSD 0.50 Å",
    "rmsd_1.25": "Butina RMSD 1.25 Å (legacy)",
}


def build_study(
    molecules: Sequence[MoleculeSpec],
    output_dir: Path | str,
    *,
    reference_method: str = "mmff94s_water",
    candidate_method: str = "etkdg",
    selections: Sequence[str] = SELECTIONS,
    caps: Sequence[int] = (1, 2, 3, 4, 8, 12, 0),
    candidate_attempts: Sequence[int] = (20,),
    reference_attempts: int = 50,
    name: str = "clustering_benchmark",
    notes: str = "",
) -> StudySpec:
    """Build the clustering benchmark matrix.

    Physics is held fixed so the only varying axis is the merging rule.  The
    default reference is the deployable MMFF94s ε=78.4 bank and the default
    candidate is an unminimized ETKDG pool, because that is what the RL reward
    actually scores.  Arm tags already carry the selection rule, so downstream
    grouping never has to parse arm names.
    """
    return build_method_study(
        molecules,
        output_dir,
        pairs=(MethodPair(reference_method, candidate_method),),
        selections=selections,
        caps=caps,
        candidate_attempts=candidate_attempts,
        reference_attempts=reference_attempts,
        name=name,
        notes=notes,
    )


def cluster_summary(
    frame: pd.DataFrame,
    *,
    score: str = "combo",
    cap: int | None = 4,
) -> pd.DataFrame:
    """Per-selection cluster counts and score at a fixed reference cap.

    ``delta_vs_diverse`` is the paired score difference against the farthest-first
    bank on molecules present in both rules, which is the only fair comparison
    when a strict rule fails to produce a conformer for some drug.
    """
    if frame.empty:
        return frame.copy()
    subset = frame if cap is None else frame[frame["reference_cap"] == cap]
    if subset.empty:
        return subset.copy()

    summary = (
        subset.groupby("arm", as_index=False)
        .agg(
            selection=("selection_mode", "first"),
            rmsd_cutoff=("rmsd_cutoff", "first"),
            cluster_count=("cluster_count", "mean"),
            references=("retained_references", "mean"),
            score=(score, "mean"),
            molecules=("molecule", "nunique"),
        )
        .sort_values("rmsd_cutoff")
        .reset_index(drop=True)
    )

    baseline = subset[subset["selection_mode"] == "diverse"]
    if baseline.empty:
        summary["delta_vs_diverse"] = float("nan")
        return summary
    base = baseline.set_index("molecule")[score]
    deltas: list[float] = []
    for _, row in summary.iterrows():
        arm_rows = subset[subset["arm"] == row["arm"]].set_index("molecule")[score]
        common = arm_rows.index.intersection(base.index)
        deltas.append(
            float((arm_rows.loc[common] - base.loc[common]).mean())
            if len(common)
            else float("nan")
        )
    summary["delta_vs_diverse"] = deltas
    return summary


__all__ = [
    "SELECTIONS",
    "SELECTION_LABELS",
    "build_study",
    "cluster_summary",
]
