"""Shared tabular transforms used by more than one component figure.

The isomer bins live in :mod:`_david.code.conformer.core.taxonomy` because the
study runners need them too; this module re-exports them so figure code has one
import site and so the canonical definition cannot drift between figure and
study tables.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from _david.code.conformer.core.taxonomy import (
    ISOMER_BIN_EDGES,
    ISOMER_BIN_LABELS,
    bin_counts,
    isomer_bin,
)


def method_label(key: str, labels: Mapping[str, str] | None = None) -> str:
    """Return a display label for a study method key."""
    if labels and key in labels:
        return labels[key]
    return (
        key.replace("mmff_high_dielectric", "MMFF94s ε=78.4")
        .replace("mmff94s_water", "MMFF94s ε=78.4")
        .replace("mmff_vacuum", "MMFF94s ε=1.0")
        .replace("mmff94s_vacuum", "MMFF94s ε=1.0")
        .replace("uff", "UFF")
        .replace("etkdg", "ETKDG (unminimized)")
    )


def pareto_mask(
    frame: pd.DataFrame,
    *,
    score: str = "score",
    cost: str = "seconds",
) -> pd.Series:
    """Return a mask of non-dominated rows for a maximise-score/minimise-cost set."""
    if frame.empty:
        return pd.Series(dtype=bool)
    ordered = frame.sort_values(cost)
    best = float("-inf")
    keep = pd.Series(False, index=frame.index)
    for index, row in ordered.iterrows():
        value = float(row[score])
        if value > best + 1e-12:
            keep.loc[index] = True
            best = value
    return keep


def slope_r2(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return the ordinary least-squares slope and R² of ``y`` on ``x``."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if len(x_values) < 2 or np.ptp(x_values) == 0.0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residual = float(np.sum((y_values - predicted) ** 2))
    total = float(np.sum((y_values - y_values.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(slope), float(r2)


__all__ = [
    "ISOMER_BIN_EDGES",
    "ISOMER_BIN_LABELS",
    "bin_counts",
    "isomer_bin",
    "method_label",
    "pareto_mask",
    "slope_r2",
]
