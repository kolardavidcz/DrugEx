"""Canonical scientific taxonomies shared by studies and figures.

One taxonomy currently matters: how molecules are grouped by stereoisomer
ambiguity.  Two modules computing "the same" bins with different edges would make
two figures silently incomparable, so the definition lives here and everything
else imports it.

Bins are right-closed intervals.  Both boundary values used by the package are
counts of *stereoisomer identities reachable by enumeration*, which differ by
protocol:

``isomer_upper_bound``
    Identities reachable by enumerating the centres the prepared input leaves
    unassigned.  This is what a study actually samples, and what
    ``studies/measure.py`` records per cell.

``masked_upper_bound``
    Identities reachable if every centre were unassigned -- the worst case an
    RNN candidate presents.  Historical panels were enumerated under this masked
    protocol, so their figures stratify on it.

Either way the bin describes the ambiguity the scorer is exposed to; a molecule
with 64 reachable identities is hard because of that, not because a study chose
to sample 8 of them.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import pandas as pd

#: Upper bound of each stereoisomer bin; ``inf`` closes the last one.
ISOMER_BIN_EDGES: tuple[float, ...] = (2, 4, 8, 15, 32, math.inf)

#: Labels matching :data:`ISOMER_BIN_EDGES`.
ISOMER_BIN_LABELS: tuple[str, ...] = ("1–2", "3–4", "5–8", "9–15", "16–32", "33+")


def isomer_bin(values: pd.Series | Sequence[float]) -> pd.Series:
    """Assign true isomer counts to the canonical logarithmic bins.

    Bins are right-closed ``(0, 2], (2, 4], ...``; a value of zero is treated
    as missing rather than as a one-isomer molecule, so a broken input can
    never be silently counted in the easiest bin.
    """
    series = pd.Series(values, dtype="float64")
    edges = [0.0, *ISOMER_BIN_EDGES]
    return pd.cut(
        series,
        bins=edges,
        labels=list(ISOMER_BIN_LABELS),
        right=True,
        include_lowest=False,
    )


def bin_counts(values: pd.Series | Sequence[float]) -> pd.Series:
    """Count observations per canonical isomer bin, preserving empty bins."""
    return (
        isomer_bin(values)
        .value_counts()
        .reindex(list(ISOMER_BIN_LABELS), fill_value=0)
        .rename("molecules")
    )


__all__ = ["ISOMER_BIN_EDGES", "ISOMER_BIN_LABELS", "bin_counts", "isomer_bin"]
