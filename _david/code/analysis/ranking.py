"""Objective-aligned post-hoc lead ranking shared by figures and the dashboard.

The RL run optimises a specific objective vector. A lead table must rank with that
same vector, otherwise the "top" molecules are the ones that maximised an objective
the run was told to ignore. The shared helper lives here rather than in
``visualization`` so the HTML dashboard can use it without importing the plotting
stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _david.code.schema import std

#: Synthetic-accessibility penalty applied when ranking leads after the run.
LEAD_SA_PENALTY: float = 0.10


def lead_ranking_weights(run_config: dict | None) -> tuple[float, float, float]:
    """Return ``(rocs, qed, sa)`` weights for post-hoc lead prioritisation.

    A lead table must not reward an objective the RL run was told to ignore.
    ``include_qed=False`` (the EGCG default) means QED was never in the reward, and
    EGCG's own QED of 0.21 makes it an anti-correlated proxy for the polyphenol
    scaffold the run is trying to reproduce: on the ``ctrl_seed42`` arm a
    ``ROCS + QED - 0.1*SA`` index is dominated by QED (Spearman 0.90 with QED versus
    0.36 with ROCS) and its top-100 shares only 9 molecules with the top-100 by ROCS.
    With QED off the index reduces to ``ROCS - 0.10 * Raw_SA``, i.e. the training
    objective plus a mild synthetic-accessibility tie-breaker.
    """
    include_qed = bool((run_config or {}).get("include_qed", False))
    return (1.0, 1.0 if include_qed else 0.0, -LEAD_SA_PENALTY)


def lead_rank_score(frame: pd.DataFrame, run_config: dict | None = None) -> pd.Series:
    """Score a molecule table with the objective-aligned lead ranking index.

    Missing columns degrade to the defaults the result analyzer already used: absent
    QED contributes zero, absent physical SA falls back to the clipped ``SA`` channel
    and finally to the 10.0 ceiling so an unparsable structure sorts last.
    """

    def column(*names: str, default: float) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(default, index=frame.index, dtype=float)

    w_rocs, w_qed, w_sa = lead_ranking_weights(run_config)
    rocs = column(std.ROCS, default=np.nan)
    qed = column(std.QED, default=0.0)
    sa = column("Raw_SA", "RawSA", std.SA, default=np.nan)
    if sa.isna().all():
        sa = pd.Series(10.0, index=frame.index, dtype=float)
    return w_rocs * rocs + w_qed * qed.fillna(0.0) + w_sa * sa.fillna(10.0)
