"""Backward-compatibility forwarding module for Query Conformer Preparation.

Primary implementation moved to _david.code.objectives.conformer.
"""
from __future__ import annotations

from _david.code.objectives.conformer import (
    MY_MOLECULE,
    ROCS_COLOR_FF,
    ROCS_MAX_CONFORMERS,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    QueryConformerPreparer,
    create_rocs_scorer,
    get_safe_parallel_jobs,
    get_seed_hash,
)

__all__: list[str] = [
    "QueryConformerPreparer",
    "create_rocs_scorer",
    "get_safe_parallel_jobs",
    "get_seed_hash",
    "MY_MOLECULE",
    "ROCS_MAX_CONFORMERS",
    "ROCS_MAX_HEAVY_ATOMS",
    "ROCS_MAX_ROTATABLE_BONDS",
    "ROCS_TIMEOUT",
    "ROCS_SCORE_TYPE",
    "ROCS_COLOR_FF",
]
