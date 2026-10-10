"""Backward-compatibility forwarding module for ResultAnalyzer.

Primary implementation moved to _david.code.analysis.result_analyzer.
"""
from __future__ import annotations

from _david.code.analysis.result_analyzer import (
    EGCG_SMILES,
    MY_MOLECULE,
    ResultAnalyzer,
    calculate_sa_score,
)

__all__: list[str] = [
    "ResultAnalyzer",
    "MY_MOLECULE",
    "EGCG_SMILES",
    "calculate_sa_score",
]
