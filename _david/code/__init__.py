"""Modular core implementation package for DrugEx LBDD pipeline."""
from __future__ import annotations

from _david.code.anti_cheating_rewards import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    GeometricMeanReward,
)
from _david.code.fallback_monitor import FallbackEvent, FallbackMonitor
from _david.code.GraphTransformer import GraphTransformerOptimizer
from _david.code.pipeline import MoleculePipeline
from _david.code.prepare_query_conformer import QueryConformerPreparer, get_safe_parallel_jobs
from _david.code.result_analyzer import ResultAnalyzer

# Backward-compatibility aliases for legacy experiments
try:
    from _david.code.legacy.scaffold_funnel import ScaffoldFunnel
    from _david.code.legacy.SequenceRNN import SequenceRNNOptimizer
    from _david.code.legacy.usrcat_scorer import USRCATScorer
except (ImportError, ModuleNotFoundError):
    ScaffoldFunnel = None  # type: ignore[assignment, misc]
    SequenceRNNOptimizer = None  # type: ignore[assignment, misc]
    USRCATScorer = None  # type: ignore[assignment, misc]

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "FallbackEvent",
    "FallbackMonitor",
    "GeometricMeanReward",
    "GraphTransformerOptimizer",
    "MoleculePipeline",
    "QueryConformerPreparer",
    "ResultAnalyzer",
    "ScaffoldFunnel",
    "SequenceRNNOptimizer",
    "USRCATScorer",
    "get_safe_parallel_jobs",
]
