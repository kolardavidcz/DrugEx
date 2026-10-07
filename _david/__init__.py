"""DrugEx Bachelor Thesis LBDD Package.

Provides access to the modular codebase under `_david.code`,
the 1-click pipeline runner `_david.run`, and backwards-compatible module aliases.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Active submodules and classes from code
from _david.code import (
    GraphTransformer,
    anti_cheating_rewards,
    dynamic_envelope,
    fallback_monitor,
    monitor,
    pipeline,
    prepare_query_conformer,
    result_analyzer,
    visualize_molecular_space,
)
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

# Legacy backward-compatibility module imports
try:
    from _david.code.legacy import SequenceRNN, scaffold_funnel, usrcat_scorer
    from _david.code.legacy.scaffold_funnel import ScaffoldFunnel
    from _david.code.legacy.SequenceRNN import SequenceRNNOptimizer
    from _david.code.legacy.usrcat_scorer import USRCATScorer
except (ImportError, ModuleNotFoundError):
    SequenceRNN = None  # type: ignore[assignment]
    scaffold_funnel = None  # type: ignore[assignment]
    usrcat_scorer = None  # type: ignore[assignment]
    ScaffoldFunnel = None  # type: ignore[assignment, misc]
    SequenceRNNOptimizer = None  # type: ignore[assignment, misc]
    USRCATScorer = None  # type: ignore[assignment, misc]

# Backwards compatibility module aliases in sys.modules
sys.modules["_david.fallback_monitor"] = fallback_monitor
sys.modules["_david.anti_cheating_rewards"] = anti_cheating_rewards
sys.modules["_david.prepare_query_conformer"] = prepare_query_conformer
sys.modules["_david.GraphTransformer"] = GraphTransformer
sys.modules["_david.pipeline"] = pipeline
sys.modules["_david.result_analyzer"] = result_analyzer
sys.modules["_david.visualize_molecular_space"] = visualize_molecular_space

if usrcat_scorer is not None:
    sys.modules["_david.usrcat_scorer"] = usrcat_scorer
    sys.modules["_david.code.usrcat_scorer"] = usrcat_scorer
if scaffold_funnel is not None:
    sys.modules["_david.scaffold_funnel"] = scaffold_funnel
    sys.modules["_david.code.scaffold_funnel"] = scaffold_funnel
if SequenceRNN is not None:
    sys.modules["_david.SequenceRNN"] = SequenceRNN
    sys.modules["_david.code.SequenceRNN"] = SequenceRNN

__all__ = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "FallbackEvent",
    "FallbackMonitor",
    "GeometricMeanReward",
    "GraphTransformer",
    "GraphTransformerOptimizer",
    "MoleculePipeline",
    "QueryConformerPreparer",
    "ResultAnalyzer",
    "ScaffoldFunnel",
    "SequenceRNN",
    "SequenceRNNOptimizer",
    "USRCATScorer",
    "anti_cheating_rewards",
    "dynamic_envelope",
    "fallback_monitor",
    "get_safe_parallel_jobs",
    "monitor",
    "pipeline",
    "prepare_query_conformer",
    "result_analyzer",
    "scaffold_funnel",
    "usrcat_scorer",
    "visualize_molecular_space",
]
