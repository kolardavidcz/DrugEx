"""DrugEx Bachelor Thesis LBDD Package.

Provides access to the modular codebase under `_david.code`,
the 1-click pipeline runner `_david.run`, and active module aliases.
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
)
from _david.code.anti_cheating_rewards import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    GeometricMeanReward,
)
from _david.code.dynamic_envelope import (
    DynamicPropertyEnvelope,
    compute_dynamic_envelope,
    create_dynamic_property_scorers,
)
from _david.code.fallback_monitor import FallbackEvent, FallbackMonitor
from _david.code.GraphTransformer import GraphTransformerOptimizer
from _david.code.pipeline import MoleculePipeline
from _david.code.prepare_query_conformer import QueryConformerPreparer, get_safe_parallel_jobs
from _david.code.result_analyzer import ResultAnalyzer

# Active module aliases in sys.modules for clean imports
sys.modules["_david.fallback_monitor"] = fallback_monitor
sys.modules["_david.anti_cheating_rewards"] = anti_cheating_rewards
sys.modules["_david.prepare_query_conformer"] = prepare_query_conformer
sys.modules["_david.GraphTransformer"] = GraphTransformer
sys.modules["_david.pipeline"] = pipeline
sys.modules["_david.result_analyzer"] = result_analyzer
sys.modules["_david.dynamic_envelope"] = dynamic_envelope

__all__ = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "DynamicPropertyEnvelope",
    "FallbackEvent",
    "FallbackMonitor",
    "GeometricMeanReward",
    "GraphTransformer",
    "GraphTransformerOptimizer",
    "MoleculePipeline",
    "QueryConformerPreparer",
    "ResultAnalyzer",
    "anti_cheating_rewards",
    "compute_dynamic_envelope",
    "create_dynamic_property_scorers",
    "dynamic_envelope",
    "fallback_monitor",
    "get_safe_parallel_jobs",
    "monitor",
    "pipeline",
    "prepare_query_conformer",
    "result_analyzer",
]
