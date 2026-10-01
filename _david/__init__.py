"""DrugEx Bachelor Thesis LBDD Package.

Provides access to the modular codebase under `_david.code`,
the 1-click pipeline runner `_david.run`, and backwards-compatible module aliases.
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import modular implementation classes from _david.code
from _david.code.fallback_monitor import FallbackEvent, FallbackMonitor
from _david.code.anti_cheating_rewards import AugmentedChebyshevReward, GeometricMeanReward
from _david.code.usrcat_scorer import USRCATScorer
from _david.code.prepare_query_conformer import QueryConformerPreparer
from _david.code.scaffold_funnel import ScaffoldFunnel
from _david.code.SequenceRNN import SequenceRNNOptimizer
from _david.code.GraphTransformer import GraphTransformerOptimizer
from _david.code.pipeline import MoleculePipeline
from _david.code.thesis_benchmark import ThesisBenchmark

# Import submodules from code
from _david.code import (
    anti_cheating_rewards,
    fallback_monitor,
    GraphTransformer,
    pipeline,
    prepare_query_conformer,
    scaffold_funnel,
    SequenceRNN,
    thesis_benchmark,
    usrcat_scorer,
    visualize_molecular_space,
)

# Backwards compatibility module aliases in sys.modules
sys.modules["_david.fallback_monitor"] = fallback_monitor
sys.modules["_david.anti_cheating_rewards"] = anti_cheating_rewards
sys.modules["_david.usrcat_scorer"] = usrcat_scorer
sys.modules["_david.prepare_query_conformer"] = prepare_query_conformer
sys.modules["_david.scaffold_funnel"] = scaffold_funnel
sys.modules["_david.SequenceRNN"] = SequenceRNN
sys.modules["_david.GraphTransformer"] = GraphTransformer
sys.modules["_david.pipeline"] = pipeline
sys.modules["_david.thesis_benchmark"] = thesis_benchmark
sys.modules["_david.visualize_molecular_space"] = visualize_molecular_space

__all__ = [
    "FallbackEvent",
    "FallbackMonitor",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
    "USRCATScorer",
    "QueryConformerPreparer",
    "ScaffoldFunnel",
    "SequenceRNNOptimizer",
    "GraphTransformerOptimizer",
    "MoleculePipeline",
    "ThesisBenchmark",
    "fallback_monitor",
    "anti_cheating_rewards",
    "usrcat_scorer",
    "prepare_query_conformer",
    "scaffold_funnel",
    "SequenceRNN",
    "GraphTransformer",
    "pipeline",
    "thesis_benchmark",
    "visualize_molecular_space",
]
