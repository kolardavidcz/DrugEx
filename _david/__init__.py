"""DrugEx Bachelor Thesis LBDD Package."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _david.code import (
    DEFAULT_MOLECULE,
    AlignedDepicter,
    ExclusiveGPUManager,
    FallbackMonitor,
    GraphTransformerOptimizer,
    MoleculePipeline,
    ParetoFrontier,
    ResultAnalyzer,
    create_environment,
    create_rocs_scorer,
    extract_seed_core_scaffold,
    generate_reference_conformers,
    get_gpu_telemetry,
    get_safe_gpus,
    get_safe_parallel_jobs,
)

__all__ = [
    "DEFAULT_MOLECULE",
    "create_environment",
    "create_rocs_scorer",
    "generate_reference_conformers",
    "extract_seed_core_scaffold",
    "GraphTransformerOptimizer",
    "AlignedDepicter",
    "ParetoFrontier",
    "ResultAnalyzer",
    "ExclusiveGPUManager",
    "FallbackMonitor",
    "get_safe_gpus",
    "get_safe_parallel_jobs",
    "get_gpu_telemetry",
    "MoleculePipeline",
]
