from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _david.code import (
    DEFAULT_MOLECULE,
    DecayingSequenceExplorer,
    FallbackMonitor,
    HardwareManager,
    HarringtonDesirabilityReward,
    MoleculePipeline,
    PipelineConfig,
    PrepareMolecule,
    ResultAnalyzer,
    RunPaths,
    SequenceRNNOptimizer,
    create_environment,
    create_rocs_scorer,
    get_gpu_telemetry,
    render_before_after_distributions,
    render_lead_structures_gallery,
    render_multiobjective_tradeoffs,
    render_reference_ensemble,
    render_training_trajectories,
    render_unfavorable_features_diagnostics,
    resolve_reward_scheme,
    std,
    unscale_to_physical_metrics,
)

__all__ = [
    "DEFAULT_MOLECULE",
    "PipelineConfig",
    "RunPaths",
    "PrepareMolecule",
    "create_environment",
    "create_rocs_scorer",
    "resolve_reward_scheme",
    "HarringtonDesirabilityReward",
    "SequenceRNNOptimizer",
    "DecayingSequenceExplorer",
    "ResultAnalyzer",
    "HardwareManager",
    "FallbackMonitor",
    "get_gpu_telemetry",
    "MoleculePipeline",
    "std",
    "unscale_to_physical_metrics",
    "render_reference_ensemble",
    "render_before_after_distributions",
    "render_multiobjective_tradeoffs",
    "render_training_trajectories",
    "render_unfavorable_features_diagnostics",
    "render_lead_structures_gallery",
]
