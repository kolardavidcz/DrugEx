from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _david.code import (
    DEFAULT_MOLECULE,
    AlignedDepicter,
    DecayingSequenceExplorer,
    HardwareManager,
    HarringtonDesirabilityReward,
    MoleculePipeline,
    PipelineConfig,
    PrepareMolecule,
    ResultAnalyzer,
    RunPaths,
    SequenceRNNOptimizer,
    bootstrap_ci,
    build_diagnostic_scorecard,
    create_environment,
    create_rocs_scorer,
    get_gpu_telemetry,
    render_before_after_distributions,
    render_lead_structures_gallery,
    render_multiobjective_tradeoffs,
    render_reference_ensemble,
    render_relative_property_shifts_tornado,
    render_rocs_shape_vs_color_decomposition,
    render_scaffold_diversity_analysis,
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
    "AlignedDepicter",
    "HardwareManager",
    "get_gpu_telemetry",
    "MoleculePipeline",
    "GridJobSpec",
    "MetaCentrumGridManager",
    "generate_8arm_grid",
    "generate_4hypothesis_grid",
    "std",
    "unscale_to_physical_metrics",
    "bootstrap_ci",
    "build_diagnostic_scorecard",
    "render_reference_ensemble",
    "render_before_after_distributions",
    "render_multiobjective_tradeoffs",
    "render_training_trajectories",
    "render_unfavorable_features_diagnostics",
    "render_lead_structures_gallery",
    "render_relative_property_shifts_tornado",
    "render_scaffold_diversity_analysis",
    "render_rocs_shape_vs_color_decomposition",
]


def __getattr__(name: str) -> object:
    if name in (
        "GridJobSpec",
        "MetaCentrumGridManager",
        "generate_8arm_grid",
        "generate_4hypothesis_grid",
    ):
        from _david.code.metacentrum import grid

        return getattr(grid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
