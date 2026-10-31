from __future__ import annotations

from typing import Any

from _david.code.chemistry import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_REFERENCE_SEED,
    enable_double_precision_mol_pickling,
    verify_disjoint_conformer_seeds,
)
from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.environment import (
    HarringtonDesirabilityReward,
    PrepareMolecule,
    create_environment,
    create_rocs_scorer,
    resolve_reward_scheme,
)
from _david.code.models import DecayingSequenceExplorer, SequenceRNNOptimizer
from _david.code.pipeline import MoleculePipeline
from _david.code.reference import (
    balanced_reference_order,
    diverse_order,
    select_conformers,
)
from _david.code.rocs import NAMED_OPTIMIZATION_METHODS, WholeCallScorer
from _david.code.schema import std
from _david.code.utils import (
    HardwareManager,
    get_gpu_telemetry,
)
from _david.code.visualization import (
    AlignedDepicter,
    ResultAnalyzer,
    bootstrap_ci,
    build_diagnostic_scorecard,
    render_before_after_distributions,
    render_lead_structures_gallery,
    render_multiobjective_tradeoffs,
    render_reference_ensemble,
    render_relative_property_shifts_tornado,
    render_rocs_shape_vs_color_decomposition,
    render_scaffold_diversity_analysis,
    render_training_trajectories,
    render_unfavorable_features_diagnostics,
    unscale_to_physical_metrics,
)

__all__ = [
    "DEFAULT_CANDIDATE_SEED",
    "DEFAULT_MOLECULE",
    "DEFAULT_REFERENCE_SEED",
    "NAMED_OPTIMIZATION_METHODS",
    "PipelineConfig",
    "RunPaths",
    "PrepareMolecule",
    "WholeCallScorer",
    "balanced_reference_order",
    "create_environment",
    "create_rocs_scorer",
    "diverse_order",
    "enable_double_precision_mol_pickling",
    "resolve_reward_scheme",
    "select_conformers",
    "verify_disjoint_conformer_seeds",
    "HarringtonDesirabilityReward",
    "SequenceRNNOptimizer",
    "DecayingSequenceExplorer",
    "ResultAnalyzer",
    "AlignedDepicter",
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
    "unscale_to_physical_metrics",
    "bootstrap_ci",
    "HardwareManager",
    "get_gpu_telemetry",
    "MoleculePipeline",
    "GridJobSpec",
    "MetaCentrumGridManager",
    "generate_8arm_grid",
    "std",
]


def __getattr__(name: str) -> Any:
    if name in ("GridJobSpec", "MetaCentrumGridManager", "generate_8arm_grid"):
        from _david.code.metacentrum import grid

        return getattr(grid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
