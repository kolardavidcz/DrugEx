"""Public re-export surface for the thesis code.

Layout
------
``_david.code`` (this package)
    Shared chemistry and scoring layer: :mod:`chemistry`, :mod:`rocs`,
    :mod:`reference`, :mod:`prior_dataset`, :mod:`schema`, :mod:`utils`.
``_david.code.training``
    RL training pipeline: :mod:`config`, :mod:`environment`, :mod:`models`,
    :mod:`pipeline`, :mod:`loss_criterion`.
``_david.code.analysis``
    Post-hoc analysis and reporting: :mod:`dashboard`, :mod:`visualization`,
    :mod:`standard_eval`, :mod:`rocs_scale`, :mod:`output_organization`.

Every name below stays importable from ``_david.code`` even though it is defined
in a subpackage, so existing call sites keep working unchanged.
"""

from __future__ import annotations

from _david.code.chemistry import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_REFERENCE_SEED,
    enable_double_precision_mol_pickling,
    verify_disjoint_conformer_seeds,
)
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
from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.training.environment import (
    HarringtonDesirabilityReward,
    PrepareMolecule,
    create_environment,
    create_rocs_scorer,
    resolve_reward_scheme,
)
from _david.code.training.models import DecayingSequenceExplorer, SequenceRNNOptimizer
from _david.code.training.pipeline import MoleculePipeline
from _david.code.analysis.visualization import (
    AlignedDepicter,
    ResultAnalyzer,
    bootstrap_ci,
    build_diagnostic_scorecard,
    render_before_after_distributions,
    render_lead_structures_gallery,
    render_multiobjective_tradeoffs,
    render_objective_conflict_correlations,
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
    "render_before_after_distributions",
    "render_multiobjective_tradeoffs",
    "render_training_trajectories",
    "render_unfavorable_features_diagnostics",
    "render_lead_structures_gallery",
    "render_objective_conflict_correlations",
    "render_scaffold_diversity_analysis",
    "unscale_to_physical_metrics",
    "bootstrap_ci",
    "HardwareManager",
    "get_gpu_telemetry",
    "MoleculePipeline",
    "std",
]
