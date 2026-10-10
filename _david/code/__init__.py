"""Modular core implementation package for DrugEx LBDD pipeline.

Organized into clean domain subpackages:
- objectives: Scorers, envelopes, feasibility floors, and anti-cheating rewards
- models: GraphTransformer generators, fragment conditioners, and RL explorers
- analysis: Result profiling, Pareto frontier ranking, and aligned depictions
- utils: GPU resource arbiters, fallback monitors, and live telemetry
- experiments: Research studies, hypothesis testing, and ablation suites
"""
from __future__ import annotations

from _david.code.analysis import (
    AlignedDepicter,
    ParetoFrontier,
    ResultAnalyzer,
    get_molecular_space_pipeline,
)
from _david.code.models import (
    GraphTransformerOptimizer,
    RL_BATCH_SIZE,
    RL_BETA,
    RL_EPOCHS,
    RL_EPSILON,
    RL_MIN_EPOCHS,
    RL_N_SAMPLES,
    RL_PATIENCE,
    get_safe_gpus,
)
from _david.code.objectives import (
    DEFAULT_PROP_THRESHOLD,
    DEFAULT_ROCS_THRESHOLD,
    DEFAULT_SA_THRESHOLD,
    MY_MOLECULE,
    ROCS_COLOR_FF,
    ROCS_MAX_CONFORMERS,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    SA_CEILING,
    SA_FLOOR,
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    DynamicPropertyEnvelope,
    GeometricMeanReward,
    ObjectiveBuilder,
    ObjectivePreset,
    QueryConformerPreparer,
    create_dynamic_property_scorers,
    create_rocs_scorer,
    get_safe_parallel_jobs,
    get_seed_hash,
)
from _david.code.pipeline import (
    DEFAULT_MOLECULE,
    MoleculePipeline,
)
from _david.code.utils import (
    ExclusiveGPUManager,
    FallbackMonitor,
    PeakRecoveryMonitor,
    display_dashboard,
    get_gpu_telemetry,
)

__all__: list[str] = [
    # Objectives
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
    "DynamicPropertyEnvelope",
    "create_dynamic_property_scorers",
    "QueryConformerPreparer",
    "create_rocs_scorer",
    "ObjectiveBuilder",
    "ObjectivePreset",
    "get_safe_parallel_jobs",
    "get_seed_hash",
    "MY_MOLECULE",
    "ROCS_SCORE_TYPE",
    "ROCS_COLOR_FF",
    "ROCS_MAX_CONFORMERS",
    "ROCS_MAX_HEAVY_ATOMS",
    "ROCS_MAX_ROTATABLE_BONDS",
    "ROCS_TIMEOUT",
    "SA_FLOOR",
    "SA_CEILING",
    "DEFAULT_ROCS_THRESHOLD",
    "DEFAULT_PROP_THRESHOLD",
    "DEFAULT_SA_THRESHOLD",
    # Models
    "GraphTransformerOptimizer",
    "get_safe_gpus",
    "RL_EPOCHS",
    "RL_EPSILON",
    "RL_BETA",
    "RL_N_SAMPLES",
    "RL_BATCH_SIZE",
    "RL_PATIENCE",
    "RL_MIN_EPOCHS",
    # Analysis
    "ResultAnalyzer",
    "ParetoFrontier",
    "AlignedDepicter",
    "get_molecular_space_pipeline",
    # Utils
    "FallbackMonitor",
    "ExclusiveGPUManager",
    "PeakRecoveryMonitor",
    "display_dashboard",
    "get_gpu_telemetry",
    # Pipeline
    "DEFAULT_MOLECULE",
    "MoleculePipeline",
]
