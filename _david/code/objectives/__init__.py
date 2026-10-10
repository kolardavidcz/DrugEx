"""Objectives, scoring, dynamic envelopes, and anti-cheating rewards for LBDD."""
from __future__ import annotations

from _david.code.objectives.builder import (
    DEFAULT_2D_THRESHOLD,
    DEFAULT_PROP_THRESHOLD,
    DEFAULT_ROCS_THRESHOLD,
    DEFAULT_SA_THRESHOLD,
    SA_CEILING,
    SA_FLOOR,
    ObjectiveBuilder,
    ObjectiveConfig,
    ObjectivePreset,
    RewardSchemeType,
    SAMode,
    SimilarityType,
)
from _david.code.objectives.conformer import (
    MY_MOLECULE,
    ROCS_COLOR_FF,
    ROCS_MAX_CONFORMERS,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    QueryConformerPreparer,
    create_rocs_scorer,
    get_safe_parallel_jobs,
    get_seed_hash,
)
from _david.code.objectives.envelopes import (
    DynamicPropertyEnvelope,
    compute_dynamic_envelope,
    create_dynamic_property_scorers,
)
from _david.code.objectives.rewards import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    GeometricMeanReward,
)

# Standard alias aliases
ROCS_THRESHOLD = DEFAULT_ROCS_THRESHOLD
PROP_THRESHOLD = DEFAULT_PROP_THRESHOLD
SA_THRESHOLD = DEFAULT_SA_THRESHOLD

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
    "DynamicPropertyEnvelope",
    "compute_dynamic_envelope",
    "create_dynamic_property_scorers",
    "QueryConformerPreparer",
    "create_rocs_scorer",
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
    "DEFAULT_2D_THRESHOLD",
    "DEFAULT_PROP_THRESHOLD",
    "DEFAULT_SA_THRESHOLD",
    "ROCS_THRESHOLD",
    "PROP_THRESHOLD",
    "SA_THRESHOLD",
    "ObjectivePreset",
    "ObjectiveBuilder",
    "ObjectiveConfig",
    "SimilarityType",
    "RewardSchemeType",
    "SAMode",
]
