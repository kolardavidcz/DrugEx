"""Backward-compatibility forwarding module for GraphTransformer Generative Optimizer.

Primary implementation moved to _david.code.models.graph_transformer.
"""
from __future__ import annotations

from _david.code.models.graph_transformer import (
    GraphTransformerOptimizer,
    get_safe_gpus,
    RL_BATCH_SIZE,
    RL_BETA,
    RL_EPOCHS,
    RL_EPSILON,
    RL_MIN_EPOCHS,
    RL_N_SAMPLES,
    RL_PATIENCE,
)
from _david.code.objectives import (
    MY_MOLECULE,
    ROCS_COLOR_FF,
    ROCS_MAX_CONFORMERS,
    ROCS_MAX_HEAVY_ATOMS,
    ROCS_MAX_ROTATABLE_BONDS,
    ROCS_SCORE_TYPE,
    ROCS_TIMEOUT,
    ROCS_THRESHOLD,
    PROP_THRESHOLD,
    SA_CEILING,
    SA_FLOOR,
    SA_THRESHOLD,
)

__all__: list[str] = [
    "GraphTransformerOptimizer",
    "get_safe_gpus",
    "MY_MOLECULE",
    "RL_BATCH_SIZE",
    "RL_BETA",
    "RL_EPOCHS",
    "RL_EPSILON",
    "RL_MIN_EPOCHS",
    "RL_N_SAMPLES",
    "RL_PATIENCE",
    "ROCS_COLOR_FF",
    "ROCS_MAX_CONFORMERS",
    "ROCS_MAX_HEAVY_ATOMS",
    "ROCS_MAX_ROTATABLE_BONDS",
    "ROCS_SCORE_TYPE",
    "ROCS_TIMEOUT",
    "ROCS_THRESHOLD",
    "PROP_THRESHOLD",
    "SA_CEILING",
    "SA_FLOOR",
    "SA_THRESHOLD",
]
