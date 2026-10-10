"""Backward-compatibility forwarding module for Anti-Cheating MORL Rewards.

Primary implementation moved to _david.code.objectives.rewards.
"""
from __future__ import annotations

from _david.code.objectives.rewards import (
    AntiCollapseSoftGateReward,
    AugmentedChebyshevReward,
    GeometricMeanReward,
)

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
]
