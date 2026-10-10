"""Backward-compatibility forwarding module for Dynamic Property Envelopes.

Primary implementation moved to _david.code.objectives.envelopes.
"""
from __future__ import annotations

from _david.code.objectives.envelopes import (
    DynamicPropertyEnvelope,
    compute_dynamic_envelope,
    create_dynamic_property_scorers,
)

__all__: list[str] = [
    "DynamicPropertyEnvelope",
    "compute_dynamic_envelope",
    "create_dynamic_property_scorers",
]
