"""Backward-compatibility forwarding module for FallbackMonitor.

Primary implementation moved to _david.code.utils.fallback_monitor.
"""
from __future__ import annotations

from _david.code.utils.fallback_monitor import FallbackEvent, FallbackMonitor

__all__: list[str] = ["FallbackEvent", "FallbackMonitor"]
