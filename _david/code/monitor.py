"""Backward-compatibility forwarding module for Pipeline Telemetry & Dashboard.

Primary implementation moved to _david.code.utils.monitor.
"""
from __future__ import annotations

from _david.code.utils.monitor import (
    PeakRecoveryMonitor,
    display_dashboard,
    find_latest_run_dir,
    format_duration,
    main,
)

__all__: list[str] = [
    "PeakRecoveryMonitor",
    "display_dashboard",
    "find_latest_run_dir",
    "format_duration",
    "main",
]
