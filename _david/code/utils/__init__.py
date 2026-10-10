"""Hardware, fallback monitoring, and telemetry utilities."""
from __future__ import annotations

from _david.code.utils.fallback_monitor import FallbackMonitor
from _david.code.utils.gpu_manager import ExclusiveGPUManager
from _david.code.utils.monitor import (
    PeakRecoveryMonitor,
    display_dashboard,
    find_latest_run_dir,
    format_duration,
)
from _david.code.utils.telemetry import get_gpu_telemetry

__all__: list[str] = [
    "FallbackMonitor",
    "ExclusiveGPUManager",
    "PeakRecoveryMonitor",
    "display_dashboard",
    "find_latest_run_dir",
    "format_duration",
    "get_gpu_telemetry",
]
