"""Backward-compatibility forwarding module for ExclusiveGPUManager.

Primary implementation moved to _david.code.utils.gpu_manager.
"""
from __future__ import annotations

from _david.code.utils.gpu_manager import ExclusiveGPUManager

__all__: list[str] = ["ExclusiveGPUManager"]
