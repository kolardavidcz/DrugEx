from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch

SAFE_MAX_CPU_CORES: int = 24


def get_safe_parallel_jobs(cap: int = SAFE_MAX_CPU_CORES) -> int:
    total = os.cpu_count() or 1
    return max(1, min(cap, total // 2 if total > 2 else total))


@dataclass
class FallbackEvent:
    timestamp: str
    fallback_type: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class FallbackMonitor:

    _events: list[FallbackEvent] = []

    @classmethod
    def record_fallback(cls, fallback_type: str, message: str, context: dict[str, Any] | None = None) -> None:
        event = FallbackEvent(
            timestamp=datetime.now().isoformat(),
            fallback_type=fallback_type,
            message=message,
            context=context or {},
        )
        cls._events.append(event)
        print(f"\n[FALLBACK MONITOR ALERT] {fallback_type}: {message}", file=sys.stderr)

    @classmethod
    def get_events(cls) -> list[FallbackEvent]:
        return list(cls._events)

    @classmethod
    def clear(cls) -> None:
        cls._events.clear()


class ExclusiveGPUManager:

    @staticmethod
    def get_device_vram(device_idx: int) -> tuple[float, float, float]:
        if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
            return 0.0, 0.0, 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free = free_bytes / (1024 ** 2)
            total = total_bytes / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        except Exception:
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            free = total - reserved
        return allocated, free, total

    @classmethod
    def select_best_gpu(cls, min_free_mb: float = 4000.0) -> int:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return -1
        best_idx, max_free = -1, -1.0
        for i in range(torch.cuda.device_count()):
            _, free, _ = cls.get_device_vram(i)
            if free > max_free:
                max_free = free
                best_idx = i
        return best_idx if max_free >= min_free_mb else 0


def get_safe_gpus(preferred_gpu: int | str | None = None) -> tuple[int, ...] | tuple[()]:
    if not torch.cuda.is_available():
        return ()
    if preferred_gpu is None or preferred_gpu == "auto":
        best = ExclusiveGPUManager.select_best_gpu()
        return (best,) if best >= 0 else ()
    if isinstance(preferred_gpu, int):
        return (preferred_gpu,) if preferred_gpu < torch.cuda.device_count() else (0,)
    if isinstance(preferred_gpu, str) and preferred_gpu.isdigit():
        idx = int(preferred_gpu)
        return (idx,) if idx < torch.cuda.device_count() else (0,)
    return (0,)


def get_gpu_telemetry() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0, "devices": []}
    devices = []
    for i in range(torch.cuda.device_count()):
        alloc, free, total = ExclusiveGPUManager.get_device_vram(i)
        devices.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "allocated_mb": round(alloc, 2),
            "free_memory_mb": round(free, 2),
            "total_memory_mb": round(total, 2),
        })
    return {"cuda_available": True, "device_count": len(devices), "devices": devices}
