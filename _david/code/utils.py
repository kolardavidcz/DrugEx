from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class FallbackEvent:
    """Record of an automated architectural fallback event for diagnostic tracking."""

    timestamp: str
    fallback_type: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class FallbackMonitor:
    """Thread-safe diagnostic monitor tracking graceful architectural fallbacks."""

    _events: list[FallbackEvent] = []

    @classmethod
    def record_fallback(
        cls,
        fallback_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an architectural fallback event and emit alert to stderr."""
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
        """Return shallow copy of recorded fallback events."""
        return list(cls._events)


class HardwareManager:
    """Central dynamic hardware solver for machine-agnostic compute resource allocation."""

    def __init__(self, cpu_ratio: float = 0.60) -> None:
        self.cpu_ratio = cpu_ratio

    @classmethod
    def get_safe_parallel_jobs(cls, ratio: float = 0.60, cap: int | None = None) -> int:
        """Calculate parallel worker allocation dynamically based on host logical core capacity."""
        total = os.cpu_count() or 1
        jobs = max(1, int(total * ratio))
        return max(1, min(cap, jobs)) if cap is not None else jobs

    @staticmethod
    def get_device_vram(device_idx: int) -> tuple[float, float, float]:
        """Query allocated, free, and total VRAM in megabytes (MB) for a specific GPU device."""
        if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
            return 0.0, 0.0, 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free = free_bytes / (1024 ** 2)
            total = total_bytes / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        except (RuntimeError, AttributeError):
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            free = max(0.0, total - reserved)
        return allocated, free, total

    #* select_best_gpu and get_safe_gpus could be programed much more elegantly
    @classmethod
    def select_best_gpu(cls, min_free_mb: float = 4000.0) -> int:
        """Select healthy GPU index with highest free VRAM, or return -1 for CPU fallback."""
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return -1
        best_idx, max_free = -1, -1.0
        for i in range(torch.cuda.device_count()):
            _, free, _ = cls.get_device_vram(i)
            if free > max_free:
                max_free = free
                best_idx = i
        return best_idx if max_free >= min_free_mb else -1

    @classmethod
    def get_safe_gpus(
        cls, preferred_gpu: int | str | None = "auto"
    ) -> tuple[int, ...] | tuple[()]:
        """Resolve single GPU device tuple or empty tuple for CPU execution."""
        if not torch.cuda.is_available() or preferred_gpu in ("cpu", "none", None, -1):
            return ()
        if preferred_gpu == "auto":
            best = cls.select_best_gpu()
            return (best,) if best >= 0 else ()
        if isinstance(preferred_gpu, int) and 0 <= preferred_gpu < torch.cuda.device_count():
            return (preferred_gpu,)
        if isinstance(preferred_gpu, str) and preferred_gpu.strip().isdigit():
            idx = int(preferred_gpu.strip())
            return (idx,) if 0 <= idx < torch.cuda.device_count() else ()
        return ()

    @classmethod
    def get_recommended_batch_size(cls, n_samples: int = 100) -> int:
        """Determine power-of-2 batch size aligned with VRAM and clamped to sample pool size."""
        best_gpu = cls.select_best_gpu()
        if best_gpu == -1:
            batch_size = 32
        else:
            _, free_mb, _ = cls.get_device_vram(best_gpu)
            if free_mb >= 16000.0:
                batch_size = 256
            elif free_mb >= 10000.0:
                batch_size = 128
            elif free_mb >= 4000.0:
                batch_size = 64
            else:
                batch_size = 32
        return max(1, min(batch_size, n_samples))

    @classmethod
    def init_runtime_environment(cls, ratio: float = 0.60) -> dict[str, str]:
        """Sole authority for initializing multiprocessing and scoring environment."""
        n_jobs = str(cls.get_safe_parallel_jobs(ratio=ratio))
        os.environ["DRUGEX_SCORE_NJOBS"] = n_jobs
        os.environ["DRUGEX_MP_CONTEXT"] = "forkserver"
        return {"DRUGEX_SCORE_NJOBS": n_jobs, "DRUGEX_MP_CONTEXT": "forkserver"}


def get_gpu_telemetry() -> dict[str, Any]:
    """Retrieve full system GPU telemetry dictionary for pipeline logging."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0, "devices": []}
    devices = []
    for i in range(torch.cuda.device_count()):
        alloc, free, total = HardwareManager.get_device_vram(i)
        devices.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "allocated_mb": round(alloc, 2),
            "free_memory_mb": round(free, 2),
            "total_memory_mb": round(total, 2),
        })
    return {"cuda_available": True, "device_count": len(devices), "devices": devices}
