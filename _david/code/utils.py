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
        return list(cls._events)


class HardwareManager:
    """Central dynamic hardware solver for machine-agnostic compute resource allocation."""

    #* _cpu_ratio sould be implicit paramether not 
    _cpu_ratio: float = 0.60

    @classmethod
    def get_safe_parallel_jobs(cls, ratio: float | None = None, cap: int | None = None) -> int:
        """Calculate worker allocation dynamically based on host logical core capacity.

        Follows host allocation ratio (max(1, int(total_cores * ratio))), leaving sufficient
        system headroom for OS services, PyTorch background threads, and I/O tasks.
        Defaults to configured _cpu_ratio (0.60 standard, 0.80 overnight).
        Optionally clamped by an upper limit cap if provided.
        """
        effective_ratio = cls._cpu_ratio if ratio is None else ratio
        total = os.cpu_count() or 1
        jobs = max(1, int(total * effective_ratio))
        if cap is not None:
            return max(1, min(cap, jobs))
        return jobs

    @staticmethod
    def get_device_vram(device_idx: int) -> tuple[float, float, float]:
        """Query allocated, free, and total VRAM in megabytes (MB) for a specific GPU device.

        Returns (allocated_mb, free_mb, total_mb). If CUDA is unavailable or device_idx
        is out of bounds, returns (0.0, 0.0, 0.0).
        """
        if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
            return 0.0, 0.0, 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free = free_bytes / (1024 ** 2)
            total = total_bytes / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        except Exception: #* use more specific exeption
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            free = max(0.0, total - reserved)
        return allocated, free, total

    @classmethod
    def select_best_gpu(cls, min_free_mb: float = 4000.0) -> int:
        """Select healthy GPU index with highest free VRAM, or return -1 for CPU fallback.

        If CUDA is unavailable, device count is 0, or all GPUs have less than min_free_mb
        available VRAM, returns -1 to indicate that no GPU is safe to use.
        """
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return -1
        best_idx, max_free = -1, -1.0
        #* thee sould be 1) check witch one is more free, ad 2) then look with would fit better
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
        """Resolve GPU device tuple honoring user preference or dynamic auto-selection.

        Returns an empty tuple () when falling back to CPU execution, which DrugEx
        natively interprets as CPU-only execution. Supports 'all' for multi-GPU training.
        """
        if not torch.cuda.is_available():
            return ()
        #* in drugex nor our codebase isnt reason for using both gpus
        #* whit this in mind simplify the coude in this and all realted files (full featured refactor)
        #* ps this code is very ugly
        if preferred_gpu == "all":
            return tuple(range(torch.cuda.device_count()))
        if preferred_gpu is None or preferred_gpu == "auto":
            best = cls.select_best_gpu()
            return (best,) if best >= 0 else ()
        if isinstance(preferred_gpu, int):
            return (
                (preferred_gpu,)
                if 0 <= preferred_gpu < torch.cuda.device_count()
                else ()
            )
        if isinstance(preferred_gpu, str) and preferred_gpu.strip().isdigit():
            idx = int(preferred_gpu.strip())
            return (idx,) if 0 <= idx < torch.cuda.device_count() else ()
        return ()

    @classmethod
    def get_recommended_batch_size(cls, n_samples: int = 100) -> int:
        """Determine power-of-2 batch size aligned with NVIDIA Tensor Core warp execution.

        Batch sizing rationale:
        - >= 12 GB free VRAM: 128 (optimal Tensor Core throughput for transformer layers)
        - >= 4 GB free VRAM: 64 (balanced memory/compute ratio)
        - CPU or low VRAM (< 4 GB): 32 (standard baseline)
        Clamped strictly to max(1, min(batch_size, n_samples)) to prevent out-of-bounds padding.
        """
        best_gpu = cls.select_best_gpu()
        #* i replaced best_gpu to explicit -1 for better readability
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
        #* not sure if we need max (1,..) if user sould put n_samples=0 then it wouldnt work (but in case of variable change, this could be critical bug\)
        return max(1, min(batch_size, n_samples))

    @classmethod
    def init_runtime_environment(cls, ratio: float | None = None) -> dict[str, str]:
        """Sole authority for initializing multiprocessing and scoring environment."""
        if ratio is not None:
            cls._cpu_ratio = ratio
        n_jobs = str(cls.get_safe_parallel_jobs())
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
