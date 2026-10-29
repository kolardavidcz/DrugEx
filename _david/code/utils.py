from __future__ import annotations

import os
from typing import Any

import torch


class HardwareManager:
    """Central dynamic hardware solver for machine-agnostic compute resource allocation."""

    @classmethod
    def get_safe_parallel_jobs(cls, ratio: float = 0.90, cap: int | None = None) -> int:
        """Calculate parallel worker allocation dynamically based on host logical core capacity."""
        if not 0 < ratio <= 1:
            raise ValueError("CPU ratio must be in (0, 1]")
        total = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        # A scheduler may allocate fewer CPUs than the host's affinity exposes.
        nodefile = os.environ.get("PBS_NODEFILE")
        if os.environ.get("PBS_JOBID") and nodefile:
            with open(nodefile) as handle:
                allocated = sum(bool(line.strip()) for line in handle)
            total = min(total, max(1, allocated))
        jobs = max(1, int(total * ratio))
        return max(1, min(cap, jobs)) if cap is not None else jobs

    @staticmethod
    def get_device_vram(device_idx: int) -> tuple[float, float, float]:
        """Query allocated, free, and total VRAM in megabytes (MB) for a specific GPU device."""
        if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
            return 0.0, 0.0, 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free = free_bytes / (1024**2)
            total = total_bytes / (1024**2)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**2)
        except (RuntimeError, AttributeError):
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (
                1024**2
            )
            free = max(0.0, total - reserved)
        return allocated, free, total

    @classmethod
    def get_safe_gpus(
        cls,
        preferred_gpu: int | str | None = "auto",
        min_free_mb: float = 4000.0,
    ) -> tuple[int, ...]:
        """Resolve single GPU device tuple or empty tuple for CPU execution based on VRAM availability."""
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() == 0
            or preferred_gpu in ("cpu", "none", None, -1)
        ):
            return ()
        if (
            isinstance(preferred_gpu, int)
            and 0 <= preferred_gpu < torch.cuda.device_count()
        ):
            return (preferred_gpu,)
        if isinstance(preferred_gpu, str) and preferred_gpu.strip().isdigit():
            idx = int(preferred_gpu.strip())
            return (idx,) if 0 <= idx < torch.cuda.device_count() else ()
        if preferred_gpu == "auto":
            free_vrams = [
                cls.get_device_vram(i)[1] for i in range(torch.cuda.device_count())
            ]
            best_idx = int(max(range(len(free_vrams)), key=lambda i: free_vrams[i]))
            return (best_idx,) if free_vrams[best_idx] >= min_free_mb else ()
        return ()

    @classmethod
    def get_device_and_gpus(
        cls,
        preferred_gpu: int | str | None = "auto",
        min_free_mb: float = 4000.0,
    ) -> tuple[torch.device, tuple[int, ...]]:
        """Resolve torch.device and GPU tuple for model execution."""
        gpus = cls.get_safe_gpus(preferred_gpu=preferred_gpu, min_free_mb=min_free_mb)
        if not gpus:
            return torch.device("cpu"), ()
        return torch.device(f"cuda:{gpus[0]}"), gpus

    @classmethod
    def init_runtime_environment(cls, ratio: float = 0.90) -> dict[str, str]:
        """Sole authority for initializing multiprocessing and scoring environment."""
        n_jobs = str(cls.get_safe_parallel_jobs(ratio=ratio))
        os.environ["DRUGEX_SCORE_NJOBS"] = n_jobs
        if "DRUGEX_MP_CONTEXT" not in os.environ:
            os.environ["DRUGEX_MP_CONTEXT"] = "fork"
        return {
            "DRUGEX_SCORE_NJOBS": n_jobs,
            "DRUGEX_MP_CONTEXT": os.environ["DRUGEX_MP_CONTEXT"],
        }


def get_gpu_telemetry() -> dict[str, Any]:
    """Retrieve full system GPU telemetry dictionary for pipeline logging."""
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0, "devices": []}
    devices = []
    for i in range(torch.cuda.device_count()):
        alloc, free, total = HardwareManager.get_device_vram(i)
        devices.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "allocated_mb": round(alloc, 2),
                "free_memory_mb": round(free, 2),
                "total_memory_mb": round(total, 2),
            }
        )
    return {"cuda_available": True, "device_count": len(devices), "devices": devices}
