"""GPU device and memory telemetry helpers."""
from __future__ import annotations

from typing import Any

# [PACKAGE: torch] WHY: Device memory and count inspection. WHAT: PyTorch deep learning framework. HOW: torch.cuda.
import torch


def get_gpu_telemetry() -> dict[str, Any]:
    """Gather current GPU device, allocation, and headroom metrics."""
    telemetry: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(i)
                free_mb = free_bytes / (1024 ** 2)
            except Exception:
                free_mb = total_mem - reserved
            telemetry["devices"].append({
                "index": i,
                "name": props.name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "max_allocated_mb": round(max_alloc, 2),
                "total_memory_mb": round(total_mem, 2),
                "free_memory_mb": round(free_mb, 2),
            })
    return telemetry
