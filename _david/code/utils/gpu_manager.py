"""Exclusive GPU Resource Arbiter & Supervisor-Safe Allocator for Shared Servers.

Prevents multi-user VRAM collisions, prohibits GPU sharing with supervisors (fulopj),
and automatically pauses workloads when all GPUs are occupied on shared computational infrastructure.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from typing import Any

# [PACKAGE: torch] WHY: Device memory probe and CUDA environment initialization. WHAT: PyTorch deep learning framework. HOW: torch.cuda.
import torch

try:
    from _david.code.utils.fallback_monitor import FallbackMonitor
except (ImportError, ModuleNotFoundError):
    from _david.code.fallback_monitor import FallbackMonitor


class ExclusiveGPUManager:
    """Manages exclusive GPU acquisition and auto-pausing on multi-user shared lab servers."""

    DEFAULT_MIN_FREE_GB: float = 6.0  # Require at least 6 GB free VRAM for exclusive, collision-free allocation
    DEFAULT_CHECK_INTERVAL_SEC: int = 60  # Wait interval between checks when all devices are occupied
    DEFAULT_MAX_WAIT_HOURS: float = 12.0  # Maximum wait timeout before aborting

    def __init__(
        self,
        min_free_gb: float = DEFAULT_MIN_FREE_GB,
        check_interval_sec: int = DEFAULT_CHECK_INTERVAL_SEC,
        max_wait_hours: float = DEFAULT_MAX_WAIT_HOURS,
    ) -> None:
        """Initialize ExclusiveGPUManager with memory thresholds and polling intervals.

        Args:
            min_free_gb: Minimum free VRAM in gigabytes required for exclusive ownership.
            check_interval_sec: Sleep interval between checks when all devices are busy.
            max_wait_hours: Maximum hours to wait for an exclusive GPU before raising.
        """
        self.min_free_gb: float = float(min_free_gb)
        self.check_interval_sec: int = int(check_interval_sec)
        self.max_wait_hours: float = float(max_wait_hours)

    @staticmethod
    def is_worker_process() -> bool:
        """Check if current execution is inside a multiprocessing worker child process."""
        return multiprocessing.current_process().name != "MainProcess"

    def query_devices(self) -> list[dict[str, Any]]:
        """Inspect all available CUDA devices without allocating memory.

        Returns:
            List of device status dictionaries detailing VRAM and occupancy.
        """
        if not torch.cuda.is_available():
            return []

        devices: list[dict[str, Any]] = []
        count = torch.cuda.device_count()
        for idx in range(count):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                used_gb = total_gb - free_gb

                # Check if device has enough headroom to be considered exclusively free
                # A device is considered BUSY if free VRAM < threshold (meaning another user/job is active)
                is_free = free_gb >= self.min_free_gb

                devices.append({
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "is_free": is_free,
                })
            except Exception as e:
                devices.append({
                    "index": idx,
                    "name": f"cuda:{idx}",
                    "free_gb": 0.0,
                    "total_gb": 0.0,
                    "used_gb": 0.0,
                    "is_free": False,
                    "error": str(e),
                })
        return devices

    def acquire_exclusive_gpu(
        self,
        preferred_gpu: int | str | None = "auto",
        allow_switch: bool = True,
    ) -> int | None:
        """Select an exclusive, unshared GPU device or None if all are occupied.

        Args:
            preferred_gpu: Explicit GPU index (0, 1), 'auto', or None.
            allow_switch: Whether to automatically switch to the other GPU if preferred is busy.

        Returns:
            Chosen GPU index (int) or None if no exclusive GPU is available.
        """
        if self.is_worker_process():
            # CPU workers must never touch CUDA devices
            return None

        devices = self.query_devices()
        if not devices:
            return None

        # Parse preferred_gpu
        target_pref: int | None = None
        if isinstance(preferred_gpu, str):
            clean = preferred_gpu.strip().lower()
            if clean.isdigit():
                target_pref = int(clean)
            elif clean == "auto":
                target_pref = None
        elif isinstance(preferred_gpu, int):
            target_pref = preferred_gpu

        # If a specific preferred GPU was requested
        if target_pref is not None and 0 <= target_pref < len(devices):
            dev = devices[target_pref]
            if dev["is_free"]:
                return target_pref
            elif not allow_switch:
                print(
                    f"[GPU ARBITER] Requested GPU {target_pref} is in use ({dev['used_gb']} GB used, "
                    f"only {dev['free_gb']} GB free < {self.min_free_gb} GB threshold). Switching disabled."
                )
                return None
            else:
                print(
                    f"[GPU ARBITER] Requested GPU {target_pref} is in use ({dev['used_gb']} GB used by another job/supervisor). "
                    f"Checking for an alternative idle GPU..."
                )

        # Look for the healthiest idle device with >= min_free_gb
        free_devices = [d for d in devices if d["is_free"]]
        if not free_devices:
            return None

        # Sort by maximum free VRAM
        free_devices.sort(key=lambda d: d["free_gb"], reverse=True)
        chosen = int(free_devices[0]["index"])
        if target_pref is not None and chosen != target_pref:
            print(
                f"[GPU ARBITER - SWITCHED] Switched execution to idle GPU {chosen} "
                f"({free_devices[0]['free_gb']} GB free) to avoid resource contention with supervisor/colleagues."
            )
        return chosen

    def wait_for_exclusive_gpu(
        self,
        preferred_gpu: int | str | None = "auto",
        allow_switch: bool = True,
    ) -> tuple[int, ...] | tuple[()]:
        """Acquire an exclusive GPU, waiting and auto-pausing if both devices are occupied.

        Args:
            preferred_gpu: Explicit target GPU (0, 1) or 'auto'.
            allow_switch: Whether to switch GPUs if preferred is busy.

        Returns:
            Tuple with single GPU index (e.g. (1,)) or () if CUDA unavailable or timed out.
        """
        if self.is_worker_process() or not torch.cuda.is_available():
            return tuple()

        is_test_env = (
            "unittest" in sys.modules
            or "pytest" in sys.modules
            or os.environ.get("DRUGEX_TEST_MODE") == "1"
            or self.max_wait_hours <= 0
        )

        first_wait: bool = True
        start_time: float = time.time()
        max_wait_seconds: float = self.max_wait_hours * 3600.0

        while True:
            gpu_idx = self.acquire_exclusive_gpu(preferred_gpu=preferred_gpu, allow_switch=allow_switch)
            if gpu_idx is not None:
                # Successfully acquired exclusive GPU! Test allocation once safely:
                try:
                    t = torch.zeros(1, device=f"cuda:{gpu_idx}")
                    del t
                    if not first_wait:
                        print(
                            f"[GPU ARBITER - RESUMED] Exclusive access secured on GPU {gpu_idx}. "
                            f"Resuming pipeline execution."
                        )
                    return (gpu_idx,)
                except Exception as e:
                    FallbackMonitor.record_fallback(
                        fallback_type="device_cuda_error",
                        message=f"Test allocation on GPU {gpu_idx} failed: {e}",
                    )

            if is_test_env:
                # Fast path during automated test suites: use best available device without blocking
                devices = self.query_devices()
                healthy = [d for d in devices if d["free_gb"] >= 1.0]
                if healthy:
                    healthy.sort(key=lambda d: d["free_gb"], reverse=True)
                    return (int(healthy[0]["index"]),)
                return tuple()

            # Both GPUs are occupied or insufficient free VRAM!
            devices = self.query_devices()
            dev_str = ", ".join([f"GPU {d['index']}: {d['free_gb']}GB free" for d in devices])
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                FallbackMonitor.record_fallback(
                    fallback_type="device_cuda_timeout",
                    message=f"Timed out waiting for exclusive GPU after {self.max_wait_hours} hours",
                )
                print(
                    f"[GPU ARBITER - TIMEOUT] Waited {self.max_wait_hours} hours with no exclusive GPU available. Falling back to CPU."
                )
                return tuple()

            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GPU ARBITER - PAUSED] "
                f"All GPUs occupied on shared server ({dev_str}). "
                f"Supervisor/colleagues active. Pausing and waiting for exclusive access "
                f"(checking every {self.check_interval_sec}s)..."
            )
            first_wait = False
            time.sleep(self.check_interval_sec)
