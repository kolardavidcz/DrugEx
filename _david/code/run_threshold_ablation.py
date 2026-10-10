"""Backward-compatibility forwarding module for Threshold Ablations.

Primary implementation moved to _david.code.experiments.ablations.
"""
from __future__ import annotations

from _david.code.experiments.ablations import ThresholdAblationRunner, main

__all__: list[str] = ["ThresholdAblationRunner", "main"]

if __name__ == "__main__":
    main()
