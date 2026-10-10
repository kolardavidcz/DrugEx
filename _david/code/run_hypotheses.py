"""Backward-compatibility forwarding module for Hypothesis Benchmark.

Primary implementation moved to _david.code.experiments.hypotheses.
"""
from __future__ import annotations

from _david.code.experiments.hypotheses import ExperimentRunner, main

__all__: list[str] = ["ExperimentRunner", "main"]

if __name__ == "__main__":
    main()
