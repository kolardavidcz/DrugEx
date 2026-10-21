"""Academic thesis experiments, hypothesis benchmarks, and ablation studies."""
from __future__ import annotations

from _david.legacy.experiments.ablations import ThresholdAblationRunner, main as run_ablations
from _david.legacy.experiments.hypotheses import ExperimentRunner, main as run_hypotheses

__all__: list[str] = [
    "ExperimentRunner",
    "ThresholdAblationRunner",
    "run_hypotheses",
    "run_ablations",
]
