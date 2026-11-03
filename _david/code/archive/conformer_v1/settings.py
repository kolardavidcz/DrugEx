"""Unified configuration settings for conformer studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from _david.code.config import DEFAULT_MOLECULE


@dataclass(frozen=True)
class StudySettings:
    """Unified configuration for chaining conformer study operations."""

    molecule: str = DEFAULT_MOLECULE
    output_dir: Path = Path("_david/output/conformer_study")
    methods: tuple[str, ...] = ("etkdg", "uff", "mmff_high_dielectric")
    reference_cap: int = 4
    candidate_attempts: int = 20
    repeats: int = 3
    run_reference_banks: bool = True
    run_benchmarks: bool = True
    generate_figures: bool = True
    cpu_ratio: float = 0.99
    gpu: int | str = "auto"
