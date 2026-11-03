"""Shared plotting style and figure provenance for the conformer components.

Figures are deliverables, not scratch output: they are written into a component
folder next to a small JSON manifest naming the input tables (with SHA-256) so a
reader can tell exactly which data produced them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from _david.code.conformer.core import provenance, storage  # noqa: E402

#: Colour-blind-safe palette shared by every component figure.
PALETTE: dict[str, str] = {
    "mmff94s_water": "#1b9e77",
    "mmff94s_vacuum": "#d95f02",
    "uff": "#7570b3",
    "etkdg": "#e7298a",
    "diverse": "#d62728",
    "rmsd_0.35": "#1f77b4",
    "rmsd_0.50": "#2ca02c",
    "rmsd_1.25": "#7f7f7f",
    "positive": "#1b9e77",
    "decoy": "#d95f02",
    "neutral": "#4d4d4d",
    "grid": "#bdbdbd",
}

#: Default DPI for component figures.
FIGURE_DPI = 180


def apply_style() -> None:
    """Apply the conservative, print-safe house style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )


def save_figure(
    figure: plt.Figure,
    path: Path | str,
    *,
    dpi: int = FIGURE_DPI,
    inputs: Mapping[str, Path | str] | None = None,
    notes: str = "",
) -> Path:
    """Save ``figure`` and a provenance sidecar, returning the PNG path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    plt.close(figure)

    manifest: dict[str, Any] = {
        "figure": target.name,
        "generated_utc": provenance.utc_now(),
        "git_revision": provenance.git_revision(),
        "git_dirty": provenance.git_dirty(),
        "versions": provenance.runtime_versions(),
        "notes": notes,
        "inputs": {},
    }
    for name, source in (inputs or {}).items():
        manifest["inputs"][name] = storage.file_record(source)
    storage.write_json(target.with_suffix(".manifest.json"), manifest)
    return target


__all__ = ["FIGURE_DPI", "PALETTE", "apply_style", "save_figure"]
