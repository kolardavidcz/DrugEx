"""Publication figures and canonical tables for the four conformer components."""

from __future__ import annotations

from _david.code.conformer.analysis.clustering import cluster_counts, figure_synthesis
from _david.code.conformer.analysis.color_shape import (
    default_scorer_summary,
    figure_tversky_grid,
)
from _david.code.conformer.analysis.common import (
    ISOMER_BIN_EDGES,
    ISOMER_BIN_LABELS,
    isomer_bin,
)
from _david.code.conformer.analysis.optimization_methods import (
    figure_drug_panel,
    figure_egcg_omnibus,
    figure_isomer_stratification,
    figure_reference_counts_vs_energy_window,
)
from _david.code.conformer.analysis.preparation import build_registry, write_component_tables
from _david.code.conformer.analysis.reproduce import regenerate
from _david.code.conformer.analysis.style import PALETTE, apply_style, save_figure

__all__ = [
    "ISOMER_BIN_EDGES",
    "ISOMER_BIN_LABELS",
    "PALETTE",
    "apply_style",
    "build_registry",
    "cluster_counts",
    "default_scorer_summary",
    "figure_drug_panel",
    "figure_egcg_omnibus",
    "figure_isomer_stratification",
    "figure_reference_counts_vs_energy_window",
    "figure_synthesis",
    "figure_tversky_grid",
    "isomer_bin",
    "regenerate",
    "save_figure",
    "write_component_tables",
]
