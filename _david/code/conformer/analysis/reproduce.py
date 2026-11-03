"""Regenerate every component figure and table from canonical data.

The entry point is intentionally boring: load the registered tables, call one
figure function per component, write component 4's tables, and return the
artifact map.  No chemistry and no study execution happen here, which is why
``python -m _david.code.conformer.cli figures`` is safe to run at any time.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from _david.code.conformer.analysis import (
    clustering,
    color_shape,
    loaders,
    optimization_methods,
)
from _david.code.conformer.analysis.preparation import build_registry, write_component_tables
from _david.code.conformer.core import datasets, storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import MoleculeSpec
from _david.code.conformer.core.paths import COMPONENT_DIRS, OUTPUT_ROOT

#: Tables each figure requires; missing optional inputs degrade gracefully.
REQUIRED: Mapping[str, tuple[str, ...]] = {
    "egcg_omnibus_grid": ("combo", "candidate_attempts"),
    "drug_panel_cells": ("score", "method"),
    "drug_panel_summary": ("mean", "mean_seconds"),
    "stereo_input_audit": ("masked_upper_bound",),
    "synthesis_grid": ("n_clusters", "score"),
    "tversky_grid": ("self_mean", "worst_decoy_p95"),
}

#: Panels combined into the canonical molecule registry.
REGISTRY_PANELS: tuple[str, ...] = (
    "drug_panel_development",
    "drug_panel_confirmation",
    "egcg",
)


def _load_optional(name: str) -> loaders.Table | None:
    try:
        return loaders.load(name)
    except Exception:  # noqa: BLE001 - an optional figure input may be absent
        return None


def _require_canonical(table: loaders.Table) -> loaders.Table:
    """Refuse to regenerate a canonical figure from an un-migrated source."""
    if not table.canonical:
        raise ValueError(
            f"Table {table.name!r} resolved to a legacy path {table.path.as_posix()}; "
            "run `cli migrate` first or drop --canonical-only"
        )
    return table


def _registry_specs() -> tuple[MoleculeSpec, ...]:
    specs: list[MoleculeSpec] = []
    for name in REGISTRY_PANELS:
        try:
            specs.extend(datasets.load_panel(name))
        except Exception:  # noqa: BLE001 - panel may not be migrated yet
            continue
    deduplicated: dict[str, MoleculeSpec] = {}
    for spec in specs:
        deduplicated.setdefault(spec.name, spec)
    return tuple(deduplicated.values())


def regenerate(
    *,
    output_root: Path | str | None = None,
    canonical_only: bool = False,
    build_registry_table: bool = True,
) -> dict[str, Path]:
    """Regenerate all component deliverables and return their paths."""
    root = Path(output_root) if output_root else OUTPUT_ROOT
    paths: dict[str, Path] = {}

    def load(name: str) -> loaders.Table:
        table = loaders.require_columns(loaders.load(name), REQUIRED[name])
        return _require_canonical(table) if canonical_only else table

    egcg = load("egcg_omnibus_grid")
    cells = load("drug_panel_cells")
    summary = load("drug_panel_summary")
    audit = load("stereo_input_audit")
    grid = load("tversky_grid")
    synthesis = load("synthesis_grid")
    hard_decoys = _load_optional("egcg_hard_decoys")
    cross_decoy = _load_optional("cross_decoy")
    if canonical_only:
        hard_decoys = None if hard_decoys is None else _require_canonical(hard_decoys)
        cross_decoy = None if cross_decoy is None else _require_canonical(cross_decoy)

    optimization_dir = root / COMPONENT_DIRS["optimization_methods"]
    color_shape_dir = root / COMPONENT_DIRS["color_shape"]
    clustering_dir = root / COMPONENT_DIRS["clustering"]
    preparation_dir = root / COMPONENT_DIRS["preparation"]

    paths["egcg_omnibus_diagnostic.png"] = optimization_methods.figure_egcg_omnibus(
        egcg, optimization_dir
    )
    paths["drug_panel_stratified_vs_trimmed.png"] = optimization_methods.figure_drug_panel(
        cells, summary, optimization_dir
    )
    paths["isomer_stratification.png"] = optimization_methods.figure_isomer_stratification(
        cells, audit, optimization_dir
    )
    paths["reference_counts_vs_energy_window.png"] = (
        optimization_methods.figure_reference_counts_vs_energy_window(egcg, optimization_dir)
    )

    paths["tversky_color_162grid_selectivity.png"] = color_shape.figure_tversky_grid(
        grid, hard_decoys, color_shape_dir
    )
    paths["default_scorer_neighbourhood.csv"] = storage.write_csv(
        color_shape_dir / "default_scorer_neighbourhood.csv",
        color_shape.default_scorer_summary(grid),
    )
    paths["drug128_definitive_synthesis.png"] = clustering.figure_synthesis(
        synthesis, cross_decoy, clustering_dir
    )

    if build_registry_table:
        specs = _registry_specs()
        if specs:
            registry, errors = build_registry(specs, cache=ArtifactCache())
            counts = _load_optional("reference_counts")
            paths.update(
                write_component_tables(
                    registry,
                    counts=None if counts is None else counts.frame,
                    audit=audit.frame,
                    output_dir=preparation_dir,
                )
            )
            if errors:
                paths["registry_errors.csv"] = storage.write_csv(
                    preparation_dir / "registry_errors.csv", pd.DataFrame(errors)
                )
            else:
                # A previous run's failures must not survive a successful run.
                stale = preparation_dir / "registry_errors.csv"
                if stale.is_file():
                    stale.unlink()

    return paths


__all__ = ["REGISTRY_PANELS", "REQUIRED", "regenerate"]
