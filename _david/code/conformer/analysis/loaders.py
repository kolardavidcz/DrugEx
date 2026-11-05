"""Resolve the tables each analysis needs, preferring canonical data.

Two modes are supported on purpose:

*Canonical* (default, what a reader should trust)
    ``_david/data/conformers/canonical/tables``.

*Legacy fallback* (migration only)
    The dated output directories that produced the published figures.  A figure
    built from legacy data records that fact in its sidecar manifest, so an
    un-migrated dependency can never masquerade as canonical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _david.code.conformer.core import datasets
from _david.code.conformer.core.paths import DAVID_ROOT
from _david.code.conformer.core.provenance import sha256_file

#: Historical roots used only until the canonical copy exists. The study tree moved to
#: ``output/conformer/self_similarity``; the flat path is kept for hosts not yet
#: reorganized, so resolve whichever exists.
def _legacy_self_similarity() -> Path:
    categorized = DAVID_ROOT / "output" / "conformer" / "self_similarity"
    flat = DAVID_ROOT / "output" / "20261028_conformer_self_similarity"
    return categorized if categorized.is_dir() else flat


LEGACY_SELF_SIMILARITY = _legacy_self_similarity()
LEGACY_SELECTIVITY = (
    LEGACY_SELF_SIMILARITY / "20261029_1400_pareto_selectivity_163_drugs"
)
LEGACY_CALIBRATION = (
    LEGACY_SELF_SIMILARITY / "20261029_1155_reward_calibration_126_mols" / "raw_data"
)

#: Logical table -> historical file, used only when the canonical copy is absent.
LEGACY_TABLES: Mapping[str, Path] = {
    "egcg_omnibus_grid": LEGACY_SELECTIVITY / "egcg_omnibus_grid.csv",
    "drug_panel_cells": LEGACY_CALIBRATION / "development" / "data.csv.gz",
    "drug_panel_summary": LEGACY_CALIBRATION / "development" / "summary.csv",
    "stereo_input_audit": LEGACY_CALIBRATION / "stereo_input_audit.csv",
    "reference_counts": LEGACY_CALIBRATION / "reference_counts.csv",
    "synthesis_grid": LEGACY_SELECTIVITY / "drug128_synthesis_grid.csv",
    "cross_decoy": LEGACY_SELECTIVITY / "drug128_cross_decoy_selectivity.csv",
    "tversky_grid": LEGACY_SELECTIVITY / "drug121_tversky_color_162grid.csv",
    "egcg_hard_decoys": LEGACY_SELECTIVITY / "egcg_hard_decoys_162grid.csv",
    "egcg_tversky_panel": LEGACY_SELECTIVITY / "egcg_tversky_vs_tanimoto_panel.csv",
}


@dataclass(frozen=True)
class Table:
    """A loaded table plus where it came from."""

    name: str
    frame: pd.DataFrame
    path: Path
    canonical: bool

    @property
    def digest(self) -> str | None:
        """SHA-256 of the source file, for figure provenance."""
        return sha256_file(self.path)


def load(name: str, *, canonical_only: bool = False) -> Table:
    """Load a registered table, preferring the canonical copy."""
    source = datasets.resolve_table(
        name,
        legacy=None if canonical_only else LEGACY_TABLES,
    )
    frame = datasets.load_table(
        name,
        legacy=None if canonical_only else LEGACY_TABLES,
    )
    return Table(name=name, frame=frame, path=source.path, canonical=source.canonical)


def require_columns(table: Table, columns: tuple[str, ...]) -> Table:
    """Fail fast when a table is missing columns a figure depends on."""
    missing = [column for column in columns if column not in table.frame.columns]
    if missing:
        raise ValueError(
            f"Table {table.name!r} at {table.path.as_posix()} is missing {missing}"
        )
    return table


def input_paths(*tables: Table) -> dict[str, Path]:
    """Map table names to paths for a figure manifest."""
    return {table.name: table.path for table in tables}


__all__ = [
    "LEGACY_CALIBRATION",
    "LEGACY_SELECTIVITY",
    "LEGACY_SELF_SIMILARITY",
    "LEGACY_TABLES",
    "Table",
    "input_paths",
    "load",
    "require_columns",
]
