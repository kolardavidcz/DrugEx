"""Canonical dataset registry and panel loading.

Long-lived tables live under ``_david/data/conformers/canonical`` because
figures, tests, and later studies all read them.  Nothing in this module
computes chemistry; it only resolves, validates, and caches *where* a table
lives so a re-analysis never depends on a dated output directory.

The registry has two halves:

``tables/``
    Wide CSV tables produced by a study or imported from a validated historical
    run.  Loading is dtype-aware: counters stay integers, labels stay strings.

``panels/``
    Molecule lists (``name,smiles[,tags]``) that turn into
    :class:`~_david.code.conformer.core.contracts.MoleculeSpec` tuples.

Legacy lookup is read-only and clearly marked: if a canonical table is missing,
:func:`load_table` may fall back to a historical output file so a migration can
be validated before the copy is committed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _david.code.conformer.core import storage
from _david.code.conformer.core.contracts import MoleculeSpec
from _david.code.conformer.core.errors import StudyError
from _david.code.conformer.core.paths import CANONICAL_ROOT

TABLE_DIR = CANONICAL_ROOT / "tables"
PANEL_DIR = CANONICAL_ROOT / "panels"

#: Logical table name -> canonical filename.
TABLES: dict[str, str] = {
    "egcg_omnibus_grid": "egcg_omnibus_grid.csv",
    "drug_panel_cells": "drug_panel_development_cells.csv.gz",
    "drug_panel_summary": "drug_panel_development_summary.csv",
    "stereo_input_audit": "stereo_input_audit.csv",
    "reference_counts": "reference_counts.csv",
    "synthesis_grid": "drug128_synthesis_grid.csv",
    "cross_decoy": "drug128_cross_decoy_selectivity.csv",
    "tversky_grid": "drug121_tversky_color_162grid.csv",
    "egcg_hard_decoys": "egcg_hard_decoys_162grid.csv",
    "egcg_tversky_panel": "egcg_tversky_vs_tanimoto_panel.csv",
}

#: Logical panel name -> canonical filename.
PANELS: dict[str, str] = {
    "drug_panel_development": "drug_panel_development.csv",
    "drug_panel_confirmation": "drug_panel_confirmation.csv",
    "egcg": "egcg.csv",
}

#: Columns that must stay strings even when pandas would infer numbers.
STRING_COLUMNS: dict[str, tuple[str, ...]] = {
    "egcg_omnibus_grid": ("energy_window", "cand_emb_rmsd", "cap_label"),
    "synthesis_grid": ("k_label",),
}


@dataclass(frozen=True)
class TableSource:
    """A resolved table location plus how it was found."""

    name: str
    path: Path
    canonical: bool


def table_path(name: str, *, root: Path | str | None = None) -> Path:
    """Return the canonical path of a registered table."""
    try:
        filename = TABLES[name]
    except KeyError as exc:
        raise StudyError(
            f"Unknown table {name!r}; expected one of {sorted(TABLES)}"
        ) from exc
    return (Path(root) if root else TABLE_DIR) / filename


def panel_path(name: str, *, root: Path | str | None = None) -> Path:
    """Return the canonical path of a registered molecule panel."""
    try:
        filename = PANELS[name]
    except KeyError as exc:
        raise StudyError(
            f"Unknown panel {name!r}; expected one of {sorted(PANELS)}"
        ) from exc
    return (Path(root) if root else PANEL_DIR) / filename


def resolve_table(
    name: str,
    *,
    legacy: Mapping[str, Path | str] | None = None,
    root: Path | str | None = None,
) -> TableSource:
    """Resolve a table, optionally falling back to an explicit legacy path."""
    canonical = table_path(name, root=root)
    if canonical.is_file():
        return TableSource(name=name, path=canonical, canonical=True)
    if legacy and name in legacy:
        candidate = Path(legacy[name])
        if candidate.is_file():
            return TableSource(name=name, path=candidate, canonical=False)
    raise StudyError(
        f"Table {name!r} is not available at {canonical.as_posix()}"
        + (f" and legacy path {legacy[name]} is missing" if legacy and name in legacy else "")
    )


def load_table(
    name: str,
    *,
    legacy: Mapping[str, Path | str] | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Load a registered table with stable dtypes.

    Default NA handling is kept on purpose: blank cells in a run table mean
    "not measured" (an excluded or failed cell), and coercing them to ``""``
    silently turns every numeric column of that table into ``object``.
    """
    source = resolve_table(name, legacy=legacy, root=root)
    dtypes = {column: "string" for column in STRING_COLUMNS.get(name, ())}
    return pd.read_csv(source.path, dtype=dtypes)


def write_table(
    name: str,
    frame: pd.DataFrame,
    *,
    root: Path | str | None = None,
) -> Path:
    """Write a frame into the canonical table directory."""
    target = table_path(name, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".gz":
        frame.to_csv(target, index=False, compression="gzip")
        return target
    return storage.write_csv(target, frame)


def load_panel(
    name: str,
    *,
    root: Path | str | None = None,
    split: str | None = None,
) -> tuple[MoleculeSpec, ...]:
    """Load a molecule panel as immutable specs.

    The panel schema is ``name,smiles[,source][,tags]`` where ``tags`` is a
    ``|``-separated list, so decoy categories and splits travel with the data
    instead of being hard-coded in a study.
    """
    path = panel_path(name, root=root)
    if not path.is_file():
        raise StudyError(f"Panel {name!r} is not available at {path.as_posix()}")
    frame = pd.read_csv(path, keep_default_na=False)
    if "name" not in frame.columns or "smiles" not in frame.columns:
        raise StudyError(f"Panel {path.name} must define 'name' and 'smiles' columns")
    if split is not None:
        if "split" not in frame.columns:
            raise StudyError(f"Panel {path.name} has no 'split' column")
        frame = frame[frame["split"] == split]
        if frame.empty:
            raise StudyError(f"Panel {name!r} has no rows for split {split!r}")

    specs: list[MoleculeSpec] = []
    for _, row in frame.iterrows():
        raw_tags = str(row.get("tags", "") or "")
        tags = tuple(tag for tag in raw_tags.split("|") if tag)
        split_value = str(row.get("split", "") or "")
        if split_value and not any(tag.startswith("split:") for tag in tags):
            tags = (*tags, f"split:{split_value}")
        specs.append(
            MoleculeSpec(
                name=str(row["name"]),
                smiles=str(row["smiles"]),
                source=str(row.get("source", "panel") or "panel"),
                tags=tags,
            )
        )
    return tuple(specs)


def write_panel(
    name: str,
    specs: tuple[MoleculeSpec, ...] | list[MoleculeSpec],
    *,
    root: Path | str | None = None,
) -> Path:
    """Persist molecule specs as a canonical panel."""
    target = panel_path(name, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "name": spec.name,
                "smiles": spec.smiles,
                "source": spec.source,
                "tags": "|".join(spec.tags),
            }
            for spec in specs
        ]
    )
    return storage.write_csv(target, frame)


__all__ = [
    "PANELS",
    "PANEL_DIR",
    "STRING_COLUMNS",
    "TABLES",
    "TABLE_DIR",
    "TableSource",
    "load_panel",
    "load_table",
    "panel_path",
    "resolve_table",
    "table_path",
    "write_panel",
    "write_table",
]
