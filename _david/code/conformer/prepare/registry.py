"""Registry of canonical, verified input molecules.

The registry is intentionally a *wide table* plus one SDF per identity.  It
answers three questions without touching a study cache:

1. Has this exact structure already been prepared?
2. Which preparation contract was used, and did it change the input?
3. What are the descriptor values that downstream stratification relies on?

It never deletes or rewrites an existing identity; conflicting re-registrations
are rejected so a silent chemistry change cannot propagate into old figures.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from _david.code.conformer.core import storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import EmbedSpec, MoleculeSpec
from _david.code.conformer.core.errors import PreparationError
from _david.code.conformer.core.paths import canonical_root
from _david.code.conformer.core.provenance import utc_now
from _david.code.conformer.prepare.standardize import (
    PreparedMolecule,
    prepare_many,
    prepare_molecule,
)

INDEX_NAME = "index.csv"
COLUMNS = (
    "name",
    "input_smiles",
    "standard_smiles",
    "canonical_smiles",
    "isomer_smiles",
    "warnings",
    "heavy_atoms",
    "molecular_weight",
    "rotatable_bonds",
    "saturated_rings",
    "aromatic_rings",
    "h_bond_donors",
    "h_bond_acceptors",
    "stereo_centers",
    "unspecified_stereo_centers",
    "isomer_upper_bound",
    "qed",
    "sa_score",
    "formal_charge",
    "fraction_csp3",
    "source",
    "registered_utc",
)


def _sdf_path(root: Path, canonical_smiles: str) -> Path:
    """Return the deterministic SDF path for one canonical identity."""
    from _david.code.conformer.core.contracts import fingerprint

    return root / f"{fingerprint(canonical_smiles)[:16]}.sdf"


def _row(prepared: PreparedMolecule) -> dict[str, Any]:
    """Flatten a prepared molecule into the registry schema."""
    properties = prepared.properties
    return {
        "name": prepared.spec.name,
        "input_smiles": prepared.spec.smiles,
        "standard_smiles": prepared.standard_smiles,
        "canonical_smiles": prepared.canonical_smiles,
        "isomer_smiles": "|".join(prepared.isomer_smiles),
        "warnings": "|".join(prepared.warnings),
        "source": prepared.spec.source,
        "registered_utc": utc_now(),
        **{
            key: properties.get(key)
            for key in (
                "heavy_atoms",
                "molecular_weight",
                "rotatable_bonds",
                "saturated_rings",
                "aromatic_rings",
                "h_bond_donors",
                "h_bond_acceptors",
                "stereo_centers",
                "unspecified_stereo_centers",
                "isomer_upper_bound",
                "qed",
                "sa_score",
                "formal_charge",
                "fraction_csp3",
            )
        },
    }


def load_registry(root: Path | str | None = None) -> pd.DataFrame:
    """Load the registry index, returning an empty typed frame when absent."""
    index_path = (Path(root) if root else canonical_root()) / INDEX_NAME
    if not index_path.is_file():
        return pd.DataFrame(columns=list(COLUMNS))
    return storage.read_csv(index_path)


def register(
    prepared: PreparedMolecule,
    *,
    root: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register one prepared molecule and write its canonical SDF.

    Re-registering the same canonical identity with a different preparation
    contract raises :class:`PreparationError` unless ``overwrite`` is true.
    """
    base = Path(root) if root else canonical_root()
    base.mkdir(parents=True, exist_ok=True)
    frame = load_registry(base)
    row = _row(prepared)

    same_identity = frame[frame["canonical_smiles"] == prepared.canonical_smiles]
    if not same_identity.empty:
        existing = same_identity.iloc[0].to_dict()
        if not overwrite and existing["standard_smiles"] != row["standard_smiles"]:
            raise PreparationError(
                "Identity already registered with a different standardization "
                f"({existing['standard_smiles']!r} -> {row['standard_smiles']!r})"
            )
        frame = frame[frame["canonical_smiles"] != prepared.canonical_smiles]

    sdf_path = _sdf_path(base, prepared.canonical_smiles)
    storage.write_sdf(sdf_path, prepared.isomers)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame = frame.drop_duplicates(subset=["canonical_smiles", "name"], keep="last")
    frame = frame.sort_values(["name", "registered_utc"]).reset_index(drop=True)
    storage.write_csv(base / INDEX_NAME, frame[list(COLUMNS)])
    return {**row, "sdf": sdf_path.as_posix()}


def register_many(
    specs: Iterable[MoleculeSpec],
    embed: EmbedSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
    root: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Prepare and register a batch, isolating per-molecule failures."""
    prepared, errors = prepare_many(specs, embed, cache=cache)
    rows: list[dict[str, Any]] = []
    for molecule in prepared:
        try:
            rows.append(register(molecule, root=root, overwrite=overwrite))
        except PreparationError as exc:
            errors.append(
                {"molecule": molecule.spec.name, "smiles": molecule.spec.smiles, "error": str(exc)}
            )
    return pd.DataFrame(rows, columns=list(COLUMNS) + ["sdf"]), errors


def find(
    query: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Find a registered molecule by name, input SMILES, or canonical SMILES."""
    frame = load_registry(root)
    if frame.empty:
        return None
    mask = (
        (frame["name"] == query)
        | (frame["input_smiles"] == query)
        | (frame["canonical_smiles"] == query)
        | (frame["standard_smiles"] == query)
    )
    matches = frame[mask]
    return matches.iloc[0].to_dict() if not matches.empty else None


def resolve_molecule(
    name: str,
    smiles: str | None = None,
    *,
    source: str = "user",
    embed: EmbedSpec | None = None,
    cache: ArtifactCache | None = None,
    root: Path | str | None = None,
) -> PreparedMolecule:
    """Load a registered molecule, or prepare and register it on first use."""
    entry = find(name, root=root)
    if entry is not None:
        isomer_smiles = str(entry["isomer_smiles"]).split("|")
        return prepare_molecule(
            MoleculeSpec(name=name, smiles=str(entry["input_smiles"]), source=str(entry["source"])),
            embed or EmbedSpec(isomer_cap=len(isomer_smiles)),
            cache=cache,
        )
    if smiles is None:
        raise PreparationError(f"Unknown molecule {name!r} and no SMILES supplied")
    prepared = prepare_molecule(MoleculeSpec(name=name, smiles=smiles, source=source), embed, cache=cache)
    register(prepared, root=root)
    return prepared


__all__ = [
    "COLUMNS",
    "INDEX_NAME",
    "find",
    "load_registry",
    "register",
    "register_many",
    "resolve_molecule",
]
