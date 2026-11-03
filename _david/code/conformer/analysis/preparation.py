"""Component 4 tables: molecule registry, preparation audit, reference counts.

This component has no headline figure on purpose.  Its deliverable is the
canonical data every other component consumes: one row per molecule recording
the input structure, the standardized structure, the enumerated stereoisomers,
descriptor values, and the number of conformers that survived each expensive
stage.

Nothing here needs a 3D conformer: :func:`build_registry` performs the cheap
chemical transformations only, so registering a new molecule is fast and safe
to repeat.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from _david.code.conformer.core import storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import EmbedSpec, MoleculeSpec
from _david.code.conformer.core.errors import PreparationError
from _david.code.conformer.core.paths import component_dir
from _david.code.conformer.prepare.standardize import prepare_molecule

#: Enumeration cap used when auditing inputs.  Deliberately far above any study
#: budget: the registry exists to *reveal* truncation, not to hide it.
REGISTRY_ISOMER_CAP: int = 32

#: Registry columns, in the order a reader expects them.
REGISTRY_COLUMNS: tuple[str, ...] = (
    "name",
    "input_smiles",
    "standard_smiles",
    "canonical_smiles",
    "isomers_enumerated",
    "isomer_upper_bound",
    "hit_enumeration_cap",
    "enumeration_cap",
    "heavy_atoms",
    "molecular_weight",
    "rotatable_bonds",
    "aromatic_rings",
    "h_bond_donors",
    "h_bond_acceptors",
    "stereo_centers",
    "unspecified_stereo_centers",
    "formal_charge",
    "qed",
    "sa_score",
    "fraction_csp3",
    "warnings",
    "source",
    "tags",
)


def build_registry(
    specs: Sequence[MoleculeSpec],
    *,
    embed: EmbedSpec | None = None,
    cache: ArtifactCache | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Standardize each input and return the canonical registry table.

    ``isomer_upper_bound`` is RDKit's naive count over unassigned stereocentres;
    it is only an *upper* bound because symmetric inputs collapse (ring
    cis/trans equivalence), so ``isomers_enumerated`` can legitimately be
    smaller without any truncation.  ``hit_enumeration_cap`` is therefore the
    only truncation flag the registry can assert without an uncapped
    enumeration.
    """
    spec_embed = embed or EmbedSpec(isomer_cap=REGISTRY_ISOMER_CAP)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for spec in specs:
        try:
            prepared = prepare_molecule(spec, spec_embed, cache=cache)
        except PreparationError as exc:
            errors.append({"molecule": spec.name, "smiles": spec.smiles, "error": str(exc)})
            continue
        properties = prepared.properties
        upper_bound = int(properties.get("isomer_upper_bound") or prepared.n_isomers)
        rows.append(
            {
                "name": spec.name,
                "input_smiles": spec.smiles,
                "standard_smiles": prepared.standard_smiles,
                "canonical_smiles": prepared.canonical_smiles,
                "isomers_enumerated": prepared.n_isomers,
                "isomer_upper_bound": upper_bound,
                "hit_enumeration_cap": prepared.n_isomers >= spec_embed.isomer_cap,
                "enumeration_cap": spec_embed.isomer_cap,
                "heavy_atoms": properties.get("heavy_atoms"),
                "molecular_weight": properties.get("molecular_weight"),
                "rotatable_bonds": properties.get("rotatable_bonds"),
                "aromatic_rings": properties.get("aromatic_rings"),
                "h_bond_donors": properties.get("h_bond_donors"),
                "h_bond_acceptors": properties.get("h_bond_acceptors"),
                "stereo_centers": properties.get("stereo_centers"),
                "unspecified_stereo_centers": properties.get("unspecified_stereo_centers"),
                "formal_charge": properties.get("formal_charge"),
                "qed": properties.get("qed"),
                "sa_score": properties.get("sa_score"),
                "fraction_csp3": properties.get("fraction_csp3"),
                "warnings": "|".join(prepared.warnings),
                "source": spec.source,
                "tags": "|".join(spec.tags),
            }
        )
    frame = pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))
    return frame.sort_values("name").reset_index(drop=True), errors


def reference_count_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-molecule conformer counts by phase and method.

    ``survival`` is the fraction of generated conformers that also survive the
    energy window; ``convergence`` is the fraction that a force field converged.
    """
    if counts.empty:
        return counts.copy()
    summary = (
        counts.groupby(["phase", "method"], as_index=False)
        .agg(
            molecules=("molecule_id", "nunique"),
            isomers=("enumerated_isomers", "mean"),
            generated=("generated_valid", "mean"),
            converged=("converged_valid", "mean"),
            survived=("energy_survivors", "mean"),
            preparation_failures=("chemical_preparation_ok", lambda values: int((~values.astype(bool)).sum())),
        )
        .reset_index(drop=True)
    )
    summary["convergence"] = summary["converged"] / summary["generated"].replace(0, pd.NA)
    summary["survival"] = summary["survived"] / summary["generated"].replace(0, pd.NA)
    return summary


def isomer_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Molecules whose enumerated isomers were truncated by the enumeration cap."""
    if audit.empty:
        return audit.copy()
    frame = audit.copy()
    frame["truncated"] = frame["masked_enumerated_up_to8"] < frame["masked_upper_bound"]
    columns = [
        "molecule_id",
        "name",
        "split",
        "specified_stereo",
        "original_upper_bound",
        "masked_upper_bound",
        "masked_enumerated_up_to8",
        "truncated",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values(
        ["truncated", "masked_upper_bound"], ascending=[False, False]
    )


def write_component_tables(
    registry: pd.DataFrame,
    *,
    counts: pd.DataFrame | None = None,
    audit: pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Write the registry and optional audit tables into component 4."""
    target = Path(output_dir) if output_dir else component_dir("preparation")
    target.mkdir(parents=True, exist_ok=True)
    written = {"molecule_registry.csv": storage.write_csv(target / "molecule_registry.csv", registry)}
    if counts is not None and not counts.empty:
        written["reference_counts.csv"] = storage.write_csv(target / "reference_counts.csv", counts)
        written["reference_count_summary.csv"] = storage.write_csv(
            target / "reference_count_summary.csv", reference_count_summary(counts)
        )
    if audit is not None and not audit.empty:
        written["isomer_audit.csv"] = storage.write_csv(target / "isomer_audit.csv", isomer_audit(audit))
    return written


__all__ = [
    "REGISTRY_COLUMNS",
    "REGISTRY_ISOMER_CAP",
    "build_registry",
    "isomer_audit",
    "reference_count_summary",
    "write_component_tables",
]
