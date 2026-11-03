"""Molecule preparation and canonical reference-count audit (component 4).

Question this study answers
---------------------------
What does the DrugEx standardization pipeline actually do to an input, how many
stereoisomers does it enumerate, and how many conformers survive embedding,
force-field convergence, and the energy window?  These counts are the resource
budget every downstream study inherits, so they are measured once and cached.

The audit is deliberately separate from scoring: a molecule can fail before any
3D geometry exists, and that failure must be a first-class row rather than a
missing value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from _david.code.chemistry import Optimization
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import (
    ArmSpec,
    EmbedSpec,
    MinimizeSpec,
    MoleculeSpec,
    ReferenceSpec,
    SelectSpec,
    default_candidate_spec,
)
from _david.code.conformer.core.errors import PreparationError
from _david.code.conformer.ensemble.embed import embed_ensemble
from _david.code.conformer.ensemble.minimize import minimize_ensemble
from _david.code.conformer.ensemble.select import cluster_statistics
from _david.code.conformer.prepare.registry import register
from _david.code.conformer.prepare.standardize import prepare_molecule
from _david.code.conformer.studies.base import CellRequest

#: Force-field conditions audited per molecule.
AUDIT_METHODS: dict[str, MinimizeSpec] = {
    "mmff94s_water": MinimizeSpec(
        profile=Optimization("MMFF94s", 78.4), energy_window=8.0
    ),
    "mmff94s_vacuum": MinimizeSpec(
        profile=Optimization("MMFF94s", 1.0), energy_window=8.0
    ),
    "uff": MinimizeSpec(profile=Optimization("UFF", 1.0), energy_window=8.0),
    "etkdg": MinimizeSpec(profile=None, energy_window=None),
}

#: Preparation contract used by the audit when the caller supplies none.
AUDIT_EMBED = EmbedSpec(attempts=50, isomer_cap=8, seed=104729, rmsd=0.35)


@dataclass(frozen=True)
class PreparationExecutor:
    """Measure preparation and reference-pool survival for one molecule + method."""

    cache_root: str | None = None
    cache_enabled: bool = True
    register_molecules: bool = False
    cluster_cutoff: float = 0.35

    def _cache(self) -> ArtifactCache:
        return ArtifactCache(
            Path(self.cache_root) if self.cache_root else None,
            enabled=self.cache_enabled,
        )

    def __call__(self, request: CellRequest) -> dict[str, Any]:
        cache = self._cache()
        embed_spec = request.arm.reference.embed
        prepared = prepare_molecule(request.molecule, embed_spec, cache=cache)
        if self.register_molecules:
            register(prepared)

        embed = embed_ensemble(prepared, embed_spec, cache=cache)
        minimize = minimize_ensemble(
            embed.records,
            request.arm.reference.minimize,
            cache=cache,
            cache_payload={"embed": embed.cache_key or len(embed.records)},
        )
        clusters, per_isomer = cluster_statistics(minimize.records, self.cluster_cutoff)
        return {
            "molecule": request.molecule.name,
            "input_smiles": request.molecule.smiles,
            "standard_smiles": prepared.standard_smiles,
            "canonical_smiles": prepared.canonical_smiles,
            "method": request.arm.name,
            "status": "ok",
            "warnings": "|".join(prepared.warnings),
            "isomers": prepared.n_isomers,
            "isomer_upper_bound": prepared.properties.get("isomer_upper_bound"),
            "heavy_atoms": prepared.properties.get("heavy_atoms"),
            "molecular_weight": prepared.properties.get("molecular_weight"),
            "rotatable_bonds": prepared.properties.get("rotatable_bonds"),
            "stereo_centers": prepared.properties.get("stereo_centers"),
            "qed": prepared.properties.get("qed"),
            "sa_score": prepared.properties.get("sa_score"),
            "conformers_generated": embed.n_records,
            "conformers_converged": minimize.converged_count,
            "conformers_survived": minimize.n_records,
            "window_removed": minimize.window_removed,
            "clusters": clusters,
            "clusters_per_isomer": "|".join(str(value) for value in per_isomer),
            "embed_seconds": embed.seconds,
            "minimize_seconds": minimize.seconds,
        }


def audit_arm(method: str, embed: EmbedSpec = AUDIT_EMBED) -> ArmSpec:
    """Return the single-purpose arm used to audit one force-field condition."""
    if method not in AUDIT_METHODS:
        raise ValueError(f"Unknown audit method {method!r}")
    return ArmSpec(
        name=method,
        reference=ReferenceSpec(
            embed=embed,
            minimize=AUDIT_METHODS[method],
            select=SelectSpec(),
        ),
        candidate=default_candidate_spec(),
        description=f"preparation audit under {method}",
    )


def audit_molecules(
    specs: Sequence[MoleculeSpec],
    *,
    methods: Sequence[str] = tuple(AUDIT_METHODS),
    embed: EmbedSpec = AUDIT_EMBED,
    cache: ArtifactCache | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Return a preparation audit table and a list of preparation failures."""
    executor = PreparationExecutor(
        cache_root=str(cache.root) if cache is not None else None,
        cache_enabled=bool(cache.enabled) if cache is not None else False,
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for method in methods:
        arm_spec = audit_arm(method, embed)
        for spec in specs:
            request = CellRequest(molecule=spec, arm=arm_spec)
            try:
                rows.append(executor(request))
            except PreparationError as exc:
                errors.append(
                    {
                        "molecule": spec.name,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            except (RuntimeError, ValueError) as exc:
                errors.append(
                    {
                        "molecule": spec.name,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows), errors


def survival_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean generated/converged/surviving conformers per method.

    ``conversion`` is the fraction of embedded conformers that survive the
    force field and the energy window, which is the quantity that matters when
    budgeting a reference pool.
    """
    if frame.empty:
        return frame.copy()
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            molecules=("molecule", "nunique"),
            isomers=("isomers", "mean"),
            generated=("conformers_generated", "mean"),
            converged=("conformers_converged", "mean"),
            survived=("conformers_survived", "mean"),
            clusters=("clusters", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["conversion"] = summary["survived"] / summary["generated"].replace(0, pd.NA)
    return summary


__all__ = [
    "AUDIT_EMBED",
    "AUDIT_METHODS",
    "PreparationExecutor",
    "audit_arm",
    "audit_molecules",
    "survival_summary",
]
