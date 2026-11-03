"""End-to-end reference and candidate construction with cache chaining."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import (
    CandidateSpec,
    ReferenceSpec,
    fingerprint,
)
from _david.code.conformer.ensemble.embed import EmbedResult, embed_ensemble
from _david.code.conformer.ensemble.minimize import MinimizeResult, minimize_ensemble
from _david.code.conformer.ensemble.select import SelectionResult, select_ensemble
from _david.code.conformer.prepare.standardize import PreparedMolecule


@dataclass(frozen=True)
class ReferenceBank:
    """A deployable reference bank plus every intermediate that produced it."""

    records: tuple[Chem.Mol, ...]
    embed: EmbedResult
    minimize: MinimizeResult
    selection: SelectionResult

    @property
    def n_records(self) -> int:
        return len(self.records)

    @property
    def cluster_count(self) -> int:
        return self.selection.cluster_count


@dataclass(frozen=True)
class CandidateEnsemble:
    """A candidate conformer pool used only for scoring, never as a reference."""

    records: tuple[Chem.Mol, ...]
    embed: EmbedResult
    minimize: MinimizeResult

    @property
    def n_records(self) -> int:
        return len(self.records)


def pool_fingerprint(records: tuple[Chem.Mol, ...]) -> str:
    """Return a cheap identity fingerprint for a conformer pool.

    Coordinates are deliberately excluded: the pooling functions are
    deterministic given their specs, so identity + record count + conformer
    labels are sufficient to distinguish pools while keeping hashing cheap.
    """
    return fingerprint(
        [
            (
                record.GetProp("EnumeratedIsomerSMILES")
                if record.HasProp("EnumeratedIsomerSMILES")
                else Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(record)), isomericSmiles=True),
                record.GetIntProp("ConformerIndex") if record.HasProp("ConformerIndex") else -1,
                record.GetProp("ForceField") if record.HasProp("ForceField") else "raw",
            )
            for record in records
        ]
    )


def build_candidate(
    prepared: PreparedMolecule,
    spec: CandidateSpec,
    *,
    cache: ArtifactCache | None = None,
) -> CandidateEnsemble:
    """Embed and optionally minimize a candidate ensemble."""
    embed = embed_ensemble(prepared, spec.embed, cache=cache)
    pool_key = embed.cache_key or pool_fingerprint(embed.records)
    minimize = minimize_ensemble(
        embed.records,
        spec.minimize,
        cache=cache,
        cache_payload={"embed": pool_key},
    )
    return CandidateEnsemble(records=minimize.records, embed=embed, minimize=minimize)


def build_reference(
    prepared: PreparedMolecule,
    spec: ReferenceSpec,
    *,
    cache: ArtifactCache | None = None,
) -> ReferenceBank:
    """Embed, relax, and select a deployable reference bank."""
    embed = embed_ensemble(prepared, spec.embed, cache=cache)
    pool_key = embed.cache_key or pool_fingerprint(embed.records)
    minimize = minimize_ensemble(
        embed.records,
        spec.minimize,
        cache=cache,
        cache_payload={"embed": pool_key},
    )
    selection = select_ensemble(
        minimize.records,
        spec.select,
        cache=cache,
        cache_payload={"minimize": minimize.cache_key or pool_fingerprint(minimize.records)},
    )
    return ReferenceBank(
        records=selection.records,
        embed=embed,
        minimize=minimize,
        selection=selection,
    )


def reference_from_records(
    records: tuple[Chem.Mol, ...],
    *,
    spec: ReferenceSpec,
) -> tuple[Chem.Mol, ...]:
    """Select a reference bank from an already-built pool without re-embedding.

    This is the reuse path for archived studies: load a saved SDF, then apply a
    new :class:`SelectSpec` to it.  It returns only the selected records; callers
    that need cluster statistics should call :func:`select_ensemble` directly.
    """
    result = select_ensemble(records, spec.select)
    return result.records


__all__ = [
    "CandidateEnsemble",
    "ReferenceBank",
    "build_candidate",
    "build_reference",
    "pool_fingerprint",
    "reference_from_records",
]
