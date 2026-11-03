"""Reference-bank selection: energy, Butina RMSD, balanced, and diverse."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import time

from rdkit import Chem

from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import SelectSpec, SelectionMode
from _david.code.conformer.core.errors import SelectionError
from _david.code.reference import (
    balanced_reference_order,
    diverse_order,
    select_conformers,
)


@dataclass(frozen=True)
class SelectionResult:
    """Ordered reference bank plus the clustering statistics that justify it."""

    records: tuple[Chem.Mol, ...]
    spec: SelectSpec
    indices: tuple[int, ...]
    pool_size: int
    cluster_count: int
    per_isomer_clusters: tuple[int, ...]
    retained: int
    seconds: float
    cache_key: str | None = None

    @property
    def n_records(self) -> int:
        return len(self.records)


def _identity(record: Chem.Mol) -> str:
    if record.HasProp("EnumeratedIsomerSMILES"):
        return record.GetProp("EnumeratedIsomerSMILES")
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(record)), isomericSmiles=True)


def _energy(record: Chem.Mol) -> float:
    return (
        float(record.GetDoubleProp("Energy_kcal_mol"))
        if record.HasProp("Energy_kcal_mol")
        else 0.0
    )


def cluster_statistics(
    records: tuple[Chem.Mol, ...],
    cutoff: float,
) -> tuple[int, tuple[int, ...]]:
    """Return total and per-isomer Butina cluster counts at ``cutoff``."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[_identity(record)].append(index)
    per_isomer: list[int] = []
    for indices in groups.values():
        if len(indices) < 2:
            per_isomer.append(len(indices))
            continue
        representatives = select_conformers([records[i] for i in indices], cutoff)
        per_isomer.append(len(representatives))
    return sum(per_isomer), tuple(per_isomer)


def _energy_order(records: tuple[Chem.Mol, ...]) -> list[int]:
    """Round-robin across isomers, lowest energy first inside each isomer."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[_identity(record)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (_energy(records[i]), i))
    ordered: list[int] = []
    round_index = 0
    while True:
        added = False
        for indices in groups.values():
            if round_index < len(indices):
                ordered.append(indices[round_index])
                added = True
        if not added:
            break
        round_index += 1
    return ordered


def _cap(order: list[int], cap: int) -> list[int]:
    if cap == 0:
        return list(order)
    return list(order[:cap])


def _select_uncached(records: tuple[Chem.Mol, ...], spec: SelectSpec) -> SelectionResult:
    started = time.perf_counter()
    if not records:
        raise SelectionError("Cannot select from an empty conformer pool")
    if spec.mode is SelectionMode.ENERGY:
        order = _energy_order(records)
    elif spec.mode is SelectionMode.BUTINA:
        order = select_conformers(records, spec.rmsd_cutoff)
    elif spec.mode is SelectionMode.BALANCED:
        clustered = select_conformers(records, spec.rmsd_cutoff)
        order = balanced_reference_order(records, clustered)
    elif spec.mode is SelectionMode.DIVERSE:
        order = diverse_order(records, prefix=max(spec.cap, 1))
    else:  # pragma: no cover - exhaustive over the enum
        raise SelectionError(f"Unsupported selection mode: {spec.mode!r}")

    selected = _cap(order, spec.cap)
    if not selected:
        raise SelectionError("Selection retained zero conformers")

    cluster_count, per_isomer = cluster_statistics(records, spec.rmsd_cutoff)
    return SelectionResult(
        records=tuple(records[index] for index in selected),
        spec=spec,
        indices=tuple(selected),
        pool_size=len(records),
        cluster_count=cluster_count,
        per_isomer_clusters=per_isomer,
        retained=len(selected),
        seconds=time.perf_counter() - started,
    )


def select_ensemble(
    records: tuple[Chem.Mol, ...],
    spec: SelectSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
    cache_payload: dict[str, object] | None = None,
) -> SelectionResult:
    """Reduce a pool to a reference bank, reusing a cached selection."""
    spec = spec or SelectSpec()
    if cache_payload is None:
        # Never key a selection on the pool *size* alone: two different pools of
        # equal length would share a cache entry and silently swap banks.
        from _david.code.conformer.ensemble.pipeline import pool_fingerprint

        cache_payload = {"pool": pool_fingerprint(records)}
    payload = {
        "input": cache_payload,
        "select": spec,
    }
    key = cache.key("select", payload) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get_sdf("select", key)
        if cached is not None:
            meta = cache.read_meta("select", key).get("metadata", {})
            return SelectionResult(
                records=tuple(cached),
                spec=spec,
                indices=tuple(int(value) for value in meta.get("indices", [])),
                pool_size=int(meta.get("pool_size", len(cached))),
                cluster_count=int(meta.get("cluster_count", 0)),
                per_isomer_clusters=tuple(
                    int(value) for value in meta.get("per_isomer_clusters", [])
                ),
                retained=int(meta.get("retained", len(cached))),
                seconds=0.0,
                cache_key=key,
            )

    result = _select_uncached(records, spec)
    if cache is not None and key is not None:
        cache.put_sdf(
            "select",
            key,
            result.records,
            metadata={
                "indices": list(result.indices),
                "pool_size": result.pool_size,
                "cluster_count": result.cluster_count,
                "per_isomer_clusters": list(result.per_isomer_clusters),
                "retained": result.retained,
                "seconds": result.seconds,
            },
        )
        return SelectionResult(**{**result.__dict__, "cache_key": key})
    return result


def selection_overlap(left: SelectionResult, right: SelectionResult) -> float:
    """Jaccard overlap of two banks selected from the same pool.

    Overlap is computed on pool indices, so two banks that choose different
    conformers of the same isomer are correctly reported as different.  If the
    two selections came from different pools the comparison is meaningless and
    an empty result is returned.
    """
    if left.pool_size != right.pool_size:
        return 0.0
    left_set = set(left.indices)
    right_set = set(right.indices)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def selection_histogram(records: tuple[Chem.Mol, ...], indices: tuple[int, ...]) -> dict[str, int]:
    """Count retained conformers per stereoisomer identity."""
    return dict(Counter(_identity(records[index]) for index in indices))


__all__ = [
    "SelectionResult",
    "cluster_statistics",
    "select_ensemble",
    "selection_histogram",
    "selection_overlap",
]
