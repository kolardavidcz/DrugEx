"""Force-field relaxation and per-stereoisomer energy-window filtering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time

from rdkit import Chem

from _david.code.chemistry import Optimization, optimize_conformers
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import MinimizeSpec
from _david.code.conformer.core.errors import MinimizationError


@dataclass(frozen=True)
class MinimizeResult:
    """Relaxed conformer pool plus energy accounting."""

    records: tuple[Chem.Mol, ...]
    spec: MinimizeSpec
    input_count: int
    converged_count: int
    window_removed: int
    lowest_energy: float | None
    statuses: tuple[int, ...]
    seconds: float
    cache_key: str | None = None

    @property
    def n_records(self) -> int:
        return len(self.records)


def _isomer_key(record: Chem.Mol) -> str:
    """Group by enumerated stereoisomer identity, never across identities."""
    if record.HasProp("EnumeratedIsomerSMILES"):
        return record.GetProp("EnumeratedIsomerSMILES")
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(record)), isomericSmiles=True)


def _apply_energy_window(
    records: list[Chem.Mol],
    window: float | None,
) -> tuple[list[Chem.Mol], int, float | None]:
    """Keep conformers within ``window`` of their own isomer's minimum energy."""
    if window is None:
        return records, 0, None

    groups: dict[str, list[Chem.Mol]] = defaultdict(list)
    for record in records:
        groups[_isomer_key(record)].append(record)

    kept: list[Chem.Mol] = []
    removed = 0
    global_min: float | None = None
    for group in groups.values():
        energies = [
            float(record.GetDoubleProp("Energy_kcal_mol"))
            for record in group
            if record.HasProp("Energy_kcal_mol")
        ]
        if not energies:
            kept.extend(group)
            continue
        baseline = min(energies)
        global_min = baseline if global_min is None else min(global_min, baseline)
        for record in group:
            if not record.HasProp("Energy_kcal_mol"):
                kept.append(record)
                continue
            energy = float(record.GetDoubleProp("Energy_kcal_mol"))
            if energy - baseline <= window:
                record.SetDoubleProp("DeltaE_kcal_mol", energy - baseline)
                kept.append(record)
            else:
                removed += 1
    return kept, removed, global_min


def _minimize_uncached(records: tuple[Chem.Mol, ...], spec: MinimizeSpec) -> MinimizeResult:
    """Run the configured force field and filter by the energy window."""
    started = time.perf_counter()
    if not spec.enabled:
        relaxed = [Chem.Mol(record) for record in records]
        for record in relaxed:
            record.SetProp("ForceField", "none")
        kept, removed, _ = _apply_energy_window(relaxed, spec.energy_window)
        return MinimizeResult(
            records=tuple(kept),
            spec=spec,
            input_count=len(records),
            converged_count=len(records),
            window_removed=removed,
            lowest_energy=None,
            statuses=(),
            seconds=time.perf_counter() - started,
        )

    profile: Optimization = spec.profile  # type: ignore[assignment]
    try:
        optimized, statuses = optimize_conformers(records, profile)
    except (RuntimeError, ValueError) as exc:
        raise MinimizationError(f"{profile.name} failed: {exc}") from exc
    if not optimized:
        raise MinimizationError(
            f"{profile.name} converged zero of {len(records)} conformers "
            f"(statuses={sorted(set(statuses))})"
        )

    kept, removed, global_min = _apply_energy_window(optimized, spec.energy_window)
    if not kept:
        raise MinimizationError("Energy window removed every converged conformer")
    return MinimizeResult(
        records=tuple(kept),
        spec=spec,
        input_count=len(records),
        converged_count=len(optimized),
        window_removed=removed,
        lowest_energy=global_min,
        statuses=tuple(statuses),
        seconds=time.perf_counter() - started,
    )


def minimize_ensemble(
    records: tuple[Chem.Mol, ...],
    spec: MinimizeSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
    cache_payload: dict[str, object] | None = None,
) -> MinimizeResult:
    """Relax a conformer pool, reusing a cached result when possible.

    ``cache_payload`` should uniquely identify the *input* pool.  Callers
    normally pass ``{"embed_key": result.cache_key}``; when the embed step was
    not cached, pass a fingerprint of the pool metadata instead.
    """
    spec = spec or MinimizeSpec()
    payload = {"input": cache_payload or {"count": len(records)}, "minimize": spec}
    key = cache.key("minimize", payload) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get_sdf("minimize", key)
        if cached is not None:
            meta = cache.read_meta("minimize", key).get("metadata", {})
            return MinimizeResult(
                records=tuple(cached),
                spec=spec,
                input_count=int(meta.get("input_count", len(records))),
                converged_count=int(meta.get("converged_count", len(cached))),
                window_removed=int(meta.get("window_removed", 0)),
                lowest_energy=(
                    None
                    if meta.get("lowest_energy") is None
                    else float(meta["lowest_energy"])
                ),
                statuses=tuple(int(value) for value in meta.get("statuses", [])),
                seconds=float(meta.get("seconds", 0.0)),
                cache_key=key,
            )

    result = _minimize_uncached(records, spec)
    if cache is not None and key is not None:
        cache.put_sdf(
            "minimize",
            key,
            result.records,
            metadata={
                "input_count": result.input_count,
                "converged_count": result.converged_count,
                "window_removed": result.window_removed,
                "lowest_energy": result.lowest_energy,
                "statuses": list(result.statuses),
                "seconds": result.seconds,
            },
        )
        return MinimizeResult(**{**result.__dict__, "cache_key": key})
    return result


__all__ = ["MinimizeResult", "minimize_ensemble"]
