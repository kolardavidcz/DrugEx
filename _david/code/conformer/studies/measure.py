"""The one measurement every study shares: build a bank, build a pool, score it.

Studies differ in *which* arms they vary, not in how a single cell is measured.
Keeping that measurement in one place means the optimization-method, clustering,
and scoring tables are directly comparable: same preparation, same seeds, same
lattice of counts, same score scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import ScoreScale, ScoreSpec
from _david.code.conformer.ensemble.pipeline import build_candidate, build_reference
from _david.code.conformer.prepare.registry import resolve_molecule
from _david.code.conformer.prepare.standardize import prepare_molecule
from _david.code.conformer.scoring.rocs import score_ensemble
from _david.code.conformer.studies.base import CellRequest

#: Default score scale for study tables.  ``NORMALIZED`` reports
#: ``0.5 * (shape + color)`` on ``[0, 1]``, which is the RL reward axis; the
#: physical TanimotoCombo on ``[0, 2]`` remains recoverable as ``2 * combo``.
STUDY_SCALE = ScoreScale.NORMALIZED


def _reference_method_label(request: CellRequest) -> str:
    return request.arm.reference.minimize.label


def _candidate_method_label(request: CellRequest) -> str:
    return request.arm.candidate.minimize.label


@dataclass(frozen=True)
class MeasureExecutor:
    """Execute one cell end-to-end and return a flat, comparable row.

    The object holds only primitives, so it pickles into a worker process.  All
    chemistry is rebuilt inside :meth:`__call__` from the :class:`MoleculeSpec`.
    """

    cache_root: str | None = None
    cache_enabled: bool = True
    score: ScoreSpec = ScoreSpec(scale=STUDY_SCALE)
    register: bool = False

    def _cache(self) -> ArtifactCache:
        from pathlib import Path

        return ArtifactCache(
            Path(self.cache_root) if self.cache_root else None,
            enabled=self.cache_enabled,
        )

    def __call__(self, request: CellRequest) -> dict[str, Any]:
        cache = self._cache()
        started = time.perf_counter()
        prepared = prepare_molecule(request.molecule, request.arm.reference.embed, cache=cache)
        if self.register:
            resolve_molecule(
                request.molecule.name,
                request.molecule.smiles,
                source=request.molecule.source,
                embed=request.arm.reference.embed,
                cache=cache,
            )

        reference = build_reference(prepared, request.arm.reference, cache=cache)
        candidate = build_candidate(prepared, request.arm.candidate, cache=cache)
        score = score_ensemble(
            candidate.records,
            reference.records,
            self.score,
            cache=cache,
            candidate_key=candidate.embed.cache_key or candidate.minimize.cache_key,
            reference_key=reference.selection.cache_key,
        )
        elapsed = time.perf_counter() - started

        return {
            "molecule": request.molecule.name,
            "input_smiles": request.molecule.smiles,
            "canonical_smiles": prepared.canonical_smiles,
            "heavy_atoms": prepared.properties.get("heavy_atoms"),
            "rotatable_bonds": prepared.properties.get("rotatable_bonds"),
            "isomers": prepared.n_isomers,
            "isomer_upper_bound": prepared.properties.get("isomer_upper_bound"),
            "arm": request.arm.name,
            "reference_method": _reference_method_label(request),
            "candidate_method": _candidate_method_label(request),
            "reference_minimized": request.arm.reference.minimize.enabled,
            "candidate_minimized": request.arm.candidate.minimize.enabled,
            "selection_mode": request.arm.reference.select.mode.value,
            "rmsd_cutoff": request.arm.reference.select.rmsd_cutoff,
            "reference_cap": request.arm.reference.select.cap,
            "reference_embed_attempts": request.arm.reference.embed.attempts,
            "candidate_attempts": request.arm.candidate.embed.attempts,
            "reference_pool": reference.minimize.n_records,
            "reference_converged": reference.minimize.converged_count,
            "reference_window_removed": reference.minimize.window_removed,
            "cluster_count": reference.cluster_count,
            "retained_references": reference.n_records,
            "actual_candidates": candidate.n_records,
            "combo": score.combo,
            "shape": score.shape,
            "color": score.color,
            "best_reference": score.best_reference,
            "best_candidate": score.best_candidate,
            "score_scale": self.score.scale.value,
            "reference_seconds": reference.embed.seconds + reference.minimize.seconds,
            "selection_seconds": reference.selection.seconds,
            "candidate_seconds": candidate.embed.seconds + candidate.minimize.seconds,
            "score_seconds": score.seconds,
            "total_seconds": elapsed,
            "repeat": request.repeat,
            "reference_cache_key": reference.selection.cache_key,
            "reference_warnings": "|".join(prepared.warnings),
        }


__all__ = ["MeasureExecutor", "STUDY_SCALE"]
