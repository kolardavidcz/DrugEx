"""Molecule-level ROCS scoring with explicit scale and provenance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time

from rdkit import Chem

from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import ScoreScale, ScoreSpec, ScoreVariant
from _david.code.conformer.ensemble.pipeline import pool_fingerprint
from _david.code.conformer.scoring.tversky import score_tversky
from _david.code.rocs import score_pair


@dataclass(frozen=True)
class EnsembleScore:
    """Best molecule-level score for one candidate pool against one bank."""

    combo: float
    shape: float
    color: float
    best_candidate: int
    best_reference: int
    n_candidate_conformers: int
    n_reference_conformers: int
    variant: ScoreVariant
    scale: ScoreScale
    seconds: float
    cache_key: str | None = None


def _scale(value: float, scale: ScoreScale) -> float:
    return value * 0.5 if scale is ScoreScale.NORMALIZED else value


def score_ensemble(
    candidates: Sequence[Chem.Mol],
    references: Sequence[Chem.Mol],
    spec: ScoreSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
    candidate_key: str | None = None,
    reference_key: str | None = None,
) -> EnsembleScore:
    """Score every candidate/reference conformer pair and keep the best.

    Tanimoto uses the authoritative production scorer.  Tversky uses the
    asymmetric Gaussian-overlap formulation in :mod:`scoring.tversky`.
    """
    spec = spec or ScoreSpec()
    if not candidates:
        raise ValueError("Cannot score an empty candidate ensemble")
    if not references:
        raise ValueError("Cannot score against an empty reference bank")

    candidate_key = candidate_key or pool_fingerprint(tuple(candidates))
    reference_key = reference_key or pool_fingerprint(tuple(references))
    payload = {
        "candidates": candidate_key,
        "references": reference_key,
        "score": spec,
    }
    key = cache.key("score", payload) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get_json("score", key)
        if cached is not None:
            return EnsembleScore(
                combo=float(cached["combo"]),
                shape=float(cached["shape"]),
                color=float(cached["color"]),
                best_candidate=int(cached["best_candidate"]),
                best_reference=int(cached["best_reference"]),
                n_candidate_conformers=int(cached["n_candidate_conformers"]),
                n_reference_conformers=int(cached["n_reference_conformers"]),
                variant=ScoreVariant(cached["variant"]),
                scale=ScoreScale(cached["scale"]),
                # A cache hit costs nothing in this run; reporting the original
                # latency would corrupt any cost/accuracy figure built on it.
                seconds=0.0,
                cache_key=key,
            )

    started = time.perf_counter()
    if spec.variant is ScoreVariant.TANIMOTO:
        best = (0.0, 0.0, 0.0)
        best_candidate = 0
        best_reference = 0
        for candidate_index, candidate in enumerate(candidates):
            for reference_index, reference in enumerate(references):
                combo, shape, color = score_pair(candidate, reference)
                if combo > best[0]:
                    best = (combo, shape, color)
                    best_candidate = candidate_index
                    best_reference = reference_index
        combo, shape, color = best
    elif spec.variant is ScoreVariant.TVERSKY:
        combo, shape, color = score_tversky(
            candidates,
            references,
            alpha=spec.alpha,
            beta=spec.beta,
            shape_weight=spec.shape_weight,
            color_weight=spec.color_weight,
        )
        best_candidate = 0
        best_reference = 0
    else:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"Unsupported score variant: {spec.variant!r}")

    result = EnsembleScore(
        combo=_scale(combo, spec.scale),
        shape=shape,
        color=color,
        best_candidate=best_candidate,
        best_reference=best_reference,
        n_candidate_conformers=len(candidates),
        n_reference_conformers=len(references),
        variant=spec.variant,
        scale=spec.scale,
        seconds=time.perf_counter() - started,
        cache_key=key,
    )
    if cache is not None and key is not None:
        cache.put_json(
            "score",
            key,
            {
                "combo": result.combo,
                "shape": result.shape,
                "color": result.color,
                "best_candidate": result.best_candidate,
                "best_reference": result.best_reference,
                "n_candidate_conformers": result.n_candidate_conformers,
                "n_reference_conformers": result.n_reference_conformers,
                "variant": result.variant.value,
                "scale": result.scale.value,
                "seconds": result.seconds,
            },
        )
    return result


def score_self_recovery(
    candidates: Sequence[Chem.Mol],
    references: Sequence[Chem.Mol],
    spec: ScoreSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
    candidate_key: str | None = None,
    reference_key: str | None = None,
) -> EnsembleScore:
    """Convenience alias used by self-recovery studies."""
    return score_ensemble(
        candidates,
        references,
        spec,
        cache=cache,
        candidate_key=candidate_key,
        reference_key=reference_key,
    )


__all__ = ["EnsembleScore", "score_ensemble", "score_self_recovery"]
