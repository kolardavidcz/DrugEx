"""Color/shape scoring studies (component 2).

The default production scorer is symmetric TanimotoCombo with equal shape and
color weights (``0.5 S + 0.5 C``, physical ``[0, 2]``).  This module measures the
alternative: an asymmetric Tversky overlap over the whole
``(alpha, beta) x (w_shape, w_color)`` grid, i.e. 18 x 9 = 162 combinations.

Why a grid instead of one number
--------------------------------
Asymmetric scores are not monotone in their parameters.  Draining ``beta``
(candidate matter ignored) rewards fragment-shaped candidates; draining
``alpha`` rewards bloated blobs.  The grid exposes those two traps and the
saddle between them, so a default can be justified rather than asserted.

Reduction
---------
Each conformer pair yields one raw ``(shape, color)`` Tanimoto.  The 162
molecule-level scores are an exact algebraic re-weighting of that pair, so the
expensive part (alignment) is computed once per molecule pair, cached as NPZ,
and re-used for every grid cell and every later re-analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _david.code.conformer.core import storage
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import (
    CandidateSpec,
    MoleculeSpec,
    ReferenceSpec,
    default_candidate_spec,
    default_reference_spec,
    fingerprint,
)
from _david.code.conformer.core.errors import StudyError
from _david.code.conformer.core.provenance import build_manifest, utc_now
from _david.code.conformer.ensemble.pipeline import build_candidate, build_reference
from _david.code.conformer.prepare.standardize import prepare_molecule
from _david.code.conformer.scoring.tversky import (
    ALPHAS,
    BETAS,
    WEIGHT_GRID,
    PairwiseTversky,
    grid_scores,
    pairwise_tversky,
)

#: Deployable defaults reused by the color/shape study.
DEFAULT_REFERENCE = default_reference_spec()
DEFAULT_CANDIDATE = default_candidate_spec()

#: Thresholds whose false-positive rate is reported.
FALSE_POSITIVE_THRESHOLDS: tuple[float, ...] = (0.35, 0.25)

PAIR_NAMESPACE = "tversky_pair"


@dataclass(frozen=True)
class BankKeys:
    """Cache keys identifying one molecule's candidate pool and reference bank."""

    molecule: str
    candidate_key: str
    reference_key: str
    n_candidates: int
    n_references: int


def _cache(cache_root: str | None, enabled: bool = True) -> ArtifactCache:
    return ArtifactCache(Path(cache_root) if cache_root else None, enabled=enabled)


def _build_bank_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build one molecule's candidate pool and reference bank inside a worker."""
    molecule = MoleculeSpec(
        name=str(payload["name"]),
        smiles=str(payload["smiles"]),
        source=str(payload.get("source", "user")),
        tags=tuple(payload.get("tags", ())),
    )
    reference_spec: ReferenceSpec = payload["reference"]
    candidate_spec: CandidateSpec = payload["candidate"]
    cache = _cache(payload.get("cache_root"), bool(payload.get("cache_enabled", True)))

    prepared = prepare_molecule(molecule, reference_spec.embed, cache=cache)
    reference = build_reference(prepared, reference_spec, cache=cache)
    candidate = build_candidate(prepared, candidate_spec, cache=cache)
    if reference.selection.cache_key is None or candidate.minimize.cache_key is None:
        raise StudyError("Bank building must run with the artifact cache enabled")
    return {
        "molecule": molecule.name,
        "candidate_key": candidate.minimize.cache_key,
        "reference_key": reference.selection.cache_key,
        "n_candidates": candidate.n_records,
        "n_references": reference.n_records,
    }


def _pair_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compute (or reuse) the raw pairwise Tversky matrices for one molecule pair."""
    cache = _cache(payload.get("cache_root"), bool(payload.get("cache_enabled", True)))
    pair_key = str(payload["pair_key"])
    cached = cache.get_npz(PAIR_NAMESPACE, pair_key)
    if cached is not None:
        return {"pair_key": pair_key, "computed": False}

    candidates = storage.read_sdf(
        cache.entry_path("minimize", str(payload["candidate_key"]), "sdf")
    )
    references = storage.read_sdf(
        cache.entry_path("select", str(payload["reference_key"]), "sdf")
    )
    if not candidates or not references:
        raise StudyError(
            f"Empty cached bank for pair {pair_key} "
            f"(candidates={len(candidates)}, references={len(references)})"
        )
    pairwise = pairwise_tversky(candidates, references)
    cache.put_npz(
        PAIR_NAMESPACE,
        pair_key,
        {
            "shape": pairwise.shape,
            "color": pairwise.color,
            "reference_volumes": pairwise.reference_volumes,
            "candidate_volumes": pairwise.candidate_volumes,
        },
        metadata={
            "candidate_key": payload["candidate_key"],
            "reference_key": payload["reference_key"],
        },
    )
    return {"pair_key": pair_key, "computed": True}


def load_pairwise(
    cache: ArtifactCache,
    candidate_key: str,
    reference_key: str,
) -> PairwiseTversky:
    """Load a cached pairwise Tversky payload."""
    pair_key = fingerprint({"candidate": candidate_key, "reference": reference_key})
    payload = cache.get_npz(PAIR_NAMESPACE, pair_key)
    if payload is None:
        raise StudyError(f"Pairwise matrices missing from cache for {pair_key}")
    return PairwiseTversky(
        shape=payload["shape"],
        color=payload["color"],
        reference_volumes=payload["reference_volumes"],
        candidate_volumes=payload["candidate_volumes"],
    )


def _run_tasks(func: Any, tasks: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers > 1:
        from drugex.training.scorers.parallel import pool_context

        with pool_context().Pool(processes=workers) as pool:
            return list(pool.map(func, tasks, chunksize=1))
    return [func(task) for task in tasks]


def _grid_metadata() -> list[dict[str, Any]]:
    """One record per grid cell, in the flat order ``grid_scores`` uses."""
    return [
        {"alpha": float(alpha), "beta": float(beta)}
        for alpha, beta in zip(ALPHAS, BETAS)
    ]


def _decoy_group(tags: Sequence[str], default: str = "decoy") -> str:
    """Return the first ``group:<name>`` tag, or a generic fallback."""
    for tag in tags:
        if tag.startswith("group:"):
            return tag.split(":", 1)[1]
    return default


def run_tversky_grid(
    molecules: Sequence[MoleculeSpec],
    output_dir: Path | str,
    *,
    reference: ReferenceSpec = DEFAULT_REFERENCE,
    candidate: CandidateSpec = DEFAULT_CANDIDATE,
    cache: ArtifactCache | None = None,
    workers: int = 1,
    include_self: bool = True,
    notes: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every molecule against its own bank and every other molecule's bank.

    Returns the 162-row selectivity table and a provenance manifest.  The table
    schema is stable and consumed by :mod:`...analysis.color_shape`.
    """
    if len(molecules) < 2 and not include_self:
        raise StudyError("Cross-molecule scoring needs at least two molecules")
    cache = cache or ArtifactCache()
    if not cache.enabled:
        raise StudyError("The Tversky grid relies on the artifact cache")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bank_tasks = [
        {
            "name": spec.name,
            "smiles": spec.smiles,
            "source": spec.source,
            "tags": list(spec.tags),
            "reference": reference,
            "candidate": candidate,
            "cache_root": cache.root.as_posix(),
            "cache_enabled": cache.enabled,
        }
        for spec in molecules
    ]
    bank_rows = _run_tasks(_build_bank_worker, bank_tasks, workers)
    banks: dict[str, BankKeys] = {
        str(row["molecule"]): BankKeys(
            molecule=str(row["molecule"]),
            candidate_key=str(row["candidate_key"]),
            reference_key=str(row["reference_key"]),
            n_candidates=int(row["n_candidates"]),
            n_references=int(row["n_references"]),
        )
        for row in bank_rows
    }
    tags = {spec.name: tuple(spec.tags) for spec in molecules}

    pair_plan: list[tuple[str, str]] = []
    if include_self:
        pair_plan.extend((spec.name, spec.name) for spec in molecules)
    pair_plan.extend(
        pair
        for pair in itertools.permutations((spec.name for spec in molecules), 2)
    )
    pair_tasks = [
        {
            "pair_key": fingerprint(
                {
                    "candidate": banks[query].candidate_key,
                    "reference": banks[decoy].reference_key,
                }
            ),
            "candidate_key": banks[query].candidate_key,
            "reference_key": banks[decoy].reference_key,
            "cache_root": cache.root.as_posix(),
            "cache_enabled": cache.enabled,
        }
        for query, decoy in pair_plan
    ]
    pair_rows = _run_tasks(_pair_worker, pair_tasks, workers)
    computed = sum(1 for row in pair_rows if row["computed"])

    names = [spec.name for spec in molecules]
    index = {name: position for position, name in enumerate(names)}
    grid_shape = (len(ALPHAS), len(WEIGHT_GRID))
    self_scores = np.full((len(names), *grid_shape), np.nan, dtype=np.float64)
    decoy_scores = np.zeros((len(pair_plan), *grid_shape), dtype=np.float64)
    group_scores: dict[str, list[np.ndarray]] = {}
    worst_per_query = np.zeros((len(names), *grid_shape), dtype=np.float64)

    for position, (query, decoy) in enumerate(pair_plan):
        pairwise = load_pairwise(
            cache, banks[query].candidate_key, banks[decoy].reference_key
        )
        scores = grid_scores(pairwise)
        if query == decoy:
            self_scores[index[query]] = scores
            continue
        decoy_scores[position] = scores
        query_position = index[query]
        worst_per_query[query_position] = np.maximum(
            worst_per_query[query_position], scores
        )
        group = _decoy_group(tags.get(decoy, ()), default="decoy")
        group_scores.setdefault(group, []).append(scores)

    self_valid = self_scores[~np.isnan(self_scores).any(axis=(1, 2))]
    if self_valid.size == 0:
        raise StudyError("No molecule produced a self-recovery score")

    metadata = _grid_metadata()
    decoy_mask = np.ones(len(pair_plan), dtype=bool)
    if include_self:
        decoy_mask = np.asarray(
            [query != decoy for query, decoy in pair_plan], dtype=bool
        )
    pooled_decoys = decoy_scores[decoy_mask]

    rows: list[dict[str, Any]] = []
    for t_index, meta in enumerate(metadata):
        alpha = float(meta["alpha"])
        beta = float(meta["beta"])
        for w_index, (weight_label, w_shape, w_color) in enumerate(WEIGHT_GRID):
            self_cell = self_valid[:, t_index, w_index]
            decoy_cell = pooled_decoys[:, t_index, w_index]
            worst_cell = worst_per_query[:, t_index, w_index]
            record: dict[str, Any] = {
                "tversky_label": _tversky_label(alpha, beta),
                "scale_group": _scale_group(alpha, beta),
                "alpha": alpha,
                "beta": beta,
                "alpha_plus_beta": alpha + beta,
                "weight_label": weight_label,
                "w_shape": float(w_shape),
                "w_color": float(w_color),
                "molecules": len(names),
                "self_mean": float(np.mean(self_cell)),
                "self_p10": float(np.percentile(self_cell, 10)),
                "decoy_mean": float(np.mean(decoy_cell)),
                "decoy_p90": float(np.percentile(decoy_cell, 90)),
                "decoy_p95": float(np.percentile(decoy_cell, 95)),
                "worst_decoy_mean": float(np.mean(worst_cell)),
                "worst_decoy_p95": float(np.percentile(worst_cell, 95)),
                "margin_self_minus_decoy_mean": float(
                    np.mean(self_cell) - np.mean(decoy_cell)
                ),
                "margin_self_minus_decoy_p95": float(
                    np.mean(self_cell) - np.percentile(decoy_cell, 95)
                ),
                "margin_self_minus_worst_decoy_p95": float(
                    np.mean(self_cell) - np.percentile(worst_cell, 95)
                ),
            }
            for threshold in FALSE_POSITIVE_THRESHOLDS:
                record[f"fp_rate_decoy_ge_{threshold:.2f}"] = float(
                    np.mean(decoy_cell >= threshold)
                )
            for group, arrays in group_scores.items():
                stacked = np.stack(arrays)
                record[f"{group}_mean"] = float(np.mean(stacked[:, t_index, w_index]))
                record[f"{group}_p95"] = float(
                    np.percentile(stacked[:, t_index, w_index], 95)
                )
            rows.append(record)

    frame = pd.DataFrame(rows)
    storage.write_csv(out / "tversky_color_grid.csv", frame)
    manifest = build_manifest(
        study="tversky_color_grid",
        payload={
            "reference": reference,
            "candidate": candidate,
            "molecules": [spec.identity for spec in molecules],
        },
        inputs={"molecules": str(len(molecules)), "pairs": str(len(pair_plan))},
        extra={
            "notes": notes,
            "completed_utc": utc_now(),
            "pairs_computed": computed,
            "pairs_reused": len(pair_rows) - computed,
            "bank_rows": [row for row in bank_rows],
            "cache": cache.summary(),
        },
    )
    storage.write_json(out / "manifest.json", manifest)
    return frame, manifest


def _tversky_label(alpha: float, beta: float) -> str:
    total = round(alpha + beta)
    return f"sum{total}_{alpha:.1f}_{beta:.1f}"


def _scale_group(alpha: float, beta: float) -> str:
    total = alpha + beta
    if abs(total - 1.0) < 1e-9:
        return "alpha+beta=1 (Dice scale)"
    return "alpha+beta=2 (Tanimoto scale)"


def selectivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank grid cells by selectivity margin, best first."""
    if frame.empty:
        return frame.copy()
    columns = [
        "tversky_label",
        "alpha",
        "beta",
        "weight_label",
        "w_shape",
        "w_color",
        "self_mean",
        "decoy_mean",
        "worst_decoy_p95",
        "margin_self_minus_decoy_mean",
        "margin_self_minus_worst_decoy_p95",
        "fp_rate_decoy_ge_0.35",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values(
        "margin_self_minus_worst_decoy_p95", ascending=False
    )


__all__ = [
    "DEFAULT_CANDIDATE",
    "DEFAULT_REFERENCE",
    "FALSE_POSITIVE_THRESHOLDS",
    "BankKeys",
    "load_pairwise",
    "run_tversky_grid",
    "selectivity_summary",
]
