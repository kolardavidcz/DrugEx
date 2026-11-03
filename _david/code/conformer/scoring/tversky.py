"""Asymmetric Tversky shape/color scoring on the RDKit Gaussian overlap scale.

The mathematics is the same one used by the historical 162-combination
selectivity audit.  It is kept in a small, testable module so the study runners
can vectorize a whole ``(alpha, beta) x (w_shape, w_color)`` grid without
re-deriving geometry for every combination.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign
from drugex.training.scorers.rocs_rdkit import (
    _AXIS_SIGN_FLIPS,
    _canonicalize_conformer_inplace,
    _flip_conformer,
)

# 9 Dice-scale settings (alpha + beta = 1) and 9 Tanimoto-scale settings
# (alpha + beta = 2), matching the published selectivity audit.
TVERSKY_GRID: tuple[tuple[str, str, float, float], ...] = (
    *(
        (f"sum1_{alpha:.1f}_{1.0 - alpha:.1f}", "alpha+beta=1 (Dice scale)", round(alpha, 2), round(1.0 - alpha, 2))
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    ),
    *(
        (f"sum2_{alpha:.1f}_{2.0 - alpha:.1f}", "alpha+beta=2 (Tanimoto scale)", round(alpha, 2), round(2.0 - alpha, 2))
        for alpha in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
    ),
)

WEIGHT_GRID: tuple[tuple[str, float, float], ...] = tuple(
    (f"{shape:.1f}S_{1.0 - shape:.1f}C", round(shape, 2), round(1.0 - shape, 2))
    for shape in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
)

ALPHAS = np.asarray([entry[2] for entry in TVERSKY_GRID], dtype=np.float64)
BETAS = np.asarray([entry[3] for entry in TVERSKY_GRID], dtype=np.float64)


@dataclass(frozen=True)
class SelfVolumes:
    """Self overlap and self feature overlap for one 3D conformer."""

    shape: float
    color: float


@dataclass(frozen=True)
class PairwiseTversky:
    """Pairwise raw Tanimoto matrices before asymmetric re-weighting."""

    shape: np.ndarray
    color: np.ndarray
    reference_volumes: np.ndarray
    candidate_volumes: np.ndarray

    @property
    def n_references(self) -> int:
        return int(self.shape.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.shape.shape[1])


def self_volumes(mol: Chem.Mol) -> SelfVolumes:
    """Return ``(sov, sof)`` for the conformer in ``mol``."""
    options = rdShapeAlign.ShapeInputOptions()
    options.useColors = True
    info = rdShapeAlign.PrepareConformer(mol, -1, options)
    return SelfVolumes(shape=float(info.sov), color=float(info.sof))


def _canonical_copy(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.Mol(mol)
    _canonicalize_conformer_inplace(copy, -1)
    return copy


def align_pair_raw(candidate: Chem.Mol, reference: Chem.Mol) -> tuple[float, float]:
    """Return ``(shape, color)`` Tanimoto at the best equal-weight pose.

    Inputs must already be canonicalized.  All four proper-axis flips are tried
    exactly as in the production scorer.
    """
    best_combo = -1.0
    best = (0.0, 0.0)
    for signs in _AXIS_SIGN_FLIPS:
        probe = _flip_conformer(candidate, -1, signs)
        try:
            shape, color = rdShapeAlign.AlignMol(
                reference,
                probe,
                refConfId=-1,
                probeConfId=-1,
                useColors=True,
                opt_param=0.5,
            )
        except (RuntimeError, ValueError):
            continue
        combo = 0.5 * (float(shape) + float(color))
        if combo > best_combo:
            best_combo = combo
            best = (float(shape), float(color))
    return best


def pairwise_tversky(
    candidates: Sequence[Chem.Mol],
    references: Sequence[Chem.Mol],
) -> PairwiseTversky:
    """Compute the raw Tanimoto matrices and self-volumes for one bank/pool pair."""
    if not candidates or not references:
        raise ValueError("Both candidates and references must be non-empty")
    canonical_refs = [_canonical_copy(mol) for mol in references]
    canonical_cands = [_canonical_copy(mol) for mol in candidates]
    shape = np.zeros((len(canonical_refs), len(canonical_cands)), dtype=np.float64)
    color = np.zeros_like(shape)
    for row, reference in enumerate(canonical_refs):
        for column, candidate in enumerate(canonical_cands):
            shape[row, column], color[row, column] = align_pair_raw(candidate, reference)
    ref_volumes = np.asarray(
        [(volumes.shape, volumes.color) for volumes in map(self_volumes, canonical_refs)],
        dtype=np.float64,
    )
    cand_volumes = np.asarray(
        [(volumes.shape, volumes.color) for volumes in map(self_volumes, canonical_cands)],
        dtype=np.float64,
    )
    return PairwiseTversky(
        shape=shape,
        color=color,
        reference_volumes=ref_volumes,
        candidate_volumes=cand_volumes,
    )


def tversky_from_tanimoto(
    tanimoto: np.ndarray,
    reference_self: np.ndarray,
    candidate_self: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Convert Gaussian Tanimoto overlaps into asymmetric Tversky scores.

    Derivation: Tanimoto ``T = O / (S_r + S_c - O)``.  Solving for the overlap
    ``O`` and re-normalizing with the asymmetric denominator
    ``O + a (S_r - O) + b (S_c - O)`` gives the Tversky score directly from the
    publicly exposed Tanimoto and self-overlaps.
    """
    overlap = np.where(
        tanimoto > 0.0,
        tanimoto * (reference_self + candidate_self) / (1.0 + tanimoto),
        0.0,
    )
    denominator = (
        overlap
        + alpha * (reference_self - overlap)
        + beta * (candidate_self - overlap)
    )
    return np.where(
        denominator > 1e-12, np.clip(overlap / denominator, 0.0, 1.0), 0.0
    )


def grid_scores(pairwise: PairwiseTversky) -> np.ndarray:
    """Return the full ``(18 Tversky) x (9 weight)`` score tensor.

    The tensor has shape ``(18, 9)``; every cell is the maximum over the
    ``K x M`` reference/candidate conformer cross product, which is how a
    molecule-level ROCS score is defined.
    """
    n_refs = pairwise.n_references
    n_cands = pairwise.n_candidates
    shape = pairwise.shape.ravel()[None, :]
    color = pairwise.color.ravel()[None, :]

    reference_shape = np.repeat(pairwise.reference_volumes[:, 0], n_cands)[None, :]
    reference_color = np.repeat(pairwise.reference_volumes[:, 1], n_cands)[None, :]
    candidate_shape = np.tile(pairwise.candidate_volumes[:, 0], n_refs)[None, :]
    candidate_color = np.tile(pairwise.candidate_volumes[:, 1], n_refs)[None, :]

    overlap_shape = np.where(
        shape > 0.0,
        shape * (reference_shape + candidate_shape) / (1.0 + shape),
        0.0,
    )
    overlap_color = np.where(
        color > 0.0,
        color * (reference_color + candidate_color) / (1.0 + color),
        0.0,
    )

    alphas = ALPHAS[:, None]
    betas = BETAS[:, None]
    denominator_shape = (
        overlap_shape
        + alphas * (reference_shape - overlap_shape)
        + betas * (candidate_shape - overlap_shape)
    )
    tversky_shape = np.where(
        denominator_shape > 1e-12,
        np.clip(overlap_shape / denominator_shape, 0.0, 1.0),
        0.0,
    )
    denominator_color = (
        overlap_color
        + alphas * (reference_color - overlap_color)
        + betas * (candidate_color - overlap_color)
    )
    tversky_color = np.where(
        denominator_color > 1e-12,
        np.clip(overlap_color / denominator_color, 0.0, 1.0),
        0.0,
    )

    weights_shape = np.asarray([entry[1] for entry in WEIGHT_GRID], dtype=np.float64)[None, :, None]
    weights_color = np.asarray([entry[2] for entry in WEIGHT_GRID], dtype=np.float64)[None, :, None]
    tensor = (
        weights_shape * tversky_shape[:, None, :]
        + weights_color * tversky_color[:, None, :]
    )
    return np.max(tensor, axis=2)


def _grid_index(alpha: float, beta: float, shape_weight: float, color_weight: float) -> tuple[int, int]:
    """Locate the nearest grid cell for a requested Tversky/weight setting.

    ``ALPHAS`` alone is ambiguous: ``alpha=0.3`` appears both as a Dice-scale
    ``(0.3, 0.7)`` and a Tanimoto-scale ``(0.3, 1.7)`` cell.  Both coordinates
    must therefore be matched, and an off-grid request is rejected instead of
    being silently snapped to a different asymmetry.
    """
    tversky_distance = np.abs(ALPHAS - alpha) + np.abs(BETAS - beta)
    tversky_index = int(np.argmin(tversky_distance))
    if tversky_distance[tversky_index] > 1e-6:
        raise ValueError(
            f"Tversky (alpha={alpha:g}, beta={beta:g}) is not on the 18-cell grid "
            f"{[(a, b) for a, b in zip(ALPHAS, BETAS)]}"
        )
    total_weight = shape_weight + color_weight
    if total_weight <= 0:
        raise ValueError("shape_weight + color_weight must be positive")
    target_shape_weight = shape_weight / total_weight
    weights = np.asarray([entry[1] for entry in WEIGHT_GRID], dtype=np.float64)
    weight_distance = np.abs(weights - target_shape_weight)
    weight_index = int(np.argmin(weight_distance))
    if weight_distance[weight_index] > 1e-6:
        raise ValueError(
            f"Shape weight {target_shape_weight:g} is not on the 9-cell grid "
            f"{[entry[1] for entry in WEIGHT_GRID]}"
        )
    return tversky_index, weight_index


def score_tversky(
    candidates: Sequence[Chem.Mol],
    references: Sequence[Chem.Mol],
    alpha: float,
    beta: float,
    shape_weight: float,
    color_weight: float,
) -> tuple[float, float, float]:
    """Score one candidate pool against one reference bank at a single setting."""
    tversky_index, weight_index = _grid_index(alpha, beta, shape_weight, color_weight)
    pairwise = pairwise_tversky(candidates, references)
    tensor = grid_scores(pairwise)
    combo = float(tensor[tversky_index, weight_index])
    shape = float(pairwise.shape.max())
    color = float(pairwise.color.max())
    return combo, shape, color


__all__ = [
    "ALPHAS",
    "BETAS",
    "PairwiseTversky",
    "SelfVolumes",
    "TVERSKY_GRID",
    "WEIGHT_GRID",
    "align_pair_raw",
    "grid_scores",
    "pairwise_tversky",
    "score_tversky",
    "self_volumes",
    "tversky_from_tanimoto",
]
