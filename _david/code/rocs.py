"""Shared RDKit ROCS shape-and-color helpers and persistent whole-call parallel scorer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from typing import Any

from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.rocs_rdkit import (
    _AXIS_SIGN_FLIPS,
    _canonicalize_conformer_inplace,
    _flip_conformer,
)
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from _david.code.chemistry import (
    Optimization,
    enable_double_precision_mol_pickling,
)

NAMED_OPTIMIZATION_METHODS: dict[str, Optimization] = {
    "etkdg": Optimization("none"),
    "uff": Optimization("UFF"),
    "mmff_pocket": Optimization("MMFF94s", 6.5),
    "mmff_high_dielectric": Optimization("MMFF94s", 78.4),
}

_WORKER_SCORER: Any = None


def has_color(mol: Chem.Mol) -> bool:
    """Return True when RDKit's native 5-pattern pharmacophore color model finds features."""
    probe = Chem.Mol(mol)
    probe.RemoveAllConformers()
    options = rdShapeAlign.ShapeInputOptions()
    options.useColors = True
    return float(rdShapeAlign.PrepareConformer(mol, -1, options).sof) > 0.0


def score_pair(
    candidate: Chem.Mol,
    reference: Chem.Mol,
) -> tuple[float, float, float]:
    """Return (normalized_combo, shape_tanimoto, color_tanimoto) from the winning pose.

    Replicates the exact inertial-frame canonicalization and 4-sign proper-rotation
    pose guard used by ``drugex.training.scorers.rocs_rdkit._score_single_reference``.
    """
    reference_copy = Chem.Mol(reference)
    candidate_copy = Chem.Mol(candidate)
    _canonicalize_conformer_inplace(reference_copy, -1)
    _canonicalize_conformer_inplace(candidate_copy, -1)
    best: tuple[float, float, float] = (0.0, 0.0, 0.0)
    for signs in _AXIS_SIGN_FLIPS:
        probe = _flip_conformer(candidate_copy, -1, signs)
        try:
            shape, color = rdShapeAlign.AlignMol(
                reference_copy,
                probe,
                refConfId=-1,
                probeConfId=-1,
                useColors=True,
                opt_param=0.5,
            )
        except (RuntimeError, ValueError):
            continue
        combo = 0.5 * (float(shape) + float(color))
        if combo > best[0]:
            best = (combo, float(shape), float(color))
    return best


def _init_whole_call_worker(
    reference_path: str,
    setting: Mapping[str, Any],
) -> None:
    """Initialize a persistent single-threaded ROCS scorer inside a worker process."""
    global _WORKER_SCORER
    from _david.code.environment import create_rocs_scorer

    enable_double_precision_mol_pickling()
    method_key = str(setting.get("method", "etkdg"))
    optimization = (
        setting["optimization"]
        if isinstance(setting.get("optimization"), Optimization)
        else NAMED_OPTIMIZATION_METHODS[method_key]
    )
    minimize = bool(
        setting.get("minimize_candidates", method_key != "etkdg" and optimization.method != "none")
    )
    _WORKER_SCORER = create_rocs_scorer(
        Path(reference_path),
        max_conformers=int(setting.get("candidate_attempts", setting.get("max_conformers", 20))),
        max_isomers=int(setting.get("max_isomers", 4)),
        random_seed=int(setting.get("random_seed", 42)),
        minimize_candidates=minimize,
        optimization=optimization,
        timeout=int(setting.get("timeout", 10)),
        conformer_threads=1,
        n_jobs=1,
    )
    _WORKER_SCORER.conformer_generator.embedding_rmsd = setting.get("embedding_rmsd", None)
    _WORKER_SCORER.conformer_generator.protonate = bool(setting.get("protonate", False))


def _score_single_smiles(smiles: str) -> float:
    """Score one unique candidate SMILES in a persistent worker."""
    return float(_WORKER_SCORER.getScores([smiles])[0, 0])


class WholeCallScorer(Scorer):
    """Persistent worker-pool adapter that parallelizes whole per-molecule ROCS calls.

    Unlike ``RDKitROCSScorer``'s native batching (which generates and minimizes all
    candidate conformers serially in the main process before parallelizing only the
    3D Gaussian overlays), ``WholeCallScorer`` dispatches each unique SMILES to a
    persistent worker process that performs conformer generation, optional force-field
    minimization, and ROCS alignment end-to-end with an isolated per-molecule
    ``ETKDGv3`` iteration budget.
    """

    def __init__(
        self,
        base: Scorer,
        reference_path: Path | str,
        setting: Mapping[str, Any],
        workers: int,
    ) -> None:
        super().__init__(base.modifier)
        self.base = base
        self._pool = ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_whole_call_worker,
            initargs=(str(Path(reference_path).resolve()), dict(setting)),
        )
        list(self._pool.map(_score_single_smiles, ["C"]))

    def close(self) -> None:
        """Shut down the underlying worker pool."""
        self._pool.shutdown(wait=True)

    def __enter__(self) -> WholeCallScorer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def getKey(self) -> Any:
        return self.base.getKey()

    def getScores(
        self,
        mols: Sequence[Chem.Mol | str | None],
        frags: Any = None,
    ) -> np.ndarray:
        smiles = self.base._convert_to_smiles(mols)
        unique, mapping = self.base._deduplicate_smiles(smiles)
        scores = np.zeros((len(mols), 1), dtype=float)
        if not unique:
            return scores
        for unique_index, value in enumerate(
            self._pool.map(_score_single_smiles, unique, chunksize=1)
        ):
            for row_index in mapping[unique_index]:
                scores[row_index, 0] = value
        return scores
