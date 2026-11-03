"""Authoritative RDKit ROCS shape-and-color alignment baseline.

Focuses strictly on canonical TanimotoCombo (Shape + Color in [0, 2]).
Experimental 162-combination multi-metric Tversky and custom Gaussian re-weighters
have been isolated into `_david/code/to_do/rocs_multimetric.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
import multiprocessing as mp
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.parallel import pool_context
from drugex.training.scorers.rocs_rdkit import (
    RDKitROCSScorer,
    _AXIS_SIGN_FLIPS,
    _canonicalize_conformer_inplace,
    _flip_conformer,
)
from _david.code.chemistry import (
    CandidateConformerGenerator,
    Optimization,
    enable_double_precision_mol_pickling,
)
from _david.code.schema import std
from _david.code.utils import HardwareManager

if TYPE_CHECKING:
    from _david.code.config import PipelineConfig

# Authoritative standard optimization method presets
NAMED_OPTIMIZATION_METHODS: dict[str, Optimization] = {
    "etkdg": Optimization("none", 1.0),
    "uff": Optimization("UFF", 1.0),
    "mmff_vacuum": Optimization("MMFF94", 1.0),
    "mmff_high_dielectric": Optimization("MMFF94s", 78.4),
}

_WORKER_SCORER: Any = None


def has_color(mol: Chem.Mol) -> bool:
    """Return True when RDKit's native 5-pattern pharmacophore color model finds features."""
    probe = Chem.Mol(mol)
    if probe.GetNumConformers() == 0 or not probe.GetConformer().Is3D():
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(probe, AllChem.ETKDGv3())
    if probe.GetNumConformers() == 0:
        return False
    options = rdShapeAlign.ShapeInputOptions()
    options.useColors = True
    return float(rdShapeAlign.PrepareConformer(probe, -1, options).sof) > 0.0


def score_pair(candidate: Chem.Mol, reference: Chem.Mol) -> tuple[float, float, float]:
    """Return winning pose (tanimoto_combo, shape_tanimoto, color_tanimoto) on [0, 2] scale.

    Evaluates canonical RDKit Shape + Color TanimotoCombo alignment with 4 canonical axis flips.
    Used by greedy diversity selection and pairwise conformer analyses.
    """
    ref_copy = Chem.Mol(reference)
    cand_copy = Chem.Mol(candidate)
    _canonicalize_conformer_inplace(ref_copy, -1)
    _canonicalize_conformer_inplace(cand_copy, -1)
    best = (0.0, 0.0, 0.0)
    for signs in _AXIS_SIGN_FLIPS:
        probe = _flip_conformer(cand_copy, -1, signs)
        try:
            shape, color = rdShapeAlign.AlignMol(
                ref_copy, probe, refConfId=-1, probeConfId=-1, useColors=True, opt_param=0.5
            )
        except (RuntimeError, ValueError):
            continue
        combo = float(shape) + float(color)
        if combo > best[0]:
            best = (combo, float(shape), float(color))
    return best


class CanonicalROCSScorer(RDKitROCSScorer):
    """Authoritative ROCS scorer returning canonical std.ROCS key without monkey-patching."""

    def getKey(self) -> list[str]:
        return [std.ROCS]


def create_rocs_scorer(
    references: str | Path | Chem.Mol | list[Chem.Mol] | list[str],
    cfg: PipelineConfig | None = None,
    *,
    n_jobs: int | None = None,
    max_conformers: int | None = None,
    max_isomers: int | None = None,
    max_heavy_atoms: int | None = 45,
    max_rotatable_bonds: int | None = 15,
    conformer_threads: int | None = None,
    timeout: int | None = None,
    timeout_seconds: int | None = None,
    random_seed: int | None = None,
    minimize_candidates: bool | None = None,
    optimization: Optimization | None = None,
    score_variant: str | None = None,
) -> CanonicalROCSScorer:
    """Instantiate a multi-threaded ``CanonicalROCSScorer`` from a config or explicit values.

    Scoring is deliberately canonical TanimotoCombo only. Experimental Tversky
    variants are calibrated in ``_david/code/to_do/rocs_multimetric.py`` and are
    intentionally unreachable from here: a training run must never be able to
    drift onto a metric that the thesis did not select.
    """
    enable_double_precision_mol_pickling()
    active_variant = score_variant if score_variant is not None else (cfg.score_variant if cfg else std.VARIANT_TANIMOTO_COMBO)
    if active_variant != std.VARIANT_TANIMOTO_COMBO:
        raise ValueError(
            f"score_variant {active_variant!r} is not available. Training scores with canonical ROCS "
            f"TanimotoCombo ({std.VARIANT_TANIMOTO_COMBO!r}) only; experimental Tversky variants live "
            "in _david/code/to_do/rocs_multimetric.py and are never reachable from the pipeline."
        )
    ref_input = str(references) if isinstance(references, Path) else references

    c_max_conf = max_conformers if max_conformers is not None else (cfg.max_conformers if cfg else 20)
    c_max_iso = max_isomers if max_isomers is not None else (cfg.max_isomers if cfg else 4)
    c_heavy = max_heavy_atoms if (max_heavy_atoms != 45 or cfg is None) else (cfg.max_heavy_atoms if cfg else 45)
    c_rot = max_rotatable_bonds if (max_rotatable_bonds != 15 or cfg is None) else (cfg.max_rotatable_bonds if cfg else 15)
    c_threads = conformer_threads if conformer_threads is not None else (cfg.conformer_threads if cfg else 0)
    c_timeout = timeout_seconds if timeout_seconds is not None else (timeout if timeout is not None else (cfg.timeout_seconds if cfg else 30))
    c_seed = random_seed if random_seed is not None else (cfg.random_seed if cfg else 42)
    c_min = minimize_candidates if minimize_candidates is not None else (cfg.minimize_candidates if cfg else True)
    c_opt = optimization if optimization is not None else (cfg.optimization if cfg else Optimization("MMFF94s", 78.4))
    c_ratio = cfg.cpu_ratio if cfg else 0.99
    safe_jobs = n_jobs if n_jobs is not None else HardwareManager.get_safe_parallel_jobs(ratio=c_ratio)

    conf_gen = CandidateConformerGenerator(
        max_conformers=c_max_conf,
        max_isomers=c_max_iso,
        max_heavy_atoms=c_heavy,
        max_rotatable_bonds=c_rot,
        num_threads=c_threads,
        show_progress=False,
        timeout=c_timeout,
        random_seed=c_seed,
        minimize_candidates=c_min,
        optimization=c_opt,
        n_jobs=safe_jobs,
    )

    return CanonicalROCSScorer(
        references=ref_input,
        conformer_generator=conf_gen,
        score_type=std.ROCS_SCORE_TYPE,
        score_variant=std.VARIANT_TANIMOTO,
        use_colors=True,
        show_progress=False,
        n_jobs=safe_jobs,
    )


def _init_whole_call_worker(
    reference_path: str,
    setting: Mapping[str, Any],
) -> None:
    """Initialize a persistent single-threaded ROCS scorer inside a worker process."""
    global _WORKER_SCORER
    from _david.code.config import PipelineConfig

    enable_double_precision_mol_pickling()
    method_key = str(setting.get("method", "etkdg"))
    optimization = (
        setting["optimization"]
        if isinstance(setting.get("optimization"), Optimization)
        else NAMED_OPTIMIZATION_METHODS.get(method_key, Optimization("none", 1.0))
    )
    minimize = bool(
        setting.get("minimize_candidates", method_key != "etkdg" and optimization.method != "none")
    )
    worker_cfg = PipelineConfig(
        max_conformers=int(setting.get("candidate_attempts", setting.get("max_conformers", 20))),
        max_isomers=int(setting.get("max_isomers", 4)),
        random_seed=int(setting.get("random_seed", 42)),
        minimize_candidates=minimize,
        optimization=optimization,
        timeout_seconds=int(setting.get("timeout", 10)),
        conformer_threads=1,
        cpu_ratio=0.99,
    )
    _WORKER_SCORER = create_rocs_scorer(
        Path(reference_path),
        cfg=worker_cfg,
        n_jobs=1,
    )
    _WORKER_SCORER.conformer_generator.embedding_rmsd = setting.get("embedding_rmsd", None)
    _WORKER_SCORER.conformer_generator.protonate = bool(setting.get("protonate", False))


def _score_single_smiles(smiles: str) -> float:
    """Score one unique candidate SMILES in a persistent worker."""
    return float(_WORKER_SCORER.getScores([smiles])[0, 0])


class WholeCallScorer(Scorer):
    """Persistent worker-pool adapter that parallelizes whole per-molecule ROCS calls."""

    def __init__(
        self,
        base: Scorer,
        reference_path: Path | str,
        setting: Mapping[str, Any],
        workers: int,
        mp_context: str = "forkserver",
    ) -> None:
        super().__init__()
        self._base = base
        self._ref_path = str(Path(reference_path).resolve())
        self._setting = dict(setting)
        self._workers = max(1, int(workers))
        self._mp_context = mp_context
        self._pool: Any = None

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            ctx = mp.get_context(self._mp_context)
            self._pool = pool_context(
                self._workers,
                initializer=_init_whole_call_worker,
                initargs=(self._ref_path, self._setting),
                context=ctx,
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            finally:
                self._pool = None

    def __enter__(self) -> WholeCallScorer:
        self._ensure_pool()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def getKey(self) -> list[str]:
        return getattr(self._base, "getKey", lambda: [std.ROCS])()

    def getScores(
        self,
        mols: list[str | Chem.Mol],
        frags: list[str | Chem.Mol] | None = None,
    ) -> np.ndarray:
        if not mols:
            return np.zeros((0, 1), dtype=np.float64)
        smiles_list = [
            Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else str(m) for m in mols
        ]
        pool = self._ensure_pool()
        chunksize = max(1, len(smiles_list) // (self._workers * 4))
        results = pool.map(_score_single_smiles, smiles_list, chunksize=chunksize)
        return np.asarray(results, dtype=np.float64).reshape(-1, 1)
