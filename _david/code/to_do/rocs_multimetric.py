"""Shared RDKit ROCS shape-and-color helpers and persistent whole-call parallel scorer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from typing import Any

from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.parallel import (
    ScoreTimeout,
    capped_n_jobs,
    mol_timeout,
    molecule_time_limit,
    pool_context,
)
from drugex.training.scorers.rocs_rdkit import (
    _AXIS_SIGN_FLIPS,
    RDKitROCSScorer,
    _annotate_color_features,
    _canonicalize_conformer_inplace,
    _flip_conformer,
    _self_overlaps,
    _tversky_from_tanimoto,
)
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from _david.code.chemistry import (
    Optimization,
    enable_double_precision_mol_pickling,
)
from _david.code.utils import HardwareManager

NAMED_OPTIMIZATION_METHODS: dict[str, Optimization] = {
    "etkdg": Optimization("none"),
    "uff": Optimization("UFF"),
    "mmff_vacuum": Optimization("MMFF94s", 1.0),
    "mmff_high_dielectric": Optimization("MMFF94s", 78.4),
}

# Canonical (tversky_alpha, tversky_beta, shape_weight, color_weight) presets calibrated
# on the 121x121 = 14,641 drug-pair x 162-combination selectivity grid (O'Hagan & Kell 2016):
# - Option 1 (Ultra-Clean Selectivity, α+β=2 Tanimoto scale + 0.2S+0.8C: <0.85% decoy FP at 0.35):
#     "selectivity_color_heavy"  -> (1.0, 1.0, 0.2, 0.8)
#     "tversky_mild_color_heavy" -> (0.8, 1.2, 0.2, 0.8)
# - Option 2 (Aggressive Sub-Scaffold Hopping, α+β=2 (0.6, 1.4) + 0.3S+0.7C or 0.2S+0.8C):
#     "tversky_scaffold_hop"        -> (0.6, 1.4, 0.3, 0.7)
#     "tversky_scaffold_hop_strict" -> (0.6, 1.4, 0.2, 0.8)
ROCS_SIMILARITY_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "tanimoto": (1.0, 1.0, 0.5, 0.5),
    "tanimoto_combo": (1.0, 1.0, 0.5, 0.5),
    "selectivity_color_heavy": (1.0, 1.0, 0.2, 0.8),
    "tversky_mild_color_heavy": (0.8, 1.2, 0.2, 0.8),
    "tversky_scaffold_hop": (0.6, 1.4, 0.3, 0.7),
    "tversky_scaffold_hop_strict": (0.6, 1.4, 0.2, 0.8),
    "tversky_subshape": (0.30, 0.70, 0.5, 0.5),
    "tversky_ref": (0.95, 0.05, 0.5, 0.5),
    "tversky_fit": (0.05, 0.95, 0.5, 0.5),
}

_WORKER_SCORER: Any = None
_CUSTOM_ROCS_WORKER_STATE: dict[str, Any] = {}

def resolve_rocs_similarity_params(
    score_variant: str = "tanimoto_combo",
    tversky_alpha: float | None = None,
    tversky_beta: float | None = None,
    shape_weight: float | None = None,
    color_weight: float | None = None,
) -> tuple[str, float, float, float, float]:
    """Resolve (base_drugex_variant, alpha, beta, shape_weight, color_weight)."""
    key = score_variant.lower().strip()
    if key not in ROCS_SIMILARITY_PRESETS:
        raise ValueError(
            f"Unknown score_variant {score_variant!r}; valid presets: {sorted(ROCS_SIMILARITY_PRESETS)}"
        )
    def_a, def_b, def_ws, def_wc = ROCS_SIMILARITY_PRESETS[key]
    alpha = float(def_a if tversky_alpha is None else tversky_alpha)
    beta = float(def_b if tversky_beta is None else tversky_beta)
    ws = float(def_ws if shape_weight is None else shape_weight)
    wc = float(def_wc if color_weight is None else color_weight)
    if alpha < 0.0 or beta < 0.0 or (alpha + beta) <= 0.0:
        raise ValueError("Tversky alpha and beta must be non-negative with alpha + beta > 0")
    if ws < 0.0 or wc < 0.0 or (ws + wc) <= 0.0:
        raise ValueError("ROCS shape_weight and color_weight must be non-negative with positive sum")
    base_variant = (
        key if key in {"tanimoto", "tversky_subshape", "tversky_ref", "tversky_fit"} else "tanimoto"
    )
    return base_variant, alpha, beta, ws, wc

#NOTE since all drug_like molecules has color and nobudy eevr ytest this, we sould chalange mengif of it (eg just error)
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


def _score_weighted_single_reference(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    score_type: str,
    use_colors: bool,
    inject_color: bool = False,
    tversky_alpha: float = 1.0,
    tversky_beta: float = 1.0,
    shape_weight: float = 0.5,
    color_weight: float = 0.5,
    color_ff: str = "default",
) -> float:
    """Compute best 3D alignment score with configurable Tversky (alpha, beta) and (shape_weight, color_weight).

    When ``score_type == "TanimotoCombo"``, returns a score on the raw ``[0, 2]`` scale:
        ``2.0 * (w_S * S + w_C * C) / (w_S + w_C)``
    so that ``ClippedScore(0.0, 2.0)`` normalizes the output into ``[0, 1]`` identically
    to ``RDKitROCSScorer``.
    """
    if query_mol is None or ref_mol is None:
        return 0.0
    if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return 0.0

    if (inject_color or color_ff != "default") and use_colors:
        if not query_mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(query_mol, color_ff=color_ff)
        if not ref_mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(ref_mol, color_ff=color_ff)

    opt_param = 1.0 if score_type == "shape" else (0.0 if score_type == "color" else 0.5)
    use_tversky = not (abs(tversky_alpha - 1.0) <= 1e-9 and abs(tversky_beta - 1.0) <= 1e-9)
    if use_tversky:
        opts = rdShapeAlign.ShapeInputOptions()
        opts.useColors = use_colors
        ref_self = {
            rc.GetId(): _self_overlaps(ref_mol, rc.GetId(), opts)
            for rc in ref_mol.GetConformers()
        }
        qry_self = {
            qc.GetId(): _self_overlaps(query_mol, qc.GetId(), opts)
            for qc in query_mol.GetConformers()
        }

    w_sum = shape_weight + color_weight
    ws_scaled = 2.0 * shape_weight / w_sum
    wc_scaled = 2.0 * color_weight / w_sum

    ref_canon = Chem.Mol(ref_mol)
    for ref_conf in ref_mol.GetConformers():
        _canonicalize_conformer_inplace(ref_canon, ref_conf.GetId())

    best_score = 0.0
    for query_conf in query_mol.GetConformers():
        probe_canon = Chem.Mol(query_mol)
        _canonicalize_conformer_inplace(probe_canon, query_conf.GetId())
        for ref_conf in ref_mol.GetConformers():
            for signs in _AXIS_SIGN_FLIPS:
                probe_copy = _flip_conformer(probe_canon, query_conf.GetId(), signs)
                try:
                    result = rdShapeAlign.AlignMol(
                        ref_canon,
                        probe_copy,
                        refConfId=ref_conf.GetId(),
                        probeConfId=query_conf.GetId(),
                        useColors=use_colors,
                        opt_param=opt_param,
                    )
                except (RuntimeError, ValueError):
                    continue
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    continue
                shape_score, color_score = float(result[0]), float(result[1])
                if use_tversky:
                    sov_r, sof_r = ref_self[ref_conf.GetId()]
                    sov_q, sof_q = qry_self[query_conf.GetId()]
                    shape_score = _tversky_from_tanimoto(
                        shape_score, sov_r, sov_q, tversky_alpha, tversky_beta
                    )
                    color_score = _tversky_from_tanimoto(
                        color_score, sof_r, sof_q, tversky_alpha, tversky_beta
                    )
                if score_type == "shape":
                    score = shape_score
                elif score_type == "color":
                    score = color_score
                else:
                    score = ws_scaled * shape_score + wc_scaled * color_score
                if score > best_score:
                    best_score = score
    return best_score


def _custom_rocs_worker_init(
    reference_mols: list[Chem.Mol],
    group_to_indices: list[list[int]],
    score_type: str,
    use_colors: bool,
    inject_color: bool,
    tversky_alpha: float,
    tversky_beta: float,
    shape_weight: float,
    color_weight: float,
    color_ff: str,
) -> None:
    """Initialize immutable worker state for ``ConfigurableRDKitROCSScorer``."""
    global _CUSTOM_ROCS_WORKER_STATE
    enable_double_precision_mol_pickling()
    _CUSTOM_ROCS_WORKER_STATE = {
        "reference_mols": reference_mols,
        "group_to_indices": group_to_indices,
        "score_type": score_type,
        "use_colors": use_colors,
        "inject_color": inject_color,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "shape_weight": shape_weight,
        "color_weight": color_weight,
        "color_ff": color_ff,
    }


def _score_custom_rocs_worker(
    args: tuple[int, list[Chem.Mol]],
) -> tuple[int, list[float]]:
    """Score one molecule's conformer list in a worker process with custom Tversky/Color weights."""
    mol_id, mol_conformers = args
    st = _CUSTOM_ROCS_WORKER_STATE
    reference_mols: list[Chem.Mol] = st.get("reference_mols", [])
    group_to_indices: list[list[int]] = st.get("group_to_indices", [])
    num_groups = len(group_to_indices) if group_to_indices else (1 if reference_mols else 0)
    if not mol_conformers or num_groups == 0:
        return mol_id, [0.0] * num_groups

    group_scores = [0.0] * num_groups
    try:
        with molecule_time_limit(mol_timeout()):
            for conf_mol in mol_conformers:
                if conf_mol is None or conf_mol.GetNumConformers() == 0:
                    continue
                for group_idx, ref_indices in enumerate(group_to_indices):
                    for ref_idx in ref_indices:
                        score = _score_weighted_single_reference(
                            conf_mol,
                            reference_mols[ref_idx],
                            st["score_type"],
                            st["use_colors"],
                            st["inject_color"],
                            st["tversky_alpha"],
                            st["tversky_beta"],
                            st["shape_weight"],
                            st["color_weight"],
                            st["color_ff"],
                        )
                        if score > group_scores[group_idx]:
                            group_scores[group_idx] = score
    except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError):
        return mol_id, [0.0] * num_groups
    return mol_id, group_scores


class ConfigurableRDKitROCSScorer(RDKitROCSScorer):
    """RDKitROCSScorer extension supporting arbitrary Tversky (alpha, beta) and (shape_weight, color_weight).

    When standard equal-weight parameters are used, delegates directly to ``RDKitROCSScorer``.
    When custom Tanimoto-scale Tversky parameters (e.g. ``alpha=0.6, beta=1.4``) or asymmetric
    shape/color weights (e.g. ``0.2S + 0.8C`` or ``0.3S + 0.7C``) are configured, evaluates the
    exact weighted Gaussian overlap in both sequential and parallel worker modes.
    """

    def __init__(
        self,
        conformer_generator: Any,
        references: Any,
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
        inject_color: bool = False,
        score_variant: str = "tanimoto",
        color_ff: str = "default",
        tversky_alpha: float | None = None,
        tversky_beta: float | None = None,
        shape_weight: float | None = None,
        color_weight: float | None = None,
    ) -> None:
        base_variant, alpha, beta, ws, wc = resolve_rocs_similarity_params(
            score_variant=score_variant,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            shape_weight=shape_weight,
            color_weight=color_weight,
        )
        super().__init__(
            conformer_generator=conformer_generator,
            references=references,
            score_type=score_type,
            use_colors=use_colors,
            show_progress=show_progress,
            n_jobs=n_jobs,
            inject_color=inject_color,
            score_variant=base_variant,
            color_ff=color_ff,
        )
        self.preset_name: str = score_variant
        self.tversky_alpha: float = alpha
        self.tversky_beta: float = beta
        self.shape_weight: float = ws
        self.color_weight: float = wc
        raw_jobs = n_jobs if n_jobs is not None else -1
        resolved_jobs = (
            HardwareManager.get_safe_parallel_jobs()
            if raw_jobs == -1
            else capped_n_jobs(int(raw_jobs))
        )
        self._n_jobs: int = max(1, resolved_jobs)
        if hasattr(self, "conformer_generator") and hasattr(
            self.conformer_generator, "n_jobs"
        ):
            self.conformer_generator.n_jobs = self._n_jobs
        # Check whether standard DrugEx path is exact (ws == wc and standard alpha/beta)
        is_equal_weights = abs(ws - wc) <= 1e-9
        is_std_ab = (
            (base_variant == "tanimoto" and abs(alpha - 1.0) <= 1e-9 and abs(beta - 1.0) <= 1e-9)
            or (base_variant == "tversky_subshape" and abs(alpha - 0.30) <= 1e-9 and abs(beta - 0.70) <= 1e-9)
            or (base_variant == "tversky_ref" and abs(alpha - 0.95) <= 1e-9 and abs(beta - 0.05) <= 1e-9)
            or (base_variant == "tversky_fit" and abs(alpha - 0.05) <= 1e-9 and abs(beta - 0.95) <= 1e-9)
        )
        self._use_custom_weights: bool = not (is_equal_weights and is_std_ab)

    @property
    def n_jobs(self) -> int:
        val = getattr(self, "_n_jobs", None)
        if val is None or val < 1:
            return max(1, capped_n_jobs(HardwareManager.get_safe_parallel_jobs()))
        return val

    @n_jobs.setter
    def n_jobs(self, value: int | None) -> None:
        if value is None or value == -1:
            self._n_jobs = max(
                1, capped_n_jobs(HardwareManager.get_safe_parallel_jobs())
            )
        else:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("n_jobs must be a positive integer, -1, or None")
            self._n_jobs = max(1, capped_n_jobs(int(value)))
        if hasattr(self, "conformer_generator") and hasattr(
            self.conformer_generator, "n_jobs"
        ):
            self.conformer_generator.n_jobs = self._n_jobs

    def _score_sequential(
        self,
        unique_count: int,
        conformers_by_mol: dict[int, list[Chem.Mol]],
        num_groups: int,
    ) -> np.ndarray:
        if not self._use_custom_weights:
            return super()._score_sequential(unique_count, conformers_by_mol, num_groups)
        scores_unique = np.zeros((unique_count, num_groups))
        for mol_id in range(unique_count):
            mol_conformers = conformers_by_mol.get(mol_id, [])
            if not mol_conformers:
                continue
            group_scores = np.zeros(num_groups)
            for conf_mol in mol_conformers:
                for group_idx, ref_indices in enumerate(self.group_to_indices):
                    for ref_idx in ref_indices:
                        score = _score_weighted_single_reference(
                            conf_mol,
                            self.reference_mols[ref_idx],
                            self.score_type,
                            self.use_colors,
                            self.inject_color,
                            self.tversky_alpha,
                            self.tversky_beta,
                            self.shape_weight,
                            self.color_weight,
                            self.color_ff,
                        )
                        if score > group_scores[group_idx]:
                            group_scores[group_idx] = score
            scores_unique[mol_id] = group_scores
        return scores_unique

    def getScores(
        self,
        mols: list[Chem.Mol],
        frags: list[Chem.Mol] | None = None,
    ) -> np.ndarray:
        if not self._use_custom_weights or self.n_jobs == 1:
            return super().getScores(mols, frags=frags)

        # Parallel execution with custom Tversky / Color weights
        num_groups = len(self.group_to_indices)
        if num_groups == 0:
            raise ValueError("No reference groups configured")
        if not mols:
            return np.zeros((0, num_groups))
        scores = np.zeros((len(mols), num_groups))
        smiles_list = self._convert_to_smiles(mols)
        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)
        if not unique_smiles:
            return scores

        from collections import defaultdict
        import os
        import tempfile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            conf_file = self.conformer_generator.genConformers(unique_smiles, tmpdir)
            if not os.path.exists(conf_file) or os.path.getsize(conf_file) == 0:
                return scores
            conformers_by_mol: dict[int, list[Chem.Mol]] = defaultdict(list)
            try:
                suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
                if suppl is None:
                    return scores
                for conf_mol in suppl:
                    if conf_mol is None:
                        continue
                    try:
                        mol_id = int(conf_mol.GetProp("_Name").split("+")[0].split("_")[1])
                        conformers_by_mol[mol_id].append(conf_mol)
                    except (KeyError, ValueError, IndexError):
                        continue
            except Exception:
                return scores

            if not conformers_by_mol:
                return scores

            unique_count = len(unique_smiles)
            worker_args = [(mid, conformers_by_mol.get(mid, [])) for mid in range(unique_count)]
            chunksize = max(1, unique_count // (max(1, self.n_jobs) * 4))
            scores_unique = np.zeros((unique_count, num_groups))
            try:
                with pool_context().Pool(
                    self.n_jobs,
                    initializer=_custom_rocs_worker_init,
                    initargs=(
                        self.reference_mols,
                        self.group_to_indices,
                        self.score_type,
                        self.use_colors,
                        self.inject_color,
                        self.tversky_alpha,
                        self.tversky_beta,
                        self.shape_weight,
                        self.color_weight,
                        self.color_ff,
                    ),
                ) as pool:
                    results = pool.map(_score_custom_rocs_worker, worker_args, chunksize=chunksize)
                for mid, group_scores in results:
                    if 0 <= mid < unique_count and len(group_scores) == num_groups:
                        scores_unique[mid] = np.asarray(group_scores)
            except Exception:
                scores_unique = self._score_sequential(unique_count, conformers_by_mol, num_groups)

        for unique_id, original_indices in unique_to_original.items():
            if unique_id < scores_unique.shape[0]:
                for original_idx in original_indices:
                    scores[original_idx] = scores_unique[unique_id]
        return scores


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
        # Default 10s per-molecule ETKDG timeout prevents pathological stalls in study worker pools.
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
