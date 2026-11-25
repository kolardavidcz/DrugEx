"""3D ROCS shape + colour TanimotoCombo scorer extending DrugEx's RDKitROCSScorer."""

from __future__ import annotations

import gc
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from _david.code.scorers.utils import SCORE_CHUNK, pool_can_run, resolve_workers
from drugex.training.scorers.parallel import (
    ScoreTimeout,
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
)

logger = logging.getLogger(__name__)

NO_OVERLAY = (0.0, 0.0)
_T = TypeVar("_T")


def lookup_by_smiles(mapping: Mapping[str, _T], value: Any) -> _T | None:
    """Look up ``value`` in ``mapping`` directly, falling back to canonical SMILES."""
    found = mapping.get(value)
    if found is None and isinstance(value, str):
        mol = Chem.MolFromSmiles(value)
        if mol is not None:
            found = mapping.get(Chem.MolToSmiles(mol))
    return found


def pair_components(
    query: Chem.Mol,
    reference: Chem.Mol,
    *,
    opt_param: float,
    use_colors: bool = True,
    color_ff: str = "fdef",
) -> tuple[float, float]:
    """Align ``query`` onto ``reference`` and return winning ``(ShapeTanimoto, ColorTanimoto)``.

    The winning overlay maximizes ``ShapeTanimoto + ColorTanimoto`` over all conformer pairs
    and inertial-axis flips; ``opt_param`` only sets ``rdShapeAlign``'s pose objective.
    """
    if query is None or reference is None:
        return NO_OVERLAY
    if query.GetNumConformers() == 0 or reference.GetNumConformers() == 0:
        return NO_OVERLAY

    if use_colors and color_ff != "default":
        for mol in (query, reference):
            if not mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
                _annotate_color_features(mol, color_ff=color_ff)

    ref_canon = Chem.Mol(reference)
    for conf in reference.GetConformers():
        _canonicalize_conformer_inplace(ref_canon, conf.GetId())

    best, best_total = NO_OVERLAY, 0.0
    for q_conf in query.GetConformers():
        probe_canon = Chem.Mol(query)
        _canonicalize_conformer_inplace(probe_canon, q_conf.GetId())
        for r_conf in ref_canon.GetConformers():
            for signs in _AXIS_SIGN_FLIPS:
                probe = _flip_conformer(probe_canon, q_conf.GetId(), signs)
                try:
                    shape, colour = map(
                        float,
                        rdShapeAlign.AlignMol(
                            ref_canon,
                            probe,
                            refConfId=r_conf.GetId(),
                            probeConfId=q_conf.GetId(),
                            useColors=use_colors,
                            opt_param=opt_param,
                        ),
                    )
                except (RuntimeError, ValueError, TypeError):
                    continue
                if shape + colour > best_total:
                    best, best_total = (shape, colour), shape + colour
    return best


@dataclass(frozen=True)
class PoseSearch:
    """Stateless worker payload scoring one molecule's conformers against reference groups."""

    references: list[Chem.Mol]
    groups: list[list[int]]
    opt_param: float
    use_colors: bool = True
    color_ff: str = "fdef"

    def best(self, conformers: list[Chem.Mol]) -> list[tuple[float, float]]:
        """Best ``(shape, colour)`` per reference group within the per-molecule time limit."""
        try:
            with molecule_time_limit(mol_timeout()):
                return [self._best_of_group(conformers, indices) for indices in self.groups]
        except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError) as error:
            logger.warning("scoring failed for one molecule (%s); scored 0.0", error)
            return [NO_OVERLAY for _ in self.groups]

    def _best_of_group(
        self, conformers: list[Chem.Mol], indices: list[int]
    ) -> tuple[float, float]:
        best = NO_OVERLAY
        for conf in conformers:
            for idx in indices:
                found = pair_components(
                    conf,
                    self.references[idx],
                    opt_param=self.opt_param,
                    use_colors=self.use_colors,
                    color_ff=self.color_ff,
                )
                if sum(found) > sum(best):
                    best = found
        return best


_SEARCH: PoseSearch | None = None


def _worker_init(search: PoseSearch) -> None:
    global _SEARCH
    _SEARCH = search


def _worker_score(task: tuple[int, list[Chem.Mol]]) -> tuple[int, list[tuple[float, float]]]:
    if _SEARCH is None:
        raise RuntimeError("ROCS worker pool was not initialized")
    mol_id, conformers = task
    return mol_id, _SEARCH.best(conformers)


class ComboROCSScorer(RDKitROCSScorer):
    """``RDKitROCSScorer`` with configurable ``alignment_shape_weight`` and shape/colour diagnostics."""

    def __init__(self, *args: Any, alignment_shape_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= alignment_shape_weight <= 1.0:
            raise ValueError(
                f"alignment_shape_weight must lie in [0, 1], got {alignment_shape_weight!r}"
            )
        self.alignment_shape_weight = float(alignment_shape_weight)
        self.components: dict[str, tuple[float, float]] = {}
        self._by_mol_id: dict[int, tuple[float, float]] = {}
        # Keep worker count locally; parent's n_jobs=1 routes execution into _score_sequential.
        self.workers = self.n_jobs
        self.n_jobs = 1

    def _score_sequential(
        self,
        unique_count: int,
        conformers_by_mol: dict[int, list[Chem.Mol]],
        num_groups: int,
    ) -> np.ndarray:
        self._by_mol_id = {}
        search = PoseSearch(
            self.reference_mols,
            self.group_to_indices,
            self.alignment_shape_weight,
            self.use_colors,
            self.color_ff,
        )
        tasks = [(mol_id, conformers_by_mol.get(mol_id, [])) for mol_id in range(unique_count)]
        worker_count = resolve_workers(self.workers)

        if pool_can_run(worker_count, unique_count, min_items=8):
            try:
                with pool_context().Pool(worker_count, _worker_init, (search,)) as pool:
                    results = pool.map(_worker_score, tasks)
            except Exception as error:
                logger.error("parallel ROCS failed (%s); aborting", error)
                raise RuntimeError(f"parallel ROCS failed: {error}") from error
        else:
            results = [(mol_id, search.best(conformers)) for mol_id, conformers in tasks]

        scores = np.zeros((unique_count, num_groups))
        for mol_id, groups in results:
            for group_index, (shape, colour) in enumerate(groups):
                scores[mol_id, group_index] = shape + colour
            self._by_mol_id[mol_id] = groups[0] if groups else NO_OVERLAY
        return scores

    def getScores(self, mols: Any, frags: Any = None) -> np.ndarray:
        """Score ``mols`` in memory-bounded chunks while recording winning shape/colour splits."""
        self._by_mol_id = {}
        self.components = {}
        smiles_list = self._convert_to_smiles(mols)
        unique, unique_to_original = self._deduplicate_smiles(smiles_list)
        scores = np.zeros((len(mols), len(self.group_to_indices)))
        if not unique:
            return scores

        all_components: dict[str, tuple[float, float]] = {}
        for start in range(0, len(unique), SCORE_CHUNK):
            piece = unique[start : start + SCORE_CHUNK]
            self._by_mol_id = {}
            piece_scores = super().getScores(piece, frags)
            for local_id, smi in enumerate(piece):
                all_components[smi] = self._by_mol_id.get(local_id, NO_OVERLAY)
                for orig_idx in unique_to_original[start + local_id]:
                    scores[orig_idx] = piece_scores[local_id]
            if len(unique) > SCORE_CHUNK:
                gc.collect()
        self.components = all_components
        return scores

    def component_frame(self, smiles: list[str]) -> dict[str, list[float]]:
        """Aligned ``ShapeTanimoto`` and ``ColorTanimoto`` lists for ``smiles``."""
        found = [lookup_by_smiles(self.components, smi) or NO_OVERLAY for smi in smiles]
        return {
            "ShapeTanimoto": [shape for shape, _ in found],
            "ColorTanimoto": [colour for _, colour in found],
        }
