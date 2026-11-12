from __future__ import annotations

import logging
import multiprocessing
import os
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers.parallel import (
    ScoreTimeout,
    capped_n_jobs,
    mol_timeout,
    molecule_time_limit,
    pool_context,
)
from drugex.training.scorers.rocs_rdkit import (
    _AXIS_SIGN_FLIPS,
    _annotate_color_features,
    _canonicalize_conformer_inplace,
    _flip_conformer,
    RDKitROCSScorer,
)

from _david.code.scorers.utils import pool_is_healthy

logger = logging.getLogger(__name__)

NO_OVERLAY = (0.0, 0.0)


def pair_components(
    query: Chem.Mol,
    reference: Chem.Mol,
    *,
    opt_param: float,
    use_colors: bool = True,
    color_ff: str = "fdef",
) -> tuple[float, float]:
    """The winner is chosen on shape + colour whatever ``opt_param`` is."""
    if query is None or reference is None:
        return NO_OVERLAY
    if query.GetNumConformers() == 0 or reference.GetNumConformers() == 0:
        return NO_OVERLAY

    if use_colors and color_ff != "default":
        for mol in (query, reference):
            if not mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
                _annotate_color_features(mol, color_ff=color_ff)

    reference_canonical = Chem.Mol(reference)
    for conformer in reference.GetConformers():
        _canonicalize_conformer_inplace(reference_canonical, conformer.GetId())

    best, best_total = NO_OVERLAY, 0.0
    for query_conformer in query.GetConformers():
        probe_canonical = Chem.Mol(query)
        _canonicalize_conformer_inplace(probe_canonical, query_conformer.GetId())
        for reference_conformer in reference_canonical.GetConformers():
            for signs in _AXIS_SIGN_FLIPS:
                probe = _flip_conformer(probe_canonical, query_conformer.GetId(), signs)
                try:
                    result = rdShapeAlign.AlignMol(
                        reference_canonical,
                        probe,
                        refConfId=reference_conformer.GetId(),
                        probeConfId=query_conformer.GetId(),
                        useColors=use_colors,
                        opt_param=opt_param,
                    )
                except (RuntimeError, ValueError):
                    continue
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    continue
                shape, colour = float(result[0]), float(result[1])
                if shape + colour > best_total:
                    best, best_total = (shape, colour), shape + colour
    return best


@dataclass(frozen=True)
class PoseSearch:
    references: list[Chem.Mol]
    groups: list[list[int]]
    opt_param: float
    use_colors: bool = True
    color_ff: str = "fdef"

    def best(self, conformers: list[Chem.Mol]) -> list[tuple[float, float]]:
        try:
            with molecule_time_limit(mol_timeout()):
                return [self._best_of_group(conformers, indices) for indices in self.groups]
        except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError) as error:
            logger.warning("scoring failed for one molecule (%s); scored 0.0", error)
            return [NO_OVERLAY for _ in self.groups]

    def _best_of_group(self, conformers, indices) -> tuple[float, float]:
        best = NO_OVERLAY
        for conformer in conformers:
            for index in indices:
                found = pair_components(
                    conformer,
                    self.references[index],
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


def _worker_score(task: tuple[int, list[Chem.Mol]]):
    mol_id, conformers = task
    return mol_id, _SEARCH.best(conformers)


class ComboROCSScorer(RDKitROCSScorer):
    """``RDKitROCSScorer`` with a configurable alignment objective and recorded components.

    The components stay diagnostics: ``getKey`` returns one key.
    """

    def __init__(self, *args, alignment_shape_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= alignment_shape_weight <= 1.0:
            raise ValueError(
                f"alignment_shape_weight must lie in [0, 1], got {alignment_shape_weight!r}"
            )
        self.alignment_shape_weight = float(alignment_shape_weight)
        self.components: dict[str, tuple[float, float]] = {}
        self._by_mol_id: dict[int, tuple[float, float]] = {}
        # The parent's own pool would call native's worker, not this class.
        self.workers = self.n_jobs
        self.n_jobs = 1

    def _score_sequential(self, unique_count, conformers_by_mol, num_groups) -> np.ndarray:
        self._by_mol_id = {}
        search = PoseSearch(
            self.reference_mols,
            self.group_to_indices,
            self.alignment_shape_weight,
            self.use_colors,
            self.color_ff,
        )
        tasks = [(mol_id, conformers_by_mol.get(mol_id, [])) for mol_id in range(unique_count)]
        workers = capped_n_jobs(self.workers if self.workers > 0 else (os.cpu_count() or 1))

        results = None
        if not (
            workers <= 1
            or unique_count < 8
            or multiprocessing.current_process().daemon
            or not pool_is_healthy()
        ):
            try:
                with pool_context().Pool(workers, _worker_init, (search,)) as pool:
                    results = pool.map(_worker_score, tasks)
            except Exception as error:
                logger.warning("parallel ROCS failed (%s); rescoring sequentially", error)
        if results is None:
            results = [(mol_id, search.best(conformers)) for mol_id, conformers in tasks]

        scores = np.zeros((unique_count, num_groups))
        for mol_id, groups in results:
            for group_index, (shape, colour) in enumerate(groups):
                scores[mol_id, group_index] = shape + colour
            self._by_mol_id[mol_id] = groups[0] if groups else NO_OVERLAY
        return scores

    def getScores(self, mols, frags=None) -> np.ndarray:
        # Also here: the parent returns early on a missing conformer file.
        self._by_mol_id = {}
        self.components = {}
        scores = super().getScores(mols, frags)
        unique, _ = self._deduplicate_smiles(self._convert_to_smiles(mols))
        self.components = {
            smiles: self._by_mol_id.get(index, NO_OVERLAY) for index, smiles in enumerate(unique)
        }
        return scores

    def component_frame(self, smiles: list[str]) -> dict[str, list[float]]:
        found = []
        for value in smiles:
            pair = self.components.get(value)
            if pair is None and isinstance(value, str):
                mol = Chem.MolFromSmiles(value)
                if mol is not None:
                    pair = self.components.get(Chem.MolToSmiles(mol))
            found.append(pair or NO_OVERLAY)
        return {
            "ShapeTanimoto": [shape for shape, _ in found],
            "ColorTanimoto": [colour for _, colour in found],
        }
