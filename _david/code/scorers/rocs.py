"""The shape/colour scorer: one pose search, three numbers, one objective.

Native DrugEx computes ShapeTanimoto and ColorTanimoto in the same pose loop and then
throws one of them away -- native's ``score_type`` picks which number comes back. Reading all three
therefore costs three full pose searches through the native scorer, which is the dominant
cost of a run. This subclass owns the loop, so it reports the configured score *and*
records the two components of the winning pose, for free.

It also exposes what native keeps internal: ``AlignMol``'s ``opt_param``, the weight the
pose optimiser maximises (1.0 shape-only, 0.5 equal, 0.0 colour-only). Native derives it
from ``score_type``, so changing the alignment objective there also changes which number
is reported and on what scale. Here the two are separate: the alignment objective moves,
the reported score stays ``shape + colour`` on [0, 2], and runs remain comparable.

**The components are diagnostics, never objectives.** ``getKey`` returns exactly one key,
so ``DrugExEnvironment.getScorerKeys()`` sees one column and ``ParetoCrowdingDistance``
optimises one shape objective. Returning three keys would silently turn a two-objective
run into a four-objective one -- the class of bug this codebase exists to avoid. There is
a test pinning the key count.

Only ``_score_sequential`` is overridden, so conformer generation, SMILES de-duplication,
conformer-name parsing and reference grouping all stay native.
"""

from __future__ import annotations

import logging
import multiprocessing
import os

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

from _david.code.scorers.conformers import pool_is_healthy

logger = logging.getLogger(__name__)

#: Set in each pool worker by :func:`_worker_init`.
_WORKER: dict = {}


def pair_components(
    query: Chem.Mol,
    reference: Chem.Mol,
    *,
    opt_param: float,
    use_colors: bool = True,
    color_ff: str = "fdef",
) -> tuple[float, float]:
    """Best overlay of ``query`` on ``reference``: the winning pose's (shape, colour).

    The search is native's, reproduced: canonicalise both molecules into their inertial
    frame, then start the probe from each of four proper axis-sign assignments, over every
    (query conformer x reference conformer) pair, and keep the pose with the best
    ``shape + colour``. Winner is chosen on the combo regardless of ``opt_param``, so the
    alignment objective moves the pose without moving the definition of "best".
    """
    if query is None or reference is None:
        return 0.0, 0.0
    if query.GetNumConformers() == 0 or reference.GetNumConformers() == 0:
        return 0.0, 0.0

    if use_colors and color_ff != "default":
        if not query.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(query, color_ff=color_ff)
        if not reference.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(reference, color_ff=color_ff)

    reference_canonical = Chem.Mol(reference)
    for conformer in reference.GetConformers():
        _canonicalize_conformer_inplace(reference_canonical, conformer.GetId())

    best = (0.0, 0.0)
    best_total = 0.0
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
                # Native guards the shape of the return before unpacking; a malformed
                # result must cost this one pose, not the epoch.
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    continue
                shape, colour = float(result[0]), float(result[1])
                total = shape + colour
                if total > best_total:
                    best_total = total
                    best = (shape, colour)
    return best


def _worker_init(references, groups, opt_param, use_colors, color_ff) -> None:
    _WORKER.update(
        references=references,
        groups=groups,
        opt_param=opt_param,
        use_colors=use_colors,
        color_ff=color_ff,
    )


def _best_for_molecule(conformers, references, groups, opt_param, use_colors, color_ff):
    """Per reference group, the best (shape, colour) over every conformer-reference pair.

    Guarded exactly as native's worker is: a per-molecule time limit (so
    ``DRUGEX_SCORE_MOL_TIMEOUT`` keeps meaning something, since the manifest records it as
    an active knob) and a catch-all. Without both, one pathological molecule aborts the
    whole ``pool.map``, forces a sequential rescore of the entire batch, and takes the run
    with it. A molecule that cannot be scored is worst, not fatal.
    """
    try:
        with molecule_time_limit(mol_timeout()):
            return _best_for_molecule_unguarded(
                conformers, references, groups, opt_param, use_colors, color_ff
            )
    except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError) as error:
        logger.warning("scoring failed for one molecule (%s); scored 0.0", error)
        return [(0.0, 0.0) for _ in groups]


def _best_for_molecule_unguarded(conformers, references, groups, opt_param, use_colors, color_ff):
    out = []
    for indices in groups:
        best = (0.0, 0.0)
        for conformer in conformers:
            for index in indices:
                shape, colour = pair_components(
                    conformer,
                    references[index],
                    opt_param=opt_param,
                    use_colors=use_colors,
                    color_ff=color_ff,
                )
                if shape + colour > best[0] + best[1]:
                    best = (shape, colour)
        out.append(best)
    return out


def _worker_score(args):
    mol_id, conformers = args
    return mol_id, _best_for_molecule(
        conformers,
        _WORKER["references"],
        _WORKER["groups"],
        _WORKER["opt_param"],
        _WORKER["use_colors"],
        _WORKER["color_ff"],
    )


class ComboROCSScorer(RDKitROCSScorer):
    """``RDKitROCSScorer`` with a configurable alignment objective and recorded components.

    ``alignment_shape_weight`` is ``AlignMol``'s ``opt_param``. After each ``getScores``,
    ``components`` maps SMILES to the winning pose's ``(shape, colour)``.
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
        # The parent dispatches to its own multiprocessing path unless n_jobs == 1, and
        # that path calls native's worker, not this class -- the components would come
        # back empty and the alignment weight would be silently ignored. Keep the real
        # worker count here and pin the parent to its sequential branch, which is the one
        # that calls `_score_sequential`; the pooling happens inside that override.
        self.workers = self.n_jobs
        self.n_jobs = 1

    def _score_sequential(self, unique_count, conformers_by_mol, num_groups) -> np.ndarray:
        """Native's signature, our loop: records components and parallelises itself."""
        self._by_mol_id = {}
        tasks = [(mol_id, conformers_by_mol.get(mol_id, [])) for mol_id in range(unique_count)]
        workers = capped_n_jobs(self.workers if self.workers > 0 else (os.cpu_count() or 1))

        results: list[tuple[int, list[tuple[float, float]]]]
        if (
            workers <= 1
            or unique_count < 8
            or multiprocessing.current_process().daemon
            or not pool_is_healthy()
        ):
            results = [
                (
                    mol_id,
                    _best_for_molecule(
                        conformers,
                        self.reference_mols,
                        self.group_to_indices,
                        self.alignment_shape_weight,
                        self.use_colors,
                        self.color_ff,
                    ),
                )
                for mol_id, conformers in tasks
            ]
        else:
            try:
                with pool_context().Pool(
                    workers,
                    initializer=_worker_init,
                    initargs=(
                        self.reference_mols,
                        self.group_to_indices,
                        self.alignment_shape_weight,
                        self.use_colors,
                        self.color_ff,
                    ),
                ) as pool:
                    results = pool.map(_worker_score, tasks)
            except Exception as error:
                # Native falls back to sequential on pool failure rather than scoring zero.
                logger.warning("parallel ROCS failed (%s); rescoring sequentially", error)
                results = [
                    (
                        mol_id,
                        _best_for_molecule(
                            conformers,
                            self.reference_mols,
                            self.group_to_indices,
                            self.alignment_shape_weight,
                            self.use_colors,
                            self.color_ff,
                        ),
                    )
                    for mol_id, conformers in tasks
                ]

        scores = np.zeros((unique_count, num_groups))
        for mol_id, groups in results:
            for group_index, (shape, colour) in enumerate(groups):
                scores[mol_id, group_index] = self._reported(shape, colour)
            # The components belong to the first group; this experiment has exactly one.
            self._by_mol_id[mol_id] = groups[0] if groups else (0.0, 0.0)
        return scores

    def _reported(self, shape: float, colour: float) -> float:
        """The objective column: always ``shape + colour``.

        The winning pose is chosen on ``shape + colour`` (``pair_components``), so
        reporting a single component here would return that component *of the
        combo-optimal pose*, not the best achievable value of it -- a quietly different
        number. Shape and colour are published as diagnostics instead, where they belong
        to the pose that was actually selected.
        """
        return shape + colour

    def getScores(self, mols, frags=None) -> np.ndarray:
        # Cleared here, not only inside _score_sequential: the parent returns early when
        # the conformer file is missing or unreadable, and that path never reaches the
        # override. The previous batch's components would then be mapped onto this
        # batch's SMILES -- a transient filesystem fault turning into plausible
        # diagnostics for molecules that were never scored.
        self._by_mol_id = {}
        self.components = {}
        scores = super().getScores(mols, frags)
        # Map components back onto SMILES using native's own de-duplication, so the keys
        # match what the caller passed in. String work only; no rescoring.
        unique, _ = self._deduplicate_smiles(self._convert_to_smiles(mols))
        self.components = {
            smiles: self._by_mol_id.get(index, (0.0, 0.0)) for index, smiles in enumerate(unique)
        }
        return scores

    def component_frame(self, smiles: list[str]) -> dict[str, list[float]]:
        """``{"ShapeTanimoto": [...], "ColorTanimoto": [...]}`` aligned with ``smiles``.

        Components are keyed by the canonical SMILES native produced while scoring, so a
        caller holding the generator's raw output has to be canonicalised to match --
        otherwise a non-canonical string silently misses and reads as 0.0, which looks
        like a terrible score rather than a lookup failure.
        """
        shape, colour = [], []
        for value in smiles:
            found = self.components.get(value)
            if found is None and isinstance(value, str):
                mol = Chem.MolFromSmiles(value)
                if mol is not None:
                    found = self.components.get(Chem.MolToSmiles(mol))
            s, c = found if found is not None else (0.0, 0.0)
            shape.append(s)
            colour.append(c)
        return {"ShapeTanimoto": shape, "ColorTanimoto": colour}
