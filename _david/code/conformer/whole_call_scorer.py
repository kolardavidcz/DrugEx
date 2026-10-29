"""Prototype: persist references and parallelize complete per-molecule scoring.

Uses the existing public factory in each worker, with no nested alignment pool.
Canonical deduplication happens before dispatch. RDKit molecules never cross a
process boundary; references load from the same saved SDF and tasks carry SMILES.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

import numpy as np
from drugex.training.scorers.interfaces import Scorer

from _david.code.conformer.reference_bank import METHODS
from _david.code.environment import create_rocs_scorer

_WORKER_SCORER = None


def initialize_worker(reference, setting):
    global _WORKER_SCORER
    _WORKER_SCORER = create_rocs_scorer(
        reference,
        max_conformers=int(setting["candidate_attempts"]),
        max_isomers=4,
        random_seed=42,
        minimize_candidates=setting["method"] != "etkdg",
        optimization=METHODS[setting["method"]],
        timeout=10,
        conformer_threads=1,
        n_jobs=1,
    )
    _WORKER_SCORER.conformer_generator.embedding_rmsd = None
    _WORKER_SCORER.conformer_generator.protonate = False


def score_one(smiles):
    # Each call constructs fresh embedding parameters for ONE molecular graph.
    # This also avoids a native batch's mutable maxIterations leaking from its
    # first molecule into later, differently sized molecules.
    return float(_WORKER_SCORER.getScores([smiles])[0, 0])


class WholeCallScorer(Scorer):
    """One reference group; preserve score scale, modifiers and input row order.

    This is a reviewable prototype, not a replacement production default.
    Scores must be checked: the legacy batch path can inherit its first graph's
    embedding-iteration budget; singleton calls follow per-graph RDKit defaults.
    """

    def __init__(self, prototype, reference: Path, setting: dict, workers: int):
        super().__init__(prototype.getModifier())
        if len(prototype.group_to_indices) != 1:
            raise ValueError("Whole-call prototype supports one reference group")
        if (
            prototype.score_type != "TanimotoCombo"
            or prototype.score_variant != "tanimoto"
            or not prototype.use_colors
            or prototype.inject_color
            or prototype.color_ff != "default"
        ):
            raise ValueError(
                "Whole-call prototype currently supports native-colour TanimotoCombo only"
            )
        self.prototype = prototype
        self.prototype.n_jobs = 1
        self.reference = str(reference.resolve())
        self.setting = {key: setting[key] for key in ("method", "candidate_attempts")}
        self.workers = max(1, workers)
        self.pool = None

    def getKey(self):
        return self.prototype.getKey()

    def getScores(self, mols, frags=None):
        if not mols:
            return np.zeros((0, 1))
        smiles = self.prototype._convert_to_smiles(mols)
        unique, mapping = self.prototype._deduplicate_smiles(smiles)
        result = np.zeros((len(mols), 1))
        if not unique:
            return result
        if self.workers == 1 or len(unique) == 1:
            # No pool startup is needed for a one-molecule validation call.
            values = [float(self.prototype.getScores([smi])[0, 0]) for smi in unique]
        else:
            if self.pool is None:
                self.pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker,
                    initargs=(self.reference, self.setting),
                )
            values = list(self.pool.map(score_one, unique, chunksize=1))
        for index, originals in mapping.items():
            result[originals, 0] = values[index]
        return result

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
