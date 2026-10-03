"""Retrosynthetic Accessibility (RAscore) scoring module using XGBoost CASP models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

# [PACKAGE: numpy] WHY: Array allocation for float64 scores. WHAT: Scientific computing library.
import numpy as np
# [PACKAGE: rdkit] WHY: Molecule conversion and SMILES export. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from rdkit.Chem import MolToSmiles

from drugex.training.scorers.interfaces import Scorer

if TYPE_CHECKING:
    from drugex.training.scorers.interfaces import ScoreModifier

# [PACKAGE: RAscore] WHY: Retrosynthetic accessibility scoring using XGBoost CASP models. WHAT: Chemical retrosynthesis predictor.
from RAscore import RAscore_XGB  # type: ignore

logger = logging.getLogger(__name__)

XGB_MODEL_PATH: str | None = None


class RetrosyntheticAccessibilityScorer(Scorer):
    """Scorer evaluating retrosynthetic accessibility via RAscore XGBoost models."""

    def __init__(self, modifier: ScoreModifier | None = None) -> None:
        """Initialize retrosynthetic accessibility scorer with optional score modifier."""
        super().__init__(modifier=modifier)
        self.scorer: Any = RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Compute synthetic accessibility scores in [0.0, 1.0] for input molecules."""
        scores = np.zeros(shape=len(mols), dtype="float64")
        for i, mol in enumerate(mols):
            if mol is None:
                scores[i] = 0.0
                continue
            if isinstance(mol, Chem.Mol):
                smi = MolToSmiles(mol)
            elif isinstance(mol, str):
                smi = mol
            else:
                scores[i] = 0.0
                continue

            try:
                scores[i] = float(self.scorer.predict(smi))
            except Exception as e:
                logger.debug("RAscore calculation failed for molecule %d (%s): %s", i, smi, e)
                scores[i] = 0.0
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return "RAscore"
