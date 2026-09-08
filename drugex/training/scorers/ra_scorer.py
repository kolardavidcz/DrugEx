"""Retrosynthetic Accessibility (RAscore) scoring module using XGBoost CASP models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles

from drugex.training.scorers.interfaces import Scorer

if TYPE_CHECKING:
    from drugex.training.scorers.interfaces import ScoreModifier

from RAscore import RAscore_XGB  # type: ignore

logger = logging.getLogger(__name__)

XGB_MODEL_PATH: str | None = None


class RetrosyntheticAccessibilityScorer(Scorer):
    """Scorer evaluating retrosynthetic accessibility via RAscore.

    Given a SMILES string or RDKit molecule, returns a probability score in [0.0, 1.0]
    indicating the likelihood of finding a synthetic route by computer-aided synthesis
    planning (CASP) tools (AiZynthFinder), evaluated using an XGBoost model.

    Parameters
    ----------
    modifier : ScoreModifier | None, optional
        Modifier applied to the raw RA scores, by default None.

    Raises
    ------
    ImportError
        If the optional `RAscore` package is not installed.
    """

    def __init__(self, modifier: ScoreModifier | None = None) -> None:
        """Initialize the Retrosynthetic Accessibility Scorer.

        Parameters
        ----------
        modifier : ScoreModifier | None, optional
            A modifier that can be used to modify the scores returned by this scorer.
        """
        super().__init__(modifier=modifier)
        self.scorer: Any = RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Get RA scores for a list of molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            A sequence of SMILES strings or RDKit Mol objects representing molecules.
        frags : Sequence[str | None] | None, optional
            A list of fragments used to generate the molecules. Unused by this scorer.

        Returns
        -------
        np.ndarray
            A 1D numpy array of float64 scores in [0.0, 1.0]. Invalid or unparseable molecules
            receive 0.0.
        """
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
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'RAscore'
        """
        return "RAscore"
