"""
Abstract interfaces for scoring functions, score modifiers, and conformer generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Union


import numpy as np

if TYPE_CHECKING:
    # [PACKAGE: rdkit] WHY: Chemical structure type hinting. WHAT: Cheminformatics toolkit.
    from rdkit import Chem


class ScoreModifier(ABC):
    """Abstract base class defining non-linear score transformations."""

    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply the transformation function on raw property values."""
        pass


class Scorer(ABC):
    """Abstract base class for all objective evaluators and reward functions."""

    def __init__(self, modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]] = None) -> None:
        self.modifier = modifier

    @abstractmethod
    def getScores(
        self, mols: Sequence[Union[str, Chem.Mol]], frags: Optional[Sequence[Union[str, Chem.Mol]]] = None
    ) -> np.ndarray:
        """Calculate raw objective scores for input molecules."""
        pass

    def __call__(
        self, mols: Sequence[Union[str, Chem.Mol]], frags: Optional[Sequence[Union[str, Chem.Mol]]] = None
    ) -> np.ndarray:
        """Evaluate molecules and return modifier-transformed scores."""
        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores: np.ndarray) -> np.ndarray:
        """Apply active modifier transformation to raw scores."""
        if self.modifier is not None:
            if isinstance(self.modifier, list):
                # Apply modifier to each score column in multi-task case
                modified = scores.copy()
                for i, modifier in enumerate(self.modifier):
                    modified[:, i] = modifier(modified[:, i])
                return modified
            else:
                return self.modifier(scores)
        else:
            return scores

    @abstractmethod
    def getKey(self) -> Union[str, list[str]]:
        """Return the unique identifier key or list of keys for this scorer."""
        pass

    def setModifier(self, modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]]) -> None:
        """Update the active score modifier."""
        self.modifier = modifier

    def getModifier(self) -> Optional[Union[ScoreModifier, list[ScoreModifier]]]:
        """Retrieve the active score modifier."""
        return self.modifier


class ConformerGenerator(ABC):
    """Abstract base class for 3D conformer ensemble generation used in ROCS shape alignment."""

    @abstractmethod
    def genConformers(self, smiles_list: list[str], out_dir: str) -> str:
        """Generate 3D conformer ensembles for a list of SMILES and write to SDF."""
        pass
