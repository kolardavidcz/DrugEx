"""
Abstract interfaces for scoring functions, score modifiers, and conformer generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from rdkit import Chem


class ScoreModifier(ABC):
    """
    Abstract base class defining non-linear score transformations.

    Modifiers transform raw biological or physicochemical property values (e.g. IC50, MW, Tanimoto)
    into normalized desirability scores in the range [0.0, 1.0] for reinforcement learning rewards.
    """

    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply the transformation function on raw property values.

        Parameters
        ----------
        x : float or np.ndarray
            Scalar value or 1D/2D NumPy array of raw property scores.

        Returns
        -------
        transformed : float or np.ndarray
            Normalized score(s) mapped to desirable ranges.
        """
        pass


class Scorer(ABC):
    """
    Abstract base class for all objective evaluators and reward functions.

    Used by `drugex.training.environment.Environment` to calculate scalar or vector
    rewards for de novo generated molecules.

    Parameters
    ----------
    modifier : ScoreModifier or list of ScoreModifier, optional
        Transformation function(s) applied to raw scores to normalize them into [0.0, 1.0].
        If a list is provided for a multi-task scorer, each modifier corresponds to a distinct task.
    """

    def __init__(
        self,
        modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]] = None
    ) -> None:
        self.modifier = modifier

    @abstractmethod
    def getScores(
        self,
        mols: Sequence[Union[str, Chem.Mol]],
        frags: Optional[Sequence[Union[str, Chem.Mol]]] = None
    ) -> np.ndarray:
        """
        Calculate raw objective scores for input molecules.

        Parameters
        ----------
        mols : sequence of str or rdkit.Chem.Mol
            List of molecules provided as SMILES strings or RDKit Mol instances.
        frags : sequence of str or rdkit.Chem.Mol, optional
            Input fragment scaffolds corresponding to each molecule.

        Returns
        -------
        scores : np.ndarray
            Raw scores array. Shape is `(n_mols,)` for single-objective scorers,
            or `(n_mols, n_tasks)` for multi-task / multi-objective scorers.
        """
        pass

    def __call__(
        self,
        mols: Sequence[Union[str, Chem.Mol]],
        frags: Optional[Sequence[Union[str, Chem.Mol]]] = None
    ) -> np.ndarray:
        """
        Evaluate molecules and return modifier-transformed scores.

        Parameters
        ----------
        mols : sequence of str or rdkit.Chem.Mol
            Molecules to evaluate.
        frags : sequence of str or rdkit.Chem.Mol, optional
            Input fragments.

        Returns
        -------
        scores : np.ndarray
            Transformed scores array mapped by the active modifier.
        """
        return self.getModifiedScores(self.getScores(mols, frags))

    def getModifiedScores(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply active modifier transformation to raw scores.

        Parameters
        ----------
        scores : np.ndarray
            Raw score matrix or vector.

        Returns
        -------
        modified_scores : np.ndarray
            Transformed score array.
        """
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
        """
        Return the unique identifier key or list of keys for this scorer.

        Returns
        -------
        key : str or list of str
            Objective column name(s) stored in evaluation DataFrames.
        """
        pass

    def setModifier(
        self,
        modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]]
    ) -> None:
        """
        Update the active score modifier.

        Parameters
        ----------
        modifier : ScoreModifier or list of ScoreModifier, optional
            New score modifier instance.
        """
        self.modifier = modifier

    def getModifier(self) -> Optional[Union[ScoreModifier, list[ScoreModifier]]]:
        """
        Retrieve the active score modifier.

        Returns
        -------
        modifier : ScoreModifier or list of ScoreModifier, optional
            Current modifier instance, or None if raw scores are emitted.
        """
        return self.modifier


class ConformerGenerator(ABC):
    """
    Abstract base class for 3D conformer ensemble generation used in ROCS shape alignment.
    """

    @abstractmethod
    def genConformers(self, smiles_list: list[str], out_dir: str) -> str:
        """
        Generate 3D conformer ensembles for a list of SMILES and write to SDF.

        Parameters
        ----------
        smiles_list : list of str
            List of input compound SMILES strings.
        out_dir : str
            Output directory path for the written SDF file.

        Returns
        -------
        sdf_path : str
            Absolute path to the generated SDF file containing 3D coordinates.
        """
        pass