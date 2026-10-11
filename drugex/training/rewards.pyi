from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Sequence
from typing import Union
from drugex.logs import logger
from drugex.training.interfaces import RewardScheme
from drugex.utils.fingerprints import get_fingerprint
from drugex.utils import get_Pareto_fronts
from rdkit import DataStructs


class ParetoRewardScheme(RewardScheme, ABC):
    """
    Abstract reward scheme prioritizing Pareto frontiers with secondary diversity ranking.

    Multi-objective scalarization prevents reward collapse on trivial 2D descriptors (unlike naive
    linear scalarization) by ranking candidates across non-dominated Pareto frontiers (or Augmented
    Chebyshev non-linear scalarization) before intra-front crowd diversification.
    """

    def getParetoFronts(self, scores: np.ndarray) -> List[np.ndarray]:
        """
        Returns Pareto fronts.

        Parameters
        ----------
        scores : np.ndarray
            Matrix of scores for the multiple objectives

        Returns
        -------
        list
            `list` of Pareto fronts. Each front is a `list` of indices of the molecules in the Pareto front. 
            Most dominant front is the first one.
        """

        ...

    @abstractmethod
    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = ..., scores: Optional[np.ndarray] = ...) -> Any:
        """
        Ranks molecules within each Pareto front and returns the indices of the molecules in the ranked order.

        Parameters
        ----------
        fronts : list
            `list` of Pareto fronts. Each front is a `list` of indices of the molecules in the Pareto front.
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives

        Returns
        -------
        rank : np.array
            Indices of the ranked SMILES sequences
        """

        ...

    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: Sequence[float] | Sequence[Any]) -> np.ndarray:
        """
        Returns the rewards for the given SMILES sequences. The reward is calculated 
        based on Pareto fronts and the rank of the molecule within the Pareto front.
        The rank of molecule in a Pareto front depends on the distance metric used by
        the specific ranking strategy.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        thresholds : list
            List of thresholds for the multiple objectives (not used, only for compatibility)

        Returns
        -------
        rewards : np.array
            Rewards for the given SMILES sequences
        """

        ...


class ParetoCrowdingDistance(ParetoRewardScheme):
    """
    Reward scheme that uses the NSGA-II crowding distance 
    ranking strategy to rank the solutions in the same Pareto frontier.

    Paper: Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: 
    NSGA-II. IEEE transactions on evolutionary computation 6.2 (2002): 182-197."
    """

    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = ..., scores: Optional[np.ndarray] = ...) -> Any:
        """
        Crowding distance algorithm to rank the solutions in the same pareto frontier.

        Parameters
        ----------
        fronts : list
            `list` of Pareto fronts. Each front is a `list` of indices of the molecules in the Pareto front.
        smiles : list
            List of SMILES sequence to be ranked (not used in the calculation -> just a requirement of the interface because some ranking strategies need it)"
        scores : np.ndarray
            matrix of scores for the multiple objectives

        Returns
        -------
        rank : np.array
            Indices of the SMILES sequences ranked with the NSGA-II crowding distance method from worst to best
        """

        ...


class ParetoTanimotoDistance(ParetoRewardScheme):
    """Reward scheme that uses the Tanimoto distance ranking strategy to rank the solutions in the same Pareto frontier."""

    def __init__(self, distance_metric: str = ...) -> None:
        """
        Args:
            distance_metric: 'mean', 'min' or 'mutual' - how to compare Tanimoto
                similarities of molecules in the same front
        """

        ...

    @staticmethod
    def calc_fps(mols: List[Chem.Mol], fp_type: str = ...) -> List[Optional[ExplicitBitVect]]:
        """
        Calculate fingerprints for a list of molecules.

        Parameters
        ----------
        mols : list
            List of RDKit molecules
        fp_type : str
            Type of fingerprint to use

        Returns
        -------
        fps : list
            List of RDKit fingerprints
        """

        ...

    def getFPs(self, smiles: List[str]) -> List[Optional[ExplicitBitVect]]:
        """
        Calculate fingerprints for a list of molecules.

        Args:
            smiles: smiles to calculate fingerprints for

        Returns:
            list of RDKit fingerprints
        """

        ...

    def _min_mean_similarity_ranking(self, fronts: Any, fps: Any) -> Any:
        ...

    def _mutual_similarity_ranking(self, fronts: Any, fps: Any) -> Any:
        """
        Alternative Tanimoto similarity based ranking strategy that tries to more closely 
        emulate the crowding distance with fingerprint similarities. Adapted from the 
        original code by @XuhanLiu (https://github.com/XuhanLiu/DrugEx/blob/cd384f4a8ed4982776e92293f77afd4ea78644f9/utils/nsgaii.py#L92).
        """

        ...

    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = ..., scores: Optional[np.ndarray] = ...) -> Any:  # type: ignore[override]
        """
        Get the rank of the molecules in the Pareto front based on the Tanimoto distance
        of molecules in a front.

        Parameters
        ----------
        fronts : list
            List of Pareto fronts
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            Array of scores for each molecule (not used)

        Returns
        -------
        rank : list
            List of indices of molecules, ranked from worst to best
        """

        ...


class WeightedSum(RewardScheme):
    """Reward scheme that uses the weighted sum ranking strategy to rank the solutions."""

    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: Sequence[float] | Sequence[Any]) -> np.ndarray:  # type: ignore[override]
        """
        Reward scheme that uses the weighted sum ranking strategy to rank the solutions.

        Parameters
        ----------
        smiles : list
            List of SMILES sequence to be ranked
        scores : np.ndarray
            matrix of scores for the multiple objectives
        thresholds : np.ndarray
            Thresholds for the multiple objectives

        Returns
        -------
        rewards : np.ndarray
            Array of rewards for the SMILES sequences
        """

        ...

