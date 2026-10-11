from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader
from drugex.data.interfaces import DataSet
from drugex.training.scorers.interfaces import Scorer

from abc import ABC
from abc import abstractmethod
import os
from pathlib import Path
from typing import Literal
from typing import Union
from drugex import DEFAULT_DEVICE
from drugex import DEFAULT_GPUS
from drugex.logs import logger
from torch import nn


class ModelEvaluator(ABC):
    """A simple function to score a model based on the generated molecules and input fragments if applicable."""

    @abstractmethod
    def __call__(self, mols: Sequence[str], frags: Optional[Sequence[str]] = ...) -> pd.DataFrame:
        """
        Score molecules.

        Parameters
        ----------
        mols : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules.
        """

        ...


class RewardScheme(ABC):
    """Reward scheme that enables ranking of molecules based on the calculated objectives and other criteria."""

    @abstractmethod
    def __call__(self, smiles: List[Any], scores: np.ndarray, thresholds: Sequence[float] | Sequence[Any]) -> np.ndarray:
        """
        Calculate the rewards for generated molecules and rank them according to teh given `RankingStrategy`.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to rank.
        scores : np.ndarray
            Matrix of scores for the multiple objectives
        thresholds : list
            List of thresholds for the objectives.

        Returns
        -------
        np.ndarray
            Array of rewards for the molecules.
        """

        ...


class Environment(ModelEvaluator):
    """Definition of the generic environment class for DrugEx. Reference implementation is `DrugExEnvironment`."""

    def __init__(self, scorers: List[Scorer], thresholds: Optional[List[float]] = ..., reward_scheme: Optional[RewardScheme] = ...) -> None:
        """
        Initialize environment with objective functions and their desirability thresholds. Molecules scoring above thresholds are desirable.

        Parameters
        ----------
        scorers : list
            List of objective functions to use for scoring molecules.
        thresholds : list, optional
            List of desirability thresholds for the objective functions.
            If `None`, all thresholds are set to 0.99.
        reward_scheme : RewardScheme, optional
            The reward scheme to use for ranking solutions. If `None`, the `DefaultRewardScheme` is used.

        Raises
        ------
        AssertionError
            If the number of scorers and thresholds does not match. In case of,
            multi-task scorer, a threshold should be provided for each task.

        Notes
        -----
        The `scorers` and `thresholds` are passed to the `RewardScheme` as well.
        """

        ...

    def __call__(self, smiles: Sequence[str], frags: Optional[Sequence[str]] = ...) -> pd.DataFrame:
        """
        Score molecules.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules.
        """

        ...

    @abstractmethod
    def getScores(self, smiles: List[str], frags: Optional[List[str]] = ..., no_multifrag_smiles: bool = ...) -> pd.DataFrame:
        """
        Calculate the scores of all objectives per molecule and qualify generated molecules (valid, accurate, desired).

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores and qualifications for the molecules.
        """

        ...

    @abstractmethod
    def getUnmodifiedScores(self, smiles: List[str]) -> pd.DataFrame:
        """
        Calculate the scores without applying modifiers of all objectives per molecule.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the scores for the molecules.
        """

        ...

    def getRewards(self, smiles: List[str], frags: Optional[List[str]] = ...) -> np.ndarray:
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning.

        Parameters
        ----------
        smiles : list
            List of SMILES strings of the molecules to score.
        frags : list, optional
            List of SMILES strings of the fragments used to generate the molecules.

        Returns
        -------
        np.ndarray
            Array of rewards for the molecules.
        """

        ...

    def getScorerKeys(self) -> List[str]:
        """
        Get the keys of the scorers.

        Returns
        -------
        list
            List of keys of the scorers.
        """

        ...


class ModelProvider(ABC):
    """Any instance that contains a DrugEx `Model` or its serialized form (i.e a state dictionary)."""

    @abstractmethod
    def getModel(self) -> Union[Model, dict[str, Any]]:
        """
        Return the current model as a `Model` instance or in serialized form.

        Returns
        -------
        Model or dict
            The current model or its serialized form.
        """

        ...


class Model(nn.Module, ModelProvider, ABC):
    """Generic base class for all PyTorch models in DrugEx. Manages the GPU or CPU gpus available to the model."""

    def __init__(self, device: Union[torch.device, str] = ..., use_gpus: Sequence[int] = ...) -> None:
        """
        Initialize the model with the given device and GPUs.

        Parameters
        ----------
        device : torch.device, optional
            The device to use for the model. If `None`, the default device is used.
        use_gpus : list, optional
            List of GPUs to use for the model. If `None`, the default GPUs are used.
        """

        ...

    def updateDevices(self, device: torch.device, gpus: Sequence[int]) -> None:
        """
        Update the device and GPUs used by the model.

        Parameters
        ----------
        device : torch.device
            The device to use for the model.
        gpus : list
            List of GPUs to use for the model.
        """

        ...

    @abstractmethod
    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """
        Use this method to handle a request to change the used GPUs. This method is automatically called when the class is instantiated, but may need to be called again in subclasses to move all data to the required devices.

        Subclasses should also make sure to set "self.device" to the currently used device and "self.gpus" to GPU ids of the currently used GPUs

        Parameters
        ----------
        gpus : tuple
            Tuple of GPU ids to use.
        """

        ...

    @abstractmethod
    def fit(self, train_loader: Any, valid_loader: Any, epochs: int = ..., monitor: Any = ..., **kwargs) -> None:
        """
        Train and validate the model with a given training and validation loader (see `DataSet` and its implementations docs to learn how to generate them).

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.
        valid_loader : torch.utils.data.DataLoader
            The validation data loader.
        epochs : int, optional
            The number of epochs to train the model for.
        monitor : TrainingMonitor, optional
            A `TrainingMonitor` instance to monitor the training process.     
        **kwargs
            Additional keyword arguments to pass to the training loop.
        """

        ...

    def loadStatesFromFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """
        Load the model states from a file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            The path to the file containing the model states.
        """

        ...

    def loadStates(self, state_dict: Dict[str, Any], strict: bool = ...) -> None:
        """
        Load the model states from a dictionary.

        Parameters
        ----------
        state_dict : dict
            The dictionary containing the model states.
        strict : bool, optional
            Whether to raise an error if the dictionary contains keys that do not match the model.
        """

        ...


class TrainingMonitor(ModelProvider, ABC):
    """Interface used to monitor model training."""

    @abstractmethod
    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = ...) -> None:
        """
        Save the state dictionary of the `Model` instance currently being trained or serialize the model any other way.

        Parameters
        ----------
        model : Model
            The model to save.
        identifier : str
            Suffix added to saved model.
        """

        ...

    @abstractmethod
    def savePerformanceInfo(self, performance_dict: Dict[str, Any], df_smiles: Optional[pd.DataFrame] = ...) -> None:
        """
        Save the performance data for the current epoch.

        Parameters
        ----------
        performance_dict : dict
            A dictionary with the performance data.
        df_smiles : pd.DataFrame
            A DataFrame with the SMILES of the molecules generated in the current epoch.
        """

        ...

    @abstractmethod
    def saveProgress(self, model: Model, current_step: Optional[int] = ..., current_epoch: Optional[int] = ..., total_steps: Optional[int] = ..., total_epochs: Optional[int] = ..., *args, **kwargs) -> None:
        """
        Notifies the monitor of the current progress of the training.

        Parameters
        ----------
        model : Model
            The model being trained.
        current_step : int, optional
            The current training step (i.e. batch).
        current_epoch : int, optional
            The current epoch.
        total_steps : int, optional
            The total number of training steps.
        total_epochs : int, optional
            The total number of epochs.
        *args
            Additional arguments depending on the model type.
        **kwargs
            Additional keyword arguments depending on the model type.
        """

        ...

    @abstractmethod
    def endStep(self, step: int, epoch: int) -> None:
        """
        Notify the monitor that a step of the training has finished.

        Parameters
        ----------
        step : int
            The current training step (i.e. batch).
        epoch : int
            The current epoch.
        """

        ...

    @abstractmethod
    def close(self) -> None:
        """Close this monitor. Training has finished."""

        ...

    @abstractmethod
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        """
        Return the scheme implemented by the monitor to save model snapshots.

        Returns
        -------
        Literal['best', 'all', 'improvement']
            The scheme implemented by the monitor to save model snapshots.
        """

        ...

