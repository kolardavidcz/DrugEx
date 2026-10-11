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
from drugex.training.interfaces import Environment
from drugex.training.monitors import TrainingMonitor as Monitor
from drugex.training.generators.interfaces import Generator
from drugex.training.interfaces import Environment, Model, ModelProvider, TrainingMonitor

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from scipy.stats import gmean
from torch import nn
from tqdm import tqdm
from drugex import DEFAULT_GPUS
from drugex import DEFAULT_DEVICE
from drugex.logs import logger
from drugex.training.interfaces import Model
from drugex.training.monitors import NullMonitor


class Explorer(Model, ABC):
    """Implements the DrugEx exploration strategy for DrugEx models under the reinforcement learning framework."""

    def __init__(self, agent: Generator, env: Environment, mutate: Optional[Generator] = ..., crover: Optional[Generator] = ..., no_multifrag_smiles: bool = ..., batch_size: int = ..., epsilon: float = ..., beta: float = ..., n_samples: int = ..., device: Union[torch.device, str] = ..., use_gpus: Sequence[int] = ...) -> None:
        """
        Initialize the explorer.

        Parameters
        ----------
        agent : Model
            The agent model optimized by the explorer.
        env : Environment
            The environment in which the agent operates.
        mutate : Generator
            The pre-trained network which increases the exploration of the chemical space.
        crover : Generator
            The iteratively updated network which increases the exploitation of the chemical space.
        no_multifrag_smiles : bool
            If True, only single-fragment SMILES are considered valid.
        batch_size : int
            The batch size used for training.
        epsilon : float
            The probability of using the mutate network to generate new molecules.
        beta : float
            The baseline for the reward function.
        n_samples : int
            The number of samples to generate in each iteration. If -1, the whole dataset is used per epoch.
        device : str
            The device on which the model is trained.
        use_gpus : tuple   
            The GPUs to use for training.
        """

        ...

    def attachToGPUs(self, gpus: tuple) -> Any:  # type: ignore[override]
        """
        Attach the model to GPUs

        Parameters
        ----------
        gpus : tuple
            The GPUs to use for training.
        """

        ...

    def getNovelMoleculeMetrics(self, scores: pd.DataFrame) -> Dict[str, Any]:
        """
        Get metrics for novel molecules

        Parameters
        ----------
        scores : pd.DataFrame
            The scores for each molecule.

        Returns
        -------
        dict
            The metrics:
                - valid_ratio (float): ratio of valid molecules
                - accurate_ratio (float, opt): the ratio of SMILES that are valid and have the desired fragments
                - unique_ratio (float): ratio of valid and unique molecules
                - desired_ratio (float): ratio of valid, unique and desired molecules
                - avg_amean (float): average arithmetic mean score of valid and unique molecules
                - avg_gmean (float): average geometric mean score of valid and unique molecules
        """

        ...

    def saveBestState(self, value: float, epoch: int, it: int) -> Any:
        """
        Save best state based on selection criteria

        Parameters
        ----------
        value : float
            The value of the selection criteria.
        epoch : int
            The current epoch.
        it : int
        """

        ...

    def logPerformanceAndCompounds(self, epoch: int, metrics: Dict[str, Any], scores: pd.DataFrame) -> Any:
        """
        Log performance of model

        Parameters
        ----------
        epoch : int
            The current epoch.
        metrics : dict
            The metrics.
        scores : pd.DataFrame
            The scores for each molecule.
        """

        ...

    @abstractmethod
    def fit(self, train_loader: Any, valid_loader: Any = ..., epochs: int = ..., monitor: Any = ...) -> Any:  # type: ignore[override]
        """Abstract RL training loop updating agent policy weights across iterations until early stopping or epoch exhaustion."""

        ...

    @abstractmethod
    def policy_gradient(self, *args, **kwargs) -> Any:
        """Policy gradient training."""

        ...

    def getModel(self) -> torch.nn.Module:  # type: ignore[override]
        """
        Returns the current state of the agent

        Returns
        -------
        torch.nn.Module
            The current state of the agent
        """

        ...


class FragExplorer(Explorer):
    """
    Implements the DrugEx exploration strategy for DrugEx models under 
    the reinforcement learning framework for fragment-based generators
    """

    @abstractmethod
    def getBatchOutputs(self: Any, src: Any, net: Any) -> Any:
        """Outputs (frags, smiles) and loss of the agent for a batch of fragments-molecule pairs."""

        ...

    def policy_gradient(self, loader: torch.utils.data.DataLoader) -> float:
        """
        Policy gradient training.

        Novel molecules are generated by the agent and are scored by the environment.
        The policy gradient is calculated using the REINFORCE algorithm and the agent is updated.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Data loader for training data

        Returns
        -------
        float
            The average loss of the agent
        """

        ...

    @abstractmethod
    def sampleEncodedPairsToLoader(self, net: Any, loader: Any) -> Any:
        """Sample new fragments-molecule pairs from a data loader."""

        ...

    @abstractmethod
    def sample_input(self, loader: Any, is_test: Any = ...) -> Any:
        """Sample a batch of fragments-molecule pairs from the training data loader."""

        ...

    def fit(self, train_loader: Any, valid_loader: Any = ..., epochs: int = ..., patience: int = ..., criteria: str = ..., min_epochs: int = ..., monitor: Any = ...) -> Any:  # type: ignore[override]
        """
        Fit the graph explorer to the training data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Data loader for training data
        valid_loader : torch.utils.data.DataLoader
            Data loader for validation data
        epochs : int
            Number of epochs to train for
        patience : int
            Number of epochs to wait for improvement before early stopping
        criteria : str
            Criteria to use for early stopping
        min_epochs : int
            Minimum number of epochs to train for
        monitor : Monitor
            Monitor to use for logging and saving model
        """

        ...

