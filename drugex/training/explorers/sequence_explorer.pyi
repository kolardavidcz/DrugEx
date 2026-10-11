from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.training.interfaces import Environment
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.monitors import TrainingMonitor as Monitor

import time
from copy import deepcopy
from drugex import DEFAULT_DEVICE
from drugex import DEFAULT_GPUS
from drugex import utils
from drugex.logs import logger
from drugex.training.explorers.interfaces import Explorer
from drugex.training.generators.utils import unique
from drugex.training.monitors import NullMonitor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm


class SequenceExplorer(Explorer):
    """
    `Explorer` to optimize a sequence-based agent (RNN) with the given `Environment`.

    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6
    """

    def __init__(self, agent: SequenceRNN, env: Environment, mutate: Optional[SequenceRNN] = ..., crover: Optional[SequenceRNN] = ..., no_multifrag_smiles: bool = ..., batch_size: int = ..., epsilon: float = ..., beta: float = ..., n_samples: int = ..., optim: Optional[torch.optim.Optimizer] = ..., device: Union[torch.device, str] = ..., use_gpus: Sequence[int] = ...) -> None:
        """Initialize sequence RNN explorer with agent, environment, mutation/crossover networks, and optimizer."""

        ...

    def forward(self) -> List[Any]:
        """
        Generate molecules with the given `agent` network

        Returns
        -------
        smiles : list
            The generated SMILES.
        seqs : torch.Tensor
            The generated encoded sequences.
        """

        ...

    def policy_gradient(self, smiles: List[Any] = ..., seqs: torch.Tensor = ..., epoch: Optional[int] = ..., epochs: Optional[int] = ...) -> float:
        """
        Policy gradient training.

        Novel molecules are scored by the environment.
        The policy gradient is calculated using the REINFORCE algorithm and the agent is updated.

        Parameters
        ----------
        smiles : list
            The generated SMILES.
        seqs : torch.Tensor
            The generated encoded sequences. 
        epoch : int, optional
            The current epoch number, used for logging.
        epochs : int, optional
            The total number of epochs to train, used for logging.

        Returns
        -------
        loss : float
            The loss of the policy gradient.
        """

        ...

    def fit(self, train_loader: Any = ..., valid_loader: Any = ..., monitor: Any = ..., epochs: int = ..., patience: int = ..., reload_interval: int = ..., criteria: str = ..., min_epochs: int = ...) -> Any:  # type: ignore[override]
        """
        Fit the graph explorer to the training data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            ignored, for compatibility with `FragExplorer`
        valid_loader : torch.utils.data.DataLoader
            ignored, for compatibility with `FragExplorer`
        epochs : int
            Number of epochs to train for
        monitor : Monitor
            Monitor to use for logging and saving model
        patience : int
            Number of epochs to wait for improvement before early stopping
        reload_interval : int
            Every nth epoch reset the agent (and the crover) network to the best state
        criteria : str
            Criteria to use for early stopping: 'desired_ratio', 'avg_amean' or 'avg_gmean'
        min_epochs : int
            Minimum number of epochs to train for
        """

        ...

