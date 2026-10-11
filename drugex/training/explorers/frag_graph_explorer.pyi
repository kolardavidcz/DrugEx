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
from drugex.training.explorers.interfaces import Explorer
from drugex.training.generators.graph_transformer import GraphTransformer

from torch.optim import Adam
from drugex import utils
from drugex import DEFAULT_DEVICE
from drugex import DEFAULT_GPUS
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from drugex.logs import logger
from drugex.training.explorers.interfaces import FragExplorer
from drugex.training.generators.utils import tri_mask


class FragGraphExplorer(FragExplorer):
    """`Explorer` to optimize a graph-based fragment-using agent with the given `Environment`."""

    def __init__(self, agent: GraphTransformer, env: Optional[Environment], mutate: Optional[GraphTransformer] = ..., crover: Optional[GraphTransformer] = ..., batch_size: int = ..., epsilon: float = ..., beta: float = ..., n_samples: int = ..., optim: Optional[torch.optim.Optimizer] = ..., device: Union[torch.device, str] = ..., use_gpus: Sequence[int] = ..., no_multifrag_smiles: bool = ...) -> None:
        """Initialize graph explorer with agent network, environment, mutation/crossover networks, and optimizer."""

        ...

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the agent.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, 80, 5).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 80, 5).
        """

        ...

    def sample_input(self, loader: torch.utils.data.DataLoader, is_test: bool = ...) -> torch.utils.data.DataLoader:
        """
        Sample a batch of fragments-molecule pairs from the dataset.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Data loader for original fragments-molecule pairs
        is_test : bool
            Whether to sample from the validation set or not

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for sampled fragments-molecule pairs
        """

        ...

    def getBatchOutputs(self: Any, net: Any, src: Any) -> List[Any]:  # type: ignore[override]
        """
        Outputs (frags, smiles) and loss of the agent for a batch of fragments-molecule pairs.

        Parameters
        ----------
        net : torch.nn.Module
            Agent
        src : torch.Tensor
            Fragments-molecule pairs

        Returns
        -------
        frags : list
            List of fragments
        smiles : list
            List of SMILES
        loss : torch.Tensor
            Loss of the agent
        """

        ...

    def sampleEncodedPairsToLoader(self, net: torch.nn.Module, loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """
        Sample new fragments-molecule pairs from a data loader.

        Parameters
        ----------
        net : torch.nn.Module
            Agent
        loader : torch.utils.data.DataLoader
            Data loader for original fragments-molecule pairs

        Returns
        -------
        torch.utils.data.DataLoader
            Data loader for sampled fragments-molecule pairs
        """

        ...

