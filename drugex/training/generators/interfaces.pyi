"""
Base interfaces and abstract classes for molecule generators in DrugEx.
"""

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
from drugex.training.interfaces import Environment
from drugex.training.monitors import TrainingMonitor
from drugex.training.monitors import TrainingMonitor as Monitor
from drugex.training.interfaces import ModelEvaluator

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Optional
from typing import Sequence
from tqdm.auto import tqdm
from drugex.logs import logger
from drugex.training.interfaces import Model
from drugex.training.interfaces import TrainingMonitor
from drugex.training.monitors import NullMonitor
from drugex.training.scorers.smiles import SmilesChecker


class Generator(Model, ABC):
    """
    Abstract base class for all molecule generators in DrugEx.

    Defines the unified lifecycle contract for generative architectures, including
    sampling, single-epoch training, multi-metric validation, molecule filtering,
    and objective-based evaluation.

    Inherits from `drugex.training.interfaces.Model` and `torch.nn.Module`.
    """

    @abstractmethod
    def sample(self: Any, *args, **kwargs) -> Tuple[list[str], Optional[list[str]]] | list[str]:
        """
        Sample molecules from the generator.

        Parameters
        ----------
        *args : Any
            Architecture-specific positional sampling arguments (e.g. `batch_size` or `loader`).
        **kwargs : Any
            Architecture-specific keyword arguments.

        Returns
        -------
        smiles : list of str
            Sampled SMILES representations of generated compounds.
        frags : list of str, optional
            Input fragments corresponding to each generated molecule (for fragment-based models).
        """

        ...

    @abstractmethod
    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Train the generator network for a single epoch.

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader providing training batches of encoded sequences or graphs.
        epoch : int
            Current epoch index (1-indexed).
        epochs : int
            Total planned training epochs.

        Returns
        -------
        loss : float
            Representative training loss for the current epoch.
        """

        ...

    @abstractmethod
    def validateNet(self, loader: Any = ..., evaluator: Any = ..., no_multifrag_smiles: bool = ..., n_samples: Any = ...) -> Tuple[dict[str, float], pd.DataFrame]:
        """
        Validate generator performance, compute validation loss, and evaluate sample quality.

        Parameters
        ----------
        loader : DataLoader, optional
            Validation DataLoader to calculate validation set likelihood or loss.
        evaluator : Environment, optional
            Reinforcement learning environment or model evaluator providing property scores.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES containing '.' are classified invalid
            (default: True).
        n_samples : int, optional
            Number of molecules to sample for statistical quality checks (used by sequence models).

        Returns
        -------
        valid_metrics : dict of str to float
            Validation metrics dictionary (e.g. `'valid_ratio'`, `'accurate_ratio'`, `'loss_valid'`).
        smiles_scores : pd.DataFrame
            DataFrame containing sampled SMILES, validity indicators, and property scores.
        """

        ...

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate molecules until target quota is satisfied, applying filtering and scoring.

        Returns
        -------
        df_smiles : pd.DataFrame
            DataFrame containing generated SMILES, optional input fragments, and objective scores.
        """

        ...

    def filterNewMolecules(self, df_old: pd.DataFrame, df_new: pd.DataFrame, with_frags: bool = ..., drop_duplicates: bool = ..., drop_undesired: bool = ..., evaluator: Optional[Environment] = ..., no_multifrag_smiles: bool = ...) -> pd.DataFrame:
        """
        Filter newly generated molecules against cumulative history and quality criteria.

        Performs chemical validity verification, canonicalization, deduplication against
        both current batch and historical generation pool, fragment consistency checks,
        and threshold filtering via the provided evaluator.

        Parameters
        ----------
        df_old : pd.DataFrame
            DataFrame of previously accepted molecules. Must contain `'SMILES'` column.
        df_new : pd.DataFrame
            DataFrame of newly sampled molecules to filter. Must contain `'SMILES'` column.
        with_frags : bool, optional
            If True, expects `'Frags'` column and verifies fragment presence in generated compounds
            (default: True).
        drop_duplicates : bool, optional
            If True, removes duplicate SMILES existing within `df_new` or already present in `df_old`
            (default: True).
        drop_undesired : bool, optional
            If True, filters out molecules that fail the minimum desirability thresholds of `evaluator`.
            Requires `evaluator` to be non-None (default: True).
        evaluator : Environment, optional
            Reinforcement learning environment or scoring instance used to evaluate desirability.
        no_multifrag_smiles : bool, optional
            If True, only single-fragment connected SMILES are considered valid (default: True).

        Returns
        -------
        df_filtered : pd.DataFrame
            Filtered subset of `df_new` retaining only valid, unique, and desirable molecules.
        """

        ...

    def evaluate(self, smiles: List[str], frags: Optional[list[str]] = ..., evaluator: Optional[Environment] = ..., no_multifrag_smiles: bool = ..., unmodified_scores: bool = ...) -> pd.DataFrame:
        """
        Evaluate molecules for chemical validity and property scores.

        If `evaluator` is omitted, performs structural validity and fragment containment
        verification via `SmilesChecker`. If `evaluator` is provided, calculates full
        multi-objective reward scores across target properties.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to evaluate.
        frags : list of str, optional
            List of input fragments corresponding to each SMILES (for fragment-constrained models).
        evaluator : Environment, optional
            Environment instance scoring molecules across bioactivity, QSAR, and physicochemical objectives.
        no_multifrag_smiles : bool, optional
            If True, disconnected multi-fragment SMILES are penalized as invalid (default: True).
        unmodified_scores : bool, optional
            If True, calculates raw scores before non-linear score modifier transformations (default: False).

        Returns
        -------
        scores : pd.DataFrame
            DataFrame indexed by molecule, containing validity indicators and score columns.
        """

        ...

    def logPerformanceAndCompounds(self, epoch: int, metrics: Dict[str, Any], scores: pd.DataFrame) -> None:
        """
        Log performance metrics and generated molecules to the monitor.

        Parameters
        ----------
        epoch : int
            The current epoch index.
        metrics : dict of str to Any
            Dictionary containing performance metrics.
        scores : pd.DataFrame
            DataFrame with generated molecules, validity flags, and objective scores.
        """

        ...

    def fit(self, train_loader: Any, valid_loader: Any = ..., epochs: int = ..., patience: int = ..., evaluator: Optional[Environment] = ..., monitor: Any = ..., no_multifrag_smiles: bool = ..., loss_tolerance: Optional[float] = ..., **kwargs) -> None:  # type: ignore[override]
        """
        Train and validate the generator network over multiple epochs with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing training batches.
        valid_loader : DataLoader, optional
            DataLoader providing hold-out validation batches.
        epochs : int, optional
            Maximum number of training epochs (default: 100).
        patience : int, optional
            Number of epochs to wait without improvement before stopping early (default: 50).
        evaluator : Environment, optional
            Environment or ModelEvaluator scoring compound quality during validation.
        monitor : TrainingMonitor, optional
            Monitor instance logging progress, metrics, and checkpoints (default: NullMonitor).
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / salt SMILES are penalized as invalid (default: True).
        loss_tolerance : float, optional
            Allowed increase in validation loss if valid_ratio improves.
        **kwargs : Any
            Additional keyword arguments for compatibility with Model.fit.
        """

        ...

    def getModel(self) -> Dict[str, Any]:
        """
        Return an in-memory deep copy of the model state dictionary.

        Returns
        -------
        model_state : dict of str to Any
            PyTorch state dictionary containing model weights and parameters.
        """

        ...


class FragGenerator(Generator):
    """
    Abstract base class for fragment-constrained molecule generators.

    Extends `Generator` to support fragment-based de novo design, where molecules
    are elaborated, connected, or scaffold-hopped starting from pre-defined chemical moieties.
    """

    def init_states(self) -> None:
        """
        Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to all multi-dimensional weight tensors
        (excluding 1D biases and embedding layers) and attaches the model to assigned GPUs.
        """

        ...

    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """
        Attach model to specified GPU device IDs and transfer model tensors.

        Parameters
        ----------
        gpus : sequence of int
            Sequence or tuple of GPU hardware device IDs.
        """

        ...

    @abstractmethod
    def loaderFromFrags(self, frags: List[str], batch_size: int = ..., n_proc: int = ...) -> DataLoader:
        """
        Encode input fragments and construct an iterable DataLoader.

        Parameters
        ----------
        frags : list of str
            List of input fragments in SMILES format.
        batch_size : int, optional
            Batch size for DataLoader batches (default: 32).
        n_proc : int, optional
            Worker processes for parallel fragment encoding (default: 1).

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding encoded fragment tensors.
        """

        ...

    @abstractmethod
    def decodeLoaders(self, src: torch.Tensor, trg: torch.Tensor) -> Tuple[List[str], list[str]]:
        """
        Decode source fragment tensors and generated target tensors into SMILES strings.

        Parameters
        ----------
        src : torch.Tensor
            Batch of encoded input fragments.
        trg : torch.Tensor
            Batch of encoded generated molecules.

        Returns
        -------
        frags : list of str
            Decoded input fragment SMILES.
        smiles : list of str
            Decoded generated molecule SMILES.
        """

        ...

    @abstractmethod
    def iterLoader(self, loader: DataLoader) -> Any:
        """
        Yield source tensor batches from the fragment DataLoader.

        Parameters
        ----------
        loader : DataLoader
            Input DataLoader.

        Yields
        ------
        src : torch.Tensor
            Batch of input fragment tensors.
        """

        ...

    def generate(self, input_frags: Optional[list[str]] = ..., input_dataset: Optional[DataSet] = ..., num_samples: int = ..., batch_size: int = ..., n_proc: int = ..., keep_frags: bool = ..., drop_duplicates: bool = ..., drop_invalid: bool = ..., evaluator: Optional[Environment] = ..., no_multifrag_smiles: bool = ..., drop_undesired: bool = ..., raw_scores: bool = ..., progress: bool = ..., tqdm_kwargs: Optional[dict[str, Any]] = ...) -> pd.DataFrame:
        """
        Generate molecules starting from input fragments or a dataset.

        Iteratively generates molecules by elaborating or linking input fragments until
        `num_samples` valid compounds are collected.

        Parameters
        ----------
        input_frags : list of str, optional
            List of input fragment SMILES to incorporate into generated compounds.
            Mutually exclusive with `input_dataset`.
        input_dataset : DataSet, optional
            Pre-encoded fragment dataset (e.g. `SmilesFragDataSet` or `GraphFragDataSet`).
            Mutually exclusive with `input_frags`.
        num_samples : int, optional
            Total target count of valid molecules to generate (default: 100).
        batch_size : int, optional
            Batch size per generation iteration (default: 32).
        n_proc : int, optional
            Processes used for encoding if `input_frags` is provided (default: 1).
        keep_frags : bool, optional
            If True, retains `'Frags'` column in the output DataFrame (default: True).
        drop_duplicates : bool, optional
            If True, duplicate molecules are discarded (default: True).
        drop_invalid : bool, optional
            If True, chemically invalid molecules are excluded (default: True).
        evaluator : Environment, optional
            Scoring environment evaluating generated compounds.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES are marked invalid (default: True).
        drop_undesired : bool, optional
            If True, compounds failing evaluator desirability criteria are discarded (default: False).
        raw_scores : bool, optional
            If True, includes raw unpenalized scores from scorers (default: True).
        progress : bool, optional
            If True, displays a tqdm progress bar tracking generation (default: True).
        tqdm_kwargs : dict, optional
            Additional keyword arguments passed to `tqdm` (default: None).

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing generated compounds with columns `['SMILES', 'Frags']`
            and scorer property scores if evaluator was specified.
        """

        ...

