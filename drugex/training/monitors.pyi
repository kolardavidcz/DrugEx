from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.training.interfaces import TrainingMonitor as TrainingMonitor, Model as Model

import copy
import os
import os.path
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union
from drugex.logs import logger
from drugex.training.interfaces import TrainingMonitor
from drugex.training.interfaces import Model

__all__ = ['TrainingMonitor', 'Model', 'NullMonitor', 'FileMonitor', 'PeakRecoveryMonitor']

class NullMonitor(TrainingMonitor):
    """No-op training monitor discarding metrics and checkpoints during silent runs."""

    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        """Always returns 'all' to maintain interface compatibility with trainer serialization workflows."""

        ...

    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = ...) -> None:
        """Silent no-op discarding model weights without saving."""

        ...

    def savePerformanceInfo(self, performance_dict: Dict[str, Any], df_smiles: Optional[pd.DataFrame] = ...) -> None:
        """Silent no-op discarding metrics dictionaries and candidate SMILES tables."""

        ...

    def saveProgress(self, model: Model, current_step: Optional[int] = ..., current_epoch: Optional[int] = ..., total_steps: Optional[int] = ..., total_epochs: Optional[int] = ..., *args, **kwargs) -> None:
        """Silent no-op ignoring step and epoch progression notifications."""

        ...

    def endStep(self, step: int, epoch: int) -> None:
        """Silent no-op receiving batch step completion signal."""

        ...

    def close(self) -> None:
        """Silent no-op terminating monitor session without disk I/O."""

        ...

    def getModel(self) -> None:  # type: ignore[override]
        """Returns None as no model state is serialized or retained."""

        ...


class FileMonitor(TrainingMonitor):
    """A simple `TrainingMonitor` implementation with file outputs."""

    def __init__(self, path: Union[str, os.PathLike, Path], save_smiles: bool = ..., save_model_option: Literal['best', 'all', 'improvement'] = ..., reset_directory: bool = ..., on_model_update: Optional[Callable[[Model], None]] = ...) -> None:
        """
        Initialize the file monitor.

        The monitor will create three/four files:
        - `path`_fit.tsv - a TSV file with the performance data for each epoch
        - `path`_fit.log - a log file with the training progress
        - `path`_smiles.tsv - a TSV file with the SMILES of the molecules generated in each epoch (if `save_smiles` is True)
        - `path`.pkg - a PyTorch package with the model state

        Parameters
        ----------
        path : str or Path
            The path and prefix of the files to be created (i.e. /tmp/drugex_rl/experiment_01). This will ensure all files
            are saved to the given directory and have the given prefix.
        save_smiles : bool
            Whether to save the SMILES of the molecules generated in each epoch.
        save_model_option : str
            Determines which models to save during training. Use this parameter with care as saving a large number of
            models is extremely memory-intensive. Possible values:
            - 'all' : Save all models.
            - 'improvement': Save all models that improve upon the previous best model.
            - 'best' (default): Save only the final best model.
            WARNING: Setting this option to 'all' or 'best' can be extremely memory-intensive, Use with caution and 
            ensure you have sufficient memory resources.
        reset_directory : bool
            Whether to reset the directory where the files are to be saved. If `True`, all files
            with the given prefix will be removed from the directory upon creation of the monitor.
        on_model_update : callable
            A callable that will be called after each model update/epoch. The callable will be passed a `Model` subclass
            instance as the only argument.
        """

        ...

    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = ...) -> None:
        """Save the model state."""

        ...

    def saveProgress(self, model: Model, current_step: Optional[int] = ..., current_epoch: Optional[int] = ..., total_steps: Optional[int] = ..., total_epochs: Optional[int] = ..., loss: Optional[float] = ..., *args, **kwargs) -> None:
        """
        Save the current training progress: epoch, step, loss.

        Parameters
        ----------
        model : Model
            The model currently being trained.
        current_step : int
            The current step.
        current_epoch : int
            The current epoch.
        total_steps : int
            The total number of steps.
        total_epochs : int 
            The total number of epochs.
        loss : float
            The current training loss.
        """

        ...

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

    def saveEpochData(self, df: pd.DataFrame) -> None:
        """Appends epoch performance evaluation DataFrame to the primary telemetry file ({path}_fit.tsv)."""

        ...

    def saveMolecules(self, df: pd.DataFrame) -> None:
        """Appends batch of generated candidate SMILES and objective scores to {path}_smiles.tsv."""

        ...

    @staticmethod
    def appendTableToFile(df: pd.DataFrame, outfile: Any) -> None:
        """Appends tabular DataFrame rows to a target TSV file on disk, automatically creating header rows on initial write."""

        ...

    def endStep(self, step: int, epoch: int) -> None:
        """Signals completion of a single batch step within an epoch (no-op in FileMonitor base)."""

        ...

    def close(self) -> None:
        """Flushes pending write buffers and closes active text file handles (outLog)."""

        ...

    def getModel(self) -> Optional[dict[str, Any]]:  # type: ignore[override]
        """Returns cached model state dictionary from the most recent saveModel call, or None if uninitialized."""

        ...

    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        """Returns active model serialization strategy ('best', 'all', or 'improvement')."""

        ...


class PeakRecoveryMonitor(FileMonitor):
    """
    Training monitor with rolling desirability tracking, checkpoint persistence, and automatic peak model recovery.

    Tracks multi-objective desirability ('desired_ratio') across epochs, persists model weights
    as '{path}_peak.pkg' upon achieving a new best value, and automatically restores peak historical
    weights onto the agent neural network in close() or via restore_peak_model().
    """

    def __init__(self, path: Optional[Union[str, os.PathLike, Path]] = ..., output_prefix: Optional[Union[str, os.PathLike, Path]] = ..., save_smiles: bool = ..., save_model_option: Literal['best', 'all', 'improvement'] = ..., reset_directory: bool = ..., criteria: str = ..., patience: int = ..., min_epochs: int = ..., on_model_update: Optional[Callable[[Model], None]] = ...) -> None:
        """Initialize PeakRecoveryMonitor with file path, criteria, and early stopping limits."""

        ...

    def saveModel(self, model: Any, identifier: Any = ...) -> None:
        """Capture target explorer/model reference and delegate serialization to FileMonitor."""

        ...

    def step(self, epoch: int, desirability: float, model_state: Any) -> bool:
        """
        Record epoch desirability and persist peak checkpoint on improvement.

        Parameters:
            epoch: Current training epoch index.
            desirability: Multi-objective desirability metric value (e.g. desired_ratio).
            model_state: Model state dictionary or object exposing state_dict().

        Returns:
            bool: True if training should continue, False if early stopping triggered.
        """

        ...

    def savePerformanceInfo(self, performance_dict: Dict[str, Any], df_smiles: Any = ...) -> None:
        """Track rolling performance metrics and store peak model checkpoint."""

        ...

    def restore_peak_model(self, agent: Any = ...) -> Any:
        """Restore peak model weights onto agent neural network or return peak state dict."""

        ...

    def close(self) -> None:
        """Auto-restore peak model weights on training completion or early stopping."""

        ...

