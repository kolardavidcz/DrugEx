"""
Scorer integration wrapping QSPRpred surrogate predictive models.
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
QSPRModel: Any = ...

from typing import TYPE_CHECKING
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union
from drugex.logs import logger
from drugex.training.scorers.interfaces import ScoreModifier
from drugex.training.scorers.interfaces import Scorer
from qsprpred.tasks import TargetProperty
from qsprpred.tasks import TargetTasks


class QSPRPredScorer(Scorer):
    """
    Scorer wrapping a QSPRpred surrogate predictive model.

    Evaluates generated molecules using quantitative structure-property relationship (QSPR)
    models (e.g. Random Forest, SVM, Deep Neural Networks, Chemprop). Supports regression,
    binary classification, multi-class classification, multi-task prediction, and
    applicability domain estimation.

    Parameters
    ----------
    model : QSPRModel
        Trained QSPRpred model instance used to make predictions.
    use_probas : bool, optional
        If True, returns continuous class probabilities for classification models.
        If False, returns hard discrete class predictions (default: True).
    tasks : list of str, optional
        Subset of target tasks to compute if `model` is a multi-task predictor.
        If None, all tasks defined on the model are scored (default: None).
    classes : list of int, optional
        Specific class indices to extract probabilities for in multi-class classification.
        If None, all classes are returned as separate task columns (default: None).
    app_domain : bool or str, optional
        Controls applicability domain (AD) evaluation:
        - `False`: applicability domain is ignored.
        - `True`: adds an additional score column `"{key}_app_domain"` with boolean/distance flags.
        - `'invalid'`: assigns `invalids_score` to any molecule falling outside the applicability domain.
        Default is False.
    invalids_score : float or list of float, optional
        Default score assigned to unparseable SMILES or out-of-domain compounds.
        Can be a scalar or a list matching the task count (default: 0.0).
    modifier : ScoreModifier or list of ScoreModifier, optional
        Score transformation function(s) mapping model outputs to desirability rewards.
    **kwargs : Any
        Additional keyword arguments forwarded to `model.predictMols`.

    Attributes
    ----------
    model : QSPRModel
        Underlying surrogate model.
    use_probas : bool
        Probability prediction flag.
    tasks : list of str, optional
        Selected task names.
    app_domain : bool or str
        Applicability domain setting.
    invalidsScore : float or list of float
        Fallback penalty score for invalid inputs.
    """

    def __init__(self, model: QSPRModel, use_probas: bool = ..., tasks: Optional[list[str]] = ..., classes: Optional[list[int]] = ..., app_domain: Union[bool, str] = ..., invalids_score: Union[float, list[float]] = ..., modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]] = ..., **kwargs) -> None:
        ...

    def getScores(self, mols: Sequence[Union[str, Chem.Mol]], frags: Optional[Sequence[Union[str, Chem.Mol]]] = ...) -> np.ndarray:
        """
        Calculate surrogate model predictions or probabilities for input molecules.

        Parameters
        ----------
        mols : sequence of str or rdkit.Chem.Mol
            List of SMILES or RDKit molecules to predict.
        frags : sequence of str or rdkit.Chem.Mol, optional
            Ignored (retained for Scorer interface parity).

        Returns
        -------
        scores : np.ndarray
            Predicted values array of shape `(n_mols,)` for single-task,
            or `(n_mols, n_tasks)` for multi-task.
        """

        ...

    def _get_valid_molecules(self, mols: Sequence[Union[str, Chem.Mol]]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert input to RDKit Mol objects and filter valid entries."""

        ...

    def _get_predictions(self, mols: np.ndarray) -> np.ndarray:
        """Query QSPRpred model predictions and applicability domain."""

        ...

    def handle_regression_task(self, scores: np.ndarray) -> np.ndarray:
        """Filter selected columns for multi-task regression."""

        ...

    def handle_classification_task(self, scores: Union[list[np.ndarray], np.ndarray]) -> np.ndarray:
        """Extract positive class or multi-class probability columns."""

        ...

    def handle_app_domain(self, scores: np.ndarray, app: Any) -> np.ndarray:
        """Apply applicability domain gating or column concatenation."""

        ...

    def getKey(self) -> Union[str, list[str]]:
        """
        Return the objective column key(s) generated by this QSPRpred model.

        Returns
        -------
        key : str or list of str
            Evaluation column name(s).
        """

        ...

    @property
    def nTasks(self) -> int:
        """Number of distinct task score outputs."""

        ...

