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
from drugex.training.scorers.smiles import SmilesChecker


class DrugExEnvironment(Environment):
    """Original implementation of the environment scoring strategy for DrugEx v3."""

    def getScores(self, smiles: List[str], frags: Optional[List[str]] = ..., no_multifrag_smiles: bool = ...) -> pd.DataFrame:
        """
        This method is used to get the scores from the scorers and to check molecule validity and desireability.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to score.
        frags : list of str, optional
            List of SMILES strings of fragments to check for.
        no_multifrag_smiles : bool, optional
            Whether to check for SMILES strings that contain more than one fragment.

        Returns
        -------
        scores : pd.DataFrame
            Dataframe with the scores from the scorers and the validity and desireability of the molecules.
        """

        ...

    def getUnmodifiedScores(self, smiles: List[str]) -> pd.DataFrame:
        """
        This method is used to get the scores from the scorers without any modifications.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to score.

        Returns
        -------
        preds : pd.DataFrame
            Dataframe with the scores from the scorers.
        """

        ...

