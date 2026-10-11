from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from rdkit.Chem.MolStandardize import rdMolStandardize
from drugex.logs import logger
from drugex.molecules.converters.interfaces import MolConverter
from drugex.molecules.converters.interfaces import ConversionException


class StandardizationException(ConversionException):
    """Custom exception class to recognize and catch standardization errors."""

    ...


class DefaultStandardizer(MolConverter):
    """Original standardization implementation from the original DrugEx v3."""

    def __call__(self, smiles: str) -> str:
        """
        Takes smiles of the input molecule and converts it to a standardized represenation.

        Raises:
            StandardizationException: thrown when the standardizer encountered a failure

        Args:
            smiles: input molecule as SMILES

        Returns:
            converted SMILES as `str`
        """

        ...


class CleanSMILES(MolConverter):
    """Converter used to clean SMILES strings at some places. At the moment the reasons for its existence are unclear..."""

    def __init__(self, is_deep: bool = ...) -> None:
        ...

    def __call__(self, smile: str) -> str:
        """Normalizes radical and isotope brackets, applies charge parent standardization, and returns canonical SMILES."""

        ...

