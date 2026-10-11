from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.molecules.converters.interfaces import MolConverter

from drugex.molecules.converters.default import SmilesToDrEx
from drugex.molecules.interfaces import BaseMolSupplier


class StandardizedSupplier(BaseMolSupplier):
    """Supplies standardized molecules from the input molecules. It requires a standardizer."""

    def __init__(self, mols: Iterable[Any], standardizer: MolConverter) -> None:
        """
        Initialize a standardized supplier with the given standardizer.

        Args:
            mols: A set of input molecules. It will be converted to an iterator if not one already.
            standardizer: a `MolConverter` to transform items in 'mols' to the standardized form. It should return the same representation as in the original 'mols' (i.e. SMILES if the original input was SMILES).
        """

        ...

    def next(self) -> Any:
        """
        Defines access to the next item to be processed.

        Returns:
            next molecule for processing
        """

        ...


class ListSupplier(BaseMolSupplier):
    """Basic supplier that converts molecules in a list to the desired representation (SMILES string to `DrExMol` by default)."""

    def __init__(self, mols: Iterable[Any], converter: MolConverter = ..., hide_duplicates: bool = ...) -> None:
        """
        Initialize list supplier.

        Args:
            mols (list): A list of molecules. By default, representation as SMILES is assumed.
            converter (MolConverter): Converter to transform molecules in 'mols' to the desired representation, `SmilesToDrEx` by default.
        """

        ...

    def next(self) -> Any:
        """Return the next raw molecule item from the list iterator."""

        ...

