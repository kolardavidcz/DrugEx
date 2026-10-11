"""
THESE FUNCTIONS WERE ADDED BY HELLE FROM:
https://github.com/BenevolentAI/guacamol/blob/8247bbd5e927fbc3d328865d12cf83cb7019e2d6/guacamol/utils/data.py#L11
to solve AttributeError: module 'utils' has no attribute 'canonicalize_list'
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
T = TypeVar('T')

from typing import Optional
from typing import Iterable
from typing import List


def canonicalize(smiles: str, include_stereocenters: bool = ...) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    ...


def remove_duplicates(list_with_duplicates: Iterable[T]) -> List[T]:
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.
    Args:
        list_with_duplicates: list that possibly contains duplicates
    Returns:
        A list with no duplicates.
    """

    ...


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters: bool = ...) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.
    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings
    Returns:
        The canonicalized and filtered input smiles.
    """

    ...

