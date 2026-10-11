"""
Path resolution and data discovery utilities for DrugEx datasets.
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

import os
from typing import List
from typing import Sequence
from typing import Tuple
from drugex import VERSION
from drugex.logs import logger


def getVocPaths(data_path: str, voc_files: Sequence[str], mol_type: str) -> List[str]:
    """
    Retrieve filesystem paths to vocabulary files, logging fallbacks if absent.

    Parameters
    ----------
    data_path : str
        Base path to the data directory.
    voc_files : sequence of str
        List of candidate vocabulary file names.
    mol_type : str, optional
        Representation format (`'smiles'` or `'graph'`) used for suffixed fallbacks (default: `'smiles'`).

    Returns
    -------
    voc_paths : list of str
        List of resolved, existing vocabulary file paths.
    """

    ...


def getDataPaths(data_path: str, input_prefix: str, mol_type: str, unique_frags: bool) -> Tuple[str, str]:
    """
    Resolve filesystem paths for training and validation datasets.

    Parameters
    ----------
    data_path : str
        Base directory containing dataset files.
    input_prefix : str
        Prefix string identifying dataset files.
    mol_type : str, optional
        Representation format (`'smiles'` or `'graph'`) (default: `'smiles'`).
    unique_frags : bool, optional
        If True, targets unique fragment splits rather than standard training splits (default: False).

    Returns
    -------
    paths : tuple of (str, str)
        Resolved `(train_path, test_path)`.

    Raises
    ------
    AssertionError
        If resolved train or test file paths do not exist on disk.
    """

    ...

