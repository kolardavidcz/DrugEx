"""
Synthetic Accessibility (SA) score calculation based on Ertl and Schuffenhauer (2009).
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
from rdkit.Chem import RingInfo

import gzip
import math
import os.path as op
import pickle
from typing import TYPE_CHECKING
from typing import Dict
from typing import Tuple
from rdkit.Chem import rdMolDescriptors

_fscores: Dict[int, float] | None

def readFragmentScores(name: str = ...) -> None:
    """
    Read precomputed fragment contribution scores from gzipped pickle file.

    Parameters
    ----------
    name : str, optional
        Base filename or path to the fragment score file (without extension),
        by default 'fpscores' located in the same directory.
    """

    ...


def numBridgeheadsAndSpiro(mol: Chem.Mol, ri: RingInfo | None = ...) -> Tuple[int, int]:
    """
    Calculate the number of bridgehead and spiro atoms in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule instance.
    ri : RingInfo | None, optional
        Precomputed ring info (unused, kept for API compatibility), by default None.

    Returns
    -------
    Tuple[int, int]
        (nBridgeheads, nSpiro) atom counts.
    """

    ...


def calculateScore(m: Chem.Mol) -> float:
    """
    Calculate the Synthetic Accessibility (SA) score for an RDKit molecule.

    The score ranges from 1.0 (very easy to synthesize) to 10.0 (very difficult to synthesize),
    combining historical fragment frequency contributions with molecular complexity penalties
    (rings, stereocenters, macrocycles, bridgeheads, spiro systems).

    Parameters
    ----------
    m : Chem.Mol
        RDKit molecule instance to score.

    Returns
    -------
    float
        Synthetic accessibility score normalized between 1.0 and 10.0.
    """

    ...

