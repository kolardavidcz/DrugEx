from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect

from rdkit.Chem import AllChem
from rdkit.Chem import Mol
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.AtomPairs.Sheridan import GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D


class _FingerprintCalculator:
    """
    Calculate the fingerprint while avoiding a series of if-else.
    See recipe 8.21 of the book "Python Cookbook".
    To support a new type of fingerprint, just add a function "get_fpname(self, mol)".
    """

    def get_fingerprint(self, mol: Mol, fp_type: str) -> Any:
        ...

    def get_AP(self, mol: Mol) -> Any:
        ...

    def get_PHCO(self, mol: Mol) -> Any:
        ...

    def get_BPF(self, mol: Mol) -> Any:
        ...

    def get_BTF(self, mol: Mol) -> Any:
        ...

    def get_PATH(self, mol: Mol) -> Any:
        ...

    def get_ECFP4(self, mol: Mol) -> Any:
        ...

    def get_ECFP6(self, mol: Mol) -> Any:
        ...

    def get_FCFP4(self, mol: Mol) -> Any:
        ...

    def get_FCFP6(self, mol: Mol) -> Any:
        ...


def get_fingerprint(mol: Mol, fp_type: str) -> ExplicitBitVect | Any:
    """Calculate the requested fingerprint representation for an RDKit molecule."""

    ...

