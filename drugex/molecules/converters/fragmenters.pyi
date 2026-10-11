from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import re
from itertools import combinations
from typing import List
from typing import Tuple
from typing import Union
from rdkit.Chem import Recap
from rdkit.Chem import BRICS
from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.converters.standardizers import CleanSMILES


class Fragmenter(CleanSMILES):
    """Reference implementation of the original fragmenter used in DrugEx v3."""

    def __init__(self, n_frags: int, n_combs: int, method: str = ..., deep_clean: bool = ..., max_bonds: int = ..., allow_single: bool = ...) -> None:
        """
        Args:
            n_frags: number of fragments to generate per compound
            n_combs: maximum number of combinations of the found leaf fragments
            method: fragmentation method to use. Possible values: ('recap', 'brics')
            deep_clean: deep clean the SMILES before fragmentation (see `CleanSMILES`)
            max_bonds: only accept molecules with the number of bonds below or equal to this threshold
            allow_single: return the fragment also for molecules that result in only one fragment
        """

        ...

    def getFragments(self, mol: Chem.Mol) -> Union[np.ndarray, None]:
        """
        Get fragments form an RDKit molecule

        Args:
            mol: instance of `rdkit.Chem.Mol`

        Returns:
            `numpy.array` of generated fragments
        """

        ...

    def filterFragments(self, frags: List[str]) -> List[str]:
        """
        Filter fragments to remove those that are contained in other fragments or are too small, 
        and keep only the largest ones.

        Args:
            frags: `list` of fragments

        Returns:
            `list` of filtered fragments
        """

        ...

    def __call__(self, smiles: str) -> Union[List[Tuple[str, str]], None]:  # type: ignore[override]
        """
        Generate fragment-molecule pairs for a given SMILES string.

        Args:
            smiles: SMILES of the molecule to fragment

        Returns:
            a list of `tuple`s of format  (fragment, smiles), smiles is the same as the input in "smiles"
        """

        ...


class FragmenterWithSelectedFragment(Fragmenter):
    """
    Fragmenter that only returns fragments-molecule pairs where the input fragments contain the fragment 
    specified in the constructor.

    If `exclusive=True`, only return fragments that contain the specified fragment and nothing else
    """

    def __init__(self, fragment: str, n_frags: int, n_combs: int, method: str = ..., deep_clean: bool = ..., max_bonds: int = ..., allow_single: bool = ..., exclusive: bool = ...) -> None:
        """
        Args:
            fragment: fragment to search for
            n_frags: number of fragments to generate per compound
            n_combs: maximum number of combinations of the found leaf fragments
            method: fragmentation method to use. Possible values: ('recap', 'brics')
            deep_clean: deep clean the SMILES before fragmentation (see `CleanSMILES`)
            max_bonds: only accept molecules with the number of bonds below or equal to this threshold
            allow_single: return the fragment also for molecules that result in only one fragment
            exclusive: if True, only return fragments that contain the specified fragment and nothing else
        """

        ...

    def __call__(self, smiles: str) -> Union[List[Tuple[str, str]], None]:  # type: ignore[override]
        """
        Generate fragment-molecule pairs for a given SMILES string and only return those that contain the specified fragment.

        Args:
            smiles: SMILES of the molecule to fragment

        Returns:
            a list of `tuple`s of format  (fragment, smiles), smiles is the same as the input in "smiles"
        """

        ...

