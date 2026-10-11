"""
SMILES validity and fragment accuracy verification scorers.
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

from typing import Sequence


class SmilesChecker:
    """
    Utility class to validate SMILES strings and verify fragment substructure constraints.

    Provides chemical validity checking through RDKit parsing and validates that
    generated molecules preserve required fragment substructures (e.g. for scaffold-based
    or fragment-based molecule generation).
    """

    @staticmethod
    def checkSmiles(smiles: Sequence[str | None], frags: Sequence[str | None] | None = ..., no_multifrag_smiles: bool = ...) -> pd.DataFrame:
        """
        Check chemical validity of SMILES strings and verify fragment preservation.

        Parameters
        ----------
        smiles : Sequence[str | None]
            List or sequence of SMILES strings to evaluate.
        frags : Sequence[str | None] | None, optional
            Corresponding list of fragment SMILES (dot-separated if multiple fragments)
            that each generated molecule must contain, by default None.
        no_multifrag_smiles : bool, optional
            If True, SMILES strings containing disconnected components ('.') are considered
            invalid and marked with 0, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame with index matching input SMILES and boolean integer columns:
            - 'Valid': 1 if SMILES is a valid, single-fragment molecule (if no_multifrag_smiles=True),
              0 otherwise.
            - 'Accurate': (if frags is provided) 1 if all fragment substructures are present
              in the molecule, 0 otherwise.
        """

        ...

