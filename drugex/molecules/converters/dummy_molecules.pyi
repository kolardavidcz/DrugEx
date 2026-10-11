from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from drugex.logs import logger
from drugex.molecules.converters.interfaces import MolConverter


class dummyMolsFromFragments(MolConverter):
    """A converter to create dummy molecules from fragments."""

    @staticmethod
    def addCCToFragments(frag: str) -> Optional[str]:
        """
        Add CC to a single fragment to build a dummy molecule.

        Args:
            frag: fragment to be converted to a dummy molecule

        Returns:
        """

        ...

    @staticmethod
    def bridgeFragments(frags: str) -> Optional[str]:
        """
        Bridge multiple fragments together into one dummy molecule.

        Args:
            frags: fragments to be bridged together

        Returns:
            a `str` of a SMILES representation of the bridged fragments
        """

        ...

    def __call__(self, frag: str) -> Optional[List[Tuple[str, str]]]:
        """
        Create dummy molecule by adding CC to single fragments or bridge multiple fragments together.

        Args:
            frag (str): fragment(s) to be converted to a dummy molecule
        Return:
            a `list` of `tuple`s of format  (fragment, smiles), fragment is the same as the input in the 'frag' argument
        """

        ...

