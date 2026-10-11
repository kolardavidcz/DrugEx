from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from drugex.molecules.interfaces import Molecule


class InitializationException(Exception):
    """Raised when DrExMol instantiation is attempted with an empty molecule."""

    ...


class DrExMol(Molecule):
    """Concrete Molecule implementation wrapping an RDKit Mol with canonical SMILES and metadata."""

    def __init__(self, rd_mol: Chem.Mol, identifier: Optional[str] = ...) -> None:
        ...

    @property
    def smiles(self) -> str:
        ...

    def annotate(self, key: str, value: Any) -> None:
        """Attaches computational annotation key-value pair to molecule."""

        ...

    def getAnnotation(self, key: str) -> Any:
        """Retrieves annotation property value for specified key."""

        ...

    def getMetadata(self) -> Dict[str, Any]:
        """Returns dictionary of attached property annotations."""

        ...

    def getUniqueID(self) -> str:
        """Returns unique canonical identifier (canonical SMILES)."""

        ...

    def asRDKit(self) -> Chem.Mol:
        """Returns underlying RDKit Chem.Mol instance."""

        ...

