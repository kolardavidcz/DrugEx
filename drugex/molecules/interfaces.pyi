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

from abc import ABC
from abc import abstractmethod
from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException


class Molecule(ABC):
    """Abstract base class representing a standardized chemical entity with annotations and unique identifiers."""

    @abstractmethod
    def annotate(self, key: str, value: Any) -> None:
        """Attaches a metadata property key-value pair to the molecule."""

        ...

    @abstractmethod
    def getAnnotation(self, key: str) -> Any:
        """Retrieves stored annotation value for specified key."""

        ...

    @abstractmethod
    def getMetadata(self) -> Dict[str, Any]:
        """Returns full dictionary of attached property annotations."""

        ...

    def __eq__(self, other: Any) -> bool:
        """Tests equivalence against another Molecule based on getUniqueID()."""

        ...

    def __hash__(self) -> int:
        """Computes integer hash code based on getUniqueID() string."""

        ...

    def __str__(self) -> str:
        """Returns human-readable string representation containing class and unique ID."""

        ...

    @abstractmethod
    def getUniqueID(self) -> str:
        """Returns unique canonical identifier string (e.g. canonical SMILES)."""

        ...

    @abstractmethod
    def asRDKit(self) -> Chem.Mol:
        """Returns the underlying RDKit Chem.Mol representation."""

        ...


class ItemException(Exception):
    """
    Supplier-level exception raised when accessing or generating an item in a MolSupplier fails due to missing indices, corrupt storage records, or supplier iteration faults. Caught inside MolSupplier.__next__() to log diagnostic warnings and proceed to the next item without aborting iteration.

    Parameters
    ----------
    *args : Any
        Supplier index, file path, or record details where item access failed.

    Notes
    -----
    Allows MolSupplier subclasses to log indexed item read errors while continuing stream ingestion.
    """

    ...


class MolSupplier(ABC):
    """
    Generic class that defines the interface for data suppliers from molecules. Implementations of this class
    are used to wrap functionality that can be reused and evaluated in parallel with the `ParallelSupplierEvaluator`.

    Suppliers are simply just Python generators that produce the desired output one item at a time. It is also possible to implement the `convert` method to customize the produced output.
    """

    def __iter__(self) -> Iterator[Any]:
        """Returns the supplier generator itself for iterator protocol compliance."""

        ...

    def __next__(self) -> Any:
        """Fetches and converts the next valid non-None item, skipping items that raise ItemException or ConversionException."""

        ...

    @abstractmethod
    def next(self) -> Any:
        """
        Implement this method so that it provides iteration over molecules item by item. It should fetch next item from a generator, line from a file or next item from a remote API. If there are no more items, raise `StopIteration`.

        Raises:
            StopIteration:  no more items to  return

        Returns:
            molecule: one instance of a molecule
            annotations (optional): molecule associated metadata as a `dict`
        """

        ...

    def convert(self, representation: Any) -> Any:
        """
        Can be used to convert a molecule from the supplied representation to a different one. This method is called automatically on the output of `next`. By default, it returns the produced representation as is.

        Parameters
        ----------
        representation - the output produced by `next`

        Returns
        -------

        molecule - molecule converted from "representation" to the desired output
        """

        ...

    def toList(self) -> List[Any]:
        """Materializes the entire streamed supplier output into an in-memory Python list."""

        ...


class BaseMolSupplier(MolSupplier, ABC):
    """Extended `MolSupplier` that produces instances of `DrExMol` that implements identification of duplicates and other useful features."""

    def __init__(self, converter: MolConverter, hide_duplicates: bool = ...) -> None:
        """
        Args:
            converter: a `MolConverter` that produces an instance of `Molecule` from the given input.
            hide_duplicates: If `True`, the returned instances will be tested on uniqueness and only the first encountered item will be processed.
        """

        ...

    def __next__(self) -> Any:
        """Advances stream, converts item, and skips duplicate molecules if hide_duplicates=True."""

        ...

    def convert(self, representation: Any) -> Any:
        """Apply converter to raw representation, raising ConversionException if output is empty."""

        ...

    @abstractmethod
    def next(self) -> Any: ...

