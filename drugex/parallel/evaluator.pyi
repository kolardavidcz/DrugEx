from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.parallel.interfaces import ResultCollector
from drugex.molecules.interfaces import MolSupplier

import concurrent
from tqdm.auto import tqdm
from drugex.parallel.interfaces import ParallelException
from drugex.parallel.interfaces import ParallelProcessor


class ParallelSupplierEvaluator(ParallelProcessor):
    """Class implementing parallel evaluation of `MolSupplier` instances on input data (see `ParallelSupplierEvaluator.apply`)."""

    def __init__(self, supplier_class: Type[MolSupplier], n_proc: Optional[int] = ..., chunk_size: int = ..., chunks: Optional[int] = ..., args: Optional[List[Any]] = ..., kwargs: Optional[Dict[str, Any]] = ...) -> None:
        """
        Initialize this instance with a `MolSupplier` and other parameters. Note that the supplier is passed as a class and not an instance. This helps to avoid some issues with serialization between processes and, thus, `ParallelSupplierEvaluator` serves only as a template for execution. Also note that the `ParallelSupplierEvaluator` assumes that the first argument of the `MolSupplier` constructor accepts the data to be processed.

        Results of the calculation invoked by `ParallelSupplierEvaluator.apply` are concatenated and returned as a `list` unless 'return_suppliers' is specified.

        Args:
            supplier_class: Class of the `MolSupplier` to use for evaluation.
            args:
            kwargs:
        """

        ...

    def initSupplier(self, supplier_class: Type[MolSupplier], chunk: Sequence[Any]) -> MolSupplier:
        """
        Initialize a `MolSupplier` instance on the given chung of data.
        Args:
            supplier_class: `MolSupplier` to initialize.
            chunk: Data chunk.

        Returns:
            initialized `MolSupplier`
        """

        ...

    def run(self, chunk: Sequence[Any], error: Callable[[Exception], None]) -> Tuple[List[Any], MolSupplier]:
        """
        Initialize and start evaluation of the `MolSupplier` instance on the given chunk of data.

        Args:
            chunk: Current chunk of data.
        Returns:
            result of the `MolSupplier.toList()` method for the given chunk.
        """

        ...

    def error(self, data: Any) -> None:
        """
        Catch and log an error occurring in the parallel process.

        Raises:
            `ParallelException`

        Args:
            data: error data
        """

        ...

    def apply(self, data: Sequence[Any], collector: ResultCollector, error: Optional[Callable[[Exception], None]] = ..., desc_string: Optional[str] = ...) -> List[Any]:  # type: ignore[override]
        """
        Apply the `ParallelSupplierEvaluator.run()` across a `Pool` of workers.

        Args:
            data: input data to divide into chunks and process in parallel
            collector: a `ResultCollector` that receives results of parallel processes
            error: a callable to handle errors during evaluation of each parallel supplier
            desc_string: progress bar description string
        Returns:
            `None`
        """

        ...

