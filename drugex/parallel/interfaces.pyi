from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.data.interfaces import DataSet
from typing import Sized

import multiprocessing
from abc import ABC
from abc import abstractmethod


class ParallelException(Exception):
    """
    Execution-level exception encapsulating failures encountered during multi-process high-throughput molecular processing. Raised when parallel worker subprocesses terminate unexpectedly, inter-process communication (IPC) queues fail, or worker execution exceeds timeout limits. Preserves worker tracebacks across multiprocessing boundaries for robust pipeline diagnostics.

    Parameters
    ----------
    *args : Any
        Multiprocessing worker pool error diagnostic message or exception cause.

    Notes
    -----
    Provides diagnostic traceback across multiprocessing pool boundaries for robust training workflows.
    """

    ...


class ResultCollector(ABC):
    """
    Abstract result aggregator interface for parallel multiprocessing workers. Implements the callable protocol (__call__) to ingest batch outputs emitted by parallel worker processes.

    Notes
    -----
    Subclasses include ListCollector (in-memory lists) and DataSet (disk-backed TSV files).
    """

    @abstractmethod
    def __call__(self, result: Any) -> None:
        """Abstract method ingesting intermediate batch result emitted from a worker process."""

        ...


class ListCollector(ResultCollector, ABC):
    """
    Abstract base class for result collectors maintaining in-memory list buffers of parallel worker results.

    Notes
    -----
    Direct base class for ListExtend (merging iterables) and ListAppend (appending single records).
    """

    @abstractmethod
    def getList(self) -> List[Any]:
        """Abstract method returning the list of aggregated results."""

        ...


class ParallelProcessor(ABC):
    """Simple interface to define parameters for parallel processing of data."""

    def __init__(self, n_proc: Optional[int] = ..., chunk_size: Optional[int] = ..., chunks: Optional[int] = ...) -> None:
        """
        Args:
            n_proc: Number of processes to initialize. Defaults to all available CPUs.
            chunk_size: Maximum size of a chunk to process by a single CPU (can help bring down memory usage, but more processing overhead). If `None`, it is set to "len(data) / n_proc" by `getChunkSize`.
            chunks: Number of chunks to divide the input data into. Defaults to 'n_proc'. If both "chunks" and "chunk_size" are specified, "chunk_size" takes precedence (see `getChunkSize`).
        """

        ...

    def getChunkSize(self, data: Sized) -> int:
        """
        Determine the chunk size from data.

        Args:
            data: input data (needs to have `len`)

        Returns:
            `int` representing the size of one chunk sent to the parallel process
        """

        ...

    @abstractmethod
    def apply(self: Any, data: Any, collector: Any) -> None:  # type: ignore[override]
        """
        Apply the processor to the given data.

        Args:
            data: input data (format depends on the implementation)
            collector: a `ResultCollector` that is used to collect data produced from each process.

        Returns:
            `None`.
        """

        ...

