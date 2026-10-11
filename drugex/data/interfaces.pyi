"""
Abstract base classes and core interfaces for DrugEx data management.
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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from drugex.data.corpus.interfaces import Vocabulary

import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from drugex.logs import logger
from drugex.parallel.interfaces import ResultCollector


class DataSplitter(ABC):
    """Abstract strategy for partitioning datasets into train, test, and validation subsets."""

    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """
        Partition the input data into subsets.

        Parameters
        ----------
        data : Any
            Input dataset to partition.

        Returns
        -------
        splits : Any
            Collection of partitioned data subsets.
        """

        ...


class DataToLoader(ABC):
    """Abstract converter from raw dataset arrays into PyTorch DataLoaders."""

    @abstractmethod
    def __call__(self, data: Any, batch_size: int, vocabulary: Any) -> DataLoader:
        """
        Convert array data into a PyTorch DataLoader.

        Parameters
        ----------
        data : Any
            Raw data matrix.
        batch_size : int
            Mini-batch size.
        vocabulary : Any
            Vocabulary used for token mapping and tensor shape constraints.

        Returns
        -------
        loader : DataLoader
            Configured PyTorch DataLoader.
        """

        ...


class DataSet(ResultCollector, ABC):
    """Abstract base class managing disk serialization, vocabulary integration, and DataLoader generation for DrugEx models."""

    def __init__(self, path: Union[str, os.PathLike, Path], rewrite: bool = ..., save_voc: bool = ..., voc_file: Optional[Union[str, os.PathLike, Path]] = ...) -> None:
        """
        Initialize the dataset instance.

        Parameters
        ----------
        path : str or Path
            Filesystem path for storing the serialized dataset.
        rewrite : bool, optional
            Whether to delete existing dataset files at `path` upon initialization (default: False).
        save_voc : bool, optional
            Whether to serialize the associated vocabulary alongside the dataset (default: True).
        voc_file : str or Path, optional
            Explicit file path for saving the vocabulary. If None, defaults to `f"{path}.vocab"`.
        """

        ...

    def reset(self) -> None:
        """Remove existing dataset and vocabulary files from disk."""

        ...

    def getVocPath(self) -> Path:
        """
        Determine the filesystem path for the vocabulary file.

        Returns
        -------
        path : Path
            Resolved vocabulary file path.
        """

        ...

    def sendDataToFile(self, data: Sequence[Any], columns: Optional[Sequence[str]] = ...) -> None:
        """
        Append a batch of records to the on-disk TSV dataset file.

        Parameters
        ----------
        data : sequence of Any
            Batch of records to serialize.
        columns : sequence of str, optional
            TSV header column names. If None, generated as `'Col1'`, `'Col2'`, etc.
        """

        ...

    def getData(self, chunk_size: Optional[int] = ...) -> np.ndarray:
        """
        Load dataset records from disk into a NumPy array.

        Parameters
        ----------
        chunk_size : int, optional
            Size of chunks to read iteratively if dataset is large (default: None).

        Returns
        -------
        data : np.ndarray
            Loaded dataset matrix.
        """

        ...

    def updateVoc(self, voc: Vocabulary) -> None:
        """
        Combine an incoming vocabulary into the active dataset vocabulary.

        Parameters
        ----------
        voc : Vocabulary
            Vocabulary instance to merge.
        """

        ...

    def getVoc(self) -> Optional[Vocabulary]:
        """
        Return the active vocabulary associated with this dataset.

        Returns
        -------
        voc : Vocabulary or None
            Active token vocabulary.
        """

        ...

    def setVoc(self, voc: Vocabulary) -> None:
        """
        Set the active vocabulary for this dataset.

        Parameters
        ----------
        voc : Vocabulary
            Vocabulary to assign.
        """

        ...

    def fromFile(self, path: Union[str, os.PathLike[str], Path], vocs: Sequence[Union[str, os.PathLike[str], Path]] = ..., voc_class: Optional[Type[Vocabulary]] = ...) -> None:
        """
        Load dataset from an existing file and initialize its vocabulary.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to existing TSV data file.
        vocs : sequence of (str or Path), optional
            Paths to vocabulary files to load.
        voc_class : type of Vocabulary, optional
            Vocabulary class to instantiate.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist on disk.
        """

        ...

    def asDataLoader(self, batch_size: int, splitter: Optional[DataSplitter] = ..., split_converter: Optional[Union[DataToLoader, Callable[[np.ndarray, int, Any], DataLoader]]] = ..., n_samples: int = ..., n_samples_ratio: Optional[float] = ...) -> Union[DataLoader, List[DataLoader]]:
        """
        Convert on-disk data into one or more PyTorch DataLoaders.

        Parameters
        ----------
        batch_size : int
            Desired batch size for the DataLoader.
        splitter : DataSplitter, optional
            Optional partition strategy to split dataset into train/test subsets.
        split_converter : DataToLoader or callable, optional
            Custom converter mapping partition arrays into DataLoaders (defaults to `self.dataToLoader`).
        n_samples : int, optional
            Desired fixed sample size. If larger than dataset length, samples are replicated (default: -1).
        n_samples_ratio : float, optional
            Scaling factor applied to `n_samples` before partitioning.

        Returns
        -------
        loaders : DataLoader or list of DataLoader
            Single DataLoader if unsplit, or list of DataLoaders matching splitter partitions.
        """

        ...

    @staticmethod
    @abstractmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: Any) -> DataLoader:
        """
        Convert a raw data array into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Input dataset array.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : Any
            Active vocabulary instance.

        Returns
        -------
        loader : DataLoader
            Constructed DataLoader.
        """

        ...

    def createLoaders(self, data: np.ndarray, batch_size: int, splitter: Optional[DataSplitter] = ..., converter: Optional[Union[DataToLoader, Callable[[np.ndarray, int, Any], DataLoader]]] = ...) -> List[Any]:
        """
        Split data and construct DataLoader instances for each partition.

        Parameters
        ----------
        data : np.ndarray
            Dataset array.
        batch_size : int
            Mini-batch size.
        splitter : DataSplitter, optional
            Partitioning strategy.
        converter : callable, optional
            Conversion function mapping arrays to DataLoaders.

        Returns
        -------
        loaders : list of Any
            List of DataLoaders or split arrays.
        """

        ...

    def readVocs(self, paths: Sequence[Union[str, os.PathLike[str], Path]], voc_class: Type[Vocabulary], *args, **kwargs) -> None:  # type: ignore[override]
        """
        Read vocabulary files from disk and combine them into the active vocabulary.

        Parameters
        ----------
        paths : sequence of (str or Path)
            File paths to vocabulary definitions.
        voc_class : type of Vocabulary
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """

        ...


class FragmentPairEncoder(ABC):
    """Abstract encoder for fragment-molecule pairs used in conditioned DrugEx models."""

    @abstractmethod
    def encodeMol(self, mol: Any) -> Tuple[Any, Any]:
        """
        Encode parent molecule sequence into token strings and index codes.

        Parameters
        ----------
        mol : Any
            Parent molecule representation (e.g. SMILES string).

        Returns
        -------
        result : tuple of (list of str or None, list of int or None)
            Extracted token sequence and corresponding numerical index codes.
        """

        ...

    @abstractmethod
    def encodeFrag(self, mol: Any, mol_tokens: Any, frag: Any) -> Optional[List[int]]:
        """
        Encode fragment within parent molecule context.

        Parameters
        ----------
        mol : Any
            Parent molecule representation.
        mol_tokens : Any
            Extracted parent molecule token sequence.
        frag : Any
            Fragment representation.

        Returns
        -------
        codes : list of int or None
            Encoded numerical representation of the fragment-molecule pair.
        """

        ...

    @abstractmethod
    def getVoc(self) -> Optional[Vocabulary]:
        """
        Return the active vocabulary used for encoding.

        Returns
        -------
        vocabulary : Vocabulary or None
            Active vocabulary instance.
        """

        ...

