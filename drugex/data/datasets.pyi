"""
Dataset implementations for sequence- and graph-based DrugEx models.
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
from drugex.data.corpus.interfaces import Vocabulary
from drugex.data.interfaces import DataSplitter
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.generators.graph_transformer import GraphTransformer

import os
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.interfaces import DataSet
from drugex.data.interfaces import DataToLoader


class SmilesDataSet(DataSet):
    """
    Encoded SMILES dataset for single-network sequence-based DrugEx models (`SequenceRNN`).

    Holds tokenized and integer-encoded SMILES representations for recurrent neural network
    training, mapping chemical tokens to vocabulary indices.
    """

    columns: Tuple[str, str] = ...
    def __init__(self, path: Union[str, os.PathLike, Path], voc: Optional[VocSmiles] = ..., rewrite: bool = ...) -> None:
        """
        Initialize the SMILES dataset.

        Parameters
        ----------
        path : str or Path
            Filesystem path to the dataset file.
        voc : VocSmiles, optional
            Vocabulary mapping SMILES tokens to indices. If None, defaults to `VocSmiles(False)`.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        """

        ...

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader:
        """
        Convert a 2D NumPy array of token indices into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Integer matrix of encoded SMILES sequences of shape `(n_samples, max_len)`.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : VocSmiles
            Vocabulary defining sequence dimensions and padding.

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding token index tensors.
        """

        ...

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """
        Collect and append encoded sequences generated from `SequenceCorpus`.

        Parameters
        ----------
        result : tuple
            A tuple `(encoded_data, corpus_instance)` where `encoded_data` is a list of encoded
            sequences and `corpus_instance` provides the active vocabulary.
        """

        ...

    def getColumns(self) -> List[str]:
        """
        Generate TSV column header names corresponding to sequence positions.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, `'C1'`, ..., `'C{max_len-1}'`.
        """

        ...

    def readVocs(self, paths: Sequence[Union[str, os.PathLike[str], Path]], voc_class: Type[VocSmiles], *args, **kwargs) -> None:  # type: ignore[override]
        """
        Read and combine vocabulary files into the active vocabulary.

        Parameters
        ----------
        paths : sequence of (str or Path)
            Paths to vocabulary files.
        voc_class : type of VocSmiles
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """

        ...


class SmilesFragDataSet(DataSet):
    """
    Encoded SMILES dataset for sequence-based encoder-decoder DrugEx models (`SequenceTransformer`).

    Stores paired representations of input fragment scaffolds and output molecules
    for conditioned chemical sequence generation.
    """

    columns: Tuple[str, str] = ...
    def __init__(self, path: str, voc: Optional[VocSmiles] = ..., rewrite: bool = ..., save_voc: bool = ..., voc_file: Optional[str] = ...) -> None:
        """
        Initialize the fragment-molecule SMILES dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset file.
        voc : VocSmiles, optional
            Active SMILES vocabulary. If None, defaults to `VocSmiles(True)` with fragment tokens.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        save_voc : bool, optional
            Whether to persist the vocabulary to disk alongside the dataset (default: True).
        voc_file : str, optional
            Custom path to save the vocabulary file (default: None).
        """

        ...

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """
        Collect encoded fragment-molecule pairs from `FragmentCorpusEncoder`.

        Parameters
        ----------
        result : tuple
            A tuple `(pairs_data, supplier)` where `pairs_data` contains encoded fragment-molecule
            tuples and `supplier` provides the active vocabulary.
        """

        ...

    def createLoaders(self, data: np.ndarray, batch_size: int, splitter: Optional[DataSplitter] = ..., converter: Optional[Callable[[np.ndarray, int, VocSmiles], DataLoader]] = ...) -> List[Any]:
        """
        Split data and construct PyTorch DataLoaders for each partition.

        Parameters
        ----------
        data : np.ndarray
            Dataset array to partition and convert.
        batch_size : int
            Number of samples per mini-batch.
        splitter : DataSplitter, optional
            Splitter strategy to partition data (e.g. train/test).
        converter : callable, optional
            Function converting raw split arrays into DataLoaders.

        Returns
        -------
        loaders : list of DataLoader or np.ndarray
            List of created DataLoaders for each partition.
        """

        ...

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader:
        """
        Convert concatenated fragment and molecule embeddings into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Array of shape `(n_samples, 2 * max_len)` containing concatenated fragment and molecule indices.
        batch_size : int
            Batch size for DataLoader.
        vocabulary : VocSmiles
            Vocabulary defining sequence dimensions.

        Returns
        -------
        loader : DataLoader
            DataLoader yielding `(molecule_tensor, fragment_tensor)` batches.
        """

        ...

    def getColumns(self) -> List[str]:
        """
        Generate TSV column header names for concatenated fragment and molecule sequences.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, ..., `'C{2*max_len-1}'`.
        """

        ...

    def readVocs(self, paths: Sequence[Union[str, os.PathLike[str], Path]], voc_class: Type[VocSmiles], *args, **kwargs) -> None:  # type: ignore[override]
        """
        Read and combine vocabulary files into the active vocabulary.

        Parameters
        ----------
        paths : sequence of (str or Path)
            Paths to vocabulary files.
        voc_class : type of VocSmiles
            Vocabulary class to instantiate.
        *args : Any
            Positional arguments forwarded to `voc_class.fromFile`.
        **kwargs : Any
            Keyword arguments forwarded to `voc_class.fromFile`.
        """

        ...


class GraphFragDataSet(DataSet):
    """
    Encoded molecular graph dataset for fragment-conditioned graph transformers (`GraphTransformer`).

    Stores paired fragment-molecule graph action encodings (atom types, loci, bond types)
    for step-wise autoregressive graph generation.
    """

    def __init__(self, path: str, voc: Optional[VocGraph] = ..., rewrite: bool = ..., save_voc: bool = ..., voc_file: Optional[str] = ...) -> None:
        """
        Initialize the graph fragment dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset file.
        voc : VocGraph, optional
            Graph vocabulary defining atom, bond, and loci tokens. If None, defaults to `VocGraph()`.
        rewrite : bool, optional
            Whether to overwrite existing dataset files at `path` (default: False).
        save_voc : bool, optional
            Whether to persist the vocabulary to disk alongside the dataset (default: True).
        voc_file : str, optional
            Custom path to save the vocabulary file (default: None).
        """

        ...

    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None:
        """
        Collect encoded graph actions from `FragmentCorpusEncoder`.

        Parameters
        ----------
        result : tuple
            A tuple `(pairs_data, supplier)` where `pairs_data` contains encoded graph action
            tuples and `supplier` provides the active graph vocabulary.
        """

        ...

    def getColumns(self) -> List[str]:
        """
        Generate TSV column header names for 5-tuple graph action sequences.

        Returns
        -------
        columns : list of str
            Column names `'C0'`, ..., `'C{5*max_len-1}'`.
        """

        ...

    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocGraph) -> DataLoader:
        """
        Convert a 2D/3D NumPy array of graph action encodings into a PyTorch DataLoader.

        Parameters
        ----------
        data : np.ndarray
            Array of graph action encodings.
        batch_size : int
            Number of samples per mini-batch.
        vocabulary : VocGraph
            Vocabulary defining graph length capacity.

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding graph action tensors.
        """

        ...

