"""
Encoders, suppliers, and splitters for fragment-conditioned molecular generation.
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
from drugex.parallel.interfaces import ResultCollector
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.generators.graph_transformer import GraphTransformer

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.interfaces import DataSplitter
from drugex.data.interfaces import FragmentPairEncoder
from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.interfaces import MolSupplier
from drugex.parallel.collectors import ListExtend
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import ParallelProcessor


class SequenceFragmentEncoder(FragmentPairEncoder):
    """Fragment-molecule pair encoder for sequence-based models (`SequenceTransformer`)."""

    def __init__(self, vocabulary: VocSmiles = ..., update_voc: bool = ..., throw: bool = ...) -> None:
        """
        Initialize the sequence fragment encoder.

        Parameters
        ----------
        vocabulary : VocSmiles, optional
            Vocabulary supporting fragment delimiter tokens (default: `VocSmiles(True)`).
        update_voc : bool, optional
            Whether to dynamically append newly encountered tokens to vocabulary (default: True).
        throw : bool, optional
            Whether to strip newly discovered tokens failing constraints (default: False).
        """

        ...

    def encodeMol(self, sequence: str) -> Tuple[Any, Any]:
        """
        Tokenize and integer-encode parent molecule SMILES sequence.

        Parameters
        ----------
        sequence : str
            Parent molecule SMILES string.

        Returns
        -------
        result : tuple of (list of str or None, list of int or None)
            Extracted token sequence and corresponding numerical index codes.
        """

        ...

    def encodeFrag(self, mol: str, mol_tokens: Sequence[str], frag: str) -> Optional[List[int]]:
        """
        Encode fragment scaffold within parent molecule context.

        Parameters
        ----------
        mol : str
            Parent molecule SMILES string.
        mol_tokens : sequence of str
            Tokenized parent molecule representation.
        frag : str
            Scaffold fragment SMILES string.

        Returns
        -------
        code : list of int or None
            Integer-encoded fragment sequence, or None if encoding fails.
        """

        ...

    def getVoc(self) -> VocSmiles:
        """
        Return active SMILES vocabulary.

        Returns
        -------
        voc : VocSmiles
            Active vocabulary instance.
        """

        ...


class GraphFragmentEncoder(FragmentPairEncoder):
    """Encodes fragment-molecule pairs for graph-based models (`GraphTransformer`)."""

    def __init__(self, vocabulary: VocGraph = ...) -> None:
        """
        Initialize graph fragment encoder.

        Parameters
        ----------
        vocabulary : VocGraph, optional
            Graph vocabulary defining atom, bond, and loci tokens (default: `VocGraph()`).
        """

        ...

    def encodeMol(self, smiles: str) -> Tuple[Any, Any]:
        """
        Pass parent molecule SMILES through as raw token representations.

        Parameters
        ----------
        smiles : str
            Input parent molecule SMILES string.

        Returns
        -------
        result : tuple of (str, str)
            Pair `(smiles, smiles)` matching interface contract.
        """

        ...

    def encodeFrag(self, mol: str, mol_tokens: Any, frag: str) -> Optional[List[int]]:
        """
        Encode paired fragment scaffold and parent molecule into graph actions.

        Parameters
        ----------
        mol : str
            Parent molecule SMILES.
        mol_tokens : Any
            Parent molecule token representation.
        frag : str
            Fragment scaffold SMILES.

        Returns
        -------
        code : list of int or None
            Encoded graph action tuple indices, or None if encoding fails.
        """

        ...

    def getVoc(self) -> VocGraph:
        """
        Return active graph vocabulary.

        Returns
        -------
        voc : VocGraph
            Active graph vocabulary instance.
        """

        ...


class FragmentPairsEncodedSupplier(MolSupplier):
    """Iteratively transforms raw fragment-molecule pairs into model-ready integer encodings."""

    def __init__(self, pairs: Iterable[Tuple[str, str]], encoder: FragmentPairEncoder) -> None:
        """
        Initialize from fragment-molecule pairs.

        Parameters
        ----------
        pairs : iterable of (str, str)
            Sequence of `(fragment_smiles, molecule_smiles)` tuples.
        encoder : FragmentPairEncoder
            Encoder implementing `encodeMol` and `encodeFrag`.
        """

        ...

    def next(self) -> Tuple[List[int], List[int]]:
        """
        Fetch and encode next fragment-molecule pair.

        Returns
        -------
        pair : tuple of (list of int, list of int)
            Encoded `(encoded_fragment, encoded_molecule)`.

        Raises
        ------
        MoleculeEncodingException
            If parent molecule fails to encode.
        FragmentEncodingException
            If fragment scaffold fails to encode.
        """

        ...


class FragmentPairsSupplier(MolSupplier):
    """Generates fragment-molecule candidate pairs from input molecules via a fragmenter algorithm."""

    def __init__(self, molecules: Iterable[Any], fragmenter: Any, max_bonds: Optional[int] = ...) -> None:
        """
        Initialize supplier with input molecules and fragmenter.

        Parameters
        ----------
        molecules : iterable
            Input molecules as list or iterator.
        fragmenter : callable
            Fragmentation strategy returning `(fragment, molecule)` pairs.
        max_bonds : int, optional
            Maximum allowed number of cut bonds (default: None).
        """

        ...

    def next(self) -> Optional[Tuple[str, str]]:
        """
        Yield next available fragment-molecule pair.

        Returns
        -------
        pair : tuple of (str, str) or None
            A `(fragment, molecule)` pair, or None upon batch completion.
        """

        ...


class FragmentCorpusEncoder(ParallelProcessor):
    """Fragments and encodes chemical compounds in parallel across CPU worker processes."""

    def __init__(self, fragmenter: Any, encoder: FragmentPairEncoder, pairs_splitter: Optional[DataSplitter] = ..., n_proc: Optional[int] = ..., chunk_size: Optional[int] = ...) -> None:
        """
        Initialize the parallel fragment corpus encoder.

        Parameters
        ----------
        fragmenter : callable
            Fragmentation strategy decomposing molecules into fragments.
        encoder : FragmentPairEncoder
            Encoder translating molecules and fragments into vocabulary tokens.
        pairs_splitter : DataSplitter, optional
            Splitter strategy partitioning generated pairs (default: None).
        n_proc : int, optional
            Number of parallel CPU worker processes (default: all available CPUs).
        chunk_size : int, optional
            Maximum chunk size per worker process (default: None).
        """

        ...

    def getFragmentPairs(self, mols: Iterable[str], collector: ResultCollector) -> None:
        """
        Decompose molecules into fragment pairs in parallel.

        Parameters
        ----------
        mols : iterable of str
            Input molecule SMILES strings.
        collector : ResultCollector
            Collector accumulating generated pairs.
        """

        ...

    def splitFragmentPairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Sequence[Tuple[str, str]]]:
        """
        Partition generated fragment-molecule pairs using configured splitter.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Generated pairs.

        Returns
        -------
        splits : sequence of sequence of (str, str)
            Partitioned pair subsets.
        """

        ...

    def encodeFragments(self, pairs: Sequence[Tuple[str, str]], collector: Optional[ResultCollector]) -> None:
        """
        Encode fragment-molecule pairs into numerical tokens in parallel.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Fragment-molecule pairs to encode.
        collector : ResultCollector, optional
            Collector accumulating encoded token batches.
        """

        ...

    def apply(self, mols: Iterable[str], fragmentPairsCollector: Optional[ResultCollector] = ..., encodingCollectors: Optional[Sequence[ResultCollector]] = ...) -> None:
        """
        Decompose molecules and encode fragment-molecule pairs across parallel CPU workers.

        Parameters
        ----------
        mols : iterable of str
            Molecule SMILES strings.
        fragmentPairsCollector : ResultCollector, optional
            Optional collector receiving intermediate fragment-molecule pairs.
        encodingCollectors : sequence of ResultCollector, optional
            List of collectors corresponding to each partition split.

        Raises
        ------
        RuntimeError
            If `encodingCollectors` length does not match partition count.
        """

        ...


class FragmentPairsSplitter(DataSplitter):
    """Partitions fragment-molecule pairs so that scaffold fragments do not overlap between splits."""

    def __init__(self, ratio: float = ..., max_test_samples: float = ..., train_collector: Optional[ResultCollector] = ..., test_collector: Optional[ResultCollector] = ..., unique_collector: Optional[ResultCollector] = ..., make_unique: bool = ..., seed: Optional[int] = ...) -> None:
        """
        Initialize fragment pair splitter.

        Parameters
        ----------
        ratio : float, optional
            Fraction of unique fragments assigned to test partition (default: 0.2).
        max_test_samples : float, optional
            Maximum count threshold for test samples (default: 10000).
        train_collector : ResultCollector, optional
            Collector accumulating training pairs.
        test_collector : ResultCollector, optional
            Collector accumulating testing pairs.
        unique_collector : ResultCollector, optional
            Collector accumulating single-representative unique fragment pairs.
        make_unique : bool, optional
            Whether to return a unique fragment subset (default: False).
        seed : int, optional
            Random state seed for reproducible sampling (default: None).
        """

        ...

    def __call__(self, pairs: Sequence[Tuple[str, str]]) -> Union[Tuple[List[Any], List[Any], List[Any]], Tuple[List[Any], List[Any]]]:
        """
        Partition fragment-molecule pairs based on fragment identity.

        Parameters
        ----------
        pairs : sequence of (str, str)
            Input `(fragment, molecule)` pairs.

        Returns
        -------
        splits : tuple
            `(test_pairs, train_pairs, unique_pairs)` if `make_unique=True`, else `(test_pairs, train_pairs)`.
        """

        ...

