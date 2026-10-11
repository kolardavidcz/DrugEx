"""
Abstract interfaces for vocabularies and molecule corpus suppliers.
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
T_Voc = TypeVar('T_Voc', bound=Vocabulary)

from abc import ABC
from abc import abstractmethod
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class Vocabulary(ABC):
    """
    Abstract base class representing a chemical token vocabulary.

    A vocabulary maintains a mapping between discrete tokens ('words') and integer indices,
    enabling conversion between textual/graph molecule representations and numeric tensors
    for generative model training and sampling.

    Parameters
    ----------
    words : Sequence[str]
        Initial collection of tokens forming the vocabulary.
    """

    def __init__(self, words: Sequence[str]) -> None:
        """
        Initialize the vocabulary with a sequence of words.

        Parameters
        ----------
        words : Sequence[str]
            Tokens included in this vocabulary.
        """

        ...

    def __add__(self: T_Voc, other: Vocabulary) -> T_Voc:
        """
        Combine tokens from two vocabularies into a new vocabulary instance.

        Parameters
        ----------
        other : Vocabulary
            Another vocabulary to merge.

        Returns
        -------
        Vocabulary
            New vocabulary containing the concatenated token list.
        """

        ...

    @abstractmethod
    def encode(self, tokens: Any, frags: Any = ...) -> Any:
        """
        Encode token sequence or molecule representation into a numeric tensor/array.

        Parameters
        ----------
        tokens : Any
            Tokens or molecule representations to encode.
        frags : Any, optional
            Fragment constraints or subgraphs, by default None.

        Returns
        -------
        Any
            Numeric array or tensor of token indices.
        """

        ...

    @abstractmethod
    def decode(self, representation: Any, *args, **kwargs) -> Any:
        """
        Decode integer indices or matrices back into human-readable SMILES or molecule representations.

        Parameters
        ----------
        representation : Any
            Tensor or array containing token indices or adjacency matrices.
        *args : Any
            Additional positional arguments (e.g. `is_tk`, `is_smiles`).
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Decoded SMILES string or tuple of (fragments, smiles).
        """

        ...

    @staticmethod
    @abstractmethod
    def fromFile(path: Union[str, os.PathLike[str], Path], *args, **kwargs) -> Vocabulary:
        """
        Load vocabulary words from an external file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to vocabulary file.
        *args : Any
            Additional configuration parameters.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Vocabulary
            Initialized vocabulary instance.
        """

        ...

    @abstractmethod
    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """
        Persist vocabulary tokens to a text or TSV file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Destination file path.
        """

        ...


class SequenceVocabulary(Vocabulary, ABC):
    """
    Abstract base class for sequence-based (SMILES) vocabularies.

    Supports control tokens (`'GO'`, `'EOS'`), special delimiter characters (`'.'`),
    length filtering, and dynamic vocabulary vocabulary expansion or vocabulary pruning.

    Parameters
    ----------
    encode_frags : bool
        If True, includes fragment separation control tokens (e.g. `'_'`).
    words : Sequence[str]
        Base collection of SMILES tokens.
    max_len : int, optional
        Maximum allowable sequence length in tokens, by default 100.
    min_len : int, optional
        Minimum allowable sequence length in tokens, by default 10.
    """

    def __init__(self, encode_frags: bool, words: Sequence[str], max_len: int = ..., min_len: int = ...) -> None:
        """
        Initialize the SequenceVocabulary.

        Parameters
        ----------
        encode_frags : bool
            Whether to support fragment delimiter tokens.
        words : Sequence[str]
            Tokens defining this vocabulary.
        max_len : int, optional
            Max allowed tokens per sequence, by default 100.
        min_len : int, optional
            Min allowed tokens per sequence, by default 10.
        """

        ...

    @abstractmethod
    def splitSequence(self, seq: str) -> List[str]:
        """
        Split a SMILES string into discrete character/bracket tokens.

        Parameters
        ----------
        seq : str
            SMILES sequence string.

        Returns
        -------
        List[str]
            List of individual tokens terminated by `'EOS'`.
        """

        ...

    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """
        Save vocabulary tokens (excluding control tokens) separated by newlines.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Destination file path.
        """

        ...

    def addWordsFromSeq(self, seq: str, ignoreConstraints: bool = ...) -> Optional[List[str]]:
        """
        Tokenize a sequence and dynamically register any newly encountered tokens into the vocabulary.

        Parameters
        ----------
        seq : str
            SMILES sequence.
        ignoreConstraints : bool, optional
            Whether to bypass min/max length filtering, by default False.

        Returns
        -------
        Optional[List[str]]
            Extracted tokens if length criteria are satisfied, otherwise None.
        """

        ...

    def removeIfNew(self, seq: str, ignoreConstraints: bool = ...) -> Optional[List[str]]:
        """
        Tokenize a sequence, rejecting and returning None if it contains out-of-vocabulary tokens.

        Parameters
        ----------
        seq : str
            SMILES sequence.
        ignoreConstraints : bool, optional
            Whether to bypass min/max length filtering, by default False.

        Returns
        -------
        Optional[List[str]]
            Extracted tokens if all tokens exist in vocabulary and meet length requirements, else None.
        """

        ...

    def updateIndex(self) -> None:
        """Rebuild token-to-index (`tk2ix`) and index-to-token (`ix2tk`) lookup dictionaries."""

        ...


class Corpus(MolSupplier, ABC):
    """
    Abstract data supplier stream generating encoded training samples from chemical representations.

    Parameters
    ----------
    molecules : Iterable[Any] | MolSupplier
        Stream or collection of molecule inputs.
    """

    def __init__(self, molecules: Union[Iterable[Any], MolSupplier]) -> None:
        """
        Initialize the corpus data supplier.

        Parameters
        ----------
        molecules : Union[Iterable[Any], MolSupplier]
            Stream or iterable of input molecule representations.
        """

        ...

    def next(self) -> Any:
        """
        Retrieve next raw item from input stream.

        Returns
        -------
        Any
            Next input molecule representation.
        """

        ...

    def convert(self, representation: Any) -> Any:
        """
        Convert a single raw representation into encoded model sample data.

        Parameters
        ----------
        representation : Any
            Raw molecule representation.

        Returns
        -------
        Any
            Processed and encoded representation, or next item on error.
        """

        ...

    @abstractmethod
    def processMolecule(self, molecule: Any) -> Any:
        """
        Process and encode a single molecule for generative model training.

        Parameters
        ----------
        molecule : Any
            Molecule representation (e.g. SMILES string or graph object).

        Returns
        -------
        Any
            Encoded numeric tensor or list of token indices.
        """

        ...

    @abstractmethod
    def getVoc(self) -> Vocabulary:
        """
        Return the Vocabulary instance used by this corpus.

        Returns
        -------
        Vocabulary
            Active vocabulary instance.
        """

        ...

