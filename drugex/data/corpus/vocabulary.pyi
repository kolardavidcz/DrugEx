"""
Concrete vocabulary implementations for SMILES and molecular graph representations.
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
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.generators.graph_transformer import GraphTransformer

import os
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
from drugex.data.corpus.interfaces import SequenceVocabulary
from drugex.data.corpus.interfaces import Vocabulary
from drugex.logs import logger
from drugex.molecules.converters.standardizers import CleanSMILES


class VocSmiles(SequenceVocabulary):
    """
    Vocabulary for encoding and decoding SMILES strings for `SequenceRNN` and `SequenceTransformer` models.

    Splits SMILES sequences into chemical tokens (atoms, bonds, rings, brackets),
    maps tokens to discrete integer indices, and converts tensors back into canonical SMILES.
    Also handles replacement of multi-character halogens (`'Cl'` -> `'L'`, `'Br'` -> `'R'`)
    for uniform single-token processing.

    Parameters
    ----------
    encode_frags : bool, optional
        Whether to include fragment-specific tokens (`'_'`), by default False.
    words : Sequence[str], optional
        Initial word/token list, by default `defaultWords`.
    max_len : int, optional
        Maximum sequence token length, by default 100.
    min_len : int, optional
        Minimum sequence token length, by default 10.
    """

    defaultWords: Tuple[str, ...] = ...
    def __init__(self, encode_frags: bool = ..., words: Sequence[str] = ..., max_len: int = ..., min_len: int = ...) -> None:
        """
        Initialize the SMILES vocabulary.

        Parameters
        ----------
        encode_frags : bool, optional
            Whether to support fragment delimiter tokens, by default False.
        words : Sequence[str], optional
            Tokens included in vocabulary, by default defaultWords.
        max_len : int, optional
            Max allowed sequence length, by default 100.
        min_len : int, optional
            Min allowed sequence length, by default 10.
        """

        ...

    def encode(self, tokens: Sequence[Sequence[str]], frags: Any = ...) -> torch.LongTensor:
        """
        Encode a batch of tokenized sequences into a 2D integer tensor.

        Parameters
        ----------
        tokens : Sequence[Sequence[str]]
            List of token sequences (e.g. `[['C', 'C', '(=O)', 'O'], ...]`).
        frags : Any, optional
            Fragment constraints (unused).

        Returns
        -------
        torch.LongTensor
            Tensor of shape `(len(tokens), max_len)` with token indices padded with 0.
        """

        ...

    def decode(self, tensor: Union[torch.Tensor, Sequence[int]], is_tk: bool = ..., is_smiles: bool = ...) -> str:
        """
        Decode a sequence of integer token indices back into a SMILES string.

        Parameters
        ----------
        tensor : Union[torch.Tensor, Sequence[int]]
            1D array or tensor of token indices.
        is_tk : bool, optional
            Whether elements are already token strings (True) or integer indices (False),
            by default True.
        is_smiles : bool, optional
            Whether to reconstruct halogen symbols (`'L'` -> `'Cl'`, `'R'` -> `'Br'`),
            by default True.

        Returns
        -------
        str
            Reconstructed SMILES string truncated at `'EOS'`.
        """

        ...

    def parseDecoded(self, smiles: str) -> str:
        """
        Substitute single-letter halogen placeholders back into valid SMILES elements.

        Parameters
        ----------
        smiles : str
            Internal SMILES containing 'L' for Cl and 'R' for Br.

        Returns
        -------
        str
            Standard SMILES with 'Cl' and 'Br'.
        """

        ...

    def splitSequence(self, smile: str) -> List[str]:
        """
        Split a SMILES string into tokens, replacing halogens and parsing bracket atoms.

        Parameters
        ----------
        smile : str
            Input SMILES string.

        Returns
        -------
        List[str]
            List of discrete tokens terminated with `'EOS'`.
        """

        ...

    @staticmethod
    def fromFile(path: Union[str, os.PathLike[str], Path], encode_frags: bool = ..., min_len: int = ..., max_len: int = ...) -> VocSmiles:
        """
        Load vocabulary words from a whitespace-separated file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to vocabulary file.
        encode_frags : bool, optional
            Whether to support fragment delimiter tokens, by default False.
        min_len : int, optional
            Min sequence length, by default 10.
        max_len : int, optional
            Max sequence length, by default 100.

        Returns
        -------
        VocSmiles
            Initialized SMILES vocabulary instance.
        """

        ...

    def calc_voc_fp(self, smiles: Sequence[str], prefix: Optional[str] = ...) -> np.ndarray:
        """
        Calculate vocabulary token fingerprint arrays for a batch of SMILES.

        Parameters
        ----------
        smiles : Sequence[str]
            List of SMILES strings.
        prefix : Optional[str], optional
            Optional prefix token to prepend to each sequence, by default None.

        Returns
        -------
        np.ndarray
            2D integer numpy array of shape `(len(smiles), max_len)`.
        """

        ...


class VocNonGPT(VocSmiles):
    """
    Vocabulary variant for legacy sequence models (`Seq2Seq` and `EncDec`).

    Supports asymmetric source and target sequence lengths (`src_len` vs `trg_len`)
    and vertical pipe prefixes for source tokens.

    Parameters
    ----------
    words : Sequence[str]
        Base token words.
    src_len : int, optional
        Source sequence maximum length, by default 1000.
    trg_len : int, optional
        Target sequence maximum length, by default 100.
    max_len : int, optional
        Maximum length fallback, by default 100.
    min_len : int, optional
        Minimum length, by default 10.
    """

    def __init__(self, words: Sequence[str], src_len: int = ..., trg_len: int = ..., max_len: int = ..., min_len: int = ...) -> None:
        """
        Initialize the VocNonGPT vocabulary.

        Parameters
        ----------
        words : Sequence[str]
            Base tokens.
        src_len : int, optional
            Source sequence length limit, by default 1000.
        trg_len : int, optional
            Target sequence length limit, by default 100.
        max_len : int, optional
            Max length fallback, by default 100.
        min_len : int, optional
            Min length, by default 10.
        """

        ...

    def encode(self, input: Sequence[Sequence[str]], is_smiles: bool = ...) -> torch.LongTensor:
        """
        Encode sequences according to source or target mode.

        Parameters
        ----------
        input : Sequence[Sequence[str]]
            Token sequences to encode.
        is_smiles : bool, optional
            If True, uses target length and standard tokens; if False, uses source length,
            by default True.

        Returns
        -------
        torch.LongTensor
            Encoded integer tensor.
        """

        ...

    def decode(self, matrix: Union[torch.Tensor, Sequence[int]], is_smiles: bool = ..., is_tk: bool = ...) -> str:  # type: ignore[override]
        """
        Decode index matrix back into a SMILES string.

        Parameters
        ----------
        matrix : Union[torch.Tensor, Sequence[int]]
            Indices to decode.
        is_smiles : bool, optional
            Whether to parse halogen substitutions, by default True.
        is_tk : bool, optional
            Whether elements are already token strings, by default False.

        Returns
        -------
        str
            Decoded SMILES string.
        """

        ...

    @staticmethod
    def fromFile(path: Union[str, os.PathLike[str], Path], src_len: int = ..., trg_len: int = ..., max_len: int = ..., min_len: int = ...) -> VocNonGPT:  # type: ignore[override]
        """
        Load legacy sequence vocabulary from a text file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to vocabulary file.
        src_len : int, optional
            Source sequence length, by default 1000.
        trg_len : int, optional
            Target sequence length, by default 100.
        max_len : int, optional
            Max length, by default 100.
        min_len : int, optional
            Min length, by default 10.

        Returns
        -------
        VocNonGPT
            Initialized vocabulary instance.
        """

        ...


class VocGraph(Vocabulary):
    """
    Vocabulary for graph-based molecule representations (`GraphTransformer`).

    Encodes molecules into atom valence state tokens (e.g. `'4C'`, `'3O+'`, `'1Cl'`),
    formal charges, bond connectivity indices, and fragment attachment flags.

    Parameters
    ----------
    words : Sequence[str], optional
        Initial list of atom-valence words, by default `defaultWords`.
    max_len : int, optional
        Maximum number of graph action steps, by default 80.
    n_frags : int, optional
        Maximum number of scaffold fragment components, by default 4.
    """

    defaultWords: Tuple[str, ...] = ...
    def __init__(self, words: Sequence[str] = ..., max_len: int = ..., n_frags: int = ...) -> None:
        """
        Initialize graph vocabulary.

        Parameters
        ----------
        words : Sequence[str], optional
            Atom-valence words, by default defaultWords.
        max_len : int, optional
            Maximum action steps, by default 80.
        n_frags : int, optional
            Max scaffold fragments, by default 4.
        """

        ...

    @staticmethod
    def parseWord(word: str) -> Tuple[str, int, int, int, str]:
        """
        Parse an atom-valence word into elemental and charge metadata.

        Parameters
        ----------
        word : str
            Atom-valence string (e.g. `'4C'`, `'3O+'`, `'*'`).

        Returns
        -------
        Tuple[str, int, int, int, str]
            `(element_with_charge, valence, charge_number, atomic_number, original_word)`
        """

        ...

    @staticmethod
    def fromFile(path: Union[str, os.PathLike[str], Path], word_col: str = ..., max_len: int = ..., n_frags: int = ...) -> VocGraph:  # type: ignore[override]
        """
        Load graph vocabulary from a TSV file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Path to TSV file.
        word_col : str, optional
            Column name containing token words, by default 'Word'.
        max_len : int, optional
            Maximum action steps, by default 80.
        n_frags : int, optional
            Fragment capacity, by default 4.

        Returns
        -------
        VocGraph
            Loaded graph vocabulary instance.
        """

        ...

    @staticmethod
    def fromDataFrame(df: pd.DataFrame, word_col: str = ..., max_len: int = ..., n_frags: int = ...) -> VocGraph:
        """
        Instantiate graph vocabulary from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing token column.
        word_col : str, optional
            Column name, by default 'Word'.
        max_len : int, optional
            Max steps, by default 80.
        n_frags : int, optional
            Max fragments, by default 4.

        Returns
        -------
        VocGraph
            Graph vocabulary instance.
        """

        ...

    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """
        Save vocabulary words and parsed atom properties to a TSV file.

        Parameters
        ----------
        path : Union[str, os.PathLike, Path]
            Output file path.
        """

        ...

    def toDataFrame(self) -> pd.DataFrame:
        """
        Export parsed vocabulary words into a structured pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns `['Ele', 'Val', 'Ch', 'Nr', 'Word']`.
        """

        ...

    def get_atom_tk(self, atom: Chem.Atom) -> int:
        """
        Compute the vocabulary index corresponding to an RDKit Atom's valence state.

        Parameters
        ----------
        atom : Chem.Atom
            RDKit atom instance.

        Returns
        -------
        int
            Vocabulary token index.
        """

        ...

    def encode(self, smiles: Sequence[str], subs: Optional[Sequence[str]] = ...) -> np.ndarray:
        """
        Encode molecules and scaffold fragments into 5-tuple graph construction actions.

        Parameters
        ----------
        smiles : Sequence[str]
            Sequence of SMILES strings.
        subs : Optional[Sequence[str]], optional
            Corresponding scaffold fragment SMILES strings, by default None.

        Returns
        -------
        np.ndarray
            3D integer numpy array of graph action tuples: `(atom_tk, curr_atom, prev_atom, bond_type, frag_flag)`.
        """

        ...

    def decode(self, matrix: Union[np.ndarray, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Decode graph construction action matrices into reconstructed fragment and molecule SMILES.

        Parameters
        ----------
        matrix : Union[np.ndarray, torch.Tensor]
            3D matrix of action tuples `(atom, curr_atom, prev_atom, bond, frag)`.

        Returns
        -------
        Tuple[List[str], List[str]]
            `(fragments_smiles_list, molecule_smiles_list)`
        """

        ...

