"""
Sequence corpus implementations for converting SMILES datasets into training token streams.
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

from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from drugex.data.corpus.interfaces import Corpus
from drugex.data.corpus.interfaces import SequenceVocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.molecules.interfaces import MolSupplier


class SequenceCorpus(Corpus):
    """
    Corpus supplier encoding molecular sequences (SMILES) for sequence-based models (RNN and GPT-2).

    Parameters
    ----------
    molecules : Union[Iterable[str], MolSupplier]
        Stream or collection of SMILES strings.
    vocabulary : SequenceVocabulary | None, optional
        Vocabulary instance used for tokenization and index mapping, by default VocSmiles(False).
    update_voc : bool, optional
        If True, new tokens encountered in the dataset are dynamically added to the vocabulary,
        by default True.
    throw : bool, optional
        If True, molecules containing tokens absent from the vocabulary are skipped, by default False.
    check_unique : bool, optional
        If True, duplicate sequences in `molecules` are skipped, by default True.
    """

    def __init__(self, molecules: Union[Iterable[str], MolSupplier], vocabulary: Optional[SequenceVocabulary] = ..., update_voc: bool = ..., throw: bool = ..., check_unique: bool = ...) -> None:
        """
        Initialize the SequenceCorpus.

        Parameters
        ----------
        molecules : Union[Iterable[str], MolSupplier]
            Input SMILES dataset or supplier.
        vocabulary : SequenceVocabulary | None, optional
            Target vocabulary, by default VocSmiles(False).
        update_voc : bool, optional
            Whether to update vocabulary with unseen tokens, by default True.
        throw : bool, optional
            Whether to discard molecules with unseen tokens, by default False.
        check_unique : bool, optional
            Whether to enforce molecule uniqueness, by default True.
        """

        ...

    def saveVoc(self, path: str) -> None:
        """
        Save current vocabulary tokens to a text file.

        Parameters
        ----------
        path : str
            Destination file path.
        """

        ...

    def getVoc(self) -> SequenceVocabulary:
        """
        Return the active SequenceVocabulary instance.

        Returns
        -------
        SequenceVocabulary
            Currently active sequence vocabulary.
        """

        ...

    def processMolecule(self, seq: str) -> Optional[List[int]]:
        """
        Tokenize and integer-encode a single SMILES sequence.

        Parameters
        ----------
        seq : str
            Input SMILES sequence string.

        Returns
        -------
        Optional[List[int]]
            List of encoded integer token indices, or None if molecule was rejected
            (duplicate or invalid length/tokens).
        """

        ...

