from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from argparse import Namespace
from drugex.data.corpus.interfaces import Vocabulary

import os
import json
import time
import argparse
from drugex.logs.utils import enable_file_logger
from drugex.logs.utils import backUpFiles
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.converters.fragmenters import FragmenterWithSelectedFragment
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.processing import Standardization
from drugex.data.processing import CorpusEncoder
from drugex.data.processing import RandomTrainTestSplitter
from drugex.data.datasets import SmilesDataSet
from drugex.data.datasets import SmilesFragDataSet
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import FragmentPairsSplitter
from drugex.data.fragments import SequenceFragmentEncoder
from drugex.data.fragments import GraphFragmentEncoder
from drugex.data.fragments import FragmentCorpusEncoder
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.corpus.vocabulary import VocGraph


def DatasetArgParser() -> argparse.Namespace:
    """Parse command-line arguments for dataset creation, molecular representations, and fragmentation strategies."""

    ...


def load_molecules(base_dir: str, input_file: str) -> List[str]:
    """
    Loads raw SMILES from input file and transform to rdkit molecule
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with input file
        input_file  (str)         : file containing SMILES, can be '(compressed) 'tsv' or 'csv' file
    Returns:
        mols (list)                : list of SMILES extracted from input_file
    """

    ...


class Dataset:
    """Base dataset manager configuring output file paths and vocabulary initialization."""

    def __init__(self, args: argparse.Namespace) -> None:
        ...

    def setVocabulary(self) -> VocSmiles:
        """
        Set up vocabulary for sequence-based datasets.    
        Returns
        -------
        voc : VocSmiles or None
            Vocabulary object
        update_voc : bool
            If True, update vocabulary
        """

        ...


class SequenceDataset(Dataset):
    """Dataset manager for unfragmented linear SMILES corpora used in RNN generator training."""

    def __init__(self, args: argparse.Namespace) -> None:
        ...

    def __call__(self, smiles_list: List[str]) -> None:
        """Prepare data for SMILES-based RNN generator"""

        ...


class FragmentDataset(Dataset):
    """Base class for fragment-derived molecular datasets configuring fragmentation and pair splitters."""

    def __init__(self, args: argparse.Namespace) -> None:
        ...

    def setPairCollectors(self) -> Dict[str, Callable[[Any], None]]:
        """
        Set up pair collectors for fragment-based datasets.    
        Returns
        -------
        pair_collectors : dict
            Dictionary containing pair collectors
        """

        ...


class FragSequenceDataset(FragmentDataset):
    """Dataset manager for fragmented SMILES sequences used in sequence transformer training."""

    def __init__(self, args: argparse.Namespace) -> None:
        ...

    def __call__(self, smiles_list: List[str]) -> None:
        """Prepare data for SMILES-based transformer generator"""

        ...


class FragGraphDataset(FragmentDataset):
    """Dataset manager for fragmented molecular graph representations used in graph transformer training."""

    def __init__(self, args: argparse.Namespace) -> None:
        ...

    def __call__(self, smiles_list: List[str]) -> None:
        """Prepare data for graph-based transformer generator"""

        ...

