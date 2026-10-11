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
from argparse import Namespace
from drugex.training.interfaces import Environment
from drugex.data.corpus.interfaces import Vocabulary
from drugex.training.explorers.interfaces import Explorer
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.training.generators.interfaces import Generator
from drugex.training.monitors import TrainingMonitor

import os
import json
import argparse
import warnings
from qsprpred.models import QSPRModel
from qsprpred.tasks import ModelTasks
from drugex import VERSION
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import GraphFragDataSet
from drugex.data.datasets import SmilesDataSet
from drugex.data.datasets import SmilesFragDataSet
from drugex.logs.utils import backUpFiles
from drugex.logs.utils import enable_file_logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.explorers import FragSequenceExplorer
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.generators import SequenceRNN
from drugex.training.generators import SequenceTransformer
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.rewards import ParetoTanimotoDistance
from drugex.training.rewards import WeightedSum
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.modifiers import SmoothHump
from drugex.training.scorers.properties import LigandEfficiency
from drugex.training.scorers.properties import LipophilicEfficiency
from drugex.training.scorers.properties import Property
from drugex.training.scorers.properties import Uniqueness
from drugex.training.scorers.similarity import FraggleSimilarity
from drugex.training.scorers.similarity import TverskyFingerprintSimilarity
from drugex.training.scorers.similarity import TverskyGraphSimilarity
from drugex.training.scorers.qsprpred import QSPRPredScorer


def GeneratorArgParser() -> argparse.ArgumentParser:
    """Define and read command line arguments"""

    ...


class DataPreparation:
    """
    Class for preparing data for training and testing from input files.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    """

    def __init__(self, base_dir: str, voc_files: Sequence[str], input: str, batch_size: int, n_samples: int, mt: str) -> None:
        """
        Parameters
        ----------
        base_dir : str
            Path to base directory.
        voc_files : list
            List of vocabulary file names.
        input : str
            Name of input file or prefix of input files.
        batch_size : int
            Batch size.
        n_samples : int
            Number of samples.
        mt : str
            Molecule type.
        """

        ...

    def getVocPaths(self, custom_type: Any = ...) -> List[Any]:
        """
        Get paths to vocabulary files. If none are found, use internal defaults.

        Returns
        -------
        list
            List of paths to vocabulary files.
        """

        ...

    def getDataPaths(self) -> Tuple[str, str]:
        """
        Get paths to training and test data files.

        Returns
        -------
        Tuple[str, str]
            Paths to training and test data files.
        """

        ...


class FragGraphDataPreparation(DataPreparation):
    """
    Data preprocessing for graph-based fragment generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'graph'
    """

    def __init__(self, base_dir: str, voc_files: Sequence[str], input: str, batch_size: int, n_samples: int, unique_frags: bool = ...) -> None:
        """Constructs and configures the fragment graph data preparation pipeline for Graph Transformers."""

        ...

    def __call__(self) -> Tuple[VocGraph, DataLoader, DataLoader]:
        """
        Get data loaders for training and testing and vocabulary.

        Returns
        -------
        Tuple[VocGraph, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        ...


class FragSmilesDataPreparation(DataPreparation):
    """
    Data preprocessing for SMILES-based fragment generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'smiles'
    """

    def __init__(self, base_dir: str, voc_files: Sequence[str], input: str, batch_size: int, n_samples: int, unique_frags: bool = ...) -> None:
        """Constructs and configures the paired fragment SMILES data preparation pipeline for Transformers."""

        ...

    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]:
        """
        Get data loaders for training and testing and vocabulary.

        Returns
        -------
        Tuple[VocSmiles, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        ...


class SmilesDataPreparation(DataPreparation):
    """
    Data preprocessing for SMILES-based molecule generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'smiles'
    """

    def __init__(self, base_dir: str, voc_files: Sequence[str], input: str, batch_size: int, n_samples: int, unique_frags: bool = ...) -> None:
        """Constructs and configures the SMILES sequence data preparation pipeline for RNNs."""

        ...

    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]:
        """
        Get data loaders for training and testing and vocabulary.

        Returns
        -------
        Tuple[VocGraph, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        ...


class SetUpGenerator:
    """Set up generator object for training or molecule generation"""

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Set up generator object for training or molecule generation

        Parameters
        ----------
        args : object
            Command line arguments
        """

        ...

    def setGeneratorAlgorithm(self, voc: Vocabulary) -> Generator:
        """
        Set generator algorithm based on molecule type and generator algorithm type

        Parameters
        ----------
        voc : object
            Vocabulary object

        Returns
        -------
        generator : object
            DrugEx Generator object
        """

        ...

    def prepareInputs(self) -> Tuple[Vocabulary, DataLoader, DataLoader]:
        """
        Prepare inputs for training or molecule generation

        Returns
        -------
        Tuple[Vocabulary, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        ...

    def loadStatesFromFile(self, generator: Generator, generator_path: str) -> None:
        """
        Load pretrained weights from file

        Parameters
        ----------
        generator : object
            DrugEx Generator object
        generator_path : str
            Path to pretrained weights

        Returns
        -------
        generator : object
            DrugEx Generator object
        """

        ...


class Pretrain(SetUpGenerator):
    """Pretrain generator"""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes pretraining orchestrator attributes from parsed CLI namespace object."""

        ...

    def __call__(self) -> Generator:
        """Pretrain generator"""

        ...


class Finetune(SetUpGenerator):
    """Finetune generator"""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes transfer-learning orchestrator attributes from parsed CLI namespace object."""

        ...

    def __call__(self) -> Generator:
        """Finetune generator"""

        ...


class Reinforce(SetUpGenerator):
    """Train generator with reinforcement learning"""

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Parameters
        ----------
        args : Namespace
            Command line arguments
        """

        ...

    def setExplorer(self, agent: object, prior: object, env: object) -> object:
        """
        Set explorer algorithm and parameters

        Parameters
        ----------
        agent : object
            DrugEx Generator object to be fitted
        prior : object
            DrugEx Generator object to be used as prior
        env : object
            DrugEx Environment object to score molecules

        Returns
        -------
        explorer : object
            DrugEx Explorer object
        """

        ...

    def __call__(self) -> Explorer:
        """Train generator with reinforcement learning"""

        ...


def getModifiers(task: ModelTasks, scheme: str, activity_threshold: float) -> Tuple[ClippedScore, ClippedScore, ClippedScore]:
    """
    Gets the modifiers for a given task, scheme and activity threshold combination.

    Arguments:
        task (str): The task to get the modifiers for.
        scheme (str): The scheme to get the modifiers for.
        activity_threshold (float): The activity threshold to use for the modifiers when relevant.

    Returns:
        list: The active modifiers for the given task.
    """

    ...


def CreateEnvironment(base_dir: str, predictor: Optional[List[str]], scheme: str, active_targets: Optional[List[str]] = ..., inactive_targets: Optional[List[str]] = ..., window_targets: Optional[List[str]] = ..., activity_threshold: float = ..., qed: bool = ..., unique: bool = ..., sa_score: bool = ..., ra_score: bool = ..., mw: bool = ..., mw_ths: Optional[List[int]] = ..., logP: bool = ..., logP_ths: Optional[List[float]] = ..., tpsa: bool = ..., tpsa_ths: Optional[List[float]] = ..., sim_smiles: Optional[str] = ..., sim_type: str = ..., sim_th: float = ..., sim_tw: Optional[List[float]] = ..., le: bool = ..., le_ths: Optional[List[float]] = ..., lipe: bool = ..., lipe_ths: Optional[List[float]] = ..., logger: Optional[Any] = ...) -> Optional[DrugExEnvironment]:
    """
    Sets up the objectives of the desirability function.

    Arguments:
        base_dir (str)              : folder containing 'qspr' folder with saved environment-predictor models
        predictor (list)                  : environment-predictor algoritm
        task (str)                  : environment-predictor task: 'REG' or 'CLS'
        scheme (str)                : optimization scheme: 'WS' for weighted sum, 'PRTD' for Pareto ranking with Tanimoto distance or 'PRCD' for Pareto ranking with crowding distance.
        active_targets (lst), opt   : list of active target IDs
        inactive_targets (lst), opt : list of inactive target IDs
        window_targets (lst), opt   : list of target IDs for selectivity window
        activity_threshold (float), opt : activity threshold in case of 'CLS'
        qed (bool), opt             : if True, 'quantitative estimate of drug-likeness' included in the desirability function
        unique (bool), opt          : if Trye, molecule uniqueness in an epoch included in the desirability function
        ra_score (bool), opt        : if True, 'Retrosythesis Accessibility score' included in the desirability function
        mw (bool), opt              : if True, large molecules are penalized in the desirability function
        mw_ths (list), opt          : molecular weight thresholds to penalize large molecules
        logP (bool), opt            : if True, molecules with logP values are penalized in the desirability function
        logP_ths (list), opt        : logP thresholds to penalize large molecules
        tpsa (bool), opt            : if True, tpsa used in the desirability function
        tpsa_ths (list), opt        : tpsa thresholds
        sim_smiles (str), opt       : reference molecules used for similarity calculation
        sim_type (str), opt         : type of fingerprint or 'graph' used for similarity calculation
        sim_th (float), opt         : threshold for similarity desirability  
        sim_tw (list), opt          : tversky similarity weights
        le (bool), opt              : if True, ligand efficiency used instead of activity for active targets in the desirability function
        le_ths (list), opt          : ligand efficiency thresholds
        lipe (bool), opt            : if True, lipophilic efficiency used instead of activity for active targets in the desirability function
        lipe_ths (list), opt        : lipophilic efficiency thresholds
        log (str), opt              : log instance

    Returns:
        DrugExEnvironment (torch model) : environment-predictor model
    """

    ...

