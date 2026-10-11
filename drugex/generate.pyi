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

import argparse
import json
import os
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import GraphFragDataSet
from drugex.data.datasets import SmilesFragDataSet
from drugex.data.utils import getVocPaths
from drugex.logs.utils import backUpFiles
from drugex.logs.utils import enable_file_logger
from drugex.train import CreateEnvironment
from drugex.train import SetUpGenerator
from drugex.train import DataPreparation


def DesignArgParser(txt: Optional[List[str]] = ...) -> Dict[str, Any]:
    """Define and read command line arguments"""

    ...


def DesignerFragsDataPreparation(voc_files: List[Any], data_path: str, input_file: str, mol_type: str) -> Union[SmilesFragDataSet, GraphFragDataSet]:
    """
    Reads and preprocesses the vocabulary and input data for a graph-based generator

    Arguments:
        voc_files (list)            : list of vocabulary file prefixes
        data_path (str)             : name of the folder input files
        input_file (str)            : name of file containing input fragments
        mol_type (str)              : molecule type
        alg (str)                   : generator algoritm
        batch_size (int), opt       : batch size
        n_samples (int), opt        : number of molecules to generate
    Returns:
        voc                         : atom vocabulary
        loader                      : torch DataLoader containing input fragements
    """

    ...


def Design(args: Namespace) -> List[str]:
    """Execute generative sampling using a trained agent and score generated compounds in the scoring environment."""

    ...

