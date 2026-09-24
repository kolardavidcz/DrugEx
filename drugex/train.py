#!/usr/bin/env python
"""
DrugEx Generator Training and Optimization Module.

This module provides both a command-line interface (CLI) and an object-oriented
Python API for pretraining, fine-tuning, and reinforcement learning of molecular
generators (SequenceRNN, SequenceTransformer, and GraphTransformer).

Dual-Use Architecture
---------------------
1. **Interactive / Script Development API**:
   Classes such as :class:`Pretrain`, :class:`Finetune`, :class:`Reinforce`, and
   :class:`DataPreparation` can be instantiated directly with keyword arguments or
   dictionaries in custom development scripts (e.g. `_david/david.py`, Jupyter
   notebooks, or workflow orchestrators). Developers do **not** need to manually
   re-implement tokenization, splitters, DataLoaders, model loading, or training loops.
   Pre-existing in-memory DataLoaders or models can also be injected directly.

2. **Command-Line Interface (CLI)**:
   The module can be invoked directly from the shell via `python -m drugex.train`,
   parsing command-line flags via :func:`GeneratorArgParser`.

Examples
--------
**1. Fine-tuning a pretrained model in a Python script:**

>>> from drugex.train import Finetune
>>> finetuner = Finetune(
...     base_dir=".",
...     input="ligand_corpus",
...     agent_path="data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT.pkg",
...     mol_type="smiles",
...     algorithm="rnn",
...     epochs=20,
...     batch_size=64,
...     use_gpus=[0]
... )
>>> finetuned_model = finetuner.fit()
>>> sample_molecules = finetuned_model.generate(num_samples=100)

**2. Multi-Objective Reinforcement Learning (RL):**

>>> from drugex.train import Reinforce
>>> rl = Reinforce(
...     base_dir=".",
...     input="ligand_corpus",
...     agent_path="generators/my_finetuned_agent.pkg",
...     prior_path="data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT.pkg",
...     mol_type="smiles",
...     algorithm="rnn",
...     scheme="PRCD",
...     qed=True,
...     sa_score=True,
...     epochs=50,
...     batch_size=128,
...     use_gpus=[0]
... )
>>> explorer = rl.fit()

**3. Direct DataPreparation for DataLoaders and Vocabularies:**

>>> from drugex.train import SmilesDataPreparation
>>> prep = SmilesDataPreparation(
...     base_dir=".",
...     voc_files=["voc_smiles.txt"],
...     input="ligand_corpus",
...     batch_size=64
... )
>>> voc, train_loader, valid_loader = prep()

**4. Building a Multi-Objective Scoring Environment:**

>>> from drugex.train import CreateEnvironment
>>> env = CreateEnvironment(
...     base_dir=".",
...     scheme="PRCD",
...     qed=True,
...     sa_score=True,
...     mw=True,
...     mw_ths=[200, 500]
... )
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader
from qsprpred.models import QSPRModel
from qsprpred.tasks import ModelTasks

from drugex import VERSION
from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.datasets import (GraphFragDataSet, SmilesDataSet,
                                  SmilesFragDataSet)
from drugex.logs import logger
from drugex.logs.utils import backUpFiles, enable_file_logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import (FragGraphExplorer, FragSequenceExplorer,
                                       SequenceExplorer)
from drugex.training.generators import (GraphTransformer, SequenceRNN,
                                        SequenceTransformer)
from drugex.training.explorers.interfaces import Explorer
from drugex.training.interfaces import Model, TrainingMonitor
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import (ParetoCrowdingDistance,
                                     ParetoTanimotoDistance, WeightedSum)
from drugex.training.scorers.modifiers import ClippedScore, SmoothHump
from drugex.training.scorers.properties import (LigandEfficiency,
                                                LipophilicEfficiency, Property,
                                                Uniqueness)
from drugex.training.scorers.similarity import (FraggleSimilarity,
                                                TverskyFingerprintSimilarity,
                                                TverskyGraphSimilarity)
from drugex.training.scorers.qsprpred import QSPRPredScorer

warnings.filterwarnings("ignore")

# Module-level logger fallback for programmatic imports outside CLI __main__
log = logger
    
def GeneratorArgParser(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and read command line arguments for DrugEx generator training workflows.

    Configures flags for data I/O, generator architectures, reinforcement learning
    policies, and multi-objective desirability functions.

    Parameters
    ----------
    args_list : Optional[List[str]], optional
        Command line arguments list to parse (defaults to None, which reads sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace with normalized GPU lists and output path prefixes.
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O parameters
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' (and 'qspr' if a QSPRpred model is used for scoring) and where 'generators' will be created to save output files.")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Prefix (using separate files for training and validation) or \
                          full name of input file(s) (used both for training and validation) containing encoded molecules or fragments-molecule pairs.")  
    parser.add_argument('-vfs', '--voc_files', type=str, nargs='*', default=None,
                        help="List of vocabulary files name (with or without the '.txt.vocab' extension). If None, set to be the first word of input and if no vocabulary file is found, a default vocabulary is used.")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Prefix of output files. If None, set to be the first word of input.")
    # Input models 
    parser.add_argument('-ag', '--agent_path', type=str, default=None,
                        help="Name of model (w/o .pkg extension) or full path to agent model. Only used in FT and RL modes.")
    parser.add_argument('-pr', '--prior_path', type=str, default=None,
                        help="Name of model (w/o .pkg extension) of full path to prior model. Only used in RL mode.")
    
    # General parameters
    parser.add_argument('-tm', '--training_mode', type=str, default='RL',
                        help="Mode, of the training: 'PT' for pretraining, 'FT' for fine-tuning and 'RL' for reinforcement learning") 
    parser.add_argument('-mt', '--mol_type', type=str, default='graph',
                        help="Molecule encoding type: 'smiles' or 'graph'")    
    parser.add_argument('-a', '--algorithm', type=str, default='trans',
                        help="Generator algorithm: 'trans' (graph- or smiles-based transformer model) or "\
                             "'rnn' (smiles-based recurrent neural network).")
    parser.add_argument('-gru', '--use_gru', action='store_true',
                        help="If on, use GRU units for the RNN model. Ignore if algorithm is not 'rnn'")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-gpu', '--use_gpus', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-pa', '--patience', type=int, default=50,
                        help="Number of epochs to wait before early stop if no progress on test set score.")
    
    
    # RL parameters
    parser.add_argument('-ns', '--n_samples', type=int, default=-1, 
                        help="During RL, n_samples and 0.2*n_samples random input fragments are used for training and validation at each epoch. If -1, all input data is used at once each epoch.") 
    parser.add_argument('-eps', '--epsilon', type=float, default=0.1,
                        help="Exploring rate: probability of using the prior model instead of the agent model.")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline used by the policy gradient algorithm.")
    parser.add_argument('-s', '--scheme', type=str, default='PRCD',
                        help="Multi-objective reward calculation scheme: 'WS' for weighted sum, 'PRTD' for Pareto ranking with Tanimoto distance or 'PRCD' for Pareto ranking with crowding distance.")

    # Affinity models
    parser.add_argument('-p', '--predictor', type=str, nargs='*', default=[],
                        help="The path to the serialized metadata of a QSPRPred model (ie. 'RF_meta.json'), this path is relative to the directory specified with '-b, --base_dir'. If different environments are required give environment of targets in order active, inactive, window.")
    parser.add_argument('-at', '--activity_threshold', type=float, default=6.5,
                        help="Activity threshold used by the modifier in case of a regressor activity predictor")
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=[],
                        help="Names of models that predict activity.")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Names of models that predict inactivity.")
    parser.add_argument('-tw', '--window_targets', type=str, nargs='*', default=[],
                        help="Names of models for which selectivity window is calculated.")
    parser.add_argument('-le', '--ligand_efficiency', action='store_true', 
                        help="If on, use the ligand efficiency instead of the simple affinity as objective for active targets.")
    parser.add_argument('-le_ths', '--le_thresholds', type=float, nargs=2, default=[0.0, 0.5],
                        help='Thresholds used calculate ligand efficiency clipped scores in the desirability function.')
    parser.add_argument('-lipe', '--lipophilic_efficiency', action='store_true',
                        help="If on, use the ligand lipophilic efficiency instead of the simple affinity as objective for active targets.")
    parser.add_argument('-lipe_ths', '--lipe_thresholds', type=float, nargs=2, default=[4.0, 6.0],
                        help='Thresholds used calculate lipophilic efficiency clipped scores in the desirability function.')
    
    # Pre-implemented properties
    parser.add_argument('-qed', '--qed', action='store_true',
                        help="If on, QED is used in desirability function")
    parser.add_argument('-unq', '--uniqueness', action='store_true',
                        help="If on, molecule uniqueness is used in desirability function")
    parser.add_argument('-sas', '--sa_score', action='store_true',
                        help="If on, Synthetic Accessibility score is used in desirability function")       
    parser.add_argument('-ras', '--ra_score', action='store_true',
                        help="If on, Retrosynthesis Accessibility score is used in desirability function")
    parser.add_argument('-mw', '--molecular_weight', action='store_true',
                        help='If on, compounds with molecular weights outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-mw_ths', '--mw_thresholds', type=int, nargs='*', default=[200, 600],
                        help='Thresholds used calculate molecular weights clipped scores in the desirability function.')
    parser.add_argument('-logP', '--logP', action='store_true',
                        help='If on, compounds with logP values outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-logP_ths', '--logP_thresholds', type=float, nargs='*', default=[-5, 5],
                        help='Thresholds used calculate logP clipped scores in the desirability function')
    parser.add_argument('-tpsa', '--tpsa', action='store_true',
                        help='If on, topology polar surface area is used in desirability function')
    parser.add_argument('-tpsa_ths', '--tpsa_thresholds', type=float, nargs=2, default=[0, 140],
                        help='Thresholds used calculate TPSA clipped scores in the desirability function')
    parser.add_argument('-sim_mol', '--similarity_mol', type=str, default=None,
                        help='SMILES string of a reference molecule to which the similarity is used as an objective. Similarity metric and threshold set by --sim_metric and --sim_th.')
    parser.add_argument('-sim_type', '--similarity_type', type=str, default='fraggle',
                        help="'fraggle' for Fraggle similarity, 'graph' for Tversky similarity between graphs or fingerprints name ('AP', 'PHCO', 'BPF', 'BTF', 'PATH', 'ECFP4', 'ECFP6', 'FCFP4', 'FCFP6') for Tversky similarity between fingeprints")
    parser.add_argument('-sim_th', '--similarity_threshold', type=float, default=0.5,
                        help="Threshold for molecular similarity to reference molecule")
    parser.add_argument('-sim_tw', '--similarity_tversky_weights', nargs=2, type=float, default=[0.7, 0.3],
                        help="Weights (alpha and beta) for Tversky similarity. If both equal to 1.0, Tanimoto similarity.")       
    
    parser.add_argument('-d', '--debug', action='store_true')

            
    args = parser.parse_args(args_list)
    
    # Setting output file prefix from input file
    if args.output is None:
        args.output = args.input.split('_')[0] if args.input else 'drugex_model'

    # Setting vocabulary file names from input file
    if args.voc_files is None:
        args.voc_files = [args.input.split('_')[0]] if args.input else []
    
    # Setting target IDs as union of active, inactive and window targets
    args.targets = args.active_targets + args.inactive_targets + args.window_targets

    # Setting output file prefix 
    args.output_long = '_'.join([args.output, args.mol_type, args.algorithm, args.training_mode])
    args.output_file_base = f'{args.base_dir}/generators/{args.output_long}'

    if isinstance(args.use_gpus, str):
        args.use_gpus = [int(x.strip()) for x in args.use_gpus.split(',') if x.strip()]

    return args

class DataPreparation:
    """Base class for preparing dataset loaders and vocabularies for generator training.

    Handles resolving vocabulary file paths and dataset file paths from either
    an explicit file name or common file prefix patterns within the DrugEx data folder.

    Parameters
    ----------
    base_dir : str
        Path to the project base directory containing the 'data' subfolder.
    voc_files : Optional[List[str]]
        List of vocabulary file names or prefixes (with or without '.txt.vocab').
    input : Optional[str]
        Name of the input file or prefix of split files (e.g. 'ligands_train.txt').
    batch_size : int, optional
        Batch size for PyTorch DataLoader (default: 256).
    n_samples : int, optional
        Number of samples per epoch during training; -1 uses the full dataset (default: -1).
    mt : str, optional
        Molecule encoding type: 'smiles' or 'graph' (default: 'smiles').
    """

    def __init__(
        self,
        base_dir: str,
        voc_files: Optional[List[str]],
        input: Optional[str],
        batch_size: int = 256,
        n_samples: int = -1,
        mt: str = 'smiles'
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = f'{base_dir}/data'
        self.voc_files = voc_files or []
        self.input = input
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.mol_type = mt
        self.unique_frags = False

    def getVocPaths(self, custom_type: Optional[str] = None) -> List[str]:
        """Get paths to vocabulary files. If none are found, use internal defaults.

        Parameters
        ----------
        custom_type : Optional[str], optional
            Override molecule type (e.g. 'corpus') when searching for vocabulary file names.

        Returns
        -------
        List[str]
            List of absolute or relative paths to detected vocabulary files.
        """
        mol_type = self.mol_type if not custom_type else custom_type
        voc_paths: List[str] = []
        for voc_file in self.voc_files:
            if os.path.exists(f'{self.data_dir}/{voc_file}'):
                # Vocabulary from full path name
                voc_paths.append(f'{self.data_dir}/{voc_file}')
            elif os.path.exists(f'{self.data_dir}/{voc_file}_{mol_type}.txt.vocab'):
                # Vocabulary from file name without extension
                voc_paths.append(f'{self.data_dir}/{voc_file}_{mol_type}.txt.vocab')
            else:
                log.warning(f'Could not find vocabulary file {voc_file} or {voc_file}.txt.vocab in {self.data_dir}.')
                
        if len(voc_paths) == 0:
            log.warning(f'No vocabulary files found. Using internal defaults for DrugEx v{VERSION}.')

        return voc_paths
    
    def getDataPaths(self) -> Tuple[str, str]:
        """Get paths to training and validation/test data files.

        Returns
        -------
        Tuple[str, str]
            Paths to training and test/validation data files.
        """
        # If exact data path was given as input, that data is both used for training and testing
        if os.path.exists(f'{self.data_dir}/{self.input}'):
            train_path = f'{self.data_dir}/{self.input}'
            test_path = train_path
        # Else if prefix was given, read separate train and test sets
        else:
            train_set = 'unique' if getattr(self, 'unique_frags', False) else 'train'
            train_path = f'{self.data_dir}/{self.input}_{train_set}_{self.mol_type}.txt'
            test_path = f'{self.data_dir}/{self.input}_test_{self.mol_type}.txt'
 
        assert os.path.exists(train_path), f'{train_path} does not exist'
        assert os.path.exists(test_path), f'{test_path} does not exist'          
            
        log.info(f'Loading training data from {train_path}')
        log.info(f'Loading validation data from {test_path}')

        return train_path, test_path   

class FragGraphDataPreparation(DataPreparation):
    """Data preprocessing and DataLoader constructor for graph-based fragment generation.

    Prepares training and validation PyTorch `DataLoader`s and a `VocGraph` vocabulary
    from graph-encoded fragment-molecule pair datasets.

    Parameters
    ----------
    base_dir : str
        Path to base directory containing 'data/'.
    voc_files : Optional[List[str]]
        List of vocabulary file names or prefixes.
    input : Optional[str]
        Name of input file or prefix of train/test files.
    batch_size : int, optional
        Base batch size (default: 256).
    n_samples : int, optional
        Number of samples per epoch (-1 for all data, default: -1).
    unique_frags : bool, optional
        If True, only unique fragments are used for training (default: False).
    """

    def __init__(
        self,
        base_dir: str,
        voc_files: Optional[List[str]],
        input: Optional[str],
        batch_size: int = 256,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None:
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'graph')
        self.unique_frags = unique_frags

    def __call__(self) -> Tuple[VocGraph, DataLoader, DataLoader]:
        """Load vocabulary and construct graph fragment DataLoaders.

        Returns
        -------
        Tuple[VocGraph, DataLoader, DataLoader]
            Merged VocGraph vocabulary, training DataLoader, and validation DataLoader.
        """
        # Get vocabulary and data paths
        voc_paths = self.getVocPaths()
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = GraphFragDataSet(train_path)
        if voc_paths:
            dataset_train.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size * 4, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = GraphFragDataSet(test_path)
        if voc_paths:
            dataset_test.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size * 10, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc() + dataset_test.getVoc()
        
        return voc, train_loader, valid_loader


class FragSmilesDataPreparation(DataPreparation):
    """Data preprocessing and DataLoader constructor for SMILES-based fragment generation.

    Prepares training and validation PyTorch `DataLoader`s and a `VocSmiles` vocabulary
    from SMILES-encoded fragment-molecule pair datasets.

    Parameters
    ----------
    base_dir : str
        Path to base directory containing 'data/'.
    voc_files : Optional[List[str]]
        List of vocabulary file names or prefixes.
    input : Optional[str]
        Name of input file or prefix of train/test files.
    batch_size : int, optional
        Base batch size (default: 256).
    n_samples : int, optional
        Number of samples per epoch (-1 for all data, default: -1).
    unique_frags : bool, optional
        If True, only unique fragments are used for training (default: False).
    """

    def __init__(
        self,
        base_dir: str,
        voc_files: Optional[List[str]],
        input: Optional[str],
        batch_size: int = 256,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None:
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'smiles')
        self.unique_frags = unique_frags

    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]:
        """Load vocabulary and construct SMILES fragment DataLoaders.

        Returns
        -------
        Tuple[VocSmiles, DataLoader, DataLoader]
            VocSmiles vocabulary, training DataLoader, and validation DataLoader.
        """
        # Get vocabulary and data paths
        voc_paths = self.getVocPaths()
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = SmilesFragDataSet(train_path)
        dataset_train.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=True)
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = SmilesFragDataSet(test_path)
        dataset_test.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=True)
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc()
        
        return voc, train_loader, valid_loader
    

class SmilesDataPreparation(DataPreparation):
    """Data preprocessing and DataLoader constructor for SMILES sequence generation.

    Prepares training and validation PyTorch `DataLoader`s and a `VocSmiles` vocabulary
    from standardized SMILES corpus datasets (e.g. for RNN or Sequence Transformer).

    Parameters
    ----------
    base_dir : str
        Path to base directory containing 'data/'.
    voc_files : Optional[List[str]]
        List of vocabulary file names or prefixes.
    input : Optional[str]
        Name of input file or prefix of train/test files.
    batch_size : int, optional
        Base batch size (default: 256).
    n_samples : int, optional
        Number of samples per epoch (-1 for all data, default: -1).
    unique_frags : bool, optional
        Unused for whole-molecule SMILES (kept for API consistency, default: False).
    """

    def __init__(
        self,
        base_dir: str,
        voc_files: Optional[List[str]],
        input: Optional[str],
        batch_size: int = 256,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None:
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'smiles')
        self.unique_frags = False

    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]:
        """Load vocabulary and construct SMILES sequence DataLoaders.

        Returns
        -------
        Tuple[VocSmiles, DataLoader, DataLoader]
            VocSmiles vocabulary, training DataLoader, and validation DataLoader.
        """
        # Get vocabulary and data paths
        voc_paths = self.getVocPaths(custom_type="corpus")
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = SmilesDataSet(train_path)
        if voc_paths:
            dataset_train.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=False)
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = SmilesDataSet(test_path)
        dataset_test.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=False)
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc()
        
        return voc, train_loader, valid_loader

class SetUpGenerator:
    """Base class for configuring and running DrugEx generator workflows.

    Coordinates vocabulary resolution, dataset loading, model initialization
    (GraphTransformer, SequenceTransformer, SequenceRNN), checkpoint restoration,
    training monitoring, and execution.

    Supports both CLI execution via `argparse.Namespace` and direct programmatic
    instantiation in regular Python development scripts (e.g. `_david/david.py`,
    notebooks, or workflow pipelines) via keyword arguments or dict.

    Parameters
    ----------
    args : Optional[Union[argparse.Namespace, Dict[str, Any], Any]], optional
        Command line arguments namespace, configuration dictionary, or object.
    **kwargs : Any
        Keyword arguments overriding or supplying configuration options directly.
        Recognized options include:
        - `base_dir` (str): Working directory path containing 'data/' and 'generators/' (default: '.').
        - `input` (str, optional): Input dataset file name or prefix.
        - `output` (str, optional): Output model name / prefix.
        - `voc_files` (List[str], optional): Vocabulary file names or prefixes.
        - `agent_path` (str, optional): Path to pretrained agent model weights (.pkg).
        - `prior_path` (str, optional): Path to pretrained prior model weights (.pkg, RL mode).
        - `training_mode` (str): Mode: 'PT' (pretraining), 'FT' (fine-tuning), 'RL' (reinforcement).
        - `mol_type` (str): Molecule encoding: 'smiles' or 'graph' (default: 'smiles').
        - `algorithm` (str): Architecture: 'trans' (Transformer) or 'rnn' (SequenceRNN) (default: 'rnn').
        - `use_gru` (bool): Use GRU instead of LSTM for SequenceRNN (default: False).
        - `epochs` (int): Number of training epochs (default: 1000).
        - `batch_size` (int): Training batch size (default: 256).
        - `use_gpus` (List[int] or str): GPU device IDs (default: [0]).
        - `patience` (int): Early stopping patience in epochs (default: 50).
        - `loss_tolerance` (float, optional): Loss tolerance threshold for early stopping.
        - `n_samples` (int): Sample count per epoch (-1 for all data, default: -1).
        - `epsilon` (float): Exploration rate for RL (default: 0.1).
        - `beta` (float): Reward baseline for policy gradient (default: 0.0).
        - `scheme` (str): Multi-objective reward scheme: 'PRCD', 'PRTD', or 'WS' (default: 'PRCD').
        - `predictor` (List[str]): Paths to QSPRPred model metadata files.
        - `active_targets`, `inactive_targets`, `window_targets` (List[str]): Target IDs.
        - `activity_threshold` (float): Activity cutoff for classification/regression (default: 6.5).
        - `qed`, `uniqueness`, `sa_score`, `ra_score` (bool): Property objectives.
        - `mw`, `mw_thresholds` (bool, List[int]): Molecular weight penalty.
        - `logP`, `logP_thresholds` (bool, List[float]): logP penalty.
        - `tpsa`, `tpsa_thresholds` (bool, List[float]): TPSA penalty.
        - `similarity_mol`, `similarity_type`, `similarity_threshold`, `similarity_tversky_weights`: Similarity objectives.
        - `ligand_efficiency`, `le_thresholds` (bool, List[float]): Ligand efficiency objective.
        - `lipophilic_efficiency`, `lipe_thresholds` (bool, List[float]): Lipophilic efficiency objective.
        - `train_loader` (DataLoader, optional): Pre-existing training DataLoader to bypass file loading.
        - `valid_loader` or `test_loader` (DataLoader, optional): Pre-existing validation DataLoader.
        - `voc` (VocSmiles or VocGraph, optional): Pre-existing vocabulary instance.
        - `agent` (Model, optional): Pre-existing generator model instance.
        - `prior` (Model, optional): Pre-existing prior generator model instance.
        - `environment` (DrugExEnvironment, optional): Pre-existing environment instance.
        - `monitor` (TrainingMonitor, optional): Custom progress monitor instance.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        'base_dir': '.',
        'input': None,
        'voc_files': None,
        'output': None,
        'agent_path': None,
        'prior_path': None,
        'training_mode': 'RL',
        'mol_type': 'smiles',
        'algorithm': 'rnn',
        'use_gru': False,
        'epochs': 1000,
        'batch_size': 256,
        'use_gpus': [0],
        'patience': 50,
        'loss_tolerance': None,
        'n_samples': -1,
        'epsilon': 0.1,
        'beta': 0.0,
        'scheme': 'PRCD',
        'predictor': [],
        'activity_threshold': 6.5,
        'active_targets': [],
        'inactive_targets': [],
        'window_targets': [],
        'targets': [],
        'ligand_efficiency': False,
        'le_thresholds': [0.0, 0.5],
        'lipophilic_efficiency': False,
        'lipe_thresholds': [4.0, 6.0],
        'qed': False,
        'uniqueness': False,
        'sa_score': False,
        'ra_score': False,
        'molecular_weight': False,
        'mw_thresholds': [200, 600],
        'logP': False,
        'logP_thresholds': [-5.0, 5.0],
        'tpsa': False,
        'tpsa_thresholds': [0.0, 140.0],
        'similarity_mol': None,
        'similarity_type': 'fraggle',
        'similarity_threshold': 0.5,
        'similarity_tversky_weights': [0.7, 0.3],
        'debug': False,
        'unique_frags': False,
        'train_loader': None,
        'valid_loader': None,
        'test_loader': None,
        'voc': None,
        'agent': None,
        'prior': None,
        'environment': None,
        'monitor': None,
    }

    training_mode: str = 'RL'

    def __init__(
        self,
        args: Optional[Union[argparse.Namespace, Dict[str, Any], Any]] = None,
        **kwargs: Any
    ) -> None:
        # 1. Populate default attributes
        for key, value in self.DEFAULT_CONFIG.items():
            setattr(self, key, value)

        # Apply class-level training_mode default (PT for Pretrain, FT for Finetune, RL for Reinforce)
        self.training_mode = getattr(self.__class__, 'training_mode', 'RL')

        # 2. Merge options from args namespace or dictionary
        if args is not None:
            if isinstance(args, dict):
                for key, value in args.items():
                    setattr(self, key, value)
            elif hasattr(args, '__dict__'):
                for key, value in vars(args).items():
                    setattr(self, key, value)
            elif hasattr(args, '_asdict'):
                for key, value in args._asdict().items():
                    setattr(self, key, value)

        # 3. Apply keyword arguments (highest precedence)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # 4. Resolve derived paths and properties
        if self.output is None:
            self.output = self.input.split('_')[0] if self.input else 'drugex_model'

        if self.voc_files is None:
            self.voc_files = [self.input.split('_')[0]] if self.input else []

        if not self.targets:
            self.targets = list(self.active_targets) + list(self.inactive_targets) + list(self.window_targets)

        self._output_long: Optional[str] = getattr(self, '_output_long', None)
        self._output_file_base: Optional[str] = getattr(self, '_output_file_base', None)

        if isinstance(self.use_gpus, str):
            self.use_gpus = [int(x.strip()) for x in self.use_gpus.split(',') if x.strip()]
        elif isinstance(self.use_gpus, int):
            self.use_gpus = [self.use_gpus]

        if self.valid_loader is not None and self.test_loader is None:
            self.test_loader = self.valid_loader
        elif self.test_loader is not None and self.valid_loader is None:
            self.valid_loader = self.test_loader

    @property
    def output_long(self) -> str:
        """Get the formatted output identifier prefix."""
        if getattr(self, '_output_long', None) is not None:
            return self._output_long
        return '_'.join([str(self.output), str(self.mol_type), str(self.algorithm), str(self.training_mode)])

    @output_long.setter
    def output_long(self, value: Optional[str]) -> None:
        self._output_long = value

    @property
    def output_file_base(self) -> str:
        """Get the base path for output files and checkpoints."""
        if getattr(self, '_output_file_base', None) is not None:
            return self._output_file_base
        return f'{self.base_dir}/generators/{self.output_long}'

    @output_file_base.setter
    def output_file_base(self, value: Optional[str]) -> None:
        self._output_file_base = value

    def setGeneratorAlgorithm(self, voc: Union[VocSmiles, VocGraph]) -> Model:
        """Instantiate the generator model architecture based on molecule type and algorithm.

        Parameters
        ----------
        voc : Union[VocSmiles, VocGraph]
            Vocabulary instance mapping tokens/graph elements.

        Returns
        -------
        Model
            Instantiated DrugEx generator model (GraphTransformer, SequenceTransformer, or SequenceRNN).
        """
        if self.mol_type == 'graph' and self.algorithm == 'trans':
            generator = GraphTransformer(voc, use_gpus=self.use_gpus)
        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            generator = SequenceTransformer(voc, use_gpus=self.use_gpus)
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            generator = SequenceRNN(voc, is_lstm=not self.use_gru, use_gpus=self.use_gpus)
        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return generator
    
    def prepareInputs(self) -> Tuple[Optional[Union[VocSmiles, VocGraph]], DataLoader, DataLoader]:
        """Prepare vocabulary and training/validation DataLoaders.

        Checks if in-memory loaders were passed; otherwise delegates to the appropriate
        `DataPreparation` class based on molecule type and algorithm.

        Returns
        -------
        Tuple[Optional[Union[VocSmiles, VocGraph]], DataLoader, DataLoader]
            Vocabulary, training DataLoader, and validation DataLoader.
        """
        if self.train_loader is not None and self.test_loader is not None:
            return self.voc, self.train_loader, self.test_loader

        if self.mol_type == 'graph' and self.algorithm == 'trans':
            voc, train_loader, test_loader = FragGraphDataPreparation(
                self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples, self.unique_frags
            )()
        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            voc, train_loader, test_loader = FragSmilesDataPreparation(
                self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples, self.unique_frags
            )()
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            voc, train_loader, test_loader = SmilesDataPreparation(
                self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples
            )()
        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return voc, train_loader, test_loader  
    
    def loadStatesFromFile(self, generator: Model, generator_path: str) -> Model:
        """Load pretrained model weights from file.

        Parameters
        ----------
        generator : Model
            DrugEx generator instance to load weights into.
        generator_path : str
            Full path to weights file or model prefix relative to generators folder.

        Returns
        -------
        Model
            Generator instance with restored weights.
        """
        if os.path.exists(f'{generator_path}'):
            generator.loadStatesFromFile(f'{generator_path}')
        elif os.path.exists(f'{self.base_dir}/generators/{generator_path}.pkg'):
            generator.loadStatesFromFile(f'{self.base_dir}/generators/{generator_path}.pkg')
        else:
            raise ValueError(f'Could not find pretrained model at {generator_path} or {self.base_dir}/generators/{generator_path}.pkg')

        return generator

    def getMonitor(self) -> TrainingMonitor:
        """Get the configured TrainingMonitor or instantiate a default FileMonitor.

        Returns
        -------
        TrainingMonitor
            Monitor instance used to track training progress and checkpoints.
        """
        if self.monitor is not None:
            return self.monitor
        os.makedirs(os.path.dirname(self.output_file_base) or '.', exist_ok=True)
        return FileMonitor(self.output_file_base, save_smiles=True)


class Pretrain(SetUpGenerator):
    """Pretrain a DrugEx generator model from scratch.

    Initializes a new generator (SequenceRNN, SequenceTransformer, or GraphTransformer),
    constructs training/validation DataLoaders from input data files or in-memory loaders,
    and trains the network using teacher forcing.

    Can be executed via CLI or programmatically in Python scripts.

    Parameters
    ----------
    args : Optional[Union[argparse.Namespace, Dict[str, Any], Any]], optional
        Parsed command line arguments or config dictionary.
    **kwargs : Any
        Direct keyword arguments (e.g. base_dir, input, mol_type, algorithm, epochs, batch_size, use_gpus).

    Examples
    --------
    >>> from drugex.train import Pretrain
    >>> # Programmatic pretraining from Python script
    >>> pretrainer = Pretrain(
    ...     base_dir=".",
    ...     input="chembl_corpus",
    ...     mol_type="smiles",
    ...     algorithm="rnn",
    ...     epochs=50,
    ...     batch_size=128,
    ...     use_gpus=[0]
    ... )
    >>> pretrained_agent = pretrainer.fit()
    """

    training_mode: str = 'PT'

    def __init__(
        self,
        args: Optional[Union[argparse.Namespace, Dict[str, Any], Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(args, **kwargs)
        self.training_mode = 'PT'
        self.unique_frags = False

    def fit(self) -> Model:
        """Execute pretraining workflow and return the trained generator model.

        Returns
        -------
        Model
            Trained DrugEx generator instance.
        """
        return self()

    def __call__(self) -> Model:
        """Execute pretraining workflow.

        Returns
        -------
        Model
            Trained DrugEx generator instance.
        """
        voc, train_loader, test_loader = self.prepareInputs()
        agent = self.agent if self.agent is not None else self.setGeneratorAlgorithm(voc)
        monitor = self.getMonitor()

        log.info('Pretraining generator...')
        fit_kwargs: Dict[str, Any] = {}
        if self.loss_tolerance is not None:
            fit_kwargs['loss_tolerance'] = self.loss_tolerance

        agent.fit(
            train_loader,
            test_loader,
            epochs=self.epochs,
            monitor=monitor,
            patience=self.patience,
            **fit_kwargs
        )
        return agent


class Finetune(SetUpGenerator):
    """Fine-tune a pretrained DrugEx generator on a focused dataset.

    Loads pretrained model weights from file or accepts a pre-instantiated model,
    prepares DataLoaders for focused compounds, and optimizes the model for the target domain.

    Can be executed via CLI or programmatically in Python scripts (such as `_david/david.py`).

    Parameters
    ----------
    args : Optional[Union[argparse.Namespace, Dict[str, Any], Any]], optional
        Parsed command line arguments or config dictionary.
    **kwargs : Any
        Direct keyword arguments (e.g. base_dir, input, agent_path, mol_type, algorithm, epochs, batch_size, use_gpus).

    Examples
    --------
    >>> from drugex.train import Finetune
    >>> # In your development script:
    >>> finetuner = Finetune(
    ...     base_dir=".",
    ...     input="ligand_corpus",
    ...     agent_path="data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT.pkg",
    ...     mol_type="smiles",
    ...     algorithm="rnn",
    ...     epochs=20,
    ...     batch_size=64,
    ...     use_gpus=[0]
    ... )
    >>> finetuned_model = finetuner.fit()
    >>> sample_df = finetuned_model.generate(num_samples=50)

    Or using in-memory DataLoaders directly:
    >>> finetuner = Finetune(
    ...     agent_path="path/to/pretrained.pkg",
    ...     train_loader=train_loader,
    ...     valid_loader=valid_loader,
    ...     voc=voc,
    ...     epochs=20
    ... )
    >>> finetuned_model = finetuner.fit()
    """

    training_mode: str = 'FT'

    def __init__(
        self,
        args: Optional[Union[argparse.Namespace, Dict[str, Any], Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(args, **kwargs)
        self.training_mode = 'FT'
        self.unique_frags = False

    def fit(self) -> Model:
        """Execute fine-tuning workflow and return the fine-tuned generator model.

        Returns
        -------
        Model
            Fine-tuned DrugEx generator instance.
        """
        return self()

    def __call__(self) -> Model:
        """Execute fine-tuning workflow.

        Returns
        -------
        Model
            Fine-tuned DrugEx generator instance.
        """
        voc, train_loader, test_loader = self.prepareInputs()
        if self.agent is not None:
            agent = self.agent
        else:
            agent = self.setGeneratorAlgorithm(voc)
            if self.agent_path:
                agent = self.loadStatesFromFile(agent, self.agent_path)

        monitor = self.getMonitor()

        log.info('Finetuning generator...')
        fit_kwargs: Dict[str, Any] = {}
        if self.loss_tolerance is not None:
            fit_kwargs['loss_tolerance'] = self.loss_tolerance

        agent.fit(
            train_loader,
            test_loader,
            epochs=self.epochs,
            monitor=monitor,
            patience=self.patience,
            **fit_kwargs
        )
        return agent


class Reinforce(SetUpGenerator):
    """Train a DrugEx generator via Multi-Objective Reinforcement Learning (RL).

    Sets up the generator agent and exploration prior models, configures the
    multi-objective desirability environment (`DrugExEnvironment`), instantiates
    an explorer (`SequenceExplorer`, `FragSequenceExplorer`, or `FragGraphExplorer`),
    and optimizes the agent via policy gradients.

    Can be executed via CLI or programmatically in Python scripts.

    Parameters
    ----------
    args : Optional[Union[argparse.Namespace, Dict[str, Any], Any]], optional
        Parsed command line arguments or config dictionary.
    **kwargs : Any
        Direct keyword arguments configuring agent, prior, environment, and explorer parameters.

    Examples
    --------
    >>> from drugex.train import Reinforce
    >>> # Reinforcement learning with QED, SAScore, and QSPR bioactivity
    >>> rl = Reinforce(
    ...     base_dir=".",
    ...     input="focused_dataset",
    ...     agent_path="generators/my_finetuned_agent.pkg",
    ...     prior_path="data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT.pkg",
    ...     mol_type="smiles",
    ...     algorithm="rnn",
    ...     scheme="PRCD",
    ...     qed=True,
    ...     sa_score=True,
    ...     epochs=100,
    ...     batch_size=128,
    ...     use_gpus=[0]
    ... )
    >>> trained_explorer = rl.fit()
    """

    training_mode: str = 'RL'

    def __init__(
        self,
        args: Optional[Union[argparse.Namespace, Dict[str, Any], Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(args, **kwargs)
        self.training_mode = 'RL'
        self.unique_frags = True

    def setExplorer(self, agent: Model, prior: Model, env: DrugExEnvironment) -> Explorer:
        """Instantiate the explorer algorithm based on molecule type and generator architecture.

        Parameters
        ----------
        agent : Model
            DrugEx generator to optimize during RL.
        prior : Model
            DrugEx generator providing exploration mutation baseline.
        env : DrugExEnvironment
            Environment computing multi-objective desirability scores.

        Returns
        -------
        Explorer
            Configured explorer instance (SequenceExplorer, FragSequenceExplorer, or FragGraphExplorer).
        """
        kwargs = {
            'mutate': prior,
            'batch_size': self.batch_size,
            'epsilon': self.epsilon,
            'beta': self.beta,
            'n_samples': self.n_samples,
            'use_gpus': self.use_gpus
        }

        if self.mol_type == 'graph' and self.algorithm == 'trans':
            explorer = FragGraphExplorer(agent, env, **kwargs)        
        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            explorer = FragSequenceExplorer(agent, env, **kwargs)
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            explorer = SequenceExplorer(agent, env, **kwargs)
        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return explorer        

    def fit(self) -> Explorer:
        """Execute reinforcement learning and return the trained explorer instance.

        Returns
        -------
        Explorer
            Trained DrugEx explorer instance containing the optimized agent.
        """
        return self()

    def __call__(self) -> Explorer:
        """Train generator with reinforcement learning.

        Returns
        -------
        Explorer
            Trained DrugEx explorer instance containing the optimized agent.
        """
        voc, train_loader, test_loader = self.prepareInputs()

        # Set agent algorithm and load pretrained weights
        if self.agent is not None:
            agent = self.agent
        else:
            agent = self.setGeneratorAlgorithm(voc)
            if self.agent_path:
                agent = self.loadStatesFromFile(agent, self.agent_path)

        # Set prior algorithm and load pretrained weights
        if self.prior is not None:
            prior = self.prior
        else:
            prior = self.setGeneratorAlgorithm(voc)
            if self.prior_path:
                prior = self.loadStatesFromFile(prior, self.prior_path)

        # Set environment
        if self.environment is not None:
            environment = self.environment
        else:
            environment = CreateEnvironment(
                self.base_dir,
                self.predictor,
                self.scheme,
                active_targets=self.active_targets,
                inactive_targets=self.inactive_targets,
                window_targets=self.window_targets,
                activity_threshold=self.activity_threshold,
                qed=self.qed,
                unique=self.uniqueness, 
                sa_score=self.sa_score,
                ra_score=self.ra_score,
                mw=self.molecular_weight,
                mw_ths=self.mw_thresholds,
                logP=self.logP,
                logP_ths=self.logP_thresholds,
                tpsa=self.tpsa,
                tpsa_ths=self.tpsa_thresholds,
                sim_smiles=self.similarity_mol,
                sim_type=self.similarity_type,
                sim_th=self.similarity_threshold,
                sim_tw=self.similarity_tversky_weights,
                le=self.ligand_efficiency,
                le_ths=self.le_thresholds,
                lipe=self.lipophilic_efficiency,
                lipe_ths=self.lipe_thresholds,
                logger=log
            )

        if not environment:
            raise ValueError('No environment specified. Please, make sure you specified at least one objective function.')

        # Set evolver
        evolver = self.setExplorer(agent, prior, environment)

        # Set monitoring
        monitor = self.getMonitor()

        # Fit
        log.info('Training generator with reinforcement learning...')
        evolver.fit(train_loader, test_loader, epochs=self.epochs, monitor=monitor, patience=self.patience)
        return evolver

def getModifiers(
    task: Union[ModelTasks, str],
    scheme: str,
    activity_threshold: float
) -> Tuple[ClippedScore, ClippedScore, ClippedScore]:
    """Get objective score modifiers for a given QSPR task, reward scheme, and activity threshold.

    Parameters
    ----------
    task : Union[ModelTasks, str]
        Target prediction task (classification or regression).
    scheme : str
        Multi-objective scheme: 'WS' (Weighted Sum) or Pareto ('PRTD', 'PRCD').
    activity_threshold : float
        Activity threshold used for desirability scaling.

    Returns
    -------
    Tuple[ClippedScore, ClippedScore, ClippedScore]
        Active target modifier, inactive target modifier, and selectivity window modifier.
    """
    pad = 3.5
    pad_window = 1.5
    is_classification = (
        (hasattr(task, 'isClassification') and task.isClassification()) or 
        (hasattr(ModelTasks, 'SINGLECLASS') and task in (ModelTasks.SINGLECLASS, getattr(ModelTasks, 'MULTICLASS', None))) or
        str(task).upper() in ('CLS', 'CLASSIFICATION', 'SINGLECLASS', 'MULTICLASS')
    )

    if scheme == 'WS':
        # Weighted Sum (WS) reward scheme
        if is_classification:
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold + pad)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold - pad)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)

    else:
        # Pareto Front reward scheme (PRTD or PRCD)
        if is_classification:
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)

    return active, inactive, window
    

def CreateEnvironment(
    base_dir: str = '.', 
    predictor: Optional[List[str]] = None,
    scheme: str = 'PRCD', 
    active_targets: Optional[List[str]] = None, 
    inactive_targets: Optional[List[str]] = None, 
    window_targets: Optional[List[str]] = None,
    activity_threshold: float = 6.5, 
    qed: bool = False, 
    unique: bool = False,
    sa_score: bool = False,
    ra_score: bool = False, 
    mw: bool = False,
    mw_ths: Optional[List[int]] = None,
    logP: bool = False,
    logP_ths: Optional[List[float]] = None,
    tpsa: bool = False,
    tpsa_ths: Optional[List[float]] = None,
    sim_smiles: Optional[str] = None,
    sim_type: str = 'ECFP6',
    sim_th: float = 0.6,
    sim_tw: Optional[List[float]] = None,
    le: bool = False,
    le_ths: Optional[List[float]] = None,
    lipe: bool = False,
    lipe_ths: Optional[List[float]] = None,
    logger: Optional[Any] = None
) -> Optional[DrugExEnvironment]:
    """Create and configure a multi-objective DrugExEnvironment for reinforcement learning.

    Instantiates QSPR predictor models, physicochemical property scorers (QED, SA score,
    RA score, MW, logP, TPSA), similarity objectives (Fraggle, Tversky fingerprint/graph),
    and efficiency metrics (Ligand Efficiency, Lipophilic Efficiency), mapping each to a
    desirability modifier and combining them under the chosen multi-objective scheme.

    Parameters
    ----------
    base_dir : str, optional
        Project directory containing models and data (default: '.').
    predictor : Optional[List[str]], optional
        List of paths to QSPRPred model metadata JSON files relative to base_dir.
    scheme : str, optional
        Multi-objective optimization scheme: 'PRCD' (Pareto Crowding Distance),
        'PRTD' (Pareto Tanimoto Distance), or 'WS' (Weighted Sum) (default: 'PRCD').
    active_targets : Optional[List[str]], optional
        Target IDs for which high activity is desired.
    inactive_targets : Optional[List[str]], optional
        Target IDs for which low activity (inactivity) is desired.
    window_targets : Optional[List[str]], optional
        Target IDs for selectivity window optimization.
    activity_threshold : float, optional
        Activity cutoff threshold used by score modifiers (default: 6.5).
    qed : bool, optional
        Include Quantitative Estimate of Drug-likeness (QED) (default: False).
    unique : bool, optional
        Include epoch molecule uniqueness penalty (default: False).
    sa_score : bool, optional
        Include Synthetic Accessibility (SA) score (default: False).
    ra_score : bool, optional
        Include Retrosynthetic Accessibility (RA) score (default: False).
    mw : bool, optional
        Penalize molecules outside molecular weight range (default: False).
    mw_ths : Optional[List[int]], optional
        Molecular weight lower and upper bounds (default: [200, 600]).
    logP : bool, optional
        Penalize molecules outside logP range (default: False).
    logP_ths : Optional[List[float]], optional
        logP lower and upper bounds (default: [-5.0, 5.0]).
    tpsa : bool, optional
        Penalize molecules outside TPSA range (default: False).
    tpsa_ths : Optional[List[float]], optional
        TPSA lower and upper bounds (default: [0.0, 140.0]).
    sim_smiles : Optional[str], optional
        Reference molecule SMILES for similarity scoring (default: None).
    sim_type : str, optional
        Similarity metric: 'fraggle', 'graph', or fingerprint type e.g. 'ECFP6' (default: 'ECFP6').
    sim_th : float, optional
        Desirability threshold for similarity (default: 0.6).
    sim_tw : Optional[List[float]], optional
        Tversky similarity weights [alpha, beta] (default: [1.0, 1.0]).
    le : bool, optional
        Use Ligand Efficiency (LE) for active targets (default: False).
    le_ths : Optional[List[float]], optional
        Ligand Efficiency thresholds (default: [0.0, 1.0]).
    lipe : bool, optional
        Use Lipophilic Efficiency (LipE) for active targets (default: False).
    lipe_ths : Optional[List[float]], optional
        Lipophilic Efficiency thresholds (default: [4.0, 6.0]).
    logger : Optional[Any], optional
        Logger instance for logging environment creation (defaults to drugex logger).

    Returns
    -------
    Optional[DrugExEnvironment]
        Configured environment instance ready for RL explorer, or None if no objectives specified.

    Examples
    --------
    >>> from drugex.train import CreateEnvironment
    >>> env = CreateEnvironment(
    ...     scheme="PRCD",
    ...     qed=True,
    ...     sa_score=True,
    ...     mw=True,
    ...     mw_ths=[200, 500]
    ... )
    """
    logger = logger or log
    predictor = predictor or []
    active_targets = active_targets or []
    inactive_targets = inactive_targets or []
    window_targets = window_targets or []
    mw_ths = mw_ths or [200, 600]
    logP_ths = logP_ths or [-5.0, 5.0]
    tpsa_ths = tpsa_ths or [0.0, 140.0]
    sim_tw = sim_tw or [1.0, 1.0]
    le_ths = le_ths or [0.0, 1.0]
    lipe_ths = lipe_ths or [4.0, 6.0]

    # TODO: update cli documentation/tutorials to reflect new scheme abbreviations
    schemes = {
        "PRTD": ParetoTanimotoDistance(),
        "PRCD": ParetoCrowdingDistance(),
        "WS": WeightedSum()
    }
    objs = []
    ths = []
    targets = active_targets + inactive_targets + window_targets
    predictors = [QSPRModel.fromFile(os.path.join(base_dir, x)) for x in predictor]
    predictors = {x.name: x for x in predictors}
    assert len(predictors) == len(targets) and all([x in targets for x in predictors]), f'Predictors do not match targets: {predictors} != {targets}'
    for target in targets:
        model = predictors[target]
        task = model.task
        if task == ModelTasks.MULTICLASS:
            raise NotImplementedError('Classification models with more than 2 classes are not supported. Invalid model: {}'.format(target))

        active, inactive, window = getModifiers(task, scheme, activity_threshold)
        scorer = QSPRPredScorer(model)
        if target in active_targets:
            scorer.setModifier(active)
            if le or lipe:
                if task == ModelTasks.CLASSIFICATION:
                    raise NotImplementedError('Ligand efficiency and lipophilic efficiency are only available for regression tasks')
                if le:
                    objs.append(LigandEfficiency(qsar_scorer=scorer,
                                modifier=ClippedScore(lower_x=le_ths[0], upper_x=le_ths[1])))
                    ths.append(0.5)
                if lipe:
                    objs.append(LipophilicEfficiency(qsar_scorer=scorer,
                                modifier=ClippedScore(lower_x=lipe_ths[0], upper_x=lipe_ths[1])))
                    ths.append(0.5)
            else:
                objs.append(scorer)
                ths.append(0.5 if scheme == 'WS' else 0.99)
        elif target in inactive_targets:
            scorer.setModifier(inactive)
            objs.append(scorer)
            ths.append(0.5 if scheme == 'WS' else 0.99)
        elif target in window_targets:
            scorer.setModifier(window)
            objs.append(scorer)
            ths.append(0.5)
        else:
            raise ValueError('Target {} not found in active, inactive or window targets'.format(target))

    if qed:
        objs.append(Property('QED', modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.5)
    if unique:
        objs.append(Uniqueness(modifier=ClippedScore(lower_x=1.0, upper_x=0.0)))
        ths.append(0.0)
    if sa_score:
        objs.append(Property('SA', modifier=ClippedScore(lower_x=10, upper_x=1.0)))
        ths.append(0.5)
    if ra_score:
        from drugex.training.scorers.ra_scorer import \
            RetrosyntheticAccessibilityScorer
        objs.append(RetrosyntheticAccessibilityScorer(modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.0)
    if mw:
        objs.append(Property('MW', modifier=SmoothHump(lower_x=mw_ths[0], upper_x=mw_ths[1], sigma=100)))
        ths.append(0.99)
    if logP:
        objs.append(Property('logP', modifier=SmoothHump(lower_x=logP_ths[0], upper_x=logP_ths[1], sigma=1)))
        ths.append(0.99)
    if tpsa:
        objs.append(Property('TPSA', modifier=SmoothHump(lower_x=tpsa_ths[0], upper_x=tpsa_ths[1], sigma=10)))
        ths.append(0.99)
    if sim_smiles:
        if sim_type == 'fraggle':
            objs.append(FraggleSimilarity(sim_smiles))
        elif sim_type == 'graph':
            objs.append(TverskyGraphSimilarity(sim_smiles, sim_tw[0], sim_tw[1]))
        else:
            objs.append(TverskyFingerprintSimilarity(sim_smiles, sim_type, sim_tw[0], sim_tw[1]))
        ths.append(sim_th)

    logger.info('DrugExEnvironment created using {} objectives: {}'.format(len(objs), [o.getKey() for o in objs]))

    return DrugExEnvironment(objs, ths, schemes[scheme]) if objs else None
    
if __name__ == "__main__":
    
    # Parse command line arguments
    args = GeneratorArgParser()

    # Create backup of files
    backup_msg = backUpFiles(args.base_dir, 'generators', (args.output_long,))
    
    # Create directory for generators
    if not os.path.exists(f'{args.base_dir}/generators'):
        os.mkdir(f'{args.base_dir}/generators')

    # Create log file
    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'generators'),
        args.output_long + '.log',
        args.debug,
        __name__,
        vars(args)
    )
    log = logSettings.log
    log.info(backup_msg)

    # Save arguments to json file
    with open(f'{args.output_file_base}.json', 'w') as f:
        json.dump(vars(args), f)

    # Train generator
    if args.training_mode == 'PT':
        Pretrain(args)()
    elif args.training_mode == 'FT':
        Finetune(args)()
    elif args.training_mode == 'RL':
        Reinforce(args)()
    else:
        raise ValueError(f"--mode should be either 'PT', 'FT' or 'RL', you gave {args.training_mode}")
