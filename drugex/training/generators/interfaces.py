"""
Base interfaces and abstract classes for molecule generators in DrugEx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generator as PyGenerator, Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from drugex.logs import logger
from drugex.training.interfaces import Model, TrainingMonitor
from drugex.training.monitors import NullMonitor
from rdkit import Chem
from drugex.training.scorers.smiles import SmilesChecker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from drugex.data.interfaces import DataSet
    from drugex.training.environment import Environment


class Generator(Model, ABC):
    """
    Abstract base class for all molecule generators in DrugEx.

    Defines the unified lifecycle contract for generative architectures, including
    sampling, single-epoch training, multi-metric validation, molecule filtering,
    and objective-based evaluation.

    Inherits from `drugex.training.interfaces.Model` and `torch.nn.Module`.
    """

    @abstractmethod
    def sample(self, *args, **kwargs) -> tuple[list[str], Optional[list[str]]] | list[str]:
        """
        Sample molecules from the generator.

        Parameters
        ----------
        *args : Any
            Architecture-specific positional sampling arguments (e.g. `batch_size` or `loader`).
        **kwargs : Any
            Architecture-specific keyword arguments.

        Returns
        -------
        smiles : list of str
            Sampled SMILES representations of generated compounds.
        frags : list of str, optional
            Input fragments corresponding to each generated molecule (for fragment-based models).
        """
        pass

    @abstractmethod
    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Train the generator network for a single epoch.

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader providing training batches of encoded sequences or graphs.
        epoch : int
            Current epoch index (1-indexed).
        epochs : int
            Total planned training epochs.

        Returns
        -------
        loss : float
            Representative training loss for the current epoch.
        """
        pass

    @abstractmethod
    def validateNet(
        self,
        loader: Optional[DataLoader] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        n_samples: Optional[int] = None
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Validate generator performance, compute validation loss, and evaluate sample quality.

        Parameters
        ----------
        loader : DataLoader, optional
            Validation DataLoader to calculate validation set likelihood or loss.
        evaluator : Environment, optional
            Reinforcement learning environment or model evaluator providing property scores.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES containing '.' are classified invalid
            (default: True).
        n_samples : int, optional
            Number of molecules to sample for statistical quality checks (used by sequence models).

        Returns
        -------
        valid_metrics : dict of str to float
            Validation metrics dictionary (e.g. `'valid_ratio'`, `'accurate_ratio'`, `'loss_valid'`).
        smiles_scores : pd.DataFrame
            DataFrame containing sampled SMILES, validity indicators, and property scores.
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.DataFrame:
        """
        Generate molecules until target quota is satisfied, applying filtering and scoring.

        Returns
        -------
        df_smiles : pd.DataFrame
            DataFrame containing generated SMILES, optional input fragments, and objective scores.
        """
        pass

    def filterNewMolecules(
        self,
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        with_frags: bool = True,
        drop_duplicates: bool = True,
        drop_undesired: bool = True,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True
    ) -> pd.DataFrame:
        """
        Filter newly generated molecules against cumulative history and quality criteria.

        Performs chemical validity verification, canonicalization, deduplication against
        both current batch and historical generation pool, fragment consistency checks,
        and threshold filtering via the provided evaluator.

        Parameters
        ----------
        df_old : pd.DataFrame
            DataFrame of previously accepted molecules. Must contain `'SMILES'` column.
        df_new : pd.DataFrame
            DataFrame of newly sampled molecules to filter. Must contain `'SMILES'` column.
        with_frags : bool, optional
            If True, expects `'Frags'` column and verifies fragment presence in generated compounds
            (default: True).
        drop_duplicates : bool, optional
            If True, removes duplicate SMILES existing within `df_new` or already present in `df_old`
            (default: True).
        drop_undesired : bool, optional
            If True, filters out molecules that fail the minimum desirability thresholds of `evaluator`.
            Requires `evaluator` to be non-None (default: True).
        evaluator : Environment, optional
            Reinforcement learning environment or scoring instance used to evaluate desirability.
        no_multifrag_smiles : bool, optional
            If True, only single-fragment connected SMILES are considered valid (default: True).

        Returns
        -------
        df_filtered : pd.DataFrame
            Filtered subset of `df_new` retaining only valid, unique, and desirable molecules.
        """
        # Make sure both valid molecules and include input fragments if needed
        use_frags = with_frags and ('Frags' in df_new.columns and df_new['Frags'].notna().any())
        scores = SmilesChecker.checkSmiles(
            df_new.SMILES.tolist(),
            frags=df_new.Frags.tolist() if use_frags else None,
            no_multifrag_smiles=no_multifrag_smiles
        )
        df_new = pd.concat([df_new, scores], axis=1)

        if use_frags:
            df_new = df_new[df_new.Accurate == 1].reset_index(drop=True)
        else:
            df_new = df_new[df_new.Valid == 1].reset_index(drop=True)

        # Canonicalize SMILES
        canonical = []
        for s in df_new.SMILES:
            mol = Chem.MolFromSmiles(s) if s else None
            canonical.append(Chem.MolToSmiles(mol) if mol is not None else s)
        df_new['SMILES'] = canonical

        # Drop duplicates
        if drop_duplicates:
            df_new = df_new.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
            df_new = df_new[~df_new.SMILES.isin(df_old.SMILES)].reset_index(drop=True)

        # Score molecules with evaluator if provided
        if evaluator:
            # Compute desirability scores
            scores = self.evaluate(
                df_new.SMILES.tolist(),
                frags=df_new.Frags.tolist() if use_frags else None,
                evaluator=evaluator,
                no_multifrag_smiles=no_multifrag_smiles
            )
            df_new['Desired'] = scores['Desired'].values

            # Drop undesired molecules
            if drop_undesired:
                df_new = df_new[df_new.Desired == 1].reset_index(drop=True)
        elif drop_undesired:
            raise ValueError("Evaluator must be provided if 'drop_undesired' is True")

        return df_new

    def evaluate(
        self,
        smiles: list[str],
        frags: Optional[list[str]] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        unmodified_scores: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate molecules for chemical validity and property scores.

        If `evaluator` is omitted, performs structural validity and fragment containment
        verification via `SmilesChecker`. If `evaluator` is provided, calculates full
        multi-objective reward scores across target properties.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings to evaluate.
        frags : list of str, optional
            List of input fragments corresponding to each SMILES (for fragment-constrained models).
        evaluator : Environment, optional
            Environment instance scoring molecules across bioactivity, QSAR, and physicochemical objectives.
        no_multifrag_smiles : bool, optional
            If True, disconnected multi-fragment SMILES are penalized as invalid (default: True).
        unmodified_scores : bool, optional
            If True, calculates raw scores before non-linear score modifier transformations (default: False).

        Returns
        -------
        scores : pd.DataFrame
            DataFrame indexed by molecule, containing validity indicators and score columns.
        """
        if evaluator is None:
            scores = SmilesChecker.checkSmiles(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)
        else:
            if unmodified_scores:
                scores = evaluator.getUnmodifiedScores(smiles)
            else:
                scores = evaluator.getScores(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)

        return scores

    def logPerformanceAndCompounds(
        self,
        epoch: int,
        metrics: dict[str, Any],
        scores: pd.DataFrame
    ) -> None:
        """Log performance metrics and generated molecules to the monitor.

        Parameters
        ----------
        epoch : int
            The current epoch index.
        metrics : dict of str to Any
            Dictionary containing performance metrics.
        scores : pd.DataFrame
            DataFrame with generated molecules, validity flags, and objective scores.
        """
        # Add epoch to metrics and order columns
        metrics['Epoch'] = epoch
        metrics = {
            k: metrics[k]
            for k in ['Epoch', 'loss_train', 'loss_valid', 'valid_ratio', 'accurate_ratio', 'best_epoch']
            if k in metrics.keys()
        }

        # Add epoch to scores and order columns
        scores['Epoch'] = epoch
        if 'Frags' in scores.columns:
            first_cols = ['Epoch', 'SMILES', 'Frags', 'Valid', 'Accurate']
        else:
            first_cols = ['Epoch', 'SMILES', 'Valid']
        scores = pd.concat([scores[first_cols], scores.drop(first_cols, axis=1)], axis=1)

        # Save performance info and generate smiles
        self.monitor.savePerformanceInfo(metrics, df_smiles=scores)
        self.monitor.endStep(None, epoch)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        patience: int = 50,
        evaluator: Optional[Environment] = None,
        monitor: Optional[TrainingMonitor] = None,
        no_multifrag_smiles: bool = True,
        loss_tolerance: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Train and validate the generator network over multiple epochs with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing training batches.
        valid_loader : DataLoader, optional
            DataLoader providing hold-out validation batches.
        epochs : int, optional
            Maximum number of training epochs (default: 100).
        patience : int, optional
            Number of epochs to wait without improvement before stopping early (default: 50).
        evaluator : Environment, optional
            Environment or ModelEvaluator scoring compound quality during validation.
        monitor : TrainingMonitor, optional
            Monitor instance logging progress, metrics, and checkpoints (default: NullMonitor).
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / salt SMILES are penalized as invalid (default: True).
        loss_tolerance : float, optional
            Allowed increase in validation loss if valid_ratio improves.
        **kwargs : Any
            Additional keyword arguments for compatibility with Model.fit.
        """
        self.monitor = monitor if monitor else NullMonitor()
        best = float('inf')
        last_save = -1
        valid_ratio_best = 0.0

        for epoch in tqdm(range(epochs), desc='Fitting model'):
            epoch += 1
            is_best = False

            # Train model
            loss_train = self.trainNet(train_loader, epoch, epochs)

            # Validate model
            valid_metrics, smiles_scores = self.validateNet(
                loader=valid_loader,
                evaluator=evaluator,
                no_multifrag_smiles=no_multifrag_smiles,
                n_samples=train_loader.batch_size * 2
            )

            # Determine best model based on validation loss or valid ratio
            if 'loss_valid' in valid_metrics.keys():
                value = valid_metrics['loss_valid']
            else:
                value = 1 - valid_metrics['valid_ratio']
            valid_metrics['loss_train'] = loss_train

            if loss_tolerance is None or 'loss_valid' not in valid_metrics.keys():
                if value < best:
                    is_best = True
                    best, last_save = value, epoch
                valid_metrics['best_epoch'] = last_save
            else:
                # allow a small increase in validation loss if valid ratio improves
                if value < best:
                    is_best = True
                    best, last_save = value, epoch
                    valid_ratio_best = valid_metrics['valid_ratio']
                elif value <= best + loss_tolerance:
                    if valid_metrics['valid_ratio'] > valid_ratio_best:
                        is_best = True
                        valid_ratio_best = valid_metrics['valid_ratio']
                        last_save = epoch
                valid_metrics['best_epoch'] = last_save

            # Save model
            save_model_option = self.monitor.getSaveModelOption()
            if save_model_option == 'all' or is_best:
                self.monitor.saveModel(self, epoch if save_model_option in ('all', 'improvement') else None)
                logger.info(f"Model was saved at epoch {epoch}")

            # Log performance and generated compounds
            self.logPerformanceAndCompounds(epoch, valid_metrics, smiles_scores)

            del loss_train, valid_metrics, smiles_scores

            # Early stopping
            if epoch - last_save > patience:
                break

        torch.cuda.empty_cache()
        self.monitor.close()

    def getModel(self) -> dict[str, Any]:
        """
        Return an in-memory deep copy of the model state dictionary.

        Returns
        -------
        model_state : dict of str to Any
            PyTorch state dictionary containing model weights and parameters.
        """
        return deepcopy(self.state_dict())


class FragGenerator(Generator):
    """
    Abstract base class for fragment-constrained molecule generators.

    Extends `Generator` to support fragment-based de novo design, where molecules
    are elaborated, connected, or scaffold-hopped starting from pre-defined chemical moieties.
    """

    def init_states(self) -> None:
        """
        Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to all multi-dimensional weight tensors
        (excluding 1D biases and embedding layers) and attaches the model to assigned GPUs.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)

    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """
        Attach model to specified GPU device IDs and transfer model tensors.

        Parameters
        ----------
        gpus : sequence of int
            Sequence or tuple of GPU hardware device IDs.
        """
        self.gpus = tuple(gpus)
        self.to(self.device)

    @abstractmethod
    def loaderFromFrags(
        self,
        frags: list[str],
        batch_size: int = 32,
        n_proc: int = 1
    ) -> DataLoader:
        """
        Encode input fragments and construct an iterable DataLoader.

        Parameters
        ----------
        frags : list of str
            List of input fragments in SMILES format.
        batch_size : int, optional
            Batch size for DataLoader batches (default: 32).
        n_proc : int, optional
            Worker processes for parallel fragment encoding (default: 1).

        Returns
        -------
        loader : DataLoader
            PyTorch DataLoader yielding encoded fragment tensors.
        """
        pass

    @abstractmethod
    def decodeLoaders(
        self,
        src: torch.Tensor,
        trg: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        """
        Decode source fragment tensors and generated target tensors into SMILES strings.

        Parameters
        ----------
        src : torch.Tensor
            Batch of encoded input fragments.
        trg : torch.Tensor
            Batch of encoded generated molecules.

        Returns
        -------
        frags : list of str
            Decoded input fragment SMILES.
        smiles : list of str
            Decoded generated molecule SMILES.
        """
        pass

    @abstractmethod
    def iterLoader(self, loader: DataLoader) -> PyGenerator[torch.Tensor, None, None]:
        """
        Yield source tensor batches from the fragment DataLoader.

        Parameters
        ----------
        loader : DataLoader
            Input DataLoader.

        Yields
        ------
        src : torch.Tensor
            Batch of input fragment tensors.
        """
        pass

    def generate(
        self,
        input_frags: Optional[list[str]] = None,
        input_dataset: Optional[DataSet] = None,
        num_samples: int = 100,
        batch_size: int = 32,
        n_proc: int = 1,
        keep_frags: bool = True,
        drop_duplicates: bool = True,
        drop_invalid: bool = True,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        drop_undesired: bool = False,
        raw_scores: bool = True,
        progress: bool = True,
        tqdm_kwargs: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate molecules starting from input fragments or a dataset.

        Iteratively generates molecules by elaborating or linking input fragments until
        `num_samples` valid compounds are collected.

        Parameters
        ----------
        input_frags : list of str, optional
            List of input fragment SMILES to incorporate into generated compounds.
            Mutually exclusive with `input_dataset`.
        input_dataset : DataSet, optional
            Pre-encoded fragment dataset (e.g. `SmilesFragDataSet` or `GraphFragDataSet`).
            Mutually exclusive with `input_frags`.
        num_samples : int, optional
            Total target count of valid molecules to generate (default: 100).
        batch_size : int, optional
            Batch size per generation iteration (default: 32).
        n_proc : int, optional
            Processes used for encoding if `input_frags` is provided (default: 1).
        keep_frags : bool, optional
            If True, retains `'Frags'` column in the output DataFrame (default: True).
        drop_duplicates : bool, optional
            If True, duplicate molecules are discarded (default: True).
        drop_invalid : bool, optional
            If True, chemically invalid molecules are excluded (default: True).
        evaluator : Environment, optional
            Scoring environment evaluating generated compounds.
        no_multifrag_smiles : bool, optional
            If True, multi-fragment / disconnected SMILES are marked invalid (default: True).
        drop_undesired : bool, optional
            If True, compounds failing evaluator desirability criteria are discarded (default: False).
        raw_scores : bool, optional
            If True, includes raw unpenalized scores from scorers (default: True).
        progress : bool, optional
            If True, displays a tqdm progress bar tracking generation (default: True).
        tqdm_kwargs : dict, optional
            Additional keyword arguments passed to `tqdm` (default: None).

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing generated compounds with columns `['SMILES', 'Frags']`
            and scorer property scores if evaluator was specified.
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        if input_dataset and input_frags:
            raise ValueError('Only one of input_dataset and input_frags can be provided')
        elif not input_dataset and not input_frags:
            raise ValueError('Either input_loader or input_frags must be provided')
        elif input_frags:
            loader = self.loaderFromFrags(input_frags, batch_size=batch_size, n_proc=n_proc)
        else:
            loader = input_dataset.asDataLoader(batch_size)

        net = nn.DataParallel(self, device_ids=self.gpus)

        if progress:
            tqdm_kwargs.update({'total': num_samples, 'desc': 'Generating molecules'})
            pbar = tqdm(**tqdm_kwargs)

        df_all = pd.DataFrame(columns=['SMILES', 'Frags'])
        while not len(df_all) >= num_samples:
            with torch.no_grad():
                for src in self.iterLoader(loader):
                    trg = net(src.to(self.device))
                    new_frags, new_smiles = self.decodeLoaders(src, trg)
                    df_new = pd.DataFrame({'SMILES': new_smiles, 'Frags': new_frags})

                    if drop_invalid:
                        df_new = self.filterNewMolecules(
                            df_all,
                            df_new,
                            drop_duplicates=drop_duplicates,
                            drop_undesired=drop_undesired,
                            evaluator=evaluator,
                            no_multifrag_smiles=no_multifrag_smiles
                        )

                    df_all = pd.concat([df_all, df_new], axis=0, ignore_index=True)

                    if progress:
                        pbar.update(len(df_new) if pbar.n + len(df_new) <= num_samples else num_samples - pbar.n)

                    if len(df_all) >= num_samples:
                        break

        if progress:
            pbar.close()

        df = df_all.head(num_samples)

        if evaluator:
            df = pd.concat([
                df,
                self.evaluate(
                    df.SMILES.tolist(),
                    frags=df.Frags.tolist(),
                    evaluator=evaluator,
                    no_multifrag_smiles=no_multifrag_smiles,
                    unmodified_scores=raw_scores
                )[evaluator.getScorerKeys()]
            ], axis=1)

        if not keep_frags:
            df.drop('Frags', axis=1, inplace=True)

        return df.round(3)