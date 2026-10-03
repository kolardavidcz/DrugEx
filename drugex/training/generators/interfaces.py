"""Base interfaces and abstract classes for molecule generators in DrugEx."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generator as PyGenerator, Optional, Sequence

# [PACKAGE: pandas] WHY: Tabular structure outputs and generation scoring results. WHAT: Data manipulation library.
import pandas as pd
# [PACKAGE: torch] WHY: Tensor neural network models, parameters initialization, and multi-GPU execution. WHAT: PyTorch deep learning framework.
import torch
import torch.nn as nn
# [PACKAGE: tqdm] WHY: Generation progress reporting across sampling loops. WHAT: Progress bar utility.
from tqdm.auto import tqdm

from drugex.logs import logger
from drugex.training.interfaces import Model, TrainingMonitor
from drugex.training.monitors import NullMonitor
# [PACKAGE: rdkit] WHY: Chemical SMILES validation and canonicalization. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from drugex.training.scorers.smiles import SmilesChecker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from drugex.data.interfaces import DataSet
    from drugex.training.environment import Environment


class Generator(Model, ABC):
    """Base abstract interface defining training, validation, and generation contracts for DrugEx models."""

    @abstractmethod
    def sample(self, *args, **kwargs) -> tuple[list[str], Optional[list[str]]] | list[str]:
        """Sample candidate SMILES representations and optional condition fragments from the generator."""
        pass

    @abstractmethod
    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        """Execute a single training epoch over DataLoader batches and return epoch loss."""
        pass

    @abstractmethod
    def validateNet(
        self,
        loader: Optional[DataLoader] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        n_samples: Optional[int] = None
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """Evaluate generator performance, calculate validation loss, and measure sample quality metrics."""
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> pd.DataFrame:
        """Sample candidate molecules until requested sample quota is satisfied with filtering and scoring."""
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
        """Filter newly generated molecules against cumulative history, validity rules, and desirability thresholds."""
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
        """Evaluate candidate molecules for chemical validity and compute objective scores via environment."""
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
        """Record epoch validation metrics and generated molecule scores to monitor."""
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
        """Train and validate generator over multiple epochs with early stopping and monitor checkpointing."""
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
        """Return deep copy of current model state dictionary."""
        return deepcopy(self.state_dict())


class FragGenerator(Generator):
    """Abstract base class for fragment-constrained molecule generators.

    Generation is strictly fragment-conditioned via loaderFromFrags() or dummyMolsFromFragments(),
    and cannot sample ex nihilo from scratch without conditioning input fragments.
    """

    def init_states(self) -> None:
        """Initialize multi-dimensional model weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attachToGPUs(self.gpus)

    def attachToGPUs(self, gpus: Sequence[int]) -> None:
        """Attach model to specified GPU device IDs and transfer model tensors."""
        self.gpus = tuple(gpus)
        self.to(self.device)

    @abstractmethod
    def loaderFromFrags(
        self,
        frags: list[str],
        batch_size: int = 32,
        n_proc: int = 1
    ) -> DataLoader:
        """Encode input fragments and construct an iterable DataLoader."""
        pass

    @abstractmethod
    def decodeLoaders(
        self,
        src: torch.Tensor,
        trg: torch.Tensor
    ) -> tuple[list[str], list[str]]:
        """Decode source fragment tensors and generated target tensors into SMILES strings."""
        pass

    @abstractmethod
    def iterLoader(self, loader: DataLoader) -> PyGenerator[torch.Tensor, None, None]:
        """Yield source tensor batches from the fragment DataLoader."""
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
        """Iteratively generate molecules starting from input fragments until quota is fulfilled."""
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