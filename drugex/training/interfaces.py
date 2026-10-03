from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Literal, Union

# [PACKAGE: numpy] WHY: Array manipulations and reward vector calculations. WHAT: Numerical computing library.
import numpy as np
# [PACKAGE: torch] WHY: Tensor computing, GPU device abstraction, and neural net modules. WHAT: PyTorch deep learning framework.
import torch
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.logs import logger
from torch import nn


class ModelEvaluator(ABC):
    """Abstract scoring evaluator evaluating generated molecules and input fragments."""

    @abstractmethod
    def __call__(self, mols, frags=None):
        """Score generated molecules and optional input fragments into tabular evaluations."""
        pass




class RewardScheme(ABC):
    """Abstract reward scheme calculating scalar rewards and ranking solutions from multi-objective scores."""


    class RewardException(Exception):
        """Exception raised during reward calculation or solution ranking errors."""
        pass



    @abstractmethod
    def __call__(self, smiles, scores, thresholds):
        """Compute scalar reinforcement learning rewards and ranks from multi-objective evaluation matrix."""
        pass

class Environment(ModelEvaluator):
    """Reinforcement learning environment coordinating objective scorers, thresholds, and reward schemes."""

    def __init__(self, scorers, thresholds=None, reward_scheme=None):
        """Initialize environment with objective scorers, desirability thresholds, and reward scheme."""

        self.scorers = scorers
        num_scorers = len(self.getScorerKeys()) if isinstance(self.scorers, list) else 1
        self.thresholds = thresholds if thresholds is not None else [0.99] * num_scorers
        assert num_scorers == len(self.thresholds), f"Number of scorers ({num_scorers}) and thresholds ({len(self.thresholds)}) do not match."
        self.rewardScheme = reward_scheme

    def __call__(self, smiles, frags=None):
        """Score candidate molecules against the environment objectives."""
        return self.getScores(smiles, frags)

    @abstractmethod
    def getScores(self, smiles, frags=None, no_multifrag_smiles=True):
        """Calculate multi-objective scores and candidate molecule validity against thresholds."""
        pass

    @abstractmethod
    def getUnmodifiedScores(self, smiles):
        """Calculate raw objective scores without post-processing modifiers."""
        pass

    def getRewards(self, smiles, frags=None):
        """Compute reinforcement learning scalar rewards using the configured reward scheme."""

        # Get scores
        scores = self.getScores(smiles, frags=frags)
        scores['SMILES'] = smiles  
        self.scores = scores

        # Initialize rewards to 0
        rewards = np.zeros((len(smiles),1))
        valid_idx = scores[scores.Valid == 1].index.tolist()
        # Compute rewards for valid molecules
        if len(valid_idx) > 0:
            smiles_valid = scores[scores.Valid == 1].SMILES.tolist()
            scores_valid = scores.loc[scores.Valid == 1, self.getScorerKeys()].values
            rewards_valid = self.rewardScheme(smiles_valid, scores_valid, self.thresholds)
            rewards[valid_idx] = rewards_valid
        else:
            logger.warning("No valid molecules generated. All rewards are 0.")

        return rewards

    def getScorerKeys(self):
        """Return list of identifier keys across all configured scorers."""
        keys = []
        for scorer in self.scorers:
            scorer_keys = scorer.getKey()
            if isinstance(scorer_keys, list):
                keys.extend(scorer_keys)
            else:
                keys.append(scorer_keys)
        return keys

class ModelProvider(ABC):
    """Interface for components providing or serializing DrugEx model state."""

    @abstractmethod
    def getModel(self):
        """Return the current model instance or serialized state dictionary."""
        pass

class Model(nn.Module, ModelProvider, ABC):
    """Base PyTorch model managing device placement, GPU allocations, and state persistence."""

    def __init__(self, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS):
        """Initialize model with execution device and allocated GPU identifiers."""
        super().__init__()

        self.device = None
        self.gpus = None
        self.updateDevices(device, use_gpus)

    def updateDevices(self, device, gpus):
        """Update execution device and GPU allocations for the model."""

        if device.type == 'cpu':
            self.device = torch.device('cpu')
            self.gpus = (-1,)
        elif device.type == 'cuda':
            self.device = torch.device(f'cuda:{gpus[0]}')
            self.attachToGPUs(gpus)
        else:
            raise ValueError(f"Unknown device: {device}")

    @abstractmethod
    def attachToGPUs(self, gpus):
        """Transfer model parameters and state to target GPU device allocations."""
        pass

    @abstractmethod
    def fit(self, train_loader, valid_loader, epochs=1000, monitor=None, **kwargs):
        """Train and validate model with train/validation data loaders and optional training monitor."""
        pass

    def loadStatesFromFile(self, path: Union[str, os.PathLike[str], Path]) -> None:
        """Load serialized model weights from file onto the current device."""
        self.loadStates(torch.load(path, map_location=self.device, weights_only=True))

    def loadStates(self, state_dict, strict=True):
        """Load model weights from state dictionary."""
        self.load_state_dict(state_dict, strict=strict)

class TrainingMonitor(ModelProvider, ABC):
    """Interface for monitoring model training progress, performance metrics, and checkpoints."""

    @abstractmethod
    def saveModel(self, model, identifier=None):
        """Serialize and persist model state dictionary with optional checkpoint suffix."""
        pass

    @abstractmethod
    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        """Record epoch performance metrics and optional generated SMILES table."""
        pass

    @abstractmethod
    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        """Notify monitor of current training step, epoch, and progress counters."""
        pass

    @abstractmethod
    def endStep(self, step, epoch):
        """Signal completion of a single training step/batch within an epoch."""
        pass

    @abstractmethod
    def close(self):
        """Finalize and close monitor resources upon training completion."""
        pass

    @abstractmethod
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        """Return checkpoint saving scheme ('best', 'all', or 'improvement')."""
        pass