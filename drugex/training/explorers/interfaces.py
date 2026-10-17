from abc import ABC, abstractmethod
from copy import deepcopy


import numpy as np

# [PACKAGE: scipy] WHY: Geometric mean calculation across multi-objective metrics. WHAT: Scientific computing library.
from scipy.stats import gmean

# [PACKAGE: torch] WHY: Tensor calculations, device placement, and neural network data parallelism. WHAT: PyTorch deep learning framework.
import torch
from torch import nn

# [PACKAGE: tqdm] WHY: Terminal progress bars for training epochs and iterations. WHAT: Progress bar utility.
from tqdm import tqdm

from drugex import DEFAULT_GPUS, DEFAULT_DEVICE
from drugex.logs import logger
from drugex.training.interfaces import Model
from drugex.training.monitors import NullMonitor


class Explorer(Model, ABC):
    """Reinforcement learning exploration framework optimizing DrugEx agent networks in objective environments."""

    def __init__(
        self,
        agent,
        env,
        mutate=None,
        crover=None,
        no_multifrag_smiles=True,
        batch_size=128,
        epsilon=0.1,
        beta=0.0,
        n_samples=-1,
        device=DEFAULT_DEVICE,
        use_gpus=DEFAULT_GPUS,
    ):
        """Initialize explorer with agent, environment, mutation/crossover networks, and RL hyperparameters."""

        super().__init__(device=device, use_gpus=use_gpus)
        self.agent = agent
        self.mutate = mutate
        self.crover = crover
        self.batchSize = batch_size
        self.epsilon = epsilon
        self.beta = beta
        self.env = env
        self.nSamples = n_samples
        self.no_multifrag_smiles = no_multifrag_smiles
        self.bestState = None
        self.best_value = 0
        self.last_save = -1
        self.last_iter = -1

    def attachToGPUs(self, gpus):
        """Delegate GPU device placement to agent, mutation, and crossover sub-networks."""
        if hasattr(self, "agent"):
            self.agent.attachToGPUs(gpus)
        if hasattr(self, "mutate") and self.mutate:
            self.mutate.attachToGPUs(gpus)
        if hasattr(self, "crover") and self.crover:
            self.crover.attachToGPUs(gpus)
        self.gpus = gpus

    def getNovelMoleculeMetrics(self, scores):
        """Compute aggregate generation statistics: validity, fragment accuracy, uniqueness, desirability, and means."""

        dct = {}
        ntot = len(scores)

        # Valid compounds
        valid = scores[scores.Valid == 1]
        dct["valid_ratio"] = len(valid) / ntot

        # Accurate compounds
        if "Accurate" in scores.columns:
            accurate = valid[valid.Accurate == 1]
            dct["accurate_ratio"] = len(accurate) / ntot
        else:
            accurate = valid

        # Unique compounds
        unique = accurate.drop_duplicates(subset="SMILES")
        dct["unique_ratio"] = len(unique) / ntot

        # Desired compounds
        dct["desired_ratio"] = unique.Desired.sum() / ntot

        # Average artithmetic and geometric mean score
        if len(unique) == 0:
            dct["avg_amean"] = 0.0
            dct["avg_gmean"] = 0.0
        else:
            dct["avg_amean"] = unique[self.env.getScorerKeys()].values.mean()
            dct["avg_gmean"] = unique[self.env.getScorerKeys()].apply(gmean, axis=1).mean()

        return dct

    def saveBestState(self, value, epoch, it):
        """Persist current top-performing agent weights and checkpoint metadata."""
        self.monitor.saveModel(self)
        self.bestState = self.getModel()
        self.best_value = value
        self.last_save = epoch
        self.last_iter = it

    def logPerformanceAndCompounds(self, epoch, metrics, scores):
        """Record epoch summary metrics and scored compound population to monitor."""
        # Add epoch and best epoch to metrics, and order columns
        metrics["Epoch"] = epoch
        metrics = {k: metrics[k] for k in ["Epoch"] + [k for k in metrics.keys() if k != "Epoch"]}
        metrics["best_epoch"] = self.last_save

        # Add epoch to scores, and order columns
        scores["Epoch"] = epoch
        if "Accurate" in scores.columns:
            scores = scores[["Epoch", "SMILES", "Frags", "Valid", "Accurate", "Desired"] + self.env.getScorerKeys()]
        else:
            scores = scores[["Epoch", "SMILES", "Valid", "Desired"] + self.env.getScorerKeys()]

        self.monitor.savePerformanceInfo(metrics, df_smiles=scores)
        self.monitor.endStep(None, epoch)

    @abstractmethod
    def fit(self, train_loader, valid_loader=None, epochs=1000, monitor=None):
        pass

    @abstractmethod
    def policy_gradient(self, *args, **kwargs):
        """Execute policy gradient training step."""
        pass

    def getModel(self):
        """Return deep copy of current agent neural network weights."""
        return deepcopy(self.agent.state_dict())


class FragExplorer(Explorer):
    """Reinforcement learning explorer specialized for fragment-based molecular generation architectures."""

    @abstractmethod
    def getBatchOutputs(self, src, net):
        """Decode batch fragments and SMILES, and compute agent training loss."""
        pass

    def policy_gradient(self, loader, epoch=None, epochs=None):
        """Optimize agent via REINFORCE policy gradients computed over sampled fragment pairs."""
        net = nn.DataParallel(self.agent, device_ids=self.gpus)
        total_steps = len(loader)

        for step_idx, src in enumerate(tqdm(loader, desc="Calculating policy gradient...", leave=False)):
            # Decode fragments and smiles, and get loss
            frags, smiles, loss = self.getBatchOutputs(net, src)

            # Get rewards
            reward = self.env.getRewards(smiles, frags=frags)

            # Filter out molecules with multiple fragments by setting reward to 0
            if self.no_multifrag_smiles:
                reward = np.array([r if s.count(".") == 0 else [0] for s, r in zip(smiles, reward)])
            reward = torch.Tensor(reward).to(self.device)

            # Train model with policy gradient
            self.optim.zero_grad()
            loss = loss * (reward - self.beta)
            loss = -loss.mean()
            loss.backward()
            self.optim.step()

            self.monitor.saveProgress(self, step_idx, epoch, total_steps, epochs, loss=loss.item())

        return loss.item()

    @abstractmethod
    def sampleEncodedPairsToLoader(self, net, loader):
        """Sample encoded fragment-molecule pairs from generator into a data loader."""
        pass

    @abstractmethod
    def sample_input(self, loader, is_test=False):
        """Subsample input fragment pairs from data loader for epoch batching."""
        pass

    def fit(
        self,
        train_loader,
        valid_loader=None,
        epochs=1000,
        patience=50,
        criteria="desired_ratio",
        min_epochs=100,
        monitor=None,
    ):
        """Execute RL training loop with policy gradient optimization, early stopping, and periodic evaluation."""

        self.monitor = monitor if monitor else NullMonitor()
        self.monitor.saveModel(self)
        self.bestState = self.getModel()

        n_iters = 1 if self.crover is None else 10
        net = nn.DataParallel(self, device_ids=self.gpus)
        logger.info(" ")

        for it in range(n_iters):
            if n_iters > 1:
                logger.info("\n----------\nITERATION %d/%d\n----------" % (it, n_iters))
            for epoch in tqdm(range(epochs), desc="Fitting graph explorer"):
                epoch += 1
                is_best = False

                # If nSamples is set, sample a subset of the training data at each epoch
                if self.nSamples > 0:
                    if epoch == 1:
                        train_loader_original = train_loader
                        valid_loader_original = valid_loader
                    train_loader = self.sample_input(train_loader_original)
                    valid_loader = self.sample_input(valid_loader_original, is_test=True)

                # Sample encoded molecules from the network
                loader = self.sampleEncodedPairsToLoader(net, train_loader)

                # Train the agent with policy gradient
                train_loss = self.policy_gradient(loader, epoch=epoch, epochs=epochs)

                # Evaluate model on validation set
                smiles, frags = self.agent.sample(valid_loader)
                scores = self.agent.evaluate(
                    smiles, frags, evaluator=self.env, no_multifrag_smiles=self.no_multifrag_smiles
                )
                scores["SMILES"], scores["Frags"] = smiles, frags

                # Compute metrics
                metrics = self.getNovelMoleculeMetrics(scores)
                metrics["loss_train"] = train_loss

                # Save evaluate criteria and set best model
                if metrics[criteria] > self.best_value:
                    is_best = True
                    self.saveBestState(metrics[criteria], epoch, it)

                # Save (intermediate) models
                save_model_option = monitor.getSaveModelOption()
                if save_model_option == "all" or is_best == True:
                    monitor.saveModel(self, epoch if save_model_option in ("all", "improvement") else None)
                    logger.info(f"Model saved at epoch {epoch}")

                # Log performance and generated compounds
                self.logPerformanceAndCompounds(epoch, metrics, scores)

                # Early stopping
                if (epoch >= min_epochs) and (epoch - self.last_save > patience):
                    break

            if self.crover is not None:
                logger.warning("Behavior of using crover with a Transformer is not tested yet. Use at your own risk.")
                self.agent.load_state_dict(self.bestState)
                self.crover.load_state_dict(self.bestState)
            if it - self.last_iter > 1:
                break

        torch.cuda.empty_cache()
        self.monitor.close()
