from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.logs import logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN

from _david.code.config import PipelineConfig
from _david.code.utils import HardwareManager

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class DecayingSequenceExplorer(SequenceExplorer):
    """SequenceExplorer with scheduled exploration noise (epsilon) decay."""

    def __init__(
        self,
        agent: SequenceRNN,
        env: DrugExEnvironment,
        mutate: SequenceRNN | None = None,
        crover: Any = None,
        no_multifrag_smiles: bool = True,
        batch_size: int = 128,
        epsilon: float = 0.20,
        epsilon_end: float = 0.05,
        decay_epochs: int = 50,
        decay_type: str = "cosine",
        use_decay: bool = True,
        beta: float = 0.0,
        n_samples: int = 500,
        optim: Any = None,
        device: Any = DEFAULT_DEVICE,
        use_gpus: Any = DEFAULT_GPUS,
    ) -> None:
        super().__init__(
            agent=agent,
            env=env,
            mutate=mutate,
            crover=crover,
            no_multifrag_smiles=no_multifrag_smiles,
            batch_size=batch_size,
            epsilon=epsilon,
            beta=beta,
            n_samples=n_samples,
            optim=optim,
            device=device,
            use_gpus=use_gpus,
        )
        self.epsilon_start: float = float(epsilon)
        self.epsilon_end: float = float(epsilon_end)
        self.decay_epochs: int = int(decay_epochs)
        self.decay_type: str = str(decay_type).lower()
        self.use_decay: bool = bool(use_decay)
        self.current_epoch: int = 0

    def get_decayed_epsilon(self, epoch: int) -> float:
        """Compute scheduled exploration noise epsilon for the given 1-based epoch."""
        if not self.use_decay or self.decay_epochs <= 1:
            return self.epsilon_start
        if epoch >= self.decay_epochs:
            return self.epsilon_end
        if epoch <= 1:
            return self.epsilon_start

        progress = (epoch - 1) / (self.decay_epochs - 1)
        if self.decay_type == "linear":
            return self.epsilon_start + progress * (
                self.epsilon_end - self.epsilon_start
            )
        elif self.decay_type == "cosine":
            # ~ [Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017]
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                self.epsilon_end
                + (self.epsilon_start - self.epsilon_end) * cosine_factor
            )
        else:
            raise ValueError(
                f"Unsupported decay_type: '{self.decay_type}'. Choose 'cosine' or 'linear'."
            )

    def forward(self) -> tuple[Any, Any]:
        """Advance epoch counter, update exploration noise if scheduled, and generate sample batch."""
        self.current_epoch += 1
        if self.use_decay:
            self.epsilon = self.get_decayed_epsilon(self.current_epoch)
        return super().forward()

    def logPerformanceAndCompounds(
        self, epoch: int, metrics: dict[str, Any], scores: pd.DataFrame
    ) -> None:
        """Record exact current exploration noise (epsilon) alongside standard epoch metrics."""
        metrics["epsilon"] = float(self.epsilon)
        super().logPerformanceAndCompounds(epoch, metrics, scores)


class SequenceRNNOptimizer:
    """Pretrained SMILES SequenceRNN agent lifecycle, frozen mutate prior builder, and explorer factory."""

    def __init__(
        self,
        models_dir: Path | str,
        gpu: int | str | None = "auto",
        agent_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
    ) -> None:
        HardwareManager.init_runtime_environment(ratio=0.90)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.use_gpus = HardwareManager.get_device_and_gpus(
            preferred_gpu=gpu
        )
        self.agent_path = agent_path
        self.vocab_path = vocab_path

    def load_models(
        self,
        pkg_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
    ) -> tuple[SequenceRNN, SequenceRNN, VocSmiles]:
        """Instantiate active SequenceRNN agent and an independent, frozen mutate exploration prior."""
        cfg = PipelineConfig(
            agent_path=pkg_path or self.agent_path,
            vocab_path=vocab_path or self.vocab_path,
        )
        pkg, voc_file = cfg.resolve_pretrained_paths()

        logger.info(f"Loading SMILES vocabulary from: {voc_file}")
        voc = VocSmiles.fromFile(str(voc_file), encode_frags=False)

        # Active trainable RL agent
        logger.info(f"Instantiating trainable SequenceRNN agent from: {pkg}")
        agent = SequenceRNN(
            voc, is_lstm=True, device=self.device, use_gpus=self.use_gpus
        )
        agent.loadStatesFromFile(str(pkg))

        # Separate frozen exploration prior for mutate
        logger.info(f"Instantiating frozen SequenceRNN prior from: {pkg}")
        prior = SequenceRNN(
            voc, is_lstm=True, device=self.device, use_gpus=self.use_gpus
        )
        prior.loadStatesFromFile(str(pkg))
        for p in prior.parameters():
            p.requires_grad_(False)
        prior.eval()

        # Strict identity verification: mutate MUST NOT be agent
        assert prior is not agent, (
            "Exploration prior MUST be an independent object instance, not agent!"
        )
        assert id(prior) != id(agent), (
            "Identity check failed: prior and agent share identical memory address!"
        )

        return agent, prior, voc

    def build_explorer(
        self,
        agent: SequenceRNN,
        env: DrugExEnvironment,
        mutate: SequenceRNN,
        config: PipelineConfig,
    ) -> SequenceExplorer:
        """Construct configured SequenceExplorer with decoupled stopping criteria and exploration schedule."""
        if config.use_epsilon_decay:
            return DecayingSequenceExplorer(
                agent=agent,
                env=env,
                mutate=mutate,
                batch_size=config.batch_size,
                epsilon=config.epsilon,
                epsilon_end=config.epsilon_end,
                decay_epochs=config.decay_epochs,
                decay_type=config.decay_type,
                use_decay=True,
                beta=config.beta,
                n_samples=config.n_samples,
                device=self.device,
                use_gpus=self.use_gpus,
            )
        return SequenceExplorer(
            agent=agent,
            env=env,
            mutate=mutate,
            batch_size=config.batch_size,
            epsilon=config.epsilon,
            beta=config.beta,
            n_samples=config.n_samples,
            device=self.device,
            use_gpus=self.use_gpus,
        )
