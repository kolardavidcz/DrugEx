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
        config: PipelineConfig | None = None,
        batch_size: int | None = None,
        epsilon: float | None = None,
        epsilon_end: float | None = None,
        decay_epochs: int | None = None,
        decay_type: str | None = None,
        use_decay: bool | None = None,
        beta: float | None = None,
        n_samples: int | None = None,
        optim: Any = None,
        device: Any = DEFAULT_DEVICE,
        use_gpus: Any = DEFAULT_GPUS,
    ) -> None:
        cfg = config or PipelineConfig()
        active_batch = cfg.batch_size if batch_size is None else batch_size
        active_eps = cfg.epsilon if epsilon is None else epsilon
        active_eps_end = cfg.epsilon_end if epsilon_end is None else epsilon_end
        active_decay_epochs = cfg.decay_epochs if decay_epochs is None else decay_epochs
        active_decay_type = cfg.decay_type if decay_type is None else decay_type
        active_use_decay = cfg.use_epsilon_decay if use_decay is None else use_decay
        active_beta = cfg.beta if beta is None else beta
        active_samples = cfg.n_samples if n_samples is None else n_samples
        super().__init__(
            agent=agent,
            env=env,
            mutate=mutate,
            crover=crover,
            no_multifrag_smiles=no_multifrag_smiles,
            batch_size=active_batch,
            epsilon=active_eps,
            beta=active_beta,
            n_samples=active_samples,
            optim=optim,
            device=device,
            use_gpus=use_gpus,
        )
        self.epsilon_start: float = float(active_eps)
        self.epsilon_end: float = float(active_eps_end)
        self.decay_epochs: int = int(active_decay_epochs)
        self.decay_type: str = str(active_decay_type).lower()
        self.use_decay: bool = bool(active_use_decay)
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
        config: PipelineConfig | None = None,
        gpu: int | str | None = None,
        agent_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
        cpu_ratio: float | None = None,
    ) -> None:
        cfg = config or PipelineConfig()
        active_ratio = cfg.cpu_ratio if cpu_ratio is None else cpu_ratio
        active_gpu = cfg.gpu if gpu is None else gpu
        HardwareManager.init_runtime_environment(ratio=active_ratio)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.use_gpus = HardwareManager.get_device_and_gpus(
            preferred_gpu=active_gpu
        )
        self.agent_path = cfg.agent_path if agent_path is None else agent_path
        self.vocab_path = cfg.vocab_path if vocab_path is None else vocab_path

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
        cls = DecayingSequenceExplorer if config.use_epsilon_decay else SequenceExplorer
        return cls(
            agent=agent,
            env=env,
            mutate=mutate,
            device=self.device,
            use_gpus=self.use_gpus,
            **config.explorer_kwargs(),
        )
