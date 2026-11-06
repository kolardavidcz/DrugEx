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
from _david.code.training.loss_criterion import HoldoutSample, build_holdout
from _david.code.schema import std
from _david.code.utils import HardwareManager

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]


def decayed_epsilon(
    epoch: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    decay_start_epoch: int,
    decay_epochs: int,
    decay_type: str,
    use_decay: bool,
) -> float:
    """Exploration noise epsilon for a 1-based epoch (single source of truth).

    Used by ``DecayingSequenceExplorer`` during training and by the training-dynamics
    figure, so the plotted schedule is exactly the schedule the agent followed.
    """
    if not use_decay or decay_epochs <= decay_start_epoch:
        return epsilon_start
    if epoch <= decay_start_epoch:
        return epsilon_start
    if epoch >= decay_epochs:
        return epsilon_end

    progress = (epoch - decay_start_epoch) / (decay_epochs - decay_start_epoch)
    if decay_type == "linear":
        return epsilon_start + progress * (epsilon_end - epsilon_start)
    if decay_type == "cosine":
        # ~ [Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017]
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return epsilon_end + (epsilon_start - epsilon_end) * cosine_factor
    raise ValueError(
        f"Unsupported decay_type: '{decay_type}'. Choose 'cosine' or 'linear'."
    )



class DecayingSequenceExplorer(SequenceExplorer):
    """SequenceExplorer with scheduled exploration noise (epsilon) decay."""

    def __init__(
        self,
        agent: SequenceRNN,
        env: DrugExEnvironment,
        mutate: SequenceRNN | None = None,
        config: PipelineConfig | None = None,
        crover: Any = None,
        no_multifrag_smiles: bool = True,
        optim: Any = None,
        device: Any = DEFAULT_DEVICE,
        use_gpus: Any = DEFAULT_GPUS,
    ) -> None:
        cfg = config or PipelineConfig()
        super().__init__(
            agent=agent,
            env=env,
            mutate=mutate,
            crover=crover,
            no_multifrag_smiles=no_multifrag_smiles,
            batch_size=cfg.batch_size,
            epsilon=cfg.epsilon,
            beta=cfg.beta,
            n_samples=cfg.n_samples,
            optim=optim,
            device=device,
            use_gpus=use_gpus,
        )
        self.epsilon_start: float = float(cfg.epsilon)
        self.epsilon_end: float = float(cfg.epsilon_end)
        self.decay_start_epoch: int = int(cfg.decay_start_epoch)
        self.decay_epochs: int = int(cfg.decay_epochs)
        self.decay_type: str = str(cfg.decay_type).lower()
        self.use_decay: bool = bool(cfg.use_epsilon_decay)
        self.current_epoch: int = 0

    def get_decayed_epsilon(self, epoch: int) -> float:
        """Compute scheduled exploration noise epsilon for the given 1-based epoch."""
        return decayed_epsilon(
            epoch,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            decay_start_epoch=self.decay_start_epoch,
            decay_epochs=self.decay_epochs,
            decay_type=self.decay_type,
            use_decay=self.use_decay,
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


class LossSelectedSequenceExplorer(DecayingSequenceExplorer):
    """Explorer whose checkpoint is chosen by held-out reward-weighted NLL.

    The core loop keeps the epoch maximizing ``metrics[criteria]``. ``J`` from
    :mod:`_david.code.training.loss_criterion` is lower-is-better, so this class records its
    negation as ``val_loss_neg`` and selects on that column. ``best_value`` starts
    at ``-inf`` because every negated ``J`` is negative and the base class starts
    at ``0``, which no epoch could ever beat.

    Selection is gated: an epoch whose validation arm falls below
    ``min_valid_ratio`` or ``min_unique_ratio`` still records ``val_loss`` (so the
    run shows what the criterion wanted) but exposes ``NaN`` as
    ``val_loss_neg``. ``NaN > best_value`` is always false in the core comparison,
    so a collapsed epoch can neither win nor be logged as a misleading ``-inf``.
    """

    def __init__(
        self,
        holdout: HoldoutSample,
        min_valid_ratio: float,
        min_unique_ratio: float,
        agent: SequenceRNN,
        env: DrugExEnvironment,
        mutate: SequenceRNN | None = None,
        config: PipelineConfig | None = None,
        crover: Any = None,
        no_multifrag_smiles: bool = True,
        optim: Any = None,
        device: Any = DEFAULT_DEVICE,
        use_gpus: Any = DEFAULT_GPUS,
    ) -> None:
        super().__init__(
            agent=agent,
            env=env,
            mutate=mutate,
            config=config,
            crover=crover,
            no_multifrag_smiles=no_multifrag_smiles,
            optim=optim,
            device=device,
            use_gpus=use_gpus,
        )
        self.holdout = holdout
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_unique_ratio = float(min_unique_ratio)
        self.best_value = float("-inf")

    def fit(self, *args: Any, criteria: str | None = None, **kwargs: Any) -> None:
        """Run the core training loop, forcing selection on ``val_loss_neg``.

        Args:
            criteria: Accepted only to reject a mismatched request; the criterion
                is a property of this explorer, not of the caller.

        Raises:
            ValueError: When a criterion other than the loss criterion is asked for.
        """
        if criteria not in (None, std.VAL_LOSS, std.VAL_LOSS_NEG, "loss"):
            raise ValueError(
                f"LossSelectedSequenceExplorer always selects on {std.VAL_LOSS_NEG!r}; "
                f"passing criteria={criteria!r} would silently restore a different "
                "checkpoint rule."
            )
        super().fit(*args, criteria=std.VAL_LOSS_NEG, **kwargs)

    def getNovelMoleculeMetrics(self, scores: pd.DataFrame) -> dict[str, Any]:
        """Add the held-out loss and its eligibility gate to the standard metrics.

        The epoch's own validation arm is scored first (validity, uniqueness,
        desirability), because a collapsed arm makes the loss meaningless even when
        the policy still assigns mass to the holdout: the reward the run reports
        would be unreachable.

        Args:
            scores: Scored validation molecules of the current epoch.

        Returns:
            The base metrics plus ``val_loss``, ``val_loss_neg`` and
            ``val_loss_eligible`` (1.0/0.0).
        """
        metrics = super().getNovelMoleculeMetrics(scores)
        eligible = (
            metrics[std.VALID_RATIO] >= self.min_valid_ratio
            and metrics[std.UNIQUE_RATIO] >= self.min_unique_ratio
        )
        # ``val_loss`` is always recorded, even when the gate rejects the epoch:
        # the diagnostic "J fell while the generator collapsed" is the whole point
        # of the floor, and it is only visible if the value survives. Only the
        # selection column is blanked, so a rejected epoch can never be the best.
        value = self.holdout.evaluate(self.agent)
        metrics[std.VAL_LOSS] = value
        metrics[std.VAL_LOSS_ELIGIBLE] = float(eligible)
        metrics[std.VAL_LOSS_NEG] = -value if eligible else float("nan")
        return metrics


class SequenceRNNOptimizer:
    """Pretrained SMILES SequenceRNN agent lifecycle, frozen mutate prior builder, and explorer factory."""

    def __init__(
        self,
        models_dir: Path | str,
        config: PipelineConfig | None = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        HardwareManager.init_runtime_environment(ratio=self.cfg.cpu_ratio)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device, self.use_gpus = HardwareManager.get_device_and_gpus(
            preferred_gpu=self.cfg.gpu
        )
        self.agent_path = self.cfg.agent_path
        self.vocab_path = self.cfg.vocab_path

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
        config: PipelineConfig | None = None,
    ) -> SequenceExplorer:
        """Construct the configured explorer.

        The two classes take disjoint constructor arguments: ``SequenceExplorer``
        accepts the RL knobs but not a ``PipelineConfig``, while
        ``DecayingSequenceExplorer`` reads them from the config and rejects the
        knobs. Passing one union of keywords to both raises ``TypeError`` at the
        start of Stage 2, so each branch is built explicitly here and pinned by
        ``test_build_explorer_constructs_both_branches``.
        """
        cfg = config or self.cfg
        common: dict[str, Any] = dict(
            agent=agent,
            env=env,
            mutate=mutate,
            device=self.device,
            use_gpus=self.use_gpus,
        )
        if cfg.criteria == std.VAL_LOSS:
            # The loss criterion is a property of the explorer, not only of the
            # config: it needs the frozen holdout built from the run's own reward
            # scheme and vocabulary, and it must select on the negated column.
            holdout = build_holdout(cfg, agent.voc, env.rewardScheme)
            return LossSelectedSequenceExplorer(
                holdout=holdout,
                min_valid_ratio=cfg.val_min_valid_ratio,
                min_unique_ratio=cfg.val_min_unique_ratio,
                config=cfg,
                **common,
            )
        if cfg.use_epsilon_decay:
            return DecayingSequenceExplorer(config=cfg, **common)
        return SequenceExplorer(
            batch_size=cfg.batch_size,
            epsilon=cfg.epsilon,
            beta=cfg.beta,
            n_samples=cfg.n_samples,
            **common,
        )
