from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.logs import logger
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer

from _david.code.environment import PrepareMolecule
from _david.code.utils import HardwareManager

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class DecayingFragGraphExplorer(FragGraphExplorer):
    """FragGraphExplorer with scheduled exploration noise (epsilon) decay."""

    def __init__(
        self,
        agent: Any,
        env: DrugExEnvironment,
        mutate: Any = None,
        crover: Any = None,
        batch_size: int = 128,
        epsilon: float = 0.25,
        epsilon_end: float = 0.05,
        decay_epochs: int = 50,
        decay_type: str = "cosine",
        beta: float = 0.0,
        n_samples: int = -1,
        optim: Any = None,
        device: Any = DEFAULT_DEVICE,
        use_gpus: Any = DEFAULT_GPUS,
        no_multifrag_smiles: bool = True,
    ) -> None:
        super().__init__(
            agent=agent,
            env=env,
            mutate=mutate,
            crover=crover,
            batch_size=batch_size,
            epsilon=epsilon,
            beta=beta,
            n_samples=n_samples,
            optim=optim,
            device=device,
            use_gpus=use_gpus,
            no_multifrag_smiles=no_multifrag_smiles,
        )
        self.epsilon_start: float = float(epsilon)
        self.epsilon_end: float = float(epsilon_end)
        self.decay_epochs: int = int(decay_epochs)
        self.decay_type: str = str(decay_type).lower()
        self.current_epoch: int = 0

    def get_decayed_epsilon(self, epoch: int) -> float:
        """Compute scheduled exploration noise epsilon for the given 1-based epoch.

        Supports 'cosine' (Loshchilov & Hutter) and 'linear' schedules.
        Beyond decay_epochs, epsilon remains clamped at epsilon_end.
        """
        if self.decay_epochs <= 1 or epoch >= self.decay_epochs:
            return self.epsilon_end
        if epoch <= 1:
            return self.epsilon_start

        progress = (epoch - 1) / (self.decay_epochs - 1)
        if self.decay_type == "linear":
            return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        elif self.decay_type == "cosine":
            #~ [Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017]
            #~ Half-cycle cosine annealing schedule provides smooth transition from broad exploration to exploitation.
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * cosine_factor
        else:
            raise ValueError(f"Unsupported decay_type: '{self.decay_type}'. Choose 'cosine' or 'linear'.")

    def sampleEncodedPairsToLoader(self, net: torch.nn.Module, loader: DataLoader) -> DataLoader:
        """Update epsilon exploration noise for current epoch before sampling training batch."""
        self.current_epoch += 1
        new_eps = self.get_decayed_epsilon(self.current_epoch)
        self.epsilon = new_eps
        if hasattr(net, "module"):
            net.module.epsilon = new_eps
        logger.info(
            f"[Exploration Schedule] Epoch {self.current_epoch}: epsilon = {self.epsilon:.4f} "
            f"(start={self.epsilon_start:.2f}, end={self.epsilon_end:.2f}, "
            f"decay_epochs={self.decay_epochs}, type={self.decay_type})"
        )
        return super().sampleEncodedPairsToLoader(net, loader)

    def logPerformanceAndCompounds(self, epoch: int, metrics: dict[str, Any], scores: pd.DataFrame) -> None:
        """Enrich epoch summary metrics with exact current exploration noise (epsilon)."""
        metrics["epsilon"] = float(self.epsilon)
        super().logPerformanceAndCompounds(epoch, metrics, scores)


class GraphTransformerOptimizer:
    """Pretrained GraphTransformer agent lifecycle, fragment dataset preparation, and explorer builder."""

    def __init__(
        self,
        models_dir: Path | str,
        raw_data_dir: Path | str,
        gpu: int | str | None = "auto",
        agent_path: Path | str | None = None,
    ) -> None:
        HardwareManager.init_runtime_environment()
        self.MODELS_DIR = Path(models_dir)
        self.RAW_DATA_DIR = Path(raw_data_dir)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        pretrained_dir = (
            PROJECT_ROOT
            / "data"
            / "models"
            / "pretrained"
            / "graph-trans"
            / "Papyrus05.5_graph_trans_PT"
        )
        pkg_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"
        vocab_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"
        if not pkg_path.exists():
            pkg_path = pretrained_dir.with_suffix(".pkg")
            vocab_path = pretrained_dir.with_suffix(".vocab")

        self.GPUS = HardwareManager.get_safe_gpus(preferred_gpu=gpu)
        self.VOC = VocGraph.fromFile(str(vocab_path))

        agent_pkg = Path(agent_path) if agent_path else pkg_path
        self.AGENT = GraphTransformer(voc_trg=self.VOC, use_gpus=self.GPUS)
        self.AGENT.loadStatesFromFile(str(agent_pkg))

        self.PRIOR = GraphTransformer(voc_trg=self.VOC, use_gpus=self.GPUS)
        self.PRIOR.loadStatesFromFile(str(pkg_path))

    def prepare_seed_fragments(
        self,
        molecule_smiles: str,
        n_prompts: int = 100,
        scaffolds_path: Path | str | None = None,
        val_samples: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[DataLoader, DataLoader, GraphFragDataSet]:
        """Generate training and decoupled validation DataLoaders for seed fragments."""
        seed_tsv = self.RAW_DATA_DIR / "seed_fragments.tsv"
        seed_dataset = GraphFragDataSet(str(seed_tsv), rewrite=True)

        encoder = FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=GraphFragmentEncoder(self.VOC),
            pairs_splitter=None,
            n_proc=1,
        )

        scaffolds: list[str] = []
        if scaffolds_path and Path(scaffolds_path).is_file():
            with open(scaffolds_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines[1:] if len(lines) > 1 and "\t" in lines[0] else lines:
                parts = line.split("\t")
                if parts[0]:
                    scaffolds.append(parts[0])

        if not scaffolds:
            scaffolds.append(
                PrepareMolecule().extract_seed_core_scaffold(molecule_smiles)
            )
        n_repeats = max(1, n_prompts // len(scaffolds) + 1)
        expanded_prompts = (scaffolds * n_repeats)[:n_prompts]
        encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])

        # Decouple validation data from training data by slicing raw data array
        raw_data = seed_dataset.getData()
        val_size = min(val_samples if val_samples else max(50, int(n_prompts * 0.25)), n_prompts)
        val_data = raw_data[:val_size]

        eff_batch_size = batch_size or HardwareManager.get_recommended_batch_size(n_prompts)
        train_loader = seed_dataset.dataToLoader(
            raw_data, batch_size=eff_batch_size, vocabulary=self.VOC
        )
        val_loader = seed_dataset.dataToLoader(
            val_data, batch_size=eff_batch_size, vocabulary=self.VOC
        )
        return train_loader, val_loader, seed_dataset

    def build_explorer(
        self,
        env: DrugExEnvironment,
        epsilon: float = 0.25,
        beta: float = 0.0,
        batch_size: int = 64,
        epsilon_end: float | None = None,
        decay_epochs: int = 50,
        decay_type: str = "cosine",
    ) -> FragGraphExplorer:
        """Construct FragGraphExplorer or DecayingFragGraphExplorer reinforcement learning agent."""
        if epsilon_end is not None:
            return DecayingFragGraphExplorer(
                agent=self.AGENT,
                env=env,
                mutate=self.PRIOR,
                epsilon=epsilon,
                epsilon_end=epsilon_end,
                decay_epochs=decay_epochs,
                decay_type=decay_type,
                beta=beta,
                batch_size=batch_size,
                use_gpus=self.GPUS,
            )
        return FragGraphExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=epsilon,
            beta=beta,
            batch_size=batch_size,
            use_gpus=self.GPUS,
        )
