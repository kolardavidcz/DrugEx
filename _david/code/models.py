from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from _david.code.environment import PrepareMolecule
from _david.code.utils import HardwareManager

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@dataclass
class AlgorithmConfig:
    """Hyperparameters for GraphTransformer reinforcement learning exploration.

    WHY:
    Encapsulates all RL exploration hyperparameters in an immutable, validated
    dataclass while separating algorithm parameters from dynamic runtime hardware sizing.

    WHAT:
    - epochs: Number of RL policy gradient training epochs.
    - n_samples: Number of molecules generated and aligned per training step.
    - val_samples: Optional explicit size for decoupled validation set.
    - epsilon: Exploration probability for epsilon-greedy or entropy regularization.
    - beta: Prior loss penalization baseline weight.
    - patience: Early stopping patience based on criteria.
    - min_epochs: Minimum epochs before early stopping can trigger.
    - criteria: Monitoring objective metric (e.g. 'desired_ratio').
    - seed: Global pseudo-random number generator seed.
    """

    epochs: int = 35
    n_samples: int = 200
    val_samples: int | None = None
    epsilon: float = 0.2
    beta: float = 0.0
    patience: int = 25
    min_epochs: int = 5
    criteria: str = "desired_ratio"
    seed: int = 42


class GraphTransformerOptimizer:

    def __init__(self, output_dir: Path | str, gpu: int | str | None = "auto") -> None:
        HardwareManager.init_runtime_environment()
        self.OUTPUT_DIR = Path(output_dir)
        self.MODEL_DIR = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        pretrained_dir = PROJECT_ROOT / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"
        pkg_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"
        vocab_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"
        if not pkg_path.exists():
            pkg_path = pretrained_dir.with_suffix(".pkg")
            vocab_path = pretrained_dir.with_suffix(".vocab")

        self.GPUS = HardwareManager.get_safe_gpus(preferred_gpu=gpu)
        self.VOC = VocGraph.fromFile(str(vocab_path))

        self.AGENT = GraphTransformer(voc_trg=self.VOC, use_gpus=self.GPUS)
        self.AGENT.loadStatesFromFile(str(pkg_path))

        self.PRIOR = GraphTransformer(voc_trg=self.VOC, use_gpus=self.GPUS)
        self.PRIOR.loadStatesFromFile(str(pkg_path))

    def prepare_seed_fragments(
        self,
        molecule_smiles: str,
        n_prompts: int = 100,
        scaffolds_path: Path | str | None = None,
        val_samples: int | None = None,
    ) -> tuple[DataLoader, DataLoader, GraphFragDataSet]:
        """Generate training and decoupled validation DataLoaders for seed fragments.

        WHY:
        Passing the full training prompt dataset (e.g. N = 200) as both training and validation
        loaders forced DrugEx explorer.fit() to generate conformers and compute 3D ROCS alignments
        twice per epoch (400 alignments total), cutting throughput in half. Decoupling validation
        with a bounded sample (val_size <= 50) preserves statistically rigorous tracking of
        desired_ratio (SE ~ 6.8%) while cutting validation compute by 75-80%.

        WHAT:
        Extracts seed fragments, encodes into GraphFragDataSet, slices the raw data array
        raw_data[:val_size], and constructs separate train_loader and val_loader via dataToLoader.

        HOW:
        val_size = min(val_samples or min(50, max(25, int(n_prompts * 0.20))), n_prompts).
        Batch size is determined dynamically via HardwareManager.get_recommended_batch_size.
        """
        seed_tsv = self.MODEL_DIR / "seed_fragments.tsv"
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
            scaffolds.append(PrepareMolecule().extract_seed_core_scaffold(molecule_smiles))
        n_repeats = max(1, n_prompts // len(scaffolds) + 1)
        expanded_prompts = (scaffolds * n_repeats)[:n_prompts]
        encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])

        # Decouple validation data from training data by slicing underlying raw data array
        raw_data = seed_dataset.getData()
        val_size = min(val_samples or min(50, max(25, int(n_prompts * 0.20))), n_prompts)
        val_data = raw_data[:val_size]

        batch_size = HardwareManager.get_recommended_batch_size(n_prompts)
        train_loader = seed_dataset.dataToLoader(
            raw_data, batch_size=batch_size, vocabulary=self.VOC
        )
        val_loader = seed_dataset.dataToLoader(
            val_data, batch_size=batch_size, vocabulary=self.VOC
        )
        return train_loader, val_loader, seed_dataset

    def build_explorer(
        self,
        env: DrugExEnvironment,
        epsilon: float = 0.0,
        batch_size: int = 64,
    ) -> FragGraphExplorer:
        """Construct FragGraphExplorer reinforcement learning agent."""
        return FragGraphExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=epsilon,
            beta=0.0,
            batch_size=batch_size,
            use_gpus=self.GPUS,
        )
