"""Graph-Transformer generative model, Prior/Agent initialization, and fragment preparation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import FragmentCorpusEncoder, GraphFragmentEncoder
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.utils import get_safe_gpus

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class GraphTransformerOptimizer:
    """Manages Pretrained Graph-Transformer Prior and Agent networks for RL exploration."""

    def __init__(self, output_dir: Path | str, gpu: int | str | None = "auto") -> None:
        self.OUTPUT_DIR = Path(output_dir)
        self.MODEL_DIR = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        pretrained_dir = PROJECT_ROOT / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"
        pkg_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.pkg"
        vocab_path = pretrained_dir / "Papyrus05.5_graph_trans_PT.vocab"
        if not pkg_path.exists():
            pkg_path = pretrained_dir.with_suffix(".pkg")
            vocab_path = pretrained_dir.with_suffix(".vocab")

        self.GPUS = get_safe_gpus(preferred_gpu=gpu)
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
    ) -> tuple[DataLoader, GraphFragDataSet]:
        """Prepares fragment dataset and DataLoader from Murcko core scaffold."""
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
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol:
                scaff = MurckoScaffold.GetScaffoldForMol(Chem.RemoveHs(mol))
                smi = Chem.MolToSmiles(scaff, isomericSmiles=False) if (scaff and scaff.GetNumHeavyAtoms() > 0) else molecule_smiles
                scaffolds.append(smi)
        n_repeats = max(1, n_prompts // len(scaffolds) + 1)
        expanded_prompts = (scaffolds * n_repeats)[:n_prompts]
        encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
        loader = seed_dataset.asDataLoader(batch_size=32)
        return loader, seed_dataset

    def build_explorer(
        self,
        env: DrugExEnvironment,
        epsilon: float = 0.0,
        batch_size: int = 50,
    ) -> FragGraphExplorer:
        """Constructs FragGraphExplorer bound to environment and mutation policy."""
        return FragGraphExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=epsilon,
            beta=0.0,
            batch_size=batch_size,
            use_gpus=self.GPUS,
        )
