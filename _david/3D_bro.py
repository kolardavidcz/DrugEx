from __future__ import annotations

import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd
from pandas.core.frame import DataFrame
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qsprpred import data as qspr_data  # noqa: F401
from qsprpred.models import scikit_learn as qspr_models  # noqa: F401
from scaffviz.clustering import manifold  # noqa: F401
from scaffviz.depiction import plot as scaff_plot  # noqa: F401

import drugex  # noqa: F401
from drugex import parallel, training, utils  # noqa: F401
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import (
    FragmentCorpusEncoder,
    FragmentPairsSplitter,
    GraphFragmentEncoder,
)
from drugex.data.processing import Standardization
from drugex.logs import logger
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.monitors import FileMonitor, NullMonitor
from drugex.training.rewards import (
    ParetoCrowdingDistance,
    ParetoRewardScheme,
    ParetoTanimotoDistance,
    WeightedSum,
)
from drugex.training.scorers import (
    modifiers,
    properties,
    qsprpred,
    sascorer,
    similarity,
)
from drugex.training.scorers.modifiers import (
    ClippedScore,
    ScoreModifier,
    SmoothClippedScore,
)
from drugex.training.scorers.properties import Property, Uniqueness
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.similarity import (
    FraggleSimilarity,
    TverskyGraphSimilarity,
)

try:
    from _david import receptor_similar
except ModuleNotFoundError:
    import receptor_similar

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

# --- Hardware & System Constants ---
N_PROCESSES = 12
CHUNK_SIZE = 1000
GPUS = [1]

# --- Fragmentation & Graph Constants ---
FRAG_MIN_FRAGS = 4
FRAG_MAX_FRAGS = 4
FRAG_METHOD = "brics"
FRAG_SPLIT_RATIO = 0.10
FRAG_MAX_PAIRS = 100

# --- Transfer Learning Constants ---
TL_EPOCHS = 20
TL_BATCH_SIZE = 64
TL_LOSS_TOLERANCE = 0.02
TL_TEST_SIZE = 0.10
TL_MAX_MOLECULES = 200

# --- Reinforcement Learning Constants ---
RL_EPOCHS = 100
RL_EPSILON = 0.20
RL_BETA = 0.00
RL_BATCH_SIZE = 64
RL_PATIENCE = 20

# --- Scorer & Objective Constants ---
TVERSKY_ALPHA = 0.70
TVERSKY_BETA = 0.30
TVERSKY_THRESHOLD = 0.40

QED_LOWER_X = 0.40
QED_UPPER_X = 0.80
QED_THRESHOLD = 0.50

SA_LOWER_X = 4.50
SA_UPPER_X = 2.50
SA_THRESHOLD = 0.50


class MY_ERROR(RuntimeError):
    def __init__(self, message: str = "Unrecoverable error in DrugEx execution pipeline."):
        super().__init__(message)


class DrugExVisualizer:
    @staticmethod
    @contextmanager
    def save_plot(output_path: Path | str, **subplots_kwargs):
        fig, ax = plt.subplots(**subplots_kwargs)
        try:
            yield ax
        finally:
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def save_molecules_grid(generated_sample: DataFrame, out_dir: Path, num_mols: int = 25):
        mols = [
            m for s in generated_sample["SMILES"][:num_mols]
            if (m := Chem.MolFromSmiles(s)) is not None
        ]
        if not mols:
            raise MY_ERROR("No valid SMILES strings could be parsed to generate a 2D molecule grid.")

        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[250, 250])
        img.save(str(out_dir / "generated_molecules.png"))

    @staticmethod
    def visualize_transfer_learning(finetuned: GraphTransformer, model_prefix: Path, out_dir: Path, input_dataset: GraphFragDataSet | None = None, input_fragments: list[str] | None = None, num_samples: int = 100):
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "loss_valid", "valid_ratio"] if c in df_info.columns]
            with DrugExVisualizer.save_plot(out_dir / "training_loss.png") as ax:
                df_info[metric_cols].plot.line(logy=True, ax=ax)
                ax.set_title("Transfer Learning Loss & Validity Dynamics")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Log Value")
                ax.grid(True, linestyle="--", alpha=0.6)

        if input_dataset is not None:
            generated_sample = finetuned.generate(input_dataset=input_dataset, num_samples=num_samples)
        elif input_fragments is not None:
            generated_sample = finetuned.generate(input_fragments, num_samples=num_samples)
        else:
            generated_sample = finetuned.generate(num_samples=num_samples)

        generated_sample.to_csv(out_dir / "generated_molecules.tsv", sep="\t", index=False)
        DrugExVisualizer.save_molecules_grid(generated_sample, out_dir)
        return generated_sample

    @staticmethod
    def visualize_reinforcement_learning(agent: GraphTransformer, env: DrugExEnvironment, model_prefix: Path, out_dir: Path, input_dataset: GraphFragDataSet | None = None, input_fragments: list[str] | None = None, num_samples: int = 100):
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
            with DrugExVisualizer.save_plot(out_dir / "rl_training_curves.png") as ax:
                df_info[metric_cols].plot.line(ax=ax)
                ax.set_title("Reinforcement Learning Training Dynamics")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Metric Value")
                ax.grid(True, linestyle="--", alpha=0.6)

        if input_dataset is not None:
            generated_sample = agent.generate(input_dataset=input_dataset, num_samples=num_samples)
        elif input_fragments is not None:
            generated_sample = agent.generate(input_fragments, num_samples=num_samples)
        else:
            generated_sample = agent.generate(num_samples=num_samples)

        scores = env.getScores(generated_sample["SMILES"])
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        DrugExVisualizer.save_molecules_grid(generated_sample, out_dir)
        return scored_df


class ALL_METHODS:
    def __init__(self, output_dir: Path | None = None):
        self.ROOT_PATH = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR = self.ROOT_PATH / "_david" / "outputs"

        if output_dir is None:
            self.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.OUTPUT_DIR = self.BASE_OUTPUT_DIR / self.RUN_ID
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.OUTPUT_DIR = output_dir
            self.RUN_ID = output_dir.name

        self.MODEL_PRETRAINED = self.ROOT_PATH / "data/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT"
        self.VOC = VocGraph.fromFile(self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.vocab")

    molecules_save_image = staticmethod(DrugExVisualizer.save_molecules_grid)
    MY_save_plot = staticmethod(DrugExVisualizer.save_plot)
    vizualize = staticmethod(DrugExVisualizer.visualize_transfer_learning)
    vizualize_rl = staticmethod(DrugExVisualizer.visualize_reinforcement_learning)


class MY_3D(ALL_METHODS):
    def __init__(self, output_dir: Path | None = None):
        super().__init__(output_dir)

    def prepare_fragment_corpus(self, smiles_list: list[str], output_prefix: Path):
        encoder = FragmentCorpusEncoder(
            fragmenter=Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, FRAG_METHOD),
            encoder=GraphFragmentEncoder(VocGraph(n_frags=FRAG_MAX_FRAGS)),
            pairs_splitter=FragmentPairsSplitter(FRAG_SPLIT_RATIO, FRAG_MAX_PAIRS),
            n_proc=N_PROCESSES,
        )
        train_ds = GraphFragDataSet(f"{output_prefix}_train.tsv", rewrite=True)
        test_ds = GraphFragDataSet(f"{output_prefix}_test.tsv", rewrite=True)
        encoder.apply(smiles_list, encodingCollectors=[test_ds, train_ds])
        return train_ds, test_ds
