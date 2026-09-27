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

import drugex
from drugex import data, logs, parallel, training, utils
from drugex.data import datasets, fragments, processing
from drugex.data.corpus import vocabulary
from drugex.logs import logger
from drugex.molecules.converters import fragmenters
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.scorers import (
    conformer_generators,
    modifiers,
    properties,
    qsprpred,
    ra_scorer,
    rocs_rdkit,
    sascorer,
    similarity,
)

from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.data.fragments import (
    FragmentCorpusEncoder,
    FragmentPairsSplitter,
    GraphFragmentEncoder,
)
from drugex.data.processing import Standardization
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
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import (
    ClippedScore,
    ScoreModifier,
    SmoothClippedScore,
)
from drugex.training.scorers.properties import Property, Uniqueness
from drugex.training.scorers.qsprpred import QSPRPredScorer
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
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
TL_EPOCHS = 5
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
ROCS_SCORE_TYPE = "TanimotoCombo"
ROCS_MAX_CONFS = 2
ROCS_MAX_ISOMERS = 1
ROCS_TIMEOUT = 10
ROCS_LOWER_X = 0.40
ROCS_UPPER_X = 1.20
ROCS_THRESHOLD = 0.50

RA_LOWER_X = 0.20
RA_UPPER_X = 0.70
RA_THRESHOLD = 0.50

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
            generated_sample = finetuned.generate(input_frags=input_fragments, num_samples=num_samples)
        else:
            test_file = out_dir.parent / "ligand_test.tsv"
            if test_file.exists():
                generated_sample = finetuned.generate(input_dataset=GraphFragDataSet(str(test_file)), num_samples=num_samples)
            else:
                raise MY_ERROR("GraphTransformer requires input_dataset or input_fragments for generation.")

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
            generated_sample = agent.generate(input_frags=input_fragments, num_samples=num_samples)
        else:
            test_file = out_dir.parent / "ligand_test.tsv"
            if test_file.exists():
                generated_sample = agent.generate(input_dataset=GraphFragDataSet(str(test_file)), num_samples=num_samples)
            else:
                raise MY_ERROR("GraphTransformer requires input_dataset or input_fragments for generation.")

        scores = env.getScores(generated_sample["SMILES"])
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        DrugExVisualizer.save_molecules_grid(generated_sample, out_dir)
        return scored_df


    ####################################################################
    ####################################################################
    #   █████████  ███████████   █████████   ███████████   ███████████ #
    #  ███░░░░░███░█░░░███░░░█  ███░░░░░███ ░░███░░░░░███ ░█░░░███░░░█ #
    # ░███    ░░░ ░   ░███  ░  ░███    ░███  ░███    ░███ ░   ░███  ░  #
    # ░░█████████     ░███     ░███████████  ░██████████      ░███     #
    #  ░░░░░░░░███    ░███     ░███░░░░░███  ░███░░░░░███     ░███     #
    #  ███    ░███    ░███     ░███    ░███  ░███    ░███     ░███     #
    # ░░█████████     █████    █████   █████ █████   █████    █████    #
    #  ░░░░░░░░░     ░░░░░    ░░░░░   ░░░░░ ░░░░░   ░░░░░    ░░░░░     #
    ####################################################################
    ####################################################################


class ALL_METHODS:
    def __init__(self, output_dir: Path | None = None):
        self.ROOT_PATH = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR = self.ROOT_PATH / "_david" / "outputs_Graph"

        if output_dir is None:
            self.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.OUTPUT_DIR = self.BASE_OUTPUT_DIR / self.RUN_ID
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.OUTPUT_DIR = output_dir
            self.RUN_ID = output_dir.name

        self.MODEL_PRETRAINED = self.ROOT_PATH / "data/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT"
        self.VOC = vocabulary.VocGraph.fromFile(self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.vocab")

    molecules_save_image = staticmethod(DrugExVisualizer.save_molecules_grid)
    MY_save_plot = staticmethod(DrugExVisualizer.save_plot)
    vizualize = staticmethod(DrugExVisualizer.visualize_transfer_learning)
    vizualize_rl = staticmethod(DrugExVisualizer.visualize_reinforcement_learning)


class MY_TRANSFER_LEARNING(ALL_METHODS):
    def __init__(self, output_dir: Path | None = None):
        super().__init__(output_dir=output_dir)
        self.MODEL = self.OUTPUT_DIR / "transfer_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        self.PRETRAINED = generators.GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.PRETRAINED.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.pkg")

    def __receptor_similar_molecules_get_SMILES(self, molecule_smile: str):
        pipeline = receptor_similar.MoleculeBioactivityPipeline()
        result_df = pipeline.run(molecule_smile, max_molecules=TL_MAX_MOLECULES).molecules

        if len(result_df) == 0:
            raise MY_ERROR("No active target ligands discovered across Papyrus bioactivity database.")

        raw_smiles = result_df["SMILES"]
        standardized_smiles = processing.Standardization(n_proc=N_PROCESSES, chunk_size=CHUNK_SIZE).apply(raw_smiles)

        if standardized_smiles is None:
            raise MY_ERROR("Chemical standardization failed to process target ligands.")

        unique_smiles = utils.remove_duplicates(standardized_smiles)

        pd.Series(unique_smiles, name="SMILES").to_csv(self.MODEL / "training_ligands.tsv", sep="\t", index=False)

        return unique_smiles

    def __fragmentation_and_encoding(self, smiles_train: list[str]):
        encoder = fragments.FragmentCorpusEncoder(
            fragmenter=fragmenters.Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, FRAG_METHOD),
            encoder=fragments.GraphFragmentEncoder(self.VOC),
            pairs_splitter=fragments.FragmentPairsSplitter(FRAG_SPLIT_RATIO, FRAG_MAX_PAIRS),
            n_proc=N_PROCESSES,
            chunk_size=CHUNK_SIZE,
        )

        test_dataset = datasets.GraphFragDataSet(self.OUTPUT_DIR / "ligand_test.tsv", rewrite=True)
        train_dataset = datasets.GraphFragDataSet(self.OUTPUT_DIR / "ligand_train.tsv", rewrite=True)

        encoder.apply(smiles_train, encodingCollectors=[test_dataset, train_dataset])
        self.VOC.toFile(self.OUTPUT_DIR / "pretrained.vocab")

        train_loader = train_dataset.asDataLoader(TL_BATCH_SIZE)
        valid_loader = test_dataset.asDataLoader(TL_BATCH_SIZE)

        return train_loader, valid_loader, test_dataset

    def __transfer_learning(self, train_loader: DataLoader, valid_loader: DataLoader, model_prefix: Path):
        finetuned = generators.GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        finetuned.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.pkg")

        monitor = monitors.FileMonitor(model_prefix, save_smiles=True, reset_directory=True)
        finetuned.fit(
            train_loader,
            valid_loader,
            epochs=TL_EPOCHS,
            monitor=monitor,
            loss_tolerance=TL_LOSS_TOLERANCE,
        )

        self.VOC.toFile(self.MODEL / "finetuned.vocab")

        return finetuned

    def TRANSFER_LEARNING(self, molecule_smile: str):
        smiles_train = self.__receptor_similar_molecules_get_SMILES(molecule_smile)
        train_loader, valid_loader, test_dataset = self.__fragmentation_and_encoding(smiles_train)

        model_prefix = self.MODEL / "david_finetuned"
        finetuned = self.__transfer_learning(train_loader, valid_loader, model_prefix)
        generated_sample = DrugExVisualizer.visualize_transfer_learning(
            finetuned, model_prefix, self.MODEL, input_dataset=test_dataset
        )

        return finetuned, generated_sample, self.OUTPUT_DIR


class RAINFORCMENT_LEARNING(ALL_METHODS):
    def __init__(self, finetuned: GraphTransformer, out_dir: Path):
        super().__init__(output_dir=out_dir)
        self.MODEL = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        self.VOC = finetuned.voc_trg if hasattr(finetuned, "voc_trg") else vocabulary.VocGraph.fromFile(
            self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.vocab"
        )

        self.AGENT = finetuned
        self.PRIOR = generators.GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.PRIOR.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_graph_trans_PT.pkg")

    def _prepare_seed_fragments(self, molecule_smile: str):
        seed_tsv = self.MODEL / "seed_fragments.tsv"
        seed_dataset = datasets.GraphFragDataSet(str(seed_tsv), rewrite=True)
        corpus_encoder = fragments.FragmentCorpusEncoder(
            fragmenter=fragmenters.Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, FRAG_METHOD),
            encoder=fragments.GraphFragmentEncoder(self.VOC),
            pairs_splitter=None,
            n_proc=1,
        )

        funnel_path = self.ROOT_PATH / "_david" / "funnel_scaffolds.tsv"
        if funnel_path.exists():
            df_funnel = pd.read_csv(funnel_path, sep="\t")
            if "Scaffold" in df_funnel.columns and len(df_funnel) > 0:
                scaffolds = df_funnel["Scaffold"].dropna().unique().tolist()
                std_scaffolds = processing.Standardization(n_proc=1).apply(scaffolds)
                if std_scaffolds:
                    corpus_encoder.apply(std_scaffolds, encodingCollectors=[seed_dataset])
                    seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
                    return seed_loader, seed_dataset

        standardized_smile = processing.Standardization(n_proc=1).apply([molecule_smile])[0]
        corpus_encoder.apply([standardized_smile], encodingCollectors=[seed_dataset])
        seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
        return seed_loader, seed_dataset

    def RAINFORCEMENT_LEARNING(self, molecule_smile: str):
        ref_sdf = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        if not ref_sdf.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_sdf.parent)
            ref_sdf = preparer.generate_reference_sdf(molecule_smile, ref_sdf)

        conf_gen = conformer_generators.RDKitConformerGenerator(
            max_conformers=ROCS_MAX_CONFS,
            max_isomers=ROCS_MAX_ISOMERS,
            num_threads=0,
            timeout=ROCS_TIMEOUT,
            show_progress=False,
        )
        rocs_scorer = rocs_rdkit.RDKitROCSScorer(
            conformer_generator=conf_gen,
            references=str(ref_sdf),
            score_type=ROCS_SCORE_TYPE,
            use_colors=True,
            show_progress=False,
            n_jobs=-1,
        )
        rocs_scorer.setModifier(
            SmoothClippedScore(
                lower_x=ROCS_LOWER_X,
                upper_x=ROCS_UPPER_X,
            )
        )

        scorers = [
            rocs_scorer,
            RetrosyntheticAccessibilityScorer(
                modifier=SmoothClippedScore(
                    lower_x=RA_LOWER_X,
                    upper_x=RA_UPPER_X,
                ),
            ),
            Property(
                prop="QED",
                modifier=SmoothClippedScore(
                    lower_x=QED_LOWER_X,
                    upper_x=QED_UPPER_X,
                ),
            ),
            Property(
                prop="SA",
                modifier=SmoothClippedScore(
                    lower_x=SA_LOWER_X,
                    upper_x=SA_UPPER_X,
                ),
            ),
        ]

        thresholds = [ROCS_THRESHOLD, RA_THRESHOLD, QED_THRESHOLD, SA_THRESHOLD]
        reward_scheme = rewards.ParetoTanimotoDistance()

        env = DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=reward_scheme,
        )

        seed_loader, seed_dataset = self._prepare_seed_fragments(molecule_smile)

        explorer = FragGraphExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=RL_EPSILON,
            beta=RL_BETA,
            batch_size=RL_BATCH_SIZE,
            use_gpus=GPUS,
        )

        model_prefix = self.MODEL / "david_agent"
        monitor = FileMonitor(model_prefix, save_smiles=True, reset_directory=True)

        explorer.fit(
            seed_loader,
            seed_loader,
            monitor=monitor,
            epochs=RL_EPOCHS,
            patience=RL_PATIENCE,
            criteria="desired_ratio",
        )

        scored_df = DrugExVisualizer.visualize_reinforcement_learning(
            self.AGENT, env, model_prefix, self.MODEL, input_dataset=seed_dataset
        )
        return self.AGENT, scored_df


if __name__ == "__main__":
    my_molecule = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

    TL = MY_TRANSFER_LEARNING()
    finetuned, generated_sample, OUT_DIR = TL.TRANSFER_LEARNING(my_molecule)

    RL = RAINFORCMENT_LEARNING(finetuned, OUT_DIR)
    agent, scored_sample = RL.RAINFORCEMENT_LEARNING(my_molecule)
