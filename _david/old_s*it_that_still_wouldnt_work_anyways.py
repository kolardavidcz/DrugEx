from __future__ import annotations

# --- Standard Library ---
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# --- Numerical Data, Cheminformatics & PyTorch ---
import matplotlib
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")

# --- Plotting, QSPRPred & Chemical Space Visualization ---
import matplotlib.pyplot as plt
from qsprpred import data as qspr_data  # noqa: F401
from qsprpred.data.descriptors import fingerprints as qspr_fps  # noqa: F401
from qsprpred.models import scikit_learn as qspr_models  # noqa: F401
from scaffviz.clustering import manifold  # noqa: F401
from scaffviz.depiction import plot as scaff_plot  # noqa: F401

# --- DrugEx Core & Submodule Namespaces ---
import drugex  # noqa: F401
from drugex import data, logs, molecules, parallel, training, utils  # noqa: F401
from drugex.data import datasets, fragments, processing  # noqa: F401
from drugex.data.corpus import corpus, vocabulary
from drugex.logs import logger
from drugex.molecules import converters, mol, suppliers  # noqa: F401
from drugex.molecules.converters import default as default_converters  # noqa: F401
from drugex.molecules.converters import standardizers  # noqa: F401
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.scorers import (
    conformer_generators,  # noqa: F401
    modifiers,
    oe_color_smarts,  # noqa: F401
    properties,
    protonation,  # noqa: F401
    qsprpred,  # noqa: F401
    sascorer,  # noqa: F401
    similarity,
    smiles as smiles_scorers,  # noqa: F401
)

# --- DrugEx Direct Class Imports (convenience: namespace.Class or bare Class) ---
from drugex.data.corpus.vocabulary import VocGraph, VocSmiles  # noqa: F401
from drugex.data.datasets import GraphFragDataSet, SmilesDataSet, SmilesFragDataSet  # noqa: F401
from drugex.data.processing import CorpusEncoder, RandomTrainTestSplitter, Standardization
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer, FragSequenceExplorer, SequenceExplorer  # noqa: F401
from drugex.training.generators import GraphTransformer, SequenceRNN, SequenceTransformer  # noqa: F401
from drugex.training.monitors import FileMonitor, NullMonitor  # noqa: F401
from drugex.training.rewards import ParetoCrowdingDistance, ParetoRewardScheme, ParetoTanimotoDistance, WeightedSum  # noqa: F401
from drugex.training.scorers.modifiers import ClippedScore, ScoreModifier, SmoothClippedScore  # noqa: F401
from drugex.training.scorers.properties import Property, Uniqueness  # noqa: F401
from drugex.training.scorers.qsprpred import QSPRPredScorer  # noqa: F401
from drugex.training.scorers.similarity import FraggleSimilarity, TverskyFingerprintSimilarity, TverskyGraphSimilarity  # noqa: F401

# --- Local Project Modules ---
from _david import receptor_similar

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")


# =====================================================================
# Exceptions
# =====================================================================

class PipelineExecutionError(RuntimeError):
    """Raised when an unrecoverable failure occurs in the DrugEx pipeline."""

    def __init__(self, message: str = "Unrecoverable error in DrugEx execution pipeline.") -> None:
        super().__init__(message)


# Backwards compatibility alias
MY_ERROR = PipelineExecutionError


# =====================================================================
# Configuration Structs (Strongly-Typed Dataclasses)
# =====================================================================

@dataclass(frozen=True)
class PipelineHardwareConfig:
    """Execution resources, chunking, and diagnostic flags.

    Parameters
    ----------
    n_processes : int
        Number of CPU worker cores for parallel standardization and encoding.
    chunk_size : int
        Batch chunk size per CPU process to regulate memory usage.
    gpus : Tuple[int, ...]
        List/tuple of GPU device IDs to allocate for PyTorch models.
    save_proof : bool
        Whether to dump intermediate TSV/vocab artifacts for inspection.
    """

    n_processes: int = 12
    chunk_size: int = 1000
    gpus: Tuple[int, ...] = (1,)
    save_proof: bool = False


@dataclass(frozen=True)
class TransferLearningConfig:
    """Hyperparameters governing SMILES-RNN Transfer Learning (fine-tuning).

    Parameters
    ----------
    epochs : int
        Number of fine-tuning epochs through the target ligand corpus.
    batch_size : int
        Mini-batch size for DataLoader gradient updates.
    loss_tolerance : float
        Early stopping threshold on validation loss convergence.
    test_size : float
        Fraction of curated ligands reserved for the validation split.
    max_target_molecules : int
        Maximum number of similar active ligands to curate from Papyrus.
    """

    epochs: int = 20
    batch_size: int = 32
    loss_tolerance: float = 0.02
    test_size: float = 0.10
    max_target_molecules: int = 200


@dataclass(frozen=True)
class TverskyScorerConfig:
    """Configuration for Tversky fingerprint structural similarity.

    Parameters
    ----------
    alpha : float
        Penalty weight for reference features missing in the query candidate.
    beta : float
        Penalty weight for extra/new features added to the query candidate.
    fp_type : str
        Fingerprint representation identifier (e.g. 'ECFP4', 'FCFP4').
    threshold : float
        Desirability threshold for multi-objective Pareto optimization.
    """

    alpha: float = 0.70
    beta: float = 0.30
    fp_type: str = "ECFP4"
    threshold: float = 0.40


@dataclass(frozen=True)
class SmoothClippedScorerConfig:
    """Configuration for smooth sigmoidal property desirabilities.

    Parameters
    ----------
    lower_x : float
        Value where score approaches zero (or start of sigmoid decline).
    upper_x : float
        Value where score approaches unity (or peak desirability).
    threshold : float
        Desirability threshold for multi-objective Pareto optimization.
    """

    lower_x: float
    upper_x: float
    threshold: float = 0.50


@dataclass(frozen=True)
class ReinforcementLearningConfig:
    """Hyperparameters and objective specifications for DrugEx RL (SequenceExplorer).

    Parameters
    ----------
    epochs : int
        Maximum number of exploration epochs.
    epsilon : float
        Mutation exploration rate (sampling from frozen prior).
    beta : float
        Entropy baseline subtraction factor in REINFORCE.
    n_samples : int
        Number of candidate molecules generated per epoch.
    batch_size : int
        Mini-batch size for policy gradient parameter updates.
    patience : int
        Early stopping patience based on criteria performance.
    tversky : TverskyScorerConfig
        Configuration for reference lead fingerprint similarity.
    qed : SmoothClippedScorerConfig
        Configuration for Quantitative Estimate of Drug-likeness.
    sa : SmoothClippedScorerConfig
        Configuration for Synthetic Accessibility scoring.
    """

    epochs: int = 100
    epsilon: float = 0.10
    beta: float = 0.00
    n_samples: int = 1000
    batch_size: int = 128
    patience: int = 20
    tversky: TverskyScorerConfig = field(default_factory=TverskyScorerConfig)
    qed: SmoothClippedScorerConfig = field(
        default_factory=lambda: SmoothClippedScorerConfig(lower_x=0.40, upper_x=0.80, threshold=0.50)
    )
    sa: SmoothClippedScorerConfig = field(
        default_factory=lambda: SmoothClippedScorerConfig(lower_x=4.50, upper_x=2.50, threshold=0.50)
    )

    @property
    def thresholds(self) -> List[float]:
        """Consolidated list of objective thresholds matching environment scorers."""
        return [self.tversky.threshold, self.qed.threshold, self.sa.threshold]


# =====================================================================
# Visualization & Depiction Utilities
# =====================================================================

class DrugExVisualizer:
    """Static plotting and chemical structure depiction utilities."""

    @staticmethod
    @contextmanager
    def save_plot(output_path: Path | str, **subplots_kwargs: Any) -> Generator[plt.Axes, None, None]:
        """Context manager that automatically saves and closes a matplotlib figure.

        Parameters
        ----------
        output_path : Path or str
            Destination path for the PNG file.
        **subplots_kwargs : Any
            Forwarded arguments to `plt.subplots` (e.g. figsize=(10, 4)).

        Yields
        ------
        matplotlib.axes.Axes
            Active axes instance for plotting.
        """
        fig, ax = plt.subplots(**subplots_kwargs)
        try:
            yield ax
        finally:
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def save_molecules_grid(generated_sample: DataFrame, out_dir: Path, num_mols: int = 25) -> None:
        """Render and save a 2D chemical structure grid of generated molecules.

        Parameters
        ----------
        generated_sample : DataFrame
            DataFrame containing a 'SMILES' column.
        out_dir : Path
            Destination directory for the PNG file.
        num_mols : int, optional
            Maximum number of structures to render (default: 25).

        Raises
        ------
        PipelineExecutionError
            If no valid molecules can be parsed from the provided SMILES list.
        """
        mols = [
            m for s in generated_sample["SMILES"][:num_mols]
            if (m := Chem.MolFromSmiles(s)) is not None
        ]
        if not mols:
            raise PipelineExecutionError("No valid SMILES strings could be parsed to generate a 2D molecule grid.")

        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=[250, 250])
        img.save(str(out_dir / "generated_molecules.png"))

    @staticmethod
    def visualize_transfer_learning(
        finetuned: generators.SequenceRNN,
        model_prefix: Path,
        out_dir: Path,
        num_samples: int = 100
    ) -> DataFrame:
        """Plot loss curves and generate sample validation molecules for fine-tuned models.

        Parameters
        ----------
        finetuned : generators.SequenceRNN
            Trained fine-tuned SequenceRNN generator.
        model_prefix : Path
            Prefix for monitoring logs (e.g. `out_dir / "david_finetuned"`).
        out_dir : Path
            Destination directory for figures and TSV outputs.
        num_samples : int, optional
            Number of molecules to sample from the model (default: 100).

        Returns
        -------
        DataFrame
            Sample of generated molecules.
        """
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            performance_for_epoch: DataFrame = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "loss_valid", "valid_ratio"] if c in performance_for_epoch.columns]
            with DrugExVisualizer.save_plot(out_dir / "training_loss.png") as ax:
                performance_for_epoch[metric_cols].plot.line(logy=True, ax=ax)
                ax.set_title("Transfer Learning Loss & Validity Dynamics")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Log Value")
                ax.grid(True, linestyle="--", alpha=0.6)

        generated_sample: DataFrame = finetuned.generate(num_samples=num_samples)
        generated_sample.to_csv(out_dir / "generated_molecules.tsv", sep="\t", index=False)
        DrugExVisualizer.save_molecules_grid(generated_sample, out_dir)
        return generated_sample

    @staticmethod
    def visualize_reinforcement_learning(
        agent: generators.SequenceRNN,
        env: DrugExEnvironment,
        model_prefix: Path,
        out_dir: Path,
        num_samples: int = 100
    ) -> DataFrame:
        """Visualize RL training curves, sample molecules from trained agent, score and depict them.

        Parameters
        ----------
        agent : generators.SequenceRNN
            Trained RL sequence RNN agent.
        env : DrugExEnvironment
            DrugEx RL environment containing configured scorers.
        model_prefix : Path
            Base prefix for model monitor logs (e.g. `out_dir / "david_agent"`).
        out_dir : Path
            Output directory for saving plots, TSVs, and depiction images.
        num_samples : int, optional
            Number of candidate molecules to generate (default: 100).

        Returns
        -------
        DataFrame
            Table of generated molecules annotated with individual objective scores.
        """
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            df_info: DataFrame = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
            with DrugExVisualizer.save_plot(out_dir / "rl_training_curves.png") as ax:
                df_info[metric_cols].plot.line(ax=ax)
                ax.set_title("Reinforcement Learning Training Dynamics")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Metric Value")
                ax.grid(True, linestyle="--", alpha=0.6)

        generated_sample: DataFrame = agent.generate(num_samples=num_samples)
        scores: DataFrame = env.getScores(generated_sample["SMILES"])
        scored_df: DataFrame = pd.concat(
            [generated_sample.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1
        )
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        DrugExVisualizer.save_molecules_grid(generated_sample, out_dir)
        return scored_df


# =====================================================================
# Base Workflow Infrastructure
# =====================================================================

class BaseWorkflow:
    """Base manager handling workspace paths, output tracking, and pre-trained models.

    Parameters
    ----------
    output_dir : Path or None, optional
        Pre-existing output directory to attach to, or None to generate a new timestamped run.
    hardware_config : PipelineHardwareConfig or None, optional
        Hardware, multiprocessing, and GPU configuration.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        hardware_config: Optional[PipelineHardwareConfig] = None
    ) -> None:
        self.hardware: PipelineHardwareConfig = hardware_config or PipelineHardwareConfig()

        # Workspace Root & Base Output Directory
        self.ROOT_PATH: Path = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "outputs"

        # Timestamped Run Directory
        if output_dir is None:
            self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.OUTPUT_DIR: Path = self.BASE_OUTPUT_DIR / self.RUN_ID
            self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        else:
            self.OUTPUT_DIR = output_dir
            self.RUN_ID = output_dir.name

        # Pre-trained Model Checkpoint & Vocabulary
        self.MODEL_PRETRAINED: Path = self.ROOT_PATH / "data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT"
        self.VOC: VocSmiles = vocabulary.VocSmiles.fromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.vocab")

    # Delegate visualizer methods for backward compatibility
    molecules_save_image = staticmethod(DrugExVisualizer.save_molecules_grid)
    MY_save_plot = staticmethod(DrugExVisualizer.save_plot)
    vizualize = staticmethod(DrugExVisualizer.visualize_transfer_learning)
    vizualize_rl = staticmethod(DrugExVisualizer.visualize_reinforcement_learning)


# Backwards compatibility alias
ALL_METHODS = BaseWorkflow


# =====================================================================
# Phase 1: Transfer Learning Workflow
# =====================================================================

class TransferLearningWorkflow(BaseWorkflow):
    """Orchestrates Phase 1: Bioactivity curation, standardization, and fine-tuning.

    Parameters
    ----------
    output_dir : Path or None, optional
        Shared output directory for the run.
    config : TransferLearningConfig or None, optional
        Hyperparameters for Transfer Learning.
    hardware_config : PipelineHardwareConfig or None, optional
        Hardware and multiprocessing configuration.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config: Optional[TransferLearningConfig] = None,
        hardware_config: Optional[PipelineHardwareConfig] = None
    ) -> None:
        super().__init__(output_dir=output_dir, hardware_config=hardware_config)
        self.config: TransferLearningConfig = config or TransferLearningConfig()

        self.MODEL: Path = self.OUTPUT_DIR / "transfer_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        self.PRETRAINED: generators.SequenceRNN = generators.SequenceRNN(
            self.VOC,
            is_lstm=True,
            use_gpus=self.hardware.gpus
        )
        self.PRETRAINED.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.pkg")

    def extract_target_ligands(self, molecule_smile: str) -> List[str]:
        """Harvest active, property-profiled ligands for the query targets from Papyrus.

        Parameters
        ----------
        molecule_smile : str
            Query reference SMILES (e.g. EGCG).

        Returns
        -------
        List[str]
            Standardized SMILES strings for fine-tuning.
        """
        pipeline = receptor_similar.MoleculeBioactivityPipeline()
        result_df: DataFrame = pipeline.run(
            molecule_smile,
            max_molecules=self.config.max_target_molecules
        ).molecules

        n_molecules = len(result_df)
        if n_molecules == 0:
            raise PipelineExecutionError("No active target ligands discovered across Papyrus bioactivity database.")

        smiles_series: Series = result_df["SMILES"]
        standardized_smiles = processing.Standardization(
            n_proc=self.hardware.n_processes,
            chunk_size=self.hardware.chunk_size
        ).apply(smiles_series)

        if standardized_smiles is None:
            raise PipelineExecutionError("Chemical standardization failed to process target ligands.")

        clean_smiles: List[str] = [str(s) for s in standardized_smiles if s]

        if self.hardware.save_proof:
            pd.Series(clean_smiles, name="SMILES").to_csv(
                self.MODEL / "training_ligands.tsv", sep="\t", index=False
            )

        return clean_smiles

    def tokenize_corpus(self, smiles_train: Sequence[str]) -> Tuple[datasets.SmilesDataSet, processing.CorpusEncoder]:
        """Tokenize target ligand SMILES using the pre-trained vocabulary.

        Parameters
        ----------
        smiles_train : Sequence[str]
            Cleaned SMILES sequences to tokenize.

        Returns
        -------
        Tuple[datasets.SmilesDataSet, processing.CorpusEncoder]
            Corpus data collector and encoder instances.
        """
        encoder = processing.CorpusEncoder(
            corpus.SequenceCorpus,
            {
                "vocabulary": self.VOC,
                "update_voc": False,
                "throw": True,
            },
            n_proc=self.hardware.n_processes,
            chunk_size=self.hardware.chunk_size
        )
        data_collector = datasets.SmilesDataSet(self.OUTPUT_DIR / "ligand_corpus.tsv", rewrite=True)
        encoder.apply(smiles_train, collector=data_collector)
        return data_collector, encoder

    def create_dataloaders(self, data_collector: datasets.SmilesDataSet) -> Tuple[DataLoader, DataLoader]:
        """Split tokenized corpus into training and validation DataLoaders.

        Parameters
        ----------
        data_collector : datasets.SmilesDataSet
            Tokenized SMILES dataset.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            Training and validation PyTorch DataLoaders.
        """
        batch_size = self.config.batch_size

        if self.hardware.save_proof:
            splitter = processing.RandomTrainTestSplitter(self.config.test_size, 10000)
            train, test = splitter(data_collector.getData())

            pd.DataFrame(train, columns=data_collector.getColumns()).to_csv(
                self.OUTPUT_DIR / "ligand_train.tsv", header=True, index=False, sep="\t"
            )
            pd.DataFrame(test, columns=data_collector.getColumns()).to_csv(
                self.OUTPUT_DIR / "ligand_test.tsv", header=True, index=False, sep="\t"
            )
            self.VOC.toFile(self.OUTPUT_DIR / "pretrained.vocab")

            train_loader = datasets.SmilesDataSet.dataToLoader(train, batch_size=batch_size, vocabulary=self.VOC)
            valid_loader = datasets.SmilesDataSet.dataToLoader(test, batch_size=batch_size, vocabulary=self.VOC)
            return train_loader, valid_loader

        loaders = data_collector.asDataLoader(
            batch_size=batch_size,
            splitter=processing.RandomTrainTestSplitter(self.config.test_size, 10000),
        )
        if not isinstance(loaders, (list, tuple)) or len(loaders) != 2:
            raise PipelineExecutionError("Expected exactly two DataLoaders (train, valid) from dataset splitter.")

        return loaders[0], loaders[1]

    def fit_transfer_learning(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_prefix: Path
    ) -> generators.SequenceRNN:
        """Execute transfer learning on SequenceRNN and checkpoint weights.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        valid_loader : DataLoader
            Validation data loader.
        model_prefix : Path
            File prefix for training checkpoints and TSV progress logs.

        Returns
        -------
        generators.SequenceRNN
            Fine-tuned generator.
        """
        finetuned = generators.SequenceRNN(
            self.VOC,
            is_lstm=True,
            use_gpus=self.hardware.gpus
        )
        finetuned.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.pkg")

        monitor = monitors.FileMonitor(model_prefix, save_smiles=True, reset_directory=True)
        finetuned.fit(
            train_loader,
            valid_loader,
            epochs=self.config.epochs,
            monitor=monitor,
            loss_tolerance=self.config.loss_tolerance,
        )

        if self.hardware.save_proof:
            self.VOC.toFile(self.MODEL / "finetuned.vocab")

        return finetuned

    def run(self, molecule_smile: str) -> Tuple[generators.SequenceRNN, DataFrame, Path]:
        """Execute the end-to-end Transfer Learning workflow.

        Parameters
        ----------
        molecule_smile : str
            Query reference SMILES.

        Returns
        -------
        Tuple[generators.SequenceRNN, DataFrame, Path]
            Fine-tuned model, sample generated molecules DataFrame, and shared output directory path.
        """
        smiles_train = self.extract_target_ligands(molecule_smile)
        data_collector, _ = self.tokenize_corpus(smiles_train)
        train_loader, valid_loader = self.create_dataloaders(data_collector)

        model_prefix = self.MODEL / "david_finetuned"
        finetuned = self.fit_transfer_learning(train_loader, valid_loader, model_prefix)
        generated_sample = DrugExVisualizer.visualize_transfer_learning(finetuned, model_prefix, self.MODEL)

        return finetuned, generated_sample, self.OUTPUT_DIR

    # Backwards compatibility alias
    TRANSFER_LEARNING = run


# Backwards compatibility alias
MY_TRANSFER_LEARNING = TransferLearningWorkflow


# =====================================================================
# Phase 2: Reinforcement Learning Workflow
# =====================================================================

class ReinforcementLearningWorkflow(BaseWorkflow):
    """Orchestrates Phase 2: Pareto Multi-Objective Reinforcement Learning (SequenceExplorer).

    Parameters
    ----------
    finetuned : SequenceRNN
        Fine-tuned generator resulting from Phase 1.
    output_dir : Path
        Shared run directory inherited from Phase 1.
    config : ReinforcementLearningConfig or None, optional
        Hyperparameters and multi-objective thresholds for RL.
    hardware_config : PipelineHardwareConfig or None, optional
        Hardware and GPU resource allocation.
    """

    def __init__(
        self,
        finetuned: SequenceRNN,
        output_dir: Path,
        config: Optional[ReinforcementLearningConfig] = None,
        hardware_config: Optional[PipelineHardwareConfig] = None
    ) -> None:
        super().__init__(output_dir=output_dir, hardware_config=hardware_config)
        self.config: ReinforcementLearningConfig = config or ReinforcementLearningConfig()

        self.MODEL: Path = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL.mkdir(parents=True, exist_ok=True)

        self.VOC = finetuned.voc if hasattr(finetuned, "voc") else vocabulary.VocSmiles.fromFile(
            self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.vocab"
        )

        # Agent: Policy network being optimized
        self.AGENT: SequenceRNN = finetuned

        # Prior: Independent, frozen reference network used for mutation exploration
        self.PRIOR: SequenceRNN = generators.SequenceRNN(
            self.VOC,
            is_lstm=True,
            use_gpus=self.hardware.gpus
        )
        self.PRIOR.loadStatesFromFile(self.MODEL_PRETRAINED / "Papyrus05.5_smiles_rnn_PT.pkg")

    def build_environment(self, reference_smiles: str) -> DrugExEnvironment:
        """Construct the Pareto multi-objective optimization environment.

        Parameters
        ----------
        reference_smiles : str
            Query reference SMILES for fingerprint similarity (e.g. EGCG).

        Returns
        -------
        DrugExEnvironment
            Environment configured with Tversky, QED, and SA scorers.
        """
        scorers = [
            # 1. Structural similarity to reference lead
            similarity.TverskyFingerprintSimilarity(
                smiles=reference_smiles,
                fp_type=self.config.tversky.fp_type,
                alpha=self.config.tversky.alpha,
                beta=self.config.tversky.beta,
            ),
            # 2. Oral drug-likeness (QED)
            Property(
                prop="QED",
                modifier=SmoothClippedScore(
                    lower_x=self.config.qed.lower_x,
                    upper_x=self.config.qed.upper_x,
                ),
            ),
            # 3. Synthetic accessibility (SA)
            Property(
                prop="SA",
                modifier=SmoothClippedScore(
                    lower_x=self.config.sa.lower_x,
                    upper_x=self.config.sa.upper_x,
                ),
            ),
        ]

        reward_scheme = rewards.ParetoCrowdingDistance()
        return DrugExEnvironment(
            scorers=scorers,
            thresholds=self.config.thresholds,
            reward_scheme=reward_scheme,
        )

    def run(self, molecule_smile: str) -> Tuple[generators.SequenceRNN, DataFrame]:
        """Execute Reinforcement Learning exploration and Pareto lead evaluation.

        Parameters
        ----------
        molecule_smile : str
            Query reference SMILES.

        Returns
        -------
        Tuple[generators.SequenceRNN, DataFrame]
            Trained RL agent and DataFrame of scored candidate molecules.
        """
        env = self.build_environment(molecule_smile)

        explorer = SequenceExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=self.config.epsilon,
            beta=self.config.beta,
            n_samples=self.config.n_samples,
            batch_size=self.config.batch_size,
            use_gpus=self.hardware.gpus,
        )

        model_prefix = self.MODEL / "david_agent"
        monitor = FileMonitor(model_prefix, save_smiles=True, reset_directory=True)

        explorer.fit(
            monitor=monitor,
            epochs=self.config.epochs,
            patience=self.config.patience,
            criteria="desired_ratio"
        )

        scored_df = DrugExVisualizer.visualize_reinforcement_learning(
            self.AGENT,
            env,
            model_prefix,
            self.MODEL
        )
        return self.AGENT, scored_df

    # Backwards compatibility alias
    RAINFORCEMENT_LEARNING = run


# Backwards compatibility alias
RAINFORCMENT_LEARNING = ReinforcementLearningWorkflow


# =====================================================================
# Main Execution Entrypoint
# =====================================================================

if __name__ == "__main__":
    query_molecule: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

    # Phase 1: Transfer Learning
    tl_workflow = TransferLearningWorkflow()
    finetuned_agent, tl_samples, run_dir = tl_workflow.run(query_molecule)

    # Phase 2: Reinforcement Learning
    rl_workflow = ReinforcementLearningWorkflow(finetuned=finetuned_agent, output_dir=run_dir)
    final_agent, scored_leads = rl_workflow.run(query_molecule)
