from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import warnings

#? [Matplotlib] Non-interactive Agg backend: disables GUI window requirement for headless server execution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import torch

from drugex.data.corpus import vocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesDataSet
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoTanimotoDistance
from drugex.training.scorers import conformer_generators, modifiers, properties, ra_scorer, rocs_rdkit
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
from drugex.training.scorers.similarity import TverskyFingerprintSimilarity

warnings.filterwarnings("ignore")


#? [PyTorch CUDA] Lightweight device probe: tests tensor allocation on GPU 0 to safely catch driver faults and fall back to CPU.
def get_safe_gpus():
    if not torch.cuda.is_available():
        return tuple()
    try:
        t = torch.zeros(1, device="cuda:0")
        del t
        return (0,)
    except Exception:
        return tuple()


GPUS = get_safe_gpus()

MY_MOLECULE = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

RL_EPOCHS = 100
RL_EPSILON = 0.10
RL_BETA = 0.00
RL_N_SAMPLES = 200
RL_BATCH_SIZE = 64
RL_PATIENCE = 20

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

QED_LOWER_X = 0.30
QED_UPPER_X = 0.70
QED_THRESHOLD = 0.50

SA_LOWER_X = 4.50
SA_UPPER_X = 2.50
SA_THRESHOLD = 0.50


class SequenceRNNOptimizer:
    """Stage 1 Generative Optimizer executing policy-gradient RL on a SequenceRNN model.

    Optimizes an autoregressive LSTM language model pretrained on Papyrus 05.5
    under multi-objective rewards including 3D ROCS shape/color similarity,
    retrosynthetic accessibility (RAscore), inverted 2D novelty, QED, and SA.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.ROOT_PATH = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR = self.ROOT_PATH / "_david" / "outputs_RNN"

        if output_dir is None:
            self.RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.OUTPUT_DIR = self.BASE_OUTPUT_DIR / self.RUN_ID
        else:
            self.OUTPUT_DIR = Path(output_dir)
            self.RUN_ID = self.OUTPUT_DIR.name
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.MODEL_DIR = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL = self.MODEL_DIR

        self.PRETRAINED_DIR = self.ROOT_PATH / "data" / "models" / "pretrained" / "smiles-rnn" / "Papyrus05.5_smiles_rnn_PT"
        self.MODEL_PRETRAINED = self.PRETRAINED_DIR

        self.VOC = VocSmiles.fromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.vocab")
        self.AGENT = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        self.AGENT.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.pkg")

        self.PRIOR = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        self.PRIOR.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.pkg")

    def build_environment(self, ref_sdf_path: Path | str, seed_smiles: str = MY_MOLECULE) -> DrugExEnvironment:
        """Construct multi-objective DrugExEnvironment for reinforcement learning.

        Args:
            ref_sdf_path: Path to reference multi-conformer 3D SDF for ROCS.
            seed_smiles: SMILES string of active seed query for scaffold hopping.

        Returns:
            Configured DrugExEnvironment with ParetoTanimotoDistance reward scheme.
        """
        ref_path_str = str(ref_sdf_path)

        conf_gen = RDKitConformerGenerator(
            max_conformers=ROCS_MAX_CONFS,
            max_isomers=ROCS_MAX_ISOMERS,
            num_threads=0,
            timeout=ROCS_TIMEOUT,
            show_progress=False,
        )
        rocs_scorer = RDKitROCSScorer(
            conformer_generator=conf_gen,
            references=ref_path_str,
            score_type=ROCS_SCORE_TYPE,
            use_colors=True,
            show_progress=False,
            n_jobs=4,
        )
        rocs_scorer.setModifier(
            SmoothClippedScore(
                lower_x=ROCS_LOWER_X,
                upper_x=ROCS_UPPER_X,
            )
        )

        ra_scorer_inst = RetrosyntheticAccessibilityScorer(
            modifier=SmoothClippedScore(
                lower_x=RA_LOWER_X,
                upper_x=RA_UPPER_X,
            )
        )

        #? [Inverted 2D Novelty] ECFP6 Tversky (alpha=beta=1.0) with inverted sigmoid to penalize seed duplication and force scaffold hopping.
        novelty_scorer = TverskyFingerprintSimilarity(
            smiles=seed_smiles,
            fp_type="ECFP6",
            alpha=1.0,
            beta=1.0,
            modifier=SmoothClippedScore(
                lower_x=0.60,
                upper_x=0.25,
            ),
        )

        qed_scorer = Property(
            prop="QED",
            modifier=SmoothClippedScore(
                lower_x=QED_LOWER_X,
                upper_x=QED_UPPER_X,
            )
        )

        sa_scorer = Property(
            prop="SA",
            modifier=SmoothClippedScore(
                lower_x=SA_LOWER_X,
                upper_x=SA_UPPER_X,
            )
        )

        scorers = [rocs_scorer, ra_scorer_inst, novelty_scorer, qed_scorer, sa_scorer]
        thresholds = [ROCS_THRESHOLD, RA_THRESHOLD, 0.50, QED_THRESHOLD, SA_THRESHOLD]
        reward_scheme = ParetoTanimotoDistance()

        env = DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=reward_scheme,
        )
        return env

    def run_reinforcement_learning(
        self, molecule_smile: str = MY_MOLECULE, epochs: int = RL_EPOCHS, n_samples: int = RL_N_SAMPLES
    ) -> tuple[SequenceRNN, pd.DataFrame]:
        """Execute policy-gradient reinforcement learning using SequenceExplorer.

        Args:
            molecule_smile: SMILES string of active query molecule.
            epochs: Total policy gradient training epochs.
            n_samples: Trajectories sampled per epoch.

        Returns:
            Tuple containing trained SequenceRNN agent and scored candidates DataFrame.
        """
        ref_sdf = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        if not ref_sdf.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_sdf.parent)
            ref_sdf = preparer.generate_reference_sdf(molecule_smile, ref_sdf)

        env = self.build_environment(ref_sdf, seed_smiles=molecule_smile)

        explorer = SequenceExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=RL_EPSILON,
            beta=RL_BETA,
            n_samples=n_samples,
            batch_size=RL_BATCH_SIZE,
            use_gpus=GPUS,
        )

        model_prefix = self.MODEL_DIR / "david_agent"
        monitor = FileMonitor(model_prefix, save_smiles=True, reset_directory=True)

        patience = min(RL_PATIENCE, max(5, epochs // 2))
        explorer.fit(
            monitor=monitor,
            epochs=epochs,
            patience=patience,
            criteria="desired_ratio",
        )

        scored_df = self.save_visualizations(self.AGENT, env, model_prefix, self.MODEL_DIR, num_samples=n_samples)
        return self.AGENT, scored_df

    def RAINFORCEMENT_LEARNING(self, molecule_smile: str, epochs: int | None = None, n_samples: int | None = None):
        ep = epochs if epochs is not None else RL_EPOCHS
        ns = n_samples if n_samples is not None else RL_N_SAMPLES
        return self.run_reinforcement_learning(molecule_smile=molecule_smile, epochs=ep, n_samples=ns)

    def save_visualizations(
        self, agent: SequenceRNN, env: DrugExEnvironment, model_prefix: Path, out_dir: Path, num_samples: int = 100
    ) -> pd.DataFrame:
        """Generate candidates, evaluate multi-objective scores, and save diagnostic plots.

        Args:
            agent: Trained SequenceRNN generator.
            env: Multi-objective DrugExEnvironment.
            model_prefix: File prefix where FileMonitor logged training data.
            out_dir: Directory where plots, tables, and depiction grids are stored.
            num_samples: Number of de novo SMILES to sample.

        Returns:
            DataFrame of scored generated candidates.
        """
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            #? [Pandas] Parses TSV training monitor log file into DataFrame.
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
            #? [Matplotlib] Renders multi-metric RL training dynamics convergence curves.
            fig, ax = plt.subplots(figsize=(8, 5))
            df_info[metric_cols].plot.line(ax=ax)
            ax.set_title("Sequence-RNN Reinforcement Learning Training Dynamics", weight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.savefig(out_dir / "rl_training_curves.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

        generated_sample = agent.generate(num_samples=num_samples)
        scores = env.getScores(generated_sample["SMILES"])
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        scored_df.to_csv(out_dir / "stage1_hits.tsv", sep="\t", index=False)
        if self.OUTPUT_DIR != out_dir:
            scored_df.to_csv(self.OUTPUT_DIR / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
            scored_df.to_csv(self.OUTPUT_DIR / "stage1_hits.tsv", sep="\t", index=False)

        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        mols = [m for s in generated_sample["SMILES"][:25] if (m := Chem.MolFromSmiles(s)) is not None]
        if mols:
            #? [RDKit Draw] Renders 2D molecular structure grid PNG.
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
            img.save(str(out_dir / "generated_molecules.png"))

        return scored_df


# Backward-compatibility adapters for legacy script interoperability
ALL_METHODS = SequenceRNNOptimizer


class MY_TRANSFER_LEARNING(SequenceRNNOptimizer):
    def TRANSFER_LEARNING(self, molecule_smile: str):
        return self.AGENT, pd.DataFrame({"SMILES": [molecule_smile]}), self.OUTPUT_DIR


class RAINFORCMENT_LEARNING(SequenceRNNOptimizer):
    def __init__(self, finetuned: SequenceRNN | Path, output_dir: Path | None = None):
        super().__init__(output_dir=output_dir)
        if isinstance(finetuned, SequenceRNN):
            self.AGENT = finetuned


if __name__ == "__main__":
    optimizer = SequenceRNNOptimizer()
    agent, scored_sample = optimizer.run_reinforcement_learning(MY_MOLECULE, epochs=1, n_samples=50)
    print(f"SequenceRNN test completed. Output saved to: {optimizer.OUTPUT_DIR}")
