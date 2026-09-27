from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import torch

from drugex.data import datasets, fragments, processing
from drugex.data.corpus import vocabulary
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoTanimotoDistance
from drugex.training.scorers import conformer_generators, modifiers, properties, ra_scorer, rocs_rdkit
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

warnings.filterwarnings("ignore")


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

# --- Fragmentation Constants ---
FRAG_MIN_FRAGS = 1
FRAG_MAX_FRAGS = 10
FRAG_METHOD = "brics"

# --- Training Hyperparameters ---
RL_EPOCHS = 100
RL_EPSILON = 0.10
RL_BETA = 0.00
RL_N_SAMPLES = 200
RL_BATCH_SIZE = 64
RL_PATIENCE = 20

# --- Scorer Bounds & Thresholds ---
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


class GraphTransformerOptimizer:
    def __init__(self, output_dir: Path | None = None):
        self.ROOT_PATH = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR = self.ROOT_PATH / "_david" / "outputs_Graph"

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

        self.PRETRAINED_DIR = self.ROOT_PATH / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"
        self.MODEL_PRETRAINED = self.PRETRAINED_DIR

        self.VOC = VocGraph.fromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.vocab")
        self.AGENT = GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.AGENT.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.pkg")

        self.PRIOR = GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.PRIOR.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.pkg")

    def build_environment(self, ref_sdf_path: Path | str):
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
            n_jobs=2,
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

        scorers = [rocs_scorer, ra_scorer_inst, qed_scorer, sa_scorer]
        thresholds = [ROCS_THRESHOLD, RA_THRESHOLD, QED_THRESHOLD, SA_THRESHOLD]
        reward_scheme = ParetoTanimotoDistance()

        env = DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=reward_scheme,
        )
        return env

    def _prepare_seed_fragments(self, molecule_smile: str, n_prompts: int = RL_N_SAMPLES):
        seed_tsv = self.MODEL_DIR / "seed_fragments.tsv"
        seed_dataset = GraphFragDataSet(str(seed_tsv), rewrite=True)
        corpus_encoder = fragments.FragmentCorpusEncoder(
            fragmenter=Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, FRAG_METHOD),
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
                    n_repeats = max(1, n_prompts // len(std_scaffolds) + 1)
                    expanded_prompts = (std_scaffolds * n_repeats)[:n_prompts]
                    corpus_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
                    seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
                    return seed_loader, seed_dataset

        standardized_smile = processing.Standardization(n_proc=1).apply([molecule_smile])[0]
        expanded_prompts = [standardized_smile] * max(1, n_prompts // 5)
        corpus_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
        seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
        return seed_loader, seed_dataset

    def run_reinforcement_learning(self, molecule_smile: str = MY_MOLECULE, epochs: int = RL_EPOCHS, n_samples: int = RL_N_SAMPLES):
        ref_sdf = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        if not ref_sdf.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_sdf.parent)
            ref_sdf = preparer.generate_reference_sdf(molecule_smile, ref_sdf)

        env = self.build_environment(ref_sdf)
        seed_loader, seed_dataset = self._prepare_seed_fragments(molecule_smile, n_prompts=n_samples)

        explorer = FragGraphExplorer(
            agent=self.AGENT,
            env=env,
            mutate=self.PRIOR,
            epsilon=RL_EPSILON,
            beta=RL_BETA,
            batch_size=RL_BATCH_SIZE,
            use_gpus=GPUS,
        )

        model_prefix = self.MODEL_DIR / "david_agent"
        monitor = FileMonitor(model_prefix, save_smiles=True, reset_directory=True)

        explorer.fit(
            seed_loader,
            seed_loader,
            monitor=monitor,
            epochs=epochs,
            patience=RL_PATIENCE,
            criteria="desired_ratio",
        )

        scored_df = self.save_visualizations(self.AGENT, env, model_prefix, self.MODEL_DIR, input_dataset=seed_dataset, num_samples=n_samples)
        return self.AGENT, scored_df

    def RAINFORCEMENT_LEARNING(self, molecule_smile: str):
        return self.run_reinforcement_learning(molecule_smile=molecule_smile, epochs=RL_EPOCHS, n_samples=RL_N_SAMPLES)

    def save_visualizations(self, agent: GraphTransformer, env: DrugExEnvironment, model_prefix: Path, out_dir: Path, input_dataset: GraphFragDataSet, num_samples: int = 100):
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
            fig, ax = plt.subplots(figsize=(8, 5))
            df_info[metric_cols].plot.line(ax=ax)
            ax.set_title("Graph-Transformer Reinforcement Learning Dynamics", weight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric Value")
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.savefig(out_dir / "rl_training_curves.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

        generated_sample = agent.generate(input_dataset=input_dataset, num_samples=num_samples)
        scores = env.getScores(generated_sample["SMILES"])
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)

        mols = [m for s in generated_sample["SMILES"][:25] if (m := Chem.MolFromSmiles(s)) is not None]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
            img.save(str(out_dir / "generated_molecules.png"))

        return scored_df


# Backward-compatibility adapter classes
ALL_METHODS = GraphTransformerOptimizer


class MY_TRANSFER_LEARNING(GraphTransformerOptimizer):
    def TRANSFER_LEARNING(self, molecule_smile: str):
        return self.AGENT, pd.DataFrame({"SMILES": [molecule_smile]}), self.OUTPUT_DIR


class RAINFORCMENT_LEARNING(GraphTransformerOptimizer):
    def __init__(self, finetuned: GraphTransformer | Path, out_dir: Path | None = None):
        super().__init__(output_dir=out_dir)
        if isinstance(finetuned, GraphTransformer):
            self.AGENT = finetuned


if __name__ == "__main__":
    optimizer = GraphTransformerOptimizer()
    agent, scored_sample = optimizer.run_reinforcement_learning(MY_MOLECULE, epochs=1, n_samples=50)
    print(f"GraphTransformer test completed. Output saved to: {optimizer.OUTPUT_DIR}")
