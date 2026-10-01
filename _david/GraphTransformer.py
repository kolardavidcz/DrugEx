from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
import warnings

#? [Matplotlib] Non-interactive Agg backend: disables GUI window requirement for headless server execution.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#? [NumPy] Array operations for score manipulations and numerical metrics.
import numpy as np
#? [Pandas] Tabular DataFrame operations for generated molecule logging, sorting, and TSV export.
import pandas as pd
#? [RDKit] Molecular graph operations, chemical sanitization, and 2D grid depictions.
from rdkit import Chem
from rdkit.Chem import Draw
#? [PyTorch] Deep learning library for GPU device querying and graph transformer tensor allocations.
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
from drugex.training.scorers.modifiers import Gaussian, SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

from _david.usrcat_scorer import USRCATScorer

warnings.filterwarnings("ignore")


#? [PyTorch CUDA] Lightweight device probe: tests tensor allocation on GPU 0 to safely catch driver faults and fall back to CPU.
def get_safe_gpus() -> tuple[int, ...] | tuple[()]:
    """Safely detect available and functioning CUDA devices.

    Returns:
        Tuple containing GPU device index (0,) if CUDA is functional, or empty tuple if unavailable.
    """
    if not torch.cuda.is_available():
        return tuple()
    try:
        t = torch.zeros(1, device="cuda:0")
        del t
        return (0,)
    except Exception:
        return tuple()


GPUS: tuple[int, ...] | tuple[()] = get_safe_gpus()

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"

# --- Fragmentation Constants ---
FRAG_MIN_FRAGS: int = 1
FRAG_MAX_FRAGS: int = 4
FRAG_METHOD: str = "brics"

# --- Training Hyperparameters ---
RL_EPOCHS: int = 100
RL_EPSILON: float = 0.10
RL_BETA: float = 0.00
RL_N_SAMPLES: int = 200
RL_BATCH_SIZE: int = 64
RL_PATIENCE: int = 20

# --- 3D Shape Scorer Bounds & Thresholds ---
ROCS_SCORE_TYPE: str = "TanimotoCombo"
ROCS_MAX_CONFS: int = 10
ROCS_MAX_ISOMERS: int = 1
ROCS_TIMEOUT: int = 15
ROCS_LOWER_X: float = 0.45
ROCS_UPPER_X: float = 1.10
ROCS_THRESHOLD: float = 0.50

# --- USRCAT 3D Pharmacophore Bounds ---
USRCAT_LOWER_X: float = 0.50
USRCAT_UPPER_X: float = 0.85
USRCAT_THRESHOLD: float = 0.55

# --- Physical Size / Volume Bounds (EGCG target: 33 heavy atoms, MW ~458) ---
TARGET_HEAVY_ATOMS: float = 33.0
SIGMA_HEAVY_ATOMS: float = 6.0
HEAVY_THRESHOLD: float = 0.45

# --- Synthesizability & Drug-likeness Bounds ---
RA_LOWER_X: float = 0.30
RA_UPPER_X: float = 0.80
RA_THRESHOLD: float = 0.50

QED_LOWER_X: float = 0.35
QED_UPPER_X: float = 0.75
QED_THRESHOLD: float = 0.50

SA_LOWER_X: float = 4.50
SA_UPPER_X: float = 2.50
SA_THRESHOLD: float = 0.50


class GraphTransformerOptimizer:
    """Stage 3 Lead Optimizer executing fragment-conditioned RL with Graph-Transformer.

    Skeptical Academic Rationale:
    1. Conditioned Lead Elaboration: Takes Bemis-Murcko core frameworks discovered in Stage 2
       funnel and uses self-attention graph transformers to grow pharmacophore decorations.
    2. Overcomes Conformer Starvation: Evaluates 10 conformers per candidate (upgraded from 2)
       to avoid penalizing viable 3D binders due to under-sampled rotatable torsions.
    3. Prevents Fragment-Mode Collapse: Introduces a Gaussian heavy-atom size constraint (mu=33)
       so the model cannot cheat the Pareto reward by growing trivial 15-atom fragments from prompts.
    4. Dual 3D Objective: Simultaneously optimizes Gaussian shape/color (ROCS) and USRCAT 3D moments.

    Attributes:
        ROOT_PATH: Root directory of repository.
        OUTPUT_DIR: Target folder for saving training logs, checkpoints, and generated candidate tables.
        VOC: Graph vocabulary object loaded from pretrained Papyrus model.
        AGENT: Trainable GraphTransformer policy model.
        PRIOR: Frozen GraphTransformer prior model providing mutation and KL penalty anchor.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize GraphTransformerOptimizer with pretrained models and vocabularies.

        Args:
            output_dir: Optional custom Path to store run outputs and models.
        """
        self.ROOT_PATH: Path = Path(__file__).resolve().parent.parent
        self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "outputs_Graph"

        if output_dir is None:
            #? [DateTime] Generates ISO-like timestamp string for unique run output directory naming.
            self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.OUTPUT_DIR: Path = self.BASE_OUTPUT_DIR / self.RUN_ID
        else:
            self.OUTPUT_DIR = Path(output_dir)
            self.RUN_ID = self.OUTPUT_DIR.name
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.MODEL_DIR: Path = self.OUTPUT_DIR / "reinforcement_learning"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL: Path = self.MODEL_DIR

        self.PRETRAINED_DIR: Path = self.ROOT_PATH / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"
        self.MODEL_PRETRAINED: Path = self.PRETRAINED_DIR

        self.VOC: VocGraph = VocGraph.fromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.vocab")
        self.AGENT: GraphTransformer = GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.AGENT.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.pkg")

        self.PRIOR: GraphTransformer = GraphTransformer(voc_trg=self.VOC, use_gpus=GPUS)
        self.PRIOR.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_graph_trans_PT.pkg")

    def build_environment(self, ref_sdf_path: Path | str) -> DrugExEnvironment:
        """Construct multi-objective DrugExEnvironment for reinforcement learning.

        Args:
            ref_sdf_path: Path to reference multi-conformer 3D SDF for ROCS and USRCAT.

        Returns:
            Configured DrugExEnvironment with ParetoTanimotoDistance reward scheme.
        """
        ref_path_str: str = str(ref_sdf_path)

        #? [RDKit Conformer Generator] Multi-conformer generator with ETKDGv3 and expanded max_heavy_atoms=50.
        conf_gen = RDKitConformerGenerator(
            max_conformers=ROCS_MAX_CONFS,
            max_isomers=ROCS_MAX_ISOMERS,
            max_heavy_atoms=50,
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

        #? [Pure LBDD USRCAT Scorer] Ultrafast 60-dimensional shape and pharmacophore moment similarity.
        usrcat_scorer = USRCATScorer(
            references=ref_path_str,
            num_conformers=ROCS_MAX_CONFS,
            modifier=SmoothClippedScore(
                lower_x=USRCAT_LOWER_X,
                upper_x=USRCAT_UPPER_X,
            ),
        )

        #? [Heavy Atom Volume Scorer] Gaussian modifier centered on EGCG (33 atoms) to penalize small-fragment cheats.
        size_scorer = Property(
            prop="Heavy",
            modifier=Gaussian(
                mu=TARGET_HEAVY_ATOMS,
                sigma=SIGMA_HEAVY_ATOMS,
            ),
        )

        #? [Retrosynthetic Accessibility Scorer] RAscore machine-learning model estimating synthesizability probability.
        ra_scorer_inst = RetrosyntheticAccessibilityScorer(
            modifier=SmoothClippedScore(
                lower_x=RA_LOWER_X,
                upper_x=RA_UPPER_X,
            )
        )

        #? [Drug-likeness QED Scorer] Quantitative Estimate of Drug-likeness.
        qed_scorer = Property(
            prop="QED",
            modifier=SmoothClippedScore(
                lower_x=QED_LOWER_X,
                upper_x=QED_UPPER_X,
            ),
        )

        #? [Synthetic Accessibility SA Scorer] Ertl & Schuffenhauer fragment penalty index.
        sa_scorer = Property(
            prop="SA",
            modifier=SmoothClippedScore(
                lower_x=SA_LOWER_X,
                upper_x=SA_UPPER_X,
            ),
        )

        scorers = [rocs_scorer, usrcat_scorer, size_scorer, ra_scorer_inst, qed_scorer, sa_scorer]
        thresholds = [ROCS_THRESHOLD, USRCAT_THRESHOLD, HEAVY_THRESHOLD, RA_THRESHOLD, QED_THRESHOLD, SA_THRESHOLD]
        reward_scheme = ParetoTanimotoDistance()

        env = DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=reward_scheme,
        )
        return env

    def _prepare_seed_fragments(
        self, molecule_smile: str, n_prompts: int = RL_N_SAMPLES, scaffolds_path: Path | str | None = None
    ) -> tuple[torch.utils.data.DataLoader, GraphFragDataSet]:
        """Encode lead scaffolds or seed fragments into PyTorch DataLoader for GraphTransformer.

        Args:
            molecule_smile: Active seed query SMILES string.
            n_prompts: Target prompt batch count.
            scaffolds_path: Optional explicit Path to funnel scaffolds TSV file.

        Returns:
            Tuple containing PyTorch DataLoader and GraphFragDataSet.
        """
        seed_tsv: Path = self.MODEL_DIR / "seed_fragments.tsv"
        seed_dataset = GraphFragDataSet(str(seed_tsv), rewrite=True)
        #? [DrugEx FragmentCorpusEncoder] Decomposes prompts into graph fragments; allow_single=True enables single-ring retention.
        corpus_encoder = fragments.FragmentCorpusEncoder(
            fragmenter=Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, FRAG_METHOD, allow_single=True),
            encoder=fragments.GraphFragmentEncoder(self.VOC),
            pairs_splitter=None,
            n_proc=1,
        )

        funnel_candidates: List[Path] = []
        if scaffolds_path is not None and Path(scaffolds_path).exists():
            funnel_candidates.append(Path(scaffolds_path))
        funnel_candidates.extend([
            self.ROOT_PATH / "_david" / "funnel_scaffolds_expanded.tsv",
            self.ROOT_PATH / "_david" / "funnel_scaffolds.tsv",
        ])

        for fpath in funnel_candidates:
            if fpath.exists():
                #? [Pandas] Reads funnel scaffold TSV table to retrieve lead centroids.
                df_funnel = pd.read_csv(fpath, sep="\t")
                if "Scaffold" in df_funnel.columns and len(df_funnel) > 0:
                    scaffolds = df_funnel["Scaffold"].dropna().unique().tolist()
                    std_scaffolds = processing.Standardization(n_proc=1).apply(scaffolds)
                    if std_scaffolds:
                        n_repeats = max(1, n_prompts // len(std_scaffolds) + 1)
                        expanded_prompts = (std_scaffolds * n_repeats)[:n_prompts]
                        corpus_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
                        if len(seed_dataset) > 0:
                            seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
                            return seed_loader, seed_dataset

        # Fallback to seed molecule fragmentation (EGCG possesses ester/ether bonds readily cleaved by BRICS/RECAP)
        standardized_smile = processing.Standardization(n_proc=1).apply([molecule_smile])[0]
        expanded_prompts = [standardized_smile] * max(1, n_prompts // 5)
        corpus_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
        if len(seed_dataset) == 0:
            # Fallback to RECAP method if BRICS produced 0 pairs
            recap_encoder = fragments.FragmentCorpusEncoder(
                fragmenter=Fragmenter(FRAG_MIN_FRAGS, FRAG_MAX_FRAGS, "recap", allow_single=True),
                encoder=fragments.GraphFragmentEncoder(self.VOC),
                pairs_splitter=None,
                n_proc=1,
            )
            recap_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])

        if len(seed_dataset) == 0:
            raise RuntimeError(f"Could not encode any valid fragment prompts from scaffolds or {molecule_smile}")

        seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
        return seed_loader, seed_dataset

    def run_reinforcement_learning(
        self,
        molecule_smile: str = MY_MOLECULE,
        epochs: int = RL_EPOCHS,
        n_samples: int = RL_N_SAMPLES,
        scaffolds_path: Path | str | None = None,
    ) -> tuple[GraphTransformer, pd.DataFrame]:
        """Execute fragment-conditioned RL exploration on GraphTransformer.

        Args:
            molecule_smile: Active seed query SMILES.
            epochs: Total training epochs.
            n_samples: Number of samples generated per epoch.
            scaffolds_path: Optional Path to Stage 2 funnel scaffolds.

        Returns:
            Tuple containing trained GraphTransformer agent and scored DataFrame.
        """
        ref_sdf = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        if not ref_sdf.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_sdf.parent)
            ref_sdf = preparer.generate_reference_sdf(molecule_smile, ref_sdf)

        env = self.build_environment(ref_sdf)
        seed_loader, seed_dataset = self._prepare_seed_fragments(molecule_smile, n_prompts=n_samples, scaffolds_path=scaffolds_path)

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

        patience = min(RL_PATIENCE, max(5, epochs // 2))
        explorer.fit(
            seed_loader,
            seed_loader,
            monitor=monitor,
            epochs=epochs,
            patience=patience,
            criteria="desired_ratio",
        )

        scored_df = self.save_visualizations(self.AGENT, env, model_prefix, self.MODEL_DIR, input_dataset=seed_dataset, num_samples=n_samples)
        return self.AGENT, scored_df

    def RAINFORCEMENT_LEARNING(
        self,
        molecule_smile: str,
        epochs: int | None = None,
        n_samples: int | None = None,
        scaffolds_path: Path | str | None = None,
    ) -> tuple[GraphTransformer, pd.DataFrame]:
        """Backward-compatibility adapter for legacy pipeline calls.

        Args:
            molecule_smile: Query molecule SMILES.
            epochs: Optional custom epoch count.
            n_samples: Optional custom sample count.
            scaffolds_path: Optional custom scaffolds path.

        Returns:
            Tuple of trained agent and scored DataFrame.
        """
        ep = epochs if epochs is not None else RL_EPOCHS
        ns = n_samples if n_samples is not None else RL_N_SAMPLES
        return self.run_reinforcement_learning(molecule_smile=molecule_smile, epochs=ep, n_samples=ns, scaffolds_path=scaffolds_path)

    def save_visualizations(
        self,
        agent: GraphTransformer,
        env: DrugExEnvironment,
        model_prefix: Path,
        out_dir: Path,
        input_dataset: GraphFragDataSet,
        num_samples: int = 100,
    ) -> pd.DataFrame:
        """Sample candidates, score across multi-objective environment, and save diagnostic plots.

        Args:
            agent: Trained GraphTransformer generator.
            env: Multi-objective DrugExEnvironment.
            model_prefix: File prefix where FileMonitor logged training data.
            out_dir: Target output directory for plots and tables.
            input_dataset: GraphFragDataSet prompt dataset.
            num_samples: Number of de novo molecules to generate.

        Returns:
            DataFrame of scored candidate leads.
        """
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            #? [Pandas] Parses TSV training monitor log file into DataFrame.
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
            #? [Matplotlib] Renders multi-metric RL training dynamics convergence curves.
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
        #? [Pandas] Concatenates generated candidate rows with calculated score columns.
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        scored_df.to_csv(out_dir / "stage3_leads.tsv", sep="\t", index=False)
        if hasattr(self, "OUTPUT_DIR") and self.OUTPUT_DIR != out_dir:
            scored_df.to_csv(self.OUTPUT_DIR / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
            scored_df.to_csv(self.OUTPUT_DIR / "stage3_leads.tsv", sep="\t", index=False)

        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        mols = [m for s in generated_sample["SMILES"][:25] if (m := Chem.MolFromSmiles(s)) is not None]
        if mols:
            #? [RDKit Draw] Renders 2D molecular structure grid PNG.
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
            img.save(str(out_dir / "generated_molecules.png"))

        return scored_df


# Backward-compatibility adapter classes
ALL_METHODS = GraphTransformerOptimizer


class MY_TRANSFER_LEARNING(GraphTransformerOptimizer):
    """Adapter for legacy transfer learning calls."""

    def TRANSFER_LEARNING(self, molecule_smile: str) -> tuple[GraphTransformer, pd.DataFrame, Path]:
        """Execute legacy transfer learning placeholder.

        Args:
            molecule_smile: Query molecule SMILES.

        Returns:
            Tuple of agent, DataFrame, and output directory.
        """
        return self.AGENT, pd.DataFrame({"SMILES": [molecule_smile]}), self.OUTPUT_DIR


class RAINFORCMENT_LEARNING(GraphTransformerOptimizer):
    """Adapter for legacy reinforcement learning class naming."""

    def __init__(
        self,
        finetuned: GraphTransformer | Path,
        out_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            finetuned: Trained agent or path.
            out_dir: Optional output directory.
            output_dir: Optional output directory alias.
        """
        target_dir = output_dir or out_dir
        super().__init__(output_dir=target_dir)
        if isinstance(finetuned, GraphTransformer):
            self.AGENT = finetuned


if __name__ == "__main__":
    optimizer = GraphTransformerOptimizer()
    agent, scored_sample = optimizer.run_reinforcement_learning(MY_MOLECULE, epochs=1, n_samples=50)
    print(f"GraphTransformer test completed. Output saved to: {optimizer.OUTPUT_DIR}")
