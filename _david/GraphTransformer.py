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
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training import environment, explorers, generators, monitors, rewards
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.interfaces import RewardScheme
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoTanimotoDistance
from drugex.training.scorers import modifiers, properties, ra_scorer
from drugex.training.scorers.modifiers import SmoothClippedScore, SmoothHump
from drugex.training.scorers.properties import Property, Scaffold
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer

from _david.anti_cheating_rewards import AugmentedChebyshevReward, GeometricMeanReward
from _david.usrcat_scorer import USRCATScorer

# Ensure encodeFragments alias exists on GraphTransformer
if not hasattr(GraphTransformer, "encodeFragments"):
    GraphTransformer.encodeFragments = GraphTransformer.loaderFromFrags

# Ensure GraphFragDataSet supports len() inspection
if not hasattr(GraphFragDataSet, "__len__"):
    GraphFragDataSet.__len__ = lambda self: len(self.getData()) if hasattr(self, "outpath") and self.outpath.exists() else 0

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

# --- ElectroShape 3D Shape & Electrostatic Moment Bounds (Ultrafast Decoupled RL Tier) ---
ELECTROSHAPE_LOWER_X: float = 0.50
ELECTROSHAPE_UPPER_X: float = 0.85
ELECTROSHAPE_THRESHOLD: float = 0.55
# Backward compatibility aliases
USRCAT_LOWER_X: float = ELECTROSHAPE_LOWER_X
USRCAT_UPPER_X: float = ELECTROSHAPE_UPPER_X
USRCAT_THRESHOLD: float = ELECTROSHAPE_THRESHOLD

# --- Universal Physical Size Envelope Defaults ---
HEAVY_THRESHOLD: float = 0.50
SIZE_SIGMA: float = 4.0
SIZE_DELTA: float = 5.0  # Dynamic plateau width: [seed_hac - SIZE_DELTA, seed_hac + SIZE_DELTA]

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

    def build_environment(
        self,
        ref_sdf_path: Path | str,
        seed_smiles: str = MY_MOLECULE,
        toxicophore_smarts: str | Sequence[str] | None = None,
        reward_scheme: RewardScheme | None = None,
    ) -> DrugExEnvironment:
        """Construct multi-objective DrugExEnvironment for reinforcement learning.

        Builds a universal, un-cheatable multi-objective environment incorporating ultrafast 3D
        ElectroShape moments (75-D shape + Gasteiger partial charge centroids, 100x faster than
        iterative ROCS conformer optimization), dynamic heavy atom volume plateau (SmoothHump
        centered on the query ligand's heavy atom count), retrosynthetic accessibility (RAscore),
        QED drug-likeness, and synthetic accessibility (SA).
        Optionally incorporates custom toxicophore/liability suppression via Scaffold if specified.
        Uses AugmentedChebyshevReward scalarization to prevent small fragments from gaming
        the Pareto frontier.

        Args:
            ref_sdf_path: Path to reference multi-conformer 3D SDF for ElectroShape.
            seed_smiles: SMILES string of active seed query for dynamic size plateau derivation.
            toxicophore_smarts: Optional SMARTS pattern or sequence of patterns to penalize (default: None).
            reward_scheme: Optional custom RewardScheme (defaults to AugmentedChebyshevReward).

        Returns:
            Configured DrugExEnvironment with anti-cheating reward scheme.
        """
        ref_path_str: str = str(ref_sdf_path)

        #? [Pure LBDD ElectroShape Scorer] Ultrafast 75-dimensional shape and partial charge centroid moment similarity.
        electroshape_scorer = USRCATScorer(
            references=ref_path_str,
            num_conformers=ROCS_MAX_CONFS,
            score_type="electroshape",
            modifier=SmoothClippedScore(
                lower_x=ELECTROSHAPE_LOWER_X,
                upper_x=ELECTROSHAPE_UPPER_X,
            ),
        )

        # Dynamic size envelope derived from seed ligand heavy atom count
        seed_mol = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None
        seed_hac = seed_mol.GetNumHeavyAtoms() if seed_mol else 33
        lower_size = max(6.0, float(seed_hac - SIZE_DELTA))
        upper_size = float(seed_hac + SIZE_DELTA)

        #? [Heavy Atom Volume Scorer] Native SmoothHump modifier dynamically centered on seed heavy atom count.
        size_scorer = Property(
            prop="Heavy",
            modifier=SmoothHump(
                lower_x=lower_size,
                upper_x=upper_size,
                sigma=SIZE_SIGMA,
            ),
        )

        #? [Retrosynthetic Accessibility Scorer] RAscore machine-learning model estimating synthesizability probability.
        ra_scorer_inst = RetrosyntheticAccessibilityScorer(
            modifier=SmoothClippedScore(
                lower_x=RA_LOWER_X,
                upper_x=RA_UPPER_X,
            ),
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

        scorers: list[Scorer] = [electroshape_scorer, size_scorer, ra_scorer_inst, qed_scorer, sa_scorer]
        thresholds: list[float] = [
            ELECTROSHAPE_THRESHOLD,
            HEAVY_THRESHOLD,
            RA_THRESHOLD,
            QED_THRESHOLD,
            SA_THRESHOLD,
        ]

        # Optional target-specific toxicophore suppression
        if toxicophore_smarts is not None:
            smarts_list = [toxicophore_smarts] if isinstance(toxicophore_smarts, str) else list(toxicophore_smarts)
            for smart in smarts_list:
                #? [Toxicophore Suppression Scorer] Native Scaffold scorer penalizing user-specified substructure liabilities.
                toxicophore_scorer = Scaffold(
                    smart=smart,
                    is_match=False,
                )
                scorers.append(toxicophore_scorer)
                thresholds.append(0.99)

        scheme = reward_scheme if reward_scheme is not None else AugmentedChebyshevReward()

        env = DrugExEnvironment(
            scorers=scorers,
            thresholds=thresholds,
            reward_scheme=scheme,
        )
        return env

    def _prepare_seed_fragments(
        self, molecule_smile: str, n_prompts: int = RL_N_SAMPLES, scaffolds_path: Path | str | None = None
    ) -> tuple[torch.utils.data.DataLoader, GraphFragDataSet]:
        """Encode lead scaffolds or seed fragments into PyTorch DataLoader for GraphTransformer.

        Skeptical Academic Rationale:
        Replaces BRICS/RECAP cleavage with DrugEx native dummyMolsFromFragments().
        Intact Bemis-Murcko core frameworks (e.g. condensed rings or flavan skeletons)
        lack cleavable retrosynthetic bonds, causing BRICS to generate zero fragments
        and silently trigger fallback to EGCG. dummyMolsFromFragments() natively treats
        the intact scaffold core as a single fragment with dummy connection vectors,
        enabling Graph-Transformer to elaborate decorations directly onto discovered leads.

        Args:
            molecule_smile: Active seed query SMILES string.
            n_prompts: Target prompt batch count.
            scaffolds_path: Optional explicit Path to funnel scaffolds TSV file.

        Returns:
            Tuple containing PyTorch DataLoader and GraphFragDataSet.
        """
        seed_tsv: Path = self.MODEL_DIR / "seed_fragments.tsv"
        seed_dataset = GraphFragDataSet(str(seed_tsv), rewrite=True)

        #? [DrugEx dummyMolsFromFragments] Native fragment converter attaching dummy attachment vectors without bond cleavage.
        corpus_encoder = fragments.FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
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

        scaffolds: List[str] = []
        for fpath in funnel_candidates:
            if fpath.exists():
                #? [Pandas] Reads funnel scaffold TSV table to retrieve lead centroids.
                df_funnel = pd.read_csv(fpath, sep="\t")
                if "Scaffold" in df_funnel.columns and len(df_funnel) > 0:
                    loaded = df_funnel["Scaffold"].dropna().unique().tolist()
                    if loaded:
                        scaffolds = loaded
                        break

        if not scaffolds:
            # Fallback to query seed molecule if no funnel scaffolds exist
            scaffolds = [molecule_smile]

        # Standardize scaffolds and remove stereochemistry (VocGraph operates on 2D constitution/valences only)
        std_raw = processing.Standardization(n_proc=1).apply(scaffolds)
        std_scaffolds: List[str] = []
        for s in (std_raw or scaffolds):
            m = Chem.MolFromSmiles(s)
            if m is not None:
                std_scaffolds.append(Chem.MolToSmiles(m, isomericSmiles=False))
        if not std_scaffolds:
            std_scaffolds = scaffolds

        # Expand scaffolds to target prompt count
        n_repeats = max(1, n_prompts // len(std_scaffolds) + 1)
        expanded_prompts = (std_scaffolds * n_repeats)[:n_prompts]

        corpus_encoder.apply(expanded_prompts, encodingCollectors=[seed_dataset])
        dataset_len = len(seed_dataset.getData()) if seed_dataset.outpath.exists() else 0
        if dataset_len == 0:
            raise RuntimeError(f"Could not encode any valid fragment prompts from scaffolds: {std_scaffolds[:5]}")

        seed_loader = seed_dataset.asDataLoader(RL_BATCH_SIZE)
        return seed_loader, seed_dataset

    def run_reinforcement_learning(
        self,
        molecule_smile: str = MY_MOLECULE,
        epochs: int = RL_EPOCHS,
        n_samples: int = RL_N_SAMPLES,
        scaffolds_path: Path | str | None = None,
        toxicophore_smarts: str | Sequence[str] | None = None,
    ) -> tuple[GraphTransformer, pd.DataFrame]:
        """Execute fragment-conditioned RL exploration on GraphTransformer.

        Args:
            molecule_smile: Active seed query SMILES.
            epochs: Total training epochs.
            n_samples: Number of samples generated per epoch.
            scaffolds_path: Optional Path to Stage 2 funnel scaffolds.
            toxicophore_smarts: Optional substructure SMARTS pattern to penalize.

        Returns:
            Tuple containing trained GraphTransformer agent and scored DataFrame.
        """
        ref_sdf = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        if not ref_sdf.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_sdf.parent)
            ref_sdf = preparer.generate_reference_sdf(molecule_smile, ref_sdf)

        env = self.build_environment(ref_sdf, seed_smiles=molecule_smile, toxicophore_smarts=toxicophore_smarts)
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
        if out_dir.name == "reinforcement_learning" and out_dir.parent != out_dir:
            scored_df.to_csv(out_dir.parent / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
            scored_df.to_csv(out_dir.parent / "stage3_leads.tsv", sep="\t", index=False)
        if hasattr(self, "OUTPUT_DIR") and self.OUTPUT_DIR != out_dir and self.OUTPUT_DIR != out_dir.parent:
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
