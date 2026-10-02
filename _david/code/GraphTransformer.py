"""Stage 3 Lead Optimization executing fragment-conditioned RL on GraphTransformer."""
from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from drugex.data import fragments, processing
from drugex.data.corpus.vocabulary import VocGraph
from drugex.data.datasets import GraphFragDataSet
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators import GraphTransformer
from drugex.training.interfaces import RewardScheme
from drugex.training.monitors import FileMonitor
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import SmoothClippedScore, SmoothHump
from drugex.training.scorers.properties import Property, Scaffold
from drugex.training.scorers.ra_scorer import RetrosyntheticAccessibilityScorer
from rdkit import Chem
from rdkit.Chem import Draw

from _david.code.anti_cheating_rewards import AugmentedChebyshevReward
from _david.code.fallback_monitor import FallbackMonitor
from _david.code.usrcat_scorer import USRCATScorer

# Ensure encodeFragments alias exists on GraphTransformer
if not hasattr(GraphTransformer, "encodeFragments"):
    GraphTransformer.encodeFragments = GraphTransformer.loaderFromFrags

# Ensure GraphFragDataSet supports len() inspection
if not hasattr(GraphFragDataSet, "__len__"):
    GraphFragDataSet.__len__ = lambda self: len(self.getData()) if hasattr(self, "outpath") and self.outpath.exists() else 0

warnings.filterwarnings("ignore")


#? [PyTorch CUDA] Lightweight device probe: tests tensor allocation on GPU 0 to safely catch driver faults and fall back to CPU.
def get_safe_gpus() -> tuple[int, ...] | tuple[()]:
    """Safely detect available and functioning CUDA devices."""
    if not torch.cuda.is_available():
        FallbackMonitor.record_fallback(
            fallback_type="device_cuda_unavailable",
            message="torch.cuda.is_available() is False; falling back to CPU",
        )
        return tuple()
    try:
        t = torch.zeros(1, device="cuda:0")
        del t
        return (0,)
    except Exception as e:
        FallbackMonitor.record_fallback(
            fallback_type="device_cuda_error",
            message=f"CUDA test allocation failed: {e}; falling back to CPU",
        )
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
ELECTROSHAPE_LOWER_X: float = 0.12
ELECTROSHAPE_UPPER_X: float = 0.35
ELECTROSHAPE_THRESHOLD: float = 0.22
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
    """Stage 3 Generative Optimizer executing fragment-conditioned RL on GraphTransformer."""

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize GraphTransformerOptimizer with pretrained models and vocabularies."""
        self.ROOT_PATH: Path = PROJECT_ROOT
        self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "output" / "stage3_graph_transformer"

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
        """Construct multi-objective DrugExEnvironment for reinforcement learning."""
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
        seed_hac = seed_mol.GetNumHeavyAtoms() if seed_mol else 30
        delta = max(3.0, seed_hac * 0.15)
        lower_size = max(6.0, float(seed_hac - delta))
        upper_size = float(seed_hac + delta)
        sigma = max(2.5, seed_hac * 0.12)

        #? [Heavy Atom Volume Scorer] Native SmoothHump modifier dynamically centered on seed heavy atom count.
        size_scorer = Property(
            prop="Heavy",
            modifier=SmoothHump(
                lower_x=lower_size,
                upper_x=upper_size,
                sigma=sigma,
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
        self,
        molecule_smile: str,
        n_prompts: int = 100,
        scaffolds_path: Path | str | None = None,
    ) -> tuple[torch.utils.data.DataLoader, GraphFragDataSet]:
        """Prepare fragment dataset and DataLoader from Stage 2 funnel scaffolds."""
        seed_tsv: Path = self.MODEL_DIR / "seed_fragments.tsv"
        seed_dataset = GraphFragDataSet(str(seed_tsv), rewrite=True)

        #? [DrugEx dummyMolsFromFragments] Native fragment converter attaching dummy attachment vectors without bond cleavage.
        corpus_encoder = fragments.FragmentCorpusEncoder(
            fragmenter=dummyMolsFromFragments(),
            encoder=fragments.GraphFragmentEncoder(self.VOC),
            pairs_splitter=None,
            n_proc=1,
        )

        scaffolds: list[str] = []
        if scaffolds_path is not None and Path(scaffolds_path).exists():
            try:
                df_scaff = pd.read_csv(scaffolds_path, sep="\t")
                if "Scaffold" in df_scaff.columns and len(df_scaff) > 0:
                    scaffolds = df_scaff["Scaffold"].dropna().unique().tolist()
            except Exception:
                pass

        if not scaffolds:
            # Direct core scaffold conditioning from active seed query
            from rdkit.Chem.Scaffolds import MurckoScaffold
            seed_mol = Chem.MolFromSmiles(molecule_smile)
            if seed_mol:
                core_scaff = MurckoScaffold.GetScaffoldForMol(seed_mol)
                if core_scaff and core_scaff.GetNumHeavyAtoms() > 0:
                    scaffolds = [Chem.MolToSmiles(core_scaff, isomericSmiles=False)]
                else:
                    scaffolds = [Chem.MolToSmiles(seed_mol, isomericSmiles=False)]
            else:
                scaffolds = [molecule_smile]

        # Standardize scaffolds and remove stereochemistry (VocGraph operates on 2D constitution/valences only)
        std_raw = processing.Standardization(n_proc=1).apply(scaffolds)
        std_scaffolds: list[str] = []
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
        """Execute fragment-conditioned RL exploration on GraphTransformer."""
        from _david.code.prepare_query_conformer import QueryConformerPreparer
        preparer = QueryConformerPreparer(self.ROOT_PATH / "_david" / "references")
        seed_hash = preparer.get_seed_hash(molecule_smile)
        ref_candidates = [
            self.OUTPUT_DIR / "references" / f"seed_{seed_hash}_3d.sdf",
            self.OUTPUT_DIR.parent / "references" / f"seed_{seed_hash}_3d.sdf",
            self.OUTPUT_DIR.parent / "references" / "query_molecule_3d.sdf",
            self.ROOT_PATH / "_david" / "references" / f"seed_{seed_hash}_3d.sdf",
            self.ROOT_PATH / "_david" / "references" / "query_molecule_3d.sdf",
            self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf",
        ]
        ref_sdf = next((p for p in ref_candidates if p.exists()), None)
        if ref_sdf is None:
            ref_sdf = self.OUTPUT_DIR / "references" / f"seed_{seed_hash}_3d.sdf"
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

        scored_df = self.save_visualizations(
            self.AGENT, env, model_prefix, self.MODEL_DIR, input_dataset=seed_dataset, num_samples=n_samples
        )
        return self.AGENT, scored_df

    def RAINFORCEMENT_LEARNING(
        self,
        molecule_smile: str,
        epochs: int | None = None,
        n_samples: int | None = None,
        scaffolds_path: Path | str | None = None,
    ) -> tuple[GraphTransformer, pd.DataFrame]:
        """Backward-compatibility wrapper for legacy pipeline calls."""
        ep = epochs if epochs is not None else RL_EPOCHS
        ns = n_samples if n_samples is not None else RL_N_SAMPLES
        return self.run_reinforcement_learning(
            molecule_smile=molecule_smile, epochs=ep, n_samples=ns, scaffolds_path=scaffolds_path
        )

    def save_visualizations(
        self,
        agent: GraphTransformer,
        env: DrugExEnvironment,
        model_prefix: Path,
        out_dir: Path,
        input_dataset: GraphFragDataSet | None = None,
        num_samples: int = 100,
    ) -> pd.DataFrame:
        """Generate leads, evaluate multi-objective scores, and save diagnostic plots."""
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

        if input_dataset is not None:
            loader = input_dataset.asDataLoader(RL_BATCH_SIZE)
            gen_smiles, gen_frags = agent.sample(loader)
            generated_sample = pd.DataFrame({"SMILES": gen_smiles, "FRAGS": gen_frags})
        else:
            _, fallback_dataset = self._prepare_seed_fragments(MY_MOLECULE, n_prompts=num_samples)
            loader = fallback_dataset.asDataLoader(RL_BATCH_SIZE)
            gen_smiles, gen_frags = agent.sample(loader)
            generated_sample = pd.DataFrame({"SMILES": gen_smiles, "FRAGS": gen_frags})

        if not generated_sample.empty:
            scores = env.getScores(generated_sample["SMILES"])
        else:
            scores = pd.DataFrame(columns=[s.getKey() for s in env.getScorers()])
        #? [Pandas] Concatenates generated SMILES with evaluated objective scores.
        scored_df = pd.concat([generated_sample.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
        scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
        scored_df.to_csv(out_dir / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
        scored_df.to_csv(out_dir / "stage2_leads.tsv", sep="\t", index=False)
        scored_df.to_csv(out_dir / "stage3_leads.tsv", sep="\t", index=False)
        if out_dir.name == "reinforcement_learning" and out_dir.parent != out_dir:
            scored_df.to_csv(out_dir.parent / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
            scored_df.to_csv(out_dir.parent / "stage2_leads.tsv", sep="\t", index=False)
            scored_df.to_csv(out_dir.parent / "stage3_leads.tsv", sep="\t", index=False)
        if out_dir != self.OUTPUT_DIR and out_dir.parent != self.OUTPUT_DIR:
            scored_df.to_csv(self.OUTPUT_DIR / "rl_generated_candidates_scored.tsv", sep="\t", index=False)
            scored_df.to_csv(self.OUTPUT_DIR / "stage2_leads.tsv", sep="\t", index=False)
            scored_df.to_csv(self.OUTPUT_DIR / "stage3_leads.tsv", sep="\t", index=False)

        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        mols = [m for s in generated_sample["SMILES"][:25] if (m := Chem.MolFromSmiles(s)) is not None]
        if mols:
            #? [RDKit Draw] Renders 2D molecular structure grid PNG.
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
            img.save(str(out_dir / "generated_molecules.png"))

        return scored_df


# Backward-compatibility adapters for legacy script interoperability
ALL_METHODS = GraphTransformerOptimizer


class MY_TRANSFER_LEARNING(GraphTransformerOptimizer):
    """Adapter for legacy transfer learning calls."""

    def TRANSFER_LEARNING(self, molecule_smile: str) -> tuple[GraphTransformer, pd.DataFrame, Path]:
        """Execute legacy transfer learning placeholder."""
        return self.AGENT, pd.DataFrame({"SMILES": [molecule_smile]}), self.OUTPUT_DIR


class RAINFORCMENT_LEARNING(GraphTransformerOptimizer):
    """Adapter for legacy reinforcement learning class naming."""

    def __init__(self, finetuned: GraphTransformer | Path, output_dir: Path | None = None) -> None:
        """Initialize adapter."""
        super().__init__(output_dir=output_dir)
        if isinstance(finetuned, GraphTransformer):
            self.AGENT = finetuned


if __name__ == "__main__":
    optimizer = GraphTransformerOptimizer()
    agent, scored_sample = optimizer.run_reinforcement_learning(MY_MOLECULE, epochs=1, n_samples=50)
    print(f"GraphTransformer test completed. Output saved to: {optimizer.OUTPUT_DIR}")
    FallbackMonitor.print_summary()
