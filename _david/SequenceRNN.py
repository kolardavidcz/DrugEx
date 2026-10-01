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


#? ============================================================================
#? GPU HARDWARE DETECTION & RUNTIME PROBING
#? ============================================================================
#? WHAT:
#?   Probes for CUDA device availability and executes a lightweight test allocation.
#? WHY:
#?   Prevents unhandled PyTorch CUDA initialization crashes when running in headless
#?   servers, virtualized WSL2 environments, or systems where CUDA driver/runtime
#?   mismatches occur. Falls back gracefully to CPU mode without interrupting execution.
#? UNDER THE HOOD:
#?   Calls torch.cuda.is_available(), attempts to allocate a 1-element float tensor on
#?   'cuda:0', and deallocates it immediately. If any RuntimeError or OOM occurs,
#?   catches the exception and returns an empty tuple () indicating CPU-only execution.
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

#? ============================================================================
#? REINFORCEMENT LEARNING HYPERPARAMETERS (STAGE 1 EXPLORATION)
#? ============================================================================
#? WHAT:
#?   Policy-gradient optimization settings governing the Markov Decision Process (MDP):
#?   - RL_EPOCHS (100): Total policy gradient iteration epochs.
#?   - RL_EPSILON (0.10): Epsilon-greedy token sampling probability from the frozen Prior.
#?   - RL_BETA (0.00): Prior regularization penalty (KL divergence coefficient).
#?   - RL_N_SAMPLES (200): Trajectories sampled per epoch for Monte Carlo reward estimation.
#?   - RL_BATCH_SIZE (64): Mini-batch size for policy gradient parameter updates.
#?   - RL_PATIENCE (20): Early stopping epochs if 'desired_ratio' ceases to improve.
#?
#? WHY (Bachelor Thesis - Stage 1 Exploration):
#?   - Epsilon-greedy exploration (0.10) injects token-level mutations from the general
#?     bioactive Prior, preventing the Agent policy from collapsing into repetitive local minima.
#?   - Sampling N=200 trajectories per epoch provides statistically robust policy gradient
#?     variance reduction without overwhelming CPU conformer generation overhead.
#?
#? UNDER THE HOOD:
#?   The agent interacts with the chemical space environment in discrete time steps (tokens).
#?   Epsilon-greedy sampling chooses token a_t ~ P_prior with probability epsilon, and
#?   a_t ~ P_agent with probability 1 - epsilon. Policy gradients are computed using REINFORCE
#?   with baseline subtraction: grad_theta = E[ grad_theta log P(S) * (R(S) - b) ].
RL_EPOCHS = 100
RL_EPSILON = 0.10
RL_BETA = 0.00
RL_N_SAMPLES = 200
RL_BATCH_SIZE = 64
RL_PATIENCE = 20

#? ============================================================================
#? MULTI-OBJECTIVE SCORER BOUNDS & DESIRABILITY THRESHOLDS
#? ============================================================================
#? WHAT:
#?   Numerical boundary parameters defining the smooth sigmoidal desirability mapping
#?   for each objective in the multi-objective reward function:
#?   - ROCS (3D Shape & Pharmacophore Color Alignment)
#?   - RAscore (Retrosynthetic Accessibility)
#?   - Inverted 2D Novelty (Scaffold Hopping Distance)
#?   - QED (Quantitative Estimate of Drug-likeness)
#?   - SA (Synthetic Accessibility Complexity Penalty)
#?
#? WHY:
#?   Raw physical values (e.g., ROCS [0, 2], SA [1, 10]) inhabit completely different scales
#?   and non-linear desirability profiles. Setting explicit lower_x and upper_x bounds
#?   maps each metric into a normalized [0, 1] desirability score, enabling Pareto ranking.
#?
#? UNDER THE HOOD:
#?   DrugEx SmoothClippedScore computes a continuous, differentiable transition curve:
#?     S(x) = 1 / (1 + 10^(alpha * (x_0 - x)))
#?   Values beyond upper_x are smoothly clamped to 1.0; values below lower_x drop to 0.0.
#?   The THRESHOLD constants define the cutoff for considering a candidate 'desired'
#?   in Pareto dominance calculations.
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


#? ============================================================================
#? CLASS: SequenceRNNOptimizer
#? ============================================================================
#? WHAT:
#?   Stage 1 Generative Optimizer executing policy-gradient reinforcement learning
#?   on an autoregressive LSTM language model (SequenceRNN) conditioned on multi-objective rewards.
#?
#? WHY:
#?   Serves as the broad chemical exploration engine of our Bachelor Thesis pipeline.
#?   It explores chemical space to find novel molecules that mimic the 3D binding shape of EGCG
#?   while simultaneously optimizing synthesizability and escaping catechol/polyphenol structures.
#?
#? UNDER THE HOOD:
#?   Utilizes dual neural networks:
#?     1. AGENT (SequenceRNN): The active policy network undergoing gradient updates.
#?     2. PRIOR (SequenceRNN): Frozen reference policy preserving valid chemical grammar.
#?   Both are initialized from a language model pretrained on Papyrus 05.5 (~1.5M bioactive SMILES).
class SequenceRNNOptimizer:
    def __init__(self, output_dir: Path | None = None):
        #? WHAT:
        #?   Resolves project paths and creates timestamped run directories.
        #? WHY:
        #?   Prevents filesystem collisions between consecutive runs and ensures artifacts
        #?   (models, checkpoints, logs, plots) are isolated and reproducible.
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

        #? ====================================================================
        #? VOCABULARY & DUAL-NETWORK (AGENT / PRIOR) INITIALIZATION
        #? ====================================================================
        #? WHAT:
        #?   Loads token vocabulary from file, constructs Agent and Prior LSTM networks,
        #?   and restores pretrained Papyrus 05.5 model weights.
        #? WHY:
        #?   Pretraining provides essential knowledge of chemical syntax (valences, aromaticity,
        #?   ring closure numbering). Without pretraining, an RL agent would waste millions of steps
        #?   generating syntactically invalid strings. The frozen Prior acts as a regularization
        #?   anchor preventing catastrophic forgetting during RL fine-tuning.
        #? UNDER THE HOOD:
        #?   - VocSmiles: Tokenizes SMILES into chemical tokens (e.g. 'C', 'c', '[nH]', '(', '=', '1')
        #?     and maps each token to an integer index.
        #?   - SequenceRNN (LSTM): Multi-layer Long Short-Term Memory network. At step t,
        #?     computes hidden state h_t, c_t = LSTM(embed(token_t), h_{t-1}, c_{t-1}) and
        #?     projects h_t via a linear layer to vocabulary logits. Autoregressive sampling
        #?     draws next token t_{t+1} ~ Softmax(W * h_t + b).
        #?   - loadStatesFromFile: Deserializes PyTorch state dict containing learned LSTM matrices.
        self.VOC = VocSmiles.fromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.vocab")
        self.AGENT = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        self.AGENT.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.pkg")

        self.PRIOR = SequenceRNN(self.VOC, is_lstm=True, use_gpus=GPUS)
        self.PRIOR.loadStatesFromFile(self.PRETRAINED_DIR / "Papyrus05.5_smiles_rnn_PT.pkg")

    #? ========================================================================
    #? MULTI-OBJECTIVE REINFORCEMENT LEARNING ENVIRONMENT (MDP ENGINE)
    #? ========================================================================
    #? WHAT:
    #?   Constructs the DrugExEnvironment MDP reward generator comprising five
    #?   complementary and competing objective scoring functions:
    #?     1. ROCS 3D Shape & Color Overlap (TanimotoCombo against EGCG reference)
    #?     2. Retrosynthetic Accessibility Score (RAscore AI synthesizability)
    #?     3. Inverted 2D Morgan Fingerprint Similarity (Forced Scaffold Hopping)
    #?     4. Quantitative Estimate of Drug-likeness (QED)
    #?     5. Synthetic Accessibility Complexity Penalty (SA score)
    #?
    #? WHY (Bachelor Thesis - Multi-Parameter Optimization & Scaffold Hopping):
    #?   - Achieving 3D shape similarity alone often yields unsynthesizable molecules or
    #?     trivial analogs that retain toxic catechols.
    #?   - The combination of ROCS (rewarding 3D bio-isosterism) and Inverted 2D Novelty
    #?     (penalizing 2D chemical similarity to EGCG) is the CORE SCIENTIFIC ENGINE of our thesis:
    #?     it forces the neural network to discover novel, non-polyphenolic core scaffolds
    #?     that project pharmacophore features into the exact same 3D binding coordinates.
    #?
    #? UNDER THE HOOD:
    #?   - RDKitConformerGenerator: Generates 3D coordinates on the fly for de novo SMILES
    #?     using ETKDG. Limited to max_conformers=2 with a 10s timeout to prevent RL workers
    #?     from hanging on pathological macrocycles.
    #?   - RDKitROCSScorer: Integrates overlapping 3D Gaussian electron density spheres
    #?     between generated molecules and the minimized EGCG reference SDF. Computes
    #?     TanimotoCombo = ShapeTanimoto + ColorTanimoto (range 0.0 - 2.0).
    #?   - SmoothClippedScore: Maps raw metric x to smooth sigmoidal desirability S(x) in [0, 1].
    #?     Formula: S(x) = 1 / (1 + 10^(alpha * (x_0 - x))). Eliminates step-function discontinuities,
    #?     providing non-zero gradients across the entire transition interval.
    #?   - RetrosyntheticAccessibilityScorer (RAscore): Deep neural network / random forest
    #?     trained on 200,000+ AI-planned synthetic routes from AiZynthFinder. Predicts the
    #?     probability P(synthesizable) in [0, 1] directly from Morgan fingerprints.
    #?   - TverskyFingerprintSimilarity (Inverted 2D Novelty):
    #?     Computes Tversky index T(A, B) = |A ∩ B| / (|A ∩ B| + alpha|A \ B| + beta|B \ A|)
    #?     (with alpha=beta=1.0, equivalent to Tanimoto over 2048-bit ECFP6).
    #?     Using inverted SmoothClippedScore(lower_x=0.60, upper_x=0.25):
    #?       - Similarity > 0.60 -> Desirability drops to 0.0 (penalizes EGCG-like catechols).
    #?       - Similarity < 0.25 -> Desirability rises to 1.0 (rewards radical scaffold hops).
    #?   - Property('QED'): Bickerton et al. Drug-likeness desirability function combining
    #?     MW, ALogP, HBD, HBA, PSA, ROTB, AROM, and structural alerts.
    #?   - Property('SA'): Ertl & Schuffenhauer synthetic complexity penalty (1=easy, 10=hard),
    #?     inverted so lower SA values yield higher desirability.
    #?   - ParetoTanimotoDistance: Non-dominated sorting identifies the Pareto frontier
    #?     in 5D objective space. Solutions on the non-dominated front receive top scalar rewards,
    #?     weighted by chemical Tanimoto crowding distance to penalize redundant clusters and
    #?     encourage chemotype diversity.
    #?   - DrugExEnvironment: Bundles scorers, thresholds, and Pareto reward calculation into
    #?     a unified API that evaluates candidate SMILES batches and returns scalar rewards R.
    def build_environment(self, ref_sdf_path: Path | str, seed_smiles: str = MY_MOLECULE):
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

        # Inverted 2D Novelty: Penalizes 2D Morgan fingerprint similarity to seed to force chemotype hopping
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

    #? ========================================================================
    #? REINFORCEMENT LEARNING EXECUTION LOOP (POLICY ITERATION)
    #? ========================================================================
    #? WHAT:
    #?   Executes policy-gradient reinforcement learning on the SequenceRNN agent
    #?   using SequenceExplorer, supervised by FileMonitor for convergence logging.
    #? WHY:
    #?   Transforms the broad Papyrus pre-trained chemical language model into a targeted
    #?   generator specialized in producing synthesizable, non-polyphenolic 3D mimics of EGCG.
    #? UNDER THE HOOD:
    #?   - SequenceExplorer: Coordinates the REINFORCE policy iteration cycle:
    #?       1. Trajectory Rollout: Agent samples N=200 SMILES autoregressively token by token.
    #?       2. Epsilon-Greedy Exploration: With probability epsilon=0.10, tokens are sampled from
    #?          the frozen Prior to maintain diversity and prevent mode collapse.
    #?       3. Validity Filtering: Invalid chemical syntax receives reward R=0.
    #?       4. Multi-Objective Evaluation: Valid molecules are evaluated in parallel by DrugExEnvironment.
    #?       5. Pareto Reward Assignment: Computes non-dominated ranks and Tanimoto crowding distances.
    #?       6. Policy Gradient Loss:
    #?            L_RL = - 1/N * sum_i [ sum_t log P_agent(t_{i,t} | t_{i,<t}) * (R_i - baseline) ]
    #?                    + beta * KL(P_agent || P_prior)
    #?       7. BPTT & Optimization: Backpropagates gradients through time and updates LSTM weights via Adam.
    #?   - FileMonitor: Logs loss_train, valid_ratio, unique_ratio, and desired_ratio per epoch to TSV.
    #?   - Early Stopping: Tracks 'desired_ratio' and aborts training if no improvement is seen
    #?     within the patience window, saving checkpoint weights.
    def run_reinforcement_learning(self, molecule_smile: str = MY_MOLECULE, epochs: int = RL_EPOCHS, n_samples: int = RL_N_SAMPLES):
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

    #? ========================================================================
    #? TRAINING VISUALIZATIONS & ARTIFACT EXPORT
    #? ========================================================================
    #? WHAT:
    #?   Renders training curve diagnostics, generates and scores final de novo candidates,
    #?   exports candidate tables (stage1_hits.tsv), and draws 2D molecular structure grids.
    #? WHY:
    #?   Provides empirical proof of reinforcement learning convergence, chemical validity,
    #?   and generates candidate datasets required by Stage 2 (Scaffold Funnel).
    #? UNDER THE HOOD:
    #?   - Matplotlib: Parses monitor TSV log and plots training loss, validity, uniqueness,
    #?     and desired ratio curves.
    #?   - agent.generate: Autoregressively samples num_samples candidate SMILES from the trained Agent.
    #?   - env.getScores: Runs the full multi-objective evaluation on generated candidates.
    #?   - RDKit Draw.MolsToGridImage: Generates 2D depiction layout via depiction coordinates
    #?     and renders a 5-column molecular structure grid PNG image.
    def save_visualizations(self, agent: SequenceRNN, env: DrugExEnvironment, model_prefix: Path, out_dir: Path, num_samples: int = 100):
        fit_file = Path(f"{model_prefix}_fit.tsv")
        if fit_file.exists():
            df_info = pd.read_csv(fit_file, sep="\t")
            metric_cols = [c for c in ["loss_train", "valid_ratio", "unique_ratio", "desired_ratio"] if c in df_info.columns]
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

        mols = [m for s in generated_sample["SMILES"][:25] if (m := Chem.MolFromSmiles(s)) is not None]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250))
            img.save(str(out_dir / "generated_molecules.png"))

        return scored_df


#? ============================================================================
#? BACKWARD-COMPATIBILITY ADAPTER CLASSES
#? ============================================================================
#? WHAT:
#?   Compatibility wrappers providing backward-compatible class and method names
#?   (MY_TRANSFER_LEARNING, RAINFORCMENT_LEARNING, ALL_METHODS).
#? WHY:
#?   Ensures seamless interoperability with legacy benchmark scripts and tutorials
#?   without altering method signatures or breaking external callers.
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
