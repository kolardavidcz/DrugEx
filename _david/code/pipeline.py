from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from drugex.logs import logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from rdkit import Chem

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.environment import PrepareMolecule, create_environment
from _david.code.models import SequenceRNNOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager
from _david.code.visualization import ResultAnalyzer, render_reference_ensemble


class MoleculePipeline:
    """Production Ligand-Based De Novo Design (LBDD) SequenceRNN pipeline in DrugEx."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        paths: RunPaths | None = None,
        output_dir: Path | str | None = None,
        **kwargs,
    ) -> None:
        self.config: PipelineConfig = config or PipelineConfig(**kwargs)
        HardwareManager.init_runtime_environment(ratio=self.config.cpu_ratio)

        if paths is not None:
            self.paths: RunPaths = paths
        elif output_dir is not None:
            self.paths = RunPaths.resolve(base_dir=output_dir)
        else:
            self.paths = RunPaths.resolve()

        self.preparer: PrepareMolecule = PrepareMolecule(
            num_conformers=100,
            max_reference_conformers=self.config.max_reference_conformers,
            random_seed=self.config.random_seed,
            energy_window_kcal=self.config.energy_window_kcal,
        )

    @staticmethod
    def sample_and_score(
        agent: SequenceRNN,
        env: DrugExEnvironment,
        n_samples: int,
        label: str,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        """Sample raw sequences from agent and evaluate multi-objective scores through the environment."""
        t0 = time.time()
        gen_df = agent.generate(
            num_samples=n_samples,
            batch_size=batch_size,
            drop_invalid=False,
            drop_duplicates=False,
            evaluator=None,
            progress=False,
        )
        smiles_list = gen_df["SMILES"].tolist()
        scores_df = env.getScores(smiles_list)
        df = pd.DataFrame({"SMILES": smiles_list})
        for col in scores_df.columns:
            df[col] = scores_df[col].values

        mols = [
            Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
            for s in smiles_list
        ]
        df[std.CANONICAL_SMILES] = [Chem.MolToSmiles(m) if m else None for m in mols]
        elapsed = time.time() - t0
        logger.info(
            f"[{label.upper()}] Sampled and scored {n_samples} molecules in {elapsed / 60:.1f} min."
        )
        return df

    def run(self, molecule: str | Path = DEFAULT_MOLECULE) -> Path:
        """Execute end-to-end multi-objective SequenceRNN RL exploration and lead prioritization."""
        self.paths.create_directories()

        ref_sdf = self.paths.raw_data_dir / "query_molecule_3d.sdf"
        ref_png = self.paths.figures_dir / "reference_3d_superposition.png"

        # Stage 0: 3D Reference Conformer Ensemble
        query_desc = str(molecule)[:40]
        logger.info(
            f"\n[Stage 0] Generating 3D reference conformer ensemble for: {query_desc}..."
        )
        self.preparer.generate_reference_conformers(molecule, output_path=ref_sdf)
        render_reference_ensemble(ref_sdf, output_png=ref_png)

        # Stage 1: Load Pretrained Models & Sample Baseline Arm
        logger.info(
            "\n[Stage 1] Loading pretrained models and sampling pre-training baseline arm..."
        )
        optimizer = SequenceRNNOptimizer(
            models_dir=self.paths.models_dir,
            gpu=self.config.gpu,
            agent_path=self.config.agent_path,
            vocab_path=self.config.vocab_path,
        )
        agent, prior, voc = optimizer.load_models()
        env = create_environment(ref_sdf, config=self.config)

        logger.info(
            f"Sampling {self.config.n_eval_samples} baseline molecules from untrained prior..."
        )
        df_before = self.sample_and_score(
            agent=agent,
            env=env,
            n_samples=self.config.n_eval_samples,
            label="baseline",
            batch_size=self.config.batch_size,
        )
        df_before.to_csv(
            self.paths.raw_data_dir / "baseline_raw.csv.gz",
            index=False,
            compression="gzip",
        )

        if self.config.threshold_percentile is not None and not df_before.empty:
            p = float(self.config.threshold_percentile)
            t_rocs = float(np.percentile(df_before[std.ROCS].dropna(), p))
            t_sa = float(np.percentile(df_before[std.SA].dropna(), p))
            t_qed = float(np.percentile(df_before[std.QED].dropna(), p))
            env.thresholds = [t_rocs, t_sa, t_qed]
            logger.info(
                f"Calibrated dynamic empirical baseline thresholds (P{p:.0f}): "
                f"ROCS>={t_rocs:.3f}, SA>={t_sa:.3f}, QED>={t_qed:.3f}"
            )

        # Stage 2: Reinforcement Learning Exploration
        logger.info(
            f"\n[Stage 2] Executing RL exploration (epochs={self.config.epochs}, samples={self.config.n_samples}, "
            f"epsilon={self.config.epsilon}, beta={self.config.beta}, reward_scheme={self.config.reward_scheme})..."
        )
        explorer = optimizer.build_explorer(
            agent=agent,
            env=env,
            mutate=prior,
            config=self.config,
        )

        output_base = self.paths.models_dir / "SequenceRNN_reinforced"
        monitor = FileMonitor(str(output_base), save_smiles=True, reset_directory=True)

        t_start = time.time()
        explorer.fit(
            monitor=monitor,
            epochs=self.config.epochs,
            patience=self.config.patience,
            reload_interval=self.config.epochs + 1,
            criteria=self.config.criteria,
            min_epochs=self.config.min_epochs,
        )
        monitor.close()
        logger.info(
            f"RL training complete in {(time.time() - t_start) / 60:.1f} minutes."
        )

        # Save final-epoch checkpoint
        final_pkg = self.paths.models_dir / "SequenceRNN_reinforced_final_epoch.pkg"
        torch.save(agent.getModel(), str(final_pkg))
        logger.info(f"Saved final-epoch checkpoint to: {final_pkg}")

        # Stage 3: Reinforced Agent Evaluation Arm
        logger.info(
            f"\n[Stage 3] Sampling {self.config.n_eval_samples} molecules from reinforced agent..."
        )
        df_after = self.sample_and_score(
            agent=agent,
            env=env,
            n_samples=self.config.n_eval_samples,
            label="reinforced",
            batch_size=self.config.batch_size,
        )
        df_after.to_csv(
            self.paths.raw_data_dir / "reinforced_raw.csv.gz",
            index=False,
            compression="gzip",
        )

        # Stage 4: Result Analysis & Visualization Suite
        logger.info(
            "\n[Stage 4] Generating publication deliverables, comparative analytics, and lead structures..."
        )
        analyzer = ResultAnalyzer(self.paths)
        analyzer.analyze(
            df_after=df_after,
            df_before=df_before,
            ref_smiles=str(molecule),
            monitor_prefix=str(output_base),
        )

        logger.info(
            f"\n[Pipeline Complete] Authoritative deliverables generated at: {self.paths.root}"
        )
        return self.paths.root


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="DrugEx SequenceRNN De Novo Drug Design Pipeline"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="RL training epochs (default: 50)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Molecules per training epoch (default: 500)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Molecules per evaluation arm (default: 1000)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.20, help="Exploration rate (default: 0.20)"
    )
    parser.add_argument(
        "--use-decay",
        action="store_true",
        help="Enable scheduled exploration noise decay",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Terminal exploration rate (default: 0.05)",
    )
    parser.add_argument(
        "--decay-type",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Schedule decay type (default: cosine)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.0, help="Policy baseline penalty (default: 0.0)"
    )
    parser.add_argument(
        "--reward-scheme",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "pareto_crowding", "harrington"],
        help="Multi-objective reward scheme (default: weighted_sum)",
    )
    parser.add_argument(
        "--score-variant",
        type=str,
        default="tanimoto_combo",
        choices=["tanimoto_combo", "tversky_subshape"],
        help="3D ROCS scoring variant (default: tanimoto_combo)",
    )
    parser.add_argument(
        "--minimize-candidates",
        action="store_true",
        help="Apply MMFF94 after candidate ETKDG generation (default: off)",
    )
    parser.add_argument(
        "--cpu-ratio",
        type=float,
        default=0.90,
        help="CPU core utilization ratio (default: 0.90)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Authoritative output root directory",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default=DEFAULT_MOLECULE,
        help="Target reference SMILES / SDF path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(
        epochs=args.epochs,
        n_samples=args.samples,
        n_eval_samples=args.eval_samples,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        use_epsilon_decay=args.use_decay,
        decay_type=args.decay_type,
        beta=args.beta,
        reward_scheme=args.reward_scheme,
        score_variant=args.score_variant,
        minimize_candidates=args.minimize_candidates,
        cpu_ratio=args.cpu_ratio,
    )
    pipe = MoleculePipeline(config=cfg, output_dir=args.output_dir)
    pipe.run(molecule=args.molecule)
