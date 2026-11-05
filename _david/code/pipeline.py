from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from drugex.logs import logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from rdkit import Chem, rdBase

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

#: Bump when a PipelineConfig field is added, renamed or changes meaning. A manifest
#: without this version was written by a release whose config lacks the newer keys,
#: so the audit must treat those keys as unknown rather than as the current default.
CONFIG_SCHEMA_VERSION: int = 1
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.chemistry import DEFAULT_REFERENCE_SEED
from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.environment import PrepareMolecule, create_environment
from _david.code.models import SequenceRNNOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager
from _david.code.visualization import ResultAnalyzer

# Human-friendly spellings accepted by ``--criteria``. The canonical values in
# ``std`` are what the checkpoint monitor receives; the short forms exist so
# ``--criteria gmean`` (the gold standard) does not require knowing the schema
# prefix.
CRITERIA_ALIASES: dict[str, str] = {
    "gmean": std.AVG_GMEAN,
    "avg_gmean": std.AVG_GMEAN,
    "amean": std.AVG_AMEAN,
    "mean": std.AVG_AMEAN,
    "avg_amean": std.AVG_AMEAN,
    "desired_ratio": std.DESIRED_RATIO,
    "valid_ratio": std.VALID_RATIO,
    "unique_ratio": std.UNIQUE_RATIO,
}


def restore_best_checkpoint(
    agent: Any,
    models_dir: Path,
    best_epoch: Any,
) -> dict[str, Any]:
    """Load the best-validation package into ``agent`` for the evaluation arm.

    ``Explorer.fit`` writes the best weights to ``SequenceRNN_reinforced.pkg`` but
    only copies them back into the live agent on a ``reload_interval`` boundary; the
    pipeline disables that reset, so the in-memory policy is the last epoch. The arm
    must describe the model the run is named after, hence this explicit restore.
    """
    best_pkg = Path(models_dir) / "SequenceRNN_reinforced.pkg"
    has_checkpoint = best_pkg.is_file()
    if has_checkpoint and isinstance(best_epoch, int) and best_epoch >= 1:
        agent.loadStatesFromFile(str(best_pkg))
        agent.eval()
        logger.info(
            f"Restored the best checkpoint (epoch {best_epoch}) for the evaluation arm."
        )
        return {
            "evaluation_source": "best_epoch",
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": hashlib.sha256(best_pkg.read_bytes()).hexdigest(),
        }
    logger.warning(
        "No best-epoch checkpoint was recorded; the evaluation arm samples the "
        "last epoch instead."
    )
    return {
        "evaluation_source": "last_epoch",
        "best_epoch": None,
        "best_checkpoint_sha256": (
            hashlib.sha256(best_pkg.read_bytes()).hexdigest() if has_checkpoint else None
        ),
    }


class MoleculePipeline:
    """Production Ligand-Based De Novo Design (LBDD) SequenceRNN pipeline in DrugEx."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        paths: RunPaths | None = None,
        output_dir: Path | str | None = None,
        run_tag: str | None = None,
        **kwargs,
    ) -> None:
        self.config: PipelineConfig = config or PipelineConfig(**kwargs)
        HardwareManager.init_runtime_environment(ratio=self.config.cpu_ratio)

        self.run_tag: str | None = run_tag
        if paths is not None:
            self.paths: RunPaths | None = paths
        elif output_dir is not None:
            self.paths = RunPaths.resolve(base_dir=output_dir)
        else:
            # Resolved in ``run()`` so the default directory can carry the
            # molecule/tag (``_david/output/20261104_1530_egcg``).
            self.paths = None

        self.preparer: PrepareMolecule = PrepareMolecule(config=self.config)
        #: Which model produced the evaluation arm and which epoch it came from.
        #: Filled during Stage 3 and merged into the manifest by ``run``.
        self.evaluation_provenance: dict[str, Any] = {}

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
        if std.ROCS in df.columns and "Raw_ROCS" not in df.columns:
            has_modifier = any(
                hasattr(s, "getKey")
                and std.ROCS in s.getKey()
                and getattr(s, "modifier", None) is not None
                for s in env.scorers
            )
            if has_modifier:
                upper = 2.0
                for s in env.scorers:
                    if (
                        hasattr(s, "getKey")
                        and std.ROCS in s.getKey()
                        and getattr(s, "modifier", None) is not None
                    ):
                        upper = getattr(s.modifier, "upper_x", 2.0)
                        break
                df["Raw_ROCS"] = df[std.ROCS] * upper
            else:
                # Scorer has no modifier; getScores returned raw TanimotoCombo [0, 2].
                df["Raw_ROCS"] = df[std.ROCS].copy()

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

    @staticmethod
    def _molecule_tag(molecule: str | Path) -> str | None:
        """Use a molecule file's stem as the default run tag; SMILES get none."""
        path = Path(str(molecule))
        if path.suffix.lower() in {".sdf", ".mol", ".mol2", ".smi", ".csv"}:
            return path.stem
        return None

    def _load_prior_baseline(self, reference_sdf: Path) -> pd.DataFrame:
        """Load the frozen pre-training arm and prove it belongs to this target.

        A dataset sampled for a different molecule would silently poison every
        threshold and every comparison, so the canonical SMILES must match. The
        run also records the dataset identity next to the arm it replaced.
        """
        from _david.code.prior_dataset import load_arm, resolve_dataset, sha256_file

        name = str(self.config.prior_dataset)
        frame, dataset_manifest = load_arm(
            name, self.config, n=self.config.n_eval_samples
        )
        expected = None
        for mol in Chem.SDMolSupplier(str(reference_sdf), removeHs=False):
            if mol is not None:
                expected = Chem.MolToSmiles(Chem.RemoveHs(mol))
                break
        recorded = dataset_manifest.get("query_canonical_smiles")
        if expected and recorded and expected != recorded:
            raise ValueError(
                f"prior dataset {name!r} was built for {recorded!r}, "
                f"but this run targets {expected!r}"
            )
        target = resolve_dataset(name)
        provenance = {
            "name": target.name,
            "path": str(target),
            "manifest_sha256": sha256_file(target / "manifest.json"),
            "csv_sha256": sha256_file(target / "prior_molecules.csv.gz"),
            "seed": dataset_manifest.get("seed"),
            "n_used": int(len(frame)),
            "query_canonical_smiles": recorded,
        }
        (self.paths.raw_data_dir / "baseline_prior_dataset.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        logger.info(
            f"Loaded {len(frame)} baseline molecules from frozen prior dataset "
            f"{target.name} (sha256 {provenance['manifest_sha256'][:12]}…)"
        )
        return frame

    def run(self, molecule: str | Path = DEFAULT_MOLECULE) -> Path:
        """Record the actual configuration and execution outcome beside run data."""
        if self.paths is None:
            self.paths = RunPaths.resolve(
                tag=self.run_tag or self._molecule_tag(molecule)
            )
        if self.config.training_seed is not None:
            random.seed(self.config.training_seed)
            np.random.seed(self.config.training_seed)
            torch.manual_seed(self.config.training_seed)
        # Hash every module that can move a score, not only the top-level ones: the
        # conformer and environment subpackages are part of the measurement.
        excluded = {"tests", "archive", "__pycache__", "run_audit"}
        sources = sorted(
            path
            for path in Path(__file__).parent.rglob("*.py")
            if not excluded.intersection(path.relative_to(Path(__file__).parent).parts)
        )
        manifest = dict(
            state="running",
            run_id=self.paths.root.name,
            query=str(molecule),
            config=asdict(self.config),
            config_schema_version=CONFIG_SCHEMA_VERSION,
            host=socket.gethostname(),
            pbs_job_id=os.environ.get("PBS_JOBID"),
            source_revision=os.environ.get("DRUGEX_SOURCE_REVISION"),
            source_sha256={
                str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sources
            },
            versions=dict(
                python=platform.python_version(),
                rdkit=rdBase.rdkitVersion,
                torch=torch.__version__,
            ),
            training_seed=self.config.training_seed,
            determinism="Seeded RNGs when requested; deterministic GPU kernels are not enforced",
            analysis_scope={
                "final_arm": "raw_data/reinforced_raw.csv.gz",
                "final_arm_purpose": (
                    "distributional statistics, diagnostic scorecard, best-of tables "
                    "and every cross-run comparison"
                ),
                "training_trace": "models/SequenceRNN_reinforced_smiles.tsv",
                "training_trace_purpose": (
                    "longitudinal diagnostics and candidate discovery only; never "
                    "pooled into the final-arm statistics"
                ),
                "statistics_source": "final_arm",
            },
        )
        target = self.paths.root / "manifest.json"
        target.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        started = time.perf_counter()
        try:
            result = self._run(molecule)
            manifest["state"] = "complete"
            return result
        except Exception as exc:
            manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            manifest.update(self.evaluation_provenance)
            manifest["elapsed_seconds"] = time.perf_counter() - started
            ref = self.paths.raw_data_dir / "query_molecule_3d.sdf"
            if ref.exists():
                manifest["reference_sha256"] = hashlib.sha256(
                    ref.read_bytes()
                ).hexdigest()
            target.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    def _run(self, molecule: str | Path) -> Path:
        """Execute end-to-end multi-objective SequenceRNN RL exploration and lead prioritization."""
        self.paths.create_directories()

        ref_sdf = self.paths.raw_data_dir / "query_molecule_3d.sdf"

        # Stage 0: 3D Reference Conformer Ensemble
        logger.info(
            f"\n[Stage 0] Generating 3D reference conformer ensemble for: {molecule}"
        )
        self.preparer.generate_reference_conformers(molecule, output_path=ref_sdf)

        # Stage 1: Load Pretrained Models & Sample Baseline Arm
        logger.info(
            "\n[Stage 1] Loading pretrained models and sampling pre-training baseline arm..."
        )
        optimizer = SequenceRNNOptimizer(
            models_dir=self.paths.models_dir,
            config=self.config,
        )
        agent, prior, voc = optimizer.load_models()
        env = create_environment(ref_sdf, config=self.config)

        if self.config.prior_dataset:
            logger.info(
                f"Reusing frozen prior dataset {self.config.prior_dataset!r} "
                f"instead of sampling the baseline..."
            )
            df_before = self._load_prior_baseline(ref_sdf)
        else:
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
            if self.config.include_qed and std.QED in df_before.columns:
                t_qed = float(np.percentile(df_before[std.QED].dropna(), p))
                env.thresholds = [t_rocs, t_sa, t_qed]
                logger.info(
                    f"Calibrated dynamic empirical baseline thresholds (P{p:.0f}): "
                    f"ROCS>={t_rocs:.3f}, SA>={t_sa:.3f}, QED>={t_qed:.3f}"
                )
            else:
                env.thresholds = [t_rocs, t_sa]
                logger.info(
                    f"Calibrated dynamic empirical baseline thresholds (P{p:.0f}): "
                    f"ROCS>={t_rocs:.3f}, SA>={t_sa:.3f}"
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

        # Restoration of the best epoch. ``Explorer.fit`` only copies ``bestState`` back
        # into the agent every ``reload_interval`` epochs, and the pipeline disables that
        # reset so the live policy cannot silently rewind mid-run. The checkpoint on disk
        # is therefore the best epoch while the in-memory agent is the last one, and the
        # evaluation arm must describe the model the run is named after.
        self.evaluation_provenance = restore_best_checkpoint(
            agent,
            self.paths.models_dir,
            getattr(explorer, "last_save", None),
        )

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

        # A run is the moment when the overview is most out of date. Refresh it,
        # but never let a reporting failure mark a finished experiment as failed.
        try:
            from _david.code.dashboard import generate as generate_dashboard

            dashboard = generate_dashboard(self.paths.root.parent)
            logger.info(f"[Dashboard] Refreshed run overview: {dashboard}")
        except Exception as exc:  # noqa: BLE001 - reporting is best effort
            logger.warning(f"[Dashboard] Refresh skipped: {type(exc).__name__}: {exc}")
        return self.paths.root

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments with all defaults sourced from ``PipelineConfig``."""
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="DrugEx SequenceRNN De Novo Drug Design Pipeline"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help=f"RL training epochs (default: {defaults.epochs})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=defaults.n_samples,
        help=f"Molecules per training epoch (default: {defaults.n_samples})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help=f"Mini-batch size per Adam policy-gradient step (default: {defaults.batch_size})",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=defaults.n_eval_samples,
        help=f"Molecules per evaluation arm (default: {defaults.n_eval_samples})",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=defaults.epsilon,
        help=f"Exploration rate (default: {defaults.epsilon})",
    )
    parser.add_argument(
        "--use-decay",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_epsilon_decay,
        help=f"Enable scheduled exploration noise decay (default: {defaults.use_epsilon_decay})",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=defaults.epsilon_end,
        help=f"Terminal exploration rate (default: {defaults.epsilon_end})",
    )
    parser.add_argument(
        "--decay-start-epoch",
        type=int,
        default=defaults.decay_start_epoch,
        help=f"Epoch where epsilon decay begins (default: {defaults.decay_start_epoch})",
    )
    parser.add_argument(
        "--decay-epochs",
        type=int,
        default=defaults.decay_epochs,
        help=f"Epsilon decay duration epochs (default: {defaults.decay_epochs})",
    )
    parser.add_argument(
        "--decay-type",
        type=str,
        default=defaults.decay_type,
        choices=["cosine", "linear"],
        help=f"Schedule decay type (default: {defaults.decay_type})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=defaults.beta,
        help=f"Policy baseline penalty (default: {defaults.beta})",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default=defaults.criteria,
        choices=sorted(CRITERIA_ALIASES),
        help=f"RL selection criteria for best checkpoint (default: {defaults.criteria})",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=defaults.weights,
        help=(
            "Per-objective reward weights (ROCS SA [QED]); normalised by their sum. "
            f"Default auto-calibrates to {defaults.weights} "
            "(0.50/0.25/0.25 with --include-qed)"
        ),
    )
    parser.add_argument(
        "--reward-scheme",
        type=str,
        default=defaults.reward_scheme,
        choices=[std.REWARD_WEIGHTED_SUM, std.REWARD_PARETO, std.REWARD_HARRINGTON],
        help=f"Multi-objective reward scheme (default: {defaults.reward_scheme})",
    )
    parser.add_argument(
        "--score-variant",
        type=str,
        default=std.VARIANT_TANIMOTO_COMBO,
        choices=[std.VARIANT_TANIMOTO_COMBO],
        help=(
            "3D ROCS scoring variant. Only canonical TanimotoCombo is available; the flag is kept "
            "so historical submission specs replay verbatim."
        ),
    )
    parser.add_argument(
        "--color-ff",
        type=str,
        default=defaults.color_ff,
        choices=["oe_mills_dean", "default", "fdef"],
        help=(
            "Pharmacophore typing for the ROCS color channel (default: "
            f"{defaults.color_ff}); forwarded to RDKitROCSScorer(color_ff=...)"
        ),
    )
    parser.add_argument(
        "--rocs-clip-upper",
        type=float,
        default=defaults.rocs_clip_upper,
        help=f"ROCS raw score upper clipping bound (default: {defaults.rocs_clip_upper})",
    )
    parser.add_argument(
        "--rocs-use-modifier",
        action=argparse.BooleanOptionalAction,
        default=defaults.rocs_use_modifier,
        help=f"Apply ClippedScore modifier to ROCS score (default: {defaults.rocs_use_modifier})",
    )
    parser.add_argument(
        "--include-qed",
        action=argparse.BooleanOptionalAction,
        default=defaults.include_qed,
        help=f"Include QED as an RL multi-objective scoring component (default: {defaults.include_qed})",
    )
    parser.add_argument(
        "--sa-score-lower",
        type=float,
        default=defaults.sa_score_lower,
        help=f"SA raw score lower desirability cutoff (default: {defaults.sa_score_lower})",
    )
    parser.add_argument(
        "--sa-score-upper",
        type=float,
        default=defaults.sa_score_upper,
        help=f"SA raw score upper desirability cutoff (default: {defaults.sa_score_upper})",
    )
    parser.add_argument(
        "--minimize-candidates",
        action=argparse.BooleanOptionalAction,
        default=defaults.minimize_candidates,
        help=f"Minimize candidates with the selected optimization profile (default: {defaults.minimize_candidates})",
    )
    parser.add_argument(
        "--conformers",
        type=int,
        default=defaults.max_conformers,
        help=f"Candidate conformers per isomer (default: {defaults.max_conformers})",
    )
    parser.add_argument(
        "--isomers",
        type=int,
        default=defaults.max_isomers,
        help=f"Candidate stereoisomer limit (default: {defaults.max_isomers})",
    )
    parser.add_argument(
        "--reference-conformers",
        type=int,
        default=defaults.max_reference_conformers,
        help=f"Reference conformer cap K (default: {defaults.max_reference_conformers})",
    )
    parser.add_argument(
        "--reference-isomers",
        type=int,
        default=defaults.max_reference_isomers,
        help=f"Reference stereoisomer limit (default: {defaults.max_reference_isomers})",
    )
    parser.add_argument(
        "--reference-selection",
        choices=["rmsd", "diverse"],
        default=defaults.reference_selection_mode,
        help=f"Reference selection strategy (default: {defaults.reference_selection_mode})",
    )
    parser.add_argument(
        "--reference-attempts",
        type=int,
        default=defaults.reference_attempts,
        help=f"Reference ETKDGv3 attempts per isomer (default: {defaults.reference_attempts})",
    )
    parser.add_argument(
        "--reference-rmsd",
        type=float,
        default=defaults.reference_rmsd_cutoff,
        help=f"Reference Butina RMSD cutoff in Å (default: {defaults.reference_rmsd_cutoff})",
    )
    parser.add_argument(
        "--reference-seed",
        type=int,
        default=defaults.reference_random_seed,
        help=f"Reference conformer RNG seed (default: {defaults.reference_random_seed})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.random_seed,
        help=f"Candidate conformer RNG seed (default: {defaults.random_seed})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=defaults.timeout_seconds,
        help=f"Per-molecule candidate ETKDG conformer timeout in seconds (0 = unlimited, default: {defaults.timeout_seconds})",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=defaults.training_seed,
        help=f"Global Python/NumPy/Torch RNG seed (default: {defaults.training_seed})",
    )
    parser.add_argument(
        "--optimization",
        choices=["MMFF94", "MMFF94s", "UFF", "none"],
        default=None,
        help=f"Force-field optimization method (default: {defaults.optimization})",
    )
    parser.add_argument(
        "--dielectric",
        type=float,
        default=None,
        help="MMFF dielectric constant override (requires --optimization MMFF94/MMFF94s)",
    )
    parser.add_argument(
        "--conformer-threads",
        type=int,
        default=defaults.conformer_threads,
        help=f"RDKit C++ threads per conformer call (default: {defaults.conformer_threads})",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=defaults.min_epochs,
        help=f"Minimum RL epochs before early stopping (default: {defaults.min_epochs})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=defaults.patience,
        help=f"Early stopping patience epochs (default: {defaults.patience})",
    )
    parser.add_argument("--agent-path", type=Path, default=defaults.agent_path)
    parser.add_argument("--vocab-path", type=Path, default=defaults.vocab_path)
    parser.add_argument(
        "--cpu-ratio",
        type=float,
        default=defaults.cpu_ratio,
        help=f"CPU core utilization ratio (default: {defaults.cpu_ratio})",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=str(defaults.gpu),
        help=f"Preferred GPU device ('auto' prefers GPU 1 when free, or integer index / 'cpu', default: {defaults.gpu})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Exact run directory (overrides the default "
            "_david/output/YYYYmmdd_HHMM[_tag] naming)"
        ),
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Optional suffix for the default run directory, "
            "e.g. --run-tag egcg_gold -> _david/output/20261104_1530_egcg_gold"
        ),
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default=DEFAULT_MOLECULE,
        help="Target reference SMILES / SDF path",
    )
    parser.add_argument(
        "--prior-dataset",
        type=str,
        default=defaults.prior_dataset,
        help=(
            "Reuse a frozen pre-training arm from _david/output/datasets instead of "
            "re-sampling (name or path; default: sample per run)"
        ),
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Keep CLI-to-configuration translation inspectable and usable by preflight."""
    from _david.code.chemistry import Optimization

    defaults = PipelineConfig()
    if args.dielectric is not None and args.optimization not in {"MMFF94", "MMFF94s"}:
        raise ValueError("A dielectric override requires an explicit MMFF profile")
    if args.optimization is None:
        opt = defaults.optimization
    elif args.optimization == "none":
        opt = None
    else:
        opt = Optimization(
            args.optimization,
            1.0 if args.dielectric is None else args.dielectric,
        )
    return PipelineConfig(
        epochs=args.epochs,
        n_samples=args.samples,
        batch_size=args.batch_size,
        n_eval_samples=args.eval_samples,
        prior_dataset=args.prior_dataset,
        epsilon=args.epsilon,
        epsilon_end=args.epsilon_end,
        use_epsilon_decay=args.use_decay,
        decay_start_epoch=args.decay_start_epoch,
        decay_epochs=args.decay_epochs,
        decay_type=args.decay_type,
        beta=args.beta,
        criteria=CRITERIA_ALIASES[args.criteria],
        reward_scheme=args.reward_scheme,
        weights=tuple(args.weights) if args.weights is not None else None,
        score_variant=args.score_variant,
        color_ff=args.color_ff,
        rocs_clip_upper=args.rocs_clip_upper,
        rocs_use_modifier=args.rocs_use_modifier,
        include_qed=args.include_qed,
        sa_score_lower=args.sa_score_lower,
        sa_score_upper=args.sa_score_upper,
        minimize_candidates=args.minimize_candidates,
        gpu=int(args.gpu) if str(args.gpu).strip().isdigit() else args.gpu,
        cpu_ratio=args.cpu_ratio,
        max_conformers=args.conformers,
        max_isomers=args.isomers,
        max_reference_conformers=args.reference_conformers,
        max_reference_isomers=args.reference_isomers,
        reference_selection_mode=args.reference_selection,
        reference_attempts=args.reference_attempts,
        reference_rmsd_cutoff=args.reference_rmsd,
        reference_random_seed=args.reference_seed,
        random_seed=args.seed,
        training_seed=args.training_seed,
        timeout_seconds=args.timeout,
        optimization=opt,
        conformer_threads=args.conformer_threads,
        min_epochs=args.min_epochs,
        patience=args.patience,
        agent_path=args.agent_path,
        vocab_path=args.vocab_path,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = config_from_args(args)
    pipe = MoleculePipeline(
        config=cfg, output_dir=args.output_dir, run_tag=args.run_tag
    )
    pipe.run(molecule=args.molecule)
