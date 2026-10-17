from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from drugex.training.monitors import PeakRecoveryMonitor

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.analysis import ResultAnalyzer
from _david.code.config import PipelineConfig, RunPaths
from _david.code.environment import (
    DEFAULT_MOLECULE,
    PrepareMolecule,
    create_environment,
)
from _david.code.models import GraphTransformerOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager, get_gpu_telemetry
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_reference_ensemble,
    render_training_curves,
)


class MoleculePipeline:
    """Production Ligand-Based De Novo Design (LBDD) pipeline in DrugEx.

    WHY:
    Orchestrates end-to-end multi-objective 3D ROCS shape/color alignment and synthetic
    accessibility optimization with an authoritative single source of truth and direct-to-disk targets.

    WHAT:
    - Stage 0: Generates pocket reference 3D conformers and extracts seed core scaffold.
    - Stage 1: RL exploration using GraphTransformer with continuous multi-objective reward.
    - Stage 2: Final candidate generation, lead prioritization, and direct-to-disk presentation deliverables.
    """

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
            max_reference_conformers=1,
            random_seed=self.config.random_seed,
        )

    def run(self, molecule: str = DEFAULT_MOLECULE) -> Path:
        """Execute end-to-end multi-objective LBDD RL exploration and lead prioritization."""
        self.paths.create_directories()

        ref_sdf = self.paths.raw_data_dir / "query_molecule_3d.sdf"
        ref_png = self.paths.raw_data_dir / "reference_3d_superposition.png"
        scaffold_tsv = self.paths.raw_data_dir / "seed_core_scaffold.tsv"

        # Stage 0: 3D Reference Conformers & Core Scaffold
        print(f"\n[Stage 0] Generating 3D reference conformers for: {molecule[:40]}...")
        self.preparer.generate_reference_conformers(molecule, output_path=ref_sdf)
        render_reference_ensemble(ref_sdf, output_png=ref_png)
        self.preparer.extract_seed_core_scaffold(molecule, output_path=scaffold_tsv)

        # Stage 1: Reinforcement Learning Exploration
        print(
            f"[Stage 1] Executing RL exploration (epochs={self.config.epochs}, samples={self.config.n_samples}, criteria={self.config.criteria})..."
        )
        optimizer = GraphTransformerOptimizer(
            models_dir=self.paths.models_dir,
            raw_data_dir=self.paths.raw_data_dir,
            gpu=self.config.gpu,
        )
        env, epsilon = create_environment(
            ref_sdf,
            thresholds=self.config.thresholds,
            epsilon=self.config.epsilon,
            max_conformers=self.config.max_conformers,
            timeout=self.config.timeout_seconds,
        )

        train_loader, val_loader, _ = optimizer.prepare_seed_fragments(
            molecule,
            n_prompts=self.config.n_samples,
            scaffolds_path=scaffold_tsv,
            val_samples=self.config.val_samples,
        )
        batch_size = HardwareManager.get_recommended_batch_size(self.config.n_samples)
        explorer = optimizer.build_explorer(
            env,
            epsilon=epsilon,
            beta=self.config.beta,
            batch_size=batch_size,
        )

        # Model prefix in raw_data keeps intermediate monitoring TSVs in raw_data
        model_prefix = self.paths.raw_data_dir / "david_agent"
        monitor = PeakRecoveryMonitor(
            model_prefix,
            save_smiles=True,
            reset_directory=False,
            criteria=self.config.criteria,
        )

        explorer.fit(
            train_loader,
            val_loader,
            epochs=self.config.epochs,
            patience=self.config.patience,
            min_epochs=min(self.config.min_epochs, self.config.epochs),
            criteria=self.config.criteria,
            monitor=monitor,
        )
        monitor.restore_peak_model(optimizer.AGENT)

        # Move final checkpoint packages directly to models_dir
        for pkg in self.paths.raw_data_dir.glob("*.pkg"):
            shutil.move(str(pkg), str(self.paths.models_dir / pkg.name))

        # Standardize smiles TSV if generated
        smiles_tsv = Path(f"{model_prefix}_smiles.tsv")
        if smiles_tsv.exists():
            s_df = pd.read_csv(smiles_tsv, sep="\t")
            std.standardize(s_df).to_csv(smiles_tsv, sep="\t", index=False)

        # Render training curves straight to root deliverables
        fit_tsv = Path(f"{model_prefix}_fit.tsv")
        if fit_tsv.exists():
            render_training_curves(
                fit_tsv,
                output_png=self.paths.root / "rl_training_curves.png",
            )
        if self.config.render_pareto and smiles_tsv.exists():
            render_epoch_pareto_evolution(
                smiles_tsv,
                output_png=self.paths.root / "pareto_epoch_evolution.png",
            )

        # Stage 2: Final Candidate Generation from Peak Agent
        eval_samples = self.config.n_eval_samples
        print(f"\n[Final Evaluation] Generating {eval_samples} candidate molecules from peak agent...")
        eval_loader, _, _ = optimizer.prepare_seed_fragments(
            molecule,
            n_prompts=eval_samples,
            scaffolds_path=scaffold_tsv,
        )
        smiles_list, frags_list = optimizer.AGENT.sample(eval_loader)
        df = env.getScores(smiles_list, frags=frags_list)
        df.insert(0, std.SMILES, smiles_list)
        df.insert(1, std.FRAGS, frags_list)
        df = std.standardize(df)

        candidates_tsv = self.paths.raw_data_dir / "generated_candidates.tsv"
        df.to_csv(candidates_tsv, sep="\t", index=False)

        # Stage 3: Lead Prioritization & Deliverables
        print("[Stage 2] Profiling generated leads and Pareto frontier...")
        analyzer = ResultAnalyzer(
            deliverables_dir=self.paths.root,
            raw_data_dir=self.paths.raw_data_dir,
        )
        analyzer.analyze(
            candidates_tsv,
            ref_sdf=ref_sdf,
            render_pareto=self.config.render_pareto,
        )

        status = {
            "run_id": self.paths.root.name,
            "status": "SUCCESS",
            "gpu_telemetry": get_gpu_telemetry(),
        }
        with open(self.paths.logs_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4)

        # Generate summary.md report straight in root deliverables
        summary_md = self.paths.root / "summary.md"
        top_leads_tsv = self.paths.root / "top_prioritized_leads.tsv"
        top_leads_df = (
            pd.read_csv(top_leads_tsv, sep="\t")
            if top_leads_tsv.is_file()
            else pd.DataFrame()
        )

        md_lines = [
            f"# DrugEx Run Summary: {self.paths.root.name}",
            "",
            f"- **Target Molecule**: `{molecule[:60]}...`",
            f"- **Epochs**: `{self.config.epochs}`",
            f"- **Criteria**: `{self.config.criteria}`",
            f"- **Candidates Evaluated**: `{eval_samples}`",
            "",
            "## Presentation Deliverables",
            "- [Lead Structures 2D](lead_structures_2d.png)",
            "- [Property Distributions (2x5 Profile)](property_distributions.png)",
            "- [RL Training Curves](rl_training_curves.png)",
            "- [Top Prioritized Leads (TSV)](top_prioritized_leads.tsv)",
            "",
            "## Top Prioritized Chemical Leads",
            "",
        ]
        if not top_leads_df.empty:
            cols = [
                c
                for c in [
                    std.ROCS,
                    std.SA,
                    std.QED,
                    std.MW,
                    std.PARETO_RANK,
                    std.CANONICAL_SMILES,
                ]
                if c in top_leads_df.columns
            ]
            md_lines.append("| " + " | ".join(cols) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, row in top_leads_df[cols].head(5).iterrows():
                vals = [
                    f"{float(row[c]):.3f}"
                    if isinstance(row[c], (float, int)) and c != std.PARETO_RANK
                    else str(row[c])
                    for c in cols
                ]
                md_lines.append("| " + " | ".join(vals) + " |")
        else:
            md_lines.append("*No leads generated.*")
        summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        print(f"\n[DONE] Deliverables saved directly to: {self.paths.root}")
        return self.paths.root


def parse_args() -> tuple[PipelineConfig, RunPaths, str, bool]:
    """Parse CLI arguments deriving all defaults strictly from PipelineConfig."""
    parser = argparse.ArgumentParser(description="DrugEx LBDD Pipeline Runner")
    defaults = PipelineConfig()

    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of RL epochs")
    parser.add_argument("--patience", type=int, default=defaults.patience, help="Early stopping patience")
    parser.add_argument("--n-samples", type=int, default=defaults.n_samples, help="RL prompt batch size")
    parser.add_argument("--val-samples", type=int, default=defaults.val_samples, help="Validation sample size")
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=defaults.n_eval_samples,
        help="Number of final candidates generated from peak agent",
    )
    parser.add_argument("--criteria", default=defaults.criteria, help="Peak model monitoring criteria")
    parser.add_argument("--epsilon", type=float, default=defaults.epsilon, help="Exploration noise epsilon")
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE, help="Target molecule SMILES or file path")
    parser.add_argument("--gpu", default=defaults.gpu, help="GPU device ID or 'auto'/'all'/'cpu'")
    parser.add_argument(
        "--cpu-ratio",
        type=float,
        default=defaults.cpu_ratio,
        help="Fraction of CPU cores allocated for parallel scoring (default: None, uses safe 0.60)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Explicit custom output directory")
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Optional experimental phase folder name (e.g. '05_new_era')",
    )
    parser.add_argument(
        "--detach", action="store_true", default=False, help="Run as detached background daemon process"
    )
    parser.add_argument(
        "--pareto", action="store_true", default=defaults.render_pareto, help="Enable Pareto evolution visualization"
    )
    args, _ = parser.parse_known_args()

    config = PipelineConfig(
        epochs=args.epochs,
        patience=args.patience,
        n_samples=args.n_samples,
        val_samples=args.val_samples,
        n_eval_samples=args.n_eval_samples,
        criteria=args.criteria,
        epsilon=args.epsilon,
        gpu=args.gpu,
        cpu_ratio=args.cpu_ratio,
        render_pareto=args.pareto,
    )
    paths = RunPaths.resolve(base_dir=args.out_dir, phase=args.phase)
    return config, paths, args.molecule, args.detach


def main() -> None:
    config, paths, molecule, detach = parse_args()
    is_worker = os.environ.get("DRUGEX_DETACHED_WORKER") == "1"

    if detach and not is_worker:
        paths.create_directories()
        log_file = paths.logs_dir / "pipeline.log"
        pid_file = paths.logs_dir / "run.pid"

        env = dict(os.environ)
        env["DRUGEX_DETACHED_WORKER"] = "1"
        env["DRUGEX_OUTPUT_DIR"] = str(paths.root)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--epochs", str(config.epochs),
            "--patience", str(config.patience),
            "--n-samples", str(config.n_samples),
            "--val-samples", str(config.val_samples),
            "--n-eval-samples", str(config.n_eval_samples),
            "--criteria", str(config.criteria),
            "--epsilon", str(config.epsilon),
            "--molecule", molecule,
            "--gpu", str(config.gpu),
            "--out-dir", str(paths.root),
        ]
        if config.render_pareto:
            cmd.append("--pareto")
        if config.cpu_ratio is not None:
            cmd.extend(["--cpu-ratio", str(config.cpu_ratio)])

        with open(log_file, "a", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                cmd,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                start_new_session=(sys.platform != "win32"),
            )

        with open(pid_file, "w", encoding="utf-8") as f_pid:
            f_pid.write(str(proc.pid))

        print(f"Spawned detached background process PID: {proc.pid}")
        print(f"Output directory: {paths.root}")
        print(f"Live log: tail -f {log_file}")
        return

    pipeline = MoleculePipeline(config=config, paths=paths)
    pipeline.run(molecule)


if __name__ == "__main__":
    main()
