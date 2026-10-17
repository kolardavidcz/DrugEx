from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from drugex.training.monitors import PeakRecoveryMonitor

from _david.code.analysis import ResultAnalyzer
from _david.code.environment import (
    DEFAULT_MOLECULE,
    PrepareMolecule,
    create_environment,
)
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_reference_ensemble,
    render_training_curves,
)
from _david.code.models import AlgorithmConfig, GraphTransformerOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager, get_gpu_telemetry

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class MoleculePipeline:
    def __init__(
        self,
        output_dir: Path | str | None = None,
        mode: str = "full",
        epochs: int = 35,
        n_samples: int = 200,
        val_samples: int | None = 100,
        gpu: int | str | None = "auto",
        patience: int = 25,
        render_pareto: bool = False,
        config: AlgorithmConfig | None = None,
    ) -> None:
        HardwareManager.init_runtime_environment()
        self.RENDER_PARETO: bool = render_pareto
        self.MODE: str = mode.lower()
        if config is not None:
            self.config = config
        else:
            self.config = AlgorithmConfig(
                epochs=1 if self.MODE == "test" else epochs,
                n_samples=10 if self.MODE == "test" else n_samples,
                val_samples=10 if self.MODE == "test" else val_samples,
                n_eval_samples=10 if self.MODE == "test" else 200,
                patience=patience,
            )
        self.EPOCHS: int = self.config.epochs
        self.N_SAMPLES: int = self.config.n_samples
        self.VAL_SAMPLES: int | None = self.config.val_samples
        self.GPU = gpu
        self.PATIENCE = self.config.patience
        self.preparer: PrepareMolecule = PrepareMolecule()

        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.BASE_OUTPUT_DIR: Path = (
            Path(output_dir)
            if output_dir
            else (PROJECT_ROOT / "_david" / "output" / self.RUN_ID)
        )
        self.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, molecule: str = DEFAULT_MOLECULE) -> Path:
        ref_dir = self.BASE_OUTPUT_DIR / "references"
        ref_sdf = ref_dir / "query_molecule_3d.sdf"
        ref_png = ref_dir / "reference_3d_superposition.png"
        scaffold_tsv = ref_dir / "seed_core_scaffold.tsv"

        print(f"\n[Stage 0] Generating 3D reference conformers for: {molecule[:40]}...")
        self.preparer.generate_reference_conformers(molecule, output_path=ref_sdf)
        render_reference_ensemble(ref_sdf, output_png=ref_png)
        self.preparer.extract_seed_core_scaffold(molecule, output_path=scaffold_tsv)

        stage1_dir = self.BASE_OUTPUT_DIR / "stage1_graph_transformer"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        candidates_tsv = stage1_dir / "generated_candidates.tsv"

        if self.MODE != "benchmark_only":
            print(
                f"[Stage 1] Executing RL exploration (epochs={self.EPOCHS}, samples={self.N_SAMPLES})..."
            )
            optimizer = GraphTransformerOptimizer(stage1_dir, gpu=self.GPU)
            env, epsilon = create_environment(ref_sdf)

            train_loader, val_loader, _ = optimizer.prepare_seed_fragments(
                molecule,
                n_prompts=self.N_SAMPLES,
                scaffolds_path=scaffold_tsv,
                val_samples=self.VAL_SAMPLES,
            )
            batch_size = HardwareManager.get_recommended_batch_size(self.N_SAMPLES)
            explorer = optimizer.build_explorer(
                env, epsilon=epsilon, batch_size=batch_size
            )

            model_prefix = optimizer.MODEL_DIR / "david_agent"
            monitor = PeakRecoveryMonitor(
                model_prefix, save_smiles=True, reset_directory=False
            )

            explorer.fit(
                train_loader,
                val_loader,
                epochs=self.EPOCHS,
                patience=self.PATIENCE,
                min_epochs=min(5, self.EPOCHS),
                monitor=monitor,
            )
            monitor.restore_peak_model(optimizer.AGENT)

            smiles_tsv = Path(f"{model_prefix}_smiles.tsv")
            if smiles_tsv.exists():
                s_df = pd.read_csv(smiles_tsv, sep="\t")
                std.standardize(s_df).to_csv(smiles_tsv, sep="\t", index=False)

            render_training_curves(
                f"{model_prefix}_fit.tsv",
                output_png=optimizer.MODEL_DIR / "rl_training_curves.png",
            )
            if self.RENDER_PARETO:
                render_epoch_pareto_evolution(
                    smiles_tsv,
                    output_png=optimizer.MODEL_DIR / "pareto_epoch_evolution.png",
                )

            eval_samples = 10 if self.MODE == "test" else getattr(self.config, "n_eval_samples", 200)
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

            stage2_dir = self.BASE_OUTPUT_DIR / "stage2_result_analysis"
            stage2_dir.mkdir(parents=True, exist_ok=True)
            eval_tsv = stage2_dir / f"generated_{eval_samples}_molecules.tsv"
            df.to_csv(eval_tsv, sep="\t", index=False)
            df.to_csv(candidates_tsv, sep="\t", index=False)

        print("[Stage 2] Profiling generated leads and Pareto frontier...")
        stage2_dir = self.BASE_OUTPUT_DIR / "stage2_result_analysis"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        eval_tsv = stage2_dir / f"generated_{getattr(self.config, 'n_eval_samples', 200)}_molecules.tsv"
        target_candidates = eval_tsv if eval_tsv.exists() else candidates_tsv

        analyzer = ResultAnalyzer(self.BASE_OUTPUT_DIR)
        analyzer.analyze(
            target_candidates, ref_sdf=ref_sdf, render_pareto=self.RENDER_PARETO
        )

        # Restructure output folder: move status/logs to logs/ and place key showcase files in root
        logs_dir = self.BASE_OUTPUT_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        status = {
            "run_id": self.RUN_ID,
            "mode": self.MODE,
            "status": "SUCCESS",
            "gpu_telemetry": get_gpu_telemetry(),
        }
        with open(
            logs_dir / "pipeline_status.json", "w", encoding="utf-8"
        ) as f:
            json.dump(status, f, indent=4)

        for log_name in ["pipeline.log", "run.pid"]:
            src = self.BASE_OUTPUT_DIR / log_name
            if src.is_file():
                shutil.move(str(src), str(logs_dir / log_name))

        # Showcase condition: only 4 files in run root (property_distributions, lead image, lead tsv, rl_curves)
        for fname in ["property_distributions.png", "lead_structures_2d.png", "top_prioritized_leads.tsv"]:
            src = stage2_dir / fname
            if src.is_file():
                shutil.copy2(src, self.BASE_OUTPUT_DIR / fname)

        if self.MODE != "benchmark_only":
            rl_curves = optimizer.MODEL_DIR / "rl_training_curves.png"
            if rl_curves.is_file():
                shutil.copy2(rl_curves, self.BASE_OUTPUT_DIR / "rl_training_curves.png")

        print(f"\n[DONE] Outputs saved to: {self.BASE_OUTPUT_DIR}")
        return self.BASE_OUTPUT_DIR
