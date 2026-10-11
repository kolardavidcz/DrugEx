"""End-to-end LBDD generation pipeline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from drugex.training.monitors import PeakRecoveryMonitor

from _david.code.analysis import ResultAnalyzer
from _david.code.environment import (
    DEFAULT_MOLECULE,
    create_environment,
    extract_seed_core_scaffold,
    generate_reference_conformers,
)
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_reference_ensemble,
    render_training_curves,
)
from _david.code.models import GraphTransformerOptimizer
from _david.code.utils import get_gpu_telemetry

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class MoleculePipeline:
    """End-to-end ligand-based de novo lead generation pipeline."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        mode: str = "full",
        epochs: int = 35,
        n_samples: int = 200,
        gpu: int | str | None = "auto",
        patience: int = 25,
    ) -> None:
        self.MODE: str = mode.lower()
        self.EPOCHS: int = 1 if self.MODE == "test" else epochs
        self.N_SAMPLES: int = 10 if self.MODE == "test" else n_samples
        self.GPU = gpu
        self.PATIENCE = patience

        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.BASE_OUTPUT_DIR: Path = Path(output_dir) if output_dir else (PROJECT_ROOT / "_david" / "output" / self.RUN_ID)
        self.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, molecule: str = DEFAULT_MOLECULE) -> Path:
        """Executes full pipeline: conformer curation, RL exploration, and lead analysis."""
        ref_dir = self.BASE_OUTPUT_DIR / "references"
        ref_sdf = ref_dir / "query_molecule_3d.sdf"
        ref_png = ref_dir / "reference_3d_superposition.png"
        scaffold_tsv = ref_dir / "seed_core_scaffold.tsv"

        print(f"\n[Stage 0] Generating 3D reference conformers for: {molecule[:40]}...")
        generate_reference_conformers(molecule, output_path=ref_sdf)
        render_reference_ensemble(ref_sdf, output_png=ref_png)
        extract_seed_core_scaffold(molecule, output_path=scaffold_tsv)

        # Stage 1: Graph-Transformer RL Exploration
        stage1_dir = self.BASE_OUTPUT_DIR / "stage1_graph_transformer"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        candidates_tsv = stage1_dir / "generated_candidates.tsv"

        if self.MODE != "benchmark_only":
            print(f"[Stage 1] Executing RL exploration (epochs={self.EPOCHS}, samples={self.N_SAMPLES})...")
            optimizer = GraphTransformerOptimizer(stage1_dir, gpu=self.GPU)
            env, epsilon = create_environment(ref_sdf)

            loader, _ = optimizer.prepare_seed_fragments(molecule, n_prompts=self.N_SAMPLES, scaffolds_path=scaffold_tsv)
            explorer = optimizer.build_explorer(env, epsilon=epsilon, batch_size=50)

            model_prefix = optimizer.MODEL_DIR / "david_agent"
            monitor = PeakRecoveryMonitor(model_prefix, save_smiles=True, reset_directory=False)

            explorer.fit(loader, loader, epochs=self.EPOCHS, monitor=monitor)
            monitor.restore_peak_model(optimizer.AGENT)

            # Stage 1 Visualizations: RL Training Dynamics & Micro-Pareto Evolution
            render_training_curves(f"{model_prefix}_fit.tsv", output_png=optimizer.MODEL_DIR / "rl_training_curves.png")
            render_epoch_pareto_evolution(f"{model_prefix}_smiles.tsv", output_png=optimizer.MODEL_DIR / "pareto_epoch_evolution.png")

            smiles_list, frags_list = optimizer.AGENT.sample(loader)
            df = env.getScores(smiles_list, frags=frags_list)
            df.insert(0, "SMILES", smiles_list)
            df.insert(1, "FRAGS", frags_list)
            df.to_csv(candidates_tsv, sep="\t", index=False)

        # Stage 2: Result Analysis & Lead Profiling
        print("[Stage 2] Profiling generated leads and Pareto frontier...")
        analyzer = ResultAnalyzer(self.BASE_OUTPUT_DIR)
        analyzer.analyze(candidates_tsv, ref_sdf=ref_sdf)

        # Save status summary
        status = {
            "run_id": self.RUN_ID,
            "mode": self.MODE,
            "status": "SUCCESS",
            "gpu_telemetry": get_gpu_telemetry(),
        }
        with open(self.BASE_OUTPUT_DIR / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4)

        print(f"\n[DONE] Outputs saved to: {self.BASE_OUTPUT_DIR}")
        return self.BASE_OUTPUT_DIR
