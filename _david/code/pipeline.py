from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from drugex.training.monitors import PeakRecoveryMonitor

from _david.code.analysis import ResultAnalyzer
from _david.code.environment import (
    DEFAULT_MOLECULE,
    PrepareMolecule,
    create_environment,
)
from _david.code.models import AlgorithmConfig, GraphTransformerOptimizer
from _david.code.schema import std
from _david.code.utils import HardwareManager, get_gpu_telemetry
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_reference_ensemble,
    render_training_curves,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class MoleculePipeline:
    """Production Ligand-Based De Novo Design (LBDD) pipeline in DrugEx.

    WHY:
    Orchestrates end-to-end multi-objective 3D ROCS shape/color alignment and synthetic
    accessibility optimization with a clean direct-to-disk directory structure.

    WHAT:
    - Stage 0: Generates pocket reference 3D conformers and extracts seed core scaffold.
    - Stage 1: RL exploration using GraphTransformer with continuous multi-objective reward.
    - Stage 2: Final candidate generation, lead prioritization, and 2D/3D presentation assets.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        is_test: bool = False,
        epochs: int = 35,
        n_samples: int = 200,
        val_samples: int | None = 100,
        n_eval_samples: int = 200,
        gpu: int | str | None = "auto",
        patience: int = 25,
        cpu_ratio: float | None = None,
        render_pareto: bool = False,
        config: AlgorithmConfig | None = None,
    ) -> None:
        HardwareManager.init_runtime_environment(ratio=cpu_ratio)
        self.cpu_ratio: float | None = cpu_ratio
        self.RENDER_PARETO: bool = render_pareto
        self.IS_TEST: bool = is_test

        if config is not None:
            self.config = config
        else:
            self.config = AlgorithmConfig(
                epochs=1 if is_test else epochs,
                n_samples=10 if is_test else n_samples,
                val_samples=10 if is_test else val_samples,
                n_eval_samples=10 if is_test else n_eval_samples,
                patience=patience,
                criteria="avg_amean",
            )

        self.EPOCHS: int = self.config.epochs
        self.N_SAMPLES: int = self.config.n_samples
        self.VAL_SAMPLES: int | None = self.config.val_samples
        self.N_EVAL_SAMPLES: int = getattr(self.config, "n_eval_samples", 200)
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
        models_dir = self.BASE_OUTPUT_DIR / "models"
        raw_data_dir = self.BASE_OUTPUT_DIR / "raw_data"
        logs_dir = raw_data_dir / "logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        ref_sdf = raw_data_dir / "query_molecule_3d.sdf"
        ref_png = raw_data_dir / "reference_3d_superposition.png"
        scaffold_tsv = raw_data_dir / "seed_core_scaffold.tsv"

        # Stage 0: 3D Reference Conformers & Core Scaffold
        print(f"\n[Stage 0] Generating 3D reference conformers for: {molecule[:40]}...")
        self.preparer.generate_reference_conformers(molecule, output_path=ref_sdf)
        render_reference_ensemble(ref_sdf, output_png=ref_png)
        self.preparer.extract_seed_core_scaffold(molecule, output_path=scaffold_tsv)

        # Stage 1: Reinforcement Learning Exploration
        print(
            f"[Stage 1] Executing RL exploration (epochs={self.EPOCHS}, samples={self.N_SAMPLES}, criteria={self.config.criteria})..."
        )
        optimizer = GraphTransformerOptimizer(raw_data_dir, gpu=self.GPU)
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
            model_prefix,
            save_smiles=True,
            reset_directory=False,
            criteria=self.config.criteria,
        )

        explorer.fit(
            train_loader,
            val_loader,
            epochs=self.EPOCHS,
            patience=self.PATIENCE,
            min_epochs=min(5, self.EPOCHS),
            criteria=self.config.criteria,
            monitor=monitor,
        )
        monitor.restore_peak_model(optimizer.AGENT)

        smiles_tsv = Path(f"{model_prefix}_smiles.tsv")
        if smiles_tsv.exists():
            s_df = pd.read_csv(smiles_tsv, sep="\t")
            std.standardize(s_df).to_csv(smiles_tsv, sep="\t", index=False)

        render_training_curves(
            f"{model_prefix}_fit.tsv",
            output_png=self.BASE_OUTPUT_DIR / "rl_training_curves.png",
        )
        if self.RENDER_PARETO:
            render_epoch_pareto_evolution(
                smiles_tsv,
                output_png=self.BASE_OUTPUT_DIR / "pareto_epoch_evolution.png",
            )

        # Stage 2: Final Candidate Generation from Peak Agent
        eval_samples = self.N_EVAL_SAMPLES
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

        eval_tsv = raw_data_dir / f"generated_{eval_samples}_molecules.tsv"
        candidates_tsv = raw_data_dir / "generated_candidates.tsv"
        df.to_csv(eval_tsv, sep="\t", index=False)
        df.to_csv(candidates_tsv, sep="\t", index=False)

        # Stage 3: Lead Prioritization & Deliverables
        print("[Stage 2] Profiling generated leads and Pareto frontier...")
        analyzer = ResultAnalyzer(self.BASE_OUTPUT_DIR)
        analyzer.analyze(
            eval_tsv, ref_sdf=ref_sdf, render_pareto=self.RENDER_PARETO
        )

        # Clean auxiliary result files to raw_data/
        result_json = self.BASE_OUTPUT_DIR / "result_analysis_summary.json"
        if result_json.is_file():
            shutil.move(str(result_json), str(raw_data_dir / "result_analysis_summary.json"))
        aligned_png = self.BASE_OUTPUT_DIR / "aligned_lead_depictions.png"
        if aligned_png.is_file():
            shutil.move(str(aligned_png), str(raw_data_dir / "aligned_lead_depictions.png"))

        # Move model checkpoints to models/
        for pkg_file in optimizer.MODEL_DIR.glob("*.pkg"):
            shutil.move(str(pkg_file), str(models_dir / pkg_file.name))

        # Move intermediate training files from optimizer.MODEL_DIR to raw_data/
        for f in optimizer.MODEL_DIR.iterdir():
            dest = raw_data_dir / f.name
            if f.is_file():
                shutil.move(str(f), str(dest))
            elif f.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(f), str(dest))
        shutil.rmtree(optimizer.MODEL_DIR, ignore_errors=True)

        # Move top-level log files if present to logs/
        for log_name in ["pipeline.log", "run.pid"]:
            src = self.BASE_OUTPUT_DIR / log_name
            if src.is_file():
                shutil.move(str(src), str(logs_dir / log_name))

        status = {
            "run_id": self.RUN_ID,
            "status": "SUCCESS",
            "gpu_telemetry": get_gpu_telemetry(),
        }
        with open(logs_dir / "pipeline_status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4)

        # Generate summary.md report straight in root
        summary_md = self.BASE_OUTPUT_DIR / "summary.md"
        top_leads_tsv = self.BASE_OUTPUT_DIR / "top_prioritized_leads.tsv"
        top_leads_df = (
            pd.read_csv(top_leads_tsv, sep="\t")
            if top_leads_tsv.is_file()
            else pd.DataFrame()
        )

        md_lines = [
            f"# DrugEx Run Summary: {self.RUN_ID}",
            "",
            f"- **Target Molecule**: `{molecule[:60]}...`",
            f"- **Epochs**: `{self.EPOCHS}`",
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

        print(f"\n[DONE] Outputs saved to: {self.BASE_OUTPUT_DIR}")
        return self.BASE_OUTPUT_DIR
