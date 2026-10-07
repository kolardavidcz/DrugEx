"""End-to-End Ligand-Based De Novo Drug Design Pipeline in DrugEx."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import torch
from rdkit import Chem

from _david.code import (
    GraphTransformer,
    prepare_query_conformer,
    thesis_benchmark,
)
from _david.code.fallback_monitor import FallbackMonitor
from _david.code.result_analyzer import ResultAnalyzer


def get_gpu_telemetry() -> dict[str, Any]:
    """Gather current GPU device and memory metrics."""
    telemetry: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
            total_mem = props.total_memory / (1024 ** 2)
            telemetry["devices"].append({
                "index": i,
                "name": props.name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "max_allocated_mb": round(max_alloc, 2),
                "total_memory_mb": round(total_mem, 2),
            })
    return telemetry


DEFAULT_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
DEFAULT_EPOCHS: int = 10
DEFAULT_N_SAMPLES: int = 200
DEFAULT_BATCH_SIZE: int = 64
TEST_EPOCHS: int = 1
TEST_N_SAMPLES: int = 50
MAX_SCAFFOLDS: int = 5
RANDOM_STATE: int = 42


class MoleculePipeline:
    """End-to-End Ligand-Based De Novo Drug Design Pipeline in DrugEx."""

    def __init__(
        self,
        output_dir: Path | None = None,
        mode: str = "full",
        epochs: int = DEFAULT_EPOCHS,
        n_samples: int = DEFAULT_N_SAMPLES,
    ) -> None:
        """Initialize MoleculePipeline with execution mode and hyperparameter settings.

        Args:
            output_dir: Custom root output directory for run artifacts.
            mode: Pipeline mode ('full', 'test', or 'benchmark_only').
            epochs: RL training epochs per stage.
            n_samples: Number of molecules sampled per epoch.
        """
        self.ROOT_PATH: Path = PROJECT_ROOT
        self.MODE: str = mode.lower()
        self.EPOCHS: int = TEST_EPOCHS if self.MODE == "test" else epochs
        self.N_SAMPLES: int = TEST_N_SAMPLES if self.MODE == "test" else n_samples
        #? [DateTime] Generates ISO-like timestamp string for run directory isolation.
        self.RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_dir is None:
            self.BASE_OUTPUT_DIR: Path = self.ROOT_PATH / "_david" / "output" / self.RUN_ID
        else:
            self.BASE_OUTPUT_DIR = Path(output_dir)
        self.BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.PRETRAINED_RNN: Path = self.ROOT_PATH / "data" / "models" / "pretrained" / "smiles-rnn" / "Papyrus05.5_smiles_rnn_PT"
        self.PRETRAINED_GRAPH: Path = self.ROOT_PATH / "data" / "models" / "pretrained" / "graph-trans" / "Papyrus05.5_graph_trans_PT"

        # Execution tracking and telemetry
        self._start_time: datetime = datetime.now()
        self._stage_statuses: dict[str, str] = {}
        self.update_status("INITIALIZING", "RUNNING")

    def log(self, message: str) -> None:
        """Write timestamped message to both stdout and pipeline.log.

        Args:
            message: Text string to log.
        """
        print(message)
        log_file = self.BASE_OUTPUT_DIR / "pipeline.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")

    def update_status(
        self, current_stage: str, status: str, details: dict[str, Any] | None = None
    ) -> None:
        """Update live status file pipeline_status.json for external monitoring.

        Args:
            current_stage: Human-readable stage identifier.
            status: Stage status string ('RUNNING', 'SUCCESS', 'FAILED').
            details: Optional dictionary of supplementary execution metrics.
        """
        elapsed = (datetime.now() - self._start_time).total_seconds() if hasattr(self, "_start_time") else 0.0
        self._stage_statuses[current_stage] = status
        status_data: dict[str, Any] = {
            "run_id": self.RUN_ID,
            "mode": self.MODE,
            "current_stage": current_stage,
            "overall_status": status,
            "elapsed_seconds": round(elapsed, 2),
            "updated_at": datetime.now().isoformat(),
            "stages": dict(self._stage_statuses),
            "gpu_telemetry": get_gpu_telemetry(),
            "details": details or {},
        }
        status_file = self.BASE_OUTPUT_DIR / "pipeline_status.json"
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=4)

    def parse_molecules(self, molecules_input: str | Path | Sequence[str]) -> list[str]:
        """Parse, validate, and standardize input query molecules from string, file, or list.

        Args:
            molecules_input: Input SMILES string, path to molecule file, or sequence of SMILES.

        Returns:
            List of valid canonical SMILES strings.
        """
        raw_smiles: list[str] = []

        if isinstance(molecules_input, (list, tuple)):
            for item in molecules_input:
                s = str(item).strip()
                if s:
                    raw_smiles.append(s)
        elif isinstance(molecules_input, Path) or (isinstance(molecules_input, str) and Path(molecules_input).exists()):
            file_path = Path(molecules_input)
            if file_path.suffix.lower() == ".sdf":
                #? [RDKit SDMolSupplier] Parses MDL multi-record SDF coordinates.
                suppl = Chem.SDMolSupplier(str(file_path))
                for m in suppl:
                    if m is not None:
                        raw_smiles.append(Chem.MolToSmiles(m))
            else:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tokens = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
                        if tokens:
                            raw_smiles.append(tokens[0])
        elif isinstance(molecules_input, str):
            for part in molecules_input.replace(",", "\n").replace(";", "\n").splitlines():
                part = part.strip()
                if part:
                    raw_smiles.append(part)

        valid_molecules: list[str] = []
        for smi in raw_smiles:
            #? [RDKit] Validates chemical syntax and converts to canonical isomeric SMILES.
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                can_smi = Chem.MolToSmiles(m)
                if can_smi not in valid_molecules:
                    valid_molecules.append(can_smi)

        if not valid_molecules:
            raise ValueError(f"No valid molecules found in input: {molecules_input}")

        return valid_molecules

    def stage0_prepare_reference_conformer(self, primary_molecule: str) -> tuple[Path, Path]:
        """Generate 3D reference conformer ensemble, moments, and core scaffold for query molecule.

        Args:
            primary_molecule: SMILES string of the primary proof molecule.

        Returns:
            Tuple of (reference SDF path, seed core scaffold TSV path).
        """
        ref_dir: Path = self.BASE_OUTPUT_DIR / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_sdf: Path = ref_dir / "query_molecule_3d.sdf"
        seed_core_tsv: Path = ref_dir / "seed_core_scaffold.tsv"

        print("\n[Stage 0] Generating 3D reference conformer ensemble...")
        preparer = prepare_query_conformer.QueryConformerPreparer(references_dir=ref_dir)
        preparer.generate_reference_sdf(smiles=primary_molecule, output_path=ref_sdf)
        preparer.extract_seed_core_scaffold(smiles=primary_molecule, output_path=seed_core_tsv)

        # Ensure shared references directory is also populated
        shared_ref = self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf"
        shared_ref.parent.mkdir(parents=True, exist_ok=True)
        if not shared_ref.exists():
            preparer.generate_reference_sdf(smiles=primary_molecule, output_path=shared_ref)

        print(f"[Stage 0] Reference conformers saved to: {ref_sdf}")
        print(f"[Stage 0] Seed core scaffold saved to: {seed_core_tsv}")
        return ref_sdf, seed_core_tsv

    def stage1_graph_transformer_generation(
        self,
        primary_molecule: str,
        ref_sdf: Path,
        seed_core_tsv: Path | str | None = None,
    ) -> tuple[pd.DataFrame, Path]:
        """Execute Stage 1 Graph-Transformer lead generation conditioned on query seed core framework.

        Args:
            primary_molecule: SMILES of the proof molecule.
            ref_sdf: Path to 3D conformer ensemble reference SDF.
            seed_core_tsv: Path to TSV containing the seed core scaffold.

        Returns:
            Tuple of (generated candidates DataFrame, TSV file path).
        """
        stage1_dir: Path = self.BASE_OUTPUT_DIR / "stage1_graph_transformer"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        candidates_tsv: Path = stage1_dir / "generated_candidates.tsv"

        print("\n[Stage 1] Graph-Transformer Ligand-Based Lead Generation (Seed Core RL)...")
        print(f"[Stage 1] Executing Graph-Transformer RL exploration (epochs={self.EPOCHS}, samples={self.N_SAMPLES})...")

        from drugex.training.explorers import FragGraphExplorer
        from drugex.training.monitors import PeakRecoveryMonitor

        optimizer = GraphTransformer.GraphTransformerOptimizer(output_dir=stage1_dir)
        env = optimizer.build_environment(
            ref_sdf,
            seed_smiles=primary_molecule,
        )
        seed_loader, seed_dataset = optimizer._prepare_seed_fragments(
            primary_molecule, n_prompts=self.N_SAMPLES, scaffolds_path=seed_core_tsv
        )

        #? [DrugEx FragGraphExplorer] Self-attention graph transformer optimizing lead frameworks against environment.
        explorer = FragGraphExplorer(
            agent=optimizer.AGENT,
            env=env,
            mutate=optimizer.PRIOR,
            epsilon=GraphTransformer.RL_EPSILON,
            beta=GraphTransformer.RL_BETA,
            batch_size=GraphTransformer.RL_BATCH_SIZE,
            use_gpus=GraphTransformer.GPUS,
        )

        model_prefix = optimizer.MODEL_DIR / "david_agent"
        patience = min(GraphTransformer.RL_PATIENCE, max(5, self.EPOCHS // 2))
        min_epochs = min(5, self.EPOCHS)

        monitor = PeakRecoveryMonitor(
            model_prefix,
            save_smiles=True,
            reset_directory=True,
            criteria="desired_ratio",
            patience=patience,
            min_epochs=min_epochs,
        )

        explorer.fit(
            seed_loader,
            seed_loader,
            monitor=monitor,
            epochs=self.EPOCHS,
            patience=patience,
            min_epochs=min_epochs,
            criteria="desired_ratio",
        )

        # Restore peak model weights to protect against late-epoch collapse
        monitor.restore_peak_model(optimizer.AGENT)

        df_candidates = optimizer.save_visualizations(
            optimizer.AGENT, env, model_prefix, optimizer.MODEL_DIR, input_dataset=seed_dataset, num_samples=self.N_SAMPLES
        )
        if not candidates_tsv.exists():
            df_candidates.to_csv(candidates_tsv, sep="\t", index=False)

        print(f"[Stage 1] Completed. Generated {len(df_candidates)} candidates saved to: {candidates_tsv}")
        return df_candidates, candidates_tsv

    def stage2_result_analysis(
        self,
        candidate_df: pd.DataFrame,
        primary_molecule: str,
        ref_sdf: Path,
    ) -> tuple[pd.DataFrame, Path, Path]:
        """Execute Stage 2 comprehensive result analysis, 3D ROCS profiling, and visualization.

        Args:
            candidate_df: DataFrame of raw generated molecules from Stage 1.
            primary_molecule: Query seed proof SMILES string.
            ref_sdf: Path to 3D conformer ensemble reference SDF.

        Returns:
            Tuple of (prioritized leads DataFrame, leads TSV path, analysis directory).
        """
        stage2_dir: Path = self.BASE_OUTPUT_DIR / "stage2_result_analysis"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        print("\n[Stage 2] Comprehensive Result Analysis & Lead Profiling...")
        analyzer = ResultAnalyzer(output_dir=stage2_dir, ref_sdf_path=ref_sdf, seed_smiles=primary_molecule)

        # 1. Full 3D, 2D, and physicochemical evaluation
        df_analyzed = analyzer.analyze_candidates(candidate_df)

        # 2. Prioritize elite leads
        df_leads = analyzer.prioritize_leads(df_analyzed, top_k=MAX_SCAFFOLDS)

        # 3. Generate publication-grade visualizations
        analyzer.visualize_leads(df_leads, df_analyzed)

        # 4. Export structured statistical summary
        analyzer.export_summary(df_analyzed, df_leads)

        leads_tsv = stage2_dir / "top_prioritized_leads.tsv"

        print(f"[Stage 2] Prioritized {len(df_leads)} top elite lead molecules:")
        for idx, row in df_leads.iterrows():
            rocs_val = float(row.get("ROCS", 0.0))
            qed_val = float(row.get("QED", 0.0))
            sa_val = float(row.get("SA", row.get("SAscore", 0.0)))
            des_val = float(row.get("Desirability", 0.0))
            print(
                f"  • Lead #{idx+1}: {row['SMILES'][:45]}... | "
                f"ROCS: {rocs_val:.2f} | QED: {qed_val:.2f} | "
                f"SA: {sa_val:.2f} | Desirability: {des_val:.2f}"
            )
        print(f"[Stage 2] Summary analysis saved to: {stage2_dir / 'result_analysis_summary.json'}")
        print(f"[Stage 2] Top leads saved to: {leads_tsv}")
        return df_leads, leads_tsv, stage2_dir

    def _find_existing_candidates(self, default_molecule: str | None = None) -> pd.DataFrame:
        """Locate any existing generated candidate TSV file for benchmark_only mode.

        Args:
            default_molecule: Optional fallback molecule SMILES if no previous runs found.

        Returns:
            DataFrame containing candidate SMILES.
        """
        search_paths: list[Path] = [
            self.BASE_OUTPUT_DIR / "stage1_graph_transformer" / "generated_candidates.tsv",
            self.BASE_OUTPUT_DIR / "stage2_graph_transformer" / "stage2_leads.tsv",
            self.BASE_OUTPUT_DIR / "stage1_sequence_rnn" / "stage1_hits.tsv",
        ]
        # Also check global output directory for recent runs
        root_output = self.ROOT_PATH / "_david" / "output"
        if root_output.exists():
            for rdir in sorted(root_output.iterdir(), reverse=True):
                if rdir.is_dir() and rdir != self.BASE_OUTPUT_DIR:
                    search_paths.extend([
                        rdir / "stage1_graph_transformer" / "generated_candidates.tsv",
                        rdir / "stage2_graph_transformer" / "stage2_leads.tsv",
                        rdir / "stage1_sequence_rnn" / "stage1_hits.tsv",
                    ])
        for p in search_paths:
            if p.exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    if len(df) > 0 and ("SMILES" in df.columns or "smiles" in df.columns):
                        return df
                except Exception:
                    continue

        fallback_smi = default_molecule or DEFAULT_MOLECULE
        return pd.DataFrame({"SMILES": [fallback_smi]})

    # Backward-compatibility wrappers
    def stage1_sequence_rnn_hit_generation(self, primary_molecule: str, ref_sdf: Path) -> tuple[pd.DataFrame, Path]:
        """Backward-compatibility alias."""
        return self.stage1_graph_transformer_generation(primary_molecule, ref_sdf)

    def stage2_graph_transformer_optimization(
        self, primary_molecule: str, ref_sdf: Path, scaffolds_path: Path | str | None = None
    ) -> tuple[pd.DataFrame, Path]:
        """Backward-compatibility alias."""
        return self.stage1_graph_transformer_generation(primary_molecule, ref_sdf, seed_core_tsv=scaffolds_path)

    def stage3_post_generation_lead_selection(
        self, candidate_df: pd.DataFrame, primary_molecule: str, ref_sdf: Path | None = None
    ) -> tuple[pd.DataFrame, Path]:
        """Backward-compatibility alias."""
        df_leads, leads_tsv, _ = self.stage2_result_analysis(candidate_df, primary_molecule, ref_sdf or (self.BASE_OUTPUT_DIR / "references" / "query_molecule_3d.sdf"))
        return df_leads, leads_tsv

    def stage2_bemis_murcko_funnel(
        self, stage1_df: pd.DataFrame, primary_molecule: str, ref_sdf: Path | None = None
    ) -> tuple[pd.DataFrame, Path]:
        """Backward-compatibility alias."""
        return self.stage3_post_generation_lead_selection(stage1_df, primary_molecule, ref_sdf=ref_sdf)

    def stage3_graph_transformer_optimization(
        self, primary_molecule: str, ref_sdf: Path, scaffolds_path: Path | str | None = None
    ) -> tuple[pd.DataFrame, Path]:
        """Backward-compatibility alias."""
        return self.stage1_graph_transformer_generation(primary_molecule, ref_sdf, seed_core_tsv=scaffolds_path)

    def stage4_benchmark_and_visualizations(
        self,
        stage1_tsv: Path,
        stage2_tsv: Path | None = None,
        leads_tsv: Path | None = None,
        ref_sdf: Path | None = None,
        primary_molecule: str = DEFAULT_MOLECULE,
    ) -> tuple[Path, pd.DataFrame]:
        """Backward-compatibility alias."""
        benchmark_dir: Path = self.BASE_OUTPUT_DIR / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark = thesis_benchmark.ThesisBenchmark(output_dir=benchmark_dir, ref_sdf=ref_sdf, seed_smiles=primary_molecule)
        table_df = benchmark.run_benchmark(rnn_path=stage1_tsv, graph_path=stage2_tsv, funnel_path=leads_tsv)
        return benchmark_dir, table_df

    def run(self, molecules: str | Path | Sequence[str] = DEFAULT_MOLECULE) -> dict[str, Any]:
        """Execute the streamlined Graph-Transformer drug design pipeline.

        Args:
            molecules: Query seed molecule SMILES string or file path.

        Returns:
            Dictionary containing run summary and output file locations.
        """
        self._start_time = datetime.now()
        self.log("=" * 80)
        self.log(" DRUGEX LIGAND-BASED SIMILAR MOLECULE PIPELINE (GRAPH-TRANSFORMER)")
        self.log(f" Run ID:    {self.RUN_ID}")
        self.log(f" Mode:      {self.MODE.upper()}")
        self.log(f" Directory: {self.BASE_OUTPUT_DIR}")
        gpu_telemetry = get_gpu_telemetry()
        if gpu_telemetry["cuda_available"] and gpu_telemetry["devices"]:
            dev0 = gpu_telemetry["devices"][0]
            self.log(f" GPU 0:     {dev0['name']} (Alloc: {dev0['allocated_mb']}MB, Res: {dev0['reserved_mb']}MB, Total: {dev0['total_memory_mb']}MB)")
        self.log("=" * 80)

        try:
            # 1. Parse and validate query molecule
            valid_molecules = self.parse_molecules(molecules)
            primary_molecule = valid_molecules[0]
            self.log(f"Input molecules loaded: {len(valid_molecules)}")
            self.log(f"Primary query seed molecule: {primary_molecule}")

            # 2. Stage 0: 3D reference conformer and seed core scaffold generation
            self.update_status("Stage 0: 3D Reference Conformation", "RUNNING")
            ref_sdf, seed_core_tsv = self.stage0_prepare_reference_conformer(primary_molecule)
            self.update_status("Stage 0: 3D Reference Conformation", "SUCCESS")

            if self.MODE == "benchmark_only":
                self.log("[Pipeline] Running in benchmark_only mode. Looking for existing run files...")
                self.update_status("Stage 2: Result Analysis", "RUNNING")
                df_existing = self._find_existing_candidates()
                candidates_tsv = self.BASE_OUTPUT_DIR / "stage1_graph_transformer" / "generated_candidates.tsv"
                df_leads, leads_tsv, analysis_dir = self.stage2_result_analysis(
                    df_existing, primary_molecule, ref_sdf
                )
                self.update_status("Stage 2: Result Analysis", "SUCCESS", {"prioritized_leads_count": len(df_leads)})
            else:
                # 3. Stage 1: Graph-Transformer Molecular Generation & RL
                self.update_status("Stage 1: Graph-Transformer Generation", "RUNNING")
                df_stage1, candidates_tsv = self.stage1_graph_transformer_generation(
                    primary_molecule, ref_sdf, seed_core_tsv=seed_core_tsv
                )
                self.update_status("Stage 1: Graph-Transformer Generation", "SUCCESS", {"generated_count": len(df_stage1)})

                # 4. Stage 2: Comprehensive Result Analysis & Lead Profiling
                self.update_status("Stage 2: Result Analysis", "RUNNING")
                df_leads, leads_tsv, analysis_dir = self.stage2_result_analysis(
                    df_stage1, primary_molecule, ref_sdf
                )
                self.update_status("Stage 2: Result Analysis", "SUCCESS", {"prioritized_leads_count": len(df_leads)})

            duration = (datetime.now() - self._start_time).total_seconds()

            # Audit and save fallbacks
            audit_file = self.BASE_OUTPUT_DIR / "fallback_audit.json"
            FallbackMonitor.save_audit_json(audit_file)
            FallbackMonitor.print_summary()

            # Save complete run summary metadata
            summary_info: dict[str, Any] = {
                "run_id": self.RUN_ID,
                "mode": self.MODE,
                "primary_molecule": primary_molecule,
                "duration_seconds": round(duration, 2),
                "output_directory": str(self.BASE_OUTPUT_DIR),
                "reference_sdf": str(ref_sdf),
                "seed_core_scaffold_file": str(seed_core_tsv),
                "generated_candidates_file": str(candidates_tsv),
                "prioritized_leads_file": str(leads_tsv),
                "analysis_directory": str(analysis_dir),
                "fallback_audit_file": str(audit_file),
                "fallbacks_count": len(FallbackMonitor.get_events()),
                "gpu_telemetry": get_gpu_telemetry(),
                "stage_statuses": dict(self._stage_statuses),
            }
            with open(self.BASE_OUTPUT_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_info, f, indent=4)

            self.update_status("COMPLETED", "SUCCESS", {"duration_seconds": round(duration, 2)})

            self.log("\n" + "=" * 80)
            self.log(" PIPELINE EXECUTION FINISHED SUCCESSFULLY")
            self.log(f" Total Duration: {duration:.2f} seconds")
            self.log(f" All outputs organized in: {self.BASE_OUTPUT_DIR}")
            self.log("=" * 80)
            return summary_info

        except Exception as e:
            self.update_status("FAILED", "FAILED", {"error": str(e)})
            self.log(f"\n[FATAL ERROR] Pipeline failed with unhandled exception: {e}")
            raise


def main() -> None:
    """CLI and IDE entry point for running the MoleculePipeline."""
    parser = argparse.ArgumentParser(description="End-to-End Ligand-Based De Novo Drug Design Pipeline in DrugEx")
    parser.add_argument("--molecules", "-m", type=str, default=DEFAULT_MOLECULE, help="Input SMILES string or comma-separated SMILES")
    parser.add_argument("--input-file", "-f", type=Path, default=None, help="Path to input molecules file (.smi, .tsv, .csv, .sdf)")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Target output directory for pipeline run")
    parser.add_argument("--mode", choices=["full", "test", "benchmark_only"], default="full", help="Execution mode ('full' for training, 'test' for fast verification, 'benchmark_only' for existing data)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of RL epochs per stage")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of molecules sampled per epoch")

    args = parser.parse_args()

    input_data = args.input_file if args.input_file else args.molecules
    pipeline = MoleculePipeline(output_dir=args.out_dir, mode=args.mode, epochs=args.epochs, n_samples=args.n_samples)
    pipeline.run(input_data)


if __name__ == "__main__":
    main()
