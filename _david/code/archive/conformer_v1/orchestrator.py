"""Unified Conformer Study Orchestrator.

Chains reference generation, benchmark execution, and statistical analysis
via a unified StudySettings configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from _david.code.conformer.settings import StudySettings
from _david.code.utils import HardwareManager


class ConformerStudyOrchestrator:
    """Orchestrator that executes modular conformer pipeline stages."""

    def __init__(self, settings: StudySettings | None = None) -> None:
        self.settings = settings or StudySettings()
        HardwareManager.init_runtime_environment(ratio=self.settings.cpu_ratio)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_reference_banks(self) -> Path:
        """Stage 1: Generate matched multi-method reference banks."""
        from _david.code.conformer.reference.reference_bank import prepare_banks

        banks_dir = self.settings.output_dir / "reference_banks"
        banks_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Orchestrator Stage 1] Preparing reference banks in {banks_dir}...")
        prepare_banks(self.settings.molecule, str(banks_dir))
        return banks_dir

    def run_benchmarks(self, banks_dir: Path | None = None) -> Path:
        """Stage 2: Run multi-arm batch scoring benchmark grid."""
        from _david.code.conformer.benchmarks.batch_probe import run_batch_probe

        bench_dir = self.settings.output_dir / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Orchestrator Stage 2] Executing conformer benchmarks in {bench_dir}...")
        df, _ = run_batch_probe(
            batch_sizes=[32, 64],
            worker_counts=[1, HardwareManager.get_safe_parallel_jobs(ratio=self.settings.cpu_ratio)],
            minimize_candidates=False,
            repeats=1,
        )
        df.to_csv(bench_dir / "batch_probe_results.csv", index=False)
        return bench_dir

    def generate_analysis_figures(self) -> Path:
        """Stage 3: Render publication-ready analytical figures and Pareto frontiers."""
        from _david.code.conformer.analysis.visualization import render_conformer_study

        fig_dir = self.settings.output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Orchestrator Stage 3] Generating analysis figures in {fig_dir}...")
        bench_csv = self.settings.output_dir / "benchmarks" / "batch_probe_results.csv"
        if bench_csv.is_file():
            df = pd.read_csv(bench_csv)
            try:
                render_conformer_study(df, fig_dir / "01_conformer_study_summary.png")
            except Exception as e:
                print(f"[Orchestrator Stage 3] Figure rendering note: {e}")
        return fig_dir

    def run(self) -> Path:
        """Execute all configured stages in end-to-end pipeline."""
        banks_dir = None
        if self.settings.run_reference_banks:
            banks_dir = self.prepare_reference_banks()
        if self.settings.run_benchmarks:
            self.run_benchmarks(banks_dir=banks_dir)
        if self.settings.generate_figures:
            self.generate_analysis_figures()
        print(f"[Orchestrator Complete] Deliverables generated in {self.settings.output_dir}")
        return self.settings.output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conformer Study Unified Orchestrator")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_david/output/conformer_study"),
        help="Target output directory",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["reference", "benchmarks", "analysis", "all"],
        default=["all"],
        help="Stages to run (default: all)",
    )
    parser.add_argument(
        "--k-ref",
        type=int,
        default=4,
        help="Reference ensemble cap K (default: 4)",
    )
    parser.add_argument(
        "--c-cand",
        type=int,
        default=20,
        help="Candidate conformer attempts C (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stages = set(args.stages)
    run_all = "all" in stages
    settings = StudySettings(
        output_dir=args.output_dir,
        reference_cap=args.k_ref,
        candidate_attempts=args.c_cand,
        run_reference_banks=run_all or "reference" in stages,
        run_benchmarks=run_all or "benchmarks" in stages,
        generate_figures=run_all or "analysis" in stages,
    )
    orchestrator = ConformerStudyOrchestrator(settings)
    orchestrator.run()


if __name__ == "__main__":
    main()
