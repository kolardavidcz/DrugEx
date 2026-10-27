"""Run or analyze the thesis conformer benchmark from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path

from _david.code.conformer.benchmark import BenchmarkConfig, ConformerBenchmark


class BenchmarkCommand:
    """Keep smoke checks, full measurements, and offline analysis explicit."""

    @staticmethod
    def run() -> None:
        """Parse a fresh output directory and optional panel sources."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--analyze-only", action="store_true")
        parser.add_argument("--confirm-production", action="store_true")
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Reuse completed work after validating config and input hashes",
        )
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--source", type=Path, action="append")
        args = parser.parse_args()
        if args.confirm_production:
            from _david.code.conformer.benchmark_confirmation import (
                ProductionConfirmation,
            )

            ProductionConfirmation(args.output).run()
            return
        if not args.analyze_only:
            config = BenchmarkConfig(workers=args.workers)
            if args.smoke:
                config = BenchmarkConfig(
                    reference_budgets=(8,),
                    reference_counts=(1, 3),
                    candidate_counts=(1, 2),
                    reference_seeds=(42,),
                    candidate_seeds=(0xC0FFEE,),
                    generated_per_source=2,
                    candidate_methods=("training_etkdg",),
                    workers=args.workers,
                )
            sources = args.source or [
                Path("_david/output/run17_baseline_50ep/raw_data/baseline_raw.csv.gz"),
                Path(
                    "_david/output/run18_epsilon_zero_50ep/raw_data/reinforced_raw.csv.gz"
                ),
            ]
            ConformerBenchmark(args.output, config, resume=args.resume).run(sources)
        from _david.code.conformer.benchmark_analysis import BenchmarkAnalysis

        BenchmarkAnalysis(args.output).run()


if __name__ == "__main__":
    BenchmarkCommand.run()
