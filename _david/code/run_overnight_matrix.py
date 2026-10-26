from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.pipeline import MoleculePipeline


def run_experiment(name: str, config: PipelineConfig, output_dir: Path) -> None:
    """Run a single end-to-end experiment and log execution timeline."""
    t0 = time.time()
    print(f"\n{'=' * 80}")
    print(f"STARTING EXPERIMENT: {name} at {datetime.now().isoformat()}")
    print(f"Output Directory: {output_dir}")
    print(
        f"Config: epochs={config.epochs}, samples={config.n_samples}, epsilon={config.epsilon}, "
        f"decay={config.use_epsilon_decay}, beta={config.beta}, scheme={config.reward_scheme}"
    )
    print(f"{'=' * 80}\n")

    paths = RunPaths(
        root=output_dir,
        models_dir=output_dir / "models",
        raw_data_dir=output_dir / "raw_data",
        logs_dir=output_dir / "logs",
    )
    pipeline = MoleculePipeline(config=config, paths=paths)
    pipeline.run(molecule=DEFAULT_MOLECULE)

    elapsed = (time.time() - t0) / 3600.0
    print(f"\n{'=' * 80}")
    print(f"COMPLETED EXPERIMENT: {name} in {elapsed:.2f} hours")
    print(f"{'=' * 80}\n")


def main() -> None:
    """Orchestrate sequential overnight execution of Runs 1 through 4."""
    base_output = PROJECT_ROOT / "_david" / "output"
    base_output.mkdir(parents=True, exist_ok=True)

    experiments: list[tuple[str, PipelineConfig, Path]] = [
        (
            "Run 1: Supervisor Baseline (epsilon=0.20, beta=0.0)",
            PipelineConfig(
                epochs=50,
                n_samples=500,
                n_eval_samples=1000,
                epsilon=0.20,
                beta=0.0,
                reward_scheme="weighted_sum",
                score_variant="tanimoto_combo",
                cpu_ratio=0.90,
            ),
            base_output / "run17_baseline_50ep",
        ),
        (
            "Run 2: Super-Supervisor Pure REINFORCE (epsilon=0.00)",
            PipelineConfig(
                epochs=50,
                n_samples=500,
                n_eval_samples=1000,
                epsilon=0.00,
                beta=0.0,
                reward_scheme="weighted_sum",
                score_variant="tanimoto_combo",
                cpu_ratio=0.90,
            ),
            base_output / "run18_epsilon_zero_50ep",
        ),
        (
            "Run 3: Cosine Decaying Exploration (epsilon=0.20 -> 0.05)",
            PipelineConfig(
                epochs=50,
                n_samples=500,
                n_eval_samples=1000,
                epsilon=0.20,
                epsilon_end=0.05,
                use_epsilon_decay=True,
                decay_type="cosine",
                decay_epochs=50,
                beta=0.0,
                reward_scheme="weighted_sum",
                score_variant="tanimoto_combo",
                cpu_ratio=0.90,
            ),
            base_output / "run19_cosine_decay_50ep",
        ),
        (
            "Run 4: Policy Penalization Baseline (beta=0.05, epsilon=0.20)",
            PipelineConfig(
                epochs=50,
                n_samples=500,
                n_eval_samples=1000,
                epsilon=0.20,
                beta=0.05,
                reward_scheme="weighted_sum",
                score_variant="tanimoto_combo",
                cpu_ratio=0.90,
            ),
            base_output / "run20_beta_penalty_50ep",
        ),
    ]

    print(f"Overnight Batch Execution: {len(experiments)} experiments scheduled.")
    start_all = time.time()

    for _idx, (name, cfg, out_dir) in enumerate(experiments, start=1):
        if (out_dir / "summary.md").is_file():
            print(f"Skipping {name} (already completed at {out_dir}).")
            continue
        try:
            run_experiment(name, cfg, out_dir)
        except Exception as e:
            print(f"ERROR in {name}: {e}", file=sys.stderr)

    total_hours = (time.time() - start_all) / 3600.0
    print(f"\nALL OVERNIGHT EXPERIMENTS COMPLETED in {total_hours:.2f} hours total.")


if __name__ == "__main__":
    main()
