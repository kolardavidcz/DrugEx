from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.pipeline import MoleculePipeline
from _david.code.schema import std


class CurriculumOrchestrator:
    """Manages multi-run adaptive reinforcement learning curriculum to optimize 3D leads toward target ROCS."""

    def __init__(
        self,
        base_dir: Path | str,
        molecule: str | Path = DEFAULT_MOLECULE,
        max_iterations: int = 4,
        target_rocs: float = 1.60,
        epochs_per_iter: int = 40,
        patience: int = 10,
        min_epochs: int = 25,
        n_samples: int = 100,
        val_samples: int = 50,
        n_eval_samples: int = 200,
        initial_epsilon: float = 0.20,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.molecule = molecule
        self.max_iterations = max_iterations
        self.target_rocs = target_rocs
        self.epochs_per_iter = epochs_per_iter
        self.patience = patience
        self.min_epochs = min_epochs
        self.n_samples = n_samples
        self.val_samples = val_samples
        self.n_eval_samples = n_eval_samples
        self.initial_epsilon = initial_epsilon

    def run(self) -> Path:
        """Execute adaptive curriculum iterations, updating hyperparameters and warm-starting agents."""
        history: list[dict[str, Any]] = []
        all_leads_frames: list[pd.DataFrame] = []

        current_thresholds = (0.60, 0.50, 0.35)
        current_epsilon = self.initial_epsilon
        current_agent_path: Path | None = None

        print("\n========================================================")
        print("Starting Multi-Run Curriculum Optimization")
        print(f"Target ROCS: >= {self.target_rocs:.2f}")
        print(f"Max Iterations: {self.max_iterations} | Epochs per run: {self.epochs_per_iter}")
        print(f"Output Root: {self.base_dir}")
        print("========================================================\n")

        for iter_idx in range(1, self.max_iterations + 1):
            iter_name = f"iteration_{iter_idx:02d}"
            iter_dir = self.base_dir / iter_name
            paths = RunPaths.resolve(base_dir=iter_dir)

            config = PipelineConfig(
                epochs=self.epochs_per_iter,
                patience=self.patience,
                min_epochs=self.min_epochs,
                n_samples=self.n_samples,
                val_samples=self.val_samples,
                n_eval_samples=self.n_eval_samples,
                criteria=std.AVG_AMEAN,
                epsilon=current_epsilon,
                beta=0.05,
                thresholds=current_thresholds,
                agent_path=current_agent_path,
            )

            print(f"\n>>> [Iteration {iter_idx}/{self.max_iterations}] Starting Run: {iter_name} <<<")
            print(f"    Thresholds (ROCS, SA, QED): {current_thresholds}")
            print(f"    Exploration Epsilon: {current_epsilon:.3f}")
            print(f"    Warm-Start Agent: {current_agent_path.name if current_agent_path else 'None (Prior)'}")

            pipeline = MoleculePipeline(config=config, paths=paths)
            pipeline.run(self.molecule)

            # Analyze results from this iteration
            leads_tsv = iter_dir / "top_prioritized_leads.tsv"
            if leads_tsv.is_file():
                df_leads = pd.read_csv(leads_tsv, sep="\t")
            else:
                df_leads = pd.DataFrame()

            max_rocs = float(df_leads[std.ROCS].max()) if not df_leads.empty and std.ROCS in df_leads.columns else 0.0
            top10_rocs = float(df_leads.head(10)[std.ROCS].mean()) if not df_leads.empty and std.ROCS in df_leads.columns else 0.0
            top10_sa = float(df_leads.head(10)[std.SA].mean()) if not df_leads.empty and std.SA in df_leads.columns else 0.0
            top10_qed = float(df_leads.head(10)[std.QED].mean()) if not df_leads.empty and std.QED in df_leads.columns else 0.0
            top_lead_smi = str(df_leads.iloc[0][std.SMILES]) if not df_leads.empty and std.SMILES in df_leads.columns else "N/A"

            if not df_leads.empty:
                df_tagged = df_leads.copy()
                df_tagged["Iteration"] = iter_idx
                all_leads_frames.append(df_tagged)

            iter_record = {
                "iteration": iter_idx,
                "name": iter_name,
                "epochs": self.epochs_per_iter,
                "epsilon": current_epsilon,
                "rocs_threshold": current_thresholds[0],
                "max_rocs": max_rocs,
                "mean_top10_rocs": top10_rocs,
                "mean_top10_sa": top10_sa,
                "mean_top10_qed": top10_qed,
                "top_lead_smi": top_lead_smi,
                "dir": str(iter_dir),
            }
            history.append(iter_record)

            print(f"\n[Iteration {iter_idx} Metrics]")
            print(f"  Max ROCS: {max_rocs:.3f} | Top-10 Mean ROCS: {top10_rocs:.3f}")
            print(f"  Top-10 Mean SA: {top10_sa:.3f} | Top-10 Mean QED: {top10_qed:.3f}")
            print(f"  Top Lead: {top_lead_smi[:45]}...")

            # Check target goal
            if max_rocs >= self.target_rocs:
                print(f"\n[GOAL ACHIEVED] Max ROCS {max_rocs:.3f} reached target {self.target_rocs:.2f}!")
                break

            # Stopping condition: Diminishing returns after at least 2 iterations
            if iter_idx >= 2:
                prev_max = history[iter_idx - 2]["max_rocs"]
                prev_mean = history[iter_idx - 2]["mean_top10_rocs"]
                if (max_rocs - prev_max < 0.01) and (top10_rocs - prev_mean < 0.01):
                    print("\n[STOPPING] Optimization reached performance plateau (delta < 0.01).")
                    break

            # Warm-start agent checkpoint selection
            peak_pkg = paths.models_dir / "david_agent_peak.pkg"
            final_pkg = paths.models_dir / "david_agent.pkg"
            current_agent_path = peak_pkg if peak_pkg.is_file() else (final_pkg if final_pkg.is_file() else None)

            # Hyperparameter progression: Epsilon decay
            current_epsilon = max(0.05, round(current_epsilon * 0.75, 3))

            # Adaptive threshold progression
            if top10_rocs >= 1.50 or max_rocs >= 1.55:
                current_thresholds = (0.75, 0.50, 0.35)
            elif top10_rocs >= 1.35 or max_rocs >= 1.45:
                current_thresholds = (0.70, 0.50, 0.35)
            elif top10_rocs >= 1.20 or max_rocs >= 1.35:
                current_thresholds = (0.65, 0.50, 0.35)

        # Consolidate deliverables
        self._generate_deliverables(history, all_leads_frames)
        return self.base_dir

    def _generate_deliverables(
        self,
        history: list[dict[str, Any]],
        all_leads_frames: list[pd.DataFrame],
    ) -> None:
        """Render multi-run progression chart, consolidated top leads, and executive markdown summary."""
        if not history:
            return

        # 1. Progression Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
        fig.suptitle("Multi-Run Curriculum Reinforcement Learning Progression", fontsize=14, fontweight="bold")

        iters = [h["iteration"] for h in history]
        max_rocs = [h["max_rocs"] for h in history]
        mean_rocs = [h["mean_top10_rocs"] for h in history]
        mean_sa = [h["mean_top10_sa"] for h in history]
        mean_qed = [h["mean_top10_qed"] for h in history]

        # Left: ROCS Progression
        axes[0].plot(iters, max_rocs, marker="o", color="#2563EB", linewidth=2.5, label="Max ROCS")
        axes[0].plot(iters, mean_rocs, marker="s", color="#3B82F6", linestyle="--", linewidth=1.8, label="Top-10 Mean ROCS")
        axes[0].axhline(self.target_rocs, color="#DC2626", linestyle=":", linewidth=2.0, label=f"Target ({self.target_rocs:.2f})")
        axes[0].set_title("3D ROCS Optimization Trajectory", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Curriculum Iteration")
        axes[0].set_ylabel("ROCS TanimotoCombo [0, 2]")
        axes[0].set_xticks(iters)
        axes[0].set_ylim(bottom=0.5, top=2.0)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="lower right")

        # Right: Secondary Objectives (SA & QED)
        axes[1].plot(iters, mean_sa, marker="^", color="#10B981", linewidth=2.0, label="Top-10 Mean SA [1, 10]")
        axes[1].plot(iters, mean_qed, marker="d", color="#8B5CF6", linewidth=2.0, label="Top-10 Mean QED [0, 1]")
        axes[1].set_title("Synthesizability & Drug-Likeness Stability", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Curriculum Iteration")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_xticks(iters)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="best")

        plt.tight_layout()
        prog_png = self.base_dir / "curriculum_progression.png"
        plt.savefig(prog_png, bbox_inches="tight")
        plt.close(fig)

        # 2. Consolidated Top Leads
        if all_leads_frames:
            combined_df = pd.concat(all_leads_frames, ignore_index=True)
            if std.CANONICAL_SMILES in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=[std.CANONICAL_SMILES])
            sort_cols = [c for c in [std.ROCS, std.SA, std.QED] if c in combined_df.columns]
            if sort_cols:
                combined_df = combined_df.sort_values(by=sort_cols, ascending=[False, True, False][:len(sort_cols)])
            overall_leads_tsv = self.base_dir / "overall_top_leads.tsv"
            combined_df.head(20).to_csv(overall_leads_tsv, sep="\t", index=False)

        # 3. Markdown Executive Summary
        md_lines = [
            "# DrugEx Multi-Run Curriculum Optimization Summary",
            "",
            f"- **Target Molecule**: `{str(self.molecule)[:40]}...`",
            f"- **Target ROCS**: `{self.target_rocs:.2f}`",
            f"- **Completed Iterations**: `{len(history)}/{self.max_iterations}`",
            f"- **Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "## Iteration Performance Matrix",
            "",
            "| Iteration | Epsilon | ROCS Thresh | Max ROCS | Top-10 ROCS | Top-10 SA | Top-10 QED | Deliverables |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
        ]
        for h in history:
            rel_link = f"[{h['name']}]({h['name']}/)"
            md_lines.append(
                f"| {h['iteration']} | {h['epsilon']:.3f} | {h['rocs_threshold']:.2f} | "
                f"**{h['max_rocs']:.3f}** | {h['mean_top10_rocs']:.3f} | {h['mean_top10_sa']:.2f} | "
                f"{h['mean_top10_qed']:.2f} | {rel_link} |"
            )

        md_lines.extend(
            [
                "",
                "## Deliverables",
                "- [Curriculum Progression Plot](curriculum_progression.png)",
                "- [Overall Top Leads (TSV)](overall_top_leads.tsv)",
                "",
            ]
        )

        with open(self.base_dir / "curriculum_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        print(f"\n[DONE] Curriculum summary and deliverables saved directly to: {self.base_dir}")


def main() -> None:
    """CLI runner entrypoint for multi-run curriculum optimization."""
    parser = argparse.ArgumentParser(description="DrugEx Multi-Run Curriculum Optimization")
    parser.add_argument("--molecule", type=str, default=DEFAULT_MOLECULE, help="Target molecule SMILES or file path")
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum number of adaptive iterations (2-5)")
    parser.add_argument("--epochs", type=int, default=40, help="RL epochs per iteration")
    parser.add_argument("--target-rocs", type=float, default=1.60, help="Target ROCS TanimotoCombo threshold")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for curriculum runs")

    args = parser.parse_args()
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).resolve().parent.parent / "output" / "overnight_runs" / ts

    orchestrator = CurriculumOrchestrator(
        base_dir=base_dir,
        molecule=args.molecule,
        max_iterations=args.max_iterations,
        target_rocs=args.target_rocs,
        epochs_per_iter=args.epochs,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
