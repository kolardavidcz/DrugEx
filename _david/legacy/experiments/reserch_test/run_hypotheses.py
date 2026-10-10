"""
====================================================================================================
BACHELOR THESIS RESEARCH STUDY: LIGAND-BASED DE NOVO MOLECULAR DESIGN IN DRUGEX
COMPARATIVE BENCHMARK OF 3 FUNDAMENTAL PHARMACOPHORIC HYPOTHESES (EGCG / DNMT1 MODEL)
====================================================================================================

AUTHOR: David Kolář (Academic Thesis Pair-Programming with Antigravity AI)
TARGET HARDWARE: dichtator (dual RTX PRO 4000 Blackwell GPUs, 125 GB RAM, 48 vCPU cores)
OFFLINE LOG-OFF EXECUTION: Fully detached background daemon surviving terminal close & VPN disconnect.

----------------------------------------------------------------------------------------------------
THE THREE CORE HYPOTHESES EVALUATED IN THIS STUDY
----------------------------------------------------------------------------------------------------

HYPOTHESIS 1: Single-Reference Template Rigidity (Baseline Overfitting)
  • Statement:
      Conditioning reinforcement learning (RL) on a single 3D conformer ensemble of a single ligand
      (Epigallocatechin gallate, EGCG, CHEMBL297453) creates an overly rigid optimization landscape.
      The Gaussian volume and Mills-Dean color overlap criteria strictly reward structural mimics
      that match EGCG's exact spatial conformation, leading to early policy collapse into local minima
      and severely restricting chemical novelty beyond the flavan-3-ol polyphenol topology.
  • Implementation:
      - Reference: _david/references/query_egcg_3d.sdf (Single EGCG MMFF94 conformer ensemble)
      - Objective: RDKitROCSScorer(oe_mills_dean, TanimotoCombo) against EGCG alone
      - Generator: FragGraphExplorer conditioned on EGCG Murcko scaffold seed

HYPOTHESIS 2: Multi-Reference Active Ensemble Flexibility (Chemotypic Ensemble)
  • Statement:
      Evaluating candidate overlays against an ensemble of 5 structurally diverse, high-affinity
      human DNMT1 actives via native DrugEx maximum reduction (max_i ROCS(candidate, active_i))
      broadens the policy exploration basin. Because candidates can satisfy any of several
      distinct bioactive binding modes across diverse chemotypes, policy collapse is mitigated,
      and early exploration yields improved overall reward trajectories; however, the model may
      still favor the easiest-to-match dominant whole-molecule shape in the reference pool.
  • Implementation:
      - Reference: _david/references/dnmt1_actives_multi_3d.sdf (5 diverse Butina cluster champions)
      - Objective: Native DrugEx multi-reference RDKitROCSScorer taking max score per candidate
      - Generator: FragGraphExplorer conditioned on EGCG Murcko scaffold seed

HYPOTHESIS 3: True Consensus Pharmacophore Scaffold Hopping (Backbone-Decoupled Spatial Model)
  • Statement:
      Evaluating candidates against an agglomerative consensus pharmacophore pseudo-molecule
      (derived from spatial clustering of interaction centroids across all diverse DNMT1 actives at
      2.0 Å tolerance and >= 50% occurrence threshold, inspired by fulopjoz/pharmacophore-toolkit)
      completely decouples pharmacophore feature satisfaction from any specific chemical backbone.
      By rewarding essential binding interaction points (hydrogen bond donors, acceptors, aromatics)
      without whole-molecule shape rigidity, this strategy enables genuine scaffold hopping,
      yielding significantly higher Bemis-Murcko framework diversity, superior drug-likeness (QED),
      and robust multi-objective desirability.
  • Implementation:
      - Reference: _david/references/dnmt1_consensus_pharmacophore.sdf (23 consensus centroid atoms)
      - Objective: RDKitROCSScorer against the consensus pharmacophore pseudo-molecule
      - Generator: FragGraphExplorer conditioned on EGCG Murcko scaffold seed

----------------------------------------------------------------------------------------------------
MATHEMATICAL REWARD FORMULATION & SCORER CONFIGURATION
----------------------------------------------------------------------------------------------------
All 3 hypotheses share an identical, academically rigorous multi-objective reinforcement learning
environment to guarantee strict scientific comparability:

1. Primary 3D ROCS Alignment Score (Normalized to [0.0, 1.0]):
     S_3D(x) = ROCS(x, Ref) / 2.0   (TanimotoCombo on [0, 2] scale mapped to [0, 1])

2. Secondary Physicochemical Envelope Scorers (with SmoothHump Gaussian-plateau modifiers):
     - Heavy Atom Count (HAC): Relative to query [0.70 * X, 1.30 * X]
     - Lipophilic Efficiency (CLogP): Additive delta [X - 1.20, X + 1.20], sigma=0.80
     - Polar Surface Area (TPSA): Relative to query [0.70 * X, 1.30 * X]
     - Hydrogen Bond Donors (HBD): Additive delta [X - 2, X + 2], sigma=1.0
     - Hydrogen Bond Acceptors (HBA): Additive delta [X - 2, X + 2], sigma=1.0

3. Feasibility Floor:
     - Synthetic Accessibility (SAscore): Feasibility floor (1.0 for SA <= 5.5, decaying to 0.0 at SA >= 6.0)

4. Anti-Collapse Soft-Gated Reward Formulation:
     R(x) = w_3D * S_3D(x) + w_props * G(S_3D(x)) * sum_i(w_i * S_prop,i(x)) + S_SA_floor(x)
     where:
       G(S_3D) = min(1.0, max(0.0, S_3D / 0.60))
       w_3D = 0.70,  w_props = 0.30

----------------------------------------------------------------------------------------------------
LOG-OFF / BACKGROUND DAEMON ARCHITECTURE
----------------------------------------------------------------------------------------------------
When launched with `--detach` (or invoked as default for offline execution):
  - Spawns a fully detached background process (setsid / double-fork) independent of the terminal.
  - Automatically survives Cisco VPN disconnects and client logoffs.
  - Redirects stdout and stderr to `_david/output/experiments_<RUN_ID>/experiments.log`.
  - Maintains `_david/output/experiments_<RUN_ID>/status.json` with live per-epoch progress.
  - Automatically probes GPU 0 and GPU 1 on `dichtator` and dynamically allocates the GPU with
    maximum free VRAM via `get_safe_gpus("auto")`.
  - Runs all 3 experiments sequentially (or any individual experiment selected via `--experiment`).
  - Caps multiprocessing to safe parallel core limits (DRUGEX_SCORE_NJOBS <= 24).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent if (SCRIPT_DIR.parent.parent.parent / "drugex").exists() else SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from _david.code import GraphTransformer
from _david.code.pipeline import MoleculePipeline, get_gpu_telemetry

EGCG_SMILES: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
DEFAULT_EPOCHS: int = 30
DEFAULT_N_SAMPLES: int = 200

EXPERIMENT_SPECS: dict[int, dict[str, str]] = {
    1: {
        "name": "Single_Reference_EGCG_Baseline",
        "title": "Experiment 1: Single-Reference EGCG 3D ROCS (Rigid Template Baseline)",
        "ref_sdf": "_david/references/query_egcg_3d.sdf",
        "seed_scaffold": "_david/references/seed_core_scaffold.tsv",
        "description": "Evaluates RL convergence against a single 3D conformer ensemble of EGCG.",
    },
    2: {
        "name": "Multi_Reference_DNMT1_Ensemble",
        "title": "Experiment 2: Multi-Reference DNMT1 Ensemble (Max Reduction over 5 Actives)",
        "ref_sdf": "_david/references/dnmt1_actives_multi_3d.sdf",
        "seed_scaffold": "_david/references/seed_core_scaffold.tsv",
        "description": "Evaluates RL convergence against 5 diverse DNMT1 actives via max reduction.",
    },
    3: {
        "name": "Consensus_Pharmacophore_Model",
        "title": "Experiment 3: Deterministic Consensus Pharmacophore Model (Scaffold Hopping)",
        "ref_sdf": "_david/references/dnmt1_consensus_pharmacophore.sdf",
        "seed_scaffold": "_david/references/seed_core_scaffold.tsv",
        "description": "Evaluates RL convergence against 23 agglomerative consensus centroids.",
    },
}


class ExperimentRunner:
    """Orchestrates comparative hypothesis testing across Experiments 1, 2, and 3."""

    def __init__(
        self,
        experiment_ids: Sequence[int] | None = None,
        epochs: int = DEFAULT_EPOCHS,
        n_samples: int = DEFAULT_N_SAMPLES,
        gpu: str | int | None = "auto",
        output_dir: Path | str | None = None,
        include_hb: bool = False,
        drop_off: bool = True,
        patience: int = 15,
        rocs_threshold: float = 0.30,
        prop_threshold: float = 0.30,
        sa_threshold: float = 0.30,
        criteria: str = "desired_ratio",
    ) -> None:
        """Initialize ExperimentRunner with target hypotheses and execution hyperparameters.

        Args:
            experiment_ids: List of experiment IDs to execute (1, 2, 3) or None for all.
            epochs: RL training epochs per experiment (default: 30).
            n_samples: Molecules sampled per epoch (default: 200).
            gpu: Target GPU index (0, 1), 'auto', or None.
            output_dir: Custom root output directory for experiment outputs.
            include_hb: Whether to include H-donor/acceptor envelope optimization (default: False).
            drop_off: Whether to enable early stopping drop-off monitoring (default: True).
            patience: Number of consecutive degradation epochs before early stopping triggers (default: 15).
            rocs_threshold: Desirability passing threshold for 3D ROCS alignment (default: 0.30).
            prop_threshold: Desirability passing threshold for secondary dynamic envelopes (default: 0.30).
            sa_threshold: Desirability passing threshold for single SA synthesizability floor (default: 0.30).
            criteria: Evaluation metric driving PeakRecoveryMonitor checkpoint tracking (default: 'desired_ratio').
        """
        self.repo_root: Path = REPO_ROOT
        self.experiment_ids: list[int] = list(experiment_ids) if experiment_ids else [1]
        self.epochs: int = epochs
        self.n_samples: int = n_samples
        self.gpu_arg: str | int | None = gpu
        self.include_hb: bool = include_hb
        self.drop_off: bool = drop_off
        self.patience: int = patience
        self.rocs_threshold: float = float(rocs_threshold)
        self.prop_threshold: float = float(prop_threshold)
        self.sa_threshold: float = float(sa_threshold)
        self.criteria: str = criteria

        from _david.code.GraphTransformer import get_safe_gpus
        self.safe_gpus: tuple[int, ...] | tuple[()] = get_safe_gpus(preferred_gpu=gpu)
        self.selected_gpu_idx: int = self.safe_gpus[0] if self.safe_gpus else 0

        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir: Path = self.repo_root / "_david" / "output" / f"experiments_{self.run_id}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status_file: Path = self.output_dir / "status.json"
        self.log_file: Path = self.output_dir / "experiments.log"
        self.pid_file: Path = self.output_dir / "process.pid"
        self._start_time: datetime = datetime.now()

    def log(self, message: str) -> None:
        """Log timestamped message to stdout and experiments.log."""
        print(message)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")

    def update_status(
        self,
        overall_status: str,
        current_experiment: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Write structured real-time status update to status.json."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        status_data: dict[str, Any] = {
            "run_id": self.run_id,
            "overall_status": overall_status,
            "current_experiment": current_experiment,
            "experiment_ids": self.experiment_ids,
            "epochs": self.epochs,
            "n_samples": self.n_samples,
            "selected_gpu": self.selected_gpu_idx,
            "elapsed_seconds": round(elapsed, 2),
            "pid": os.getpid(),
            "updated_at": datetime.now().isoformat(),
            "gpu_telemetry": get_gpu_telemetry(),
            "details": details or {},
        }
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=4)

    def run_all(self) -> dict[str, Any]:
        """Execute all configured experiments sequentially with continuous tracking."""
        with open(self.pid_file, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

        self.log("=" * 85)
        self.log(" STARTING LBDD COMPARATIVE EXPERIMENT STUDY (3 HYPOTHESES BENCHMARK)")
        self.log(f" Run ID:             {self.run_id}")
        self.log(f" Target Experiments: {self.experiment_ids}")
        self.log(f" Epochs Per Run:     {self.epochs}")
        self.log(f" Samples Per Epoch:  {self.n_samples}")
        self.log(f" Target GPU:         GPU {self.selected_gpu_idx}")
        self.log(f" Output Root:        {self.output_dir}")
        self.log(f" Process PID:        {os.getpid()}")
        self.log("=" * 85)

        self.update_status("RUNNING", details={"stage": "Initialization"})

        results: dict[int, dict[str, Any]] = {}
        for exp_id in self.experiment_ids:
            spec = EXPERIMENT_SPECS[exp_id]
            self.log("\n" + "#" * 85)
            self.log(f" >>> STARTING {spec['title']}")
            self.log(f" Description: {spec['description']}")
            self.log("#" * 85)

            exp_dir = self.output_dir / f"exp{exp_id}_{spec['name']}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            ref_sdf = self.repo_root / spec["ref_sdf"]
            seed_scaffold = self.repo_root / spec["seed_scaffold"]

            if not ref_sdf.exists():
                self.log(f"[ERROR] Reference SDF missing: {ref_sdf}. Aborting experiment {exp_id}.")
                results[exp_id] = {"status": "FAILED", "error": f"Reference SDF missing: {ref_sdf}"}
                continue

            self.update_status(
                "RUNNING",
                current_experiment=exp_id,
                details={"experiment_title": spec["title"], "stage": f"Running Exp {exp_id}"},
            )

            exp_start = datetime.now()
            try:
                pipeline = MoleculePipeline(
                    output_dir=exp_dir,
                    mode="full",
                    epochs=self.epochs,
                    n_samples=self.n_samples,
                    gpu=self.selected_gpu_idx,
                    include_hb=self.include_hb,
                    drop_off=self.drop_off,
                    patience=self.patience,
                    rocs_threshold=self.rocs_threshold,
                    prop_threshold=self.prop_threshold,
                    sa_threshold=self.sa_threshold,
                    criteria=self.criteria,
                )
                run_summary = pipeline.run(
                    molecules=EGCG_SMILES,
                    custom_ref_sdf=ref_sdf,
                    custom_seed_core_tsv=seed_scaffold,
                )
                exp_duration = (datetime.now() - exp_start).total_seconds()
                results[exp_id] = {
                    "status": "SUCCESS",
                    "duration_seconds": round(exp_duration, 2),
                    "output_dir": str(exp_dir),
                    "summary": run_summary,
                }
                self.log(f"[SUCCESS] {spec['title']} completed in {exp_duration:.2f} seconds.")

            except Exception as e:
                exp_duration = (datetime.now() - exp_start).total_seconds()
                self.log(f"[FAILED] {spec['title']} encountered error after {exp_duration:.2f}s: {e}")
                results[exp_id] = {
                    "status": "FAILED",
                    "duration_seconds": round(exp_duration, 2),
                    "error": str(e),
                }

        # Cross-experiment comparative analysis
        self.log("\n" + "=" * 85)
        self.log(" GENERATING CROSS-HYPOTHESIS COMPARATIVE ANALYSIS MATRIX")
        self.log("=" * 85)
        comparative_df = self.generate_comparative_matrix(results)

        total_duration = (datetime.now() - self._start_time).total_seconds()
        self.update_status(
            "COMPLETED",
            details={
                "total_duration_seconds": round(total_duration, 2),
                "experiments_completed": len([r for r in results.values() if r.get("status") == "SUCCESS"]),
            },
        )

        self.log("\n" + "=" * 85)
        self.log(f" ALL EXPERIMENTS FINISHED (Total Time: {total_duration / 60.0:.2f} minutes)")
        self.log(f" Results & summaries stored in: {self.output_dir}")
        self.log("=" * 85)

        return {
            "run_id": self.run_id,
            "duration_seconds": round(total_duration, 2),
            "results": results,
            "comparative_table": comparative_df.to_dict(orient="records") if comparative_df is not None else [],
        }

    def generate_comparative_matrix(self, results: dict[int, dict[str, Any]]) -> pd.DataFrame | None:
        """Compile comparative metrics across all completed experiments into tabular summary."""
        records: list[dict[str, Any]] = []

        for exp_id, res in results.items():
            spec = EXPERIMENT_SPECS[exp_id]
            if res.get("status") != "SUCCESS":
                records.append({
                    "Experiment_ID": exp_id,
                    "Hypothesis": spec["name"],
                    "Status": "FAILED",
                    "Duration_s": res.get("duration_seconds", 0.0),
                })
                continue

            exp_dir = Path(res["output_dir"])
            summary_json_path = exp_dir / "stage2_result_analysis" / "result_analysis_summary.json"
            if not summary_json_path.exists():
                records.append({
                    "Experiment_ID": exp_id,
                    "Hypothesis": spec["name"],
                    "Status": "SUCCESS_NO_SUMMARY",
                    "Duration_s": res.get("duration_seconds", 0.0),
                })
                continue

            with open(summary_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            vun = data.get("vun_metrics", {})
            scaff = data.get("bemis_murcko_scaffold_diversity", data.get("scaffold_diversity", {}))
            elite = data.get("top_leads", data.get("prioritized_leads", data.get("elite_leads", [])))
            means = data.get("physicochemical_distributions", {})

            scaff_count = scaff.get("unique_scaffold_count", scaff.get("distinct_scaffolds", 0))
            scaff_ratio = scaff.get("diversity_ratio", scaff.get("scaffold_diversity_ratio", 0.0))
            leads_count = data.get("top_leads_count", len(elite) if isinstance(elite, list) else 0)
            des_mean = means.get("Desirability", {}).get("mean", 0.0)
            if des_mean == 0.0 and isinstance(elite, list) and len(elite) > 0:
                des_vals = [float(row.get("Desirability", 0.0)) for row in elite if isinstance(row, dict)]
                if des_vals:
                    des_mean = float(np.mean(des_vals))

            records.append({
                "Experiment_ID": exp_id,
                "Hypothesis": spec["name"],
                "Status": "SUCCESS",
                "Duration_s": res.get("duration_seconds", 0.0),
                "Total_Generated": vun.get("total_generated", 0),
                "Valid_Count": vun.get("valid_count", 0),
                "Validity_%": vun.get("validity_percentage", 0.0),
                "Uniqueness_%": vun.get("uniqueness_percentage", 0.0),
                "Novelty_%": vun.get("novelty_percentage", 0.0),
                "Distinct_Scaffolds": scaff_count,
                "Scaffold_Diversity_Ratio": round(scaff_ratio, 4),
                "Prioritized_Leads_Count": leads_count,
                "ROCS_Mean": means.get("ROCS", {}).get("mean", 0.0),
                "QED_Mean": means.get("QED", {}).get("mean", 0.0),
                "SA_Mean": means.get("SA", {}).get("mean", 0.0),
                "Desirability_Mean": round(des_mean, 4),
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        tsv_path = self.output_dir / "comparative_analysis.tsv"
        json_path = self.output_dir / "comparative_analysis.json"
        df.to_csv(tsv_path, sep="\t", index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)

        self.log(f"Comparative summary table saved to: {tsv_path}")
        self.log(f"Comparative JSON summary saved to: {json_path}")
        print("\n" + df.to_string(index=False) + "\n")
        return df


def spawn_detached_runner(args_list: list[str], output_dir: Path) -> int:
    """Spawn fully detached background process that survives SSH/VPN disconnects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiments.log"
    pid_file = output_dir / "process.pid"

    # [PACKAGE: subprocess] WHY: Detach process with start_new_session=True to survive terminal closure. WHAT: Subprocess management. HOW: subprocess.Popen.
    cmd = [sys.executable, str(Path(__file__).resolve())] + args_list + ["--no-detach"]

    with open(log_file, "a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(REPO_ROOT),
        )

    with open(pid_file, "w", encoding="utf-8") as pf:
        pf.write(str(proc.pid))

    return proc.pid


def print_status(target_dir: Path | None = None) -> None:
    """Print human-readable real-time status from latest or specified run directory."""
    if target_dir is None:
        out_root = REPO_ROOT / "_david" / "output"
        if not out_root.exists():
            print("No output directories found.")
            return
        candidates = sorted(
            [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("experiments_")],
            reverse=True,
        )
        if not candidates:
            print("No experiment directories found in _david/output/.")
            return
        target_dir = candidates[0]

    status_file = target_dir / "status.json"
    pid_file = target_dir / "process.pid"
    log_file = target_dir / "experiments.log"

    print("=" * 80)
    print(f" EXPERIMENT RUN STATUS: {target_dir.name}")
    print("=" * 80)

    pid_alive = False
    if pid_file.exists():
        pid_str = pid_file.read_text(encoding="utf-8").strip()
        try:
            pid = int(pid_str)
            os.kill(pid, 0)
            pid_alive = True
            print(f" • PID:            {pid} (ACTIVE / RUNNING)")
        except (ValueError, OSError):
            print(f" • PID:            {pid_str} (INACTIVE / TERMINATED)")

    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f" • Overall Status: {data.get('overall_status')}")
        print(f" • Current Exp:    {data.get('current_experiment', 'N/A')}")
        print(f" • Elapsed:        {data.get('elapsed_seconds', 0.0):.1f} seconds")
        print(f" • Selected GPU:   GPU {data.get('selected_gpu')}")
        print(f" • Updated At:     {data.get('updated_at')}")
    else:
        print(" • status.json not yet created.")

    if log_file.exists():
        print("-" * 80)
        print(" RECENT LOG OUTPUT (Last 15 lines):")
        print("-" * 80)
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[-15:]:
                print(line.rstrip())
    print("=" * 80)


def stop_runner(target_dir: Path | None = None) -> None:
    """Gracefully terminate a running background experiment process."""
    if target_dir is None:
        out_root = REPO_ROOT / "_david" / "output"
        candidates = sorted(
            [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("experiments_")],
            reverse=True,
        )
        if not candidates:
            print("No experiment directories found.")
            return
        target_dir = candidates[0]

    pid_file = target_dir / "process.pid"
    if not pid_file.exists():
        print(f"No process.pid found in {target_dir}.")
        return

    pid_str = pid_file.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to process PID {pid}. Background job terminated.")
    except Exception as e:
        print(f"Failed to terminate PID {pid_str}: {e}")


def main() -> None:
    """CLI Entry Point for Comparative Hypothesis Runner."""
    parser = argparse.ArgumentParser(
        description="Comparative Benchmark of 3 LBDD Pharmacophore Hypotheses in DrugEx (Log-Off Daemon)"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="1",
        help="Experiment to run: '1', '2', '3', or 'all' (default: '1')",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="RL epochs per experiment (default: 30)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per epoch (default: 200)")
    parser.add_argument("--gpu", default="auto", help="GPU index (0, 1) or 'auto' (default: auto)")
    parser.add_argument("--include-hb", action="store_true", help="Include H-donor and acceptor envelope optimization (default: False)")
    parser.add_argument("--no-drop-off", action="store_true", help="Disable drop off (early stopping), forcing full epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs (default: 15)")
    parser.add_argument("--rocs-threshold", type=float, default=0.30, help="ROCS desirability passing threshold (default: 0.30)")
    parser.add_argument("--prop-threshold", type=float, default=0.30, help="Dynamic property envelope passing threshold (default: 0.30)")
    parser.add_argument("--sa-threshold", type=float, default=0.30, help="SA synthesizability floor passing threshold (default: 0.30)")
    parser.add_argument("--criteria", type=str, default="desired_ratio", choices=["desired_ratio", "avg_amean", "avg_gmean"], help="Peak recovery selection criteria")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Root output directory")
    parser.add_argument("--detach", action="store_true", help="Launch detached background daemon surviving SSH logout (default behavior)")
    parser.add_argument("--no-detach", action="store_true", help="Run synchronously in current shell session")
    parser.add_argument("--status", action="store_true", help="Inspect real-time status of latest background run")
    parser.add_argument("--stop", action="store_true", help="Terminate active background run")

    args = parser.parse_args()

    if args.status:
        print_status(args.out_dir)
        return

    if args.stop:
        stop_runner(args.out_dir)
        return

    # Parse experiment list
    if args.experiment.lower() == "all":
        exp_ids = [1, 2, 3]
    else:
        exp_ids = [int(x.strip()) for x in args.experiment.split(",") if x.strip()]

    # If detached mode requested or default (when not --no-detach):
    if args.detach or not args.no_detach:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_out = args.out_dir or (REPO_ROOT / "_david" / "output" / f"experiments_{run_id}")
        fwd_args = [
            f"--experiment={args.experiment}",
            f"--epochs={args.epochs}",
            f"--n-samples={args.n_samples}",
            f"--gpu={args.gpu}",
            f"--out-dir={target_out}",
            f"--patience={args.patience}",
            f"--rocs-threshold={args.rocs_threshold}",
            f"--prop-threshold={args.prop_threshold}",
            f"--sa-threshold={args.sa_threshold}",
            f"--criteria={args.criteria}",
        ]
        if args.include_hb:
            fwd_args.append("--include-hb")
        if args.no_drop_off:
            fwd_args.append("--no-drop-off")

        pid = spawn_detached_runner(fwd_args, target_out)

        print("\n" + "=" * 85)
        print(" EXPERIMENT STUDY LAUNCHED DETACHED IN BACKGROUND")
        print("=" * 85)
        print(f" • Process PID:  {pid}")
        print(f" • Output Dir:   {target_out}")
        print(f" • Log File:     {target_out / 'experiments.log'}")
        print(f" • Status File:  {target_out / 'status.json'}")
        print(f" • Hypotheses:   Experiments {exp_ids}")
        print(f" • Epochs:       {args.epochs} per experiment")
        print(f" • Drop-Off:     {'Enabled (Early Stopping on degradation)' if not args.no_drop_off else 'Disabled (Forced full epochs)'}")
        print(f" • Patience:     {args.patience} epochs")
        print(f" • Thresholds:   ROCS >= {args.rocs_threshold}, Props >= {args.prop_threshold}, SA >= {args.sa_threshold}")
        print(f" • HBD/HBA:      {'Enabled' if args.include_hb else 'Disabled (Pure ROCS + Physical Envelope)'}")
        print(f" • Samples:      {args.n_samples} per epoch")
        print("=" * 85)
        print(" >>> YOU CAN NOW SAFELY CLOSE SSH AND DISCONNECT CISCO VPN! <<<")
        print(" The process is running independently in the background on dichtator.")
        print(" To inspect live status later:")
        print(f"   python _david/code/reserch_test/run_hypotheses.py --status")
        print(f"   tail -f {target_out / 'experiments.log'}")
        print("=" * 85 + "\n")
        return

    # Synchronous execution
    runner = ExperimentRunner(
        experiment_ids=exp_ids,
        epochs=args.epochs,
        n_samples=args.n_samples,
        gpu=args.gpu,
        output_dir=args.out_dir,
        include_hb=args.include_hb,
        drop_off=not args.no_drop_off,
        patience=args.patience,
        rocs_threshold=args.rocs_threshold,
        prop_threshold=args.prop_threshold,
        sa_threshold=args.sa_threshold,
        criteria=args.criteria,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
