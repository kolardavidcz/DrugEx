"""Comparative Threshold, Exploration Rate & Anti-Cheating Ablation Suite (Log-Off Daemon).

====================================================================================================
ACADEMIC HYPOTHESIS & FIRST-PRINCIPLES STUDY DESIGN (BACHELOR THESIS LBDD)
====================================================================================================

This script evaluates six fundamental research hypotheses on multi-objective reinforcement
learning (MORL) dynamics in ligand-based de novo drug design against the 3D reference Epigallocatechin
gallate (EGCG):

----------------------------------------------------------------------------------------------------
STUDY 1: PURE 3D ROCS + SA FLOOR (50 EPOCHS, ANTI-COLLAPSE SOFT-GATE)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Stripping secondary property constraints (HAC, CLogP) isolates the pure
    interaction between 3D conformational shape/color alignment (RDKit ROCS) and synthetic
    feasibility (SA floor <= 5.5). Run for 50 epochs to observe late-stage convergence.
  • Academic Test: Does removing property envelope bounds allow unconstrained shape complementarity,
    or does it trigger "Volume-Grease Exploitation" (MW > 600, CLogP > 5.5)?
  • Configuration: include_envelope = False, epochs = 50, use_anti_cheating = True, epsilon = 0.10.

----------------------------------------------------------------------------------------------------
STUDY 2: PURE 3D ROCS + SA FLOOR WITHOUT ANTI-CHEATING (NATIVE WEIGHTED-SUM, 35 EPOCHS)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Standard DrugEx WeightedSum reward scalarization without soft gating permits Pareto
    cheating even in a 2-objective setup, because trivial molecules with SA=1.0 harvest rewards
    despite failing 3D ROCS (S_3D = 0).
  • Academic Test: Does native WeightedSum lead to policy degradation / cheating compared to Study 1?
  • Configuration: include_envelope = False, epochs = 35, use_anti_cheating = False, epsilon = 0.10.

----------------------------------------------------------------------------------------------------
STUDY 3: BALANCED PHYSICAL ENVELOPE (PROP_THRESHOLD = 0.30, 35 EPOCHS)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Applying a loose Gaussian envelope on HAC and CLogP at tau = 0.30 with
    AntiCollapseSoftGateReward eliminates volume-grease inflation while preventing early drop-out.
  • Academic Test: Baseline multi-objective lead optimization benchmark.
  • Configuration: include_envelope = True, prop_threshold = 0.30, epochs = 35, epsilon = 0.10.

----------------------------------------------------------------------------------------------------
STUDY 4: PERMISSIVE PHYSICAL ENVELOPE (PROP_THRESHOLD = 0.20, 35 EPOCHS)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Setting prop_threshold = 0.20 broadens the Gaussian property acceptance window,
    measuring whether a more permissive property filter enhances policy exploration.
  • Academic Test: Phase transition between tau = 0.20 and tau = 0.30.
  • Configuration: include_envelope = True, prop_threshold = 0.20, epochs = 35, epsilon = 0.10.

----------------------------------------------------------------------------------------------------
STUDY 5: REDUCED EXPLORATION RATE (PROP_THRESHOLD = 0.30, MUTATE EPSILON = 0.05, 35 EPOCHS)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Halving prior mutation rate (epsilon = 0.05) shifts 95% of generation to the active
    agent policy, testing whether lower prior noise speeds up optimization toward the 3D target.
  • Academic Test: Impact of reducing prior sampling rate from 10% to 5%.
  • Configuration: include_envelope = True, prop_threshold = 0.30, epochs = 35, epsilon = 0.05.

----------------------------------------------------------------------------------------------------
STUDY 6: ZERO MUTATION EXPLORATION RATE (PROP_THRESHOLD = 0.30, MUTATE EPSILON = 0.00, 35 EPOCHS)
----------------------------------------------------------------------------------------------------
  • Hypothesis: Setting epsilon = 0.0 completely shuts off sampling from the frozen Prior network,
    forcing 100% of generation through the active learning agent network.
  • Academic Test: Does turning off prior sampling trigger catastrophic mode collapse (generating
    identical repetitive scaffolds), or does it yield rapid greedy convergence on the 3D template?
  • Configuration: include_envelope = True, prop_threshold = 0.30, epochs = 35, epsilon = 0.00.

====================================================================================================
EXECUTION PROTOCOL:
====================================================================================================
  • Runs as a background daemon surviving SSH/VPN log-off on dichtator.
  • Automatic GPU selection queries free VRAM (targeting Blackwell RTX PRO 4000).
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

# [PACKAGE: numpy] WHY: Computes mean desirability and summary statistics across elite candidate pools. WHAT: Numerical computing library. HOW: np.mean.
import numpy as np
# [PACKAGE: pandas] WHY: Tabular summary DataFrame construction and TSV export. WHAT: Data manipulation library. HOW: pd.DataFrame, df.to_csv.
import pandas as pd
# [PACKAGE: torch] WHY: Device memory probe and CUDA environment initialization. WHAT: PyTorch deep learning framework. HOW: torch.cuda.
import torch

from _david.code import GraphTransformer
from _david.code.pipeline import MoleculePipeline, get_gpu_telemetry

EGCG_SMILES: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
DEFAULT_EPOCHS: int = 35
DEFAULT_N_SAMPLES: int = 200
DEFAULT_MIN_EPOCHS: int = 25

ABLATION_SPECS: dict[int, dict[str, Any]] = {
    1: {
        "name": "Pure_3D_ROCS_and_SA_50ep_Epsilon_020",
        "title": "Study 1: Pure 3D ROCS + SA Floor (50 Epochs, Drop-Off >= 25, Epsilon = 0.20)",
        "include_envelope": False,
        "epochs": 50,
        "min_epochs": 25,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.20,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Pure 3D conformational shape/color alignment (RDKit ROCS) + synthesizability floor (SA <= 5.5) with elevated Prior exploration noise (epsilon = 0.20) and drop-off active after 25 epochs.",
    },
    2: {
        "name": "Envelope_PropThresh_020_Epsilon_020_35ep",
        "title": "Study 2: Permissive Envelope (tau = 0.20, 35 Epochs, Drop-Off >= 25, Epsilon = 0.20)",
        "include_envelope": True,
        "epochs": 35,
        "min_epochs": 25,
        "rocs_threshold": 0.20,
        "prop_threshold": 0.20,
        "sa_threshold": 0.20,
        "epsilon": 0.20,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Permissive physical envelope on Heavy Atom Count and CLogP at tau = 0.20 with elevated Prior exploration noise (epsilon = 0.20) and drop-off active after 25 epochs.",
    },
    3: {
        "name": "Historical_Pure_3D_ROCS_and_SA_50ep",
        "title": "Overnight Benchmark 1: Pure 3D ROCS + SA Floor (50 Epochs, Soft-Gate Anti-Collapse, Epsilon = 0.10)",
        "include_envelope": False,
        "epochs": 50,
        "min_epochs": 10,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.10,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Pure 3D shape + feasibility floor with AntiCollapseSoftGateReward run for 50 epochs (baseline epsilon = 0.10).",
    },
    4: {
        "name": "Historical_Pure_3D_ROCS_and_SA_NoAntiCheating_35ep",
        "title": "Overnight Benchmark 2: Pure 3D ROCS + SA Floor WITHOUT Anti-Cheating (Native WeightedSum, 35 Epochs)",
        "include_envelope": False,
        "epochs": 35,
        "min_epochs": 10,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.10,
        "use_anti_cheating": False,
        "patience": 15,
        "description": "Direct ablation control: tests whether native WeightedSum leads to Pareto cheating on 3D vs SA.",
    },
    5: {
        "name": "Historical_Envelope_PropThresh_030_35ep",
        "title": "Overnight Benchmark 3: Multi-Objective Envelope (HAC + CLogP), prop_threshold = 0.30 (35 Epochs, Epsilon = 0.10)",
        "include_envelope": True,
        "epochs": 35,
        "min_epochs": 10,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.10,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Balanced lead optimization physical envelope at tau = 0.30 with default epsilon = 0.10.",
    },
    6: {
        "name": "Historical_Envelope_PropThresh_020_35ep",
        "title": "Overnight Benchmark 4: Multi-Objective Envelope (HAC + CLogP), prop_threshold = 0.20 (35 Epochs, Epsilon = 0.10)",
        "include_envelope": True,
        "epochs": 35,
        "min_epochs": 10,
        "rocs_threshold": 0.20,
        "prop_threshold": 0.20,
        "sa_threshold": 0.20,
        "epsilon": 0.10,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Permissive envelope tolerance at tau = 0.20 testing broader Gaussian property acceptance.",
    },
    7: {
        "name": "Historical_Envelope_PropThresh_030_Epsilon_005_35ep",
        "title": "Overnight Benchmark 5: Envelope tau = 0.30 with mutate epsilon = 0.05 (35 Epochs)",
        "include_envelope": True,
        "epochs": 35,
        "min_epochs": 10,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.05,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Reduced prior mutation rate (5%), placing 95% gradient weight on active agent policy.",
    },
    8: {
        "name": "Historical_Envelope_PropThresh_030_Epsilon_000_35ep",
        "title": "Overnight Benchmark 6: Envelope tau = 0.30 with mutate epsilon = 0.00 (Pure Greedy/Agent, 35 Epochs)",
        "include_envelope": True,
        "epochs": 35,
        "min_epochs": 10,
        "rocs_threshold": 0.30,
        "prop_threshold": 0.30,
        "sa_threshold": 0.30,
        "epsilon": 0.00,
        "use_anti_cheating": True,
        "patience": 15,
        "description": "Zero prior mutation rate (0% Prior, 100% Agent): tests for mode collapse vs accelerated convergence.",
    },
}

HISTORICAL_FOLDER_SPECS: dict[str, dict[str, Any]] = {
    "study1_Pure_3D_ROCS_and_SA_50ep": ABLATION_SPECS[3],
    "study2_Pure_3D_ROCS_and_SA_NoAntiCheating_35ep": ABLATION_SPECS[4],
    "study3_Envelope_PropThresh_030_35ep": ABLATION_SPECS[5],
    "study4_Envelope_PropThresh_020_35ep": ABLATION_SPECS[6],
    "study5_Envelope_PropThresh_030_Epsilon_005_35ep": ABLATION_SPECS[7],
    "study6_Envelope_PropThresh_030_Epsilon_000_35ep": ABLATION_SPECS[8],
}


def write_study_readme(
    study_dir: Path,
    spec: dict[str, Any],
    results: dict[str, Any] | None = None,
) -> Path:
    """Generate an academically rigorous, comprehensive README.md in the study output folder.

    Args:
        study_dir: Target directory of the study run.
        spec: Configuration specification dictionary for the study.
        results: Optional results dictionary containing run summary or status.

    Returns:
        Path to the generated README.md.
    """
    readme_path = study_dir / "README.md"
    summary_path = study_dir / "stage2_result_analysis" / "result_analysis_summary.json"
    leads_path = study_dir / "stage2_result_analysis" / "top_prioritized_leads.tsv"
    pipe_summary_path = study_dir / "pipeline_summary.json"

    # Load result data if available
    analysis_data: dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
        except Exception:
            pass

    pipe_data: dict[str, Any] = {}
    if pipe_summary_path.exists():
        try:
            with open(pipe_summary_path, "r", encoding="utf-8") as f:
                pipe_data = json.load(f)
        except Exception:
            pass

    vun = analysis_data.get("vun_metrics", {})
    dists = analysis_data.get("physicochemical_distributions", {})
    scaff = analysis_data.get("bemis_murcko_scaffold_diversity", analysis_data.get("scaffold_diversity", {}))
    leads = analysis_data.get("top_leads", [])
    duration_s = pipe_data.get("duration_seconds", results.get("duration_seconds") if results else None)

    title = spec.get("title", study_dir.name)
    desc = spec.get("description", "")
    epochs = spec.get("epochs", 35)
    min_epochs = spec.get("min_epochs", 25)
    patience = spec.get("patience", 15)
    epsilon = spec.get("epsilon", 0.10)
    rocs_t = spec.get("rocs_threshold", 0.30)
    prop_t = spec.get("prop_threshold", 0.30)
    sa_t = spec.get("sa_threshold", 0.30)
    envelope = spec.get("include_envelope", True)
    anti_cheating = spec.get("use_anti_cheating", True)

    lines: list[str] = [
        f"# {title}",
        "",
        "## 1. Academic Hypothesis & Scientific Motivation",
        f"{desc}",
        "",
        rf"- **Core Scientific Question**: How does this specific exploration noise ($\epsilon$) and objective boundary set affect the multi-objective Pareto frontier and prevent policy degradation during 3D lead optimization?",
        rf"- **Exploration vs Exploitation**: Prior mutation sampling rate $\epsilon = {epsilon:.2f}$ determines the stochastic injection of general chemical space fragments from Papyrus vs exploitation of the active Graph-Transformer policy.",
        f"- **Reward Architecture**: {'Continuous `AntiCollapseSoftGateReward` (eliminating Pareto cheating by gating secondary envelopes behind 3D ROCS alignment)' if anti_cheating else 'Standard DrugEx `WeightedSum` (ablation control without soft gating)'}.",
        "",
        "## 2. Experimental Configuration & Hyperparameters",
        "| Hyperparameter | Value | Description & Rationale |",
        "|:---|:---|:---|",
        f"| **Target Horizon** | `{epochs} epochs` | Total reinforcement learning exploration budget |",
        f"| **Drop-Off (Early Stopping)** | `min_epochs={min_epochs}`, `patience={patience}` | Drop-off monitoring only activates after epoch {min_epochs} to guarantee sufficient exploration |",
        rf"| **Prior Mutation Rate ($\epsilon$)** | `{epsilon:.2f}` | Fraction of tokens/fragments mutated via frozen Prior |",
        rf"| **3D ROCS Scorer** | RDKit ROCS (`oe_mills_dean`, `TanimotoCombo`) | Alignment against 3D query template ($\tau = {rocs_t:.2f}$) |",
        rf"| **Property Envelope** | `{'Enabled (HAC + CLogP)' if envelope else 'Disabled (Pure 3D Shape & Feasibility)'}` | Dynamic envelope bounded around query ($\tau = {prop_t:.2f}$) |",
        rf"| **Synthesizability Floor** | Single SAscore floor ($\le 5.5$) | Unpenalized feasibility floor ($\tau = {sa_t:.2f}$) |",
        f"| **Reward Scalarization** | `{'AntiCollapseSoftGateReward' if anti_cheating else 'Native WeightedSum'}` | Mathematical gating to prevent lipophilic/volume cheating |",
        f"| **Sampling per Epoch** | `200 molecules` | High-fidelity policy rollout estimation |",
        "",
        "## 3. Directory Layout & Key Artifacts",
        "```",
        f"{study_dir.name}/",
        "├── stage1_graph_transformer/        # RL training checkpoints & generation",
        "│   ├── david_agent.fit.tsv           # Epoch-by-epoch rolling loss and desirability",
        "│   ├── david_agent_peak.pt           # Peak desirability model weights restored for inference",
        "│   └── generated_candidates.tsv      # Raw generated molecule pool",
        "├── stage2_result_analysis/          # Quantitative profiling & lead selection",
        "│   ├── result_analysis_summary.json  # Comprehensive VUN metrics and distributions",
        "│   ├── top_prioritized_leads.tsv     # Ranked elite lead candidates",
        "│   ├── multi_property_distributions.png # 2x4 descriptor distributions",
        "│   └── top_prioritized_leads_2d.png  # High-resolution 2D chemical structures",
        "├── fallback_audit.json               # Zero-fallback execution audit",
        "├── pipeline_summary.json             # Pipeline execution metadata & GPU telemetry",
        "├── pipeline.log                      # Complete execution transcript",
        "└── README.md                         # This academic report",
        "```",
    ]

    if vun or dists:
        des_vals = [float(row.get("Desirability", 0.0)) for row in leads if isinstance(row, dict)]
        mean_des = round(float(np.mean(des_vals)), 4) if des_vals else "N/A"
        dur_str = f"{duration_s / 60.0:.1f} min" if duration_s else "N/A"

        lines.extend([
            "",
            "## 4. Empirical Performance & Lead Quality",
            "",
            "### Summary Metrics",
            "| Metric | Result | Benchmark Context |",
            "|:---|:---|:---|",
            f"| **Execution Duration** | `{dur_str}` | Full RL training + conformer scoring |",
            f"| **Total Generated** | `{vun.get('total_generated', 'N/A')}` | Raw generated token sequences |",
            f"| **Validity** | `{vun.get('validity_percentage', 0.0):.1f}%` ({vun.get('valid_count', 'N/A')} molecules) | 100% chemically valid graphs |",
            f"| **Internal Uniqueness** | `{vun.get('uniqueness_percentage', 0.0):.1f}%` | Non-redundant novel chemical matter |",
            f"| **Novelty (vs Papyrus)** | `{vun.get('novelty_percentage', 0.0):.1f}%` | Genuine de novo generation |",
            f"| **Distinct Murcko Scaffolds** | `{scaff.get('unique_scaffold_count', scaff.get('distinct_scaffolds', 'N/A'))}` | Diverse core topology count |",
            f"| **Scaffold Diversity Ratio** | `{scaff.get('diversity_ratio', scaff.get('scaffold_diversity_ratio', 0.0)):.4f}` | Scaffolds / valid molecules |",
            f"| **Mean 3D ROCS Alignment** | `{dists.get('ROCS', {}).get('mean', 'N/A')}` | 3D Gaussian shape & Mills-Dean color overlap |",
            f"| **Mean Molecular Weight (MW)** | `{dists.get('MW', {}).get('mean', 'N/A')}` g/mol | Controlled size drift |",
            f"| **Mean CLogP (Lipophilicity)** | `{dists.get('LogP', {}).get('mean', 'N/A')}` | Wildcard grease indicator |",
            rf"| **Mean Synthesizability (SA)** | `{dists.get('SA', {}).get('mean', 'N/A')}` | Feasibility ($\le 5.5$ is unpenalized) |",
            f"| **Mean Desirability** | `{mean_des}` | Top prioritized lead quality |",
        ])

        if leads:
            lines.extend([
                "",
                "### Top Prioritized Elite Leads",
                "| # | SMILES | ROCS | MW | CLogP | SA | Desirability | Scaffold |",
                "|:---|:---|:---|:---|:---|:---|:---|:---|",
            ])
            for idx, lead in enumerate(leads[:5], 1):
                smi = lead.get("SMILES", "")
                short_smi = smi if len(smi) <= 35 else smi[:32] + "..."
                rocs_val = round(float(lead.get("ROCS", 0.0)), 3)
                mw_val = round(float(lead.get("MW", 0.0)), 1)
                logp_val = round(float(lead.get("LogP", 0.0)), 2)
                sa_val = round(float(lead.get("SA", 0.0)), 2)
                des_val = round(float(lead.get("Desirability", 0.0)), 4)
                scaff_val = lead.get("Scaffold", "")
                short_scaff = scaff_val if len(scaff_val) <= 25 else scaff_val[:22] + "..."
                lines.append(f"| {idx} | `{short_smi}` | `{rocs_val}` | `{mw_val}` | `{logp_val}` | `{sa_val}` | `{des_val}` | `{short_scaff}` |")

    readme_content = "\n".join(lines) + "\n"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    return readme_path


def generate_readmes_for_directory(target_dir: Path) -> list[Path]:
    """Generate or update README.md for all study subdirectories in target_dir."""
    created_paths: list[Path] = []
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist.")
        return created_paths

    for child in sorted(target_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("study"):
            continue
        spec = HISTORICAL_FOLDER_SPECS.get(child.name)
        if spec is None:
            try:
                study_num = int(child.name[5:].split("_")[0])
                spec = ABLATION_SPECS.get(study_num)
            except ValueError:
                pass
        if spec is None:
            spec = {
                "title": child.name.replace("_", " "),
                "name": child.name,
                "description": "DrugEx RL Lead Optimization Study.",
                "epochs": 35,
                "min_epochs": 25,
                "patience": 15,
                "epsilon": 0.10,
                "rocs_threshold": 0.30,
                "prop_threshold": 0.30,
                "sa_threshold": 0.30,
                "include_envelope": True,
                "use_anti_cheating": True,
            }
        p = write_study_readme(child, spec)
        created_paths.append(p)
        print(f"Generated study documentation: {p}")
    return created_paths


class ThresholdAblationRunner:
    """Orchestrates comparative hypothesis testing across target studies."""

    def __init__(
        self,
        study_ids: Sequence[int] | None = None,
        epochs: int = DEFAULT_EPOCHS,
        n_samples: int = DEFAULT_N_SAMPLES,
        gpu: str | int | None = "auto",
        output_dir: Path | str | None = None,
        drop_off: bool = True,
        patience: int = 15,
        min_epochs: int = DEFAULT_MIN_EPOCHS,
    ) -> None:
        """Initialize ThresholdAblationRunner with target studies and execution hyperparameters."""
        self.repo_root: Path = REPO_ROOT
        self.study_ids: list[int] = list(study_ids) if study_ids else [1, 2]
        self.epochs: int = epochs
        self.n_samples: int = n_samples
        self.gpu_arg: str | int | None = gpu
        self.drop_off: bool = drop_off
        self.patience: int = patience
        self.min_epochs: int = int(min_epochs)

        from _david.code.GraphTransformer import get_safe_gpus
        self.safe_gpus: tuple[int, ...] | tuple[()] = get_safe_gpus(preferred_gpu=gpu)
        self.selected_gpu_idx: int | None = self.safe_gpus[0] if len(self.safe_gpus) > 0 else None

        # Unique output directory for this benchmark session
        self.run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir: Path = self.repo_root / "_david" / "output" / f"ablation_{self.run_id}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file: Path = self.output_dir / "ablation.log"
        self.status_file: Path = self.output_dir / "status.json"
        self.pid_file: Path = self.output_dir / "process.pid"

        self._start_time: datetime = datetime.now()
        self.write_pid()
        self.update_status("INITIALIZING", details={"studies": self.study_ids})

    def log(self, message: str) -> None:
        """Log message with timestamp to stdout and ablation.log."""
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t_str}] {message}"
        print(line)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def write_pid(self) -> None:
        """Record current process ID for background process monitoring."""
        try:
            with open(self.pid_file, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

    def update_status(
        self,
        overall_status: str,
        current_study: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Update JSON status payload for live telemetry inspection."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        status_payload: dict[str, Any] = {
            "run_id": self.run_id,
            "pid": os.getpid(),
            "overall_status": overall_status,
            "current_study": current_study,
            "elapsed_seconds": round(elapsed, 2),
            "selected_gpu": f"GPU {self.selected_gpu_idx}" if self.selected_gpu_idx is not None else "CPU",
            "epochs_per_study": self.epochs,
            "samples_per_epoch": self.n_samples,
            "drop_off": self.drop_off,
            "patience": self.patience,
            "updated_at": datetime.now().isoformat(),
            "details": details or {},
        }
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(status_payload, f, indent=2)
        except Exception:
            pass

    def run_all(self) -> dict[str, Any]:
        """Execute all configured studies sequentially, persisting leads and cross-study comparisons."""
        self.log("=" * 85)
        self.log(f" STARTING THRESHOLD & ENVELOPE ABLATION STUDY: {self.run_id}")
        self.log(f" Target Studies:    {self.study_ids}")
        self.log(f" Epochs / Samples:  {self.epochs} epochs, {self.n_samples} samples/epoch")
        self.log(f" Selected GPU:      Device {self.selected_gpu_idx}")
        self.log(f" Output Directory:  {self.output_dir}")
        self.log("=" * 85)

        ref_sdf = self.repo_root / "_david/references/query_egcg_3d.sdf"
        seed_scaffold = self.repo_root / "_david/references/seed_core_scaffold.tsv"

        results: dict[int, dict[str, Any]] = {}

        for study_id in self.study_ids:
            spec = ABLATION_SPECS[study_id]
            self.log("\n" + "#" * 85)
            self.log(f" >>> STARTING {spec['title']}")
            self.log(f" Description: {spec['description']}")
            self.log(f" Envelope:    {spec['include_envelope']} | ROCS_tau: {spec['rocs_threshold']} | Prop_tau: {spec['prop_threshold']} | SA_tau: {spec['sa_threshold']}")
            self.log("#" * 85)

            study_dir = self.output_dir / f"study{study_id}_{spec['name']}"
            study_dir.mkdir(parents=True, exist_ok=True)

            self.update_status(
                "RUNNING",
                current_study=study_id,
                details={"study_title": spec["title"], "stage": f"Running Study {study_id}"},
            )

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            study_start = datetime.now()
            try:
                study_epochs = spec.get("epochs", self.epochs)
                study_patience = spec.get("patience", self.patience)
                study_min_epochs = spec.get("min_epochs", self.min_epochs)
                study_epsilon = spec.get("epsilon", 0.10)
                study_anti_cheating = spec.get("use_anti_cheating", True)

                # Dynamically verify exclusive GPU before each study (pauses if supervisor/colleagues active)
                from _david.code.GraphTransformer import get_safe_gpus
                study_safe_gpus = get_safe_gpus(preferred_gpu=self.gpu_arg)
                study_gpu_idx = study_safe_gpus[0] if len(study_safe_gpus) > 0 else None

                # Initialize study documentation before RL starts
                write_study_readme(study_dir, spec)

                pipeline = MoleculePipeline(
                    output_dir=study_dir,
                    mode="full",
                    epochs=study_epochs,
                    n_samples=self.n_samples,
                    gpu=study_gpu_idx,
                    include_hb=False,
                    include_envelope=spec["include_envelope"],
                    drop_off=self.drop_off,
                    patience=study_patience,
                    min_epochs=study_min_epochs,
                    rocs_threshold=spec["rocs_threshold"],
                    prop_threshold=spec["prop_threshold"],
                    sa_threshold=spec["sa_threshold"],
                    epsilon=study_epsilon,
                    use_anti_cheating=study_anti_cheating,
                    criteria="desired_ratio",
                )
                run_summary = pipeline.run(
                    molecules=EGCG_SMILES,
                    custom_ref_sdf=ref_sdf,
                    custom_seed_core_tsv=seed_scaffold,
                )
                study_duration = (datetime.now() - study_start).total_seconds()
                results[study_id] = {
                    "status": "SUCCESS",
                    "duration_seconds": round(study_duration, 2),
                    "output_dir": str(study_dir),
                    "summary": run_summary,
                }
                self.log(f" [SUCCESS] Study {study_id} completed in {study_duration:.1f}s.")
                # Update documentation with full empirical metrics
                readme_p = write_study_readme(study_dir, spec, results[study_id])
                self.log(f" Generated study documentation: {readme_p}")

            except Exception as e:
                study_duration = (datetime.now() - study_start).total_seconds()
                self.log(f" [FAILED] Study {study_id} failed with error: {e}")
                results[study_id] = {
                    "status": "FAILED",
                    "duration_seconds": round(study_duration, 2),
                    "error": str(e),
                }
                write_study_readme(study_dir, spec, results[study_id])

            finally:
                # Explicitly purge PyTorch GPU cache and garbage collect between sequential studies
                try:
                    del pipeline
                except NameError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Cross-study comparative matrix compilation
        self.log("\n" + "=" * 85)
        self.log(" GENERATING CROSS-STUDY COMPARATIVE ABLATION MATRIX")
        self.log("=" * 85)
        comparative_df = self.generate_comparative_matrix(results)

        total_duration = (datetime.now() - self._start_time).total_seconds()
        self.update_status(
            "COMPLETED",
            details={
                "total_duration_seconds": round(total_duration, 2),
                "studies_completed": len([r for r in results.values() if r.get("status") == "SUCCESS"]),
            },
        )

        self.log("\n" + "=" * 85)
        self.log(f" ALL STUDIES FINISHED (Total Time: {total_duration / 60.0:.2f} minutes)")
        self.log(f" Results & summaries stored in: {self.output_dir}")
        self.log("=" * 85)

        return {
            "run_id": self.run_id,
            "duration_seconds": round(total_duration, 2),
            "results": results,
            "comparative_table": comparative_df.to_dict(orient="records") if comparative_df is not None else [],
        }

    def generate_comparative_matrix(self, results: dict[int, dict[str, Any]]) -> pd.DataFrame | None:
        """Compile comparative metrics across all completed studies into tabular summary."""
        records: list[dict[str, Any]] = []

        for study_id, res in results.items():
            spec = ABLATION_SPECS[study_id]
            if res.get("status") != "SUCCESS":
                records.append({
                    "Study_ID": study_id,
                    "Study_Name": spec["name"],
                    "Status": "FAILED",
                    "Duration_s": res.get("duration_seconds", 0.0),
                })
                continue

            exp_dir = Path(res["output_dir"])
            summary_json_path = exp_dir / "stage2_result_analysis" / "result_analysis_summary.json"
            if not summary_json_path.exists():
                records.append({
                    "Study_ID": study_id,
                    "Study_Name": spec["name"],
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
                "Study_ID": study_id,
                "Study_Name": spec["name"],
                "Envelope_Enabled": spec["include_envelope"],
                "Prop_Threshold": spec["prop_threshold"],
                "Target_Epochs": spec.get("epochs", self.epochs),
                "Mutate_Epsilon": spec.get("epsilon", 0.10),
                "Anti_Cheating_Reward": spec.get("use_anti_cheating", True),
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
                "MW_Mean": means.get("MW", {}).get("mean", 0.0),
                "HAC_Mean": means.get("HAC", {}).get("mean", 0.0),
                "LogP_Mean": means.get("LogP", {}).get("mean", 0.0),
                "SA_Mean": means.get("SA", {}).get("mean", 0.0),
                "Desirability_Mean": round(des_mean, 4),
            })

        if not records:
            return None

        df = pd.DataFrame(records)
        tsv_path = self.output_dir / "comparative_ablation.tsv"
        json_path = self.output_dir / "comparative_ablation.json"
        df.to_csv(tsv_path, sep="\t", index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        self.log(f"Comparative summary table saved to: {tsv_path}")
        self.log(f"Comparative JSON summary saved to: {json_path}\n")
        self.log(df.to_string(index=False))

        return df


def spawn_detached_runner(args_list: list[str], target_out_dir: Path) -> int:
    """Launch runner as detached background daemon surviving terminal closure."""
    target_out_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_out_dir / "ablation.log"
    pid_file = target_out_dir / "process.pid"

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
    """Print human-readable real-time status from latest or specified ablation directory."""
    if target_dir is None:
        out_root = REPO_ROOT / "_david" / "output"
        if not out_root.exists():
            print("No output directories found.")
            return
        candidates = sorted(
            [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("ablation_")],
            reverse=True,
        )
        if not candidates:
            print("No ablation directories found in _david/output/.")
            return
        target_dir = candidates[0]

    status_file = target_dir / "status.json"
    pid_file = target_dir / "process.pid"
    log_file = target_dir / "ablation.log"

    print("=" * 80)
    print(f" ABLATION STUDY STATUS: {target_dir.name}")
    print("=" * 80)

    pid_alive = False
    if pid_file.exists():
        pid_str = pid_file.read_text(encoding="utf-8").strip()
        try:
            pid = int(pid_str)
            os.kill(pid, 0)
            pid_alive = True
            print(f" • PID:            {pid} (ACTIVE / RUNNING)")
        except (ProcessLookupError, ValueError, OSError):
            print(f" • PID:            {pid_str} (INACTIVE / TERMINATED)")

    if status_file.exists():
        try:
            with open(status_file, encoding="utf-8") as f:
                data = json.load(f)
            print(f" • Overall Status: {data.get('overall_status', 'UNKNOWN')}")
            print(f" • Current Study:  {data.get('current_study', 'None')}")
            print(f" • Elapsed:        {data.get('elapsed_seconds', 0.0)} seconds")
            print(f" • Selected GPU:   {data.get('selected_gpu', 'UNKNOWN')}")
            print(f" • Updated At:     {data.get('updated_at', 'UNKNOWN')}")
        except Exception as e:
            print(f" • Status Read Error: {e}")

    print("-" * 80)
    print(" RECENT LOG OUTPUT (Last 15 lines):")
    print("-" * 80)
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-15:]:
                print(line.rstrip())
        except Exception as e:
            print(f"Log read error: {e}")
    else:
        print("Log file not yet created.")
    print("=" * 80 + "\n")


def stop_runner(target_dir: Path | None = None) -> None:
    """Terminate the active background runner via SIGTERM."""
    if target_dir is None:
        out_root = REPO_ROOT / "_david" / "output"
        candidates = sorted(
            [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("ablation_")],
            reverse=True,
        )
        if not candidates:
            print("No ablation directories found.")
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
    """CLI Entry Point for Threshold Ablation Study."""
    parser = argparse.ArgumentParser(
        description="Comparative Threshold & Envelope Ablation Study in DrugEx (Log-Off Daemon)"
    )
    parser.add_argument(
        "--studies",
        "-s",
        type=str,
        default="1,2",
        help="Studies to run: comma-separated list '1,2' or 'all' (default: '1,2')",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="RL epochs per study (default: 35)")
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS, help="Minimum epochs before drop-off early stopping activates (default: 25)")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Samples per epoch (default: 200)")
    parser.add_argument("--gpu", default="auto", help="GPU index (0, 1) or 'auto' (default: auto)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs (default: 15)")
    parser.add_argument("--no-drop-off", action="store_true", help="Disable drop off (early stopping), forcing full epochs")
    parser.add_argument("--out-dir", "-o", type=Path, default=None, help="Root output directory")
    parser.add_argument("--detach", action="store_true", help="Launch detached background daemon surviving SSH logout (default behavior)")
    parser.add_argument("--no-detach", action="store_true", help="Run synchronously in current shell session")
    parser.add_argument("--status", action="store_true", help="Inspect real-time status of latest background run")
    parser.add_argument("--stop", action="store_true", help="Terminate active background run")
    parser.add_argument("--generate-readmes", type=Path, nargs="?", const=REPO_ROOT / "_david" / "output", help="Generate/update README.md files for study directories")

    args = parser.parse_args()

    if args.generate_readmes:
        target_dir = args.generate_readmes
        if target_dir.name.startswith("study"):
            generate_readmes_for_directory(target_dir.parent)
        else:
            studies = [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("study")]
            if not studies:
                ablation_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("ablation_")], reverse=True)
                if ablation_dirs:
                    target_dir = ablation_dirs[0]
            generate_readmes_for_directory(target_dir)
        return

    if args.status:
        print_status(args.out_dir)
        return

    if args.stop:
        stop_runner(args.out_dir)
        return

    # Parse study list
    if args.studies.lower() == "all":
        study_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        study_ids = [int(x.strip()) for x in args.studies.split(",") if x.strip()]

    # If detached mode requested or default (when not --no-detach):
    if args.detach or not args.no_detach:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_out = args.out_dir or (REPO_ROOT / "_david" / "output" / f"ablation_{run_id}")
        fwd_args = [
            f"--studies={args.studies}",
            f"--epochs={args.epochs}",
            f"--min-epochs={args.min_epochs}",
            f"--n-samples={args.n_samples}",
            f"--gpu={args.gpu}",
            f"--out-dir={target_out}",
            f"--patience={args.patience}",
        ]
        if args.no_drop_off:
            fwd_args.append("--no-drop-off")

        pid = spawn_detached_runner(fwd_args, target_out)

        print("\n" + "=" * 85)
        print(" ABLATION STUDY LAUNCHED DETACHED IN BACKGROUND")
        print("=" * 85)
        print(f" • Process PID:  {pid}")
        print(f" • Output Dir:   {target_out}")
        print(f" • Log File:     {target_out / 'ablation.log'}")
        print(f" • Status File:  {target_out / 'status.json'}")
        print(f" • Studies:      {study_ids}")
        print(f" • Epochs:       {args.epochs} per study (with drop-off active after epoch {args.min_epochs})")
        print(f" • Drop-Off:     {'Enabled (Patience = ' + str(args.patience) + ', Min Epochs = ' + str(args.min_epochs) + ')' if not args.no_drop_off else 'Disabled (Forced full epochs)'}")
        print(f" • Samples:      {args.n_samples} per epoch")
        print("=" * 85)
        print(" >>> YOU CAN NOW SAFELY CLOSE SSH AND DISCONNECT CISCO VPN! <<<")
        print(" The process is running independently in the background on dichtator.")
        print(" To inspect live status later:")
        print(f"   python _david/code/reserch_test/run_threshold_ablation.py --status")
        print(f"   tail -f {target_out / 'ablation.log'}")
        print("=" * 85 + "\n")
        return

    # Synchronous execution
    runner = ThresholdAblationRunner(
        study_ids=study_ids,
        epochs=args.epochs,
        n_samples=args.n_samples,
        gpu=args.gpu,
        output_dir=args.out_dir,
        drop_off=not args.no_drop_off,
        patience=args.patience,
        min_epochs=args.min_epochs,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
