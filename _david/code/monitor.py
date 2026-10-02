"""Live Terminal Monitor and Dashboard for DrugEx LBDD Pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root and _david/code are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

#? [Pandas] Parses TSV training monitor tables for live RL loss & convergence metrics.
import pandas as pd


def find_latest_run_dir(base_dir: Path | None = None) -> Path | None:
    """Locate the most recent pipeline run output directory."""
    search_root = base_dir or (SCRIPT_DIR.parent / "output")
    if not search_root.exists():
        return None

    run_dirs = [d for d in search_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not run_dirs:
        return None

    run_dirs.sort(key=lambda d: d.name, reverse=True)
    return run_dirs[0]


def format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable MM:SS or HH:MM:SS string."""
    total_sec = int(round(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


def display_dashboard(run_dir: Path) -> None:
    """Read all telemetry files and print a formatted pipeline status dashboard."""
    status_file = run_dir / "pipeline_status.json"
    summary_file = run_dir / "pipeline_summary.json"
    audit_file = run_dir / "fallback_audit.json"
    log_file = run_dir / "pipeline.log"
    attrition_file = run_dir / "stage2_scaffold_funnel" / "funnel_attrition.json"

    status_data: dict[str, Any] = {}
    if status_file.exists():
        try:
            with open(status_file, encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass

    summary_data: dict[str, Any] = {}
    if summary_file.exists():
        try:
            with open(summary_file, encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    audit_data: dict[str, Any] = {}
    if audit_file.exists():
        try:
            with open(audit_file, encoding="utf-8") as f:
                audit_data = json.load(f)
        except Exception:
            pass

    run_id = status_data.get("run_id") or summary_data.get("run_id") or run_dir.name
    mode = status_data.get("mode") or summary_data.get("mode") or "UNKNOWN"
    overall_status = status_data.get("overall_status") or ("SUCCESS" if summary_file.exists() else "RUNNING")
    current_stage = status_data.get("current_stage") or ("COMPLETED" if summary_file.exists() else "RUNNING")
    elapsed = status_data.get("elapsed_seconds") or summary_data.get("duration_seconds") or 0.0

    # Color indicators (ANSI)
    is_tty = sys.stdout.isatty()
    green = "\033[92m" if is_tty else ""
    yellow = "\033[93m" if is_tty else ""
    red = "\033[91m" if is_tty else ""
    bold = "\033[1m" if is_tty else ""
    reset = "\033[0m" if is_tty else ""

    status_color = green if overall_status in ["SUCCESS", "COMPLETED"] else (yellow if overall_status == "RUNNING" else red)

    print("=" * 80)
    print(f" {bold}DRUGEX LBDD PIPELINE TELEMETRY MONITOR{reset}")
    print("=" * 80)
    print(f"  • Run Directory:  {run_dir}")
    print(f"  • Run ID:         {run_id}")
    print(f"  • Execution Mode: {mode.upper()}")
    print(f"  • Status:         {status_color}{overall_status}{reset}")
    print(f"  • Current Stage:  {current_stage}")
    print(f"  • Elapsed Time:   {format_duration(elapsed)}")
    print("-" * 80)

    # 1. Hardware & GPU Telemetry
    gpu_telemetry = status_data.get("gpu_telemetry", {})
    if gpu_telemetry.get("cuda_available") and gpu_telemetry.get("devices"):
        print(f" {bold}Hardware Telemetry (CUDA Online):{reset}")
        for dev in gpu_telemetry["devices"]:
            name = dev.get("name", "Unknown GPU")
            idx = dev.get("index", 0)
            alloc = dev.get("allocated_mb", 0.0)
            res = dev.get("reserved_mb", 0.0)
            total = dev.get("total_memory_mb", 0.0)
            pct = (alloc / total * 100.0) if total > 0 else 0.0
            print(f"  [GPU {idx}] {name}: {alloc:.1f} MB allocated ({pct:.1f}%) | {res:.1f} MB reserved | {total:.1f} MB total")
        print("-" * 80)

    # 2. Pipeline Stages Progress
    stages = status_data.get("stages", {})
    if stages:
        print(f" {bold}Pipeline Stage Statuses:{reset}")
        for st_name, st_status in stages.items():
            st_color = green if st_status == "SUCCESS" else (yellow if st_status == "RUNNING" else (red if st_status == "FAILED" else reset))
            print(f"  • {st_name:<38} [{st_color}{st_status}{reset}]")
        print("-" * 80)

    # 3. RL Training Convergence (Stage 1 & Stage 3)
    for _, st_name, st_path in [
        (1, "Stage 1: Sequence-RNN", run_dir / "stage1_sequence_rnn" / "reinforcement_learning" / "david_agent_fit.tsv"),
        (3, "Stage 3: Graph-Transformer", run_dir / "stage3_graph_transformer" / "reinforcement_learning" / "david_agent_fit.tsv"),
    ]:
        if st_path.exists():
            try:
                df_fit = pd.read_csv(st_path, sep="\t")
                if len(df_fit) > 0:
                    last_row = df_fit.iloc[-1]
                    epoch = int(last_row.get("epoch", len(df_fit)))
                    loss = float(last_row.get("loss_train", 0.0))
                    val_ratio = float(last_row.get("valid_ratio", 0.0)) * 100.0
                    uniq_ratio = float(last_row.get("unique_ratio", 0.0)) * 100.0
                    des_ratio = float(last_row.get("desired_ratio", 0.0)) * 100.0
                    print(f" {bold}{st_name} (Latest Epoch {epoch}):{reset}")
                    print(f"    Loss: {loss:.4f} | Valid: {val_ratio:.1f}% | Unique: {uniq_ratio:.1f}% | Desired: {des_ratio:.1f}%")
                    print("-" * 80)
            except Exception:
                pass

    # 4. Stage 2 Funnel Attrition
    if attrition_file.exists():
        try:
            with open(attrition_file, encoding="utf-8") as f:
                att = json.load(f)
            total_cand = att.get("total_input_candidates", 0)
            tier1 = att.get("tier1_survivors_count", 0)
            tier2 = att.get("tier2_rocs_survivors_count", 0)
            scaffs = att.get("final_scaffolds_selected", 0)
            clusters = att.get("butina_clusters_count", 0)
            print(f" {bold}Stage 2 Chemotype Funnel Attrition:{reset}")
            print(f"    Input Candidates: {total_cand} -> Tier 1 Survivors: {tier1} -> ROCS Verified: {tier2}")
            print(f"    Discovered Frameworks: {scaffs} centroid(s) across {clusters} Butina cluster(s)")
            print("-" * 80)
        except Exception:
            pass

    # 5. Fallback Audit Alerts
    fallbacks_count = audit_data.get("total_fallbacks", 0)
    if fallbacks_count > 0:
        counts_by_type = audit_data.get("counts_by_type", {})
        print(f" {bold}Fallback Monitoring ({yellow}{fallbacks_count} event(s) recorded{reset}):{reset}")
        for ftype, fcount in counts_by_type.items():
            print(f"    • {ftype}: {fcount} event(s)")
        print("-" * 80)
    elif audit_file.exists():
        print(f" {bold}Fallback Monitoring: {green}Zero fallbacks triggered (optimal algorithms){reset}")
        print("-" * 80)

    # 6. Recent Log Lines
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8") as f:
                lines = f.readlines()
            recent_lines = [line.strip() for line in lines[-6:] if line.strip()]
            if recent_lines:
                print(f" {bold}Recent Pipeline Log Entries:{reset}")
                for line in recent_lines:
                    print(f"  {line}")
                print("-" * 80)
        except Exception:
            pass


def main() -> int:
    """CLI entrypoint for pipeline monitoring."""
    parser = argparse.ArgumentParser(description="Live Terminal Monitor for DrugEx LBDD Pipeline")
    parser.add_argument("--dir", "-d", type=Path, default=None, help="Target run directory to monitor")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuously refresh and watch telemetry")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Refresh interval in seconds for --watch")
    args = parser.parse_args()

    target_dir = args.dir or find_latest_run_dir()
    if target_dir is None:
        print("[ERROR] No pipeline runs found in _david/output/. Run the pipeline first with `python _david/run.py`.", file=sys.stderr)
        return 1

    if not args.watch:
        display_dashboard(target_dir)
        return 0

    try:
        while True:
            # Clear terminal screen
            os.system("cls" if os.name == "nt" else "clear")
            display_dashboard(target_dir)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[Monitor Stopped]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
