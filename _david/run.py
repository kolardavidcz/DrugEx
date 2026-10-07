"""IDE Direct Runner for DrugEx Ligand-Based De Novo Drug Design.

Press F5 or click Run in your IDE (VS Code / PyCharm) to execute directly.
Supports background detachment so you can safely close VS Code / SSH without killing the run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.pipeline import DEFAULT_MOLECULE, MoleculePipeline
from _david.code.prepare_query_conformer import get_safe_parallel_jobs

# [PACKAGE: os] WHY: Cap CPU multiprocessing workers across DrugEx to at most 24 cores (half host cores) and enforce forkserver to prevent mutex deadlocks. WHAT: Process env configuration. HOW: os.environ.setdefault.
SAFE_CORES: int = get_safe_parallel_jobs()
os.environ.setdefault("DRUGEX_SCORE_NJOBS", str(SAFE_CORES))
os.environ.setdefault("DRUGEX_MP_CONTEXT", "forkserver")

# =============================================================================
# TIER 1: EXPERIMENTER HYPERPARAMETERS (TWEAK FREELY IN IDE)
# =============================================================================
# Execution mode: 'full' (train RL), 'test' (1-epoch sanity check), 'benchmark_only'
MODE: str = "full"

# Number of RL epochs (recommend: 10 or 20 on dichtator GPU, 1 on local CPU)
EPOCHS: int = 10

# Number of molecules sampled per RL epoch
N_SAMPLES: int = 200

# Target query template (default: EGCG flavan-3-ol ester proof molecule)
# Can be any valid SMILES string or Path to .sdf / .smi file
MOLECULE: str = DEFAULT_MOLECULE

# Background detachment (True = safe to close VS Code / SSH; False = interactive terminal)
DETACH: bool = True

# Target GPU device ('auto' automatically selects the GPU with highest free VRAM; or 0, 1 for explicit index)
GPU: str | int = "auto"

# Custom output directory: None auto-generates timestamped folder in _david/output/
OUTPUT_DIR: Path | None = None


def main() -> None:
    """Entry point for IDE and detached daemon execution."""
    global MODE, EPOCHS, N_SAMPLES, DETACH, MOLECULE, OUTPUT_DIR, GPU

    is_worker = os.environ.get("DRUGEX_DETACHED_WORKER") == "1"

    # Allow optional CLI flag override if invoked with arguments outside IDE
    if len(sys.argv) > 1 and not is_worker:
        import argparse
        parser = argparse.ArgumentParser(description="DrugEx LBDD Runner")
        parser.add_argument("--mode", default=MODE, choices=["full", "test", "benchmark_only"])
        parser.add_argument("--epochs", type=int, default=EPOCHS)
        parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
        parser.add_argument("--molecule", default=MOLECULE)
        parser.add_argument("--gpu", default=GPU, help="GPU target: 'auto', 0, 1")
        parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
        parser.add_argument("--no-detach", action="store_false", dest="detach", default=DETACH)
        args, _ = parser.parse_known_args()
        MODE = args.mode
        EPOCHS = args.epochs
        N_SAMPLES = args.n_samples
        MOLECULE = args.molecule
        GPU = args.gpu
        OUTPUT_DIR = args.out_dir
        DETACH = args.detach

    # Resolve target output directory
    if OUTPUT_DIR is not None:
        target_dir = Path(OUTPUT_DIR)
    else:
        cached_dir = os.environ.get("DRUGEX_OUTPUT_DIR")
        if cached_dir:
            target_dir = Path(cached_dir)
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = PROJECT_ROOT / "_david" / "output" / run_id

    target_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # CASE 1: SPAWN DETACHED BACKGROUND DAEMON (SURVIVES SSH / VPN DISCONNECT)
    # -------------------------------------------------------------------------
    if DETACH and not is_worker:
        log_file = target_dir / "pipeline.log"
        pid_file = target_dir / "run.pid"

        worker_env = dict(os.environ)
        worker_env["DRUGEX_DETACHED_WORKER"] = "1"
        worker_env["DRUGEX_OUTPUT_DIR"] = str(target_dir)
        worker_env["DRUGEX_SCORE_NJOBS"] = str(SAFE_CORES)
        worker_env["DRUGEX_MP_CONTEXT"] = "forkserver"
        worker_env["DRUGEX_GPU"] = str(GPU)

        popen_kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": worker_env,
            "stdin": subprocess.DEVNULL,
        }

        # Cross-platform detachment: setsid on POSIX, DETACHED_PROCESS on Windows
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True
        else:
            detached_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
            if detached_flags:
                popen_kwargs["creationflags"] = detached_flags
            else:
                popen_kwargs["start_new_session"] = True

        with open(log_file, "a", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())],
                stdout=f_out,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )

        with open(pid_file, "w", encoding="utf-8") as f_pid:
            f_pid.write(str(proc.pid))

        print("\n" + "=" * 80)
        print(" DRUGEX LBDD: DETACHED DAEMON SPAWNED ON DICHTATOR")
        print("=" * 80)
        print(f" • Process ID (PID): {proc.pid}")
        print(f" • Execution Mode:   {MODE.upper()} ({EPOCHS} epochs, {N_SAMPLES} samples/epoch)")
        print(f" • Target GPU:       {GPU}")
        print(f" • Output Directory: {target_dir}")
        print(f" • Live Log File:    {log_file}")
        print("-" * 80)
        print(" MONITORING COMMANDS:")
        print(f"   Watch live logs:   tail -f {log_file}")
        print(f"   Watch live status: watch -n 5 cat {target_dir}/pipeline_status.json")
        print(f"   Check GPU compute: nvidia-smi")
        print("-" * 80)
        print(" [SAFE TO DISCONNECT]")
        print(" You can now safely close VS Code, terminate your SSH session, or")
        print(" disconnect Cisco VPN. The job will continue running in the background")
        print(" until training, peak restoration, and 9-panel analysis complete!")
        print("=" * 80 + "\n")
        return

    # -------------------------------------------------------------------------
    # CASE 2: EXECUTE PIPELINE WORKER (INTERACTIVE OR DETACHED DAEMON)
    # -------------------------------------------------------------------------
    cached_gpu = os.environ.get("DRUGEX_GPU")
    if cached_gpu:
        GPU = int(cached_gpu) if cached_gpu.isdigit() else cached_gpu

    print("=" * 80)
    print(" DRUGEX LBDD: PIPELINE WORKER ACTIVE")
    print(f" • Mode:       {MODE}")
    print(f" • Epochs:     {EPOCHS}")
    print(f" • Samples:    {N_SAMPLES}")
    print(f" • Target GPU: {GPU}")
    print(f" • Directory:  {target_dir}")
    print("=" * 80)

    pipeline = MoleculePipeline(
        output_dir=target_dir,
        mode=MODE,
        epochs=EPOCHS,
        n_samples=N_SAMPLES,
        gpu=GPU,
    )
    pipeline.run(MOLECULE)


if __name__ == "__main__":
    main()
