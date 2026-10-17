from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _david.code.environment import DEFAULT_MOLECULE
from _david.code.pipeline import MoleculePipeline
from _david.code.utils import HardwareManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DrugEx LBDD Runner")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run fast verification test mode (1 epoch, 10 samples)",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Number of RL epochs")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--n-samples", type=int, default=200, help="RL prompt batch size")
    parser.add_argument("--val-samples", type=int, default=100, help="Validation sample size")
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=200,
        help="Number of final candidates generated from peak agent",
    )
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE, help="Target molecule SMILES")
    parser.add_argument("--gpu", default="auto", help="GPU device ID or 'auto'/'all'/'cpu'")
    parser.add_argument(
        "--cpu-ratio",
        type=float,
        default=None,
        help="Fraction of CPU cores allocated for parallel 3D scoring (default: None, uses safe 0.60; use 0.80 or 0.90 for dedicated runs)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Custom output directory")
    parser.add_argument(
        "--detach", action="store_true", default=False, help="Run as background daemon process"
    )
    parser.add_argument(
        "--pareto", action="store_true", default=False, help="Enable Pareto evolution visualization"
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    HardwareManager.init_runtime_environment(ratio=args.cpu_ratio)
    is_worker = os.environ.get("DRUGEX_DETACHED_WORKER") == "1"

    target_dir = args.out_dir or (
        PROJECT_ROOT / "_david" / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.detach and not is_worker:
        log_file = target_dir / "pipeline.log"
        pid_file = target_dir / "run.pid"

        env = dict(os.environ)
        env["DRUGEX_DETACHED_WORKER"] = "1"
        env["DRUGEX_OUTPUT_DIR"] = str(target_dir)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--n-samples", str(args.n_samples),
            "--val-samples", str(args.val_samples),
            "--n-eval-samples", str(args.n_eval_samples),
            "--molecule", args.molecule,
            "--gpu", str(args.gpu),
            "--out-dir", str(target_dir),
        ]
        if args.test:
            cmd.append("--test")
        if args.pareto:
            cmd.append("--pareto")
        if args.cpu_ratio is not None:
            cmd.extend(["--cpu-ratio", str(args.cpu_ratio)])

        with open(log_file, "a", encoding="utf-8") as f_out:
            proc = subprocess.Popen(
                cmd,
                stdout=f_out,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                start_new_session=(sys.platform != "win32"),
            )

        with open(pid_file, "w", encoding="utf-8") as f_pid:
            f_pid.write(str(proc.pid))

        print(f"Spawned detached background process PID: {proc.pid}")
        print(f"Output directory: {target_dir}")
        print(f"Live log: tail -f {log_file}")
        return

    pipeline = MoleculePipeline(
        output_dir=target_dir,
        is_test=args.test,
        epochs=args.epochs,
        patience=args.patience,
        n_samples=args.n_samples,
        val_samples=args.val_samples,
        n_eval_samples=args.n_eval_samples,
        gpu=args.gpu,
        cpu_ratio=args.cpu_ratio,
        render_pareto=args.pareto,
    )
    pipeline.run(args.molecule)


if __name__ == "__main__":
    main()
