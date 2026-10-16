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
    parser.add_argument("--mode", default="full", choices=["full", "test", "benchmark_only"])
    parser.add_argument("--test", action="store_const", const="test", dest="mode")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=100, help="Number of validation samples per epoch")
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--detach", action="store_true", dest="detach", default=False, help="Run as background daemon process"
    )
    parser.add_argument(
        "--no-detach", action="store_false", dest="detach", help="Run interactively in foreground (default)"
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    HardwareManager.init_runtime_environment()
    args = parse_args()
    is_worker = os.environ.get("DRUGEX_DETACHED_WORKER") == "1"

    target_dir = args.out_dir or (PROJECT_ROOT / "_david" / "output" / datetime.now().strftime("%Y%m%d_%H%M%S"))
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
            "--mode", args.mode,
            "--epochs", str(args.epochs),
            "--n-samples", str(args.n_samples),
            "--val-samples", str(args.val_samples),
            "--molecule", args.molecule,
            "--gpu", str(args.gpu),
            "--out-dir", str(target_dir),
            "--no-detach",
        ]

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
        mode=args.mode,
        epochs=args.epochs,
        n_samples=args.n_samples,
        val_samples=args.val_samples,
        gpu=args.gpu,
    )
    pipeline.run(args.molecule)


if __name__ == "__main__":
    main()
