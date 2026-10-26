from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class GridJobSpec:
    """Resource specifications and execution parameters for a MetaCentrum PBS Pro job."""

    experiment_name: str
    args: str
    n_cpus: int = 32
    mem_gb: int = 64
    n_gpus: int = 1
    gpu_cap: str = "cuda80"
    scratch_gb: int = 30
    walltime: str = "14:00:00"


class MetaCentrumGridManager:
    """Class-oriented manager for orchestrating parallel DrugEx experiments on MetaCentrum grid."""

    def __init__(
        self,
        remote_host: str = "meta",
        remote_user: str = "kolarv",
        storage_site: str = "praha1",
    ) -> None:
        self.remote_host = remote_host
        self.remote_user = remote_user
        self.storage_root = f"/storage/{storage_site}/home/{remote_user}/DrugEx"

    def test_connection(self) -> bool:
        """Verify SSH accessibility to MetaCentrum frontend."""
        cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            self.remote_host,
            "echo OK",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0 and "OK" in res.stdout

    def sync_codebase_to_grid(self, local_root: Path | str = PROJECT_ROOT) -> None:
        """Synchronize local codebase and pretrained models to MetaCentrum persistent storage."""
        dest = f"{self.remote_host}:{self.storage_root}/"
        cmd = [
            "rsync",
            "-avz",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=_david/output",
            "--exclude=.venv",
            f"{local_root}/",
            dest,
        ]
        subprocess.run(cmd, check=True)

    def submit_job(self, spec: GridJobSpec) -> str:
        """Submit a single DrugEx experiment to the PBS Pro queue."""
        qsub_cmd = (
            f"qsub -N {spec.experiment_name} "
            f"-l select=1:ncpus={spec.n_cpus}:mem={spec.mem_gb}gb:ngpus={spec.n_gpus}:"
            f"gpu_cap={spec.gpu_cap}:scratch_local={spec.scratch_gb}gb "
            f"-l walltime={spec.walltime} "
            f"-v EXP_NAME='{spec.experiment_name}',EXP_ARGS='{spec.args}' "
            f"{self.storage_root}/_david/metacentrum/job_template.sh"
        )
        ssh_cmd = ["ssh", self.remote_host, qsub_cmd]
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        job_id = res.stdout.strip()
        return job_id

    def submit_matrix(self, specs: Sequence[GridJobSpec]) -> list[str]:
        """Submit multiple experiments concurrently to run in parallel across the grid."""
        job_ids: list[str] = []
        for spec in specs:
            jid = self.submit_job(spec)
            job_ids.append(jid)
        return job_ids

    def get_queue_status(self) -> str:
        """Query PBS Pro queue status for user jobs."""
        ssh_cmd = ["ssh", self.remote_host, f"qstat -u {self.remote_user}"]
        res = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return res.stdout.strip()

    def sync_results_from_grid(self, local_output_dir: Path | str) -> None:
        """Synchronize completed experiment figures, metrics, and models back to local storage."""
        src = f"{self.remote_host}:{self.storage_root}/_david/output/"
        dest = f"{local_output_dir}/"
        cmd = ["rsync", "-avz", src, dest]
        subprocess.run(cmd, check=True)


def main() -> None:
    """CLI utility for managing MetaCentrum grid synchronization and job tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="DrugEx MetaCentrum Grid CLI")
    parser.add_argument(
        "action",
        choices=["test", "sync", "status", "pull"],
        help="Action to execute: 'test' connection, 'sync' code, check 'status', or 'pull' results",
    )
    parser.add_argument(
        "--host", default="meta", help="SSH host alias (default: 'meta')"
    )
    parser.add_argument(
        "--user", default="kolarv", help="MetaCentrum username (default: 'kolarv')"
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_david" / "output_grid"),
        help="Local directory for pulling results",
    )
    args = parser.parse_args()

    mgr = MetaCentrumGridManager(remote_host=args.host, remote_user=args.user)

    if args.action == "test":
        ok = mgr.test_connection()
        print(
            f"MetaCentrum connection to '{args.host}': {'SUCCESS' if ok else 'FAILED (Key not yet authorized)'}"
        )
    elif args.action == "sync":
        print(f"Synchronizing codebase to {args.host}:{mgr.storage_root}...")
        mgr.sync_codebase_to_grid()
        print("Synchronization completed successfully.")
    elif args.action == "status":
        print(mgr.get_queue_status() or "No active jobs in queue.")
    elif args.action == "pull":
        print(f"Pulling deliverables to {args.output_dir}...")
        mgr.sync_results_from_grid(args.output_dir)
        print("Pull completed.")


if __name__ == "__main__":
    main()
