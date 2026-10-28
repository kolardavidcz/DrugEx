from __future__ import annotations

import subprocess
import json
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from _david.code.config import PROJECT_ROOT, PipelineConfig


@dataclass(frozen=True)
class GridJobSpec:
    """Resource specifications and execution parameters for a MetaCentrum PBS Pro job."""

    experiment_name: str
    args: str
    n_cpus: int = 16
    mem_gb: int = 32
    n_gpus: int = 1
    gpu_cap: str = "compute_80"
    scratch_gb: int = 15
    walltime: str = "12:00:00"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.experiment_name):
            raise ValueError(
                "Use a unique experiment name containing letters, digits, '_' or '-'"
            )
        if min(self.n_cpus, self.mem_gb, self.scratch_gb) < 1 or self.n_gpus < 0:
            raise ValueError("Invalid PBS resources")


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
        """Stage a committed release, excluding outputs and preserving older releases."""
        root = Path(local_root)
        pkg, vocab = PipelineConfig().resolve_pretrained_paths()
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ):
            raise ValueError("Commit reviewed task changes before staging a release")
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        release = f"{self.storage_root}/releases/{revision}"
        command = (
            f"mkdir -p {shlex.quote(self.storage_root + '/releases')} && "
            f"mkdir {shlex.quote(release)} && tar -xf - -C {shlex.quote(release)}"
        )
        with tempfile.TemporaryFile() as archive:
            subprocess.run(
                ["git", "-C", str(root), "archive", revision],
                stdout=archive,
                check=True,
            )
            archive.seek(0)
            subprocess.run(
                ["ssh", self.remote_host, command], stdin=archive, check=True
            )
        model_dir = f"{self.storage_root}/models"
        subprocess.run(
            ["ssh", self.remote_host, f"mkdir -p {shlex.quote(model_dir)}"], check=True
        )
        subprocess.run(
            ["rsync", "-a", str(pkg), str(vocab), f"{self.remote_host}:{model_dir}/"],
            check=True,
        )

    def submit_job(self, spec: GridJobSpec, revision: str) -> str:
        """Submit a single DrugEx experiment to the PBS Pro queue."""
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A full committed source revision is required")
        release = f"{self.storage_root}/releases/{revision}"
        run_dir = f"{self.storage_root}/jobs/{spec.experiment_name}"
        spec_path = f"{run_dir}/spec.json"
        args = shlex.split(spec.args)
        args += [
            "--agent-path",
            f"{self.storage_root}/models/Papyrus05.5_smiles_rnn_PT.pkg",
            "--vocab-path",
            f"{self.storage_root}/models/Papyrus05.5_smiles_rnn_PT.vocab",
        ]
        payload = json.dumps(
            dict(
                name=spec.experiment_name, args=args, source=release, revision=revision
            )
        )
        write_spec = (
            f"mkdir -p {shlex.quote(self.storage_root + '/jobs')} && "
            f"mkdir {shlex.quote(run_dir)} && cat > {shlex.quote(spec_path)}"
        )
        subprocess.run(
            ["ssh", self.remote_host, write_spec], input=payload, text=True, check=True
        )
        resources = f"select=1:ncpus={spec.n_cpus}:mem={spec.mem_gb}gb:scratch_local={spec.scratch_gb}gb"
        if spec.n_gpus:
            resources += f":ngpus={spec.n_gpus}:gpu_cap={spec.gpu_cap}"
        command = [
            "qsub",
            "-N",
            spec.experiment_name,
            "-l",
            resources,
            "-l",
            f"walltime={spec.walltime}",
            "-v",
            f"RUN_SPEC={spec_path},STORAGE_DIR={self.storage_root}",
            f"{release}/_david/code/metacentrum/job_template.sh",
        ]
        ssh_cmd = ["ssh", self.remote_host, shlex.join(command)]
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        job_id = res.stdout.strip()
        return job_id

    def submit_matrix(self, specs: Sequence[GridJobSpec], revision: str) -> list[str]:
        """Submit multiple experiments concurrently to run in parallel across the grid."""
        job_ids: list[str] = []
        for spec in specs:
            jid = self.submit_job(spec, revision)
            job_ids.append(jid)
        return job_ids

    def get_queue_status(self) -> str:
        """Query PBS Pro queue status for user jobs."""
        ssh_cmd = ["ssh", self.remote_host, f"qstat -u {self.remote_user}"]
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
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
        default=str(PROJECT_ROOT / "_david" / "output"),
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
