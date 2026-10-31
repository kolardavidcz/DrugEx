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
    gpu_cap: str | None = None
    scratch_gb: int = 15
    walltime: str = "24:00:00"
    spec: float | None = 10.0

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
        remote_user: str = "kolardavidcz",
        storage_site: str = "brno2",
    ) -> None:
        self.remote_host = remote_host
        self.remote_user = remote_user
        # On MetaCentrum, /storage/brno2/home is a symlink owned by nobody; use canonical brno2/brno2
        if storage_site == "brno2":
            self.storage_root = f"/storage/{storage_site}/{storage_site}/{remote_user}/DrugEx"
        else:
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
            f"mkdir -p {shlex.quote(release)} && tar -xf - -C {shlex.quote(release)}"
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
            f"mkdir -p {shlex.quote(run_dir)} && cat > {shlex.quote(spec_path)}"
        )
        subprocess.run(
            ["ssh", self.remote_host, write_spec], input=payload, text=True, check=True
        )
        resources = f"select=1:ncpus={spec.n_cpus}:mem={spec.mem_gb}gb:scratch_local={spec.scratch_gb}gb"
        if spec.spec is not None:
            resources += f":spec={spec.spec}"
        if spec.n_gpus:
            resources += f":ngpus={spec.n_gpus}"
            if spec.gpu_cap:
                resources += f":gpu_cap={spec.gpu_cap}"
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


def generate_8arm_grid(
    prefix: str = "exp",
    molecule: str | None = None,
    n_cpus: int = 16,
    mem_gb: int = 32,
    n_gpus: int = 1,
    walltime: str = "24:00:00",
    scratch_gb: int = 15,
    spec: float | None = 10.0,
    gpu_cap: str | None = None,
) -> list[GridJobSpec]:
    """Generate canonical 8-arm PBS Pro experiment grid (2 QED x 2 Reward x 2 Conformer/Opt).

    Factors:
        1. QED Factor (2 levels):
           - without_qed: include_qed=False (--no-include-qed)
           - with_qed:    include_qed=True  (--include-qed)
        2. Reward Factor (2 levels):
           - pareto_crowding: ParetoCrowdingDistance (--reward-scheme pareto_crowding)
           - harrington:      HarringtonDesirabilityReward (--reward-scheme harrington)
        3. Conformer / Optimization Factor (2 levels):
           - unminimized_c30_k1: minimize_candidates=False, max_conformers=30, max_reference_conformers=1
           - minimized_c20_k4:   minimize_candidates=True,  max_conformers=20, max_reference_conformers=4

    Shared Hyperparameters:
        epochs=200, min_epochs=50, patience=50, epsilon=0.05, score_variant="tanimoto_combo".
    """
    qed_levels = [
        ("without_qed", "--no-include-qed"),
        ("with_qed", "--include-qed"),
    ]
    reward_levels = [
        ("pareto_crowding", "--reward-scheme pareto_crowding"),
        ("harrington", "--reward-scheme harrington"),
    ]
    conformer_levels = [
        (
            "unminimized_c30_k1",
            "--no-minimize-candidates --conformers 30 --reference-conformers 1",
        ),
        (
            "minimized_c20_k4",
            "--minimize-candidates --conformers 20 --reference-conformers 4",
        ),
    ]

    base_args = [
        "--epochs 200",
        "--min-epochs 50",
        "--patience 50",
        "--epsilon 0.05",
        "--score-variant tanimoto_combo",
    ]
    if molecule:
        base_args.append(f"--molecule {shlex.quote(molecule)}")

    specs: list[GridJobSpec] = []
    for qed_tag, qed_arg in qed_levels:
        for reward_tag, reward_arg in reward_levels:
            for conf_tag, conf_arg in conformer_levels:
                name = f"{prefix}_{qed_tag}_{reward_tag}_{conf_tag}"
                if len(name) > 64:
                    short_qed = "noqed" if qed_tag == "without_qed" else "qed"
                    short_rew = "pareto" if reward_tag == "pareto_crowding" else "harr"
                    short_conf = "unmin_c30" if "unminimized" in conf_tag else "min_c20"
                    suffix = f"_{short_qed}_{short_rew}_{short_conf}"
                    max_pfx = max(1, 64 - len(suffix))
                    name = f"{prefix[:max_pfx]}{suffix}"
                arm_args = " ".join([*base_args, qed_arg, reward_arg, conf_arg])
                specs.append(
                    GridJobSpec(
                        experiment_name=name,
                        args=arm_args,
                        n_cpus=n_cpus,
                        mem_gb=mem_gb,
                        n_gpus=n_gpus,
                        gpu_cap=gpu_cap,
                        scratch_gb=scratch_gb,
                        walltime=walltime,
                        spec=spec,
                    )
                )
    return specs


def main() -> None:
    """CLI utility for managing MetaCentrum grid synchronization and job tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="DrugEx MetaCentrum Grid CLI")
    parser.add_argument(
        "action",
        choices=["test", "sync", "status", "pull", "submit", "grid", "submit-grid"],
        help="Action to execute: 'test' connection, 'sync' code, check 'status', 'pull' results, 'submit' job, inspect 'grid', or 'submit-grid'",
    )
    parser.add_argument(
        "--host", default="meta", help="SSH host alias (default: 'meta')"
    )
    parser.add_argument(
        "--user", default="kolardavidcz", help="MetaCentrum username (default: 'kolardavidcz')"
    )
    parser.add_argument(
        "--storage-site", default="brno2", help="MetaCentrum storage site (default: 'brno2')"
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "_david" / "output"),
        help="Local directory for pulling results",
    )
    parser.add_argument(
        "--name",
        default="rl_default_tanimoto_combo",
        help="Experiment name prefix for submission / grid generation (default: 'rl_default_tanimoto_combo')",
    )
    parser.add_argument(
        "--args",
        default="",
        help="CLI arguments passed to _david.code.pipeline (default: empty, using config defaults)",
    )
    parser.add_argument(
        "--molecule",
        default=None,
        help="Optional target query molecule SMILES or SDF path for grid experiments",
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=16,
        help="Requested CPU cores (default: 16)",
    )
    parser.add_argument(
        "--mem",
        type=int,
        default=32,
        help="Requested RAM in GB (default: 32)",
    )
    parser.add_argument(
        "--scratch",
        type=int,
        default=15,
        help="Requested local scratch in GB (default: 15)",
    )
    parser.add_argument(
        "--spec",
        type=float,
        default=10.0,
        help="Minimum CPU SPEC benchmark score for high-performance cores (default: 10.0)",
    )
    parser.add_argument(
        "--walltime",
        default="24:00:00",
        help="Requested walltime HH:MM:SS (default: '24:00:00')",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Requested GPU count (default: 1; set 0 for CPU-only nodes)",
    )
    parser.add_argument(
        "--gpu-cap",
        default=None,
        help="Optional GPU capability filter, e.g. 'sm_89', 'sm_90' (default: None)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Committed Git commit hash to deploy (default: HEAD)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing codebase before submitting",
    )
    args = parser.parse_args()

    mgr = MetaCentrumGridManager(
        remote_host=args.host, remote_user=args.user, storage_site=args.storage_site
    )

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
    elif args.action == "grid":
        grid_prefix = (
            "rl_grid"
            if args.name == "rl_default_tanimoto_combo"
            else args.name
        )
        specs = generate_8arm_grid(
            prefix=grid_prefix,
            molecule=args.molecule,
            n_cpus=args.ncpus,
            mem_gb=args.mem,
            n_gpus=args.ngpus,
            gpu_cap=args.gpu_cap,
            scratch_gb=args.scratch,
            walltime=args.walltime,
            spec=args.spec,
        )
        print(f"Generated {len(specs)}-arm experiment grid:")
        for idx, spec in enumerate(specs, 1):
            print(f"[{idx}/8] {spec.experiment_name}")
            print(f"      Args: {spec.args}")
    elif args.action == "submit-grid":
        if not args.no_sync:
            print(f"Synchronizing codebase to {args.host}:{mgr.storage_root}...")
            mgr.sync_codebase_to_grid()
        revision = (
            args.revision
            or subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        )
        grid_prefix = (
            "rl_grid"
            if args.name == "rl_default_tanimoto_combo"
            else args.name
        )
        specs = generate_8arm_grid(
            prefix=grid_prefix,
            molecule=args.molecule,
            n_cpus=args.ncpus,
            mem_gb=args.mem,
            n_gpus=args.ngpus,
            gpu_cap=args.gpu_cap,
            scratch_gb=args.scratch,
            walltime=args.walltime,
            spec=args.spec,
        )
        print(f"Submitting {len(specs)} grid experiments to PBS Pro...")
        job_ids = mgr.submit_matrix(specs, revision)
        for spec, jid in zip(specs, job_ids):
            print(f"Submitted {spec.experiment_name}: {jid}")
    elif args.action == "submit":
        if not args.no_sync:
            print(f"Synchronizing codebase to {args.host}:{mgr.storage_root}...")
            mgr.sync_codebase_to_grid()
        revision = (
            args.revision
            or subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        )
        spec = GridJobSpec(
            experiment_name=args.name,
            args=args.args,
            n_cpus=args.ncpus,
            mem_gb=args.mem,
            n_gpus=args.ngpus,
            gpu_cap=args.gpu_cap,
            scratch_gb=args.scratch,
            walltime=args.walltime,
            spec=args.spec,
        )
        print(
            f"Submitting job '{spec.experiment_name}' with {spec.n_cpus} CPUs (spec>={spec.spec}) and {spec.n_gpus} GPU(s)..."
        )
        job_id = mgr.submit_job(spec, revision)
        print(f"Job successfully submitted to PBS Pro: {job_id}")


if __name__ == "__main__":
    main()
