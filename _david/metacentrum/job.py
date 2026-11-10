"""What happens on the PBS worker node. Entered by ``pbs_launcher.sh``, stdlib only.

It must start before any project import works, so it imports nothing from ``_david``.

The contract, written by ``submit.py`` and read here:

``RELEASE_DIR``  immutable ``git archive`` of one commit on storage, passed by the launcher.
``RUN_SPEC``     JSON file with ``name``, ``args``, ``source`` and ``revision``.
``STORAGE_DIR``  persistent tree holding ``.venv`` and ``_david/output``.
``SCRATCHDIR``   the node's scratch allocation, where the run actually happens.

Partial results are copied back on success, on a traceback, and on the SIGTERM PBS sends
at walltime, because a 23-hour run that is killed still holds every epoch it finished.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from contextlib import suppress
from pathlib import Path

#: Proof that a staged release is complete rather than half-extracted.
ENTRY_POINT = Path("_david") / "run.py"

SPEC_KEYS = ("name", "args", "source", "revision")
REQUIRED_ENV = ("PBS_JOBID", "SCRATCHDIR", "STORAGE_DIR", "RUN_SPEC")

#: One BLAS/OMP thread per process. The ROCS scorer and the relaxation pool already use
#: every allocated core, so a library that also threads inside each worker oversubscribes
#: the node by a factor of ncpus. ``forkserver`` avoids the fork-after-threads deadlock
#: that wedges a fork-Pool created after torch has started its own threads.
THREAD_ENV = {
    "MPLBACKEND": "Agg",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "DRUGEX_MP_CONTEXT": "forkserver",
    # The pose kernel wraps every molecule in molecule_time_limit(mol_timeout()). Without
    # this the limit is None, so the guard exists but never fires and one pathological
    # alignment can hold an epoch until the walltime kills the job.
    "DRUGEX_SCORE_MOL_TIMEOUT": "120",
}

#: PBS points TMPDIR at /var/tmp/pbs.<jobid> on the root filesystem, which carries a ~1 GB
#: per-user quota; one conformer batch is larger than that. Temp files go on scratch.
TEMP_ENV_KEYS = ("TMPDIR", "TMP", "TEMP")


class JobError(RuntimeError):
    """A fatal condition, reported as one actionable line instead of a traceback."""


def read_spec(path: Path) -> dict:
    """Load and validate ``spec.json``. A missing key is a staging bug, not a warning."""
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise JobError(f"job spec is gone: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JobError(f"job spec is not valid JSON: {path} ({exc})") from exc
    missing = [key for key in SPEC_KEYS if key not in spec]
    if missing:
        raise JobError(f"job spec {path} is missing {', '.join(missing)}")
    if not isinstance(spec["args"], list) or not all(isinstance(a, str) for a in spec["args"]):
        raise JobError(f"job spec {path} has a non-list 'args'")
    return spec


def allocated_cpus(env: dict[str, str]) -> int:
    """Cores PBS actually gave this job, counted from ``PBS_NODEFILE``.

    Load-bearing: ``os.cpu_count()`` reports the whole node, so on a shared 64-core
    machine a 16-core allocation would start 64 scoring workers on 16 cores. ``NCPUS`` has
    been seen reporting 1 for a multi-core job, so the nodefile is the authority.
    """
    nodefile = Path(env.get("PBS_NODEFILE", ""))
    if nodefile.is_file():
        lines = [line for line in nodefile.read_text().splitlines() if line.strip()]
        if lines:
            return len(lines)
    with suppress(ValueError, TypeError):
        return max(1, int(env.get("NCPUS", "")))
    return os.cpu_count() or 1


def job_environment(
    env: dict[str, str], release_copy: Path, temp_root: Path, revision: str | None = None
) -> dict[str, str]:
    """Worker environment: pinned threads, temp on scratch, the scratch release importable."""
    merged = dict(env)
    merged.update(THREAD_ENV)
    # A release is an extracted `git archive` with no .git directory, so the run cannot
    # discover its own revision. Without this every cluster manifest records
    # `revision: null` and the provenance that justifies the whole manifest is missing
    # exactly where it matters most.
    if revision:
        merged["DRUGEX_SOURCE_REVISION"] = revision
    for key in TEMP_ENV_KEYS:
        merged[key] = str(temp_root)
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = f"{release_copy}:{existing}" if existing else str(release_copy)
    return merged


def run_command(python: Path, output_root: Path, cpus: int, spec: dict) -> list[str]:
    """The pipeline invocation. Job-owned settings come first so a delta can override."""
    return [
        str(python),
        "-u",
        "-m",
        "_david.run",
        "--tag",
        spec["name"],
        "--set",
        f"output_root={output_root}",
        "--set",
        f"n_jobs={cpus}",
        *spec["args"],
    ]


def free_destination(root: Path, name: str) -> Path:
    """``root/name``, or the first free ``name_2``, ``name_3`` ... as the run's own rule."""
    candidate, suffix = root / name, 2
    while candidate.exists():
        candidate = root / f"{name}_{suffix}"
        suffix += 1
    return candidate


def copy_results(
    output_root: Path, storage_output: Path, submission: Path | None = None
) -> list[Path]:
    """Copy each run directory from scratch to storage, never merging into an existing one."""
    copied = []
    if not output_root.is_dir():
        return copied
    if submission and submission.is_file():
        for run_dir in (p for p in output_root.iterdir() if p.is_dir()):
            shutil.copy2(submission, run_dir / "submission.json")
    storage_output.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        destination = free_destination(storage_output, run_dir.name)
        shutil.copytree(run_dir, destination, symlinks=True)
        copied.append(destination)
    return copied


def clean_scratch(scratch: Path) -> None:
    """Free the allocation with PBS's own helper where it exists."""
    cleaner = shutil.which("clean_scratch")
    if cleaner:
        subprocess.run([cleaner], check=False)
        return
    if scratch.is_dir():
        for child in scratch.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


class _SignalExit(SystemExit):
    """Unwind through the copy-back path, then exit with the shell's 128+signal code."""


def install_signal_handlers() -> None:
    def handler(signum: int, _frame) -> None:
        raise _SignalExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    release_arg = Path(argv[1]).resolve() if argv[:1] == ["--release"] and len(argv) > 1 else None
    env = dict(os.environ)

    try:
        missing = [name for name in REQUIRED_ENV if not env.get(name)]
        if missing:
            raise JobError(
                f"missing PBS environment: {', '.join(missing)}. Submit through "
                "`python -m _david.submit submit`."
            )
        storage = Path(env["STORAGE_DIR"])
        scratch = Path(env["SCRATCHDIR"])
        spec_path = Path(env["RUN_SPEC"])
        python = storage / ".venv" / "bin" / "python"
        if not (python.is_file() and os.access(python, os.X_OK)):
            raise JobError(f"missing validated environment: {python}")
        spec = read_spec(spec_path)
        release = release_arg or Path(spec["source"]).resolve()
        if release_arg and Path(spec["source"]).resolve() != release_arg:
            raise JobError(
                f"spec points at {spec['source']} but the launcher came from {release_arg}"
            )
        if not (release / ENTRY_POINT).is_file():
            raise JobError(f"release is incomplete or gone: {release}")
    except JobError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cpus = allocated_cpus(env)
    output_root = scratch / "output"
    storage_output = storage / "_david" / "output"
    print("================ DrugEx v2 job ================")
    for key, value in (
        ("job", env.get("PBS_JOBID", "?")),
        ("name", spec["name"]),
        ("node", os.uname().nodename),
        ("queue", env.get("PBS_QUEUE", "unknown")),
        ("cpus", cpus),
        ("scratch", scratch),
        ("release", release),
        ("revision", spec["revision"]),
        ("deltas", " ".join(spec["args"]) or "(none)"),
        ("results", storage_output),
    ):
        print(f"{key:<9}: {value}")
    print("===============================================")

    install_signal_handlers()
    release_copy = scratch / "DrugEx"
    temp_root = scratch / "tmp"
    status = 1
    # Bound before anything that can raise. copytree can fail on a full scratch, and the
    # `finally` below passes `submission` to copy_results -- an UnboundLocalError there
    # would replace a clear "no space left on device" with a confusing traceback and skip
    # both the copy-back and the scratch cleanup.
    submission = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        if release_copy.exists():
            shutil.rmtree(release_copy)
        shutil.copytree(release, release_copy, symlinks=True)
        # Written inside the run directory, not beside it: copy_results only walks
        # directories, so a file at the output root was deleted with scratch and the
        # record of what was submitted never reached storage.
        submission = spec_path

        process = subprocess.run(
            run_command(python, output_root, cpus, spec),
            cwd=release_copy,
            env=job_environment(env, release_copy, temp_root, spec.get("revision")),
        )
        # A child killed by a signal reports -N; shells and PBS report 128+N.
        status = 128 - process.returncode if process.returncode < 0 else process.returncode
    except _SignalExit as exc:
        print(f"signalled, copying partial results back (exit {exc.code})", file=sys.stderr)
        status = int(exc.code or 143)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        # A second signal during the transfer must not interrupt the only copy of the data.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        try:
            for destination in copy_results(output_root, storage_output, submission):
                print(f"results copied to {destination}")
        except OSError as exc:
            print(f"result transfer FAILED ({exc}); scratch kept at {scratch}", file=sys.stderr)
            return 1
        clean_scratch(scratch)
    return status


if __name__ == "__main__":
    with suppress(BrokenPipeError):
        sys.exit(main())
