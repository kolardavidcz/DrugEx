"""PBS job body: everything that happens on the worker node, in Python.

The PBS entry point is ``pbs_launcher.sh``, a six-line shim that only ``exec``s this
file with the stored virtualenv. Keeping the body here (instead of in the launcher)
means the scratch/copy-back contract is ordinary, unit-tested Python: run-directory
naming, environment pinning, argument forwarding, result transfer and signal exit
codes no longer live in a bash ``trap`` that can only be checked by reading it.

Contract (written by ``submit.py``, read here):

``RUN_SPEC``      JSON file with ``name``, ``args``, ``source`` and ``revision``.
``STORAGE_DIR``   persistent tree holding ``.venv`` and ``_david/output``.
``SCRATCHDIR``    PBS scratch allocation.
``PBS_JOBID``     PBS job id, printed in the header.

Deliberately stdlib-only: it must start before any project import works.
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
from datetime import datetime
from pathlib import Path

#: Matches ``config.RUN_STAMP_FORMAT``; duplicated because importing the package
#: would pull in torch before the job is validated. ``test_job_body`` asserts parity.
RUN_STAMP_FORMAT = "%Y%m%d_%H%M"

THREAD_ENV = {
    "MPLBACKEND": "Agg",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "DRUGEX_MP_CONTEXT": "forkserver",
}

REQUIRED_ENV = ("PBS_JOBID", "SCRATCHDIR", "STORAGE_DIR", "RUN_SPEC")
SPEC_KEYS = ("name", "args", "source", "revision")

#: File that proves a staged release is complete. It moved to ``training/`` in the
#: 2026-11-06 cleanup; the stale path only surfaced on the first job submitted
#: afterwards, which died 37 s in with "Release is incomplete or gone".
PIPELINE_ENTRY = Path("_david") / "code" / "training" / "pipeline.py"


class JobError(RuntimeError):
    """Fatal condition in the job body; printed as a single actionable line."""


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def read_spec(path: Path) -> dict:
    """Load and validate ``spec.json``; a missing key is a staging bug, not a warning."""
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise JobError(f"Job spec is gone: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JobError(f"Job spec is not valid JSON: {path} ({exc})") from exc
    missing = [key for key in SPEC_KEYS if key not in spec]
    if missing:
        raise JobError(f"Job spec {path} is missing {', '.join(missing)}")
    if not isinstance(spec["args"], list) or not all(isinstance(a, str) for a in spec["args"]):
        raise JobError(f"Job spec {path} has a non-list 'args'")
    return spec


def require_environment(env: dict[str, str]) -> None:
    """Fail fast with one line naming the variable PBS or submit.py forgot."""
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        raise JobError(
            f"Missing PBS environment: {', '.join(missing)}. "
            "Submit through `python -m _david.code.metacentrum.submit submit`."
        )


def job_python(storage: Path) -> Path:
    """The validated interpreter on storage; its absence is fatal, never repaired."""
    python = storage / ".venv" / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise JobError(f"Missing validated environment: {python}")
    return python


def job_cpus(env: dict[str, str]) -> str:
    """Core count from ``PBS_NODEFILE``; ``NCPUS`` reported 1 for a 4-core job."""
    nodefile = Path(env.get("PBS_NODEFILE", ""))
    if env.get("PBS_NODEFILE") and nodefile.is_file():
        return str(len(nodefile.read_text().splitlines()))
    return env.get("NCPUS") or "?"


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #


def claim_results_dir(root: Path, name: str, when: datetime | None = None) -> Path:
    """``<root>/YYYYmmdd_HHMM_<name>``, or ``…_2`` if another run owns that minute."""
    stamp = (when or datetime.now()).strftime(RUN_STAMP_FORMAT)
    candidate = root / f"{stamp}_{name}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{name}_{suffix}"
        suffix += 1
    return candidate


def results_layout(storage: Path, scratch: Path, name: str, when: datetime | None = None) -> tuple[Path, Path]:
    """Return ``(results_dir_on_storage, out_dir_in_scratch)`` with the same run id."""
    results = claim_results_dir(storage / "_david" / "output", name, when)
    return results, scratch / "output" / results.name


def header_rows(
    env: dict[str, str], spec: dict, release: Path, run_id: str, results: Path
) -> list[tuple[str, str]]:
    """The first lines of the PBS log, which are the run's provenance record."""
    return [
        ("job", env.get("PBS_JOBID", "?")),
        ("name", spec["name"]),
        ("node", os.uname().nodename),
        ("queue", env.get("PBS_QUEUE", "unknown")),
        ("cpus", job_cpus(env)),
        ("scratch", env.get("SCRATCHDIR", "?")),
        ("release", str(release)),
        ("revision", spec["revision"]),
        ("run id", run_id),
        ("results", str(results)),
    ]


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def pipeline_command(python: Path, out_dir: Path, spec: dict) -> list[str]:
    """``python -u -m _david.code.training.pipeline --output-dir <run> <pipeline flags…>``."""
    return [
        str(python),
        "-u",
        "-m",
        "_david.code.training.pipeline",
        "--output-dir",
        str(out_dir),
        *spec["args"],
    ]


def job_environment(env: dict[str, str], release_copy: Path, spec: dict) -> dict[str, str]:
    """Worker environment: pinned threads, forkserver, and the scratch release first."""
    merged = dict(env)
    merged.update(THREAD_ENV)
    merged["DRUGEX_SOURCE_REVISION"] = spec["revision"]
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = f"{release_copy}:{existing}" if existing else str(release_copy)
    return merged


def prepare_run(release: Path, release_copy: Path, spec_path: Path, out_dir: Path) -> None:
    """Create the scratch release and the run directory before the pipeline starts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if release_copy.exists():
        shutil.rmtree(release_copy)
    shutil.copytree(release, release_copy, symlinks=True)
    shutil.copy2(spec_path, out_dir / "submission.json")


def run_pipeline(
    command: list[str], cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    """Run the pipeline, mirroring stdout+stderr to the PBS log and ``execution.log``."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            with process.stdout:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
        except BaseException:
            # Walltime SIGTERM unwinds through here; stop the child before the copy-back
            # so the result directory is not modified while it is being transferred.
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            raise
        status = process.wait()
        # A child killed by a signal reports -N here; shells and PBS report 128+N.
        return 128 - status if status < 0 else status


def copy_results(out_dir: Path, results_dir: Path) -> None:
    """Copy the scratch run directory to storage; raises if the copy cannot complete."""
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir, results_dir, symlinks=True, dirs_exist_ok=True)


def clean_scratch(scratch: Path) -> None:
    """Free the allocation: ``clean_scratch`` when PBS provides it, else remove contents."""
    cleaner = shutil.which("clean_scratch")
    if cleaner:
        subprocess.run([cleaner], check=False)
        return
    if not scratch.is_dir():
        return
    for child in scratch.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


class _SignalExit(SystemExit):
    """Unwind through the copy-back path, then exit with the shell's 128+signal code."""


def install_signal_handlers() -> None:
    """SIGTERM (walltime) and SIGINT must still copy partial results back."""

    def handler(signum: int, _frame) -> None:
        raise _SignalExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, handler)


def finalize(out_dir: Path, results_dir: Path, scratch: Path, status: int) -> int:
    """Copy results back, then clean scratch. A failed copy keeps scratch and exits 1."""
    if out_dir.exists():
        try:
            copy_results(out_dir, results_dir)
        except OSError as exc:
            print(f"Result transfer FAILED ({exc}); scratch kept at {scratch}", file=sys.stderr)
            return 1
        print(f"Results copied to {results_dir} (exit status {status})")
    clean_scratch(scratch)
    return status


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    release_arg = None
    if argv[:1] == ["--release"]:
        if len(argv) < 2:
            print("Usage: job.py [--release <path>]", file=sys.stderr)
            return 2
        release_arg = Path(argv[1]).resolve()
    env = dict(os.environ)
    try:
        require_environment(env)
        storage = Path(env["STORAGE_DIR"])
        scratch = Path(env["SCRATCHDIR"])
        spec_path = Path(env["RUN_SPEC"])
        python = job_python(storage)
        spec = read_spec(spec_path)
        release = release_arg or Path(spec["source"]).resolve()
        if release_arg and Path(spec["source"]).resolve() != release_arg:
            raise JobError(
                f"Spec points at {spec['source']} but the launcher came from {release_arg}"
            )
        if not (release / PIPELINE_ENTRY).is_file():
            raise JobError(f"Release is incomplete or gone: {release}")
    except JobError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results_dir, out_dir = results_layout(storage, scratch, spec["name"])
    print("================ DrugEx job ================")
    for key, value in header_rows(env, spec, release, results_dir.name, results_dir):
        print(f"{key:<11}: {value}")
    print("============================================")

    install_signal_handlers()
    release_copy = scratch / "DrugEx"
    status = 1
    try:
        prepare_run(release, release_copy, spec_path, out_dir)
        status = run_pipeline(
            pipeline_command(python, out_dir, spec),
            cwd=release_copy,
            env=job_environment(env, release_copy, spec),
            log_path=out_dir / "execution.log",
        )
        if status == 0:
            print(f"Pipeline finished for {results_dir.name}")
    except _SignalExit as exc:
        status = int(exc.code or 143)
    except Exception:  # noqa: BLE001 - a worker traceback must not skip the copy-back
        traceback.print_exc()
        status = 1
    finally:
        # A second signal during transfer must not interrupt the only copy of the results.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        status = finalize(out_dir, results_dir, scratch, status)
    return status


if __name__ == "__main__":
    with suppress(BrokenPipeError):
        sys.exit(main())
