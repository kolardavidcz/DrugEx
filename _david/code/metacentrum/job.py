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

ENTRY_POINT = Path("_david") / "code" / "run.py"

SPEC_KEYS = ("name", "args", "source", "revision")
REQUIRED_ENV = ("PBS_JOBID", "SCRATCHDIR", "STORAGE_DIR", "RUN_SPEC")

#: The heaviest measured cell reached ~40 GB with 16 scoring workers (~2.5 GB each) and was
#: OOM-killed. The job derives workers from the *memory request* with this budget and a
#: reserve, so a 32 GB allocation runs 12 workers rather than 16 and stays below its cgroup.
MEM_GB_PER_WORKER = 2.5
MEMORY_RESERVE_GB = 2.0

THREAD_ENV = {
    "MPLBACKEND": "Agg",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "DRUGEX_MP_CONTEXT": "forkserver",
    "DRUGEX_SCORE_MOL_TIMEOUT": "120",
}

TEMP_ENV_KEYS = ("TMPDIR", "TMP", "TEMP")


class JobError(RuntimeError):
    pass


class _SignalExit(SystemExit):
    pass


def read_spec(path: Path) -> dict:
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise JobError(f"job spec is gone: {path}") from error
    except json.JSONDecodeError as error:
        raise JobError(f"job spec is not valid JSON: {path} ({error})") from error
    missing = [key for key in SPEC_KEYS if key not in spec]
    if missing:
        raise JobError(f"job spec {path} is missing {', '.join(missing)}")
    if not isinstance(spec["args"], list) or not all(isinstance(a, str) for a in spec["args"]):
        raise JobError(f"job spec {path} has a non-list 'args'")
    return spec


def staged_job(env: dict[str, str], release_arg: Path | None) -> tuple[dict, Path]:
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        raise JobError(
            f"missing PBS environment: {', '.join(missing)}. Submit through "
            "`python -m _david.code.metacentrum.submit submit`."
        )
    python = venv_python(Path(env["STORAGE_DIR"]))
    if not (python.is_file() and os.access(python, os.X_OK)):
        raise JobError(f"missing validated environment: {python}")
    spec = read_spec(Path(env["RUN_SPEC"]))
    declared = Path(spec["source"]).resolve()
    if release_arg and declared != release_arg:
        raise JobError(f"spec points at {declared} but the launcher came from {release_arg}")
    release = release_arg or declared
    if not (release / ENTRY_POINT).is_file():
        raise JobError(f"release is incomplete or gone: {release}")
    return spec, release


def venv_python(storage: Path) -> Path:
    return storage / ".venv" / "bin" / "python"


def allocated_cpus(env: dict[str, str]) -> int:
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
    merged = dict(env) | THREAD_ENV
    if revision:
        merged["DRUGEX_SOURCE_REVISION"] = revision
    for key in TEMP_ENV_KEYS:
        merged[key] = str(temp_root)
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = f"{release_copy}:{existing}" if existing else str(release_copy)
    return merged


def workers_for(spec: dict, cpus: int) -> int:
    """Scoring workers that fit the memory request, never more than the allocated cores.

    PBS reserves ``mem`` for the whole job but does not grow it while the job runs. This
    cap keeps the scoring pool below that reservation: 32 GB -> 12 workers, 64 GB -> all
    allocated cores. Old specs without ``mem_gb`` keep the previous all-cores behaviour.
    """
    memory = spec.get("mem_gb")
    if not isinstance(memory, (int, float)) or memory <= MEMORY_RESERVE_GB:
        return cpus
    fit = int((float(memory) - MEMORY_RESERVE_GB) // MEM_GB_PER_WORKER)
    return max(1, min(cpus, fit))


def run_command(python: Path, output_root: Path, cpus: int, spec: dict) -> list[str]:
    return [
        str(python),
        "-u",
        "-m",
        "_david.code.run",
        "--tag",
        spec["name"],
        "--set",
        f"output_root={output_root}",
        "--set",
        f"n_jobs={workers_for(spec, cpus)}",
        *spec["args"],
    ]


def free_destination(root: Path, name: str) -> Path:
    candidate, suffix = root / name, 2
    while candidate.exists():
        candidate = root / f"{name}_{suffix}"
        suffix += 1
    return candidate


def copy_results(
    output_root: Path, storage_output: Path, submission: Path | None = None
) -> list[Path]:
    if not output_root.is_dir():
        return []
    directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    if submission and submission.is_file():
        for run_dir in directories:
            shutil.copy2(submission, run_dir / "submission.json")
    storage_output.mkdir(parents=True, exist_ok=True)
    copied = []
    for run_dir in directories:
        destination = free_destination(storage_output, run_dir.name)
        shutil.copytree(run_dir, destination, symlinks=True)
        copied.append(destination)
    return copied


def clean_scratch(scratch: Path) -> None:
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


def install_signal_handlers() -> None:
    def handler(signum: int, _frame) -> None:
        raise _SignalExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, handler)


def _banner(
    env: dict[str, str],
    spec: dict,
    cpus: int,
    workers: int,
    scratch: Path,
    release: Path,
    results: Path,
) -> None:
    rows = (
        ("job", env.get("PBS_JOBID", "?")),
        ("name", spec["name"]),
        ("node", os.uname().nodename),
        ("queue", env.get("PBS_QUEUE", "unknown")),
        ("cpus", cpus),
        ("workers", workers),
        ("memory", f"{spec.get('mem_gb', '?')} GB"),
        ("scratch", scratch),
        ("release", release),
        ("revision", spec["revision"]),
        ("deltas", " ".join(spec["args"]) or "(none)"),
        ("results", results),
    )
    print("================ DrugEx v2 job ================")
    for key, value in rows:
        print(f"{key:<9}: {value}")
    print("===============================================")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    release_arg = Path(argv[1]).resolve() if argv[:1] == ["--release"] and len(argv) > 1 else None
    env = dict(os.environ)
    try:
        spec, release = staged_job(env, release_arg)
    except JobError as error:
        print(str(error), file=sys.stderr)
        return 1

    storage, scratch = Path(env["STORAGE_DIR"]), Path(env["SCRATCHDIR"])
    output_root, storage_output = scratch / "output", storage / "_david" / "output"
    release_copy, temp_root = scratch / "DrugEx", scratch / "tmp"
    cpus = allocated_cpus(env)
    workers = workers_for(spec, cpus)
    _banner(env, spec, cpus, workers, scratch, release, storage_output)

    install_signal_handlers()
    status = 1
    submission = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        if release_copy.exists():
            shutil.rmtree(release_copy)
        shutil.copytree(release, release_copy, symlinks=True)
        submission = Path(env["RUN_SPEC"])

        process = subprocess.run(
            run_command(venv_python(storage), output_root, cpus, spec),
            cwd=release_copy,
            env=job_environment(env, release_copy, temp_root, spec.get("revision")),
        )
        returncode = process.returncode
        status = 128 - returncode if returncode < 0 else returncode
    except _SignalExit as error:
        print(f"signalled, copying partial results back (exit {error.code})", file=sys.stderr)
        status = int(error.code or 143)
    except Exception:
        traceback.print_exc()
        status = 1
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        try:
            for destination in copy_results(output_root, storage_output, submission):
                print(f"results copied to {destination}")
        except OSError as error:
            print(f"result transfer FAILED ({error}); scratch kept at {scratch}", file=sys.stderr)
            return 1
        clean_scratch(scratch)
    return status


if __name__ == "__main__":
    with suppress(BrokenPipeError):
        sys.exit(main())
