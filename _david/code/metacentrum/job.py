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

#: What a spec may ask the job to run. ``train`` is the default so a spec written before
#: this key existed still means what it meant.
MODES = ("train", "rebuild")

#: Memory a scoring worker keeps alive scales with the conformers it embeds per molecule.
#: Fitted from the PBS peaks of finished jobs (``memreport``): 10 conformers needed
#: 1.0 GB/worker, 20 needed 2.0, 30 needed 2.8. The margin covers molecule-size variation
#: inside a bucket; the reserve is the parent process. A job with no recorded shape is
#: budgeted as ``DEFAULT_MAX_CONFORMERS``.
MEM_GB_PER_CONFORMER = 0.1
MEMORY_SAFETY = 1.15
MEM_GB_PER_WORKER_FLOOR = 1.0
MEMORY_RESERVE_GB = 2.0
#: ``Config.max_conformers`` default. Duplicated on purpose: importing ``Config`` would
#: pull torch into every job body, including a rebuild that only scores molecules. The test
#: suite asserts the two stay in step.
DEFAULT_MAX_CONFORMERS = 20

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
    spec.setdefault("mode", "train")
    if spec["mode"] not in MODES:
        raise JobError(f"job spec {path} has mode {spec['mode']!r}, not one of {MODES}")
    if spec["mode"] == "rebuild":
        runs = spec.get("runs")
        if not isinstance(runs, list) or not runs or not all(isinstance(r, str) for r in runs):
            raise JobError(f"job spec {path} is a rebuild with no run directories")
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


def _delta(spec: dict, name: str) -> str | None:
    """The value a spec overrides ``name`` with, if it does (last one wins, as in ``--set``)."""
    value = None
    for argument in spec.get("args") or []:
        key, separator, raw = str(argument).partition("=")
        if separator and key == name and raw:
            value = raw
    return value


def spec_max_conformers(spec: dict) -> int:
    """The conformer budget a train spec asks for, or the default it inherits."""
    raw = _delta(spec, "max_conformers")
    if raw is None:
        return DEFAULT_MAX_CONFORMERS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_CONFORMERS


def gb_per_worker(conformers: int) -> float:
    """Memory to budget per scoring worker for this conformer count."""
    return (
        max(MEM_GB_PER_WORKER_FLOOR, MEM_GB_PER_CONFORMER * conformers) * MEMORY_SAFETY
    )


def workers_for(spec: dict, cpus: int, *, conformers: int | None = None) -> int:
    """Scoring workers that fit the memory request, never more than the allocated cores.

    PBS reserves ``mem`` for the whole job but does not grow it while the job runs. This
    cap keeps the scoring pool below that reservation: the budget per worker follows the
    conformer count of the run, so a 30-conformer job gets fewer workers than a 10-conformer
    one out of the same 32 GB. ``conformers`` overrides the spec, because a rebuild job
    replays a run whose shape is recorded in that run's own ``config.json``. Old specs
    without ``mem_gb`` keep the previous all-cores behaviour.
    """
    memory = spec.get("mem_gb")
    if not isinstance(memory, (int, float)) or memory <= MEMORY_RESERVE_GB:
        return cpus
    budget = gb_per_worker(conformers or spec_max_conformers(spec))
    fit = int((float(memory) - MEMORY_RESERVE_GB) // budget)
    return max(1, min(cpus, fit))


def run_max_conformers(storage_output: Path, run: str) -> int | None:
    """The conformer count a run recorded, for sizing the pool that rebuilds its arm."""
    try:
        config = json.loads((storage_output / run / "config.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    value = config.get("max_conformers")
    return value if isinstance(value, int) and value > 0 else None


def commands_for(
    python: Path, output_root: Path, storage_output: Path, cpus: int, spec: dict
) -> list[list[str]]:
    """One argv per unit of work: a train job has one, a rebuild has one per run.

    A rebuild reads the run directory where it lives on storage and writes the missing arm
    back into it, so a killed job leaves the run unchanged rather than half-copied. Each
    rebuilt run gets the worker count its own conformer budget allows, because one PBS job
    carries runs of different shapes.
    """
    if spec.get("mode") == "rebuild":
        commands = []
        for run in spec["runs"]:
            workers = workers_for(
                spec, cpus, conformers=run_max_conformers(storage_output, run)
            )
            commands.append(
                [
                    str(python),
                    "-u",
                    "-m",
                    "_david.code.reconstruct.score_arms",
                    str(storage_output / run),
                    "--n-jobs",
                    str(workers),
                ]
            )
        return commands
    workers = workers_for(spec, cpus)
    return [
        [
            str(python),
            "-u",
            "-m",
            "_david.code.run",
            "--tag",
            spec["name"],
            "--set",
            f"output_root={output_root}",
            "--set",
            f"n_jobs={workers}",
            *spec["args"],
        ]
    ]


def workers_label(spec: dict, storage_output: Path, cpus: int) -> str:
    """One number for a train job; a range for a rebuild, whose runs differ in shape."""
    if spec.get("mode") != "rebuild":
        return str(workers_for(spec, cpus))
    counts = [
        workers_for(spec, cpus, conformers=run_max_conformers(storage_output, run))
        for run in spec["runs"]
    ]
    low, high = min(counts), max(counts)
    return str(low) if low == high else f"{low}..{high} per run"


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
    rows = [
        ("job", env.get("PBS_JOBID", "?")),
        ("name", spec["name"]),
        ("mode", spec.get("mode", "train")),
        ("node", os.uname().nodename),
        ("queue", env.get("PBS_QUEUE", "unknown")),
        ("cpus", cpus),
        ("workers", workers),
        ("memory", f"{spec.get('mem_gb', '?')} GB"),
        ("scratch", scratch),
        ("release", release),
        ("revision", spec["revision"]),
    ]
    if spec.get("mode") == "rebuild":
        rows.append(("runs", f"{len(spec['runs'])}: {' '.join(spec['runs'])}"))
    else:
        rows.append(("deltas", " ".join(spec["args"]) or "(none)"))
        rows.append(("results", results))
    print("================ DrugEx v2 job ================")
    for key, value in rows:
        print(f"{key:<9}: {value}")
    # Flushed: stdout is buffered while logging writes to stderr, so without this the
    # banner appears after the first setup lines in a PBS log.
    print("===============================================", flush=True)


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
    workers = workers_label(spec, storage_output, cpus)
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

        commands = commands_for(
            venv_python(storage), output_root, storage_output, cpus, spec
        )
        env_for_job = job_environment(env, release_copy, temp_root, spec.get("revision"))
        status, failed = 0, []
        active: subprocess.Popen | None = None
        try:
            for command in commands:
                # One unit of work failing does not discard the ones after it: a rebuild job
                # carries several independent runs, and each is useful on its own.
                active = subprocess.Popen(command, cwd=release_copy, env=env_for_job)
                returncode = active.wait()
                active = None
                code = 128 - returncode if returncode < 0 else returncode
                if code:
                    unit = command[-1] if spec.get("mode") == "rebuild" else spec["name"]
                    failed.append((unit, code))
                    print(f"FAILED ({code}): {' '.join(command)}", file=sys.stderr)
                status = status or code
        except _SignalExit:
            if active is not None and active.poll() is None:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    signal.signal(signum, signal.SIG_IGN)
                with suppress(OSError):
                    active.terminate()
                try:
                    active.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        active.kill()
                    with suppress(OSError):
                        active.wait()
            raise
        if failed:
            print(f"{len(failed)} of {len(commands)} unit(s) failed: {failed}", file=sys.stderr)
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
