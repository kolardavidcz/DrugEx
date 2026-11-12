"""Run this pipeline on MetaCentrum: ``python -m _david.code.metacentrum.submit <command>``.

    check                 ssh, the storage venv, the staged release, the queue
    sync                  stage HEAD on storage without submitting
    submit --name <name>  stage HEAD, upload the spec, qsub  [-- name=value ...]
    status                the user's queue
    logs <name>           the PBS stdout of the newest job of that name
    cancel <name|id> ...  qdel
    pull                  copy finished runs back into _david/output

One job at a time; the planned sweep lives in ``campaign.py`` and calls into this. A job runs
``git archive HEAD``, so a fix travels through a commit, never through an edit on storage.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from _david.code.config import PROJECT_ROOT, Config

HOST = "meta"
USER = "kolardavidcz"
PACKAGE = "_david/code"
STORAGE = f"/storage/brno2/brno2/{USER}/DrugEx"
RESERVED = ("output_root", "n_jobs")
CPU_FLOOR = "8.0"
REQUIRED = (
    "__init__.py",
    "config.py",
    "training.py",
    "run.py",
    "archive/__init__.py",
    "archive/explorer.py",
    "reconstruct/__init__.py",
    "reconstruct/provenance.py",
    "reconstruct/score_arms.py",
    "scorers/__init__.py",
    "scorers/conformers.py",
    "scorers/rocs.py",
    "scorers/objectives.py",
    "scorers/utils.py",
    "analysis/__init__.py",
    "analysis/__main__.py",
    "analysis/report.py",
    "analysis/figures.py",
    "metacentrum/__init__.py",
    "metacentrum/job.py",
    "metacentrum/pbs_launcher.sh",
)


@dataclass(frozen=True)
class Job:
    name: str
    args: tuple[str, ...] = ()
    ncpus: int = 16
    mem_gb: int = 64
    scratch_gb: int = 15
    walltime: str = "24:00:00"
    queue: str | None = None

    @property
    def resources(self) -> str:
        return (
            f"select=1:ncpus={self.ncpus}:mem={self.mem_gb}gb:"
            f"scratch_local={self.scratch_gb}gb:spec={CPU_FLOOR}"
        )


def _run(command: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, input=stdin, capture_output=True)
    if result.returncode != 0:
        output = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise SystemExit(f"{command[0]} failed ({result.returncode}): {output}")
    return result


def _git(*args: str) -> str:
    return _run(["git", "-C", str(PROJECT_ROOT), *args]).stdout.decode()


def remote(command: str, stdin: bytes | None = None) -> str:
    """An expired ticket comes back as the fix for it."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", HOST, command], input=stdin, capture_output=True
    )
    output = (result.stdout + result.stderr).decode("utf-8", "replace")
    if result.returncode != 0:
        if any(word in output.lower() for word in ("kerberos", "credential", "permission denied")):
            raise SystemExit(
                f"ssh {HOST} was refused. The ticket is interactive and needs 2FA, so run "
                f"it yourself:\n    ssh {HOST} kinit {USER}@META\n\n{output.strip()}"
            )
        raise SystemExit(f"remote command failed on {HOST}: {command}\n{output.strip()}")
    return result.stdout.decode("utf-8", "replace")


def validate_deltas(deltas: list[str]) -> None:
    """Parsed here, so a bad override fails now and not after the queue wait."""
    from _david.code.run import config_from_args
    from _david.code.scorers.objectives import reward_scheme

    malformed = [delta for delta in deltas if delta.startswith("-") or "=" not in delta]
    if malformed:
        raise SystemExit(
            "deltas are bare NAME=VALUE pairs; the --set is added here. Offending: "
            + " ".join(malformed)
        )
    try:
        reward_scheme(config_from_args(list(deltas)).scheme)  # a string field: parsing is not enough
    except SystemExit as error:
        raise SystemExit(f"delta rejected locally, nothing submitted: {error}") from None
    except (ValueError, TypeError) as error:
        raise SystemExit(
            f"delta rejected locally, nothing submitted: {type(error).__name__}: {error}"
        ) from None


def head_revision() -> str:
    """HEAD, refusing to stage a release that is not what the working tree holds."""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(
            "tracked files are modified. A job runs `git archive HEAD`, so these changes "
            "would not be in it; commit them first:\n" + dirty
        )

    listing = _git("ls-tree", "-r", "--name-only", "HEAD", "--", PACKAGE)
    tracked = {line.strip() for line in listing.splitlines() if line.strip()}
    missing = [name for name in REQUIRED if f"{PACKAGE}/{name}" not in tracked]
    if missing:
        raise SystemExit(
            "these files are not in HEAD, so `git archive HEAD` would stage a release "
            "without them and the job would fail on the cluster:\n  "
            + "\n  ".join(f"{PACKAGE}/{name}" for name in missing)
            + "\n\ncommit them first."
        )

    local = {
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / PACKAGE).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh", ".toml", ".md"}
        and "__pycache__" not in path.parts
    }
    untracked = sorted(local - tracked)
    if untracked:
        print(
            "warning: these files exist locally but not in HEAD, so they will not be in the "
            "release:\n  " + "\n  ".join(untracked)
        )
    return _git("rev-parse", "HEAD").strip()


def sync(revision: str | None = None) -> str:
    """Stage an immutable ``$STORAGE/releases/<sha>`` and repoint ``current``."""
    revision = revision or head_revision()
    release = f"{STORAGE}/releases/{revision}"
    archive = _run(["git", "-C", str(PROJECT_ROOT), "archive", revision]).stdout
    remote(f"mkdir -p {shlex.quote(release)} && tar -xf - -C {shlex.quote(release)}", stdin=archive)

    pkg, vocab = Config().pretrained()
    relative = pkg.parent.relative_to(PROJECT_ROOT)
    remote(f"mkdir -p {shlex.quote(f'{release}/{relative}')}")
    _run(["rsync", "-a", str(pkg), str(vocab), f"{HOST}:{release}/{relative}/"])

    remote(f"ln -sfn {shlex.quote(release)} {shlex.quote(STORAGE + '/current')}")
    print(f"release : {release}")
    return revision


def stage_spec(job: Job, revision: str) -> str:
    directory = f"{STORAGE}/jobs/{job.name}"
    path = f"{directory}/spec.json"
    spec = {
        "name": job.name,
        "args": list(job.args),
        "source": f"{STORAGE}/releases/{revision}",
        "revision": revision,
    }
    payload = json.dumps(spec, indent=2).encode()
    remote(f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}", stdin=payload)
    return path


def submit(job: Job, revision: str | None = None) -> str:
    """Stage, upload and queue one job; a campaign stages once and passes ``revision`` in."""
    for delta in job.args:
        name = delta.split("=", 1)[0]
        if name in RESERVED:
            raise SystemExit(
                f"{name} is owned by the job (it is the scratch directory the results are "
                "copied back from); drop it from the deltas"
            )
    revision = revision or sync()
    spec = stage_spec(job, revision)
    release = f"{STORAGE}/releases/{revision}"
    command = ["qsub", "-N", job.name]
    if job.queue:
        command += ["-q", job.queue]  # frontends cannot submit into an execution queue
    command += [
        "-l",
        job.resources,
        "-l",
        f"walltime={job.walltime}",
        "-v",
        f"RUN_SPEC={spec},STORAGE_DIR={STORAGE},RELEASE_DIR={release}",
        f"{release}/{PACKAGE}/metacentrum/pbs_launcher.sh",
    ]
    print(f"queue    : {job.queue or 'routed by walltime (' + job.walltime + ')'}")
    print(f"resources: {job.resources}, walltime={job.walltime}")
    print(f"deltas   : {' '.join(job.args) or '(none)'}")
    job_id = remote(shlex.join(command)).strip()
    print(f"\nsubmitted {job.name}: {job_id}")
    return job_id


def check() -> None:
    """Everything a submission depends on, in the order it would fail."""
    print(remote("hostname").strip(), "reachable")
    venv = f"{STORAGE}/.venv/bin/python"
    print("storage venv:", remote(f"test -x {venv} && echo present || echo MISSING").strip())
    revision = _git("rev-parse", "HEAD").strip()
    staged = remote(f"test -d {STORAGE}/releases/{revision} && echo yes || echo no").strip()
    print(f"HEAD {revision[:12]} staged: {staged}")
    print(queue_status() or "queue: nothing of yours is queued")


def queue_status() -> str:
    return remote(f"qstat -u {USER}").strip()


def logs(name: str, lines: int, follow: bool) -> None:
    path = remote(f"ls -t $HOME/{shlex.quote(name)}.o* 2>/dev/null | head -1").strip()
    if not path:
        raise SystemExit(f"no PBS log for {name!r} yet; `status` shows whether it has started")
    if follow:
        subprocess.run(["ssh", "-t", HOST, f"tail -n {lines} -f {shlex.quote(path)}"])
    else:
        print(remote(f"tail -n {lines} {shlex.quote(path)}"))


def pull(name: str | None) -> None:
    """MetaCentrum to here only; nothing is pushed the other way."""
    destination = PROJECT_ROOT / "_david" / "output"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--info=progress2"]
    if name:
        # A run directory is `<YYYYmmdd_HHMM>_<tag>`: the job name is only a suffix.
        command += ["--include=*/", f"--include=*_{name}/***", "--exclude=*"]
    command += [f"{HOST}:{STORAGE}/_david/output/", f"{destination}/"]
    subprocess.run(command, check=True)
    print(f"pulled into {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("sync")
    sub.add_parser("status")

    run = sub.add_parser("submit", help="stage HEAD and queue one job")
    run.add_argument("--name", required=True, help="PBS job name and the run's tag")
    run.add_argument("--ncpus", type=int, default=Job.ncpus)
    run.add_argument("--mem", type=int, default=Job.mem_gb, help="GB")
    run.add_argument("--scratch", type=int, default=Job.scratch_gb, help="GB")
    run.add_argument(
        "--walltime",
        default=Job.walltime,
        help="compute budget, not a queue knob: it is what routes the queue",
    )
    run.add_argument("--queue", default=None, help="only when you know you need one")
    run.add_argument(
        "deltas",
        nargs="*",
        metavar="NAME=VALUE",
        help="Config overrides, each forwarded as `--set NAME=VALUE`",
    )

    log = sub.add_parser("logs")
    log.add_argument("name")
    log.add_argument("-n", "--lines", type=int, default=60)
    log.add_argument("-f", "--follow", action="store_true")

    sub.add_parser("cancel").add_argument("targets", nargs="+")
    sub.add_parser("pull").add_argument(
        "--name", default=None, help="one run directory instead of all"
    )

    args = parser.parse_args(argv)
    if args.command == "check":
        check()
    elif args.command == "sync":
        sync()
    elif args.command == "status":
        print(queue_status() or "nothing of yours is queued")
    elif args.command == "submit":
        validate_deltas(args.deltas)
        submit(
            Job(
                name=args.name,
                args=tuple(part for delta in args.deltas for part in ("--set", delta)),
                ncpus=args.ncpus,
                mem_gb=args.mem,
                scratch_gb=args.scratch,
                walltime=args.walltime,
                queue=args.queue,
            )
        )
    elif args.command == "logs":
        logs(args.name, args.lines, args.follow)
    elif args.command == "cancel":
        for target in args.targets:
            print(remote(f"qdel {shlex.quote(target)}").strip() or f"cancelled {target}")
    elif args.command == "pull":
        pull(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
