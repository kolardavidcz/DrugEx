"""Run this pipeline on MetaCentrum: ``python -m _david.metacentrum.submit <command>``.

    check                 ssh, the storage venv, the staged release, the queue
    sync                  stage HEAD on storage without submitting
    submit --name <name>  stage HEAD, upload the spec, qsub  [-- name=value ...]
    status                the user's queue
    logs <name>           the PBS stdout of the newest job of that name
    cancel <name|id> ...  qdel
    pull                  copy finished runs back into _david/output

One job at a time; the planned sweep lives in ``campaign.py`` and calls into this. A job
runs ``git archive HEAD``, so uncommitted work does not exist for it and a bug fix
travels through a commit, never through an edit on storage.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from _david.config import PROJECT_ROOT, Config

#: ssh alias and account. Never switch aliases: the Kerberos ticket is bound to this one.
PACKAGE = "_david"

HOST = "meta"
USER = "kolardavidcz"

#: Persistent tree on storage. ``.venv`` (~5.7 GB) lives here and is never deleted.
STORAGE = f"/storage/brno2/brno2/{USER}/DrugEx"

#: Settings the job owns; a delta that set them would fight the copy-back contract.
#: Settings the job owns. A delta overriding one of these would undo the thing the job
#: computed it for: n_jobs comes from allocated_cpus (the PBS allocation, not the
#: node), so `--set n_jobs=-1` would reintroduce the oversubscription it prevents.
RESERVED = ("output_root", "n_jobs")

#: ``spec=8.0`` keeps the job off the slowest nodes: ROCS is CPU-bound, so a slow node
#: costs hours. Everything else is the shape of a default CPU ROCS run.
CPU_FLOOR = "8.0"


@dataclass(frozen=True)
class Job:
    name: str
    args: tuple[str, ...] = ()
    ncpus: int = 16
    mem: int = 32  # GB
    scratch: int = 15  # GB
    walltime: str = "24:00:00"
    queue: str | None = None  # None routes by walltime, which is the only way that works

    @property
    def resources(self) -> str:
        return (
            f"select=1:ncpus={self.ncpus}:mem={self.mem}gb:"
            f"scratch_local={self.scratch}gb:spec={CPU_FLOOR}"
        )


def _run(command: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
    """Run a local command, raising with its own stderr rather than a bare exit code."""
    result = subprocess.run(command, input=stdin, capture_output=True)
    if result.returncode != 0:
        output = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        raise SystemExit(f"{command[0]} failed ({result.returncode}): {output}")
    return result


def remote(command: str, stdin: bytes | None = None) -> str:
    """Run one command on the cluster, turning an expired ticket into the fix for it."""
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


#: Everything a job needs to exist inside the release. A file missing from HEAD makes the
#: job fail on the cluster, minutes of queue time later, with an import error.
REQUIRED = (
    "__init__.py",
    "config.py",
    "provenance.py",
    "run.py",
    "scorers/__init__.py",
    "scorers/conformers.py",
    "scorers/rocs.py",
    "scorers/objectives.py",
    "training/__init__.py",
    "training/explorer.py",
    "analysis/__init__.py",
    "score_arms.py",
    "analysis/__main__.py",
    "analysis/report.py",
    "analysis/figures.py",
    "metacentrum/__init__.py",
    "metacentrum/job.py",
    "metacentrum/pbs_launcher.sh",
)


def _tracked_at_head() -> set[str]:
    """Paths under this package that ``git archive HEAD`` would actually contain."""
    listing = _run(
        ["git", "-C", str(PROJECT_ROOT), "ls-tree", "-r", "--name-only", "HEAD", "--", PACKAGE]
    ).stdout.decode()
    return {line.strip() for line in listing.splitlines() if line.strip()}


def validate_deltas(deltas: list[str]) -> None:
    """Parse the overrides here, so a bad one fails now rather than on the worker node."""
    from _david.run import config_from_args
    from _david.scorers.objectives import reward_scheme

    try:
        # Resolved, not just parsed: `scheme` is a string field, so a typo survives
        # config_from_args and would surface on the worker node after the queue wait.
        reward_scheme(config_from_args(list(deltas)).scheme)
    except SystemExit as error:
        raise SystemExit(f"delta rejected locally, nothing submitted: {error}") from None
    except (ValueError, TypeError) as error:
        # _coerce raises ValueError on a bad value and Config.__post_init__ on an
        # impossible combination; neither is a SystemExit, and both would otherwise
        # surface only on the worker node after the queue wait.
        raise SystemExit(
            f"delta rejected locally, nothing submitted: {type(error).__name__}: {error}"
        ) from None


def head_revision() -> str:
    """HEAD, refusing to stage a release that is not what the working tree holds.

    Two different failures, both of which produce a release that silently lacks the code:
    a tracked file modified since the commit, and a file never committed at all. The
    second is the dangerous one -- ``git status --untracked-files=no`` is quiet about it,
    so an entirely untracked package passes the check and ships an empty release.
    """
    dirty = _run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"]
    ).stdout.decode()
    if dirty:
        raise SystemExit(
            "tracked files are modified. A job runs `git archive HEAD`, so these changes "
            "would not be in it; commit them first:\n" + dirty
        )

    tracked = _tracked_at_head()
    missing = [name for name in REQUIRED if f"{PACKAGE}/{name}" not in tracked]
    if missing:
        raise SystemExit(
            "these files are not in HEAD, so `git archive HEAD` would stage a release "
            "without them and the job would fail on the cluster:\n  "
            + "\n  ".join(f"{PACKAGE}/{name}" for name in missing)
            + "\n\ncommit them first."
        )

    # Recursive: the package has subpackages now, and a top-level glob would be silent
    # about a new untracked module inside one of them -- which is exactly the file that
    # would be missing from the release.
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
            "warning: these files exist locally but not in HEAD, so they will not be in "
            "the release:\n  " + "\n  ".join(untracked)
        )
    return _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]).stdout.decode().strip()


def sync(revision: str | None = None) -> str:
    """Stage ``git archive HEAD`` as ``$STORAGE/releases/<sha>`` and repoint ``current``.

    A release is immutable: nothing is ever written into a tree a job might be running
    from, so a job always keeps the code it started with. The pretrained checkpoint is
    rsynced into the release at the same path the local tree holds it, so the staged
    release is self-contained and ``Config.pretrained()`` resolves without a cluster knob.
    """
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
    """Upload the spec. It is the record of what the job was actually asked to do."""
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
    """Stage, upload and queue one job; print what was requested before qsub runs.

    ``revision`` is for a campaign: staging is identical for every job of one sweep, and
    ``sync`` rsyncs a 150 MB checkpoint, so it is done once and passed in. Omitted, this
    stages HEAD itself and behaves exactly as a single submission always has.
    """
    for delta in job.args:
        for reserved in RESERVED:
            if delta.startswith(f"{reserved}=") or delta == reserved:
                raise SystemExit(
                    f"{reserved} is owned by the job (it is the scratch directory the "
                    "results are copied back from); drop it from the deltas"
                )
    revision = revision or sync()
    spec = stage_spec(job, revision)
    release = f"{STORAGE}/releases/{revision}"
    command = ["qsub", "-N", job.name]
    if job.queue:
        # Only when the user asks. Frontends cannot submit into an execution queue:
        # q_2h..q_2d are from_route_only, and p2e_1d refuses the host outright.
        command += ["-q", job.queue]
    command += [
        "-l",
        job.resources,
        "-l",
        f"walltime={job.walltime}",
        "-v",
        f"RUN_SPEC={spec},STORAGE_DIR={STORAGE},RELEASE_DIR={release}",
        f"{release}/_david/metacentrum/pbs_launcher.sh",
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
    revision = _run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]).stdout.decode().strip()
    staged = remote(f"test -d {STORAGE}/releases/{revision} && echo yes || echo no").strip()
    print(f"HEAD {revision[:12]} staged: {staged}")
    print(remote(f"qstat -u {USER}").strip() or "queue: nothing of yours is queued")


def logs(name: str, lines: int, follow: bool) -> None:
    """PBS stdout of the newest job with that name. ``$HOME`` is where PBS leaves it."""
    path = remote(f"ls -t $HOME/{shlex.quote(name)}.o* 2>/dev/null | head -1").strip()
    if not path:
        raise SystemExit(f"no PBS log for {name!r} yet; `status` shows whether it has started")
    if follow:
        subprocess.run(["ssh", "-t", HOST, f"tail -n {lines} -f {shlex.quote(path)}"])
    else:
        print(remote(f"tail -n {lines} {shlex.quote(path)}"))


def pull(name: str | None) -> None:
    """Copy finished runs back. MetaCentrum to here only; nothing is pushed into output."""
    # A run directory is `<YYYYmmdd_HHMM>_<tag>`, so the job name is a suffix, never the
    # whole name. Matching it as a prefix pattern is what makes `pull --name foo` find
    # anything at all.
    destination = PROJECT_ROOT / "_david" / "output"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--info=progress2"]
    if name:
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
    run.add_argument("--mem", type=int, default=Job.mem, help="GB")
    run.add_argument("--scratch", type=int, default=Job.scratch, help="GB")
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

    kill = sub.add_parser("cancel")
    kill.add_argument("targets", nargs="+")

    get = sub.add_parser("pull")
    get.add_argument("--name", default=None, help="one run directory instead of all")

    args = parser.parse_args(argv)

    if args.command == "check":
        check()
    elif args.command == "sync":
        sync()
    elif args.command == "status":
        print(remote(f"qstat -u {USER}").strip() or "nothing of yours is queued")
    elif args.command == "submit":
        # Each delta is a bare `name=value`; it is this that adds the `--set` flag. A
        # delta that already carries one would be forwarded as `--set --set name=value`
        # and die on the worker node after the queue wait, so reject it here instead.
        bad = [d for d in args.deltas if d.startswith("-") or "=" not in d]
        if bad:
            raise SystemExit(
                "deltas are bare NAME=VALUE pairs; this adds the --set itself. Offending: "
                + " ".join(bad)
            )
        validate_deltas(args.deltas)
        deltas = [part for delta in args.deltas for part in ("--set", delta)]
        submit(
            Job(
                name=args.name,
                args=tuple(deltas),
                ncpus=args.ncpus,
                mem=args.mem,
                scratch=args.scratch,
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
