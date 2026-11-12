"""Run this pipeline on MetaCentrum: ``python -m _david.code.metacentrum.submit <command>``.

    check                     ssh, the storage venv, the staged release, the queue
    sync                      stage HEAD on storage without submitting
    submit --name <name>      stage, upload the spec, qsub  [-- name=value ...]
    status                    the user's queue, from long-format qstat
    watch [-i 60]             live epochs of running jobs, read on the login node
    logs <name> [-f]          the PBS stdout of the job, not merely of its name
    cancel <name|id> ...      qdel, resolving names through qselect
    pull [--name <tag>]       copy finished runs back into _david/output

One job at a time; the planned sweep lives in ``campaign.py`` and calls into this. A job runs
``git archive <revision>``, so a fix travels through a commit, never through an edit on
storage. Resource defaults are the ones measured for this pipeline; ``--revision`` reruns a
release that is already staged, and ``--no-sync`` is for when you know it is there.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from _david.code.config import PROJECT_ROOT, Config

HOST = "meta"
USER = "kolardavidcz"
PACKAGE = "_david/code"
STORAGE = f"/storage/brno2/brno2/{USER}/DrugEx"
RESERVED = ("output_root", "n_jobs")
CPU_FLOOR = "8.0"

#: Memory a scoring worker keeps alive. Doubling the pool without doubling memory is an
#: OOM kill after the queue wait, so a job refuses a starved core/memory pairing.
MEM_GB_PER_CORE = 2

#: Default allocation: fast to place (32 GB queues roughly 3x faster than 64 GB on our
#: account). The job body caps scoring workers to fit it; a run measured above it asks 64.
DEFAULT_MEM_GB = 32

#: The default and only lane used for training: 24 h routes to ``q_1d``. A training run
#: needs the whole day; shrinking the walltime to reach a shallower queue truncates it.
DEFAULT_WALLTIME = "24:00:00"
DEFAULT_QUEUE = "q_1d"

#: Walltime ceiling (hours) -> execution queue chosen by the ``default`` router. Frontends
#: cannot submit to an execution queue directly (``q_2h``... are ``from_route_only``), so
#: the requested walltime is what actually routes the job. Verified on tarkil 2026-11-04.
QUEUE_ROUTING: tuple[tuple[int, str], ...] = (
    (2, "q_2h"),
    (4, "q_4h"),
    (24, "q_1d"),
    (48, "q_2d"),
    (96, "q_4d"),
    (168, "q_1w"),
    (336, "q_2w"),
)

_WALLTIME_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)$")
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")

#: Files the job body imports. A release missing one of these fails only after the queue
#: wait, so `head_revision` checks them before anything is staged.
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
    "metacentrum/monitor.py",
    "metacentrum/queues.py",
    "metacentrum/pbs_launcher.sh",
)


def walltime_seconds(value: str) -> int:
    """``HH:MM:SS`` -> seconds; raises ``ValueError`` on anything else."""
    match = _WALLTIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid walltime {value!r}: expected HH:MM:SS")
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def routed_queue(walltime: str) -> str:
    """Execution queue the ``default`` router picks for this walltime."""
    seconds = walltime_seconds(walltime)
    for hours, queue in QUEUE_ROUTING:
        if seconds <= hours * 3600:
            return queue
    raise ValueError(
        f"Walltime {walltime} exceeds the longest routed queue "
        f"({QUEUE_ROUTING[-1][0]} h); submit to a special queue explicitly."
    )


@dataclass(frozen=True)
class Job:
    """One PBS job: a unique name plus the deltas that differ from ``Config`` defaults."""

    name: str
    args: tuple[str, ...] = ()
    ncpus: int = 16
    mem_gb: int = DEFAULT_MEM_GB
    scratch_gb: int = 15
    walltime: str = DEFAULT_WALLTIME
    queue: str | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"Invalid job name {self.name!r}: letters, digits, '_' or '-', starting "
                "with a letter, at most 64 characters. It also becomes the run tag."
            )
        if min(self.ncpus, self.mem_gb, self.scratch_gb) < 1:
            raise ValueError("PBS resources must be positive")
        needed = MEM_GB_PER_CORE * self.ncpus
        if self.mem_gb < needed:
            raise ValueError(
                f"{self.ncpus} cores need --mem {needed}: the scoring pool keeps "
                f"~{MEM_GB_PER_CORE} GB per worker alive, so cores cannot be raised alone."
            )
        walltime_seconds(self.walltime)
        if self.queue is None:
            routed_queue(self.walltime)  # fail now if no routed queue can hold it

    @property
    def queue_label(self) -> str:
        """What the operator should expect: an explicit queue, or the routed default."""
        if self.queue:
            return self.queue
        return f"{routed_queue(self.walltime)} (default router)"

    @property
    def resources(self) -> str:
        return (
            f"select=1:ncpus={self.ncpus}:mem={self.mem_gb}gb:"
            f"scratch_local={self.scratch_gb}gb:spec={CPU_FLOOR}"
        )


def validate_resources(
    ncpus: int,
    mem_gb: int,
    scratch_gb: int,
    walltime: str,
    queue: str | None,
) -> None:
    """Validate a resource block once, before a plan prints or a batch starts."""
    try:
        Job(
            name="probe",
            ncpus=ncpus,
            mem_gb=mem_gb,
            scratch_gb=scratch_gb,
            walltime=walltime,
            queue=queue,
        )
    except ValueError as error:
        raise SystemExit(f"invalid resources: {error}") from None


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
        lowered = output.lower()
        if any(word in lowered for word in ("kerberos", "credential", "permission denied")):
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
        "mem_gb": job.mem_gb,
        "source": f"{STORAGE}/releases/{revision}",
        "revision": revision,
    }
    payload = json.dumps(spec, indent=2).encode()
    remote(f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}", stdin=payload)
    return path


def _release_exists(revision: str) -> bool:
    probe = f"test -d {shlex.quote(f'{STORAGE}/releases/{revision}')} && echo yes || echo no"
    return remote(probe).strip() == "yes"


def submit(job: Job, revision: str | None = None, no_sync: bool = False) -> str:
    """Stage, upload and queue one job; a campaign stages once and passes ``revision`` in."""
    for delta in job.args:
        name = delta.split("=", 1)[0]
        if name in RESERVED:
            raise SystemExit(
                f"{name} is owned by the job (it is the scratch directory the results are "
                "copied back from); drop it from the deltas"
            )

    if no_sync and not revision:
        raise SystemExit("--no-sync needs --revision: it never guesses which release to use")
    if no_sync:
        if not _release_exists(revision):
            raise SystemExit(
                f"release {revision} is not staged; run `sync` or drop --no-sync"
            )
    elif revision:
        sync(revision)
    else:
        revision = sync()

    release = f"{STORAGE}/releases/{revision}"
    spec = stage_spec(job, revision)
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
    print(f"queue    : {job.queue_label}")
    print(f"resources: {job.resources}, walltime={job.walltime}")
    print(f"deltas   : {' '.join(job.args) or '(none)'}")
    job_id = remote(shlex.join(command)).strip()
    print(f"\nsubmitted {job.name}: {job_id}")
    return job_id


# --------------------------------------------------------------------------- #
# supervision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RemoteJob:
    """One row of ``qstat -f`` that a human cares about."""

    job_id: str
    name: str
    state: str
    queue: str = ""
    node: str = ""
    elapsed: str = ""
    walltime: str = ""
    comment: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.job_id.split(".")[0]

    @property
    def running(self) -> bool:
        return self.state == "R"


def parse_qstat_full(text: str) -> list[RemoteJob]:
    """Parse long-format ``qstat -f`` blocks (PBS Pro emits ``Key = Value`` lines).

    PBS output is column-based in the short format and shifts with long job names, so the
    long format is parsed instead of slicing whitespace. ``qstat -u <user> -f`` on this
    build prints the short table regardless, which is why :func:`list_jobs` resolves ids
    first and expands them one ``qstat -f`` call.
    """
    jobs: list[RemoteJob] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            if current and "Job_Name" in current:
                jobs.append(_job_from_fields(current))
            current = {"Job_Id": line.split(":", 1)[1].strip()}
            continue
        if current is None:
            continue
        match = re.match(r"\s+([A-Za-z_][\w.\-]*)\s*=\s*(.*)$", line)
        if match:
            current[match.group(1)] = match.group(2).strip()
    if current and "Job_Name" in current:
        jobs.append(_job_from_fields(current))
    return jobs


def _node_name(fields: dict[str, str]) -> str:
    """Resolvable worker name: ``exec_host2`` has the domain, ``exec_host`` may be short."""
    for key in ("exec_host2", "exec_host"):
        raw = fields.get(key, "").strip()
        if raw:
            return raw.split("/")[0].split(":")[0].strip()
    return ""


def _job_from_fields(fields: dict[str, str]) -> RemoteJob:
    return RemoteJob(
        job_id=fields.get("Job_Id", ""),
        name=fields.get("Job_Name", "?"),
        state=fields.get("job_state", "?"),
        queue=fields.get("queue", ""),
        node=_node_name(fields),
        elapsed=fields.get("resources_used.walltime", ""),
        walltime=fields.get("Resource_List.walltime", ""),
        comment=fields.get("comment", ""),
        extra=fields,
    )


def list_jobs() -> list[RemoteJob]:
    """Current queue contents for the user (running + queued + held)."""
    ids = remote(f"qselect -u {shlex.quote(USER)}").split()
    if not ids:
        return []
    listing = remote("qstat -f " + " ".join(shlex.quote(job) for job in ids))
    return parse_qstat_full(listing)


def queue_status() -> str:
    """A stable table: state, queue, elapsed/limit, id, name."""
    jobs = list_jobs()
    if not jobs:
        return ""
    jobs.sort(key=lambda job: (job.state != "R", job.name, job.job_id))
    rows = [f"{'ID':<12} {'S':<2} {'QUEUE':<10} {'ELAPSED/LIMIT':<15} {'NAME'}"]
    for job in jobs:
        elapsed = f"{job.elapsed or '?'}/{job.walltime or '?'}"
        rows.append(
            f"{job.short_id:<12} {job.state:<2} {job.queue:<10} {elapsed:<15} {job.name}"
        )
        if job.comment:
            rows.append(f"{'':<12}   {job.comment}")
    return "\n".join(rows)


def log_path(name: str) -> str:
    """PBS stdout for this job, not merely for its name.

    Reusing ``--name`` leaves earlier attempts' logs in ``$HOME``; a newest-by-mtime glob
    then shows a stale failure while the current job is still in setup. Resolve the live
    job id from PBS first and look for ``<name>.o<id>``; a finished job with no live entry
    falls back to the newest file, which is its last attempt.
    """
    quoted = shlex.quote(name)
    ids = [job.short_id for job in list_jobs() if job.name == name]
    for job_id in ids:
        found = remote(
            f"ls $HOME/{quoted}.o{shlex.quote(job_id)}* 2>/dev/null | head -1"
        ).strip()
        if found:
            return found
    if ids:
        return ""  # live job, log not written yet: never show an older attempt
    return remote(f"ls -t $HOME/{quoted}.o* 2>/dev/null | head -1").strip()


def logs(name: str, lines: int, follow: bool) -> None:
    path = log_path(name)
    if not path:
        if any(job.name == name for job in list_jobs()):
            raise SystemExit(
                f"{name!r} is queued or just started and has not written its PBS log yet; "
                "`status` shows the state, `watch` shows epochs once they exist"
            )
        raise SystemExit(f"no PBS log for {name!r}; `status` shows whether it ever ran")
    if follow:
        subprocess.run(["ssh", "-t", HOST, f"tail -n {lines} -f {shlex.quote(path)}"])
    else:
        print(remote(f"tail -n {lines} {shlex.quote(path)}"))


def cancel(targets: list[str]) -> list[str]:
    """``qdel`` by job id or by job name (all matching jobs)."""
    resolved: list[str] = []
    for target in targets:
        if re.fullmatch(r"\d+(?:\..*)?", target):
            resolved.append(target.split(".")[0])
            continue
        out = remote(f"qselect -u {shlex.quote(USER)} -N {shlex.quote(target)}")
        matches = [line.strip() for line in out.splitlines() if line.strip()]
        if not matches:
            print(f"no queued job named {target!r}")
        resolved.extend(matches)
    if not resolved:
        return []
    remote("qdel " + " ".join(shlex.quote(job) for job in resolved))
    return resolved


def pull(name: str | None) -> None:
    """MetaCentrum to here only; nothing is pushed the other way."""
    remote_root = f"{STORAGE}/_david/output"
    if name is not None and not _NAME_RE.fullmatch(name):
        raise SystemExit(
            f"invalid run tag {name!r}: it is the suffix after the run timestamp"
        )
    probe = (
        f"ls -d {remote_root}/*_{shlex.quote(name)} 2>/dev/null | head -1"
        if name
        else f"test -d {shlex.quote(remote_root)} && echo {shlex.quote(remote_root)} || true"
    )
    if not remote(probe).strip():
        target = f"{remote_root}/*_{name}" if name else remote_root
        raise SystemExit(
            f"nothing to pull: {HOST}:{target} does not exist yet. A run directory appears "
            "there when a job finishes or is killed."
        )

    destination = PROJECT_ROOT / "_david" / "output"
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-az", "--info=progress2"]
    if name:
        # A run directory is `<YYYYmmdd_HHMM>_<tag>`: the job name is only a suffix.
        command += ["--include=*/", f"--include=*_{name}/***", "--exclude=*"]
    command += [f"{HOST}:{remote_root}/", f"{destination}/"]
    subprocess.run(command, check=True)
    print(f"pulled into {destination}")


def check() -> None:
    """Everything a submission depends on, in the order it would fail."""
    print(remote("hostname").strip(), "reachable")
    venv = f"{STORAGE}/.venv/bin/python"
    print("storage venv:", remote(f"test -x {venv} && echo present || echo MISSING").strip())
    revision = _git("rev-parse", "HEAD").strip()
    staged = "yes" if _release_exists(revision) else "no"
    print(f"HEAD {revision[:12]} staged: {staged}")
    current = remote(f"readlink {shlex.quote(STORAGE + '/current')} 2>/dev/null || true").strip()
    print("current release:", current or "(none)")
    print(
        "default lane:",
        f"{DEFAULT_WALLTIME} -> {DEFAULT_QUEUE} (default router; the training lane)",
    )
    print("queue depths:", "python -m _david.code.metacentrum.queues")
    print(queue_status() or "queue: nothing of yours is queued")


def watch(interval: int, lines: int, job: str | None) -> None:
    """Run the read-only monitor on the login node, where qstat and node ssh exist."""
    script = f"{STORAGE}/current/{PACKAGE}/metacentrum/monitor.py"
    python = f"{STORAGE}/.venv/bin/python"
    arguments = [python, script, "--user", USER, "--lines", str(lines)]
    if job:
        arguments += ["--job", job]
    if interval:
        arguments += ["--watch", str(interval)]
    command = " ".join(shlex.quote(part) for part in arguments)
    ssh = ["ssh"]
    if interval:
        ssh.append("-t")
    raise SystemExit(subprocess.run([*ssh, HOST, command]).returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("sync")
    sub.add_parser("status")

    run = sub.add_parser("submit", help="stage a release and queue one job")
    run.add_argument("--name", required=True, help="PBS job name and the run's tag")
    run.add_argument("--ncpus", type=int, default=Job.ncpus)
    run.add_argument(
        "--mem",
        type=int,
        default=Job.mem_gb,
        help=f"GB; default {Job.mem_gb}, which places ~3x faster than 64 GB. The job "
        "derives its scoring workers from this number, so a smaller request only makes "
        "epochs slower",
    )
    run.add_argument("--scratch", type=int, default=Job.scratch_gb, help="GB")
    run.add_argument(
        "--walltime",
        default=Job.walltime,
        help="compute budget, not a queue knob: it is what routes the queue",
    )
    run.add_argument("--queue", default=None, help="only when you know you need one")
    run.add_argument(
        "--revision",
        default=None,
        help="stage this commit instead of HEAD (a rerun of a known-good release)",
    )
    run.add_argument(
        "--no-sync",
        action="store_true",
        help="use --revision exactly as staged; never archive or upload",
    )
    run.add_argument(
        "deltas",
        nargs="*",
        metavar="NAME=VALUE",
        help="Config overrides, each forwarded as `--set NAME=VALUE`",
    )

    status = sub.add_parser("watch", help="live epochs of running jobs")
    status.add_argument("-i", "--interval", type=int, default=0, help="refresh every N seconds")
    status.add_argument("-n", "--lines", type=int, default=5, help="epoch rows per job")
    status.add_argument("--job", default=None, help="only this job id or name")

    log = sub.add_parser("logs")
    log.add_argument("name")
    log.add_argument("-n", "--lines", type=int, default=60)
    log.add_argument("-f", "--follow", action="store_true")

    sub.add_parser("cancel", help="qdel by job id or name").add_argument(
        "targets", nargs="+"
    )
    sub.add_parser("pull").add_argument(
        "--name", default=None, help="one run tag instead of all"
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
        validate_resources(
            args.ncpus, args.mem, args.scratch, args.walltime, args.queue
        )
        submit(
            Job(
                name=args.name,
                args=tuple(part for delta in args.deltas for part in ("--set", delta)),
                ncpus=args.ncpus,
                mem_gb=args.mem,
                scratch_gb=args.scratch,
                walltime=args.walltime,
                queue=args.queue,
            ),
            revision=args.revision,
            no_sync=args.no_sync,
        )
    elif args.command == "watch":
        watch(args.interval, args.lines, args.job)
    elif args.command == "logs":
        logs(args.name, args.lines, args.follow)
    elif args.command == "cancel":
        cancelled = cancel(args.targets)
        if cancelled:
            print("cancelled " + " ".join(cancelled))
    elif args.command == "pull":
        pull(args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
