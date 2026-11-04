"""Submit and supervise DrugEx runs on MetaCentrum.

This module is the *only* command an operator (or an agent) needs. It stages an
immutable release of the committed tree, writes the per-job specification and
calls ``qsub``; the PBS work itself is done by ``job_template.sh``.

Read ``RUNBOOK.md`` next to this file for the recipes, ``help/`` for the
long-form tutorials. Pipeline hyperparameters are **not** duplicated here: every
default comes from ``_david.code.config.PipelineConfig`` and reaches the job
through the pipeline CLI, so this file only has to know about PBS.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from _david.code.config import PROJECT_ROOT, PipelineConfig

DEFAULT_HOST = "meta"
DEFAULT_USER = "kolardavidcz"
DEFAULT_SITE = "brno2"

#: PBS script executed inside the job; shipped with the release, never edited per run.
TEMPLATE_NAME = "job_template.sh"
MODEL_STEM = "Papyrus05.5_smiles_rnn_PT"

#: Flags the template owns. Letting a caller set them would silently override
#: the release/scratch contract or point the job at a model that is not staged.
RESERVED_FLAGS = ("--output-dir", "--agent-path", "--vocab-path")

#: Walltime ceiling (hours) → execution queue reached through the ``default`` router.
#: The frontends may not submit to an execution queue directly (``q_2h``…``q_2d`` and
#: ``p2e_1d`` are either ``from_route_only`` or restricted by ``acl_hosts``), so the
#: requested walltime is what actually chooses the queue. Verified on tarkil 2026-11-04.
QUEUE_ROUTING: tuple[tuple[int, str], ...] = (
    (2, "q_2h"),
    (4, "q_4h"),
    (24, "q_1d"),
    (48, "q_2d"),
    (96, "q_4d"),
    (168, "q_1w"),
    (336, "q_2w"),
)

#: The one lane every job uses: 24 h, which the ``default`` router places in ``q_1d``.
#: Walltime is not a queue-jumping knob — ask for the compute the run needs, not a
#: shorter slot to land in a shallower queue.
DEFAULT_WALLTIME = "24:00:00"

_WALLTIME_RE = re.compile(r"^(\d{1,3}):([0-5]\d):([0-5]\d)$")
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_KERBEROS_MARKERS = (
    "15010",
    "No credentials were supplied",
    "Key has expired",
    "Ticket expired",
)


# --------------------------------------------------------------------------- #
# paths and release staging
# --------------------------------------------------------------------------- #


def storage_root(user: str = DEFAULT_USER, site: str = DEFAULT_SITE) -> str:
    """Persistent project directory for ``user`` on ``site``.

    ``/storage/brno2/home`` is a symlink owned by ``nobody``, so the canonical
    ``/storage/brno2/brno2`` spelling is used instead of the home alias.
    """
    if site == "brno2":
        return f"/storage/{site}/{site}/{user}/DrugEx"
    return f"/storage/{site}/home/{user}/DrugEx"


def release_path(storage: str, revision: str) -> str:
    """Immutable release directory for one commit."""
    return f"{storage}/releases/{revision}"


def spec_path(storage: str, name: str) -> str:
    """Per-experiment job specification kept on storage as provenance."""
    return f"{storage}/jobs/{name}/spec.json"


def template_path(storage: str, revision: str) -> str:
    """PBS script shipped inside the release, so a job can never drift from its code."""
    return f"{release_path(storage, revision)}/_david/code/metacentrum/{TEMPLATE_NAME}"


def walltime_seconds(value: str) -> int:
    """``HH:MM:SS`` → seconds; raises ``ValueError`` on anything else."""
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


# --------------------------------------------------------------------------- #
# job specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobSpec:
    """One PBS job: a unique name plus the pipeline flags that differ from the defaults."""

    name: str
    args: tuple[str, ...] = ()
    ncpus: int = 16
    mem_gb: int = 32
    scratch_gb: int = 15
    walltime: str = DEFAULT_WALLTIME
    queue: str | None = None
    spec: float | None = 8.0
    ngpus: int = 0
    gpu_cap: str | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"Invalid job name {self.name!r}: use letters, digits, '_' or '-', "
                "starting with a letter (max 64 chars). It also becomes the run tag."
            )
        if min(self.ncpus, self.mem_gb, self.scratch_gb) < 1 or self.ngpus < 0:
            raise ValueError("PBS resources must be positive (ngpus may be 0)")
        walltime_seconds(self.walltime)
        for flag in RESERVED_FLAGS:
            if any(arg == flag or arg.startswith(f"{flag}=") for arg in self.args):
                raise ValueError(
                    f"{flag} is owned by the MetaCentrum template and must not be passed"
                )

    @property
    def queue_label(self) -> str:
        """What the operator should expect: an explicit queue, or the routed default."""
        if self.queue:
            return self.queue
        return f"{routed_queue(self.walltime)} (default router)"

    @property
    def resources(self) -> str:
        """``select=…`` expression for ``qsub -l``."""
        line = (
            f"select=1:ncpus={self.ncpus}:mem={self.mem_gb}gb"
            f":scratch_local={self.scratch_gb}gb"
        )
        if self.spec is not None:
            line += f":spec={self.spec}"
        if self.ngpus:
            line += f":ngpus={self.ngpus}"
            if self.gpu_cap:
                line += f":gpu_cap={self.gpu_cap}"
        return line


def spec_payload(spec: JobSpec, storage: str, revision: str) -> dict:
    """JSON handed to ``job_template.sh``.

    ``--agent-path``/``--vocab-path`` are appended here because the checkpoint is
    staged on storage, not shipped in the release.
    """
    args = [*spec.args]
    args += [
        "--agent-path",
        f"{storage}/models/{MODEL_STEM}.pkg",
        "--vocab-path",
        f"{storage}/models/{MODEL_STEM}.vocab",
    ]
    return {
        "name": spec.name,
        "args": args,
        "source": release_path(storage, revision),
        "revision": revision,
    }


def qsub_command(spec: JobSpec, storage: str, revision: str, host: str = DEFAULT_HOST) -> list[str]:
    """Full ``ssh <host> qsub …`` argv for one job."""
    remote = ["qsub", "-N", spec.name]
    if spec.queue:
        remote += ["-q", spec.queue]
    remote += [
        "-l",
        spec.resources,
        "-l",
        f"walltime={spec.walltime}",
        "-v",
        f"RUN_SPEC={spec_path(storage, spec.name)},STORAGE_DIR={storage}",
        template_path(storage, revision),
    ]
    return ["ssh", host, shlex.join(remote)]


# --------------------------------------------------------------------------- #
# remote plumbing
# --------------------------------------------------------------------------- #


def _kerberos_error(action: str, host: str, output: str) -> RuntimeError | None:
    if not any(marker in output for marker in _KERBEROS_MARKERS):
        return None
    return RuntimeError(
        f"Kerberos ticket missing or expired on '{host}' during {action}.\n"
        f"Fix it interactively: ssh {host} kinit {DEFAULT_USER}@META\n"
        "A 2FA code is required; it cannot be refreshed non-interactively."
    )


def remote(host: str, command: str, stdin: str | bytes | None = None) -> subprocess.CompletedProcess:
    """Run ``command`` on ``host``, turning PBS/Kerberos noise into one clear error.

    ``stdin`` may be a byte stream (a ``git archive`` tarball); subprocess refuses bytes
    input in text mode, so binary transfers run without ``text=True`` and the captured
    output is decoded afterwards.
    """
    binary = isinstance(stdin, (bytes, bytearray))
    res = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        input=stdin,
        capture_output=True,
        text=not binary,
    )
    if binary:
        res = subprocess.CompletedProcess(
            res.args,
            res.returncode,
            res.stdout.decode("utf-8", "replace"),
            res.stderr.decode("utf-8", "replace"),
        )
    if res.returncode != 0:
        error = _kerberos_error("SSH command", host, res.stdout + res.stderr)
        if error:
            raise error
        raise RuntimeError(
            f"Remote command failed ({res.returncode}) on '{host}': {command}\n"
            f"{(res.stderr or res.stdout).strip()}"
        )
    return res


def git_head(root: Path = PROJECT_ROOT) -> str:
    """Revision that will be staged; the caller decides whether to sync it."""
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _assert_clean_tracked_tree(root: Path = PROJECT_ROOT) -> None:
    """Untracked files never enter ``git archive``, tracked edits silently would."""
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise RuntimeError(
            "Tracked files are modified; commit them before staging a release "
            "(otherwise the job runs HEAD, not your working tree):\n" + dirty
        )


def sync_release(
    host: str = DEFAULT_HOST,
    storage: str = storage_root(),
    root: Path = PROJECT_ROOT,
    revision: str | None = None,
) -> str:
    """Stage ``git archive <revision>`` on storage, repoint ``current``, refresh models.

    Returns the staged revision. Nothing is ever rsynced into a live tree, so a
    running job always keeps the code it started with.
    """
    root = Path(root)
    revision = revision or git_head(root)
    _assert_clean_tracked_tree(root)
    release = release_path(storage, revision)

    archive = subprocess.run(["git", "-C", str(root), "archive", revision], check=True, capture_output=True).stdout
    remote(
        host,
        f"mkdir -p {shlex.quote(storage + '/releases')} "
        f"{shlex.quote(release)} && tar -xf - -C {shlex.quote(release)}",
        stdin=archive,
    )
    remote(host, f"ln -sfn {shlex.quote(release)} {shlex.quote(storage + '/current')}")

    pkg, vocab = PipelineConfig().resolve_pretrained_paths()
    remote(host, f"mkdir -p {shlex.quote(storage + '/models')}")
    subprocess.run(
        ["rsync", "-a", str(pkg), str(vocab), f"{host}:{storage}/models/"],
        check=True,
        capture_output=True,
    )
    return revision


def stage_spec(host: str, storage: str, spec: JobSpec, revision: str) -> str:
    """Upload the job specification; it is what the job actually executed."""
    path = spec_path(storage, spec.name)
    payload = json.dumps(spec_payload(spec, storage, revision), indent=2)
    remote(
        host,
        f"mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path)}",
        stdin=payload,
    )
    return path


def submit_job(
    spec: JobSpec,
    revision: str,
    host: str = DEFAULT_HOST,
    storage: str = storage_root(),
) -> str:
    """Stage the spec, submit, and return the PBS job id.

    There is no preview mode: the printed resource block in ``_do_submit`` is the
    only summary, and the job is always handed to PBS.
    """
    command = qsub_command(spec, storage, revision, host)
    stage_spec(host, storage, spec, revision)
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0:
        error = _kerberos_error("job submission", host, out.stdout + out.stderr)
        if error:
            raise error
        raise RuntimeError(f"qsub failed:\n{out.stdout}{out.stderr}".strip())
    return out.stdout.strip().splitlines()[-1].strip()


# --------------------------------------------------------------------------- #
# queue inspection
# --------------------------------------------------------------------------- #


@dataclass
class RemoteJob:
    """One row of ``qstat -f`` that a human cares about."""

    job_id: str
    name: str
    state: str
    queue: str = ""
    node: str = ""
    elapsed: str = ""
    walltime: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.job_id.split(".")[0]

    @property
    def running(self) -> bool:
        return self.state == "R"


def parse_qstat_full(text: str) -> list[RemoteJob]:
    """Parse long-format ``qstat -f <job id>`` blocks (PBS Pro emits ``Key = Value`` lines).

    PBS output is column-based in the short format and shifts with long job
    names, so the long format is parsed instead of slicing whitespace. Note that
    ``qstat -u <user> -f`` on this build prints the *short* table regardless, which is
    why :func:`list_jobs` resolves ids first.
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
    """Fully qualified worker name.

    ``exec_host`` may be a bare short name (``eluo1-2/4*4``) that does not resolve from
    the login node, while ``exec_host2`` carries the domain and a port
    (``eluo1-2.hw.elixir-czech.cz:15002/4*4``). Prefer the resolvable one.
    """
    for key in ("exec_host2", "exec_host"):
        raw = fields.get(key, "").strip()
        if raw:
            return raw.split("/")[0].split(":")[0].strip()
    return ""


def _job_from_fields(fields: dict[str, str]) -> RemoteJob:
    node = _node_name(fields)
    return RemoteJob(
        job_id=fields.get("Job_Id", ""),
        name=fields.get("Job_Name", "?"),
        state=fields.get("job_state", "?"),
        queue=fields.get("queue", ""),
        node=node,
        elapsed=fields.get("resources_used.walltime", ""),
        walltime=fields.get("Resource_List.walltime", ""),
        extra=fields,
    )


def list_jobs(host: str = DEFAULT_HOST, user: str = DEFAULT_USER) -> list[RemoteJob]:
    """Current queue contents for ``user`` (running + queued + held).

    This PBS build prints the *short* table when ``-f`` is combined with ``-u``, so the
    job ids are collected first and expanded one ``qstat -f`` call. Column parsing of the
    short table would break on the long job names we use.
    """
    ids = remote(host, f"qselect -u {shlex.quote(user)}").stdout.split()
    if not ids:
        return []
    listing = remote(host, "qstat -f " + " ".join(shlex.quote(job) for job in ids))
    return parse_qstat_full(listing.stdout)


def cancel(host: str, user: str, targets: list[str]) -> list[str]:
    """``qdel`` by job id or by job name (all matching jobs)."""
    resolved: list[str] = []
    for target in targets:
        if target.isdigit():
            resolved.append(target)
        else:
            out = remote(host, f"qselect -u {shlex.quote(user)} -N {shlex.quote(target)}").stdout
            resolved.extend(line.strip() for line in out.splitlines() if line.strip())
    if not resolved:
        return []
    remote(host, "qdel " + " ".join(shlex.quote(job) for job in resolved))
    return resolved


def log_path(host: str, name: str) -> str:
    """Newest PBS stdout/stderr file for a job name (``$HOME/<name>.o<seq>``)."""
    out = remote(host, f"ls -t $HOME/{shlex.quote(name)}.o* 2>/dev/null | head -1").stdout
    return out.strip()


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #


def pull_results(
    host: str,
    storage: str,
    dest: Path,
    only: str | None = None,
) -> None:
    """Copy finished runs from storage into the local ``_david/output`` tree.

    The storage output root only exists once a job has copied something back, so a missing
    directory is reported as "nothing yet" instead of an rsync failure.
    """
    remote_root = f"{storage}/_david/output"
    probe_target = f"{remote_root}/{only}" if only else remote_root
    probe = (
        f'if [ -d {shlex.quote(probe_target)} ]; then echo yes; '
        "else echo no; fi"
    )
    if remote(host, probe).stdout.strip() != "yes":
        raise RuntimeError(
            f"Nothing to pull: {host}:{probe_target} does not exist yet. "
            "A run directory appears there when a job finishes (or is killed)."
        )

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{host}:{remote_root}/"
    if only:
        src += f"{only}/"
        dest = dest / only
        dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--info=progress2", src, f"{dest}/"], check=True)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def split_cli(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``… submit --name x -- --criteria gmean`` into CLI and pipeline parts.

    ``--`` is handled here rather than by argparse because a subparser would
    otherwise try to interpret pipeline flags as its own.
    """
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def _split_args(raw: str) -> tuple[str, ...]:
    return tuple(shlex.split(raw))


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH alias (default: meta)")
    parser.add_argument("--user", default=DEFAULT_USER, help="MetaCentrum username")
    parser.add_argument(
        "--storage-site", default=DEFAULT_SITE, help="Storage site (default: brno2)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m _david.code.metacentrum.submit",
        description="Run DrugEx on MetaCentrum. Recipes: see RUNBOOK.md next to this file.",
    )
    actions = parser.add_subparsers(dest="action", required=True)

    check = actions.add_parser("check", help="verify SSH, environment, release and queue")
    _common(check)

    sync = actions.add_parser("sync", help="stage HEAD as a new release and repoint `current`")
    _common(sync)

    submit = actions.add_parser("submit", help="stage, specify and submit one job")
    _common(submit)
    submit.add_argument("--name", required=True, help="unique job name; also the run tag")
    submit.add_argument(
        "--args", default="", help='extra pipeline flags, e.g. --args "--criteria gmean"'
    )
    submit.add_argument("--ncpus", type=int, default=JobSpec.ncpus)
    submit.add_argument("--mem", type=int, default=JobSpec.mem_gb, help="RAM in GB")
    submit.add_argument("--scratch", type=int, default=JobSpec.scratch_gb, help="scratch GB")
    submit.add_argument(
        "--walltime",
        default=JobSpec.walltime,
        help="HH:MM:SS compute budget (default: 24 h → q_1d)",
    )
    submit.add_argument(
        "--queue",
        default=None,
        help=(
            "route/special queue only, e.g. gpu_long (default: none; PBS routes the "
            "job by walltime into q_1d). Execution queues reject direct submissions."
        ),
    )
    submit.add_argument("--spec", type=float, default=JobSpec.spec, help="min node SPEC score")
    submit.add_argument("--ngpus", type=int, default=JobSpec.ngpus)
    submit.add_argument("--gpu-cap", default=None)
    submit.add_argument("--revision", default=None, help="commit to run (default: HEAD)")
    submit.add_argument("--no-sync", action="store_true", help="reuse an already staged release")

    status = actions.add_parser("status", help="queue table for the current user")
    _common(status)

    logs = actions.add_parser("logs", help="tail the PBS log of a job (live with -f)")
    _common(logs)
    logs.add_argument("name", help="job name as passed to submit --name")
    logs.add_argument("-n", "--lines", type=int, default=40)
    logs.add_argument("-f", "--follow", action="store_true")

    cancel_cmd = actions.add_parser("cancel", help="delete jobs by id or by name")
    _common(cancel_cmd)
    cancel_cmd.add_argument("targets", nargs="+", help="job ids or job names")

    pull = actions.add_parser("pull", help="copy finished runs into _david/output")
    _common(pull)
    pull.add_argument("--name", default=None, help="pull only this run directory")
    pull.add_argument("--dest", default=str(PROJECT_ROOT / "_david" / "output"))

    return parser


def _print_jobs(jobs: list[RemoteJob]) -> None:
    if not jobs:
        print("Queue is empty for this user.")
        return
    print(f"{'JOB ID':<10} {'STATE':<5} {'QUEUE':<10} {'ELAPSED':<10} {'WALLTIME':<10} NAME")
    for job in jobs:
        print(
            f"{job.short_id:<10} {job.state:<5} {job.queue:<10} "
            f"{job.elapsed or '-':<10} {job.walltime or '-':<10} {job.name}"
        )
    running = sum(1 for job in jobs if job.running)
    print(f"\n{len(jobs)} job(s): {running} running, {len(jobs) - running} queued/held")


def _do_check(host: str, user: str, storage: str) -> None:
    script = (
        'echo "host: $(hostname)"; '
        f'test -x {shlex.quote(storage)}/.venv/bin/python '
        f'&& echo "python: $({shlex.quote(storage)}/.venv/bin/python -V 2>&1)" '
        '|| echo "python: MISSING"; '
        f'echo "current: $(readlink -f {shlex.quote(storage)}/current 2>/dev/null || echo none)"; '
        f'echo "models: $(ls {shlex.quote(storage)}/models 2>/dev/null | tr "\\n" " ")"; '
        f'echo "jobs: $(qstat -u {shlex.quote(user)} 2>/dev/null | tail -n +6 | wc -l)"'
    )
    print(remote(host, script).stdout.strip())


def _assert_unquoted_pipeline_tokens(tokens: list[str]) -> None:
    """Catch ``-- "--epochs 1 --samples 10"``: one quoted blob is not a flag list.

    Everything after ``--`` is forwarded verbatim, exactly like ``argv`` in a shell.
    A token that starts with ``-`` and contains whitespace can only be a quoted group
    of flags, which argparse would pass to the pipeline as one nonsense argument.
    """
    for token in tokens:
        if token.startswith("-") and any(char.isspace() for char in token):
            raise SystemExit(
                f"Pipeline flags after '--' must be separate arguments, but got one "
                f"quoted blob: {token!r}.\n"
                "Drop the surrounding quotes, or use --args \"…\" instead."
            )


def _do_submit(args: argparse.Namespace, pipeline_args: list[str], storage: str) -> None:
    if args.args and pipeline_args:
        raise SystemExit("Pass pipeline flags either with --args or after --, not both")
    _assert_unquoted_pipeline_tokens(pipeline_args)
    spec = JobSpec(
        name=args.name,
        args=tuple(pipeline_args) if pipeline_args else _split_args(args.args),
        ncpus=args.ncpus,
        mem_gb=args.mem,
        scratch_gb=args.scratch,
        walltime=args.walltime,
        queue=args.queue,
        spec=args.spec,
        ngpus=args.ngpus,
        gpu_cap=args.gpu_cap,
    )
    revision = args.revision or git_head()
    if not args.no_sync:
        print(f"Staging {revision[:8]} → {storage}/releases/ …")
        revision = sync_release(host=args.host, storage=storage, revision=revision)
    else:
        print(f"Reusing staged revision {revision[:8]}")

    print(f"  release : {release_path(storage, revision)}")
    print(f"  queue   : {spec.queue_label}")
    print(f"  resources: {spec.resources}, walltime={spec.walltime}")
    print(f"  pipeline: {shlex.join(spec.args) or '(all PipelineConfig defaults)'}")
    job_id = submit_job(spec, revision, host=args.host, storage=storage)
    print(f"\nSubmitted {spec.name}: {job_id}")
    print(f"Watch it: python -m _david.code.metacentrum.submit logs {spec.name} -f")


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    cli, pipeline_args = split_cli(raw)
    args = build_parser().parse_args(cli)
    storage = storage_root(args.user, args.storage_site)

    try:
        if args.action == "check":
            _do_check(args.host, args.user, storage)
        elif args.action == "sync":
            revision = sync_release(host=args.host, storage=storage)
            print(f"Staged {revision} → {storage}/current")
        elif args.action == "submit":
            _do_submit(args, pipeline_args, storage)
        elif args.action == "status":
            _print_jobs(list_jobs(args.host, args.user))
        elif args.action == "logs":
            path = log_path(args.host, args.name)
            if not path:
                print(
                    f"No PBS log for '{args.name}' in $HOME "
                    "(not started yet, or already rotated)."
                )
                return 1
            print(f"# {path}")
            if args.follow:
                subprocess.run(
                    ["ssh", "-t", args.host, f"tail -n {args.lines} -f {shlex.quote(path)}"]
                )
            else:
                print(remote(args.host, f"tail -n {args.lines} {shlex.quote(path)}").stdout)
        elif args.action == "cancel":
            deleted = cancel(args.host, args.user, args.targets)
            print("Deleted: " + ", ".join(deleted) if deleted else "Nothing matched.")
        elif args.action == "pull":
            pull_results(args.host, storage, Path(args.dest), args.name)
            print(f"Pulled into {args.dest}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
