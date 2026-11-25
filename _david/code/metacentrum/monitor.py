#!/usr/bin/env python3
"""Live epoch monitor for DrugEx jobs on MetaCentrum.

Runs on a MetaCentrum login node, where ``qstat`` and SSH to the worker nodes exist:

    ssh meta ".venv/bin/python /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/monitor.py"

or, from the workstation, ``python -m _david.code.metacentrum.submit watch``.

It reads the queue in long format, finds each running job's scratch run directory on its
worker node, and prints the tail of ``rl_fit.tsv`` with an epoch rate and an ETA against the
walltime limit. When scratch is not reachable (the job is still starting, or the node
filesystem is not visible) it falls back to that job's PBS log in ``$HOME``.

Stdlib only and read-only: it must keep working when the staged release is old or only
half-synced, so it never imports the DrugEx package.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SENTINEL = "@@@DRUGEX@@@"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-T"]
FIT_NAME = "rl_fit.tsv"
PBS_LOG_LINES = 12
STALE_EPOCH_SECONDS = 1800


def run(argv: list[str], timeout: int = 30) -> str:
    """Return stdout of ``argv`` (empty on any failure; this is a monitor, not a guard)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout


def _run_probe_ssh(argv: list[str], timeout: int = 30) -> tuple[str, str]:
    """Run an SSH probe and return ``(stdout, error_summary)``."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", f"ssh timed out after {timeout}s"
    except OSError as exc:
        return "", str(exc)
    if proc.returncode != 0 and not proc.stdout.strip():
        err = (proc.stderr or "").strip().splitlines()
        return "", err[-1] if err else f"ssh exited {proc.returncode}"
    return proc.stdout, ""


@dataclass
class Job:
    job_id: str
    short_id: str
    name: str
    state: str
    queue: str
    node: str
    elapsed: int | None
    limit: int | None
    comment: str = ""
    #: The request, in the units the table prints: cores and whole GB.
    ncpus: int | None = None
    memory_gb: int | None = None
    scratch_dir: str = ""
    user: str = ""


def parse_walltime(value: str) -> int | None:
    """``HH:MM:SS`` (PBS) to seconds; ``None`` when unset or malformed."""
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        return None
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    while len(numbers) < 3:
        numbers.insert(0, 0)
    hours, minutes, seconds = numbers
    return hours * 3600 + minutes * 60 + seconds


def parse_qstat_full(text: str) -> list[Job]:
    """Parse long-format ``qstat -f <id>`` blocks (``Key = Value`` lines + continuations)."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    last_key: str | None = None
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            if current:
                blocks.append(current)
            current = {"Job_Id": line.split(":", 1)[1].strip()}
            last_key = None
        elif current:
            if last_key and line.startswith("\t"):
                current[last_key] += line[1:].rstrip()
                continue
            match = re.match(r"\s+([A-Za-z_][\w.\-]*)\s*=\s*(.*)$", line)
            if match:
                last_key = match.group(1)
                current[last_key] = match.group(2).strip()
    if current:
        blocks.append(current)

    jobs: list[Job] = []
    for fields in blocks:
        job_id = fields.get("Job_Id", "")
        if not job_id:
            continue
        jobs.append(
            Job(
                job_id=job_id,
                short_id=job_id.split(".")[0],
                name=fields.get("Job_Name", "?"),
                state=fields.get("job_state", "?"),
                queue=fields.get("queue", ""),
                node=_node_name(fields),
                elapsed=parse_walltime(fields.get("resources_used.walltime", "")),
                limit=parse_walltime(fields.get("Resource_List.walltime", "")),
                comment=fields.get("comment", ""),
                ncpus=_as_int(fields.get("Resource_List.ncpus") or fields.get("NCPUS")),
                memory_gb=_memory_gb(fields.get("Resource_List.mem", "")),
                scratch_dir=_scratch_dir(fields),
                user=_job_owner(fields),
            )
        )
    return jobs


def _scratch_dir(fields: dict[str, str]) -> str:
    var_list = fields.get("Variable_List", "")
    match = re.search(r"(?:^|,)SCRATCHDIR=([^,]+)", var_list)
    return match.group(1).strip() if match else ""


def _job_owner(fields: dict[str, str]) -> str:
    owner = fields.get("Job_Owner", "").strip()
    return owner.split("@", 1)[0] if owner else ""


def _as_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


_MEM_UNIT_GB: dict[str, float] = {
    "": 1.0,
    "b": 1.0 / (1024**3),
    "kb": 1.0 / (1024**2),
    "mb": 1.0 / 1024,
    "gb": 1.0,
    "tb": 1024.0,
}


def _memory_gb(value: str) -> int | None:
    """Parse PBS memory strings (``'32gb'``, ``'32768mb'``, ``'33554432kb'``) to whole GB."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)", value or "")
    if not match:
        return None
    scale = _MEM_UNIT_GB.get(match.group(2).lower())
    if scale is None:
        return None
    return round(float(match.group(1)) * scale)


def _node_name(fields: dict[str, str]) -> str:
    """Resolvable worker name: ``exec_host2`` has the domain, ``exec_host`` may be short."""
    for key in ("exec_host2", "exec_host"):
        raw = fields.get(key, "").strip()
        if raw:
            return raw.split("/")[0].split(":")[0].strip()
    return "unknown"


def queue_snapshot(user: str) -> list[Job]:
    """Long-format job blocks for ``user``.

    This PBS build returns the *short* table for ``qstat -u <user> -f``, so ids are
    resolved first and expanded with ``qstat -f <id>...``.
    """
    ids = run(["qselect", "-u", user]).split()
    if not ids:
        return []
    return parse_qstat_full(run(["qstat", "-f", *ids]))


# --------------------------------------------------------------------------- #
# worker scratch probe
# --------------------------------------------------------------------------- #


def probe_node(
    node: str,
    short_id: str,
    lines: int,
    *,
    scratch_dir: str = "",
    user: str = "",
) -> dict[str, object]:
    """One SSH round trip: scratch dir, run dir, epoch count/age, tail, manifest summary."""
    if not node or node == "unknown":
        return {}
    if user and re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        globs = (
            f"/scratch*/job_{short_id}* "
            f"/scratch*/{user}/job_{short_id}* "
            f"/scratch*/*/{user}/job_{short_id}*"
        )
    else:
        globs = f"/scratch*/job_{short_id}* /scratch*/*/job_{short_id}* /scratch*/*/*/job_{short_id}*"
    hint = scratch_dir.replace("'", "'\"'\"'") if scratch_dir else ""
    script = f"""
set +e
S=""
HINT='{hint}'
if [ -n "$HINT" ] && [ -d "$HINT" ]; then
    S="$HINT"
else
    for d in {globs}; do
        if [ -d "$d" ]; then S="$d"; break; fi
    done
fi
echo "SCRATCH=$S"
[ -n "$S" ] || exit 0
R=$(ls -td "$S"/output/*/ 2>/dev/null | head -1)
echo "RUN=$R"
[ -n "$R" ] || exit 0
F="${{R}}{FIT_NAME}"
echo "FIT=$F"
if [ -f "$F" ]; then
    echo "COUNT=$(wc -l < "$F")"
    echo "MTIME=$(stat -c %Y "$F" 2>/dev/null)"
    echo "HEADER<<{SENTINEL}"
    head -1 "$F"
    echo "{SENTINEL}"
    echo "ROWS<<{SENTINEL}"
    tail -n {lines} "$F"
    echo "{SENTINEL}"
fi
M="${{R}}manifest.json"
if [ -f "$M" ]; then
    echo "EPOCHS=$(grep -m1 '"epochs"' "$M" | tr -dc '0-9')"
    echo "STATE=$(grep -m1 '"state"' "$M" | cut -d'"' -f4)"
fi
# Prior sampling ends when prior.tsv is written, which is the boundary between the
# fixed cost before epoch 1 and the per-epoch work. config.json gives n_final, which
# sizes the post-training tail.
echo "PRIOR=$(stat -c %Y "${{R}}"prior.tsv 2>/dev/null)"
echo "FINAL=$(grep -m1 '"n_final"' "${{R}}"config.json 2>/dev/null | tr -dc '0-9')"
"""
    stdout, err = _run_probe_ssh(["ssh", *SSH_OPTS, node, script])
    if err:
        return {"probe_error": err}
    return parse_probe(stdout)


def parse_probe(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.endswith(f"<<{SENTINEL}"):
            key = line[: -len(f"<<{SENTINEL}")].strip()
            index += 1
            chunk: list[str] = []
            while index < len(lines) and lines[index].strip() != SENTINEL:
                chunk.append(lines[index])
                index += 1
            result[key.lower()] = chunk
        elif "=" in line:
            key, value = line.split("=", 1)
            result[key.strip().lower()] = value.strip()
        index += 1
    return result


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def human_seconds(value: float | int | None) -> str:
    if value is None:
        return "?"
    seconds = int(value)
    if seconds < 0:
        return "?"
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def pbs_log_tail(name: str, lines: int, short_id: str | None = None) -> tuple[str, str]:
    """PBS stdout for this job, not merely for its name.

    A reused ``--name`` leaves the logs of earlier attempts in ``$HOME``; preferring
    ``<name>.o<short_id>*`` keeps a stale failure out of the report.
    """
    pattern = f"{name}.o{short_id}*" if short_id else f"{name}.o*"
    candidates = list(Path.home().glob(pattern))
    if not candidates:
        return "", ""
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            tail = handle.readlines()[-lines:]
    except OSError:
        return str(path), ""
    return str(path), "".join(tail).rstrip()


#: What happens after the epoch loop: three 5000-molecule arms (best-epoch, at-stop,
#: gmean) and then the figures. Measured over 132 finished runs: median 36 min, p10 24,
#: p90 68 -- and flat across 10/20/30 conformers, because the arms are what they are.
TAIL_MINUTES = 36.0
TAIL_REFERENCE_N = 5000


def tail_seconds(n_final: int | None) -> float:
    """The work a run still owes after its last epoch, sized by what it re-scores."""
    if not isinstance(n_final, int) or n_final <= 0:
        return TAIL_MINUTES * 60.0
    return TAIL_MINUTES * 60.0 * n_final / TAIL_REFERENCE_N


def head_seconds(job: Job, prior_mtime: object, now: float) -> float | None:
    """How much of ``elapsed`` was spent before epoch 1 rather than in it.

    ``prior.tsv`` is written when the prior sampling finishes, immediately before the
    loop starts, and the run starts at ``now - elapsed``. Leaving that setup cost inside
    the per-epoch rate (median 55 min, so a full epoch's worth at 10 conformers) makes
    every early ETA several times too large, and it keeps decaying for the whole run.
    """
    if not job.elapsed:
        return None
    try:
        written = float(str(prior_mtime))
    except (TypeError, ValueError):
        return None
    return max(written - (now - job.elapsed), 0.0)


def job_report(job: Job, lines: int) -> dict[str, object]:
    probe = (
        probe_node(
            job.node,
            job.short_id,
            lines,
            scratch_dir=job.scratch_dir,
            user=job.user,
        )
        if job.state == "R"
        else {}
    )
    now = time.time()
    report: dict[str, object] = {
        "id": job.short_id,
        "name": job.name,
        "state": job.state,
        "queue": job.queue,
        "node": job.node,
        "elapsed_s": job.elapsed,
        "limit_s": job.limit,
        "comment": job.comment,
    }

    rows = probe.get("rows")
    report["run_dir"] = probe.get("run", "")
    report["scratch"] = probe.get("scratch", "")
    if probe.get("probe_error"):
        report["probe_error"] = probe["probe_error"]
    if isinstance(rows, list) and rows:
        header = probe.get("header")
        report["header"] = header[0] if isinstance(header, list) and header else ""
        report["rows"] = rows
        count = _as_int(probe.get("count")) or 0
        done = max(count - 1, 0)
        report["epochs_done"] = done
        try:
            mtime = float(str(probe.get("mtime")))
            report["last_epoch_age_s"] = max(now - mtime, 0)
        except (TypeError, ValueError):
            report["last_epoch_age_s"] = None
        target = str(probe.get("epochs") or "").strip()
        target_epochs = int(target) if target.isdigit() else None
        report["epochs_target"] = target_epochs
        report["manifest_state"] = probe.get("state", "")
        report["n_final"] = _as_int(probe.get("final"))
        head = head_seconds(job, probe.get("prior"), now)
        report["head_s"] = head
        if job.elapsed and done:
            working = job.elapsed - head if head is not None else job.elapsed
            per_epoch = max(working, 0.0) / done
            report["sec_per_epoch"] = per_epoch
            # One epoch is not a rate: the setup it sits next to is several times its
            # length, so an ETA from it is either 60 h or 36 min depending on rounding.
            if target_epochs and done >= 2:
                report["eta_s"] = max(target_epochs - done, 0) * per_epoch + tail_seconds(
                    report["n_final"]
                )
    elif job.state == "R":
        path, tail = pbs_log_tail(job.name, PBS_LOG_LINES, job.short_id)
        report["pbs_log"] = path
        report["log_tail"] = tail

    return report


# --------------------------------------------------------------------------- #
# the one-shot table
# --------------------------------------------------------------------------- #

#: Each probe is an SSH round trip to a different worker; they are independent, so a
#: 25-job table runs them side by side instead of taking a minute.
PROBE_WORKERS = 8


def probe_all(jobs: list[Job], lines: int = 0) -> list[dict[str, object]]:
    """One report per job, probing the running ones concurrently."""
    running = [job for job in jobs if job.state == "R"]
    probes: dict[str, dict[str, object]] = {}
    if running:
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
            futures = {job.short_id: pool.submit(job_report, job, lines) for job in running}
            probes = {key: future.result() for key, future in futures.items()}
    return [probes[job.short_id] if job.state == "R" else job_report(job, lines) for job in jobs]


def _rate(seconds: object) -> str:
    """Minutes per epoch, because a PBS board is read in minutes, not seconds."""
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "-"
    minutes = seconds / 60
    return f"{minutes:.0f}m" if minutes < 100 else f"{minutes / 60:.1f}h"


def _epoch_cell(report: dict[str, object]) -> str:
    done = report.get("epochs_done")
    if not isinstance(done, int):
        return "-"
    return f"{done}/{report.get('epochs_target') or '?'}"


def _eta_cell(job: Job, report: dict[str, object]) -> str:
    """Time left, marked ``!`` when it cannot fit in what remains of the walltime.

    The mark is one character wide on purpose: a ``NOTE`` column costs more width than
    the whole rest of the table and is empty on most rows.
    """
    eta, elapsed, limit = report.get("eta_s"), job.elapsed, job.limit
    if not isinstance(eta, (int, float)):
        return "-"
    if elapsed and limit and eta > limit - elapsed:
        return f"!{human_seconds(eta)}"
    return human_seconds(eta)


def table_rows(jobs: list[Job], reports: list[dict[str, object]]) -> list[str]:
    """One row per job: what it is, how far it got, how fast, and how long is left."""
    by_id = {report["id"]: report for report in reports}
    ordered = sorted(jobs, key=lambda job: (job.state != "R", job.name))
    width = max([len(job.name) for job in ordered] + [4])
    rows = [f"{'NAME':<{width}} {'QUEUE':<6} {'ELAPSED':>10} {'EPOCH':>7} {'RATE':>5} {'ETA':>10}"]
    for job in ordered:
        report = by_id.get(job.short_id, {})
        rows.append(
            f"{job.name:<{width}} {job.queue:<6} {human_seconds(job.elapsed):>10} "
            f"{_epoch_cell(report):>7} {_rate(report.get('sec_per_epoch')):>5} "
            f"{_eta_cell(job, report):>10}"
        )
    return rows


def print_table(jobs: list[Job], reports: list[dict[str, object]]) -> None:
    print(
        f" METACENTRUM  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
        f"{sum(job.state == 'R' for job in jobs)} running, "
        f"{sum(job.state != 'R' for job in jobs)} queued"
    )
    if not jobs:
        print("\n nothing of yours is queued or running")
        return
    print()
    for row in table_rows(jobs, reports):
        print(f" {row}")

    by_id = {report["id"]: report for report in reports}
    overrun: list[str] = []
    stale: list[str] = []
    unreachable: list[str] = []
    queued: list[str] = []
    has_eta = False
    for job in jobs:
        report = by_id.get(job.short_id, {})
        if isinstance(report.get("eta_s"), (int, float)):
            has_eta = True
        if job.state == "R":
            if _eta_cell(job, report).startswith("!"):
                overrun.append(job.name)
            age = report.get("last_epoch_age_s")
            if isinstance(age, (int, float)) and age > STALE_EPOCH_SECONDS:
                stale.append(f"{job.name} {age / 60:.0f}m")
            probe_error = report.get("probe_error")
            if probe_error:
                unreachable.append(f"{job.name} ({job.node}: {probe_error})")
        elif job.comment:
            queued.append(f"{job.name}: {job.comment}")

    print()
    if overrun:
        print(f" ! ETA overruns the walltime left ({len(overrun)}): {', '.join(overrun)}")
    if stale:
        print(f" no new epoch for 30+ min: {', '.join(stale)}")
    for line in unreachable:
        print(f" unreachable {line}")
    for line in queued:
        print(f" queued {line}")
    if has_eta:
        print(
            f" ETA = epochs left at RATE, plus the {TAIL_MINUTES:.0f} min tail "
            f"(3 arms at n_final={TAIL_REFERENCE_N} and the figures)"
        )


def print_human(jobs: list[Job], reports: list[dict[str, object]]) -> None:
    print("=" * 78)
    print(f" METACENTRUM LIVE MONITOR   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    queued = [job for job in jobs if job.state != "R"]
    running = [job for job in jobs if job.state == "R"]

    if queued:
        print(f"\nQueued / held ({len(queued)}):")
        for job in queued:
            comment = f"  -- {job.comment}" if job.comment else ""
            print(f"  {job.short_id:<12} {job.state:<2} {job.queue:<10} {job.name}{comment}")

    if not running:
        print("\nNothing running.")
        print("=" * 78)
        return

    by_id = {report["id"]: report for report in reports}
    print(f"\nRunning ({len(running)}):")
    for job in running:
        report = by_id.get(job.short_id, {})
        print(f"\n>>> {job.short_id}  {job.name}  [{job.queue}]  node {job.node}")
        print(
            f"    walltime {human_seconds(job.elapsed)} / {human_seconds(job.limit)}"
            " | elapsed / limit"
        )

        rows = report.get("rows")
        if isinstance(rows, list) and rows:
            done = int(report.get("epochs_done") or 0)
            target = report.get("epochs_target")
            fraction = f" ({done * 100 // target}%)" if isinstance(target, int) and target else ""
            print(f"    epoch {done}/{target or '?'}{fraction}")
            per_epoch = report.get("sec_per_epoch")
            if per_epoch:
                print(f"    rate {float(per_epoch) / 60:.1f} min/epoch", end="")
            eta = report.get("eta_s")
            if eta is not None:
                print(f" | ETA {human_seconds(eta)}", end="")
            age = report.get("last_epoch_age_s")
            if isinstance(age, float):
                print(f" | last row {age / 60:.1f} min ago", end="")
            print()
            if (
                isinstance(eta, (int, float))
                and job.limit
                and job.elapsed
                and float(eta) > job.limit - job.elapsed
            ):
                print("    ! ETA exceeds the remaining walltime")
            if isinstance(age, float) and age > STALE_EPOCH_SECONDS:
                print("    ! no new epoch for 30+ min (stuck molecule or dead process?)")
            header = report.get("header")
            if header:
                print(f"    {header}")
            for row in rows:
                print(f"    {row}")
            state = report.get("manifest_state")
            if state and state != "running":
                print(f"    manifest state: {state}")
        else:
            run_dir = report.get("run_dir")
            probe_error = report.get("probe_error")
            if run_dir:
                print(f"    scratch run dir: {run_dir}")
                print("    (setup still running -- no epoch rows yet)")
            elif probe_error:
                print(f"    (node probe failed: {probe_error})")
            else:
                print("    (scratch not reachable yet -- job starting, or another node fs)")
            log_path = report.get("pbs_log")
            if log_path:
                print(f"    PBS log: {log_path}")
            tail = report.get("log_tail")
            if tail:
                for line in str(tail).splitlines()[-PBS_LOG_LINES:]:
                    print(f"    {line}")

    print("\n" + "=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live DrugEx epoch monitor on MetaCentrum.")
    parser.add_argument("--user", default=os.environ.get("USER", "kolardavidcz"))
    parser.add_argument("--job", default=None, help="only this job id (short form ok)")
    parser.add_argument(
        "--lines", type=int, default=5, help="epoch rows to show per running job (default 5)"
    )
    parser.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--table",
        action="store_true",
        help="one row per job: state, queue, elapsed, live epoch, rate, ETA, request, node",
    )
    args = parser.parse_args(argv)

    if shutil.which("qstat") is None:
        print(
            "qstat not found. Run this on a MetaCentrum login node (ssh meta).",
            file=sys.stderr,
        )
        return 2

    try:
        while True:
            jobs = queue_snapshot(args.user)
            if args.job:
                wanted = args.job.split(".")[0]
                jobs = [job for job in jobs if job.short_id == wanted or job.name == args.job]
            reports = probe_all(jobs, args.lines)

            if args.json:
                print(
                    json.dumps(
                        {
                            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "user": args.user,
                            "jobs": reports,
                        },
                        indent=2,
                    )
                )
            else:
                if args.watch and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                if args.table:
                    print_table(jobs, reports)
                else:
                    print_human(jobs, reports)

            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
