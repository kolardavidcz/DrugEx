#!/usr/bin/env python3
"""Live epoch monitor for DrugEx jobs on MetaCentrum.

Run this on the login node (it needs ``qstat`` and SSH to the worker nodes)::

    ssh meta "python3 /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/help/monitor_live_epochs.py"

It reads ``qstat -u <user> -f``, finds each running job's scratch directory on its worker
node, and prints the tail of ``models/SequenceRNN_reinforced_fit.tsv`` with an epoch rate and
an ETA against the walltime limit. When the scratch directory is not reachable (job still
starting, or a node filesystem we cannot see) it falls back to the PBS log in ``$HOME``.

Stdlib only, read-only, no imports from the DrugEx package: it must keep working when the
staged release is old or half-synced.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

SENTINEL = "@@@DRUGEX@@@"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4"]
FIT_SUFFIX = "models/SequenceRNN_reinforced_fit.tsv"
PBS_LOG_LINES = 12


def run(argv: list[str], timeout: int = 20) -> str:
    """Return stdout of ``argv`` (empty on any failure; this is a monitor, not a guard)."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout


# --------------------------------------------------------------------------- #
# PBS state
# --------------------------------------------------------------------------- #


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
    comment: str


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
    """Parse long-format ``qstat -f <id>`` blocks (``Key = Value`` lines)."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            if current:
                blocks.append(current)
            current = {"Job_Id": line.split(":", 1)[1].strip()}
        elif current:
            match = re.match(r"\s+([A-Za-z_][\w.\-]*)\s*=\s*(.*)$", line)
            if match:
                current[match.group(1)] = match.group(2).strip()
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
            )
        )
    return jobs


def _node_name(fields: dict[str, str]) -> str:
    """Resolvable worker name: `exec_host2` has the domain, `exec_host` may be short."""
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


def probe_node(node: str, short_id: str, lines: int) -> dict[str, object]:
    """One SSH round trip: scratch dir, run dir, epoch count/age, tail, manifest summary."""
    if not node or node == "unknown":
        return {}
    script = f"""
set +e
S=""
for d in /scratch*/job_{short_id}* /scratch*/*/job_{short_id}*; do
    if [ -d "$d" ]; then S="$d"; break; fi
done
echo "SCRATCH=$S"
[ -n "$S" ] || exit 0
R=$(ls -td "$S"/output/*/ 2>/dev/null | head -1)
echo "RUN=$R"
[ -n "$R" ] || exit 0
F="${{R}}{FIT_SUFFIX}"
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
"""
    output = run(["ssh", *SSH_OPTS, node, script])
    return parse_probe(output)


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


def pbs_log_tail(name: str, lines: int) -> tuple[str, str]:
    """Newest ``$HOME/<name>.o*`` on this login node (the PBS stdout)."""
    candidates = glob.glob(os.path.expanduser(f"~/{name}.o*"))
    if not candidates:
        return "", ""
    path = max(candidates, key=os.path.getmtime)
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            tail = handle.readlines()[-lines:]
    except OSError:
        return path, ""
    return path, "".join(tail).rstrip()


def job_report(job: Job, lines: int) -> dict[str, object]:
    probe = probe_node(job.node, job.short_id, lines) if job.state == "R" else {}
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
    if isinstance(rows, list) and rows:
        header = probe.get("header")
        report["header"] = header[0] if isinstance(header, list) and header else ""
        report["rows"] = rows
        count = int(str(probe.get("count") or "0"))
        done = max(count - 1, 0)
        report["epochs_done"] = done
        try:
            mtime = float(str(probe.get("mtime")))
            report["last_epoch_age_s"] = max(now - mtime, 0)
        except ValueError:
            report["last_epoch_age_s"] = None
        target = str(probe.get("epochs") or "").strip()
        target_epochs = int(target) if target.isdigit() else None
        report["epochs_target"] = target_epochs
        report["manifest_state"] = probe.get("state", "")
        if job.elapsed and done:
            per_epoch = job.elapsed / done
            report["sec_per_epoch"] = per_epoch
            if target_epochs:
                report["eta_s"] = max(target_epochs - done, 0) * per_epoch
        report["run_dir"] = probe.get("run", "")
    elif job.state == "R":
        path, tail = pbs_log_tail(job.name, PBS_LOG_LINES)
        report["pbs_log"] = path
        report["log_tail"] = tail

    return report


def print_human(jobs: list[Job], reports: list[dict[str, object]]) -> None:
    print("=" * 78)
    print(f" METACENTRUM LIVE MONITOR   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    queued = [job for job in jobs if job.state != "R"]
    running = [job for job in jobs if job.state == "R"]

    if queued:
        print(f"\nQueued / held ({len(queued)}):")
        for job in queued:
            comment = f"  — {job.comment}" if job.comment else ""
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
            if isinstance(eta, (int, float)) and job.limit and job.elapsed:
                remaining = job.limit - job.elapsed
                if float(eta) > remaining:
                    print("    ! ETA exceeds the remaining walltime")
            if isinstance(age, float) and age > 1800:
                print("    ! no new epoch for 30+ min (stuck molecule or dead process?)")
            header = report.get("header")
            if header:
                print(f"    {header}")
            for row in rows:
                print(f"    {row}")
            if report.get("manifest_state") and report.get("manifest_state") != "running":
                print(f"    manifest state: {report.get('manifest_state')}")
        else:
            run_dir = report.get("run_dir")
            if run_dir:
                print(f"    scratch run dir: {run_dir}")
                print("    (setup still running — no epoch rows yet)")
            else:
                print("    (scratch not reachable yet — job still starting, or another node fs)")
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
            reports = [job_report(job, args.lines) for job in jobs]

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
                print_human(jobs, reports)

            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
