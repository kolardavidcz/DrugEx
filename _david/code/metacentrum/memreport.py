"""What finished jobs actually used: ``python -m _david.code.metacentrum.memreport``.

``job.py`` derives the scoring-worker count from a memory budget per worker. PBS records the
request and the peak of every finished job, so after a batch that budget can be checked
against measurements instead of trusted: this reads ``qselect``/``qstat`` and the job specs
on storage, joins them, and prints, per job, requested vs peak GB, peak per scoring worker,
the budget its shape got and a verdict. It submits, cancels and edits nothing.

Two fingerprints decide the reading:

- a peak equal to the request with ``Exit_status = 137`` was killed by the cgroup, so the
  request was too small for the pool that ran;
- a job that *finished* at 95 % or more of its request is censored the same way: the number
  stopped growing at the limit, so it bounds the need from below and cannot confirm a budget.

And one measurement that overrides the per-worker reading: the scorers build every conformer
of the batch in the parent process before the pool runs, so a job's peak follows
``n_final x max_conformers`` and moves little with the pool size. Six rebuild jobs at 13
workers and earlier training jobs at 16 workers all peaked at exactly 32.0 GB for that
reason, which is why the report shows the total peak per shape beside the per-worker split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from _david.code.metacentrum.job import (
    MEM_GB_PER_CONFORMER,
    MEMORY_RESERVE_GB,
    MEMORY_SAFETY,
    gb_per_worker,
    spec_max_conformers,
    workers_for,
)
from _david.code.metacentrum.submit import STORAGE, USER, RemoteJob, parse_qstat_full, remote

#: PBS writes ``32gb`` in a request and ``33554432kb`` in usage; both are powers of 1024.
_UNIT_BYTES = {
    "": 1,
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
    "pb": 1024**5,
}
_MEM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")

#: Above this fraction of the request a job is too close to its cgroup to trust.
TIGHT_FRACTION = 0.85
#: A job that finished at this fraction of its request is censored exactly like one that
#: was killed: the number stopped growing at the cgroup, so it is a floor on what the shape
#: needs, not a measurement of it. The rebuild batch is why this matters — six jobs at 13
#: and 16 workers all peaked at exactly 32.0 GB, so dividing by the pool turns a ceiling
#: into a convincing "per worker" fit that is not a fit at all.
CENSORED_FRACTION = 0.95
#: PBS reports a SIGKILL as 128 + 9; a job killed by the cgroup lands here.
OOM_EXIT_STATUS = 137
#: A job that died in setup peaks at a few MB. Counting that as a measurement of the
#: scoring pool would "prove" that the budget is generous, so it is not one.
MIN_USEFUL_PEAK_GB = 1.0


def parse_mem(value: str | None) -> float | None:
    """``'32gb'`` / ``'33554432kb'`` -> GB; ``None`` when PBS recorded nothing usable."""
    if value is None:
        return None
    match = _MEM_RE.match(value)
    if not match:
        return None
    number, unit = match.groups()
    factor = _UNIT_BYTES.get(unit.lower())
    if factor is None:
        return None
    return float(number) * factor / _UNIT_BYTES["gb"]


@dataclass(frozen=True)
class Finished:
    """One finished job: what it asked for, what it got, and what that implies."""

    job_id: str
    name: str
    state: str
    exit_status: int | None
    requested_gb: float | None
    peak_gb: float | None
    ncpus: int | None
    #: The pool the job intended to run; ``None`` when the spec is gone or covered a rebuild.
    workers: int | None
    #: The conformer budget of the run; ``None`` for a rebuild, whose spec spans shapes.
    conformers: int | None

    @property
    def short_id(self) -> str:
        return self.job_id.split(".", 1)[0]

    @property
    def budget_gb(self) -> float | None:
        """What the cap allowed this job's shape per worker."""
        return gb_per_worker(self.conformers) if self.conformers else None

    @property
    def per_worker_gb(self) -> float | None:
        """Peak per scoring worker; ``None`` without both a peak and a known pool."""
        if self.peak_gb is None or not self.workers or self.workers < 2:
            return None
        return self.peak_gb / self.workers

    @property
    def fill(self) -> float | None:
        if not self.peak_gb or not self.requested_gb:
            return None
        return self.peak_gb / self.requested_gb

    @property
    def censored(self) -> bool:
        """Whether the request capped the peak, so the peak only bounds the need."""
        fill = self.fill
        return fill is not None and fill >= CENSORED_FRACTION

    @property
    def verdict(self) -> str:
        if self.exit_status == OOM_EXIT_STATUS:
            return "OOM (killed at the request)"
        if self.peak_gb is None or self.peak_gb == 0:
            # A job cancelled in the queue keeps state F with no usage: it never ran, so
            # it says nothing about what a scoring pool needs.
            return "never started"
        if self.censored:
            # Finished, but at the limit: peak/pool here is (request/pool), not a need.
            return "censored at the request"
        per_worker, budget = self.per_worker_gb, self.budget_gb
        if per_worker is not None and budget is not None and per_worker > budget:
            return "over its budget"
        fill = self.fill
        if fill is not None and fill >= TIGHT_FRACTION:
            return "tight"
        return "ok"


def finished_from_jobs(
    jobs: Sequence[RemoteJob], specs: dict[str, dict[str, Any]]
) -> list[Finished]:
    """Join parsed ``qstat -x -f`` entries with the specs that produced them."""
    rows: list[Finished] = []
    for job in jobs:
        fields = job.extra
        spec = specs.get(job.name, {})
        rebuild = spec.get("mode") == "rebuild"
        ncpus = _as_int(fields.get("Resource_List.ncpus") or fields.get("resources_used.ncpus"))
        rows.append(
            Finished(
                job_id=job.job_id,
                name=job.name,
                state=job.state,
                exit_status=_as_int(fields.get("Exit_status")),
                requested_gb=parse_mem(fields.get("Resource_List.mem")),
                peak_gb=parse_mem(fields.get("resources_used.mem")),
                ncpus=ncpus,
                # A spec written before the pool was recorded ran every allocated core.
                workers=_as_int(spec.get("workers")) or (ncpus if spec and not rebuild else None),
                conformers=None if rebuild or not spec else spec_max_conformers(spec),
            )
        )
    return rows


def _as_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _job_number(job_id: str) -> int:
    try:
        return int(job_id.split(".", 1)[0])
    except ValueError:
        return 0


def specs_from_stream(text: str) -> dict[str, dict]:
    """Parse the ``### <name>/spec.json`` blocks one remote ``cat`` produces."""
    specs: dict[str, dict] = {}
    name, lines = None, []
    for line in text.splitlines():
        if line.startswith("### "):
            if name is not None:
                _keep_spec(specs, name, lines)
            name = line[4:].strip().split("/", 1)[0]
            lines = []
        elif name is not None:
            lines.append(line)
    if name is not None:
        _keep_spec(specs, name, lines)
    return specs


def _keep_spec(specs: dict[str, dict], name: str, lines: list[str]) -> None:
    try:
        parsed = json.loads("\n".join(lines))
    except json.JSONDecodeError:
        return
    if isinstance(parsed, dict):
        specs[name] = parsed


def _specs() -> dict[str, dict]:
    """Every staged spec, in one round trip; a job cleaned off storage simply has none."""
    script = (
        f"cd {shlex.quote(STORAGE + '/jobs')} 2>/dev/null || exit 0\n"
        'for f in */spec.json; do [ -f "$f" ] || continue; echo "### $f"; cat "$f"; echo; done'
    )
    return specs_from_stream(remote(script))


def _finished_ids() -> list[str]:
    """Finished job ids, newest first: ``-x`` reaches history, ``-s F`` drops the queue."""
    ids = remote(f"qselect -u {shlex.quote(USER)} -x -s F").split()
    return sorted(ids, key=_job_number, reverse=True)


def _usable(row: Finished) -> bool:
    """A peak that can inform the budget: a real pool, a real peak, no cgroup ceiling."""
    return (
        row.per_worker_gb is not None
        and row.conformers is not None
        and row.exit_status != OOM_EXIT_STATUS
        and not row.censored
        and (row.peak_gb or 0) >= MIN_USEFUL_PEAK_GB
    )


def _ceilinged(row: Finished) -> bool:
    """A peak that was cut off by the request, and so is a lower bound, not a measurement."""
    return (
        row.per_worker_gb is not None
        and row.conformers is not None
        and (row.peak_gb or 0) >= MIN_USEFUL_PEAK_GB
        and (row.exit_status == OOM_EXIT_STATUS or row.censored)
    )


def summary(rows: list[Finished]) -> dict[str, Any]:
    """The numbers the worker budget is calibrated from.

    Only uncensored jobs inform the budget. A job killed at its cgroup, and equally one that
    finished at 100 % of its request, recorded the request as its peak: that number is a
    floor on what the shape needs, not evidence of what sufficed. Both are reported as lower
    bounds, because the fix for them is a larger ``--mem``, not a different worker count.

    ``worst_total_gb`` is the observable that actually scales with the work: the scorers
    build every conformer of the batch in the parent process before the pool runs, so the
    peak follows ``n_final x max_conformers`` and barely moves with the pool size.
    """
    buckets: dict[int, list[float]] = {}
    floors: dict[int, list[float]] = {}
    totals: dict[int, float] = {}
    censored_names: list[str] = []
    oom_names: list[str] = []
    tight_names: list[str] = []
    over_names: list[str] = []
    for row in rows:
        if row.censored:
            censored_names.append(row.name)
        if row.exit_status == OOM_EXIT_STATUS:
            oom_names.append(row.name)
        verdict = row.verdict
        if verdict == "tight":
            tight_names.append(row.name)
        elif verdict == "over its budget":
            over_names.append(row.name)

        if row.conformers is None or (row.peak_gb or 0.0) < MIN_USEFUL_PEAK_GB:
            continue
        if _usable(row):
            assert row.per_worker_gb is not None
            buckets.setdefault(row.conformers, []).append(row.per_worker_gb)
            totals[row.conformers] = max(totals.get(row.conformers, 0.0), float(row.peak_gb))
        elif row.censored or row.exit_status == OOM_EXIT_STATUS:
            totals[row.conformers] = max(totals.get(row.conformers, 0.0), float(row.peak_gb))
            if _ceilinged(row):
                assert row.per_worker_gb is not None
                floors.setdefault(row.conformers, []).append(row.per_worker_gb)

    worst = {conformers: max(values) for conformers, values in buckets.items()}
    lower = {conformers: max(values) for conformers, values in floors.items()}

    ratio = max(
        (value / conformers for conformers, value in worst.items()),
        default=None,
    )
    suggested = MEM_GB_PER_CONFORMER
    short = [c for c, value in worst.items() if value > gb_per_worker(c)]
    if ratio is not None and short:
        # Raise the coefficient to the smallest 0.01 that covers the worst measured shape.
        suggested = math.ceil(ratio * 100) / 100
    shapes = sorted(set(worst) | set(lower) | set(totals))
    return {
        "jobs": len(rows),
        "measured": sum(len(values) for values in buckets.values()),
        "censored": censored_names,
        "oom": oom_names,
        "tight": tight_names,
        "over_budget": over_names,
        "per_conformer": {
            str(conformers): {
                "jobs": len(buckets.get(conformers, [])),
                "worst_gb_per_worker": worst.get(conformers),
                "lower_bound_gb_per_worker": lower.get(conformers),
                "worst_total_gb": totals.get(conformers),
                "budget_gb_per_worker": gb_per_worker(conformers),
            }
            for conformers in shapes
        },
        "current_gb_per_conformer": MEM_GB_PER_CONFORMER,
        "safety": MEMORY_SAFETY,
        "reserve_gb": MEMORY_RESERVE_GB,
        "suggested_gb_per_conformer": suggested,
    }


def _table(rows: list[Finished]) -> str:
    header = (
        f"{'JOB':<11} {'NAME':<22} {'EXIT':>4} {'REQ':>6} {'PEAK':>6} "
        f"{'PEAK/W':>6} {'CONF':>4} {'BUDGET':>6} {'WKRS':>4} {'FILL':>5}  VERDICT"
    )
    lines = [header]
    for row in rows:
        peak = f"{row.peak_gb:>6.1f}" if row.peak_gb is not None else f"{'-':>6}"
        request = f"{row.requested_gb:>6.1f}" if row.requested_gb is not None else f"{'-':>6}"
        per_worker = (
            f"{row.per_worker_gb:>6.2f}" if row.per_worker_gb is not None else f"{'-':>6}"
        )
        budget = f"{row.budget_gb:>6.2f}" if row.budget_gb is not None else f"{'-':>6}"
        conformers = f"{row.conformers:>4}" if row.conformers is not None else f"{'-':>4}"
        workers = f"{row.workers:>4}" if row.workers is not None else f"{'-':>4}"
        fill = f"{row.fill * 100:>4.0f}%" if row.fill is not None else f"{'-':>5}"
        exit_status = "-" if row.exit_status is None else str(row.exit_status)
        lines.append(
            f"{row.short_id:<11} {row.name[:22]:<22} {exit_status:>4} {request} {peak} "
            f"{per_worker} {conformers} {budget} {workers} {fill}  {row.verdict}"
        )
    return "\n".join(lines)


def _cell(value: float | None, width: int, places: int = 2) -> str:
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def _summary_text(stats: dict) -> str:
    if not stats["measured"]:
        lines = [
            "no finished job gives an uncensored peak: the recent ones either never "
            "started, died in setup, or stopped at their cgroup"
        ]
        if stats["oom"]:
            lines.append(
                f"{len(stats['oom'])} job(s) were killed by the cgroup "
                f"({', '.join(stats['oom'][:4])}{' …' if len(stats['oom']) > 4 else ''})"
            )
        if stats["censored"]:
            lines.append(
                f"{len(stats['censored'])} job(s) finished at or above "
                f"{CENSORED_FRACTION:.0%} of their request: their peak is a floor on what "
                "the shape needs, not a measurement of it"
            )
        lines.append("the worker budget cannot be checked from measurements yet")
        return "\n".join(lines)

    lines = [
        f"{stats['measured']} job(s) give an uncensored peak; "
        f"budget model: {stats['current_gb_per_conformer']} GB per conformer "
        f"per worker x {stats['safety']} safety + {stats['reserve_gb']} GB reserve",
        f"{'conf':>4} {'jobs':>4} {'worst GB/w':>10} {'>= GB/w':>8} {'peak GB':>8} "
        f"{'budget':>7}",
    ]
    for conformers, bucket in stats["per_conformer"].items():
        lines.append(
            f"{conformers:>4} {bucket['jobs']:>4} "
            f"{_cell(bucket['worst_gb_per_worker'], 10)} "
            f"{_cell(bucket['lower_bound_gb_per_worker'], 8)} "
            f"{_cell(bucket['worst_total_gb'], 8, 1)} "
            f"{_cell(bucket['budget_gb_per_worker'], 7)}"
        )
    lines.append(
        "`>= GB/w` is a lower bound from jobs the request capped (killed or 100% full), "
        "not a measurement"
    )
    lines.append(
        "32 GB then runs "
        + ", ".join(
            f"{workers_for({'mem_gb': 32}, 32, conformers=int(c))} workers at {c} conformers"
            for c in stats["per_conformer"]
        )
    )
    if stats["suggested_gb_per_conformer"] != stats["current_gb_per_conformer"]:
        lines.append(
            "a measured shape exceeds its budget: raise MEM_GB_PER_CONFORMER to "
            f"{stats['suggested_gb_per_conformer']} GB"
        )
    if stats["over_budget"]:
        lines.append(
            f"{len(stats['over_budget'])} job(s) used more per worker than their shape's "
            "budget"
        )
    if stats["censored"]:
        lines.append(
            f"{len(stats['censored'])} job(s) finished at or above {CENSORED_FRACTION:.0%} "
            "of their request, so their peak is a floor: they cannot confirm a budget, and "
            "a pool small enough to fit them is not thereby shown to be safe"
        )
    if stats["oom"]:
        lines.append(
            f"{len(stats['oom'])} job(s) were killed by the cgroup "
            f"({', '.join(stats['oom'][:4])}{' …' if len(stats['oom']) > 4 else ''}): "
            "the peak equals the request, so that shape needs a larger --mem, and the "
            "per-worker column of such a job is (request / pool), not a need"
        )
    if stats["tight"]:
        lines.append(
            f"{len(stats['tight'])} job(s) finished within "
            f"{TIGHT_FRACTION:.0%} of their request: their cap has no headroom"
        )
    return "\n".join(lines)


def _as_dict(row: Finished) -> dict:
    return {
        "job_id": row.job_id,
        "name": row.name,
        "state": row.state,
        "exit_status": row.exit_status,
        "requested_gb": row.requested_gb,
        "peak_gb": row.peak_gb,
        "ncpus": row.ncpus,
        "workers": row.workers,
        "conformers": row.conformers,
        "budget_gb": row.budget_gb,
        "per_worker_gb": row.per_worker_gb,
        "fill": row.fill,
        "censored": row.censored,
        "verdict": row.verdict,
    }


def report(last: int = 20, as_json: bool = False) -> str:
    """One pass over the most recent finished jobs; returns the text to print."""
    ids = _finished_ids()[: max(1, last)]
    if not ids:
        return "no finished jobs on MetaCentrum"
    listing = remote("qstat -x -f " + " ".join(shlex.quote(job) for job in ids))
    rows = finished_from_jobs(parse_qstat_full(listing), _specs())
    # qstat answers in request order, so keep the newest-first order of the ids.
    order = {job: index for index, job in enumerate(ids)}
    rows.sort(key=lambda row: order.get(row.job_id, len(order)))
    stats = summary(rows)
    if as_json:
        return json.dumps(
            {"jobs": [_as_dict(row) for row in rows], "summary": stats}, indent=2
        )
    return f"{_table(rows)}\n\n{_summary_text(stats)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--last", type=int, default=20, help="how many finished jobs to read (default 20)"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    print(report(args.last, args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
