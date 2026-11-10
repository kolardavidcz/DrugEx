"""The planned sweep: 148 runs in three batches.

    python -m _david.metacentrum.campaign              # all three batches, nothing submitted
    python -m _david.metacentrum.campaign parent       # one batch, exactly as it would be sent
    python -m _david.metacentrum.campaign parent --submit   # stage HEAD once, then qsub all 96

Nothing leaves this machine without ``--submit``; the default prints the plan. Submission
validates every delta of the whole batch first, so a typo stops the campaign before the
first ``qsub`` rather than after fifty of them.

A batch is a full factorial over its axes and nothing else. Every run is the same
pipeline, so two runs can differ only in the axes their names spell out -- and every
setting another batch varies is pinned in `COMMON`, which keeps a spec readable years
later and immune to a changed default.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

from _david.metacentrum.submit import Job, submit, sync, validate_deltas

#: Pinned in every run of every batch. Each is an axis some *other* batch varies, so
#: writing it out makes each spec state the whole experiment instead of inheriting half
#: of it from whatever ``config.py`` happened to default to on the day it was submitted.
COMMON: dict[str, object] = {
    "scheme": "PRCD",
    "minimize": True,
    "reference_conformers": 4,
    "max_conformers": 20,
    "color_ff": "fdef",
    "alignment_shape_weight": 0.5,
}

#: How an axis is written into a job name. ``qstat`` shows the name and nothing else, so
#: a name has to carry every axis that varies inside its batch. Floats are written as
#: hundredths (``e05`` = 0.05, ``s30`` = 0.3) because a dot in a PBS job name is asking
#: for trouble.
TOKEN = {
    "minimize": lambda v: "m1" if v else "m0",
    "epsilon": lambda v: f"e{round(v * 100):02d}",
    "reference_conformers": lambda v: f"r{v}",
    "max_conformers": lambda v: f"c{v}",
    "include_qed": lambda v: "q1" if v else "q0",
    "color_ff": lambda v: {"default": "fd", "fdef": "ff", "oe_mills_dean": "fo"}[v],
    "alignment_shape_weight": lambda v: f"s{round(v * 100):02d}",
}


@dataclass(frozen=True)
class Batch:
    """One full factorial. ``axes`` is ordered: it fixes both the name and the order."""

    prefix: str
    axes: dict[str, tuple]
    purpose: str
    fixed: dict = field(default_factory=dict)


BATCHES: dict[str, Batch] = {
    "parent": Batch(
        prefix="p",
        purpose="the main grid: does relaxation help, and under what exploration and bank",
        axes={
            "minimize": (True, False),
            "epsilon": (0.1, 0.05, 0.02, 0.0),
            "reference_conformers": (4, 1),
            "max_conformers": (10, 20, 30),
            "include_qed": (False, True),
        },
    ),
    "ws": Batch(
        prefix="w",
        purpose=(
            "DrugEx's WeightedSum instead of Pareto ranking. Note what this changes: "
            "WeightedSum builds its weights from the thresholds, so in these four runs "
            "the thresholds are part of the reward, not only the stop condition"
        ),
        fixed={"scheme": "WS"},
        axes={
            "epsilon": (0.05, 0.0),
            "include_qed": (False, True),
        },
    ),
    "color": Batch(
        prefix="c",
        purpose="colour typing and the alignment objective, at the parent batch's centre",
        axes={
            "epsilon": (0.05, 0.0),
            "color_ff": ("default", "fdef", "oe_mills_dean"),
            "alignment_shape_weight": (0.5, 0.3, 0.2, 0.1),
            "include_qed": (False, True),
        },
    ),
}


def runs(batch: Batch) -> list[tuple[str, dict]]:
    """Every run of one batch, as ``(job name, settings)``, in a fixed order."""
    out = []
    for values in itertools.product(*batch.axes.values()):
        chosen = dict(zip(batch.axes, values))
        name = batch.prefix + "_" + "".join(TOKEN[key](value) for key, value in chosen.items())
        out.append((name, {**COMMON, **batch.fixed, **chosen}))
    return out


def deltas(settings: dict) -> list[str]:
    """The settings as ``name=value`` strings, in the order ``submit`` wants them."""
    return [f"{key}={value}" for key, value in sorted(settings.items())]


def duplicates() -> list[tuple[str, str]]:
    """Runs whose settings exactly repeat an earlier batch's run.

    The `color` batch is centred on a cell the `parent` batch already contains, so four
    configurations appear twice. They are kept rather than silently dropped -- the seed is
    fixed, so a repeat is a free reproducibility check -- but the plan has to say so, or
    someone will later read four identical runs as four independent measurements.
    """
    seen: dict[tuple, str] = {}
    repeats = []
    for batch in BATCHES.values():
        for name, settings in runs(batch):
            key = tuple(sorted((k, str(v)) for k, v in settings.items()))
            if key in seen:
                repeats.append((name, seen[key]))
            else:
                seen[key] = name
    return repeats


def show(chosen: list[str]) -> list[tuple[str, dict]]:
    """Print the plan for the named batches and return every run in it."""
    planned: list[tuple[str, dict]] = []
    for name in chosen:
        batch = BATCHES[name]
        rows = runs(batch)
        print(f"\n{name} -- {len(rows)} runs: {batch.purpose}")
        for key, values in batch.axes.items():
            print(f"    {key:<24} {', '.join(str(v) for v in values)}")
        print()
        for job, settings in rows:
            print(f"    {job:<16} {' '.join(deltas(settings))}")
        planned.extend(rows)
    return planned


def launch(planned: list[tuple[str, dict]], resources: dict, limit: int | None) -> None:
    """Validate every delta, stage HEAD once, then queue the runs."""
    for _, settings in planned:
        validate_deltas(deltas(settings))
    chosen = planned[:limit] if limit else planned
    print(f"\nvalidated {len(planned)} runs; submitting {len(chosen)}")
    revision = sync()
    done = 0
    try:
        for name, settings in chosen:
            flat = tuple(part for delta in deltas(settings) for part in ("--set", delta))
            submit(Job(name=name, args=flat, **resources), revision=revision)
            done += 1
    finally:
        # A qsub that fails halfway leaves real jobs in the queue. Saying how many is the
        # difference between resubmitting the remainder and duplicating the whole batch.
        print(f"\nsubmitted {done} of {len(chosen)} ({len(chosen) - done} not queued)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "batch",
        nargs="?",
        default="all",
        choices=("all", *BATCHES),
        help="which batch to plan or submit (default: all three, planning only)",
    )
    parser.add_argument("--submit", action="store_true", help="actually queue the runs")
    parser.add_argument("--limit", type=int, default=None, help="queue only the first N")
    parser.add_argument("--ncpus", type=int, default=Job.ncpus)
    parser.add_argument("--mem", type=int, default=Job.mem, help="GB")
    parser.add_argument("--scratch", type=int, default=Job.scratch, help="GB")
    parser.add_argument("--walltime", default=Job.walltime)
    parser.add_argument("--queue", default=None)
    args = parser.parse_args(argv)

    chosen = list(BATCHES) if args.batch == "all" else [args.batch]
    planned = show(chosen)

    hours = args.walltime.split(":")[0]
    print(f"\n{len(planned)} runs, {args.ncpus} cpus each, walltime {args.walltime}")
    print(f"worst case {len(planned) * args.ncpus * int(hours)} core-hours if none stops early")
    for repeat, original in duplicates():
        if any(repeat == name for name, _ in planned):
            print(f"note: {repeat} repeats {original} exactly (same seed: a replicate)")

    if not args.submit:
        print("\nnothing submitted. Add --submit to queue these.")
        return 0
    launch(
        planned,
        {
            "ncpus": args.ncpus,
            "mem": args.mem,
            "scratch": args.scratch,
            "walltime": args.walltime,
            "queue": args.queue,
        },
        args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
