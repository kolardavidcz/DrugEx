"""The planned sweep, as batches of runs.

    python -m _david.code.metacentrum.campaign                 # every batch, nothing submitted
    python -m _david.code.metacentrum.campaign parent          # one batch, as it would be sent
    python -m _david.code.metacentrum.campaign parent --submit  # stage HEAD once, then qsub it

Nothing leaves this machine without ``--submit``; the default prints the plan. Submission
validates every delta of the batch first, so a typo stops the campaign before the first
``qsub`` rather than after fifty of them.

A batch is a full factorial over its axes and nothing else: two runs differ only in the axes
their names spell out, and every setting another batch varies is pinned in ``COMMON``.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

from _david.code.metacentrum.submit import (
    STORAGE,
    Job,
    remote,
    submit,
    sync,
    validate_deltas,
    validate_resources,
)

# Each is an axis some *other* batch varies, so a spec states the whole experiment.
COMMON: dict[str, object] = {
    "scheme": "PRCD",
    "minimize": True,
    "reference_conformers": 4,
    "max_conformers": 20,
    "color_ff": "fdef",
    "alignment_shape_weight": 0.5,
}

# qstat shows the name and nothing else; floats become hundredths because a dot in a PBS
# job name asks for trouble.
TOKEN = {
    "minimize": lambda v: "m1" if v else "m0",
    "epsilon": lambda v: f"e{round(v * 100):02d}",
    "reference_conformers": lambda v: f"r{v}",
    "max_conformers": lambda v: f"c{v}",
    "include_qed": lambda v: "q1" if v else "q0",
    "color_ff": lambda v: {"default": "fd", "fdef": "ff", "oe_mills_dean": "fo"}[v],
    "alignment_shape_weight": lambda v: f"s{round(v * 100):02d}",
}

#: The planned grid includes the heaviest cell (~40 GB around epoch 30), so it buys the
#: slower-to-place 64 GB request. A 32 GB ad-hoc run caps its worker pool instead; pass
#: `--mem 32` to a batch only when every cell is known to stay below it.
MEMORY_GB = 64


@dataclass(frozen=True)
class Batch:
    """One full factorial; ``axes`` is ordered and fixes the job names."""

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
    "prtd": Batch(
        prefix="t",
        purpose=(
            "DrugEx's Tanimoto-distance ranking instead of crowding distance. The parent "
            "grid is PRCD throughout, so nothing in it prices the one native mechanism "
            "that puts chemical diversity into the reward: PRTD ranks the molecules of a "
            "Pareto front by ECFP6 distance, so one far from its neighbours outranks one "
            "in a crowd. Run at the relaxed centre with a single reference, which is "
            "where the parent grid's own runs collapsed onto a single chemical series"
        ),
        fixed={"scheme": "PRTD", "reference_conformers": 1},
        axes={
            "epsilon": (0.1, 0.05),
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
    # The WeightedSum sweep: every batch below runs WS, where the thresholds are part
    # of the reward (DrugEx builds the weights from them), so each axis is priced under
    # the scheme it changes the most.
    "wso": Batch(
        prefix="wso",
        purpose=(
            "WeightedSum on the relaxed bank: exploration breadth against conformer "
            "count, single reference"
        ),
        fixed={"scheme": "WS", "minimize": True, "reference_conformers": 1},
        axes={
            "epsilon": (0.2, 0.1, 0.05, 0.0),
            "max_conformers": (10, 20),
            "include_qed": (False, True),
        },
    ),
    "wsu": Batch(
        prefix="wsu",
        purpose=(
            "WeightedSum without relaxation: raw ETKDG poses, the four-conformer "
            "bank and the larger candidate counts"
        ),
        fixed={"scheme": "WS", "minimize": False, "reference_conformers": 4},
        axes={
            "epsilon": (0.1, 0.05, 0.0),
            "max_conformers": (20, 30),
            "include_qed": (False, True),
        },
    ),
    "wcm": Batch(
        prefix="wcm",
        purpose="colour typing under WeightedSum, relaxed single-reference centre",
        fixed={"scheme": "WS", "minimize": True, "reference_conformers": 1},
        axes={
            "epsilon": (0.2, 0.1, 0.05),
            "color_ff": ("default", "fdef", "oe_mills_dean"),
            "include_qed": (False, True),
        },
    ),
    "wcw": Batch(
        prefix="wcw",
        purpose=(
            "colour/shape weight split under WeightedSum, default colour typing, "
            "single reference"
        ),
        fixed={"scheme": "WS", "minimize": True, "reference_conformers": 1, "color_ff": "default"},
        axes={
            "epsilon": (0.2, 0.1, 0.05),
            "alignment_shape_weight": (0.5, 0.3, 0.2),
            "include_qed": (False, True),
        },
    ),
}


def runs(batch: Batch) -> list[tuple[str, dict]]:
    """One batch as ``(job name, settings)`` pairs."""
    plan = []
    for values in itertools.product(*batch.axes.values()):
        chosen = dict(zip(batch.axes, values))
        name = batch.prefix + "_" + "".join(TOKEN[key](value) for key, value in chosen.items())
        plan.append((name, {**COMMON, **batch.fixed, **chosen}))
    return plan


def deltas(settings: dict) -> list[str]:
    return [f"{key}={value}" for key, value in sorted(settings.items())]


def duplicates() -> list[tuple[str, str]]:
    """Runs repeating an earlier batch's settings: replicates, never independent."""
    seen: dict[tuple, str] = {}
    repeats = []
    for batch in BATCHES.values():
        for name, settings in runs(batch):
            key = tuple(sorted((key, str(value)) for key, value in settings.items()))
            if key in seen:
                repeats.append((name, seen[key]))
            else:
                seen[key] = name
    return repeats


def show(chosen: list[str], memory_gb: int) -> list[tuple[str, dict]]:
    """Print the plan; returns every run in it."""
    planned: list[tuple[str, dict]] = []
    for name in chosen:
        batch = BATCHES[name]
        rows = runs(batch)
        print(f"\n{name} -- {len(rows)} runs: {batch.purpose}")
        for key, values in batch.axes.items():
            print(f"    {key:<24} {', '.join(str(value) for value in values)}")
        print()
        for job, settings in rows:
            print(f"    {job:<16} {memory_gb:>3} GB  {' '.join(deltas(settings))}")
        planned.extend(rows)
    return planned


def already_submitted() -> set[str]:
    """Job names with a spec on storage; a spec outlives the queue entry."""
    listing = remote(f"ls {STORAGE}/jobs 2>/dev/null || true")
    return {line.strip() for line in listing.splitlines() if line.strip()}


def launch(
    planned: list[tuple[str, dict]], resources: dict, limit: int | None, again: bool = False
) -> None:
    """Validate every delta, stage HEAD once, then queue the runs."""
    for _, settings in planned:
        validate_deltas(deltas(settings))
    repeats = sorted(already_submitted() & {name for name, _ in planned}) if not again else []
    if repeats:
        print(f"\nskipping {len(repeats)} already submitted: {' '.join(repeats[:8])}")
        print("    (pass --again to queue them a second time)")
        planned = [(name, settings) for name, settings in planned if name not in set(repeats)]
    if not planned:
        print("\nevery run of this batch was submitted before; nothing to do")
        return

    chosen = planned[:limit] if limit else planned
    print(f"\nvalidated {len(planned)} runs; submitting {len(chosen)}")
    revision = sync()
    done = 0
    try:
        for name, settings in chosen:
            args = tuple(part for delta in deltas(settings) for part in ("--set", delta))
            try:
                job = Job(name=name, args=args, **resources)
            except ValueError as error:
                raise SystemExit(f"invalid job {name}: {error}") from None
            submit(job, revision=revision)
            done += 1
    finally:
        # A qsub failing halfway leaves real jobs queued; the count allows a resume.
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
        help="which batch to plan or submit (default: all of them, planning only)",
    )
    parser.add_argument("--submit", action="store_true", help="actually queue the runs")
    parser.add_argument("--limit", type=int, default=None, help="queue only the first N")
    parser.add_argument(
        "--again", action="store_true", help="queue runs that were submitted before"
    )
    parser.add_argument("--ncpus", type=int, default=Job.ncpus)
    parser.add_argument(
        "--mem",
        type=int,
        default=MEMORY_GB,
        help=f"GB for every run; default {MEMORY_GB} because the grid includes the ~40 GB "
        "cell. `--mem 32` places faster but the job body caps workers to fit it",
    )
    parser.add_argument("--scratch", type=int, default=Job.scratch_gb, help="GB")
    parser.add_argument("--walltime", default=Job.walltime)
    parser.add_argument("--queue", default=None)
    args = parser.parse_args(argv)
    validate_resources(args.ncpus, args.mem, args.scratch, args.walltime, args.queue)

    planned = show(list(BATCHES) if args.batch == "all" else [args.batch], args.mem)
    hours = int(args.walltime.split(":")[0])
    print(
        f"\n{len(planned)} runs, {args.ncpus} cpus each, {args.mem} GB each, "
        f"walltime {args.walltime}"
    )
    print(f"worst case {len(planned) * args.ncpus * hours} core-hours if none stops early")
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
            "mem_gb": args.mem,
            "scratch_gb": args.scratch,
            "walltime": args.walltime,
            "queue": args.queue,
        },
        args.limit,
        args.again,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
