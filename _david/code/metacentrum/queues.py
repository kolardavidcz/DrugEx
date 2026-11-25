"""Queue depth for the three MetaCentrum lanes this thesis uses.

    python -m _david.code.metacentrum.queues
    python -m _david.code.metacentrum.queues --json

It reads ``qstat -Qf`` on the login node and prints, for ``q_4h``, ``q_1d`` and ``q_2d``,
how many jobs are running, queued and held, plus each queue's walltime ceiling. Depth moves
hourly: use it to decide when to submit, not as a promise about the wait.

``q_1d`` is the default training lane (24 h walltime routes there). ``q_4h`` is for quick
probes only — a training run cannot finish in four hours — and ``q_2d`` is only for a
minimised run whose honest budget exceeds a day.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from dataclasses import dataclass

from _david.code.metacentrum.submit import DEFAULT_QUEUE, remote

#: The lanes worth watching. Other routed queues are shallower or need a grant.
QUEUES = ("q_4h", "q_1d", "q_2d")

ROLE = {
    "q_4h": "probe/inspection only -- NOT for training",
    "q_1d": "DEFAULT -- every training run",
    "q_2d": "only a minimised run honestly >24 h",
}


@dataclass(frozen=True)
class QueueDepth:
    name: str
    running: int
    queued: int
    held: int
    total: int
    max_walltime: str

    @property
    def pressure(self) -> float | None:
        """Queued per running job: a rough depth comparison, not an ETA."""
        if not self.running:
            return None
        return self.queued / self.running


def parse_state_count(value: str) -> dict[str, int]:
    """``Transit:0 Queued:3873 Running:1581`` -> ``{"Queued": 3873, ...}``."""
    states: dict[str, int] = {}
    for token in value.split():
        key, separator, number = token.partition(":")
        if not separator:
            continue
        try:
            states[key] = int(number)
        except ValueError:
            continue
    return states


def parse_queues(text: str) -> list[QueueDepth]:
    """Parse long-format ``qstat -Qf`` blocks (including tab-indented continuations)."""
    fields: list[dict[str, str]] = []
    current: dict[str, str] = {}
    last_key: str | None = None
    for line in text.splitlines():
        if line.startswith("Queue:"):
            if current:
                fields.append(current)
            current = {"Queue": line.split(":", 1)[1].strip()}
            last_key = None
            continue
        if not current:
            continue
        if last_key is not None and line.startswith("\t"):
            current[last_key] += line[1:].rstrip()
            continue
        match = re.match(r"\s+([A-Za-z_][\w.\-]*)\s*=\s*(.*)$", line)
        if match:
            last_key = match.group(1)
            current[last_key] = match.group(2).strip()
    if current:
        fields.append(current)

    depths = []
    for values in fields:
        states = parse_state_count(values.get("state_count", ""))
        running = states.get("Running", 0)
        queued = states.get("Queued", 0)
        held = states.get("Held", 0)
        try:
            total = int(values.get("total_jobs") or sum(states.values()))
        except ValueError:
            total = sum(states.values())
        depths.append(
            QueueDepth(
                name=values.get("Queue", "?"),
                running=running,
                queued=queued,
                held=held,
                total=total,
                max_walltime=values.get("resources_max.walltime", "?"),
            )
        )
    return depths


def fetch() -> list[QueueDepth]:
    """Every queue, then keep the three lanes; a retired queue is reported as missing."""
    depth = {entry.name: entry for entry in parse_queues(remote("qstat -Qf"))}
    return [
        depth.get(name, QueueDepth(name=name, running=0, queued=0, held=0, total=0,
                                   max_walltime="MISSING"))
        for name in QUEUES
    ]


def show(depths: list[QueueDepth]) -> None:
    print(f"{'QUEUE':<7} {'RUNNING':>8} {'QUEUED':>7} {'HELD':>5} {'TOTAL':>7} "
          f"{'MAX WALLTIME':>12} {'QUEUED/RUNNING':>14}")
    for entry in depths:
        pressure = "n/a" if entry.pressure is None else f"{entry.pressure:.1f}x"
        print(
            f"{entry.name:<7} {entry.running:>8} {entry.queued:>7} {entry.held:>5} "
            f"{entry.total:>7} {entry.max_walltime:>12} {pressure:>14}"
        )
    print()
    for name in QUEUES:
        marker = "  <-- default" if name == DEFAULT_QUEUE else ""
        print(f"{name:<6} {ROLE[name]}{marker}")
    print()
    print("q_1d is the default: the 24 h walltime routes there and every training run asks")
    print("for it. q_4h is for probes only. q_2d is only for a run that honestly needs >24 h.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    depths = fetch()
    if args.json:
        print(json.dumps([dataclasses.asdict(entry) for entry in depths], indent=2))
    else:
        show(depths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
