"""Normalize ``_david/output`` into timestamped RL run folders plus category folders.

The audit, the dashboard and every figure address a run by directory name, so the
layout is a contract: each RL arm lives directly under ``_david/output`` as
``YYYYmmdd_HHMM_<note>``, and everything that is not an RL arm lives in a category
folder (``conformer/``, ``datasets/``, ``archive/``, ``_metacentrum_jobs/``).

This module turns that contract into a reproducible, logged operation instead of a
one-off ``mv`` session: :func:`plan` never touches the filesystem and :func:`apply`
executes the plan and appends every action to ``ORGANIZATION_LOG.md``. Planning an
already normalized tree returns an empty plan, so the operation is idempotent.

Classification is data-driven, not name-guessing:

* **RL arm** - the directory holds ``manifest.json`` with a ``config`` block, or a
  ``raw_data/reinforced_raw.csv.gz`` arm, or ``models/SequenceRNN_reinforced_*``, so
  pre-manifest runs from the grid era are still recognized.
* **Group** - a non-conforming directory whose children are RL arms (the historical
  ``20261027_reinforcement_learning/`` container). Its children are flattened into the
  output root, its loose files move to ``archive/group_files/``.
* **Category** - an explicit, curated mapping (:data:`CATEGORY_MOVES`) for study trees
  that are not RL arms.
* **Duplicate** - same ``manifest.json`` and same reinforced arm by SHA-256. The copy
  that already conforms and sits shallowest wins; the other is archived, never deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Category folders that are part of the layout and are never reorganized themselves.
CATEGORY_ROOTS: tuple[str, ...] = (
    "archive",
    "conformer",
    "datasets",
    "_metacentrum_jobs",
    ".agents",
)

#: Non-RL trees that belong in a category folder. Curated so nothing is guessed.
CATEGORY_MOVES: dict[str, str] = {
    "20261028_conformer_self_similarity": "conformer/self_similarity",
}

GROUP_ARCHIVE_DIR = "archive/group_files"
DUPLICATE_ARCHIVE_DIR = "archive/duplicate_runs"
EMPTY_ARCHIVE_DIR = "archive/empty_runs"

_STAMP = re.compile(r"^(\d{8})_(\d{4})_")
_RUN_FILE = "manifest.json"
_REINFORCED_ARM = Path("raw_data") / "reinforced_raw.csv.gz"
_MODEL_GLOB = "models/SequenceRNN_reinforced*"
LOG_NAME = "ORGANIZATION_LOG.md"


def has_stamp(name: str) -> bool:
    """True when a directory name already starts with ``YYYYmmdd_HHMM_``."""
    return _STAMP.match(name) is not None


def standard_name(name: str, started: datetime | None) -> str:
    """Prefix a missing ``YYYYmmdd_HHMM_`` stamp, keeping the original note intact."""
    if has_stamp(name) or started is None:
        return name
    return f"{started:%Y%m%d_%H%M}_{name}"


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_rl_run(path: Path) -> bool:
    """True when ``path`` holds the artifacts of an RL arm (manifest, arm or models)."""
    if not path.is_dir():
        return False
    manifest = path / _RUN_FILE
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict) and "config" in payload:
            return True
    if (path / _REINFORCED_ARM).is_file():
        return True
    return any(path.glob(_MODEL_GLOB))


def started_at(path: Path) -> datetime | None:
    """Earliest artifact mtime: the reference SDF is written before Stage 1 starts."""
    stamps = [
        item.stat().st_mtime
        for item in path.rglob("*")
        if item.is_file() and not item.name.startswith(".")
    ]
    return datetime.fromtimestamp(min(stamps)) if stamps else None


def content_key(path: Path) -> tuple[str, str]:
    """Identity of an arm: manifest plus the delivered population."""
    return (_sha256(path / _RUN_FILE), _sha256(path / _REINFORCED_ARM))


def is_empty_tree(path: Path) -> bool:
    return not any(item.is_file() for item in path.rglob("*"))


@dataclass
class Action:
    """One filesystem action in a plan."""

    kind: str  # flatten | group_file | category | duplicate | rename | empty
    source: Path
    target: Path
    reason: str

    def row(self, root: Path) -> str:
        def rel(path: Path) -> str:
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return path.as_posix()

        return f"| {self.kind} | `{rel(self.source)}` | `{rel(self.target)}` | {self.reason} |"


@dataclass
class Plan:
    """An ordered, inspectable list of actions; empty when the tree already conforms."""

    root: Path
    actions: list[Action] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.actions)

    def render(self, actions: list[Action] | None = None) -> str:
        chosen = self.actions if actions is None else actions
        if not chosen:
            return "Nothing to do: the output tree already follows the layout contract.\n"
        lines = ["| action | from | to | why |", "| :--- | :--- | :--- | :--- |"]
        lines += [action.row(self.root) for action in chosen]
        return "\n".join(lines) + "\n"


def plan(root: Path | str) -> Plan:
    """Compute the normalization plan without touching the filesystem."""
    root = Path(root)

    result = Plan(root=root)
    entry_dirs = [
        entry
        for entry in sorted(root.iterdir())
        if entry.is_dir() and entry.name not in CATEGORY_ROOTS
    ]

    # : logical final path -> directory that currently holds that content.
    real: dict[Path, Path] = {}

    # 1. Dissolve non-conforming group folders that only contain RL arms.
    for entry in entry_dirs:
        if has_stamp(entry.name) or is_rl_run(entry):
            continue
        children = [child for child in sorted(entry.iterdir()) if child.is_dir()]
        # A stamped child is a suite that already follows the contract; flatten it too so
        # the container can dissolve instead of keeping one folder nested two levels deep.
        run_children = [
            child for child in children if is_rl_run(child) or has_stamp(child.name)
        ]
        if not run_children:
            continue
        for child in run_children:
            target = root / child.name
            result.actions.append(
                Action(
                    kind="flatten",
                    source=child,
                    target=target,
                    reason="RL arm inside a non-conforming group folder",
                )
            )
            real[target] = child
        for loose in sorted(entry.iterdir()):
            if loose.is_dir():
                continue
            result.actions.append(
                Action(
                    kind="group_file",
                    source=loose,
                    target=root / GROUP_ARCHIVE_DIR / loose.name,
                    reason=f"loose file of the dissolved group `{entry.name}`",
                )
            )

    # 2. Curated category moves for non-RL study trees.
    for entry in entry_dirs:
        target_rel = CATEGORY_MOVES.get(entry.name)
        if target_rel is None or is_rl_run(entry):
            continue
        result.actions.append(
            Action(
                kind="category",
                source=entry,
                target=root / target_rel,
                reason="not an RL arm; belongs in its category folder",
            )
        )

    # 3. RL arms that will sit at the output root after flattening.
    moved_sources = {action.source for action in result.actions}
    rl_candidates = [
        entry
        for entry in entry_dirs
        if is_rl_run(entry) and entry.name not in CATEGORY_MOVES
    ]
    rl_logical = [entry for entry in rl_candidates]
    for logical, source in real.items():
        rl_logical.append(logical)
        rl_candidates.append(source)

    # 4. Duplicates by content: keep the conforming, shallowest copy.
    groups: dict[tuple[str, str], list[Path]] = {}
    for logical in rl_logical:
        key = content_key(real.get(logical, logical))
        if key == ("", ""):
            continue
        groups.setdefault(key, []).append(logical)

    # Anything already archived wins over a live copy: an arm that was archived for a
    # reason must not reappear at the root just because it was copied there once.
    archive_index: dict[tuple[str, str], Path] = {}
    archive_root = root / "archive"
    ignored_archive = {
        Path(DUPLICATE_ARCHIVE_DIR).name,
        Path(GROUP_ARCHIVE_DIR).name,
        Path(EMPTY_ARCHIVE_DIR).name,
    }
    if archive_root.is_dir():
        for candidate in sorted(archive_root.iterdir()):
            if not candidate.is_dir() or candidate.name in ignored_archive:
                continue
            bucket = [candidate, *[d for d in candidate.iterdir() if d.is_dir()]]
            for sub in bucket:
                if not is_rl_run(sub):
                    continue
                key = content_key(sub)
                if key != ("", ""):
                    archive_index.setdefault(key, sub)

    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = sorted(
            members,
            key=lambda p: (not has_stamp(p.name), len(p.relative_to(root).parts), p.name),
        )[0]
        for other in members:
            if other == canonical:
                continue
            result.actions.append(
                Action(
                    kind="duplicate",
                    source=real.get(other, other),
                    target=root / DUPLICATE_ARCHIVE_DIR / other.name,
                    reason=f"byte-identical copy of `{canonical.name}`",
                )
            )

    for logical in rl_logical:
        source = real.get(logical, logical)
        if any(action.source == source for action in result.actions):
            continue
        archived = archive_index.get(content_key(source))
        if archived is None:
            continue
        result.actions.append(
            Action(
                kind="duplicate",
                source=source,
                target=root / DUPLICATE_ARCHIVE_DIR / logical.name,
                reason=f"byte-identical copy of `{archived.relative_to(root).as_posix()}`",
            )
        )

    # 5. Stamp every remaining RL arm that lacks a timestamp.
    handled = {action.source for action in result.actions}
    for logical in rl_logical:
        source = real.get(logical, logical)
        if source in handled or has_stamp(logical.name):
            continue
        result.actions.append(
            Action(
                kind="rename",
                source=source,
                target=root / standard_name(logical.name, started_at(source)),
                reason="RL arm without a YYYYmmdd_HHMM_ stamp",
            )
        )

    # 6. Empty leftovers go to the archive instead of cluttering the root.
    for entry in entry_dirs:
        if is_rl_run(entry) or entry.name in CATEGORY_MOVES:
            continue
        if entry in moved_sources or not is_empty_tree(entry):
            continue
        result.actions.append(
            Action(
                kind="empty",
                source=entry,
                target=root / EMPTY_ARCHIVE_DIR / entry.name,
                reason="empty directory with no artifacts",
            )
        )
    return result


def apply(root: Path | str, plan_obj: Plan | None = None) -> list[Action]:
    """Execute a plan and append the executed actions to ``ORGANIZATION_LOG.md``."""
    root = Path(root)
    plan_obj = plan_obj if plan_obj is not None else plan(root)
    executed = []
    for action in plan_obj.actions:
        if not action.source.exists():
            continue
        action.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(action.source), str(action.target))
        executed.append(action)

    # A dissolved group leaves its own (now empty) directory behind.
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in CATEGORY_ROOTS:
            continue
        if is_empty_tree(entry):
            shutil.rmtree(entry)
            executed.append(
                Action(
                    kind="empty",
                    source=entry,
                    target=root / EMPTY_ARCHIVE_DIR / entry.name,
                    reason="container left empty after its runs were moved",
                )
            )

    log = root / LOG_NAME
    previous = log.read_text().rstrip() if log.is_file() else "# Output organization log"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"\n## {stamp} — normalized to `YYYYmmdd_HHMM_<note>` + category folders\n\n"
    log.write_text(previous + "\n" + header + plan_obj.render(executed))
    return executed


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m _david.code.output_organization [plan|apply] [--root DIR]``."""
    import argparse

    from _david.code.config import OUTPUT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply"), nargs="?", default="plan")
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)

    target = plan(args.root)
    print(target.render())
    if args.action == "apply" and target:
        executed = apply(args.root, target)
        print(f"Executed {len(executed)} action(s); see {args.root / LOG_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
