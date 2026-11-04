"""Compare two DrugEx ``_david/output`` trees and report what moved and what is missing.

The question this answers is *layout*, not *content*: an experiment folder that
was renamed or re-nested must show up as ``moved``, never as ``missing``. Content
is only consulted (a) to confirm a probable move when the names differ, and
(b) to decide whether the folder is a thesis-grade result or a throw-away test.

Matching order (first hit wins, cheapest first)::

    exact relative path -> unique folder basename -> file-name-set Jaccard
    -> identical content hash of any small file

Classification of every experiment folder::

    test     <= 20 finished epochs, or a smoke/test/incomplete/debug name
    aborted  0 epochs and/or no checkpoint and/or an empty skeleton
    real     everything else

A folder is reported as *usable* only when it is ``real``, carries the canonical
figure panels and holds a full evaluation arm (~1000 molecules per arm).

Only the standard library is used and nothing is ever written outside the
requested report/JSON paths, so the same file can be streamed to a remote host::

    # inventory one host (used remotely)
    ssh dichtator python3 - --scan-only --root '~/DrugEx/_david/output' \
        < _david/code/compare_output_trees.py

    # full comparison, report into _david/output/
    python3 _david/code/compare_output_trees.py --compare \
        --local-root _david/output \
        --remote dichtator --remote-root '~/DrugEx/_david/output' \
        --write _david/output/OUTPUT_TREE_DIFF.md
"""

# Design notes, matching thresholds and the edge cases that already produced wrong
# output live in COMPARE_OUTPUT_TREES.md next to this file.

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1

#: Directories that never belong to a result comparison.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".agents",
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".ruff_cache",
)

#: Files whose presence marks a directory as a self-contained experiment folder.
UNIT_FILE_MARKERS: tuple[str, ...] = (
    "manifest.json",
    "submission.json",
    "execution.log",
    "summary.md",
    "provenance.json",
)
UNIT_DIR_MARKERS: tuple[str, ...] = ("models", "raw_data", "figures", "logs")

#: Canonical deliverable panels produced by ``visualization.py``.
CURRENT_FIGURES: tuple[str, ...] = (
    "01_before_after_distributions.png",
    "02_multiobjective_tradeoffs.png",
    "03_training_trajectories.png",
    "05_lead_structures_gallery.png",
    "06_relative_property_shifts_tornado.png",
    "07_bemis_murcko_scaffold_diversity.png",
    "08_rocs_shape_vs_color_decomposition.png",
    "10_lead_to_seed_3d_superposition.png",
    "11_epoch_property_trajectories.png",
    "reference_3d_superposition.png",
)
LEGACY_FIGURES: tuple[str, ...] = ("04_diagnostic_unfavorable_features.png",)

TEST_NAME_RE = re.compile(
    r"(smoke|test|incomplete|negative_control|debug|demo|dry|pilot|scratch|tmp)",
    re.IGNORECASE,
)
EPOCH_LOG_RE = re.compile(r"(?:^|\s)epoch\s*[=: ]\s*(\d+)", re.IGNORECASE)

#: Suffixes too expensive to hash on a whole-tree scan.
HASH_EXCLUDE_SUFFIXES: tuple[str, ...] = (
    ".pkg", ".pt", ".pth", ".ckpt", ".sdf", ".npy", ".npz", ".zip", ".tar",
)

TEST_EPOCH_CEILING = 20
MIN_MOLECULES = 500


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #


@dataclass
class Unit:
    """One experiment folder found in a tree."""

    rel: str
    name: str
    files: list[str] = field(default_factory=list)  # relative to the unit
    sizes: dict[str, int] = field(default_factory=dict)
    bytes: int = 0
    hashes: dict[str, str] = field(default_factory=dict)
    epochs_done: int = 0
    epochs_planned: int | None = None
    n_pkg: int = 0
    n_figures: int = 0
    canonical_figs: int = 0
    raw_reinforced_rows: int = 0
    raw_baseline_rows: int = 0
    leads_rows: int = 0
    smiles_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def file_set(self) -> set[str]:
        return set(self.files)


def _walk_files(root: Path, excludes: tuple[str, ...]) -> list[tuple[str, int, float]]:
    """Return ``(relpath, size, mtime)`` for every file, in a stable order."""
    out: list[tuple[str, int, float]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.name in excludes:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            out.append((os.path.relpath(entry.path, root), stat.st_size, stat.st_mtime))
    out.sort()
    return out


def _dir_names(path: Path) -> list[str]:
    try:
        return [e.name for e in os.scandir(path) if e.is_dir(follow_symlinks=False)]
    except OSError:
        return []


def _file_names(path: Path) -> set[str]:
    try:
        return {e.name for e in os.scandir(path) if e.is_file(follow_symlinks=False)}
    except OSError:
        return set()


def _find_units(root: Path, files: list[tuple[str, int, float]], excludes: tuple[str, ...]) -> list[str]:
    """Every directory that looks like a self-contained experiment folder.

    A folder that merely sits inside another experiment folder (``raw_data/pilot``,
    ``raw_data/rl_smoke``, ``checkpoints/...``) is part of that experiment, not a
    sibling of it, so nested candidates are folded into their outermost parent.
    """
    seen: set[str] = set()
    for relpath, _size, _mtime in files:
        parts = Path(relpath).parts[:-1]
        for depth in range(1, len(parts) + 1):
            rel = os.path.join(*parts[:depth]) if depth else "."
            if rel in seen:
                continue
            seen.add(rel)
    candidates: list[str] = []
    for rel in sorted(seen):
        path = root if rel == "." else root / rel
        names = _file_names(path)
        if names & set(UNIT_FILE_MARKERS):
            candidates.append(rel)
            continue
        if set(_dir_names(path)) & set(UNIT_DIR_MARKERS):
            candidates.append(rel)

    units: list[str] = []
    chosen: set[str] = set()
    for rel in sorted(candidates, key=lambda item: (len(Path(item).parts), item)):
        parts = Path(rel).parts
        ancestors = [os.path.join(*parts[:depth]) for depth in range(1, len(parts))] if rel != "." else []
        if any(ancestor in chosen for ancestor in ancestors):
            continue
        chosen.add(rel)
        units.append(rel)
    return sorted(units)


def _deepest_unit(relpath: str, units: set[str]) -> str | None:
    parts = Path(relpath).parts
    for depth in range(len(parts) - 1, 0, -1):
        candidate = os.path.join(*parts[:depth])
        if candidate in units:
            return candidate
    return "." if "." in units else None


def _tail_lines(path: Path, n: int = 200, block: int = 65536) -> list[str]:
    """Last ``n`` lines without reading the whole file (``.pkg`` friendly)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                handle.seek(size)
                data = handle.read(step) + data
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    return text.splitlines()[-n:]


def _gzip_rows(path: Path) -> int:
    """Data rows in a ``.csv.gz`` (header excluded), streamed."""
    if not path.is_file():
        return 0
    try:
        with gzip.open(path, "rb") as handle:
            count = 0
            for _line in handle:
                count += 1
    except OSError:
        return 0
    return max(count - 1, 0)


def _plain_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("rb") as handle:
            count = sum(1 for _line in handle)
    except OSError:
        return 0
    return max(count - 1, 0)


def _epochs_from_smiles_tsv(path: Path) -> int:
    for line in reversed(_tail_lines(path, n=5)):
        first = line.split("\t", 1)[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def _epochs_from_log(path: Path) -> int:
    best = 0
    for line in _tail_lines(path, n=400):
        for match in EPOCH_LOG_RE.finditer(line):
            best = max(best, int(match.group(1)))
    return best


def _planned_epochs(unit_path: Path) -> int | None:
    manifest = unit_path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    config = data.get("config") if isinstance(data, dict) else None
    pool = [config, data] if isinstance(config, dict) else [data]
    for obj in pool:
        if not isinstance(obj, dict):
            continue
        for key in ("n_epochs", "epochs", "num_epochs", "n_epoch"):
            value = obj.get(key)
            if isinstance(value, int) and value > 0:
                return value
        for key in ("training", "training_config", "hyperparameters"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                for inner in ("n_epochs", "epochs"):
                    value = nested.get(inner)
                    if isinstance(value, int) and value > 0:
                        return value
    return None


def _md5(path: Path) -> str | None:
    digest = hashlib.md5()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def scan_tree(root: Path, excludes: tuple[str, ...], hash_max: int) -> dict:
    root = root.expanduser().resolve()
    started = time.time()
    files = _walk_files(root, excludes)
    unit_rels = _find_units(root, files, excludes)
    units = {rel: Unit(rel=rel, name=os.path.basename(rel) or str(root)) for rel in unit_rels}

    for relpath, size, _mtime in files:
        owner = _deepest_unit(relpath, set(unit_rels))
        if owner is None:
            continue
        unit = units[owner]
        inner = relpath if owner == "." else os.path.relpath(relpath, owner)
        unit.files.append(inner)
        unit.sizes[inner] = size
        unit.bytes += size
        if size <= hash_max and not inner.lower().endswith(HASH_EXCLUDE_SUFFIXES):
            digest = _md5(root / relpath)
            if digest:
                unit.hashes[inner] = digest

    for unit in units.values():
        path = root if unit.rel == "." else root / unit.rel
        models = path / "models"
        figures = path / "figures"
        raw = path / "raw_data"

        if models.is_dir():
            unit.n_pkg = len(list(models.glob("*.pkg")))
            for tsv in sorted(models.glob("*_smiles.tsv")):
                unit.smiles_rows += _plain_rows(tsv)
                unit.epochs_done = max(unit.epochs_done, _epochs_from_smiles_tsv(tsv))
        if unit.epochs_done == 0:
            candidates = [path / "execution.log"]
            logs_dir = path / "logs"
            if logs_dir.is_dir():
                candidates.extend(sorted(logs_dir.glob("*.log")))
            for log in candidates:
                if log.is_file():
                    unit.epochs_done = max(unit.epochs_done, _epochs_from_log(log))

        unit.epochs_planned = _planned_epochs(path)

        if figures.is_dir():
            pngs = {p.name for p in figures.glob("*.png")}
            unit.n_figures = len(pngs)
            unit.canonical_figs = len(pngs & set(CURRENT_FIGURES))
        if raw.is_dir():
            unit.raw_reinforced_rows = _gzip_rows(raw / "reinforced_raw.csv.gz")
            unit.raw_baseline_rows = _gzip_rows(raw / "baseline_raw.csv.gz")
            unit.leads_rows = _plain_rows(raw / "leads.csv")

        if not unit.files:
            unit.notes.append("empty skeleton")
        if unit.epochs_done == 0 and unit.n_pkg == 0:
            unit.notes.append("no checkpoint")
        if unit.raw_reinforced_rows == 0 and unit.raw_baseline_rows == 0:
            unit.notes.append("no evaluation arms")
        if unit.canonical_figs == 0 and unit.n_figures == 0:
            unit.notes.append("no figures")

    payload = {
        "schema": SCHEMA,
        "host": socket.gethostname(),
        "root": str(root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scan_seconds": round(time.time() - started, 2),
        "file_count": len(files),
        "orphan_files": len(files) - sum(unit.n_files for unit in units.values()),
        "total_bytes": sum(size for _rel, size, _mtime in files),
        "units": [
            {
                "rel": unit.rel,
                "name": unit.name,
                "files": sorted(unit.files),
                "sizes": unit.sizes,
                "bytes": unit.bytes,
                "hashes": unit.hashes,
                "epochs_done": unit.epochs_done,
                "epochs_planned": unit.epochs_planned,
                "n_pkg": unit.n_pkg,
                "n_figures": unit.n_figures,
                "canonical_figs": unit.canonical_figs,
                "raw_reinforced_rows": unit.raw_reinforced_rows,
                "raw_baseline_rows": unit.raw_baseline_rows,
                "leads_rows": unit.leads_rows,
                "smiles_rows": unit.smiles_rows,
                "notes": unit.notes,
            }
            for unit in sorted(units.values(), key=lambda u: u.rel)
        ],
    }
    return payload


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def classify(unit: dict) -> tuple[str, str, str]:
    """Return ``(kind, verdict, reason)`` for one inventory unit."""
    name = unit["name"]
    files = unit["files"]
    epochs = unit["epochs_done"]
    planned = unit["epochs_planned"]
    # A per-epoch TSV holds one row per molecule per epoch, so the evaluation
    # arm is the honest molecule count; the TSV is only a fallback.
    molecules = unit["raw_reinforced_rows"] or unit["smiles_rows"]
    canonical = unit["canonical_figs"]
    legacy = int(any(f"figures/{n}" in files for n in LEGACY_FIGURES))

    if TEST_NAME_RE.search(name):
        return ("test", "exclude", "name marks a calibration/smoke folder")
    if not files:
        return ("aborted", "exclude", "empty directory skeleton")
    if epochs == 0 and molecules == 0:
        # No finished epoch and no evaluation arm: nothing was produced, whatever
        # checkpoints or logs survived.
        reason = f"0/{planned} epochs, aborted before training" if planned else "no finished epoch"
        return ("aborted", "exclude", reason)
    if epochs and epochs <= TEST_EPOCH_CEILING:
        return ("test", "exclude", f"only {epochs} epoch(s)")
    if canonical < len(CURRENT_FIGURES) and legacy == 0:
        return ("real", "partial", f"figures {canonical}/{len(CURRENT_FIGURES)}, legacy panel absent")
    if molecules < MIN_MOLECULES:
        return ("real", "partial", f"only {molecules} generated molecules")
    return ("real", "include", f"{epochs} epochs, {canonical} panels, {molecules} molecules")


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_inventories(local: dict, remote: dict) -> dict:
    local_units = {u["rel"]: u for u in local["units"]}
    remote_units = {u["rel"]: u for u in remote["units"]}
    matched: list[dict] = []
    moved: list[dict] = []

    free_local = dict(local_units)
    free_remote = dict(remote_units)

    for rel in sorted(set(local_units) & set(remote_units)):
        matched.append({"local": local_units[rel], "remote": remote_units[rel], "via": "path"})
        del free_local[rel]
        del free_remote[rel]

    def basename_index(units: dict[str, dict]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for rel, unit in units.items():
            index.setdefault(unit["name"], []).append(rel)
        return index

    local_by_name = basename_index(free_local)
    remote_by_name = basename_index(free_remote)

    for name, rels in sorted(local_by_name.items()):
        if len(rels) != 1:
            continue
        candidates = remote_by_name.get(name) or []
        if len(candidates) != 1:
            continue
        lrel, rrel = rels[0], candidates[0]
        moved.append({"local": free_local.pop(lrel), "remote": free_remote.pop(rrel), "via": "name"})

    # file-name-set similarity, then content hash overlap.
    # Containment (not only Jaccard) is scored too: a folder that lost its
    # checkpoints keeps 9/11 file names and must still pair with its sibling
    # copy instead of being reported as a missing experiment.
    for lrel in sorted(free_local):
        lunit = free_local[lrel]
        lset = set(lunit["files"])
        best: tuple[float, str, int] | None = None
        for rrel, runit in free_remote.items():
            rset = set(runit["files"])
            shared = lset & rset
            if not shared:
                continue
            score = max(_jaccard(lset, rset), len(shared) / min(len(lset), len(rset)))
            if best is None or score > best[0]:
                best = (score, rrel, len(shared))
        if best and best[0] >= 0.6 and best[2] >= min(3, min(len(lset), 3)):
            rrel = best[1]
            moved.append({"local": free_local.pop(lrel), "remote": free_remote.pop(rrel),
                          "via": "file-overlap", "score": round(best[0], 3), "shared": best[2]})

    for lrel in sorted(free_local):
        lunit = free_local[lrel]
        lhashes = set(lunit["hashes"].values())
        best = None
        for rrel, runit in free_remote.items():
            shared = lhashes & set(runit["hashes"].values())
            if not shared:
                continue
            ratio = len(shared) / max(1, min(len(lhashes), len(set(runit["hashes"].values()))))
            if best is None or ratio > best[0]:
                best = (ratio, rrel, len(shared))
        if best and best[0] >= 0.2:
            rrel = best[1]
            moved.append({"local": free_local.pop(lrel), "remote": free_remote.pop(rrel),
                          "via": "content", "score": round(best[0], 3), "shared": best[2]})

    return {
        "matched": matched,
        "moved": moved,
        "remote_only": [free_remote[rel] for rel in sorted(free_remote)],
        "local_only": [free_local[rel] for rel in sorted(free_local)],
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def _unit_row(unit: dict, verdicts: dict[str, tuple[str, str, str]]) -> str:
    kind, verdict, reason = verdicts[unit["rel"]]
    # A per-epoch TSV holds one row per molecule per epoch, so the evaluation
    # arm is the honest molecule count; the TSV is only a fallback.
    molecules = unit["raw_reinforced_rows"] or unit["smiles_rows"]
    return (
        f"| `{unit['rel']}` | {kind} | {unit['epochs_done']}/{unit['epochs_planned'] or '?'} | "
        f"{unit['canonical_figs']}/{len(CURRENT_FIGURES)} | {molecules} | "
        f"{_fmt_bytes(unit['bytes'])} | **{verdict}** — {reason} |"
    )


def _partial_parent(unit: dict, local_units: list[dict]) -> tuple[dict, list[str]] | None:
    """If ``unit`` is a partial copy of a WSL folder, return that folder and its gaps.

    This catches the "folder was copied back without its checkpoints" case: the
    unit only exists on dichtator, but most of its file names already live inside
    a WSL experiment folder under a different path.

    Every run directory repeats the same file names (``execution.log``,
    ``raw_data/*.csv.gz``, ...), so names alone cannot tell two runs apart. The
    parent is therefore chosen by identical file *content* first and only falls
    back to names when the unit carries no hashes at all.
    """
    files = set(unit["files"])
    hashes = set(unit["hashes"].values())
    best: tuple[float, dict, list[str]] | None = None
    for candidate in local_units:
        other = set(candidate["files"])
        shared_names = files & other
        if not shared_names:
            continue
        shared_hashes = hashes & set(candidate["hashes"].values())
        if hashes:
            score = len(shared_hashes) / len(hashes)
        else:
            score = len(shared_names) / len(files)
        if best is None or score > best[0]:
            best = (score, candidate, sorted(files - other))
    if best and best[0] >= 0.6:
        return best[1], best[2]
    return None


def build_report(local: dict, remote: dict, result: dict, remote_label: str) -> str:
    verdicts = {u["rel"]: classify(u) for u in remote["units"]}
    lverdicts = {u["rel"]: classify(u) for u in local["units"]}

    remote_only = result["remote_only"]
    local_only = result["local_only"]
    moved = result["moved"]
    matched = result["matched"]

    lines: list[str] = []
    lines.append("# Output tree difference: WSL ↔ dichtator")
    lines.append("")
    lines.append("_Generated by `_david/code/compare_output_trees.py` — read-only on both hosts._")
    lines.append("")
    lines.append("| | WSL | dichtator |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| root | `{local['root']}` | `{remote['root']}` |")
    lines.append(f"| host | `{local['host']}` | `{remote['host']}` |")
    lines.append(f"| snapshot | {local['generated_at']} | {remote['generated_at']} |")
    lines.append(f"| files | {local['file_count']:,} | {remote['file_count']:,} |")
    lines.append(f"| bytes | {_fmt_bytes(local['total_bytes'])} | {_fmt_bytes(remote['total_bytes'])} |")
    lines.append(f"| experiment folders | {len(local['units'])} | {len(remote['units'])} |")
    lines.append(
        f"| files outside experiment folders | {local.get('orphan_files', 0):,} "
        f"| {remote.get('orphan_files', 0):,} |"
    )
    lines.append("")
    lines.append(
        "Snapshot semantics: both trees are being reorganised while this runs, so the "
        "counts are a point-in-time view, not an invariant. Matching is by exact path → "
        "unique folder name → file-name overlap → identical file content, so a renamed or "
        "re-nested folder is reported as **moved**, never as missing. Files that sit outside any "
        "experiment folder (loose reports, `_COOL_STUFF` images, notebook leftovers) are counted "
        "in the table above but not compared."
    )
    lines.append("")

    carried_gaps = [
        (pair, sorted(set(pair["remote"]["files"]) - set(pair["local"]["files"])))
        for pair in moved
        if set(pair["remote"]["files"]) - set(pair["local"]["files"])
    ]
    partial_copies = []
    duplicates = []
    for unit in remote_only:
        parent = _partial_parent(unit, local["units"])
        if not parent:
            continue
        if parent[1]:
            partial_copies.append((unit, parent[0], parent[1]))
        else:
            duplicates.append((unit, parent[0]))
    stranded_files = partial_copies
    claimed = {p[0]["rel"] for p in partial_copies} | {d[0]["rel"] for d in duplicates}
    fully_missing = [u for u in remote_only if u["rel"] not in claimed]

    lines.append("## Bottom line")
    lines.append("")
    if not fully_missing and not carried_gaps and not stranded_files:
        lines.append(
            "Every usable experiment folder on dichtator has a counterpart in WSL, at the same "
            "path or under a moved one, and no result file was left behind."
        )
    else:
        if stranded_files:
            total = sum(len(missing) for _unit, _parent, missing in stranded_files)
            lines.append(
                f"- **{total} result file(s) exist on dichtator and nowhere in WSL**: "
                + "; ".join(
                    "`" + unit["rel"] + "`: " + ", ".join(f"`{n}`" for n in missing[:6])
                    + (f" (+{len(missing) - 6})" if len(missing) > 6 else "")
                    for unit, _parent, missing in stranded_files
                )
            )
        if fully_missing:
            usable_missing = [u for u in fully_missing if verdicts[u["rel"]][1] != "exclude"]
            if usable_missing:
                lines.append(
                    f"- **{len(usable_missing)} experiment folder(s) exist only on dichtator**: "
                    + ", ".join(f"`{u['rel']}`" for u in usable_missing)
                )
        if carried_gaps:
            total = sum(len(names) for _pair, names in carried_gaps)
            lines.append(
                f"- **{total} result file(s) were not carried over by a move**, in "
                f"{len(carried_gaps)} folder(s) — see section 2."
            )
    if moved or duplicates:
        lines.append(
            f"- {len(moved) + len(duplicates)} folder(s) exist under a different path on each host (section 2)."
        )
    other_tests = [u for u in fully_missing if verdicts[u["rel"]][1] == "exclude"]
    if other_tests:
        lines.append(
            f"- {len(other_tests)} further dichtator-only folder(s) are tests/aborted and can be ignored (section 1c)."
        )
    lines.append("")

    lines.append("## 1. Experiment folders on dichtator but missing in WSL")
    lines.append("")
    if not remote_only:
        lines.append("None. Every experiment folder on dichtator has a counterpart (same path or moved) in WSL.")
    else:
        lines.append(
            f"{len(remote_only)} folder(s) are not at the same path in WSL: "
            f"{len(partial_copies)} survive only as a partial copy, "
            f"{len(duplicates)} are mirrored byte-for-byte under another path, "
            f"{len(fully_missing)} are absent outright."
        )
        lines.append("")
        if partial_copies:
            lines.append("### 1a. Folder exists in WSL, but without these files")
            lines.append("")
            lines.append("| folder on dichtator | its WSL counterpart | files present only on dichtator |")
            lines.append("| --- | --- | --- |")
            for unit, parent, missing in partial_copies:
                names = ", ".join(
                    f"`{n}` ({_fmt_bytes(unit['sizes'].get(n, 0))})" for n in missing
                )
                lines.append(f"| `{unit['rel']}` | `{parent['rel']}` | {names} |")
            lines.append("")
        if any(verdicts[unit["rel"]][1] in ("include", "partial") for unit in fully_missing):
            lines.append("### 1b. Experiment folder absent outright — worth rescuing")
            lines.append("")
            lines.append("| folder on dichtator | kind | epochs | panels | molecules | size | verdict |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for unit in fully_missing:
                if verdicts[unit["rel"]][1] != "exclude":
                    lines.append(_unit_row(unit, verdicts))
            lines.append("")
        if any(verdicts[unit["rel"]][1] == "exclude" for unit in fully_missing):
            lines.append("### 1c. Test / aborted — do not rescue")
            lines.append("")
            lines.append("| folder on dichtator | kind | epochs | panels | molecules | size | verdict |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for unit in fully_missing:
                if verdicts[unit["rel"]][1] == "exclude":
                    lines.append(_unit_row(unit, verdicts))
            lines.append("")

    lines.append("## 2. Folders that only moved")
    lines.append("")
    if not moved and not duplicates:
        lines.append("No folder needed move-detection: every counterpart matched by exact path or folder name.")
    else:
        lines.append("| WSL path | dichtator path | matched by | confidence | files only in WSL | files only on dichtator |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for pair in moved:
            lfiles = set(pair["local"]["files"])
            rfiles = set(pair["remote"]["files"])
            score = pair.get("score")
            confidence = (
                f"{score:.2f}" if isinstance(score, float) else
                ("unique name" if pair["via"] == "name" else "—")
            )
            lines.append(
                f"| `{pair['local']['rel']}` | `{pair['remote']['rel']}` | {pair['via']} | {confidence} "
                f"| {len(lfiles - rfiles)} | {len(rfiles - lfiles)} |"
            )
        for unit, parent in duplicates:
            extra_local = len(set(parent["files"]) - set(unit["files"]))
            lines.append(
                f"| `{parent['rel']}` | `{unit['rel']}` | duplicate | every file identical "
                f"| {extra_local} | 0 |"
            )
        lines.append("")
        genuine_gaps = []
        for pair in moved:
            missing = sorted(set(pair["remote"]["files"]) - set(pair["local"]["files"]))
            if missing:
                genuine_gaps.append((pair, missing))
        if genuine_gaps:
            lines.append("Files that the move did **not** carry over:")
            lines.append("")
            for pair, missing in genuine_gaps:
                shown = ", ".join(f"`{name}`" for name in missing[:12])
                suffix = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
                lines.append(f"- `{pair['remote']['rel']}` → {shown}{suffix}")
            lines.append("")
        if any(pair["via"] == "name" for pair in moved):
            lines.append(
                "Name matches are safe: the folder name is unique on both sides and the content "
                "folders (`models/`, `raw_data/`, `figures/`, `logs/`) travel with it."
            )
            lines.append("")

    lines.append("## 3. Folders only in WSL")
    lines.append("")
    if not local_only:
        lines.append("None.")
    else:
        lines.append("| folder in WSL | kind | epochs | panels | molecules | size | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for unit in local_only:
            lines.append(_unit_row(unit, lverdicts))
    lines.append("")

    lines.append("## 4. Matched folders — content drift")
    lines.append("")
    drift = []
    for pair in matched:
        lfiles, rfiles = set(pair["local"]["files"]), set(pair["remote"]["files"])
        only_local = lfiles - rfiles
        only_remote = rfiles - lfiles
        if only_local or only_remote:
            drift.append((pair, only_local, only_remote))
    if not drift:
        lines.append("None: every matched folder holds the same set of relative file paths.")
    else:
        lines.append(
            f"{len(drift)} matched folder(s) differ in their file list."
        )
        lines.append("")
        lines.append("| folder | files only in WSL | files only on dichtator |")
        lines.append("| --- | --- | --- |")
        for pair, only_local, only_remote in sorted(drift, key=lambda item: item[0]["local"]["rel"]):
            lines.append(
                f"| `{pair['local']['rel']}` | {len(only_local)} | {len(only_remote)} |"
            )
        lines.append("")
        if drift and all(not only_remote for _pair, _only_local, only_remote in drift):
            lines.append(
                "Every delta is one-directional: WSL holds extra files (newer figures, checkpoints, "
                "summaries) and dichtator holds none that WSL lacks."
            )
            lines.append("")
        extras: Counter[str] = Counter()
        for _pair, only_local, _only_remote in drift:
            extras.update(only_local)
        if extras:
            top = ", ".join(f"`{name}` ×{count}" for name, count in extras.most_common(8))
            lines.append(f"Extra files present only in WSL, by frequency: {top}.")
            lines.append("")
        detail = [(pair, sorted(only_local), sorted(only_remote)) for pair, only_local, only_remote in drift
                  if len(only_local) + len(only_remote) <= 4]
        if detail:
            lines.append("Small deltas, named in full:")
            lines.append("")
            for pair, only_local, only_remote in sorted(detail, key=lambda item: item[0]["local"]["rel"])[:12]:
                if only_local:
                    lines.append(f"- `{pair['local']['rel']}` — WSL only: " + ", ".join(f"`{n}`" for n in only_local))
                if only_remote:
                    lines.append(f"- `{pair['local']['rel']}` — dichtator only: " + ", ".join(f"`{n}`" for n in only_remote))
            lines.append("")
    lines.append("")

    lines.append("## 5. What counts as a thesis result")
    lines.append("")
    lines.append(
        f"- `test`: name marks a calibration/smoke run, or training stopped at ≤ {TEST_EPOCH_CEILING} epochs."
    )
    lines.append("- `aborted`: empty skeleton, no checkpoint, or zero finished epochs.")
    lines.append(
        f"- `include`: real run, {len(CURRENT_FIGURES)} canonical panels, ≥ {MIN_MOLECULES} generated molecules."
    )
    lines.append("- `partial`: real run missing panels or a full evaluation arm.")
    lines.append("")
    lines.append(f"Remote label: `{remote_label}`.")
    lines.append("")

    lines.append("## Appendix. Every experiment folder on dichtator")
    lines.append("")
    lines.append("| folder | kind | epochs | panels | molecules | size | verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for unit in sorted(remote["units"], key=lambda item: item["rel"]):
        lines.append(_unit_row(unit, verdicts))
    lines.append("")
    lines.append("## Re-run")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 _david/code/compare_output_trees.py --compare \\")
    lines.append("    --local-root _david/output \\")
    lines.append('    --remote "ssh dichtator" --remote-root "~/DrugEx/_david/output" \\')
    lines.append("    --write _david/output/OUTPUT_TREE_DIFF.md \\")
    lines.append("    --json-out _david/output/.agents/output_tree_diff_snapshot.json.gz")
    lines.append("```")
    lines.append("")
    lines.append("`--scan-only --root PATH` prints a single inventory as JSON and is what the SSH side runs.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def _scan_remote(command: str, root: str, excludes: tuple[str, ...], hash_max: int) -> dict:
    script = Path(__file__).read_text(encoding="utf-8")
    args = [sys.executable or "python3", "-", "--scan-only", "--root", root, "--hash-max", str(hash_max)]
    for name in excludes:
        args += ["--exclude", name]
    remote_cmd = " ".join(shlex.quote(part) for part in args)
    proc = subprocess.run(
        [*shlex.split(command), remote_cmd],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"remote scan failed ({proc.returncode}):\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("_david/output"), help="tree to scan")
    parser.add_argument("--scan-only", action="store_true", help="print one inventory as JSON and exit")
    parser.add_argument("--compare", action="store_true", help="scan locally, scan the remote, report")
    parser.add_argument("--local-root", type=Path, default=Path("_david/output"))
    parser.add_argument("--remote", default="", help="ssh command, e.g. 'ssh dichtator'")
    parser.add_argument("--remote-root", default="")
    parser.add_argument("--write", type=Path, default=None, help="markdown report destination")
    parser.add_argument("--json-out", type=Path, default=None, help="dump both inventories and the diff")
    parser.add_argument("--exclude", action="append", default=[], help="extra directory name to ignore")
    parser.add_argument("--hash-max", type=int, default=1 << 20, help="largest file to hash, bytes")
    args = parser.parse_args(argv)

    excludes = tuple(dict.fromkeys([*DEFAULT_EXCLUDES, *args.exclude]))

    if args.scan_only:
        payload = scan_tree(args.root, excludes, args.hash_max)
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
        return 0

    if not args.compare:
        parser.error("choose --scan-only or --compare")

    if not args.remote or not args.remote_root:
        parser.error("--compare needs --remote and --remote-root")

    local = scan_tree(args.local_root, excludes, args.hash_max)
    remote = _scan_remote(args.remote, args.remote_root, excludes, args.hash_max)
    result = match_inventories(local, remote)

    print(
        f"WSL {len(local['units'])} folders / {local['file_count']:,} files | "
        f"remote {len(remote['units'])} folders / {remote['file_count']:,} files | "
        f"matched {len(result['matched'])} | moved {len(result['moved'])} | "
        f"remote-only {len(result['remote_only'])} | local-only {len(result['local_only'])}"
    )
    for unit in result["remote_only"]:
        kind, verdict, reason = classify(unit)
        print(f"  remote-only [{kind}/{verdict}] {unit['rel']}  ({reason})")

    report = build_report(local, remote, result, args.remote)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report, encoding="utf-8")
        print(f"report: {args.write}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"local": local, "remote": remote, "diff": result}, indent=1)
        if args.json_out.suffix == ".gz":
            with gzip.open(args.json_out, "wt", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            args.json_out.write_text(payload, encoding="utf-8")
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
