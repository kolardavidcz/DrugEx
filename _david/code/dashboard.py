"""Generate one self-contained HTML overview of every RL run in ``output/``.

The static table in ``RL_RUN_AUDIT.md`` answers "what did we run"; it cannot
answer "what did we *not* run", "which runs are the same design" or "which
molecule is best across all arms". This module reads the normalized output tree
(``YYYYmmdd_HHMM_<note>`` RL arms only, never ``archive/`` or the category
folders), normalizes each manifest's config to today's field names, clusters the
runs by the settings that actually differ, and writes a single offline HTML file
with the runs, the done/not-done design grid, the clusters and a best-of-the-best
molecule table (inline SVG depictions, no CDN).

A note on "not run": an empty cell in the design grid means *no run in this
output tree* has that combination. It is a proposal, not a claim that the
combination is bad or impossible.

CLI::

    python -m _david.code.dashboard [--root _david/output] [--out path.html]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, QED, rdMolDescriptors
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score

from _david.code.config import OUTPUT_ROOT, PipelineConfig
from _david.code.schema import std

LOGGER = logging.getLogger(__name__)

STAMP_RE = re.compile(r"^(\d{8})_(\d{4})_")
CATEGORY_ROOTS: tuple[str, ...] = (
    "archive",
    "conformer",
    "datasets",
    "_metacentrum_jobs",
    ".agents",
    "dashboard",
)
DASHBOARD_DIR = "dashboard"
INDEX_NAME = "index.html"

#: Fields the overview compares. Administrative fields (paths, hardware) are
#: deliberately absent: their differences are not experiments.
CANONICAL_FIELDS: tuple[str, ...] = (
    "reward_scheme",
    "criteria",
    "epsilon",
    "epsilon_end",
    "use_epsilon_decay",
    "decay_start_epoch",
    "decay_epochs",
    "decay_type",
    "beta",
    "batch_size",
    "n_samples",
    "n_eval_samples",
    "epochs",
    "min_epochs",
    "patience",
    "include_qed",
    "weights",
    "threshold_percentile",
    "rocs_use_modifier",
    "sa_score_lower",
    "sa_score_upper",
    "prior_dataset",
)

#: Keys that older manifests carry and today's config no longer has. Reporting
#: them as "retired" is honest; silently comparing them would invent deltas.
RETIRED_FIELDS: tuple[str, ...] = (
    "rocs_color_weight",
    "rocs_shape_weight",
    "rocs_tversky_alpha",
    "rocs_tversky_beta",
    "energy_window_kcal",
)

#: Cluster split order: objective first, then exploration, then cost.
SPLIT_PRIORITY: tuple[str, ...] = (
    "reward_scheme",
    "criteria",
    "include_qed",
    "epsilon",
    "beta",
    "epsilon_end",
    "use_epsilon_decay",
    "batch_size",
    "n_samples",
    "n_eval_samples",
    "epochs",
    "min_epochs",
    "patience",
    "decay_start_epoch",
    "decay_epochs",
    "decay_type",
    "weights",
    "threshold_percentile",
    "rocs_use_modifier",
    "sa_score_lower",
    "sa_score_upper",
    "prior_dataset",
)

MAX_BEST_PER_RUN = 3
MAX_BEST_GLOBAL = 12
CANDIDATE_POOL = 40


def is_run_dir(path: Path) -> bool:
    """True for a stamped RL arm sitting directly under the output root."""
    return path.is_dir() and STAMP_RE.match(path.name) is not None


def _is_arm(path: Path) -> bool:
    from _david.code.output_organization import is_rl_run

    return is_rl_run(path)


def discover_runs(root: Path | str = OUTPUT_ROOT) -> list[Path]:
    """Stamped RL arms at the output root, chronological, categories excluded.

    A stamped directory that is a *container* of arms (the 6-arm suite) is not a
    run itself, but any child that recorded a manifest is one; it is returned so
    the overview does not silently drop a finished arm.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not is_run_dir(entry) or entry.name in CATEGORY_ROOTS:
            continue
        if _is_arm(entry):
            found.append(entry)
            continue
        for child in sorted(entry.iterdir()):
            if child.is_dir() and _is_arm(child) and (child / "manifest.json").is_file():
                found.append(child)
    return found


def started_at(name: str) -> datetime | None:
    match = STAMP_RE.match(name.split("/", 1)[0])
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M")
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Project a manifest config onto the current field names.

    Returns only the canonical fields (``None`` where the manifest predates the
    field) plus a ``retired`` dict so the UI can say *why* a difference has no
    comparison.
    """
    raw = raw or {}
    normalized = {key: _jsonable(raw.get(key)) for key in CANONICAL_FIELDS}
    normalized["retired"] = {
        key: _jsonable(raw[key]) for key in RETIRED_FIELDS if key in raw
    }
    return normalized


@dataclass
class RunSummary:
    """Everything the dashboard shows about one RL arm."""

    name: str
    path: Path
    started: str
    state: str
    elapsed_hours: float | None
    host: str
    pbs_job_id: str
    source_revision: str
    config: dict[str, Any]
    evaluation_source: str = ""
    best_epoch: int | None = None
    standard_source: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    standard_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started": self.started,
            "state": self.state,
            "elapsed_hours": self.elapsed_hours,
            "host": self.host,
            "pbs_job_id": self.pbs_job_id,
            "source_revision": self.source_revision,
            "config": self.config,
            "evaluation_source": self.evaluation_source,
            "best_epoch": self.best_epoch,
            "standard_source": self.standard_source,
            "metrics": self.metrics,
            "standard_metrics": self.standard_metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }


def _read_manifest(run: Path) -> dict[str, Any]:
    manifest = run / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        payload = json.loads(manifest.read_text())
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def standard_model(run: Path) -> str:
    """Name the model behind ``standard_eval``: the best checkpoint or a rescore.

    ``evaluate_run`` reloads ``models/SequenceRNN_reinforced.pkg`` and samples;
    when no checkpoint exists it only rescores the delivered molecules.

    Which epoch that package holds depends on the release that produced the run:
    the current pipeline keeps the best-validation model in
    ``SequenceRNN_reinforced.pkg`` and writes the end-of-training model to
    ``SequenceRNN_reinforced_final_epoch.pkg``, while older runs have only the
    single package and it is whatever the Explorer saved last. The label says
    ``best@N`` only when that separation proves the semantics; otherwise it says
    ``saved`` rather than claiming an epoch the file may not hold.
    """
    manifest_path = run / "standard_eval" / "manifest.json"
    if not manifest_path.is_file():
        return ""
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not payload.get("resampled"):
        return "rescore"
    best = _best_epoch(run)
    split = (run / "models" / "SequenceRNN_reinforced_final_epoch.pkg").is_file()
    manifest = _read_manifest(run)
    recorded_best = str(manifest.get("evaluation_source") or "") == "best_epoch"
    if best is not None and (split or recorded_best):
        return f"best@{best}"
    return "saved" if best is None else "saved (single package)"


def _best_epoch(run: Path) -> int | None:
    fit = run / "models" / "SequenceRNN_reinforced_fit.tsv"
    if not fit.is_file():
        return None
    try:
        frame = pd.read_csv(fit, sep="\t")
    except (OSError, ValueError):
        return None
    if "best_epoch" not in frame.columns or frame.empty:
        return None
    value = pd.to_numeric(frame["best_epoch"], errors="coerce").dropna()
    return int(value.iloc[-1]) if not value.empty else None


def _arm_columns(run: Path) -> tuple[Path | None, pd.DataFrame | None]:
    arm = run / "raw_data" / "reinforced_raw.csv.gz"
    if not arm.is_file():
        return None, None
    try:
        return arm, pd.read_csv(arm)
    except (OSError, ValueError) as exc:
        LOGGER.warning("unreadable arm %s: %s", arm, exc)
        return arm, None


def _standard_arm(run: Path) -> pd.DataFrame | None:
    """The comparable re-evaluated arm written by ``standard_eval``, if present."""
    arm = run / "standard_eval" / "reinforced_standard.csv.gz"
    if not arm.is_file():
        return None
    try:
        return pd.read_csv(arm)
    except (OSError, ValueError) as exc:
        LOGGER.warning("unreadable standard arm %s: %s", arm, exc)
        return None


def _raw_rocs(frame: pd.DataFrame, rocs_factor: float = 2.0) -> pd.Series:
    """Raw TanimotoCombo, recovered with the run's audited clip, never a guess."""
    if "Raw_ROCS" in frame.columns:
        return pd.to_numeric(frame["Raw_ROCS"], errors="coerce")
    values = pd.to_numeric(frame.get(std.ROCS, pd.Series(dtype=float)), errors="coerce")
    return values * float(rocs_factor) if not values.empty and values.max() <= 1.0001 else values


def _raw_sa(frame: pd.DataFrame) -> pd.Series | None:
    if "Raw_SA" in frame.columns:
        return pd.to_numeric(frame["Raw_SA"], errors="coerce")
    return None


def _qed_series(frame: pd.DataFrame) -> pd.Series:
    if std.QED in frame.columns:
        return pd.to_numeric(frame[std.QED], errors="coerce")
    return pd.Series(
        [
            QED.qed(mol) if (mol := Chem.MolFromSmiles(str(smi))) is not None else np.nan
            for smi in frame.get(std.SMILES, [])
        ],
        index=frame.index,
        dtype=float,
    )


def summarize_arm(
    frame: pd.DataFrame, run_config: dict[str, Any], rocs_factor: float = 2.0
) -> dict[str, Any]:
    """Baseline/reinforced medians that need no re-scoring of the whole arm."""
    valid = frame[frame[std.VALID] == 1] if std.VALID in frame.columns else frame
    rocs = _raw_rocs(valid, rocs_factor)
    qed = _qed_series(valid)
    sa_scaled = pd.to_numeric(valid.get(std.SA, pd.Series(dtype=float)), errors="coerce")
    metrics: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_valid": int(len(valid)),
        "valid_ratio": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "rocs_median": float(rocs.median()) if rocs.notna().any() else None,
        "rocs_tail_median": float(rocs.quantile(0.9))
        if rocs.notna().any()
        else None,
        "qed_median": float(qed.median()) if qed.notna().any() else None,
        "sa_median": float(sa_scaled.median()) if sa_scaled.notna().any() else None,
    }
    raw_sa = _raw_sa(valid)
    if raw_sa is not None and raw_sa.notna().any():
        metrics["raw_sa_median"] = float(raw_sa.median())
    metrics["eval_n_configured"] = run_config.get("n_eval_samples")
    return metrics


def _candidate_table(
    run_name: str,
    frame: pd.DataFrame,
    pool: int = CANDIDATE_POOL,
    *,
    arm: str = "historical",
    rocs_factor: float = 2.0,
) -> pd.DataFrame:
    """Cheap pre-ranking, then exact desirability on the survivors.

    Raw SA scoring dominates the runtime, so only the ``pool`` best molecules by
    ROCS+QED are scored for SA and then ranked with the figure-05 desirability
    (ROCS + QED − 0.1·SA).
    """
    if frame.empty or std.SMILES not in frame.columns:
        return pd.DataFrame(
            columns=["run", "arm", "SMILES", "ROCS", "QED", "SA", "score"]
        )
    valid = frame[frame[std.VALID] == 1] if std.VALID in frame.columns else frame
    if valid.empty:
        return pd.DataFrame(
            columns=["run", "arm", "SMILES", "ROCS", "QED", "SA", "score"]
        )
    work = pd.DataFrame(
        {
            "SMILES": valid[std.SMILES].astype(str),
            "ROCS": _raw_rocs(valid, rocs_factor).fillna(0.0),
            "QED": _qed_series(valid).fillna(0.0),
        }
    )
    work["pre"] = work["ROCS"] + work["QED"]
    work = work.nlargest(min(pool, len(work)), "pre")
    unique = work.drop_duplicates("SMILES").copy()
    unique["SA"] = [
        float(calculate_sa_score(mol))
        if (mol := Chem.MolFromSmiles(smi)) is not None
        else np.nan
        for smi in unique["SMILES"]
    ]
    unique["score"] = unique["ROCS"] + unique["QED"] - 0.1 * unique["SA"].fillna(10.0)
    unique["run"] = run_name
    unique["arm"] = arm
    return unique[["run", "arm", "SMILES", "ROCS", "QED", "SA", "score"]]


def _svg(smiles: str, width: int = 220, height: int = 160) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Draw.MolToSVG(mol, width=width, height=height)
    except (RuntimeError, ValueError):
        return ""


def collect(
    root: Path | str = OUTPUT_ROOT,
    *,
    include_molecules: bool = True,
) -> dict[str, Any]:
    """Scan the tree and build the JSON payload behind the dashboard."""
    root = Path(root)
    runs: list[RunSummary] = []
    candidates: list[pd.DataFrame] = []
    for run_dir in discover_runs(root):
        manifest = _read_manifest(run_dir)
        config = normalize_config(manifest.get("config"))
        elapsed = manifest.get("elapsed_seconds")
        display_name = run_dir.relative_to(root).as_posix()
        summary = RunSummary(
            name=display_name,
            path=run_dir,
            started=(started_at(display_name) or datetime.fromtimestamp(0)).isoformat(
                timespec="minutes"
            ),
            state=str(manifest.get("state", "unknown")),
            elapsed_hours=round(float(elapsed) / 3600.0, 2) if elapsed else None,
            host=str(manifest.get("host", "")),
            pbs_job_id=str(manifest.get("pbs_job_id") or ""),
            source_revision=str(manifest.get("source_revision") or ""),
            config=config,
            evaluation_source=str(manifest.get("evaluation_source") or ""),
            best_epoch=manifest.get("best_epoch"),
            error=manifest.get("error"),
        )
        from _david.code.rocs_scale import ensure_verdict

        scale = ensure_verdict(run_dir)
        rocs_factor = 1.0 if scale.is_raw else scale.factor
        arm, frame = _arm_columns(run_dir)
        standard = _standard_arm(run_dir)
        summary.artifacts = {
            "manifest": bool(manifest),
            "reinforced_arm": arm is not None and frame is not None,
            "model": any((run_dir / "models").glob("SequenceRNN_reinforced*.pkg")),
            "figures": (run_dir / "figures").is_dir(),
            "standard_eval": (run_dir / "standard_eval").is_dir(),
            "standard_arm": standard is not None,
            "rocs_scale": scale.verdict,
        }
        if frame is not None:
            summary.metrics = summarize_arm(frame, config, rocs_factor)
        if standard is not None:
            # The standard arm is scored with the frozen environment, which has no
            # ROCS modifier, so its ROCS is physical and must never be rescaled.
            summary.standard_metrics = summarize_arm(standard, config, rocs_factor=1.0)
            summary.standard_source = standard_model(run_dir)
        # A comparable standard arm outranks the historical one for cross-run
        # statements and best-of tables; the historical metrics stay visible.
        chosen = standard if standard is not None else frame
        if chosen is not None and include_molecules:
            candidates.append(
                _candidate_table(
                    summary.name,
                    chosen,
                    arm="standard" if standard is not None else "historical",
                    rocs_factor=rocs_factor,
                )
            )
        runs.append(summary)

    best = _best_molecules(candidates)
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "runs": [run.payload() for run in runs],
        "best_molecules": best,
        "cluster_tree": cluster_runs(
            [{"name": run.name, "config": run.config} for run in runs]
        ),
        "design_grids": design_grids(runs),
        "totals": {
            "runs": len(runs),
            "complete": sum(1 for run in runs if run.state == "complete"),
            "failed": sum(1 for run in runs if run.state == "failed"),
            "running": sum(1 for run in runs if run.state == "running"),
            "with_arm": sum(1 for run in runs if run.artifacts["reinforced_arm"]),
            "with_standard_eval": sum(
                1 for run in runs if run.artifacts["standard_eval"]
            ),
            "with_standard_arm": sum(
                1 for run in runs if run.artifacts.get("standard_arm")
            ),
        },
    }


def _best_molecules(candidates: Iterable[pd.DataFrame]) -> dict[str, Any]:
    frames = [frame for frame in candidates if not frame.empty]
    if not frames:
        return {"per_run": {}, "global": []}
    pooled = pd.concat(frames, ignore_index=True)
    per_run: dict[str, list[dict[str, Any]]] = {}
    for run_name, group in pooled.groupby("run"):
        top = group.nlargest(min(MAX_BEST_PER_RUN, len(group)), "score")
        per_run[str(run_name)] = [_molecule_row(row) for _, row in top.iterrows()]
    top_global = pooled.drop_duplicates("SMILES").nlargest(MAX_BEST_GLOBAL, "score")
    return {
        "per_run": per_run,
        "global": [_molecule_row(row) for _, row in top_global.iterrows()],
    }


def _molecule_row(row: pd.Series) -> dict[str, Any]:
    smiles = str(row["SMILES"])
    return {
        "run": str(row["run"]),
        "smiles": smiles,
        "arm": str(row.get("arm", "historical")),
        "canonical": _canonical(smiles),
        "rocs": round(float(row["ROCS"]), 4),
        "qed": round(float(row["QED"]), 4),
        "sa": round(float(row["SA"]), 3) if pd.notna(row["SA"]) else None,
        "score": round(float(row["score"]), 4),
        "svg": _svg(smiles),
    }


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def cluster_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively split runs on the setting that separates them most.

    A node whose runs share every canonical field is a leaf that collapses them
    into one design ("3 runs, same settings"), which is exactly the "preserve
    togetherness" the static table loses.
    """
    if not runs:
        return {}
    varying = [
        key
        for key in SPLIT_PRIORITY
        if len({json.dumps(run["config"].get(key), sort_keys=True) for run in runs}) > 1
    ]
    if not varying:
        return {
            "kind": "leaf",
            "runs": [run["name"] for run in runs],
            "signature": {
                key: runs[0]["config"].get(key) for key in CANONICAL_FIELDS
            },
        }
    key = varying[0]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        buckets.setdefault(json.dumps(run["config"].get(key), sort_keys=True), []).append(run)
    children = [
        {"value": _display_value(value), **cluster_runs(members)}
        for value, members in sorted(buckets.items(), key=lambda item: item[0])
    ]
    return {"kind": "split", "field": key, "children": children}


def _display_value(value: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def design_grids(runs: list[RunSummary]) -> list[dict[str, Any]]:
    """Two matrices whose empty cells are combinations never run."""
    grids = []
    for row_key, col_key in (
        ("reward_scheme", "criteria"),
        ("epsilon", "beta"),
    ):
        rows = sorted(
            {_grid_value(run.config.get(row_key)) for run in runs},
            key=_grid_sort_key,
        )
        cols = sorted(
            {_grid_value(run.config.get(col_key)) for run in runs},
            key=_grid_sort_key,
        )
        cells: dict[str, dict[str, list[str]]] = {}
        for run in runs:
            row = _grid_value(run.config.get(row_key))
            col = _grid_value(run.config.get(col_key))
            cells.setdefault(row, {}).setdefault(col, []).append(run.name)
        grids.append(
            {
                "row_field": row_key,
                "col_field": col_key,
                "rows": rows,
                "cols": cols,
                "cells": cells,
            }
        )
    return grids


def _grid_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "·".join(_grid_value(item) for item in value)
    return str(value)


def _grid_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, value)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       color: #0f172a; background: #f8fafc; }
header { padding: 22px 28px 14px; background: #0f172a; color: #f8fafc; }
header h1 { margin: 0 0 4px; font-size: 21px; font-weight: 650; letter-spacing: -0.01em; }
header p { margin: 0; color: #cbd5e1; font-size: 12.5px; }
.cards { display: flex; gap: 12px; padding: 18px 28px 4px; flex-wrap: wrap; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px;
        min-width: 128px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
.card b { display: block; font-size: 22px; font-weight: 650; }
.card span { color: #64748b; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }
nav { display: flex; gap: 4px; padding: 14px 28px 0; flex-wrap: wrap; }
nav button { border: 1px solid #e2e8f0; background: #fff; color: #334155; padding: 7px 13px;
             border-radius: 7px; font-size: 13px; cursor: pointer; }
nav button.active { background: #0f172a; color: #fff; border-color: #0f172a; }
main { padding: 16px 28px 48px; }
section { display: none; }
section.active { display: block; }
h2 { font-size: 15px; margin: 18px 0 8px; }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 12.5px;
        border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
th, td { padding: 7px 9px; text-align: left; border-bottom: 1px solid #eef2f7; vertical-align: top; }
th { background: #f1f5f9; font-weight: 600; cursor: pointer; user-select: none; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.state { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.complete { background: #dcfce7; color: #166534; }
.failed { background: #fee2e2; color: #991b1b; }
.running { background: #dbeafe; color: #1e40af; }
.unknown { background: #f1f5f9; color: #475569; }
.muted { color: #94a3b8; }
.grid { border-collapse: separate; border-spacing: 4px; }
.grid th { background: transparent; cursor: default; }
.grid td { border: 1px solid #e2e8f0; border-radius: 6px; min-width: 92px; text-align: center;
           background: #fff; padding: 8px 6px; font-size: 12px; }
.grid td.done { background: #dcfce7; border-color: #86efac; color: #14532d; font-weight: 600; }
.grid td.empty { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
.tree ul { list-style: none; margin: 2px 0 2px 18px; padding: 0; border-left: 1px dashed #cbd5e1; }
.tree li { margin: 3px 0 3px 10px; }
.tree .field { font-weight: 600; color: #1e293b; }
.tree .value { color: #0e7490; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.tree .leaf { color: #475569; }
.mols { display: grid; grid-template-columns: repeat(auto-fill, minmax(215px, 1fr)); gap: 12px; }
.mol { background: #fff; border: 1px solid #e2e8f0; border-radius: 9px; padding: 8px; }
.mol svg { width: 100%; height: auto; display: block; }
.mol .meta { font-size: 11px; color: #475569; line-height: 1.35; overflow-wrap: anywhere; }
.mol .score { font-weight: 650; color: #0f172a; }
.note { background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; padding: 9px 12px;
        border-radius: 8px; font-size: 12.5px; margin: 10px 0; }
.chart { background: #fff; border: 1px solid #e2e8f0; border-radius: 9px; padding: 6px; }
.chart svg { width: 100%; height: auto; display: block; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { background: #eef2ff; color: #3730a3; border-radius: 999px; padding: 2px 9px; font-size: 11.5px; }
"""

_JS = """
const P = JSON.parse(document.getElementById('payload').textContent);
const esc = (v) => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, d=3) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const cfg = (run, key) => {
  const v = run.config[key];
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (Array.isArray(v)) return v.map(x => typeof x === 'number' ? +x.toFixed(3) : x).join(' · ');
  if (typeof v === 'number') return String(+v.toFixed(4));
  return String(v);
};
const stateBadge = (s) => `<span class="state ${esc(s)}">${esc(s)}</span>`;

function tabs() {
  document.querySelectorAll('nav button').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    };
  });
  document.querySelector('nav button').click();
}

function sortable(table) {
  table.querySelectorAll('th').forEach((th, idx) => {
    th.onclick = () => {
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      const asc = th.dataset.asc !== '1';
      th.dataset.asc = asc ? '1' : '0';
      rows.sort((a, b) => {
        const av = a.cells[idx].dataset.sort ?? a.cells[idx].textContent;
        const bv = b.cells[idx].dataset.sort ?? b.cells[idx].textContent;
        const an = parseFloat(av), bn = parseFloat(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
      rows.forEach(r => body.appendChild(r));
    };
  });
}

function modelLabel(run) {
  if (run.evaluation_source === 'best_epoch' && run.best_epoch) return `best@${run.best_epoch}`;
  if (run.evaluation_source === 'last_epoch') return 'last';
  return 'unrecorded';
}

function renderRuns() {
  const cols = ['Run', 'Started', 'State', 'Arm model', 'ROCS med', 'ROCS p90', 'QED med', 'SA des', 'Valid',
                'Eval n', 'Arm n', 'Std ROCS', 'Std QED', 'Std SA des', 'Hours', 'Host / job', 'Std model', 'Standard eval'];
  const head = cols.map(c => `<th>${esc(c)}</th>`).join('');
  const rows = P.runs.map(r => {
    const m = r.metrics || {};
    const sm = r.standard_metrics || {};
    return `<tr>
      <td data-sort="${esc(r.name)}"><b>${esc(r.name)}</b><br><span class="muted">${esc(r.source_revision.slice(0,7) || 'no revision')}</span></td>
      <td data-sort="${esc(r.started)}">${esc(r.started.replace('T',' '))}</td>
      <td>${stateBadge(r.state)}</td>
      <td>${esc(modelLabel(r))}</td>
      <td class="num" data-sort="${m.rocs_median ?? ''}">${fmt(m.rocs_median)}</td>
      <td class="num" data-sort="${m.rocs_tail_median ?? ''}">${fmt(m.rocs_tail_median)}</td>
      <td class="num" data-sort="${m.qed_median ?? ''}">${fmt(m.qed_median)}</td>
      <td class="num" data-sort="${m.sa_median ?? ''}">${fmt(m.sa_median)}</td>
      <td class="num" data-sort="${m.valid_ratio ?? ''}">${m.valid_ratio === undefined ? '—' : (100*m.valid_ratio).toFixed(0)+' %'}</td>
      <td class="num" data-sort="${cfg(r,'n_eval_samples')}">${cfg(r,'n_eval_samples')}</td>
      <td class="num" data-sort="${m.n_rows ?? ''}">${m.n_rows ?? '—'}</td>
      <td class="num" data-sort="${sm.rocs_median ?? ''}">${fmt(sm.rocs_median)}</td>
      <td class="num" data-sort="${sm.qed_median ?? ''}">${fmt(sm.qed_median)}</td>
      <td class="num" data-sort="${sm.sa_median ?? ''}">${fmt(sm.sa_median)}</td>
      <td class="num" data-sort="${r.elapsed_hours ?? ''}">${fmt(r.elapsed_hours, 1)}</td>
      <td>${esc(r.host || '—')}<br><span class="muted">${esc(r.pbs_job_id || '')}</span></td>
      <td>${esc(r.standard_source || '—')}</td>
      <td>${r.artifacts.standard_arm ? '✓ comparable' : (r.artifacts.standard_eval ? 'dir only' : '—')}</td>
    </tr>`;
  }).join('');
  const table = document.getElementById('runs-table');
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${rows}</tbody>`;
  sortable(table);
}

function renderGrids() {
  const host = document.getElementById('grids-content');
  host.innerHTML = P.design_grids.map(g => {
    const head = `<tr><th>${esc(g.row_field)} \\\\ ${esc(g.col_field)}</th>${g.cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>`;
    const rows = g.rows.map(r => `<tr><th>${esc(r)}</th>${g.cols.map(c => {
      const names = (g.cells?.[r]?.[c]) || [];
      if (!names.length) return `<td class="empty" title="not run">—</td>`;
      return `<td class="done" title="${esc(names.join('\\n'))}">${names.length} run${names.length > 1 ? 's' : ''}</td>`;
    }).join('')}</tr>`).join('');
    return `<h2>${esc(g.row_field)} × ${esc(g.col_field)}</h2>
      <p class="muted">Green = at least one run; orange = this combination has never been run in this tree.</p>
      <table class="grid">${head}${rows}</table>`;
  }).join('');
}

function tree(node, depth = 0) {
  if (!node || !node.kind) return '';
  if (node.kind === 'leaf') {
    const names = node.runs.map(n => `<span class="chip">${esc(n)}</span>`).join(' ');
    return `<li><span class="leaf">${node.runs.length} run(s) with identical settings</span>
      <div class="chips">${names}</div></li>`;
  }
  const children = node.children.map(child =>
    `<li><span class="value">${esc(fmtValue(child.value))}</span>${tree(child, depth + 1)}</li>`
  ).join('');
  return `<li><span class="field">${esc(node.field)}</span><ul>${children}</ul></li>`;
}

function fmtValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (Array.isArray(v)) return v.join(' · ');
  if (typeof v === 'number') return String(+v.toFixed(4));
  return String(v);
}

function renderClusters() {
  const host = document.getElementById('clusters');
  host.innerHTML = `<div class="tree"><ul>${tree(P.cluster_tree)}</ul></div>`;
}

function renderMolecules() {
  const host = document.getElementById('molecules');
  const global = P.best_molecules.global || [];
  const perRun = P.best_molecules.per_run || {};
  const card = (m, showRun) => `<div class="mol">
      ${m.svg || '<div class="muted">no depiction</div>'}
      <div class="meta">
        <div class="score">${fmt(m.score)} <span class="muted">= ROCS + QED − 0.1·SA</span></div>
        <div>ROCS ${fmt(m.rocs)} · QED ${fmt(m.qed)} · SA ${fmt(m.sa, 2)}</div>
        ${showRun ? `<div><b>${esc(m.run)}</b> <span class="chip">${esc(m.arm)}</span></div>` : ''}
        <div class="muted">${esc(m.smiles)}</div>
      </div></div>`;
  host.innerHTML = `
    <h2>Best of the best (all stored arms)</h2>
    <p class="muted">Ranked by the figure-05 desirability on the raw TanimotoCombo; only the
      candidate pool per run is SA-scored, the final order uses exact SA.</p>
    <div class="mols">${global.map(m => card(m, true)).join('')}</div>
    <h2>Per run</h2>
    ${Object.entries(perRun).map(([run, mols]) => `
      <h3>${esc(run)}</h3>
      <div class="mols">${mols.map(m => card(m, false)).join('')}</div>`).join('')}`;
}


function renderComparison() {
  const W = 780, H = 430, L = 58, R = 22, T = 18, B = 46;
  const xmin = 0, xmax = 2, ymin = 0, ymax = 1;
  const px = (v) => L + (v - xmin) / (xmax - xmin) * (W - L - R);
  const py = (v) => T + (ymax - v) / (ymax - ymin) * (H - T - B);
  const hist = [], std = [];
  P.runs.forEach(r => {
    const m = r.metrics || {}, sm = r.standard_metrics || {};
    if (m.rocs_median != null && m.qed_median != null)
      hist.push({name: r.name, x: m.rocs_median, y: m.qed_median, valid: m.valid_ratio, n: m.n_valid});
    if (sm.rocs_median != null && sm.qed_median != null)
      std.push({name: r.name, x: sm.rocs_median, y: sm.qed_median, valid: sm.valid_ratio, n: sm.n_valid});
  });
  const grid = [];
  for (let i = 0; i <= 4; i++) {
    const vx = xmin + (xmax - xmin) * i / 4;
    const vy = ymin + (ymax - ymin) * i / 4;
    grid.push(`<line x1="${px(vx)}" y1="${py(ymin)}" x2="${px(vx)}" y2="${py(ymax)}" stroke="#eef2f7"/>`);
    grid.push(`<text x="${px(vx)}" y="${H - B + 16}" font-size="11" text-anchor="middle" fill="#64748b">${vx.toFixed(2)}</text>`);
    grid.push(`<line x1="${L}" y1="${py(vy)}" x2="${W - R}" y2="${py(vy)}" stroke="#eef2f7"/>`);
    grid.push(`<text x="${L - 8}" y="${py(vy) + 4}" font-size="11" text-anchor="end" fill="#64748b">${vy.toFixed(2)}</text>`);
  }
  const byName = Object.fromEntries(hist.map(p => [p.name, p]));
  const links = std.filter(p => byName[p.name]).map(p => {
    const q = byName[p.name];
    return `<line x1="${px(q.x)}" y1="${py(q.y)}" x2="${px(p.x)}" y2="${py(p.y)}" stroke="#cbd5e1" stroke-dasharray="3 3"/>`;
  }).join('');
  const dot = (p, color) => `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="${4 + 2 * Math.min(1, (p.valid ?? 0))}"
      fill="${color}" fill-opacity="0.82" stroke="#fff" stroke-width="1">
      <title>${esc(p.name)}\nROCS ${fmt(p.x)} · QED ${fmt(p.y)} · valid ${p.valid == null ? '—' : (100*p.valid).toFixed(0)+' %'}\nn=${p.n ?? '—'}</title></circle>`;
  const dots = hist.map(p => dot(p, '#94a3b8')).join('') + std.map(p => dot(p, '#2563eb')).join('');
  const empty = (!hist.length && !std.length) ? '<div class="note">No arm has both ROCS and QED medians yet.</div>' : '';
  document.getElementById('comparison').innerHTML = `${empty}
    <h2>ROCS vs QED across runs</h2>
    <p class="muted">Each point is one run's median over valid molecules. Grey = the historical arm,
      blue = the standard re-evaluated arm; a dashed line joins the two arms of the same run.
      Point size grows with validity. Axes are fixed (raw TanimotoCombo 0–2, QED 0–1), so position
      is comparable across pages.</p>
    <div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="ROCS versus QED scatter">
      ${grid.join('')}
      <line x1="${L}" y1="${py(ymin)}" x2="${W - R}" y2="${py(ymin)}" stroke="#94a3b8"/>
      <line x1="${L}" y1="${py(ymin)}" x2="${L}" y2="${py(ymax)}" stroke="#94a3b8"/>
      <text x="${(L + W - R) / 2}" y="${H - 8}" font-size="12" text-anchor="middle" fill="#334155">median raw ROCS (TanimotoCombo)</text>
      <text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#334155">median QED</text>
      ${links}${dots}
    </svg></div>
    <p class="muted">Historical points: ${hist.length} · standard points: ${std.length}</p>`;
}

function render() {
  document.getElementById('generated').textContent = P.generated;
  document.getElementById('totals').innerHTML = [
    ['Runs', P.totals.runs], ['Complete', P.totals.complete], ['Failed', P.totals.failed],
    ['Running', P.totals.running], ['With arm', P.totals.with_arm],
    ['Standard eval', P.totals.with_standard_eval],
    ['Standard arm', P.totals.with_standard_arm],
  ].map(([label, value]) => `<div class="card"><b>${value}</b><span>${label}</span></div>`).join('');
  renderRuns(); renderGrids(); renderClusters(); renderMolecules(); renderComparison(); tabs();
}
render();
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DrugEx run overview</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>DrugEx RL run overview</h1>
  <p>Generated <span id="generated"></span> from every <code>YYYYmmdd_HHMM_</code> arm under the
     output root. Offline file: no CDN, no server.</p>
</header>
<div class="cards" id="totals"></div>
<nav>
  <button data-tab="runs">Runs</button>
  <button data-tab="grids">Design grid</button>
  <button data-tab="clusters">Clusters</button>
  <button data-tab="comparison">Comparison</button>
  <button data-tab="molecules">Best molecules</button>
</nav>
<main>
  <section id="runs"><h2>All runs</h2><p class="muted">Medians are computed from the stored
    reinforced arm; QED is recomputed from SMILES when the run did not score it.</p>
    <table id="runs-table"></table></section>
  <section id="grids"><h2>What we ran</h2><div id="grids-content"></div></section>
  <section id="clusters"><h2>Design tree</h2>
    <p class="muted">Splits on the setting that separates the runs most, objective fields first.
      A leaf collapses runs whose canonical settings are identical.</p><div id="clusters"></div></section>
  <section id="comparison"><div id="comparison"></div></section>
  <section id="molecules"><div id="molecules"></div></section>
</main>
<script id="payload" type="application/json">{payload}</script>
<script>{js}</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    """Serialise the payload into the self-contained page (``<`` escaped)."""
    data = json.dumps(payload, ensure_ascii=False, default=str).replace("<", "\\u003c")
    return (
        _TEMPLATE.replace("{css}", _CSS)
        .replace("{payload}", data)
        .replace("{js}", _JS)
    )


def generate(
    root: Path | str = OUTPUT_ROOT,
    out_path: Path | str | None = None,
    *,
    include_molecules: bool = True,
) -> Path:
    """Scan, render and write the dashboard; returns the HTML path."""
    root = Path(root)
    payload = collect(root, include_molecules=include_molecules)
    target = Path(out_path) if out_path else root / DASHBOARD_DIR / INDEX_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(payload), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-molecules", action="store_true")
    args = parser.parse_args(argv)
    target = generate(
        args.root, args.out, include_molecules=not args.no_molecules
    )
    print(f"dashboard: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
