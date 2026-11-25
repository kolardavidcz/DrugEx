"""Scan every run for mode collapse: ``python -m _david.code.analysis.collapse``.

The per-epoch metrics cannot see collapse: they are computed on ``n_eval`` (50) draws, and
a policy that repeats one series every few hundred draws still looks unique at n=50. The
5,000-molecule final arms can, file by file, so this module reads every arm table of every
run under the roots it is given, computes per-arm distribution statistics, and labels each
run. Both schemas are understood: this repo's arms (``prior.tsv``, ``final_*.tsv``) and the
legacy WITH_BUGS repo's ``raw_data/*_raw.csv.gz``.

Outputs ``collapse_scan.json`` and ``collapse_scan.tsv``; the self-contained atlas is the
tree atlas (``collapse_tree_atlas.html``, see :mod:`_david.code.analysis.atlas_v2`), which
reads the same scan.

Two rules fix the scope. Every judgement — verdict, frontier, medals — reads the **stop
arm** (``final_desired_ratio_at_stop.tsv``), the live final policy, never the gmean
checkpoint's arm. And only runs from :data:`SINCE` onwards are counted: everything earlier
belongs to the legacy WITH_BUGS regime and this repo's first probes.

    python -m _david.code.analysis.collapse [--roots DIR ...] [--since YYYYMMDD]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gzip
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.analysis._helpers import (
    frame_rocs_scale,
    iter_run_dirs,
    pareto_mask,
    run_date,
)
from _david.code.config import PROJECT_ROOT
from _david.code.reconstruct.provenance import atomic_write_text

logger = logging.getLogger(__name__)

#: The legacy checkout. Not scanned by default: every run in it predates :data:`SINCE`.
LEGACY_ROOT = PROJECT_ROOT.parent / "WITH_BUGS_drugex" / "_david" / "output"

#: Only runs from this date onwards are counted (``YYYYMMDD`` prefix of the run dir name).
SINCE = "20261110"

#: Arms in reporting order, named by role.
ARM_ROLES = ("prior", "best", "stop", "gmean")

#: Verdict thresholds on this campaign's distribution.
COLLAPSED_UNIQUE = 0.35
CONCENTRATED_UNIQUE = 0.70
SERIES_LOCK_TOP10 = 0.60
VALIDITY_FLOOR = 0.85
FRAGMENT_DRIFT_HAC = 0.80
NO_LEARNING_GAIN = 1.5
PILOT_MAX_N = 1000
SPARK_POINTS = 140

LEGACY_ARMS = {
    "prior": "baseline_raw.csv.gz",
    "best": "reinforced_raw.csv.gz",
    "stop": "reinforced_final_epoch_raw.csv.gz",
}
CURRENT_ARMS = {
    "prior": "prior.tsv",
    "best": "final_desired_ratio.tsv",
    "stop": "final_desired_ratio_at_stop.tsv",
    "gmean": "final_gmean.tsv",
}

LEGACY_AXIS_MAP = {
    "reward_scheme": "scheme",
    "epsilon": "epsilon",
    "minimize_candidates": "minimize",
    "max_reference_conformers": "reference_conformers",
    "max_conformers": "max_conformers",
    "include_qed": "include_qed",
    "color_ff": "color_ff",
    "shape_color_weight": "alignment_shape_weight",
}

LEGACY_SCHEME = {
    "pareto_crowding": "PRCD",
    "pareto_tanimoto": "PRTD",
    "weighted_sum": "WS",
    "harrington": "HARR",
}

COLLAPSE_CACHE_VERSION = 2

#: canonical SMILES -> (heavy atoms, molecular weight, Murcko scaffold).
_MOLECULES: dict[str, tuple[int, float, str] | None] = {}
_run_date = run_date


def _describe(canonical: str) -> tuple[int, float, str] | None:
    """Heavy-atom count, molecular weight, and Murcko scaffold of one canonical SMILES."""
    if canonical in _MOLECULES:
        return _MOLECULES[canonical]
    mol = Chem.MolFromSmiles(canonical) if canonical else None
    entry = None
    if mol is not None:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        entry = (mol.GetNumHeavyAtoms(), float(Descriptors.MolWt(mol)), scaffold or "")
    _MOLECULES[canonical] = entry
    return entry


# --------------------------------------------------------------------------------------
# Per-arm statistics, computed from the arm table itself
# --------------------------------------------------------------------------------------


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def arm_stats(frame: pd.DataFrame, rocs_column: str | None, scale: float) -> dict[str, Any]:
    """Compute per-arm distribution metrics matching :func:`_david.code.run.write_arm`."""
    n = len(frame)
    if n == 0:
        return {
            "n": 0,
            "valid_ratio": None,
            "unique_ratio": None,
            "top1_share": None,
            "top10_share": None,
            "desired_ratio": None,
            "rocs_median": None,
            "rocs_max": None,
        }

    valid_mask = _numeric(frame, "Valid").fillna(0) == 1
    canon = (
        frame.loc[valid_mask, "Canonical"]
        if "Canonical" in frame.columns
        else pd.Series(dtype=str)
    )
    if canon.empty and "Canonical_SMILES" in frame.columns:
        canon = frame.loc[valid_mask, "Canonical_SMILES"]
    canon = canon.fillna("").astype(str)
    canon = canon[canon != ""]

    counts = canon.value_counts()
    n_unique = int(counts.size)
    stats: dict[str, Any] = {
        "n": n,
        "valid_ratio": round(float(valid_mask.mean()), 4),
        "unique_ratio": round(n_unique / n, 4),
        "top1_share": round(float(counts.iloc[0]) / n, 4) if n_unique else 0.0,
        "top10_share": round(float(counts.head(10).sum()) / n, 4) if n_unique else 0.0,
    }

    desired_col = _numeric(frame, "Desired")
    unique_rows = frame.loc[valid_mask].drop_duplicates(
        subset="Canonical" if "Canonical" in frame.columns else None
    )
    if len(unique_rows) and "Desired" in frame.columns:
        stats["desired_ratio"] = round(
            float(desired_col.loc[unique_rows.index].fillna(0).sum()) / n, 4
        )
    else:
        stats["desired_ratio"] = None

    rocs = _numeric(frame, rocs_column or "") * scale if rocs_column else None
    if rocs is not None:
        rocs = rocs.loc[unique_rows.index] if len(unique_rows) else rocs
        rocs = rocs.dropna()
    stats["rocs_median"] = round(float(rocs.median()), 3) if rocs is not None and len(rocs) else None
    stats["rocs_max"] = round(float(rocs.max()), 3) if rocs is not None and len(rocs) else None
    return stats


def chemistry_stats(canonicals: pd.Series) -> dict[str, Any]:
    """Medians and distinct Murcko scaffold concentration over one arm's unique valid molecules."""
    unique = [c for c in canonicals.fillna("").astype(str).unique() if c]
    described = [d for d in (_describe(c) for c in unique) if d is not None]
    if not described:
        return {"hac_median": None, "mw_median": None, "scaffolds": 0, "scaffold_top": None}
    hac = np.array([d[0] for d in described], dtype=float)
    scaffold_counts = pd.Series([d[2] for d in described if d[2]], dtype=str).value_counts()
    return {
        "hac_median": round(float(np.median(hac)), 1),
        "mw_median": round(float(np.median([d[1] for d in described])), 1),
        "scaffolds": int(scaffold_counts.size),
        "scaffold_top": round(float(scaffold_counts.iloc[0]) / len(described), 4)
        if scaffold_counts.size
        else None,
    }


def _top_molecules(
    frame: pd.DataFrame, rocs_column: str | None, scale: float, count: int
) -> list[dict[str, Any]]:
    """The most repeated molecules of an arm, with each one's best ROCS."""
    if "Canonical" not in frame.columns or not len(frame):
        return []
    valid = frame[_numeric(frame, "Valid").fillna(0) == 1].copy()
    valid["_canonical"] = valid["Canonical"].fillna("")
    repeats = valid["_canonical"].value_counts()
    valid["_rocs"] = _numeric(valid, rocs_column or "") * scale if rocs_column else math.nan
    out: list[dict[str, Any]] = []
    for canonical, n in repeats.head(count).items():
        if not canonical:
            continue
        rows = valid.loc[valid["_canonical"] == canonical, "_rocs"].dropna()
        out.append(
            {
                "smiles": canonical,
                "count": int(n),
                "rocs": round(float(rows.max()), 3) if len(rows) else None,
            }
        )
    return out


def _best_desired_molecules(
    frame: pd.DataFrame, rocs_column: str | None, scale: float, count: int = 10
) -> list[dict[str, Any]]:
    """Top N unique desired molecules of an arm, ranked by ROCS."""
    if not len(frame) or rocs_column is None:
        return []
    valid = frame[_numeric(frame, "Valid").fillna(0) == 1].copy()
    desired_mols = (
        valid[_numeric(valid, "Desired").fillna(0) == 1].copy()
        if "Desired" in valid.columns
        else valid.copy()
    )
    if not len(desired_mols):
        desired_mols = valid
    desired_mols["_rocs"] = _numeric(desired_mols, rocs_column) * scale
    top = (
        desired_mols.dropna(subset=["_rocs"])
        .drop_duplicates(subset=["Canonical" if "Canonical" in desired_mols.columns else "SMILES"])
        .nlargest(count, "_rocs")
    )
    out: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        smi = str(row.get("Canonical") or row.get("SMILES") or "")
        if smi:
            out.append({"smiles": smi, "rocs": round(float(row["_rocs"]), 3)})
    return out


def _lead_molecule(
    frame: pd.DataFrame, rocs_column: str | None, scale: float
) -> dict[str, Any] | None:
    """The single best desired molecule of an arm."""
    top = _best_desired_molecules(frame, rocs_column, scale, count=1)
    return top[0] if top else None


# --------------------------------------------------------------------------------------
# Schema adapters: one run directory -> frames and axes, whichever repo it lives in
# --------------------------------------------------------------------------------------


def _read_table(path: Path, sep: str = "\t") -> pd.DataFrame:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return pd.read_csv(handle, sep=",")
        return pd.read_csv(path, sep=sep)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        logger.warning("%s unreadable (%s)", path.name, error)
        return pd.DataFrame()


def _rocs_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if "TanimotoCombo" in column or column == "ROCS":
            return column
    return None


def _load_current(run: Path) -> dict[str, pd.DataFrame]:
    return {
        role: _read_table(run / name)
        for role, name in CURRENT_ARMS.items()
        if (run / name).is_file()
    }


def _load_legacy(run: Path) -> dict[str, pd.DataFrame]:
    return {
        role: _read_table(run / "raw_data" / name)
        for role, name in LEGACY_ARMS.items()
        if (run / "raw_data" / name).is_file()
    }


def _axes(config: dict[str, Any], legacy: bool) -> dict[str, Any]:
    """Normalize campaign axes under one spelling for both current and legacy schemas."""
    if not legacy:
        return {
            key: config.get(key)
            for key in (
                "scheme",
                "epsilon",
                "minimize",
                "reference_conformers",
                "max_conformers",
                "include_qed",
                "color_ff",
                "alignment_shape_weight",
            )
        }
    mapped: dict[str, Any] = {}
    for old, new in LEGACY_AXIS_MAP.items():
        value = config.get(old)
        if new == "scheme" and isinstance(value, str):
            value = LEGACY_SCHEME.get(value, value.upper() if value else value)
        mapped[new] = value
    return mapped


def _training_curve(run: Path, legacy: bool) -> dict[str, Any]:
    """Per-epoch desired/unique/valid trajectories, downsampled for atlas sparklines."""
    columns = {"desired": "desired_ratio", "unique": "unique_ratio", "valid": "valid_ratio"}
    if not legacy and (run / "rl_fit.tsv").is_file():
        frame = _read_table(run / "rl_fit.tsv")
    elif legacy and (run / "molecules" / "training_molecules.csv.gz").is_file():
        frame = _read_table(run / "molecules" / "training_molecules.csv.gz")
        if len(frame):
            grouped = frame.groupby("Epoch")
            frame = pd.DataFrame(
                {
                    "desired_ratio": grouped["Desired"].mean(),
                    "unique_ratio": grouped["Canonical"].nunique() / grouped.size(),
                    "valid_ratio": grouped["Valid"].mean(),
                }
            ).reset_index()
    else:
        return {"epochs": 0}
    if not len(frame):
        return {"epochs": 0}
    step = max(1, math.ceil(len(frame) / SPARK_POINTS))
    frame = frame.iloc[::step]
    curve: dict[str, Any] = {
        "epochs": int(frame["Epoch"].iloc[-1]) if "Epoch" in frame else len(frame)
    }
    for key, column in columns.items():
        if column in frame.columns:
            curve[key] = [round(float(v), 4) for v in frame[column]]
    return curve


# --------------------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------------------


def verdict(
    stop: dict[str, Any],
    prior: dict[str, Any],
    stop_chemistry: dict[str, Any],
    prior_chemistry: dict[str, Any],
    n_expected: int,
) -> dict[str, Any]:
    """Assign a triage verdict label, flags, and reasons from the stop arm and prior."""
    flags: list[str] = []
    reasons: list[str] = []

    if stop["n"] < 0.9 * max(n_expected, 1):
        return {
            "label": "incomplete",
            "flags": ["incomplete"],
            "reasons": [f"stop arm holds {stop['n']} of {n_expected} molecules"],
        }
    if n_expected < PILOT_MAX_N:
        return {
            "label": "pilot",
            "flags": ["pilot"],
            "reasons": [f"n_final={n_expected}: too few draws to measure uniqueness"],
        }

    unique = stop["unique_ratio"]
    prior_unique = prior["unique_ratio"]
    if unique is not None and unique < COLLAPSED_UNIQUE:
        flags.append("collapse")
        reasons.append(
            f"only {unique:.2f} unique molecules per draw (prior {prior_unique:.2f})"
        )
    if stop["top10_share"] is not None and stop["top10_share"] > SERIES_LOCK_TOP10:
        flags.append("series_lock")
        reasons.append(f"top-10 molecules are {stop['top10_share']:.0%} of the whole draw")
    if stop["valid_ratio"] is not None and stop["valid_ratio"] < VALIDITY_FLOOR:
        flags.append("validity_collapse")
        reasons.append(
            f"validity fell to {stop['valid_ratio']:.2f} (prior {prior['valid_ratio']:.2f})"
        )

    hac = stop_chemistry.get("hac_median")
    prior_hac = prior_chemistry.get("hac_median")
    if hac and prior_hac and hac < FRAGMENT_DRIFT_HAC * prior_hac:
        flags.append("fragment_drift")
        reasons.append(
            f"median heavy atoms {hac:.0f} vs prior {prior_hac:.0f}: reward bought with fragments"
        )

    desired = stop["desired_ratio"]
    prior_desired = prior["desired_ratio"]
    gained = (
        desired is not None
        and prior_desired is not None
        and desired > NO_LEARNING_GAIN * max(prior_desired, 1e-9)
    )
    if unique is not None and unique >= COLLAPSED_UNIQUE and not gained:
        flags.append("no_learning")
        reasons.append(
            f"desired {desired:.2f} barely moved from the prior's {prior_desired:.2f}"
            if desired is not None and prior_desired is not None
            else "desired ratio never left the prior's neighbourhood"
        )

    if "collapse" in flags or "series_lock" in flags or "validity_collapse" in flags:
        label = "collapsed"
    elif unique is not None and unique < CONCENTRATED_UNIQUE:
        label = "concentrated"
        reasons.append(
            f"unique {unique:.2f} of the prior's {prior_unique:.2f}: partially concentrated"
        )
    elif "no_learning" in flags:
        label = "unlearned"
    else:
        label = "clean"
    return {"label": label, "flags": flags, "reasons": reasons}


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


def _diversity_version() -> int:
    try:
        from _david.code.analysis.diversity import DIVERSITY_VERSION

        return int(DIVERSITY_VERSION)
    except Exception:
        return 0


def _collapse_sources(run: Path, legacy: bool) -> dict[str, list[int]]:
    """Size + mtime_ns fingerprints of every file :func:`scan_run` depends on."""
    names = [
        "manifest.json",
        "config.json",
        "rl_fit.tsv",
        "run.log",
        "cache_stats/diversity.json",
    ]
    if legacy:
        names.extend(f"raw_data/{name}" for name in LEGACY_ARMS.values())
        names.append("molecules/training_molecules.csv.gz")
    else:
        names.extend(CURRENT_ARMS.values())
    out: dict[str, list[int]] = {}
    for rel in names:
        path = run / rel
        if path.is_file():
            st = path.stat()
            out[rel] = [int(st.st_size), int(st.st_mtime_ns)]
        else:
            out[rel] = [-1, -1]
    return out


def _load_collapse_cache(run: Path, legacy: bool) -> dict[str, Any] | None:
    cache = run / "cache_stats" / "collapse_run.json"
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("version") != COLLAPSE_CACHE_VERSION
        or data.get("diversity_version") != _diversity_version()
        or bool(data.get("legacy")) != bool(legacy)
    ):
        return None
    if data.get("sources") != _collapse_sources(run, legacy):
        return None
    record = data.get("record")
    return record if isinstance(record, dict) else None


def _save_collapse_cache(run: Path, legacy: bool, record: dict[str, Any]) -> None:
    payload = {
        "version": COLLAPSE_CACHE_VERSION,
        "diversity_version": _diversity_version(),
        "legacy": bool(legacy),
        "sources": _collapse_sources(run, legacy),
        "record": record,
    }
    with contextlib.suppress(OSError):
        atomic_write_text(run / "cache_stats" / "collapse_run.json", json.dumps(payload))


def _diversity_summary(
    run: Path, arms: dict[str, Any], n_expected: int
) -> dict[str, Any]:
    """Attach cached chemical-diversity metrics for ``run``, refreshing stale caches in place."""
    try:
        from _david.code.analysis.diversity import _sources as _div_sources
        from _david.code.analysis.diversity import load_profile, profile_run

        profile = load_profile(run)
        if (
            profile is None
            and arms.get("stop")
            and arms.get("prior")
            and (run / "cache_stats" / "diversity.json").is_file()
        ):
            profile = profile_run(run)
            if profile is not None:
                profile["sources"] = _div_sources(run)
                atomic_write_text(run / "cache_stats" / "diversity.json", json.dumps(profile))
    except Exception:  # a missing cache is not a scan failure
        return {}
    if not profile:
        return {}

    stop_div = (profile.get("arms") or {}).get("stop") or {}
    prior_div = (profile.get("arms") or {}).get("prior") or {}
    cross = profile.get("cross") or {}
    summary: dict[str, Any] = {
        "effective_molecules": stop_div.get("effective_molecules"),
        "prior_effective_molecules": prior_div.get("effective_molecules"),
        "effective_scaffolds": stop_div.get("effective_scaffolds"),
        "prior_effective_scaffolds": prior_div.get("effective_scaffolds"),
        "internal_diversity": stop_div.get("internal_diversity"),
        "scaffold_internal_diversity": stop_div.get("scaffold_internal_diversity"),
        "sidechain_internal_diversity": stop_div.get("sidechain_internal_diversity"),
        "scaffold_entropy_ratio": stop_div.get("scaffold_entropy_ratio"),
        "nn_sim_p50": stop_div.get("nn_sim_p50"),
        "nn_sim_p90": stop_div.get("nn_sim_p90"),
        "near_dup_rate": stop_div.get("near_dup_rate"),
        "novelty_rate": cross.get("novelty_rate"),
        "prior_echo_rate": cross.get("prior_echo_rate"),
        "prior_sim_p50": cross.get("prior_sim_p50"),
    }
    stop_arm = arms.get("stop") or {}
    summary["effective_desired_yield"] = round(
        (stop_arm.get("desired_ratio") or 0.0)
        * (summary.get("effective_scaffolds") or 0.0)
        / max(n_expected, 1),
        4,
    )
    for key in ("thumb", "arms", "cross"):
        if profile.get(key):
            summary[key] = profile[key]
    return summary


def _load_training_summary(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("summary") or (manifest.get("config") or {}).get("summary") or {}
    if not summary and (run / "run.log").is_file():
        try:
            for line in (run / "run.log").read_text(encoding="utf-8", errors="replace").splitlines():
                if "training summary:" in line:
                    return ast.literal_eval(line.split("training summary: ", 1)[1].strip())
        except Exception:
            return {}
    return summary


def scan_run(run: Path, legacy: bool, *, use_cache: bool = True) -> dict[str, Any] | None:
    """Compute every collapse/diversity/curve statistic for one run directory."""
    if use_cache:
        cached = _load_collapse_cache(run, legacy)
        if cached is not None:
            return cached
    manifest_path = run / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = manifest.get("config") or {}
    if not config:
        return None

    frames = _load_legacy(run) if legacy else _load_current(run)
    arm_scales: dict[str, float] = {}
    n_expected = int(config.get("n_final") or config.get("n_eval_samples") or 5000)
    arms: dict[str, Any] = {}
    for role in ARM_ROLES:
        frame = frames.get(role)
        if frame is None or not len(frame):
            arms[role] = None
            continue
        rocs_col = _rocs_column(frame)
        arm_scale = 1.0 if legacy else frame_rocs_scale(frame, rocs_col, config)
        arm_scales[role] = arm_scale
        arms[role] = arm_stats(frame, rocs_col, arm_scale)

    stop_chemistry: dict[str, Any] = {}
    prior_chemistry: dict[str, Any] = {}
    if arms.get("stop") and arms.get("prior"):
        for role in ("stop", "prior"):
            frame = frames[role]
            valid = frame[_numeric(frame, "Valid").fillna(0) == 1]
            column = "Canonical" if "Canonical" in frame.columns else "Canonical_SMILES"
            canonicals = valid[column].dropna().astype(str)
            canonicals = canonicals[canonicals != ""].drop_duplicates()
            chem = chemistry_stats(canonicals)
            if role == "stop":
                stop_chemistry = chem
            else:
                prior_chemistry = chem

    judged = arms.get("stop") or {"n": 0}
    result = verdict(
        judged, arms.get("prior") or {"n": 0}, stop_chemistry, prior_chemistry, n_expected
    )
    diversity_summary = _diversity_summary(run, arms, n_expected)

    hac = stop_chemistry.get("hac_median")
    hac_prior = prior_chemistry.get("hac_median")
    curve = _training_curve(run, legacy)
    summary = _load_training_summary(run, manifest)

    stop_epoch = summary.get("desired_ratio_freeze_epoch")
    best_desired_epoch = summary.get("best_desired_ratio_epoch")
    gmean_epoch = summary.get("best_avg_gmean_epoch")
    desired_frozen = summary.get("desired_ratio_frozen")

    if best_desired_epoch is None and curve.get("desired"):
        des_list = curve["desired"]
        if des_list:
            best_desired_epoch = max(range(len(des_list)), key=lambda i: des_list[i]) + 1
    if stop_epoch is None:
        stop_epoch = (
            summary.get("epochs_run") or curve.get("epochs") or len(curve.get("desired") or [])
        )

    curve["stop_epoch"] = stop_epoch
    curve["best_desired_epoch"] = best_desired_epoch
    curve["gmean_epoch"] = gmean_epoch
    curve["desired_frozen"] = desired_frozen

    record: dict[str, Any] = {
        "run": run.name,
        "label": run.name,
        "repo": "legacy" if legacy else "current",
        "legacy": legacy,
        "state": manifest.get("state", "unknown"),
        "axes": _axes(config, legacy),
        "family": "ROCS+SA+QED" if config.get("include_qed") else "ROCS+SA",
        "hours": round(manifest["elapsed_seconds"] / 3600, 1)
        if manifest.get("elapsed_seconds") is not None
        else None,
        "epochs_run": summary.get("epochs_run") or curve.get("epochs"),
        "n_expected": n_expected,
        "arms": arms,
        "chemistry": {"stop": stop_chemistry, "prior": prior_chemistry},
        "diversity": diversity_summary,
        "hac_drift": round(hac - hac_prior, 1) if hac and hac_prior else None,
        "postpeak_decay": round(arms["best"]["unique_ratio"] - arms["stop"]["unique_ratio"], 4)
        if arms.get("best") and arms.get("stop") and arms["best"]["unique_ratio"] is not None
        else None,
        "curve": curve,
        "stop_epoch": stop_epoch,
        "best_desired_epoch": best_desired_epoch,
        "gmean_epoch": gmean_epoch,
        "desired_frozen": desired_frozen,
        "verdict": result["label"],
        "flags": result["flags"],
        "reasons": result["reasons"],
    }

    stop_frame = frames.get("stop")
    if stop_frame is not None and len(stop_frame):
        stop_scale = arm_scales.get("stop", 1.0)
        rocs_col = _rocs_column(stop_frame)
        if result["label"] == "collapsed":
            record["repeats"] = _top_molecules(stop_frame, rocs_col, stop_scale, 3)
        lead = _lead_molecule(stop_frame, rocs_col, stop_scale)
        if lead and lead.get("smiles"):
            record["lead"] = lead
        record["best_desired"] = _best_desired_molecules(
            stop_frame, rocs_col, stop_scale, count=10
        )

    if legacy:
        record["axes"]["scheme"] = f"{record['axes'].get('scheme') or 'PRCD'}"
    if use_cache:
        _save_collapse_cache(run, legacy, record)
    return record


def missing_arms(run: Path, record: dict[str, Any]) -> list[str]:
    """Return missing stop-arm table filenames that a killed run still owes."""
    from _david.code.run import arm_is_complete

    name = "final_desired_ratio_at_stop.tsv"
    if record["arms"].get("stop") is None or not arm_is_complete(run / name, record["n_expected"]):
        return [name]
    return []


# --------------------------------------------------------------------------------------
# Many runs: payload and highlights
# --------------------------------------------------------------------------------------


def _frontier(records: list[dict[str, Any]]) -> list[str]:
    """Runs not dominated on ``(desired, unique)`` per objective family among full-scale runs."""
    scored = [
        r
        for r in records
        if r["arms"].get("stop")
        and r["arms"]["stop"]["desired_ratio"] is not None
        and r["arms"]["stop"]["unique_ratio"] is not None
        and r["verdict"] not in ("pilot", "incomplete")
    ]
    if not scored:
        return []
    out: list[str] = []
    for include_qed in (False, True):
        group = [
            r for r in scored if bool((r.get("axes") or {}).get("include_qed")) == include_qed
        ]
        if not group:
            continue
        points = np.array(
            [
                [r["arms"]["stop"]["desired_ratio"], r["arms"]["stop"]["unique_ratio"]]
                for r in group
            ]
        )
        mask = pareto_mask(points)
        out.extend(r["run"] for r, on in zip(group, mask, strict=True) if on)
    return out


def is_legacy_layout(run: Path) -> bool:
    """Return whether ``run`` uses the legacy WITH_BUGS ``raw_data/`` directory layout."""
    return (run / "raw_data").is_dir()


def build(
    roots: list[Path],
    out: Path | None = None,
    since: str = SINCE,
    *,
    use_cache: bool = True,
) -> Path:
    """Scan ``roots`` and write ``collapse_scan.json`` and ``collapse_scan.tsv``."""
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            logger.warning("root %s does not exist; skipping", root)
            continue
        candidates.extend(iter_run_dirs(root))

    chosen_dirs: dict[str, Path] = {}
    for run in sorted(candidates, key=lambda p: ("pilots_and_other" in p.parts, str(p))):
        chosen_dirs.setdefault(run.name, run)

    records: list[dict[str, Any]] = []
    for run in sorted(chosen_dirs.values(), key=lambda p: (p.name, str(p))):
        date = run_date(run.name)
        if date is not None and date < since:
            continue
        legacy = is_legacy_layout(run)
        record = scan_run(run, legacy, use_cache=use_cache)
        if record is None:
            logger.warning("%s: no usable manifest; skipped", run.name)
            continue
        if record["state"] not in ("complete", "running") and not legacy:
            record["missing"] = missing_arms(run, record)
        records.append(record)

    if not records:
        raise SystemExit("no runs with a readable manifest under the given roots")

    frontier = set(_frontier(records))
    for record in records:
        record["frontier"] = record["run"] in frontier

    counts: dict[str, int] = {}
    for record in records:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    rebuildable = [r for r in records if r.get("missing")]

    payload = {
        "generated": datetime.now().isoformat(" ", "seconds"),
        "roots": [str(root) for root in roots],
        "since": since,
        "runs": records,
        "counts": counts,
        "totals": {
            "runs": len(records),
            "complete": sum(1 for r in records if r["state"] == "complete"),
            "killed": sum(1 for r in records if r["state"] != "complete"),
            "rebuildable": len(rebuildable),
            "frontier": len(frontier),
        },
    }

    target = out if out is not None and out.suffix == "" else (out.parent if out else roots[0])
    target.mkdir(parents=True, exist_ok=True)
    scan_path = target / "collapse_scan.json"
    atomic_write_text(scan_path, json.dumps(payload, indent=1))
    _write_tsv(payload, target / "collapse_scan.tsv")
    logger.info("wrote %s (%d runs)", scan_path, len(records))
    return scan_path


def _write_tsv(payload: dict[str, Any], path: Path) -> None:
    """Write one flat summary row per run for spreadsheet/pandas inspection."""
    rows = []
    for r in payload["runs"]:
        stop = r["arms"].get("stop") or {}
        div = r.get("diversity") or {}
        rows.append(
            {
                "run": r["run"],
                "repo": r["repo"],
                "state": r["state"],
                "verdict": r["verdict"],
                "flags": ";".join(r["flags"]),
                "frontier": r["frontier"],
                **{f"axis_{k}": v for k, v in r["axes"].items()},
                "family": r["family"],
                "epochs": r["epochs_run"],
                "hours": r["hours"],
                "unique": stop.get("unique_ratio"),
                "top10": stop.get("top10_share"),
                "desired": stop.get("desired_ratio"),
                "valid": stop.get("valid_ratio"),
                "rocs_median": stop.get("rocs_median"),
                "rocs_max": stop.get("rocs_max"),
                "hac_median": (r["chemistry"].get("stop") or {}).get("hac_median"),
                "hac_drift": r["hac_drift"],
                "postpeak_decay": r["postpeak_decay"],
                "unique_prior": (r["arms"].get("prior") or {}).get("unique_ratio"),
                "desired_prior": (r["arms"].get("prior") or {}).get("desired_ratio"),
                "eff_molecules": div.get("effective_molecules"),
                "eff_scaffolds": div.get("effective_scaffolds"),
                "internal_div": div.get("internal_diversity"),
                "scaffold_internal_div": div.get("scaffold_internal_diversity"),
                "sidechain_internal_div": div.get("sidechain_internal_diversity"),
                "scaffold_entropy": div.get("scaffold_entropy_ratio"),
                "edy": div.get("effective_desired_yield"),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------------------
# Depictions (shared with atlas_v2)
# --------------------------------------------------------------------------------------


def _depict(smiles: str) -> str:
    """Render ``smiles`` as an inline SVG snippet (empty string on parse failure)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(200, 150)
    drawer.drawOptions().clearBackground = False
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=None,
        help="run directories to scan (default: this repo's output)",
    )
    parser.add_argument(
        "--since",
        default=SINCE,
        metavar="YYYYMMDD",
        help=f"count only runs from this date onwards (default: {SINCE})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass per-run cache_stats/collapse_run.json and recompute from tables",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    roots = args.roots or [PROJECT_ROOT / "_david" / "output"]
    scan_path = build(roots, since=args.since, use_cache=not args.no_cache)
    print(f"scan data: {scan_path}")
    print(f"scan table: {scan_path.with_suffix('.tsv')}")
    print("atlas: python -m _david.code.analysis.atlas_v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
