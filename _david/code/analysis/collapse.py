"""Scan every run for mode collapse: ``python -m _david.code.analysis.collapse``.

The per-epoch metrics cannot see collapse: they are computed on ``n_eval`` (50) draws, and
a policy that repeats one series every few hundred draws still looks unique at n=50. The
5,000-molecule final arms can, file by file, so this module reads every arm table of every
run under the roots it is given, computes per-arm distribution statistics, and labels each
run. Both schemas are understood: this repo's arms (``prior.tsv``, ``final_*.tsv``) and the
legacy WITH_BUGS repo's ``raw_data/*_raw.csv.gz``.

Outputs ``collapse_scan.json``, ``collapse_scan.tsv`` and one self-contained
``collapse_atlas.html``. The verdict is a triage label with its reasons attached, not a
statistic; the atlas exists so the trade-off itself (desired vs unique) can be read off the
scatter, thresholds argued with, and the interesting runs found by eye.

Two rules fix the scope. Every judgement — verdict, frontier, medals — reads the **stop
arm** (``final_desired_ratio_at_stop.tsv``), the live final policy, never the gmean
checkpoint's arm: comparing selection rules is not comparing configurations. And only
runs from :data:`SINCE` onwards are counted: everything earlier belongs to the legacy
WITH_BUGS regime and this repo's first probes, whose numbers answer different questions.

    python -m _david.code.analysis.collapse [--roots DIR ...] [--since YYYYMMDD] [--out FILE.html]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.analysis._helpers import pareto_mask, rocs_scale
from _david.code.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: The legacy checkout. Not scanned by default: every run in it predates :data:`SINCE`,
#: so the date rule already excludes it; point ``--roots`` here to override.
LEGACY_ROOT = PROJECT_ROOT.parent / "WITH_BUGS_drugex" / "_david" / "output"

#: Only runs from this date onwards are counted (``YYYYMMDD`` prefix of the run dir name).
#: 2026-11-10 is the restart of the campaign on the fixed pipeline: earlier runs either
#: carry the WITH_BUGS scoring regime or probe it, and their arm numbers are not comparable.
SINCE = "20261110"

#: Arms in reporting order, named by role: the untrained baseline, the checkpoint the
#: desired-ratio criterion froze, the live policy at stop, and the gmean checkpoint.
ARM_ROLES = ("prior", "best", "stop", "gmean")

#: Verdict thresholds. These are triage cut points on this campaign's own distribution
#: (prior unique ~0.95, worst arms 0.25): a run below COLLAPSED_UNIQUE repeats its top
#: series more often than not, and SERIES_LOCK is the top-10 molecules' share at which a
#: gallery of the arm stops being a gallery of distinct findings.
COLLAPSED_UNIQUE = 0.35
CONCENTRATED_UNIQUE = 0.70
SERIES_LOCK_TOP10 = 0.60
VALIDITY_FLOOR = 0.85
FRAGMENT_DRIFT_HAC = 0.80
NO_LEARNING_GAIN = 1.5
PILOT_MAX_N = 1000
SPARK_POINTS = 140

#: Legacy arm file -> role. The legacy pipeline kept its best-checkpoint and final-epoch
#: arms in raw_data; it has no gmean arm.
LEGACY_ARMS = {
    "prior": "baseline_raw.csv.gz",
    "best": "reinforced_raw.csv.gz",
    "stop": "reinforced_final_epoch_raw.csv.gz",
}

#: Legacy config key -> the current Config's name for the same axis.
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

#: Legacy reward-scheme names -> the abbreviations this repo's campaign uses.
LEGACY_SCHEME = {
    "pareto_crowding": "PRCD",
    "pareto_tanimoto": "PRTD",
    "weighted_sum": "WS",
    "harrington": "HARR",
}

#: canonical SMILES -> (heavy atoms, molecular weight, Murcko scaffold). One entry per
#: distinct molecule ever scanned; the same molecules recur across runs and arms, so the
#: cache pays for the descriptor pass several times over.
_MOLECULES: dict[str, tuple[int, float, str] | None] = {}


def _describe(canonical: str) -> tuple[int, float, str] | None:
    """Heavy-atom count, molecular weight and Murcko scaffold of one canonical SMILES."""
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
    """Every cheap statistic of one arm, in write_arm's own conventions.

    ``unique_ratio`` and ``desired_ratio`` mirror ``run.write_arm`` exactly (unique valid
    molecules over the whole draw, desired over the same denominator), so a number here
    can be compared with the manifest's without a translation step.
    """
    n = len(frame)
    empty = {
        "n": n,
        "valid_ratio": None,
        "unique_ratio": None,
        "top1_share": None,
        "top10_share": None,
        "desired_ratio": None,
        "rocs_median": None,
        "rocs_max": None,
    }
    if n == 0:
        return empty

    valid_mask = _numeric(frame, "Valid").fillna(0) == 1
    canon = frame.loc[valid_mask, "Canonical"] if "Canonical" in frame.columns else pd.Series(dtype=str)
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

    # write_arm's convention: desired counts a molecule once, over the whole draw.
    desired_col = _numeric(frame, "Desired")
    unique_rows = frame.loc[valid_mask].drop_duplicates(
        subset="Canonical" if "Canonical" in frame.columns else None
    )
    if len(unique_rows) and "Desired" in frame.columns:
        stats["desired_ratio"] = round(float(desired_col.loc[unique_rows.index].fillna(0).sum()) / n, 4)
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
    """Medians and scaffold concentration over one arm's unique valid molecules.

    This is the expensive pass (one RDKit parse per distinct molecule), so it is spent
    only on the arms the verdict reads: the prior and the live policy at stop.
    """
    unique = [c for c in canonicals.fillna("").astype(str).unique() if c]
    described = [d for d in (_describe(c) for c in unique) if d is not None]
    if not described:
        return {"hac_median": None, "mw_median": None, "scaffolds": 0, "scaffold_top": None}
    hac = np.array([d[0] for d in described], dtype=float)
    scaffolds = pd.Series([d[2] for d in described if d[2]], dtype=str)
    return {
        "hac_median": round(float(np.median(hac)), 1),
        "mw_median": round(float(np.median([d[1] for d in described])), 1),
        "scaffolds": int(scaffolds.size),
        "scaffold_top": round(float(scaffolds.value_counts().iloc[0]) / len(described), 4)
        if scaffolds.size
        else None,
    }


def _top_molecules(frame: pd.DataFrame, rocs_column: str | None, scale: float, count: int) -> list[dict]:
    """The most repeated molecules of an arm, with each one's best ROCS."""
    if "Canonical" not in frame.columns or not len(frame):
        return []
    valid = frame[_numeric(frame, "Valid").fillna(0) == 1].copy()
    valid["_canonical"] = valid["Canonical"].fillna("")
    repeats = valid["_canonical"].value_counts()
    valid["_rocs"] = _numeric(valid, rocs_column or "") * scale if rocs_column else math.nan
    out = []
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


def _lead_molecule(frame: pd.DataFrame, rocs_column: str | None, scale: float) -> dict | None:
    """The single best desired molecule of an arm: what winning looked like in this run."""
    if not len(frame) or rocs_column is None:
        return None
    valid = frame[_numeric(frame, "Valid").fillna(0) == 1].copy()
    if "Desired" in valid.columns:
        valid = valid[_numeric(valid, "Desired").fillna(0) == 1]
    if not len(valid):
        valid = frame[_numeric(frame, "Valid").fillna(0) == 1]
    valid["_rocs"] = _numeric(valid, rocs_column) * scale
    valid = valid.dropna(subset=["_rocs"]).nlargest(1, "_rocs")
    if not len(valid):
        return None
    smiles = str(valid.iloc[0].get("Canonical") or valid.iloc[0].get("SMILES") or "")
    return {"smiles": smiles, "rocs": round(float(valid.iloc[0]["_rocs"]), 3)} or None


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
    files = {
        "prior": "prior.tsv",
        "best": "final_desired_ratio.tsv",
        "stop": "final_desired_ratio_at_stop.tsv",
        "gmean": "final_gmean.tsv",
    }
    return {role: _read_table(run / name) for role, name in files.items()}


def _load_legacy(run: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for role, name in LEGACY_ARMS.items():
        path = run / "raw_data" / name
        if path.is_file():
            frames[role] = _read_table(path)
    return frames


def _axes(config: dict[str, Any], legacy: bool) -> dict[str, Any]:
    """The campaign's axes, under one spelling for both schemas."""
    if not legacy:
        return {key: config.get(key) for key in (
            "scheme", "epsilon", "minimize", "reference_conformers", "max_conformers",
            "include_qed", "color_ff", "alignment_shape_weight",
        )}
    mapped = {}
    for old, new in LEGACY_AXIS_MAP.items():
        value = config.get(old)
        if new == "scheme" and isinstance(value, str):
            value = LEGACY_SCHEME.get(value, value.upper() if value else value)
        mapped[new] = value
    return mapped


def _training_curve(run: Path, legacy: bool) -> dict[str, Any]:
    """Per-epoch desired/unique/valid trajectories, downsampled for the sparklines."""
    frame = pd.DataFrame()
    if not legacy and (run / "rl_fit.tsv").is_file():
        frame = _read_table(run / "rl_fit.tsv")
        columns = {"desired": "desired_ratio", "unique": "unique_ratio", "valid": "valid_ratio"}
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
        columns = {"desired": "desired_ratio", "unique": "unique_ratio", "valid": "valid_ratio"}
    else:
        return {"epochs": 0}
    if not len(frame):
        return {"epochs": 0}
    step = max(1, math.ceil(len(frame) / SPARK_POINTS))
    frame = frame.iloc[::step]
    curve = {"epochs": int(frame["Epoch"].iloc[-1]) if "Epoch" in frame else len(frame)}
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
    """One label per run, from the live policy's arm, with every reason spelled out.

    Order of precedence: a run that never finished scoring cannot be judged; a pilot-scale
    arm cannot be judged either (uniqueness at n=10 means nothing); diversity failure
    dominates everything else because a collapsed generator's reward numbers describe one
    series, not a search.
    """
    flags: list[str] = []
    reasons: list[str] = []

    if stop["n"] < 0.9 * max(n_expected, 1):
        return {"label": "incomplete", "flags": ["incomplete"], "reasons": [
            f"stop arm holds {stop['n']} of {n_expected} molecules"]}
    if n_expected < PILOT_MAX_N:
        return {"label": "pilot", "flags": ["pilot"], "reasons": [
            f"n_final={n_expected}: too few draws to measure uniqueness"]}

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
        reasons.append(f"validity fell to {stop['valid_ratio']:.2f} (prior {prior['valid_ratio']:.2f})")

    hac = stop_chemistry.get("hac_median")
    prior_hac = prior_chemistry.get("hac_median")
    if hac and prior_hac and hac < FRAGMENT_DRIFT_HAC * prior_hac:
        flags.append("fragment_drift")
        reasons.append(
            f"median heavy atoms {hac:.0f} vs prior {prior_hac:.0f}: reward bought with fragments"
        )

    desired = stop["desired_ratio"]
    prior_desired = prior["desired_ratio"]
    gained = desired is not None and prior_desired is not None and desired > NO_LEARNING_GAIN * max(prior_desired, 1e-9)
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
        reasons.append(f"unique {unique:.2f} of the prior's {prior_unique:.2f}: partially concentrated")
    elif "no_learning" in flags:
        label = "unlearned"
    else:
        label = "clean"
    return {"label": label, "flags": flags, "reasons": reasons}


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


def scan_run(run: Path, legacy: bool) -> dict[str, Any] | None:
    """Every statistic the atlas shows, for one run directory."""
    manifest_path = run / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    config = manifest.get("config") or {}
    if not config:
        return None

    frames = _load_legacy(run) if legacy else _load_current(run)
    scale = rocs_scale(config)
    if legacy:
        # The legacy pipeline stored published TanimotoCombo directly.
        scale = 1.0

    n_expected = int(config.get("n_final") or config.get("n_eval_samples") or 5000)
    arms: dict[str, Any] = {}
    for role in ARM_ROLES:
        frame = frames.get(role)
        if frame is None or not len(frame):
            arms[role] = None
            continue
        arms[role] = arm_stats(frame, _rocs_column(frame), scale)

    stop_chemistry = prior_chemistry = {}
    if arms.get("stop") and arms.get("prior"):
        for role in ("stop", "prior"):
            frame = frames[role]
            valid = frame[_numeric(frame, "Valid").fillna(0) == 1]
            column = "Canonical" if "Canonical" in frame.columns else "Canonical_SMILES"
            canonicals = valid[column].dropna().astype(str)
            canonicals = canonicals[canonicals != ""].drop_duplicates()
            if role == "stop":
                stop_chemistry = chemistry_stats(canonicals)
            else:
                prior_chemistry = chemistry_stats(canonicals)

    judged = arms.get("stop") or {"n": 0}
    result = verdict(judged, arms.get("prior") or {"n": 0}, stop_chemistry, prior_chemistry, n_expected)

    # The chemical layer rides along from its own per-run cache (fresh-checked against
    # the arm tables); lazy import: diversity reads this module's _read_table at load.
    diversity_summary: dict[str, Any] = {}
    try:
        from _david.code.analysis.diversity import load_profile

        profile = load_profile(run)
    except Exception:  # a missing cache is not a scan failure
        profile = None
    if profile:
        stop_div = (profile.get("arms") or {}).get("stop") or {}
        prior_div = (profile.get("arms") or {}).get("prior") or {}
        cross = profile.get("cross") or {}
        diversity_summary = {
            "effective_scaffolds": stop_div.get("effective_scaffolds"),
            "prior_effective_scaffolds": prior_div.get("effective_scaffolds"),
            "internal_diversity": stop_div.get("internal_diversity"),
            "nn_sim_p50": stop_div.get("nn_sim_p50"),
            "novelty_rate": cross.get("novelty_rate"),
            "prior_echo_rate": cross.get("prior_echo_rate"),
        }

    hac = stop_chemistry.get("hac_median")
    hac_prior = prior_chemistry.get("hac_median")
    curve = _training_curve(run, legacy)
    summary = manifest.get("summary") or {}

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
        "verdict": result["label"],
        "flags": result["flags"],
        "reasons": result["reasons"],
    }

    stop_frame = frames.get("stop")
    if stop_frame is not None and len(stop_frame):
        if result["label"] == "collapsed":
            record["repeats"] = _top_molecules(stop_frame, _rocs_column(stop_frame), scale, 3)
        lead = _lead_molecule(stop_frame, _rocs_column(stop_frame), scale)
        if lead and lead.get("smiles"):
            record["lead"] = lead

    if legacy:
        record["axes"]["scheme"] = f"{record['axes'].get('scheme') or 'PRCD'}"
    return record


def missing_arms(run: Path, record: dict[str, Any]) -> list[str]:
    """Final arms a killed run still owes, in score_arms' own completeness sense."""
    from _david.code.run import arm_is_complete

    owed = []
    for role, name in (
        ("best", "final_desired_ratio.tsv"),
        ("stop", "final_desired_ratio_at_stop.tsv"),
        ("gmean", "final_gmean.tsv"),
    ):
        if record["arms"].get(role) is None or not arm_is_complete(run / name, record["n_expected"]):
            owed.append(name)
    return owed


# --------------------------------------------------------------------------------------
# Many runs: payload and highlights
# --------------------------------------------------------------------------------------


def _frontier(records: list[dict[str, Any]]) -> list[str]:
    """Runs not dominated on (desired, unique) among full-scale, scored runs."""
    scored = [
        r for r in records
        if r["arms"].get("stop")
        and r["arms"]["stop"]["desired_ratio"] is not None
        and r["arms"]["stop"]["unique_ratio"] is not None
        and r["verdict"] not in ("pilot", "incomplete")
    ]
    if not scored:
        return []
    points = np.array(
        [[r["arms"]["stop"]["desired_ratio"], r["arms"]["stop"]["unique_ratio"]] for r in scored]
    )
    mask = pareto_mask(points)
    return [r["run"] for r, on in zip(scored, mask) if on]


def _run_date(name: str) -> str | None:
    """The ``YYYYMMDD`` a run dir name carries, or None when the name is undated."""
    match = re.match(r"^(\d{8})_\d{4}_", name)
    return match.group(1) if match else None


def build(roots: list[Path], out: Path, since: str = SINCE) -> Path:
    records = []
    for root in roots:
        if not root.is_dir():
            logger.warning("root %s does not exist; skipping", root)
            continue
        for run in sorted(root.iterdir()):
            if not run.is_dir() or not (run / "manifest.json").is_file():
                continue
            date = _run_date(run.name)
            if date is not None and date < since:
                continue
            legacy = not (run / "prior.tsv").is_file()
            record = scan_run(run, legacy)
            if record is None:
                logger.warning("%s: no usable manifest; skipped", run.name)
                continue
            if record["state"] != "complete" and not legacy:
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

    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "collapse_scan.json").write_text(json.dumps(payload, indent=1))
    _write_tsv(payload, out.parent / "collapse_scan.tsv")
    render(payload, out)
    return out


def _write_tsv(payload: dict[str, Any], path: Path) -> None:
    """One flat row per run, for pandas on the user's own desk."""
    rows = []
    for r in payload["runs"]:
        stop = r["arms"].get("stop") or {}
        rows.append({
            "run": r["run"], "repo": r["repo"], "state": r["state"], "verdict": r["verdict"],
            "flags": ";".join(r["flags"]), "frontier": r["frontier"],
            **{f"axis_{k}": v for k, v in r["axes"].items()},
            "family": r["family"], "epochs": r["epochs_run"], "hours": r["hours"],
            "unique": stop.get("unique_ratio"), "top10": stop.get("top10_share"),
            "desired": stop.get("desired_ratio"), "valid": stop.get("valid_ratio"),
            "rocs_median": stop.get("rocs_median"), "rocs_max": stop.get("rocs_max"),
            "hac_median": (r["chemistry"].get("stop") or {}).get("hac_median"),
            "hac_drift": r["hac_drift"], "postpeak_decay": r["postpeak_decay"],
            "unique_prior": (r["arms"].get("prior") or {}).get("unique_ratio"),
            "desired_prior": (r["arms"].get("prior") or {}).get("desired_ratio"),
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


def _depict(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(200, 150)
    drawer.drawOptions().clearBackground = False
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def render(payload: dict[str, Any], out: Path) -> None:
    """Attach depictions for the highlighted runs, then write the atlas."""
    for record in payload["runs"]:
        if record.get("lead") and record["frontier"]:
            record["lead"]["svg"] = _depict(record["lead"]["smiles"])
        for repeat in record.get("repeats") or []:
            repeat["svg"] = _depict(repeat["smiles"])
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out.write_text(_PAGE.replace("__PAYLOAD__", serialized), encoding="utf-8")
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mode collapse atlas</title>
<style>
:root { color-scheme: light dark; --bg:#f8fafc; --fg:#0f172a; --panel:#fff; --line:#e2e8f0;
  --muted:#64748b; --head:#0f172a; --accent:#4f46e5; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0b1120; --fg:#e2e8f0; --panel:#111827; --line:#1f2937; --muted:#94a3b8; --head:#020617; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
header { padding:20px 16px 14px; background:var(--head); color:#f8fafc; }
h1 { margin:0 0 4px; font-size:20px; } header p { margin:0; color:#cbd5e1; font-size:12.5px; }
.cards { display:flex; gap:12px; padding:16px 16px 4px; flex-wrap:wrap; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:110px; }
.card b { display:block; font-size:22px; } .card span { color:var(--muted); font-size:11.5px; text-transform:uppercase; }
nav { display:flex; gap:4px; padding:14px 16px 0; flex-wrap:wrap; }
nav button { border:1px solid var(--line); background:var(--panel); color:var(--fg); padding:7px 13px; border-radius:7px; cursor:pointer; }
nav button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
main { padding:14px 16px 48px; } section { display:none; } section.active { display:block; }
h2 { font-size:15px; margin:18px 0 8px; } .note { background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12; padding:9px 12px; border-radius:8px; margin:10px 0; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--panel); font-size:12.5px; border:1px solid var(--line); }
th,td { padding:6px 9px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
th { background:var(--bg); font-weight:600; cursor:pointer; position:sticky; top:0; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.pill { padding:1px 8px; border-radius:999px; font-size:11px; font-weight:600; }
.collapsed { background:#fee2e2; color:#991b1b; } .concentrated { background:#fef3c7; color:#92400e; }
.clean { background:#dcfce7; color:#166534; } .unlearned { background:#e0e7ff; color:#3730a3; }
.pilot, .incomplete { background:var(--line); color:var(--muted); }
.state { padding:1px 7px; border-radius:999px; font-size:11px; font-weight:600; }
.complete { background:#dcfce7; color:#166534; } .running { background:#fee2e2; color:#991b1b; }
.unknown { background:var(--line); color:var(--muted); }
tr.detail { display:none; } tr.detail.open { display:table-row; }
tr.detail > td { background:var(--bg); white-space:normal; }
svg.spark { vertical-align:middle; }
.mols { display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:12px; }
.mol { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:8px; }
.mol svg { width:100%; height:auto; display:block; } .mol .meta { font-size:11px; color:var(--muted); overflow-wrap:anywhere; }
.mol b { color:var(--fg); }
.runs { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }
.run { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:10px 12px; }
.run h3 { margin:0 0 4px; font-size:13px; overflow-wrap:anywhere; }
.run .meta { font-size:11.5px; color:var(--muted); }
#scatter-box { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; }
.controls { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; align-items:center; }
.controls select { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
.legend { font-size:11.5px; color:var(--muted); }
.axisgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:14px; }
.axisplot { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; }
.axisplot h3 { margin:2px 4px 6px; font-size:12.5px; }
</style></head><body>
<header><h1>Mode collapse atlas</h1><p id="sub"></p></header>
<div class="cards" id="cards"></div>
<nav id="nav"></nav>
<main>
  <section id="runs" class="active"><h2>Every run</h2>
    <div class="note">Click a row for the verdict's evidence: all four arms, the reasons, and what the model repeated.</div>
    <div class="scroll"><table id="runtable"></table></div></section>
  <section id="map"><h2>Collapse map: diversity retained vs desired reached</h2>
    <div class="controls" id="filters"></div>
    <div class="controls"><b>x-axis:</b>
      <label><input type="radio" name="scatter-x" value="unique" checked> unique / draw</label>
      <label><input type="radio" name="scatter-x" value="eff"> eff. scaffolds (log)</label>
      <span class="legend" style="margin-left:8px">unique-ratio is not chemical breadth — switch axes to see the "concentrated" band sit below the "collapsed" one.</span></div>
    <div id="scatter-box"></div>
    <div class="legend">Dashed line: the non-dominated frontier on (desired, unique). Top-left is a collapsed policy that still collects reward; top-right is the goal.</div></section>
  <section id="axes"><h2>What the axes do to uniqueness</h2><div class="axisgrid" id="axisgrid"></div></section>
  <section id="high"><h2>Runs worth reading first</h2>
    <h2>Frontier runs</h2><div class="runs" id="frontier"></div>
    <h2>What the collapsed runs collapsed onto</h2><div class="mols" id="collapsed"></div>
    <h2>Largest post-checkpoint decay (unique: best checkpoint &rarr; stop)</h2><div class="scroll"><table id="decay"></table></div></section>
  <section id="rebuild"><h2>Killed runs, rebuildable from checkpoints</h2>
    <div class="note">These finished training and were killed while scoring final arms. All keep three checkpoints; rebuild with <code>python -m _david.code.score_arms &lt;run&gt;</code>.</div>
    <div class="scroll"><table id="rebuildtable"></table></div></section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const R = P.runs;
const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=2) => v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d);
const cell = v => typeof v === 'boolean' ? (v ? 'on' : 'off') : (v ?? '—');
const stopOf = r => r.arms.stop || {};
const priorOf = r => r.arms.prior || {};

document.getElementById('sub').textContent = `${P.totals.runs} runs · ${P.totals.complete} complete · ${P.totals.killed} killed (${P.totals.rebuildable} rebuildable) · ${P.generated}`;

const VERDICTS = ['collapsed','concentrated','clean','unlearned','pilot','incomplete'];
document.getElementById('cards').innerHTML =
  `<div class="card"><b>${P.totals.runs}</b><span>runs</span></div>` +
  VERDICTS.map(v => `<div class="card"><b>${P.counts[v] || 0}</b><span>${v}</span></div>`).join('') +
  `<div class="card"><b>${P.totals.frontier}</b><span>frontier</span></div>`;

document.getElementById('nav').innerHTML = [['runs','Runs'],['map','Collapse map'],['axes','By axis'],['high','Highlights'],['rebuild','Rebuild']]
  .map(([id, label], i) => `<button data-tab="${id}" class="${i ? '' : 'active'}">${label}</button>`).join('');
document.getElementById('nav').onclick = e => {
  const id = e.target.dataset.tab; if (!id) return;
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b === e.target));
  document.querySelectorAll('section').forEach(s => s.classList.toggle('active', s.id === id));
};

const COLS = [
  ['run', 'run', r => esc(r.run)],
  ['verdict', 'verdict', r => `<span class="pill ${r.verdict}">${r.verdict}</span>${r.frontier ? ' ★' : ''}`],
  ['state', 'state', r => `<span class="state ${['complete','running'].includes(r.state) ? r.state : 'unknown'}">${r.state}</span>`],
  ['repo', 'repo', r => esc(r.repo)],
  ['epsilon', 'ε', r => num(r.axes.epsilon, 2)],
  ['scheme', 'scheme', r => esc(r.axes.scheme)],
  ['minimize', 'min', r => cell(r.axes.minimize)],
  ['reference_conformers', 'K', r => esc(r.axes.reference_conformers)],
  ['max_conformers', 'C', r => esc(r.axes.max_conformers)],
  ['include_qed', 'qed', r => cell(r.axes.include_qed)],
  ['color_ff', 'colour', r => esc(r.axes.color_ff)],
  ['alignment_shape_weight', 'w', r => num(r.axes.alignment_shape_weight, 1)],
  ['unique_ratio', 'unique', r => num(stopOf(r).unique_ratio), 1],
  ['eff_scaffolds', 'eff.scaf', r => {
    const v = (r.diversity || {}).effective_scaffolds;
    // Unique-ratio is not chemical breadth: red when the arm holds < 100 effective
    // scaffold families, whatever the verdict says.
    return v == null ? '—' : `<span style="${v < 100 ? 'color:#dc2626;font-weight:700' : ''}">${Math.round(v).toLocaleString()}</span>`;
  }, 1],
  ['nn_sim_p50', 'NN p50', r => num((r.diversity || {}).nn_sim_p50), 1],
  ['top10_share', 'top10', r => num(stopOf(r).top10_share), 1],
  ['desired_ratio', 'desired', r => num(stopOf(r).desired_ratio), 1],
  ['valid_ratio', 'valid', r => num(stopOf(r).valid_ratio), 1],
  ['rocs_median', 'ROCS', r => num(stopOf(r).rocs_median, 2), 1],
  ['hac_drift', 'ΔHAC', r => num(r.hac_drift, 1), 1],
  ['postpeak_decay', 'decay', r => num(r.postpeak_decay), 1],
  ['epochs_run', 'epochs', r => r.epochs_run ?? '—', 1],
  ['hours', 'hours', r => num(r.hours, 1), 1],
  ['curve', 'desired/epoch', r => spark(r), 0],
];
let key = 'unique_ratio', desc = false;
const fmt = (r, k) => {
  if (k === 'unique_ratio') return stopOf(r).unique_ratio;
  if (k === 'eff_scaffolds') return (r.diversity || {}).effective_scaffolds;
  if (k === 'nn_sim_p50') return (r.diversity || {}).nn_sim_p50;
  if (k === 'desired_ratio') return stopOf(r).desired_ratio;
  if (k === 'top10_share') return stopOf(r).top10_share;
  if (k === 'valid_ratio') return stopOf(r).valid_ratio;
  if (k === 'rocs_median') return stopOf(r).rocs_median;
  if (k === 'hac_drift') return r.hac_drift;
  if (k === 'postpeak_decay') return r.postpeak_decay;
  if (k === 'epochs_run') return r.epochs_run;
  if (k === 'hours') return r.hours;
  return r.axes[k] ?? r[k];
};
const draw = () => {
  const rows = [...R].sort((a, b) => {
    const x = fmt(a, key), y = fmt(b, key);
    if (x === y) return a.run < b.run ? -1 : 1;
    if (x == null) return 1; if (y == null) return -1;
    return (x > y ? 1 : -1) * (desc ? -1 : 1);
  });
  document.getElementById('runtable').innerHTML =
    `<thead><tr>${COLS.map(([k, l]) => `<th data-k="${k}">${l}${k === key ? (desc ? ' ▾' : ' ▴') : ''}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map(r => `<tr data-run="${esc(r.run)}">${COLS.map(([, , f, n]) => `<td class="${n ? 'num' : ''}">${f(r)}</td>`).join('')}</tr>` +
      detailRow(r)).join('')}</tbody>`;
};
const detailRow = r => `<tr class="detail" id="d-${esc(r.run)}"><td colspan="${COLS.length}">
  <b>${esc(r.run)}</b> — ${esc(r.verdict)}${r.frontier ? ' · frontier' : ''}<br>
  ${r.reasons.length ? r.reasons.map(esc).join('<br>') : 'no verdict reasons recorded'}<br>
  arms: ${['prior','best','stop','gmean'].map(a => r.arms[a] ?
    `${a} (n=${r.arms[a].n}, unique ${num(r.arms[a].unique_ratio)}, desired ${num(r.arms[a].desired_ratio)}, valid ${num(r.arms[a].valid_ratio)})` :
    `${a}: missing`).join(' · ')}<br>
  ${r.repeats ? `repeated: ${r.repeats.map(m => `<span title="${esc(m.smiles)}">${m.count}× ROCS ${num(m.rocs)}</span>`).join(', ')}` : ''}
  ${r.missing ? `<br>missing final arms: ${esc(r.missing.join(', '))}` : ''}</td></tr>`;
document.getElementById('runtable').onclick = e => {
  const th = e.target.closest('th'); if (th) { const k = th.dataset.k; desc = k === key ? !desc : true; key = k; draw(); return; }
  const tr = e.target.closest('tr[data-run]'); if (!tr) return;
  document.getElementById('d-' + tr.dataset.run)?.classList.toggle('open');
};
draw();

function spark(r) {
  const series = r.curve?.desired; if (!series || !series.length) return '—';
  const w = 90, h = 22, max = Math.max(...series, 1e-9);
  const pts = series.map((v, i) => `${(i / (series.length - 1 || 1)) * w},${h - v / max * (h - 3) - 1.5}`);
  return `<svg class="spark" width="${w}" height="${h}"><polyline points="${pts.join(' ')}" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>`;
}

// ---- collapse map ----
const AXES = ['scheme','epsilon','minimize','include_qed','reference_conformers','max_conformers','color_ff'];
const filters = {};
document.getElementById('filters').innerHTML = AXES.map(a =>
  `<label>${a} <select id="f-${a}"><option value="">all</option></select></label>`).join('') +
  `<label>repo <select id="f-repo"><option value="">all</option><option>current</option><option>legacy</option></select></label>`;
for (const a of [...AXES, 'repo']) {
  const sel = document.getElementById('f-' + a);
  [...new Set(R.map(r => a === 'repo' ? r.repo : r.axes[a]))].filter(v => v != null)
    .sort((x, y) => x > y ? 1 : -1).forEach(v => {
      const o = document.createElement('option'); o.value = o.textContent = v; sel.appendChild(o);
    });
  sel.onchange = () => { filters[a] = sel.value; drawScatter(); };
}
let scatterX = 'unique';
document.querySelectorAll('input[name="scatter-x"]').forEach(rb => rb.onchange = () => { scatterX = rb.value; drawScatter(); });
function drawScatter() {
  const runs = R.filter(r => r.verdict !== 'pilot' && r.verdict !== 'incomplete'
    && Object.entries(filters).every(([a, v]) => !v || (a === 'repo' ? r.repo : String(r.axes[a])) === v));
  const W = 860, H = 480, m = { l: 52, r: 16, t: 16, b: 42 };
  const xOf = r => scatterX === 'eff' ? (r.diversity || {}).effective_scaffolds : stopOf(r).unique_ratio;
  const xs = runs.map(xOf).filter(v => v != null);
  const ys = runs.map(r => stopOf(r).desired_ratio).filter(v => v != null);
  // Effective scaffolds span 1..2500: log scale, floored at 1, so both bands stay readable.
  const logX = scatterX === 'eff';
  const xmin = logX ? 1 : Math.min(...xs, 0), xmax = logX ? Math.max(...xs, 2) : Math.max(...xs, 1), ymin = 0, ymax = Math.max(...ys, 0.1);
  const sx = v => logX ? Math.log10(Math.max(v, 1)) : v;
  const X = v => m.l + (sx(v) - sx(xmin)) / (sx(xmax) - sx(xmin) || 1) * (W - m.l - m.r);
  const Y = v => H - m.b - (v - ymin) / (ymax - ymin || 1) * (H - m.t - m.b);
  const color = { collapsed: '#dc2626', concentrated: '#d97706', clean: '#16a34a', unlearned: '#4f46e5', pilot: '#94a3b8', incomplete: '#94a3b8' };
  const fr = runs.filter(r => r.frontier).sort((a, b) => stopOf(a).desired_ratio - stopOf(b).desired_ratio);
  const ticks = (lo, hi, n) => Array.from({ length: n + 1 }, (_, i) => lo + (hi - lo) * i / n);
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
    ${ticks(0, 1, 5).map(t => `<line x1="${X(t * xmax)}" y1="${m.t}" x2="${X(t * xmax)}" y2="${H - m.b}" stroke="var(--line)"/>`).join('')}
    ${ticks(0, 1, 5).map(t => `<line x1="${m.l}" y1="${Y(t * ymax)}" x2="${W - m.r}" y2="${Y(t * ymax)}" stroke="var(--line)"/>`).join('')}
    ${(logX ? [1, 10, 100, 1000, xmax].filter((v, i, a) => a.indexOf(v) === i && v <= xmax)
             : ticks(0, 1, 5).map(t => t * xmax)).map(xv =>
      `<text x="${X(xv)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">${logX ? Math.round(xv).toLocaleString() : xv.toFixed(2)}</text>`).join('')}
    ${ticks(0, 1, 5).map(t => `<text x="${m.l - 8}" y="${Y(t * ymax) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${(t * ymax).toFixed(2)}</text>`).join('')}
    <text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12">${logX ? 'effective scaffolds in the stop arm (log)' : 'unique molecules per draw (stop arm)'}</text>
    <text x="18" y="${(m.t + H - m.b) / 2}" text-anchor="middle" font-size="12" transform="rotate(-90 18 ${(m.t + H - m.b) / 2})">desired ratio</text>`;
  if (fr.length > 1) svg += `<polyline points="${fr.map(r => `${X(xOf(r))},${Y(stopOf(r).desired_ratio)}`).join(' ')}" fill="none" stroke="var(--fg)" stroke-width="1" stroke-dasharray="4 4" opacity="0.7"/>`;
  for (const r of runs) {
    const x = xOf(r), y = stopOf(r).desired_ratio;
    if (x == null || y == null) continue;
    svg += `<g><circle cx="${X(x)}" cy="${Y(y)}" r="${r.frontier ? 6 : 4}" fill="${color[r.verdict] || '#94a3b8'}"
      opacity="0.8" stroke="${r.frontier ? 'var(--fg)' : 'none'}" stroke-width="1.5">
      <title>${esc(r.run)} — ${r.verdict}
unique ${num(stopOf(r).unique_ratio)} · desired ${num(y)} · valid ${num(stopOf(r).valid_ratio)}${(r.diversity || {}).effective_scaffolds != null ? ' · eff. scaffolds ' + Math.round(r.diversity.effective_scaffolds).toLocaleString() + ' (NN p50 ' + num(r.diversity.nn_sim_p50) + ')' : ''}
ε=${num(r.axes.epsilon)} ${esc(r.axes.scheme)} K=${esc(r.axes.reference_conformers)} C=${esc(r.axes.max_conformers)} qed=${cell(r.axes.include_qed)}</title></circle></g>`;
  }
  svg += `</svg><div class="legend">${Object.entries(color).map(([k, c]) =>
    `<svg width="10" height="10" style="vertical-align:-1px"><circle cx="5" cy="5" r="4" fill="${c}"/></svg> ${k}`).join(' · ')} · ★ = frontier · ${runs.length} runs shown</div>`;
  document.getElementById('scatter-box').innerHTML = svg;
}
drawScatter();

// ---- by axis ----
const stripAxes = ['epsilon','scheme','minimize','include_qed','reference_conformers','max_conformers','color_ff','alignment_shape_weight'];
document.getElementById('axisgrid').innerHTML = stripAxes.map(a => {
  const groups = {};
  R.forEach(r => { if (r.verdict === 'pilot' || r.verdict === 'incomplete') return;
    const v = r.axes[a]; if (v == null) return;
    (groups[v] = groups[v] || []).push(r); });
  const keys = Object.keys(groups).sort((x, y) => parseFloat(x) - parseFloat(y) || (x > y ? 1 : -1));
  const W = 420, H = 200, m = { l: 34, r: 8, t: 8, b: 24 };
  const color = { collapsed: '#dc2626', concentrated: '#d97706', clean: '#16a34a', unlearned: '#4f46e5' };
  const Y = v => H - m.b - v * (H - m.t - m.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">`;
  [0, .25, .5, .75, 1].map(y => { svg += `<line x1="${m.l}" y1="${Y(y)}" x2="${W - m.r}" y2="${Y(y)}" stroke="var(--line)"/>
    <text x="${m.l - 6}" y="${Y(y) + 4}" text-anchor="end" font-size="10" fill="var(--muted)">${y}</text>`; });
  keys.forEach((k, gi) => {
    const cx = m.l + (gi + 0.5) * (W - m.l - m.r) / keys.length;
    svg += `<text x="${cx}" y="${H - 6}" text-anchor="middle" font-size="10" fill="var(--muted)">${esc(k)}</text>`;
    groups[k].forEach((r, i) => {
      const u = stopOf(r).unique_ratio; if (u == null) return;
      const jx = cx + (i / Math.max(groups[k].length - 1, 1) - 0.5) * Math.min(46, (W - m.l - m.r) / keys.length * 0.7);
      svg += `<circle cx="${jx}" cy="${Y(u)}" r="3.5" fill="${color[r.verdict] || '#94a3b8'}" opacity="0.75"><title>${esc(r.run)} unique ${num(u)}</title></circle>`;
    });
    const us = groups[k].map(r => stopOf(r).unique_ratio).filter(v => v != null);
    if (us.length) {
      const med = us.sort((x, y) => x - y)[Math.floor(us.length / 2)];
      svg += `<line x1="${cx - 14}" y1="${Y(med)}" x2="${cx + 14}" y2="${Y(med)}" stroke="var(--fg)" stroke-width="2"/>`;
    }
  });
  return `<div class="axisplot"><h3>${a} — median bar, one dot per run (y: unique)</h3>${svg}</div>`;
}).join('');

// ---- highlights ----
const frontier = R.filter(r => r.frontier).sort((a, b) =>
  (stopOf(b).desired_ratio + stopOf(b).unique_ratio) - (stopOf(a).desired_ratio + stopOf(a).unique_ratio));
document.getElementById('frontier').innerHTML = frontier.slice(0, 10).map(r => {
  const lead = r.lead || {};
  return `<div class="run"><h3>${esc(r.run)}</h3><div class="meta">
    ${esc(r.axes.scheme)} · ε=${num(r.axes.epsilon)} · K=${esc(r.axes.reference_conformers)} · C=${esc(r.axes.max_conformers)} · qed=${cell(r.axes.include_qed)} · ${esc(r.axes.color_ff)}<br>
    desired <b>${num(stopOf(r).desired_ratio)}</b> · unique <b>${num(stopOf(r).unique_ratio)}</b> · valid ${num(stopOf(r).valid_ratio)} · ROCS med ${num(stopOf(r).rocs_median)} · ${r.epochs_run} epochs<br>
    lead: ${esc(lead.smiles || '—')} (ROCS ${num(lead.rocs)})</div>
    ${lead.svg || ''}</div>`;
}).join('') || '<div class="note">No full-scale scored run.</div>';

document.getElementById('collapsed').innerHTML = R.filter(r => r.verdict === 'collapsed')
  .sort((a, b) => stopOf(a).unique_ratio - stopOf(b).unique_ratio)
  .flatMap(r => (r.repeats || []).map((m, i) => `<div class="mol">${m.svg || ''}
    <div class="meta"><b>${m.count}× per 5000</b> · ROCS ${num(m.rocs)}<br>
    ${i === 0 ? `<b>${esc(r.run)}</b> (unique ${num(stopOf(r).unique_ratio)})` : '&nbsp;'}<br>${esc(m.smiles)}</div></div>`))
  .join('') || '<div class="note">No collapsed run.</div>';

const decay = R.filter(r => r.postpeak_decay != null).sort((a, b) => b.postpeak_decay - a.postpeak_decay).slice(0, 12);
document.getElementById('decay').innerHTML = `<thead><tr><th>run</th><th>verdict</th><th>ε</th><th>scheme</th>
  <th>unique @ best</th><th>unique @ stop</th><th>lost</th></tr></thead><tbody>${decay.map(r =>
  `<tr><td>${esc(r.run)}</td><td><span class="pill ${r.verdict}">${r.verdict}</span></td>
   <td>${num(r.axes.epsilon)}</td><td>${esc(r.axes.scheme)}</td>
   <td class="num">${num(r.arms.best?.unique_ratio)}</td><td class="num">${num(stopOf(r).unique_ratio)}</td>
   <td class="num">${num(r.postpeak_decay)}</td></tr>`).join('')}</tbody>`;

// ---- rebuild ----
const reb = R.filter(r => r.missing && r.missing.length);
document.getElementById('rebuildtable').innerHTML = `<thead><tr><th>run</th><th>state</th><th>epochs trained</th>
  <th>missing arms</th><th>command</th></tr></thead><tbody>${reb.map(r =>
  `<tr><td>${esc(r.run)}</td><td><span class="state running">${esc(r.state)}</span></td>
   <td class="num">${r.epochs_run ?? '—'}</td><td>${esc(r.missing.join(', '))}</td>
   <td><code>python -m _david.code.score_arms ${esc(r.run)}</code></td></tr>`).join('')}</tbody>`;
</script></body></html>
"""


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
        "--out",
        type=Path,
        default=None,
        help="atlas path (default: <first root>/collapse_atlas.html)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    roots = args.roots or [PROJECT_ROOT / "_david" / "output"]
    out = args.out or roots[0] / "collapse_atlas.html"
    build(roots, out, since=args.since)
    print(f"collapse atlas: {out}")
    print(f"scan data: {out.parent / 'collapse_scan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
