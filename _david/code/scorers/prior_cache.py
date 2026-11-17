"""Precalculated canonical prior calibration bundles (reference.sdf + raw prior scores).

A prior calibration set depends only on:
- reference bank construction (``reference_smiles``, ``reference_attempts``,
  ``reference_conformers``, ``reference_rmsd``)
- force-field / relaxation parameters (``minimize``, ``dielectric``, ``dielectric_model``,
  ``max_iterations``, ``FORCE_FIELD``, ``PROTONATE``, ``CANONICALIZE_TAUTOMER``)
- candidate conformer & ROCS scoring axes (``max_conformers``, ``max_isomers``,
  ``max_heavy_atoms``, ``max_rotatable_bonds``, ``timeout``, ``color_ff``,
  ``alignment_shape_weight``)
- prior draw identity (``seed``, ``n_prior``, pretrained ``.pkg`` and ``.vocab`` SHA256s)

It is independent of ``scheme``, ``epsilon``, ``beta``, ``include_qed``, ``sa_reward_*``,
``fixed_thresholds``, ``threshold_percentile``, ``threshold_from_reference``, ``epochs``,
``patience``, ``min_epochs``, ``n_samples``, ``n_eval``, ``batch_size`` and ``n_final``.

Storing raw, modifier-independent columns in ``prior_scores.tsv`` lets any run project its
exact objective columns, modifiers, and thresholds in milliseconds without re-embedding or
re-scoring the 5,000 prior molecules.

CLI::
    python -m _david.code.scorers.prior_cache backfill
    python -m _david.code.scorers.prior_cache verify
    python -m _david.code.scorers.prior_cache build-grid [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.config import PROJECT_ROOT, ROCS_PUBLISHED_MAX, Config
from _david.code.reconstruct.provenance import digest
from _david.code.scorers.conformers import (
    CANONICALIZE_TAUTOMER,
    FORCE_FIELD,
    PROTONATE,
    build_reference,
)
from _david.code.scorers.utils import check_pool, read_sdf, use_double_precision_pickling
from drugex.training.scorers.properties import FAILURE_VALUES, Property
from drugex.training.scorers.sascorer import calculateScore

logger = logging.getLogger(__name__)

PRIOR_CACHE_SCHEMA_VERSION = 1

PRIOR_SCORE_COLUMNS = (
    "SMILES",
    "Canonical",
    "Valid",
    "ShapeTanimoto",
    "ColorTanimoto",
    "TanimotoCombo_raw",
    "SA_raw",
    "QED",
)

_COLOR_FF_TOKENS = {
    "default": "fd",
    "fdef": "ff",
    "oe_mills_dean": "fo",
}


@dataclasses.dataclass(frozen=True)
class PriorBundle:
    key: str
    bundle_dir: Path
    reference_sdf: Path
    scores: pd.DataFrame
    meta: dict


def prior_cache_root(cfg: Config | None = None) -> Path:
    """Resolve the prior cache directory (env override > $STORAGE_DIR > repo _david)."""
    _ = cfg  # reserved for future per-config override
    env_dir = os.environ.get("DRUGEX_PRIOR_CACHE")
    if env_dir:
        return Path(env_dir)
    storage_dir = os.environ.get("STORAGE_DIR")
    if storage_dir:
        return Path(storage_dir) / "_david" / "prior_cache"
    return PROJECT_ROOT / "_david" / "prior_cache"


def prior_cache_axes(cfg: Config) -> dict[str, object] | None:
    """Canonical dictionary of every config axis that changes reference.sdf or raw prior scores.

    Returns ``None`` when the pretrained checkpoint/vocab paths do not exist on disk (e.g. in
    unit tests with mocked ``Config.pretrained``), signalling callers to bypass the cache.
    """
    try:
        pkg, vocab = cfg.pretrained()
    except FileNotFoundError:
        return None
    pkg_sha = digest(Path(pkg))
    vocab_sha = digest(Path(vocab))
    if not pkg_sha or not vocab_sha:
        return None

    return {
        "schema_version": PRIOR_CACHE_SCHEMA_VERSION,
        # Reference bank axes
        "reference_smiles": str(cfg.reference_smiles),
        "reference_attempts": int(cfg.reference_attempts),
        "reference_conformers": int(cfg.reference_conformers),
        "reference_rmsd": round(float(cfg.reference_rmsd), 6),
        # Shared force-field / relaxation axes
        "force_field": FORCE_FIELD if cfg.minimize else None,
        "minimize": bool(cfg.minimize),
        "dielectric": round(float(cfg.dielectric), 6) if cfg.minimize else None,
        "dielectric_model": int(cfg.dielectric_model) if cfg.minimize else None,
        "max_iterations": int(cfg.max_iterations) if cfg.minimize else None,
        "protonate": bool(PROTONATE),
        "canonicalize_tautomer": bool(CANONICALIZE_TAUTOMER),
        # Candidate conformer + ROCS scoring axes
        "max_conformers": int(cfg.max_conformers),
        "max_isomers": int(cfg.max_isomers),
        "max_heavy_atoms": int(cfg.max_heavy_atoms),
        "max_rotatable_bonds": int(cfg.max_rotatable_bonds),
        "timeout": int(cfg.timeout),
        "color_ff": str(cfg.color_ff),
        "alignment_shape_weight": round(float(cfg.alignment_shape_weight), 6),
        # Prior draw axes
        "seed": int(cfg.seed),
        "n_prior": int(cfg.n_prior),
        "pretrained_pkg_sha256": pkg_sha,
        "pretrained_vocab_sha256": vocab_sha,
    }


def prior_cache_key(cfg: Config) -> str | None:
    """Human-readable deterministic slug + 12-hex SHA256 digest, or ``None`` if unhashable."""
    axes = prior_cache_axes(cfg)
    if axes is None:
        return None
    payload = json.dumps(axes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest_12 = hashlib.sha256(payload).hexdigest()[:12]
    m_tok = "m1" if cfg.minimize else "m0"
    ff_tok = _COLOR_FF_TOKENS.get(
        cfg.color_ff, re.sub(r"[^A-Za-z0-9]+", "", str(cfg.color_ff))[:6] or "ff"
    )
    s_tok = f"s{round(float(cfg.alignment_shape_weight) * 100):02d}"
    return (
        f"{m_tok}_r{int(cfg.reference_conformers)}_c{int(cfg.max_conformers)}_"
        f"{ff_tok}_{s_tok}_n{int(cfg.n_prior)}_seed{int(cfg.seed)}_{digest_12}"
    )


def _read_prior_scores_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"SMILES": str, "Canonical": str},
        keep_default_na=False,
    )
    for col in ("Valid",):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    for col in ("ShapeTanimoto", "ColorTanimoto", "TanimotoCombo_raw", "SA_raw", "QED"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    return df


def load_prior_bundle(
    cfg: Config,
    prior_smiles: list[str] | None = None,
    *,
    cache_root: Path | None = None,
) -> PriorBundle | None:
    """Load and verify a cached prior bundle matching ``cfg``, or return ``None`` on miss/mismatch."""
    axes = prior_cache_axes(cfg)
    key = prior_cache_key(cfg)
    if axes is None or key is None:
        return None

    root = cache_root or prior_cache_root(cfg)
    bundle_dir = root / key
    meta_path = bundle_dir / "meta.json"
    ref_path = bundle_dir / "reference.sdf"
    tsv_path = bundle_dir / "prior_scores.tsv"
    if not (meta_path.is_file() and ref_path.is_file() and tsv_path.is_file()):
        return None

    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        logger.warning("prior cache %s unreadable meta.json (%s); bypassing", key, error)
        return None

    if meta.get("schema_version") != PRIOR_CACHE_SCHEMA_VERSION or meta.get("axes") != axes:
        logger.warning("prior cache %s metadata axes mismatch; bypassing", key)
        return None
    if digest(ref_path) != meta.get("reference_sha256"):
        logger.warning("prior cache %s reference.sdf checksum mismatch; bypassing", key)
        return None
    if digest(tsv_path) != meta.get("prior_scores_sha256"):
        logger.warning("prior cache %s prior_scores.tsv checksum mismatch; bypassing", key)
        return None

    try:
        df = _read_prior_scores_tsv(tsv_path)
    except Exception as error:
        logger.warning("prior cache %s unreadable prior_scores.tsv (%s); bypassing", key, error)
        return None

    if not set(PRIOR_SCORE_COLUMNS).issubset(df.columns):
        logger.warning("prior cache %s missing columns; bypassing", key)
        return None

    if prior_smiles is not None and (
        len(df) != len(prior_smiles) or df["SMILES"].tolist() != list(prior_smiles)
    ):
        logger.warning(
            "prior cache %s SMILES sequence diverges from live sample (n=%d vs %d); bypassing",
            key,
            len(df),
            len(prior_smiles),
        )
        return None

    return PriorBundle(
        key=key,
        bundle_dir=bundle_dir,
        reference_sdf=ref_path,
        scores=df,
        meta=meta,
    )


def project_prior_scores(df: pd.DataFrame, cfg: Config, scorers: list) -> pd.DataFrame:
    """Project raw cached prior metrics onto the exact DataFrame returned by ``probe.getScores``."""
    valid = df["Valid"].to_numpy(dtype=int)
    valid_mask = valid == 1

    rocs_scorer = scorers[0]
    rocs_key_raw = rocs_scorer.getKey()
    rocs_key = rocs_key_raw[0] if isinstance(rocs_key_raw, list) else rocs_key_raw
    rocs_mod = rocs_scorer.getModifier()
    rocs_raw = df["TanimotoCombo_raw"].to_numpy(dtype=float)
    rocs_vals = np.asarray(rocs_mod(rocs_raw), dtype=float) if rocs_mod is not None else rocs_raw.copy()
    rocs_vals = np.where(valid_mask, rocs_vals, 0.0)

    sa_scorer = scorers[1]
    sa_key_raw = sa_scorer.getKey()
    sa_key = sa_key_raw[0] if isinstance(sa_key_raw, list) else sa_key_raw
    sa_mod = sa_scorer.getModifier()
    sa_raw = df["SA_raw"].to_numpy(dtype=float)
    # Invalid molecules carry SA_raw=0.0, which a descending SA modifier maps to 1.0 unless masked.
    sa_input = np.where(valid_mask, sa_raw, FAILURE_VALUES["SA"])
    sa_vals = np.asarray(sa_mod(sa_input), dtype=float) if sa_mod is not None else sa_raw.copy()
    sa_vals = np.where(valid_mask, sa_vals, 0.0)

    columns: dict[str, object] = {
        "Valid": valid.astype(float),
        rocs_key: rocs_vals,
        sa_key: sa_vals,
    }
    if cfg.include_qed and len(scorers) >= 3:
        qed_scorer = scorers[2]
        qed_key_raw = qed_scorer.getKey()
        qed_key = qed_key_raw[0] if isinstance(qed_key_raw, list) else qed_key_raw
        qed_mod = qed_scorer.getModifier()
        qed_raw = df["QED"].to_numpy(dtype=float)
        qed_vals = np.asarray(qed_mod(qed_raw), dtype=float) if qed_mod is not None else qed_raw.copy()
        columns[qed_key] = np.where(valid_mask, qed_vals, 0.0)

    columns["Desired"] = np.zeros(len(df), dtype=int)
    return pd.DataFrame(columns)


def project_prior_diagnostics(df: pd.DataFrame) -> dict[str, list[float]]:
    valid_mask = df["Valid"].to_numpy(dtype=int) == 1
    shape = np.where(valid_mask, df["ShapeTanimoto"].to_numpy(dtype=float), 0.0)
    color = np.where(valid_mask, df["ColorTanimoto"].to_numpy(dtype=float), 0.0)
    return {
        "ShapeTanimoto": shape.tolist(),
        "ColorTanimoto": color.tolist(),
    }


def _compute_canonical_sa_qed(
    smiles: list[str], valid: np.ndarray, qed_existing: np.ndarray | None = None
) -> tuple[list[str], list[float], list[float]]:
    canonical: list[str] = []
    sa_raw: list[float] = []
    qed_vals: list[float] = []
    mols_for_qed: list[Chem.Mol | None] = []
    for value, is_valid in zip(smiles, valid):
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) and value else None
        if mol is None or int(is_valid) != 1:
            canonical.append("")
            sa_raw.append(0.0)
            mols_for_qed.append(None)
            continue
        canonical.append(Chem.MolToSmiles(mol))
        mols_for_qed.append(mol)
        try:
            sa_raw.append(float(calculateScore(mol)))
        except Exception:
            sa_raw.append(float(FAILURE_VALUES["SA"]))

    if qed_existing is not None:
        qed_vals = [float(v) if int(ok) == 1 else 0.0 for v, ok in zip(qed_existing, valid)]
    else:
        raw_qed = Property("QED").getScores(mols_for_qed)
        qed_vals = [float(v) if int(ok) == 1 else 0.0 for v, ok in zip(raw_qed, valid)]
    return canonical, sa_raw, qed_vals


def _write_bundle_atomic(
    bundle_dir: Path,
    *,
    key: str,
    axes: dict[str, object],
    reference_sdf: Path,
    raw_df: pd.DataFrame,
    origin: str,
) -> PriorBundle:
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = bundle_dir.parent / f".tmp.{key}.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        ref_out = tmp_dir / "reference.sdf"
        shutil.copy2(reference_sdf, ref_out)
        refs = read_sdf(ref_out)
        ref_energies = [
            round(float(mol.GetDoubleProp("Energy")), 4) for mol in refs if mol.HasProp("Energy")
        ]

        tsv_out = tmp_dir / "prior_scores.tsv"
        raw_df = raw_df.loc[:, list(PRIOR_SCORE_COLUMNS)]
        raw_df.to_csv(tsv_out, sep="\t", index=False)

        meta = {
            "schema_version": PRIOR_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "axes": axes,
            "reference_conformers_kept": len(refs),
            "reference_sha256": digest(ref_out),
            "reference_energies": ref_energies,
            "prior_scores_sha256": digest(tsv_out),
            "n_rows": len(raw_df),
            "n_valid": int((raw_df["Valid"] == 1).sum()),
            "origin": origin,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        if bundle_dir.exists():
            backup_dir = bundle_dir.parent / f".old.{key}.{os.getpid()}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            bundle_dir.replace(backup_dir)
            try:
                tmp_dir.replace(bundle_dir)
            finally:
                shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            tmp_dir.replace(bundle_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return PriorBundle(
        key=key,
        bundle_dir=bundle_dir,
        reference_sdf=bundle_dir / "reference.sdf",
        scores=raw_df,
        meta=meta,
    )


def save_prior_bundle(
    cfg: Config,
    reference_sdf: Path,
    prior_smiles: list[str],
    scores: pd.DataFrame,
    diagnostics: dict[str, list[float]],
    *,
    origin: str,
    cache_root: Path | None = None,
) -> PriorBundle | None:
    """Persist a modifier-independent prior bundle atomically for future runs."""
    axes = prior_cache_axes(cfg)
    key = prior_cache_key(cfg)
    if axes is None or key is None:
        return None
    if not reference_sdf.is_file():
        return None

    valid = scores["Valid"].to_numpy(dtype=int)
    valid_mask = valid == 1
    shape = np.where(valid_mask, np.asarray(diagnostics.get("ShapeTanimoto", [0.0] * len(prior_smiles)), dtype=float), 0.0)
    color = np.where(valid_mask, np.asarray(diagnostics.get("ColorTanimoto", [0.0] * len(prior_smiles)), dtype=float), 0.0)
    combo = shape + color
    qed_existing = scores["QED"].to_numpy(dtype=float) if "QED" in scores.columns else None
    canonical, sa_raw, qed_vals = _compute_canonical_sa_qed(prior_smiles, valid, qed_existing=qed_existing)

    raw_df = pd.DataFrame(
        {
            "SMILES": list(prior_smiles),
            "Canonical": canonical,
            "Valid": valid,
            "ShapeTanimoto": shape,
            "ColorTanimoto": color,
            "TanimotoCombo_raw": combo,
            "SA_raw": sa_raw,
            "QED": qed_vals,
        }
    )
    root = cache_root or prior_cache_root(cfg)
    return _write_bundle_atomic(
        root / key,
        key=key,
        axes=axes,
        reference_sdf=reference_sdf,
        raw_df=raw_df,
        origin=origin,
    )


def _config_from_dict(recorded: dict) -> Config:
    values = {}
    for field in dataclasses.fields(Config):
        if field.name not in recorded:
            continue
        value = recorded[field.name]
        if value is None:
            continue
        annotation = str(field.type)
        if "Path" in annotation:
            value = Path(value)
        elif "tuple" in annotation:
            value = tuple(value)
        values[field.name] = value
    return Config(**values)


def _find_rocs_column(frame: pd.DataFrame) -> str | None:
    for col in frame.columns:
        if col.startswith("RDKit_") and "Tanimoto" in col:
            return col
    return None


def backfill_from_outputs(
    output_root: Path | None = None,
    *,
    cache_root: Path | None = None,
    since: str = "20261110",
    force: bool = False,
) -> dict[str, object]:
    """Populate the prior cache from existing complete runs under ``output_root``."""
    from _david.code.analysis._helpers import iter_run_dirs
    from _david.code.run import arm_is_complete

    out_root = output_root or Config().output_root
    target_root = cache_root or prior_cache_root()
    target_root.mkdir(parents=True, exist_ok=True)

    run_dirs = iter_run_dirs(out_root)
    saved: list[str] = []
    skipped_existing: list[str] = []
    skipped_incomplete: list[str] = []
    warnings: list[str] = []

    for run_dir in sorted(run_dirs):
        if run_dir.name[:8] < since:
            continue
        cfg_path = run_dir / "config.json"
        prior_path = run_dir / "prior.tsv"
        ref_path = run_dir / "reference.sdf"
        if not (cfg_path.is_file() and prior_path.is_file() and ref_path.is_file()):
            skipped_incomplete.append(run_dir.name)
            continue

        try:
            recorded = json.loads(cfg_path.read_text())
            cfg = _config_from_dict(recorded)
        except Exception as error:
            warnings.append(f"{run_dir.name}: config parse error ({error})")
            continue

        if not arm_is_complete(prior_path, cfg.n_prior):
            skipped_incomplete.append(run_dir.name)
            continue

        axes = prior_cache_axes(cfg)
        key = prior_cache_key(cfg)
        if axes is None or key is None:
            warnings.append(f"{run_dir.name}: pretrained checkpoint/vocab unavailable")
            continue

        bundle_dir = target_root / key
        if not force and (bundle_dir / "meta.json").is_file() and load_prior_bundle(cfg, cache_root=target_root) is not None:
            skipped_existing.append(key)
            continue

        try:
            prior_df = pd.read_csv(
                prior_path,
                sep="\t",
                dtype={"SMILES": str, "Canonical": str},
                keep_default_na=False,
            )
        except Exception as error:
            warnings.append(f"{run_dir.name}: prior.tsv read failed ({error})")
            continue

        required = {"SMILES", "Canonical", "Valid", "ShapeTanimoto", "ColorTanimoto", "SA_raw"}
        if not required.issubset(prior_df.columns):
            skipped_incomplete.append(run_dir.name)
            continue

        valid = pd.to_numeric(prior_df["Valid"], errors="raise").to_numpy(dtype=int)
        valid_mask = valid == 1
        shape = np.where(valid_mask, pd.to_numeric(prior_df["ShapeTanimoto"]).to_numpy(dtype=float), 0.0)
        color = np.where(valid_mask, pd.to_numeric(prior_df["ColorTanimoto"]).to_numpy(dtype=float), 0.0)
        combo = shape + color

        rocs_col = _find_rocs_column(prior_df)
        if rocs_col is not None:
            rocs_scale = float(recorded.get("rocs_reward_scale") or ROCS_PUBLISHED_MAX)
            stored_rocs = pd.to_numeric(prior_df[rocs_col]).to_numpy(dtype=float)
            diff = np.max(np.abs(combo[valid_mask] / rocs_scale - stored_rocs[valid_mask])) if valid_mask.any() else 0.0
            if diff > 1e-4:
                warnings.append(
                    f"{run_dir.name}: Shape+Color/{rocs_scale:g} deviates from {rocs_col} by {diff:.6g}; skipping"
                )
                continue

        sa_raw = np.where(valid_mask, pd.to_numeric(prior_df["SA_raw"]).to_numpy(dtype=float), 0.0)
        smiles_list = prior_df["SMILES"].tolist()
        canonical_list = [c if ok == 1 else "" for c, ok in zip(prior_df["Canonical"].tolist(), valid)]
        if "QED" in prior_df.columns:
            qed_vals = np.where(valid_mask, pd.to_numeric(prior_df["QED"]).to_numpy(dtype=float), 0.0)
        else:
            _, _, qed_list = _compute_canonical_sa_qed(smiles_list, valid, qed_existing=None)
            qed_vals = np.asarray(qed_list, dtype=float)

        raw_df = pd.DataFrame(
            {
                "SMILES": smiles_list,
                "Canonical": canonical_list,
                "Valid": valid,
                "ShapeTanimoto": shape,
                "ColorTanimoto": color,
                "TanimotoCombo_raw": combo,
                "SA_raw": sa_raw,
                "QED": qed_vals,
            }
        )
        _write_bundle_atomic(
            bundle_dir,
            key=key,
            axes=axes,
            reference_sdf=ref_path,
            raw_df=raw_df,
            origin=f"backfilled:{run_dir.name}",
        )
        saved.append(key)
        logger.info("backfilled %s from %s (%d rows, %d valid)", key, run_dir.name, len(raw_df), int(valid_mask.sum()))

    return {
        "cache_root": str(target_root),
        "saved": sorted(set(saved)),
        "skipped_existing": len(set(skipped_existing)),
        "skipped_incomplete": len(skipped_incomplete),
        "warnings": warnings,
    }


def verify_cache(cache_root: Path | None = None) -> tuple[int, list[str]]:
    """Verify integrity and invariants of every bundle under ``cache_root``."""
    root = cache_root or prior_cache_root()
    if not root.is_dir():
        return 0, [f"cache root missing: {root}"]

    errors: list[str] = []
    checked = 0
    for bundle_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        checked += 1
        meta_path = bundle_dir / "meta.json"
        ref_path = bundle_dir / "reference.sdf"
        tsv_path = bundle_dir / "prior_scores.tsv"
        if not (meta_path.is_file() and ref_path.is_file() and tsv_path.is_file()):
            errors.append(f"{bundle_dir.name}: missing bundle file(s)")
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as error:
            errors.append(f"{bundle_dir.name}: invalid meta.json ({error})")
            continue
        if meta.get("schema_version") != PRIOR_CACHE_SCHEMA_VERSION:
            errors.append(f"{bundle_dir.name}: unexpected schema_version {meta.get('schema_version')!r}")
        if digest(ref_path) != meta.get("reference_sha256"):
            errors.append(f"{bundle_dir.name}: reference.sdf checksum mismatch")
        if digest(tsv_path) != meta.get("prior_scores_sha256"):
            errors.append(f"{bundle_dir.name}: prior_scores.tsv checksum mismatch")
        try:
            df = _read_prior_scores_tsv(tsv_path)
        except Exception as error:
            errors.append(f"{bundle_dir.name}: prior_scores.tsv error ({error})")
            continue
        if list(df.columns) != list(PRIOR_SCORE_COLUMNS):
            errors.append(f"{bundle_dir.name}: unexpected columns {list(df.columns)}")
            continue
        if len(df) != meta.get("n_rows"):
            errors.append(f"{bundle_dir.name}: row count {len(df)} != meta {meta.get('n_rows')}")
        invalid = df["Valid"] == 0
        if invalid.any():
            errors.extend(
                f"{bundle_dir.name}: invalid rows have nonzero {col}"
                for col in ("ShapeTanimoto", "ColorTanimoto", "TanimotoCombo_raw", "SA_raw", "QED")
                if not np.allclose(df.loc[invalid, col], 0.0)
            )
        combo_diff = np.max(np.abs(df["TanimotoCombo_raw"] - (df["ShapeTanimoto"] + df["ColorTanimoto"])))
        if combo_diff > 1e-6:
            errors.append(f"{bundle_dir.name}: TanimotoCombo_raw != Shape+Color (max diff {combo_diff})")
    return checked, errors


def campaign_prior_configs() -> list[tuple[str, Config]]:
    """Unique prior-affecting configurations across ``campaign.BATCHES``."""
    from _david.code.metacentrum import campaign
    from _david.code.run import config_from_args

    by_key: dict[str, tuple[str, Config]] = {}
    for batch in campaign.BATCHES.values():
        for name, settings in campaign.runs(batch):
            cfg = config_from_args([f"{k}={v}" for k, v in sorted(settings.items())])
            key = prior_cache_key(cfg)
            if key is not None and key not in by_key:
                by_key[key] = (name, cfg)
    return list(by_key.values())


def standard_grid_configs(
    *,
    color_ffs: tuple[str, ...] = ("default", "fdef", "oe_mills_dean"),
    reference_conformers: tuple[int, ...] = (1, 4),
    minimizes: tuple[bool, ...] = (True, False),
    max_conformers: tuple[int, ...] = (10, 20, 30),
    alignment_shape_weights: tuple[float, ...] = (0.5,),
) -> list[tuple[str, Config]]:
    """Explicit cartesian grid over color_ff x reference_conformers x minimize x max_conformers."""
    items: list[tuple[str, Config]] = []
    for ff in color_ffs:
        for k in reference_conformers:
            for minimize in minimizes:
                for c in max_conformers:
                    for w in alignment_shape_weights:
                        cfg = Config(
                            color_ff=ff,
                            reference_conformers=k,
                            minimize=minimize,
                            max_conformers=c,
                            alignment_shape_weight=w,
                        )
                        key = prior_cache_key(cfg)
                        if key is not None:
                            items.append((key, cfg))
    return items


def build_missing_bundles(
    configs: list[tuple[str, Config]],
    *,
    cache_root: Path | None = None,
    dry_run: bool = False,
    n_jobs: int | None = None,
) -> dict[str, object]:
    """Build and cache any prior bundles in ``configs`` that do not yet exist."""
    from _david.code.run import load_model, sample, seed_everything
    from _david.code.scorers.objectives import build_scorers, calibrate_thresholds
    from _david.code.scorers.utils import components
    from drugex.data.corpus.vocabulary import VocSmiles

    root = cache_root or prior_cache_root()
    missing: list[tuple[str, Config]] = []
    present: list[str] = []
    for label, cfg in configs:
        if n_jobs is not None:
            cfg = dataclasses.replace(cfg, n_jobs=n_jobs)
        key = prior_cache_key(cfg)
        if key is None:
            continue
        if load_prior_bundle(cfg, cache_root=root) is not None:
            present.append(key)
        else:
            missing.append((label, cfg))

    if dry_run or not missing:
        return {
            "present": len(present),
            "missing": [(label, prior_cache_key(cfg)) for label, cfg in missing],
            "built": [],
        }

    check_pool()
    use_double_precision_pickling()
    built: list[str] = []

    import tempfile

    for label, cfg in missing:
        key = prior_cache_key(cfg)
        logger.info("building missing prior bundle %s (%s)", key, label)
        seed_everything(cfg.seed)
        checkpoint, vocabulary = cfg.pretrained()
        voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)
        _ = load_model(voc, checkpoint)  # advance PyTorch RNG identically to _pipeline()
        prior = load_model(voc, checkpoint)
        for p in prior.parameters():
            p.requires_grad_(False)
        prior.eval()
        prior_smiles = sample(prior, cfg.n_prior)

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.sdf"
            references = build_reference(cfg, ref_path)
            scorers = build_scorers(cfg, references)
            _, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
            from drugex.training.environment import DrugExEnvironment
            from drugex.training.rewards import ParetoCrowdingDistance

            probe = DrugExEnvironment(scorers, [0.0] * len(scorers), ParetoCrowdingDistance())
            # Extract diagnostics computed during calibrate_thresholds via scorers directly
            probe_diag = components(probe, prior_smiles)
            save_prior_bundle(
                cfg,
                ref_path,
                prior_smiles,
                prior_scores,
                probe_diag,
                origin=f"build-grid:{label}",
                cache_root=root,
            )
        if key:
            built.append(key)
    return {
        "present": len(present),
        "missing": [(label, prior_cache_key(cfg)) for label, cfg in missing],
        "built": built,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bf = sub.add_parser("backfill", help="backfill prior cache from completed runs")
    bf.add_argument("--output-root", type=Path, default=None)
    bf.add_argument("--cache-root", type=Path, default=None)
    bf.add_argument("--since", default="20261110")
    bf.add_argument("--force", action="store_true")

    vf = sub.add_parser("verify", help="verify integrity and invariants of cached prior bundles")
    vf.add_argument("--cache-root", type=Path, default=None)

    bg = sub.add_parser("build-grid", help="inspect or prebuild missing standard prior bundles")
    bg.add_argument("--cache-root", type=Path, default=None)
    bg.add_argument("--source", choices=("campaign", "factorial"), default="campaign")
    bg.add_argument("--dry-run", action="store_true")
    bg.add_argument("--n-jobs", type=int, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "backfill":
        report = backfill_from_outputs(
            args.output_root, cache_root=args.cache_root, since=args.since, force=args.force
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "verify":
        checked, errors = verify_cache(args.cache_root)
        print(f"checked {checked} bundle(s), {len(errors)} error(s)")
        for err in errors:
            print(f"  ERROR: {err}")
        return 1 if errors else 0
    if args.command == "build-grid":
        configs = campaign_prior_configs() if args.source == "campaign" else standard_grid_configs()
        report = build_missing_bundles(
            configs, cache_root=args.cache_root, dry_run=args.dry_run, n_jobs=args.n_jobs
        )
        print(json.dumps(report, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
