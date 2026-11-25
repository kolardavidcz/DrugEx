"""Modifier-independent cache of ``reference.sdf`` + raw prior scores.

Keyed by every config axis that changes the reference bank, conformers, ROCS scores, or
prior draw (independent of RL scheme, epsilon, modifiers, QED inclusion, or thresholds).
Runs project their own modifiers and thresholds from the cached raw columns in milliseconds.

Optional prebuild CLI::
    python -m _david.code.scorers.prior_cache build-grid [--dry-run] [--n-jobs N]
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
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.config import PROJECT_ROOT, Config
from _david.code.reconstruct.provenance import digest
from _david.code.scorers.conformers import (
    CANONICALIZE_TAUTOMER,
    FORCE_FIELD,
    PROTONATE,
    build_reference,
)
from _david.code.scorers.objectives import scorer_key
from _david.code.scorers.utils import check_pool, use_double_precision_pickling
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
_COLOR_FF_TOKENS = {"default": "fd", "fdef": "ff", "oe_mills_dean": "fo"}


@dataclasses.dataclass(frozen=True)
class PriorBundle:
    """Cached ``reference.sdf`` + raw prior scores for one scoring-axis key."""

    key: str
    bundle_dir: Path
    reference_sdf: Path
    scores: pd.DataFrame
    meta: dict[str, Any]


def prior_cache_root(cfg: Config | None = None) -> Path:
    """Resolve prior cache root (``DRUGEX_PRIOR_CACHE`` > ``$STORAGE_DIR`` > repo)."""
    _ = cfg
    if env_dir := os.environ.get("DRUGEX_PRIOR_CACHE"):
        return Path(env_dir)
    if storage_dir := os.environ.get("STORAGE_DIR"):
        return Path(storage_dir) / "_david" / "prior_cache"
    return PROJECT_ROOT / "_david" / "prior_cache"


def prior_cache_axes(cfg: Config) -> dict[str, object] | None:
    """Every config axis that affects ``reference.sdf`` or raw prior scores, or ``None`` if unhashable."""
    try:
        pkg, vocab = cfg.pretrained()
    except FileNotFoundError:
        return None
    pkg_sha, vocab_sha = digest(Path(pkg)), digest(Path(vocab))
    if not pkg_sha or not vocab_sha:
        return None
    return {
        "schema_version": PRIOR_CACHE_SCHEMA_VERSION,
        "reference_smiles": str(cfg.reference_smiles),
        "reference_attempts": int(cfg.reference_attempts),
        "reference_conformers": int(cfg.reference_conformers),
        "reference_rmsd": round(float(cfg.reference_rmsd), 6),
        "force_field": FORCE_FIELD if cfg.minimize else None,
        "minimize": bool(cfg.minimize),
        "dielectric": round(float(cfg.dielectric), 6) if cfg.minimize else None,
        "dielectric_model": int(cfg.dielectric_model) if cfg.minimize else None,
        "max_iterations": int(cfg.max_iterations) if cfg.minimize else None,
        "protonate": bool(PROTONATE),
        "canonicalize_tautomer": bool(CANONICALIZE_TAUTOMER),
        "max_conformers": int(cfg.max_conformers),
        "max_isomers": int(cfg.max_isomers),
        "max_heavy_atoms": int(cfg.max_heavy_atoms),
        "max_rotatable_bonds": int(cfg.max_rotatable_bonds),
        "timeout": int(cfg.timeout),
        "color_ff": str(cfg.color_ff),
        "alignment_shape_weight": round(float(cfg.alignment_shape_weight), 6),
        "seed": int(cfg.seed),
        "n_prior": int(cfg.n_prior),
        "pretrained_pkg_sha256": pkg_sha,
        "pretrained_vocab_sha256": vocab_sha,
    }


def _key_from_axes(cfg: Config, axes: dict[str, object]) -> str:
    payload = json.dumps(axes, sort_keys=True, separators=(",", ":")).encode()
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


def prior_cache_key(cfg: Config) -> str | None:
    """Deterministic human-readable slug + SHA-256 suffix, or ``None`` if unhashable."""
    axes = prior_cache_axes(cfg)
    return _key_from_axes(cfg, axes) if axes is not None else None


def load_prior_bundle(
    cfg: Config,
    prior_smiles: list[str] | None = None,
    *,
    cache_root: Path | None = None,
) -> PriorBundle | None:
    """Load an atomically cached prior bundle matching ``cfg``, or ``None`` on miss."""
    axes = prior_cache_axes(cfg)
    if axes is None:
        return None
    key = _key_from_axes(cfg, axes)
    bundle_dir = (cache_root or prior_cache_root(cfg)) / key
    ref_path = bundle_dir / "reference.sdf"
    tsv_path = bundle_dir / "prior_scores.tsv"
    meta_path = bundle_dir / "meta.json"
    if not (ref_path.is_file() and tsv_path.is_file()):
        return None

    df = pd.read_csv(tsv_path, sep="\t", dtype={"SMILES": str, "Canonical": str}, keep_default_na=False)
    if prior_smiles is not None and df["SMILES"].tolist() != prior_smiles:
        logger.warning("prior cache %s SMILES sequence diverged from live sample; bypassing", key)
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return PriorBundle(key=key, bundle_dir=bundle_dir, reference_sdf=ref_path, scores=df, meta=meta)


def _project_column(
    scorer: Any, raw: np.ndarray, valid_mask: np.ndarray, *, invalid_input: float | None = None
) -> tuple[str, np.ndarray]:
    modifier = scorer.getModifier()
    values = np.where(valid_mask, raw, invalid_input) if invalid_input is not None else raw
    projected = np.asarray(modifier(values), dtype=float) if modifier is not None else raw.copy()
    return scorer_key(scorer), np.where(valid_mask, projected, 0.0)


def project_prior_scores(df: pd.DataFrame, cfg: Config, scorers: Sequence[Any]) -> pd.DataFrame:
    """Project raw cached prior metrics onto the exact DataFrame returned by ``probe.getScores``."""
    valid = df["Valid"].to_numpy(dtype=int)
    mask = valid == 1
    rocs_key, rocs_vals = _project_column(scorers[0], df["TanimotoCombo_raw"].to_numpy(dtype=float), mask)
    # Invalid rows store SA_raw=0.0, which a descending SA modifier would otherwise map to 1.0.
    sa_key, sa_vals = _project_column(
        scorers[1], df["SA_raw"].to_numpy(dtype=float), mask, invalid_input=float(FAILURE_VALUES["SA"])
    )
    cols: dict[str, object] = {"Valid": valid.astype(float), rocs_key: rocs_vals, sa_key: sa_vals}
    if cfg.include_qed and len(scorers) >= 3:
        qed_key, qed_vals = _project_column(scorers[2], df["QED"].to_numpy(dtype=float), mask)
        cols[qed_key] = qed_vals
    cols["Desired"] = np.zeros(len(df), dtype=int)
    return pd.DataFrame(cols)


def project_prior_diagnostics(df: pd.DataFrame) -> dict[str, list[float]]:
    """Extract validity-masked ``ShapeTanimoto`` and ``ColorTanimoto`` lists from cached scores."""
    mask = df["Valid"].to_numpy(dtype=int) == 1
    return {
        "ShapeTanimoto": np.where(mask, df["ShapeTanimoto"].to_numpy(dtype=float), 0.0).tolist(),
        "ColorTanimoto": np.where(mask, df["ColorTanimoto"].to_numpy(dtype=float), 0.0).tolist(),
    }


def _canonical_sa_qed(
    smiles: list[str], valid: np.ndarray, qed_existing: np.ndarray | None
) -> tuple[list[str], list[float], list[float]]:
    canonical, sa_raw, mols = [], [], []
    for smi, ok in zip(smiles, valid):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if mol is None or int(ok) != 1:
            canonical.append("")
            sa_raw.append(0.0)
            mols.append(None)
            continue
        canonical.append(Chem.MolToSmiles(mol))
        mols.append(mol)
        try:
            sa_raw.append(float(calculateScore(mol)))
        except Exception:
            sa_raw.append(float(FAILURE_VALUES["SA"]))
    raw_qed = qed_existing if qed_existing is not None else Property("QED").getScores(mols)
    qed = [float(v) if int(ok) == 1 else 0.0 for v, ok in zip(raw_qed, valid)]
    return canonical, sa_raw, qed


def _write_bundle_atomic(
    bundle_dir: Path,
    *,
    key: str,
    axes: dict[str, object],
    reference_sdf: Path,
    raw_df: pd.DataFrame,
    origin: str,
) -> PriorBundle:
    """Stage bundle files into a sibling temp dir and atomically swap into ``bundle_dir``."""
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = bundle_dir.parent / f".tmp.{key}.{os.getpid()}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        shutil.copy2(reference_sdf, tmp_dir / "reference.sdf")
        raw_df = raw_df.loc[:, list(PRIOR_SCORE_COLUMNS)]
        raw_df.to_csv(tmp_dir / "prior_scores.tsv", sep="\t", index=False)
        meta: dict[str, Any] = {
            "schema_version": PRIOR_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "axes": axes,
            "n_rows": len(raw_df),
            "n_valid": int((raw_df["Valid"] == 1).sum()),
            "origin": origin,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        if bundle_dir.exists():
            backup = bundle_dir.parent / f".old.{key}.{os.getpid()}"
            shutil.rmtree(backup, ignore_errors=True)
            bundle_dir.replace(backup)
            try:
                tmp_dir.replace(bundle_dir)
            except BaseException:
                if not bundle_dir.exists() and backup.exists():
                    backup.replace(bundle_dir)
                raise
            shutil.rmtree(backup, ignore_errors=True)
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
    """Persist raw prior scores and ``reference.sdf`` atomically for reuse by future runs."""
    axes = prior_cache_axes(cfg)
    if axes is None or not reference_sdf.is_file():
        return None
    key = _key_from_axes(cfg, axes)

    valid = scores["Valid"].to_numpy(dtype=int)
    mask = valid == 1
    zeros = [0.0] * len(prior_smiles)
    shape = np.where(mask, np.asarray(diagnostics.get("ShapeTanimoto", zeros), dtype=float), 0.0)
    color = np.where(mask, np.asarray(diagnostics.get("ColorTanimoto", zeros), dtype=float), 0.0)
    qed_existing = scores["QED"].to_numpy(dtype=float) if "QED" in scores.columns else None
    canonical, sa_raw, qed = _canonical_sa_qed(prior_smiles, valid, qed_existing)

    raw_df = pd.DataFrame(
        {
            "SMILES": list(prior_smiles),
            "Canonical": canonical,
            "Valid": valid,
            "ShapeTanimoto": shape,
            "ColorTanimoto": color,
            "TanimotoCombo_raw": shape + color,
            "SA_raw": sa_raw,
            "QED": qed,
        }
    )
    return _write_bundle_atomic(
        (cache_root or prior_cache_root(cfg)) / key,
        key=key,
        axes=axes,
        reference_sdf=reference_sdf,
        raw_df=raw_df,
        origin=origin,
    )


def build_missing_bundles(
    configs: list[tuple[str, Config]],
    *,
    cache_root: Path | None = None,
    dry_run: bool = False,
    n_jobs: int | None = None,
) -> dict[str, object]:
    """Prebuild any prior bundles in ``configs`` that are not yet cached."""
    from _david.code.run import _load_models, sample, seed_everything
    from _david.code.scorers.objectives import build_scorers, calibrate_thresholds
    from _david.code.scorers.utils import components

    root = cache_root or prior_cache_root()
    present, missing = [], []
    for label, cfg in configs:
        if n_jobs is not None:
            cfg = dataclasses.replace(cfg, n_jobs=n_jobs)
        key = prior_cache_key(cfg)
        if key is None:
            continue
        if load_prior_bundle(cfg, cache_root=root) is not None:
            present.append(key)
        else:
            missing.append((label, key, cfg))

    summary = [(label, key) for label, key, _ in missing]
    if dry_run or not missing:
        return {"present": len(present), "missing": summary, "built": []}

    check_pool()
    use_double_precision_pickling()
    built: list[str] = []
    for label, key, cfg in missing:
        logger.info("building prior bundle %s (%s)", key, label)
        seed_everything(cfg.seed)
        _, _, prior = _load_models(cfg)
        prior_smiles = sample(prior, cfg.n_prior)
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.sdf"
            scorers = build_scorers(cfg, build_reference(cfg, ref_path))
            _, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
            diag = components(type("_Env", (), {"scorers": scorers})(), prior_smiles)
            save_prior_bundle(
                cfg, ref_path, prior_smiles, prior_scores, diag, origin=f"build-grid:{label}", cache_root=root
            )
        built.append(key)
    return {"present": len(present), "missing": summary, "built": built}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bg = sub.add_parser("build-grid", help="prebuild missing campaign prior bundles")
    bg.add_argument("--cache-root", type=Path, default=None)
    bg.add_argument("--dry-run", action="store_true")
    bg.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from _david.code.metacentrum import campaign
    from _david.code.run import config_from_args

    by_key: dict[str, tuple[str, Config]] = {}
    for batch in campaign.BATCHES.values():
        for name, settings in campaign.runs(batch):
            cfg = config_from_args([f"{k}={v}" for k, v in sorted(settings.items())])
            if (key := prior_cache_key(cfg)) and key not in by_key:
                by_key[key] = (name, cfg)
    report = build_missing_bundles(
        list(by_key.values()), cache_root=args.cache_root, dry_run=args.dry_run, n_jobs=args.n_jobs
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
