"""Freeze the untrained-prior arm into one standard, scored molecule dataset.

Every RL run starts by sampling ``n_eval_samples`` molecules from the *same*
pretrained SMILES-RNN prior and scoring them with the target's 3D reference
ensemble. That arm costs a full ROCS pass per run, and because
``threshold_percentile`` may calibrate the training reward from it, the baseline
drifts from run to run. This module samples it once, scores it once and stores
the result with enough provenance that later runs can load it instead.

The stored contract is the **raw** measurement (``Raw_ROCS`` TanimotoCombo on
``[0, 2]``, ``Raw_SA`` on ``[1, 10]``, QED on ``[0, 1]``) plus the 2D
descriptors. Scaled columns are re-derived from the raw values and the loading
run's ``PipelineConfig``, so a dataset built with ``rocs_use_modifier=False`` is
still valid for a run that enables the modifier:

* ``ROCS = Raw_ROCS / rocs_clip_upper`` (only when ``rocs_use_modifier``)
* ``SA   = clip((sa_score_lower - Raw_SA) / (sa_score_lower - sa_score_upper))``
* ``QED  = clip(QED)``

Layout of one dataset::

    _david/output/datasets/<name>/
        prior_molecules.csv.gz   # one row per sampled molecule, raw + scaled
        reference_3d.sdf         # the scoring reference ensemble
        statistics.json          # what the prior population is like
        manifest.json            # hashes, seed, config, versions
        README.md

CLI::

    python -m _david.code.prior_dataset build [--name N] [--n 5000] [--seed S]
    python -m _david.code.prior_dataset list
    python -m _david.code.prior_dataset verify [--name N]
    python -m _david.code.prior_dataset sync [--name N] [--host meta] [--dry-run]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.config import (
    DEFAULT_MOLECULE,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    PipelineConfig,
)
from _david.code.schema import std

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
DATASET_ROOT: Path = OUTPUT_ROOT / "datasets"
CSV_NAME: str = "prior_molecules.csv.gz"
REFERENCE_NAME: str = "reference_3d.sdf"
STATISTICS_NAME: str = "statistics.json"
MANIFEST_NAME: str = "manifest.json"
README_NAME: str = "README.md"
DEFAULT_SEED: int = 20261105
DEFAULT_N: int = 5000
DIVERSITY_SAMPLE: int = 512

RAW_ROCS: str = "Raw_ROCS"
RAW_SA: str = "Raw_SA"
FSP3: str = "Fsp3"
RINGS: str = "Rings"

DESCRIPTOR_COLUMNS: tuple[str, ...] = (
    std.MW,
    std.HAC,
    std.LOGP,
    std.TPSA,
    std.HBD,
    std.HBA,
    std.ROT_BONDS,
    FSP3,
    RINGS,
)

#: Columns a loaded arm must provide before it can be used as a baseline.
REQUIRED_COLUMNS: tuple[str, ...] = (
    std.SMILES,
    std.CANONICAL_SMILES,
    RAW_ROCS,
    RAW_SA,
    std.QED,
)


def sha256_file(path: Path) -> str:
    """Content hash used in the manifest and by ``verify``."""
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_smiles(value: Any) -> str | None:
    """Canonicalise a SMILES string; invalid input yields ``None`` instead of raising."""
    if not isinstance(value, str) or not value:
        return None
    mol = Chem.MolFromSmiles(value)
    return Chem.MolToSmiles(mol) if mol is not None else None


def molecule_tag(molecule: str | Path) -> str:
    """Short, filesystem-safe tag for a dataset name."""
    if isinstance(molecule, Path):
        return molecule.stem
    text = str(molecule)
    candidate = Path(text)
    if candidate.suffix.lower() in {".sdf", ".mol", ".mol2", ".smi", ".csv"}:
        return candidate.stem
    if text == DEFAULT_MOLECULE:
        return "egcg"
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return f"mol_{hashlib.sha256(Chem.MolToSmiles(mol).encode()).hexdigest()[:8]}"
    return "molecule"


def default_name(molecule: str | Path, n: int, seed: int) -> str:
    return f"{molecule_tag(molecule)}_prior_{n}_s{seed}"


def resolve_dataset(name_or_path: str | Path | None, root: Path = DATASET_ROOT) -> Path:
    """Resolve a dataset name or directory to its absolute path."""
    if name_or_path is None:
        raise ValueError("A prior dataset name or path is required")
    candidate = Path(name_or_path)
    if candidate.is_dir():
        return candidate
    if candidate.is_absolute():
        return candidate
    for base in (root, PROJECT_ROOT / "_david" / "data" / "datasets"):
        target = base / str(name_or_path)
        if target.is_dir():
            return target
    return root / str(name_or_path)


# --------------------------------------------------------------------------- #
# raw <-> scaled score contract
# --------------------------------------------------------------------------- #


def raw_to_scaled(
    raw: pd.DataFrame, config: PipelineConfig | None = None
) -> pd.DataFrame:
    """Derive the environment's scaled score columns from the raw measurements.

    This is the inverse of :func:`unscale_to_physical_metrics` and the single
    place where a config's modifier settings are interpreted, so a dataset can be
    reused by runs whose ROCS/SA windows differ from the build run.
    """
    cfg = config or PipelineConfig()
    out = raw.copy()

    rocs = pd.to_numeric(out[RAW_ROCS], errors="coerce")
    if cfg.rocs_use_modifier:
        upper = float(cfg.rocs_clip_upper)
        out[std.ROCS] = np.clip(rocs / upper, 0.0, 1.0) if upper > 0 else rocs * 0
    else:
        out[std.ROCS] = rocs

    sa = pd.to_numeric(out[RAW_SA], errors="coerce")
    lower, upper = float(cfg.sa_score_lower), float(cfg.sa_score_upper)
    span = lower - upper
    out[std.SA] = (
        np.clip((lower - sa) / span, 0.0, 1.0) if span != 0 else sa * 0 + 0.5
    )

    qed = pd.to_numeric(out[std.QED], errors="coerce")
    out[std.QED] = np.clip(qed, 0.0, 1.0)
    return out


def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 2D descriptors and Murcko scaffold that every figure expects."""
    out = df.copy()
    mols = [
        Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s else None
        for s in out.get(std.SMILES, [])
    ]
    out[std.CANONICAL_SMILES] = [
        Chem.MolToSmiles(m) if m is not None else None for m in mols
    ]
    out[std.VALID] = [1 if m is not None else 0 for m in mols]
    out[std.HAC] = [m.GetNumHeavyAtoms() if m is not None else 0 for m in mols]
    out[std.MW] = [Descriptors.MolWt(m) if m is not None else np.nan for m in mols]
    out[std.LOGP] = [Descriptors.MolLogP(m) if m is not None else np.nan for m in mols]
    out[std.TPSA] = [
        rdMolDescriptors.CalcTPSA(m) if m is not None else np.nan for m in mols
    ]
    out[std.HBD] = [Lipinski.NumHDonors(m) if m is not None else 0 for m in mols]
    out[std.HBA] = [Lipinski.NumHAcceptors(m) if m is not None else 0 for m in mols]
    out[std.ROT_BONDS] = [
        Lipinski.NumRotatableBonds(m) if m is not None else 0 for m in mols
    ]
    out[FSP3] = [
        rdMolDescriptors.CalcFractionCSP3(m) if m is not None else np.nan for m in mols
    ]
    out[RINGS] = [Lipinski.RingCount(m) if m is not None else 0 for m in mols]
    out[std.QED] = [QED.qed(m) if m is not None else np.nan for m in mols]
    out[RAW_SA] = [
        float(calculate_sa_score(m)) if m is not None else np.nan for m in mols
    ]
    out[std.SCAFFOLD] = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ""
        for m in mols
    ]
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quantiles(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {}
    points = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    return {
        f"p{int(p * 100):02d}": float(clean.quantile(p)) for p in points
    }


def mean_pairwise_similarity(df: pd.DataFrame, limit: int = DIVERSITY_SAMPLE) -> float | None:
    """Mean ECFP4 Tanimoto over a deterministic subsample of valid molecules."""
    smiles = [
        s
        for s in df.loc[df[std.VALID] == 1, std.CANONICAL_SMILES].dropna().tolist()
    ]
    if len(smiles) < 2:
        return None
    if len(smiles) > limit:
        picks = np.linspace(0, len(smiles) - 1, num=limit).round().astype(int)
        smiles = [smiles[i] for i in picks]
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048))
    if len(fps) < 2:
        return None
    total = 0.0
    pairs = 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total += float(sum(sims))
        pairs += len(sims)
    return total / pairs if pairs else None


def compute_statistics(
    df: pd.DataFrame,
    config: PipelineConfig,
    *,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Describe the population: validity, chemistry, diversity and reward spread."""
    from _david.code.environment import resolve_reward_scheme

    stats: dict[str, Any] = {
        "n": int(len(df)),
        "n_valid": int(df[std.VALID].sum()) if std.VALID in df else 0,
        "n_unique": int(df[std.CANONICAL_SMILES].dropna().nunique())
        if std.CANONICAL_SMILES in df
        else 0,
        "n_scaffolds": int(
            df.loc[df[std.SCAFFOLD].astype(str) != "", std.SCAFFOLD].nunique()
        )
        if std.SCAFFOLD in df
        else 0,
        "properties": {
            key: _quantiles(df[key])
            for key in (RAW_ROCS, RAW_SA, std.QED, *DESCRIPTOR_COLUMNS)
            if key in df.columns
        },
    }
    stats["valid_ratio"] = (
        stats["n_valid"] / stats["n"] if stats["n"] else 0.0
    )
    stats["unique_ratio"] = (
        stats["n_unique"] / stats["n_valid"] if stats["n_valid"] else 0.0
    )
    stats["mean_pairwise_tanimoto"] = mean_pairwise_similarity(df)
    if std.SCAFFOLD in df:
        counts = (
            df.loc[df[std.SCAFFOLD].astype(str) != "", std.SCAFFOLD]
            .value_counts()
            .head(25)
        )
        stats["top_scaffolds"] = [
            {"scaffold": str(scaffold), "count": int(count)}
            for scaffold, count in counts.items()
        ]

    active = list(thresholds or config.thresholds or ())
    if len(active) != (3 if config.include_qed else 2):
        active = list((0.35, 0.50, 0.50) if config.include_qed else (0.35, 0.50))
    scheme = resolve_reward_scheme(
        config.reward_scheme,
        weights=config.weights or ((0.50, 0.25, 0.25) if config.include_qed else (0.75, 0.25)),
        rocs_raw=(not config.rocs_use_modifier),
    )
    columns = [std.ROCS, std.SA] + ([std.QED] if config.include_qed else [])
    matrix = df[columns].to_numpy(dtype=float)
    rewards = np.asarray(
        scheme(list(df[std.SMILES].astype(str)), matrix, active)
    ).reshape(-1)
    rewards = rewards[np.isfinite(rewards)]
    if rewards.size:
        stats["reward"] = {
            "scheme": config.reward_scheme,
            "weights": list(config.weights or ()),
            "thresholds": list(active),
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "p90": float(np.percentile(rewards, 90)),
        }
    return stats


def _suggested_thresholds(df: pd.DataFrame, config: PipelineConfig) -> list[float]:
    """Thresholds a run would calibrate from this arm (percentile or config)."""
    if config.threshold_percentile is None or df.empty:
        return list(config.thresholds or ())
    p = float(config.threshold_percentile)
    values = [
        float(np.percentile(df[std.ROCS].dropna(), p)),
        float(np.percentile(df[std.SA].dropna(), p)),
    ]
    if config.include_qed and std.QED in df.columns:
        values.append(float(np.percentile(df[std.QED].dropna(), p)))
    return values


def _sample_prior(
    config: PipelineConfig,
    reference_sdf: Path,
    n: int,
    batch_size: int,
) -> pd.DataFrame:
    """Sample ``n`` raw SMILES from the untrained prior (the Stage 1 arm)."""
    from _david.code.environment import create_environment
    from _david.code.models import SequenceRNNOptimizer

    with tempfile.TemporaryDirectory(prefix="prior-models-") as scratch:
        optimizer = SequenceRNNOptimizer(models_dir=Path(scratch), config=config)
        agent, _prior, _voc = optimizer.load_models()
        env = create_environment(reference_sdf, config=config)
        generated = agent.generate(
            num_samples=n,
            batch_size=batch_size,
            drop_invalid=False,
            drop_duplicates=False,
            evaluator=None,
            progress=False,
        )
    smiles = generated["SMILES"].tolist()
    LOGGER.info("generated %d sequences; scoring them against %s", len(smiles), reference_sdf.name)
    scores = env.getScores(smiles)
    frame = pd.DataFrame({std.SMILES: smiles})
    for column in scores.columns:
        frame[column] = scores[column].values
    return frame


def build(
    molecule: str | Path = DEFAULT_MOLECULE,
    *,
    name: str | None = None,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    config: PipelineConfig | None = None,
    root: Path = DATASET_ROOT,
    force: bool = False,
) -> Path:
    """Sample, score and persist one standard prior dataset; returns its directory."""
    if n < 1:
        raise ValueError("n must be positive")
    cfg = config or PipelineConfig()
    target = root / (name or default_name(molecule, n, seed))
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass force=True (or --force) to rebuild it"
        )
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _seed_everything(seed)

    from _david.code.reference import PrepareMolecule

    preparer = PrepareMolecule(config=cfg)
    reference = preparer.generate_reference_conformers(
        molecule, output_path=target / REFERENCE_NAME
    )
    LOGGER.info("reference ensemble written to %s", reference)

    raw = _sample_prior(cfg, reference, n, cfg.batch_size)
    LOGGER.info("sampled and scored %d molecules; adding descriptors", len(raw))
    # Raw physical measurements are the contract; scaled columns are derived.
    physical = add_descriptors(raw)
    physical[RAW_ROCS] = pd.to_numeric(raw[std.ROCS], errors="coerce") * (
        float(cfg.rocs_clip_upper) if cfg.rocs_use_modifier else 1.0
    )
    if "Raw_ROCS" in raw.columns:
        physical[RAW_ROCS] = pd.to_numeric(raw["Raw_ROCS"], errors="coerce")
    frame = raw_to_scaled(physical, cfg)
    frame = frame[
        [
            std.SMILES,
            std.CANONICAL_SMILES,
            RAW_ROCS,
            RAW_SA,
            std.QED,
            std.ROCS,
            std.SA,
            *DESCRIPTOR_COLUMNS,
            std.SCAFFOLD,
            std.VALID,
        ]
    ]
    csv_path = target / CSV_NAME
    frame.to_csv(csv_path, index=False, compression="gzip")

    thresholds = _suggested_thresholds(frame, cfg)
    statistics = compute_statistics(frame, cfg, thresholds=thresholds)
    statistics_path = target / STATISTICS_NAME
    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n")

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = os.environ.get("DRUGEX_SOURCE_REVISION")
    pkg, vocab = cfg.resolve_pretrained_paths()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": target.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "molecule": str(molecule),
        "query_canonical_smiles": _reference_smiles(reference, molecule),
        "n_requested": int(n),
        "n_scored": int(len(frame)),
        "seed": int(seed),
        "sampling": {
            "model": str(pkg),
            "model_sha256": sha256_file(pkg),
            "vocab": str(vocab),
            "vocab_sha256": sha256_file(vocab),
            "batch_size": int(cfg.batch_size),
        },
        "scoring": {
            "score_variant": cfg.score_variant,
            "color_ff": cfg.color_ff,
            "rocs_use_modifier": cfg.rocs_use_modifier,
            "rocs_clip_upper": float(cfg.rocs_clip_upper),
            "sa_score_lower": float(cfg.sa_score_lower),
            "sa_score_upper": float(cfg.sa_score_upper),
            "include_qed": bool(cfg.include_qed),
            "thresholds": thresholds,
            "threshold_percentile": cfg.threshold_percentile,
            "reward_scheme": cfg.reward_scheme,
            "weights": list(cfg.weights or ()),
            "reference_conformers": int(cfg.max_reference_conformers),
            "reference_seed": cfg.reference_random_seed,
        },
        "source_revision": revision,
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "torch": torch.__version__,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "files": {
            CSV_NAME: sha256_file(csv_path),
            REFERENCE_NAME: sha256_file(target / REFERENCE_NAME),
            STATISTICS_NAME: sha256_file(statistics_path),
        },
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (target / README_NAME).write_text(
        _readme(manifest, statistics, target), encoding="utf-8"
    )
    return target


def _reference_smiles(reference: Path, molecule: str | Path) -> str | None:
    """Canonical SMILES of the first reference conformer, or of the input."""
    if reference.is_file():
        for mol in Chem.SDMolSupplier(str(reference), removeHs=False):
            if mol is not None:
                return Chem.MolToSmiles(Chem.RemoveHs(mol))
    return canonical_smiles(str(molecule))


def _readme(
    manifest: dict[str, Any], statistics: dict[str, Any], target: Path
) -> str:
    props = statistics.get("properties", {})
    rows = "\n".join(
        f"| {key} | {values.get('p50', float('nan')):.3f} | "
        f"{values.get('p05', float('nan')):.3f} – {values.get('p95', float('nan')):.3f} |"
        for key, values in props.items()
        if values
    )
    thresholds = ", ".join(f"{v:.3f}" for v in manifest["scoring"]["thresholds"])
    return f"""# Prior dataset `{manifest['name']}`

The frozen pre-training arm: {manifest['n_scored']} molecules sampled from the
untrained SMILES-RNN prior with seed `{manifest['seed']}` and scored against the
reference ensemble in this directory.

| Field | Value |
| :--- | :--- |
| Molecule | `{manifest['molecule']}` |
| Valid / unique | {statistics['n_valid']} / {statistics['n_unique']} |
| Mean pairwise ECFP4 | {statistics.get('mean_pairwise_tanimoto') or float('nan'):.3f} |
| Suggested thresholds | {thresholds} |
| Model SHA-256 | `{manifest['sampling']['model_sha256'][:16]}…` |
| Source revision | `{manifest.get('source_revision') or 'uncommitted'}` |

| Property | Median | P05 – P95 |
| :--- | ---: | ---: |
{rows}

`prior_molecules.csv.gz` stores the **raw** measurements (`Raw_ROCS`, `Raw_SA`,
`QED`) plus descriptors. Scaled `ROCS`/`SA` are re-derived by the loading run's
`PipelineConfig`, so a run with different clip windows can reuse this file.
Rebuild with `python -m _david.code.prior_dataset build --force`; verify with
`python -m _david.code.prior_dataset verify`.
"""


# --------------------------------------------------------------------------- #
# load / verify / sync
# --------------------------------------------------------------------------- #


def load(name_or_path: str | Path, root: Path = DATASET_ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset arm and its manifest."""
    target = resolve_dataset(name_or_path, root=root)
    csv_path = target / CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"no {CSV_NAME} in {target}")
    manifest = json.loads((target / MANIFEST_NAME).read_text())
    return pd.read_csv(csv_path), manifest


def load_arm(
    name_or_path: str | Path,
    config: PipelineConfig | None = None,
    *,
    n: int | None = None,
    root: Path = DATASET_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset arm re-scaled for ``config`` and truncated to ``n`` rows."""
    frame, manifest = load(name_or_path, root=root)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"prior dataset {name_or_path!r} lacks required columns: {missing}"
        )
    frame = raw_to_scaled(frame, config)
    if n is not None:
        if n > len(frame):
            raise ValueError(
                f"prior dataset has {len(frame)} molecules but {n} were requested"
            )
        frame = frame.head(n).reset_index(drop=True)
    return frame, manifest


def verify(name_or_path: str | Path, root: Path = DATASET_ROOT) -> list[str]:
    """Recompute every recorded hash; returns a list of problems (empty is good)."""
    target = resolve_dataset(name_or_path, root=root)
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {MANIFEST_NAME}"]
    manifest = json.loads(manifest_path.read_text())
    problems = []
    for filename, expected in manifest.get("files", {}).items():
        actual = sha256_file(target / filename)
        if actual != expected:
            problems.append(f"{filename}: expected {expected[:12]}… got {actual[:12]}…")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return problems


def remote_storage(host: str = "meta") -> str:
    from _david.code.metacentrum.submit import storage_root

    return storage_root()


def sync(
    name_or_path: str | Path,
    *,
    host: str = "meta",
    root: Path = DATASET_ROOT,
    dry_run: bool = False,
) -> str:
    """Mirror one dataset to MetaCentrum storage and verify the checksums."""
    target = resolve_dataset(name_or_path, root=root)
    remote_dir = f"{remote_storage(host)}/datasets/{target.name}"
    command = ["rsync", "-a", "--checksum", "--delete", f"{target}/", f"{host}:{remote_dir}/"]
    if dry_run:
        return " ".join(command)
    subprocess.run(command, check=True)
    local = sha256_file(target / CSV_NAME)
    remote = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, f"sha256sum {remote_dir}/{CSV_NAME}"],
        text=True,
    ).split()[0]
    if remote != local:
        raise RuntimeError(
            f"checksum mismatch after sync: local {local[:12]}… remote {remote[:12]}…"
        )
    return remote_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "list", "verify", "sync"))
    parser.add_argument("--name", default=None, help="dataset name or directory")
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--host", default="meta")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args(argv)

    if args.action == "build":
        target = build(
            args.molecule,
            name=args.name,
            n=args.n,
            seed=args.seed,
            root=args.root,
            force=args.force,
        )
        print(f"built {target}")
        return 0

    if args.action == "list":
        root = args.root
        if not root.is_dir():
            print(f"no datasets under {root}")
            return 0
        for entry in sorted(root.iterdir()):
            if not (entry / MANIFEST_NAME).is_file():
                continue
            manifest = json.loads((entry / MANIFEST_NAME).read_text())
            print(
                f"{entry.name:44s} n={manifest.get('n_scored', '?'):>5} "
                f"seed={manifest.get('seed', '?'):>9} "
                f"created={manifest.get('created', '?')}"
            )
        return 0

    target = resolve_dataset(args.name or _single_dataset(args.root), root=args.root)
    if args.action == "verify":
        problems = verify(target)
        if problems:
            print(f"{target}: FAIL")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"{target}: OK")
        return 0

    print(sync(target, host=args.host, root=args.root, dry_run=args.dry_run))
    return 0


def _single_dataset(root: Path) -> str:
    found = [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / MANIFEST_NAME).is_file()
    ] if root.is_dir() else []
    if len(found) != 1:
        raise SystemExit(
            f"--name is required (found {len(found)} datasets under {root})"
        )
    return found[0]


if __name__ == "__main__":
    raise SystemExit(main())
