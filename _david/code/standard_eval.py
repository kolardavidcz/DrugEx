"""Re-evaluate every run's final policy on the standard prior baseline.

Runs differ in evaluation size (1 000 vs 5 000), scorer window and reward
weights, so their figures are not directly comparable. This module adds one
*comparable* arm beside the historical one:

* If the run saved a checkpoint (``models/SequenceRNN_reinforced*.pkg``), the
  final policy is reloaded, sampled at the standard size with a fixed seed and
  scored against the standard dataset's reference ensemble.
* If only the delivered arm exists, its molecules are re-scored with the same
  standard environment and the manifest records ``resampled: false``.
* If neither exists, the run is ``skip`` with a reason.

The result lands in ``<run>/standard_eval/`` and never touches
``raw_data/reinforced_raw.csv.gz``: adding a measurement must not rewrite the
history that produced the thesis figures.

CLI::

    python -m _david.code.standard_eval plan
    python -m _david.code.standard_eval run --run <name> [--n 5000] [--force]
    python -m _david.code.standard_eval run --all [--n 5000]
    python -m _david.code.standard_eval verify
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _david.code.config import OUTPUT_ROOT, PipelineConfig
from _david.code.dashboard import discover_runs
from _david.code.prior_dataset import (
    DATASET_ROOT,
    RAW_ROCS,
    RAW_SA,
    add_descriptors,
    compute_statistics,
    load,
    raw_to_scaled,
    resolve_dataset,
    sha256_file,
)
from _david.code.schema import std

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
STANDARD_DIR: str = "standard_eval"
CSV_NAME: str = "reinforced_standard.csv.gz"
MANIFEST_NAME: str = "manifest.json"
STATISTICS_NAME: str = "statistics.json"
DEFAULT_SEED: int = 42
RESAMPLE_ACTIONS: tuple[str, ...] = ("resample", "rescore", "skip", "done")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint(run: Path) -> Path | None:
    """Best available final policy, preferring the selected best checkpoint."""
    models = run / "models"
    if not models.is_dir():
        return None
    preferred = models / "SequenceRNN_reinforced.pkg"
    if preferred.is_file():
        return preferred
    candidates = sorted(models.glob("SequenceRNN_reinforced*.pkg"))
    return candidates[0] if candidates else None


@dataclass
class EvalPlan:
    """What can be done for one run, decided from artifacts alone."""

    run: Path
    action: str
    reason: str
    checkpoint: Path | None = None
    arm: Path | None = None

    @property
    def name(self) -> str:
        return self.run.name

    def as_row(self) -> str:
        return f"| `{self.name}` | {self.action} | {self.reason} |"


def plan_run(run: Path | str, n: int | None = None) -> EvalPlan:
    """Classify one run: already done, resample, rescore or skip."""
    run = Path(run)
    target = run / STANDARD_DIR / CSV_NAME
    if target.is_file():
        return EvalPlan(run=run, action="done", reason="standard arm already exists")
    checkpoint = _checkpoint(run)
    if checkpoint is not None:
        return EvalPlan(
            run=run,
            action="resample",
            reason=f"checkpoint {checkpoint.name}",
            checkpoint=checkpoint,
        )
    arm = run / "raw_data" / "reinforced_raw.csv.gz"
    if arm.is_file():
        rows = _arm_rows(arm)
        return EvalPlan(
            run=run,
            action="rescore",
            reason=f"stored arm only ({rows} rows)",
            arm=arm,
        )
    return EvalPlan(
        run=run, action="skip", reason="no checkpoint and no reinforced arm"
    )


def _arm_rows(arm: Path) -> int:
    try:
        return int(len(pd.read_csv(arm)))
    except (OSError, ValueError):
        return 0


def plan_all(
    root: Path | str = OUTPUT_ROOT, n: int | None = None
) -> list[EvalPlan]:
    return [plan_run(run, n) for run in discover_runs(root)]


def _load_dataset(dataset: str | Path | None, root: Path) -> tuple[Path, dict]:
    if dataset is None:
        candidates = sorted(
            entry
            for entry in root.iterdir()
            if entry.is_dir() and (entry / MANIFEST_NAME).is_file()
        ) if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"no prior dataset under {root}")
        dataset = candidates[-1]
    target = resolve_dataset(dataset, root=root)
    _frame, manifest = load(target, root=root)
    return target, manifest


def _standard_config(manifest: dict) -> PipelineConfig:
    """One config for every run: raw ROCS, QED always measurable, fixed SA window."""
    scoring = manifest.get("scoring", {})
    return PipelineConfig(
        include_qed=True,
        rocs_use_modifier=False,
        sa_score_lower=float(scoring.get("sa_score_lower", 5.0)),
        sa_score_upper=float(scoring.get("sa_score_upper", 3.0)),
        gpu="auto",
    )


def _score(frame: pd.DataFrame, reference: Path, config: PipelineConfig) -> pd.DataFrame:
    from _david.code.environment import create_environment

    env = create_environment(reference, config=config)
    scores = env.getScores(frame[std.SMILES].astype(str).tolist())
    out = frame.reset_index(drop=True).copy()
    for column in scores.columns:
        if column == std.QED and column in out.columns:
            continue
        out[column] = scores[column].values
    return out


def _resample(policy: Path, n: int, config: PipelineConfig, seed: int) -> pd.DataFrame:
    from _david.code.models import SequenceRNNOptimizer

    _seed_everything(seed)
    optimizer = SequenceRNNOptimizer(models_dir=policy.parent, config=config)
    agent, _prior, _voc = optimizer.load_models()
    agent.loadStatesFromFile(policy)
    agent.eval()
    generated = agent.generate(
        num_samples=n,
        batch_size=config.batch_size,
        drop_invalid=False,
        drop_duplicates=False,
        evaluator=None,
        progress=False,
    )
    return pd.DataFrame({std.SMILES: generated["SMILES"].tolist()})


def evaluate_run(
    run: Path | str,
    dataset: str | Path | None = None,
    *,
    n: int = 5000,
    seed: int = DEFAULT_SEED,
    dataset_root: Path = DATASET_ROOT,
    force: bool = False,
) -> Path | None:
    """Write ``<run>/standard_eval`` for one run; ``None`` when it cannot be done."""
    run = Path(run)
    plan = plan_run(run)
    if plan.action in {"done", "skip"}:
        if plan.action == "done" and not force:
            LOGGER.info("%s: %s", run.name, plan.reason)
            return run / STANDARD_DIR
        if plan.action == "skip":
            LOGGER.warning("%s: %s", run.name, plan.reason)
            return None

    if plan.action == "done" and force:
        checkpoint = _checkpoint(run)
        arm = run / "raw_data" / "reinforced_raw.csv.gz"
        plan = EvalPlan(
            run=run,
            action="resample" if checkpoint else ("rescore" if arm.is_file() else "skip"),
            reason="forced rebuild",
            checkpoint=checkpoint,
            arm=arm if arm.is_file() else None,
        )
        if plan.action == "skip":
            return None

    dataset_dir, dataset_manifest = _load_dataset(dataset, dataset_root)
    config = _standard_config(dataset_manifest)
    reference = dataset_dir / "reference_3d.sdf"
    started = time.time()

    if plan.action == "resample":
        frame = _resample(plan.checkpoint, n, config, seed)
        resampled = True
        used_n = int(len(frame))
    else:
        stored = pd.read_csv(plan.arm)
        if std.SMILES not in stored.columns:
            LOGGER.warning("%s: stored arm has no SMILES", run.name)
            return None
        frame = pd.DataFrame({std.SMILES: stored[std.SMILES].astype(str)})
        resampled = False
        used_n = int(len(frame))

    frame = _score(frame, reference, config)
    frame = add_descriptors(frame)
    frame[RAW_ROCS] = pd.to_numeric(frame[std.ROCS], errors="coerce")
    frame = raw_to_scaled(frame, config)
    frame = _add_standard_reward(frame, dataset_manifest, config)

    target = run / STANDARD_DIR
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / CSV_NAME
    frame.to_csv(csv_path, index=False, compression="gzip")

    statistics = compute_statistics(
        frame,
        config,
        thresholds=dataset_manifest.get("scoring", {}).get("thresholds"),
    )
    statistics_path = target / STATISTICS_NAME
    statistics_path.write_text(json.dumps(statistics, indent=2) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": run.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": plan.action,
        "resampled": resampled,
        "n_requested": int(n),
        "n_used": used_n,
        "seed": int(seed) if resampled else None,
        "checkpoint": plan.checkpoint.name if plan.checkpoint else None,
        "checkpoint_sha256": sha256_file(plan.checkpoint)
        if plan.checkpoint
        else None,
        "stored_arm_sha256": sha256_file(plan.arm) if plan.arm else None,
        "dataset": {
            "name": dataset_dir.name,
            "manifest_sha256": sha256_file(dataset_dir / MANIFEST_NAME),
            "csv_sha256": sha256_file(dataset_dir / "prior_molecules.csv.gz"),
            "reference_sha256": sha256_file(reference),
        },
        "config": {
            "include_qed": config.include_qed,
            "rocs_use_modifier": config.rocs_use_modifier,
            "sa_score_lower": config.sa_score_lower,
            "sa_score_upper": config.sa_score_upper,
        },
        "files": {
            CSV_NAME: sha256_file(csv_path),
            STATISTICS_NAME: sha256_file(statistics_path),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info(
        "%s: %s arm (%d molecules) in %.1f min -> %s",
        run.name,
        plan.action,
        used_n,
        (time.time() - started) / 60,
        target,
    )
    return target


def _add_standard_reward(
    frame: pd.DataFrame, dataset_manifest: dict, config: PipelineConfig
) -> pd.DataFrame:
    """Score with one fixed reward definition so arms are comparable columns."""
    from _david.code.environment import resolve_reward_scheme

    scoring = dataset_manifest.get("scoring", {})
    thresholds = scoring.get("thresholds") or [0.35, 0.50, 0.50]
    weights = scoring.get("weights") or [0.50, 0.25, 0.25]
    scheme = resolve_reward_scheme(
        scoring.get("reward_scheme") or std.REWARD_HARRINGTON,
        weights=tuple(weights),
        rocs_raw=True,
    )
    columns = [std.ROCS, std.SA, std.QED]
    matrix = frame[columns].to_numpy(dtype=float)
    rewards = np.asarray(
        scheme(frame[std.SMILES].astype(str).tolist(), matrix, thresholds)
    ).reshape(-1)
    frame["StandardReward"] = rewards
    return frame


# --------------------------------------------------------------------------- #
# MetaCentrum job
# --------------------------------------------------------------------------- #

JOB_NAME: str = "drugex_std_eval"
JOB_SCRIPT_NAME: str = "run_standard_eval.pbs"


def pbs_script(
    storage: str,
    release: str,
    *,
    dataset: str,
    n: int = 5000,
    seed: int = DEFAULT_SEED,
    max_runs: int = 6,
    walltime: str = "24:00:00",
) -> str:
    """PBS job that re-evaluates runs in place on storage (no scratch copy-back)."""
    return f"""#!/usr/bin/env bash
#PBS -N {JOB_NAME}
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=5gb:spec=8.0
#PBS -l walltime={walltime}
#PBS -j oe

set -euo pipefail
STORAGE_DIR={storage!r}
RELEASE_DIR={release!r}
cd "$RELEASE_DIR"
export PYTHONPATH="$RELEASE_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DRUGEX_SCORE_NJOBS=16
exec "$STORAGE_DIR/.venv/bin/python" -u -m _david.code.standard_eval run \
    --all --root "$STORAGE_DIR/_david/output" \
    --dataset {dataset!r} --dataset-root {storage + "/datasets"!r} \
    --n {int(n)} --seed {int(seed)} --max-runs {int(max_runs)}
"""


def submit_job(
    dataset: str | Path,
    *,
    host: str = "meta",
    n: int = 5000,
    seed: int = DEFAULT_SEED,
    max_runs: int = 6,
    walltime: str = "24:00:00",
    dry_run: bool = False,
) -> str:
    """Stage HEAD as a release and qsub the standard-evaluation job."""
    from _david.code.metacentrum.submit import remote, storage_root, sync_release

    storage = storage_root()
    dataset_path = resolve_dataset(dataset)
    script = pbs_script(
        storage,
        f"{storage}/current",
        dataset=f"{storage}/datasets/{dataset_path.name}",
        n=n,
        seed=seed,
        max_runs=max_runs,
        walltime=walltime,
    )
    target = f"{storage}/jobs/standard_eval/{JOB_SCRIPT_NAME}"
    if dry_run:
        return script
    revision = sync_release(host=host, storage=storage)
    remote(
        host,
        f"mkdir -p {storage}/jobs/standard_eval && cat > {target}",
        stdin=script,
    )
    result = remote(host, f"cd {storage}/jobs/standard_eval && qsub {target}")
    return f"revision {revision}: {result.stdout.strip()}"


def pull_standard_arms(
    root: Path | str = OUTPUT_ROOT, *, host: str = "meta"
) -> list[str]:
    """Copy each remote ``standard_eval/`` directory into the local run tree."""
    from _david.code.metacentrum.submit import remote, storage_root

    storage = storage_root()
    root = Path(root)
    listing = remote(
        host,
        f"find {storage}/_david/output -maxdepth 2 -type d -name {STANDARD_DIR}",
    ).stdout.splitlines()
    copied = []
    import subprocess

    for remote_dir in (line.strip() for line in listing if line.strip()):
        relative = Path(remote_dir).relative_to(f"{storage}/_david/output")
        local = root / relative
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", f"{host}:{remote_dir}/", f"{local}/"], check=True
        )
        copied.append(local.as_posix())
    return copied


def verify(root: Path | str = OUTPUT_ROOT) -> list[str]:
    """Check every standard arm's recorded hashes."""
    problems = []
    for run in discover_runs(root):
        target = run / STANDARD_DIR
        manifest_path = target / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for filename, expected in manifest.get("files", {}).items():
            actual = sha256_file(target / filename)
            if actual != expected:
                problems.append(
                    f"{run.name}/{filename}: expected {expected[:12]}… got {actual[:12]}…"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify", "submit", "pull"))
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run", action="append", default=None, dest="runs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--host", default="meta")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "plan":
        plans = plan_all(args.root, args.n)
        print("| run | action | reason |")
        print("| :--- | :--- | :--- |")
        for plan in plans:
            print(plan.as_row())
        return 0

    if args.action == "submit":
        if not args.dataset:
            parser.error("submit needs --dataset <name>")
        print(
            submit_job(
                args.dataset,
                host=args.host,
                n=args.n,
                seed=args.seed,
                max_runs=args.max_runs,
                walltime=args.walltime,
                dry_run=args.dry_run,
            )
        )
        return 0

    if args.action == "pull":
        for target in pull_standard_arms(args.root, host=args.host):
            print(f"pulled {target}")
        return 0

    if args.action == "verify":
        problems = verify(args.root)
        for problem in problems:
            print(f"- {problem}")
        print("OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    root = Path(args.root)
    selected = (
        [root / name for name in args.runs]
        if args.runs
        else discover_runs(root) if args.all else []
    )
    if not selected:
        parser.error("choose --run <name> (repeatable) or --all")
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    for run in selected:
        evaluate_run(
            run,
            args.dataset,
            n=args.n,
            seed=args.seed,
            dataset_root=args.dataset_root,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
