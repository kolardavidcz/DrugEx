"""One 3D shape-guided DrugEx RL run: ``python -m _david.code.run [--set k=v ...] [--tag name]``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import shutil
import signal
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from _david.code.config import PROJECT_ROOT, ROCS_PUBLISHED_MAX, Config, parse_dataclass_overrides
from _david.code.reconstruct.provenance import ARTIFACTS, Manifest, atomic_write_text, describe, digest
from _david.code.scorers.conformers import FORCE_FIELD, RELAXATION_TALLY, build_reference
from _david.code.scorers.objectives import (
    build_environment,
    build_scorers,
    calibrate_thresholds,
    mark_desired,
    reward_scheme,
    scorer_keys,
    set_basin_stage,
    thresholds_from_scores,
)
from _david.code.scorers.prior_cache import (
    load_prior_bundle,
    project_prior_diagnostics,
    project_prior_scores,
    save_prior_bundle,
)
from _david.code.scorers.utils import (
    check_pool,
    components,
    pool_is_healthy,
    read_sdf,
    resolve_workers,
    score_in_chunks,
    use_double_precision_pickling,
)
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.scorers.properties import FAILURE_VALUES
from drugex.training.scorers.sascorer import calculateScore

FINAL_ARMS = ("desired_ratio_at_stop",)
ARMS = ("prior", *FINAL_ARMS)
ARM_ORIGINS = ("prior", "run", "rebuilt")

logger = logging.getLogger("_david")


def config_from_args(overrides: Sequence[str]) -> Config:
    """Build a :class:`Config` from repeatable ``name=value`` CLI overrides."""
    return parse_dataclass_overrides(Config, overrides)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_directory(cfg: Config, tag: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{stamp}_{tag}" if tag else stamp
    path, suffix = Path(cfg.output_root) / name, 2
    while path.exists():
        path = Path(cfg.output_root) / f"{name}_{suffix}"
        suffix += 1
    (path / "models").mkdir(parents=True)
    return path


def load_model(voc: VocSmiles, checkpoint: Path) -> SequenceRNN:
    """Load a :class:`SequenceRNN` checkpoint onto the active device."""
    model = SequenceRNN(voc, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS)
    model.loadStatesFromFile(str(checkpoint))
    return model


def _load_models(cfg: Config) -> tuple[VocSmiles, SequenceRNN, SequenceRNN]:
    """Load vocabulary, trainable agent, and frozen prior (independent weights)."""
    checkpoint, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)
    agent = load_model(voc, checkpoint)
    prior = load_model(voc, checkpoint)
    for param in prior.parameters():
        param.requires_grad_(False)
    prior.eval()
    return voc, agent, prior


def sample(model: SequenceRNN, n: int, batch: int = 500) -> list[str]:
    """Sample exactly ``n`` SMILES from ``model`` in batches of ``batch``."""
    drawn: list[str] = []
    while len(drawn) < n:
        drawn.extend(model.sample(min(batch, n - len(drawn))))
    return drawn[:n]


def _canonical_and_sa(smiles: list[str], valid: pd.Series) -> tuple[list[str], list[float]]:
    canonical, raw_sa = [], []
    for smi, is_valid in zip(smiles, valid):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else (smi or ""))
        if mol is None or is_valid != 1:
            raw_sa.append(0.0)
            continue
        try:
            raw_sa.append(round(float(calculateScore(mol)), 4))
        except Exception:
            raw_sa.append(FAILURE_VALUES["SA"])
    return canonical, raw_sa


def write_arm(
    path: Path,
    scores: pd.DataFrame,
    smiles: list[str],
    diagnostics: dict[str, list[float]] | None = None,
    *,
    origin: str = "run",
    seed: int | None = None,
) -> dict[str, Any]:
    """Write one scored molecule arm TSV and return summary ratios for ``manifest.json``."""
    if origin not in ARM_ORIGINS:
        raise ValueError(f"origin must be one of {', '.join(ARM_ORIGINS)}, not {origin!r}")
    table = scores.copy()
    table.insert(0, "SMILES", smiles)
    for col, values in (diagnostics or {}).items():
        table[col] = values
    invalid = table["Valid"] == 0
    for col in ("Desired", *(diagnostics or {})):
        if col in table:
            table.loc[invalid, col] = 0
    table["Canonical"], table["SA_raw"] = _canonical_and_sa(smiles, table["Valid"])
    table.to_csv(path, sep="\t", index=False)

    n = len(table)
    denom = n or 1
    valid = table[~invalid]
    unique = valid.drop_duplicates(subset="Canonical")
    ratios: dict[str, Any] = {
        "n": n,
        "valid_ratio": len(valid) / denom,
        "unique_ratio": len(unique) / denom,
        "desired_ratio": float(unique["Desired"].sum()) / denom,
        "origin": origin,
        "seed": seed,
        "sampled": origin != "prior",
    }
    logger.info(
        "%s: n=%d valid=%.4f unique=%.4f desired=%.4f",
        path.name,
        n,
        ratios["valid_ratio"],
        ratios["unique_ratio"],
        ratios["desired_ratio"],
    )
    return ratios


def arm_is_complete(path: Path, n: int) -> bool:
    """Whether ``path`` holds ``n`` non-truncated arm rows."""
    try:
        table = pd.read_csv(path, sep="\t")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    required = {"SMILES", "Canonical", "Valid", "Desired", "SA_raw"}
    return (
        len(table) == n
        and required.issubset(table.columns)
        and not table[["Valid", "Desired", "SA_raw"]].isna().any().any()
    )


def recorded_arm_size(run_dir: Path, name: str) -> int | None:
    """Return the ``n`` recorded for ``name`` in ``manifest.json``, if present."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    arm = (manifest.get("arms") or {}).get(name)
    size = arm.get("n") if isinstance(arm, dict) else None
    return size if isinstance(size, int) and size > 0 else None


def incomplete_arms(run_dir: Path, n: int) -> list[str]:
    """Final arm names missing or shorter than their expected row count."""
    return [
        name
        for name in FINAL_ARMS
        if not arm_is_complete(run_dir / f"final_{name}.tsv", recorded_arm_size(run_dir, name) or n)
    ]


def serial_run_refusal(cfg: Config) -> str | None:
    """Explain why multi-core execution must refuse to run serially when the pool probe failed."""
    count = resolve_workers(cfg.n_jobs)
    if pool_is_healthy() or count <= 1:
        return None
    return (
        f"{count} cores requested but worker pool failed to start. Run as `python -m _david.code.run`, "
        "set DRUGEX_MP_CONTEXT=fork, or pass --set n_jobs=1."
    )


def _save_model(state: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _training_summary(fit_path: Path, cfg: Config) -> dict[str, Any]:
    if not fit_path.is_file():
        return {}
    fit = pd.read_csv(fit_path, sep="\t")
    if fit.empty or "desired_ratio" not in fit.columns or fit["desired_ratio"].isna().all():
        return {}
    epochs_run = int(fit["Epoch"].iloc[-1])
    best_idx = int(fit["desired_ratio"].idxmax())
    best_desired = float(fit.loc[best_idx, "desired_ratio"])
    last_save = int(fit["best_epoch"].iloc[-1]) if "best_epoch" in fit.columns else int(fit.loc[best_idx, "Epoch"])
    best_epoch = last_save if last_save > 0 else int(fit.loc[best_idx, "Epoch"])
    frozen = bool(epochs_run < cfg.epochs and epochs_run >= cfg.min_epochs and (epochs_run - last_save) > cfg.patience)
    return {
        "epochs_run": epochs_run,
        "best_desired_ratio": best_desired,
        "best_desired_ratio_epoch": best_epoch,
        "desired_ratio_frozen": frozen,
        "desired_ratio_freeze_epoch": epochs_run,
        "desired_ratio_at_stop": float(fit["desired_ratio"].iloc[-1]),
    }


def _prepare_environment(
    cfg: Config, run_dir: Path, prior_smiles: list[str], scheme: Any
) -> tuple[list[Chem.Mol], Any, list[float], pd.DataFrame, dict[str, list[float]], str]:
    """Build or load the reference bank, calibrate prior thresholds, and assemble the RL env."""
    cached = (
        load_prior_bundle(cfg, prior_smiles)
        if cfg.use_prior_cache and not cfg.track_conformer_basins
        else None
    )
    if cached is not None:
        shutil.copy2(cached.reference_sdf, run_dir / "reference.sdf")
        references = read_sdf(run_dir / "reference.sdf")
        scorers = build_scorers(cfg, references)
        prior_scores = project_prior_scores(cached.scores, cfg, scorers)
        prior_diag = project_prior_diagnostics(cached.scores)
        thresholds = thresholds_from_scores(cfg, scorers, prior_scores, scorer_keys(scorers))
        env = build_environment(scorers, thresholds, scheme)
        logger.info("prior cache hit: %s (%d molecules)", cached.key, len(prior_smiles))
        return references, env, thresholds, prior_scores, prior_diag, f"hit:{cached.key}"

    references = build_reference(cfg, run_dir / "reference.sdf")
    scorers = (
        build_scorers(cfg, references, basin_log_dir=run_dir / "conformer_basins", basin_stage="prior")
        if cfg.track_conformer_basins
        else build_scorers(cfg, references)
    )
    thresholds, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
    env = build_environment(scorers, thresholds, scheme)
    prior_diag = components(env, prior_smiles)
    saved = (
        save_prior_bundle(
            cfg, run_dir / "reference.sdf", prior_smiles, prior_scores, prior_diag, origin=f"run:{run_dir.name}"
        )
        if cfg.use_prior_cache
        else None
    )
    status = f"miss:{saved.key}" if saved is not None else "bypassed"
    return references, env, thresholds, prior_scores, prior_diag, status


def _train_policy(
    cfg: Config, run_dir: Path, agent: SequenceRNN, prior: SequenceRNN, env: Any
) -> dict[str, Any]:
    """Run policy-gradient training and persist the final checkpoint even under SIGTERM."""
    set_basin_stage(env, "train")
    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=prior,
        batch_size=cfg.batch_size,
        epsilon=cfg.epsilon,
        beta=cfg.beta,
        n_samples=cfg.n_samples,
        device=DEFAULT_DEVICE,
        use_gpus=DEFAULT_GPUS,
    )
    model_path = run_dir / "models" / "desired_ratio_at_stop.pkg"
    on_main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if on_main else None

    def _on_sigterm(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    if on_main:
        signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        explorer.fit(
            monitor=FileMonitor(str(run_dir / "rl"), save_smiles=True),
            epochs=cfg.epochs,
            patience=cfg.patience,
            reload_interval=cfg.epochs + 1,
            criteria="desired_ratio",
            min_epochs=cfg.min_epochs,
        )
    finally:
        if on_main:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if not model_path.exists():
            try:
                _save_model(explorer.getModel(), model_path)
            except Exception:
                logger.exception("failed to save %s", model_path.name)
        if on_main and previous is not None:
            signal.signal(signal.SIGTERM, previous)

    summary = _training_summary(run_dir / "rl_fit.tsv", cfg)
    logger.info("training summary: %s", summary)
    return summary


def _evaluate_final_arms(
    cfg: Config, run_dir: Path, voc: VocSmiles, env: Any
) -> dict[str, dict[str, Any]]:
    """Sample and chunk-score every final checkpoint arm."""
    set_basin_stage(env, "final")
    arms: dict[str, dict[str, Any]] = {}
    for name in FINAL_ARMS:
        try:
            model = load_model(voc, run_dir / "models" / f"{name}.pkg")
            smiles = sample(model, cfg.n_final)
            scores, diag = score_in_chunks(env, smiles)
            arms[name] = write_arm(
                run_dir / f"final_{name}.tsv", scores, smiles, diag, origin="run", seed=cfg.seed
            )
        except Exception:
            logger.exception(
                "arm %s failed; rebuild later with: python -m _david.code.reconstruct.score_arms %s",
                name,
                run_dir,
            )
    return arms


def _pipeline(
    cfg: Config, run_dir: Path, resolved: dict[str, Any], manifest: Manifest, started: float
) -> int:
    scheme = reward_scheme(cfg.scheme)
    voc, agent, prior = _load_models(cfg)

    logger.info("sampling %d molecules from prior to calibrate thresholds", cfg.n_prior)
    prior_smiles = sample(prior, cfg.n_prior)
    references, env, thresholds, prior_scores, prior_diag, cache_status = _prepare_environment(
        cfg, run_dir, prior_smiles, scheme
    )
    objectives = env.getScorerKeys()

    resolved.update(
        force_field=FORCE_FIELD if cfg.minimize else None,
        reward_scheme=type(scheme).__name__,
        rocs_reward_scale=ROCS_PUBLISHED_MAX,
        objectives=objectives,
        thresholds=thresholds,
    )
    atomic_write_text(run_dir / "config.json", json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        config=resolved,
        objectives=objectives,
        thresholds=dict(zip(objectives, thresholds)),
        threshold_source="fixed" if cfg.fixed_thresholds else f"p{cfg.threshold_percentile:g}",
        prior_cache=cache_status,
        reference_conformers=len(references),
        reference_sha256=digest(run_dir / "reference.sdf"),
        reference_energies=[
            round(float(m.GetDoubleProp("Energy")), 3) for m in references if m.HasProp("Energy")
        ],
    )

    arms: dict[str, dict[str, Any]] = {
        "prior": write_arm(
            run_dir / "prior.tsv",
            mark_desired(prior_scores, objectives, thresholds),
            prior_smiles,
            prior_diag,
            origin="prior",
            seed=cfg.seed,
        )
    }

    summary = _train_policy(cfg, run_dir, agent, prior, env)
    arms.update(_evaluate_final_arms(cfg, run_dir, voc, env))

    resolved.update(summary=summary, arms=arms)
    atomic_write_text(run_dir / "config.json", json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        summary=summary,
        arms=arms,
        relaxation=dict(sorted(RELAXATION_TALLY.items())),
        checkpoints={
            name: {"file": f"models/{name}.pkg", "sha256": digest(run_dir / "models" / f"{name}.pkg")}
            for name in FINAL_ARMS
        },
        artifacts={name: desc for name, desc in ARTIFACTS.items() if (run_dir / name).exists()},
    )

    try:
        from _david.code.analysis.report import analyse

        analyse(run_dir)
    except Exception:
        logger.exception("analysis failed; rerun: python -m _david.code.analysis %s", run_dir)

    incomplete = incomplete_arms(run_dir, cfg.n_final)
    if incomplete:
        logger.error("incomplete final arms: %s", ", ".join(incomplete))
    manifest.update(
        state="failed" if incomplete else "complete",
        finished=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=round(time.monotonic() - started, 1),
        error=f"incomplete final arms: {', '.join(incomplete)}" if incomplete else None,
    )
    logger.info("done: %s", run_dir)
    return 1 if incomplete else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
        help="override one Config field (repeatable)",
    )
    parser.add_argument("--tag", help="suffix for the run directory name")
    args = parser.parse_args(argv)

    cfg = config_from_args(args.overrides)
    run_dir = _run_directory(cfg, args.tag)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")],
        force=True,
    )
    seed_everything(cfg.seed)
    use_double_precision_pickling()

    resolved: dict[str, Any] = {
        k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()
    }
    logger.info("run directory %s (device=%s, gpus=%s)", run_dir, DEFAULT_DEVICE, DEFAULT_GPUS)

    manifest = Manifest(
        run_dir / "manifest.json",
        run=run_dir.name,
        query=cfg.reference_smiles,
        started=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        device=str(DEFAULT_DEVICE),
        gpus=list(DEFAULT_GPUS) if DEFAULT_GPUS else [],
        provenance=describe(PROJECT_ROOT),
        mp_context=check_pool(),
        config=resolved,
    )
    source = manifest.data["provenance"]["source"]  # type: ignore[index]
    logger.info("source %s (dirty=%s)", source["revision"], source["dirty"])

    if refusal := serial_run_refusal(cfg):
        logger.error("%s", refusal)
        manifest.update(state="failed", error=refusal)
        return 2

    started = time.monotonic()
    try:
        return _pipeline(cfg, run_dir, resolved, manifest, started)
    except BaseException as error:
        manifest.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
