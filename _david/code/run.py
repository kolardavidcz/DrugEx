"""One reinforcement-learning run: python -m _david.code.run [--set name=value ...] [--tag name]"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from _david.code.config import PROJECT_ROOT, ROCS_PUBLISHED_MAX, Config
from _david.code.reconstruct.provenance import ARTIFACTS, Manifest, describe, digest
from _david.code.scorers.conformers import (
    FORCE_FIELD,
    RELAXATION_TALLY,
    build_reference,
)
from _david.code.scorers.objectives import (
    build_environment,
    build_scorers,
    calibrate_thresholds,
    mark_desired,
    reward_scheme,
    set_basin_stage,
    thresholds_from_scores,
)
from _david.code.scorers.prior_cache import (
    load_prior_bundle,
    prior_cache_key,
    project_prior_diagnostics,
    project_prior_scores,
    save_prior_bundle,
)
from _david.code.scorers.utils import (
    check_pool,
    components,
    pool_is_healthy,
    read_sdf,
    score_in_chunks,
    use_double_precision_pickling,
)
from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.scorers.parallel import capped_n_jobs

FINAL_ARMS = ("desired_ratio_at_stop",)
ARMS = ("prior", *FINAL_ARMS)

logger = logging.getLogger("_david")


def _coerce(field: dataclasses.Field, raw: str):
    annotation = str(field.type)
    if "None" in annotation and raw.lower() in ("none", "null", ""):
        return None
    if "bool" in annotation:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"{field.name} is a boolean; got {raw!r}")
        return raw.lower() in ("true", "1")
    if "tuple" in annotation:
        inner = float if "float" in annotation else int if "int" in annotation else str
        text = raw.strip().strip("()[]")
        return tuple(inner(part) for part in text.split(",")) if text else ()
    for name, cast in (("int", int), ("float", float), ("Path", Path)):
        if name in annotation:
            return cast(raw)
    return raw


def config_from_args(overrides: list[str]) -> Config:
    fields = {field.name: field for field in dataclasses.fields(Config)}
    values = {}
    for item in overrides:
        name, separator, raw = item.partition("=")
        name = name.strip()
        if not separator:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        if name not in fields:
            raise SystemExit(
                f"unknown setting {name!r}. Valid settings: {', '.join(sorted(fields))}"
            )
        values[name] = _coerce(fields[name], raw.strip())
    return Config(**values)


def seed_everything(seed: int) -> None:
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
    model = SequenceRNN(voc, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS)
    model.loadStatesFromFile(str(checkpoint))
    return model


def sample(model: SequenceRNN, n: int, batch: int = 500) -> list[str]:
    drawn: list[str] = []
    while len(drawn) < n:
        drawn.extend(model.sample(min(batch, n - len(drawn))))
    return drawn[:n]


def _canonical_and_sa(smiles: list[str], valid: pd.Series) -> tuple[list[str], list[float]]:
    from drugex.training.scorers.properties import FAILURE_VALUES
    from drugex.training.scorers.sascorer import calculateScore

    canonical, raw_sa = [], []
    for value, is_valid in zip(smiles, valid):
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else (value or ""))
        if mol is None or is_valid != 1:
            raw_sa.append(0.0)
            continue
        try:
            raw_sa.append(round(float(calculateScore(mol)), 4))
        except Exception:
            raw_sa.append(FAILURE_VALUES["SA"])
    return canonical, raw_sa


#: Where an arm's molecules came from. ``prior`` is the unmodified prior draw, ``run`` an
#: arm written by the training job itself, ``rebuilt`` an arm re-sampled from a checkpoint
#: after a kill. A rebuilt arm is the same policy on a different draw, so a reader must be
#: able to tell the two apart without trusting prose.
ARM_ORIGINS = ("prior", "run", "rebuilt")


def write_arm(
    path: Path,
    scores: pd.DataFrame,
    smiles: list[str],
    diagnostics: dict | None = None,
    *,
    origin: str = "run",
    seed: int | None = None,
) -> dict:
    """Write one arm and return the ratios recorded for it in the manifest.

    Besides the ratios, an arm records ``origin`` (see ``ARM_ORIGINS``), the ``seed`` its
    draw used, and ``sampled`` -- whether the molecules were drawn from a checkpoint now.
    Those three are what distinguish an in-run arm from a rebuild of the same policy.
    """
    if origin not in ARM_ORIGINS:
        raise ValueError(f"origin is one of {', '.join(ARM_ORIGINS)}, not {origin!r}")
    scores = scores.copy()
    scores.insert(0, "SMILES", smiles)
    for column, values in (diagnostics or {}).items():
        scores[column] = values
    invalid = scores["Valid"] == 0
    # An invalid molecule has every objective zeroed, so it clears an all-zero threshold set.
    for column in ("Desired", *(diagnostics or {})):
        if column in scores:
            scores.loc[invalid, column] = 0
    scores["Canonical"], scores["SA_raw"] = _canonical_and_sa(smiles, scores["Valid"])
    scores.to_csv(path, sep="\t", index=False)

    valid = scores[~invalid]
    unique = valid.drop_duplicates(subset="Canonical")
    ratios = {
        "n": len(scores),
        "valid_ratio": len(valid) / len(scores),
        "unique_ratio": len(unique) / len(scores),
        "desired_ratio": float(unique["Desired"].sum()) / len(scores),
        "origin": origin,
        "seed": seed,
        "sampled": origin != "prior",
    }
    # Only the numeric part: a manifest entry now also carries strings and a bool.
    numeric = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in ratios.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    logger.info("%s: %s", path.name, numeric)
    return ratios


def arm_is_complete(path: Path, n: int) -> bool:
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
    """The ``n`` the run recorded for this arm, if the manifest carries one.

    ``arm_is_complete`` compares a table against an expected size. Reading that size from
    today's ``config.json`` would make every existing arm "incomplete" the day ``n_final``
    changes, and a rebuild would then replace a finished arm with a different draw. A
    recorded per-arm ``n`` pins the arm to the size it was written at.
    """
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    arm = (manifest.get("arms") or {}).get(name)
    if not isinstance(arm, dict):
        return None
    size = arm.get("n")
    return size if isinstance(size, int) and size > 0 else None


def incomplete_arms(run_dir: Path, n: int) -> list[str]:
    """Arms whose table is missing, truncated or the wrong size.

    ``n`` is the fallback: an arm written before sizes were recorded has no entry to read,
    so it is judged against the size the caller expects.
    """
    return [
        name
        for name in FINAL_ARMS
        if not arm_is_complete(
            run_dir / f"final_{name}.tsv", recorded_arm_size(run_dir, name) or n
        )
    ]


def serial_run_refusal(cfg: Config) -> str | None:
    workers = capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1))
    if pool_is_healthy() or workers <= 1:
        return None
    return (
        f"{workers} cores were requested but no worker can start, and scoring one core at "
        "a time does not finish inside any sensible walltime. Start the pipeline as "
        "`python -m _david.code.run`, or set DRUGEX_MP_CONTEXT=fork, or --set n_jobs=1 to "
        "accept a serial run."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one Config field; repeatable",
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

    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(cfg).items()
    }
    logger.info("run directory %s", run_dir)
    logger.info("device %s, gpus %s", DEFAULT_DEVICE, DEFAULT_GPUS)

    # Before the first expensive stage, so a run killed at walltime reads as `running`.
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
    source = manifest.data["provenance"]["source"]
    logger.info("source %s (dirty=%s)", source["revision"], source["dirty"])

    refusal = serial_run_refusal(cfg)
    if refusal:
        logger.error("%s", refusal)
        manifest.update(state="failed", error=refusal)
        return 2

    started = time.monotonic()
    try:
        return _pipeline(cfg, run_dir, resolved, manifest, started)
    except BaseException as error:
        manifest.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise


def _save_model(state, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _training_summary(fit_path: Path, cfg: Config) -> dict:
    if not fit_path.is_file():
        return {}
    fit = pd.read_csv(fit_path, sep="\t")
    if fit.empty or "desired_ratio" not in fit.columns:
        return {}
    epochs_run = int(fit["Epoch"].iloc[-1])
    best_idx = int(fit["desired_ratio"].idxmax())
    best_desired = float(fit.loc[best_idx, "desired_ratio"])
    last_save = int(fit["best_epoch"].iloc[-1]) if "best_epoch" in fit.columns else int(fit.loc[best_idx, "Epoch"])
    best_epoch = last_save if last_save > 0 else int(fit.loc[best_idx, "Epoch"])
    at_stop = float(fit["desired_ratio"].iloc[-1])
    frozen = bool(
        epochs_run < cfg.epochs
        and epochs_run >= cfg.min_epochs
        and (epochs_run - last_save) > cfg.patience
    )
    return {
        "epochs_run": epochs_run,
        "best_desired_ratio": best_desired,
        "best_desired_ratio_epoch": best_epoch,
        "peak_desired_ratio_any_epoch": best_desired,
        "desired_ratio_frozen": frozen,
        "desired_ratio_freeze_epoch": epochs_run,
        "desired_ratio_at_stop": at_stop,
        "checkpoint_population": {
            "desired_ratio_at_stop": {
                "valid_ratio": round(float(fit["valid_ratio"].iloc[-1]), 4)
                if "valid_ratio" in fit.columns
                else 0.0,
                "unique_ratio": round(float(fit["unique_ratio"].iloc[-1]), 4)
                if "unique_ratio" in fit.columns
                else 0.0,
            }
        },
    }


def _pipeline(
    cfg: Config, run_dir: Path, resolved: dict, manifest: Manifest, started: float
) -> int:
    # Before anything expensive: an unknown scheme name is a typo in a submitted delta.
    scheme = reward_scheme(cfg.scheme)

    checkpoint, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)
    agent = load_model(voc, checkpoint)
    # An independent copy: the agent trains, the prior stays frozen for epsilon-mutation.
    prior = load_model(voc, checkpoint)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    prior.eval()

    logger.info("sampling %d molecules from the prior to calibrate thresholds", cfg.n_prior)
    prior_smiles = sample(prior, cfg.n_prior)

    use_cache = cfg.use_prior_cache and not cfg.track_conformer_basins
    cached = load_prior_bundle(cfg, prior_smiles) if use_cache else None
    if cached is not None:
        shutil.copy2(cached.reference_sdf, run_dir / "reference.sdf")
        references = read_sdf(run_dir / "reference.sdf")
        scorers = build_scorers(cfg, references)
        prior_scores = project_prior_scores(cached.scores, cfg, scorers)
        prior_diagnostics = project_prior_diagnostics(cached.scores)
        keys = [
            key
            for scorer in scorers
            for key in (
                scorer.getKey() if isinstance(scorer.getKey(), list) else [scorer.getKey()]
            )
        ]
        thresholds = thresholds_from_scores(cfg, scorers, prior_scores, keys)
        env = build_environment(scorers, thresholds, scheme)
        prior_cache_status = f"hit:{cached.key}"
        logger.info("prior cache hit: %s (%d molecules)", cached.key, len(prior_smiles))
    else:
        references = build_reference(cfg, run_dir / "reference.sdf")
        if cfg.track_conformer_basins:
            scorers = build_scorers(
                cfg,
                references,
                basin_log_dir=run_dir / "conformer_basins",
                basin_stage="prior",
            )
        else:
            scorers = build_scorers(cfg, references)
        thresholds, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
        env = build_environment(scorers, thresholds, scheme)
        prior_diagnostics = components(env, prior_smiles)
        saved = None
        if cfg.use_prior_cache and prior_cache_key(cfg) is not None:
            saved = save_prior_bundle(
                cfg,
                run_dir / "reference.sdf",
                prior_smiles,
                prior_scores,
                prior_diagnostics,
                origin=f"run:{run_dir.name}",
            )
        prior_cache_status = f"miss:{saved.key}" if saved is not None else "bypassed"
    objectives = env.getScorerKeys()

    resolved.update(
        force_field=FORCE_FIELD if cfg.minimize else None,
        reward_scheme=type(scheme).__name__,
        # The factor the stored ROCS column was divided by, which score_arms converts out of.
        rocs_reward_scale=ROCS_PUBLISHED_MAX,
        objectives=objectives,
        thresholds=thresholds,
    )
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        config=resolved,
        objectives=objectives,
        thresholds=dict(zip(objectives, thresholds)),
        threshold_source="fixed" if cfg.fixed_thresholds else f"p{cfg.threshold_percentile:g}",
        prior_cache=prior_cache_status,
        reference_conformers=len(references),
        reference_sha256=digest(run_dir / "reference.sdf"),
        reference_energies=[
            round(float(mol.GetDoubleProp("Energy")), 3)
            for mol in references
            if mol.HasProp("Energy")
        ],
    )

    arms = {
        "prior": write_arm(
            run_dir / "prior.tsv",
            mark_desired(prior_scores, objectives, thresholds),
            prior_smiles,
            prior_diagnostics,
            origin="prior",
            seed=cfg.seed,
        )
    }

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
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(signum: int, _frame) -> None:
        raise SystemExit(128 + signum)

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
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if not model_path.exists():
            try:
                _save_model(explorer.getModel(), model_path)
            except Exception:
                logger.exception("failed to save %s", model_path.name)
        signal.signal(signal.SIGTERM, previous_sigterm)
    summary = _training_summary(run_dir / "rl_fit.tsv", cfg)
    if cfg.epochs > 0 and type(explorer).__module__.startswith("drugex.") and not summary:
        raise RuntimeError(f"RL training produced no epochs in {run_dir / 'rl_fit.tsv'}")
    logger.info("training summary: %s", summary)

    set_basin_stage(env, "final")
    # Guarded per arm: this stage runs after training, so a fault must cost one arm.
    for name in FINAL_ARMS:
        try:
            model = load_model(voc, run_dir / "models" / f"{name}.pkg")
            smiles = sample(model, cfg.n_final)
            # In slices: the whole arm at once is a ~32 GB parent for 5000 x 20 conformers,
            # at any worker count. Chunking is what lets a 32 GB job finish its own arm.
            scores, diagnostics = score_in_chunks(env, smiles)
            arms[name] = write_arm(
                run_dir / f"final_{name}.tsv",
                scores,
                smiles,
                diagnostics,
                origin="run",
                seed=cfg.seed,
            )
        except Exception:
            logger.exception(
                "arm %s failed; the others continue. Rebuild it later with: "
                "python -m _david.code.reconstruct.score_arms %s",
                name,
                run_dir,
            )

    resolved["summary"] = summary
    resolved["arms"] = arms
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        summary=summary,
        arms=arms,
        relaxation=dict(sorted(RELAXATION_TALLY.items())),
        checkpoints={
            name: {
                "file": f"models/{name}.pkg",
                "sha256": digest(run_dir / "models" / f"{name}.pkg"),
            }
            for name in FINAL_ARMS
        },
        artifacts={
            name: purpose for name, purpose in ARTIFACTS.items() if (run_dir / name).exists()
        },
    )

    try:
        from _david.code.analysis.report import analyse

        analyse(run_dir)
    except Exception:
        logger.exception(
            "analysis failed; rerun it with: python -m _david.code.analysis %s", run_dir
        )

    incomplete = incomplete_arms(run_dir, cfg.n_final)
    if incomplete:
        logger.error("incomplete final arms: %s", ", ".join(incomplete))
    manifest.update(
        state="failed" if incomplete else "complete",
        finished=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=round(time.monotonic() - started, 1),
        error="incomplete final arms: " + ", ".join(incomplete) if incomplete else None,
    )
    logger.info("done: %s", run_dir)
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
