"""One reinforcement-learning run: python -m _david.code.run [--set name=value ...] [--tag name]"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from drugex import DEFAULT_DEVICE, DEFAULT_GPUS
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.scorers.parallel import capped_n_jobs

from _david.code.config import PROJECT_ROOT, Config
from _david.code.provenance import ARTIFACTS, Manifest, describe, digest
from _david.code.scorers.conformers import (
    FORCE_FIELD,
    RELAXATION_TALLY,
    build_reference,
    check_pool,
    pool_is_healthy,
    use_double_precision_pickling,
)
from _david.code.scorers.objectives import (
    ROCS_PUBLISHED_MAX,
    build_environment,
    build_scorers,
    calibrate_thresholds,
    mark_desired,
    reward_scheme,
)
from _david.code.training.explorer import DualCheckpointExplorer

FINAL_ARMS = ("desired_ratio", "desired_ratio_at_stop", "gmean")
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


def components(env, smiles: list[str]) -> dict[str, list[float]]:
    for scorer in getattr(env, "scorers", []):
        if hasattr(scorer, "component_frame"):
            return scorer.component_frame(smiles)
    return {}


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


def write_arm(
    path: Path, scores: pd.DataFrame, smiles: list[str], diagnostics: dict | None = None
) -> dict:
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
    }
    logger.info("%s: %s", path.name, {key: round(value, 4) for key, value in ratios.items()})
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


def incomplete_arms(run_dir: Path, n: int) -> list[str]:
    return [name for name in FINAL_ARMS if not arm_is_complete(run_dir / f"final_{name}.tsv", n)]


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


def _pipeline(
    cfg: Config, run_dir: Path, resolved: dict, manifest: Manifest, started: float
) -> int:
    # Before anything expensive: an unknown scheme name is a typo in a submitted delta.
    scheme = reward_scheme(cfg.scheme)
    references = build_reference(cfg, run_dir / "reference.sdf")

    checkpoint, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)
    agent = load_model(voc, checkpoint)
    # An independent copy: the agent trains, the prior stays frozen for epsilon-mutation.
    prior = load_model(voc, checkpoint)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    prior.eval()

    scorers = build_scorers(cfg, references)
    logger.info("sampling %d molecules from the prior to calibrate thresholds", cfg.n_prior)
    prior_smiles = sample(prior, cfg.n_prior)
    thresholds, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
    env = build_environment(scorers, thresholds, scheme)
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
            components(env, prior_smiles),
        )
    }

    explorer = DualCheckpointExplorer(
        agent=agent,
        env=env,
        mutate=prior,
        batch_size=cfg.batch_size,
        epsilon=cfg.epsilon,
        beta=cfg.beta,
        n_samples=cfg.n_samples,
        device=DEFAULT_DEVICE,
        use_gpus=DEFAULT_GPUS,
        desired_ratio_path=run_dir / "models" / "desired_ratio.pkg",
        desired_ratio_at_stop_path=run_dir / "models" / "desired_ratio_at_stop.pkg",
        gmean_path=run_dir / "models" / "gmean.pkg",
        training_samples_path=run_dir / "training_samples.tsv",
        n_eval=cfg.n_eval,
    )
    summary = explorer.fit(
        monitor=FileMonitor(str(run_dir / "rl"), save_smiles=True),
        epochs=cfg.epochs,
        patience=cfg.patience,
        min_epochs=cfg.min_epochs,
    )
    logger.info("training summary: %s", summary)

    # Guarded per arm: this stage runs after training, so a fault must cost one arm.
    for name in FINAL_ARMS:
        try:
            model = load_model(voc, run_dir / "models" / f"{name}.pkg")
            smiles = sample(model, cfg.n_final)
            arms[name] = write_arm(
                run_dir / f"final_{name}.tsv",
                env.getScores(smiles),
                smiles,
                components(env, smiles),
            )
        except Exception:
            logger.exception(
                "arm %s failed; the others continue. Rebuild it later with: "
                "python -m _david.code.score_arms %s",
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
