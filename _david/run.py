"""Run the pipeline: ``python -m _david.run [--set name=value ...] [--tag name]``.

Stages: reference bank -> prior baseline and thresholds -> RL -> one evaluation arm per
checkpoint -> figures and report. Everything the run resolved is written to
``config.json`` in the run directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
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

from _david.config import PROJECT_ROOT, Config
from _david.scorers.conformers import (
    FORCE_FIELD,
    RELAXATION_TALLY,
    build_reference,
    check_pool,
    use_double_precision_pickling,
)
from _david.training.explorer import DualCheckpointExplorer
from _david.provenance import ARTIFACTS, Manifest, describe, digest
from _david.scorers.objectives import (
    build_environment,
    build_scorers,
    calibrate_thresholds,
    mark_desired,
    reward_scheme,
)

#: Arms a completed run writes, in reporting order.
ARMS = ("prior", "desired_ratio", "desired_ratio_at_stop", "gmean")

logger = logging.getLogger("_david")


def _coerce(field: dataclasses.Field, raw: str):
    """Turn one ``--set`` value into the type its field is annotated with."""
    annotation = str(field.type)
    if "None" in annotation and raw.lower() in ("none", "null", ""):
        return None
    if "bool" in annotation:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"{field.name} is a boolean; got {raw!r}")
        return raw.lower() in ("true", "1")
    # Checked before the scalar branches: "tuple[float, ...]" contains "float", so a
    # scalar branch would match first and fail on every value the field can actually take.
    if "tuple" in annotation:
        inner = float if "float" in annotation else int if "int" in annotation else str
        text = raw.strip().strip("()[]")
        if not text:
            return ()
        return tuple(inner(part) for part in text.split(","))
    if "int" in annotation:
        return int(raw)
    if "float" in annotation:
        return float(raw)
    if "Path" in annotation:
        return Path(raw)
    return raw


def config_from_args(overrides: list[str]) -> Config:
    """Build a Config, applying ``name=value`` overrides against the dataclass fields."""
    by_name = {f.name: f for f in dataclasses.fields(Config)}
    values = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if name not in by_name:
            raise SystemExit(
                f"unknown setting {name!r}. Valid settings: {', '.join(sorted(by_name))}"
            )
        values[name] = _coerce(by_name[name], raw.strip())
    return Config(**values)


def _seed_everything(seed: int) -> None:
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


def _configure_logging(logfile: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)],
        force=True,
    )


def _load(voc: VocSmiles, checkpoint: Path) -> SequenceRNN:
    model = SequenceRNN(voc, device=DEFAULT_DEVICE, use_gpus=DEFAULT_GPUS)
    model.loadStatesFromFile(str(checkpoint))
    return model


def _sample(model: SequenceRNN, n: int, batch: int = 500) -> list[str]:
    """``n`` SMILES, sampled in batches, unfiltered.

    Unfiltered on purpose: DrugEx's ``generate`` would resample until it held n *valid
    unique* molecules, which makes valid_ratio and unique_ratio 1 by construction and the
    threshold calibration conditional on validity.
    """
    out: list[str] = []
    while len(out) < n:
        out.extend(model.sample(min(batch, n - len(out))))
    return out[:n]


def _canonical(smiles: list[str]) -> list[str]:
    """Canonical SMILES per entry, falling back to the input when it will not parse."""
    out = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        out.append(Chem.MolToSmiles(mol) if mol is not None else (value or ""))
    return out


def _raw_sa(canonical: pd.Series, valid: pd.Series) -> list[float]:
    """Unmodified synthetic accessibility on [1, 10], lower is better; 0.0 for invalid."""
    from drugex.training.scorers.properties import FAILURE_VALUES
    from drugex.training.scorers.sascorer import calculateScore

    out = []
    for smiles, is_valid in zip(canonical, valid):
        mol = Chem.MolFromSmiles(smiles) if is_valid == 1 and isinstance(smiles, str) else None
        if mol is None:
            out.append(0.0)
            continue
        try:
            out.append(round(float(calculateScore(mol)), 4))
        except Exception:
            out.append(FAILURE_VALUES["SA"])
    return out


def _components(env, smiles: list[str]) -> dict[str, list[float]]:
    """ShapeTanimoto / ColorTanimoto of the winning pose, if the scorer recorded them.

    Diagnostics only: they are columns in the arm table and never objectives, so the
    reward stays exactly as many objectives as ``env.getScorerKeys()`` reports.
    """
    for scorer in getattr(env, "scorers", []):
        if hasattr(scorer, "component_frame"):
            return scorer.component_frame(smiles)
    return {}


def _write_arm(
    path: Path, scores: pd.DataFrame, smiles: list[str], components: dict | None = None
) -> dict:
    """Write one scored arm as a TSV and return the ratios a thesis table reports."""
    scores = scores.copy()
    scores.insert(0, "SMILES", smiles)
    # Native computes Desired from the thresholds alone, so an invalid row (every
    # objective zeroed) is marked desirable whenever a threshold is 0.0. Every arm gets
    # the same guard the prior arm gets.
    if "Desired" in scores:
        scores.loc[scores["Valid"] == 0, "Desired"] = 0
    for column, values in (components or {}).items():
        scores[column] = values
        # DrugExEnvironment zeroes every objective for an invalid molecule; the
        # diagnostics follow the same rule so a row always adds up.
        scores.loc[scores["Valid"] == 0, column] = 0.0
    # Canonical structure, not the generated spelling: an RNN emits "Oc1ccccc1" and
    # "c1ccccc1O" as different strings for one molecule, and deduplicating on the raw
    # string would count it twice in unique_ratio and desired_ratio, and could put the
    # same molecule in the lead gallery several times. Written to the TSV as well, so
    # every reader of the arm deduplicates the same way this did.
    scores["Canonical"] = _canonical(smiles)
    # Raw SA beside the modified column. ClippedScore(5 -> 3) pins ~68 % of a drug-like
    # prior at exactly 1.0 (measured: 68.4 % of 950 valid prior molecules), so the reported SA column cannot show whether
    # training moved synthesizability: 3.2 -> 2.8 is invisible at the ceiling, and every
    # arm reads "SA 1.0" whether or not anything happened. Diagnostic only -- it is not
    # an objective and never reaches the reward.
    scores["SA_raw"] = _raw_sa(scores["Canonical"], scores["Valid"])
    scores.to_csv(path, sep="\t", index=False)
    valid = scores[scores["Valid"] == 1]
    unique = valid.drop_duplicates(subset="Canonical")
    ratios = {
        "n": len(scores),
        "valid_ratio": len(valid) / len(scores),
        "unique_ratio": len(unique) / len(scores),
        "desired_ratio": float(unique["Desired"].sum()) / len(scores),
    }
    logger.info("%s: %s", path.name, {k: round(v, 4) for k, v in ratios.items()})
    return ratios


def _arm_is_complete(path: Path, n: int) -> bool:
    """A full readable arm, not merely a file left behind by an interrupted writer."""
    try:
        table = pd.read_csv(path, sep="\t")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    # Valid and Desired, not SMILES: an RNN can emit an empty string, which round-trips
    # through the TSV as NaN in both SMILES and Canonical. A truncated final row leaves
    # the trailing numeric columns empty, which is what this actually has to catch.
    required = {"SMILES", "Canonical", "Valid", "Desired"}
    return (
        len(table) == n
        and required.issubset(table.columns)
        and not table[["Valid", "Desired"]].isna().any().any()
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
    _configure_logging(run_dir / "run.log")
    _seed_everything(cfg.seed)
    use_double_precision_pickling()

    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
    logger.info("run directory %s", run_dir)
    logger.info("device %s, gpus %s", DEFAULT_DEVICE, DEFAULT_GPUS)

    # Written before the first expensive stage, so a run killed at walltime is still
    # identifiable as `running` rather than looking like a finished one.
    manifest = Manifest(
        run_dir / "manifest.json",
        run=run_dir.name,
        query=cfg.reference_smiles,
        started=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        device=str(DEFAULT_DEVICE),
        gpus=list(DEFAULT_GPUS) if DEFAULT_GPUS else [],
        provenance=describe(PROJECT_ROOT),
        # Recorded, because it decides whether this run had 16 cores or one of them.
        # Probe before the reference bank. A failed probe disables every production
        # pool, so scoring is deliberately serial rather than entering another unbounded
        # map with the same broken start method. The probe's own workers are then closed.
        mp_context=check_pool(),
        config=resolved,
    )
    logger.info(
        "source %s (dirty=%s)",
        manifest.data["provenance"]["source"]["revision"],
        manifest.data["provenance"]["source"]["dirty"],
    )

    started = time.monotonic()
    try:
        return _pipeline(cfg, run_dir, resolved, manifest, started)
    except BaseException as error:
        manifest.finish("failed", error=f"{type(error).__name__}: {error}")
        raise


def _pipeline(
    cfg: Config, run_dir: Path, resolved: dict, manifest: Manifest, started: float
) -> int:
    """The stages, with the manifest advanced after each one that produces an artifact."""
    # First, before the reference bank and before 5000 prior molecules are scored: an
    # unknown scheme name is a typo in a submitted delta, and it must cost a second
    # rather than the hour it takes to reach the first getRewards call.
    scheme = reward_scheme(cfg.scheme)
    references = build_reference(cfg, run_dir / "reference.sdf")

    checkpoint, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)
    agent = _load(voc, checkpoint)
    # The mutate prior is a second, independent copy of the same pretrained weights: the
    # agent is trained, the prior stays frozen as the distribution epsilon-mutation draws
    # from. Sharing one object would let training change the exploration distribution.
    prior = _load(voc, checkpoint)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    prior.eval()

    scorers = build_scorers(cfg, references)
    logger.info("sampling %d molecules from the prior to calibrate thresholds", cfg.n_prior)
    prior_smiles = _sample(prior, cfg.n_prior)
    thresholds, prior_scores = calibrate_thresholds(cfg, scorers, prior_smiles)
    env = build_environment(scorers, thresholds, scheme)

    resolved["force_field"] = FORCE_FIELD if cfg.minimize else None
    # Recorded as the class that will actually rank the rewards, not as the short name:
    # "WeightedSum" in config.json is unambiguous to a reader who has never seen `--scheme`.
    resolved["reward_scheme"] = type(scheme).__name__
    resolved["objectives"] = env.getScorerKeys()
    resolved["thresholds"] = thresholds
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        config=resolved,
        objectives=env.getScorerKeys(),
        thresholds=dict(zip(env.getScorerKeys(), thresholds)),
        threshold_source="fixed" if cfg.fixed_thresholds else f"p{cfg.threshold_percentile:g}",
        reference_conformers=len(references),
        reference_sha256=digest(run_dir / "reference.sdf"),
        # The bank decides every score in the run, so what it actually contains is
        # recorded rather than inferred from the settings: a bank smaller than requested,
        # or one whose energies are implausible, is visible in the manifest alone.
        reference_energies=[
            round(float(m.GetDoubleProp("Energy")), 3) for m in references if m.HasProp("Energy")
        ],
    )

    # The prior arm is the pre-training baseline, so it is kept as an arm of its own. Its
    # Desired column is rewritten against the thresholds it just produced.
    arms = {
        "prior": _write_arm(
            run_dir / "prior.tsv",
            mark_desired(prior_scores, env.getScorerKeys(), thresholds),
            prior_smiles,
            _components(env, prior_smiles),
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

    # Each arm is written as it completes and guarded on its own: this stage is a quarter
    # of the compute budget and it runs after training, so a fault here must cost one arm
    # rather than every table the run exists to produce.
    for name in ("desired_ratio", "desired_ratio_at_stop", "gmean"):
        try:
            model = _load(voc, run_dir / "models" / f"{name}.pkg")
            smiles = _sample(model, cfg.n_final)
            scored = env.getScores(smiles)
            arms[name] = _write_arm(
                run_dir / f"final_{name}.tsv", scored, smiles, _components(env, smiles)
            )
        except Exception:
            logger.exception(
                "arm %s failed; the others continue. Rebuild it later with: "
                "python -m _david.score_arms %s",
                name,
                run_dir,
            )

    resolved["summary"] = summary
    resolved["arms"] = arms
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    manifest.update(
        summary=summary,
        arms=arms,
        # Three of these reasons drop a conformer in the minimised arm only, so this is
        # the number behind any claim that the two arms scored the same molecules.
        relaxation=dict(sorted(RELAXATION_TALLY.items())),
        # Each arm named with the checkpoint it came from and that file's hash, so an arm
        # can never be silently re-attributed to a different model later.
        checkpoints={
            name: {
                "file": f"models/{name}.pkg",
                "sha256": digest(run_dir / "models" / f"{name}.pkg"),
            }
            for name in ("desired_ratio", "desired_ratio_at_stop", "gmean")
        },
        artifacts={
            name: purpose for name, purpose in ARTIFACTS.items() if (run_dir / name).exists()
        },
    )

    # Imported here, and never allowed to fail the run: by this point every number is on
    # disk, and `python -m _david.analysis <run_dir>` reproduces the figures.
    try:
        from _david.analysis.report import analyse

        analyse(run_dir)
    except Exception:
        logger.exception("analysis failed; rerun it with: python -m _david.analysis %s", run_dir)

    incomplete = [
        name
        for name in ARMS
        if name != "prior" and not _arm_is_complete(run_dir / f"final_{name}.tsv", cfg.n_final)
    ]
    if incomplete:
        logger.error("incomplete final arms: %s", ", ".join(incomplete))
    manifest.finish(
        "failed" if incomplete else "complete",
        finished=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=round(time.monotonic() - started, 1),
        error="incomplete final arms: " + ", ".join(incomplete) if incomplete else None,
    )
    logger.info("done: %s", run_dir)
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
