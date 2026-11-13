"""Rebuild a run's final arms from its checkpoints: python -m _david.code.reconstruct.score_arms <run>

A walltime kill after training leaves valid checkpoints and no arm tables, and fit() cannot
resume. Everything comes from the run's own config.json and models/*.pkg, so a rebuilt arm is
scored against what the run scored against -- but not on the same draw.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields, replace
from pathlib import Path

from drugex.data.corpus.vocabulary import VocSmiles

from _david.code.config import ROCS_PUBLISHED_MAX, Config
from _david.code.run import (
    FINAL_ARMS,
    incomplete_arms,
    load_model,
    sample,
    seed_everything,
    serial_run_refusal,
    write_arm,
)
from _david.code.scorers.objectives import (
    build_environment,
    build_scorers,
    reward_scheme,
)
from _david.code.scorers.utils import (
    check_pool,
    score_in_chunks,
    use_double_precision_pickling,
)

logger = logging.getLogger(__name__)


def config_from_run(run_dir: Path) -> Config:
    """The configuration the run used, not today's defaults."""
    recorded = json.loads((run_dir / "config.json").read_text())
    values = {}
    for field in fields(Config):
        value = recorded.get(field.name)
        if value is None:
            continue
        annotation = str(field.type)
        if "Path" in annotation:
            value = Path(value)
        elif "tuple" in annotation:
            value = tuple(value)
        values[field.name] = value
    return Config(**values)


def recorded_thresholds(recorded: dict) -> list[float]:
    """The run's own thresholds, converted out of the scale that run recorded them on."""
    stored = recorded["thresholds"]
    thresholds = (
        [float(stored[key]) for key in recorded["objectives"]]
        if isinstance(stored, dict)
        else [float(value) for value in stored]
    )
    scale = float(recorded.get("rocs_reward_scale", 1.0)) or 1.0
    if thresholds and scale != ROCS_PUBLISHED_MAX:
        rescaled = thresholds[0] * scale / ROCS_PUBLISHED_MAX
        logger.info(
            "ROCS threshold %.4f recorded on a [0, %g] scale, rescaled to %.4f",
            thresholds[0],
            scale,
            rescaled,
        )
        thresholds[0] = rescaled
    return thresholds


def rebuild(run_dir: Path, force: bool = False, only: tuple[str, ...] = (),
            n_jobs: int | None = None) -> dict:
    run_dir = Path(run_dir)
    recorded = json.loads((run_dir / "config.json").read_text())
    cfg = config_from_run(run_dir)
    if n_jobs is not None:
        # In memory only: a shared node splits its cores between concurrent rebuilds, but
        # the recorded config stays what the run trained with. Config is frozen.
        cfg = replace(cfg, n_jobs=n_jobs)
    reference = run_dir / "reference.sdf"
    if not reference.is_file():
        raise SystemExit(
            f"{reference} is missing: the reference bank cannot be rebuilt without "
            "re-embedding, which would not be the bank this run scored against."
        )

    missing = list(FINAL_ARMS) if force else incomplete_arms(run_dir, cfg.n_final)
    if only:
        # A triage rebuild names the arms it needs (``--only desired_ratio_at_stop``):
        # the verdict reads the stop arm, so that one arm upgrades an unfinished run.
        missing = [name for name in missing if name in only]
    if not missing:
        logger.info("every arm is already written; nothing to rebuild")
        return {}

    check_pool()
    refusal = serial_run_refusal(cfg)
    if refusal:
        raise SystemExit(refusal)
    use_double_precision_pickling()
    seed_everything(cfg.seed)

    env = build_environment(
        build_scorers(cfg, [str(reference)]),
        recorded_thresholds(recorded),
        reward_scheme(cfg.scheme),
    )
    _, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)

    rebuilt = {}
    for name in missing:
        model_path = run_dir / "models" / f"{name}.pkg"
        if not model_path.is_file():
            logger.warning("%s has no checkpoint at %s; skipping", name, model_path)
            continue
        logger.info("rebuilding arm %s from %s", name, model_path.name)
        smiles = sample(load_model(voc, model_path), cfg.n_final)
        # A whole 5000-molecule arm at 20 conformers needs ~32 GB in the parent, whatever
        # the pool size; scoring it in slices is what keeps a rebuild inside its request.
        scores, diagnostics = score_in_chunks(env, smiles)
        rebuilt[name] = write_arm(
            run_dir / f"final_{name}.tsv",
            scores,
            smiles,
            diagnostics,
            origin="rebuilt",
            seed=cfg.seed,
        )

    if rebuilt:
        _update_manifest(run_dir, cfg, rebuilt)
    return rebuilt


def _update_manifest(run_dir: Path, cfg: Config, rebuilt: dict) -> None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    manifest.setdefault("arms", {}).update(rebuilt)
    manifest.pop("error", None)
    # Recorded, not hidden: an arm sampled later is the same policy on a different draw.
    manifest["arms_rebuilt"] = sorted(rebuilt)
    still_missing = incomplete_arms(run_dir, cfg.n_final)
    manifest["state"] = "running" if still_missing else "complete"
    if still_missing:
        logger.warning("still missing: %s; state stays 'running'", ", ".join(still_missing))
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="rescore arms that already have a table"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=FINAL_ARMS,
        default=[],
        help="rebuild only these arms (default: every incomplete one)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="worker processes for scoring (default: the run's own config)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    for name, ratios in rebuild(
        args.run_dir, force=args.force, only=tuple(args.only), n_jobs=args.n_jobs
    ).items():
        print(f"{name}: {ratios}")
    try:
        from _david.code.analysis.report import analyse

        analyse(args.run_dir)
    except Exception:
        logger.exception(
            "analysis failed; rerun it with: python -m _david.code.analysis %s", args.run_dir
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
