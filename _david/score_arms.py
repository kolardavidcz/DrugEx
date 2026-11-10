"""Rebuild a run's final arms from its checkpoints: ``python -m _david.score_arms <run_dir>``.

The final stage samples and scores four arms of ``n_final`` molecules each -- about a
quarter of a run's compute, and it happens *after* training. A walltime kill or a fault
there leaves a directory with valid checkpoints, a manifest stuck at ``running`` and no
arm tables, and training cannot be resumed: ``fit`` always restarts from the pretrained
weights. Without this the whole run is lost to recover a stage that only needs the files
already on disk.

Nothing here re-derives anything: the configuration, thresholds and objectives come from
the run's own ``config.json`` and the policies from its own ``models/*.pkg``, so a rebuilt
arm is scored against exactly what the run scored against.

It is **not** the same draw. The run's RNG state at the final stage depends on all of
training, which cannot be reconstructed, so a rebuilt arm is the same policy sampled
again. Sampling is seeded from ``Config.seed`` so that two rebuilds agree with each
other, and the manifest records which arms were rebuilt. Arms that already exist are left
alone, so this is safe to re-run after a partial recovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import fields
from pathlib import Path

from drugex.data.corpus.vocabulary import VocSmiles

from _david.config import Config
from _david.run import ARMS, _components, _load, _sample, _seed_everything, _write_arm
from _david.scorers.conformers import check_pool, use_double_precision_pickling
from _david.scorers.objectives import build_environment, build_scorers, reward_scheme

logger = logging.getLogger(__name__)


def config_from_run(run_dir: Path) -> Config:
    """The configuration the run actually used, not today's defaults."""
    recorded = json.loads((run_dir / "config.json").read_text())
    declared = {f.name: f.type for f in fields(Config)}
    values = {}
    for key, value in recorded.items():
        if key not in declared or value is None:
            continue
        if "Path" in str(declared[key]):
            values[key] = Path(value)
        elif "tuple" in str(declared[key]):
            values[key] = tuple(value)
        else:
            values[key] = value
    return Config(**values)


def rebuild(run_dir: Path, force: bool = False) -> dict:
    """Score every missing arm of ``run_dir`` and update its manifest."""
    run_dir = Path(run_dir)
    recorded = json.loads((run_dir / "config.json").read_text())
    cfg = config_from_run(run_dir)
    reference = run_dir / "reference.sdf"
    if not reference.is_file():
        raise SystemExit(
            f"{reference} is missing: the reference bank cannot be rebuilt without "
            "re-embedding, which would not be the bank this run scored against."
        )

    missing = [
        name
        for name in ARMS
        if name != "prior" and (force or not (run_dir / f"final_{name}.tsv").is_file())
    ]
    if not missing:
        logger.info("every arm is already written; nothing to rebuild")
        return {}

    check_pool()
    use_double_precision_pickling()
    # Seeded so two rebuilds of the same directory agree. This cannot reproduce the
    # original draw -- see the module docstring.
    _seed_everything(cfg.seed)
    # Thresholds come from the run's own calibration. Recalibrating would need a fresh
    # prior sample and would silently change what `Desired` means for the rebuilt arms.
    # config.json keeps thresholds as a list in objective order; manifest.json keeps the
    # same values keyed by objective. Accept either, so a run recovered from one file is
    # scored against exactly the thresholds the other recorded.
    stored = recorded["thresholds"]
    thresholds = (
        [float(stored[key]) for key in recorded["objectives"]]
        if isinstance(stored, dict)
        else [float(value) for value in stored]
    )
    env = build_environment(
        build_scorers(cfg, [str(reference)]), thresholds, reward_scheme(cfg.scheme)
    )
    checkpoint, vocabulary = cfg.pretrained()
    voc = VocSmiles.fromFile(str(vocabulary), encode_frags=False)

    rebuilt = {}
    for name in missing:
        model_path = run_dir / "models" / f"{name}.pkg"
        if not model_path.is_file():
            logger.warning("%s has no checkpoint at %s; skipping", name, model_path)
            continue
        logger.info("rebuilding arm %s from %s", name, model_path.name)
        model = _load(voc, model_path)
        smiles = _sample(model, cfg.n_final)
        scored = env.getScores(smiles)
        rebuilt[name] = _write_arm(
            run_dir / f"final_{name}.tsv", scored, smiles, _components(env, smiles)
        )

    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and rebuilt:
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("arms", {}).update(rebuilt)
        manifest.pop("error", None)  # the crash that made this necessary is now resolved
        # Recorded, not hidden: an arm sampled later is the same policy on a different
        # random draw, and a reader comparing two runs should be able to see that.
        manifest["arms_rebuilt"] = sorted(rebuilt)
        # Only complete when every arm exists. Declaring a run complete while an arm is
        # still missing is exactly the corpse-that-looks-finished the state machine is
        # there to prevent.
        still_missing = [
            name
            for name in ARMS
            if name != "prior" and not (run_dir / f"final_{name}.tsv").is_file()
        ]
        manifest["state"] = "running" if still_missing else "complete"
        if still_missing:
            logger.warning("still missing: %s; state stays 'running'", ", ".join(still_missing))
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return rebuilt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="rescore arms that already have a table"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    rebuilt = rebuild(args.run_dir, force=args.force)
    for name, ratios in rebuilt.items():
        print(f"{name}: {ratios}")
    try:
        from _david.analysis.report import analyse

        analyse(args.run_dir)
    except Exception:
        logger.exception(
            "analysis failed; rerun it with: python -m _david.analysis %s", args.run_dir
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
