"""Rebuild missing final arms from saved checkpoints: ``python -m _david.code.reconstruct.score_arms <run>``."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from _david.code.config import ROCS_PUBLISHED_MAX, Config
from _david.code.reconstruct.provenance import atomic_write_text
from _david.code.run import (
    FINAL_ARMS,
    incomplete_arms,
    load_model,
    sample,
    seed_everything,
    serial_run_refusal,
    write_arm,
)
from _david.code.scorers.objectives import build_environment, build_scorers, reward_scheme
from _david.code.scorers.utils import check_pool, score_in_chunks, use_double_precision_pickling
from drugex.data.corpus.vocabulary import VocSmiles

logger = logging.getLogger(__name__)


def config_from_run(run_dir: Path) -> Config:
    """Restore the configuration recorded in ``run_dir/config.json``."""
    return Config.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))


def recorded_thresholds(recorded: dict[str, Any]) -> list[float]:
    """Return recorded thresholds, rescaling legacy [0, 2] ROCS cuts onto [0, 1]."""
    stored = recorded["thresholds"]
    thresholds = (
        [float(stored[key]) for key in recorded["objectives"]]
        if isinstance(stored, dict)
        else [float(value) for value in stored]
    )
    scale = float(recorded.get("rocs_reward_scale", 1.0)) or 1.0
    if thresholds and scale != ROCS_PUBLISHED_MAX:
        thresholds[0] = thresholds[0] * scale / ROCS_PUBLISHED_MAX
    return thresholds


def rebuild(
    run_dir: Path,
    force: bool = False,
    only: tuple[str, ...] = (),
    n_jobs: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Resample and score missing (or forced) final arms from ``run_dir/models/*.pkg``."""
    run_dir = Path(run_dir)
    recorded = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(recorded)
    if n_jobs is not None:
        cfg = replace(cfg, n_jobs=n_jobs)

    reference = run_dir / "reference.sdf"
    if not reference.is_file():
        raise SystemExit(f"{reference} is missing; cannot rebuild against the original bank")

    missing = list(FINAL_ARMS) if force else incomplete_arms(run_dir, cfg.n_final)
    if only:
        missing = [name for name in missing if name in only]
    if not missing:
        logger.info("every arm is already written; nothing to rebuild")
        return {}

    check_pool()
    if refusal := serial_run_refusal(cfg):
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

    rebuilt: dict[str, dict[str, Any]] = {}
    for name in missing:
        model_path = run_dir / "models" / f"{name}.pkg"
        if not model_path.is_file():
            logger.warning("%s has no checkpoint at %s; skipping", name, model_path)
            continue
        logger.info("rebuilding arm %s from %s", name, model_path.name)
        smiles = sample(load_model(voc, model_path), cfg.n_final)
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


def _update_manifest(run_dir: Path, cfg: Config, rebuilt: dict[str, dict[str, Any]]) -> None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("arms", {}).update(rebuilt)
    manifest.pop("error", None)
    manifest["arms_rebuilt"] = sorted(rebuilt)
    still_missing = incomplete_arms(run_dir, cfg.n_final)
    manifest["state"] = "running" if still_missing else "complete"
    atomic_write_text(path, json.dumps(manifest, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="rescore arms that already exist")
    parser.add_argument("--only", nargs="+", choices=FINAL_ARMS, default=[])
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    for name, ratios in rebuild(
        args.run_dir, force=args.force, only=tuple(args.only), n_jobs=args.n_jobs
    ).items():
        print(f"{name}: {ratios}")
    try:
        from _david.code.analysis.report import analyse

        analyse(args.run_dir)
    except Exception:
        logger.exception("analysis failed; rerun: python -m _david.code.analysis %s", args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
