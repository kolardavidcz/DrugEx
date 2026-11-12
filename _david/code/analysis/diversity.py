"""Per-run chemical-diversity profiling with a per-run cache.

    python -m _david.code.analysis.diversity [--workers N] [--force] [--limit N]

The collapse verdict sees *duplication*, but not chemical concentration: 1,200 distinct
canonicals can be one scaffold family wearing 1,200 hats, and a 50-draw per-epoch eval
cannot see either. So this module reads each run's arm tables molecule by molecule — the
stop arm (``final_desired_ratio_at_stop.tsv``, the live final policy) and its prior — and
computes the generative-modelling diversity repertoire on the unique valid molecules:

- **effective molecules / scaffolds** — exp(Shannon entropy) of the multiplicity and
  scaffold distributions: how many distinct findings the arm effectively holds, in the
  units of a uniform arm;
- **internal diversity and nearest-neighbour similarity** — 1 − mean pairwise Tanimoto,
  and each molecule's closest neighbour: a mode-collapsed arm clusters at NN ≈ 1 even
  when every canonical is distinct;
- **prior echo and novelty** — each stop-arm molecule's closest prior molecule: reward
  bought by re-describing the prior (echo) versus by finding new space (novelty);
- **a 2-D PCA thumbnail** of stop and prior unique molecules, fitted on both: the
  chemical-space picture of what training moved, for the atlases to draw.

Each profile lands in ``<run>/cache_stats/diversity.json``, keyed to the size and mtime
of the arm tables it was computed from: fresh profiles are skipped on re-run, and the
collapse scanner picks them up for the atlas payloads without recomputation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.analysis.collapse import _read_table
from _david.code.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Bump when the profile's meaning changes, so every cache invalidates at once.
DIVERSITY_VERSION = 2

#: A neighbour this close is the same finding twice; a prior match this close is an echo.
NEAR_DUP_SIM = 0.90
PRIOR_ECHO_SIM = 0.95
NOVEL_SIM = 0.60

#: Morgan fingerprint geometry and the thumbnail's point budget.
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
THUMB_POINTS = 600

STOP_ARM = "final_desired_ratio_at_stop.tsv"
PRIOR_ARM = "prior.tsv"


class ArmUniques:
    """One arm's unique valid molecules, with everything the profile needs of them.

    ``canonicals[i]`` appeared ``counts[i]`` times in the draw; ``desired[i]`` is the best
    Desired any of its copies reached (the same canonical can be scored differently in
    different rows — the draw, not the canonical, is what was scored).
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self.canonicals: list[str] = []
        self.counts: list[int] = []
        self.desired: list[int] = []
        if frame.empty or "Canonical" not in frame.columns:
            return
        valid = frame[pd.to_numeric(frame.get("Valid"), errors="coerce").fillna(0) == 1].copy()
        valid["Canonical"] = valid["Canonical"].fillna("").astype(str)
        valid = valid[valid["Canonical"] != ""]
        if "Desired" in valid.columns:
            valid["Desired"] = pd.to_numeric(valid["Desired"], errors="coerce").fillna(0)
        grouped = valid.groupby("Canonical", sort=True)
        self.canonicals = list(grouped.groups.keys())
        self.counts = [int(n) for n in grouped.size()]
        self.desired = [int(d) for d in (
            grouped["Desired"].max() if "Desired" in valid.columns else grouped.size() * 0
        )]

    def __len__(self) -> int:
        return len(self.canonicals)

    @property
    def fingerprints(self) -> list[Any]:
        """Morgan fingerprints of the unique canonicals, in the same order."""
        fps = []
        for canonical in self.canonicals:
            mol = Chem.MolFromSmiles(canonical)
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS
                )
                if mol is not None
                else None
            )
        return fps


def _entropy_shannon(weights: list[int] | np.ndarray) -> float | None:
    """exp(H) of a distribution given by counts: its effective number of outcomes."""
    counts = np.asarray(weights, dtype=float)
    positive = counts[counts > 0]
    if positive.size < 1:
        return None
    shares = positive / positive.sum()
    return float(np.exp(-(shares * np.log(shares)).sum()))


def _scaffold_counts(canonicals: list[str]) -> Counter:
    counts: Counter = Counter()
    for canonical in canonicals:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:  # pragma: no cover - RDKit refuses some valences
            continue
        if scaffold:
            counts[scaffold] += 1
    return counts


def _pairwise_profile(fps: list[Any]) -> dict[str, Any]:
    """Internal diversity and nearest-neighbour statistics over one arm's fingerprints.

    Computed row by row so the full similarity matrix is never materialised: one pass
    keeps the running mean, the global minimum (the arm's widest pair) and every
    molecule's closest neighbour.
    """
    n = len(fps)
    if n < 2:
        return {"internal_diversity": None, "nn_sim_p50": None, "nn_sim_p90": None,
                "nn_sim_max": None, "near_dup_rate": None, "min_sim": None}
    nearest = np.zeros(n)
    total, pairs = 0.0, 0
    minimum = 1.0
    for i in range(n - 1):
        rest = fps[i + 1:]
        if any(fp is None for fp in (fps[i], *rest[:0])):
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], rest))
        total += float(sims.sum())
        pairs += sims.size
        nearest[i] = max(nearest[i], float(sims.max()))
        np.maximum(nearest[i + 1:], sims, out=nearest[i + 1:])
        minimum = min(minimum, float(sims.min()))
    if pairs == 0:
        return {"internal_diversity": None, "nn_sim_p50": None, "nn_sim_p90": None,
                "nn_sim_max": None, "near_dup_rate": None, "min_sim": None}
    return {
        "internal_diversity": round(1.0 - total / pairs, 4),
        "nn_sim_p50": round(float(np.percentile(nearest, 50)), 4),
        "nn_sim_p90": round(float(np.percentile(nearest, 90)), 4),
        "nn_sim_max": round(float(nearest.max()), 4),
        "near_dup_rate": round(float((nearest >= NEAR_DUP_SIM).mean()), 4),
        "min_sim": round(minimum, 4),
    }


def _cross_profile(stop_fps: list[Any], prior_fps: list[Any]) -> dict[str, Any]:
    """Each stop-arm molecule's closest prior molecule: echo of, or distance from, home."""
    if not stop_fps or not prior_fps:
        return {"prior_sim_p50": None, "prior_echo_rate": None, "novelty_rate": None}
    closest = np.asarray([
        max(DataStructs.BulkTanimotoSimilarity(fp, prior_fps))
        for fp in stop_fps if fp is not None
    ])
    if closest.size == 0:
        return {"prior_sim_p50": None, "prior_echo_rate": None, "novelty_rate": None}
    return {
        "prior_sim_p50": round(float(np.percentile(closest, 50)), 4),
        "prior_echo_rate": round(float((closest >= PRIOR_ECHO_SIM).mean()), 4),
        "novelty_rate": round(float((closest < NOVEL_SIM).mean()), 4),
    }


def _thumb_coordinates(stop: ArmUniques, prior: ArmUniques) -> dict[str, Any] | None:
    """2-D PCA of stop and prior unique molecules, fitted on both, subsampled for the atlas.

    Fitting on the union is the point: both clouds share axes, so the thumbnail shows
    where the policy moved the prior's cloud instead of two unrelated projections.
    """
    fps = [fp for fp in (*stop.fingerprints, *prior.fingerprints) if fp is not None]
    n_stop = sum(1 for fp in stop.fingerprints if fp is not None)
    if len(fps) < 3:
        return None
    matrix = np.zeros((len(fps), FINGERPRINT_BITS), dtype=np.float32)
    for row, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, matrix[row])
    centered = matrix - matrix.mean(axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    projections = centered @ right[:2].T
    explained = singular[:2] ** 2 / max((singular ** 2).sum(), 1e-12)

    def subsample(points: np.ndarray, desired: list[int], budget: int) -> dict[str, Any]:
        step = max(1, len(points) // budget)
        picked = points[::step]
        flags = desired[::step]
        return {
            "x": [round(float(x), 2) for x in picked[:, 0]],
            "y": [round(float(y), 2) for y in picked[:, 1]],
            "desired": [int(d) for d in flags],
        }

    return {
        "explained": [round(float(v), 4) for v in explained],
        "stop": subsample(projections[:n_stop], stop.desired, THUMB_POINTS),
        "prior": subsample(projections[n_stop:], prior.desired, THUMB_POINTS),
    }


def arm_profile(uniques: ArmUniques) -> dict[str, Any]:
    """Every per-arm statistic, from one arm's unique valid molecules."""
    effective = _entropy_shannon(uniques.counts)
    scaffolds = _scaffold_counts(uniques.canonicals)
    profile = {
        "n_unique": len(uniques),
        "effective_molecules": round(effective, 1) if effective is not None else None,
        "scaffolds": len(scaffolds),
        "effective_scaffolds": round(_entropy_shannon(list(scaffolds.values())), 1)
        if scaffolds else None,
    }
    profile.update(_pairwise_profile(uniques.fingerprints))
    return profile


def profile_run(run: Path) -> dict[str, Any] | None:
    """The whole diversity profile of one run, or None when it has no readable arms."""
    prior_frame = _read_table(run / PRIOR_ARM)
    stop_path = run / STOP_ARM
    stop_frame = _read_table(stop_path) if stop_path.is_file() else pd.DataFrame()
    if prior_frame.empty:
        return None

    prior = ArmUniques(prior_frame)
    stop = ArmUniques(stop_frame)
    profile: dict[str, Any] = {
        "version": DIVERSITY_VERSION,
        "arms": {"prior": arm_profile(prior)},
    }
    if len(stop):
        profile["arms"]["stop"] = arm_profile(stop)
        profile["cross"] = _cross_profile(stop.fingerprints, prior.fingerprints)
        profile["thumb"] = _thumb_coordinates(stop, prior)
    return profile


def _sources(run: Path) -> dict[str, list[int]]:
    """The arm tables a profile depends on, as (size, mtime) pairs; -1 marks absence."""
    sources = {}
    for name in (PRIOR_ARM, STOP_ARM):
        path = run / name
        stat = path.stat() if path.is_file() else None
        sources[name] = [stat.st_size, int(stat.st_mtime)] if stat else [-1, -1]
    return sources


def profile_is_fresh(run: Path) -> bool:
    """True when the cached profile exists and every source table is unchanged."""
    cache = run / "cache_stats" / "diversity.json"
    try:
        data = json.loads(cache.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("version") != DIVERSITY_VERSION:
        return False
    recorded = data.get("sources") or {}
    return recorded == _sources(run)


def load_profile(run: Path) -> dict[str, Any] | None:
    """The run's cached profile, if it is fresh; public read path for the scanner."""
    if not profile_is_fresh(run):
        return None
    try:
        return json.loads((run / "cache_stats" / "diversity.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _worker(run: Path) -> tuple[str, str, float]:
    """Compute and cache one run's profile; returns (name, status, seconds)."""
    started = time.monotonic()
    try:
        profile = profile_run(run)
        if profile is None:
            return run.name, "no readable prior arm", 0.0
        profile["sources"] = _sources(run)
        cache = run / "cache_stats"
        cache.mkdir(exist_ok=True)
        (cache / "diversity.json").write_text(json.dumps(profile))
        return run.name, "ok", time.monotonic() - started
    except Exception as error:  # keep the batch alive; report and move on
        return run.name, f"failed: {error!r}", time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "_david" / "output")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true", help="recompute fresh profiles too")
    parser.add_argument("--limit", type=int, default=None, help="profile at most N runs (smoke tests)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RDLogger.DisableLog("rdApp.*")

    runs = sorted(
        run for run in args.root.iterdir()
        if run.is_dir() and (run / "manifest.json").is_file() and (run / PRIOR_ARM).is_file()
    )
    todo = [run for run in runs if args.force or not profile_is_fresh(run)]
    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("%d runs, %d to profile (%d fresh skipped)", len(runs), len(todo),
                len(runs) - len(todo))

    statuses: list[tuple[str, str, float]] = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, run): run for run in todo}
            for i, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                statuses.append(result)
                logger.info("[%d/%d] %s: %s (%.0fs)", i, len(todo), *result)
    failed = [s for s in statuses if s[1].startswith("failed")]
    logger.info("done: %d ok, %d failed, %.0fs total compute",
                len(statuses) - len(failed), len(failed), sum(s[2] for s in statuses))
    for name, status, _ in failed:
        logger.error("%s: %s", name, status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
