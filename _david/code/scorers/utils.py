"""Hardware plumbing: multiprocessing pools, chunked scoring, parallel embedding, and SDF I/O."""

from __future__ import annotations

import gc
import logging
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from rdkit import Chem

from _david.code.config import Config
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

logger = logging.getLogger(__name__)

_POOL_BROKEN = False
_T = TypeVar("_T")
_R = TypeVar("_R")

#: Molecules per scoring slice; keeps parent peak memory bounded (~3 GB vs ~32 GB for 5,000 x 20).
SCORE_CHUNK = 500


def components(env: Any, smiles: list[str]) -> dict[str, list[float]]:
    """Return winning ``ShapeTanimoto``/``ColorTanimoto`` lists if any scorer in ``env`` records them."""
    for scorer in getattr(env, "scorers", ()):
        if hasattr(scorer, "component_frame"):
            return scorer.component_frame(smiles)
    return {}


def score_in_chunks(
    env: Any, smiles: list[str], *, chunk: int = SCORE_CHUNK
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Score ``smiles`` in memory-bounded slices while preserving exact scores and order."""
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if not smiles:
        return env.getScores(smiles), {}

    frames: list[pd.DataFrame] = []
    diagnostics: dict[str, list[float]] = {}
    for start in range(0, len(smiles), chunk):
        piece = smiles[start : start + chunk]
        frames.append(env.getScores(piece).reset_index(drop=True))
        for col, values in components(env, piece).items():
            diagnostics.setdefault(col, []).extend(values)
        gc.collect()
    return pd.concat(frames, ignore_index=True), diagnostics


def pool_is_healthy() -> bool:
    """Whether the last :func:`check_pool` probe started workers cleanly."""
    return not _POOL_BROKEN


def pool_can_run(worker_count: int, n_items: int, *, min_items: int) -> bool:
    """Whether spawning a multiprocessing pool is legal and worthwhile."""
    return (
        worker_count > 1
        and n_items >= min_items
        and not multiprocessing.current_process().daemon
        and pool_is_healthy()
    )


def _touch(value: _T) -> _T:
    return value


def check_pool(timeout: float = 300.0) -> str:
    """Verify worker-pool startup once before expensive stages; return the context status."""
    global _POOL_BROKEN
    _POOL_BROKEN = False
    context = pool_context()
    method = context.get_start_method()
    pool = None
    try:
        pool = context.Pool(2)
        pool.map_async(_touch, (1, 2)).get(timeout=timeout)
    except Exception as error:
        _POOL_BROKEN = True
        logger.error(
            "multiprocessing %r cannot start workers (%s: %s); run as `python -m _david.code.run` "
            "or set DRUGEX_MP_CONTEXT=fork",
            method,
            type(error).__name__,
            error,
        )
        return f"{method} (broken: {type(error).__name__})"
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    logger.info("multiprocessing %r ready", method)
    return method


def use_double_precision_pickling() -> None:
    """Ensure RDKit 3D conformers cross process boundaries as float64 rather than float32."""
    Chem.SetDefaultPickleProperties(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    )


def _allocated_cpus() -> int:
    for var in ("PBS_NUM_PPN", "NCPUS"):
        raw = os.getenv(var)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    return os.cpu_count() or 1


def resolve_workers(n_jobs: int) -> int:
    """Resolve an explicit or auto (``<= 0``) worker count against PBS allocation and env caps."""
    return int(capped_n_jobs(n_jobs if n_jobs > 0 else _allocated_cpus()))


def workers(cfg: Config) -> int:
    """Effective multiprocessing worker count for ``cfg``."""
    return resolve_workers(cfg.n_jobs)


def map_tasks(fn: Callable[[_T], _R], tasks: list[_T], cfg: Config) -> list[_R]:
    """Map ``fn`` across a worker pool, failing fast if a healthy pool dies."""
    count = workers(cfg)
    if not pool_can_run(count, len(tasks), min_items=32):
        return [fn(task) for task in tasks]
    try:
        with pool_context().Pool(
            min(count, len(tasks)), initializer=use_double_precision_pickling
        ) as pool:
            return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (count * 4)))
    except Exception as error:
        logger.error("relaxation pool failed (%s); aborting", error)
        raise RuntimeError(f"relaxation pool failed: {error}") from error


def read_sdf(path: Path) -> list[Chem.Mol]:
    """Read all valid hydrogens-retained molecules from ``path``."""
    return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]


def write_sdf(path: Path, records: Iterable[Chem.Mol]) -> int:
    """Write ``records`` to ``path``, retrying once without kekulization; return dropped count."""
    unwritable = 0
    writer = Chem.SDWriter(str(path))
    try:
        for mol in records:
            try:
                writer.write(mol)
            except Exception:
                try:
                    writer.SetKekulize(False)
                    writer.write(mol)
                except Exception as error:
                    unwritable += 1
                    logger.warning("conformer could not be written to SDF: %s", error)
                finally:
                    writer.SetKekulize(True)
    finally:
        writer.close()
    return unwritable


def _embed_shard(task: tuple[int, list[str], dict[str, Any], str]) -> str:
    """Embed one SMILES shard in a worker process and rebase ``mol_<idx>`` names."""
    offset, smiles, kwargs, out_dir = task
    shard = Path(out_dir) / f"shard_{offset}"
    shard.mkdir(parents=True, exist_ok=True)
    produced = Path(RDKitConformerGenerator(**{**kwargs, "num_threads": 1}).genConformers(smiles, str(shard)))
    if not produced.is_file() or not produced.stat().st_size:
        return ""
    records = read_sdf(produced)
    for mol in records:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        head, _, isomer = name.partition("+")
        _, _, index = head.partition("_")
        if index.isdigit():
            mol.SetProp("_Name", f"mol_{offset + int(index)}" + (f"+{isomer}" if isomer else ""))
    write_sdf(produced, records)
    return str(produced)


def parallel_embed(
    generator: Any,
    smiles_list: list[str],
    out_dir: str | Path,
    *,
    native_kwargs: dict[str, Any],
    serial: Callable[[], str | Path],
) -> Path:
    """Shard ETKDGv3 embedding across workers while preserving input molecule indices."""
    count = workers(generator.cfg)
    if not pool_can_run(count, sum(1 for s in smiles_list if s), min_items=8):
        return Path(serial())

    size = math.ceil(len(smiles_list) / count)
    shards = [
        (offset, smiles_list[offset : offset + size], native_kwargs, str(out_dir))
        for offset in range(0, len(smiles_list), size)
    ]
    try:
        with pool_context().Pool(count) as pool:
            produced = pool.map(_embed_shard, shards)
    except Exception as error:
        logger.error("parallel embedding failed (%s); aborting", error)
        raise RuntimeError(f"parallel embedding failed: {error}") from error

    combined = Path(out_dir) / "conformers.sdf"
    write_sdf(combined, (mol for path in produced if path for mol in read_sdf(Path(path))))
    return combined
