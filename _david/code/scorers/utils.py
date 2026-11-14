"""Hardware-facing plumbing: worker pools, shard embedding and SDF transfer.

Nothing here decides a score. The chemistry -- force field, relaxation, stereo policy and
reference selection -- lives in :mod:`_david.code.scorers.conformers`; this module only
makes that work run on more than one core and moves the resulting records through files.
"""

from __future__ import annotations

import gc
import logging
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
from rdkit import Chem

from _david.code.config import Config
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

logger = logging.getLogger(__name__)

_POOL_BROKEN = False

#: Molecules per scoring call. The scorers embed every conformer of the batch in the parent
#: process before the pool runs, so a job's peak follows ``len(smiles) x max_conformers``
#: and barely moves with the pool size (measured: 16/32/45 GB at 10/20/30 conformers for a
#: 5000-molecule arm, and six jobs that all hit exactly 32 GB at 13 and at 16 workers). A
#: 5000-molecule arm at 20 conformers therefore needs ~32 GB however many workers it has,
#: which is what killed 23 jobs at their cgroup. Slicing bounds the parent to one chunk:
#: 500 x 20 conformers is ~3 GB, so the same arm finishes inside 32 GB.
SCORE_CHUNK = 500


def components(env, smiles: list[str]) -> dict[str, list[float]]:
    """Per-molecule shape/colour of the winning pose, when the environment records them."""
    for scorer in getattr(env, "scorers", []):
        if hasattr(scorer, "component_frame"):
            return scorer.component_frame(smiles)
    return {}


def score_in_chunks(env, smiles: list[str], *, chunk: int = SCORE_CHUNK):
    """Score ``smiles`` in slices; the scores are the same, the peak is the slice's.

    Returns the concatenated frame in the caller's order and the merged shape/colour
    diagnostics. Slicing does not change a score: every molecule is embedded in its own
    ``EmbedMultipleConfs`` call with a fixed ``params.randomSeed``, so its conformers do not
    depend on where in the batch it sat. ``Tests`` pins that on the real scorer.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    frames, diagnostics = [], {}
    for start in range(0, len(smiles), chunk):
        piece = smiles[start : start + chunk]
        frames.append(env.getScores(piece).reset_index(drop=True))
        for column, values in components(env, piece).items():
            diagnostics.setdefault(column, []).extend(values)
        gc.collect()
    if not frames:
        return env.getScores(smiles), {}
    return pd.concat(frames, ignore_index=True), diagnostics


def pool_is_healthy() -> bool:
    """Whether the last :func:`check_pool` probe found a working worker pool."""
    return not _POOL_BROKEN


def _touch(value):
    return value


def check_pool(timeout: float = 300.0) -> str:
    """Start two workers once, before the expensive stages, and never raise on failure."""
    global _POOL_BROKEN
    _POOL_BROKEN = False
    context = pool_context()
    method = context.get_start_method()
    pool = None
    try:
        pool = context.Pool(2)
        pool.map_async(_touch, (1, 2)).get(timeout=timeout)
    except Exception as error:  # noqa: BLE001 - any failure here means "no workers"
        _POOL_BROKEN = True
        logger.error(
            "multiprocessing start method %r cannot start a worker here (%s: %s). Every "
            "scoring stage will fall back to a single core. Start the pipeline as "
            "`python -m _david.code.run` (a heredoc has no importable __main__) or set "
            "DRUGEX_MP_CONTEXT=fork.",
            method,
            type(error).__name__,
            error,
        )
        return f"{method} (broken: {type(error).__name__})"
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    logger.info("multiprocessing start method %r starts workers", method)
    return method


def use_double_precision_pickling() -> None:
    """Coordinates cross a process boundary as doubles, not as the default float32."""
    Chem.SetDefaultPickleProperties(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    )


def _allocated_cpus() -> int:
    for var in ("PBS_NUM_PPN", "NCPUS"):
        raw = os.getenv(var)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    return os.cpu_count() or 1


def workers(cfg: Config) -> int:
    # OMP_NUM_THREADS is pinned to 1 in job.py to stop BLAS thread oversubscription inside
    # each worker; the worker pool size comes from cfg.n_jobs (or PBS_NUM_PPN/NCPUS).
    return capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else _allocated_cpus())


def map_tasks(fn, tasks: list, cfg: Config) -> list:
    """Map a per-record function over a pool, falling back to this process."""
    count = workers(cfg)
    if (
        len(tasks) < 32
        or count <= 1
        or multiprocessing.current_process().daemon
        or not pool_is_healthy()
    ):
        return [fn(task) for task in tasks]
    try:
        with pool_context().Pool(
            min(count, len(tasks)), initializer=use_double_precision_pickling
        ) as pool:
            return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (count * 4)))
    except Exception as error:
        logger.warning("relaxation pool failed (%s); continuing serially", error)
        return [fn(task) for task in tasks]


def read_sdf(path: Path) -> list[Chem.Mol]:
    return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]


def write_sdf(path: Path, records: Iterable[Chem.Mol]) -> int:
    """Write every record, retrying once without kekulization; returns the unwritable count."""
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


def _molecule_of(mol: Chem.Mol) -> str:
    return mol.GetProp("_Name").split("+")[0] if mol.HasProp("_Name") else "?"


def _embed_shard(task: tuple[int, list[str], dict, str]) -> str:
    """Embed one contiguous slice of the input in a fresh child process."""
    offset, smiles, kwargs, out_dir = task
    shard = Path(out_dir) / f"shard_{offset}"
    shard.mkdir(parents=True, exist_ok=True)
    generator = RDKitConformerGenerator(**{**kwargs, "num_threads": 1})
    produced = Path(generator.genConformers(smiles, str(shard)))
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
    generator: RDKitConformerGenerator,
    smiles_list: list[str],
    out_dir: str | Path,
    *,
    native_kwargs: dict,
    serial: Callable[[], str | Path],
) -> Path:
    """Split the embedding across workers, keeping the input indices stable.

    ``serial`` is the parent class's own embedding, called whenever a pool cannot help.
    """
    count = workers(generator.cfg)
    wanted = [smiles for smiles in smiles_list if smiles]
    if (
        count <= 1
        or len(wanted) < 8
        or multiprocessing.current_process().daemon
        or not pool_is_healthy()
    ):
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
        logger.warning("parallel embedding failed (%s); falling back to serial", error)
        return Path(serial())

    combined = Path(out_dir) / "conformers.sdf"
    write_sdf(combined, (mol for path in produced if path for mol in read_sdf(Path(path))))
    return combined
