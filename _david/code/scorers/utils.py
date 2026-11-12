"""Hardware-facing plumbing: worker pools, shard embedding and SDF transfer.

Nothing here decides a score. The chemistry -- force field, relaxation, stereo policy and
reference selection -- lives in :mod:`_david.code.scorers.conformers`; this module only
makes that work run on more than one core and moves the resulting records through files.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from rdkit import Chem

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

from _david.code.config import Config

logger = logging.getLogger(__name__)

_POOL_BROKEN = False


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


def workers(cfg: Config) -> int:
    return capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1))


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
