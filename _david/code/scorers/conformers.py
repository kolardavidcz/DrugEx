from __future__ import annotations

import logging
import math
import multiprocessing
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

from _david.code.config import Config

logger = logging.getLogger(__name__)

FORCE_FIELD = "MMFF94s"
RELAXATION_TALLY: Counter[str] = Counter()
PROTONATE = False
CANONICALIZE_TAUTOMER = True

_POOL_BROKEN = False


def pool_is_healthy() -> bool:
    return not _POOL_BROKEN


def _touch(value):
    return value


def check_pool(timeout: float = 300.0) -> str:
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
    Chem.SetDefaultPickleProperties(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    )


def _workers(cfg: Config) -> int:
    return capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1))


def _force_field(mol: Chem.Mol, cfg: Config):
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=FORCE_FIELD)
    props.SetMMFFDielectricConstant(cfg.dielectric)
    props.SetMMFFDielectricModel(cfg.dielectric_model)
    return AllChem.MMFFGetMoleculeForceField(mol, props)


def single_point_energy(mol: Chem.Mol, cfg: Config) -> float:
    field = _force_field(mol, cfg)
    if field is None:
        return math.inf
    try:
        energy = float(field.CalcEnergy())
    except RuntimeError:
        return math.inf
    return energy if math.isfinite(energy) else math.inf


def _stereo_key(mol: Chem.Mol) -> str | None:
    try:
        probe = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(probe)
        return Chem.MolToSmiles(Chem.RemoveHs(probe))
    except Exception:
        return None


def _relax(task: tuple[Chem.Mol, Config]) -> tuple[Chem.Mol | None, str]:
    record, cfg = task
    try:
        mol = Chem.Mol(record)
        field = _force_field(mol, cfg)
        if field is None:
            return record, "unparameterized"
        try:
            status = int(field.Minimize(maxIts=cfg.max_iterations))
            energy = float(field.CalcEnergy())
        except RuntimeError:
            return None, "solver_raised"
        if not math.isfinite(energy):
            return None, "non_finite_energy"
        before, after = _stereo_key(record), _stereo_key(mol)
        if before is None or after is None:
            return mol, "stereo_uncheckable"
        if before != after:
            #! my probably should we really keep it if we have different geometry
            return None, "stereo_changed"
        mol.SetDoubleProp("Energy", energy)
        mol.SetIntProp("MinimizeStatus", status)
        return mol, "converged" if status == 0 else "unconverged"
    except Exception as error:  # noqa: BLE001 - a per-conformer fault must stay local
        logger.debug("relaxation raised for one conformer: %s", error)
        return record, "relax_raised"


def _map(fn, tasks: list, cfg: Config) -> list:
    workers = _workers(cfg)
    if (
        len(tasks) < 32
        or workers <= 1
        or multiprocessing.current_process().daemon
        or not pool_is_healthy()
    ):
        return [fn(task) for task in tasks]
    try:
        with pool_context().Pool(
            min(workers, len(tasks)), initializer=use_double_precision_pickling
        ) as pool:
            return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
    except Exception as error:
        logger.warning("relaxation pool failed (%s); continuing serially", error)
        return [fn(task) for task in tasks]


def _read(path: Path) -> list[Chem.Mol]:
    return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]


def _write(path: Path, records: Iterable[Chem.Mol]) -> int:
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


def relax_sdf(path: Path, cfg: Config) -> list[Chem.Mol]:
    records = _read(path)
    if not records:
        return []

    embedded, names = len(records), {_molecule_of(mol) for mol in records}
    results = _map(_relax, [(mol, cfg) for mol in records], cfg)
    del records

    kept = [mol for mol, _ in results if mol is not None]
    reasons = Counter(reason for _, reason in results)
    unwritable = _write(path, kept)
    if unwritable:
        reasons["unwritable"] += unwritable
    RELAXATION_TALLY.update(reasons)

    if len(kept) != embedded:
        orphaned = names - {_molecule_of(mol) for mol in kept}
        logger.warning(
            "relaxation dropped %d/%d conformers (%s); %d molecules left with none",
            embedded - len(kept),
            embedded,
            ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())),
            len(orphaned),
        )
    elif reasons["unparameterized"]:
        logger.warning(
            "%d/%d conformers had no %s parameters and were left unrelaxed",
            reasons["unparameterized"],
            embedded,
            FORCE_FIELD,
        )
    return kept


def _embed_shard(task: tuple[int, list[str], dict, str]) -> str:
    offset, smiles, kwargs, out_dir = task
    shard = Path(out_dir) / f"shard_{offset}"
    shard.mkdir(parents=True, exist_ok=True)
    generator = RDKitConformerGenerator(**{**kwargs, "num_threads": 1})
    produced = Path(generator.genConformers(smiles, str(shard)))
    if not produced.is_file() or not produced.stat().st_size:
        return ""
    records = _read(produced)
    for mol in records:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        head, _, isomer = name.partition("+")
        _, _, index = head.partition("_")
        if index.isdigit():
            mol.SetProp("_Name", f"mol_{offset + int(index)}" + (f"+{isomer}" if isomer else ""))
    _write(produced, records)
    return str(produced)


class RelaxingConformerGenerator(RDKitConformerGenerator):
    """ETKDGv3 conformers, relaxed with MMFF94s when ``cfg.minimize`` is set."""

    def __init__(self, cfg: Config, max_conformers: int | None = None, timeout: int | None = None):
        super().__init__(
            max_conformers=cfg.max_conformers if max_conformers is None else max_conformers,
            max_isomers=cfg.max_isomers,
            max_heavy_atoms=cfg.max_heavy_atoms,
            max_rotatable_bonds=cfg.max_rotatable_bonds,
            num_threads=_workers(cfg),
            timeout=cfg.timeout if timeout is None else timeout,
            protonate=PROTONATE,
            canonicalize_tautomer=CANONICALIZE_TAUTOMER,
            show_progress=False,
        )
        self.cfg = cfg

    def _native_kwargs(self) -> dict:
        return {
            "max_conformers": self.max_conformers,
            "max_isomers": self.max_isomers,
            "max_heavy_atoms": self.max_heavy_atoms,
            "max_rotatable_bonds": self.max_rotatable_bonds,
            "timeout": self.timeout,
            "protonate": self.protonate,
            "canonicalize_tautomer": self.canonicalize_tautomer,
            "show_progress": False,
        }

    def _embed(self, smiles_list, out_dir) -> Path:
        """Embed across molecules, not one at a time."""
        workers = _workers(self.cfg)
        wanted = [smiles for smiles in smiles_list if smiles]
        if (
            workers <= 1
            or len(wanted) < 8
            or multiprocessing.current_process().daemon
            or not pool_is_healthy()
        ):
            return Path(super().genConformers(smiles_list, out_dir))

        size = math.ceil(len(smiles_list) / workers)
        shards = [
            (offset, smiles_list[offset : offset + size], self._native_kwargs(), str(out_dir))
            for offset in range(0, len(smiles_list), size)
        ]
        try:
            with pool_context().Pool(workers) as pool:
                produced = pool.map(_embed_shard, shards)
        except Exception as error:
            logger.warning("parallel embedding failed (%s); falling back to serial", error)
            return Path(super().genConformers(smiles_list, out_dir))

        combined = Path(out_dir) / "conformers.sdf"
        _write(combined, (mol for path in produced if path for mol in _read(Path(path))))
        return combined

    def genConformers(self, smiles_list, out_dir) -> str:
        path = self._embed(smiles_list, out_dir)
        written = path.is_file() and path.stat().st_size
        if not written and any(smiles_list):
            logger.warning(
                "conformer generation produced no output for %d molecule(s); every shape "
                "score in this batch will be 0.0. Check scratch/TMPDIR space.",
                sum(1 for smiles in smiles_list if smiles),
            )
        if written and self.cfg.minimize:
            relax_sdf(path, self.cfg)
        return str(path)


def _rmsd(probe: Chem.Mol, other: Chem.Mol) -> float:
    return rdMolAlign.GetBestRMS(Chem.Mol(probe), Chem.Mol(other))


def build_reference(cfg: Config, out_path: Path) -> list[Chem.Mol]:
    """The reference bank, by geometric diversity: energy only anchors and breaks ties."""
    generator = RelaxingConformerGenerator(cfg, max_conformers=cfg.reference_attempts, timeout=0)
    with tempfile.TemporaryDirectory() as tmp:
        records = _read(Path(generator.genConformers([cfg.reference_smiles], tmp)))
    if not records:
        raise RuntimeError(
            f"no conformers generated for the reference {cfg.reference_smiles!r}; check the "
            "SMILES and the heavy-atom / rotatable-bond caps"
        )
    if len(records) < cfg.reference_conformers:
        logger.warning(
            "reference embedding produced only %d distinct conformers of the %d attempted, "
            "fewer than the %d the bank asks for. Legitimate for a rigid molecule; for a "
            "flexible reference it means the embedding did not explore the space.",
            len(records),
            cfg.reference_attempts,
            cfg.reference_conformers,
        )

    ranked = sorted(((single_point_energy(mol, cfg), mol) for mol in records), key=lambda r: r[0])
    if not math.isfinite(ranked[0][0]):
        raise RuntimeError("no reference conformer could be assigned a finite energy")

    probes = [Chem.RemoveHs(Chem.Mol(mol)) for _, mol in ranked]
    spread = [_rmsd(probe, probes[0]) for probe in probes]
    kept = {0}
    while len(kept) < cfg.reference_conformers:
        candidate = max(
            (index for index in range(len(ranked)) if index not in kept),
            key=lambda index: spread[index],
            default=None,
        )
        if candidate is None or spread[candidate] <= cfg.reference_rmsd:
            break
        kept.add(candidate)
        spread = [min(value, _rmsd(probes[i], probes[candidate])) for i, value in enumerate(spread)]

    bank = []
    for index in sorted(kept, key=lambda index: ranked[index][0]):
        energy, mol = ranked[index]
        mol.SetDoubleProp("Energy", energy)
        bank.append(mol)

    _write(out_path, bank)
    scored = [energy for energy, _ in ranked if math.isfinite(energy)]
    logger.info(
        "reference bank: %d distinct conformers embedded spanning %.1f kcal/mol, %d kept by "
        "diversity at RMSD > %.2f A, energies %s, stereoisomers %s",
        len(ranked),
        scored[-1] - scored[0],
        len(bank),
        cfg.reference_rmsd,
        [round(mol.GetDoubleProp("Energy"), 2) for mol in bank],
        sorted({_stereo_key(mol) for mol in bank}),
    )
    return bank
