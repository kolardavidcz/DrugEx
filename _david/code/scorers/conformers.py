"""Force-field relaxation of 3D conformers, and the reference conformer bank.

This is the one extension the thesis adds to DrugEx scoring: the native conformer
generators embed with ETKDGv3 but never relax the result. Candidates and the reference go
through the same generator, so they are prepared identically -- anything else would have
the colour channel compare two different chemical species.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

from _david.code.config import Config

logger = logging.getLogger(__name__)

#: The only force field in this project. MMFF94s, the "static" variant: it planarises
#: delocalised trigonal nitrogen so that a single minimised structure matches the
#: time-averaged geometry crystallography observes, which is what this pipeline produces
#: -- we minimise to a local minimum and score it, never run dynamics. Halgren introduced
#: the variant for exactly this case ("MMFF94s option for energy minimization studies").
#: EGCG has no nitrogen, so the reference bank is bit-identical either way. Measured on
#: sulfanilamide: the *aniline* N planarises 131.7 deg -> 173.8 deg (0.031 A heavy-atom
#: RMSD) while the sulfonamide N does not move at all (135.4 deg under both). So the
#: variant reaches nitrogen conjugated into a ring, not sulfonamides -- an earlier note
#: here named the wrong nitrogen of the same molecule. Not a config knob -- the two arms are "relaxed" and "not relaxed",
#: never "relaxed under a different force field".
FORCE_FIELD = "MMFF94s"

#: Every relaxation outcome of the run, by reason. Three reasons drop a conformer in the
#: minimised arm only (``solver_raised``, ``non_finite_energy``, ``stereo_changed``,
#: ``unwritable``), so
#: the arm comparison is only as symmetric as these counts are small. Logged per batch and
#: totalled here, because "the two arms scored the same molecules to within X %" is a claim
#: the thesis has to be able to make with a number.
RELAXATION_TALLY: dict[str, int] = {}

#: pH reassignment stays off, as `_david/requirements.txt` records: Dimorphite-DL would
#: deprotonate EGCG's phenols and change which species the ROCS colour channel types. It
#: is set explicitly because the native generator defaults it ON, and Dimorphite-DL *is*
#: installed here, so leaving it unset would silently protonate every molecule.
PROTONATE = False

#: Native default, kept: one canonical tautomer per molecule, reference and candidates
#: alike. It runs before stereoisomer enumeration in DrugEx, which is load-bearing.
CANONICALIZE_TAUTOMER = True


def _touch(value):
    """A task a worker can run. Module level on purpose: forkserver and spawn workers
    import the function by qualified name, so a nested definition cannot be sent."""
    return value


_POOL_BROKEN = False


def pool_is_healthy() -> bool:
    """Whether the startup probe proved that production workers can return a result."""
    return not _POOL_BROKEN


def check_pool(timeout: float = 300.0) -> str:
    """Prove workers can return a result; a failed probe disables every production pool.

    Both scorer pools consult :func:`pool_is_healthy`. Recording a broken start method
    in the manifest but entering an unbounded ``pool.map`` afterwards would hang on the
    same fault the probe just found. Small batches were already serial; this makes every
    batch serial after a failed probe.

    Every parallel path here degrades *silently*: a pool that cannot start is caught,
    logged and rescored serially, so the run still finishes -- a day later than it should
    have. On 16 cores that is the difference between 20 hours and most of a week, with
    nothing in the output saying why.

    The failure is real and easy to hit. ``forkserver`` and ``spawn`` children re-import
    ``__main__`` (``multiprocessing.spawn.prepare``), so a pipeline started from a
    heredoc -- ``python - <<EOF`` -- has a ``__main__`` of ``<stdin>``, which does not
    exist on disk, and *every* worker dies with ``FileNotFoundError``. ``python -m
    _david.code.run`` has an importable main module and is safe; it is how ``job.py`` starts
    the run on the cluster, where ``DRUGEX_MP_CONTEXT=forkserver`` is set.

    The timeout is patience, not a budget: a healthy probe returns as soon as the workers
    answer, so a generous bound costs a healthy run nothing. It has to be generous because
    a forkserver child re-imports the main module, and on the cluster the venv is on NFS:
    importing torch, rdkit and drugex measured 14.4 s on the frontend with a warm-ish
    cache, and the first 20 s-bounded probe on a compute node timed out and killed 148
    queued jobs in twenty seconds each. Five minutes of patience against a 24 h walltime.
    """
    global _POOL_BROKEN
    _POOL_BROKEN = False
    context = pool_context()
    method = context.get_start_method()
    pool = None
    try:
        pool = context.Pool(2)
        # map_async(...).get(timeout), never map(): the real failure mode is not an
        # exception. A worker that dies while importing __main__ is *replaced* by the
        # pool, forever, so a plain map() never returns -- the probe would hang the run
        # it exists to protect. Measured: a broken forkserver spins until killed.
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
            # terminate, not close: close() waits for tasks that a wedged pool will
            # never finish.
            pool.terminate()
            pool.join()
    logger.info("multiprocessing start method %r starts workers", method)
    return method


def use_double_precision_pickling() -> None:
    """Pickle conformer coordinates as doubles. Call this before building any scorer.

    Not cosmetic. RDKit pickles coordinates as float32 by default, and the ROCS scorer
    ships reference molecules to its workers by pickling them, so the parallel and
    sequential paths would see slightly different geometries and return different scores
    for the same input.
    """
    Chem.SetDefaultPickleProperties(
        Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
    )


def _force_field(mol: Chem.Mol, cfg: Config):
    """An MMFF94 force field for ``mol``, or None when MMFF has no parameters for it.

    There is deliberately no fallback to another force field: relaxing part of a library
    under different physics would make two molecules' scores incomparable for a reason
    nothing in the output records.
    """
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=FORCE_FIELD)
    props.SetMMFFDielectricConstant(cfg.dielectric)
    props.SetMMFFDielectricModel(cfg.dielectric_model)
    return AllChem.MMFFGetMoleculeForceField(mol, props)


def single_point_energy(mol: Chem.Mol, cfg: Config) -> float:
    """Force-field energy of ``mol`` as it stands, without moving an atom.

    Infinity when MMFF cannot parameterize the molecule, so an unscorable conformer sorts
    last instead of sorting first on a fabricated 0.0 kcal/mol. This ranks the reference
    bank and runs whether or not ``minimize`` is set, so the relaxed and unrelaxed arms
    differ in exactly one respect rather than also in how their bank was ordered.
    """
    field = _force_field(mol, cfg)
    if field is None:
        return math.inf
    try:
        energy = float(field.CalcEnergy())
    except RuntimeError:
        return math.inf
    return energy if math.isfinite(energy) else math.inf


def _stereo_key(mol: Chem.Mol) -> str | None:
    """Canonical SMILES with stereochemistry read off the 3D coordinates, or None.

    None when RDKit cannot produce one -- a generated molecule can carry an aromatic
    perception that ``RemoveHs`` refuses to kekulize. That is a fact about this molecule,
    not a reason to lose it, and certainly not a reason to end the run.
    """
    try:
        probe = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(probe)
        return Chem.MolToSmiles(Chem.RemoveHs(probe))
    except Exception:
        return None


def _relax(args: tuple[Chem.Mol, Config]) -> tuple[Chem.Mol | None, str]:
    """Relax one conformer. Returns the result (or None) and a reason, for the counters.

    Nothing in here may raise. This runs once per conformer per epoch inside a pool, so an
    escaping exception does not cost a conformer -- it ends a run that may be twenty hours
    old. Every failure becomes a reason in the counters instead.
    """
    try:
        return _relax_one(*args)
    except Exception as error:  # noqa: BLE001 - a per-conformer fault must stay local
        logger.debug("relaxation raised for one conformer: %s", error)
        return args[0], "relax_raised"


def _relax_one(record: Chem.Mol, cfg: Config) -> tuple[Chem.Mol | None, str]:
    mol = Chem.Mol(record)
    field = _force_field(mol, cfg)
    if field is None:
        # Keeping the embedded geometry is right: dropping it would turn "MMFF does not
        # cover this molecule" into a zero shape score, which reads as a bad match.
        return record, "unparameterized"
    try:
        status = int(field.Minimize(maxIts=cfg.max_iterations))
        energy = float(field.CalcEnergy())
    except RuntimeError:
        # RDKit's BFGS raises "bad direction in linearSearch" on pathological geometries.
        # That is a property of this one conformer, not of the run.
        return None, "solver_raised"
    if not math.isfinite(energy):
        return None, "non_finite_energy"
    before, after = _stereo_key(record), _stereo_key(mol)
    if before is None or after is None:
        # Unverifiable, not wrong. Dropping here would filter on "RDKit could not
        # kekulize this", which is a chemistry-dependent filter applied to the minimized
        # arm only -- precisely the confound the relaxed/unrelaxed comparison must avoid.
        return mol, "stereo_uncheckable"
    if before != after:
        return None, "stereo_changed"
    # status 1 means "not converged within max_iterations". The geometry is still relaxed
    # and usable, so it is kept: dropping it would bias the library toward easy molecules.
    mol.SetDoubleProp("Energy", energy)
    mol.SetIntProp("MinimizeStatus", status)
    return mol, "converged" if status == 0 else "unconverged"


def _map(fn, tasks: list, cfg: Config) -> list:
    """``map`` over a process pool, serially where a pool is unavailable or not worth it."""
    workers = capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1))
    if (
        len(tasks) < 32
        or workers <= 1
        or multiprocessing.current_process().daemon
        or not pool_is_healthy()
    ):
        return [fn(task) for task in tasks]
    try:
        # DrugEx's default start method (fork) on purpose: forkserver avoids the
        # fork-after-threads deadlock upstream saw at n_jobs=62 but costs seconds to
        # bootstrap, and this pool is rebuilt every epoch. Set DRUGEX_MP_CONTEXT=forkserver
        # on the many-core hosts where that deadlock is real.
        with pool_context().Pool(
            min(workers, len(tasks)), initializer=use_double_precision_pickling
        ) as pool:
            return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
    except Exception as exc:
        logger.warning("relaxation pool failed (%s); continuing serially", exc)
        return [fn(task) for task in tasks]


def _read(path: Path) -> list[Chem.Mol]:
    """Every conformer in an SDF, hydrogens kept."""
    return [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]


def _write(path: Path, records: Iterable[Chem.Mol]) -> int:
    """Write conformers to an SDF. Returns how many could not be written at all.

    A generated molecule can carry an aromatic perception RDKit refuses to kekulize, and
    the default writer raises on it. The SDF is how conformers reach the scorer, so losing
    the molecule here would filter the minimized arm on "RDKit could not kekulize this" --
    a chemistry-dependent confound in the one comparison this extension exists to make.
    Falling back to non-kekulized output (aromatic bonds as type 4, which RDKit reads
    back) keeps the molecule instead.
    """
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
    """The ``mol_<i>`` part of a conformer's title, i.e. which molecule it belongs to.

    DrugEx maps conformers back to input molecules through this name, so it has to survive
    the rewrite; ``Chem.Mol(record)`` copies it along with every other property.
    """
    return mol.GetProp("_Name").split("+")[0] if mol.HasProp("_Name") else "?"


def relax_sdf(path: Path, cfg: Config) -> list[Chem.Mol]:
    """Relax every conformer in an SDF in place, and report what was lost.

    A molecule that loses all of its conformers scores 0.0, which is indistinguishable
    from a genuinely poor shape match, so the loss is counted and logged rather than left
    to look like a result.
    """
    records = _read(path)
    if not records:
        return []

    # _relax_one minimises a copy, so once the results are back the originals are a
    # second full copy of the batch -- 5 GB at a 5000-molecule arm -- held only for a
    # count and a set of names. Take those two facts off them and let them go.
    n_records = len(records)
    names = {_molecule_of(mol) for mol in records}
    results = _map(_relax, [(mol, cfg) for mol in records], cfg)
    del records
    kept = [mol for mol, _ in results if mol is not None]
    reasons: dict[str, int] = {}
    for _, reason in results:
        reasons[reason] = reasons.get(reason, 0) + 1
        RELAXATION_TALLY[reason] = RELAXATION_TALLY.get(reason, 0) + 1
    unwritable = _write(path, kept)
    if unwritable:
        # Another minimise-only loss: the unminimised arm never rewrites its SDF. It has
        # to reach the tally, which is the evidence for "both arms scored the same
        # molecules".
        reasons["unwritable"] = reasons.get("unwritable", 0) + unwritable
        RELAXATION_TALLY["unwritable"] = RELAXATION_TALLY.get("unwritable", 0) + unwritable

    if len(kept) != n_records:
        orphaned = names - {_molecule_of(m) for m in kept}
        logger.warning(
            "relaxation dropped %d/%d conformers (%s); %d molecules left with none",
            n_records - len(kept),
            n_records,
            ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())),
            len(orphaned),
        )
    elif reasons.get("unparameterized"):
        logger.warning(
            "%d/%d conformers had no %s parameters and were left unrelaxed",
            reasons["unparameterized"],
            n_records,
            FORCE_FIELD,
        )
    return kept


def _embed_shard(args: tuple[int, list[str], dict, str]) -> str:
    """Embed one slice of the batch with a plain native generator, in a worker process.

    ``offset`` is where this slice began in the caller's list. Native titles conformers
    ``mol_<index>`` and the scorer maps them back to input molecules by that index, so a
    worker that numbered from zero would attribute every molecule's conformers to the
    wrong SMILES. The index is rewritten to the global one here.

    ``num_threads=1``: parallelism is across molecules now, so letting each worker thread
    internally as well would oversubscribe the allocation.

    A path is returned rather than the molecules themselves. RDKit pickles properties only
    when ``SetDefaultPickleProperties`` has been called, so returning molecules would make
    the conformer titles -- and with them every molecule's identity -- depend on a global
    set in another module. Writing an SDF and handing back its path cannot lose them.
    """
    offset, smiles, kwargs, out_dir = args
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
    """ETKDGv3 conformers, relaxed with MMFF94 when ``cfg.minimize`` is set.

    The embedding, protonation, tautomer canonicalization, stereoisomer enumeration and
    the heavy-atom / rotatable-bond filters are all the native generator's. Only the
    relaxation step is added, and it is added where every caller gets it: on the SDF that
    ``genConformers`` hands back to the scorer.
    """

    def __init__(self, cfg: Config, max_conformers: int | None = None, timeout: int | None = None):
        super().__init__(
            max_conformers=cfg.max_conformers if max_conformers is None else max_conformers,
            max_isomers=cfg.max_isomers,
            max_heavy_atoms=cfg.max_heavy_atoms,
            max_rotatable_bonds=cfg.max_rotatable_bonds,
            # Native calls genConformers in the MAIN process, before it creates any pool
            # (rocs_rdkit.py:762 vs :818), and relies on RDKit's own threading inside
            # EmbedMultipleConfs. Pinning this to 1 put every embedding of the run on one
            # core while the rest of the allocation idled. Cap it at the allocation
            # instead: 0 would mean "every core on a shared node", which overruns PBS.
            num_threads=capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1)),
            timeout=cfg.timeout if timeout is None else timeout,
            protonate=PROTONATE,
            canonicalize_tautomer=CANONICALIZE_TAUTOMER,
            show_progress=False,
        )
        self.cfg = cfg

    def _native_kwargs(self) -> dict:
        """The constructor arguments a worker needs to rebuild an equivalent generator."""
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
        """Embed the batch, across molecules, instead of one molecule at a time.

        Native loops over molecules in Python and only threads *inside*
        ``EmbedMultipleConfs``, so a batch of 500 molecules spends most of its time
        single-core: tautomer canonicalization, stereoisomer enumeration, AddHs and SDF
        writing are all serial, and embedding 20 conformers cannot occupy 32 cores. The
        molecules are independent and ETKDG is seeded per call, so sharding them across
        processes produces the same conformers -- there is a test asserting exactly that.
        """
        workers = capped_n_jobs(self.cfg.n_jobs if self.cfg.n_jobs > 0 else (os.cpu_count() or 1))
        wanted = [s for s in smiles_list if s]
        # A flat floor, not a multiple of the worker count: at 32 cores the latter would
        # send every 50-molecule evaluation arm down the serial path, once per epoch.
        # Below this, forkserver startup costs more than the embedding saves.
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

        # A generator, not a list. Native streams each conformer to one SDWriter as it
        # is produced; collecting every shard first undid that, and a 5000-molecule
        # batch is ~85,000 conformer records at ~65 kB resident apiece. `_write`
        # consumes whatever it is given one molecule at a time, so this holds one
        # shard instead of the whole batch and writes the identical SDF.
        combined = Path(out_dir) / "conformers.sdf"
        _write(combined, (mol for path in produced if path for mol in _read(Path(path))))
        return combined

    def genConformers(self, smiles_list, out_dir) -> str:
        path = self._embed(smiles_list, out_dir)
        # Native returns an all-zero score matrix when the SDF is missing or empty, and
        # says so only under show_progress, which is off. The reward for that batch would
        # then be SA alone and the arm would read as a legitimately poor shape match. The
        # realistic cause is scratch or TMPDIR pressure, which nothing else checks.
        wanted = [s for s in smiles_list if s]
        if wanted and (not path.is_file() or not path.stat().st_size):
            logger.warning(
                "conformer generation produced no output for %d molecule(s); every shape "
                "score in this batch will be 0.0. Check scratch/TMPDIR space.",
                len(wanted),
            )
        if self.cfg.minimize and path.is_file() and path.stat().st_size:
            relax_sdf(path, self.cfg)
        return str(path)


def build_reference(cfg: Config, out_path: Path) -> list[Chem.Mol]:
    """Generate the reference conformer bank from ``cfg.reference_smiles``.

    Embed ``reference_attempts`` conformers through the same generator the candidates use,
    then pick ``reference_conformers`` of them by *geometric diversity*
    rather than by energy rank. The bank is one ROCS group, so a candidate's score is its
    best overlay against any conformer in it -- which is exactly why the bank should span
    the accessible shapes instead of clustering in the lowest-energy basin.

    Energy anchors and orders; it never filters, and that is deliberate. Taking the N
    lowest-energy conformers is the worst of the available strategies: on the Platinum set
    it retained a pose within 1 A of the crystal conformation for 57.6 % of ligands, versus
    67.8 % for unbiased sampling and 72.7 % for RMSD-diversity selection, and it did not
    beat unbiased sampling for any N > 1. Force-field energies rank conformers poorly in
    the first place (MMFF94 vs QM: median Spearman -0.10), and bound ligands sit 4-6
    kcal/mol above the global minimum about half the time. See ``DECISIONS.md``.
    """
    # timeout=0 (native: "no timeout"): cfg.timeout is a per-isomer wall-clock cap meant
    # to bound the ~10 % of *generated* molecules that will not embed cleanly. The
    # reference is one known molecule embedded once per run, and the work is already
    # bounded by reference_attempts. Leaving the cap on makes the bank depend on how busy
    # the machine was: a slow node truncates the embedding and the run then measures every
    # candidate against a distorted EGCG, silently, and still finishes "complete".
    generator = RelaxingConformerGenerator(cfg, max_conformers=cfg.reference_attempts, timeout=0)
    with tempfile.TemporaryDirectory() as tmp:
        records = _read(Path(generator.genConformers([cfg.reference_smiles], tmp)))

    if not records:
        raise RuntimeError(
            f"no conformers generated for the reference {cfg.reference_smiles!r}; check the "
            "SMILES and the heavy-atom / rotatable-bond caps"
        )
    # A bank smaller than requested is legitimate for a rigid molecule (ethanol has one
    # distinct conformer however many times it is embedded) and a red flag for a flexible
    # one, so it warns rather than raising -- but it must never pass unremarked: a bank
    # built from a truncated embedding scores every candidate for the rest of the run.
    # The real cure is upstream: the reference embeds with timeout=0, so a slow machine
    # can no longer truncate it.
    if len(records) < cfg.reference_conformers:
        logger.warning(
            "reference embedding produced only %d distinct conformers of the %d attempted, "
            "fewer than the %d the bank asks for. Legitimate for a rigid molecule; for a "
            "flexible reference it means the embedding did not explore the space.",
            len(records),
            cfg.reference_attempts,
            cfg.reference_conformers,
        )

    ranked = sorted(
        ((single_point_energy(mol, cfg), mol) for mol in records), key=lambda pair: pair[0]
    )
    floor = ranked[0][0]
    if not math.isfinite(floor):
        raise RuntimeError("no reference conformer could be assigned a finite energy")
    # No energy filter at all. An absolute kcal/mol window meant different things in the
    # two arms (it admitted 2 unminimised poses against 4 minimised, a bank-size confound
    # in the one comparison this extension exists to make), and a rank-based pool was
    # measured to be inert at the shipped settings: EGCG yields only ~15 distinct
    # conformers from 200 attempts after ETKDG's own 0.5 A pruning, so any pool above that
    # filtered nothing while config.json recorded it as active. Diversity selection over
    # everything embedded is also the better-supported rule -- energy-ranked selection
    # recovered a crystal pose for 57.6 % of ligands against 72.7 % for RMSD diversity.
    # Energy still anchors the bank at the global minimum and breaks ties.
    pool = ranked

    # Farthest-point selection over the window. The global minimum anchors the bank, then
    # each further pose is the one whose closest already-kept neighbour is furthest away,
    # so the bank spans the accessible shapes instead of crowding one basin. RMSD is on
    # heavy atoms: hydrogen positions add noise, not shape.
    probes = [Chem.RemoveHs(Chem.Mol(mol)) for _, mol in pool]
    kept_index = [0]
    # GetBestRMS aligns the probe it is given, so every argument is a copy.
    spread = [rdMolAlign.GetBestRMS(Chem.Mol(probe), Chem.Mol(probes[0])) for probe in probes]
    while len(kept_index) < cfg.reference_conformers:
        candidate = max(
            (i for i in range(len(pool)) if i not in kept_index),
            key=lambda i: spread[i],
            default=None,
        )
        if candidate is None or spread[candidate] <= cfg.reference_rmsd:
            break  # nothing left that is a genuinely different pose
        kept_index.append(candidate)
        for i in range(len(pool)):
            distance = rdMolAlign.GetBestRMS(Chem.Mol(probes[i]), Chem.Mol(probes[candidate]))
            spread[i] = min(spread[i], distance)

    kept = []
    for i in sorted(kept_index, key=lambda i: pool[i][0]):
        energy, mol = pool[i]
        mol.SetDoubleProp("Energy", energy)
        kept.append(mol)

    _write(out_path, kept)
    logger.info(
        "reference bank: %d embedded, %d candidates%s, %d kept by "
        "diversity at RMSD > %.2f A, energies %s, stereoisomers %s",
        len(records),
        len(pool),
        f" lowest-energy (span {pool[-1][0] - pool[0][0]:.1f} kcal/mol)" if len(pool) > 1 else "",
        len(kept),
        cfg.reference_rmsd,
        [round(m.GetDoubleProp("Energy"), 2) for m in kept],
        sorted({_stereo_key(m) for m in kept}),
    )
    return kept
