"""Reference and candidate conformers: embedding, relaxation, stereo policy and the bank.

The chemistry lives here. Worker pools and SDF transfer live in
:mod:`_david.code.scorers.utils`, so this module can be read as "what a conformer is"
rather than "how it is moved between processes".
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

from _david.code.config import Config
from _david.code.scorers.utils import (
    map_tasks,
    parallel_embed,
    read_sdf,
    workers,
    write_sdf,
)

logger = logging.getLogger(__name__)

FORCE_FIELD = "MMFF94s"
RELAXATION_TALLY: Counter[str] = Counter()
PROTONATE = False
CANONICALIZE_TAUTOMER = True


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
    """Canonical SMILES after reading stereochemistry back out of 3D coordinates."""
    try:
        probe = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(probe)
        return Chem.MolToSmiles(Chem.RemoveHs(probe))
    except Exception:
        return None


def _smiles_stereo_key(smiles: str) -> str | None:
    """The stereo key a reference SMILES declares, for comparison against its conformers."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _relax(task: tuple[Chem.Mol, Config, bool]) -> tuple[Chem.Mol | None, str]:
    """Minimize one conformer and keep it unless strict stereo checking rejects the move."""
    record, cfg, strict_stereo = task
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
            # A reference must be verifiably the supplied stereoisomer; a candidate may
            # not have one yet, because the model does not generate stereochemistry.
            return (None, "stereo_uncheckable") if strict_stereo else (mol, "stereo_uncheckable")
        if before != after:
            if strict_stereo:
                return None, "stereo_changed"
            # Candidates keep the relaxed geometry: an unassigned centre was assigned by
            # embedding, not inverted by minimization, and the score is a maximum over
            # the isomers the SMILES could be.
            return mol, "stereo_changed"
        mol.SetDoubleProp("Energy", energy)
        mol.SetIntProp("MinimizeStatus", status)
        return mol, "converged" if status == 0 else "unconverged"
    except Exception as error:  # noqa: BLE001 - a per-conformer fault must stay local
        logger.debug("relaxation raised for one conformer: %s", error)
        return record, "relax_raised"


def _molecule_of(mol: Chem.Mol) -> str:
    return mol.GetProp("_Name").split("+")[0] if mol.HasProp("_Name") else "?"


def relax_sdf(path: Path, cfg: Config, *, strict_stereo: bool = False) -> list[Chem.Mol]:
    """Relax every record in place and tally why anything was dropped or kept unrelaxed."""
    records = read_sdf(path)
    if not records:
        return []

    embedded, names = len(records), {_molecule_of(mol) for mol in records}
    results = map_tasks(_relax, [(mol, cfg, strict_stereo) for mol in records], cfg)
    del records

    kept = [mol for mol, _ in results if mol is not None]
    reasons = Counter(reason for _, reason in results)
    unwritable = write_sdf(path, kept)
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


class RelaxingConformerGenerator(RDKitConformerGenerator):
    """ETKDGv3 conformers, relaxed with MMFF94s when ``cfg.minimize`` is set.

    ``strict_stereo`` is set for the reference bank only: a candidate SMILES may leave
    stereocentres unassigned and is scored as a maximum over its possible isomers, while a
    reference must remain exactly the stereoisomer the caller supplied.
    """

    def __init__(
        self,
        cfg: Config,
        max_conformers: int | None = None,
        timeout: int | None = None,
        strict_stereo: bool = False,
    ):
        super().__init__(
            max_conformers=cfg.max_conformers if max_conformers is None else max_conformers,
            max_isomers=cfg.max_isomers,
            max_heavy_atoms=cfg.max_heavy_atoms,
            max_rotatable_bonds=cfg.max_rotatable_bonds,
            num_threads=workers(cfg),
            timeout=cfg.timeout if timeout is None else timeout,
            protonate=PROTONATE,
            canonicalize_tautomer=CANONICALIZE_TAUTOMER,
            show_progress=False,
        )
        self.cfg = cfg
        self.strict_stereo = bool(strict_stereo)

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
        return parallel_embed(
            self,
            smiles_list,
            out_dir,
            native_kwargs=self._native_kwargs(),
            # The parent's own loop, not this override: calling self.genConformers here
            # would return to this method forever.
            serial=lambda: RDKitConformerGenerator.genConformers(self, smiles_list, out_dir),
        )

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
            relax_sdf(path, self.cfg, strict_stereo=self.strict_stereo)
        return str(path)


def _rmsd(probe: Chem.Mol, other: Chem.Mol) -> float:
    return rdMolAlign.GetBestRMS(Chem.Mol(probe), Chem.Mol(other))


def build_reference(cfg: Config, out_path: Path) -> list[Chem.Mol]:
    """The reference bank, by geometric diversity: energy only anchors and breaks ties.

    Every conformer is also required to describe exactly the stereoisomer declared by
    ``cfg.reference_smiles``. Embedding an unassigned centre can enumerate a different
    isomer, and a reference bank is the yardstick every candidate is scored against, so a
    silent stereoisomer swap would change every score in the run.
    """
    generator = RelaxingConformerGenerator(
        cfg, max_conformers=cfg.reference_attempts, timeout=0, strict_stereo=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        records = read_sdf(Path(generator.genConformers([cfg.reference_smiles], tmp)))
    if not records:
        raise RuntimeError(
            f"no conformers generated for the reference {cfg.reference_smiles!r}; check the "
            "SMILES and the heavy-atom / rotatable-bond caps"
        )

    expected = _smiles_stereo_key(cfg.reference_smiles)
    if expected is None:
        raise RuntimeError(f"reference_smiles does not parse: {cfg.reference_smiles!r}")
    retained = [mol for mol in records if _stereo_key(mol) == expected]
    if len(retained) != len(records):
        logger.warning(
            "reference embedding produced %d/%d conformers that are not %s; dropping them",
            len(records) - len(retained),
            len(records),
            expected,
        )
    if not retained:
        raise RuntimeError(
            f"none of the {len(records)} reference conformers preserved the stereochemistry "
            f"of {cfg.reference_smiles!r} ({expected})"
        )
    records = retained

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

    write_sdf(out_path, bank)
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
