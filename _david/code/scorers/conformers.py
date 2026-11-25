"""Conformer chemistry: ETKDGv3 embedding, MMFF94s relaxation, stereo policy, and reference bank.

Worker pools and SDF I/O live in :mod:`_david.code.scorers.utils` so this module reads purely
as the conformational protocol.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from _david.code.config import Config
from _david.code.scorers.utils import map_tasks, parallel_embed, read_sdf, workers, write_sdf
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator

logger = logging.getLogger(__name__)

FORCE_FIELD = "MMFF94s"
RELAXATION_TALLY: Counter[str] = Counter()
# Pin protonate=False so neutral polyphenol queries (EGCG) are not deprotonated by Dimorphite-DL.
PROTONATE = False
CANONICALIZE_TAUTOMER = True


def _force_field(mol: Chem.Mol, cfg: Config) -> Any:
    """Return an MMFF94s force field for ``mol``, or ``None`` if unparameterized."""
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=FORCE_FIELD)
    props.SetMMFFDielectricConstant(cfg.dielectric)
    props.SetMMFFDielectricModel(cfg.dielectric_model)
    return AllChem.MMFFGetMoleculeForceField(mol, props)


def single_point_energy(mol: Chem.Mol, cfg: Config) -> float:
    """MMFF94s single-point energy in kcal/mol, or ``inf`` when unparameterized/invalid."""
    field = _force_field(mol, cfg)
    if field is None:
        return math.inf
    try:
        energy = float(field.CalcEnergy())
    except RuntimeError:
        return math.inf
    return energy if math.isfinite(energy) else math.inf


def _stereo_key(mol: Chem.Mol) -> str | None:
    """Canonical isomeric SMILES recovered from 3D coordinates."""
    try:
        probe = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(probe)
        return Chem.MolToSmiles(Chem.RemoveHs(probe))
    except Exception:
        return None


def _relax(task: tuple[Chem.Mol, Config, bool]) -> tuple[Chem.Mol | None, str]:
    """Minimize one conformer with MMFF94s; reject stereo drift only when ``strict_stereo``."""
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
            return (None, "stereo_uncheckable") if strict_stereo else (mol, "stereo_uncheckable")
        if before != after:
            # References must stay the exact supplied stereoisomer; candidate SMILES leave
            # stereocentres unassigned and are scored as the max over enumerated isomers.
            return (None, "stereo_changed") if strict_stereo else (mol, "stereo_changed")

        mol.SetDoubleProp("Energy", energy)
        mol.SetIntProp("MinimizeStatus", status)
        return mol, "converged" if status == 0 else "unconverged"
    except Exception as error:
        logger.debug("relaxation raised for one conformer: %s", error)
        return record, "relax_raised"


def _molecule_of(mol: Chem.Mol) -> str:
    """Prefix ``mol_<idx>`` of DrugEx's ``mol_<idx>+<isomer_idx>`` SDF record name."""
    return mol.GetProp("_Name").split("+")[0] if mol.HasProp("_Name") else "?"


def relax_sdf(path: Path, cfg: Config, *, strict_stereo: bool = False) -> list[Chem.Mol]:
    """Minimize all conformers in ``path`` in place and update :data:`RELAXATION_TALLY`."""
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
        detail = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        logger.warning(
            "relaxation dropped %d/%d conformers (%s); %d molecules left with none",
            embedded - len(kept),
            embedded,
            detail,
            len(orphaned),
        )
        if not kept and len(names) >= 10:
            raise RuntimeError(
                f"relaxation dropped all {embedded} conformers across {len(names)} molecules ({dict(reasons)})"
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
    """ETKDGv3 conformer generator with parallel sharding and optional MMFF94s relaxation."""

    def __init__(
        self,
        cfg: Config,
        max_conformers: int | None = None,
        timeout: int | None = None,
        strict_stereo: bool = False,
    ) -> None:
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

    def _native_kwargs(self) -> dict[str, Any]:
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

    def genConformers(self, smiles_list: list[str], out_dir: str | Path) -> str:
        """Generate conformers into ``out_dir``, relaxing with MMFF94s when ``cfg.minimize``."""
        path = parallel_embed(
            self,
            smiles_list,
            out_dir,
            native_kwargs=self._native_kwargs(),
            serial=lambda: RDKitConformerGenerator.genConformers(self, smiles_list, out_dir),
        )
        written = path.is_file() and path.stat().st_size > 0
        if not written and any(smiles_list):
            embeddable = sum(
                1
                for smi in smiles_list
                if isinstance(smi, str)
                and smi
                and (mol := Chem.MolFromSmiles(smi)) is not None
                and not self._filter_mol(smi, mol)
            )
            if embeddable >= 10:
                msg = f"conformer generation produced no output for {embeddable} embeddable molecule(s) in {out_dir}"
                logger.error("%s; check scratch/TMPDIR space", msg)
                raise RuntimeError(f"{msg}; check scratch/TMPDIR space")
            logger.warning(
                "conformer generation produced no output for %d molecule(s); shape scores will be 0.0",
                sum(1 for smi in smiles_list if smi),
            )
        if written and self.cfg.minimize:
            relax_sdf(path, self.cfg, strict_stereo=self.strict_stereo)
        return str(path)


def _rmsd(probe: Chem.Mol, other: Chem.Mol) -> float:
    """Symmetry-aware heavy-atom RMSD via RDKit ``GetBestRMS``."""
    return float(rdMolAlign.GetBestRMS(Chem.Mol(probe), Chem.Mol(other)))


def build_reference(cfg: Config, out_path: Path) -> list[Chem.Mol]:
    """Build the reference conformer bank by ascending MMFF94s energy + RMSD pruning (Rule B)."""
    ref_mol = Chem.MolFromSmiles(cfg.reference_smiles)
    if ref_mol is None:
        raise RuntimeError(f"reference_smiles does not parse: {cfg.reference_smiles!r}")
    expected_stereo = Chem.MolToSmiles(ref_mol)

    generator = RelaxingConformerGenerator(
        cfg, max_conformers=cfg.reference_attempts, timeout=0, strict_stereo=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        records = read_sdf(Path(generator.genConformers([cfg.reference_smiles], tmp)))
    if not records:
        raise RuntimeError(f"no conformers generated for reference {cfg.reference_smiles!r}")

    retained = [mol for mol in records if _stereo_key(mol) == expected_stereo]
    if len(retained) != len(records):
        logger.warning(
            "reference embedding dropped %d conformers not matching %s",
            len(records) - len(retained),
            expected_stereo,
        )
    if not retained:
        raise RuntimeError(
            f"none of {len(records)} reference conformers preserved stereochemistry {expected_stereo}"
        )
    if len(retained) < cfg.reference_conformers:
        logger.warning(
            "reference embedding produced %d/%d distinct conformers (fewer than requested %d)",
            len(retained),
            cfg.reference_attempts,
            cfg.reference_conformers,
        )

    ranked = sorted(((single_point_energy(mol, cfg), mol) for mol in retained), key=lambda r: r[0])
    if not math.isfinite(ranked[0][0]):
        raise RuntimeError("no reference conformer could be assigned a finite MMFF94s energy")

    # Rule B: anchor at global energy minimum, greedily keep poses with RMSD > reference_rmsd.
    probes = [Chem.RemoveHs(Chem.Mol(mol)) for _, mol in ranked]
    kept = [0]
    for idx in range(1, len(ranked)):
        if len(kept) >= cfg.reference_conformers:
            break
        if all(_rmsd(probes[idx], probes[k]) > cfg.reference_rmsd for k in kept):
            kept.append(idx)

    bank: list[Chem.Mol] = []
    for idx in kept:
        energy, mol = ranked[idx]
        mol.SetDoubleProp("Energy", energy)
        bank.append(mol)

    write_sdf(out_path, bank)
    finite_e = [e for e, _ in ranked if math.isfinite(e)]
    logger.info(
        "reference bank: %d embedded (span %.1f kcal/mol), %d kept at RMSD > %.2f A, energies %s",
        len(ranked),
        finite_e[-1] - finite_e[0],
        len(bank),
        cfg.reference_rmsd,
        [round(m.GetDoubleProp("Energy"), 2) for m in bank],
    )
    return bank
