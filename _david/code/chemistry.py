"""Small thesis-local adaptations; DrugEx core remains unchanged."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers import conformer_generators as core_preparation
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


def prepare_candidate_molecule(smiles, *, canonicalize_tautomer=True, protonate=True):
    """Training chemical state, preserving assigned stereo during tautomerization.

    RDKit otherwise removes stereo from atoms/bonds touched by any enumerated
    tautomer, even when the chosen tautomer still supports the original stereo.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles!r}")
    assigned_before = sum(
        str(s.specified) == "Specified" for s in Chem.FindPotentialStereo(mol)
    )
    if canonicalize_tautomer:
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetRemoveSp3Stereo(False)
        enumerator.SetRemoveBondStereo(False)
        mol = enumerator.Canonicalize(mol)
    if protonate and core_preparation._rdkit_protonate_smiles is not None:
        parsed = Chem.MolFromSmiles(
            core_preparation._rdkit_protonate_smiles(Chem.MolToSmiles(mol))
        )
        if parsed is not None:
            mol = parsed
    assigned_after = sum(
        str(s.specified) == "Specified" for s in Chem.FindPotentialStereo(mol)
    )
    if assigned_after < assigned_before:
        raise ValueError("Selected chemical state loses assigned stereochemistry")
    return mol


@dataclass(frozen=True)
class Optimization:
    """One explicit force-field condition, shared by references and candidates."""

    method: str = "MMFF94"
    dielectric: float = 1.0
    dielectric_model: int = 1
    max_iterations: int = 1000

    @property
    def name(self) -> str:
        if self.method in {"none", "UFF"}:
            return self.method
        model = "distance" if self.dielectric_model == 2 else "constant"
        return f"{self.method} eps={self.dielectric:g} {model}"


def enumerate_stereoisomers(mol, *, limit: int, seed: int = 42):
    """Share seeded, identity-labelled enumeration without changing assigned stereo."""
    if limit < 1:
        raise ValueError("Stereoisomer limit must be positive")
    options = StereoEnumerationOptions(
        maxIsomers=limit,
        onlyUnassigned=True,
        tryEmbedding=False,
        rand=seed,
        unique=True,
    )
    for index, iso in enumerate(EnumerateStereoisomers(mol, options)):
        iso.SetProp(
            "EnumeratedIsomerSMILES", Chem.MolToSmiles(iso, isomericSmiles=True)
        )
        iso.SetIntProp("EnumeratedIsomerIndex", index)
        yield iso


class CandidateConformerGenerator(RDKitConformerGenerator):
    """DrugEx ETKDG with a selectable seed and optional MMFF94 post-minimization."""

    def __init__(
        self,
        random_seed: int = 42,
        stereo_seed: int = 42,
        minimize_candidates: bool = False,
        optimization: Optimization | None = None,
        **kwargs,
    ) -> None:
        requested = kwargs.get("max_conformers", 10)
        if not isinstance(requested, int) or not 1 <= requested <= 1000:
            raise ValueError("Conformer attempts must be an integer from 1 to 1000")
        kwargs["max_conformers"] = min(requested, 200)
        super().__init__(**kwargs)
        # The core silently clamps at 200; explicit thesis budgets must be honored.
        self.max_conformers = requested
        self.random_seed = random_seed
        self.stereo_seed = stereo_seed
        self.minimize_candidates = minimize_candidates
        self.optimization = optimization or Optimization()
        self.last_minimization_statuses: list[int] = []
        self.last_embedding_failures: int = 0

    def _create_fresh_etkdg(self):
        params = super()._create_fresh_etkdg()
        params.randomSeed = self.random_seed
        return params

    def _get_isomers(self, mol):
        yield from enumerate_stereoisomers(
            mol, limit=self.max_isomers, seed=self.stereo_seed
        )

    def genConformers(self, smiles_list, out_dir) -> str:
        # Core has no preparation hook. Keep its filtering, ETKDG and SDF naming
        # here while routing chemistry through the shared stereo-preserving step.
        path = str(Path(out_dir) / "conformers.sdf")
        params = self._create_fresh_etkdg()
        with Chem.SDWriter(path) as writer:
            for index, smiles in enumerate(smiles_list):
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None or self._filter_mol(smiles, mol):
                    continue
                try:
                    mol = prepare_candidate_molecule(
                        smiles,
                        canonicalize_tautomer=self.canonicalize_tautomer,
                        protonate=self.protonate,
                    )
                except (ValueError, RuntimeError) as exc:
                    logging.getLogger(__name__).warning(
                        "%s: preparation failed: %s", smiles, exc
                    )
                    continue
                for iso_index, iso in enumerate(self._get_isomers(mol)):
                    iso = Chem.AddHs(iso)
                    iso.SetProp("_Name", f"mol_{index}+{iso_index}")
                    try:
                        conf_ids = AllChem.EmbedMultipleConfs(
                            iso, numConfs=self.max_conformers, params=params
                        )
                        for conf_id in conf_ids:
                            writer.write(iso, confId=conf_id)
                    except (RuntimeError, ValueError) as exc:
                        logging.getLogger(__name__).warning(
                            "%s: conformer generation failed: %s", smiles, exc
                        )
        self.last_minimization_statuses = []
        self.last_embedding_failures = 0
        try:
            records = list(Chem.SDMolSupplier(path, removeHs=False))
        except (OSError, ValueError):
            return path  # An empty generation remains an empty scorer input.
        valid = [
            mol
            for mol in records
            if mol is not None
            and mol.GetNumConformers()
            and mol.GetConformer().Is3D()
            and np.isfinite(mol.GetConformer().GetPositions()).all()
        ]
        self.last_embedding_failures = len(records) - len(valid)
        if self.last_embedding_failures and not self.minimize_candidates:
            with Chem.SDWriter(path) as writer:
                for mol in valid:
                    writer.write(mol)
        if self.minimize_candidates:
            minimized, statuses = optimize_conformers(valid, self.optimization)
            self.last_minimization_statuses = statuses
            with Chem.SDWriter(path) as writer:
                for mol in minimized:
                    writer.write(mol)
            failures = sum(status != 0 for status in statuses)
            if failures:
                logging.getLogger(__name__).warning(
                    "%s candidate minimization excluded %d/%d conformers",
                    self.optimization.name,
                    failures,
                    len(statuses),
                )
        return path


def minimize_mmff94(
    records: list[Chem.Mol], max_iterations: int = 1000
) -> tuple[list[Chem.Mol], list[int]]:
    """Compatibility wrapper for the selected default MMFF94 condition."""
    return optimize_conformers(records, Optimization(max_iterations=max_iterations))


def optimize_conformers(
    records: list[Chem.Mol], settings: Optimization
) -> tuple[list[Chem.Mol], list[int]]:
    """Minimize copies; return converged conformers and every native status.

    Status 0 means converged, 1 means more iterations are needed, and -1 means
    invalid geometry/missing parameters. No UFF fallback is used for this method.
    Round-trip results through the scorer's SDF representation so the study and
    training use the same coordinates, aromaticity perception and properties.
    """
    if settings.method not in {"MMFF94", "MMFF94s", "UFF"}:
        raise ValueError(f"Unsupported optimization method: {settings.method}")
    minimized, statuses = [], []
    for record in records:
        status = -1
        if record is not None and record.GetNumConformers():
            mol = Chem.Mol(record)
            conf = mol.GetConformer()
            field = None
            if conf.Is3D() and np.isfinite(conf.GetPositions()).all():
                if settings.method == "UFF" and AllChem.UFFHasAllMoleculeParams(mol):
                    field = AllChem.UFFGetMoleculeForceField(mol)
                elif settings.method.startswith(
                    "MMFF"
                ) and AllChem.MMFFHasAllMoleculeParams(mol):
                    props = AllChem.MMFFGetMoleculeProperties(
                        mol, mmffVariant=settings.method
                    )
                    props.SetMMFFDielectricConstant(settings.dielectric)
                    props.SetMMFFDielectricModel(settings.dielectric_model)
                    field = AllChem.MMFFGetMoleculeForceField(mol, props)
            if field is not None:
                status = field.Minimize(maxIts=settings.max_iterations)
                if status == 0:
                    check_identity(mol, record)
                    energy = float(field.CalcEnergy())
                    if not np.isfinite(energy):
                        statuses.append(-1)
                        continue
                    mol.SetProp("ForceField", settings.method)
                    mol.SetProp("Optimization", settings.name)
                    mol.SetDoubleProp("Energy_kcal_mol", energy)
                    mol.SetIntProp("OptimizationStatus", status)
                    if settings.method.startswith("MMFF"):
                        mol.SetIntProp("MMFFOptimizationStatus", status)
                    supplier = Chem.SDMolSupplier()
                    supplier.SetData(Chem.SDWriter.GetText(mol), removeHs=False)
                    mol = supplier[0]
                    check_identity(mol, record)
                    minimized.append(mol)
        statuses.append(status)
    return minimized, statuses


def standardize_smiles(smiles: str) -> str:
    """Use DrugEx's standardization steps, retaining isomeric SMILES output.

    DefaultStandardizer ends with non-isomeric serialization. There is no option
    to change that, so reproduce its RDKit operations here rather than patch core
    or bypass desalting/normalization. Fail explicitly instead of using a fallback
    with a different chemistry policy.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    mol = disconnector.Disconnect(mol)
    mol = normalizer.normalize(mol)
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = normalizer.normalize(disconnector.Disconnect(mol))
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        raise ValueError("Standardized molecule contains no carbon")
    if any(atom.GetAtomicNum() in (11, 30) for atom in mol.GetAtoms()):
        raise ValueError("Salt removal failed")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def check_identity(mol: Chem.Mol, expected: Chem.Mol) -> str:
    """Check connectivity and specified stereo, including assignment from 3D.

    Unspecified input centers may be resolved; specified centers may not change.
    Return the actual isomeric identity for inspection and winner reporting.
    """
    actual = Chem.Mol(mol)
    if actual.GetNumConformers():
        conf = actual.GetConformer()
        if not conf.Is3D() or not np.isfinite(conf.GetPositions()).all():
            raise ValueError("Conformer lacks finite 3D coordinates")
        Chem.AssignStereochemistryFrom3D(actual, replaceExistingTags=True)
    actual = Chem.RemoveHs(actual)
    expected = Chem.RemoveHs(Chem.Mol(expected))
    same_graph = Chem.MolToSmiles(actual, isomericSmiles=False) == Chem.MolToSmiles(
        expected, isomericSmiles=False
    )
    if not same_graph or not actual.HasSubstructMatch(expected, useChirality=True):
        raise ValueError(
            f"Molecular identity changed: {Chem.MolToSmiles(expected)} -> "
            f"{Chem.MolToSmiles(actual)}"
        )
    return Chem.MolToSmiles(actual, isomericSmiles=True)
