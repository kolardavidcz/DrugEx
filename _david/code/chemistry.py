"""Shared chemical standardization, stereoisomer enumeration, and 3D conformer physics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path

from drugex.training.scorers import protonation as core_protonation
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

IDENTITY_REJECTED: int = -2  # Thesis validation result, not a native force-field status.
DEFAULT_CANDIDATE_SEED: int = 42
DEFAULT_REFERENCE_SEED: int = 104729


def enable_double_precision_mol_pickling() -> None:
    """Ensure RDKit preserves float64 3D coordinates when pickling molecules across workers.

    By default, RDKit serializes conformer coordinates in single precision (float32,
    ~2.9e-5 Å rounding error), which can shift local Gaussian shape/color overlay optima
    when reference or candidate conformers cross multiprocessing queues.
    """
    Chem.SetDefaultPickleProperties(
        Chem.GetDefaultPickleProperties() | Chem.PropertyPickleOptions.CoordsAsDouble
    )


enable_double_precision_mol_pickling()


def verify_disjoint_conformer_seeds(
    seed_a: int,
    count_a: int,
    seed_b: int,
    count_b: int,
) -> None:
    """Verify that two RDKit EmbedMultipleConfs RNG streams do not collide.

    When ``enableSequentialRandomSeeds`` is False and ``randomSeed > 0``, RDKit's
    C++ conformer embedder seeds conformer index ``i`` (0-based) with
    ``effective_seed = base_seed * (i + 1)``. Two ensembles share identical conformer
    RNG draws whenever ``seed_a * (i + 1) == seed_b * (j + 1)`` within their budgets.
    """
    if seed_a <= 0 or seed_b <= 0:
        raise ValueError("Conformer random seeds must be positive integers")
    if count_a < 1 or count_b < 1:
        raise ValueError("Conformer attempt counts must be at least 1")
    if seed_a * count_a >= 2**31 or seed_b * count_b >= 2**31:
        raise ValueError(
            "Seed * max_conformers exceeds 32-bit signed integer limit (2^31 - 1)"
        )
    stream_a = {seed_a * i for i in range(1, count_a + 1)}
    stream_b = {seed_b * i for i in range(1, count_b + 1)}
    overlap = stream_a & stream_b
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            f"Conformer RNG streams overlap ({len(overlap)} shared seeds, e.g. {sample})"
        )


def prepare_candidate_molecule(
    smiles: str,
    *,
    canonicalize_tautomer: bool = True,
    protonate: bool = False,
) -> Chem.Mol:
    """Prepare a candidate SMILES for 3D embedding while preserving assigned stereochemistry.

    pH reassignment is explicit: installing an optional dependency must not change
    an existing experiment's chemical states. When requested, it cannot fail open.
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
    if protonate:
        # Same pH and first-variant policy as DrugEx, without its silent fallback.
        if not core_protonation._DIMORPHITE_AVAILABLE:
            raise ImportError(
                "pH preparation requires dimorphite_dl; refusing a silent no-op"
            )
        variants = core_protonation.dimorphite_dl.protonate_smiles(
            Chem.MolToSmiles(mol), ph_min=7.4, ph_max=7.4, precision=0.0
        )
        parsed = Chem.MolFromSmiles(variants[0]) if variants else None
        if parsed is None:
            raise ValueError("pH-7.4 preparation returned no valid chemical state")
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
        """Human-readable label describing the force-field and dielectric model."""
        if self.method in {"none", "UFF"}:
            return self.method
        model = "distance" if self.dielectric_model == 2 else "constant"
        return f"{self.method} eps={self.dielectric:g} {model}"


def enumerate_stereoisomers(
    mol: Chem.Mol,
    *,
    limit: int,
    seed: int = 42,
) -> Iterator[Chem.Mol]:
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
    """DrugEx ETKDG with a selectable seed, per-isomer parameter isolation, and optional minimization."""

    def __init__(
        self,
        random_seed: int = DEFAULT_CANDIDATE_SEED,
        stereo_seed: int = 42,
        minimize_candidates: bool = False,
        optimization: Optimization | None = None,
        embedding_rmsd: float | None = 0.5,
        protonate: bool = False,
        **kwargs: object,
    ) -> None:
        requested = kwargs.get("max_conformers", 10)
        if not isinstance(requested, int) or not 1 <= requested <= 1000:
            raise ValueError("Conformer attempts must be an integer from 1 to 1000")
        kwargs["max_conformers"] = min(requested, 200)
        kwargs["protonate"] = protonate
        super().__init__(**kwargs)
        # The core silently clamps at 200; explicit thesis budgets must be honored.
        self.max_conformers: int = requested
        self.random_seed: int = random_seed
        self.stereo_seed: int = stereo_seed
        self.minimize_candidates: bool = minimize_candidates
        self.optimization: Optimization = optimization or Optimization()
        if embedding_rmsd is not None and embedding_rmsd < 0:
            raise ValueError("Embedding RMSD must be nonnegative or None")
        self.embedding_rmsd: float | None = embedding_rmsd
        self.last_minimization_statuses: list[int] = []
        self.last_embedding_failures: int = 0
        self.last_identity_rejections: list[str] = []

    def _create_fresh_etkdg(self) -> AllChem.EmbedParameters:
        params = super()._create_fresh_etkdg()
        params.randomSeed = self.random_seed
        params.enableSequentialRandomSeeds = False
        params.pruneRmsThresh = (
            -1.0 if self.embedding_rmsd is None else self.embedding_rmsd
        )
        return params

    def _get_isomers(self, mol: Chem.Mol) -> Iterator[Chem.Mol]:
        yield from enumerate_stereoisomers(
            mol, limit=self.max_isomers, seed=self.stereo_seed
        )

    def genConformers(self, smiles_list: Sequence[str], out_dir: str | Path) -> str:
        """Generate 3D conformers for a batch of SMILES and write valid geometries to an SDF."""
        # Core has no preparation hook. Keep its filtering, ETKDG and SDF naming
        # here while routing chemistry through the shared stereo-preserving step.
        self.last_minimization_statuses = []
        self.last_embedding_failures = 0
        self.last_identity_rejections = []
        path = str(Path(out_dir) / "conformers.sdf")
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
                        # Fresh ETKDGv3 parameters per isomer prevent maxIterations leakage across batches.
                        params = self._create_fresh_etkdg()
                        conf_ids = AllChem.EmbedMultipleConfs(
                            iso, numConfs=self.max_conformers, params=params
                        )
                        for conf_id in conf_ids:
                            writer.write(iso, confId=conf_id)
                    except (RuntimeError, ValueError) as exc:
                        logging.getLogger(__name__).warning(
                            "%s: conformer generation failed: %s", smiles, exc
                        )
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
        valid, self.last_identity_rejections = filter_conformer_identities(valid)
        self.last_embedding_failures = len(records) - len(valid)
        if self.last_embedding_failures and not self.minimize_candidates:
            with Chem.SDWriter(path) as writer:
                for mol in valid:
                    writer.write(mol)
        if self.minimize_candidates:
            minimized, statuses = optimize_conformers(valid, self.optimization)
            # ~ [Kim, Bolton & Bryant, 2013; doi:10.1186/1758-2946-5-1]
            # RMSD clustering can discard poses important to shape/colour overlap.
            # Keep converged candidates without additional RMSD clustering here;
            # embedding-stage pruning is separately controlled by embedding_rmsd.
            # The paper supports this risk, not a universal speed/accuracy optimum.
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
    records: Sequence[Chem.Mol], max_iterations: int = 1000
) -> tuple[list[Chem.Mol], list[int]]:
    """Compatibility wrapper for the selected default MMFF94 condition."""
    return optimize_conformers(records, Optimization(max_iterations=max_iterations))


def optimize_conformers(
    records: Sequence[Chem.Mol], settings: Optimization
) -> tuple[list[Chem.Mol], list[int]]:
    """Minimize copies; retain only converged, identity-valid conformers.

    Status 0 means converged, 1 means more iterations are needed, and -1 means
    invalid geometry/missing parameters. IDENTITY_REJECTED (-2) means the solver
    converged but stereo/connectivity or SDF validation failed. One rejection
    never discards other conformers. No UFF fallback is used for this method.
    Round-trip results through the scorer's SDF representation so the study and
    training use the same coordinates, aromaticity perception and properties.
    """
    if settings.method not in {"MMFF94", "MMFF94s", "UFF"}:
        raise ValueError(f"Unsupported optimization method: {settings.method}")
    minimized: list[Chem.Mol] = []
    statuses: list[int] = []
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
                    energy = float(field.CalcEnergy())
                    if not np.isfinite(energy):
                        statuses.append(-1)
                        continue
                    try:
                        check_identity(mol, record)
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
                    except ValueError as exc:
                        logging.getLogger(__name__).warning(
                            "Conformer rejected after minimization: %s", exc
                        )
                        status = IDENTITY_REJECTED
                    else:
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
    # Ordinary stereo-defining hydrogens may remain explicit after 3D assignment.
    # Normalize both representations; keep isotope-labelled H and real E/Z checks.
    params = Chem.RemoveHsParameters()
    params.removeDefiningBondStereo = True
    actual = Chem.RemoveHs(actual, params)
    expected = Chem.RemoveHs(Chem.Mol(expected), params)
    same_graph = Chem.MolToSmiles(actual, isomericSmiles=False) == Chem.MolToSmiles(
        expected, isomericSmiles=False
    )
    if not same_graph or not actual.HasSubstructMatch(expected, useChirality=True):
        raise ValueError(
            f"Molecular identity changed: {Chem.MolToSmiles(expected)} -> "
            f"{Chem.MolToSmiles(actual)}"
        )
    return Chem.MolToSmiles(actual, isomericSmiles=True)


def filter_conformer_identities(
    records: Sequence[Chem.Mol],
    expected: Chem.Mol | None = None,
) -> tuple[list[Chem.Mol], list[str]]:
    """Keep valid 3D identities; return each rejection reason for accounting."""
    accepted: list[Chem.Mol] = []
    rejected: list[str] = []
    for record in records:
        target = expected
        if target is None:
            target = Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
        try:
            check_identity(record, target)
        except ValueError as exc:
            rejected.append(str(exc))
        else:
            accepted.append(record)
    return accepted, rejected
