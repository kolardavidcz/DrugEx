"""Shared chemical standardization, stereoisomer enumeration, and 3D conformer physics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from drugex.training.scorers import protonation as core_protonation
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

IDENTITY_REJECTED: int = -2  # Thesis validation result, not a native force-field status.
DEFAULT_CANDIDATE_SEED: int = 42
DEFAULT_REFERENCE_SEED: int = 104729


def enable_double_precision_mol_pickling() -> None:
    """Ensure RDKit preserves float64 3D coordinates and properties when pickling molecules.

    By default, RDKit serializes conformer coordinates in single precision (float32,
    ~2.9e-5 Å rounding error) and strips molecular properties across multiprocessing
    queues, which can shift local Gaussian shape/color overlay optima or drop metadata.
    """
    Chem.SetDefaultPickleProperties(
        Chem.GetDefaultPickleProperties()
        | Chem.PropertyPickleOptions.CoordsAsDouble
        | Chem.PropertyPickleOptions.AllProps
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


def _count_assigned_stereo(mol: Chem.Mol) -> int:
    """Count explicitly specified tetrahedral and bond stereocenters on ``mol``."""
    return sum(str(s.specified) == "Specified" for s in Chem.FindPotentialStereo(mol))


def _canonicalize_tautomer_preserving_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Canonicalize tautomers while retaining assigned sp3 and double-bond stereochemistry.

    Unlike ``drugex.training.scorers.conformer_generators._RDKIT_TAUTOMER_ENUMERATOR``
    (which leaves ``RemoveSp3Stereo=True`` and strips assigned chirality near keto/enol
    sites), this helper explicitly disables stereocenter erasure.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetRemoveSp3Stereo(False)
    enumerator.SetRemoveBondStereo(False)
    return enumerator.Canonicalize(mol)


def _protonate_at_physiological_ph(mol: Chem.Mol) -> Chem.Mol:
    """Apply explicit pH-7.4 protonation via Dimorphite-DL without silent fallback."""
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
    return parsed


def prepare_candidate_molecule(
    smiles: str,
    *,
    canonicalize_tautomer: bool = True,
    protonate: bool = False,
) -> Chem.Mol:
    """Prepare a candidate SMILES for 3D embedding while preserving assigned stereochemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles!r}")
    assigned_before = _count_assigned_stereo(mol)
    if canonicalize_tautomer:
        mol = _canonicalize_tautomer_preserving_stereo(mol)
    if protonate:
        mol = _protonate_at_physiological_ph(mol)
    if _count_assigned_stereo(mol) < assigned_before:
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


def _has_finite_3d_conformer(mol: Chem.Mol | None) -> bool:
    """Return True when ``mol`` has at least one 3D conformer with finite coordinates."""
    if mol is None or not mol.GetNumConformers():
        return False
    conf = mol.GetConformer()
    return bool(conf.Is3D() and np.isfinite(conf.GetPositions()).all())


def _write_sdf_records(path: str, records: Sequence[Chem.Mol]) -> None:
    """Write a sequence of RDKit molecule records to an SDF path."""
    with Chem.SDWriter(path) as writer:
        for mol in records:
            writer.write(mol)


class CandidateConformerGenerator(RDKitConformerGenerator):
    """DrugEx ETKDGv3 generator with selectable seed, per-isomer isolation, and optional minimization."""

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
        max_heavy = kwargs.pop("max_heavy_atoms", 35)
        max_rotors = kwargs.pop("max_rotatable_bonds", 15)
        kwargs["max_conformers"] = min(requested, 200)
        kwargs["protonate"] = protonate
        super().__init__(
            max_heavy_atoms=35 if max_heavy is None else int(max_heavy),
            max_rotatable_bonds=15 if max_rotors is None else int(max_rotors),
            **kwargs,
        )
        self.max_heavy_atoms: int | None = None if max_heavy is None else int(max_heavy)
        self.max_rotatable_bonds: int | None = (
            None if max_rotors is None else int(max_rotors)
        )
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
        self.last_filtered_molecules: list[str] = []

    def _filter_mol(self, smi: str, mol: Chem.Mol) -> bool:
        """Filter candidate molecules exceeding optional heavy-atom or rotatable-bond caps."""
        if (
            self.max_heavy_atoms is not None
            and rdMolDescriptors.CalcNumHeavyAtoms(mol) > self.max_heavy_atoms
        ):
            self.last_filtered_molecules.append(smi)
            return True
        if (
            self.max_rotatable_bonds is not None
            and rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds
        ):
            self.last_filtered_molecules.append(smi)
            return True
        return False

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

    def _embed_batch_to_sdf(self, smiles_list: Sequence[str], sdf_path: str) -> None:
        """Prepare candidates, enumerate unassigned stereoisomers, and write raw ETKDGv3 conformers."""
        with Chem.SDWriter(sdf_path) as writer:
            for index, smiles in enumerate(smiles_list):
                if not smiles:
                    continue
                raw_mol = Chem.MolFromSmiles(smiles)
                if raw_mol is None or self._filter_mol(smiles, raw_mol):
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
                    iso_h = Chem.AddHs(iso)
                    iso_h.SetProp("_Name", f"mol_{index}+{iso_index}")
                    try:
                        # Fresh ETKDGv3 parameters per isomer prevent maxIterations leakage across batches.
                        params = self._create_fresh_etkdg()
                        conf_ids = AllChem.EmbedMultipleConfs(
                            iso_h, numConfs=self.max_conformers, params=params
                        )
                        for conf_id in conf_ids:
                            writer.write(iso_h, confId=conf_id)
                    except (RuntimeError, ValueError) as exc:
                        logging.getLogger(__name__).warning(
                            "%s: conformer generation failed: %s", smiles, exc
                        )

    def genConformers(self, smiles_list: Sequence[str], out_dir: str | Path) -> str:
        """Generate 3D conformers for a batch of SMILES and write valid geometries to an SDF."""
        self.last_minimization_statuses = []
        self.last_embedding_failures = 0
        self.last_identity_rejections = []
        self.last_filtered_molecules = []
        path_obj = Path(out_dir) / "conformers.sdf"
        path = str(path_obj)
        self._embed_batch_to_sdf(smiles_list, path)
        if not path_obj.is_file() or path_obj.stat().st_size == 0:
            return path
        try:
            records = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
        except (OSError, ValueError):
            return path
        valid = [mol for mol in records if _has_finite_3d_conformer(mol)]
        valid, self.last_identity_rejections = filter_conformer_identities(valid)
        self.last_embedding_failures = len(records) - len(valid)

        if not self.minimize_candidates:
            if self.last_embedding_failures:
                _write_sdf_records(path, valid)
            return path

        if not valid:
            return path

        minimized, statuses = optimize_conformers(valid, self.optimization)
        self.last_minimization_statuses = statuses
        if not minimized:
            return path
        _write_sdf_records(path, minimized)
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


def _build_force_field(mol: Chem.Mol, settings: Optimization) -> Any:
    """Construct an RDKit UFF or MMFF94/MMFF94s force-field instance for ``mol``."""
    if settings.method == "UFF" and AllChem.UFFHasAllMoleculeParams(mol):
        return AllChem.UFFGetMoleculeForceField(mol)
    if settings.method.startswith("MMFF") and AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=settings.method)
        props.SetMMFFDielectricConstant(settings.dielectric)
        props.SetMMFFDielectricModel(settings.dielectric_model)
        return AllChem.MMFFGetMoleculeForceField(mol, props)
    return None


def _round_trip_sdf_conformer(
    mol: Chem.Mol,
    reference: Chem.Mol,
    settings: Optimization,
    energy: float,
    status: int,
) -> Chem.Mol:
    """Annotate force-field metadata, round-trip through SDF, and verify stereochemical identity."""
    check_identity(mol, reference)
    mol.SetProp("ForceField", settings.method)
    mol.SetProp("Optimization", settings.name)
    mol.SetDoubleProp("Energy_kcal_mol", energy)
    mol.SetIntProp("OptimizationStatus", status)
    if settings.method.startswith("MMFF"):
        mol.SetIntProp("MMFFOptimizationStatus", status)
    supplier = Chem.SDMolSupplier()
    supplier.SetData(Chem.SDWriter.GetText(mol), removeHs=False)
    round_tripped = supplier[0]
    check_identity(round_tripped, reference)
    return round_tripped


def _minimize_single_record(
    record: Chem.Mol | None, settings: Optimization
) -> tuple[Chem.Mol | None, int]:
    """Minimize one 3D conformer copy and validate its post-minimization identity."""
    if not _has_finite_3d_conformer(record):
        return None, -1
    mol = Chem.Mol(record)
    field = _build_force_field(mol, settings)
    if field is None:
        return None, -1
    status = int(field.Minimize(maxIts=settings.max_iterations))
    if status != 0:
        return None, status
    energy = float(field.CalcEnergy())
    if not np.isfinite(energy):
        return None, -1
    try:
        validated = _round_trip_sdf_conformer(mol, record, settings, energy, status)
    except ValueError as exc:
        logging.getLogger(__name__).warning(
            "Conformer rejected after minimization: %s", exc
        )
        return None, IDENTITY_REJECTED
    return validated, 0


def _minimize_chunk_worker(
    args: tuple[list[Chem.Mol | None], Optimization],
) -> tuple[list[Chem.Mol], list[int]]:
    """Minimize a chunk of conformer records in a worker process with double-precision pickling."""
    enable_double_precision_mol_pickling()
    chunk, settings = args
    minimized: list[Chem.Mol] = []
    statuses: list[int] = []
    for record in chunk:
        validated, status = _minimize_single_record(record, settings)
        statuses.append(status)
        if validated is not None:
            minimized.append(validated)
    return minimized, statuses


def optimize_conformers(
    records: Sequence[Chem.Mol],
    settings: Optimization,
    n_jobs: int | None = None,
) -> tuple[list[Chem.Mol], list[int]]:
    """Minimize copies; retain only converged, identity-valid conformers.

    Status 0 means converged, 1 means more iterations are needed, and -1 means
    invalid geometry/missing parameters. IDENTITY_REJECTED (-2) means the solver
    converged but stereo/connectivity or SDF validation failed.
    """
    if settings.method not in {"MMFF94", "MMFF94s", "UFF"}:
        raise ValueError(f"Unsupported optimization method: {settings.method}")

    import multiprocessing
    from drugex.training.scorers.parallel import pool_context
    from _david.code.utils import HardwareManager

    enable_double_precision_mol_pickling()
    workers = (
        HardwareManager.get_safe_parallel_jobs() if n_jobs is None else int(n_jobs)
    )
    if (
        len(records) >= 32
        and workers > 1
        and not multiprocessing.current_process().daemon
    ):
        num_chunks = min(len(records), workers * 2)
        chunk_size = (len(records) + num_chunks - 1) // num_chunks
        tasks = [
            (list(records[i : i + chunk_size]), settings)
            for i in range(0, len(records), chunk_size)
        ]
        try:
            with pool_context().Pool(
                min(workers, len(tasks)),
                initializer=enable_double_precision_mol_pickling,
            ) as pool:
                chunk_results = pool.map(_minimize_chunk_worker, tasks, chunksize=1)
            minimized: list[Chem.Mol] = []
            statuses: list[int] = []
            for m_chunk, s_chunk in chunk_results:
                minimized.extend(m_chunk)
                statuses.extend(s_chunk)
            return minimized, statuses
        except Exception:
            pass

    minimized = []
    statuses = []
    for record in records:
        validated, status = _minimize_single_record(record, settings)
        statuses.append(status)
        if validated is not None:
            minimized.append(validated)
    return minimized, statuses


def standardize_smiles(smiles: str) -> str:
    """Execute DrugEx's chemical standardization pipeline while preserving isomeric SMILES.

    Note: ``drugex.molecules.converters.standardizers.DefaultStandardizer`` ends with
    ``Chem.MolToSmiles(rd_mol, 0)`` (``isomericSmiles=False``), which erases all ``@``
    and ``/`` ``\\`` stereocenters. This function runs the identical RDKit standardization
    steps (MetalDisconnector -> Normalizer -> LargestFragmentChooser -> Uncharger) and
    serializes with ``isomericSmiles=True``.
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


def _strip_non_isotopic_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Remove explicit hydrogens (including stereo-defining H) while keeping isotopic H."""
    params = Chem.RemoveHsParameters()
    params.removeDefiningBondStereo = True
    return Chem.RemoveHs(Chem.Mol(mol), params)


def check_identity(mol: Chem.Mol, expected: Chem.Mol) -> str:
    """Check connectivity and specified stereo, including assignment from 3D coordinates.

    Unspecified input centers may be resolved in 3D; specified centers may never invert.
    Returns the canonical isomeric SMILES of ``mol`` after 3D stereo perception.
    """
    actual = Chem.Mol(mol)
    if actual.GetNumConformers():
        if not _has_finite_3d_conformer(actual):
            raise ValueError("Conformer lacks finite 3D coordinates")
        Chem.AssignStereochemistryFrom3D(actual, replaceExistingTags=True)
    actual_clean = _strip_non_isotopic_hydrogens(actual)
    expected_clean = _strip_non_isotopic_hydrogens(expected)
    same_connectivity = Chem.MolToSmiles(
        actual_clean, isomericSmiles=False
    ) == Chem.MolToSmiles(expected_clean, isomericSmiles=False)
    if not same_connectivity or not actual_clean.HasSubstructMatch(
        expected_clean, useChirality=True
    ):
        raise ValueError(
            f"Molecular identity changed: {Chem.MolToSmiles(expected_clean)} -> "
            f"{Chem.MolToSmiles(actual_clean)}"
        )
    return Chem.MolToSmiles(actual_clean, isomericSmiles=True)


def filter_conformer_identities(
    records: Sequence[Chem.Mol],
    expected: Chem.Mol | None = None,
) -> tuple[list[Chem.Mol], list[str]]:
    """Keep valid 3D identities; return each rejection reason for accounting."""
    accepted: list[Chem.Mol] = []
    rejected: list[str] = []
    for record in records:
        target = (
            expected
            if expected is not None
            else Chem.MolFromSmiles(record.GetProp("EnumeratedIsomerSMILES"))
        )
        try:
            check_identity(record, target)
        except ValueError as exc:
            rejected.append(str(exc))
        else:
            accepted.append(record)
    return accepted, rejected
