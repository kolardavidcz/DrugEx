"""CDPKit-based ROCS shape and pharmacophoric overlay similarity scorer using `CDPL.Shape`."""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from rdkit import Chem

try:
    import CDPL.Chem as CDPLChem  # type: ignore
    import CDPL.Pharm as CDPLPharm  # type: ignore
    import CDPL.Shape as CDPLShape  # type: ignore

    CDPL_AVAILABLE = True
except ImportError:
    CDPL_AVAILABLE = False
    CDPLChem = None  # type: ignore
    CDPLShape = None  # type: ignore
    CDPLPharm = None  # type: ignore

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer

logger = logging.getLogger(__name__)

MAX_OPTIMIZATION_ITERATIONS = 20
OPTIMIZATION_STOP_GRADIENT = 1.0

_DEFAULT_CDPKIT_GROUP_NAME = "_default_group"


@dataclass
class CDPKitWorkerContext:
    """Immutable state context passed to CDPKit worker processes in multiprocessing Pools.

    Parameters
    ----------
    reference_shapes : List[Any]
        List of pre-computed CDPKit `GaussianShape` instances for reference structures.
    group_to_indices : List[List[int]]
        Mapping from reference group indices to indices in `reference_shapes`.
    conf_file : str
        Path to query conformer SDF file.
    """

    reference_shapes: List[Any]
    group_to_indices: List[List[int]]
    conf_file: str


class CDPKitScoringWorker:
    """Callable worker evaluating query conformer alignment against CDPKit Gaussian shapes.

    Designed for memory efficiency in `multiprocessing.Pool` workflows. State is
    initialized once per worker process to avoid serializing heavy molecular graphs.
    """

    _context: ClassVar[Optional[CDPKitWorkerContext]] = None

    @staticmethod
    def initialize(context: CDPKitWorkerContext) -> None:
        """Initialize worker process with shared context.

        Parameters
        ----------
        context : CDPKitWorkerContext
            Immutable reference and file path context.
        """
        CDPKitScoringWorker._context = context

    def __call__(self, mol_id: int) -> Tuple[int, List[float]]:
        """Score all conformers belonging to `mol_id` against reference groups.

        Parameters
        ----------
        mol_id : int
            Unique identifier of query molecule.

        Returns
        -------
        Tuple[int, List[float]]
            `(mol_id, best_scores_per_group)`
        """
        ctx = CDPKitScoringWorker._context
        if ctx is None:
            return mol_id, []

        num_groups = len(ctx.group_to_indices) if ctx.group_to_indices else 0
        if not ctx.conf_file or not os.path.exists(ctx.conf_file) or num_groups == 0:
            return mol_id, [0.0] * num_groups

        group_scores = [0.0] * num_groups
        try:
            reader = CDPLChem.FileSDFMoleculeReader(ctx.conf_file)
            target_prefix = f"mol_{mol_id}+"
            while True:
                m = CDPLChem.BasicMolecule()
                if not reader.read(m):
                    break
                try:
                    name = CDPLChem.getName(m)
                except Exception:
                    continue
                if not name or not name.startswith(target_prefix):
                    continue

                query_shapes = _generate_shape_helper(m)
                if not query_shapes:
                    continue
                for group_idx, ref_indices in enumerate(ctx.group_to_indices):
                    best = group_scores[group_idx]
                    for ref_idx in ref_indices:
                        ref_shape = ctx.reference_shapes[ref_idx]
                        for query_shape in query_shapes:
                            score = _align_and_score_helper(query_shape, ref_shape)
                            if score > best:
                                best = score
                    group_scores[group_idx] = best
        except Exception:
            return mol_id, [0.0] * num_groups

        return mol_id, group_scores


def _generate_shape_helper(cdpkit_mol: Any) -> List[Any]:
    """Generate pharmacophore-annotated Gaussian shape representations for a molecule.

    Parameters
    ----------
    cdpkit_mol : CDPLChem.BasicMolecule
        CDPKit molecule with 3D coordinates.

    Returns
    -------
    List[CDPLShape.GaussianShape]
        List of generated shape objects (one per conformer).
    """
    try:
        CDPLPharm.prepareForPharmacophoreGeneration(cdpkit_mol)
        shape_gen = CDPLShape.GaussianShapeGenerator()
        shape_gen.generatePharmacophoreShape(True)
        shape_gen.multiConformerMode(True)
        shape_set = shape_gen.generate(cdpkit_mol)
        if shape_set.getSize() == 0:
            return []
        return [shape_set.getElement(i) for i in range(shape_set.getSize())]
    except (RuntimeError, ValueError):
        return []


def _align_and_score_helper(query_shape: Any, ref_shape: Any) -> float:
    """Align query Gaussian shape to reference shape and return optimal TanimotoCombo score.

    Parameters
    ----------
    query_shape : CDPLShape.GaussianShape
        Query Gaussian shape.
    ref_shape : CDPLShape.GaussianShape
        Reference Gaussian shape.

    Returns
    -------
    float
        Highest TanimotoCombo score in [0.0, 2.0].
    """
    try:
        aligner = CDPLShape.GaussianShapeAlignment()
        start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()
        aligner.setStartGenerator(start_generator)
        aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
        aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
        aligner.addReferenceShape(ref_shape)
        if not aligner.align(query_shape) or aligner.getNumResults() == 0:
            return 0.0
        best_score = 0.0
        for i in range(aligner.getNumResults()):
            alignment_result = aligner.getResult(i)
            score = CDPLShape.calcTanimotoComboScore(alignment_result)
            best_score = max(best_score, score)
        return best_score
    except (RuntimeError, ValueError):
        return 0.0


class CDPKitROCSScorer(Scorer):
    """ROCS-like 3D shape and pharmacophore similarity scorer powered by CDPKit `CDPL.Shape`.

    Features:
    - Multi-reference shape support with automatic Gaussian pharmacophore generation.
    - Principal axes alignment and gradient-based shape overlap optimization.
    - Dict-based reference grouping for multi-target or multi-conformer targets.
    - SMILES deduplication to minimize 3D conformer generation overhead.
    - Multiprocessing Pool execution with low-memory worker contexts.

    Parameters
    ----------
    conformer_generator : ConformerGenerator
        Conformer generator used to produce 3D conformers for query molecules.
    references : Union[str, List[str], Dict[str, List[str]]]
        Reference SDF file path(s) or dictionary mapping group names to file paths.
    show_progress : bool, optional
        Whether to print progress indicators, by default True.
    n_jobs : int, optional
        Worker process count for parallel scoring (-1 uses all available CPU cores), by default -1.

    Raises
    ------
    ImportError
        If CDPKit Python bindings (`CDPL`) are not installed.
    FileNotFoundError
        If a specified reference file does not exist.
    ValueError
        If no valid 3D shapes can be extracted from reference files.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[str, List[str], Dict[str, List[str]]],
        show_progress: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the CDPKit ROCS scorer.

        Parameters
        ----------
        conformer_generator : ConformerGenerator
            Conformer generator instance.
        references : Union[str, List[str], Dict[str, List[str]]]
            Target reference file mapping.
        show_progress : bool, optional
            Progress logging flag, by default True.
        n_jobs : int, optional
            Number of parallel processes, by default -1.
        """
        super().__init__()

        if not CDPL_AVAILABLE:
            raise ImportError(
                "CDPKit is required for CDPKitROCSScorer. "
                "Install with `pip install cdpkit` or conda."
            )

        self.conformer_generator = conformer_generator
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]

        self.shape_generator = CDPLShape.GaussianShapeGenerator()
        self.shape_generator.generatePharmacophoreShape(True)
        self.shape_generator.multiConformerMode(False)
        self.start_generator = CDPLShape.PrincipalAxesAlignmentStartGenerator()

        self.reference_mols: List[Any] = []
        self.reference_shapes: List[Any] = []
        self.group_to_indices: List[List[int]] = []
        self._load_reference_groups()

        self._is_supermol = (
            len(self.group_names) == 1 and len(self.group_to_indices[0]) == 1
        )

        if self.show_progress:
            print(f"CDPKit ROCS ready with {len(self.reference_shapes)} reference shape(s).")

    def _prepare_reference_groups(
        self, references: Union[str, List[str], Dict[str, List[str]]]
    ) -> List[Tuple[str, List[str]]]:
        """Normalize reference specification into group tuples.

        Parameters
        ----------
        references : Union[str, List[str], Dict[str, List[str]]]
            Input reference paths or dictionary.

        Returns
        -------
        List[Tuple[str, List[str]]]
            List of `(group_name, list_of_file_paths)`.
        """
        groups: List[Tuple[str, List[str]]] = []
        if isinstance(references, dict):
            for name, paths in references.items():
                norm_paths = self._normalize_reference_list(paths)
                groups.append((str(name), norm_paths))
        else:
            norm_paths = self._normalize_reference_list(references)
            groups.append((_DEFAULT_CDPKIT_GROUP_NAME, norm_paths))

        if not groups:
            raise ValueError("At least one reference group must be provided")
        return groups

    def _normalize_reference_list(self, refs: Union[str, List[str]]) -> List[str]:
        """Validate and convert reference paths into a list of strings.

        Parameters
        ----------
        refs : Union[str, List[str]]
            Single path or list of paths.

        Returns
        -------
        List[str]
            List of validated file paths.
        """
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            raise TypeError("Reference collection must be a path string or list of paths")
        for path in refs:
            if not isinstance(path, str):
                raise TypeError("Reference path must be string")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Reference file not found: {path}")
        return refs

    def _load_reference_groups(self) -> None:
        """Load molecules and compute Gaussian shapes for each reference group.

        Raises
        ------
        ValueError
            If any group yields zero valid Gaussian shapes.
        """
        for name, paths in self.group_definitions:
            indices: List[int] = []
            for path in paths:
                mols = self._load_reference_molecules(path)
                for mol in mols:
                    shape = self._generate_gaussian_shape(mol)
                    if shape is None:
                        continue
                    indices.append(len(self.reference_shapes))
                    self.reference_shapes.append(shape)
                    self.reference_mols.append(mol)
            if not indices:
                raise ValueError(f"No valid reference shapes generated for group: {name}")
            self.group_to_indices.append(indices)

    def _load_reference_molecules(self, filepath: str) -> List[Any]:
        """Read 3D molecules from an SDF file using CDPKit reader.

        Parameters
        ----------
        filepath : str
            Path to SDF file.

        Returns
        -------
        List[CDPLChem.BasicMolecule]
            Loaded molecules.
        """
        mols: List[Any] = []
        reader = CDPLChem.FileSDFMoleculeReader(filepath)
        while True:
            mol = CDPLChem.BasicMolecule()
            if not reader.read(mol):
                break
            mols.append(mol)
        return mols

    def _generate_gaussian_shape(self, mol: Any) -> Optional[Any]:
        """Generate a single pharmacophore-aware Gaussian shape for a reference molecule.

        Parameters
        ----------
        mol : CDPLChem.BasicMolecule
            Reference molecule.

        Returns
        -------
        CDPLShape.GaussianShape | None
            Generated shape or None on failure.
        """
        try:
            CDPLPharm.prepareForPharmacophoreGeneration(mol)
            shape_set = self.shape_generator.generate(mol)
            if shape_set.getSize() == 0:
                return None
            return shape_set.getElement(0)
        except (RuntimeError, ValueError):
            return None

    def _align_and_score(self, query_shape: Any, ref_shape: Any) -> float:
        """Align query shape to reference shape using internal alignment settings.

        Parameters
        ----------
        query_shape : CDPLShape.GaussianShape
            Query shape.
        ref_shape : CDPLShape.GaussianShape
            Reference shape.

        Returns
        -------
        float
            Optimal TanimotoCombo score.
        """
        try:
            aligner = CDPLShape.GaussianShapeAlignment()
            aligner.setStartGenerator(self.start_generator)
            aligner.setMaxNumOptimizationIterations(MAX_OPTIMIZATION_ITERATIONS)
            aligner.setOptimizationStopGradient(OPTIMIZATION_STOP_GRADIENT)
            aligner.addReferenceShape(ref_shape)
            if not aligner.align(query_shape) or aligner.getNumResults() == 0:
                return 0.0
            best_score = 0.0
            for i in range(aligner.getNumResults()):
                res = aligner.getResult(i)
                best_score = max(best_score, CDPLShape.calcTanimotoComboScore(res))
            return best_score
        except (RuntimeError, ValueError):
            return 0.0

    def getKey(self) -> List[str]:
        """Return list of identifier keys for the reference groups.

        Returns
        -------
        List[str]
            Formatted identifier strings.
        """
        if len(self.group_names) == 1 and self.group_names[0] == _DEFAULT_CDPKIT_GROUP_NAME:
            if self._is_supermol:
                return ["CDPKit_ROCS_Supermol_TanimotoCombo"]
            refs = len(self.reference_shapes)
            return [f"CDPKit_ROCS_Aggregate_{refs}refs_TanimotoCombo"]
        return [f"CDPKit_{name}" for name in self.group_names]

    def create_progress_bar(self, total: int, desc: str) -> Any:
        """Create a tqdm progress bar instance if progress reporting is enabled.

        Parameters
        ----------
        total : int
            Total items.
        desc : str
            Progress description.

        Returns
        -------
        tqdm | None
            Configured progress bar or None.
        """
        if not self.show_progress:
            return None
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc)
        except ImportError:
            return None

    def getScores(
        self,
        mols: Sequence[Any],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Score input molecules against reference query groups using CDPKit shape alignment.

        Parameters
        ----------
        mols : Sequence[Any]
            List or sequence of SMILES strings or RDKit molecules.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            2D numpy array of shape `(len(mols), num_groups)` containing TanimotoCombo scores in [0.0, 2.0].
        """
        num_groups = len(self.group_to_indices)
        if num_groups == 0:
            raise ValueError("No reference groups configured")
        if not mols:
            return np.zeros((0, num_groups))

        smiles_list: List[str | None] = []
        for mol in mols:
            if mol is None:
                smiles_list.append(None)
            elif isinstance(mol, str):
                smiles_list.append(mol)
            else:
                try:
                    smiles_list.append(Chem.MolToSmiles(mol))
                except Exception:
                    smiles_list.append(None)

        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)
        if not unique_smiles:
            return np.zeros((len(mols), num_groups))

        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(unique_smiles, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer file not generated")
                return np.zeros((len(mols), num_groups))

            present_ids: Set[int] = set()
            try:
                reader = CDPLChem.FileSDFMoleculeReader(conf_file)
                while True:
                    m = CDPLChem.BasicMolecule()
                    if not reader.read(m):
                        break
                    try:
                        name = CDPLChem.getName(m)
                        mol_id = int(name.split("+")[0].split("_")[1])
                        present_ids.add(mol_id)
                    except Exception:
                        continue
            except Exception as exc:
                if self.show_progress:
                    print(f"Warning: failed to read conformers: {exc}")
                return np.zeros((len(mols), num_groups))

            num_unique = len(unique_smiles)
            scores_unique = np.zeros((num_unique, num_groups), dtype=np.float32)

            context = CDPKitWorkerContext(
                reference_shapes=self.reference_shapes,
                group_to_indices=self.group_to_indices,
                conf_file=conf_file,
            )
            worker = CDPKitScoringWorker()

            if self.n_jobs == 1:
                worker.initialize(context)
                for mol_id in range(num_unique):
                    if mol_id not in present_ids:
                        continue
                    _, group_scores = worker(mol_id)
                    if len(group_scores) == num_groups:
                        scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
            else:
                worker_args = [mol_id for mol_id in range(num_unique) if mol_id in present_ids]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, len(worker_args) // (effective_jobs * 4))
                try:
                    with Pool(
                        self.n_jobs,
                        initializer=CDPKitScoringWorker.initialize,
                        initargs=(context,),
                    ) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(worker, worker_args, chunksize=chunksize),
                                        total=len(worker_args),
                                        desc=f"Scoring with {self.getKey()}",
                                    )
                                )
                            except ImportError:
                                results = pool.map(worker, worker_args, chunksize=chunksize)
                                print(f"  Scored {len(worker_args)} molecules (parallel)")
                        else:
                            results = pool.map(worker, worker_args, chunksize=chunksize)
                    for mol_id, group_scores in results:
                        if 0 <= mol_id < num_unique and len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    worker.initialize(context)
                    for mol_id in range(num_unique):
                        if mol_id not in present_ids:
                            continue
                        _, group_scores = worker(mol_id)
                        if len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores, dtype=np.float32)

            scores = np.zeros((len(mols), num_groups), dtype=np.float32)
            for unique_id, original_indices in unique_to_original.items():
                for original_idx in original_indices:
                    scores[original_idx] = scores_unique[unique_id]

            return scores

    @staticmethod
    def _deduplicate_smiles(
        smiles_list: Sequence[Optional[str]],
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """Group identical SMILES strings to avoid redundant conformer generation.

        Parameters
        ----------
        smiles_list : Sequence[Optional[str]]
            List of input SMILES.

        Returns
        -------
        Tuple[List[str], Dict[int, List[int]]]
            (unique_smiles_list, index_mapping)
        """
        unique_smiles: List[str] = []
        unique_lookup: Dict[str, int] = {}
        unique_to_original: Dict[int, List[int]] = defaultdict(list)

        for idx, smi in enumerate(smiles_list):
            if smi is None:
                continue
            existing = unique_lookup.get(smi)
            if existing is None:
                existing = len(unique_smiles)
                unique_smiles.append(smi)
                unique_lookup[smi] = existing
            unique_to_original[existing].append(idx)

        return unique_smiles, unique_to_original
