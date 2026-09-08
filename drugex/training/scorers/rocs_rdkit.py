"""RDKit-based ROCS shape and pharmacophoric color similarity scorer using `rdShapeAlign`."""

from __future__ import annotations

import logging
import os
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeAlign

from drugex.training.scorers.interfaces import ConformerGenerator, Scorer

logger = logging.getLogger(__name__)

_RDKIT_WORKER_SETTINGS: Dict[str, Any] = {}
_DEFAULT_RDKIT_GROUP_NAME = "_default_group"


def _score_single_reference(
    query_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    score_type: str,
    use_colors: bool,
) -> float:
    """Compute highest pairwise alignment score between conformers of query and reference molecules.

    Parameters
    ----------
    query_mol : Chem.Mol
        Query molecule containing generated 3D conformers.
    ref_mol : Chem.Mol
        Reference ligand molecule containing 3D coordinates.
    score_type : str
        Scoring metric: 'shape' (ShapeTanimoto), 'color' (ColorTanimoto),
        or 'TanimotoCombo' (shape + color).
    use_colors : bool
        Whether to calculate pharmacophore feature color overlays.

    Returns
    -------
    float
        Highest similarity score found across all conformer combinations.
    """
    if query_mol is None or ref_mol is None:
        return 0.0
    if query_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return 0.0

    best_score = 0.0
    for query_conf in query_mol.GetConformers():
        for ref_conf in ref_mol.GetConformers():
            try:
                probe_copy = Chem.Mol(query_mol)
                result = rdShapeAlign.AlignMol(
                    ref_mol,
                    probe_copy,
                    refConfId=ref_conf.GetId(),
                    probeConfId=query_conf.GetId(),
                    useColors=use_colors,
                )
            except (RuntimeError, ValueError):
                continue

            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue

            shape_score, color_score = result[0], result[1]
            if score_type == "shape":
                score = shape_score
            elif score_type == "color":
                score = color_score
            else:
                score = shape_score + color_score

            if score > best_score:
                best_score = score
    return best_score


def _rdkit_worker_init(
    reference_mols: List[Chem.Mol],
    group_to_indices: List[List[int]],
    score_type: str,
    use_colors: bool,
) -> None:
    """Initializer callable to distribute shared reference data to worker processes.

    Parameters
    ----------
    reference_mols : List[Chem.Mol]
        Flattened list of all reference molecules.
    group_to_indices : List[List[int]]
        Mapping of reference groups to indices in `reference_mols`.
    score_type : str
        Scoring mode.
    use_colors : bool
        Whether color matching is enabled.
    """
    global _RDKIT_WORKER_SETTINGS
    _RDKIT_WORKER_SETTINGS = {
        "reference_mols": reference_mols,
        "group_to_indices": group_to_indices,
        "score_type": score_type,
        "use_colors": use_colors,
    }


def _score_molecule_rdkit_worker(args: Tuple[int, List[Chem.Mol]]) -> Tuple[int, List[float]]:
    """Multiprocessing worker function scoring conformers of one molecule against all reference groups.

    Parameters
    ----------
    args : Tuple[int, List[Chem.Mol]]
        `(molecule_id, list_of_conformers)` for the molecule.

    Returns
    -------
    Tuple[int, List[float]]
        `(molecule_id, max_scores_per_group)`.
    """
    mol_id, mol_conformers = args
    settings = _RDKIT_WORKER_SETTINGS
    reference_mols: List[Chem.Mol] = settings.get("reference_mols", [])
    group_to_indices: List[List[int]] = settings.get("group_to_indices", [])
    score_type: str = settings.get("score_type", "TanimotoCombo")
    use_colors: bool = settings.get("use_colors", True)
    num_groups = len(group_to_indices) if group_to_indices else (1 if reference_mols else 0)
    if not mol_conformers or num_groups == 0:
        return mol_id, [0.0] * num_groups

    group_scores = [0.0] * num_groups
    try:
        for conf_mol in mol_conformers:
            if conf_mol is None or conf_mol.GetNumConformers() == 0:
                continue
            for group_idx, ref_indices in enumerate(group_to_indices):
                for ref_idx in ref_indices:
                    ref_mol = reference_mols[ref_idx]
                    score = _score_single_reference(conf_mol, ref_mol, score_type, use_colors)
                    if score > group_scores[group_idx]:
                        group_scores[group_idx] = score
    except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
        logger.warning("Error scoring molecule %d: %s", mol_id, exc)
        return mol_id, [0.0] * num_groups

    return mol_id, group_scores


class RDKitROCSScorer(Scorer):
    """ROCS-like 3D shape and pharmacophore similarity scorer powered by RDKit `rdShapeAlign`.

    Features:
    - Multi-target reference groups with flexible input (file paths, RDKit Mols).
    - ShapeTanimoto, ColorTanimoto, and TanimotoCombo (shape + color) scoring modes.
    - Automatic 3D conformer embedding for 2D reference structures.
    - SMILES deduplication to avoid redundant 3D embedding during batch scoring.
    - Multiprocessing Pool support for high-throughput batch evaluation.

    Parameters
    ----------
    conformer_generator : ConformerGenerator
        Conformer generator used to produce 3D conformers for query molecules.
    references : Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]]
        Reference structure(s) or dictionary mapping group names to references.
    score_type : str, optional
        Scoring mode: 'TanimotoCombo', 'shape', or 'color', by default 'TanimotoCombo'.
    use_colors : bool, optional
        Whether to calculate pharmacophore color overlap in addition to shape, by default True.
    show_progress : bool, optional
        Whether to print progress messages, by default True.
    n_jobs : int, optional
        Worker process count for parallel scoring (-1 uses all available CPU cores), by default -1.
    """

    def __init__(
        self,
        conformer_generator: ConformerGenerator,
        references: Union[
            str,
            List[str],
            Dict[str, List[str]],
            Chem.Mol,
            List[Chem.Mol],
            Dict[str, List[Chem.Mol]],
        ],
        score_type: str = "TanimotoCombo",
        use_colors: bool = True,
        show_progress: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the RDKit ROCS scorer.

        Parameters
        ----------
        conformer_generator : ConformerGenerator
            Conformer generator instance.
        references : Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]]
            Reference structure specification.
        score_type : str, optional
            Alignment scoring metric, by default 'TanimotoCombo'.
        use_colors : bool, optional
            Pharmacophore color matching flag, by default True.
        show_progress : bool, optional
            Progress logging flag, by default True.
        n_jobs : int, optional
            Parallel workers, by default -1.
        """
        super().__init__()

        self.conformer_generator = conformer_generator
        self.score_type = score_type
        self.use_colors = use_colors
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        self.group_definitions = self._prepare_reference_groups(references)
        self.group_names = [name for name, _ in self.group_definitions]
        self.reference_mols, self.group_to_indices = self._flatten_groups(self.group_definitions)
        self.reference_mols = [self._ensure_reference_conformers(m) for m in self.reference_mols]
        self._validate_references()
        self._single_reference = len(self.reference_mols) == 1

    def _prepare_reference_groups(
        self,
        references: Union[
            str,
            List[str],
            Dict[str, List[str]],
            Chem.Mol,
            List[Chem.Mol],
            Dict[str, List[Chem.Mol]],
        ],
    ) -> List[Tuple[str, List[Chem.Mol]]]:
        """Normalize input reference definitions into group tuples.

        Parameters
        ----------
        references : Any
            User-provided reference input.

        Returns
        -------
        List[Tuple[str, List[Chem.Mol]]]
            List of `(group_name, list_of_reference_mols)`.
        """
        groups: List[Tuple[str, List[Chem.Mol]]] = []

        if isinstance(references, dict):
            for name, refs in references.items():
                ref_mols = self._normalize_reference_collection(refs)
                groups.append((str(name), ref_mols))
        else:
            ref_mols = self._normalize_reference_collection(references)
            groups.append((_DEFAULT_RDKIT_GROUP_NAME, ref_mols))

        if not groups:
            raise ValueError("At least one reference group must be provided")
        return groups

    def _normalize_reference_collection(
        self,
        refs: Union[str, List[str], Chem.Mol, List[Chem.Mol]],
    ) -> List[Chem.Mol]:
        """Convert reference file paths or molecules into a flat list of RDKit molecules.

        Parameters
        ----------
        refs : Union[str, List[str], Chem.Mol, List[Chem.Mol]]
            Path, Mol, or list thereof.

        Returns
        -------
        List[Chem.Mol]
            List of loaded RDKit molecules.
        """
        if isinstance(refs, (str, Chem.Mol)):
            refs = [refs]  # type: ignore
        if not isinstance(refs, list):
            raise TypeError("references must be str, List[str], Chem.Mol, List[Chem.Mol], or dict thereof")

        normalized: List[Chem.Mol] = []
        for item in refs:
            if isinstance(item, str):
                normalized.extend(self._load_molecules_from_file(item))
            elif isinstance(item, Chem.Mol):
                normalized.append(item)
            else:
                raise TypeError("Reference entries must be file paths or RDKit molecules")

        if not normalized:
            raise ValueError("Reference group cannot be empty")
        return normalized

    def _flatten_groups(
        self, groups: List[Tuple[str, List[Chem.Mol]]]
    ) -> Tuple[List[Chem.Mol], List[List[int]]]:
        """Flatten grouped molecules into a single list with index pointer groups.

        Parameters
        ----------
        groups : List[Tuple[str, List[Chem.Mol]]]
            Group definitions.

        Returns
        -------
        Tuple[List[Chem.Mol], List[List[int]]]
            (all_reference_mols, group_to_indices_mapping)
        """
        reference_mols: List[Chem.Mol] = []
        group_to_indices: List[List[int]] = []

        for _, refs in groups:
            indices: List[int] = []
            for ref in refs:
                indices.append(len(reference_mols))
                reference_mols.append(ref)
            group_to_indices.append(indices)

        return reference_mols, group_to_indices

    def _load_molecules_from_file(self, path: str) -> List[Chem.Mol]:
        """Load RDKit molecules with 3D coordinates from an SDF file.

        Parameters
        ----------
        path : str
            Path to SDF file.

        Returns
        -------
        List[Chem.Mol]
            Parsed molecules with atoms.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference file not found: {path}")

        try:
            suppl = Chem.SDMolSupplier(path, removeHs=False)
            if not suppl:
                raise ValueError(f"Could not open SDF file: {path}")
            mols = [m for m in suppl if m is not None and m.GetNumAtoms() > 0]
            if not mols:
                raise ValueError(f"No molecules found in file: {path}")
            return mols
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: failed to load {path}: {exc}")
            raise ValueError(f"Failed to load molecules from {path}: {exc}") from exc

    def _ensure_reference_conformers(self, mol: Chem.Mol) -> Chem.Mol:
        """Embed a 3D conformer using ETKDGv3 if the reference molecule lacks 3D coordinates.

        Parameters
        ----------
        mol : Chem.Mol
            Reference molecule.

        Returns
        -------
        Chem.Mol
            Reference molecule guaranteed to have at least one 3D conformer.
        """
        if mol is None:
            return mol
        if mol.GetNumConformers() > 0:
            return mol
        try:
            m = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xC0FFEE
            AllChem.EmbedMolecule(m, params=params)
            return m if m.GetNumConformers() > 0 else mol
        except Exception as exc:
            if self.show_progress:
                print(f"Warning: embedding reference failed: {exc}")
            return mol

    def _validate_references(self) -> None:
        """Ensure all loaded reference molecules possess valid conformers and rebuild index maps.

        Raises
        ------
        ValueError
            If no valid reference molecules with conformers exist.
        """
        if not self.reference_mols:
            raise ValueError("At least one reference molecule is required")

        index_map: Dict[int, int] = {}
        valid_refs: List[Chem.Mol] = []
        for idx, ref_mol in enumerate(self.reference_mols):
            if ref_mol is None or ref_mol.GetNumConformers() == 0:
                if self.show_progress:
                    print(f"Warning: reference molecule at index {idx} is invalid or lacks conformers")
                continue
            index_map[idx] = len(valid_refs)
            valid_refs.append(ref_mol)

        if not valid_refs:
            raise ValueError("No valid reference molecules with conformers available")

        new_group_to_indices: List[List[int]] = []
        for name, indices in zip(self.group_names, self.group_to_indices):
            mapped = [index_map[i] for i in indices if i in index_map]
            if not mapped:
                raise ValueError(f"Reference group '{name}' has no valid molecules with conformers")
            new_group_to_indices.append(mapped)

        self.reference_mols = valid_refs
        self.group_to_indices = new_group_to_indices

    def getKey(self) -> List[str]:
        """Return list of identifier keys for the reference groups.

        Returns
        -------
        List[str]
            Identifier keys formatted by group or aggregation mode.
        """
        if len(self.group_names) == 1 and self.group_names[0] == _DEFAULT_RDKIT_GROUP_NAME:
            prefix = "RDKit_Supermol" if self._single_reference else "RDKit_Aggregate"
            refs = len(self.reference_mols)
            if prefix == "RDKit_Aggregate":
                return [f"{prefix}_{refs}refs_{self.score_type}"]
            return [f"{prefix}_{self.score_type}"]
        return [f"RDKit_{name}" for name in self.group_names]

    def _calculate_shape_score(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> float:
        """Calculate pairwise score between a query and a reference molecule.

        Parameters
        ----------
        query_mol : Chem.Mol
            Query molecule.
        ref_mol : Chem.Mol
            Reference molecule.

        Returns
        -------
        float
            Similarity score.
        """
        return _score_single_reference(query_mol, ref_mol, self.score_type, self.use_colors)

    @staticmethod
    def _deduplicate_smiles(
        smiles_list: Sequence[str | None],
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """Group identical SMILES strings to avoid redundant 3D embedding.

        Parameters
        ----------
        smiles_list : Sequence[str | None]
            List of input SMILES strings.

        Returns
        -------
        Tuple[List[str], Dict[int, List[int]]]
            (unique_smiles_list, mapping_of_unique_idx_to_original_indices)
        """
        unique_smiles: List[str] = []
        unique_lookup: Dict[str, int] = {}
        unique_to_original: Dict[int, List[int]] = defaultdict(list)

        for idx, smi in enumerate(smiles_list):
            if smi is None:
                continue
            unique_idx = unique_lookup.get(smi)
            if unique_idx is None:
                unique_idx = len(unique_smiles)
                unique_smiles.append(smi)
                unique_lookup[smi] = unique_idx
            unique_to_original[unique_idx].append(idx)

        return unique_smiles, unique_to_original

    def _convert_to_smiles(self, mols: Sequence[Any]) -> List[str | None]:
        """Convert input molecule sequence to SMILES strings.

        Parameters
        ----------
        mols : Sequence[Any]
            SMILES or RDKit molecules.

        Returns
        -------
        List[str | None]
            List of SMILES.
        """
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
        return smiles_list

    def _score_sequential(
        self,
        unique_count: int,
        conformers_by_mol: Dict[int, List[Chem.Mol]],
        num_groups: int,
    ) -> np.ndarray:
        """Score molecules sequentially without multiprocessing overhead.

        Parameters
        ----------
        unique_count : int
            Number of unique molecules to score.
        conformers_by_mol : Dict[int, List[Chem.Mol]]
            Dictionary mapping molecule IDs to their conformers.
        num_groups : int
            Number of reference groups.

        Returns
        -------
        np.ndarray
            Array of shape `(unique_count, num_groups)`.
        """
        scores_unique = np.zeros((unique_count, num_groups))

        for mol_id in range(unique_count):
            mol_conformers = conformers_by_mol.get(mol_id, [])
            if not mol_conformers:
                continue
            group_scores = np.zeros(num_groups)
            for conf_mol in mol_conformers:
                for group_idx, ref_indices in enumerate(self.group_to_indices):
                    for ref_idx in ref_indices:
                        ref_mol = self.reference_mols[ref_idx]
                        score = _score_single_reference(
                            conf_mol, ref_mol, self.score_type, self.use_colors
                        )
                        if score > group_scores[group_idx]:
                            group_scores[group_idx] = score
            scores_unique[mol_id] = group_scores
            if self.show_progress and (mol_id + 1) % 100 == 0:
                print(f"  Scored {mol_id + 1}/{unique_count} unique molecules")

        return scores_unique

    def getScores(
        self,
        mols: Sequence[Any],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Score input molecules against reference query groups using RDKit 3D shape alignment.

        Parameters
        ----------
        mols : Sequence[Any]
            List or sequence of SMILES strings or RDKit molecules.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            2D numpy array of shape `(len(mols), num_groups)` containing alignment scores.
        """
        num_groups = len(self.group_to_indices)
        if num_groups == 0:
            raise ValueError("No reference groups configured")

        if not mols:
            return np.zeros((0, num_groups))

        num_mols = len(mols)
        scores = np.zeros((num_mols, num_groups))

        if self.show_progress:
            print(f"Scoring {num_mols} molecules with {self.getKey()}...")

        smiles_list = self._convert_to_smiles(mols)
        unique_smiles, unique_to_original = self._deduplicate_smiles(smiles_list)

        if not unique_smiles:
            return scores

        with tempfile.TemporaryDirectory() as tmpdir:
            conf_file = self.conformer_generator.genConformers(unique_smiles, tmpdir)
            if not os.path.exists(conf_file):
                if self.show_progress:
                    print("Warning: conformer generation failed")
                return scores

            conformers_by_mol: Dict[int, List[Chem.Mol]] = defaultdict(list)
            try:
                suppl = Chem.SDMolSupplier(conf_file, removeHs=False)
                if suppl is None:
                    if self.show_progress:
                        print(f"Warning: Could not open SDF file: {conf_file}")
                    return scores
            except Exception as exc:
                if self.show_progress:
                    print(f"Warning: Failed to open conformer file {conf_file}: {exc}")
                return scores

            for conf_mol in suppl:
                if conf_mol is None:
                    continue
                try:
                    name = conf_mol.GetProp("_Name")
                    parts = name.split("+")[0].split("_")
                    if len(parts) < 2:
                        continue
                    mol_id = int(parts[1])
                    conformers_by_mol[mol_id].append(conf_mol)
                except (KeyError, ValueError, IndexError) as exc:
                    if self.show_progress:
                        print(f"Warning: Could not parse conformer name: {exc}")
                    continue

            unique_count = len(unique_smiles)
            scores_unique = np.zeros((unique_count, num_groups))

            if self.n_jobs == 1:
                scores_unique = self._score_sequential(unique_count, conformers_by_mol, num_groups)
            else:
                worker_args = [
                    (mol_id, conformers_by_mol.get(mol_id, []))
                    for mol_id in range(unique_count)
                ]
                effective_jobs = max(1, self.n_jobs)
                chunksize = max(1, unique_count // (effective_jobs * 4))

                try:
                    with Pool(
                        self.n_jobs,
                        initializer=_rdkit_worker_init,
                        initargs=(
                            self.reference_mols,
                            self.group_to_indices,
                            self.score_type,
                            self.use_colors,
                        ),
                    ) as pool:
                        if self.show_progress:
                            try:
                                from tqdm import tqdm

                                results = list(
                                    tqdm(
                                        pool.imap(
                                            _score_molecule_rdkit_worker,
                                            worker_args,
                                            chunksize=chunksize,
                                        ),
                                        total=len(worker_args),
                                        desc="Scoring unique molecules",
                                    )
                                )
                            except ImportError:
                                results = pool.map(
                                    _score_molecule_rdkit_worker,
                                    worker_args,
                                    chunksize=chunksize,
                                )
                                print(f"  Scored {len(worker_args)} unique molecules (parallel)")
                        else:
                            results = pool.map(
                                _score_molecule_rdkit_worker,
                                worker_args,
                                chunksize=chunksize,
                            )
                    for mol_id, group_scores in results:
                        if 0 <= mol_id < unique_count and len(group_scores) == num_groups:
                            scores_unique[mol_id] = np.asarray(group_scores)
                except Exception as exc:
                    if self.show_progress:
                        print(
                            f"Warning: parallel processing failed ({exc}), "
                            "switching to sequential mode"
                        )
                    scores_unique = self._score_sequential(unique_count, conformers_by_mol, num_groups)

        for unique_id, original_indices in unique_to_original.items():
            if unique_id >= scores_unique.shape[0]:
                continue
            for original_idx in original_indices:
                scores[original_idx] = scores_unique[unique_id]

        if self.show_progress:
            non_zero = np.count_nonzero(scores)
            avg_score = scores.mean()
            max_score = scores.max() if scores.size > 0 else 0.0
            print(
                f"Scoring complete. Average score: {avg_score:.3f}, "
                f"Max score: {max_score:.3f}, "
                f"Molecules with score > 0: {non_zero}/{scores.shape[0]}"
            )

        return scores
