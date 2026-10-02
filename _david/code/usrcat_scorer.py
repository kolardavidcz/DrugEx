"""Pure Ligand-Based 3D Shape and Pharmacophore Scorer using USRCAT and ElectroShape moments."""
from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from drugex.training.scorers.interfaces import ScoreModifier, Scorer
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from _david.code.fallback_monitor import FallbackMonitor


class USRCATScorer(Scorer):
    """Ligand-based 3D shape and pharmacophore scorer using USRCAT and ElectroShape moments."""

    def __init__(
        self,
        references: str | Path | Sequence[Chem.Mol] | np.ndarray,
        num_conformers: int = 10,
        score_type: str = "electroshape",
        modifier: ScoreModifier | None = None,
        random_seed: int = 42,
        max_isomers: int = 1,
        optimize_energy: bool = False,
    ) -> None:
        super().__init__(modifier=modifier)
        self.num_conformers: int = num_conformers
        self.score_type: str = score_type.lower()
        self.random_seed: int = random_seed
        self.max_isomers: int = max_isomers
        self.optimize_energy: bool = optimize_energy
        self.ref_moments: np.ndarray = self._load_reference_moments(references)

    def getKey(self) -> str:
        """Return the unique objective identifier key for this scorer."""
        if self.score_type == "electroshape":
            return "ElectroShape"
        elif self.score_type == "usrcat":
            return "USRCAT"
        return f"USRCAT_{self.score_type.upper()}"

    def _load_reference_moments(self, references: str | Path | Sequence[Chem.Mol] | np.ndarray) -> np.ndarray:
        """Extract and cache 3D moments for all reference conformers."""
        if isinstance(references, np.ndarray):
            if references.ndim == 1:
                return references.reshape(1, -1)
            return references

        if isinstance(references, (str, Path)):
            ref_path = Path(references)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference file not found: {ref_path}")

            if ref_path.suffix.lower() == ".npy":
                #? [NumPy] Load precomputed moment matrix directly.
                arr = np.load(ref_path)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr

            # Check if pre-computed companion .npy moments file exists alongside the SDF
            if ref_path.suffix.lower() in [".sdf", ".sd"]:
                expected_dim = 75 if self.score_type == "electroshape" else 60
                companion_candidates = [
                    ref_path.with_name(f"{ref_path.stem}_{self.score_type}.npy"),
                    ref_path.parent / f"my_molecule_{self.score_type}.npy",
                ]
                for comp_path in companion_candidates:
                    if comp_path.exists():
                        with contextlib.suppress(Exception):
                            arr = np.load(comp_path)
                            if arr.ndim == 1:
                                arr = arr.reshape(1, -1)
                            if arr.shape[1] == expected_dim:
                                return arr

            # Parse reference conformers from multi-record SDF
            suppl = Chem.SDMolSupplier(str(ref_path), removeHs=False)
            ref_mols: list[Chem.Mol] = [m for m in suppl if m is not None]
            if not ref_mols:
                raise ValueError(f"No valid RDKit molecules found in SDF: {ref_path}")
        elif isinstance(references, (list, tuple)):
            ref_mols = list(references)
        else:
            raise TypeError(f"Unsupported references type: {type(references)}")

        moments_list: list[np.ndarray] = []
        for mol in ref_mols:
            if mol is None:
                continue
            # Calculate moments across all conformers of the molecule
            n_confs = mol.GetNumConformers()
            if n_confs == 0:
                continue
            for conf_idx in range(n_confs):
                m = self._calc_moments(mol, conf_id=conf_idx)
                if m is not None:
                    moments_list.append(m)

        if not moments_list:
            raise ValueError("Failed to extract any valid 3D moments from reference conformers.")

        return np.vstack(moments_list)

    def _calc_electroshape_moments(self, mol: Chem.Mol, conf_id: int = -1) -> np.ndarray:
        """Calculate 75-dimensional ElectroShape moments for a single conformer."""
        if not mol.HasProp("_GasteigerChargesComputed"):
            with contextlib.suppress(Exception):
                AllChem.ComputeGasteigerCharges(mol)

        conf = mol.GetConformer(conf_id)
        coords = conf.GetPositions()

        charges: np.ndarray = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
        for i, atom in enumerate(mol.GetAtoms()):
            with contextlib.suppress(Exception):
                val = float(atom.GetProp("_GasteigerCharge"))
                charges[i] = 0.0 if np.isnan(val) or np.isinf(val) else val

        # 1. Geometric centroid
        c_geom = np.mean(coords, axis=0)

        # 2. Positive charge centroid
        pos_mask = charges > 1e-4
        if np.any(pos_mask):
            c_pos = np.sum(coords[pos_mask] * charges[pos_mask, np.newaxis], axis=0) / np.sum(charges[pos_mask])
        else:
            c_pos = c_geom

        # 3. Negative charge centroid
        neg_mask = charges < -1e-4
        if np.any(neg_mask):
            abs_neg = np.abs(charges[neg_mask])
            c_neg = np.sum(coords[neg_mask] * abs_neg[:, np.newaxis], axis=0) / np.sum(abs_neg)
        else:
            c_neg = c_geom

        if not np.any(pos_mask) or not np.any(neg_mask):
            FallbackMonitor.record_fallback(
                fallback_type="electroshape_unpolarized_charges",
                message="Partial charges unpolarized or zero; charge centroids fell back to geometric centroid",
                context={"num_atoms": mol.GetNumAtoms()},
            )

        # 4 & 5. Highest positive and negative charge atom coordinates
        idx_max_pos = int(np.argmax(charges))
        idx_max_neg = int(np.argmin(charges))
        p_max_pos = coords[idx_max_pos]
        p_max_neg = coords[idx_max_neg]

        ref_points = [c_geom, c_pos, c_neg, p_max_pos, p_max_neg]
        es_moments: list[float] = []

        for p in ref_points:
            dists = np.linalg.norm(coords - p, axis=1)
            mu1 = float(np.mean(dists))
            mu2 = float(np.var(dists))
            std = np.sqrt(mu2)
            mu3 = float(np.mean(((dists - mu1) / std) ** 3)) if std > 1e-6 else 0.0
            es_moments.extend([mu1, mu2, mu3])

        usrcat = np.array(rdMolDescriptors.GetUSRCAT(mol, confId=conf_id), dtype=np.float64)
        return np.concatenate([usrcat, np.array(es_moments, dtype=np.float64)])

    def _calc_moments(self, mol: Chem.Mol, conf_id: int = -1) -> np.ndarray | None:
        """Calculate 3D shape/pharmacophore moments for a single molecular conformer."""
        try:
            if self.score_type == "usr":
                usr = rdMolDescriptors.GetUSR(mol, confId=conf_id)
                return np.array(usr, dtype=np.float64)
            elif self.score_type == "electroshape":
                return self._calc_electroshape_moments(mol, conf_id=conf_id)
            else:
                usrcat = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
                return np.array(usrcat, dtype=np.float64)
        except Exception:
            return None

    def _score_single_molecule(self, mol_or_smi: str | Chem.Mol) -> float:
        """Score candidate molecule against reference moment ensemble."""
        if isinstance(mol_or_smi, str):
            mol = Chem.MolFromSmiles(mol_or_smi)
            smi_str = mol_or_smi
        else:
            mol = Chem.Mol(mol_or_smi) if mol_or_smi is not None else None
            smi_str = Chem.MolToSmiles(mol) if mol is not None else "unknown"

        if mol is None:
            return 0.0

        # Enumerate stereoisomers if requested
        if self.max_isomers > 1:
            try:
                from rdkit.Chem.EnumerateStereoisomers import (
                    EnumerateStereoisomers,
                    StereoEnumerationOptions,
                )

                opts = StereoEnumerationOptions(maxIsomers=self.max_isomers, onlyUnassigned=True, rand=self.random_seed)
                target_mols = list(EnumerateStereoisomers(mol, options=opts))
                if not target_mols:
                    target_mols = [mol]
            except Exception:
                target_mols = [mol]
        else:
            target_mols = [mol]

        cand_moments: list[np.ndarray] = []
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.pruneRmsThresh = 0.50
        params.maxIterations = 200
        params.numThreads = 1

        confs_per_iso = max(1, self.num_conformers // len(target_mols))

        for iso_mol in target_mols:
            iso_mol = Chem.AddHs(iso_mol)

            # Embed candidate conformers with standard ETKDGv3
            conf_ids = list(AllChem.EmbedMultipleConfs(iso_mol, numConfs=confs_per_iso, params=params))
            if not conf_ids:
                FallbackMonitor.record_fallback(
                    fallback_type="conformer_embedding_random_coords",
                    message="Standard ETKDGv3 conformer embedding yielded 0 conformers; relaxed bounds with useRandomCoords=True",
                    context={"smiles": smi_str, "num_conformers": confs_per_iso},
                )
                params.useRandomCoords = True
                params.maxIterations = 50
                num_fallback_confs = min(confs_per_iso, 3)
                conf_ids = list(AllChem.EmbedMultipleConfs(iso_mol, numConfs=num_fallback_confs, params=params))
                params.useRandomCoords = False
                params.maxIterations = 200

            if not conf_ids:
                continue

            # Optional MMFF94 force-field relaxation
            if self.optimize_energy:
                for cid in conf_ids:
                    with contextlib.suppress(Exception):
                        res = AllChem.MMFFOptimizeMolecule(iso_mol, confId=cid, maxIters=200)
                        if res == -1:
                            AllChem.UFFOptimizeMolecule(iso_mol, confId=cid, maxIters=200)

            for cid in conf_ids:
                m = self._calc_moments(iso_mol, conf_id=cid)
                if m is not None:
                    cand_moments.append(m)

        if not cand_moments:
            FallbackMonitor.record_fallback(
                fallback_type="conformer_embedding_failure",
                message="Conformer embedding completely failed across all stereoisomers; returning score 0.0",
                context={"smiles": smi_str},
            )
            return 0.0

        cand_mat = np.vstack(cand_moments)
        if cand_mat.shape[1] != self.ref_moments.shape[1]:
            raise ValueError(
                f"Descriptor dimension mismatch: candidate produced {cand_mat.shape[1]} moments, "
                f"but reference ensemble has {self.ref_moments.shape[1]} moments."
            )
        dim = cand_mat.shape[1]

        diff = np.abs(cand_mat[:, np.newaxis, :] - self.ref_moments[np.newaxis, :, :])
        manhattan_dists = np.sum(diff, axis=2) / float(dim)
        sim_matrix = 1.0 / (1.0 + manhattan_dists)
        return float(np.max(sim_matrix))

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol],
        frags: Sequence[str | Chem.Mol] | None = None,
    ) -> np.ndarray:
        """Calculate raw 3D USRCAT similarity scores for a batch of candidate molecules."""
        if not mols:
            return np.array([], dtype=np.float32)
        scores = [self._score_single_molecule(m) for m in mols]
        return np.array(scores, dtype=np.float32)
