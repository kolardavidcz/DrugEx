"""Prepares low-energy 3D conformer ensembles and USRCAT/ElectroShape descriptors for LBDD seed ligands.

Uses ETKDGv3 (CSD experimental torsional potentials) and MMFF94 force field minimization,
applying heavy-atom RMSD pruning and thermodynamic energy window filtering.
Pre-computes 60-D USRCAT and 75-D ElectroShape moments for alignment-free 3D similarity evaluations.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hashlib
#? [NumPy] Array operations for saving and loading pre-computed moment vectors.
import numpy as np
#? [RDKit] Cheminformatics foundation for molecule parsing, conformer embedding, force-field optimization, and SDF I/O.
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.fallback_monitor import FallbackMonitor

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
NUM_CONFORMERS: int = 50
RANDOM_SEED: int = 42
PRUNE_RMS_THRESH: float = 0.50
MAX_ENERGY_DELTA: float = 8.0


def get_seed_hash(smiles: str, length: int = 8) -> str:
    """Compute deterministic short hash for query seed SMILES.

    Args:
        smiles: Input SMILES string.
        length: Length of hex hash string to return (default: 8).

    Returns:
        Deterministic hex hash string.
    """
    mol = Chem.MolFromSmiles(smiles)
    can_smi = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else smiles.strip()
    return hashlib.sha256(can_smi.encode("utf-8")).hexdigest()[:length]


class QueryConformerPreparer:
    """Prepares low-energy 3D conformer ensembles and USRCAT/ElectroShape descriptors for LBDD seed ligands.

    Uses ETKDGv3 (CSD experimental torsional potentials) and MMFF94 force field
    minimization, applying heavy-atom RMSD pruning and thermodynamic energy window filtering.
    Additionally pre-computes 60-D USRCAT and 75-D ElectroShape shape/pharmacophore moments to enable
    ultrafast alignment-free 3D similarity evaluations.

    Attributes:
        REFERENCES_DIR: Path to directory where reference SDFs and moment arrays are stored.
        OUTPUT_SDF: Path to default output multi-conformer SDF file.
        OUTPUT_USRCAT: Path to precomputed reference USRCAT moments array (.npy).
        OUTPUT_ELECTROSHAPE: Path to precomputed reference ElectroShape moments array (.npy).
    """

    get_seed_hash = staticmethod(get_seed_hash)

    def __init__(self, references_dir: Path | None = None) -> None:
        """Initialize QueryConformerPreparer.

        Args:
            references_dir: Optional custom path for storing reference SDF and moment files.
        """
        if references_dir is None:
            self.REFERENCES_DIR: Path = SCRIPT_DIR.parent / "references"
        else:
            self.REFERENCES_DIR = Path(references_dir)
        self.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_SDF: Path = self.REFERENCES_DIR / "my_molecule_3d.sdf"
        self.OUTPUT_USRCAT: Path = self.REFERENCES_DIR / "my_molecule_usrcat.npy"
        self.OUTPUT_ELECTROSHAPE: Path = self.REFERENCES_DIR / "my_molecule_electroshape.npy"

    def generate_reference_sdf(self, smiles: str = MY_MOLECULE, output_path: Path | None = None) -> Path:
        """Generate a low-energy 3D conformer ensemble, USRCAT, and ElectroShape moments for query SMILES.

        Args:
            smiles: SMILES string of the query active ligand.
            output_path: Optional destination Path for writing the .sdf file.

        Returns:
            Path to the written .sdf file.

        Raises:
            ValueError: If SMILES cannot be parsed into an RDKit Mol.
            RuntimeError: If conformer embedding yields zero conformers.
        """
        target_path: Path = output_path or self.OUTPUT_SDF
        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        mol: Chem.Mol | None = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        #? [RDKit] Adds explicit hydrogen vertices to 2D graph: required for 3D force-field charges, van der Waals radii, and H-bond donors.
        mol = Chem.AddHs(mol)

        #? [RDKit ETKDGv3] Experimental-Torsion Knowledge Distance Geometry: incorporates CSD dihedral potentials.
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        #? [RDKit] Heavy-atom Kabsch RMSD threshold: prunes duplicate conformers with RMSD < 0.5 Å on-the-fly during stochastic embedding.
        params.pruneRmsThresh = PRUNE_RMS_THRESH

        #? [RDKit Distance Geometry] Embeds 3D coordinates via Floyd-Warshall triangle bounds smoothing + metric matrix PCA projection.
        conf_ids: List[int] = list(AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params))
        if len(conf_ids) == 0:
            #? [RDKit Distance Geometry Fallback] Relax distance constraints with random coordinates if initial projection encounters metric matrix bounds violations.
            FallbackMonitor.record_fallback(
                fallback_type="conformer_embedding_random_coords",
                message=f"Standard ETKDGv3 embedding yielded 0 conformers for seed {smiles}; falling back to useRandomCoords=True",
                context={"smiles": smiles, "num_conformers": NUM_CONFORMERS},
            )
            params.useRandomCoords = True
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params))
            if len(conf_ids) == 0:
                FallbackMonitor.record_fallback(
                    fallback_type="conformer_embedding_failure",
                    message=f"Conformer embedding failed completely for seed: {smiles}",
                    context={"smiles": smiles},
                )
                raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {smiles}")

        energies: List[Tuple[float, int]] = []
        for cid in conf_ids:
            #? [RDKit MMFF94] Computes Merck Molecular Force Field atom types and partial charges; returns None if elements unparameterized.
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                #? [RDKit UFF Fallback] Universal Force Field fallback relaxation using generalized Lennard-Jones/torsion parameters.
                FallbackMonitor.record_fallback(
                    fallback_type="forcefield_uff_fallback",
                    message="MMFF94 unparameterized for molecule; falling back to UFF force field",
                    context={"smiles": smiles, "conf_id": cid},
                )
                res = AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                energy = float(res)
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if ff is not None:
                    ff.Minimize(maxIts=500)
                    energy = float(ff.CalcEnergy())
                else:
                    FallbackMonitor.record_fallback(
                        fallback_type="forcefield_uff_fallback",
                        message="MMFF forcefield initialization returned None; falling back to UFF",
                        context={"smiles": smiles, "conf_id": cid},
                    )
                    AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                    energy = 0.0
            energies.append((energy, cid))

        energies.sort(key=lambda x: x[0])
        min_e: float = energies[0][0]
        #? [Thermodynamic Filtering] Retains conformers within Boltzmann thermal energy window (delta <= 8.0 kcal/mol) of global minimum.
        accepted_cids: List[int] = [cid for e, cid in energies if (e - min_e) <= MAX_ENERGY_DELTA]

        target_path.parent.mkdir(parents=True, exist_ok=True)
        #? [RDKit SDWriter] Serializes accepted 3D Cartesian coordinates into MDL multi-record SDF v2000 structure blocks.
        writer = Chem.SDWriter(str(target_path))
        for cid in accepted_cids:
            writer.write(mol, confId=cid)
        writer.close()

        # Compute Gasteiger charges once for ElectroShape moment evaluations
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception:
            pass

        # Precompute 60-D USRCAT and 75-D ElectroShape moments for cached alignment-free scoring
        usrcat_moments: List[np.ndarray] = []
        electroshape_moments: List[np.ndarray] = []
        for cid in accepted_cids:
            # 60-D USRCAT moments
            usrcat_vec = np.array(rdMolDescriptors.GetUSRCAT(mol, confId=cid), dtype=np.float64)
            usrcat_moments.append(usrcat_vec)

            # 75-D ElectroShape moments
            conf = mol.GetConformer(cid)
            coords = conf.GetPositions()
            charges = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
            for i, atom in enumerate(mol.GetAtoms()):
                try:
                    val = float(atom.GetProp("_GasteigerCharge"))
                    charges[i] = 0.0 if np.isnan(val) or np.isinf(val) else val
                except Exception:
                    charges[i] = 0.0

            c_geom = np.mean(coords, axis=0)
            pos_mask = charges > 1e-4
            c_pos = np.sum(coords[pos_mask] * charges[pos_mask, np.newaxis], axis=0) / np.sum(charges[pos_mask]) if np.any(pos_mask) else c_geom
            neg_mask = charges < -1e-4
            c_neg = np.sum(coords[neg_mask] * np.abs(charges[neg_mask, np.newaxis]), axis=0) / np.sum(np.abs(charges[neg_mask])) if np.any(neg_mask) else c_geom
            p_pos = coords[int(np.argmax(charges))]
            p_neg = coords[int(np.argmin(charges))]

            es_extra: List[float] = []
            for p in [c_geom, c_pos, c_neg, p_pos, p_neg]:
                dists = np.linalg.norm(coords - p, axis=1)
                mu1 = float(np.mean(dists))
                mu2 = float(np.var(dists))
                std = np.sqrt(mu2)
                mu3 = float(np.mean(((dists - mu1) / std) ** 3)) if std > 1e-6 else 0.0
                es_extra.extend([mu1, mu2, mu3])

            full_es = np.concatenate([usrcat_vec, np.array(es_extra, dtype=np.float64)])
            electroshape_moments.append(full_es)

        # Save cached moments to disk alongside SDF
        if usrcat_moments:
            u_arr = np.vstack(usrcat_moments)
            np.save(self.OUTPUT_USRCAT, u_arr)
            target_usrcat = target_path.with_name(target_path.stem + "_usrcat.npy")
            if target_usrcat.resolve() != self.OUTPUT_USRCAT.resolve():
                np.save(target_usrcat, u_arr)

        if electroshape_moments:
            es_arr = np.vstack(electroshape_moments)
            np.save(self.OUTPUT_ELECTROSHAPE, es_arr)
            target_electroshape = target_path.with_name(target_path.stem + "_electroshape.npy")
            if target_electroshape.resolve() != self.OUTPUT_ELECTROSHAPE.resolve():
                np.save(target_electroshape, es_arr)

        return target_path



    def extract_seed_core_scaffold(self, smiles: str = MY_MOLECULE, output_path: Path | None = None) -> str:
        """Extract canonical Bemis-Murcko core scaffold of query seed ligand.

        Args:
            smiles: SMILES string of active seed query.
            output_path: Optional Path to save scaffold TSV.

        Returns:
            Canonical SMILES string of the core scaffold.

        Raises:
            ValueError: If input SMILES cannot be parsed into an RDKit Mol.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse query SMILES: {smiles}")

        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            if scaff is None or scaff.GetNumHeavyAtoms() == 0:
                scaff_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                scaff_smi = Chem.MolToSmiles(scaff, isomericSmiles=False)
        except Exception:
            scaff_smi = Chem.MolToSmiles(mol, isomericSmiles=False)

        target = output_path or (self.REFERENCES_DIR / "seed_core_scaffold.tsv")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write("Scaffold\tSMILES\n")
            f.write(f"{scaff_smi}\t{smiles}\n")

        return scaff_smi
