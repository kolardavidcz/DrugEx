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

#? [NumPy] Array operations for saving and loading pre-computed moment vectors.
import numpy as np
#? [RDKit] Cheminformatics foundation for molecule parsing, conformer embedding, force-field optimization, and SDF I/O.
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from _david.code.fallback_monitor import FallbackMonitor

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
NUM_CONFORMERS: int = 50
RANDOM_SEED: int = 42
PRUNE_RMS_THRESH: float = 0.50
MAX_ENERGY_DELTA: float = 8.0


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
            np.save(self.OUTPUT_USRCAT, np.vstack(usrcat_moments))
        if electroshape_moments:
            np.save(self.OUTPUT_ELECTROSHAPE, np.vstack(electroshape_moments))

        return target_path
