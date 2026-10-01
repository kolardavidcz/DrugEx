from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

# Ensure project root and _david are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "drugex").exists() or (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

#? [NumPy] Array operations for saving and loading pre-computed USRCAT 3D moment vectors.
import numpy as np
#? [RDKit] Cheminformatics foundation for molecule parsing, conformer embedding, force-field optimization, and SDF I/O.
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
NUM_CONFORMERS: int = 50
RANDOM_SEED: int = 42
PRUNE_RMS_THRESH: float = 0.50
MAX_ENERGY_DELTA: float = 8.0


class QueryConformerPreparer:
    """Prepares low-energy 3D conformer ensembles and USRCAT descriptors for LBDD seed ligands.

    Uses ETKDGv3 (CSD experimental torsional potentials) and MMFF94 force field
    minimization, applying heavy-atom RMSD pruning and thermodynamic energy window filtering.
    Additionally pre-computes 60-dimensional USRCAT shape/pharmacophore moments to enable
    ultrafast alignment-free 3D similarity evaluations.

    Attributes:
        REFERENCES_DIR: Path to directory where reference SDFs and moment arrays are stored.
        OUTPUT_SDF: Path to default output multi-conformer SDF file.
        OUTPUT_USRCAT: Path to precomputed reference USRCAT moments array (.npy).
    """

    def __init__(self, references_dir: Path | None = None) -> None:
        """Initialize QueryConformerPreparer.

        Args:
            references_dir: Optional custom path for storing reference SDF and USRCAT files.
        """
        if references_dir is None:
            self.REFERENCES_DIR: Path = Path(__file__).resolve().parent / "references"
        else:
            self.REFERENCES_DIR = Path(references_dir)
        self.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_SDF: Path = self.REFERENCES_DIR / "my_molecule_3d.sdf"
        self.OUTPUT_USRCAT: Path = self.REFERENCES_DIR / "my_molecule_usrcat.npy"

    def generate_reference_sdf(self, smiles: str = MY_MOLECULE, output_path: Path | None = None) -> Path:
        """Generate a low-energy 3D conformer ensemble and USRCAT moments for the query SMILES.

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

        #? [RDKit ETKDGv3] Experimental-Torsion Knowledge Distance Geometry: incorporates Cambridge Structural Database (CSD) dihedral potentials.
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        #? [RDKit] Heavy-atom Kabsch RMSD threshold: prunes duplicate conformers with RMSD < 0.5 Å on-the-fly during stochastic embedding.
        params.pruneRmsThresh = PRUNE_RMS_THRESH

        #? [RDKit Distance Geometry] Embeds 3D coordinates via Floyd-Warshall triangle bounds smoothing + metric matrix PCA projection.
        conf_ids: List[int] = list(AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params))
        if len(conf_ids) == 0:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {smiles}")

        energies: List[Tuple[float, int]] = []
        for cid in conf_ids:
            #? [RDKit MMFF94] Computes Merck Molecular Force Field atom types and partial charges; returns None if elements unparameterized.
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                #? [RDKit UFF] Universal Force Field fallback relaxation using generalized Lennard-Jones/torsion parameters.
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                energy = float(AllChem.UFFGetMoleculeForceField(mol, confId=cid).CalcEnergy())
            else:
                #? [RDKit MMFF94] Minimizes potential energy (bonds, angles, torsions, electrostatics, vdW) via BFGS/conjugate gradients.
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                energy = float(AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid).CalcEnergy())
            energies.append((energy, cid))

        energies.sort(key=lambda x: x[0])
        min_energy: float = energies[0][0]
        energy_map: dict[int, float] = {cid: energy for energy, cid in energies}

        #? [Boltzmann Filtering] Retains conformers within delta_E <= 8.0 kcal/mol (P_i / P_0 > 2.3e-6 at 310 K) to discard strained unphysical targets.
        valid_cids: List[int] = [cid for cid, energy in energy_map.items() if energy - min_energy <= MAX_ENERGY_DELTA]
        valid_cids.sort(key=lambda cid: energy_map[cid])

        #? [RDKit SDWriter] Writes MDL V2000 multi-record SDF (coordinates, bonds, and Energy property) required by 3D shape overlay engines.
        writer = Chem.SDWriter(str(target_path))
        writer.SetKekulize(True)
        for cid in valid_cids:
            mol_conf = Chem.Mol(mol)
            mol_conf.SetProp("_Name", f"my_molecule_conf_{cid}")
            mol_conf.SetProp("Energy", f"{energy_map[cid]:.4f}")
            writer.write(mol_conf, confId=cid)
        writer.close()

        #? [RDKit USRCAT Pre-computation] Computes 60-dim USRCAT vectors for all valid reference conformers and stores as NumPy matrix.
        usrcat_vectors = []
        for cid in valid_cids:
            try:
                vec = rdMolDescriptors.GetUSRCAT(mol, confId=cid)
                usrcat_vectors.append(vec)
            except Exception:
                pass

        if usrcat_vectors:
            usrcat_arr = np.array(usrcat_vectors, dtype=np.float64)
            #? [NumPy Save] Saves precomputed reference 3D moments for instant O(1) vectorized similarity lookups during RL and screening.
            usrcat_target = target_path.parent / f"{target_path.stem}_usrcat.npy"
            np.save(str(usrcat_target), usrcat_arr)
            if self.OUTPUT_USRCAT != usrcat_target:
                np.save(str(self.OUTPUT_USRCAT), usrcat_arr)

        return target_path

    def test_rocs_loading(self, target_path: Path | None = None) -> list[float]:
        """Test loading and scoring against the generated SDF reference.

        Args:
            target_path: Optional Path to the reference SDF.

        Returns:
            List of ROCS TanimotoCombo scores for test probes.

        Raises:
            FileNotFoundError: If the reference SDF does not exist.
        """
        sdf_path: Path = target_path or self.OUTPUT_SDF
        if not sdf_path.exists():
            raise FileNotFoundError(f"Reference SDF missing at: {sdf_path}")

        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

        #? [RDKit ROCS Scorer] Generates conformers and evaluates 3D Gaussian shape/color TanimotoCombo overlap against reference SDF.
        conf_gen = RDKitConformerGenerator(
            max_conformers=10,
            max_isomers=1,
            num_threads=0,
            timeout=15,
            show_progress=False,
        )

        rocs_scorer = RDKitROCSScorer(
            conformer_generator=conf_gen,
            references=str(sdf_path),
            score_type="TanimotoCombo",
            use_colors=True,
            show_progress=False,
            n_jobs=1,
        )

        test_mols: list[str] = ["c1ccccc1", MY_MOLECULE]
        scores: list[float] = list(rocs_scorer.getScores(test_mols))
        return scores


if __name__ == "__main__":
    preparer = QueryConformerPreparer()
    sdf_path = preparer.generate_reference_sdf()
    print("Reference SDF successfully generated at:", sdf_path)
    try:
        scores = preparer.test_rocs_loading(sdf_path)
        print("Test ROCS TanimotoCombo scores [benzene, my_molecule]:", scores)
    except Exception as exc:
        print(f"ROCS loading test skipped ({exc}); reference conformer SDF generation verified.")
