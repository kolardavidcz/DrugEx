from __future__ import annotations

import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

import drugex
from drugex.training.scorers import conformer_generators, rocs_rdkit
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

MY_MOLECULE = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
NUM_CONFORMERS = 30
RANDOM_SEED = 42
PRUNE_RMS_THRESH = 0.50
MAX_ENERGY_DELTA = 8.0


class QueryConformerPreparer:
    def __init__(self, references_dir: Path | None = None):
        if references_dir is None:
            self.REFERENCES_DIR = Path(__file__).resolve().parent / "references"
        else:
            self.REFERENCES_DIR = references_dir
        self.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_SDF = self.REFERENCES_DIR / "my_molecule_3d.sdf"

    def generate_reference_sdf(self, smiles: str = MY_MOLECULE, output_path: Path | None = None):
        target_path = output_path or self.OUTPUT_SDF
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = RANDOM_SEED
        params.pruneRmsThresh = PRUNE_RMS_THRESH

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFORMERS, params=params)
        if len(conf_ids) == 0:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {smiles}")

        energies = []
        for cid in conf_ids:
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                energy = AllChem.UFFGetMoleculeForceField(mol, confId=cid).CalcEnergy()
            else:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
                energy = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid).CalcEnergy()
            energies.append((energy, cid))

        energies.sort(key=lambda x: x[0])
        min_energy = energies[0][0]
        valid_cids = [cid for energy, cid in energies if energy - min_energy <= MAX_ENERGY_DELTA]

        writer = Chem.SDWriter(str(target_path))
        writer.SetKekulize(True)
        for cid in valid_cids:
            mol_conf = Chem.Mol(mol)
            mol_conf.SetProp("_Name", f"my_molecule_conf_{cid}")
            mol_conf.SetProp("Energy", f"{energies[cid][0]:.4f}")
            writer.write(mol_conf, confId=cid)
        writer.close()
        return target_path

    def test_rocs_loading(self, target_path: Path | None = None):
        sdf_path = target_path or self.OUTPUT_SDF
        if not sdf_path.exists():
            raise FileNotFoundError(f"Reference SDF missing at: {sdf_path}")

        conf_gen = RDKitConformerGenerator(
            max_conformers=2,
            max_isomers=1,
            num_threads=0,
            timeout=10,
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

        test_mols = ["c1ccccc1", MY_MOLECULE]
        scores = rocs_scorer.getScores(test_mols)
        return scores


if __name__ == "__main__":
    preparer = QueryConformerPreparer()
    sdf_path = preparer.generate_reference_sdf()
    scores = preparer.test_rocs_loading(sdf_path)
    print("Reference SDF successfully generated at:", sdf_path)
    print("Test ROCS TanimotoCombo scores [benzene, my_molecule]:", scores)
