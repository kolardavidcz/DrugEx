import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.visualization import (
    compute_2d_skyline,
    compute_3d_pareto_mask,
    AlignedDepicter,
    render_epoch_property_trajectories,
    render_lead_to_seed_3d_superposition,
)
from _david.code.schema import std


class TestVisualizationSuite(unittest.TestCase):
    def test_compute_2d_skyline(self) -> None:
        x = np.array([1.0, 2.0, 1.5, 3.0])
        y = np.array([1.0, 2.0, 3.0, 1.5])
        sx, sy = compute_2d_skyline(x, y, maximize_x=True, maximize_y=True)
        self.assertNotIn(1.0, sx)
        self.assertIn(1.5, sx)
        self.assertIn(3.0, sx)

    def test_compute_3d_pareto_mask(self) -> None:
        rocs = np.array([1.5, 0.5, 1.8])
        sa = np.array([2.0, 5.0, 3.5])
        qed = np.array([0.8, 0.3, 0.5])
        mask = compute_3d_pareto_mask(rocs, sa, qed)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_aligned_depicter_mcs_fallback(self) -> None:
        seed_smi = "c1ccc2[nH]ccc2c1"
        cand_smi = "c1ccc2ncccc2c1"
        cand_mol = Chem.MolFromSmiles(cand_smi)

        depicter = AlignedDepicter(seed_smi)
        aligned_mol = depicter.align_molecule(cand_mol)
        self.assertIsNotNone(aligned_mol)
        self.assertEqual(aligned_mol.GetNumConformers(), 1)

    def test_epoch_property_trajectories_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_epoch_trajectories.png"
            tsv_path = Path(tmpdir) / "test_smiles.tsv"
            df = pd.DataFrame({
                "Epoch": [1, 1, 2, 2],
                "SMILES": ["c1ccccc1", "c1ccc(CC)cc1", "c1ccccc1C(=O)O", "c1ccccc1O"],
                "ROCS": [1.1, 0.8, 0.95, 1.05],
                "SA": [2.0, 3.0, 2.5, 2.2],
            })
            df.to_csv(tsv_path, sep="\t", index=False)

            p = render_epoch_property_trajectories(
                smiles_tsv_path=tsv_path,
                output_png=out_png,
                ref_smiles="c1ccc(Cl)cc1",
            )
            self.assertTrue(out_png.is_file())
            self.assertGreater(out_png.stat().st_size, 1000)

    def test_lead_to_seed_3d_superposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_superposition.png"
            out_sdf = Path(tmpdir) / "test_aligned.sdf"

            seed_m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
            AllChem.EmbedMolecule(seed_m)
            seed_sdf = Path(tmpdir) / "seed.sdf"
            w = Chem.SDWriter(str(seed_sdf))
            w.write(seed_m)
            w.close()

            top_leads = pd.DataFrame({
                std.SMILES: ["c1ccccc1N", "c1ccccc1F"],
                std.ROCS: [1.2, 1.1],
                std.SA: [2.1, 1.9],
                std.QED: [0.8, 0.85],
                std.MW: [93.1, 96.1],
            })

            render_lead_to_seed_3d_superposition(
                top_leads=top_leads,
                ref_sdf_or_mol=seed_sdf,
                output_png=out_png,
                output_sdf=out_sdf,
                max_leads=2,
            )
            self.assertTrue(out_png.is_file())
            self.assertGreater(out_png.stat().st_size, 1000)
            self.assertTrue(out_sdf.is_file())
            suppl = Chem.SDMolSupplier(str(out_sdf), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertEqual(len(mols), 3)


if __name__ == "__main__":
    unittest.main()
