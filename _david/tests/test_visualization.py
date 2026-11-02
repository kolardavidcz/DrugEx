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
    render_antishrinkage_diagnostic,
    render_lead_to_seed_3d_superposition,
)
from _david.code.schema import std


class TestVisualizationSuite(unittest.TestCase):
    def test_compute_2d_skyline(self) -> None:
        # Points: (1, 1), (2, 2), (1.5, 3), (3, 1.5)
        # Dominance (max x, max y):
        # (1, 1) dominated by (2, 2)
        # Non-dominated: (1.5, 3), (2, 2) - wait (2, 2) is dominated by nothing?
        # At x=2, y=2: (1.5, 3) has higher y but lower x.
        # At x=3, y=1.5: higher x but lower y.
        # So non-dominated: (1.5, 3), (2, 2), (3, 1.5).
        x = np.array([1.0, 2.0, 1.5, 3.0])
        y = np.array([1.0, 2.0, 3.0, 1.5])
        sx, sy = compute_2d_skyline(x, y, maximize_x=True, maximize_y=True)
        self.assertNotIn(1.0, sx)  # (1, 1) is dominated
        self.assertIn(1.5, sx)
        self.assertIn(3.0, sx)

    def test_compute_3d_pareto_mask(self) -> None:
        # ROCS maximize, SA minimize, QED maximize
        # Point 0: ROCS=1.5, SA=2.0, QED=0.8 (Excellent)
        # Point 1: ROCS=0.5, SA=5.0, QED=0.3 (Strictly dominated by Point 0)
        # Point 2: ROCS=1.8, SA=3.5, QED=0.5 (Trade-off: better ROCS, worse SA)
        rocs = np.array([1.5, 0.5, 1.8])
        sa = np.array([2.0, 5.0, 3.5])
        qed = np.array([0.8, 0.3, 0.5])
        mask = compute_3d_pareto_mask(rocs, sa, qed)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_aligned_depicter_mcs_fallback(self) -> None:
        # Seed has indole core, candidate has quinoline core -> different Murcko cores
        # But both share benzene ring (6 atoms MCS)
        seed_smi = "c1ccc2[nH]ccc2c1"  # indole
        cand_smi = "c1ccc2ncccc2c1"   # quinoline
        cand_mol = Chem.MolFromSmiles(cand_smi)

        depicter = AlignedDepicter(seed_smi)
        aligned_mol = depicter.align_molecule(cand_mol)
        self.assertIsNotNone(aligned_mol)
        self.assertEqual(aligned_mol.GetNumConformers(), 1)

    def test_antishrinkage_diagnostic_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "test_antishrinkage.png"
            df_after = pd.DataFrame({
                std.VALID: [1, 1, 1],
                std.SMILES: ["c1ccccc1", "c1ccc(CC)cc1", "c1ccccc1C(=O)O"],
                std.ROCS: [1.1, 0.8, 0.95],
                std.SA: [2.0, 3.0, 2.5],
                std.QED: [0.7, 0.6, 0.8],
                std.HAC: [6, 8, 8],
            })
            df_before = pd.DataFrame({
                std.VALID: [1, 1],
                std.SMILES: ["c1ccccc1O", "c1ccccc1N"],
                std.ROCS: [0.7, 0.6],
                std.HAC: [7, 7],
            })
            p = render_antishrinkage_diagnostic(
                df_after=df_after,
                output_png=out_png,
                df_before=df_before,
                ref_smiles="c1ccc(Cl)cc1",  # 7 heavy atoms
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
            self.assertEqual(len(mols), 3)  # seed + 2 leads


if __name__ == "__main__":
    unittest.main()
