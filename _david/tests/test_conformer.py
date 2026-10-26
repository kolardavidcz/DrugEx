from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem

from _david.code.conformer.analyzer import ConformerConvergenceAnalyzer
from _david.code.conformer.generator import ConformerGenerator
from _david.code.conformer.io import ConformerIOManager
from _david.code.conformer.physics import (
    ForceFieldVariant,
    PhysicalVariantFactory,
)
from _david.code.conformer.visualization import ConformerVisualizer


class TestConformerArchitecture(unittest.TestCase):
    """Unit test suite for modular conformer generation, physics, analysis, I/O, and visualization."""

    # Simple probe molecule (aspirin) for rapid, deterministic test execution
    ASPIRIN_SMILES: str = "CC(=O)Oc1ccccc1C(=O)O"

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_physical_variants_factory(self) -> None:
        """Verify the standard benchmark suite returns 5 distinct physical environments."""
        suite = PhysicalVariantFactory.get_benchmark_suite()
        self.assertEqual(len(suite), 5)
        names = [v.name for v in suite]
        self.assertIn("MMFF94s_Vacuum", names)
        self.assertIn("UFF_Vacuum", names)
        self.assertIn("MMFF94s_Pocket", names)
        self.assertIn("MMFF94s_Water", names)
        self.assertIn("MMFF94s_DistanceDep", names)

        pocket = next(v for v in suite if v.name == "MMFF94s_Pocket")
        self.assertEqual(pocket.dielectric_constant, 6.5)
        self.assertTrue(pocket.is_solvent_shielded())

    def test_conformer_generator_embedding_and_clustering(self) -> None:
        """Verify embedding, minimization under MMFF94s dielectric, and Butina clustering."""
        gen = ConformerGenerator(self.ASPIRIN_SMILES)
        variant = ForceFieldVariant(
            name="MMFF_Test",
            display_title="MMFF Test",
            ff_type="mmff",
            dielectric_model=1,
            dielectric_constant=6.5,
            energy_window_kcal=8.0,
            rmsd_cutoff=1.00,
        )

        mol, cids, energies = gen.build_ensemble(variant=variant, n_raw=20, seed=42)
        self.assertGreater(len(cids), 0)
        self.assertGreaterEqual(mol.GetNumConformers(), len(cids))
        for cid in cids:
            self.assertIn(cid, energies)
            self.assertIsInstance(energies[cid], float)

    def test_conformer_io_manager_sdf_and_pdb_export(self) -> None:
        """Verify multi-model SDF and PDB export format for VS Code molecular visualizers."""
        gen = ConformerGenerator(self.ASPIRIN_SMILES)
        variant = ForceFieldVariant(
            name="MMFF_Export",
            display_title="MMFF Export",
            ff_type="mmff",
            dielectric_constant=1.0,
            rmsd_cutoff=0.8,
        )
        mol, cids, energies = gen.build_ensemble(variant=variant, n_raw=10, seed=42)
        rg_values = {
            cid: ConformerConvergenceAnalyzer.compute_radius_of_gyration(mol, cid)
            for cid in cids
        }

        io_mgr = ConformerIOManager(self.output_dir)
        sdf_path = io_mgr.export_sdf(
            mol=mol,
            conf_ids=cids,
            energies=energies,
            rg_values=rg_values,
            filename="aspirin_test.sdf",
        )
        pdb_path = io_mgr.export_pdb(
            mol=mol,
            conf_ids=cids,
            filename="aspirin_test.pdb",
        )

        self.assertTrue(sdf_path.is_file())
        self.assertTrue(pdb_path.is_file())

        # Inspect SDF content properties
        suppl = Chem.SDMolSupplier(str(sdf_path))
        exported_mols = [m for m in suppl if m is not None]
        self.assertEqual(len(exported_mols), len(cids))
        first_mol = exported_mols[0]
        self.assertTrue(first_mol.HasProp("Energy_kcal_mol"))
        self.assertTrue(first_mol.HasProp("Radius_of_Gyration_A"))

        # Inspect PDB MODEL/ENDMDL records
        pdb_text = pdb_path.read_text(encoding="utf-8")
        self.assertIn("MODEL        1", pdb_text)
        self.assertIn("ENDMDL", pdb_text)
        self.assertTrue(pdb_text.strip().endswith("END"))

    def test_analyzer_and_grid_scan(self) -> None:
        """Verify Radius of Gyration calculation and ROCS surface scan output structure."""
        gen = ConformerGenerator(self.ASPIRIN_SMILES)
        variant = ForceFieldVariant(
            name="MMFF_Grid",
            display_title="MMFF Grid",
            ff_type="mmff",
            dielectric_constant=1.0,
            rmsd_cutoff=0.8,
        )
        ref_mol, ref_cids, _ = gen.build_ensemble(variant, n_raw=10, seed=1)
        probe_mol, probe_cids, _ = gen.build_ensemble(variant, n_raw=10, seed=2)

        analyzer = ConformerConvergenceAnalyzer()
        rg = analyzer.compute_radius_of_gyration(ref_mol, ref_cids[0])
        self.assertGreater(rg, 0.5)
        self.assertLess(rg, 10.0)

        df = analyzer.run_surface_scan(
            ref_mol=ref_mol,
            ref_cids=ref_cids,
            query_mol=probe_mol,
            query_cids=probe_cids,
            k_ref_grid=[1, 2],
            k_gen_grid=[1, 2],
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertIn("tanimoto_combo", df.columns)
        self.assertIn("shape_tanimoto", df.columns)
        self.assertIn("color_tanimoto", df.columns)
        for tc in df["tanimoto_combo"]:
            self.assertGreaterEqual(tc, 0.0)
            self.assertLessEqual(tc, 2.05)


if __name__ == "__main__":
    unittest.main()
