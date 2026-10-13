import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.analysis import AlignedDepicter, ParetoFrontier
from _david.code.environment import create_environment, generate_reference_conformers
from _david.code.pipeline import MoleculePipeline


class TestLBDDPipeline(unittest.TestCase):

    def test_environment_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "ref.sdf"
            mol = Chem.MolFromSmiles("c1ccccc1O")
            AllChem.EmbedMolecule(mol)
            writer = Chem.SDWriter(str(ref_sdf))
            writer.write(mol)
            writer.close()

            env, epsilon = create_environment(ref_sdf)
            self.assertEqual(len(env.scorers), 2)
            self.assertEqual(len(env.thresholds), 2)
            self.assertEqual(epsilon, 0.2)
            self.assertIsInstance(env.rewardScheme, (ParetoCrowdingDistance, WeightedSum))

    def test_conformer_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "test_confs.sdf"
            out = generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=5)
            self.assertTrue(out.exists())
            suppl = Chem.SDMolSupplier(str(out), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertGreater(len(mols), 0)

    def test_pareto_ranking(self) -> None:
        pareto = ParetoFrontier()
        df = pd.DataFrame({
            "SMILES": ["mol0", "mol1", "mol2"],
            "ROCS": [1.5, 1.0, 0.8],
            "SA": [2.0, 4.0, 1.5],
            "Valid": [True, True, True],
        })
        ranked = pareto.rank_pareto_fronts(df, objectives=["ROCS", "SA"], maximize=[True, False])
        self.assertEqual(ranked.loc[0, "Pareto_Rank"], 0)
        self.assertGreater(ranked.loc[1, "Pareto_Rank"], 0)

    def test_aligned_depiction(self) -> None:
        depicter = AlignedDepicter("c1ccccc1O")
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "aligned.png"
            img = depicter.create_aligned_grid(["c1ccccc1CC"], legends=["Lead 1"], output_path=img_path)
            self.assertIsNotNone(img)
            self.assertTrue(img_path.exists())

    def test_pipeline_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "run_test"
            pipeline = MoleculePipeline(output_dir=pipe_dir, mode="benchmark_only")
            pipeline.run("Oc1ccccc1")
            self.assertTrue((pipe_dir / "pipeline_status.json").exists())

    def test_visualization_suite(self) -> None:
        from _david.code.visualization import (
            render_epoch_pareto_evolution,
            render_lead_structures_2d,
            render_pareto_frontier,
            render_property_distributions,
            render_reference_ensemble,
            render_training_curves,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            ref_sdf = tmp / "ref.sdf"
            generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=3)
            ref_png = render_reference_ensemble(ref_sdf, output_png=tmp / "ref_super.png")
            self.assertTrue(ref_png.exists())

            fit_tsv = tmp / "fit.tsv"
            pd.DataFrame({
                "Epoch": [1, 2, 3],
                "loss_train": [0.8, 0.5, 0.3],
                "valid_ratio": [0.9, 0.95, 0.98],
                "desired_ratio": [0.1, 0.2, 0.4],
                "unique_ratio": [0.8, 0.75, 0.7],
            }).to_csv(fit_tsv, sep="\t", index=False)
            curves_png = render_training_curves(fit_tsv, output_png=tmp / "curves.png")
            self.assertIsNotNone(curves_png)
            self.assertTrue(curves_png.exists())

            smiles_tsv = tmp / "smiles.tsv"
            pd.DataFrame({
                "Epoch": [1, 1, 2, 2],
                "SMILES": ["c1ccccc1", "c1ccccc1O", "c1ccccc1CC", "c1ccccc1C(=O)O"],
                "ROCS": [0.4, 0.6, 0.5, 0.8],
                "SA": [0.9, 0.85, 0.8, 0.75],
            }).to_csv(smiles_tsv, sep="\t", index=False)
            pareto_evo_png = render_epoch_pareto_evolution(smiles_tsv, output_png=tmp / "pareto_evo.png")
            self.assertIsNotNone(pareto_evo_png)
            self.assertTrue(pareto_evo_png.exists())

            dist_df = pd.DataFrame({
                "SMILES": ["c1ccccc1", "c1ccccc1O", "c1ccccc1CC"],
                "ROCS": [0.5, 0.7, 0.6],
                "SA": [1.0, 0.9, 0.8],
            })
            dist_png = render_property_distributions(dist_df, output_png=tmp / "dists.png", ref_mol="c1ccccc1O")
            self.assertIsNotNone(dist_png)
            self.assertTrue(dist_png.exists())

            lead_png = render_lead_structures_2d(["c1ccccc1O", "c1ccccc1CC"], legends=["L1", "L2"], output_png=tmp / "leads.png")
            self.assertIsNotNone(lead_png)
            self.assertTrue(lead_png.exists())

            leads_df = pd.DataFrame({
                "SMILES": ["c1ccccc1O", "c1ccccc1CC"],
                "ROCS": [0.95, 0.90],
                "Desirability": [0.55, 0.50],
                "MW": [110.0, 120.0],
                "QED": [0.60, 0.65],
                "SA": [2.5, 3.0],
            })
            lead_df_png = render_lead_structures_2d(leads_df, output_png=tmp / "leads_df.png", reference="c1ccccc1")
            self.assertIsNotNone(lead_df_png)
            self.assertTrue(lead_df_png.exists())

            front_df = pd.DataFrame({
                "SMILES": ["m1", "m2"],
                "ROCS": [0.8, 0.5],
                "SA": [0.9, 0.4],
                "Pareto_Rank": [0, 1],
            })
            front_png = render_pareto_frontier(front_df, output_png=tmp / "front.png", rocs_col="ROCS", sa_col="SA")
            self.assertIsNotNone(front_png)
            self.assertTrue(front_png.exists())

    def test_adaptive_rmsd_cutoff(self) -> None:
        from _david.code.environment import get_adaptive_rmsd_cutoff
        m_rigid = Chem.MolFromSmiles("c1ccccc1")
        self.assertEqual(get_adaptive_rmsd_cutoff(m_rigid), 1.25)

    def test_seed_core_scaffold(self) -> None:
        from _david.code.environment import extract_seed_core_scaffold, DEFAULT_MOLECULE

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tsv = Path(tmp_dir) / "scaffold.tsv"

            smi_core = extract_seed_core_scaffold(DEFAULT_MOLECULE, output_path=out_tsv)
            mol_core = Chem.MolFromSmiles(smi_core)
            self.assertEqual(mol_core.GetNumHeavyAtoms(), 10)
            self.assertEqual(mol_core.GetRingInfo().NumRings(), 2)
            self.assertTrue(out_tsv.exists())

            with open(out_tsv, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                row = f.readline().strip().split("\t")
            self.assertEqual(header, "Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings")
            self.assertEqual(row[0], smi_core)
            self.assertEqual(row[2], "core_ring_system")
            self.assertEqual(row[3], "10")
            self.assertEqual(row[4], "2")

            smi_acyclic = extract_seed_core_scaffold("CCCCCC(=O)O")
            self.assertEqual(smi_acyclic, "CCCCCC")

    def test_load_query_molecule(self) -> None:
        from _david.code.environment import load_query_molecule

        mol, is_3d = load_query_molecule("c1ccccc1O")
        self.assertFalse(is_3d)
        self.assertEqual(mol.GetNumHeavyAtoms(), 7)

        # Long valid SMILES (> 255 chars) must not crash with OSError
        mol_long, is_3d = load_query_molecule("C" * 260)
        self.assertFalse(is_3d)
        self.assertEqual(mol_long.GetNumHeavyAtoms(), 260)

        # Invalid long string (> 255 chars) raises ValueError cleanly
        with self.assertRaises(ValueError):
            load_query_molecule("INVALID" * 40)

        mol_input = Chem.MolFromSmiles("c1ccccc1")
        mol_ret, is_3d = load_query_molecule(mol_input)
        self.assertIs(mol_ret, mol_input)
        self.assertFalse(is_3d)

        mol_3d = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol_3d)
        mol_ret_3d, is_3d = load_query_molecule(mol_3d)
        self.assertTrue(is_3d)

        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = Path(tmp_dir) / "query.txt"
            txt_path.write_text("c1ccccc1O  phenol\n", encoding="utf-8")
            mol_from_file, is_3d = load_query_molecule(txt_path)
            self.assertFalse(is_3d)
            self.assertEqual(mol_from_file.GetNumHeavyAtoms(), 7)

            sdf_path = Path(tmp_dir) / "query.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol_3d)
            writer.close()
            mol_from_sdf, is_3d = load_query_molecule(sdf_path)
            self.assertTrue(is_3d)

            empty_path = Path(tmp_dir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_query_molecule(empty_path)

    def test_flat_2d_conformer_rejection(self) -> None:
        from _david.code.environment import load_query_molecule

        mol_2d = Chem.MolFromSmiles("c1ccccc1")
        AllChem.Compute2DCoords(mol_2d)
        ret_mol, is_3d = load_query_molecule(mol_2d)
        self.assertFalse(is_3d, "Planar 2D conformer with z=0.0 must not be flagged as 3D")

    def test_unparameterized_element_uff_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_sdf = Path(tmp_dir) / "boronic.sdf"
            # Boronic acid is not parameterized in MMFF94 and triggers status -1
            res = generate_reference_conformers("OB(O)c1ccccc1", output_path=out_sdf, num_conformers=3)
            self.assertTrue(res.exists())
            suppl = Chem.SDMolSupplier(str(res), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertGreater(len(mols), 0)
            energy = float(mols[0].GetProp("Energy_MMFF_kcal_mol"))
            self.assertGreater(energy, 0.0, "UFF fallback energy must be valid and positive")


if __name__ == "__main__":
    unittest.main()
