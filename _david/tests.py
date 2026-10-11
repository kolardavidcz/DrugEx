import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from drugex.training.rewards import WeightedSum
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.analysis import AlignedDepicter, ParetoFrontier
from _david.code.environment import create_environment, generate_reference_conformers
from _david.code.pipeline import MoleculePipeline


class TestLBDDPipeline(unittest.TestCase):
    """Test suite for clean LBDD environment, scoring, and analysis."""

    def test_environment_creation(self) -> None:
        """Verifies Level 1 environment creation with ROCS and SA scorers."""
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
            self.assertEqual(epsilon, 0.0)
            self.assertIsInstance(env.rewardScheme, WeightedSum)

    def test_conformer_generation(self) -> None:
        """Verifies 3D conformer ensemble generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "test_confs.sdf"
            out = generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=5)
            self.assertTrue(out.exists())
            suppl = Chem.SDMolSupplier(str(out), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertGreater(len(mols), 0)

    def test_pareto_ranking(self) -> None:
        """Verifies Pareto non-dominated frontier ranking."""
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
        """Verifies aligned 2D depiction grid generation."""
        depicter = AlignedDepicter("c1ccccc1O")
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "aligned.png"
            img = depicter.create_aligned_grid(["c1ccccc1CC"], legends=["Lead 1"], output_path=img_path)
            self.assertIsNotNone(img)
            self.assertTrue(img_path.exists())

    def test_pipeline_benchmark(self) -> None:
        """Verifies end-to-end pipeline in benchmark_only mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "run_test"
            pipeline = MoleculePipeline(output_dir=pipe_dir, mode="benchmark_only")
            pipeline.run("Oc1ccccc1")
            self.assertTrue((pipe_dir / "pipeline_status.json").exists())
    def test_visualization_suite(self) -> None:
        """Verifies all publication visualization generators in visualization module."""
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

            # 1. 3D reference superposition
            ref_sdf = tmp / "ref.sdf"
            generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=3)
            ref_png = render_reference_ensemble(ref_sdf, output_png=tmp / "ref_super.png")
            self.assertTrue(ref_png.exists())

            # 2. RL Training curves
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

            # 3. Epoch Pareto evolution
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

            # 4. Property distributions
            dist_df = pd.DataFrame({
                "SMILES": ["c1ccccc1", "c1ccccc1O", "c1ccccc1CC"],
                "ROCS": [0.5, 0.7, 0.6],
                "SA": [1.0, 0.9, 0.8],
            })
            dist_png = render_property_distributions(dist_df, output_png=tmp / "dists.png", ref_mol="c1ccccc1O")
            self.assertIsNotNone(dist_png)
            self.assertTrue(dist_png.exists())

            # 5. Lead structures 2D
            lead_png = render_lead_structures_2d(["c1ccccc1O", "c1ccccc1CC"], legends=["L1", "L2"], output_png=tmp / "leads.png")
            self.assertIsNotNone(lead_png)
            self.assertTrue(lead_png.exists())

            # 6. Pareto frontier
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
        """Verify adaptive RMSD cutoff bounds and literature expectations."""
        from _david.code.environment import get_adaptive_rmsd_cutoff
        # 1. EGCG query molecule (3 rotatable bonds) -> 1.25 Å (Hawkins 2017)
        m_egcg = Chem.MolFromSmiles("C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O")
        self.assertEqual(get_adaptive_rmsd_cutoff(m_egcg), 1.25)

        # 2. Rigid benzene (0 rotatable bonds) -> 1.25 Å (Hawkins 2017 standard baseline)
        m_rigid = Chem.MolFromSmiles("c1ccccc1")
        self.assertEqual(get_adaptive_rmsd_cutoff(m_rigid), 1.25)

        # 3. Flexible chain (10 rotatable bonds) -> 1.50 Å (Friedrich 2017)
        m_flex = Chem.MolFromSmiles("CCCCCCCCCCC(=O)O")
        self.assertEqual(get_adaptive_rmsd_cutoff(m_flex), 1.50)


if __name__ == "__main__":
    unittest.main()

