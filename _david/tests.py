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
            self.assertTrue((pipe_dir / "references" / "query_molecule_3d.sdf").exists())


if __name__ == "__main__":
    unittest.main()
