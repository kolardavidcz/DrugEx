from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from drugex.training.rewards import WeightedSum
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.config import DEFAULT_MOLECULE, PipelineConfig, RunPaths
from _david.code.environment import (
    PrepareMolecule,
    SafeSAModifier,
    create_environment,
    load_query_molecule,
    load_sdf_file,
    load_smiles_file,
    load_smiles_string,
)
from _david.code.pipeline import MoleculePipeline
from _david.code.schema import std
from _david.code.utils import HardwareManager
from _david.code.visualization import (
    AlignedDepicter,
    ResultAnalyzer,
    render_lead_structures_2d,
    render_property_distributions,
    render_reference_ensemble,
    render_training_curves,
    unscale_to_physical_metrics,
)


class TestLBDDPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.preparer: PrepareMolecule = PrepareMolecule()

    def test_environment_creation(self) -> None:
        """Verify environment creation, multi-objective scorers, thresholds, and reward scheme."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "ref.sdf"
            mol = Chem.MolFromSmiles("c1ccccc1O")
            AllChem.EmbedMolecule(mol)
            writer = Chem.SDWriter(str(ref_sdf))
            writer.write(mol)
            writer.close()

            env, epsilon = create_environment(ref_sdf)
            self.assertGreater(len(env.scorers), 0)
            self.assertEqual(len(env.thresholds), len(env.scorers))
            self.assertGreater(epsilon, 0.0)
            self.assertIsInstance(env.rewardScheme, WeightedSum)

    def test_conformer_generation(self) -> None:
        """Verify 3D conformer generation with MMFF94 and UFF fallback on non-MMFF elements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_sdf = Path(tmp_dir) / "test_confs.sdf"
            out = self.preparer.generate_reference_conformers(
                "c1ccccc1O", output_path=ref_sdf, num_conformers=5
            )
            self.assertTrue(out.exists())
            suppl = Chem.SDMolSupplier(str(out), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertGreater(len(mols), 0)
            self.assertTrue(mols[0].GetConformer().Is3D())
            self.assertTrue(mols[0].HasProp("Energy_MMFF_kcal_mol"))

            boronic_sdf = Path(tmp_dir) / "boronic.sdf"
            res_uff = self.preparer.generate_reference_conformers(
                "OB(O)c1ccccc1", output_path=boronic_sdf, num_conformers=3
            )
            self.assertTrue(res_uff.exists())
            suppl_uff = Chem.SDMolSupplier(str(res_uff), removeHs=False)
            mols_uff = [m for m in suppl_uff if m is not None]
            self.assertGreater(len(mols_uff), 0)
            self.assertTrue(mols_uff[0].GetConformer().Is3D())
            self.assertGreater(
                float(mols_uff[0].GetProp("Energy_MMFF_kcal_mol")), 0.0
            )

    def test_adaptive_rmsd_cutoff(self) -> None:
        """Verify Hawkins (2017) heavy-atom RMSD cutoffs across chemical tiers with topology invariance."""
        m_benzene = Chem.MolFromSmiles("c1ccccc1")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_benzene), (1.00, 1)
        )

        m_aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_aspirin), (1.00, 1)
        )

        m_phenol_h = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_phenol_h), (1.00, 1)
        )

        m_ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_ibuprofen), (1.25, 1)
        )

        m_decanoic = Chem.MolFromSmiles("CCCCCCCCCC(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_decanoic), (1.50, 1)
        )

        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(None), (1.00, 1)
        )
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(Chem.MolFromSmiles("")),
            (1.00, 1),
        )

    def test_seed_core_scaffold(self) -> None:
        """Verify canonical Bemis-Murcko framework extraction preserving rings and linkers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tsv = Path(tmp_dir) / "scaffold.tsv"
            smi_core = self.preparer.extract_seed_core_scaffold(
                DEFAULT_MOLECULE, output_path=out_tsv
            )
            mol_core = Chem.MolFromSmiles(smi_core)
            self.assertIsNotNone(mol_core)
            self.assertEqual(mol_core.GetNumHeavyAtoms(), 25)
            self.assertEqual(mol_core.GetRingInfo().NumRings(), 4)
            self.assertTrue(out_tsv.exists())

            with open(out_tsv, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                row = f.readline().strip().split("\t")
            self.assertEqual(
                header, "Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings"
            )
            self.assertEqual(row[0], smi_core)
            self.assertEqual(row[2], "BemisMurcko")

            self.assertEqual(
                self.preparer.extract_seed_core_scaffold("c1ccccc1O"), "c1ccccc1"
            )
            self.assertEqual(
                self.preparer.extract_seed_core_scaffold("CCCCCC(=O)O"), "CCCCCC(=O)O"
            )

    def test_load_query_molecule(self) -> None:
        """Verify query molecule loading across SMILES strings, RDKit Mols, and files."""
        # 1. SMILES string loading
        mol_smi = load_smiles_string("c1ccccc1O")
        self.assertEqual(mol_smi.GetNumHeavyAtoms(), 7)

        mol, is_3d = load_query_molecule("c1ccccc1O")
        self.assertFalse(is_3d)
        self.assertEqual(mol.GetNumHeavyAtoms(), 7)

        with self.assertRaises(ValueError):
            load_query_molecule("INVALID" * 40)

        # 2. Mol object loading
        mol_input = Chem.MolFromSmiles("c1ccccc1")
        mol_ret, is_3d = load_query_molecule(mol_input)
        self.assertIs(mol_ret, mol_input)
        self.assertFalse(is_3d)

        # Embedded 3D Mol
        mol_3d = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol_3d)
        _, is_3d_emb = load_query_molecule(mol_3d)
        self.assertTrue(is_3d_emb)

        # 3. File inputs (.txt, .sdf)
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
            _, is_3d_sdf = load_query_molecule(sdf_path)
            self.assertTrue(is_3d_sdf)

            empty_path = Path(tmp_dir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_query_molecule(empty_path)

    def test_aligned_depiction(self) -> None:
        """Verify 2D core-aligned grid image generation."""
        depicter = AlignedDepicter("c1ccccc1O")
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "aligned.png"
            img = depicter.create_aligned_grid(
                ["c1ccccc1CC"], legends=["Lead 1"], output_path=img_path
            )
            self.assertIsNotNone(img)
            self.assertTrue(img_path.exists())

    def test_hardware_manager(self) -> None:
        """Verify dynamic CPU allocation, batch sizing, environment isolation, and GPU selection."""
        expected_jobs = max(1, int((os.cpu_count() or 1) * 0.60))
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(), expected_jobs)
        self.assertEqual(
            HardwareManager.get_safe_parallel_jobs(cap=2), min(2, expected_jobs)
        )
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(cap=0), 1)

        self.assertEqual(HardwareManager.get_recommended_batch_size(n_samples=10), 10)
        batch_normal = HardwareManager.get_recommended_batch_size(n_samples=200)
        self.assertIn(batch_normal, [32, 64, 128, 200])

        env_vars = HardwareManager.init_runtime_environment()
        self.assertEqual(os.environ.get("DRUGEX_SCORE_NJOBS"), str(expected_jobs))
        self.assertEqual(os.environ.get("DRUGEX_MP_CONTEXT"), "forkserver")
        self.assertEqual(env_vars["DRUGEX_SCORE_NJOBS"], str(expected_jobs))

        gpus = HardwareManager.get_safe_gpus()
        self.assertIsInstance(gpus, tuple)
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="cpu"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="none"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu=9999), ())

    def test_decoupled_validation_loader(self) -> None:
        """Verify that prepare_seed_fragments returns decoupled train and bounded validation dataloaders."""
        from _david.code.models import GraphTransformerOptimizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir) / "models"
            raw_dir = Path(tmp_dir) / "raw_data"
            optimizer = GraphTransformerOptimizer(models_dir=models_dir, raw_data_dir=raw_dir, gpu="auto")

            train_loader, val_loader, dataset = optimizer.prepare_seed_fragments(
                "c1ccccc1O", n_prompts=20, val_samples=5
            )
            self.assertIsNot(train_loader, val_loader)
            self.assertEqual(len(train_loader.dataset), 20)
            self.assertEqual(len(val_loader.dataset), 5)
            self.assertLessEqual(len(val_loader.dataset), len(train_loader.dataset))

    def test_safe_sa_modifier(self) -> None:
        """Verify SafeSAModifier enforces monotonic synthesizability and guards against uncalculable fallbacks."""
        modifier = SafeSAModifier()

        self.assertGreater(float(modifier(2.0)), float(modifier(4.0)))
        self.assertGreater(float(modifier(4.0)), float(modifier(6.0)))

        for val in [1.0, 2.5, 3.5, 5.0, 8.0, 10.0]:
            score = float(modifier(val))
            self.assertTrue(0.0 <= score <= 1.0, f"Score {score} out of [0, 1] bounds for SA {val}")

        self.assertEqual(float(modifier(0.0)), 0.0)
        self.assertEqual(float(modifier(-1.0)), 0.0)

        input_arr = np.array([0.0, 2.0, 5.0, -1.0])
        res_arr = modifier(input_arr)
        for i, x in enumerate(input_arr):
            self.assertAlmostEqual(res_arr[i], float(modifier(x)), places=5)

    def test_result_analyzer_deduplication_and_lead_prioritization(self) -> None:
        """Verify ResultAnalyzer deduplicates canonical SMILES clones and prioritizes leads."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidates_tsv = tmp_path / "candidates.tsv"

            smi_asp1 = "CC(=O)Oc1ccccc1C(=O)O"
            smi_asp2 = "O=C(O)c1ccccc1OC(=O)C"
            smi_phenol = "c1ccccc1O"
            smi_low = "CCCC"

            df = pd.DataFrame(
                {
                    std.SMILES: [smi_asp1, smi_asp2, smi_phenol, smi_low],
                    std.ROCS: [1.5, 1.5, 1.5, 0.5],
                    std.SA: [2.5, 2.5, 2.5, 5.0],
                    std.VALID: [True, True, True, True],
                }
            )
            df.to_csv(candidates_tsv, sep="\t", index=False)

            analyzer = ResultAnalyzer(tmp_path)
            saved_leads = analyzer.analyze(candidates_tsv, top_k=5)

            asp_canon = Chem.CanonSmiles(smi_asp1)
            asp_rows = saved_leads[saved_leads[std.CANONICAL_SMILES] == asp_canon]
            self.assertEqual(len(asp_rows), 1, "Duplicate canonical SMILES must be collapsed")
            self.assertGreater(saved_leads.iloc[0][std.ROCS], saved_leads.iloc[-1][std.ROCS])

    def test_data_schema_and_unscaling(self) -> None:
        """Verify std constants and physical metric unscaling."""
        self.assertEqual(std.SMILES, "SMILES")
        self.assertEqual(std.ROCS, "ROCS")
        self.assertEqual(std.SA, "SA")
        self.assertEqual(std.VALID, "Valid")

        norm_df = pd.DataFrame(
            {
                std.SMILES: ["c1ccccc1O"],
                std.ROCS: [0.60],
                std.SA: [1.0],
            }
        )
        unscaled = unscale_to_physical_metrics(norm_df)
        self.assertAlmostEqual(unscaled.loc[0, std.ROCS], 1.20, places=5)
        self.assertGreater(unscaled.loc[0, std.SA], 1.0001)

    def test_visualization_suite(self) -> None:
        """Verify end-to-end visualization suite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. Reference 3D Superposition
            ref_sdf = tmp / "ref_ensemble.sdf"
            self.preparer.generate_reference_conformers(
                "c1ccccc1O", output_path=ref_sdf, num_conformers=3
            )
            ref_png = render_reference_ensemble(
                ref_sdf, output_png=tmp / "ref_ensemble.png"
            )
            self.assertIsNotNone(ref_png)
            self.assertTrue(ref_png.exists())

            # 2. Training Curves
            fit_tsv = tmp / "fit.tsv"
            pd.DataFrame(
                {
                    std.EPOCH: [1, 2, 3],
                    std.LOSS_TRAIN: [0.5, 0.4, 0.3],
                    std.VALID_RATIO: [0.8, 0.85, 0.9],
                    std.UNIQUE_RATIO: [0.7, 0.75, 0.8],
                    std.DESIRED_RATIO: [0.2, 0.4, 0.6],
                }
            ).to_csv(fit_tsv, sep="\t", index=False)
            curves_png = render_training_curves(
                fit_tsv, output_png=tmp / "training_curves.png"
            )
            self.assertIsNotNone(curves_png)
            self.assertTrue(curves_png.exists())

            # 3. Property Distributions
            dist_df = pd.DataFrame(
                {
                    std.SMILES: ["c1ccccc1", "c1ccccc1O", "c1ccccc1CC"],
                    std.ROCS: [0.5, 0.7, 0.6],
                    std.SA: [1.0, 0.9, 0.8],
                }
            )
            dist_png = render_property_distributions(
                dist_df, output_png=tmp / "dists.png", ref_mol="c1ccccc1O"
            )
            self.assertIsNotNone(dist_png)
            self.assertTrue(dist_png.exists())

            # 4. Lead Structures 2D Cards
            leads_df = pd.DataFrame(
                {
                    std.SMILES: ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1CC"],
                    std.ROCS: [0.575, 0.90],
                    std.DESIRED: [1.0, 0.50],
                    std.QED: [0.55, 0.65],
                    std.MW: [180.15, 120.0],
                    std.SA: [1.0, 3.0],
                },
                index=[176, 204],
            )
            lead_df_png = render_lead_structures_2d(
                leads_df, output_png=tmp / "leads_df.png", reference="c1ccccc1"
            )
            self.assertIsNotNone(lead_df_png)
            self.assertTrue(lead_df_png.exists())

    def test_pipeline_execution(self) -> None:
        """Verify end-to-end MoleculePipeline execution using test-scale configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "run_test"
            paths = RunPaths.resolve(base_dir=pipe_dir)
            test_config = PipelineConfig(
                epochs=1,
                n_samples=10,
                val_samples=10,
                n_eval_samples=10,
                min_epochs=1,
            )
            pipeline = MoleculePipeline(config=test_config, paths=paths)
            pipeline.run("Oc1ccccc1")
            self.assertTrue(
                (paths.logs_dir / "pipeline_status.json").exists()
            )
            self.assertTrue((paths.root / "summary.md").exists())
            self.assertTrue(
                (paths.raw_data_dir / "query_molecule_3d.sdf").exists()
            )
            self.assertTrue(
                (paths.raw_data_dir / "generated_candidates.tsv").exists()
            )
            self.assertTrue(
                (paths.root / "lead_structures_2d.png").exists()
                or (paths.root / "property_distributions.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
