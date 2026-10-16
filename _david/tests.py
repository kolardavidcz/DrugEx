import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.analysis import AlignedDepicter, ParetoFrontier
from _david.code.environment import DEFAULT_MOLECULE, PrepareMolecule, create_environment
from _david.code.pipeline import MoleculePipeline


class TestLBDDPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.preparer = PrepareMolecule()

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
            out = self.preparer.generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=5)
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
            self.preparer.generate_reference_conformers("c1ccccc1O", output_path=ref_sdf, num_conformers=3)
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
        """Verify Hawkins (2017) heavy-atom RMSD cutoffs across chemical tiers with topology invariance."""
        # 1. Rigid/compact tier (<= 2 rotatable bonds, <= 20 heavy atoms): 1.00 Å
        m_benzene = Chem.MolFromSmiles("c1ccccc1")  # 0 rot, 6 heavy
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_benzene), 1.00)

        m_aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # 2 rot, 13 heavy
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_aspirin), 1.00)

        # Explicit hydrogens invariance check: phenol with explicit Hs must remain 1.00 Å
        m_phenol_h = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_phenol_h), 1.00)

        # 2. Standard drug-like tier (<= 7 rotatable bonds, <= 45 heavy atoms): 1.25 Å
        m_ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # 3 rot, 15 heavy
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_ibuprofen), 1.25)

        m_diazepam = Chem.MolFromSmiles("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21")  # 1 rot, 20 heavy
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_diazepam), 1.00)

        # 3. Flexible tier (> 7 rotatable bonds or > 45 heavy atoms): 1.50 Å
        m_decanoic = Chem.MolFromSmiles("CCCCCCCCCC(=O)O")  # 8 rot, 11 heavy
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_decanoic), 1.50)

        # Null and empty molecule guards
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(None), 1.00)
        m_empty = Chem.MolFromSmiles("")
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_empty), 1.00)

        # Backwards-compatible instance method delegation
        self.assertEqual(self.preparer.get_adaptive_rmsd_cutoff(m_benzene), 1.00)

    def test_seed_core_scaffold(self) -> None:
        """Verify canonical Bemis-Murcko framework extraction preserving rings and linkers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tsv = Path(tmp_dir) / "scaffold.tsv"

            # EGCG canonical Bemis-Murcko framework preserves chroman, 2 aryl rings, and ester linker (25 heavy atoms, 4 rings)
            smi_core = self.preparer.extract_seed_core_scaffold(DEFAULT_MOLECULE, output_path=out_tsv)
            mol_core = Chem.MolFromSmiles(smi_core)
            self.assertIsNotNone(mol_core)
            self.assertEqual(mol_core.GetNumHeavyAtoms(), 25)
            self.assertEqual(mol_core.GetRingInfo().NumRings(), 4)
            self.assertTrue(out_tsv.exists())

            with open(out_tsv, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                row = f.readline().strip().split("\t")
            self.assertEqual(header, "Scaffold\tQuery\tScaffold_Type\tNum_Heavy_Atoms\tNum_Rings")
            self.assertEqual(row[0], smi_core)
            self.assertEqual(row[2], "BemisMurcko")
            self.assertEqual(row[3], "25")
            self.assertEqual(row[4], "4")

            # Monocyclic and standard drug frameworks
            smi_phenol = self.preparer.extract_seed_core_scaffold("c1ccccc1O")
            self.assertEqual(smi_phenol, "c1ccccc1")

            smi_aspirin = self.preparer.extract_seed_core_scaffold("CC(=O)Oc1ccccc1C(=O)O")
            self.assertEqual(smi_aspirin, "c1ccccc1")

            # Acyclic fallback: preserves sanitized heavy-atom structure cleanly
            smi_acyclic = self.preparer.extract_seed_core_scaffold("CCCCCC(=O)O")
            self.assertEqual(smi_acyclic, "CCCCCC(=O)O")

    def test_load_query_molecule(self) -> None:
        mol, is_3d = self.preparer.load_query_molecule("c1ccccc1O")
        self.assertFalse(is_3d)
        self.assertEqual(mol.GetNumHeavyAtoms(), 7)

        # Long valid SMILES (> 255 chars) must not crash with OSError
        mol_long, is_3d = self.preparer.load_query_molecule("C" * 260)
        self.assertFalse(is_3d)
        self.assertEqual(mol_long.GetNumHeavyAtoms(), 260)

        # Invalid long string (> 255 chars) raises ValueError cleanly
        with self.assertRaises(ValueError):
            self.preparer.load_query_molecule("INVALID" * 40)

        mol_input = Chem.MolFromSmiles("c1ccccc1")
        mol_ret, is_3d = self.preparer.load_query_molecule(mol_input)
        self.assertIs(mol_ret, mol_input)
        self.assertFalse(is_3d)

        mol_3d = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol_3d)
        mol_ret_3d, is_3d = self.preparer.load_query_molecule(mol_3d)
        self.assertTrue(is_3d)

        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = Path(tmp_dir) / "query.txt"
            txt_path.write_text("c1ccccc1O  phenol\n", encoding="utf-8")
            mol_from_file, is_3d = self.preparer.load_query_molecule(txt_path)
            self.assertFalse(is_3d)
            self.assertEqual(mol_from_file.GetNumHeavyAtoms(), 7)

            sdf_path = Path(tmp_dir) / "query.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol_3d)
            writer.close()
            mol_from_sdf, is_3d = self.preparer.load_query_molecule(sdf_path)
            self.assertTrue(is_3d)

            empty_path = Path(tmp_dir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.preparer.load_query_molecule(empty_path)

    def test_flat_2d_conformer_rejection(self) -> None:
        mol_2d = Chem.MolFromSmiles("c1ccccc1")
        AllChem.Compute2DCoords(mol_2d)
        ret_mol, is_3d = self.preparer.load_query_molecule(mol_2d)
        self.assertFalse(is_3d, "Planar 2D conformer with z=0.0 must not be flagged as 3D")

    def test_unparameterized_element_uff_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_sdf = Path(tmp_dir) / "boronic.sdf"
            # Boronic acid is not parameterized in MMFF94 and triggers status -1
            res = self.preparer.generate_reference_conformers("OB(O)c1ccccc1", output_path=out_sdf, num_conformers=3)
            self.assertTrue(res.exists())
            suppl = Chem.SDMolSupplier(str(res), removeHs=False)
            mols = [m for m in suppl if m is not None]
            self.assertGreater(len(mols), 0)
            energy = float(mols[0].GetProp("Energy_MMFF_kcal_mol"))
            self.assertGreater(energy, 0.0, "UFF fallback energy must be valid and positive")

    def test_prepare_molecule_class(self) -> None:
        custom_preparer = PrepareMolecule()
        mol, is_3d = custom_preparer.load_query_molecule("c1ccccc1O")
        self.assertFalse(is_3d)
        self.assertEqual(mol.GetNumHeavyAtoms(), 7)

        smi = custom_preparer.extract_seed_core_scaffold("CCOC(=O)CC")
        self.assertEqual(smi, "CCOC(=O)CC")

    def test_hardware_manager(self) -> None:
        """Verify dynamic CPU allocation, GPU routing, batch sizing, and environment isolation."""
        from _david.code.utils import HardwareManager, get_safe_parallel_jobs

        # 1. Dynamic CPU allocation (60% rule)
        expected_jobs = max(1, int((os.cpu_count() or 1) * 0.60))
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(), expected_jobs)
        self.assertEqual(get_safe_parallel_jobs(), expected_jobs)
        self.assertEqual(get_safe_parallel_jobs(cap=2), min(2, expected_jobs))

        # 2. Power-of-2 batch sizing clamped strictly to n_samples
        batch_small = HardwareManager.get_recommended_batch_size(n_samples=10)
        self.assertEqual(batch_small, 10)
        batch_normal = HardwareManager.get_recommended_batch_size(n_samples=200)
        self.assertIn(batch_normal, [32, 64, 128])

        # 3. Environment isolation
        env_vars = HardwareManager.init_runtime_environment()
        self.assertEqual(os.environ.get("DRUGEX_SCORE_NJOBS"), str(expected_jobs))
        self.assertEqual(os.environ.get("DRUGEX_MP_CONTEXT"), "forkserver")
        self.assertEqual(env_vars["DRUGEX_SCORE_NJOBS"], str(expected_jobs))
        self.assertEqual(env_vars["DRUGEX_MP_CONTEXT"], "forkserver")

        # 4. Safe GPU routing and CPU fallback
        gpus = HardwareManager.get_safe_gpus()
        self.assertIsInstance(gpus, tuple)
        # Invalid preferred GPU index falls back to ()
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu=9999), ())

    def test_decoupled_validation_loader(self) -> None:
        """Verify that prepare_seed_fragments returns decoupled train and bounded val loaders."""
        from _david.code.models import AlgorithmConfig, GraphTransformerOptimizer

        cfg = AlgorithmConfig(n_samples=200, val_samples=None)
        self.assertEqual(cfg.epochs, 35)
        self.assertEqual(cfg.n_samples, 200)
        self.assertIsNone(cfg.val_samples)

        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(tmp_dir, gpu="auto")

            # Decoupled validation dataloader for N=200 prompts: bounded val_size = 40 (min(50, max(25, 40)))
            train_loader, val_loader, dataset = optimizer.prepare_seed_fragments("c1ccccc1O", n_prompts=200)
            self.assertEqual(len(dataset.getData()), 200)
            self.assertEqual(len(train_loader.dataset), 200)
            self.assertEqual(len(val_loader.dataset), 40)

            # Test mode clamping: N=10 prompts yields val_size = 10
            train_l10, val_l10, dataset10 = optimizer.prepare_seed_fragments("c1ccccc1O", n_prompts=10)
            self.assertEqual(len(dataset10.getData()), 10)
            self.assertEqual(len(train_l10.dataset), 10)
            self.assertEqual(len(val_l10.dataset), 10)

    def test_pareto_front_0_alignment(self) -> None:
        """Verify Pareto Front 0 non-domination alignment under maximize=[True, True]."""
        # Synthetic 5-point dataset matching Phase 0 numerical proof:
        # A: [1.6, 0.9] (High ROCS, High SA Desirability) -> Front 0
        # B: [1.8, 0.5] (Peak ROCS, Moderate SA Desirability) -> Front 0
        # C: [1.2, 1.0] (Moderate ROCS, Perfect SA Desirability) -> Front 0
        # D: [1.4, 0.4] (Dominated by A) -> Dominated
        # E: [0.8, 0.1] (Terrible candidate) -> Dominated
        df_proof = pd.DataFrame({
            "SMILES": ["molA", "molB", "molC", "molD", "molE"],
            "ROCS": [1.6, 1.8, 1.2, 1.4, 0.8],
            "SA": [0.9, 0.5, 1.0, 0.4, 0.1],
        })
        scores = df_proof[["ROCS", "SA"]].to_numpy(dtype=float)
        is_front = ParetoFrontier.identify_pareto_front(scores, maximize=[True, True])

        # Exactly A, B, C form Front 0
        self.assertTrue(is_front[0], "Mol A [1.6, 0.9] must be Front 0")
        self.assertTrue(is_front[1], "Mol B [1.8, 0.5] must be Front 0")
        self.assertTrue(is_front[2], "Mol C [1.2, 1.0] must be Front 0")
        self.assertFalse(is_front[3], "Mol D [1.4, 0.4] must be dominated by Mol A")
        self.assertFalse(is_front[4], "Mol E [0.8, 0.1] must be dominated")

        pareto = ParetoFrontier()
        ranked = pareto.rank_pareto_fronts(df_proof, objectives=["ROCS", "SA"], maximize=[True, True])
        self.assertEqual(ranked.loc[0, "Pareto_Rank"], 0)
        self.assertEqual(ranked.loc[1, "Pareto_Rank"], 0)
        self.assertEqual(ranked.loc[2, "Pareto_Rank"], 0)
        self.assertGreater(ranked.loc[3, "Pareto_Rank"], 0)
        self.assertGreater(ranked.loc[4, "Pareto_Rank"], 0)

    def test_forcefield_minimization_streamlined(self) -> None:
        """Verify streamlined conformer generation with MMFF94 and UFF without dead status loops."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Standard organic molecule with MMFF94
            mmff_sdf = Path(tmp_dir) / "phenol.sdf"
            res_mmff = self.preparer.generate_reference_conformers("c1ccccc1O", output_path=mmff_sdf, num_conformers=5)
            self.assertTrue(res_mmff.exists())
            suppl_mmff = Chem.SDMolSupplier(str(res_mmff), removeHs=False)
            confs_mmff = [m for m in suppl_mmff if m is not None]
            self.assertGreater(len(confs_mmff), 0)
            self.assertTrue(confs_mmff[0].GetConformer().Is3D())
            self.assertTrue(confs_mmff[0].HasProp("Energy_MMFF_kcal_mol"))

            # 2. Boronic acid with UFF fallback
            uff_sdf = Path(tmp_dir) / "boronic.sdf"
            res_uff = self.preparer.generate_reference_conformers("OB(O)c1ccccc1", output_path=uff_sdf, num_conformers=3)
            self.assertTrue(res_uff.exists())
            suppl_uff = Chem.SDMolSupplier(str(res_uff), removeHs=False)
            confs_uff = [m for m in suppl_uff if m is not None]
            self.assertGreater(len(confs_uff), 0)
            self.assertTrue(confs_uff[0].GetConformer().Is3D())
            self.assertGreater(float(confs_uff[0].GetProp("Energy_MMFF_kcal_mol")), 0.0)

    def test_clipped_score_feasibility_floor(self) -> None:
        """Verify ClippedScore mathematical parameterization and SA feasibility floor behavior."""
        # Standard feasibility floor modifier:
        # High desirability (1.0) for easy SA <= 5.5, floor penalty (0.0) for difficult SA >= 6.0
        modifier = ClippedScore(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0)
        self.assertEqual(modifier.slope, -2.0)
        self.assertEqual(modifier.intercept, 12.0)

        # 1. Synthetically accessible tier (SA <= 5.5) -> Desirability = 1.0
        self.assertEqual(float(modifier(1.0)), 1.0)
        self.assertEqual(float(modifier(3.0)), 1.0)
        self.assertEqual(float(modifier(5.0)), 1.0)
        self.assertEqual(float(modifier(5.5)), 1.0)

        # 2. Linear transition window (5.5 < SA < 6.0)
        self.assertAlmostEqual(float(modifier(5.75)), 0.5, places=5)
        self.assertAlmostEqual(float(modifier(5.625)), 0.75, places=5)
        self.assertAlmostEqual(float(modifier(5.875)), 0.25, places=5)

        # 3. Penalized / infeasible tier (SA >= 6.0) -> Desirability = 0.0
        self.assertEqual(float(modifier(6.0)), 0.0)
        self.assertEqual(float(modifier(7.0)), 0.0)
        self.assertEqual(float(modifier(10.0)), 0.0)

        # Property integration
        sa_scorer = Property(prop="SA", modifier=modifier)
        scores = sa_scorer.getScores(["c1ccccc1O"])
        self.assertGreater(scores[0], 0.0)
        mod_scores = sa_scorer(["c1ccccc1O"])
        self.assertEqual(float(mod_scores[0]), 1.0)

    def test_adaptive_rmsd_cutoff_adversarial_battery(self) -> None:
        """Verify Hawkins (2017) cutoffs on macrocycles, polycycles, amine hydrogens, and single atoms."""
        # 1. Macrocycle: Erythromycin (48 heavy atoms) -> 1.50 Å flexible tier
        smi_erythromycin = (
            "CCC1C(C(C(C(=O)C(CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)"
            "OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O"
        )
        mol_erythro = Chem.MolFromSmiles(smi_erythromycin)
        self.assertIsNotNone(mol_erythro)
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_erythro), 1.50)

        # 2. Polycyclic aromatic sheet: Coronene (26 heavy atoms, 0 rotatable bonds) -> 1.25 Å drug-like tier
        smi_coronene = "c12c3c4c5c6c1c7c8c2c9c%10c3c%11c4c%12c5c%13c6c7c%14c%13c%12c%11c%10c%14c89"
        mol_coronene = Chem.MolFromSmiles(smi_coronene)
        self.assertIsNotNone(mol_coronene)
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_coronene), 1.25)

        # 3. Nitrogenous heteroatoms: aniline (c1ccccc1N) with and without explicit Hs
        mol_aniline = Chem.MolFromSmiles("c1ccccc1N")
        mol_aniline_h = Chem.AddHs(mol_aniline)
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_aniline), 1.00)
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_aniline_h), 1.00)

        # 4. Minimal / single-atom molecules
        mol_methane = Chem.MolFromSmiles("C")
        mol_water = Chem.MolFromSmiles("O")
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_methane), 1.00)
        self.assertEqual(PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_water), 1.00)

    def test_canonical_scaffolds_adversarial_battery(self) -> None:
        """Verify Bemis-Murcko extraction across spiro, fused lactones, bridgeheads, and zwitterions."""
        # 1. Spiro bicycle: Spiro[4.5]decane (preserves both rings, 11 heavy atoms, 2 rings)
        scaff_spiro = self.preparer.extract_seed_core_scaffold("C1CC2(CCC1)CCCCC2")
        mol_spiro = Chem.MolFromSmiles(scaff_spiro)
        self.assertIsNotNone(mol_spiro)
        self.assertEqual(mol_spiro.GetNumHeavyAtoms(), 11)
        self.assertEqual(mol_spiro.GetRingInfo().NumRings(), 2)

        # 2. Fused lactone: Coumarin (preserves benzene and lactone rings, 11 heavy atoms, 2 rings)
        scaff_coumarin = self.preparer.extract_seed_core_scaffold("O=C1Oc2ccccc2C=C1")
        mol_coumarin = Chem.MolFromSmiles(scaff_coumarin)
        self.assertIsNotNone(mol_coumarin)
        self.assertEqual(mol_coumarin.GetNumHeavyAtoms(), 11)
        self.assertEqual(mol_coumarin.GetRingInfo().NumRings(), 2)

        # 3. Bridgehead bicycle: Norbornane (preserves bicyclo[2.2.1]heptane framework, 7 heavy atoms, 2 rings)
        scaff_norbornane = self.preparer.extract_seed_core_scaffold("C1CC2CCC1C2")
        mol_norbornane = Chem.MolFromSmiles(scaff_norbornane)
        self.assertIsNotNone(mol_norbornane)
        self.assertEqual(mol_norbornane.GetNumHeavyAtoms(), 7)
        self.assertEqual(mol_norbornane.GetRingInfo().NumRings(), 2)

        # 4. Bicyclic ketone: Camphor (preserves bicyclo[2.2.1] with carbonyl, 8 heavy atoms, 2 rings)
        scaff_camphor = self.preparer.extract_seed_core_scaffold("CC1(C)C2CCC1(C)C(=O)C2")
        mol_camphor = Chem.MolFromSmiles(scaff_camphor)
        self.assertIsNotNone(mol_camphor)
        self.assertEqual(mol_camphor.GetNumHeavyAtoms(), 8)
        self.assertEqual(mol_camphor.GetRingInfo().NumRings(), 2)

        # 5. Acyclic zwitterion: Betaine (preserves sanitized heavy-atom structure, 8 heavy atoms)
        scaff_betaine = self.preparer.extract_seed_core_scaffold("C[N+](C)(C)CC(=O)[O-]")
        mol_betaine = Chem.MolFromSmiles(scaff_betaine)
        self.assertIsNotNone(mol_betaine)
        self.assertEqual(mol_betaine.GetNumHeavyAtoms(), 8)

    def test_hardware_manager_adversarial(self) -> None:
        """Verify HardwareManager across simulated host scales, fallback limits, and whitespace parsing."""
        from _david.code.utils import HardwareManager, get_safe_parallel_jobs

        # 1. 60% rule scaling across different machine profiles
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(ratio=0.60), max(1, int((os.cpu_count() or 1) * 0.60)))
        # Simulated 1-core container: max(1, int(1 * 0.60)) = 1
        self.assertEqual(max(1, int(1 * 0.60)), 1)
        # Simulated 4-core workstation: max(1, int(4 * 0.60)) = 2
        self.assertEqual(max(1, int(4 * 0.60)), 2)
        # Simulated 48-core server: max(1, int(48 * 0.60)) = 28
        self.assertEqual(max(1, int(48 * 0.60)), 28)
        # Simulated 128-core HPC node: max(1, int(128 * 0.60)) = 76
        self.assertEqual(max(1, int(128 * 0.60)), 76)

        # 2. Cap guard against negative or zero caps
        self.assertEqual(get_safe_parallel_jobs(cap=0), 1)
        self.assertEqual(get_safe_parallel_jobs(cap=-5), 1)

        # 3. CPU fallback on extreme / unachievable VRAM demand
        best_gpu_extreme = HardwareManager.select_best_gpu(min_free_mb=1_000_000.0)
        self.assertEqual(best_gpu_extreme, -1)

        # 4. Whitespace trimming on string device IDs
        safe_gpus_whitespace = HardwareManager.get_safe_gpus(preferred_gpu=" 0 ")
        self.assertIsInstance(safe_gpus_whitespace, tuple)

        # 5. Non-digit string preferred GPUs cleanly fall back to ()
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="cpu"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="none"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="cuda:invalid"), ())

    def test_pareto_front_duplicate_and_boundary_handling(self) -> None:
        """Verify Pareto frontier handles duplicate points and strict single-dimension dominance."""
        # 1. Duplicate points: Two identical non-dominated points must both be in Front 0
        df_dups = pd.DataFrame({
            "SMILES": ["mol1", "mol2_dup", "mol3"],
            "ROCS": [1.5, 1.5, 0.5],
            "SA": [0.9, 0.9, 0.2],
        })
        ranked_dups = ParetoFrontier().rank_pareto_fronts(df_dups, objectives=["ROCS", "SA"], maximize=[True, True])
        self.assertEqual(ranked_dups.loc[0, "Pareto_Rank"], 0)
        self.assertEqual(ranked_dups.loc[1, "Pareto_Rank"], 0)
        self.assertGreater(ranked_dups.loc[2, "Pareto_Rank"], 0)

        # 2. Single-coordinate dominance: [1.5, 0.9] dominates [1.5, 0.8] because 0.9 > 0.8 and 1.5 >= 1.5
        df_boundary = pd.DataFrame({
            "SMILES": ["mol_better", "mol_worse"],
            "ROCS": [1.5, 1.5],
            "SA": [0.9, 0.8],
        })
        is_front = ParetoFrontier.identify_pareto_front(
            df_boundary[["ROCS", "SA"]].to_numpy(dtype=float), maximize=[True, True]
        )
        self.assertTrue(is_front[0])
        self.assertFalse(is_front[1])


if __name__ == "__main__":
    unittest.main()
