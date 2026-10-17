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
from drugex.training.rewards import ParetoCrowdingDistance, WeightedSum
from drugex.training.scorers.properties import Property
from rdkit import Chem
from rdkit.Chem import AllChem

from _david.code.analysis import AlignedDepicter, ParetoFrontier, ResultAnalyzer
from _david.code.environment import (
    DEFAULT_MOLECULE,
    PrepareMolecule,
    SafeSAModifier,
    create_environment,
)
from _david.code.pipeline import MoleculePipeline
from _david.code.schema import std
from _david.code.utils import HardwareManager
from _david.code.visualization import (
    render_epoch_pareto_evolution,
    render_lead_structures_2d,
    render_pareto_frontier,
    render_property_distributions,
    render_reference_ensemble,
    render_training_curves,
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
            self.assertIsInstance(
                env.rewardScheme, (ParetoCrowdingDistance, WeightedSum)
            )

    def test_conformer_generation(self) -> None:
        """Verify 3D conformer generation with MMFF94 and UFF fallback on non-MMFF elements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Standard organic molecule (MMFF94)
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

            # 2. Boronic acid: unparameterized in MMFF94, triggers UFF fallback
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
        # 1. Rigid/compact tier (<= 2 rotatable bonds, <= 20 heavy atoms): (1.00 Å, 1)
        m_benzene = Chem.MolFromSmiles("c1ccccc1")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_benzene), (1.00, 1)
        )

        m_aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_aspirin), (1.00, 1)
        )

        # Explicit hydrogens invariance check: phenol with explicit Hs must remain (1.00 Å, 1)
        m_phenol_h = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_phenol_h), (1.00, 1)
        )

        # 2. Standard drug-like tier (<= 7 rotatable bonds, <= 45 heavy atoms)
        m_ibuprofen = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_ibuprofen), (1.25, 1)
        )

        # 3. Flexible tier (> 7 rotatable bonds or > 45 heavy atoms): (1.50 Å, 1)
        m_decanoic = Chem.MolFromSmiles("CCCCCCCCCC(=O)O")
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(m_decanoic), (1.50, 1)
        )

        # Macrocycle: Erythromycin (48 heavy atoms) -> 1.50 Å flexible tier
        smi_erythromycin = (
            "CCC1C(C(C(C(=O)C(CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)"
            "OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O"
        )
        mol_erythro = Chem.MolFromSmiles(smi_erythromycin)
        self.assertEqual(
            PrepareMolecule._calculate_adaptive_rmsd_cutoff(mol_erythro), (1.50, 1)
        )

        # Null and empty molecule guards
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

            # EGCG canonical Bemis-Murcko framework preserves chroman, 2 aryl rings, and ester linker (25 heavy atoms, 4 rings)
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
            self.assertEqual(row[3], "25")
            self.assertEqual(row[4], "4")

            # Standard monocyclic framework: phenol -> benzene
            self.assertEqual(
                self.preparer.extract_seed_core_scaffold("c1ccccc1O"), "c1ccccc1"
            )

            # Acyclic fallback: preserves sanitized heavy-atom structure cleanly
            self.assertEqual(
                self.preparer.extract_seed_core_scaffold("CCCCCC(=O)O"), "CCCCCC(=O)O"
            )

    def test_load_query_molecule(self) -> None:
        """Verify query molecule loading across SMILES strings, RDKit Mols, and files."""
        # 1. SMILES input
        mol, is_3d = self.preparer.load_query_molecule("c1ccccc1O")
        self.assertFalse(is_3d)
        self.assertEqual(mol.GetNumHeavyAtoms(), 7)

        # Long valid SMILES (> 255 chars) must not crash with OS path limit
        mol_long, is_3d = self.preparer.load_query_molecule("C" * 260)
        self.assertFalse(is_3d)
        self.assertEqual(mol_long.GetNumHeavyAtoms(), 260)

        with self.assertRaises(ValueError):
            self.preparer.load_query_molecule("INVALID" * 40)

        # 2. Mol object input
        mol_input = Chem.MolFromSmiles("c1ccccc1")
        mol_ret, is_3d = self.preparer.load_query_molecule(mol_input)
        self.assertIs(mol_ret, mol_input)
        self.assertFalse(is_3d)

        # Planar 2D conformer coordinates must be rejected as non-3D
        AllChem.Compute2DCoords(mol_input)
        _, is_3d_planar = self.preparer.load_query_molecule(mol_input)
        self.assertFalse(is_3d_planar)

        # Embedded 3D Mol
        mol_3d = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol_3d)
        _, is_3d_emb = self.preparer.load_query_molecule(mol_3d)
        self.assertTrue(is_3d_emb)

        # 3. File inputs (.txt, .sdf)
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
            _, is_3d_sdf = self.preparer.load_query_molecule(sdf_path)
            self.assertTrue(is_3d_sdf)

            empty_path = Path(tmp_dir) / "empty.txt"
            empty_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.preparer.load_query_molecule(empty_path)

    def test_pareto_ranking(self) -> None:
        """Verify Pareto Front 0 non-domination alignment, duplicate handling, and single-dimension dominance."""
        pareto = ParetoFrontier()

        # 1. Non-dominated Front 0 proof battery under maximize=[True, True]
        df_proof = pd.DataFrame(
            {
                std.SMILES: ["molA", "molB", "molC", "molD", "molE"],
                std.ROCS: [1.6, 1.8, 1.2, 1.4, 0.8],
                std.SA: [0.9, 0.5, 1.0, 0.4, 0.1],
            }
        )
        scores = df_proof[[std.ROCS, std.SA]].to_numpy(dtype=float)
        is_front = ParetoFrontier.identify_pareto_front(scores, maximize=[True, True])

        # Exactly A, B, C form Front 0; D and E are dominated
        self.assertTrue(is_front[0], "Mol A [1.6, 0.9] must be Front 0")
        self.assertTrue(is_front[1], "Mol B [1.8, 0.5] must be Front 0")
        self.assertTrue(is_front[2], "Mol C [1.2, 1.0] must be Front 0")
        self.assertFalse(is_front[3], "Mol D [1.4, 0.4] must be dominated")
        self.assertFalse(is_front[4], "Mol E [0.8, 0.1] must be dominated")

        ranked = pareto.rank_pareto_fronts(
            df_proof, objectives=[std.ROCS, std.SA], maximize=[True, True]
        )
        self.assertEqual(ranked.loc[0, std.PARETO_RANK], 0)
        self.assertEqual(ranked.loc[1, std.PARETO_RANK], 0)
        self.assertEqual(ranked.loc[2, std.PARETO_RANK], 0)
        self.assertGreater(ranked.loc[3, std.PARETO_RANK], 0)
        self.assertGreater(ranked.loc[4, std.PARETO_RANK], 0)

        # 2. Duplicate points: identical non-dominated points must both reside in Front 0
        df_dups = pd.DataFrame(
            {
                std.SMILES: ["mol1", "mol2_dup", "mol3"],
                std.ROCS: [1.5, 1.5, 0.5],
                std.SA: [0.9, 0.9, 0.2],
            }
        )
        ranked_dups = pareto.rank_pareto_fronts(
            df_dups, objectives=[std.ROCS, std.SA], maximize=[True, True]
        )
        self.assertEqual(ranked_dups.loc[0, std.PARETO_RANK], 0)
        self.assertEqual(ranked_dups.loc[1, std.PARETO_RANK], 0)
        self.assertGreater(ranked_dups.loc[2, std.PARETO_RANK], 0)

        # 3. Single-coordinate dominance: [1.5, 0.9] dominates [1.5, 0.8]
        df_boundary = pd.DataFrame(
            {
                std.SMILES: ["mol_better", "mol_worse"],
                std.ROCS: [1.5, 1.5],
                std.SA: [0.9, 0.8],
            }
        )
        is_front_b = ParetoFrontier.identify_pareto_front(
            df_boundary[[std.ROCS, std.SA]].to_numpy(dtype=float),
            maximize=[True, True],
        )
        self.assertTrue(is_front_b[0])
        self.assertFalse(is_front_b[1])

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
        """Verify dynamic CPU allocation, power-of-2 batch sizing, environment isolation, and GPU selection."""
        # 1. Dynamic CPU allocation (60% rule)
        expected_jobs = max(1, int((os.cpu_count() or 1) * 0.60))
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(), expected_jobs)
        self.assertEqual(
            HardwareManager.get_safe_parallel_jobs(cap=2), min(2, expected_jobs)
        )
        # Cap guard against zero or negative caps
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(cap=0), 1)
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(cap=-5), 1)

        # 2. Power-of-2 batch sizing clamped strictly to n_samples
        self.assertEqual(HardwareManager.get_recommended_batch_size(n_samples=10), 10)
        batch_normal = HardwareManager.get_recommended_batch_size(n_samples=200)
        self.assertIn(batch_normal, [32, 64, 128, 200])
        batch_large = HardwareManager.get_recommended_batch_size(n_samples=500)
        self.assertIn(batch_large, [32, 64, 128, 256])

        # 3. Environment isolation and dynamic ratio support
        env_vars = HardwareManager.init_runtime_environment()
        self.assertEqual(os.environ.get("DRUGEX_SCORE_NJOBS"), str(expected_jobs))
        self.assertEqual(os.environ.get("DRUGEX_MP_CONTEXT"), "forkserver")
        self.assertEqual(env_vars["DRUGEX_SCORE_NJOBS"], str(expected_jobs))
        self.assertEqual(env_vars["DRUGEX_MP_CONTEXT"], "forkserver")

        expected_jobs_80 = max(1, int((os.cpu_count() or 1) * 0.80))
        self.assertEqual(HardwareManager.get_safe_parallel_jobs(ratio=0.80), expected_jobs_80)
        env_vars_80 = HardwareManager.init_runtime_environment(ratio=0.80)
        self.assertEqual(os.environ.get("DRUGEX_SCORE_NJOBS"), str(expected_jobs_80))
        self.assertEqual(env_vars_80["DRUGEX_SCORE_NJOBS"], str(expected_jobs_80))
        HardwareManager.init_runtime_environment(ratio=0.60)

        # 4. Safe GPU routing, multi-GPU and CPU fallback
        import torch
        gpus = HardwareManager.get_safe_gpus()
        self.assertIsInstance(gpus, tuple)
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="cpu"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="none"), ())
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu=9999), ())
        expected_all = tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
        self.assertEqual(HardwareManager.get_safe_gpus(preferred_gpu="all"), expected_all)

    def test_decoupled_validation_loader(self) -> None:
        """Verify that prepare_seed_fragments returns decoupled train and bounded validation dataloaders."""
        from _david.code.models import GraphTransformerOptimizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = GraphTransformerOptimizer(tmp_dir, gpu="auto")

            # Decoupled validation: train and validation loaders are separate instances
            train_loader, val_loader, dataset = optimizer.prepare_seed_fragments(
                "c1ccccc1O", n_prompts=20, val_samples=5
            )
            self.assertIsNot(train_loader, val_loader)
            self.assertEqual(len(train_loader.dataset), 20)
            self.assertEqual(len(val_loader.dataset), 5)
            self.assertLessEqual(len(val_loader.dataset), len(train_loader.dataset))

            # Bounds invariance: validation dataset size never exceeds prompt pool
            train_l10, val_l10, _ = optimizer.prepare_seed_fragments(
                "c1ccccc1O", n_prompts=10, val_samples=50
            )
            self.assertLessEqual(len(val_l10.dataset), len(train_l10.dataset))

    def test_safe_sa_modifier(self) -> None:
        """Verify SafeSAModifier enforces monotonic synthesizability and guards against uncalculable fallbacks."""
        modifier = SafeSAModifier()

        # 1. Monotonicity: easier synthesis (lower SA score) strictly yields higher or equal desirability
        self.assertGreater(float(modifier(2.0)), float(modifier(4.0)))
        self.assertGreater(float(modifier(4.0)), float(modifier(6.0)))

        # 2. Boundedness: desirability always stays within [0.0, 1.0]
        for val in [1.0, 2.5, 3.5, 5.0, 8.0, 10.0]:
            score = float(modifier(val))
            self.assertTrue(0.0 <= score <= 1.0, f"Score {score} out of [0, 1] bounds for SA {val}")

        # 3. Crash / uncalculable / unphysical fallback guard (SA <= 0.0 strictly maps to 0.0 reward)
        self.assertEqual(float(modifier(0.0)), 0.0)
        self.assertEqual(float(modifier(-1.0)), 0.0)
        self.assertEqual(float(modifier(-99.0)), 0.0)

        # 4. Vector vs. scalar invariance
        input_arr = np.array([0.0, 2.0, 5.0, -1.0])
        res_arr = modifier(input_arr)
        for i, x in enumerate(input_arr):
            self.assertAlmostEqual(res_arr[i], float(modifier(x)), places=5)

    def test_result_analyzer_deduplication_and_tie_breaking(self) -> None:
        """Verify ResultAnalyzer deduplicates canonical SMILES clones and breaks Pareto Front 0 ties by QED."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidates_tsv = tmp_path / "candidates.tsv"

            smi_asp1 = "CC(=O)Oc1ccccc1C(=O)O"
            smi_asp2 = "O=C(O)c1ccccc1OC(=O)C"  # Chemically valid alternative SMILES for aspirin
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

            # Deduplication: identical canonical SMILES must appear only once
            asp_canon = Chem.CanonSmiles(smi_asp1)
            asp_rows = saved_leads[saved_leads[std.CANONICAL_SMILES] == asp_canon]
            self.assertEqual(len(asp_rows), 1, "Duplicate canonical SMILES must be collapsed")

            # Tie-breaking: Front 0 candidates sorted by QED descending
            self.assertEqual(saved_leads.iloc[0][std.PARETO_RANK], 0)
            self.assertEqual(saved_leads.iloc[1][std.PARETO_RANK], 0)
            self.assertGreater(saved_leads.iloc[0][std.QED], saved_leads.iloc[1][std.QED])

    def test_data_schema_standardization(self) -> None:
        """Verify std normalizes dynamic DrugEx scorer names, historical variants, and prunes duplicate columns."""
        # 1. Raw DrugEx DataFrame with Supermol ROCS and Desired
        raw_df = pd.DataFrame(
            {
                std.SMILES: ["c1ccccc1"],
                "RDKit_Supermol_TanimotoCombo_oe": [0.65],
                std.SA: [1.0],
                std.DESIRED: [1],
            }
        )
        std_df = std.standardize(raw_df)
        self.assertIn(std.ROCS, std_df.columns)
        self.assertEqual(std_df.loc[0, std.ROCS], 0.65)
        self.assertEqual(std_df.loc[0, std.DESIRED], 1)

        # 2. Multi-reference DrugEx DataFrame with Aggregate 8refs ROCS
        agg_df = pd.DataFrame(
            {
                std.SMILES: ["c1ccccc1"],
                "RDKit_Aggregate_8refs_TanimotoCombo_oe": [0.72],
                "SAScore": [2.5],
                "Desirability": [0.85],
            }
        )
        std_agg = std.standardize(agg_df)
        self.assertIn(std.ROCS, std_agg.columns)
        self.assertEqual(std_agg.loc[0, std.ROCS], 0.72)
        self.assertIn(std.SA, std_agg.columns)
        self.assertEqual(std_agg.loc[0, std.SA], 2.5)
        self.assertIn(std.DESIRED, std_agg.columns)
        self.assertEqual(std_agg.loc[0, std.DESIRED], 0.85)

        # 3. Idempotence: already standardized DataFrame is unchanged
        re_std = std.standardize(std_agg)
        self.assertEqual(list(std_agg.columns), list(re_std.columns))

        # 4. Duplicate pruning: duplicate SA/SAscore and Desired/Desirability
        dup_df = pd.DataFrame(
            {
                std.SMILES: ["c1ccccc1"],
                std.ROCS: [0.65],
                std.SA: [1.0],
                "SAscore": [1.0],
                std.DESIRED: [1],
                "Desirability": [1],
                "Frags": ["c1ccccc1"],
                "logP": [1.5],
            }
        )
        std_dup = std.standardize(dup_df)
        self.assertNotIn("SAscore", std_dup.columns)
        self.assertNotIn("Desirability", std_dup.columns)
        self.assertNotIn("Frags", std_dup.columns)
        self.assertNotIn("logP", std_dup.columns)
        self.assertIn(std.SA, std_dup.columns)
        self.assertIn(std.DESIRED, std_dup.columns)
        self.assertIn(std.FRAGS, std_dup.columns)
        self.assertIn(std.LOGP, std_dup.columns)

        # 5. Physical unscaling: ROCS [0, 1] -> [0, 2] and clipped SA [0, 1] -> raw SAscore [1, 10]
        norm_df = pd.DataFrame(
            {
                std.SMILES: ["c1ccccc1O"],
                std.ROCS: [0.60],
                std.SA: [1.0],
            }
        )
        unscaled = std.unscale_to_physical_metrics(norm_df)
        self.assertAlmostEqual(unscaled.loc[0, std.ROCS], 1.20, places=5)
        self.assertGreater(unscaled.loc[0, std.SA], 1.05)  # Ertl raw SAscore > 1.05


    def test_visualization_suite(self) -> None:
        """Verify end-to-end visualization suite (training curves, evolution, property distributions, 2D leads, Pareto frontier)."""
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

            # 3. Pareto Epoch Evolution
            smiles_tsv = tmp / "smiles.tsv"
            pd.DataFrame(
                {
                    std.EPOCH: [1, 1, 2, 2],
                    std.SMILES: ["c1ccccc1", "c1ccccc1O", "c1ccccc1CC", "c1ccccc1C(=O)O"],
                    std.ROCS: [0.4, 0.6, 0.5, 0.8],
                    std.SA: [0.9, 0.85, 0.8, 0.75],
                }
            ).to_csv(smiles_tsv, sep="\t", index=False)
            pareto_evo_png = render_epoch_pareto_evolution(
                smiles_tsv, output_png=tmp / "pareto_evo.png"
            )
            self.assertIsNotNone(pareto_evo_png)
            self.assertTrue(pareto_evo_png.exists())

            # 4. Property Distributions (9-panel grid)
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

            # 5. Lead Structures 2D Cards (including unreset index verification)
            leads_df = pd.DataFrame(
                {
                    std.SMILES: ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1CC"],
                    std.ROCS: [0.575, 0.90],
                    std.DESIRED: [1.0, 0.50],
                    std.PARETO_RANK: [0, 1],
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

            # 6. Pareto Frontier
            front_df = pd.DataFrame(
                {
                    std.SMILES: ["m1", "m2"],
                    std.ROCS: [0.8, 0.5],
                    std.SA: [0.9, 0.4],
                    std.PARETO_RANK: [0, 1],
                }
            )
            front_png = render_pareto_frontier(
                front_df, output_png=tmp / "front.png", rocs_col=std.ROCS, sa_col=std.SA
            )
            self.assertIsNotNone(front_png)
            self.assertTrue(front_png.exists())

    def test_pipeline_test_mode(self) -> None:
        """Verify MoleculePipeline execution in test mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipe_dir = Path(tmp_dir) / "run_test"
            pipeline = MoleculePipeline(output_dir=pipe_dir, is_test=True)
            pipeline.run("Oc1ccccc1")
            self.assertTrue(
                (pipe_dir / "raw_data" / "logs" / "pipeline_status.json").exists()
            )
            self.assertTrue((pipe_dir / "summary.md").exists())
            self.assertTrue(
                (pipe_dir / "raw_data" / "query_molecule_3d.sdf").exists()
            )
            self.assertTrue(
                (pipe_dir / "lead_structures_2d.png").exists()
                or (pipe_dir / "property_distributions.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
