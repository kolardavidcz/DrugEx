"""Regression checks for chemical identity, scoring and RNN prior isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.training.scorers.rocs_rdkit import _score_single_reference
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from _david.code.chemistry import (
    Optimization,
    check_identity,
    minimize_mmff94,
    optimize_conformers,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE, PipelineConfig
from _david.code.environment import PrepareMolecule, create_environment
from _david.code.environment import create_rocs_scorer
from _david.code.models import SequenceRNNOptimizer


class TestPreparation(unittest.TestCase):
    def test_failed_embedding_is_not_written_as_a_2d_candidate(self):
        reference = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.assertEqual(AllChem.EmbedMolecule(reference, randomSeed=42), 0)
        scorer = create_rocs_scorer(reference, max_conformers=1, n_jobs=1)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "_david.code.chemistry.AllChem.EmbedMultipleConfs", return_value=[-1]
            ),
        ):
            path = scorer.conformer_generator.genConformers(["CCO"], directory)
            self.assertEqual(Path(path).stat().st_size, 0)
            self.assertEqual(scorer.conformer_generator.last_embedding_failures, 1)

    def test_minimized_study_score_matches_the_public_training_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.sdf"
            PrepareMolecule(
                num_conformers=5, num_threads=1
            ).generate_reference_conformers(DEFAULT_MOLECULE, path)
            references = list(Chem.SDMolSupplier(str(path), removeHs=False))
            baseline = create_rocs_scorer(
                references,
                max_conformers=5,
                timeout=0,
                conformer_threads=1,
                n_jobs=1,
            )
            raw_path = baseline.conformer_generator.genConformers(
                [DEFAULT_MOLECULE], directory
            )
            records = list(Chem.SDMolSupplier(raw_path, removeHs=False))
            minimized, _ = minimize_mmff94(records)
            self.assertTrue(minimized)
            study_score = max(
                _score_single_reference(
                    Chem.Mol(mol), Chem.Mol(ref), "TanimotoCombo", True
                )
                for mol in minimized
                for ref in references
            )
            training = create_rocs_scorer(
                references,
                max_conformers=5,
                timeout=0,
                conformer_threads=1,
                n_jobs=1,
                minimize_candidates=True,
            )
            self.assertAlmostEqual(
                float(training.getScores([DEFAULT_MOLECULE])[0, 0]),
                study_score,
                places=10,
            )

    def test_candidate_minimization_switch_is_off_by_default_and_preserves_identity(
        self,
    ):
        self.assertFalse(PipelineConfig().minimize_candidates)
        smiles = "CCCCCC"
        reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.assertEqual(AllChem.EmbedMolecule(reference, randomSeed=42), 0)
        generated = {}
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (False, True):
                env = create_environment(
                    reference,
                    config=PipelineConfig(
                        max_conformers=3,
                        minimize_candidates=enabled,
                        conformer_threads=1,
                    ),
                )
                generator = env.scorers[0].conformer_generator
                self.assertEqual(generator.minimize_candidates, enabled)
                path = generator.genConformers([smiles], directory)
                generated[enabled] = list(Chem.SDMolSupplier(path, removeHs=False))
                self.assertTrue(generated[enabled])
                for mol in generated[enabled]:
                    check_identity(mol, Chem.MolFromSmiles(smiles))
                    self.assertEqual(mol.HasProp("MMFFOptimizationStatus"), enabled)
                if enabled:
                    self.assertTrue(
                        all(
                            status == 0
                            for status in generator.last_minimization_statuses
                        )
                    )
            self.assertEqual(len(generated[False]), len(generated[True]))
            for before, after in zip(generated[False], generated[True]):

                def energy(mol):
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    return AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()

                self.assertLessEqual(energy(after), energy(before) + 1e-3)

    def test_unconverged_candidate_is_reported_and_excluded(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        minimized, statuses = minimize_mmff94([mol], max_iterations=0)
        self.assertEqual(minimized, [])
        self.assertEqual(statuses, [1])

    def test_force_fields_preserve_multiple_molecular_structures(self):
        methods = [
            Optimization(),
            Optimization("MMFF94s"),
            Optimization("MMFF94s", 6.5),
            Optimization("MMFF94s", 78.4),
            Optimization("MMFF94s", 1, 2),
            Optimization("UFF"),
        ]
        for smiles in (
            "CCCCCC",
            "c1ccccc1",
            "N[C@@H](C)C(=O)O",
            "F/C=C/F",
            DEFAULT_MOLECULE,
        ):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
            for method in methods:
                with self.subTest(smiles=smiles, method=method.name):
                    optimized, statuses = optimize_conformers([mol], method)
                    self.assertEqual(statuses, [0])
                    check_identity(optimized[0], mol)
                    self.assertTrue(
                        np.isfinite(optimized[0].GetDoubleProp("Energy_kcal_mol"))
                    )
        with tempfile.TemporaryDirectory() as directory:
            path = PrepareMolecule(
                num_conformers=5, num_threads=1, optimization=Optimization("none")
            ).generate_reference_conformers("CCCCC", Path(directory) / "none.sdf")
            mol = next(Chem.SDMolSupplier(str(path), removeHs=False))
            self.assertFalse(mol.HasProp("Energy_kcal_mol"))
            self.assertFalse(mol.GetBoolProp("EnergyWindowApplied"))

    def test_standardization_preserves_stereo_and_existing_chemistry(self):
        for smiles in (
            DEFAULT_MOLECULE,
            "N[C@@H](C)C(=O)O",
            "F/C=C/F",
            "F/C=C\\F",
            "CC(O)C(=O)[O-].[Na+]",
        ):
            with self.subTest(smiles=smiles):
                result = standardize_smiles(smiles)
                self.assertEqual(
                    Chem.MolToSmiles(Chem.MolFromSmiles(result), isomericSmiles=False),
                    DefaultStandardizer()(smiles),
                )
                if "." not in smiles:
                    check_identity(
                        Chem.MolFromSmiles(result), Chem.MolFromSmiles(smiles)
                    )
        unresolved = Chem.MolFromSmiles(standardize_smiles("FC(Cl)C(O)C"))
        self.assertEqual(len(tuple(EnumerateStereoisomers(unresolved))), 4)
        with self.assertRaises(ValueError):
            standardize_smiles("not a molecule")

    def test_reference_round_trip_preserves_geometry_identity_and_energy_window(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, smiles in enumerate(
                (DEFAULT_MOLECULE, "N[C@@H](C)C(=O)O", "F/C=C/F")
            ):
                with self.subTest(smiles=smiles):
                    path = Path(directory) / f"reference_{index}.sdf"
                    preparer = PrepareMolecule(
                        num_conformers=5, random_seed=43, num_threads=1
                    )
                    preparer.generate_reference_conformers(smiles, path)
                    records = list(Chem.SDMolSupplier(str(path), removeHs=False))
                    self.assertTrue(records)
                    for mol in records:
                        self.assertIsNotNone(mol)
                        check_identity(
                            mol, Chem.MolFromSmiles(preparer.standardize(smiles))
                        )
                        self.assertLessEqual(mol.GetDoubleProp("DeltaE_kcal_mol"), 8.0)
                        self.assertGreaterEqual(
                            mol.GetIntProp("EmbeddedConformers"), len(records)
                        )

    def test_wrong_isomer_and_mixed_reference_file_are_rejected(self):
        left = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
        right = Chem.AddHs(Chem.MolFromSmiles("N[C@H](C)C(=O)O"))
        for mol in (left, right):
            self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            check_identity(right, left)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.sdf"
            with Chem.SDWriter(str(path)) as writer:
                writer.write(left)
                writer.write(right)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                PrepareMolecule().load_molecule(path)

    def test_exact_geometry_overlay_and_score_scale(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))
        self.assertEqual(AllChem.EmbedMolecule(mol, randomSeed=42), 0)
        raw = _score_single_reference(
            Chem.Mol(mol), Chem.Mol(mol), "TanimotoCombo", True
        )
        self.assertAlmostEqual(raw, 2.0, places=5)
        env = create_environment(
            mol, config=PipelineConfig(max_conformers=7, max_isomers=8)
        )
        scorer = env.scorers[0]
        self.assertAlmostEqual(float(scorer.getModifiedScores(np.array([raw]))[0]), 1.0)
        generator = scorer.conformer_generator
        self.assertEqual(
            (generator.max_conformers, generator.max_isomers, generator.timeout),
            (7, 8, 0),
        )
        self.assertEqual(generator._create_fresh_etkdg().randomSeed, 0xC0FFEE)

    def test_candidate_generation_preserves_specified_identity(self):
        env = create_environment(
            Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")),
            config=PipelineConfig(max_conformers=5),
        )
        generator = env.scorers[0].conformer_generator
        with tempfile.TemporaryDirectory() as directory:
            path = generator.genConformers([DEFAULT_MOLECULE], directory)
            mols = list(Chem.SDMolSupplier(path, removeHs=False))
            self.assertTrue(mols)
            for mol in mols:
                check_identity(mol, Chem.MolFromSmiles(DEFAULT_MOLECULE))
            self.assertEqual(len({Chem.MolToSmiles(Chem.RemoveHs(m)) for m in mols}), 1)


class TestRNNTransfer(unittest.TestCase):
    def test_loaded_agent_and_frozen_prior_do_not_share_parameter_storage(self):
        try:
            package, vocabulary = PipelineConfig().resolve_pretrained_paths()
        except FileNotFoundError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            optimizer = SequenceRNNOptimizer(
                models_dir=directory,
                gpu="cpu",
                agent_path=package,
                vocab_path=vocabulary,
            )
            agent, prior, _ = optimizer.load_models()
            self.assertIsNot(agent, prior)
            agent_storage = {p.data_ptr() for p in agent.parameters()}
            prior_storage = {p.data_ptr() for p in prior.parameters()}
            self.assertTrue(agent_storage.isdisjoint(prior_storage))
            self.assertTrue(any(p.requires_grad for p in agent.parameters()))
            self.assertTrue(all(not p.requires_grad for p in prior.parameters()))
            self.assertFalse(prior.training)


if __name__ == "__main__":
    unittest.main()
