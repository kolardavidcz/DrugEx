"""Guard the specified stereoisomer through thesis reference preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from drugex.molecules.converters.standardizers import DefaultStandardizer

from _david.code.config import DEFAULT_MOLECULE
from _david.code.conformer.self_recovery import EGCGSelfRecovery, StudySettings
from _david.code.environment import PrepareMolecule


class TestReferenceIdentity(unittest.TestCase):
    """Reject an EGCG reference whose stereochemical identity changed."""

    def test_reference_standardization_retains_specified_stereochemistry(self) -> None:
        """Preserve canonical isomeric identity rather than only the molecular formula."""
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(DEFAULT_MOLECULE))
        mol, _ = PrepareMolecule().load_molecule(DEFAULT_MOLECULE)
        self.assertEqual(Chem.MolToSmiles(mol), expected)
        self.assertNotEqual(DefaultStandardizer()(DEFAULT_MOLECULE), expected)

    def test_embedded_references_have_the_input_stereoisomer(self) -> None:
        """Verify the SDF's coordinate-derived stereochemistry after optimization."""
        expected = Chem.MolToSmiles(Chem.MolFromSmiles(DEFAULT_MOLECULE))
        with tempfile.TemporaryDirectory() as directory:
            path = PrepareMolecule().generate_reference_conformers(
                DEFAULT_MOLECULE,
                Path(directory) / "reference.sdf",
                num_conformers=32,
                random_seed=42,
            )
            references = list(Chem.SDMolSupplier(str(path), removeHs=False))
            self.assertGreater(len(references), 0)
            for mol in references:
                self.assertIsNotNone(mol)
                self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(mol)), expected)
                self.assertTrue(mol.GetConformer().Is3D())

    def test_self_recovery_rejects_a_different_stereoisomer(self) -> None:
        """Fail the study rather than silently comparing a wrong chiral reference."""
        mol = Chem.MolFromSmiles(DEFAULT_MOLECULE)
        center = Chem.FindMolChiralCenters(mol)[0][0]
        mol.GetAtomWithIdx(center).InvertChirality()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        with tempfile.TemporaryDirectory() as directory:
            study = EGCGSelfRecovery(Path(directory) / "study", StudySettings())
            with self.assertRaisesRegex(ValueError, "identity differs"):
                study.checked_record(mol)
