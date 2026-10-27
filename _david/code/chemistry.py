"""Small thesis-local adaptations; DrugEx core remains unchanged."""

from __future__ import annotations

import numpy as np
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


class SeededConformerGenerator(RDKitConformerGenerator):
    """Expose the embedding seed while retaining DrugEx's candidate algorithm."""

    def __init__(self, random_seed: int = 0xC0FFEE, **kwargs) -> None:
        super().__init__(**kwargs)
        self.random_seed = random_seed

    def _create_fresh_etkdg(self):
        params = super()._create_fresh_etkdg()
        params.randomSeed = self.random_seed
        return params


def standardize_smiles(smiles: str) -> str:
    """Use DrugEx's standardization steps, retaining isomeric SMILES output.

    DefaultStandardizer ends with non-isomeric serialization. There is no option
    to change that, so reproduce its RDKit operations here rather than patch core
    or bypass desalting/normalization. Fail explicitly instead of using a fallback
    with a different chemistry policy.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    mol = disconnector.Disconnect(mol)
    mol = normalizer.normalize(mol)
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = normalizer.normalize(disconnector.Disconnect(mol))
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        raise ValueError("Standardized molecule contains no carbon")
    if any(atom.GetAtomicNum() in (11, 30) for atom in mol.GetAtoms()):
        raise ValueError("Salt removal failed")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def check_identity(mol: Chem.Mol, expected: Chem.Mol) -> str:
    """Check connectivity and specified stereo, including assignment from 3D.

    Unspecified input centers may be resolved; specified centers may not change.
    Return the actual isomeric identity for inspection and winner reporting.
    """
    actual = Chem.Mol(mol)
    if actual.GetNumConformers():
        conf = actual.GetConformer()
        if not conf.Is3D() or not np.isfinite(conf.GetPositions()).all():
            raise ValueError("Conformer lacks finite 3D coordinates")
        Chem.AssignStereochemistryFrom3D(actual, replaceExistingTags=True)
    actual = Chem.RemoveHs(actual)
    expected = Chem.RemoveHs(Chem.Mol(expected))
    same_graph = Chem.MolToSmiles(actual, isomericSmiles=False) == Chem.MolToSmiles(
        expected, isomericSmiles=False
    )
    if not same_graph or not actual.HasSubstructMatch(expected, useChirality=True):
        raise ValueError(
            f"Molecular identity changed: {Chem.MolToSmiles(expected)} -> "
            f"{Chem.MolToSmiles(actual)}"
        )
    return Chem.MolToSmiles(actual, isomericSmiles=True)
