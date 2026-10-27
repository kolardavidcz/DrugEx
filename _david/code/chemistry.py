"""Small thesis-local adaptations; DrugEx core remains unchanged."""

from __future__ import annotations

import logging

import numpy as np
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


class CandidateConformerGenerator(RDKitConformerGenerator):
    """DrugEx ETKDG with a selectable seed and optional MMFF94 post-minimization."""

    def __init__(
        self, random_seed: int = 0xC0FFEE, minimize_candidates: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.random_seed = random_seed
        self.minimize_candidates = minimize_candidates
        self.last_minimization_statuses: list[int] = []

    def _create_fresh_etkdg(self):
        params = super()._create_fresh_etkdg()
        params.randomSeed = self.random_seed
        return params

    def genConformers(self, smiles_list, out_dir) -> str:
        path = super().genConformers(smiles_list, out_dir)
        self.last_minimization_statuses = []
        if self.minimize_candidates:
            records = list(Chem.SDMolSupplier(path, removeHs=False))
            minimized, statuses = minimize_mmff94(records)
            self.last_minimization_statuses = statuses
            with Chem.SDWriter(path) as writer:
                for mol in minimized:
                    writer.write(mol)
            failures = sum(status != 0 for status in statuses)
            if failures:
                logging.getLogger(__name__).warning(
                    "MMFF94 candidate minimization excluded %d/%d conformers",
                    failures,
                    len(statuses),
                )
        return path


def minimize_mmff94(
    records: list[Chem.Mol], max_iterations: int = 1000
) -> tuple[list[Chem.Mol], list[int]]:
    """Minimize copies; return converged conformers and every native status.

    Status 0 means converged, 1 means more iterations are needed, and -1 means
    invalid geometry/missing parameters. No UFF fallback is used for this method.
    """
    minimized, statuses = [], []
    for record in records:
        status = -1
        if record is not None and record.GetNumConformers():
            mol = Chem.Mol(record)
            conf = mol.GetConformer()
            if (
                conf.Is3D()
                and np.isfinite(conf.GetPositions()).all()
                and AllChem.MMFFHasAllMoleculeParams(mol)
            ):
                status = AllChem.MMFFOptimizeMolecule(
                    mol, mmffVariant="MMFF94", maxIters=max_iterations
                )
                if status == 0:
                    check_identity(mol, record)
                    mol.SetProp("ForceField", "MMFF94")
                    mol.SetIntProp("MMFFOptimizationStatus", status)
                    minimized.append(mol)
        statuses.append(status)
    return minimized, statuses


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
