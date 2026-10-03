"""Guacamol-derived molecule processing utilities for SMILES canonicalization and deduplication."""
from typing import Optional, Iterable, List

# [PACKAGE: rdkit] WHY: Cheminformatics engine WHAT: Parses SMILES to molecules and renders canonical SMILES
from rdkit import Chem


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit, preserving optional stereochemistry and returning None if invalid."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None

def remove_duplicates(list_with_duplicates):
    """Deduplicate an iterable while preserving the first-seen insertion order."""
    unique_set = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list

def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """Canonicalize, filter unparseable entries, and deduplicate an iterable of SMILES strings."""
    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)
