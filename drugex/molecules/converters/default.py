# [PACKAGE: rdkit] WHY: Chemical molecule parsing from SMILES strings. WHAT: Chem.MolFromSmiles parses string representations into Mol objects.
from rdkit import Chem

from drugex.molecules.converters.interfaces import ConversionException, MolConverter
from drugex.molecules.mol import DrExMol


class SmilesToDrEx(MolConverter):
    """Converter parsing a SMILES string into a DrExMol instance."""

    def __call__(self, smiles):
        mol = None
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as exp:
            raise ConversionException(exp)
        if not mol:
            raise ConversionException(f"Failed to parse SMILES: {smiles}")
        return DrExMol(mol, identifier=smiles.strip())


class RDKitToDrEx(MolConverter):
    """Converter wrapping an existing RDKit Mol object into a DrExMol instance."""

    def __call__(self, rd_mol):
        if not rd_mol:
            raise ConversionException("RDKit supplied an empty molecule.")
        else:
            return DrExMol(rd_mol)

class DrExToSMILES(MolConverter):
    """Converter extracting the canonical SMILES string from a DrExMol instance."""

    def __call__(self, drugex_mol):
        return drugex_mol.smiles

class Identity(MolConverter):
    """Identity converter passing input items through unchanged."""

    def __call__(self, item):
        return item