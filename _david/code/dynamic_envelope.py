"""Dynamic physicochemical property envelope calibration for universal reference ligands."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

# [PACKAGE: rdkit] WHY: Extract molecular physicochemical properties (heavy atom count, CLogP, TPSA) from RDKit Mol. WHAT: Cheminformatics toolkit. HOW: Lipinski.HeavyAtomCount, Crippen.MolLogP, AllChem.CalcTPSA.
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski

# [PACKAGE: drugex] WHY: Construct DrugEx Property scorers with SmoothHump Gaussian-plateau modifiers. WHAT: Deep learning drug design framework. HOW: drugex.training.scorers.properties.Property, drugex.training.scorers.modifiers.SmoothHump.
from drugex.training.scorers.modifiers import SmoothHump
from drugex.training.scorers.properties import Property


def _parse_molecule(query: str | Path | Chem.Mol) -> Chem.Mol:
    """Parse query input into a valid RDKit Mol object."""
    if isinstance(query, Chem.Mol):
        return query

    query_str: str = str(query)
    query_path: Path = Path(query_str)
    if query_path.is_file():
        # [PACKAGE: rdkit] WHY: Parse SDF or SMILES from file path. WHAT: Molecular file format supplier. HOW: Chem.SDMolSupplier.
        if query_path.suffix.lower() in {".sdf", ".sd"}:
            suppl = Chem.SDMolSupplier(str(query_path), removeHs=False)
            for mol in suppl:
                if mol is not None:
                    return mol
            raise ValueError(f"No valid molecule could be read from SDF: {query_path}")
        else:
            with open(query_path, "r", encoding="utf-8") as f:
                first_line: str = f.readline().strip()
                smi: str = first_line.split()[0] if first_line else ""
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return mol
            raise ValueError(f"Could not parse SMILES from file: {query_path}")

    mol = Chem.MolFromSmiles(query_str)
    if mol is not None:
        return mol

    raise ValueError(f"Could not parse query input into RDKit Mol: {query!r}")


class DynamicPropertyEnvelope:
    """Dynamically calibrates target physicochemical envelopes relative to a reference ligand.

    Calibrates Heavy Atom Count (HAC), CLogP, and Topological Polar Surface Area (TPSA)
    around the reference molecule values using SmoothHump plateau modifiers, with zero
    hardcoded molecular bounds or SMARTS rules.
    """

    def __init__(
        self,
        target_hac: float,
        target_clogp: float,
        target_tpsa: float,
        hac_lower: float,
        hac_upper: float,
        hac_sigma: float,
        clogp_lower: float,
        clogp_upper: float,
        clogp_sigma: float,
        tpsa_lower: float,
        tpsa_upper: float,
        tpsa_sigma: float,
    ) -> None:
        """Initialize calibrated property envelope parameters."""
        self.target_hac: float = float(target_hac)
        self.target_clogp: float = float(target_clogp)
        self.target_tpsa: float = float(target_tpsa)

        self.hac_lower: float = float(hac_lower)
        self.hac_upper: float = float(hac_upper)
        self.hac_sigma: float = float(hac_sigma)

        self.clogp_lower: float = float(clogp_lower)
        self.clogp_upper: float = float(clogp_upper)
        self.clogp_sigma: float = float(clogp_sigma)

        self.tpsa_lower: float = float(tpsa_lower)
        self.tpsa_upper: float = float(tpsa_upper)
        self.tpsa_sigma: float = float(tpsa_sigma)

    @classmethod
    def from_molecule(cls, mol: Chem.Mol) -> DynamicPropertyEnvelope:
        """Compute dynamic envelope directly from an RDKit Mol instance."""
        hac: float = float(Lipinski.HeavyAtomCount(mol))
        clogp: float = float(Crippen.MolLogP(mol))
        tpsa: float = float(AllChem.CalcTPSA(mol))

        # HAC envelope: [0.70 * X, 1.30 * X] bounded to prevent inversion
        hac_lower: float = max(1.0, 0.70 * hac)
        hac_upper: float = max(hac_lower + 1.0, 1.30 * hac)
        hac_sigma: float = max(2.0, 0.10 * hac)

        # CLogP envelope: Additive [X - 1.20, X + 1.20] with sigma = 0.80
        # (Additive delta is required because CLogP can be zero or negative)
        clogp_delta: float = 1.20
        clogp_lower: float = clogp - clogp_delta
        clogp_upper: float = clogp + clogp_delta
        clogp_sigma: float = 0.80

        # TPSA envelope: [0.70 * X, 1.30 * X] with low-TPSA fallback to prevent inversion
        tpsa_lower: float = max(0.0, 0.70 * tpsa)
        tpsa_upper: float = max(tpsa + 10.0 if tpsa < 10.0 else 1.30 * tpsa, 1.30 * tpsa)
        tpsa_sigma: float = max(5.0, 0.12 * tpsa)

        return cls(
            target_hac=hac,
            target_clogp=clogp,
            target_tpsa=tpsa,
            hac_lower=hac_lower,
            hac_upper=hac_upper,
            hac_sigma=hac_sigma,
            clogp_lower=clogp_lower,
            clogp_upper=clogp_upper,
            clogp_sigma=clogp_sigma,
            tpsa_lower=tpsa_lower,
            tpsa_upper=tpsa_upper,
            tpsa_sigma=tpsa_sigma,
        )

    @classmethod
    def from_query(cls, query: str | Path | Chem.Mol) -> DynamicPropertyEnvelope:
        """Compute dynamic envelope from query SMILES, SDF file path, or Mol."""
        mol = _parse_molecule(query)
        return cls.from_molecule(mol)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Serialize envelope bounds to dictionary format."""
        return {
            "Heavy": {
                "target": self.target_hac,
                "lower_x": self.hac_lower,
                "upper_x": self.hac_upper,
                "sigma": self.hac_sigma,
            },
            "logP": {
                "target": self.target_clogp,
                "lower_x": self.clogp_lower,
                "upper_x": self.clogp_upper,
                "sigma": self.clogp_sigma,
            },
            "TPSA": {
                "target": self.target_tpsa,
                "lower_x": self.tpsa_lower,
                "upper_x": self.tpsa_upper,
                "sigma": self.tpsa_sigma,
            },
        }

    def create_scorers(self) -> list[Property]:
        """Instantiate DrugEx Property scorers configured with SmoothHump modifiers."""
        hac_scorer = Property(
            prop="Heavy",
            modifier=SmoothHump(
                lower_x=self.hac_lower,
                upper_x=self.hac_upper,
                sigma=self.hac_sigma,
            ),
        )
        logp_scorer = Property(
            prop="logP",
            modifier=SmoothHump(
                lower_x=self.clogp_lower,
                upper_x=self.clogp_upper,
                sigma=self.clogp_sigma,
            ),
        )
        tpsa_scorer = Property(
            prop="TPSA",
            modifier=SmoothHump(
                lower_x=self.tpsa_lower,
                upper_x=self.tpsa_upper,
                sigma=self.tpsa_sigma,
            ),
        )
        return [hac_scorer, logp_scorer, tpsa_scorer]


def compute_dynamic_envelope(query: str | Path | Chem.Mol) -> dict[str, dict[str, float]]:
    """Compute dynamic physicochemical envelope dictionary for a query ligand."""
    envelope = DynamicPropertyEnvelope.from_query(query)
    return envelope.to_dict()


def create_dynamic_property_scorers(query: str | Path | Chem.Mol) -> list[Property]:
    """Create DrugEx Property scorers with calibrated SmoothHump modifiers for query ligand."""
    envelope = DynamicPropertyEnvelope.from_query(query)
    return envelope.create_scorers()
