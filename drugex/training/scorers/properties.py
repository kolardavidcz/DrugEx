"""Physicochemical property, substructure, and efficiency scoring functions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Sequence, Tuple


import numpy as np

# [PACKAGE: tqdm] WHY: Progress bar tracking during substructure matching across candidate molecules. WHAT: Progress bar utility.
import tqdm

# [PACKAGE: rdkit] WHY: Calculation of physicochemical properties, atom counting, substructure matching, and SMILES canonicalization. WHAT: Cheminformatics toolkit.
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.QED import qed

from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import Gaussian
from drugex.training.scorers.sascorer import calculateScore

if TYPE_CHECKING:
    from drugex.training.scorers.interfaces import ScoreModifier


class Property(Scorer):
    """Calculates standard physicochemical molecular properties using RDKit."""

    def __init__(self, prop: str = "MW", modifier: ScoreModifier | None = None) -> None:
        """Initialize the Property scorer."""
        super().__init__(modifier)
        self.prop = prop
        self.prop_dict: Dict[str, Callable[[Chem.Mol], float]] = {
            "MW": desc.MolWt,
            "logP": Crippen.MolLogP,
            "HBA": AllChem.CalcNumLipinskiHBA,
            "HBD": AllChem.CalcNumLipinskiHBD,
            "Rotable": AllChem.CalcNumRotatableBonds,
            "Amide": AllChem.CalcNumAmideBonds,
            "Bridge": AllChem.CalcNumBridgeheadAtoms,
            "Hetero": AllChem.CalcNumHeteroatoms,
            "Heavy": Lipinski.HeavyAtomCount,
            "Spiro": AllChem.CalcNumSpiroAtoms,
            "FCSP3": AllChem.CalcFractionCSP3,
            "Ring": Lipinski.RingCount,
            "Aliphatic": AllChem.CalcNumAliphaticRings,
            "Aromatic": AllChem.CalcNumAromaticRings,
            "Saturated": AllChem.CalcNumSaturatedRings,
            "HeteroR": AllChem.CalcNumHeterocycles,
            "TPSA": AllChem.CalcTPSA,
            "Valence": desc.NumValenceElectrons,
            "MR": Crippen.MolMR,
            "QED": qed,
            "SA": calculateScore,
            "Bertz": BertzCT,
        }
        if self.prop not in self.prop_dict:
            raise KeyError(f"Unknown property: {self.prop}. Available properties: {list(self.prop_dict.keys())}")

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate property scores for input molecules."""
        scores = np.zeros(len(mols), dtype="float64")
        calc_fn = self.prop_dict[self.prop]
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            try:
                scores[i] = calc_fn(mol)
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return property identifier string."""
        return self.prop


class AtomCounter(Scorer):
    """Scorer that counts occurrences of a specific chemical element in a molecule."""

    def __init__(self, element: str, modifier: ScoreModifier | None = None) -> None:
        """Initialize the AtomCounter scorer."""
        super().__init__(modifier)
        self.element = element

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Count the number of atoms of the target element in each molecule."""
        scores = np.zeros(len(mols), dtype="float64")
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            try:
                if self.element in ["", "H"]:
                    mol_hs = Chem.AddHs(mol)
                else:
                    mol_hs = mol

                if self.element == "":
                    scores[i] = len(mol_hs.GetAtoms())
                else:
                    scores[i] = sum(1 for a in mol_hs.GetAtoms() if a.GetSymbol() == self.element)
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return f"AtomCounter (element={self.element})"


class Isomer(Scorer):
    """Scoring function rewarding closeness to a target molecular formula."""

    def __init__(
        self,
        formula: str,
        mean_func: Literal["geometric", "arithmetic"] = "geometric",
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the Isomer scorer."""
        super().__init__(modifier)
        self.objs, self.mods = self.scoring_functions(formula)
        self.mean_func = mean_func

    @staticmethod
    def parse_molecular_formula(formula: str) -> List[Tuple[str, int]]:
        """Parse a molecular formula string into element symbols and stoichiometry counts."""
        matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        results: List[Tuple[str, int]] = []
        for match in matches:
            count = 1 if not match[1] else int(match[1])
            results.append((match[0], count))
        return results

    def scoring_functions(self, formula: str) -> Tuple[List[Scorer], List[Gaussian]]:
        """Construct constituent atom counter scorers and Gaussian modifiers for the formula."""
        element_occurrences = self.parse_molecular_formula(formula)
        total_n_atoms = sum(elem[1] for elem in element_occurrences)

        objs: List[Scorer] = [AtomCounter(element) for element, _ in element_occurrences]
        mods: List[Gaussian] = [Gaussian(mu=n_atoms, sigma=1.0) for _, n_atoms in element_occurrences]

        # Overall total atom count scorer
        objs.append(AtomCounter(""))
        mods.append(Gaussian(mu=total_n_atoms, sigma=2.0))

        return objs, mods

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate isomer similarity scores for the given molecules."""
        score_matrix = np.array([self.mods[i](obj(mols)) for i, obj in enumerate(self.objs)])
        if self.mean_func == "geometric":
            scores = score_matrix.prod(axis=0) ** (1.0 / len(score_matrix))
        else:
            scores = np.mean(score_matrix, axis=0)
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return f"Isomer (mean_func={self.mean_func})"


class Scaffold(Scorer):
    """Scorer checking presence or absence of a SMARTS substructure pattern."""

    def __init__(self, smart: str, is_match: bool = True, modifier: ScoreModifier | None = None) -> None:
        """Initialize the Scaffold scorer."""
        super().__init__(modifier)
        self.smart = smart
        self.frag = Chem.MolFromSmarts(smart)
        self.is_match = is_match

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Check substructure pattern match against each molecule."""
        scores = np.zeros(len(mols), dtype="float64")
        for i, mol in enumerate(tqdm.tqdm(mols, leave=False)):
            if mol is None:
                continue
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            try:
                match = mol.HasSubstructMatch(self.frag)
                scores[i] = 1.0 if (match == self.is_match) else 0.0
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return f"Scaffold(smart={self.smart},is_match={self.is_match})"


class Uniqueness(Scorer):
    """Calculates internal uniqueness of a batch of molecules."""

    def __init__(self, modifier: ScoreModifier | None = None) -> None:
        """Initialize Uniqueness scorer."""
        super().__init__(modifier)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Compute frequency ratios for molecules in the current batch."""
        n = len(mols)
        scores = np.zeros(n, dtype="float64")
        if n <= 1:
            return scores

        # Canonicalize to SMILES strings for accurate comparison
        smiles_list: List[str | None] = [Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else m for m in mols]
        for i, mol in enumerate(smiles_list):
            if mol is not None:
                scores[i] = (smiles_list.count(mol) - 1) / (n - 1)
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return "Unique"


class LipophilicEfficiency(Scorer):
    """Calculates Lipophilic Efficiency (LipE): LipE = pChEMBL - logP."""

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = None) -> None:
        """Initialize the LipophilicEfficiency scorer."""
        super().__init__(modifier)
        self.qsar_scorer = qsar_scorer
        self.key = f"LipE_{qsar_scorer.getKey()}"

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate LipE = activity - logP for molecules."""
        p_chembl = self.qsar_scorer.getScores(mols)
        log_p = Property("logP").getScores(mols)
        return p_chembl - log_p

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return self.key


class LigandEfficiency(Scorer):
    """Calculates Ligand Efficiency (LE): LE = 1.4 * pChEMBL / nHeavyAtoms."""

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = None) -> None:
        """Initialize the LigandEfficiency scorer."""
        super().__init__(modifier)
        self.qsar_scorer = qsar_scorer
        self.key = f"LE_{qsar_scorer.getKey()}"

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate LE = 1.4 * activity / nHeavyAtoms for molecules."""
        pChEMBL = np.asarray(self.qsar_scorer.getScores(mols), dtype="float64")
        nAtoms = np.zeros(len(mols), dtype="float64")
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            try:
                nAtoms[i] = float(mol.GetNumHeavyAtoms())
            except Exception:
                continue
        scores = np.zeros(len(mols), dtype="float64")
        valid_mask = nAtoms > 0
        scores[valid_mask] = 1.4 * pChEMBL[valid_mask] / nAtoms[valid_mask]
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer."""
        return self.key
