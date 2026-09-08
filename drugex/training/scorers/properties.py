"""Physicochemical property, substructure, and efficiency scoring functions."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
import tqdm
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
    """Calculates standard physicochemical molecular properties using RDKit.

    Supported properties:
    - 'MW': Molecular weight
    - 'logP': Octanol-water partition coefficient (Crippen)
    - 'HBA': Lipinski Hydrogen Bond Acceptors
    - 'HBD': Lipinski Hydrogen Bond Donors
    - 'Rotable': Rotatable bond count
    - 'Amide': Number of amide bonds
    - 'Bridge': Number of bridgehead atoms
    - 'Hetero': Number of heteroatoms
    - 'Heavy': Heavy atom count
    - 'Spiro': Number of spiro atoms
    - 'FCSP3': Fraction of SP3 hybridized carbons
    - 'Ring': Total ring count
    - 'Aliphatic': Aliphatic ring count
    - 'Aromatic': Aromatic ring count
    - 'Saturated': Saturated ring count
    - 'HeteroR': Heterocyclic ring count
    - 'TPSA': Topological Polar Surface Area
    - 'Valence': Number of valence electrons
    - 'MR': Molar refractivity
    - 'QED': Quantitative Estimate of Drug-likeness
    - 'SA': Synthetic Accessibility score (Ertl & Schuffenhauer)
    - 'Bertz': Bertz topological complexity index

    Parameters
    ----------
    prop : str, optional
        Property key to evaluate, by default 'MW'.
    modifier : ScoreModifier | None, optional
        Modifier applied to the raw property score, by default None.
    """

    def __init__(self, prop: str = "MW", modifier: ScoreModifier | None = None) -> None:
        """Initialize the Property scorer.

        Parameters
        ----------
        prop : str, optional
            The property name to compute, by default 'MW'.
        modifier : ScoreModifier | None, optional
            Modifier applied to the computed property values, by default None.
        """
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
            raise KeyError(
                f"Unknown property: {self.prop}. Available properties: {list(self.prop_dict.keys())}"
            )

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate property scores for input molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Sequence of SMILES strings or RDKit molecules.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of calculated property scores. Unparseable molecules return 0.0.
        """
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
        """Return property identifier string.

        Returns
        -------
        str
            Property name (e.g. 'MW', 'logP').
        """
        return self.prop


class AtomCounter(Scorer):
    """Scorer that counts occurrences of a specific chemical element in a molecule.

    Parameters
    ----------
    element : str
        Chemical symbol of element to count (e.g. 'C', 'N', 'F', 'H'), or empty string
        '' to count total atoms including hydrogens.
    modifier : ScoreModifier | None, optional
        Score modifier to apply, by default None.
    """

    def __init__(self, element: str, modifier: ScoreModifier | None = None) -> None:
        """Initialize the AtomCounter scorer.

        Parameters
        ----------
        element : str
            The element symbol to count within the molecules.
        modifier : ScoreModifier | None, optional
            A `ScoreModifier` object to modify the scores, by default None.
        """
        super().__init__(modifier)
        self.element = element

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Count the number of atoms of the target element in each molecule.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Sequence of SMILES strings or RDKit molecules.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of atom counts.
        """
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
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'AtomCounter (element=<element>)'
        """
        return f"AtomCounter (element={self.element})"


class Isomer(Scorer):
    """Scoring function rewarding closeness to a target molecular formula.

    Penalizes deviations from the required stoichiometric counts of each element
    and the total atom count using Gaussian modifier penalties.

    For example, for target formula 'C2H4', the score is the geometric or arithmetic
    mean of:
    - C count with Gaussian(mu=2, sigma=1)
    - H count with Gaussian(mu=4, sigma=1)
    - Total atom count with Gaussian(mu=6, sigma=2)

    Parameters
    ----------
    formula : str
        Target molecular formula (e.g. 'C12H16N2O').
    mean_func : Literal['geometric', 'arithmetic'], optional
        Averaging method across element components, by default 'geometric'.
    modifier : ScoreModifier | None, optional
        Overall modifier applied to final averaged score, by default None.
    """

    def __init__(
        self,
        formula: str,
        mean_func: Literal["geometric", "arithmetic"] = "geometric",
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the Isomer scorer.

        Parameters
        ----------
        formula : str
            The molecular formula to score against.
        mean_func : Literal['geometric', 'arithmetic'], optional
            Which function to use for averaging the scores, by default 'geometric'.
        modifier : ScoreModifier | None, optional
            A `ScoreModifier` object to modify the scores, by default None.
        """
        super().__init__(modifier)
        self.objs, self.mods = self.scoring_functions(formula)
        self.mean_func = mean_func

    @staticmethod
    def parse_molecular_formula(formula: str) -> List[Tuple[str, int]]:
        """Parse a molecular formula string into element symbols and stoichiometry counts.

        Parameters
        ----------
        formula : str
            Molecular formula (e.g. 'C2H4Cl2').

        Returns
        -------
        List[Tuple[str, int]]
            List of (element_symbol, count) tuples.
        """
        matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        results: List[Tuple[str, int]] = []
        for match in matches:
            count = 1 if not match[1] else int(match[1])
            results.append((match[0], count))
        return results

    def scoring_functions(self, formula: str) -> Tuple[List[Scorer], List[Gaussian]]:
        """Construct constituent atom counter scorers and Gaussian modifiers for the formula.

        Parameters
        ----------
        formula : str
            Target molecular formula.

        Returns
        -------
        Tuple[List[Scorer], List[Gaussian]]
            (scoring_functions, modifiers) for individual elements and total atoms.
        """
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
        """Calculate isomer similarity scores for the given molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            Averaged isomer scores in [0.0, 1.0].
        """
        score_matrix = np.array([self.mods[i](obj(mols)) for i, obj in enumerate(self.objs)])
        if self.mean_func == "geometric":
            scores = score_matrix.prod(axis=0) ** (1.0 / len(score_matrix))
        else:
            scores = np.mean(score_matrix, axis=0)
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Isomer (mean_func=<mean_func>)'
        """
        return f"Isomer (mean_func={self.mean_func})"


class Scaffold(Scorer):
    """Scorer checking presence or absence of a SMARTS substructure pattern.

    Parameters
    ----------
    smart : str
        SMARTS query pattern.
    is_match : bool
        If True, returns 1.0 when substructure matches; if False, returns 1.0 when it does NOT match.
    modifier : ScoreModifier | None, optional
        Modifier applied to the binary match score, by default None.
    """

    def __init__(self, smart: str, is_match: bool = True, modifier: ScoreModifier | None = None) -> None:
        """Initialize the Scaffold scorer.

        Parameters
        ----------
        smart : str
            The SMARTS pattern to match.
        is_match : bool, optional
            Whether the pattern match is desired (True) or penalized (False), by default True.
        modifier : ScoreModifier | None, optional
            Modifier applied to scores, by default None.
        """
        super().__init__(modifier)
        self.smart = smart
        self.frag = Chem.MolFromSmarts(smart)
        self.is_match = is_match

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Check substructure pattern match against each molecule.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            Binary float array (1.0 or 0.0) indicating whether condition was met.
        """
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
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Scaffold(smart=<smart>,is_match=<is_match>)'
        """
        return f"Scaffold(smart={self.smart},is_match={self.is_match})"


class Uniqueness(Scorer):
    """Calculates internal uniqueness of a batch of molecules.

    The score penalizes duplicated molecules in the current generated batch:
    `score = (count(mol) - 1) / (len(batch) - 1)`

    Parameters
    ----------
    modifier : ScoreModifier | None, optional
        Modifier applied to uniqueness score, by default None.
    """

    def __init__(self, modifier: ScoreModifier | None = None) -> None:
        """Initialize Uniqueness scorer."""
        super().__init__(modifier)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Compute frequency ratios for molecules in the current batch.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Generated molecules to evaluate.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            Array of duplicate frequency fractions.
        """
        n = len(mols)
        scores = np.zeros(n, dtype="float64")
        if n <= 1:
            return scores

        # Canonicalize to SMILES strings for accurate comparison
        smiles_list: List[str | None] = [
            Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else m
            for m in mols
        ]
        for i, mol in enumerate(smiles_list):
            if mol is not None:
                scores[i] = (smiles_list.count(mol) - 1) / (n - 1)
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Unique'
        """
        return "Unique"


class LipophilicEfficiency(Scorer):
    """Calculates Lipophilic Efficiency (LipE): `LipE = pChEMBL - logP`.

    LipE estimates the potency of a drug molecule per unit of lipophilicity,
    helping identify compounds that achieve target affinity through specific
    binding interactions rather than non-specific hydrophobic partitioning.

    Parameters
    ----------
    qsar_scorer : Scorer
        QSAR/QSPR activity scorer producing pChEMBL / pIC50 values.
    modifier : ScoreModifier | None, optional
        Modifier applied to the computed LipE scores, by default None.
    """

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = None) -> None:
        """Initialize the LipophilicEfficiency scorer.

        Parameters
        ----------
        qsar_scorer : Scorer
            Scorer predicting target activity in logarithmic molar units.
        modifier : ScoreModifier | None, optional
            Score modifier, by default None.
        """
        super().__init__(modifier)
        self.qsar_scorer = qsar_scorer
        self.key = f"LipE_{qsar_scorer.getKey()}"

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate LipE = activity - logP for molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to evaluate.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            Calculated LipE scores.
        """
        p_chembl = self.qsar_scorer.getScores(mols)
        log_p = Property("logP").getScores(mols)
        return p_chembl - log_p

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'LipE_<qsar_scorer_key>'
        """
        return self.key


class LigandEfficiency(Scorer):
    """Calculates Ligand Efficiency (LE): `LE = 1.4 * pChEMBL / nHeavyAtoms`.

    Normalizes binding affinity by molecular size (heavy atom count),
    quantifying average binding energy per non-hydrogen atom (in kcal/mol).

    Parameters
    ----------
    qsar_scorer : Scorer
        QSAR/QSPR activity scorer producing pChEMBL / pIC50 values.
    modifier : ScoreModifier | None, optional
        Modifier applied to the computed LE scores, by default None.
    """

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = None) -> None:
        """Initialize the LigandEfficiency scorer.

        Parameters
        ----------
        qsar_scorer : Scorer
            Scorer predicting target activity in logarithmic molar units.
        modifier : ScoreModifier | None, optional
            Score modifier, by default None.
        """
        super().__init__(modifier)
        self.qsar_scorer = qsar_scorer
        self.key = f"LE_{qsar_scorer.getKey()}"

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate LE = 1.4 * activity / nHeavyAtoms for molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to evaluate.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            Calculated Ligand Efficiency scores.
        """
        pChEMBL = self.qsar_scorer.getScores(mols)
        nAtoms = [mol.GerNumAtoms() for mol in mols]
        scores = 1.4 * pChEMBL / nAtoms
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'LE_<qsar_scorer_key>'
        """
        return self.key
