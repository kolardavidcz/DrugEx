"""
Physicochemical property, substructure, and efficiency scoring functions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from drugex.training.scorers.interfaces import ScoreModifier

import re
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Sequence
from typing import Tuple
import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.QED import qed
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import Gaussian
from drugex.training.scorers.sascorer import calculateScore


class Property(Scorer):
    """
    Calculates standard physicochemical molecular properties using RDKit.

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

    def __init__(self, prop: str = ..., modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the Property scorer.

        Parameters
        ----------
        prop : str, optional
            The property name to compute, by default 'MW'.
        modifier : ScoreModifier | None, optional
            Modifier applied to the computed property values, by default None.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Calculate property scores for input molecules.

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

        ...

    def getKey(self) -> str:
        """
        Return property identifier string.

        Returns
        -------
        str
            Property name (e.g. 'MW', 'logP').
        """

        ...


class AtomCounter(Scorer):
    """
    Scorer that counts occurrences of a specific chemical element in a molecule.

    Parameters
    ----------
    element : str
        Chemical symbol of element to count (e.g. 'C', 'N', 'F', 'H'), or empty string
        '' to count total atoms including hydrogens.
    modifier : ScoreModifier | None, optional
        Score modifier to apply, by default None.
    """

    def __init__(self, element: str, modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the AtomCounter scorer.

        Parameters
        ----------
        element : str
            The element symbol to count within the molecules.
        modifier : ScoreModifier | None, optional
            A `ScoreModifier` object to modify the scores, by default None.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Count the number of atoms of the target element in each molecule.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'AtomCounter (element=<element>)'
        """

        ...


class Isomer(Scorer):
    """
    Scoring function rewarding closeness to a target molecular formula.

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

    def __init__(self, formula: str, mean_func: Literal['geometric', 'arithmetic'] = ..., modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the Isomer scorer.

        Parameters
        ----------
        formula : str
            The molecular formula to score against.
        mean_func : Literal['geometric', 'arithmetic'], optional
            Which function to use for averaging the scores, by default 'geometric'.
        modifier : ScoreModifier | None, optional
            A `ScoreModifier` object to modify the scores, by default None.
        """

        ...

    @staticmethod
    def parse_molecular_formula(formula: str) -> List[Tuple[str, int]]:
        """
        Parse a molecular formula string into element symbols and stoichiometry counts.

        Parameters
        ----------
        formula : str
            Molecular formula (e.g. 'C2H4Cl2').

        Returns
        -------
        List[Tuple[str, int]]
            List of (element_symbol, count) tuples.
        """

        ...

    def scoring_functions(self, formula: str) -> Tuple[List[Scorer], List[Gaussian]]:
        """
        Construct constituent atom counter scorers and Gaussian modifiers for the formula.

        Parameters
        ----------
        formula : str
            Target molecular formula.

        Returns
        -------
        Tuple[List[Scorer], List[Gaussian]]
            (scoring_functions, modifiers) for individual elements and total atoms.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Calculate isomer similarity scores for the given molecules.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Isomer (mean_func=<mean_func>)'
        """

        ...


class Scaffold(Scorer):
    """
    Scorer checking presence or absence of a SMARTS substructure pattern.

    Parameters
    ----------
    smart : str
        SMARTS query pattern.
    is_match : bool
        If True, returns 1.0 when substructure matches; if False, returns 1.0 when it does NOT match.
    modifier : ScoreModifier | None, optional
        Modifier applied to the binary match score, by default None.
    """

    def __init__(self, smart: str, is_match: bool = ..., modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the Scaffold scorer.

        Parameters
        ----------
        smart : str
            The SMARTS pattern to match.
        is_match : bool, optional
            Whether the pattern match is desired (True) or penalized (False), by default True.
        modifier : ScoreModifier | None, optional
            Modifier applied to scores, by default None.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Check substructure pattern match against each molecule.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Scaffold(smart=<smart>,is_match=<is_match>)'
        """

        ...


class Uniqueness(Scorer):
    """
    Calculates internal uniqueness of a batch of molecules.

    The score penalizes duplicated molecules in the current generated batch:
    `score = (count(mol) - 1) / (len(batch) - 1)`

    Parameters
    ----------
    modifier : ScoreModifier | None, optional
        Modifier applied to uniqueness score, by default None.
    """

    def __init__(self, modifier: ScoreModifier | None = ...) -> None:
        """Initialize Uniqueness scorer."""

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Compute frequency ratios for molecules in the current batch.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'Unique'
        """

        ...


class LipophilicEfficiency(Scorer):
    """
    Calculates Lipophilic Efficiency (LipE): `LipE = pChEMBL - logP`.

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

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the LipophilicEfficiency scorer.

        Parameters
        ----------
        qsar_scorer : Scorer
            Scorer predicting target activity in logarithmic molar units.
        modifier : ScoreModifier | None, optional
            Score modifier, by default None.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Calculate LipE = activity - logP for molecules.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'LipE_<qsar_scorer_key>'
        """

        ...


class LigandEfficiency(Scorer):
    """
    Calculates Ligand Efficiency (LE): `LE = 1.4 * pChEMBL / nHeavyAtoms`.

    Normalizes binding affinity by molecular size (heavy atom count),
    quantifying average binding energy per non-hydrogen atom (in kcal/mol).

    Parameters
    ----------
    qsar_scorer : Scorer
        QSAR/QSPR activity scorer producing pChEMBL / pIC50 values.
    modifier : ScoreModifier | None, optional
        Modifier applied to the computed LE scores, by default None.
    """

    def __init__(self, qsar_scorer: Scorer, modifier: ScoreModifier | None = ...) -> None:
        """
        Initialize the LigandEfficiency scorer.

        Parameters
        ----------
        qsar_scorer : Scorer
            Scorer predicting target activity in logarithmic molar units.
        modifier : ScoreModifier | None, optional
            Score modifier, by default None.
        """

        ...

    def getScores(self, mols: Sequence[str | Chem.Mol | None], frags: Sequence[str | None] | None = ...) -> np.ndarray:
        """
        Calculate LE = 1.4 * activity / nHeavyAtoms for molecules.

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

        ...

    def getKey(self) -> str:
        """
        Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'LE_<qsar_scorer_key>'
        """

        ...

