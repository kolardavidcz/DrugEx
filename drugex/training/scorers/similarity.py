"""Molecular similarity scoring functions based on fingerprints, Maximum Common Substructures (MCS), Fraggle, and Graph Edit Distance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import networkx
import numpy as np
import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem.Fraggle import FraggleSim

from drugex.training.scorers.interfaces import Scorer
from drugex.utils.fingerprints import get_fingerprint

if TYPE_CHECKING:
    from drugex.training.scorers.interfaces import ScoreModifier

logger = logging.getLogger(__name__)


class TverskyFingerprintSimilarity(Scorer):
    """Scoring function evaluating Tversky similarity between molecular fingerprints and a reference.

    When `alpha=1.0` and `beta=1.0`, this reduces to standard Tanimoto similarity.
    Adjusting alpha/beta allows biasing towards substructure or superstructure similarity.

    Parameters
    ----------
    smiles : str
        SMILES string of reference compound.
    fp_type : str
        Fingerprint type identifier accepted by `drugex.utils.fingerprints.get_fingerprint`
        (e.g. 'MorganFP', 'MACCS').
    alpha : float, optional
        Weight of unique features of the reference compound, by default 1.0.
    beta : float, optional
        Weight of unique features of the query compound, by default 1.0.
    modifier : ScoreModifier | None, optional
        Modifier applied to computed similarity scores, by default None.
    """

    def __init__(
        self,
        smiles: str,
        fp_type: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the TverskyFingerprintSimilarity scorer.

        Parameters
        ----------
        smiles : str
            The SMILES string of the reference molecule.
        fp_type : str
            The type of fingerprint to use.
        alpha : float, optional
            The weight of the features of the reference compound, by default 1.0.
        beta : float, optional
            The weight of the features of the compound to be scored, by default 1.0.
        modifier : ScoreModifier | None, optional
            A modifier that can be used to modify the scores returned by this scorer, by default None.
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid reference SMILES for TverskyFingerprintSimilarity: {smiles}")
        self.fp_type = fp_type
        self.fp = get_fingerprint(self.mol, fp_type=fp_type)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate Tversky fingerprint similarity scores for input molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of similarity scores in [0.0, 1.0].
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
                fp = get_fingerprint(mol, fp_type=self.fp_type)
                scores[i] = DataStructs.TverskySimilarity(self.fp, fp, self.alpha, self.beta)
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            Key specifying fingerprint type, weights, and reference SMILES.
        """
        return f"Fingerprint similarity (fp_type={self.fp_type}, Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"


class TverskyGraphSimilarity(Scorer):
    """Scoring function evaluating Maximum Common Substructure (MCS) Tversky similarity.

    Quantifies the graph overlap between each molecule and the reference structure:
    `score = |MCS| / (|MCS| + alpha * |Reference - MCS| + beta * |Query - MCS|)`

    Parameters
    ----------
    smiles : str
        SMILES string of reference molecule.
    alpha : float, optional
        Weight of reference features, by default 1.0.
    beta : float, optional
        Weight of query molecule features, by default 1.0.
    modifier : ScoreModifier | None, optional
        Modifier applied to the scores, by default None.
    """

    def __init__(
        self,
        smiles: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the TverskyGraphSimilarity scorer.

        Parameters
        ----------
        smiles : str
            The SMILES string of the reference molecule.
        alpha : float, optional
            The weight of the features of the reference molecule, by default 1.0.
        beta : float, optional
            The weight of the features of the compound to be scored, by default 1.0.
        modifier : ScoreModifier | None, optional
            A ScorerModifier object to modify the scores, by default None.
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid reference SMILES for TverskyGraphSimilarity: {smiles}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate Tversky graph similarity scores using MCS.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of MCS graph similarity scores in [0.0, 1.0].
        """
        scores = np.zeros(len(mols), dtype="float64")
        nref = self.mol.GetNumAtoms()
        for i, mol in enumerate(tqdm.tqdm(mols, leave=False)):
            if mol is None:
                continue
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            try:
                mcs = rdFMCS.FindMCS([self.mol, mol])
                nmcs = mcs.numAtoms
                diff_ref = nref - nmcs
                diff_mol = mol.GetNumAtoms() - nmcs
                denominator = nmcs + self.alpha * diff_ref + self.beta * diff_mol
                scores[i] = (nmcs / denominator) if denominator > 0 else 0.0
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            Key specifying weights and reference SMILES.
        """
        return f"Graph similarity (Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"


class FraggleSimilarity(Scorer):
    """Scoring function based on Fraggle fragment-based similarity.

    Fraggle evaluates structural similarity by fragmenting molecules along acyclic
    single bonds and comparing the matching query fragments to reference fragments.
    Reference: Hussain & Rea, RDKit UGM 2013.

    Parameters
    ----------
    smiles : str
        Reference compound SMILES.
    trevsky_th : float, optional
        Tversky similarity threshold used by Fraggle, by default 0.8.
    modifier : ScoreModifier | None, optional
        Score modifier, by default None.
    """

    def __init__(
        self,
        smiles: str,
        trevsky_th: float = 0.8,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initiate the Fraggle similarity scorer.

        Parameters
        ----------
        smiles : str
            Reference compound SMILES.
        trevsky_th : float, optional
            Tversky threshold used by Fraggle, by default 0.8.
        modifier : ScoreModifier | None, optional
            Score modifier to apply, by default None.
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid reference SMILES for FraggleSimilarity: {smiles}")
        self.th = float(trevsky_th)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Calculate Fraggle similarity scores for given molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of Fraggle similarity scores in [0.0, 1.0].
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
                sim, _ = FraggleSim.GetFraggleSimilarity(self.mol, mol, tverskyThresh=self.th)
                scores[i] = float(sim)
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            Key specifying threshold and reference SMILES.
        """
        return f"Fraggle similarity (Tversky threshold={self.th}, smiles={self.smiles})"


class GraphEditInverseDistance(Scorer):
    """Scoring function based on inverse Graph Edit Distance (GED) between molecular graphs.

    `score = 1.0 / sqrt(GED)`

    Notes
    -----
    Computing exact graph edit distance is NP-hard. This scorer is computationally heavy
    and intended primarily for small molecules or low-volume benchmark tasks.

    Parameters
    ----------
    smiles : str
        Reference compound SMILES.
    modifier : ScoreModifier | None, optional
        Score modifier, by default None.
    """

    def __init__(self, smiles: str, modifier: ScoreModifier | None = None) -> None:
        """Initialize the GraphEditInverseDistance scorer.

        Parameters
        ----------
        smiles : str
            Reference molecule SMILES string.
        modifier : ScoreModifier | None, optional
            Score modifier, by default None.
        """
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid reference SMILES for GraphEditInverseDistance: {smiles}")
        self.graph = self.get_graph(self.mol)

    @staticmethod
    def get_graph(mol: Chem.Mol) -> networkx.Graph:
        """Convert an RDKit molecule into an atom-annotated NetworkX graph.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.

        Returns
        -------
        networkx.Graph
            NetworkX graph with atom atomic numbers on diagonals/nodes.
        """
        mol_copy = Chem.Mol(mol)
        Chem.Kekulize(mol_copy)
        atoms = [atom.GetAtomicNum() for atom in mol_copy.GetAtoms()]
        am = Chem.GetAdjacencyMatrix(mol_copy, useBO=True)
        for i, atom in enumerate(atoms):
            am[i, i] = atom

        return networkx.from_numpy_matrix(am)

    def getScores(
        self,
        mols: Sequence[str | Chem.Mol | None],
        frags: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Compute inverse GED scores for input molecules.

        Parameters
        ----------
        mols : Sequence[str | Chem.Mol | None]
            Molecules to score.
        frags : Sequence[str | None] | None, optional
            Fragment constraints (unused).

        Returns
        -------
        np.ndarray
            1D array of inverse graph edit distances.
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
                graph = self.get_graph(mol)
                dist = 0.0
                for v in networkx.optimize_graph_edit_distance(
                    self.graph, graph, edge_match=lambda a, b: a["weight"] == b["weight"]
                ):
                    dist = float(v)
                scores[i] = (1.0 / np.sqrt(dist)) if dist > 0 else 1.0
            except Exception:
                continue
        return scores

    def getKey(self) -> str:
        """Return the unique identifier string for this scorer.

        Returns
        -------
        str
            'GraphEditInverseDistance(smiles=<smiles>)'
        """
        return f"GraphEditInverseDistance(smiles={self.smiles})"
