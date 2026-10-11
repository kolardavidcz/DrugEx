"""Molecular similarity scoring functions based on fingerprints, Maximum Common Substructures (MCS), Fraggle, and Graph Edit Distance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

# [PACKAGE: networkx] WHY: Molecular graph construction and graph edit distance calculation. WHAT: Network analysis library.
import networkx

import numpy as np

# [PACKAGE: tqdm] WHY: Progress bar tracking during similarity calculations. WHAT: Progress bar utility.
import tqdm

# [PACKAGE: rdkit] WHY: Molecular graph/fingerprint parsing, MCS calculation, and Fraggle similarity. WHAT: Cheminformatics toolkit.
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem.Fraggle import FraggleSim

from drugex.training.scorers.interfaces import Scorer
from drugex.utils.fingerprints import get_fingerprint

if TYPE_CHECKING:
    from drugex.training.scorers.interfaces import ScoreModifier

logger = logging.getLogger(__name__)


class TverskyFingerprintSimilarity(Scorer):
    """Scoring function evaluating Tversky similarity between molecular fingerprints and a reference."""

    def __init__(
        self,
        smiles: str,
        fp_type: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the TverskyFingerprintSimilarity scorer."""
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
        """Calculate Tversky fingerprint similarity scores for input molecules."""
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
        """Return the unique identifier string for this scorer."""
        return f"Fingerprint similarity (fp_type={self.fp_type}, Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"


class TverskyGraphSimilarity(Scorer):
    """Scoring function evaluating Maximum Common Substructure (MCS) Tversky similarity."""

    def __init__(
        self,
        smiles: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initialize the TverskyGraphSimilarity scorer."""
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
        """Calculate Tversky graph similarity scores using MCS."""
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
        """Return the unique identifier string for this scorer."""
        return f"Graph similarity (Tversky weights={self.alpha},{self.beta}, smiles={self.smiles})"


class FraggleSimilarity(Scorer):
    """Scoring function based on Fraggle fragment-based similarity."""

    def __init__(
        self,
        smiles: str,
        trevsky_th: float = 0.8,
        modifier: ScoreModifier | None = None,
    ) -> None:
        """Initiate the Fraggle similarity scorer."""
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
        """Calculate Fraggle similarity scores for given molecules."""
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
        """Return the unique identifier string for this scorer."""
        return f"Fraggle similarity (Tversky threshold={self.th}, smiles={self.smiles})"


class GraphEditInverseDistance(Scorer):
    """Scoring function based on inverse Graph Edit Distance (GED) between molecular graphs."""

    def __init__(self, smiles: str, modifier: ScoreModifier | None = None) -> None:
        """Initialize the GraphEditInverseDistance scorer."""
        super().__init__(modifier)
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None:
            raise ValueError(f"Invalid reference SMILES for GraphEditInverseDistance: {smiles}")
        self.graph = self.get_graph(self.mol)

    @staticmethod
    def get_graph(mol: Chem.Mol) -> networkx.Graph:
        """Convert an RDKit molecule into an atom-annotated NetworkX graph."""
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
        """Compute inverse GED scores for input molecules."""
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
        """Return the unique identifier string for this scorer."""
        return f"GraphEditInverseDistance(smiles={self.smiles})"
