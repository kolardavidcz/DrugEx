from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Sequence, Union

# Ensure project root and _david are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "drugex").exists() or (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

#? [NumPy] Vectorized numerical operations for pairwise Manhattan distance and maximum similarity aggregation.
import numpy as np
#? [RDKit] Cheminformatics foundation for molecule parsing, conformer embedding, and 3D moment calculations.
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from drugex.training.scorers.interfaces import ScoreModifier, Scorer


class USRCATScorer(Scorer):
    """Pure Ligand-Based 3D Shape and Pharmacophore Scorer using USRCAT and ElectroShape moments.

    Skeptical Academic Rationale:
    1. ProLIF (Protein-Ligand Interaction Fingerprints) is strictly designed for Structure-Based
       Drug Design (SBDD) requiring explicit atomic coordinates of the macromolecular receptor binding
       pocket (e.g., CCR2 5T1A). In 100% pure Ligand-Based Drug Design (LBDD), receptor coordinates do
       not exist. Introducing artificial docked poses or pseudo-receptors introduces scoring function bias
       and violates the pure LBDD premise.
    2. 2D Pharmacophore Filters (e.g. topological distance paths) collapse 3D conformational geometry
       into 1D/2D graph steps, completely missing out-of-plane orientations, steric volume, and chiral
       envelopes (such as EGCG's tri-aromatic galloyl ester architecture).
    3. USRCAT (Ultrafast Shape Recognition with CREDO Atom Types; Ballester et al., 2010) extends
       classical USR by evaluating distance distributions (mean, variance, skewness) to 4 reference
       geometric points (centroid, closest to centroid, furthest from centroid, furthest from furthest)
       across 5 pharmacophore atom subsets (all atoms, hydrophobic, aromatic, hydrogen-bond donors,
       hydrogen-bond acceptors), generating a 60-dimensional rotationally and translationally invariant
       fingerprint.
    4. Unlike iterative Gaussian volume superposition (ROCS/rdShapeAlign) which requires expensive numerical
       optimization over Euler angles and easily gets trapped in local rotational minima, USRCAT similarity
       is an exact O(1) inverse Manhattan distance, executing in microseconds per conformer pair.

    Attributes:
        references: Path to multi-conformer reference SDF or list of Chem.Mol objects.
        num_conformers: Number of 3D conformers to generate for candidate molecules during scoring.
        score_type: Type of 3D moment to calculate ('usrcat', 'usr', or 'electroshape').
        ref_moments: Pre-computed 2D NumPy array of shape (N_ref_confs, 60) holding reference USRCAT vectors.
    """

    def __init__(
        self,
        references: Union[str, Path, Sequence[Chem.Mol]],
        num_conformers: int = 10,
        score_type: str = "usrcat",
        modifier: ScoreModifier | None = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize USRCATScorer with reference conformer ensemble.

        Args:
            references: Path to multi-record reference SDF or sequence of RDKit Mol conformers.
            num_conformers: Candidate conformers generated per molecule (default: 10).
            score_type: Descriptor type ('usrcat', 'usr', or 'electroshape').
            modifier: Optional DrugEx ScoreModifier to scale raw similarity into [0.0, 1.0].
            random_seed: Random seed for stochastic ETKDG conformer embedding.

        Raises:
            FileNotFoundError: If references path does not exist.
            ValueError: If no valid conformers can be extracted from references.
        """
        super().__init__(modifier=modifier)
        self.num_conformers: int = num_conformers
        self.score_type: str = score_type.lower()
        self.random_seed: int = random_seed
        self.ref_moments: np.ndarray = self._load_reference_moments(references)

    def _load_reference_moments(self, references: Union[str, Path, Sequence[Chem.Mol]]) -> np.ndarray:
        """Extract and cache 3D moments for all reference conformers.

        Args:
            references: File path to reference SDF or sequence of Chem.Mol objects.

        Returns:
            2D NumPy array of shape (N_refs, descriptor_dim).
        """
        ref_mols: List[Chem.Mol] = []
        if isinstance(references, (str, Path)):
            ref_path = Path(references)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference conformer file not found: {ref_path}")
            #? [RDKit SDMolSupplier] Parses MDL multi-record SDF without stripping explicit hydrogens.
            suppl = Chem.SDMolSupplier(str(ref_path), removeHs=False)
            for m in suppl:
                if m is not None:
                    ref_mols.append(m)
        else:
            ref_mols = [m for m in references if m is not None]

        if not ref_mols:
            raise ValueError("No valid reference molecules found for USRCAT scoring.")

        moments_list: List[np.ndarray] = []
        for mol in ref_mols:
            #? [RDKit Mol] Iterates through internal 3D conformers stored in the Mol object.
            for conf_idx in range(mol.GetNumConformers()):
                desc = self._calc_moments(mol, conf_id=conf_idx)
                if desc is not None:
                    moments_list.append(desc)

        if not moments_list:
            raise ValueError("Could not extract any 3D USRCAT moments from reference conformers.")

        #? [NumPy] Stacks individual 60-dim moment vectors into an (N_refs, 60) matrix.
        return np.vstack(moments_list)

    def _calc_moments(self, mol: Chem.Mol, conf_id: int = -1) -> np.ndarray | None:
        """Calculate 3D shape/pharmacophore moments for a single molecular conformer.

        Args:
            mol: RDKit Mol object containing at least one 3D conformer.
            conf_id: Conformer index to evaluate (-1 for default/first conformer).

        Returns:
            1D NumPy array containing the moment descriptors, or None on calculation failure.
        """
        try:
            if self.score_type == "usr":
                #? [RDKit USR] Ultrafast Shape Recognition: 12-dimensional geometric distance moments.
                usr = rdMolDescriptors.GetUSR(mol, confId=conf_id)
                return np.array(usr, dtype=np.float64)
            elif self.score_type == "electroshape":
                #? [RDKit Gasteiger] Computes partial atomic charges for electrostatic moment calculation.
                AllChem.ComputeGasteigerCharges(mol)
                usr = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
                # Compute charge moments across atoms
                conf = mol.GetConformer(conf_id)
                charges = [float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0 for atom in mol.GetAtoms()]
                q_arr = np.nan_to_num(np.array(charges, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                q_moments = [float(np.mean(q_arr)), float(np.var(q_arr)), float(np.mean(np.abs(q_arr)))]
                return np.concatenate([np.array(usr, dtype=np.float64), np.array(q_moments, dtype=np.float64)])
            else:
                #? [RDKit USRCAT] 60-dimensional Ultrafast Shape Recognition with CREDO Atom Types.
                usrcat = rdMolDescriptors.GetUSRCAT(mol, confId=conf_id)
                return np.array(usrcat, dtype=np.float64)
        except Exception:
            return None

    def _score_single_molecule(self, mol_or_smi: Union[str, Chem.Mol]) -> float:
        """Score candidate molecule against reference USRCAT moment ensemble.

        Args:
            mol_or_smi: Candidate SMILES string or RDKit Mol.

        Returns:
            Maximum USRCAT similarity score across candidate and reference conformers [0.0, 1.0].
        """
        if isinstance(mol_or_smi, str):
            #? [RDKit] Parses candidate SMILES string into C++ Mol representation.
            mol = Chem.MolFromSmiles(mol_or_smi)
        else:
            mol = Chem.Mol(mol_or_smi) if mol_or_smi is not None else None

        if mol is None:
            return 0.0

        #? [RDKit] Adds explicit hydrogens required for geometric centroids and pharmacophore H-bonds.
        mol = Chem.AddHs(mol)

        #? [RDKit ETKDGv3] Experimental-Torsion Knowledge Distance Geometry conformer embedding.
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.pruneRmsThresh = 0.50

        # Embed candidate conformers
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conformers, params=params))
        if not conf_ids:
            return 0.0

        cand_moments: List[np.ndarray] = []
        for cid in conf_ids:
            m = self._calc_moments(mol, conf_id=cid)
            if m is not None:
                cand_moments.append(m)

        if not cand_moments:
            return 0.0

        #? [NumPy] Array of candidate moments of shape (N_cand_confs, dim).
        cand_mat = np.vstack(cand_moments)
        dim = cand_mat.shape[1]

        #? [NumPy Broadcasting] Computes pairwise L1 Manhattan distances between all candidate and reference conformers:
        # cand_mat shape: (N_c, 1, dim), ref_moments shape: (1, N_r, dim)
        diff = np.abs(cand_mat[:, np.newaxis, :] - self.ref_moments[np.newaxis, :, :])
        manhattan_dists = np.sum(diff, axis=2) / float(dim)

        #? [Ballester USRCAT Similarity] S = 1.0 / (1.0 + (1/N) * sum(|u - v|)).
        sim_matrix = 1.0 / (1.0 + manhattan_dists)

        # Maximum similarity achieved by any conformer pair
        max_sim = float(np.max(sim_matrix))
        return max_sim

    def getScores(
        self,
        mols: Sequence[Union[str, Chem.Mol]],
        frags: Sequence[Union[str, Chem.Mol]] | None = None,
    ) -> np.ndarray:
        """Calculate raw 3D USRCAT similarity scores for a batch of candidate molecules.

        Args:
            mols: Sequence of candidate SMILES strings or RDKit Mol objects.
            frags: Optional sequence of fragment prompts (unused, present for DrugEx Scorer interface).

        Returns:
            1D NumPy array of shape (len(mols),) containing raw USRCAT similarities in [0.0, 1.0].
        """
        scores = [self._score_single_molecule(m) for m in mols]
        #? [NumPy] Converts Python score list to float32 NumPy vector for DrugEx Environment compatibility.
        return np.array(scores, dtype=np.float32)
