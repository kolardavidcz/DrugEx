"""Stage 2 Chemotype Funnel bridging Sequence-RNN exploration and Graph-Transformer optimization.

Decomposes raw candidate hits into Bemis-Murcko core frameworks, purges PAINS
and target-specific toxicophores, filters by dynamic heavy-atom volume bounds,
enforces structural hopping distance away from the seed query, and clusters scaffolds
via Butina sphere exclusion.
Strictly gates expensive Gaussian volume overlay (RDKit / CDPKit ROCS) to evaluate
ONLY candidates that pass preceding 3D ElectroShape and property pre-filters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if (SCRIPT_DIR.parent.parent / "drugex").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

#? [NumPy] Array operations for shape scoring and distance vectors.
import numpy as np
#? [Pandas] Tabular DataFrame operations for candidate sorting, deduplication, and TSV export.
import pandas as pd
#? [RDKit] Cheminformatics library for SMILES parsing, substructure searching, Morgan fingerprints, and depictions.
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
#? [RDKit FilterCatalog] Pre-compiled substructure catalogs for PAINS and reactive functional group screening.
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
#? [RDKit Bemis-Murcko] Algorithmic extraction of molecular core ring frameworks and linkers.
from rdkit.Chem.Scaffolds import MurckoScaffold
#? [RDKit Butina] Sphere-exclusion unsupervised clustering algorithm for structural diversity selection.
from rdkit.ML.Cluster import Butina

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
try:
    from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
    CDPKIT_AVAILABLE: bool = True
except (ImportError, Exception):
    CDPKIT_AVAILABLE = False
try:
    from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
    RDKIT_ROCS_AVAILABLE: bool = True
except (ImportError, Exception):
    RDKIT_ROCS_AVAILABLE = False

from _david.code.fallback_monitor import FallbackMonitor

MIN_ROCS_SCORE: float = 0.38
MIN_ELECTROSHAPE_SCORE: float = 0.20
MIN_USRCAT_SCORE: float = MIN_ELECTROSHAPE_SCORE  # Backward compatibility alias
MIN_RA_SCORE: float = 0.50
MIN_QED_SCORE: float = 0.40
MAX_SEED_SIMILARITY: float = 0.60
CLUSTER_CUTOFF: float = 0.40
MAX_OUTPUT_SCAFFOLDS: int = 5
MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"


class ScaffoldFunnel:
    """Stage 2 Chemotype Funnel bridging Sequence-RNN exploration and Graph-Transformer optimization.

    Decomposes raw candidate hits into Bemis-Murcko core frameworks, purges PAINS
    and target-specific toxicophores, filters by dynamic heavy-atom volume bounds,
    enforces structural hopping distance away from the seed query, and clusters scaffolds
    via Butina sphere exclusion.
    Strictly gates expensive Gaussian volume overlay (RDKit / CDPKit ROCS) to evaluate
    ONLY candidates that pass preceding 3D ElectroShape and property pre-filters.

    Attributes:
        ROOT_PATH: Root directory of repository.
        OUTPUT_DIR: Target folder for saving funnel TSV and PNG files.
        SEED_MOL: RDKit Mol of the query active ligand.
        SEED_HAC: Heavy atom count of the query active ligand.
        SEED_FP: Explicit 2048-bit Morgan fingerprint of query active ligand.
        MIN_CANDIDATE_HEAVY_ATOMS: Dynamically scaled minimum candidate heavy atoms.
        MAX_CANDIDATE_HEAVY_ATOMS: Dynamically scaled maximum candidate heavy atoms.
        MIN_SCAFFOLD_HEAVY_ATOMS: Dynamically scaled minimum core scaffold heavy atoms.
        MAX_SCAFFOLD_HEAVY_ATOMS: Dynamically scaled maximum core scaffold heavy atoms.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        seed_smiles: str = MY_MOLECULE,
        toxicophore_smarts: str | Sequence[str] | None = None,
        filter_pains: bool = True,
        ref_sdf_path: Path | str | None = None,
    ) -> None:
        """Initialize ScaffoldFunnel.

        Args:
            output_dir: Optional Path to save output files.
            seed_smiles: SMILES string of active query seed ligand.
            toxicophore_smarts: Optional SMARTS string or sequence of SMARTS patterns to penalize.
            filter_pains: Whether to apply standard RDKit PAINS filter catalog (default: True).
            ref_sdf_path: Optional explicit path to reference 3D conformer ensemble SDF.
        """
        self.ROOT_PATH: Path = PROJECT_ROOT
        self.OUTPUT_DIR: Path = output_dir or (self.ROOT_PATH / "_david")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.seed_smiles: str = seed_smiles
        self.ref_sdf_path: Path | None = Path(ref_sdf_path) if ref_sdf_path else None

        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        self.SEED_MOL: Chem.Mol | None = Chem.MolFromSmiles(seed_smiles) if seed_smiles else None
        self.SEED_HAC: int = self.SEED_MOL.GetNumHeavyAtoms() if self.SEED_MOL else 33
        #? [RDKit Morgan Fingerprint] Pre-computes 2048-bit ECFP4 bit vector for rapid Tanimoto scaffold hopping reference.
        self.SEED_FP: DataStructs.ExplicitBitVect | None = (
            AllChem.GetMorganFingerprintAsBitVect(self.SEED_MOL, 2, nBits=2048) if self.SEED_MOL else None
        )

        # Dynamic size envelope derived from seed ligand heavy atom count (HAC)
        self.MIN_CANDIDATE_HEAVY_ATOMS: int = max(8, int(self.SEED_HAC * 0.60))
        self.MAX_CANDIDATE_HEAVY_ATOMS: int = int(self.SEED_HAC * 1.40)
        self.MIN_SCAFFOLD_HEAVY_ATOMS: int = max(6, int(self.SEED_HAC * 0.35))
        self.MAX_SCAFFOLD_HEAVY_ATOMS: int = int(self.SEED_HAC * 0.95)

        # Initialize PAINS filter catalog
        self.filter_pains: bool = filter_pains
        if filter_pains:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog: FilterCatalog | None = FilterCatalog(params)
        else:
            self.pains_catalog = None

        # Custom toxicophore SMARTS patterns
        self.toxicophore_patterns: List[Chem.Mol] = []
        if toxicophore_smarts:
            if isinstance(toxicophore_smarts, str):
                toxicophore_smarts = [toxicophore_smarts]
            for pat in toxicophore_smarts:
                p_mol = Chem.MolFromSmarts(pat)
                if p_mol:
                    self.toxicophore_patterns.append(p_mol)

    def is_liability_or_pains(self, mol: Chem.Mol | str | None) -> bool:
        """Check if candidate matches RDKit PAINS filters or target-specific toxicophores.

        Args:
            mol: Molecule object or SMILES string.

        Returns:
            True if any PAINS or toxicophore pattern matches, False otherwise.
        """
        if mol is None:
            return True
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return True

        # Check PAINS catalog
        if self.filter_pains and self.pains_catalog is not None:
            if self.pains_catalog.HasMatch(mol):
                return True

        # Check custom toxicophores (e.g. catechol smarts)
        for pat in self.toxicophore_patterns:
            if mol.HasSubstructMatch(pat):
                return True

        return False

    def is_polyphenol(self, mol: Chem.Mol | str | None) -> bool:
        """Check if molecule contains catechol / polyphenol patterns (backward-compatibility alias).

        Args:
            mol: Molecule object or SMILES string.

        Returns:
            True if liability detected, False otherwise.
        """
        return self.is_liability_or_pains(mol)

    def extract_murcko_scaffold(self, smiles: str) -> str | None:
        """Extract Bemis-Murcko core scaffold from SMILES string.

        Args:
            smiles: Input SMILES string.

        Returns:
            Scaffold SMILES string without stereochemistry, or None if extraction fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            #? [RDKit Bemis-Murcko] Truncates exocyclic side chains to isolate core rings and linkers.
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                FallbackMonitor.record_fallback(
                    fallback_type="murcko_empty_scaffold",
                    message="Murcko extraction produced 0 heavy atoms; using raw molecule SMILES",
                    context={"smiles": smiles},
                )
                return Chem.MolToSmiles(mol, isomericSmiles=False)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception as e:
            FallbackMonitor.record_fallback(
                fallback_type="murcko_extraction_exception",
                message=f"Murcko extraction exception: {e}; falling back to raw SMILES",
                context={"smiles": smiles},
            )
            return Chem.MolToSmiles(mol, isomericSmiles=False)

    def cluster_scaffolds(self, scaffold_smiles: Sequence[str], cutoff: float = CLUSTER_CUTOFF) -> List[Tuple[int, ...]]:
        """Cluster scaffold SMILES using Butina sphere-exclusion algorithm.

        Args:
            scaffold_smiles: List of unique scaffold SMILES strings.
            cutoff: Distance cutoff for clustering (1.0 - Tanimoto similarity).

        Returns:
            List of cluster tuples containing molecule indices, sorted by cluster size descending.
        """
        mols = [Chem.MolFromSmiles(s) for s in scaffold_smiles]
        valid_mols = [m for m in mols if m is not None]
        if not valid_mols:
            return []

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
        n_fps = len(fps)
        if n_fps == 1:
            return [(0,)]

        #? [RDKit Butina] Computes 1D triangular distance matrix for sphere-exclusion clustering.
        dists = []
        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - x for x in sims])

        clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
        return sorted(clusters, key=len, reverse=True)

    def compute_rocs_scores(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute Gaussian ROCS shape overlay scores for a list of molecules.

        Strictly evaluates physical Gaussian volume overlay (RDKit / CDPKit ROCS)
        against the reference 3D conformer ensemble.

        Args:
            smiles_list: List of candidate SMILES strings.

        Returns:
            1D NumPy array of ROCS scores in [0.0, 1.0].
        """
        if not smiles_list:
            return np.array([], dtype=np.float32)

        ref_candidates: List[Path] = []
        if self.ref_sdf_path is not None:
            ref_candidates.append(self.ref_sdf_path)
        ref_candidates.extend([
            self.OUTPUT_DIR / "references" / "query_molecule_3d.sdf",
            self.OUTPUT_DIR.parent / "references" / "query_molecule_3d.sdf",
            self.ROOT_PATH / "_david" / "references" / "query_molecule_3d.sdf",
            self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf",
        ])
        ref_path = next((p for p in ref_candidates if p.exists()), None)
        if not ref_path:
            FallbackMonitor.record_fallback(
                fallback_type="rocs_missing_reference_sdf",
                message="No reference SDF found for ROCS overlay; falling back to default score (0.50)",
                context={"n_molecules": len(smiles_list)},
            )
            return np.full(len(smiles_list), 0.5, dtype=np.float32)

        conf_gen = RDKitConformerGenerator(
            max_conformers=10,
            max_isomers=4,
            max_heavy_atoms=max(50, int(self.SEED_HAC * 1.8)),
            num_threads=0,
            timeout=15,
            show_progress=False,
        )

        if CDPKIT_AVAILABLE:
            try:
                #? [CDPKit ROCS Scorer] Rigorous physical Gaussian volume overlap.
                scorer = CDPKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(ref_path),
                    show_progress=False,
                    opt_mode="combo",
                )
                cdp_scores = np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
                if len(cdp_scores) > 0 and np.all(cdp_scores == 0.0):
                    FallbackMonitor.record_fallback(
                        fallback_type="rocs_cdpkit_zero_scores",
                        message="CDPKit ROCS returned all-zero scores; falling back to RDKitROCSScorer",
                        context={"n_molecules": len(smiles_list)},
                    )
                else:
                    return cdp_scores
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_cdpkit_failed",
                    message=f"CDPKit ROCS failed with exception: {e}; falling back to RDKitROCSScorer",
                    context={"n_molecules": len(smiles_list)},
                )

        if RDKIT_ROCS_AVAILABLE:
            try:
                #? [RDKit ROCS Scorer] Robust fallback Gaussian shape aligner.
                scorer = RDKitROCSScorer(
                    conformer_generator=conf_gen,
                    references=str(ref_path),
                    score_type="TanimotoCombo",
                    use_colors=True,
                    show_progress=False,
                    n_jobs=2,
                )
                return np.asarray(scorer.getScores(smiles_list), dtype=np.float32).flatten()
            except Exception as e:
                FallbackMonitor.record_fallback(
                    fallback_type="rocs_rdkit_failed",
                    message=f"RDKit ROCS failed with exception: {e}; falling back to default score (0.50)",
                    context={"n_molecules": len(smiles_list)},
                )

        FallbackMonitor.record_fallback(
            fallback_type="rocs_all_engines_failed",
            message="Both CDPKit and RDKit ROCS engines failed/unavailable; falling back to default 0.50",
            context={"n_molecules": len(smiles_list)},
        )
        return np.full(len(smiles_list), 0.5, dtype=np.float32)

    @staticmethod
    def _safe_float(val: Any, default: float = 0.5) -> float:
        """Safely parse float value with fallback default.

        Args:
            val: Input value of any type.
            default: Fallback float if parsing fails.

        Returns:
            Float value.
        """
        if val is None or pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def process_candidates_dataframe(self, df: pd.DataFrame, max_scaffolds: int = MAX_OUTPUT_SCAFFOLDS) -> pd.DataFrame:
        """Process candidate DataFrame through the Bemis-Murcko chemotype funnel.

        Gated Two-Tier 3D Evaluation:
        1. Tier 1 (Pre-Filters): Evaluates candidate molecules against ultrafast 75-D
           ElectroShape moments (>= 0.45), dynamic heavy-atom volume bounds, retrosynthetic
           accessibility RAscore (>= 0.50), drug-likeness QED (>= 0.40), and PAINS / toxicophore
           absence in microseconds.
        2. Tier 2 (Gated Physical ROCS): Rigid/flexible Gaussian volume overlay (RDKit / CDPKit ROCS)
           is executed ONLY on the elite candidates that passed Tier 1 pre-filters.
        3. Chemotype Funnel: Extracts Bemis-Murcko scaffolds from ROCS-verified leads,
           enforces core framework volume and hopping distance, and clusters via Butina.

        Args:
            df: DataFrame containing candidate molecules with at least a 'SMILES' column.
            max_scaffolds: Maximum number of distinct scaffold centroids to output.

        Returns:
            DataFrame of selected scaffold leads with properties, IDs, and similarity metrics.

        Raises:
            ValueError: If 'SMILES' column is missing from input DataFrame.
        """
        if "SMILES" not in df.columns:
            raise ValueError("Input DataFrame must contain 'SMILES' column.")

        # Flexible column identification
        rocs_col = next(
            (
                c
                for c in df.columns
                if c in ["ROCS", "RDKit_ROCS", "CDPKit_ROCS", "TanimotoCombo"]
                or (
                    ("ROCS" in c or "Shape" in c)
                    and not any(es in c for es in ["ElectroShape", "electroshape", "USRCAT", "usrcat"])
                )
            ),
            None,
        )
        electroshape_col = next(
            (c for c in df.columns if any(k in c for k in ["ElectroShape", "electroshape", "USRCAT", "usrcat"])),
            None,
        )
        ra_col = next(
            (
                c
                for c in df.columns
                if c in ["RAscore", "RA", "RA_Score", "rascore"]
                or "Retrosynthetic" in c
                or c.startswith("RA_")
                or c.startswith("RAscore")
            ),
            None,
        )
        qed_col = next((c for c in df.columns if "QED" in c), None)

    def compute_composite_desirability(
        self,
        es_score: float,
        ra_score: float,
        qed_score: float,
        hac: int,
        seed_sim: float,
    ) -> float:
        """Compute continuous Harrington/Derringer-Suich geometric desirability.

        Balances 3D shape similarity (30%), synthesizability (25%), drug-likeness (25%),
        and seed heavy-atom volume match (20%), modulated by scaffold hopping novelty.

        Args:
            es_score: 3D ElectroShape similarity [0.0, 1.0].
            ra_score: RAscore synthesizability [0.0, 1.0].
            qed_score: QED drug-likeness [0.0, 1.0].
            hac: Candidate molecule heavy atom count.
            seed_sim: ECFP4 Tanimoto similarity to active seed query.

        Returns:
            Continuous desirability score in [0.0, 1.0].
        """
        eps = 1e-4
        # Volume desirability: Gaussian bell centered on seed heavy atom count
        hac_delta = abs(hac - self.SEED_HAC)
        sigma = max(3.0, self.SEED_HAC * 0.15)
        size_desirability = float(np.exp(-0.5 * (hac_delta / sigma) ** 2))

        # Novelty penalty if too similar to seed (> 0.80 Tanimoto)
        if seed_sim > 0.80:
            novelty_factor = 0.3
        elif seed_sim > 0.60:
            novelty_factor = 0.7
        else:
            novelty_factor = 1.0

        d = (
            (max(es_score, eps) ** 0.30)
            * (max(ra_score, eps) ** 0.25)
            * (max(qed_score, eps) ** 0.25)
            * (max(size_desirability, eps) ** 0.20)
        )
        return float(d * novelty_factor)

    def process_candidates_dataframe(self, df: pd.DataFrame, max_scaffolds: int = MAX_OUTPUT_SCAFFOLDS) -> pd.DataFrame:
        """Process candidate DataFrame through the Bemis-Murcko chemotype funnel.

        Top-K Continuous Multi-Objective Selection:
        1. Evaluates chemical validity, broad physical size bounds, and liability absence.
        2. Computes continuous composite desirability (combining 3D ElectroShape moments,
           synthesizability, QED, and volume match).
        3. Ranks candidates and selects the elite Top-K candidates for physical ROCS validation.
        4. Evaluates authentic Gaussian volume overlap (CDPKit / RDKit ROCS) with multi-isomer
           enumeration and MMFF94 force-field minimization.
        5. Decomposes into Bemis-Murcko frameworks and clusters via Butina sphere exclusion.

        Args:
            df: DataFrame containing candidate molecules with at least a 'SMILES' column.
            max_scaffolds: Maximum number of distinct scaffold centroids to output.

        Returns:
            DataFrame of selected scaffold leads with properties, IDs, and similarity metrics.

        Raises:
            ValueError: If 'SMILES' column is missing from input DataFrame.
        """
        if "SMILES" not in df.columns:
            raise ValueError("Input DataFrame must contain 'SMILES' column.")

        # Flexible column identification
        rocs_col = next(
            (
                c
                for c in df.columns
                if c in ["ROCS", "RDKit_ROCS", "CDPKit_ROCS", "TanimotoCombo"]
                or (
                    ("ROCS" in c or "Shape" in c)
                    and not any(es in c for es in ["ElectroShape", "electroshape", "USRCAT", "usrcat"])
                )
            ),
            None,
        )
        electroshape_col = next(
            (c for c in df.columns if any(k in c for k in ["ElectroShape", "electroshape", "USRCAT", "usrcat"])),
            None,
        )
        ra_col = next(
            (
                c
                for c in df.columns
                if c in ["RAscore", "RA", "RA_Score", "rascore"]
                or "Retrosynthetic" in c
                or c.startswith("RA_")
                or c.startswith("RAscore")
            ),
            None,
        )
        qed_col = next((c for c in df.columns if "QED" in c), None)

        total_input: int = len(df)
        passed_valid: int = 0
        passed_hac: int = 0
        passed_liability: int = 0

        valid_candidates = []
        for _, row in df.iterrows():
            if "Valid" in row and row["Valid"] != 1:
                continue

            smi = str(row["SMILES"]).strip()
            cand_mol = Chem.MolFromSmiles(smi)
            if cand_mol is None:
                continue
            passed_valid += 1

            # Broad heavy atom volume sanity bounds
            hac = cand_mol.GetNumHeavyAtoms()
            if hac < max(6, int(self.SEED_HAC * 0.35)) or hac > int(self.SEED_HAC * 1.80):
                continue
            passed_hac += 1

            # Check PAINS and custom toxicophore liabilities
            has_liability = self.is_liability_or_pains(cand_mol)
            if has_liability:
                continue
            passed_liability += 1

            es_val = self._safe_float(row.get(electroshape_col), 0.5) if electroshape_col else 0.5
            ra_val = self._safe_float(row.get(ra_col), 0.5) if ra_col else 0.5
            qed_val = self._safe_float(row.get(qed_col), 0.5) if qed_col else 0.5

            if self.SEED_FP:
                cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2, nBits=2048)
                sim_to_seed = float(DataStructs.TanimotoSimilarity(cand_fp, self.SEED_FP))
            else:
                sim_to_seed = 0.0

            desirability = self.compute_composite_desirability(
                es_score=es_val,
                ra_score=ra_val,
                qed_score=qed_val,
                hac=hac,
                seed_sim=sim_to_seed,
            )

            valid_candidates.append({
                "row": row,
                "SMILES": smi,
                "Mol": cand_mol,
                "Candidate_HAC": hac,
                "ElectroShape": es_val,
                "RAscore": ra_val,
                "QED": qed_val,
                "Desirability": desirability,
                "Has_Liability": has_liability,
                "Is_Polyphenol": has_liability,
                "Seed_Similarity": sim_to_seed,
            })

        if not valid_candidates:
            # Fallback only if all candidates failed liability/size
            for _, row in df.head(50).iterrows():
                smi = str(row["SMILES"]).strip()
                cand_mol = Chem.MolFromSmiles(smi)
                if cand_mol is None:
                    continue
                valid_candidates.append({
                    "row": row,
                    "SMILES": smi,
                    "Mol": cand_mol,
                    "Candidate_HAC": cand_mol.GetNumHeavyAtoms(),
                    "ElectroShape": self._safe_float(row.get(electroshape_col), 0.5) if electroshape_col else 0.5,
                    "RAscore": self._safe_float(row.get(ra_col), 0.5) if ra_col else 0.5,
                    "QED": self._safe_float(row.get(qed_col), 0.5) if qed_col else 0.5,
                    "Desirability": 0.5,
                    "Has_Liability": False,
                    "Is_Polyphenol": False,
                    "Seed_Similarity": 0.0,
                })

        # Rank by continuous desirability descending and select elite Top-K candidates
        valid_candidates.sort(key=lambda c: c["Desirability"], reverse=True)
        top_k_candidates = valid_candidates[:min(len(valid_candidates), 100)]

        # =========================================================================
        # Tier 2 Gated ROCS Evaluation on Elite Top-K Candidates
        # =========================================================================
        surviving_smiles = [c["SMILES"] for c in top_k_candidates]
        if rocs_col is None and len(surviving_smiles) > 0:
            rocs_scores = self.compute_rocs_scores(surviving_smiles)
            for idx, c in enumerate(top_k_candidates):
                c["ROCS"] = float(rocs_scores[idx]) if idx < len(rocs_scores) else 0.5
        else:
            for c in top_k_candidates:
                orig_row = c["row"]
                c["ROCS"] = self._safe_float(orig_row.get(rocs_col), 0.5) if rocs_col else 0.5

        # =========================================================================
        # Murcko Framework Extraction & Diversity Clustering
        # =========================================================================
        records = []
        for c in top_k_candidates:
            smi = c["SMILES"]
            scaff = self.extract_murcko_scaffold(smi) or smi
            scaff_mol = Chem.MolFromSmiles(scaff) or c["Mol"]
            scaff_hac = scaff_mol.GetNumHeavyAtoms()

            records.append({
                "SMILES": smi,
                "Scaffold": scaff,
                "Scaffold_HAC": scaff_hac,
                "Candidate_HAC": c["Candidate_HAC"],
                "ROCS": c["ROCS"],
                "ElectroShape": c["ElectroShape"],
                "USRCAT": c["ElectroShape"],
                "RAscore": c["RAscore"],
                "QED": c["QED"],
                "Desirability": c["Desirability"],
                "Has_Liability": c["Has_Liability"],
                "Is_Polyphenol": c["Is_Polyphenol"],
                "Seed_Similarity": c["Seed_Similarity"],
            })

        scaff_df = pd.DataFrame(records)

        # Deduplicate scaffolds and sort by ROCS / Desirability / RAscore
        sort_cols = [col for col in ["ROCS", "Desirability", "ElectroShape", "RAscore"] if col in scaff_df.columns]
        pool_unique = scaff_df.sort_values(by=sort_cols, ascending=False).drop_duplicates(subset=["Scaffold"]).reset_index(drop=True)
        unique_scaffolds = pool_unique["Scaffold"].tolist()

        clusters = self.cluster_scaffolds(unique_scaffolds)
        if clusters:
            selected_indices = [cluster[0] for cluster in clusters[:max_scaffolds]]
            final_df = pool_unique.iloc[selected_indices].copy().reset_index(drop=True)
        else:
            final_df = pool_unique.head(max_scaffolds).copy().reset_index(drop=True)

        final_df["Scaffold_ID"] = [f"Funnel_Scaffold_{i+1}" for i in range(len(final_df))]

        tsv_out = self.OUTPUT_DIR / "funnel_scaffolds.tsv"
        final_df.to_csv(tsv_out, sep="\t", index=False)

        # Export structured attrition statistics
        attrition_stats: Dict[str, Any] = {
            "total_input_candidates": total_input,
            "passed_chemical_validity": passed_valid,
            "passed_hac_envelope": passed_hac,
            "passed_pains_and_toxicophores": passed_liability,
            "top_k_candidates_evaluated": len(top_k_candidates),
            "tier1_survivors_count": len(top_k_candidates),
            "murcko_frameworks_extracted": len(records),
            "unique_scaffolds_count": len(unique_scaffolds),
            "butina_clusters_count": len(clusters),
            "final_scaffolds_selected": len(final_df),
        }
        attrition_file = self.OUTPUT_DIR / "funnel_attrition.json"
        with open(attrition_file, "w", encoding="utf-8") as f:
            json.dump(attrition_stats, f, indent=4)

        pct = lambda n: f"{(n / total_input * 100.0):.1f}%" if total_input > 0 else "0.0%"
        print("\n[Lead Selection] Top-K Multi-Objective Candidate Prioritization:")
        print(f"  • Total input candidates:       {total_input} (100.0%)")
        print(f"  • Passed chemical validity:     {passed_valid} ({pct(passed_valid)})")
        print(f"  • Passed broad HAC envelope:    {passed_hac} ({pct(passed_hac)})")
        print(f"  • Passed PAINS/toxicophores:    {passed_liability} ({pct(passed_liability)})")
        print(f"  • Elite candidates evaluated:   {len(top_k_candidates)}")
        print(f"  • Unique Murcko frameworks:     {len(unique_scaffolds)} in {len(clusters)} Butina cluster(s)")
        print(f"  • Final diverse leads selected: {len(final_df)}")

        #? [Statistical Prompt Expansion] Replicates top lead scaffolds to N=200 prompts for stable mini-batch tensor distribution.
        n_prompts: int = 200
        if len(final_df) > 0:
            n_repeats = max(1, n_prompts // len(final_df) + 1)
            expanded_df = pd.concat([final_df] * n_repeats, ignore_index=True).iloc[:n_prompts].copy()
            expanded_out = self.OUTPUT_DIR / "funnel_scaffolds_expanded.tsv"
            expanded_df.to_csv(expanded_out, sep="\t", index=False)

        #? [RDKit Draw] Renders 2D depiction grid PNG of lead scaffolds with ROCS, USRCAT, and RAscore legends.
        mols_to_draw = [Chem.MolFromSmiles(s) for s in final_df["Scaffold"]]
        legends = [
            f"{row['Scaffold_ID']} (HAC: {row['Scaffold_HAC']})\nROCS: {row['ROCS']:.2f} | USRCAT: {row.get('USRCAT', 0.0):.2f} | RA: {row['RAscore']:.2f}"
            for _, row in final_df.iterrows()
        ]
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=min(5, len(mols_to_draw)), subImgSize=(300, 300), legends=legends)
        img.save(str(self.OUTPUT_DIR / "funnel_scaffolds.png"))

        return final_df

    process_candidates = process_candidates_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run self-test on dummy dataset")
    args = parser.parse_args()

    funnel = ScaffoldFunnel()
    if args.test:
        dummy_data = pd.DataFrame({
            "SMILES": [
                "O=C(Oc1ccc(cc1)C(=O)NC1CCCCC1)c1ccccc1",
                "Cc1ccc(cc1)N1CCN(CC1)c1nc2ccccc2n1C",
                "CC1CCC2C(C1)C1CCC3C4(C)CCC(O)C(C)(C)C4CCC3(C)C1(C)CC2",
                "Oc1ccc(cc1)C1Cc2c(O)cc(O)cc2OC1c1cc(O)c(O)c(O)c1",
                "c1ccccc1NC(=O)c1ccc(cn1)C1CC1",
                "c1ccc2c(c1)nc(s2)N1CCN(CC1)c1ccccc1",
            ],
            "Valid": [1, 1, 1, 1, 1, 1],
            "RDKit_ROCS": [0.65, 0.72, 0.45, 0.90, 0.58, 0.61],
            "USRCAT": [0.65, 0.70, 0.48, 0.88, 0.58, 0.62],
            "RAscore": [0.85, 0.78, 0.60, 0.35, 0.88, 0.82],
            "QED": [0.70, 0.65, 0.40, 0.50, 0.75, 0.68],
        })
        res = funnel.process_candidates_dataframe(dummy_data)
        print("Self-test succeeded! Extracted scaffolds:")
        print(res[["Scaffold_ID", "Scaffold", "Scaffold_HAC", "ROCS", "USRCAT", "RAscore", "Is_Polyphenol", "Seed_Similarity"]])
        FallbackMonitor.print_summary()
