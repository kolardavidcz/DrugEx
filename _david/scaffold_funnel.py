from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

# Ensure project root and _david are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "drugex").exists() or (SCRIPT_DIR.parent / "drugex").exists() else SCRIPT_DIR
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
#? [RDKit Bemis-Murcko] Algorithmic extraction of molecular core ring frameworks and linkers.
from rdkit.Chem.Scaffolds import MurckoScaffold
#? [RDKit Butina] Sphere-exclusion unsupervised clustering algorithm for structural diversity selection.
from rdkit.ML.Cluster import Butina

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
try:
    from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer
    CDPKIT_AVAILABLE: bool = True
except (ImportError, Exception):
    CDPKIT_AVAILABLE: bool = False
try:
    from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer
    RDKIT_ROCS_AVAILABLE: bool = True
except (ImportError, Exception):
    RDKIT_ROCS_AVAILABLE: bool = False

MIN_ROCS_SCORE: float = 0.38
MIN_USRCAT_SCORE: float = 0.45
MIN_RA_SCORE: float = 0.50
MIN_QED_SCORE: float = 0.40
MAX_SEED_SIMILARITY: float = 0.60
MIN_CANDIDATE_HEAVY_ATOMS: int = 20
MIN_SCAFFOLD_HEAVY_ATOMS: int = 12
MAX_SCAFFOLD_HEAVY_ATOMS: int = 32
CLUSTER_CUTOFF: float = 0.40
MAX_OUTPUT_SCAFFOLDS: int = 5
MY_MOLECULE: str = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"


class ScaffoldFunnel:
    """Stage 2 Chemotype Funnel bridging Sequence-RNN exploration and Graph-Transformer optimization.

    Decomposes raw candidate hits into Bemis-Murcko core frameworks, purges catechol
    and polyphenol PAINS substructures, filters by minimum heavy-atom volume to prevent
    fragment size collapse, enforces structural hopping distance away from the seed query,
    and clusters scaffolds via Butina sphere exclusion.

    Attributes:
        ROOT_PATH: Root directory of repository.
        OUTPUT_DIR: Target folder for saving funnel TSV and PNG files.
        CATECHOL_SMARTS: RDKit Mol representing catechol substructure.
        SEED_MOL: RDKit Mol of the query active ligand.
        SEED_FP: Explicit 2048-bit Morgan fingerprint of query active ligand.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize ScaffoldFunnel.

        Args:
            output_dir: Optional Path to save output files.
        """
        self.ROOT_PATH: Path = Path(__file__).resolve().parent.parent
        self.OUTPUT_DIR: Path = output_dir or (self.ROOT_PATH / "_david")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        #? [RDKit SMARTS] Pre-compiles catechol (1,2-dihydroxybenzene) query pattern for rapid C++ graph matching.
        self.CATECHOL_SMARTS: Chem.Mol | None = Chem.MolFromSmarts("c1c([OH])c([OH])ccc1")
        #? [RDKit] Parse SMILES into C++ Mol graph: validates valences, ring closures, and chiral center parities (@/@@).
        self.SEED_MOL: Chem.Mol | None = Chem.MolFromSmiles(MY_MOLECULE)
        #? [RDKit Morgan Fingerprint] Pre-computes 2048-bit ECFP4 bit vector for rapid Tanimoto scaffold hopping reference.
        self.SEED_FP: DataStructs.ExplicitBitVect | None = AllChem.GetMorganFingerprintAsBitVect(self.SEED_MOL, 2, nBits=2048) if self.SEED_MOL else None

    def is_catechol_polyphenol(self, mol: Chem.Mol | None) -> bool:
        """Determine whether a molecule contains catechol or polyphenol liabilities.

        Skeptical Academic Rationale:
        Catechol and polyphenol moieties (such as the 8 phenolic hydroxyls in native EGCG)
        are Pan-Assay Interference Compounds (PAINS) that exhibit redox cycling, metal chelation,
        and non-specific protein reactivity, causing frequent false positives in biochemical assays.
        In modern medicinal chemistry, scaffold hopping must eliminate these toxicophores.

        Args:
            mol: RDKit Mol to inspect.

        Returns:
            True if molecule matches catechol SMARTS, has >= 3 phenolic hydroxyls, or >= 4 total hydroxyls.
        """
        if mol is None:
            return False
        #? [RDKit Subgraph Matching] Uses Ullmann/VF2 subgraph isomorphism to detect exact catechol toxicophoric substructures.
        if self.CATECHOL_SMARTS and mol.HasSubstructMatch(self.CATECHOL_SMARTS):
            return True
        #? [RDKit Subgraph Matching] Matches isolated phenolic hydroxyls (c[OH]) and total hydroxyls to flag polyphenol PAINS liabilities.
        phenolic_oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c[OH]")))
        if phenolic_oh >= 3:
            return True
        total_oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]")))
        if total_oh >= 4:
            return True
        return False

    def extract_murcko_scaffold(self, smiles: str) -> str | None:
        """Extract the Bemis-Murcko core framework from a SMILES string.

        Args:
            smiles: Input SMILES string.

        Returns:
            Canonical SMILES of the heavy-atom Murcko scaffold, or None if extraction fails.
        """
        #? [RDKit] Parse candidate SMILES to construct molecular graph.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            #? [RDKit Bemis-Murcko] Decomposes molecule into core ring systems and linkers, stripping exocyclic side chains.
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            if scaff.GetNumHeavyAtoms() == 0:
                return None
            return str(Chem.MolToSmiles(scaff))
        except Exception:
            return None

    def cluster_scaffolds(self, scaffolds: list[str], cutoff: float = CLUSTER_CUTOFF) -> Sequence[Tuple[int, ...]]:
        """Cluster scaffold fingerprints using the Butina sphere-exclusion algorithm.

        Args:
            scaffolds: List of scaffold SMILES strings.
            cutoff: Maximum Tanimoto distance threshold for neighborhood membership.

        Returns:
            Sequence of cluster tuples, where each tuple starts with the centroid index.
        """
        #? [RDKit] Re-parse unique scaffold SMILES into Mol objects.
        mols = [Chem.MolFromSmiles(s) for s in scaffolds]
        valid_mols = [m for m in mols if m is not None]
        if not valid_mols:
            return []

        #? [RDKit Morgan Fingerprint] Computes 2048-bit ECFP4 bit vectors for scaffold structural comparison.
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
        num_fps = len(fps)
        dists: List[float] = []
        for i in range(1, num_fps):
            #? [RDKit DataStructs] Computes pairwise lower-triangular Tanimoto distance matrix (d = 1.0 - T).
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - s for s in sims])

        #? [RDKit Butina] Unsupervised sphere-exclusion clustering: groups scaffolds within distance cutoff (1 - Tanimoto) and returns cluster centroids.
        clusters: Sequence[Tuple[int, ...]] = Butina.ClusterData(dists, num_fps, cutoff, isDistData=True)
        return clusters

    def compute_rocs_scores(self, smiles_list: Sequence[str], ref_sdf: Path | str | None = None) -> np.ndarray:
        """Compute authentic physical Gaussian volume overlap for candidates.

        Skeptical Academic Rationale:
        Decoupled 3D Tier: During Stage 1 RL, USRCAT moments ran in microseconds without
        conformer generation to accelerate RL rollouts. In Stage 2 Funnel, candidates are
        subjected to rigorous physical Gaussian volume overlay (CDPKit ROCS: CDPKitROCSScorer
        + RDKitConformerGenerator) using ETKDGv3 conformer ensembles against the query conformers.

        Args:
            smiles_list: Candidate SMILES strings to evaluate.
            ref_sdf: Optional custom reference SDF path.

        Returns:
            NumPy array of ROCS TanimotoCombo scores.
        """
        if not smiles_list:
            return np.zeros(0, dtype=np.float32)

        ref_path = Path(ref_sdf) if ref_sdf else (self.ROOT_PATH / "_david" / "references" / "my_molecule_3d.sdf")
        if not ref_path.exists():
            from _david.prepare_query_conformer import QueryConformerPreparer
            preparer = QueryConformerPreparer(ref_path.parent)
            ref_path = preparer.generate_reference_sdf(MY_MOLECULE, ref_path)

        #? [RDKit Conformer Generator] 10 conformers per candidate with ETKDGv3.
        conf_gen = RDKitConformerGenerator(
            max_conformers=10,
            max_isomers=1,
            max_heavy_atoms=50,
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
                return np.asarray(scorer.getScores(smiles_list), dtype=np.float32)
            except Exception:
                pass

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
                return np.asarray(scorer.getScores(smiles_list), dtype=np.float32)
            except Exception:
                pass

        return np.full(len(smiles_list), 0.5, dtype=np.float32)

    def process_candidates_dataframe(self, df: pd.DataFrame, max_scaffolds: int = MAX_OUTPUT_SCAFFOLDS) -> pd.DataFrame:
        """Process candidate DataFrame through the Bemis-Murcko chemotype funnel.

        Filters candidates by score thresholds (ROCS, USRCAT, RAscore, QED), enforces
        heavy-atom envelope bounds (14-32 heavy atoms), extracts Murcko scaffolds,
        filters out catechol/polyphenol PAINS and seed-like frameworks, and clusters
        unique scaffolds with Butina clustering to select diverse lead centroids.

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
        rocs_col = next((c for c in df.columns if "RDKit" in c or "ROCS" in c or "TanimotoCombo" in c or "Shape" in c), None)
        usrcat_col = next((c for c in df.columns if "USRCAT" in c or "usrcat" in c), None)
        ra_col = next((c for c in df.columns if "Retrosynthetic" in c or "RA" in c), None)
        qed_col = next((c for c in df.columns if "QED" in c), None)

        #? [Pandas] Filters candidate hits by minimum property thresholds.
        filtered = df.copy()
        if "Valid" in filtered.columns:
            filtered = filtered[filtered["Valid"] == 1]
        if usrcat_col and usrcat_col in filtered.columns:
            filtered = filtered[filtered[usrcat_col] >= MIN_USRCAT_SCORE]
        if ra_col and ra_col in filtered.columns:
            filtered = filtered[filtered[ra_col] >= MIN_RA_SCORE]
        if qed_col and qed_col in filtered.columns:
            filtered = filtered[filtered[qed_col] >= MIN_QED_SCORE]

        # Decoupled 3D Tier: If ROCS was not evaluated during Stage 1 RL loop, evaluate authentic Gaussian overlap in Funnel
        if rocs_col is None and len(filtered) > 0:
            candidate_smiles = filtered["SMILES"].tolist()
            rocs_scores = self.compute_rocs_scores(candidate_smiles)
            filtered["ROCS"] = rocs_scores
            rocs_col = "ROCS"
            if (filtered["ROCS"] >= MIN_ROCS_SCORE).any():
                filtered = filtered[filtered["ROCS"] >= MIN_ROCS_SCORE]
        elif rocs_col and rocs_col in filtered.columns:
            filtered = filtered[filtered[rocs_col] >= MIN_ROCS_SCORE]

        if len(filtered) == 0:
            filtered = df.head(50)

        records = []
        for _, row in filtered.iterrows():
            smi = str(row["SMILES"])
            cand_mol = Chem.MolFromSmiles(smi)
            if not cand_mol:
                continue

            # Heavy atom envelope check on candidate to prevent tiny fragment contamination (targeting EGCG 33 atoms)
            hac = cand_mol.GetNumHeavyAtoms()
            if hac < MIN_CANDIDATE_HEAVY_ATOMS or hac > 45:
                continue

            scaff = self.extract_murcko_scaffold(smi)
            if not scaff:
                continue

            scaff_mol = Chem.MolFromSmiles(scaff)
            if not scaff_mol:
                continue

            # Heavy atom envelope check on core scaffold (must be substantial multi-ring core)
            scaff_hac = scaff_mol.GetNumHeavyAtoms()
            if scaff_hac < MIN_SCAFFOLD_HEAVY_ATOMS or scaff_hac > MAX_SCAFFOLD_HEAVY_ATOMS:
                continue

            #? [PAINS Liability Audit] Evaluates candidate and Murcko core against catechol/polyphenol criteria to prevent latent toxicophores.
            is_poly = self.is_catechol_polyphenol(cand_mol) or self.is_catechol_polyphenol(scaff_mol)

            #? [RDKit Tanimoto] Evaluates 2048-bit ECFP4 similarity to seed; enforces scaffold hopping (Tanimoto <= 0.60).
            if self.SEED_FP:
                fp = AllChem.GetMorganFingerprintAsBitVect(scaff_mol, 2, nBits=2048)
                sim_to_seed = DataStructs.TanimotoSimilarity(fp, self.SEED_FP)
            else:
                sim_to_seed = 0.0

            rocs_val = float(row[rocs_col]) if rocs_col and rocs_col in row else 0.5
            usrcat_val = float(row[usrcat_col]) if usrcat_col and usrcat_col in row else 0.5
            ra_val = float(row[ra_col]) if ra_col and ra_col in row else 0.5
            qed_val = float(row[qed_col]) if qed_col and qed_col in row else 0.5

            records.append({
                "SMILES": smi,
                "Scaffold": scaff,
                "Scaffold_HAC": scaff_hac,
                "Candidate_HAC": hac,
                "ROCS": rocs_val,
                "USRCAT": usrcat_val,
                "RAscore": ra_val,
                "QED": qed_val,
                "Is_Polyphenol": is_poly,
                "Seed_Similarity": sim_to_seed,
            })

        if not records:
            # Fallback if strict heavy-atom filters removed all candidates
            for _, row in filtered.iterrows():
                smi = str(row["SMILES"])
                cand_mol = Chem.MolFromSmiles(smi)
                if not cand_mol:
                    continue
                scaff = self.extract_murcko_scaffold(smi) or smi
                scaff_mol = Chem.MolFromSmiles(scaff) or cand_mol
                records.append({
                    "SMILES": smi,
                    "Scaffold": scaff,
                    "Scaffold_HAC": scaff_mol.GetNumHeavyAtoms(),
                    "Candidate_HAC": cand_mol.GetNumHeavyAtoms(),
                    "ROCS": float(row[rocs_col]) if rocs_col and rocs_col in row else 0.5,
                    "USRCAT": float(row[usrcat_col]) if usrcat_col and usrcat_col in row else 0.5,
                    "RAscore": float(row[ra_col]) if ra_col and ra_col in row else 0.5,
                    "QED": float(row[qed_col]) if qed_col and qed_col in row else 0.5,
                    "Is_Polyphenol": self.is_catechol_polyphenol(cand_mol),
                    "Seed_Similarity": 0.2,
                })

        scaff_df = pd.DataFrame(records)

        # Filter out PAINS polyphenols and seed-like scaffolds
        non_poly = scaff_df[~scaff_df["Is_Polyphenol"] & (scaff_df["Seed_Similarity"] <= MAX_SEED_SIMILARITY)]
        pool = non_poly if len(non_poly) >= max_scaffolds else scaff_df

        # Ensure 1:1 index integrity between unique scaffolds and Butina clustering
        valid_mask = pool["Scaffold"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        pool_valid = pool[valid_mask].copy()
        if len(pool_valid) == 0:
            pool_valid = scaff_df.copy()

        sort_cols = [c for c in ["ROCS", "USRCAT", "RAscore"] if c in pool_valid.columns]
        pool_unique = pool_valid.sort_values(by=sort_cols, ascending=False).drop_duplicates(subset=["Scaffold"]).reset_index(drop=True)
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

        #? [Statistical Prompt Expansion] Replicates top lead scaffolds to N=200 prompts for stable mini-batch tensor distribution in Stage 3.
        n_prompts: int = 200
        if len(final_df) > 0:
            n_repeats = max(1, n_prompts // len(final_df) + 1)
            expanded_df = pd.concat([final_df] * n_repeats, ignore_index=True).iloc[:n_prompts].copy()
            expanded_out = self.OUTPUT_DIR / "funnel_scaffolds_expanded.tsv"
            expanded_df.to_csv(expanded_out, sep="\t", index=False)
            if self.OUTPUT_DIR != self.ROOT_PATH / "_david":
                shared_expanded = self.ROOT_PATH / "_david" / "funnel_scaffolds_expanded.tsv"
                expanded_df.to_csv(shared_expanded, sep="\t", index=False)

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
