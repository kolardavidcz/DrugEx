from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

MIN_ROCS_SCORE = 0.40
MIN_RA_SCORE = 0.50
MIN_QED_SCORE = 0.40
MAX_SEED_SIMILARITY = 0.60
CLUSTER_CUTOFF = 0.40
MAX_OUTPUT_SCAFFOLDS = 5
MY_MOLECULE = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"


class ScaffoldFunnel:
    def __init__(self, output_dir: Path | None = None):
        self.ROOT_PATH = Path(__file__).resolve().parent.parent
        self.OUTPUT_DIR = output_dir or (self.ROOT_PATH / "_david")
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CATECHOL_SMARTS = Chem.MolFromSmarts("c1c([OH])c([OH])ccc1")
        self.SEED_MOL = Chem.MolFromSmiles(MY_MOLECULE)
        self.SEED_FP = AllChem.GetMorganFingerprintAsBitVect(self.SEED_MOL, 2, nBits=2048) if self.SEED_MOL else None

    def is_catechol_polyphenol(self, mol: Chem.Mol):
        if mol is None:
            return False
        if self.CATECHOL_SMARTS and mol.HasSubstructMatch(self.CATECHOL_SMARTS):
            return True
        oh_matches = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OH]")))
        if oh_matches >= 4:
            return True
        return False

    def extract_murcko_scaffold(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            if scaff.GetNumHeavyAtoms() == 0:
                return None
            return Chem.MolToSmiles(scaff)
        except Exception:
            return None

    def cluster_scaffolds(self, scaffolds: list[str], cutoff: float = CLUSTER_CUTOFF):
        mols = [Chem.MolFromSmiles(s) for s in scaffolds]
        valid_mols = [m for m in mols if m is not None]
        if not valid_mols:
            return []

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in valid_mols]
        num_fps = len(fps)
        dists = []
        for i in range(1, num_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - s for s in sims])

        clusters = Butina.ClusterData(dists, num_fps, cutoff, isDistData=True)
        return clusters

    def process_candidates_dataframe(self, df: pd.DataFrame, max_scaffolds: int = MAX_OUTPUT_SCAFFOLDS):
        if "SMILES" not in df.columns:
            raise ValueError("Input DataFrame must contain 'SMILES' column.")

        # Identify ROCS and RA columns flexibly
        rocs_col = next((c for c in df.columns if "RDKit" in c or "ROCS" in c), None)
        ra_col = next((c for c in df.columns if "Retrosynthetic" in c or "RA" in c), None)
        qed_col = next((c for c in df.columns if "QED" in c), None)

        filtered = df.copy()
        if "Valid" in filtered.columns:
            filtered = filtered[filtered["Valid"] == 1]
        if rocs_col and rocs_col in filtered.columns:
            filtered = filtered[filtered[rocs_col] >= MIN_ROCS_SCORE]
        if ra_col and ra_col in filtered.columns:
            filtered = filtered[filtered[ra_col] >= MIN_RA_SCORE]
        if qed_col and qed_col in filtered.columns:
            filtered = filtered[filtered[qed_col] >= MIN_QED_SCORE]

        if len(filtered) == 0:
            filtered = df.head(50)

        records = []
        for _, row in filtered.iterrows():
            smi = str(row["SMILES"])
            scaff = self.extract_murcko_scaffold(smi)
            if not scaff:
                continue

            scaff_mol = Chem.MolFromSmiles(scaff)
            if not scaff_mol:
                continue

            is_poly = self.is_catechol_polyphenol(scaff_mol)

            # Similarity to seed
            if self.SEED_FP:
                fp = AllChem.GetMorganFingerprintAsBitVect(scaff_mol, 2, nBits=2048)
                sim_to_seed = DataStructs.TanimotoSimilarity(fp, self.SEED_FP)
            else:
                sim_to_seed = 0.0

            rocs_val = float(row[rocs_col]) if rocs_col and rocs_col in row else 0.5
            ra_val = float(row[ra_col]) if ra_col and ra_col in row else 0.5
            qed_val = float(row[qed_col]) if qed_col and qed_col in row else 0.5

            records.append({
                "SMILES": smi,
                "Scaffold": scaff,
                "ROCS": rocs_val,
                "RAscore": ra_val,
                "QED": qed_val,
                "Is_Polyphenol": is_poly,
                "Seed_Similarity": sim_to_seed,
            })

        if not records:
            return pd.DataFrame()

        scaff_df = pd.DataFrame(records)

        # Prioritize novel non-polyphenols
        non_poly = scaff_df[~scaff_df["Is_Polyphenol"] & (scaff_df["Seed_Similarity"] <= MAX_SEED_SIMILARITY)]
        pool = non_poly if len(non_poly) >= max_scaffolds else scaff_df

        pool_unique = pool.sort_values(by=["ROCS", "RAscore"], ascending=False).drop_duplicates(subset=["Scaffold"])
        unique_scaffolds = pool_unique["Scaffold"].tolist()

        clusters = self.cluster_scaffolds(unique_scaffolds)
        selected_indices = [cluster[0] for cluster in clusters[:max_scaffolds]]

        final_df = pool_unique.iloc[selected_indices].copy().reset_index(drop=True)
        final_df["Scaffold_ID"] = [f"Funnel_Scaffold_{i+1}" for i in range(len(final_df))]

        tsv_out = self.OUTPUT_DIR / "funnel_scaffolds.tsv"
        final_df.to_csv(tsv_out, sep="\t", index=False)

        # Generate depiction grid
        mols_to_draw = [Chem.MolFromSmiles(s) for s in final_df["Scaffold"]]
        legends = [
            f"{row['Scaffold_ID']}\nROCS: {row['ROCS']:.2f} | RA: {row['RAscore']:.2f}"
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
            "RAscore": [0.85, 0.78, 0.60, 0.35, 0.88, 0.82],
            "QED": [0.70, 0.65, 0.40, 0.50, 0.75, 0.68],
        })
        res = funnel.process_candidates_dataframe(dummy_data)
        print("Self-test succeeded! Extracted scaffolds:")
        print(res[["Scaffold_ID", "Scaffold", "ROCS", "RAscore", "Is_Polyphenol", "Seed_Similarity"]])
