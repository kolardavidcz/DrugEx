"""Result analysis, Pareto frontier extraction, and aligned 2D depiction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold


class AlignedDepicter:
    """Generates 2D molecular structure depictions aligned to a reference scaffold."""

    def __init__(self, reference: str | Chem.Mol) -> None:
        self.ref_mol = Chem.MolFromSmiles(reference) if isinstance(reference, str) else reference
        self.ref_core = MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            rdDepictor.Compute2DCoords(self.ref_mol)
            self.ref_core = self.ref_mol

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        """Aligns 2D layout of molecule matching reference core."""
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)
        if self.ref_core and mol_copy.HasSubstructMatch(self.ref_core):
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, self.ref_core)
            except Exception:
                pass
        return mol_copy

    def create_aligned_grid(
        self,
        molecules: Sequence[str | Chem.Mol],
        legends: Sequence[str] | None = None,
        output_path: Path | str | None = None,
        mols_per_row: int = 4,
        sub_img_size: tuple[int, int] = (350, 300),
    ) -> Any:
        """Renders grid image of aligned molecules."""
        mol_objs = []
        for m in molecules:
            mol_obj = Chem.MolFromSmiles(m) if isinstance(m, str) else m
            if mol_obj:
                mol_objs.append(self.align_molecule(mol_obj))

        img = Draw.MolsToGridImage(
            mol_objs,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=list(legends) if legends else None,
            useSVG=False,
        )
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))
        return img


class ParetoFrontier:
    """Identifies non-dominated Pareto frontiers across multi-objective evaluations."""

    @staticmethod
    def identify_pareto_front(scores: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
        """Identifies non-dominated points in an N x M score matrix."""
        n_points = scores.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        mult = np.array([1.0 if m else -1.0 for m in maximize], dtype=np.float64)
        scaled = scores * mult

        for i in range(n_points):
            dominated = np.any(
                np.all(scaled >= scaled[i], axis=1) & np.any(scaled > scaled[i], axis=1)
            )
            if dominated:
                is_pareto[i] = False
        return is_pareto

    def rank_pareto_fronts(
        self,
        df: pd.DataFrame,
        objectives: Sequence[str],
        maximize: Sequence[bool],
    ) -> pd.DataFrame:
        """Computes Pareto rank layers (0 = non-dominated, 1 = second front, etc.)."""
        df_ranked = df.copy()
        df_ranked["Pareto_Rank"] = -1
        valid_mask = df_ranked["Valid"].astype(bool) if "Valid" in df_ranked.columns else np.ones(len(df_ranked), dtype=bool)

        available_indices = df_ranked[valid_mask].index.to_numpy()
        current_rank = 0

        while len(available_indices) > 0:
            sub_scores = df_ranked.loc[available_indices, list(objectives)].to_numpy(dtype=np.float64)
            is_front = self.identify_pareto_front(sub_scores, maximize)
            front_indices = available_indices[is_front]
            df_ranked.loc[front_indices, "Pareto_Rank"] = current_rank
            available_indices = available_indices[~is_front]
            current_rank += 1

        return df_ranked


class ResultAnalyzer:
    """Extracts top lead candidates, Pareto frontiers, and aligned depictions."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.analysis_dir = self.output_dir / "stage2_result_analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        candidates_tsv: Path | str,
        ref_sdf: Path | str | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Analyzes generated molecules, extracts Pareto frontier, and generates visualizations."""
        tsv_path = Path(candidates_tsv)
        if not tsv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(tsv_path, sep="\t")
        if df.empty or "SMILES" not in df.columns:
            return df

        if "Valid" not in df.columns:
            df["Valid"] = [Chem.MolFromSmiles(s) is not None for s in df["SMILES"]]
        valid_df = df[df["Valid"].astype(bool)].copy()

        # Resolve primary score columns
        rocs_col = [c for c in valid_df.columns if any(k in c.upper() for k in ("ROCS", "SIMILARITY", "TANIMOTO"))]
        sa_col = [c for c in valid_df.columns if "SA" in c.upper() and c not in ("SMILES", "DESIRED")]

        rocs_name = rocs_col[0] if rocs_col else None
        sa_name = sa_col[0] if sa_col else None

        if rocs_name and sa_name:
            # Detect whether SA is modified score in [0, 1] (maximize) or raw score 1-10 (minimize)
            max_sa = valid_df[sa_name].max()
            maximize_sa = bool(max_sa <= 1.0)
            pareto = ParetoFrontier()
            valid_df = pareto.rank_pareto_fronts(valid_df, objectives=[rocs_name, sa_name], maximize=[True, maximize_sa])
            valid_df = valid_df.sort_values(by=["Pareto_Rank", rocs_name], ascending=[True, False])

            # Scatter plot of Pareto frontier
            plt.figure(figsize=(7, 5))
            front_0 = valid_df[valid_df["Pareto_Rank"] == 0]
            others = valid_df[valid_df["Pareto_Rank"] > 0]
            plt.scatter(others[sa_name], others[rocs_name], c="gray", alpha=0.4, label="Dominated")
            plt.scatter(front_0[sa_name], front_0[rocs_name], c="crimson", alpha=0.9, label="Pareto Front (Rank 0)")
            plt.xlabel(f"{sa_name} ({'higher is better' if maximize_sa else 'lower is better'})")
            plt.ylabel(f"{rocs_name} (higher is better)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "pareto_frontier.png", dpi=150)
            plt.close()

        top_leads = valid_df.head(top_k)
        top_leads.to_csv(self.analysis_dir / "top_prioritized_leads.tsv", sep="\t", index=False)

        # Aligned depictions
        if ref_sdf and Path(ref_sdf).exists() and not top_leads.empty:
            suppl = Chem.SDMolSupplier(str(ref_sdf), removeHs=False)
            ref_mol = next((m for m in suppl if m is not None), None)
            if ref_mol:
                depicter = AlignedDepicter(reference=ref_mol)
                legends = [f"Rank {i+1}" for i in range(len(top_leads))]
                depicter.create_aligned_grid(
                    top_leads["SMILES"].tolist(),
                    legends=legends,
                    output_path=self.analysis_dir / "aligned_lead_depictions.png",
                )

        summary = {
            "total_generated": len(df),
            "valid_molecules": int(valid_df["Valid"].sum()),
            "top_k_extracted": len(top_leads),
        }
        with open(self.analysis_dir / "result_analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return top_leads
