from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.visualization import (
    AlignedDepicter,
    render_lead_structures_2d,
    render_pareto_frontier,
    render_property_distributions,
)


class ParetoFrontier:

    @staticmethod
    def identify_pareto_front(scores: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
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
        tsv_path = Path(candidates_tsv)
        if not tsv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(tsv_path, sep="\t")
        if df.empty or "SMILES" not in df.columns:
            return df

        if "Valid" not in df.columns:
            df["Valid"] = [Chem.MolFromSmiles(s) is not None for s in df["SMILES"]]
        valid_df = df[df["Valid"].astype(bool)].copy()

        rocs_col = [c for c in valid_df.columns if any(k in c.upper() for k in ("ROCS", "SIMILARITY", "TANIMOTO"))]
        sa_col = [c for c in valid_df.columns if "SA" in c.upper() and c not in ("SMILES", "DESIRED")]

        rocs_name = rocs_col[0] if rocs_col else None
        sa_name = sa_col[0] if sa_col else None

        if rocs_name and sa_name:
            max_sa = valid_df[sa_name].max()
            maximize_sa = bool(max_sa <= 1.0)
            pareto = ParetoFrontier()
            valid_df = pareto.rank_pareto_fronts(valid_df, objectives=[rocs_name, sa_name], maximize=[True, maximize_sa])
            valid_df = valid_df.sort_values(by=["Pareto_Rank", rocs_name], ascending=[True, False])

            render_pareto_frontier(
                valid_df,
                output_png=self.analysis_dir / "pareto_frontier.png",
                rocs_col=rocs_name,
                sa_col=sa_name,
            )

        top_leads = valid_df.head(top_k)
        top_leads.to_csv(self.analysis_dir / "top_prioritized_leads.tsv", sep="\t", index=False)

        ref_mol: Chem.Mol | None = None
        if ref_sdf and Path(ref_sdf).exists():
            suppl = Chem.SDMolSupplier(str(ref_sdf), removeHs=False)
            ref_mol = next((m for m in suppl if m is not None), None)

        render_property_distributions(
            valid_df,
            output_png=self.analysis_dir / "property_distributions.png",
            ref_mol=ref_mol,
        )

        if ref_mol and not top_leads.empty:
            lead_png = self.analysis_dir / "lead_structures_2d.png"
            render_lead_structures_2d(
                top_leads,
                output_png=lead_png,
                reference=ref_mol,
            )
            if lead_png.exists():
                shutil.copyfile(lead_png, self.analysis_dir / "aligned_lead_depictions.png")

        summary = {
            "total_generated": len(df),
            "valid_molecules": int(valid_df["Valid"].sum()),
            "top_k_extracted": len(top_leads),
        }
        with open(self.analysis_dir / "result_analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        return top_leads
