from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from _david.code.schema import std
from _david.code.visualization import (
    AlignedDepicter,
    render_lead_structures_2d,
    render_pareto_frontier,
    render_property_distributions,
)


class ParetoFrontier:
    @staticmethod
    def identify_pareto_front(
        scores: np.ndarray, maximize: Sequence[bool]
    ) -> np.ndarray:
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
        df_ranked[std.PARETO_RANK] = -1
        valid_mask = (
            df_ranked[std.VALID].astype(bool)
            if std.VALID in df_ranked.columns
            else np.ones(len(df_ranked), dtype=bool)
        )

        available_indices = df_ranked[valid_mask].index.to_numpy()
        current_rank = 0

        while len(available_indices) > 0:
            sub_scores = df_ranked.loc[available_indices, list(objectives)].to_numpy(
                dtype=np.float64
            )
            is_front = self.identify_pareto_front(sub_scores, maximize)
            front_indices = available_indices[is_front]
            df_ranked.loc[front_indices, std.PARETO_RANK] = current_rank
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
        render_pareto: bool = False,
    ) -> pd.DataFrame:
        tsv_path = Path(candidates_tsv)
        if not tsv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(tsv_path, sep="\t")
        if df.empty or std.SMILES not in df.columns:
            return df

        if std.VALID not in df.columns:
            df[std.VALID] = [Chem.MolFromSmiles(s) is not None for s in df[std.SMILES]]
        valid_df = df[df[std.VALID].astype(bool)].copy()

        if std.ROCS in valid_df.columns and std.SA in valid_df.columns:
            max_sa = valid_df[std.SA].max()
            maximize_sa = bool(max_sa <= 1.0)
            pareto = ParetoFrontier()
            valid_df = pareto.rank_pareto_fronts(
                valid_df,
                objectives=[std.ROCS, std.SA],
                maximize=[True, maximize_sa],
            )

            if std.QED not in valid_df.columns:
                valid_df[std.QED] = [
                    float(QED.qed(m))
                    if (m := Chem.MolFromSmiles(s)) is not None
                    else 0.0
                    for s in valid_df[std.SMILES]
                ]

            if std.MW not in valid_df.columns:
                valid_df[std.MW] = [
                    float(Descriptors.MolWt(m))
                    if (m := Chem.MolFromSmiles(s)) is not None
                    else 0.0
                    for s in valid_df[std.SMILES]
                ]

            valid_df = valid_df.sort_values(
                by=[std.PARETO_RANK, std.ROCS, std.SA, std.QED],
                ascending=[True, False, not maximize_sa, False],
            )

            if render_pareto:
                render_pareto_frontier(
                    valid_df,
                    output_png=self.analysis_dir / "pareto_frontier.png",
                    rocs_col=std.ROCS,
                    sa_col=std.SA,
                )

        # Deduplicate candidates by canonical SMILES to extract distinct chemical leads
        valid_df[std.CANONICAL_SMILES] = [
            Chem.CanonSmiles(s) if (s and Chem.MolFromSmiles(s) is not None) else s
            for s in valid_df[std.SMILES]
        ]
        unique_leads = valid_df.drop_duplicates(subset=[std.CANONICAL_SMILES])
        top_leads = unique_leads.head(top_k).reset_index(drop=True)
        top_leads.to_csv(
            self.analysis_dir / "top_prioritized_leads.tsv", sep="\t", index=False
        )

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
                shutil.copyfile(
                    lead_png, self.analysis_dir / "aligned_lead_depictions.png"
                )

        summary = {
            "total_generated": len(df),
            "valid_molecules": int(valid_df[std.VALID].sum()),
            "top_k_extracted": len(top_leads),
        }
        with open(
            self.analysis_dir / "result_analysis_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=4)

        return top_leads
