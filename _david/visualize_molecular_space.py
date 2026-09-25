"""Molecular Chemical Space Visualization Pipeline using QSPRPred, Scaffviz, and DrugEx.

This module automates the comparative 2D projection and depiction of chemical space:
1. Samples 1,000 random background molecules across the chemical universe (colored in gray).
2. Harvests 1,000 candidate ligands interacting with targets binding the query compound
   using `receptor_similar.MoleculeBioactivityPipeline.run` (ranked by property profile
   similarity where the highest score has the most vibrant blue color, fading towards white).
3. Highlights the query molecule (EGCG) with a prominent, distinct star marker.
4. Generates two comparative versions:
   - Version 1: Papyrus-only (curated chemogenomics benchmark, pAffinity >= 6.0, 11 targets).
   - Version 2: Papyrus + ChEMBL (broad polypharmacological bioactivity space, pAffinity >= 5.0, 16 targets, 3,900 candidate pool).
5. Exports both high-resolution publication PNGs and interactive standalone HTML explorers.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from rdkit import Chem

from papyrus_scripts.reader import read_papyrus
from qsprpred.data import MoleculeTable
from qsprpred.data.descriptors.fingerprints import MorganFP
from scaffviz.clustering.manifold import TSNE
from scaffviz.data.manifold_table import ManifoldTable
from scaffviz.depiction.plot import Plot

# Ensure DrugEx and local _david module are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "drugex").exists() or (SCRIPT_DIR / "_david").exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from _david import receptor_similar
except ImportError:
    import receptor_similar


@dataclass(frozen=True)
class MolecularSpaceConfig:
    """Configuration settings for molecular chemical space projection and visualization.

    Parameters
    ----------
    query_smiles : str
        SMILES string of the reference query molecule (e.g. Epigallocatechin gallate / EGCG).
    n_background : int, default=1000
        Number of random background molecules to sample from the chemical space.
    n_candidates : int, default=1000
        Number of bioactivity candidate molecules to extract from the pipeline.
    fp_radius : int, default=3
        Morgan fingerprint circular radius (radius=3 corresponds to ECFP6).
    fp_nbits : int, default=2048
        Bit length for the folded Morgan fingerprint bitvector.
    tsne_perplexity : float, default=35.0
        Perplexity parameter balancing local vs global manifold preservation in t-SNE.
    tsne_n_iter : int, default=1000
        Maximum iterations for t-SNE optimization.
    random_state : int, default=42
        Seed for pseudo-random number generator for reproducible sampling and projection.
    n_jobs : int, default=8
        Number of parallel CPU worker threads for fingerprinting and manifold calculation.
    """

    query_smiles: str = (
        "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
    )
    n_background: int = 1000
    n_candidates: int = 1000
    fp_radius: int = 3
    fp_nbits: int = 2048
    tsne_perplexity: float = 35.0
    tsne_n_iter: int = 1000
    random_state: int = 42
    n_jobs: int = 8


class ChemicalSpaceDataset:
    """Manages collection, curation, and assembly of chemical subsets into a unified dataset.

    Parameters
    ----------
    papyrus_path : str | Path
        Path to local Papyrus chemogenomics repository directory.
    config : MolecularSpaceConfig
        Pipeline configuration settings.
    """

    def __init__(self, papyrus_path: str | Path, config: MolecularSpaceConfig) -> None:
        self.papyrus_path: str = str(papyrus_path)
        self.config: MolecularSpaceConfig = config

    def sample_background_molecules(
        self,
        n_samples: int = 1000,
        exclude_smiles: Optional[Set[str]] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Sample random background molecules across the Papyrus chemical database.

        Parameters
        ----------
        n_samples : int, default=1000
            Number of background molecules to retain.
        exclude_smiles : set of str, optional
            Set of candidate and query SMILES strings to exclude to avoid overlap.
        random_seed : int, default=42
            Random seed for sampling.

        Returns
        -------
        pd.DataFrame
            DataFrame containing standardized background molecules with columns
            ``['SMILES', 'Group', 'property_similarity', 'Source', 'Target']``.
        """
        exclude: Set[str] = exclude_smiles or set()

        raw_papyrus = read_papyrus(
            version="latest",
            source_path=self.papyrus_path,
            plusplus=True,
            chunksize=1
        )
        lf = raw_papyrus.lazy() if isinstance(raw_papyrus, pl.DataFrame) else raw_papyrus

        collected = (
            lf.select(["SMILES"])
            .filter(pl.col("SMILES").is_not_null())
            .limit(n_samples * 25)
            .collect()
            .to_pandas()
        )

        valid_smiles: List[str] = []
        for smi in collected["SMILES"].dropna().unique():
            if smi not in exclude and len(smi) > 4:
                valid_smiles.append(smi)
            if len(valid_smiles) >= n_samples * 3:
                break

        rng = np.random.default_rng(random_seed)
        selected_smiles = rng.choice(valid_smiles, size=min(n_samples, len(valid_smiles)), replace=False)

        bg_df = pd.DataFrame({
            "SMILES": selected_smiles,
            "Group": "Background Space (1,000 Random Molecules)",
            "property_similarity": np.nan,
            "Source": "Papyrus (Random Chemical Universe)",
            "Target": "None (Background)"
        })
        return bg_df

    def format_candidate_molecules(
        self,
        candidate_df: pd.DataFrame,
        group_label: str
    ) -> pd.DataFrame:
        """Standardize candidate ligands extracted from the bioactivity pipeline.

        Parameters
        ----------
        candidate_df : pd.DataFrame
            DataFrame returned by `MoleculeBioactivityPipeline.run`.
        group_label : str
            Descriptive cohort label for visualization legend and hovering.

        Returns
        -------
        pd.DataFrame
            Curated candidate DataFrame with uniform schema.
        """
        df = candidate_df.copy()
        smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
        df["SMILES"] = df[smiles_col]
        df["Group"] = group_label

        if "property_similarity" not in df.columns:
            df["property_similarity"] = 0.50

        if "source" in df.columns:
            df["Source"] = df["source"]
        else:
            df["Source"] = "Papyrus Bioactivity"

        if "target_accession" in df.columns:
            df["Target"] = df["target_accession"]
        else:
            df["Target"] = "Shared Receptors"

        keep_cols = ["SMILES", "Group", "property_similarity", "Source", "Target"]
        for extra in ["pAffinity", "MW", "logP", "TPSA", "QED"]:
            if extra in df.columns:
                keep_cols.append(extra)

        return df[keep_cols].reset_index(drop=True)

    def format_query_molecule(self, query_smiles: str) -> pd.DataFrame:
        """Format the reference query molecule into a standalone single-row DataFrame.

        Parameters
        ----------
        query_smiles : str
            Query SMILES string.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame representing the reference query point.
        """
        return pd.DataFrame([{
            "SMILES": query_smiles,
            "Group": "Query Molecule (my_molecule / EGCG)",
            "property_similarity": 1.0,
            "Source": "Input Reference",
            "Target": "Query Reference Ligand",
            "pAffinity": np.nan
        }])

    def assemble_unified_table(
        self,
        background_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        query_df: pd.DataFrame,
        table_name: str,
        store_dir: Path
    ) -> Tuple[MoleculeTable, pd.DataFrame]:
        """Combine background, candidate, and query cohorts into a QSPRPred MoleculeTable.

        Parameters
        ----------
        background_df : pd.DataFrame
            1,000 random background molecules.
        candidates_df : pd.DataFrame
            1,000 bioactivity-pipeline candidate molecules.
        query_df : pd.DataFrame
            Single-row reference query molecule.
        table_name : str
            Unique identifier for the MoleculeTable cache.
        store_dir : Path
            Directory for backing store files.

        Returns
        -------
        Tuple[MoleculeTable, pd.DataFrame]
            Instantiated MoleculeTable with calculated descriptors and the combined DataFrame.
        """
        combined = pd.concat([background_df, candidates_df, query_df], ignore_index=True)
        # Deduplicate SMILES keeping candidates/query precedence
        combined = combined.drop_duplicates(subset=["SMILES"], keep="last").reset_index(drop=True)

        store_dir.mkdir(parents=True, exist_ok=True)
        table = MoleculeTable(
            name=table_name,
            df=combined,
            smiles_col="SMILES",
            store_dir=str(store_dir),
            overwrite=True,
            n_jobs=self.config.n_jobs
        )

        # Compute Morgan fingerprints (ECFP6 equivalent)
        table.addDescriptors([
            MorganFP(radius=self.config.fp_radius, nBits=self.config.fp_nbits)
        ])

        return table, combined


class ManifoldEmbeddingEngine:
    """Executes high-dimensional circular fingerprint dimensionality reduction via t-SNE."""

    @staticmethod
    def compute_tsne_coordinates(
        table: MoleculeTable,
        config: MolecularSpaceConfig
    ) -> Tuple[str, str, pd.DataFrame]:
        """Project high-dimensional descriptors to 2D using Scaffviz TSNE manifold.

        Parameters
        ----------
        table : MoleculeTable
            MoleculeTable containing computed Morgan fingerprint descriptors.
        config : MolecularSpaceConfig
            Projection hyperparameter configuration.

        Returns
        -------
        Tuple[str, str, pd.DataFrame]
            Tuple of (x_col_name, y_col_name, enriched_dataframe).
        """
        manifold_table = ManifoldTable.fromMolTable(table)
        tsne_clusterer = TSNE(
            perplexity=config.tsne_perplexity,
            n_iter=config.tsne_n_iter,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
        x_col, y_col = manifold_table.addManifoldData(tsne_clusterer, recalculate=True)
        enriched_df = manifold_table.getDF()
        return x_col, y_col, enriched_df


class MolecularSpaceRenderer:
    """Renders pixel-perfect publication figures (Matplotlib) and interactive HTML dashboards (Plotly)."""

    # High-contrast sequential colormap: soft pale ice blue/white to deep vibrant royal blue
    VIBRANT_BLUE_CMAP: ClassVar[List[Tuple[float, str]]] = [
        (0.00, "#F0F8FF"),  # Softest ice blue (near white)
        (0.20, "#C6DBEF"),  # Pale powder blue
        (0.40, "#6BAED6"),  # Vivid cerulean
        (0.65, "#2171B5"),  # Rich royal blue
        (0.85, "#08519C"),  # Deep navy sapphire
        (1.00, "#001A66"),  # Most vibrant deep midnight blue
    ]

    @classmethod
    def render_matplotlib(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_png: Path,
        title: str,
        subtitle: str
    ) -> Path:
        """Render publication-quality high-resolution static scatter plot.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame with t-SNE coordinates, Group, and property_similarity.
        x_col : str
            X-axis coordinate column name (e.g. 'TSNE_1').
        y_col : str
            Y-axis coordinate column name (e.g. 'TSNE_2').
        output_png : Path
            Destination file path for the exported PNG.
        title : str
            Primary chart title.
        subtitle : str
            Descriptive subtitle detailing cohorts and parameters.

        Returns
        -------
        Path
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
        ax.set_facecolor("#FCFDFF")

        # Split cohorts
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        # Sort candidates ascending so highest-scoring vibrant blue points are plotted ON TOP
        df_cand = df[is_candidate].sort_values(by="property_similarity", ascending=True)
        df_query = df[is_query]

        # 1. Background trace (1,000 random molecules in neutral gray)
        ax.scatter(
            df_bg[x_col],
            df_bg[y_col],
            c="#B8C2CC",
            s=24,
            alpha=0.45,
            edgecolors="none",
            label=f"Background Chemical Space ({len(df_bg):,} Random Molecules)",
            zorder=1
        )

        # 2. Candidate trace (1,000 molecules with vibrant blue to white colormap)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "white_to_vibrant_blue",
            [c for _, c in cls.VIBRANT_BLUE_CMAP]
        )
        scores = df_cand["property_similarity"].fillna(0.0)

        scatter_cand = ax.scatter(
            df_cand[x_col],
            df_cand[y_col],
            c=scores,
            cmap=cmap,
            vmin=float(scores.min()),
            vmax=1.0,
            s=48,
            alpha=0.90,
            edgecolors="#001A66",
            linewidths=0.45,
            label=f"Bioactivity Pipeline Candidates ({len(df_cand):,} Molecules)",
            zorder=3
        )

        # 3. Query molecule trace (highlighted with prominent ruby star)
        if not df_query.empty:
            qx, qy = df_query[x_col].iloc[0], df_query[y_col].iloc[0]
            ax.scatter(
                [qx],
                [qy],
                c="#E63946",
                s=380,
                marker="*",
                edgecolors="#1A1A1A",
                linewidths=2.0,
                label="Query Molecule: EGCG (my_molecule)",
                zorder=10
            )
            ax.annotate(
                "★ EGCG (my_molecule)\n  [Property Sim: 1.000]",
                xy=(qx, qy),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=10.5,
                fontweight="bold",
                color="#8B0000",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF2F2", ec="#E63946", lw=1.3, alpha=0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#8B0000", lw=1.3),
                zorder=11
            )

        # Non-overlapping hierarchical title block
        full_title = f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=13, fontweight="bold", pad=16, color="#1A202C", loc="left", linespacing=1.4)
        ax.set_xlabel(f"{x_col} (Morgan Circular Fingerprint ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)
        ax.set_ylabel(f"{y_col} (Morgan Circular Fingerprint ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)

        # Colorbar
        cbar = fig.colorbar(scatter_cand, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Property Profile Similarity Score\n(White: Lowest $\\rightarrow$ Vibrant Blue: Highest = 1.0)", fontsize=10, fontweight="bold", color="#1A202C", labelpad=10)
        cbar.ax.tick_params(labelsize=9)

        # Styling
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="#CBD5E0")
        ax.legend(loc="upper right", framealpha=0.94, edgecolor="#CBD5E0", fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E0")

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return output_png

    @classmethod
    def render_plotly_interactive(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_html: Path,
        title: str
    ) -> Path:
        """Render standalone interactive 2D HTML explorer with hover cards and zoom.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame with t-SNE coordinates.
        x_col : str
            X-axis column.
        y_col : str
            Y-axis column.
        output_html : Path
            Destination file path for the exported HTML.
        title : str
            Chart title.

        Returns
        -------
        Path
            Path to the saved HTML file.
        """
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        df_cand = df[is_candidate].sort_values(by="property_similarity", ascending=True)
        df_query = df[is_query]

        fig = go.Figure()

        # 1. Background trace (1,000 random molecules in gray)
        bg_hover = [
            f"<b>Background Molecule</b><br>SMILES: <code>{s}</code>"
            for s in df_bg["SMILES"]
        ]
        fig.add_trace(go.Scatter(
            x=df_bg[x_col],
            y=df_bg[y_col],
            mode="markers",
            name="Background Space (1,000 Random Molecules)",
            marker=dict(size=5.5, color="#A0AEC0", opacity=0.45),
            hoverinfo="text",
            hovertext=bg_hover
        ))

        # 2. Candidate trace (1,000 molecules with vibrant blue to white colormap)
        scores = df_cand["property_similarity"].fillna(0.0)
        cand_hover = []
        for _, row in df_cand.iterrows():
            sim = row.get("property_similarity", np.nan)
            src = row.get("Source", "Papyrus")
            tgt = row.get("Target", "Unknown")
            paff = row.get("pAffinity", np.nan)
            paff_str = f"{paff:.2f}" if pd.notnull(paff) else "N/A"
            smi = row.get("SMILES", "")
            cand_hover.append(
                f"<b>Candidate Ligand</b><br>"
                f"Similarity Score: <b>{sim:.4f}</b><br>"
                f"Target Receptors: {tgt}<br>"
                f"pAffinity: {paff_str}<br>"
                f"Source: {src}<br>"
                f"SMILES: <code>{smi[:45]}...</code>"
            )

        plotly_scale = [[pos, col] for pos, col in cls.VIBRANT_BLUE_CMAP]
        fig.add_trace(go.Scatter(
            x=df_cand[x_col],
            y=df_cand[y_col],
            mode="markers",
            name="Bioactivity Candidates (1,000 Molecules)",
            marker=dict(
                size=8.5,
                color=scores,
                colorscale=plotly_scale,
                cmin=float(scores.min()),
                cmax=1.0,
                colorbar=dict(
                    title=dict(text="<b>Similarity Score</b><br>White $\\rightarrow$ Vibrant Blue", font=dict(size=12)),
                    thickness=18,
                    len=0.75,
                    x=1.02
                ),
                line=dict(color="#001A66", width=0.6),
                showscale=True
            ),
            hoverinfo="text",
            hovertext=cand_hover
        ))

        # 3. Query molecule trace (distinct ruby star marker)
        if not df_query.empty:
            q_row = df_query.iloc[0]
            qx, qy = q_row[x_col], q_row[y_col]
            q_smi = q_row["SMILES"]
            fig.add_trace(go.Scatter(
                x=[qx],
                y=[qy],
                mode="markers+text",
                name="Query: EGCG (my_molecule)",
                marker=dict(
                    symbol="star",
                    size=22,
                    color="#E63946",
                    line=dict(color="#1A1A1A", width=2)
                ),
                text=["★ Query: EGCG"],
                textposition="top center",
                textfont=dict(size=13, color="#8B0000", family="Arial Black"),
                hoverinfo="text",
                hovertext=[f"<b>Reference Query: EGCG</b><br>SMILES: <code>{q_smi}</code><br>Score: <b>1.0000</b>"]
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=17, color="#1A202C")),
            xaxis=dict(title=f"{x_col} (Morgan Fingerprint Manifold)", gridcolor="#EDF2F7", zeroline=False),
            yaxis=dict(title=f"{y_col} (Morgan Fingerprint Manifold)", gridcolor="#EDF2F7", zeroline=False),
            plot_bgcolor="#FCFDFF",
            paper_bgcolor="#FFFFFF",
            width=1200,
            height=850,
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#CBD5E0",
                borderwidth=1
            )
        )

        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")
        return output_html


class MolecularSpacePipeline:
    """Orchestrates end-to-end execution of Version 1 (Papyrus) and Version 2 (Papyrus + ChEMBL).

    Parameters
    ----------
    config : MolecularSpaceConfig, optional
        Custom configuration options.
    output_dir : Path, optional
        Destination directory for generated figures and data tables.
    """

    def __init__(
        self,
        config: Optional[MolecularSpaceConfig] = None,
        output_dir: Optional[Path] = None
    ) -> None:
        self.config: MolecularSpaceConfig = config or MolecularSpaceConfig()
        default_dir = PROJECT_ROOT / "_david" / "outputs" / "molecular_space_visualization"
        self.output_dir: Path = output_dir or default_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.papyrus_path: str = (
            receptor_similar.MoleculeBioactivityPipeline._resolve_papyrus_path()
            or str(PROJECT_ROOT / "tutorial" / "data" / "data" / ".Papyrus")
        )
        self.dataset_manager = ChemicalSpaceDataset(self.papyrus_path, self.config)

    def run_version_1_papyrus(self) -> Dict[str, Any]:
        """Execute Version 1: Candidate extraction strictly from Papyrus (high stringency pIC50 >= 6.0).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing result metrics and exported file paths.
        """
        print("\n" + "=" * 80)
        print(">>> EXECUTING VERSION 1: CHEMICAL SPACE VISUALIZATION (PAPYRUS-ONLY) <<<".center(80))
        print("=" * 80)

        t0 = time.time()
        pipeline_v1 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=False)
        result_v1 = pipeline_v1.run(
            self.config.query_smiles,
            min_paffinity=6.0,
            max_molecules=self.config.n_candidates,
            quality_filter="High",
            allow_chembl_fallback=False,
            show_progress=True
        )
        candidates_raw = result_v1.molecules

        # Format cohorts
        candidate_df = self.dataset_manager.format_candidate_molecules(
            candidates_raw,
            group_label="Papyrus Bioactivity Candidates (pIC50 >= 6.0)"
        )
        query_df = self.dataset_manager.format_query_molecule(self.config.query_smiles)
        exclude_smiles = set(candidate_df["SMILES"]).union(set(query_df["SMILES"]))

        background_df = self.dataset_manager.sample_background_molecules(
            n_samples=self.config.n_background,
            exclude_smiles=exclude_smiles,
            random_seed=42
        )

        print(f"\n[V1: Papyrus] Assembling unified dataset: {len(background_df)} background + {len(candidate_df)} candidates + 1 query...")
        table_dir = self.output_dir / "cache" / "v1_papyrus"
        table, combined_df = self.dataset_manager.assemble_unified_table(
            background_df,
            candidate_df,
            query_df,
            table_name="molspace_v1_papyrus",
            store_dir=table_dir
        )

        print(f"[V1: Papyrus] Computing t-SNE 2D projection via Scaffviz TSNE (perplexity={self.config.tsne_perplexity})...")
        x_col, y_col, enriched_df = ManifoldEmbeddingEngine.compute_tsne_coordinates(table, self.config)

        out_png = self.output_dir / "molecular_space_v1_papyrus.png"
        out_html = self.output_dir / "molecular_space_v1_papyrus.html"

        title = "Molecular Chemical Space: Version 1 (Papyrus-Only Benchmark)"
        subtitle = (
            f"1,000 Gray Background Molecules vs. 1,000 Papyrus Candidates "
            f"({len(result_v1.targets)} Receptors, pAffinity >= 6.0 | Score: {candidate_df['property_similarity'].min():.3f} -> {candidate_df['property_similarity'].max():.3f})"
        )

        MolecularSpaceRenderer.render_matplotlib(enriched_df, x_col, y_col, out_png, title, subtitle)
        MolecularSpaceRenderer.render_plotly_interactive(enriched_df, x_col, y_col, out_html, title)

        elapsed = time.time() - t0
        print(f"[V1: Papyrus] Completed in {elapsed:.2f} s!")
        print(f"  -> Static High-Res PNG: {out_png}")
        print(f"  -> Interactive Dashboard: {out_html}")

        return {
            "version": "v1_papyrus",
            "n_targets": len(result_v1.targets),
            "targets": [t["uniprot_acc"] for t in result_v1.targets],
            "n_candidates": len(candidate_df),
            "n_background": len(background_df),
            "score_min": float(candidate_df["property_similarity"].min()),
            "score_max": float(candidate_df["property_similarity"].max()),
            "png_path": str(out_png),
            "html_path": str(out_html)
        }

    def run_version_2_papyrus_chembl(self) -> Dict[str, Any]:
        """Execute Version 2: Candidate extraction across Papyrus + ChEMBL broad bioactivity space (pIC50 >= 5.0).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing result metrics and exported file paths.
        """
        print("\n" + "=" * 80)
        print(">>> EXECUTING VERSION 2: CHEMICAL SPACE VISUALIZATION (PAPYRUS + ChEMBL) <<<".center(80))
        print("=" * 80)

        t0 = time.time()
        pipeline_v2 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=True)

        # Version 2 queries broad bioactivity space (pAffinity >= 5.0, capturing all 16 polypharmacological
        # ChEMBL receptors including TERT, APP, SARS-CoV-2 replicase, BACE1, and 3,900 candidate pool)
        result_v2 = pipeline_v2.run(
            self.config.query_smiles,
            min_paffinity=5.0,  # Standard 10 uM screening cutoff in ChEMBL assays
            max_molecules=self.config.n_candidates,
            quality_filter=None,  # Comprehensive multi-tier quality spanning ChEMBL & Papyrus
            allow_chembl_fallback=True,
            show_progress=True
        )
        candidates_raw = result_v2.molecules

        candidate_df = self.dataset_manager.format_candidate_molecules(
            candidates_raw,
            group_label="Papyrus + ChEMBL Candidates (Broad Space, pIC50 >= 5.0)"
        )
        query_df = self.dataset_manager.format_query_molecule(self.config.query_smiles)
        exclude_smiles = set(candidate_df["SMILES"]).union(set(query_df["SMILES"]))

        background_df = self.dataset_manager.sample_background_molecules(
            n_samples=self.config.n_background,
            exclude_smiles=exclude_smiles,
            random_seed=101  # Distinct seed for independent background space verification
        )

        print(f"\n[V2: Papyrus + ChEMBL] Assembling unified dataset: {len(background_df)} background + {len(candidate_df)} candidates + 1 query...")
        table_dir = self.output_dir / "cache" / "v2_papyrus_chembl"
        table, combined_df = self.dataset_manager.assemble_unified_table(
            background_df,
            candidate_df,
            query_df,
            table_name="molspace_v2_papyrus_chembl",
            store_dir=table_dir
        )

        print(f"[V2: Papyrus + ChEMBL] Computing t-SNE 2D projection via Scaffviz TSNE (perplexity={self.config.tsne_perplexity})...")
        x_col, y_col, enriched_df = ManifoldEmbeddingEngine.compute_tsne_coordinates(table, self.config)

        out_png = self.output_dir / "molecular_space_v2_papyrus_chembl.png"
        out_html = self.output_dir / "molecular_space_v2_papyrus_chembl.html"

        title = "Molecular Chemical Space: Version 2 (Papyrus + ChEMBL Broad Space)"
        subtitle = (
            f"1,000 Gray Background Molecules vs. 1,000 Candidates Ranked from 3,900 Pool "
            f"({len(result_v2.targets)} Receptors, pAffinity >= 5.0 | Score: {candidate_df['property_similarity'].min():.3f} -> {candidate_df['property_similarity'].max():.3f})"
        )

        MolecularSpaceRenderer.render_matplotlib(enriched_df, x_col, y_col, out_png, title, subtitle)
        MolecularSpaceRenderer.render_plotly_interactive(enriched_df, x_col, y_col, out_html, title)

        elapsed = time.time() - t0
        print(f"[V2: Papyrus + ChEMBL] Completed in {elapsed:.2f} s!")
        print(f"  -> Static High-Res PNG: {out_png}")
        print(f"  -> Interactive Dashboard: {out_html}")

        return {
            "version": "v2_papyrus_chembl",
            "n_targets": len(result_v2.targets),
            "targets": [t["uniprot_acc"] for t in result_v2.targets],
            "n_candidates": len(candidate_df),
            "n_background": len(background_df),
            "score_min": float(candidate_df["property_similarity"].min()),
            "score_max": float(candidate_df["property_similarity"].max()),
            "png_path": str(out_png),
            "html_path": str(out_html)
        }

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """Execute both Version 1 and Version 2 consecutively.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Consolidated outputs from both pipeline runs.
        """
        res_v1 = self.run_version_1_papyrus()
        res_v2 = self.run_version_2_papyrus_chembl()
        return {"v1_papyrus": res_v1, "v2_papyrus_chembl": res_v2}


if __name__ == "__main__":
    runner = MolecularSpacePipeline()
    summary = runner.run_all()
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS SUCCESSFULLY GENERATED!".center(80))
    print("=" * 80)
