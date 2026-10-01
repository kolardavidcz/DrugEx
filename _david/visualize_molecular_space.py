import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeAlign

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
    try:
        import receptor_similar
    except ImportError:
        receptor_similar = None

# --- Hyperparameters & System Constants ---
SEED_SMILES = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
N_BACKGROUND = 1000
N_CANDIDATES = 1000
FP_RADIUS = 3
FP_NBITS = 2048
TSNE_PERPLEXITY = 35.0
TSNE_N_ITER = 1000
RANDOM_STATE = 42
N_JOBS = 8
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "_david" / "outputs" / "molecular_space_visualization"

# Custom sapphire blue colormap: highest score = vibrant blue (#001A66), lowest = white (#FFFFFF)
VIBRANT_BLUE_CMAP = [
    (0.00, "#FFFFFF"),
    (0.25, "#E1EDFE"),
    (0.50, "#85B6FF"),
    (0.75, "#1B64F2"),
    (1.00, "#001A66")
]


@dataclass(frozen=True)
class MolecularSpaceConfig:
    query_smiles: str = SEED_SMILES
    n_background: int = N_BACKGROUND
    n_candidates: int = N_CANDIDATES
    fp_radius: int = FP_RADIUS
    fp_nbits: int = FP_NBITS
    tsne_perplexity: float = TSNE_PERPLEXITY
    tsne_n_iter: int = TSNE_N_ITER
    random_state: int = RANDOM_STATE
    n_jobs: int = N_JOBS


class ConformerSimilarityEngine:
    @staticmethod
    def generate_3d_conformer(smiles: str, random_seed: int = RANDOM_STATE):
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            code = AllChem.EmbedMolecule(mol, params)
            if code == -1:
                code = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=random_seed)
            if code == -1:
                return None
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        except Exception:
            return None

    @classmethod
    def compute_3d_similarities(cls, ref_smiles: str, candidate_smiles: list, n_jobs: int = N_JOBS):
        ref_mol = cls.generate_3d_conformer(ref_smiles)
        if ref_mol is None:
            return [0.0] * len(candidate_smiles), [0.0] * len(candidate_smiles), [0.0] * len(candidate_smiles)

        def _score_single(smi: str):
            if smi == ref_smiles:
                return 1.0, 1.0, 1.0
            try:
                probe = cls.generate_3d_conformer(smi)
                if probe is None:
                    return 0.0, 0.0, 0.0
                shape_sim, color_sim = rdShapeAlign.AlignMol(ref_mol, probe)
                combo = (shape_sim + color_sim) / 2.0
                return float(shape_sim), float(color_sim), float(combo)
            except Exception:
                return 0.0, 0.0, 0.0

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            scores = list(executor.map(_score_single, candidate_smiles))

        shape_scores = [round(s[0], 4) for s in scores]
        color_scores = [round(s[1], 4) for s in scores]
        combo_scores = [round(s[2], 4) for s in scores]
        return shape_scores, color_scores, combo_scores


class ChemicalSpaceDataset:
    def __init__(self, papyrus_path: str, config: MolecularSpaceConfig):
        self.papyrus_path = papyrus_path
        self.config = config

    def sample_background_molecules(self, n_samples: int, exclude_smiles: set, random_seed: int = RANDOM_STATE):
        exclude = exclude_smiles or set()
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

        valid_smiles = []
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
            "property_similarity": 0.0,
            "sim_3d_shape": 0.0,
            "sim_3d_color": 0.0,
            "sim_3d_combo": 0.0,
            "Source": "Papyrus (Random Chemical Universe)",
            "Target": "None (Background)",
            "pAffinity": np.nan
        })
        return bg_df

    def format_candidate_molecules(self, candidates_raw, group_label: str = "Bioactivity Candidates"):
        if isinstance(candidates_raw, pd.DataFrame):
            df = candidates_raw.copy()
        elif isinstance(candidates_raw, pl.DataFrame):
            df = candidates_raw.to_pandas()
        elif isinstance(candidates_raw, list):
            df = pd.DataFrame(candidates_raw)
        else:
            raise ValueError(f"Unsupported candidate container type: {type(candidates_raw)}")

        if "SMILES" not in df.columns and "smiles" in df.columns:
            df["SMILES"] = df["smiles"]

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

        df = df[keep_cols].drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

        # Compute 3D shape and pharmacophore similarity against reference
        print(f"  -> Computing 3D shape & color alignments for {len(df)} candidate molecules...")
        shape_sim, color_sim, combo_sim = ConformerSimilarityEngine.compute_3d_similarities(
            SEED_SMILES, df["SMILES"].tolist(), n_jobs=self.config.n_jobs
        )
        df["sim_3d_shape"] = shape_sim
        df["sim_3d_color"] = color_sim
        df["sim_3d_combo"] = combo_sim

        return df

    def format_query_molecule(self, query_smiles: str):
        return pd.DataFrame([{
            "SMILES": query_smiles,
            "Group": "Query Molecule (my_molecule / EGCG)",
            "property_similarity": 1.0,
            "sim_3d_shape": 1.0,
            "sim_3d_color": 1.0,
            "sim_3d_combo": 1.0,
            "Source": "Input Reference",
            "Target": "Query Reference Ligand",
            "pAffinity": np.nan
        }])

    def assemble_unified_table(self, background_df: pd.DataFrame, candidates_df: pd.DataFrame, query_df: pd.DataFrame, table_name: str, store_dir: Path):
        combined = pd.concat([background_df, candidates_df, query_df], ignore_index=True)
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

        table.addDescriptors([
            MorganFP(radius=self.config.fp_radius, nBits=self.config.fp_nbits)
        ])

        return table, combined


class ManifoldEmbeddingEngine:
    @staticmethod
    def compute_tsne_coordinates(table: MoleculeTable, config: MolecularSpaceConfig):
        manifold_tab = ManifoldTable.fromMolTable(table)
        tsne_manifold = TSNE(
            perplexity=config.tsne_perplexity,
            n_iter=config.tsne_n_iter,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )

        manifold_tab.addManifoldData(tsne_manifold)
        df_coords = manifold_tab.getDF()

        coord_cols = [c for c in df_coords.columns if c.startswith("TSNE_") or c.startswith("tSNE_")]
        if len(coord_cols) >= 2:
            x_col, y_col = coord_cols[0], coord_cols[1]
        else:
            num_cols = df_coords.select_dtypes(include=[np.number]).columns.tolist()
            x_col, y_col = num_cols[-2], num_cols[-1]

        return x_col, y_col, df_coords


class MolecularSpaceRenderer:
    @classmethod
    def render_matplotlib(cls, df: pd.DataFrame, x_col: str, y_col: str, output_png: Path, title: str, subtitle: str):
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        df_cand = df[is_candidate].sort_values(by="property_similarity", ascending=True)
        df_query = df[is_query]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("VibrantBlue", VIBRANT_BLUE_CMAP, N=256)

        fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)
        fig.patch.set_facecolor("#FCFDFF")
        ax.set_facecolor("#FCFDFF")

        # 1. Background trace (1,000 neutral gray points)
        ax.scatter(
            df_bg[x_col],
            df_bg[y_col],
            c="#B8C2CC",
            s=18,
            alpha=0.45,
            edgecolors="none",
            label="Background Universe (1,000 Random Molecules)",
            zorder=1
        )

        # 2. Candidate trace: Color = 2D Bioactivity, Size = 3D Shape Similarity
        sim_2d = df_cand["property_similarity"].fillna(0.0).values
        sim_3d = df_cand["sim_3d_shape"].fillna(0.0).values
        # Size mapping: size scales with 3D shape similarity from 20 to 220
        marker_sizes = 20.0 + 200.0 * (sim_3d ** 1.5)

        scatter_cand = ax.scatter(
            df_cand[x_col],
            df_cand[y_col],
            c=sim_2d,
            cmap=cmap,
            vmin=float(sim_2d.min()),
            vmax=1.0,
            s=marker_sizes,
            alpha=0.88,
            edgecolors="#001A66",
            linewidths=0.55,
            label=f"Candidates ({len(df_cand):,} Mols | Size $\\propto$ 3D Shape Sim)",
            zorder=3
        )

        # 3. Query molecule trace (distinct ruby star marker)
        if not df_query.empty:
            qx, qy = df_query[x_col].iloc[0], df_query[y_col].iloc[0]
            ax.scatter(
                [qx],
                [qy],
                c="#E63946",
                s=420,
                marker="*",
                edgecolors="#1A1A1A",
                linewidths=2.2,
                label="Reference: EGCG (my_molecule)",
                zorder=10
            )
            ax.annotate(
                "★ EGCG (my_molecule)\n  [2D Sim: 1.000 | 3D Shape: 1.000]",
                xy=(qx, qy),
                xytext=(-150, 20),
                textcoords="offset points",
                fontsize=10.5,
                fontweight="bold",
                color="#8B0000",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFF2F2", ec="#E63946", lw=1.3, alpha=0.95),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#8B0000", lw=1.3),
                zorder=11
            )

        full_title = f"{title}\n{subtitle}"
        ax.set_title(full_title, fontsize=12.5, fontweight="bold", pad=16, color="#1A202C", loc="left", linespacing=1.35)
        ax.set_xlabel(f"{x_col} (Morgan ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)
        ax.set_ylabel(f"{y_col} (Morgan ECFP6 Manifold)", fontsize=11, fontweight="bold", color="#2D3748", labelpad=8)

        # Colorbar for 2D similarity
        cbar = fig.colorbar(scatter_cand, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("2D Bioactivity Profile Similarity\n(White: Lowest $\\rightarrow$ Sapphire Blue: Highest = 1.0)", fontsize=9.5, fontweight="bold", color="#1A202C", labelpad=10)
        cbar.ax.tick_params(labelsize=8.5)

        # Visual size scale legend for 3D shape similarity
        size_samples = [0.30, 0.50, 0.70, 0.90]
        size_handles = [
            mlines.Line2D([], [], color="#001A66", marker="o", linestyle="None",
                          markersize=np.sqrt(20.0 + 200.0 * (v ** 1.5)),
                          markerfacecolor="#85B6FF", markeredgecolor="#001A66", markeredgewidth=0.8,
                          label=f"3D Sim: {v:.2f}")
            for v in size_samples
        ]
        leg_sizes = ax.legend(handles=size_handles, title="3D Shape Similarity (Size)", loc="lower left", framealpha=0.92, edgecolor="#CBD5E0", fontsize=8.5, title_fontsize=9)
        ax.add_artist(leg_sizes)

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="#CBD5E0")
        ax.legend(loc="upper right", framealpha=0.92, edgecolor="#CBD5E0", fontsize=9.0)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E0")

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_png, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return output_png

    @classmethod
    def render_plotly_interactive(cls, df: pd.DataFrame, x_col: str, y_col: str, output_html: Path, title: str, subtitle: str = ""):
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        df_cand = df[is_candidate].sort_values(by="property_similarity", ascending=True)
        df_query = df[is_query]

        plotly_scale = [[pos, col] for pos, col in VIBRANT_BLUE_CMAP]

        cand_x = [float(v) for v in df_cand[x_col]]
        cand_y = [float(v) for v in df_cand[y_col]]
        cand_scores_2d = [round(float(v), 5) for v in df_cand["property_similarity"].fillna(0.0)]
        cand_scores_3d = [round(float(v), 5) for v in df_cand["sim_3d_shape"].fillna(0.0)]
        cand_colors_3d = [round(float(v), 5) for v in df_cand["sim_3d_color"].fillna(0.0)]
        cand_combos_3d = [round(float(v), 5) for v in df_cand["sim_3d_combo"].fillna(0.0)]

        # Marker size scaled by 3D shape similarity
        cand_sizes = [round(6.0 + 18.0 * (v ** 1.5), 2) for v in cand_scores_3d]

        cand_hover = []
        for _, row in df_cand.iterrows():
            sim2d = row.get("property_similarity", np.nan)
            sim3d = row.get("sim_3d_shape", np.nan)
            sim_col = row.get("sim_3d_color", np.nan)
            combo = row.get("sim_3d_combo", np.nan)
            src = row.get("Source", "Papyrus")
            tgt = row.get("Target", "Unknown")
            paff = row.get("pAffinity", np.nan)
            paff_str = f"{paff:.2f}" if pd.notnull(paff) else "N/A"
            smi = row.get("SMILES", "")
            cand_hover.append(
                f"<b>Candidate Ligand</b><br>"
                f"2D Bioactivity Similarity: <b>{sim2d:.4f}</b><br>"
                f"3D Shape Similarity: <b>{sim3d:.4f}</b> (Marker Size)<br>"
                f"3D Pharmacophore Match: <b>{sim_col:.4f}</b><br>"
                f"3D ComboScore: <b>{combo:.4f}</b><br>"
                f"Target Receptors: {tgt}<br>"
                f"pAffinity: {paff_str} | Source: {src}<br>"
                f"SMILES: <code>{smi[:42]}...</code>"
            )

        bg_hover = [f"<b>Background Molecule</b><br>SMILES: <code>{s}</code>" for s in df_bg["SMILES"]]

        fig = go.Figure()

        # 1. Background trace (1,000 neutral gray points)
        fig.add_trace(go.Scatter(
            x=[float(v) for v in df_bg[x_col]],
            y=[float(v) for v in df_bg[y_col]],
            mode="markers",
            name="Background Space (1,000 Mols)",
            marker=dict(size=5.5, color="#A0AEC0", opacity=0.45),
            hoverinfo="text",
            hovertext=bg_hover
        ))

        # 2. Candidate trace (1,000 molecules with dual-encoding: Color=2D, Size=3D)
        fig.add_trace(go.Scatter(
            x=cand_x,
            y=cand_y,
            mode="markers",
            name="Candidates (Size = 3D Shape, Color = 2D)",
            customdata=cand_scores_3d,
            marker=dict(
                size=cand_sizes,
                color=cand_scores_2d,
                colorscale=plotly_scale,
                cmin=float(min(cand_scores_2d)),
                cmax=1.0,
                colorbar=dict(
                    title=dict(text="<b>2D Bioactivity Sim</b><br>White &rarr; Sapphire Blue", font=dict(size=11.5)),
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
            qx, qy = float(q_row[x_col]), float(q_row[y_col])
            q_smi = q_row["SMILES"]
            fig.add_trace(go.Scatter(
                x=[qx],
                y=[qy],
                mode="markers+text",
                name="Query: EGCG (my_molecule)",
                marker=dict(
                    symbol="star",
                    size=26,
                    color="#E63946",
                    line=dict(color="#1A1A1A", width=2.2)
                ),
                text=["★ Query: EGCG"],
                textposition="top center",
                textfont=dict(size=13, color="#8B0000", family="Arial Black"),
                hoverinfo="text",
                hovertext=[f"<b>Reference Query: EGCG</b><br>2D Sim: <b>1.000</b> | 3D Shape: <b>1.000</b><br>SMILES: <code>{q_smi}</code>"]
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#1A202C")),
            xaxis=dict(title=f"{x_col} (Morgan Fingerprint Manifold)", gridcolor="#EDF2F7", zeroline=False),
            yaxis=dict(title=f"{y_col} (Morgan Fingerprint Manifold)", gridcolor="#EDF2F7", zeroline=False),
            plot_bgcolor="#FCFDFF",
            paper_bgcolor="#FFFFFF",
            autosize=True,
            height=780,
            margin=dict(l=60, r=40, t=50, b=60),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor="#CBD5E0",
                borderwidth=1
            )
        )

        plot_inner_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"responsive": True, "displayModeBar": True, "scrollZoom": True}
        )

        json_cand_x = json.dumps(cand_x)
        json_cand_y = json.dumps(cand_y)
        json_cand_scores_2d = json.dumps(cand_scores_2d)
        json_cand_scores_3d = json.dumps(cand_scores_3d)
        json_cand_sizes = json.dumps(cand_sizes)
        json_cand_hover = json.dumps(cand_hover)

        total_candidates = len(df_cand)
        avg_3d = float(np.mean(cand_scores_3d))
        max_3d = float(np.max(cand_scores_3d))

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{title}</title>
    <style>
        :root {{
            --primary-blue: #001A66;
            --accent-blue: #2563EB;
            --light-blue: #DBEAFE;
            --bg-gray: #F8FAFC;
            --card-border: #E2E8F0;
            --text-dark: #0F172A;
            --text-muted: #64748B;
            --emerald: #059669;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-gray);
            color: var(--text-dark);
            min-height: 100vh;
            padding: 16px 24px;
        }}
        .dashboard-container {{
            max-width: 1440px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .control-panel {{
            background: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 18px 24px;
            box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 12px;
            border-bottom: 1px solid #F1F5F9;
            padding-bottom: 12px;
        }}
        .header-title-block h1 {{
            font-size: 19px;
            font-weight: 700;
            color: var(--text-dark);
            letter-spacing: -0.02em;
        }}
        .header-title-block p {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        .header-badges {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .query-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .stats-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #EFF6FF;
            color: #1E40AF;
            border: 1px solid #BFDBFE;
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 13px;
            font-weight: 600;
        }}
        .size-indicator-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #F0FDF4;
            color: var(--emerald);
            border: 1px solid #BBF7D0;
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .sliders-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .sliders-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .slider-box {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: #F8FAFC;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #EDF2F7;
        }}
        .slider-box-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .slider-title-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .slider-title-group span.title {{
            font-size: 13.5px;
            font-weight: 700;
            color: var(--text-dark);
        }}
        .pill-3d {{
            background: var(--emerald);
            color: #FFFFFF;
            font-size: 12px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
        }}
        .pill-2d {{
            background: var(--primary-blue);
            color: #FFFFFF;
            font-size: 12px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
        }}
        .slider-instruction {{
            font-size: 11.5px;
            color: var(--text-muted);
        }}
        .slider-track-wrap {{
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .slider-endpoint {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            white-space: nowrap;
        }}
        input[type=range].slider-3d {{
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 6px;
            background: linear-gradient(90deg, #E2E8F0 0%, #A7F3D0 40%, #059669 100%);
            outline: none;
        }}
        input[type=range].slider-3d::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--emerald);
            border: 2px solid #FFFFFF;
            box-shadow: 0 2px 6px rgba(5, 150, 105, 0.45);
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        input[type=range].slider-3d::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        input[type=range].slider-2d {{
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 6px;
            background: linear-gradient(90deg, #E2E8F0 0%, #BFDBFE 40%, #001A66 100%);
            outline: none;
        }}
        input[type=range].slider-2d::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #001A66;
            border: 2px solid #FFFFFF;
            box-shadow: 0 2px 6px rgba(0, 26, 102, 0.45);
            cursor: pointer;
            transition: transform 0.15s ease;
        }}
        input[type=range].slider-2d::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        .actions-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding-top: 4px;
        }}
        .preset-buttons {{
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .preset-label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            margin-right: 4px;
        }}
        .btn {{
            appearance: none;
            border: 1px solid var(--card-border);
            background: #FFFFFF;
            color: #334155;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #F8FAFC;
            border-color: #94A3B8;
            color: #0F172A;
        }}
        .btn-active {{
            background: var(--light-blue);
            border-color: var(--accent-blue);
            color: var(--primary-blue);
            font-weight: 600;
        }}
        .toggle-group {{
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .toggle-label {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-size: 12.5px;
            font-weight: 500;
            color: #334155;
            cursor: pointer;
            user-select: none;
        }}
        .toggle-label input[type=checkbox] {{
            accent-color: var(--primary-blue);
            width: 15px;
            height: 15px;
            cursor: pointer;
        }}
        .chart-card {{
            background: #FFFFFF;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
            min-height: 800px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Control Panel Card -->
        <div class="control-panel">
            <div class="panel-header">
                <div class="header-title-block">
                    <h1>{title}</h1>
                    <p>{subtitle if subtitle else "Chemical Space Explorer • 3D Shape Similarity Encoded via Marker Size & Filtering"}</p>
                </div>
                <div class="header-badges">
                    <span class="query-badge">★ EGCG (Query Molecule, 3D: 1.000)</span>
                    <span class="size-indicator-badge">● Marker Size &prop; 3D Shape Sim (Max: {max_3d:.3f})</span>
                    <span class="stats-badge" id="stats-badge">
                        Visible: <b id="count-visible">{total_candidates:,}</b> / {total_candidates:,} (<span id="pct-visible">100.0%</span>)
                    </span>
                </div>
            </div>

            <!-- Dual Interactive Sliders Grid -->
            <div class="sliders-grid">
                <!-- Slider 1: 3D Shape Similarity (Size Channel) -->
                <div class="slider-box">
                    <div class="slider-box-header">
                        <div class="slider-title-group">
                            <span class="title">3D Shape Similarity Cutoff:</span>
                            <span class="pill-3d" id="val-3d">≥ 0.000</span>
                        </div>
                        <span class="slider-instruction">Encoded by Marker Size (Larger = More 3D Similar)</span>
                    </div>
                    <div class="slider-track-wrap">
                        <span class="slider-endpoint">0.00 (All)</span>
                        <input type="range" class="slider-3d" id="slider-3d" min="0.000" max="1.000" step="0.005" value="0.000">
                        <span class="slider-endpoint">1.00 (Exact 3D)</span>
                    </div>
                </div>

                <!-- Slider 2: 2D Bioactivity Similarity (Color Channel) -->
                <div class="slider-box">
                    <div class="slider-box-header">
                        <div class="slider-title-group">
                            <span class="title">2D Bioactivity Profile Cutoff:</span>
                            <span class="pill-2d" id="val-2d">≥ 0.000</span>
                        </div>
                        <span class="slider-instruction">Encoded by Color (Sapphire Blue = High, White = Low)</span>
                    </div>
                    <div class="slider-track-wrap">
                        <span class="slider-endpoint">0.00 (All)</span>
                        <input type="range" class="slider-2d" id="slider-2d" min="0.000" max="1.000" step="0.005" value="0.000">
                        <span class="slider-endpoint">1.00 (Exact 2D)</span>
                    </div>
                </div>
            </div>

            <!-- Presets & Toggles -->
            <div class="actions-row">
                <div class="preset-buttons">
                    <span class="preset-label">3D & 2D Presets:</span>
                    <button class="btn preset-btn btn-active" id="btn-all">Show All</button>
                    <button class="btn preset-btn" id="btn-3d-40">3D Shape ≥ 0.40</button>
                    <button class="btn preset-btn" id="btn-3d-50">3D Shape ≥ 0.50</button>
                    <button class="btn preset-btn" id="btn-top100-3d">Top 100 3D Matches</button>
                    <button class="btn preset-btn" id="btn-dual-hits">Dual 2D+3D Hits (Both ≥ 0.35)</button>
                    <button class="btn" id="btn-reset">Reset</button>
                </div>
                <div class="toggle-group">
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-bg" checked>
                        Show 1,000 Neutral Gray Background Molecules
                    </label>
                </div>
            </div>
        </div>

        <!-- Interactive Plotly Chart Card -->
        <div class="chart-card">
            {plot_inner_html}
        </div>
    </div>

    <!-- Client-side Interactive Filtering Script -->
    <script>
    const CANDIDATE_CACHE = {{
        x: {json_cand_x},
        y: {json_cand_y},
        scores_2d: {json_cand_scores_2d},
        scores_3d: {json_cand_scores_3d},
        sizes: {json_cand_sizes},
        hover: {json_cand_hover}
    }};

    document.addEventListener("DOMContentLoaded", function() {{
        function initInteractiveChemicalSpace() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv || !plotDiv.data || plotDiv.data.length < 2) {{
                setTimeout(initInteractiveChemicalSpace, 50);
                return;
            }}

            const totalCandidates = CANDIDATE_CACHE.scores_3d.length;
            const slider3D = document.getElementById('slider-3d');
            const slider2D = document.getElementById('slider-2d');
            const val3D = document.getElementById('val-3d');
            const val2D = document.getElementById('val-2d');
            const countVisible = document.getElementById('count-visible');
            const pctVisible = document.getElementById('pct-visible');
            const toggleBg = document.getElementById('toggle-bg');
            const btnAll = document.getElementById('btn-all');
            const btnReset = document.getElementById('btn-reset');
            const btn3D40 = document.getElementById('btn-3d-40');
            const btn3D50 = document.getElementById('btn-3d-50');
            const btnTop1003D = document.getElementById('btn-top100-3d');
            const btnDualHits = document.getElementById('btn-dual-hits');

            function applyFilters(cutoff3D, cutoff2D) {{
                const filtX = [];
                const filtY = [];
                const filtColors = [];
                const filtSizes = [];
                const filtScores3D = [];
                const filtHover = [];

                for (let i = 0; i < totalCandidates; i++) {{
                    if (CANDIDATE_CACHE.scores_3d[i] >= cutoff3D && CANDIDATE_CACHE.scores_2d[i] >= cutoff2D) {{
                        filtX.push(CANDIDATE_CACHE.x[i]);
                        filtY.push(CANDIDATE_CACHE.y[i]);
                        filtColors.push(CANDIDATE_CACHE.scores_2d[i]);
                        filtSizes.push(CANDIDATE_CACHE.sizes[i]);
                        filtScores3D.push(CANDIDATE_CACHE.scores_3d[i]);
                        filtHover.push(CANDIDATE_CACHE.hover[i]);
                    }}
                }}

                val3D.textContent = '≥ ' + cutoff3D.toFixed(3);
                val2D.textContent = '≥ ' + cutoff2D.toFixed(3);
                countVisible.textContent = filtScores3D.length.toLocaleString();
                pctVisible.textContent = ((filtScores3D.length / totalCandidates) * 100).toFixed(1) + '%';

                Plotly.restyle(plotDiv, {{
                    x: [filtX],
                    y: [filtY],
                    'marker.color': [filtColors],
                    'marker.size': [filtSizes],
                    hovertext: [filtHover]
                }}, [1]);
            }}

            slider3D.addEventListener('input', function(e) {{
                applyFilters(parseFloat(e.target.value), parseFloat(slider2D.value));
            }});

            slider2D.addEventListener('input', function(e) {{
                applyFilters(parseFloat(slider3D.value), parseFloat(e.target.value));
            }});

            if (toggleBg) {{
                toggleBg.addEventListener('change', function(e) {{
                    Plotly.restyle(plotDiv, {{ visible: e.target.checked ? true : false }}, [0]);
                }});
            }}

            if (btnAll) {{
                btnAll.addEventListener('click', function() {{
                    slider3D.value = 0.0;
                    slider2D.value = 0.0;
                    applyFilters(0.0, 0.0);
                }});
            }}

            if (btnReset) {{
                btnReset.addEventListener('click', function() {{
                    slider3D.value = 0.0;
                    slider2D.value = 0.0;
                    applyFilters(0.0, 0.0);
                }});
            }}

            if (btn3D40) {{
                btn3D40.addEventListener('click', function() {{
                    slider3D.value = 0.40;
                    slider2D.value = 0.0;
                    applyFilters(0.40, 0.0);
                }});
            }}

            if (btn3D50) {{
                btn3D50.addEventListener('click', function() {{
                    slider3D.value = 0.50;
                    slider2D.value = 0.0;
                    applyFilters(0.50, 0.0);
                }});
            }}

            if (btnTop1003D) {{
                btnTop1003D.addEventListener('click', function() {{
                    const sorted = [...CANDIDATE_CACHE.scores_3d].sort((a, b) => b - a);
                    const cutoff = sorted[Math.min(99, sorted.length - 1)];
                    slider3D.value = cutoff;
                    slider2D.value = 0.0;
                    applyFilters(cutoff, 0.0);
                }});
            }}

            if (btnDualHits) {{
                btnDualHits.addEventListener('click', function() {{
                    slider3D.value = 0.35;
                    slider2D.value = 0.35;
                    applyFilters(0.35, 0.35);
                }});
            }}
        }}

        initInteractiveChemicalSpace();
    }});
    </script>
</body>
</html>"""

        output_html.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(full_html)
        return output_html


class MolecularSpacePipeline:
    def __init__(self, config: MolecularSpaceConfig = None, output_dir: Path = None):
        self.config = config or MolecularSpaceConfig()
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.papyrus_path = (
            (receptor_similar.MoleculeBioactivityPipeline._resolve_papyrus_path() if receptor_similar else None)
            or str(PROJECT_ROOT / "tutorial" / "data" / "data" / ".Papyrus")
        )
        self.dataset_manager = ChemicalSpaceDataset(self.papyrus_path, self.config)

    @staticmethod
    def _enrich_coords_with_metadata(enriched_df: pd.DataFrame, combined_df: pd.DataFrame):
        df = enriched_df.copy()
        if "SMILES" not in df.columns:
            if "SMILES" in combined_df.columns and len(df) == len(combined_df):
                df["SMILES"] = combined_df["SMILES"].values
            elif df.index.name == "SMILES":
                df["SMILES"] = df.index.values
            else:
                df["SMILES"] = combined_df["SMILES"].values

        sync_cols = [
            "sim_3d_shape", "sim_3d_color", "sim_3d_combo",
            "property_similarity", "Group", "Source", "Target", "pAffinity"
        ]
        for col in sync_cols:
            if col in combined_df.columns:
                smi_map = dict(zip(combined_df["SMILES"], combined_df[col]))
                df[col] = df["SMILES"].map(smi_map)

        return df

    def run_version_1_papyrus(self):
        print("\n" + "=" * 80)
        print(">>> EXECUTING VERSION 1: CHEMICAL SPACE VISUALIZATION (PAPYRUS-ONLY) <<<".center(80))
        print("=" * 80)

        t0 = time.time()
        pipeline_v1 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=False)
        result_v1 = pipeline_v1.run(
            self.config.query_smiles,
            min_paffinity=6.0,
            max_molecules=self.config.n_candidates,
            quality_filter=None,
            allow_chembl_fallback=False,
            show_progress=True
        )
        candidates_raw = result_v1.molecules

        candidate_df = self.dataset_manager.format_candidate_molecules(
            candidates_raw,
            group_label="Papyrus Candidates (High-Affinity, pAffinity >= 6.0)"
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
        enriched_df = self._enrich_coords_with_metadata(enriched_df, combined_df)

        out_png = self.output_dir / "molecular_space_v1_papyrus.png"
        out_html = self.output_dir / "molecular_space_v1_papyrus.html"

        title = "Molecular Chemical Space: Version 1 (Papyrus-Only Benchmark)"
        subtitle = (
            f"1,000 Gray Background Molecules vs. 1,000 Papyrus Candidates "
            f"({len(result_v1.targets)} Receptors, pAffinity >= 6.0 | Size: 3D Shape Sim | Color: 2D Sim)"
        )

        MolecularSpaceRenderer.render_matplotlib(enriched_df, x_col, y_col, out_png, title, subtitle)
        MolecularSpaceRenderer.render_plotly_interactive(enriched_df, x_col, y_col, out_html, title, subtitle)

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
            "score_2d_min": float(candidate_df["property_similarity"].min()),
            "score_2d_max": float(candidate_df["property_similarity"].max()),
            "score_3d_min": float(candidate_df["sim_3d_shape"].min()),
            "score_3d_max": float(candidate_df["sim_3d_shape"].max()),
            "png_path": str(out_png),
            "html_path": str(out_html)
        }

    def run_version_2_papyrus_chembl(self):
        print("\n" + "=" * 80)
        print(">>> EXECUTING VERSION 2: CHEMICAL SPACE VISUALIZATION (PAPYRUS + ChEMBL) <<<".center(80))
        print("=" * 80)

        t0 = time.time()
        pipeline_v2 = receptor_similar.MoleculeBioactivityPipeline(allow_chembl_fallback=True)

        result_v2 = pipeline_v2.run(
            self.config.query_smiles,
            min_paffinity=5.0,
            max_molecules=self.config.n_candidates,
            quality_filter=None,
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
            random_seed=101
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
        enriched_df = self._enrich_coords_with_metadata(enriched_df, combined_df)

        out_png = self.output_dir / "molecular_space_v2_papyrus_chembl.png"
        out_html = self.output_dir / "molecular_space_v2_papyrus_chembl.html"

        title = "Molecular Chemical Space: Version 2 (Papyrus + ChEMBL Broad Space)"
        subtitle = (
            f"1,000 Gray Background Molecules vs. 1,000 Candidates Ranked from 3,900 Pool "
            f"({len(result_v2.targets)} Receptors, pAffinity >= 5.0 | Size: 3D Shape Sim | Color: 2D Sim)"
        )

        MolecularSpaceRenderer.render_matplotlib(enriched_df, x_col, y_col, out_png, title, subtitle)
        MolecularSpaceRenderer.render_plotly_interactive(enriched_df, x_col, y_col, out_html, title, subtitle)

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
            "score_2d_min": float(candidate_df["property_similarity"].min()),
            "score_2d_max": float(candidate_df["property_similarity"].max()),
            "score_3d_min": float(candidate_df["sim_3d_shape"].min()),
            "score_3d_max": float(candidate_df["sim_3d_shape"].max()),
            "png_path": str(out_png),
            "html_path": str(out_html)
        }

    def run_all(self):
        v1_summary = self.run_version_1_papyrus()
        v2_summary = self.run_version_2_papyrus_chembl()
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS WITH 3D SIMILARITY SUCCESSFULLY GENERATED!".center(80))
        print("=" * 80)
        return {"v1": v1_summary, "v2": v2_summary}


if __name__ == "__main__":
    pipeline = MolecularSpacePipeline()
    pipeline.run_all()
