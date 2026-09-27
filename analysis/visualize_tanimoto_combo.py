import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Hyperparameters & System Constants ---
SEED_SMILES = "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"
ROCS_USE_COLORS = True
ROCS_OPT_PARAM = 0.5
RANDOM_STATE = 42

VIBRANT_BLUE_CMAP = [
    (0.00, "#FFFFFF"),
    (0.25, "#E1EDFE"),
    (0.50, "#85B6FF"),
    (0.75, "#1B64F2"),
    (1.00, "#001A66")
]


class MolecularSpaceRenderer:
    @classmethod
    def render_matplotlib(cls, df: pd.DataFrame, x_col: str, y_col: str, output_png: Path, title: str, subtitle: str):
        is_bg = df["Group"].str.contains("Background", case=False, na=False)
        is_query = df["Group"].str.contains("Query", case=False, na=False)
        is_candidate = (~is_bg) & (~is_query)

        df_bg = df[is_bg]
        df_cand = df[is_candidate].sort_values(by="sim_3d_combo", ascending=True)
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

        # 2. Candidate trace: Color = 2D Bioactivity, Size = TanimotoCombo [0.0 - 2.0]
        sim_2d = df_cand["property_similarity"].fillna(0.0).values
        sim_combo = df_cand["sim_3d_combo"].fillna(0.0).values
        marker_sizes = 22.0 + 220.0 * ((sim_combo / 1.0) ** 1.6)

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
            label=f"Candidates ({len(df_cand):,} Mols | Size $\\propto$ TanimotoCombo)",
            zorder=3
        )

        # 3. Query molecule trace (distinct ruby star marker)
        if not df_query.empty:
            qx, qy = df_query[x_col].iloc[0], df_query[y_col].iloc[0]
            ax.scatter(
                [qx],
                [qy],
                c="#E63946",
                s=450,
                marker="*",
                edgecolors="#1A1A1A",
                linewidths=2.2,
                label="Reference: EGCG (my_molecule)",
                zorder=10
            )
            ax.annotate(
                "★ EGCG (my_molecule)\n  [2D Sim: 1.000 | TanimotoCombo: 2.000]\n  [Shape: 1.000 | Color: 1.000]",
                xy=(qx, qy),
                xytext=(-160, 22),
                textcoords="offset points",
                fontsize=10.0,
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

        # Visual size scale legend for 3D TanimotoCombo similarity
        size_samples = [0.50, 0.65, 0.75, 0.90]
        size_handles = [
            mlines.Line2D([], [], color="#001A66", marker="o", linestyle="None",
                          markersize=np.sqrt(22.0 + 220.0 * ((v / 1.0) ** 1.6)),
                          markerfacecolor="#85B6FF", markeredgecolor="#001A66", markeredgewidth=0.8,
                          label=f"Combo: {v:.2f}")
            for v in size_samples
        ]
        leg_sizes = ax.legend(handles=size_handles, title="TanimotoCombo (Size)", loc="lower left", framealpha=0.92, edgecolor="#CBD5E0", fontsize=8.5, title_fontsize=9)
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
        df_cand = df[is_candidate].sort_values(by="sim_3d_combo", ascending=True)
        df_query = df[is_query]

        plotly_scale = [[pos, col] for pos, col in VIBRANT_BLUE_CMAP]

        cand_x = [float(v) for v in df_cand[x_col]]
        cand_y = [float(v) for v in df_cand[y_col]]
        cand_scores_2d = [round(float(v), 5) for v in df_cand["property_similarity"].fillna(0.0)]
        cand_scores_shape = [round(float(v), 5) for v in df_cand["sim_3d_shape"].fillna(0.0)]
        cand_scores_color = [round(float(v), 5) for v in df_cand["sim_3d_color"].fillna(0.0)]
        cand_scores_combo = [round(float(v), 5) for v in df_cand["sim_3d_combo"].fillna(0.0)]

        # Marker size scaled by TanimotoCombo
        cand_sizes = [round(6.0 + 22.0 * ((v / 1.0) ** 1.6), 2) for v in cand_scores_combo]

        cand_hover = []
        for _, row in df_cand.iterrows():
            sim2d = row.get("property_similarity", np.nan)
            sim_shape = row.get("sim_3d_shape", np.nan)
            sim_color = row.get("sim_3d_color", np.nan)
            combo = row.get("sim_3d_combo", np.nan)
            src = row.get("Source", "Papyrus")
            tgt = row.get("Target", "Unknown")
            paff = row.get("pAffinity", np.nan)
            paff_str = f"{paff:.2f}" if pd.notnull(paff) else "N/A"
            smi = str(row.get("SMILES", ""))
            smi_trunc = f"{smi[:42]}..." if len(smi) > 42 else smi
            cand_hover.append(
                f"<b>Candidate Ligand</b><br>"
                f"3D TanimotoCombo: <b>{combo:.4f}</b> / 2.000 (Marker Size)<br>"
                f"&bull; 3D Shape Tanimoto: <b>{sim_shape:.4f}</b><br>"
                f"&bull; 3D Color Tanimoto: <b>{sim_color:.4f}</b><br>"
                f"2D Bioactivity Similarity: <b>{sim2d:.4f}</b> (Marker Color)<br>"
                f"Target Receptors: {tgt}<br>"
                f"pAffinity: {paff_str} | Source: {src}<br>"
                f"SMILES: <code>{smi_trunc}</code>"
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

        # 2. Candidate trace (1,000 molecules with dual-encoding: Color=2D, Size=TanimotoCombo)
        fig.add_trace(go.Scatter(
            x=cand_x,
            y=cand_y,
            mode="markers",
            name="Candidates (Size = TanimotoCombo, Color = 2D)",
            customdata=cand_scores_combo,
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
                hovertext=[f"<b>Reference Query: EGCG</b><br>2D Sim: <b>1.000</b> | TanimotoCombo: <b>2.000</b><br>Shape: 1.000 | Color: 1.000<br>SMILES: <code>{q_smi}</code>"]
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
        json_cand_scores_combo = json.dumps(cand_scores_combo)
        json_cand_sizes = json.dumps(cand_sizes)
        json_cand_hover = json.dumps(cand_hover)

        total_candidates = len(df_cand)
        avg_combo = float(np.mean(cand_scores_combo))
        max_combo = float(np.max(cand_scores_combo))
        p50_combo = float(np.percentile(cand_scores_combo, 50))
        p75_combo = float(np.percentile(cand_scores_combo, 75))
        p90_combo = float(np.percentile(cand_scores_combo, 90))
        p75_2d = float(np.percentile(cand_scores_2d, 75))

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
            --emerald-light: #D1FAE5;
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
        .rocs-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
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
        .pill-combo {{
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
        input[type=range].slider-combo {{
            -webkit-appearance: none;
            width: 100%;
            height: 8px;
            border-radius: 6px;
            background: linear-gradient(90deg, #E2E8F0 0%, #A7F3D0 40%, #059669 100%);
            outline: none;
        }}
        input[type=range].slider-combo::-webkit-slider-thumb {{
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
        input[type=range].slider-combo::-webkit-slider-thumb:hover {{
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
            background: #DBEAFE !important;
            border-color: #2563EB !important;
            color: #1E40AF !important;
            font-weight: 700 !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
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
                    <p>{subtitle if subtitle else "Chemical Space Explorer • 3D TanimotoCombo (ROCS_USE_COLORS=True) Encoded via Marker Size & Cutoffs"}</p>
                </div>
                <div class="header-badges">
                    <span class="query-badge">★ EGCG Reference (Combo: 2.000)</span>
                    <span class="rocs-badge">&#x2713; ROCS_USE_COLORS = True</span>
                    <span class="size-indicator-badge">● Size &prop; TanimotoCombo (Max Cand: {max_combo:.3f})</span>
                    <span class="stats-badge" id="stats-badge">
                        Visible: <b id="count-visible">{total_candidates:,}</b> / {total_candidates:,} (<span id="pct-visible">100.0%</span>)
                    </span>
                </div>
            </div>

            <!-- Dual Interactive Sliders Grid -->
            <div class="sliders-grid">
                <!-- Slider 1: 3D TanimotoCombo (Size Channel) -->
                <div class="slider-box">
                    <div class="slider-box-header">
                        <div class="slider-title-group">
                            <span class="title">3D TanimotoCombo Cutoff:</span>
                            <span class="pill-combo" id="val-combo">≥ 0.000</span>
                        </div>
                        <span class="slider-instruction">Shape + Color Tanimoto [0.0 - 1.0 (Cand) | Ref = 2.00]</span>
                    </div>
                    <div class="slider-track-wrap">
                        <span class="slider-endpoint">0.00 (All)</span>
                        <input type="range" class="slider-combo" id="slider-combo" min="0.000" max="1.000" step="0.005" value="0.000">
                        <span class="slider-endpoint">1.00 (Top Cand)</span>
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
                        <input type="range" class="slider-2d" id="slider-2d" min="0.000" max="0.600" step="0.005" value="0.000">
                        <span class="slider-endpoint">0.60 (High 2D)</span>
                    </div>
                </div>
            </div>

            <!-- Presets & Toggles -->
            <div class="actions-row">
                <div class="preset-buttons">
                    <span class="preset-label">3D & 2D Presets:</span>
                    <button class="btn preset-btn btn-active" id="btn-all">Show All</button>
                    <button class="btn preset-btn" id="btn-combo-median">Combo ≥ {p50_combo:.2f} (Top 50%)</button>
                    <button class="btn preset-btn" id="btn-combo-p75">Combo ≥ {p75_combo:.2f} (Top 25%)</button>
                    <button class="btn preset-btn" id="btn-combo-p90">Combo ≥ {p90_combo:.2f} (Top 10%)</button>
                    <button class="btn preset-btn" id="btn-top100-combo">Top 100 3D Matches</button>
                    <button class="btn preset-btn" id="btn-dual-hits">Dual 2D+3D Hits</button>
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
        scores_combo: {json_cand_scores_combo},
        sizes: {json_cand_sizes},
        hover: {json_cand_hover},
        p50_combo: {p50_combo:.4f},
        p75_combo: {p75_combo:.4f},
        p90_combo: {p90_combo:.4f},
        p75_2d: {p75_2d:.4f}
    }};

    document.addEventListener("DOMContentLoaded", function() {{
        function initInteractiveChemicalSpace() {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv || !plotDiv.data || plotDiv.data.length < 2) {{
                setTimeout(initInteractiveChemicalSpace, 50);
                return;
            }}

            const totalCandidates = CANDIDATE_CACHE.scores_combo.length;
            const sliderCombo = document.getElementById('slider-combo');
            const slider2D = document.getElementById('slider-2d');
            const valCombo = document.getElementById('val-combo');
            const val2D = document.getElementById('val-2d');
            const countVisible = document.getElementById('count-visible');
            const pctVisible = document.getElementById('pct-visible');
            const toggleBg = document.getElementById('toggle-bg');
            const btnAll = document.getElementById('btn-all');
            const btnReset = document.getElementById('btn-reset');
            const btnComboMedian = document.getElementById('btn-combo-median');
            const btnComboP75 = document.getElementById('btn-combo-p75');
            const btnComboP90 = document.getElementById('btn-combo-p90');
            const btnTop100Combo = document.getElementById('btn-top100-combo');
            const btnDualHits = document.getElementById('btn-dual-hits');

            function setActivePreset(activeBtn) {{
                document.querySelectorAll('.preset-btn').forEach(btn => {{
                    btn.classList.remove('btn-active');
                }});
                if (activeBtn) {{
                    activeBtn.classList.add('btn-active');
                }}
            }}

            function applyFilters(cutoffCombo, cutoff2D) {{
                const filtX = [];
                const filtY = [];
                const filtColors = [];
                const filtSizes = [];
                const filtScoresCombo = [];
                const filtHover = [];

                for (let i = 0; i < totalCandidates; i++) {{
                    if (CANDIDATE_CACHE.scores_combo[i] >= cutoffCombo && CANDIDATE_CACHE.scores_2d[i] >= cutoff2D) {{
                        filtX.push(CANDIDATE_CACHE.x[i]);
                        filtY.push(CANDIDATE_CACHE.y[i]);
                        filtColors.push(CANDIDATE_CACHE.scores_2d[i]);
                        filtSizes.push(CANDIDATE_CACHE.sizes[i]);
                        filtScoresCombo.push(CANDIDATE_CACHE.scores_combo[i]);
                        filtHover.push(CANDIDATE_CACHE.hover[i]);
                    }}
                }}

                valCombo.textContent = '≥ ' + cutoffCombo.toFixed(3);
                val2D.textContent = '≥ ' + cutoff2D.toFixed(3);
                countVisible.textContent = filtScoresCombo.length.toLocaleString();
                pctVisible.textContent = ((filtScoresCombo.length / totalCandidates) * 100).toFixed(1) + '%';

                Plotly.restyle(plotDiv, {{
                    x: [filtX],
                    y: [filtY],
                    'marker.color': [filtColors],
                    'marker.size': [filtSizes],
                    hovertext: [filtHover]
                }}, [1]);
            }}

            sliderCombo.addEventListener('input', function(e) {{
                setActivePreset(null);
                applyFilters(parseFloat(e.target.value), parseFloat(slider2D.value));
            }});

            slider2D.addEventListener('input', function(e) {{
                setActivePreset(null);
                applyFilters(parseFloat(sliderCombo.value), parseFloat(e.target.value));
            }});

            if (toggleBg) {{
                toggleBg.addEventListener('change', function(e) {{
                    Plotly.restyle(plotDiv, {{ visible: e.target.checked ? true : false }}, [0]);
                }});
            }}

            if (btnAll) {{
                btnAll.addEventListener('click', function() {{
                    setActivePreset(this);
                    sliderCombo.value = 0.0;
                    slider2D.value = 0.0;
                    applyFilters(0.0, 0.0);
                }});
            }}

            if (btnReset) {{
                btnReset.addEventListener('click', function() {{
                    setActivePreset(btnAll);
                    sliderCombo.value = 0.0;
                    slider2D.value = 0.0;
                    applyFilters(0.0, 0.0);
                }});
            }}

            if (btnComboMedian) {{
                btnComboMedian.addEventListener('click', function() {{
                    setActivePreset(this);
                    sliderCombo.value = CANDIDATE_CACHE.p50_combo;
                    slider2D.value = 0.0;
                    applyFilters(CANDIDATE_CACHE.p50_combo, 0.0);
                }});
            }}

            if (btnComboP75) {{
                btnComboP75.addEventListener('click', function() {{
                    setActivePreset(this);
                    sliderCombo.value = CANDIDATE_CACHE.p75_combo;
                    slider2D.value = 0.0;
                    applyFilters(CANDIDATE_CACHE.p75_combo, 0.0);
                }});
            }}

            if (btnComboP90) {{
                btnComboP90.addEventListener('click', function() {{
                    setActivePreset(this);
                    sliderCombo.value = CANDIDATE_CACHE.p90_combo;
                    slider2D.value = 0.0;
                    applyFilters(CANDIDATE_CACHE.p90_combo, 0.0);
                }});
            }}

            if (btnTop100Combo) {{
                btnTop100Combo.addEventListener('click', function() {{
                    setActivePreset(this);
                    const sorted = [...CANDIDATE_CACHE.scores_combo].sort((a, b) => b - a);
                    const cutoff = sorted[Math.min(99, sorted.length - 1)];
                    sliderCombo.value = cutoff;
                    slider2D.value = 0.0;
                    applyFilters(cutoff, 0.0);
                }});
            }}

            if (btnDualHits) {{
                btnDualHits.addEventListener('click', function() {{
                    setActivePreset(this);
                    sliderCombo.value = CANDIDATE_CACHE.p75_combo;
                    slider2D.value = CANDIDATE_CACHE.p75_2d;
                    applyFilters(CANDIDATE_CACHE.p75_combo, CANDIDATE_CACHE.p75_2d);
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


def re_render_all():
    analysis_dir = Path("/home/kolar/build_projects/drugex_edu/analysis")
    
    v1_csv = analysis_dir / "molecular_space_v1_papyrus_tanimotocombo.csv"
    v2_csv = analysis_dir / "molecular_space_v2_papyrus_chembl_tanimotocombo.csv"

    print("Re-rendering V1...")
    df_v1 = pd.read_csv(v1_csv)
    v1_png = analysis_dir / "molecular_space_v1_papyrus_tanimotocombo.png"
    v1_html = analysis_dir / "molecular_space_v1_papyrus_tanimotocombo.html"
    MolecularSpaceRenderer.render_matplotlib(
        df_v1, "TSNE_1", "TSNE_2", v1_png,
        title="Molecular Chemical Space: Version 1 (Papyrus-Only Benchmark)",
        subtitle="1,000 Gray Background Molecules vs. 1,000 Papyrus Candidates (ROCS_USE_COLORS=True | Size: TanimotoCombo | Color: 2D Sim)"
    )
    MolecularSpaceRenderer.render_plotly_interactive(
        df_v1, "TSNE_1", "TSNE_2", v1_html,
        title="Molecular Chemical Space: Version 1 (Papyrus-Only Benchmark)",
        subtitle="1,000 Gray Background Molecules vs. 1,000 Papyrus Candidates (ROCS_USE_COLORS=True | Size: TanimotoCombo | Color: 2D Sim)"
    )
    print("V1 rendered!")

    print("Re-rendering V2...")
    df_v2 = pd.read_csv(v2_csv)
    v2_png = analysis_dir / "molecular_space_v2_papyrus_chembl_tanimotocombo.png"
    v2_html = analysis_dir / "molecular_space_v2_papyrus_chembl_tanimotocombo.html"
    MolecularSpaceRenderer.render_matplotlib(
        df_v2, "TSNE_1", "TSNE_2", v2_png,
        title="Molecular Chemical Space: Version 2 (Papyrus + ChEMBL Broad Space)",
        subtitle="1,000 Gray Background Molecules vs. 1,000 Candidates Ranked from Broad Pool (ROCS_USE_COLORS=True | Size: TanimotoCombo | Color: 2D Sim)"
    )
    MolecularSpaceRenderer.render_plotly_interactive(
        df_v2, "TSNE_1", "TSNE_2", v2_html,
        title="Molecular Chemical Space: Version 2 (Papyrus + ChEMBL Broad Space)",
        subtitle="1,000 Gray Background Molecules vs. 1,000 Candidates Ranked from Broad Pool (ROCS_USE_COLORS=True | Size: TanimotoCombo | Color: 2D Sim)"
    )
    print("V2 rendered!")

if __name__ == "__main__":
    re_render_all()
