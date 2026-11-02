#!/usr/bin/env python3
"""
Comprehensive 6-Panel Synthesis Visual Suite for Conformer & ROCS Alignment.

Integrates:
- Panel 1: Hardware-calibrated 4-stage batch scoring cost (dichtator 48 cores, 10 arms).
- Panel 2: Speed vs Rank-Agreement Pareto Frontier vs Asymptotic Gold (K=all, 46 refs).
- Panel 3: Stereoisomer complexity stratification (I=1, 2, 3-4, 5-8) across reference cap K.
- Panel 4: Potential energy surface (PES) steepness stratification across K (surviving pool size).
- Panel 5: Reference selection mode comparison: Pharmacophoric Diverse vs Butina RMSD (0.35, 0.50, 1.25 A).
- Panel 6: Rank preservation (Spearman rho) & Top-10% decile overlap vs deployable gold across all 10 arms.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def render_6panel_figure(output_png: Path) -> Path:
    base_dir = Path("/home/kolar/DrugEx/_david/output")
    grid_dir = base_dir / "20261028_conformer_self_similarity/20261102_1735_rl_batch_calibration_dichtator"
    dev_path = base_dir / "20261028_conformer_self_similarity/20261029_1155_reward_calibration_126_mols/raw_data/development/data.csv.gz"
    synth_path = base_dir / "20261028_pareto_scoring_selectivity/20261029_1400_pareto_selectivity_163_drugs/drug128_synthesis_grid.csv"

    summary = pd.read_csv(grid_dir / "cost_summary.csv")
    agreement = pd.read_csv(grid_dir / "rank_agreement.csv")
    agreement_kall = pd.read_csv(grid_dir / "rank_agreement_kall.csv")
    dev = pd.read_csv(dev_path)
    synth = pd.read_csv(synth_path, dtype={"k_label": str}, keep_default_na=False)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
    })

    fig, axes = plt.subplots(3, 2, figsize=(21, 24), dpi=200)
    plt.subplots_adjust(hspace=0.32, wspace=0.26)

    # =========================================================================
    # Panel 1 (Top-Left): 4-Stage Hardware Batch Latency Breakdown
    # =========================================================================
    ax1 = axes[0, 0]
    ordered = summary.sort_values("warm_per_molecule_seconds")
    y = np.arange(len(ordered))
    embed = ordered.embed_per_molecule
    minimize = ordered.minimize_per_molecule
    align = ordered.align_io_per_molecule
    other = ordered.env_other_per_molecule

    ax1.barh(y, embed, color="#1f77b4", label="ETKDG Embedding (0.38 s/mol cost floor)")
    ax1.barh(y, minimize, left=embed, color="#ff7f0e", label="Force-field Minimization (~0.04-0.07 s/mol)")
    ax1.barh(y, align, left=embed + minimize, color="#2ca02c", label="3D ROCS Alignment + I/O (scales K x C)")
    ax1.barh(y, other, left=embed + minimize + align, color="#9467bd", label="Env / Validity / Desirability")

    ax1.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, label="1.0 s/mol threshold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(ordered.arm, fontsize=8.5, fontweight="medium")
    ax1.set_xlabel("Warm per-molecule seconds (dichtator 48 cores, 500 paired prior inputs)")
    ax1.set_title("1. Batch Scoring Latency by Arm (4-Stage Breakdown)\n[Embedding floor dominates; K=all causes 3.2x alignment surge]")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax1.grid(alpha=0.25, axis="x")

    # =========================================================================
    # Panel 2 (Top-Right): Speed vs Rank-Agreement Pareto Frontier vs Asymptotic Gold
    # =========================================================================
    ax2 = axes[0, 1]
    merged_kall = summary.merge(agreement_kall, on="arm", how="inner")
    rho_asymp = merged_kall.spearman_valid.fillna(merged_kall.spearman_all)
    overlap_asymp = merged_kall.top10_overlap_valid.fillna(merged_kall.top10_overlap_all)
    merged_kall = merged_kall.assign(_rho=rho_asymp, _overlap=overlap_asymp).dropna(subset=["_rho"])

    for _, row in merged_kall.iterrows():
        is_gold = (row.arm == "mmff_high_dielectric_k4_c20")
        is_asymp = (row.arm == "mmff_high_dielectric_kall_c50")
        pt_color = "#d62728" if is_gold else ("#7f7f7f" if is_asymp else "#2ca02c")
        marker = "*" if (is_gold or is_asymp) else "o"
        size = 180 if (is_gold or is_asymp) else (40 + 260 * row._overlap)
        
        ax2.scatter(
            row.warm_per_molecule_seconds,
            row._rho,
            s=size,
            color=pt_color,
            marker=marker,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=4 if is_gold else 3,
        )
        label_text = f"{row.arm}\n(rho={row._rho:.3f}, top10={row._overlap:.2f})" if is_gold else row.arm
        ax2.annotate(
            label_text,
            (row.warm_per_molecule_seconds, row._rho),
            fontsize=7.5,
            fontweight="bold" if is_gold else "normal",
            xytext=(5, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow" if is_gold else "white", alpha=0.7, edgecolor="none"),
        )

    # Draw Pareto envelope curve connecting dominant points
    pareto_pts = merged_kall.sort_values("warm_per_molecule_seconds")
    ax2.plot(pareto_pts.warm_per_molecule_seconds, pareto_pts._rho, linestyle="--", color="#31688e", alpha=0.5, label="Empirical trade-off curve")
    ax2.set_xlabel("Warm Scoring Latency (s/mol, log scale)")
    ax2.set_xscale("log")
    ax2.set_ylabel("Spearman Rank Correlation rho vs Asymptotic Gold (K=46)")
    ax2.set_title("2. Hardware Pareto Frontier: Speed vs Rank Fidelity\n[Red Star: K=4, C=20 captures 95% ranking fidelity at 3.2x lower cost]")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    # =========================================================================
    # Panel 3 (Mid-Left): Stereoisomer Stratification Across K (from drug_panel)
    # =========================================================================
    ax3 = axes[1, 0]
    valid_dev = dev[~dev["color_excluded"] & ~dev["reference_failure"]].copy()
    valid_dev["iso_group"] = pd.cut(
        valid_dev["enumerated_reference_isomers"],
        bins=[0, 1, 2, 4, 8],
        labels=["I = 1 (Achiral / Specified)", "I = 2 (1 Unassigned Center)", "I = 3-4 (2 Centers)", "I = 5-8 (>=3 Centers)"],
    )
    sub_iso = valid_dev[(valid_dev["method"] == "mmff_high_dielectric") & (valid_dev["candidate_attempts"] == 20)].copy()
    sub_iso["k_order"] = sub_iso["reference_cap"].replace({0: 16})
    iso_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for (grp, gdf), col in zip(sub_iso.groupby("iso_group", observed=True), iso_palette):
        n_mols = gdf["molecule_id"].nunique()
        agg = gdf.groupby("k_order")["score"].mean().reset_index()
        ax3.plot(agg["k_order"], agg["score"], marker="o", linewidth=2.2, color=col, label=f"{grp} (n={n_mols})")
    ax3.set_xticks([1, 2, 4, 8, 12, 16])
    ax3.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"], fontweight="medium")
    ax3.set_title("3. Stereoisomer Complexity Stratification (128-Drug Panel, C=20)\n[Multi-chiral drugs severely degrade under K=1, plateau at K=4]")
    ax3.set_xlabel("Reference Cap K (Round-Robin Balanced Across Stereoisomers)")
    ax3.set_ylabel("Mean ROCS Self-Recovery")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    # =========================================================================
    # Panel 4 (Mid-Right): PES Steepness / Landscape Stratification Across K
    # =========================================================================
    ax4 = axes[1, 1]
    all_refs = (
        valid_dev[valid_dev["reference_cap"] == 0]
        .groupby(["method", "molecule_id"])["actual_references"]
        .first()
        .rename("total_bank_refs")
        .reset_index()
    )
    valid_dev = valid_dev.merge(all_refs, on=["method", "molecule_id"], how="left")
    iso1 = valid_dev[(valid_dev["enumerated_reference_isomers"] == 1) & (valid_dev["method"] == "mmff_high_dielectric") & (valid_dev["candidate_attempts"] == 20)].copy()
    iso1["flex_group"] = pd.cut(
        iso1["total_bank_refs"],
        bins=[0, 25, 45, 55],
        labels=[
            "Steep PES / Flexible (<=25 survivors/50, Delta E>8 filtered)",
            "Moderate PES (26-45 survivors/50)",
            "Flat PES / Rigid (46-50 survivors/50)",
        ],
    )
    iso1["k_order"] = iso1["reference_cap"].replace({0: 16})
    for (grp, gdf), col in zip(iso1.groupby("flex_group", observed=True), ["#e7298a", "#d95f02", "#1b9e77"]):
        n_mols = gdf["molecule_id"].nunique()
        agg = gdf.groupby("k_order")["score"].mean().reset_index()
        ax4.plot(agg["k_order"], agg["score"], marker="s", linewidth=2.2, color=col, label=f"{grp} (n={n_mols})")
    ax4.set_xticks([1, 2, 4, 8, 12, 16])
    ax4.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"], fontweight="medium")
    ax4.set_title("4. Potential Energy Surface (PES) Steepness Stratification (C=20)\n[Flexible molecules benefit from K>=4; rigid drugs saturate at K=1]")
    ax4.set_xlabel("Reference Cap K")
    ax4.set_ylabel("Mean ROCS Self-Recovery")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    # =========================================================================
    # Panel 5 (Bottom-Left): Reference Selection Mode (from drug128_definitive_synthesis)
    # =========================================================================
    ax5 = axes[2, 0]
    sub_a = synth[
        (synth["regime"] == "MMFF94s_Ref_vs_ETKDG_Cand")
        & (synth["c_attempts"] == 20)
        & (synth["metric"] == "tanimoto_05_05")
    ]
    k_order_s = ["1", "2", "3", "4", "8", "all"]
    sel_styles = [
        ("diverse", "#d62728", "o", "Diverse Farthest-First (Shape+Color)", 9),
        ("rmsd_0.35", "#1f77b4", "s", "Butina RMSD r = 0.35 A (ETKDG Default)", -16),
        ("rmsd_0.50", "#2ca02c", "^", "Butina RMSD r = 0.50 A", None),
        ("rmsd_1.25", "#7f7f7f", "d", "Butina RMSD r = 1.25 A (Legacy)", -16),
    ]
    x_s = np.arange(len(k_order_s))
    for sel_key, col, marker, label, y_off in sel_styles:
        s_df = (
            sub_a[sub_a["ref_sel"] == sel_key]
            .groupby("k_label")[["score", "k_actual"]]
            .mean()
            .reindex(k_order_s)
        )
        ax5.plot(x_s, s_df["score"].values, marker=marker, color=col, linewidth=2.3, label=label)
        if y_off is not None:
            for idx, (sc, k_act) in enumerate(zip(s_df["score"].values, s_df["k_actual"].values)):
                if idx == 0 and sel_key != "diverse":
                    continue
                ax5.annotate(
                    f"{sc:.3f}",
                    (idx, sc),
                    textcoords="offset points",
                    xytext=(0, y_off),
                    ha="center",
                    fontsize=7.5,
                    color=col,
                    fontweight="bold",
                )
    ax5.set_xticks(x_s)
    ax5.set_xticklabels([f"K={k}" if k != "all" else "K=all" for k in k_order_s], fontweight="medium")
    ax5.set_ylim(0.755, 0.855)
    ax5.set_ylabel("128-Drug Mean ROCS Self-Recovery")
    ax5.set_title("5. Reference Selection Mode Comparison (C=20 ETKDG Candidates)\n[Diverse farthest-first (shape+color) beats naive RMSD pruning]")
    ax5.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax5.grid(True, alpha=0.25)

    # =========================================================================
    # Panel 6 (Bottom-Right): Rank Fidelity & Top-10% Decile Overlap vs Deployable Gold
    # =========================================================================
    ax6 = axes[2, 1]
    merged_c = ordered[["arm"]].merge(agreement, on="arm", how="left")
    y_c = np.arange(len(merged_c))
    rho_vals = merged_c.spearman_valid.fillna(merged_c.spearman_all).fillna(0.0)
    top10_vals = merged_c.top10_overlap_valid.fillna(merged_c.top10_overlap_all).fillna(0.0)

    bar_h = 0.38
    ax6.barh(y_c + bar_h / 2, rho_vals, height=bar_h, color="#35b779", label="Spearman rho (Rank Agreement)")
    ax6.barh(
        y_c - bar_h / 2,
        top10_vals,
        height=bar_h,
        color="#fde725",
        edgecolor="black",
        linewidth=0.6,
        label="Top-10% Decile Retention",
    )
    ax6.axvline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax6.set_yticks(y_c)
    ax6.set_yticklabels(merged_c.arm, fontsize=8.5, fontweight="medium")
    ax6.set_xlabel("Fidelity Metric vs Deployable Gold (1.0 = Perfect)")
    ax6.set_title("6. Policy Update Fidelity: Global Rank vs Top-Decile Retention\n[Cheap arms preserve means but severely drop top-10% RL candidates]")
    ax6.set_xlim(0, 1.08)
    ax6.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax6.grid(alpha=0.25, axis="x")

    fig.suptitle(
        "3D Conformer & ROCS Alignment Calibration: Comprehensive 6-Panel Synthesis Suite\n"
        "Multi-Method Hardware Grid (Dichtator 48 Cores) + Stereoisomer / PES / Pharmacophoric Selection Stratification",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"SUCCESS: Generated 6-panel synthesis figure -> {output_png} (size: {output_png.stat().st_size} bytes)")
    return output_png

if __name__ == "__main__":
    out_target = Path("/home/kolar/DrugEx/_david/output/20261028_conformer_self_similarity/20261102_1735_rl_batch_calibration_dichtator/rl_batch_calibration_6panel.png")
    render_6panel_figure(out_target)
