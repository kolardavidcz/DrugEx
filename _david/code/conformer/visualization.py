from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ConformerVisualizer:
    """Produces publication-grade figures, annotated matrices, and interactive 3D web visualizations.

    Strictly conforms to mathematical bounds:
    - ROCS TanimotoCombo is explicitly clamped and labeled in [0.00, 2.00].
    - Eliminates 3D perspective distortion using complementary annotated 2D heatmaps.
    - Embeds interactive Plotly surfaces for browser-based rotation and inspection.
    """

    @staticmethod
    def render_3d_surfaces(
        results_by_variant: Mapping[str, pd.DataFrame], output_path: Path | str
    ) -> Path:
        """Render multi-panel 3D surface plots strictly labeled [0.0, 2.05] for ROCS TanimotoCombo.

        Args:
            results_by_variant: Mapping of variant display title to scanned grid DataFrame.
            output_path: Destination image path.

        Returns:
            Resolved Path to the saved figure.
        """
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 11), dpi=300)
        positions = [(2, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5)]

        for (title, df), pos in zip(results_by_variant.items(), positions):
            ax = fig.add_subplot(*pos, projection="3d")

            piv = df.pivot(
                index="k_gen_requested",
                columns="k_ref_requested",
                values="tanimoto_combo",
            )
            x_vals = piv.columns.to_numpy()
            y_vals = piv.index.to_numpy()
            x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
            z_vals = piv.to_numpy()

            surf = ax.plot_surface(
                x_mesh,
                y_mesh,
                z_vals,
                cmap="viridis",
                vmin=0.0,
                vmax=2.0,
                edgecolor="none",
                alpha=0.88,
                antialiased=True,
            )
            ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel(r"$K_{\mathrm{ref}}$ (Reference)", fontsize=8, labelpad=6)
            ax.set_ylabel(r"$K_{\mathrm{gen}}$ (Candidate)", fontsize=8, labelpad=6)
            ax.set_zlabel("ROCS [0.0 - 2.0]", fontsize=8, labelpad=6)
            ax.set_zlim(0.0, 2.05)
            ax.view_init(elev=26, azim=-125)
            fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.08)

        plt.suptitle(
            r"ROCS TanimotoCombo Self-Similarity Surface: Reference ($K_{\mathrm{ref}}$) vs. Candidate ($K_{\mathrm{gen}}$)",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_file

    @staticmethod
    def render_heatmaps(
        results_by_variant: Mapping[str, pd.DataFrame], output_path: Path | str
    ) -> Path:
        """Render 2D heatmap matrix with explicit numeric cell annotations strictly bounded [0.0, 2.0].

        Provides distortion-free quantitative inspection for every (K_ref, K_gen) pair.

        Args:
            results_by_variant: Mapping of variant display title to grid DataFrame.
            output_path: Destination PNG image path.

        Returns:
            Resolved Path to the saved heatmap figure.
        """
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        n_vars = len(results_by_variant)
        fig, axes = plt.subplots(1, n_vars, figsize=(4.2 * n_vars, 4.8), dpi=300)

        im = None
        for idx, ((title, df), ax) in enumerate(zip(results_by_variant.items(), axes)):
            piv = df.pivot(
                index="k_gen_requested",
                columns="k_ref_requested",
                values="tanimoto_combo",
            )
            im = ax.imshow(
                piv.to_numpy(),
                origin="lower",
                cmap="viridis",
                vmin=0.5,
                vmax=2.0,
                aspect="auto",
            )
            clean_title = title.split(" (")[0]
            ax.set_title(clean_title, fontsize=9, fontweight="bold")
            ax.set_xlabel(r"$K_{\mathrm{ref}}$", fontsize=9)
            if idx == 0:
                ax.set_ylabel(r"$K_{\mathrm{gen}}$", fontsize=9)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index, fontsize=8)
            else:
                ax.set_yticks([])

            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels(piv.columns, fontsize=8)

            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    val = float(piv.iloc[i, j])
                    text_color = "black" if val > 1.6 else "white"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                    )

        if im is not None:
            fig.colorbar(
                im,
                ax=axes,
                orientation="horizontal",
                shrink=0.35,
                pad=0.20,
                label="ROCS TanimotoCombo [Theoretical Range: 0.00 - 2.00]",
            )
        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_file

    @staticmethod
    def render_interactive_plotly(
        results_by_variant: Mapping[str, pd.DataFrame], output_path: Path | str
    ) -> Path:
        """Generate interactive 3D Plotly HTML widget allowing complete 3D orbital rotation and inspection.

        Args:
            results_by_variant: Mapping of variant display title to grid DataFrame.
            output_path: Destination HTML file path.

        Returns:
            Resolved Path to the saved interactive HTML file.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        titles = list(results_by_variant.keys())
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
                [{"type": "surface"}, {"type": "surface"}, None],
            ],
            subplot_titles=titles,
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

        for (_title, df), (r, c) in zip(results_by_variant.items(), positions):
            piv = df.pivot(
                index="k_gen_requested",
                columns="k_ref_requested",
                values="tanimoto_combo",
            )
            x_vals = piv.columns.to_numpy()
            y_vals = piv.index.to_numpy()
            z_vals = piv.to_numpy()

            fig.add_trace(
                go.Surface(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    colorscale="Viridis",
                    cmin=0.5,
                    cmax=2.0,
                    showscale=(r == 1 and c == 1),
                    colorbar=dict(title="ROCS [0-2]", len=0.6, y=0.5)
                    if (r == 1 and c == 1)
                    else None,
                    hovertemplate="K_ref: %{x}<br>K_gen: %{y}<br>ROCS: %{z:.3f} / 2.000<extra></extra>",
                ),
                row=r,
                col=c,
            )

        for idx in range(1, len(titles) + 1):
            scene_name = f"scene{idx}" if idx > 1 else "scene"
            fig.update_layout(
                {
                    scene_name: dict(
                        xaxis_title="K_ref (Basins)",
                        yaxis_title="K_gen (Candidates)",
                        zaxis=dict(title="ROCS TanimotoCombo", range=[0.0, 2.05]),
                        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
                    )
                }
            )

        fig.update_layout(
            title_text="Interactive 3D ROCS Self-Similarity Surfaces (Rotate, Zoom & Hover)",
            height=900,
            width=1400,
            template="plotly_white",
        )
        fig.write_html(str(output_file), include_plotlyjs="cdn")
        return output_file

    @staticmethod
    def render_cross_sectional_recovery(
        results_by_variant: Mapping[str, pd.DataFrame], output_path: Path | str
    ) -> Path:
        """Generate 2D cross-sectional recovery curves comparing fixed K_ref and K_gen slices.

        Args:
            results_by_variant: Mapping of variant display title to grid DataFrame.
            output_path: Destination PNG image path.

        Returns:
            Resolved Path to the saved cross-sectional plot.
        """
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        ax1, ax2 = axes[0], axes[1]
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

        for (title, df), col in zip(results_by_variant.items(), colors):
            slice_df = cast(pd.DataFrame, df[df["k_ref_requested"] == 3]).sort_values(
                by="k_gen_requested"
            )
            ax1.plot(
                slice_df["k_gen_requested"],
                slice_df["tanimoto_combo"],
                marker="o",
                linewidth=2.0,
                color=col,
                label=title.split(" (")[0],
            )

        ax1.axhline(
            2.0, color="gray", linestyle="--", alpha=0.6, label="Ideal Self-Match (2.0)"
        )
        ax1.set_title(
            r"A: Self-Similarity Scaling at Fixed $K_{\mathrm{ref}}=3$",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel(
            r"Number of Generated Candidate Conformers ($K_{\mathrm{gen}}$)",
            fontsize=10,
        )
        ax1.set_ylabel("ROCS TanimotoCombo [0.0 - 2.0]", fontsize=10)
        ax1.set_ylim(0.0, 2.05)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(fontsize=8, loc="lower right")

        for (title, df), col in zip(results_by_variant.items(), colors):
            slice_df = cast(pd.DataFrame, df[df["k_gen_requested"] == 30]).sort_values(
                by="k_ref_requested"
            )
            ax2.plot(
                slice_df["k_ref_requested"],
                slice_df["tanimoto_combo"],
                marker="s",
                linewidth=2.0,
                color=col,
                label=title.split(" (")[0],
            )

        ax2.axhline(
            2.0, color="gray", linestyle="--", alpha=0.6, label="Ideal Self-Match (2.0)"
        )
        ax2.set_title(
            r"B: Reference Basin Density Impact at Fixed $K_{\mathrm{gen}}=30$",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel(r"Number of Reference Basins ($K_{\mathrm{ref}}$)", fontsize=10)
        ax2.set_ylabel("ROCS TanimotoCombo [0.0 - 2.0]", fontsize=10)
        ax2.set_ylim(0.0, 2.05)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=8, loc="lower right")

        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_file

    @staticmethod
    def render_expansion_profiles(
        expansion_data: Sequence[Mapping[str, object]], output_path: Path | str
    ) -> Path:
        """Render heavy-atom Radius of Gyration (Rg) boxplots illustrating vacuum collapse vs solvent expansion.

        Args:
            expansion_data: Sequence of records holding 'variant' and 'radius_of_gyration'.
            output_path: Destination PNG image path.

        Returns:
            Resolved Path to the saved profile figure.
        """
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        df_exp = pd.DataFrame(expansion_data)

        variants = df_exp["variant"].unique()
        colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]

        data_to_plot = [
            np.asarray(df_exp[df_exp["variant"] == v]["radius_of_gyration"])
            for v in variants
        ]

        bplot = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            tick_labels=[str(v).replace("MMFF94s_", "") for v in variants],
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, col in zip(bplot["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)

        ax.set_title(
            "Conformational Expansion Profile: Vacuum Collapse vs. Solvent Shielding",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(
            r"Heavy Atom Radius of Gyration ($R_g$, $\mathrm{\AA}$)", fontsize=10
        )
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        plt.xticks(rotation=20, ha="right", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_file
