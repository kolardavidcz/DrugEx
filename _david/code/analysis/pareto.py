"""Multi-Objective Pareto Frontier Analysis and Non-Dominated Ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

# [PACKAGE: matplotlib] WHY: Publication-grade Pareto frontier scatter plots. WHAT: Plotting library. HOW: plt.subplots, fig.savefig.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# [PACKAGE: numpy] WHY: Vectorized dominance comparisons and boolean masking. WHAT: Numerical computing library. HOW: np.asarray, np.all, np.any.
import numpy as np
# [PACKAGE: pandas] WHY: Tabular candidate filtering and ranking by Pareto optimality. WHAT: Data manipulation library. HOW: pd.DataFrame.
import pandas as pd


class ParetoFrontier:
    """Computes Pareto dominance ranks and visualizes non-dominated frontiers."""

    @staticmethod
    def identify_pareto_front(
        costs: np.ndarray,
        maximize: Sequence[bool] | None = None,
    ) -> np.ndarray:
        """Find non-dominated Pareto efficient points.

        Args:
            costs: Array of shape (N, M) containing M objective scores for N samples.
            maximize: Sequence of booleans indicating whether each objective should be maximized (True) or minimized (False).

        Returns:
            Boolean mask of shape (N,) where True marks non-dominated points (Front 0).
        """
        n_samples = costs.shape[0]
        if n_samples == 0:
            return np.zeros(0, dtype=bool)

        arr = np.asarray(costs, dtype=np.float64)
        if maximize is not None:
            # Transform to minimization problem: flip sign for maximized objectives
            arr = arr.copy()
            for col_idx, is_max in enumerate(maximize):
                if is_max:
                    arr[:, col_idx] = -arr[:, col_idx]

        is_efficient = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            if not is_efficient[i]:
                continue
            # Point j strictly dominates point i if arr[j] <= arr[i] in all dims and < in at least one dim
            dominates_i = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
            if np.any(dominates_i):
                is_efficient[i] = False
            else:
                # Point i strictly dominates any point j if arr[i] <= arr[j] and < in at least one dim
                dominated_by_i = np.all(arr[i] <= arr, axis=1) & np.any(arr[i] < arr, axis=1)
                is_efficient[dominated_by_i] = False

        return is_efficient

    def rank_pareto_fronts(
        self,
        df: pd.DataFrame,
        objectives: Sequence[str] = ("ROCS", "SA"),
        maximize: Sequence[bool] = (True, False),
    ) -> pd.DataFrame:
        """Calculate non-dominated sorting Pareto ranks for candidates in df.

        Rank 0 represents the first non-dominated frontier, Rank 1 the next, etc.

        Args:
            df: DataFrame containing candidate molecules and evaluated metrics.
            objectives: Column names of objectives to optimize simultaneously.
            maximize: Booleans for each objective: True if higher is better, False if lower is better.

        Returns:
            DataFrame with added 'Pareto_Rank' and 'Pareto_Optimal' columns.
        """
        df_out = df.copy()
        if df_out.empty:
            df_out["Pareto_Rank"] = 0
            df_out["Pareto_Optimal"] = False
            return df_out

        valid_mask = df_out.get("Valid", pd.Series(True, index=df_out.index)).values.astype(bool)
        # Check that all requested objective columns exist
        avail_objs = [obj for obj in objectives if obj in df_out.columns]
        if not avail_objs:
            df_out["Pareto_Rank"] = 0
            df_out["Pareto_Optimal"] = False
            return df_out

        costs = df_out.loc[valid_mask, avail_objs].values
        eff_mask = self.identify_pareto_front(costs, maximize=maximize[:len(avail_objs)])

        pareto_optimal = np.zeros(len(df_out), dtype=bool)
        valid_indices = df_out.index[valid_mask]
        pareto_optimal[valid_indices[eff_mask]] = True

        df_out["Pareto_Optimal"] = pareto_optimal
        df_out["Pareto_Rank"] = np.where(pareto_optimal, 0, 1)
        return df_out

    def plot_pareto_frontier(
        self,
        df: pd.DataFrame,
        x_col: str = "ROCS",
        y_col: str = "SA",
        output_path: Path | str | None = None,
        x_label: str = "3D Shape & Color Overlap (ROCS TanimotoCombo)",
        y_label: str = "Synthetic Accessibility (SAscore, lower is better)",
    ) -> Path | None:
        """Plot scatter of candidates with highlighted non-dominated Pareto frontier."""
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return None

        valid_df = df[df.get("Valid", True)].copy()
        if valid_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # Plot all candidates
        ax.scatter(
            valid_df[x_col],
            valid_df[y_col],
            c="#8D99AE",
            alpha=0.6,
            s=45,
            edgecolors="none",
            label="Generated Candidates",
        )

        # Highlight Pareto frontier
        pareto_mask = valid_df.get("Pareto_Optimal", pd.Series(False, index=valid_df.index)).values
        pareto_df = valid_df[pareto_mask].sort_values(by=x_col)

        if not pareto_df.empty:
            ax.scatter(
                pareto_df[x_col],
                pareto_df[y_col],
                c="#E63946",
                s=110,
                edgecolors="#1D3557",
                linewidths=1.5,
                zorder=5,
                label=f"Pareto Frontier (n={len(pareto_df)})",
            )
            # Step line along Pareto points
            ax.step(
                pareto_df[x_col],
                pareto_df[y_col],
                where="post",
                color="#E63946",
                linestyle="--",
                linewidth=1.8,
                alpha=0.8,
                zorder=4,
            )

        ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        ax.set_title("Multi-Objective Non-Dominated Pareto Frontier", fontsize=14, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", frameon=True, framealpha=0.9)

        plt.tight_layout()

        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            return p

        plt.close(fig)
        return None
