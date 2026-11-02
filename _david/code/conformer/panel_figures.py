"""Regenerate the 128-drug stratified reference-cap figure from saved tables.

Promoted 2026-11-01 from a one-off SSH heredoc (recovered from the session
archive that produced the original figure on 2026-10-29). The original four
panels were:

- A. 128-drug panel: fewer minimized conformers beat more unminimized ones
  (mean scoring latency vs mean ROCS self-recovery).
- B. Mean self-recovery stratified by enumerated reference stereoisomer count.
- C. Single-isomer drugs stratified by the surviving 8 kcal/mol reference pool
  (PES steepness proxy).
- D. Boxplot of actual retained references under fixed caps ``K`` versus
  tail-trimming ``all - m`` — REMOVED 2026-11-01 by user decision: not
  applicable to real training, where the reference bank is frozen once by the
  pipeline and never trimmed per epoch.

Panel A keeps the calibration timing on the x-axis but now draws per-input
p95 whiskers and a banner with the measured live-RL reference, because the
isolated serial single-molecule calibration seconds underestimate the real
RL per-row cost by roughly 3x (see ``visualization.rl_reference_measurement``
and the 2026-11-01 entry in ``results.md``).

Usage from the repository root:

.. code-block:: bash

    MPLBACKEND=Agg python -m _david.code.conformer.panel_figures \
        --calibration _david/output/20261029_1155_reward_calibration \
        --output _david/output/20261029_egcg_omnibus_diagnostic
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _david.code.conformer.reference_bank import LABELS  # noqa: E402
from _david.code.conformer.utils import atomic_json, digest, source_hashes  # noqa: E402
from _david.code.conformer.visualization import (  # noqa: E402
    FALLBACK_RL_REFERENCE,
    rl_reference_measurement,
)

FIGURE_NAME = "drug_panel_stratified_vs_trimmed.png"
STATE_FIGURE_NAME = "calibration_state_overview.png"
MANIFEST_NAME = "drug_panel_figure_manifest.json"
SOURCES: tuple[str, ...] = (
    "_david/code/conformer/panel_figures.py",
    "_david/code/conformer/visualization.py",
    "_david/code/conformer/reference_bank.py",
)

METHOD_ORDER: tuple[str, ...] = (
    "mmff_high_dielectric",
    "mmff_vacuum",
    "uff",
    "etkdg",
)
METHOD_COLORS: dict[str, tuple[str, str]] = {
    "mmff_high_dielectric": ("MMFF94s ε=78.4 (Minimized)", "#1b9e77", "o"),
    "mmff_vacuum": ("MMFF94s ε=1.0 (Minimized)", "#d95f02", "s"),
    "uff": ("UFF (Minimized)", "#7570b3", "^"),
    "etkdg": ("ETKDG Only (Unminimized)", "#e7298a", "D"),
}
ISO_GROUPS: tuple[str, ...] = (
    "I = 1 (Achiral / Specified)",
    "I = 2 (1 Unassigned Center)",
    "I = 3–4 (2 Centers)",
    "I = 5–8 (≥3 Centers)",
)


def load_development(calibration: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frozen development-panel tables from a reward calibration run."""
    raw = calibration / "raw_data" / "development"
    frame = pd.read_csv(raw / "data.csv.gz")
    summary = pd.read_csv(raw / "summary.csv")
    required = {
        "molecule_id",
        "method",
        "reference_cap",
        "candidate_attempts",
        "color_excluded",
        "reference_failure",
        "enumerated_reference_isomers",
        "actual_references",
        "score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Development table is missing columns: {sorted(missing)}")
    missing = {"mean_seconds", "p95_seconds", "mean"} - set(summary.columns)
    if missing:
        raise ValueError(f"Development summary is missing columns: {sorted(missing)}")
    return frame, summary


def _rl_banner(calibration: Path) -> str:
    reference = rl_reference_measurement(calibration) or FALLBACK_RL_REFERENCE
    factor = reference.get("factor")
    return (
        f"Measured live-RL reference: {reference['provenance']}, "
        f"{reference['serial_equivalent_seconds_per_row']:.2f} s/row serial-equivalent"
        + (f" (~{factor:.1f}x these calibration seconds)" if factor else "")
        + " — calibration x-axis is NOT RL epoch time"
    )


def render_drug_panel_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    *,
    rl_note: str,
) -> Path:
    """Render the three-panel 128-drug stratified figure (panel D removed)."""
    valid = frame[~frame.color_excluded & ~frame.reference_failure].copy()
    valid["iso_group"] = pd.cut(
        valid["enumerated_reference_isomers"],
        bins=[0, 1, 2, 4, 8],
        labels=list(ISO_GROUPS),
    )
    total_bank = (
        valid[valid.reference_cap.eq(0)]
        .groupby(["method", "molecule_id"])["actual_references"]
        .first()
        .rename("total_bank_refs")
        .reset_index()
    )
    valid = valid.merge(total_bank, on=["method", "molecule_id"], how="left")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=180)
    panel_a, panel_b, panel_c = axes

    # Panel A: fewer minimized conformers vs more unminimized conformers,
    # with per-input p95 latency whiskers (straggler-aware).
    for method in METHOD_ORDER:
        label, color, marker = METHOD_COLORS[method]
        cells = summary[summary.method.eq(method)]
        if cells.empty:
            continue
        panel_a.errorbar(
            cells.mean_seconds,
            cells["mean"],
            xerr=(cells.p95_seconds - cells.mean_seconds).clip(lower=0),
            fmt="none",
            ecolor=color,
            alpha=0.35,
            capsize=1.5,
            linewidth=1,
        )
        panel_a.scatter(
            cells.mean_seconds,
            cells["mean"],
            color=color,
            marker=marker,
            s=38,
            alpha=0.85,
            label=label,
            zorder=3,
        )
        trajectory = cells[cells.reference_cap.eq(4)].sort_values("candidate_attempts")
        panel_a.plot(
            trajectory.mean_seconds,
            trajectory["mean"],
            color=color,
            linewidth=1.6,
            linestyle="-",
            alpha=0.8,
        )
        for row in cells[
            cells.reference_cap.isin([4, 0]) & cells.candidate_attempts.isin([10, 20, 50, 75])
        ].itertuples():
            if method in ("mmff_high_dielectric", "etkdg") and (
                (row.reference_cap == 4 and row.candidate_attempts in (10, 20, 75))
                or (row.reference_cap == 0 and row.candidate_attempts in (10, 50))
            ):
                cap_str = "all" if row.reference_cap == 0 else f"K{int(row.reference_cap)}"
                panel_a.annotate(
                    f"{cap_str}/C{int(row.candidate_attempts)}",
                    (row.mean_seconds, row.mean),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7.5,
                    color=color,
                    weight="bold",
                )
    panel_a.set_xscale("log")
    panel_a.set_title(
        "A. 128-Drug Panel: Fewer Minimized Conformers Beat More Unminimized\n"
        "(whiskers: per-input p95 latency)"
    )
    panel_a.set_xlabel(
        f"Mean per-input scoring seconds (isolated serial calls, log scale)\n{rl_note}",
        fontsize=9,
    )
    panel_a.set_ylabel("Mean ROCS Self-Recovery")
    panel_a.grid(True, which="both", alpha=0.25)
    panel_a.legend(fontsize=8, loc="lower right")

    # Panel B: mean self-recovery stratified by enumerated reference
    # stereoisomer count (candidate stereo cap I=4, C=20).
    sub_iso = valid[
        valid.method.eq("mmff_high_dielectric") & valid.candidate_attempts.eq(20)
    ].copy()
    sub_iso["k_order"] = sub_iso.reference_cap.replace({0: 16})
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for (group, group_frame), color in zip(
        sub_iso.groupby("iso_group", observed=True), palette
    ):
        molecules = group_frame.molecule_id.nunique()
        aggregate = group_frame.groupby("k_order").score.mean().reset_index()
        panel_b.plot(
            aggregate.k_order,
            aggregate.score,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{group} (n={molecules})",
        )
    panel_b.set_xticks([1, 2, 4, 8, 12, 16])
    panel_b.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"])
    panel_b.set_title(
        "B. 128-Drug Panel Stratified by Enumerated Reference Stereoisomers (C=20)"
    )
    panel_b.set_xlabel("Reference Cap K (Round-Robin Balanced Across Stereoisomers)")
    panel_b.set_ylabel("Mean ROCS Self-Recovery")
    panel_b.grid(True, alpha=0.25)
    panel_b.legend(fontsize=8, loc="lower right")

    # Panel C: single-isomer drugs stratified by the surviving reference pool
    # (proxy for potential-energy-surface steepness / flexibility).
    iso1 = valid[
        valid.enumerated_reference_isomers.eq(1)
        & valid.method.eq("mmff_high_dielectric")
        & valid.candidate_attempts.eq(20)
    ].copy()
    iso1["flex_group"] = pd.cut(
        iso1.total_bank_refs,
        bins=[0, 25, 45, 55],
        labels=[
            "Steep PES / Flexible (≤25 survivors/50, ΔE>8 filtered)",
            "Moderate PES (26–45 survivors/50)",
            "Flat PES / Rigid (46–50 survivors/50)",
        ],
    )
    iso1["k_order"] = iso1.reference_cap.replace({0: 16})
    for (group, group_frame), color in zip(
        iso1.groupby("flex_group", observed=True),
        ["#e7298a", "#d95f02", "#1b9e77"],
    ):
        molecules = group_frame.molecule_id.nunique()
        aggregate = group_frame.groupby("k_order").score.mean().reset_index()
        panel_c.plot(
            aggregate.k_order,
            aggregate.score,
            marker="s",
            linewidth=2.0,
            color=color,
            label=f"{group} (n={molecules})",
        )
    panel_c.set_xticks([1, 2, 4, 8, 12, 16])
    panel_c.set_xticklabels(["K=1", "K=2", "K=4", "K=8", "K=12", "K=all"])
    panel_c.set_title("C. Single-Isomer Drugs Stratified by Surviving 8 kcal/mol Reference Count")
    panel_c.set_xlabel("Reference Cap K")
    panel_c.set_ylabel("Mean ROCS Self-Recovery")
    panel_c.grid(True, alpha=0.25)
    panel_c.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "128-Drug Panel: Stratified Reference-Cap Response (panel D removed 2026-11-01)",
        fontsize=13,
    )
    fig.tight_layout()
    target = output / FIGURE_NAME
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def write_figure_manifest(output: Path, calibration: Path, rl_note: str) -> Path:
    """Write the provenance manifest for the regenerated figure."""
    raw = calibration / "raw_data" / "development"
    target = output / MANIFEST_NAME
    atomic_json(
        target,
        dict(
            figure=str(output / FIGURE_NAME),
            generated_utc=datetime.now(timezone.utc).isoformat(),
            panels=["A. time vs recovery", "B. stereoisomer strata", "C. PES strata"],
            removed_panels=[
                "D. fixed K vs (all - m) retained-reference boxplot "
                "(user decision 2026-11-01: not applicable to real training)"
            ],
            inputs={
                "data.csv.gz": digest(raw / "data.csv.gz"),
                "summary.csv": digest(raw / "summary.csv"),
            },
            source_sha256=source_hashes(SOURCES),
            rl_note=rl_note,
        ),
    )
    return target


def render_calibration_state(
    dev: pd.DataFrame,
    confirmation: pd.DataFrame,
    output: Path,
    *,
    rl_note: str,
) -> Path:
    """Render the current calibration state: coverage, frontier, validation.

    Panel A: measured development grid per method (mean ROCS per K x C cell,
    confirmation-validated shortlist cells outlined).
    Panel B: score vs p95-time frontier with shortlist cells starred.
    Panel C: development vs confirmation mean ROCS on the shortlist cells.
    """
    cap_order = [1, 2, 4, 8, 12, 0]
    cap_labels = {cap: ("all" if cap == 0 else f"K{cap}") for cap in cap_order}
    shortlist = set(
        zip(confirmation.method, confirmation.reference_cap, confirmation.candidate_attempts)
    )
    fig = plt.figure(figsize=(22, 8), dpi=180)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.9], wspace=0.28)
    grid_a = outer[0, 0].subgridspec(2, 2, hspace=0.55, wspace=0.45)

    heatmap_axes = []
    for index, method in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(grid_a[index])
        heatmap_axes.append(ax)
        cells = dev[dev.method.eq(method)]
        matrix = cells.pivot_table(
            index="reference_cap", columns="candidate_attempts", values="mean"
        ).reindex(cap_order)
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="RdYlGn",
            vmin=0.75,
            vmax=1.0,
        )
        attempts = sorted(cells.candidate_attempts.unique())
        for row, cap in enumerate(cap_order):
            for col, count in enumerate(attempts):
                value = matrix.loc[cap, count]
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black" if value > 0.87 else "white",
                )
                if (method, cap, count) in shortlist:
                    ax.add_patch(
                        plt.Rectangle(
                            (col - 0.5, row - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#1a1a1a",
                            linewidth=2.4,
                        )
                    )
        ax.set_xticks(range(len(attempts)), [f"C{c}" for c in attempts], fontsize=8)
        ax.set_yticks(range(len(cap_order)), [cap_labels[c] for c in cap_order], fontsize=8)
        ax.set_title(f"{METHOD_COLORS[method][0]}\n(dev n={int(cells.inputs.iloc[0])}, 36 cells)", fontsize=9)
    fig.colorbar(
        image,
        ax=heatmap_axes,
        shrink=0.85,
        label="Mean self-recovery ROCS",
    )

    # Panel B: p95 time vs mean score frontier, shortlist starred.
    ax_b = fig.add_subplot(outer[0, 1])
    for method in METHOD_ORDER:
        label, color, marker = METHOD_COLORS[method]
        cells = dev[dev.method.eq(method)]
        ax_b.scatter(cells.p95_seconds, cells["mean"], s=22, alpha=0.55, color=color, label=label)
        for row in cells.itertuples():
            if (row.method, row.reference_cap, row.candidate_attempts) in shortlist:
                ax_b.scatter(
                    row.p95_seconds, row.mean, s=230, marker="*", color=color,
                    edgecolors="black", linewidths=0.8, zorder=4,
                )
                ax_b.annotate(
                    f"K{'all' if row.reference_cap == 0 else int(row.reference_cap)}/C{int(row.candidate_attempts)}",
                    (row.p95_seconds, row.mean),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
    ax_b.set_xscale("log")
    ax_b.axhline(0.95, color="black", linestyle="--", linewidth=1, label="Target mean 0.95")
    ax_b.set_xlabel(f"Per-input scoring seconds (p95, isolated serial calls)\n{rl_note}", fontsize=9)
    ax_b.set_ylabel("Mean ROCS self-recovery")
    ax_b.set_title("B. Score-time frontier (dev 128)\nstars = confirmation-validated shortlist")
    ax_b.grid(True, which="both", alpha=0.25)
    ax_b.legend(fontsize=8, loc="lower right")

    # Panel C: development vs confirmation agreement on the shortlist.
    ax_c = fig.add_subplot(outer[0, 2])
    merged = dev.merge(
        confirmation,
        on=["method", "reference_cap", "candidate_attempts"],
        suffixes=("_dev", "_conf"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        label, color, _ = METHOD_COLORS[row.method]
        ax_c.scatter(row.mean_dev, row.mean_conf, s=90, color=color, zorder=3)
        ax_c.annotate(
            f"K{'all' if row.reference_cap == 0 else int(row.reference_cap)}/C{int(row.candidate_attempts)}",
            (row.mean_dev, row.mean_conf),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    low = min(merged.mean_dev.min(), merged.mean_conf.min()) - 0.02
    high = max(merged.mean_dev.max(), merged.mean_conf.max()) + 0.02
    ax_c.plot([low, high], [low, high], "k--", linewidth=1, label="y = x")
    ax_c.set_xlabel("Development mean ROCS (128 inputs)")
    ax_c.set_ylabel("Confirmation mean ROCS (384 inputs)")
    ax_c.set_title("C. Shortlist validation: dev vs confirmation")
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(fontsize=8, loc="upper left")

    total_cells = len(dev) + len(confirmation)
    fig.suptitle(
        "Current ROCS self-recovery calibration state: 128 development x 144 cells + 384 confirmation x 6 shortlist cells "
        f"({total_cells} measured molecule-cells, 4 methods, K x C grid)\n"
        "Scores = self-recovery (each drug vs its own reference ensemble); timing = isolated serial single-molecule calls",
        fontsize=13,
    )
    target = output / STATE_FIGURE_NAME
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Reward-calibration output directory with raw_data/development tables",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory that receives the figure and manifest",
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Render the calibration-state overview instead of the drug-panel figure",
    )
    args = parser.parse_args()
    frame, summary = load_development(args.calibration)
    args.output.mkdir(parents=True, exist_ok=True)
    rl_note = _rl_banner(args.calibration)
    if args.state:
        confirmation = pd.read_csv(
            args.calibration / "raw_data" / "confirmation" / "summary.csv"
        )
        target = render_calibration_state(
            summary, confirmation, args.output, rl_note=rl_note
        )
        print(f"Saved {target}")
        return
    target = render_drug_panel_figure(frame, summary, args.output, rl_note=rl_note)
    manifest = write_figure_manifest(args.output, args.calibration, rl_note)
    print(f"Saved {target}")
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
