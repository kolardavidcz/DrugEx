"""Four requested ROCS/time sheets and Pareto diagnostics from saved measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

from _david.code.conformer.reference_bank import LABELS, METHODS
from _david.code.conformer.reward_calibration import CAPS, CANDIDATES, KEYS
from _david.code.conformer.self_recovery import atomic_json, digest


def heatmaps(summary: pd.DataFrame, output: Path, warning=""):
    """Identical grid/colour scales across methods; explicit actual all-bank counts."""
    sheets = [
        ("mean", "mean_seconds", "Mean ROCS / mean scoring time", "mean"),
        ("median", "median_seconds", "Median ROCS / median scoring time", "median"),
        (
            "p05",
            "p95_seconds",
            "Fifth-percentile ROCS / 95th-percentile scoring time",
            "p05",
        ),
        (
            "fraction_ge_095",
            "p95_seconds",
            "Fraction reaching ROCS 0.95 / 95th-percentile scoring time",
            "success",
        ),
    ]
    for score_key, time_key, title, filename in sheets:
        fig, axes = plt.subplots(4, 2, figsize=(18, 23), constrained_layout=True)
        vmax = max(float(summary[time_key].max()), 0.001)
        vmin = max(float(summary.loc[summary[time_key].gt(0), time_key].min()), 0.001)
        for row, method in enumerate(METHODS):
            data = summary[summary.method.eq(method)].set_index(
                ["reference_cap", "candidate_attempts"]
            )
            all_bank = data.loc[(0, CANDIDATES[0])]
            bank_label = f"all-bank median {all_bank.median_references:g}; max {all_bank.max_references:g}"
            for col, key in enumerate((score_key, time_key)):
                ax = axes[row, col]
                values = np.array(
                    [
                        [data.loc[(cap, count), key] for cap in CAPS]
                        for count in CANDIDATES
                    ]
                )
                if col == 0:
                    if score_key in ("mean", "median"):
                        lower = min(
                            0.94, float(np.floor(summary[score_key].min() * 100) / 100)
                        )
                        image = ax.imshow(
                            values,
                            aspect="auto",
                            cmap="RdYlGn",
                            norm=TwoSlopeNorm(vmin=lower, vcenter=0.95, vmax=1.0001),
                        )
                    else:
                        image = ax.imshow(
                            values, aspect="auto", cmap="viridis", vmin=0, vmax=1
                        )
                else:
                    image = ax.imshow(
                        np.maximum(values, 0.001),
                        aspect="auto",
                        cmap="magma_r",
                        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01)),
                    )
                ax.set_xticks(range(6), ["1", "2", "4", "8", "12", "all"])
                ax.set_yticks(range(6), [f"C{count}" for count in CANDIDATES])
                ax.set_xlabel("Total reference-structure cap (at most)")
                ax.set_ylabel("Candidate requests per stereoisomer")
                ax.set_title(
                    f"{LABELS[method]} · {'ROCS' if col == 0 else 'seconds'}\n{bank_label}",
                    fontsize=12,
                )
                for i, j in np.ndindex(values.shape):
                    value = values[i, j]
                    rgba = image.cmap(image.norm(max(value, 0.001) if col else value))
                    light = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    label = (
                        (
                            f"{value:.0%}"
                            if score_key == "fraction_ge_095"
                            else f"{value:.4f}"
                        )
                        if col == 0
                        else f"{value:.2f}"
                    )
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black" if light > 0.48 else "white",
                    )
                fig.colorbar(image, ax=ax, shrink=0.85)
        n, scored = int(summary.inputs.iloc[0]), int(summary.scored_inputs.iloc[0])
        fig.suptitle(
            f"{warning}{title}\n{n} frozen development inputs · {scored} native-colour-eligible · preparation/scoring failures retained\n"
            "R50/I8 · candidate I4 · reference-only energy/diversity selection · no RMSD or pH reassignment\n"
            "Fresh public scoring calls, one worker per input; shared host · reference setup excluded · not RL epoch time",
            fontsize=15,
        )
        fig.savefig(output / f"{filename}.png", dpi=170)
        fig.savefig(output / f"{filename}.svg")
        plt.close(fig)


def pareto(frame, summary, output, warning=""):
    """Descriptive frontier and signed gaps to each method's C75/all anchor."""
    dominated = []
    for _, row in summary.iterrows():
        other = (
            summary.mean_seconds.le(row.mean_seconds)
            & summary["mean"].ge(row["mean"])
            & (
                summary.mean_seconds.lt(row.mean_seconds)
                | summary["mean"].gt(row["mean"])
            )
        )
        dominated.append(bool(other.any()))
    summary = summary.assign(pareto=~np.array(dominated))
    summary.to_csv(output / "raw_data/development/pareto.csv", index=False)
    anchor = frame[frame.reference_cap.eq(0) & frame.candidate_attempts.eq(75)][
        ["molecule_id", "method", "score"]
    ]
    paired = frame.merge(
        anchor.rename(columns={"score": "anchor_score"}),
        on=["molecule_id", "method"],
        validate="many_to_one",
    )
    paired["score_gap"] = paired.anchor_score - paired.score
    paired.to_csv(output / "raw_data/development/paired_anchor.csv.gz", index=False)
    gaps = (
        paired[~paired.color_excluded]
        .groupby(KEYS[1:])
        .score_gap.agg(mean_gap="mean", p95_gap=lambda s: s.quantile(0.95))
        .reset_index()
    )
    summary = summary.merge(gaps, on=KEYS[1:])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for method, group in summary.groupby("method"):
        points = axes[0].scatter(
            group.mean_seconds, group["mean"], s=30, alpha=0.65, label=LABELS[method]
        )
        color = points.get_facecolors()[0]
        front = group[group.pareto].sort_values("mean_seconds")
        axes[0].scatter(
            front.mean_seconds,
            front["mean"],
            s=100,
            facecolors="none",
            edgecolors=[color],
            linewidths=2,
        )
        for _, row in front.iterrows():
            cap = "all" if row.reference_cap == 0 else str(int(row.reference_cap))
            axes[0].annotate(
                f"K{cap}/C{int(row.candidate_attempts)}",
                (row.mean_seconds, row["mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].scatter(
            group.mean_seconds,
            group.p95_gap,
            s=30,
            color=color,
            alpha=0.65,
            label=LABELS[method],
        )
    axes[0].axhline(
        0.95,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Chosen mean target 0.95",
    )
    axes[0].set_ylabel("Mean self-recovery ROCS")
    axes[0].set_title("Observed score–time frontier; circles mark nondominated cells")
    axes[1].set_ylabel("95th-percentile signed score gap to same-method C75/all")
    axes[1].set_title("Large-score-loss tail relative to a finite sampled anchor")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Mean measured per-input scoring seconds")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        f"{warning}Development-panel trade-offs · lower time and lower score gap preferred\n"
        "Reference/candidate methods matched; this is not biological accuracy or RL improvement",
        fontsize=14,
    )
    fig.savefig(output / "pareto.png", dpi=170)
    fig.savefig(output / "pareto.svg")
    plt.close(fig)


def render_rl_timing(output):
    raw = output / "raw_data/rl_egcg"
    path = raw / "manifest.json"
    if not path.exists() or json.loads(path.read_text())["state"] != "complete":
        return
    epochs = pd.read_csv(raw / "epoch_timings.csv")
    replay = pd.read_csv(raw / "batch_replays.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    stages = [
        (
            "Generation + minimization",
            epochs.generation_minimization_seconds,
            "#31688e",
        ),
        (
            "Alignment + scorer I/O/worker overhead",
            epochs.rocs_seconds - epochs.generation_minimization_seconds,
            "#35b779",
        ),
        (
            "Other environment scoring",
            epochs.environment_seconds - epochs.rocs_seconds,
            "#d5812b",
        ),
        (
            "RNN + optimizer + checkpoint/capture work",
            epochs.epoch_seconds - epochs.environment_seconds,
            "#8c6bb1",
        ),
    ]
    bottom = np.zeros(len(epochs))
    for label, values, color in stages:
        left.bar(epochs.epoch, values, bottom=bottom, label=label, color=color)
        bottom += values
    left.set_xlabel(
        "Actual RL epoch (200 requested training samples + native validation)"
    )
    left.set_ylabel("Wall seconds")
    left.set_title("Measured epoch cost; reference/model setup excluded")
    left.legend(fontsize=9)
    labels = []
    for _, row in replay.iterrows():
        cap = "all" if row.reference_cap == 0 else str(int(row.reference_cap))
        labels.append(
            f"{LABELS[row.method]}\nK{cap}/C{int(row.candidate_attempts)}\n{row.engine}, pass{int(row['repeat']) + 1}"
        )
    per_input = replay.seconds / replay.inputs
    bars = right.bar(range(len(replay)), per_input, color="#31688e")
    right.bar_label(
        bars,
        labels=[
            f"{time:.2f}s\nΔ {delta:+.3f}"
            for time, delta in zip(per_input, replay.mean_change)
        ],
        padding=4,
        fontsize=8,
    )
    right.set_xticks(range(len(replay)), labels, rotation=35, ha="right", fontsize=8)
    right.set_ylabel("Measured seconds per input row of the same frozen batch")
    right.set_title("Same-first-batch replays; score changes are reported")
    right.set_ylim(0, per_input.max() * 1.3)
    fig.suptitle(
        "Real five-epoch EGCG cost illustration · not an RL learning comparison\n"
        "Baseline MMFF94s ε78.4, K4/C20/I4 · no RMSD/pH reassignment · model seed42\n"
        "Whole-call prototype: fresh per-graph defaults and persistent workers; native batches can inherit first-graph iteration budget",
        fontsize=13,
    )
    fig.savefig(output / "rl_timing.png", dpi=170)
    fig.savefig(output / "rl_timing.svg")
    plt.close(fig)


def render(output):
    raw = output / "raw_data/development"
    summary = pd.read_csv(raw / "summary.csv")
    frame = pd.read_csv(raw / "data.csv.gz")
    status = json.loads((raw / "manifest.json").read_text()).get(
        "scientific_status", ""
    )
    warning = (
        "SUPERSEDED: reference/candidate RNG streams overlap — not valid for recommendations\n"
        if status == "superseded_seed_overlap"
        else ""
    )
    heatmaps(summary, output, warning)
    pareto(frame, summary, output, warning)
    render_rl_timing(output)
    comparison = output / "raw_data/confirmation/summary.csv"
    if comparison.exists():
        confirmed = pd.read_csv(comparison).merge(
            summary,
            on=KEYS[1:],
            suffixes=("_confirmation", "_development"),
            validate="one_to_one",
        )
        confirmed.to_csv(
            output / "raw_data/confirmation/development_comparison.csv", index=False
        )
        passing = confirmed[confirmed.mean_confirmation.ge(0.95)]
        preferred = (
            passing.sort_values("mean_seconds_confirmation").iloc[0]
            if len(passing)
            else confirmed.sort_values("mean_confirmation", ascending=False).iloc[0]
        )
        recommendation = {
            key: preferred[key].item()
            if hasattr(preferred[key], "item")
            else preferred[key]
            for key in KEYS[1:]
        }
        recommendation.update(
            target_met=bool(preferred.mean_confirmation >= 0.95),
            measured_mean=float(preferred.mean_confirmation),
            measured_seconds=float(preferred.mean_seconds_confirmation),
            scope="Selected among the predeclared development shortlist after confirmation. Finite self-recovery/cost study, not a universal RL optimum; I4 not tuned.",
        )
        atomic_json(output / "raw_data/recommended_settings.json", recommendation)
    path = raw / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rendering"] = dict(
        source_sha256=digest(__file__),
        input_summary_sha256=digest(raw / "summary.csv"),
        files_sha256={
            p.name: digest(p) for p in output.iterdir() if p.suffix in {".png", ".svg"}
        },
    )
    atomic_json(path, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)
