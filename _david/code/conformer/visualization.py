"""Conformer-study figures; RL analysis remains in code.visualization."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_reference_counts(frame: pd.DataFrame, output: Path) -> None:
    """Counts through the reference filters, with zeros and long tails visible."""
    stages = [("converged_valid_reported", "Converged + identity valid", "#31688e"),
              ("energy_survivors", "After energy window", "#35b779"),
              ("retained_rmsd_05", "After 0.5 Å RMSD", "#d5812b")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    bins = [-.5, .5, 1.5, 3.5, 7.5, 15.5, 31.5, 63.5, 127.5, 255.5, 511.5, 1024.5]
    for row, attempts in enumerate((50, 100)):
        subset = frame[frame.attempts.eq(attempts)]
        hist, ecdf = axes[row]
        for metric, label, color in stages:
            values = subset[metric].to_numpy()
            hist.hist(values, bins=bins, histtype="step", linewidth=2, color=color, label=label)
            ordered = np.sort(values)
            ecdf.step(ordered, np.arange(1, len(ordered)+1)/len(ordered),
                      where="post", linewidth=2, color=color, label=label)
        converged = subset.converged_valid_reported
        hist.text(.98, .95, f"All {len(subset)} inputs; zero outputs included\n"
                  f"Converged count: median {converged.median():.0f}, mean {converged.mean():.1f}\n"
                  f"95th percentile {converged.quantile(.95):.1f}; maximum {converged.max()}\n"
                  f"Zero converged outputs: {converged.eq(0).sum()}",
                  transform=hist.transAxes, ha="right", va="top", fontsize=10,
                  bbox=dict(facecolor="white", alpha=.88, edgecolor="none"))
        hist.set_ylabel("Number of input molecules")
        ecdf.set_ylabel("Fraction of all input molecules")
        ecdf.set_ylim(0, 1.02)
        for ax in (hist, ecdf):
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0, 1, 4, 16, 64, 256, 1024], ["0", "1", "4", "16", "64", "256", "1024"])
            ax.set_xlim(-.5, 1024)
            ax.set_xlabel("Conformers per input (summed across its stereoisomers)")
            ax.set_title(f"{attempts} requests per stereoisomer · reference stereo cap 8")
            ax.grid(alpha=.18)
        ecdf.legend(loc="lower right")
    fig.suptitle("Reference ensemble sizes in the completed 512-drug study\n"
                 "Historical preparation; embedding RMSD 0.35 Å · MMFF94s ε=78.4 · energy window 8 kcal/mol\n"
                 "Reported convergence precedes energy filtering; failed stereo preparations have incomplete intermediate counts",
                 fontsize=14)
    fig.savefig(output / "reference_counts.png", dpi=170)
    fig.savefig(output / "reference_counts.svg")
    plt.close(fig)


def render_batch_cost(frame: pd.DataFrame, output: Path) -> None:
    """Actual public-call wall times, including worker startup; two repeats."""
    configurations = [("native", 1, "Native: 1 worker"),
                      ("native", 8, "Native: 8 alignment workers"),
                      ("whole_call_pool", 8, "8 whole-call workers")]
    fig, (total, stages) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    positions = np.arange(3)
    for minimize, offset, color, label in [(False, -.18, "#31688e", "ETKDG, no minimization"),
                                           (True, .18, "#35b779", "ETKDG + MMFF94s ε78.4")]:
        groups = [frame[frame.engine.eq(engine) & frame.workers.eq(workers)
                        & frame.minimize.eq(minimize)].batch_seconds
                  for engine, workers, _ in configurations]
        heights = [group.median() for group in groups]
        bars = total.bar(positions + offset, heights, width=.34, color=color, label=label)
        total.bar_label(bars, fmt="%.1f s", padding=4)
        for x, group in zip(positions + offset, groups):
            total.scatter([x] * len(group), group, color="black", s=18, zorder=3)
    total.set_xticks(positions, [label.replace(": ", ":\n") for _, _, label in configurations])
    total.set_ylim(0, frame.batch_seconds.max() * 1.18)
    total.set_ylabel("Wall seconds for the complete 32-input call")
    total.set_title("Measured batch cost; dots show individual repeats")
    total.legend(loc="upper right", fontsize=9)
    native = frame[frame.engine.eq("native") & frame.workers.eq(8)]
    generation = native.groupby("minimize").generation_and_minimization_seconds.median().reindex([False, True])
    alignment = native.groupby("minimize").alignment_io_pool_seconds.median().reindex([False, True])
    stages.bar([0, 1], generation, color="#31688e", label="Serial generation + optional minimization")
    stages.bar([0, 1], alignment, bottom=generation, color="#d5812b", label="Alignment + I/O + worker startup")
    stages.set_xticks([0, 1], ["No minimization", "MMFF94s ε78.4"])
    stages.set_ylabel("Wall seconds")
    stages.set_title("Why parallelizing only alignment leaves a bottleneck")
    stages.legend(fontsize=9)
    fig.suptitle("RL scoring cost diagnostic · 32 frozen drug inputs · three fixed EGCG references\n"
                 "C20/I4 · no RMSD pruning · no pH reassignment · shared lab host\n"
                 "Fresh public scoring calls; no RNN training; scores checked for serial/parallel parity", fontsize=13)
    fig.savefig(output / "batch_cost.png", dpi=170)
    fig.savefig(output / "batch_cost.svg")
    plt.close(fig)


def render_joint_calibration(summary: pd.DataFrame, output: Path) -> None:
    """One readable score/time sheet for the 36-cell joint calibration."""
    reference_keys = [(n, r) for n in (50, 100) for r in (0.5, 1.25)]
    candidate_keys = [(n, r) for n in (10, 20, 40) for r in (0.0, 0.5, 1.25)]
    lookup = summary.set_index([
        "reference_attempts", "reference_rmsd", "candidate_attempts", "candidate_rmsd"
    ])
    fig, axes = plt.subplots(3, 2, figsize=(20, 14), constrained_layout=True)
    metrics = [("mean", "Mean native normalized ROCS", "viridis", (0.8, 1)),
               ("median", "Median native normalized ROCS", "viridis", (0.8, 1)),
               ("p05", "Lower tail: fifth percentile ROCS", "viridis", (0, 1)),
               ("fraction_ge_095", "Fraction of inputs reaching ROCS ≥ 0.95", "viridis", (0, 1)),
               ("seconds", "Mean candidate generation + minimization + pruning + alignment (s)", "magma_r", None),
               ("reference_seconds", "Mean one-time reference preparation (s)", "magma_r", None)]
    for ax, (metric, title, cmap, limits) in zip(axes.flat, metrics):
        values = np.array([[lookup.loc[(rn, rr, cn, cr), metric]
                            for cn, cr in candidate_keys] for rn, rr in reference_keys])
        image = ax.imshow(values, aspect="auto", cmap=cmap,
                          vmin=limits[0] if limits else None,
                          vmax=limits[1] if limits else None)
        ax.set_xticks(range(len(candidate_keys)),
                      [f"C{n}\n{'off' if r == 0 else str(r) + ' Å'}" for n, r in candidate_keys])
        ax.set_yticks(range(len(reference_keys)), [f"R{n} / {r} Å" for n, r in reference_keys])
        ax.set_xlabel("Candidate requests per stereoisomer / additional RMSD pruning")
        ax.set_ylabel("Reference requests per stereoisomer / RMSD pruning")
        ax.set_title(title, fontsize=12)
        for i, j in np.ndindex(values.shape):
            val = values[i, j]
            rgba = image.cmap(image.norm(val)) if np.isfinite(val) else (1, 1, 1, 1)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="black" if luminance > 0.48 else "white", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.8)
    inputs = int(summary.inputs.iloc[0])
    excluded = int(summary.color_excluded.iloc[0])
    fig.suptitle(f"Same-drug self-recovery · {inputs} inputs ({excluded} color exclusions) · MMFF94s ε=78.4\n"
                 "Reference stereo cap 8; candidate cap 4 · independent geometry seeds 43/42 · failures retained as zero\n"
                 "One best pair per input; shared component timing, not RL latency", fontsize=15)
    fig.savefig(output / "calibration.png", dpi=160)
    fig.savefig(output / "calibration.svg")
    plt.close(fig)


def render_conformer_study(frame: pd.DataFrame, output: Path) -> None:
    """Two rows for preparation methods; score/time occupy separate columns."""
    frame = frame.copy()
    modes = [
        ("training_ETKDG", "ETKDG only"),
        ("MMFF94_minimized", "ETKDG + MMFF94"),
    ]
    caps = sorted(frame.isomer_cap.unique())
    keys = ["mode", "seed", "requested_conformers", "requested_references"]
    equivalent = (
        frame.actual_isomers.eq(1).all()
        and frame.groupby(keys).normalized_combo.nunique(dropna=False).max() <= 1
    )
    all_caps = caps.copy()
    if equivalent:
        caps = [4 if 4 in caps else caps[0]]
    budgets = sorted(frame.requested_conformers.unique())
    requests = sorted(frame.requested_references.unique(), key=lambda r: r or np.inf)
    first = frame[frame.isomer_cap == caps[0]]
    actual = first.groupby("requested_references").actual_references.first()
    available = (
        first.groupby("requested_references").reference_budget_available.first()
        if "reference_budget_available" in frame
        else pd.Series(True, index=actual.index)
    )
    references = []
    seen_counts = set()
    for request in requests:
        count = int(actual.loc[request])
        if available.loc[request] and count in seen_counts:
            continue
        references.append(request)
        if available.loc[request]:
            seen_counts.add(count)
    fig, axes = plt.subplots(
        2, 2 * len(caps), figsize=(12 * len(caps), 10), squeeze=False
    )
    time_limit = max(float(frame.total_seconds.max()), 1e-9)
    for facet, cap in enumerate(caps):
        for row, (mode, label) in enumerate(modes):
            subset = frame[(frame.isomer_cap == cap) & (frame["mode"] == mode)]
            for column, metric in enumerate(("normalized_combo", "total_seconds")):
                ax = axes[row, 2 * facet + column]
                grouped = subset.groupby(
                    ["requested_references", "requested_conformers"]
                )[metric]
                means = (
                    grouped.mean().unstack().reindex(index=references, columns=budgets)
                )
                deviations = (
                    grouped.std().unstack().reindex(index=references, columns=budgets)
                )
                palette = plt.get_cmap("viridis" if column == 0 else "magma").copy()
                palette.set_bad("#dddddd")
                image = ax.imshow(
                    means,
                    vmin=0,
                    vmax=1 if column == 0 else time_limit,
                    cmap=palette,
                    aspect="auto",
                )
                for y, reference in enumerate(references):
                    for x in range(len(budgets)):
                        value, sd = means.iloc[y, x], deviations.iloc[y, x]
                        text = (
                            "unavailable" if not available.loc[reference] else "failed"
                        )
                        if np.isfinite(value):
                            text = f"{value:.3f}"
                            text += f"\n±{sd:.3f}" if np.isfinite(sd) else ""
                        fraction = value / (1 if column == 0 else time_limit)
                        ax.text(
                            x,
                            y,
                            text,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black"
                            if not np.isfinite(value) or fraction > 0.65
                            else "white",
                        )
                labels = [
                    f"{'all' if r == 0 or (0 in actual.index and actual.loc[r] == actual.loc[0]) else r} → {int(actual.loc[r])}"
                    if available.loc[r]
                    else f"{r} (unavailable)"
                    for r in references
                ]
                ax.set_xticks(range(len(budgets)), budgets)
                ax.set_yticks(range(len(references)), labels)
                ax.set_xlabel("Requested candidate conformers per isomer")
                ax.set_ylabel("Reference conformer count")
                quantity = (
                    "Score: shape-plus-color combo / 2"
                    if column == 0
                    else "Seconds per molecule"
                )
                counts = ", ".join(
                    map(str, sorted(map(int, subset.actual_isomers.unique())))
                )
                ax.set_title(
                    f"{label}\n{quantity} | isomer cap {cap}, actual {counts}",
                    fontsize=10,
                )
                fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    fig.suptitle(
        f"MMFF94 / RDKit Tanimoto shape + color | {frame.seed.nunique()} embedding seeds\n"
        "Molecular identity and reference-generation settings: manifest.json",
        fontsize=12,
    )
    footer = (
        "Means ± seed SD. No RNN was trained. Time includes candidate preparation and overlays, "
        "excluding reference preparation and multiprocessing startup. Grey cells lack the requested number of references. "
        "Equivalent reference counts are shown once."
    )
    if equivalent and len(all_caps) > 1:
        footer += f" Isomer caps {list(map(int, all_caps))} yielded one isomer and identical scores; cap {caps[0]} timing is shown."
    fig.text(0.5, 0.01, fill(footer, width=135 * len(caps)), ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def render_mega_study(summary: pd.DataFrame, output: Path, manifest: dict) -> None:
    """Four zoomable images; methods × isomer caps, each tile references × C."""
    from matplotlib.colors import LogNorm, Normalize

    caps = manifest["isomer_caps"]
    references = manifest["reference_caps"]
    conformers = manifest["conformers"]
    methods = manifest["methods"]
    figures = (
        ("mean", "mean_score", "mean_seconds", "Mean ROCS / 2", "Mean seconds"),
        (
            "median",
            "median_score",
            "median_seconds",
            "Median ROCS / 2",
            "Median seconds",
        ),
        (
            "tails",
            "q10_score",
            "q95_seconds",
            "10th-percentile ROCS / 2",
            "95th-percentile seconds",
        ),
        (
            "reliability",
            "failure_fraction",
            "top5_overlap",
            "Preparation failure fraction",
            "Top-5% ranking overlap",
        ),
    )
    for filename, first_metric, second_metric, first_label, second_label in figures:
        fig, axes = plt.subplots(
            2 * len(methods), len(caps), figsize=(25, 4 * len(methods)), squeeze=False
        )
        time_values = summary[second_metric].dropna()
        is_time = filename != "reliability"
        norms = [
            Normalize(0, 1),
            LogNorm(
                max(float(time_values.min()), 1e-5),
                max(float(time_values.max()), float(time_values.min()) * 1.01),
            )
            if is_time
            else Normalize(0, 1),
        ]
        palettes = [
            plt.get_cmap("Reds" if filename == "reliability" else "viridis").copy(),
            plt.get_cmap("magma" if is_time else "Greens").copy(),
        ]
        for palette in palettes:
            palette.set_bad("#dddddd")
        images = [None, None]
        for method_index, method in enumerate(methods):
            pool_size = manifest["references"][str(method_index)]["count"]
            for metric_index, (metric, label) in enumerate(
                ((first_metric, first_label), (second_metric, second_label))
            ):
                for column, cap in enumerate(caps):
                    ax = axes[2 * method_index + metric_index, column]
                    selected = summary[
                        (summary.method == method_index) & (summary.isomer_cap == cap)
                    ]
                    table = selected.pivot(
                        index="reference_cap", columns="conformers", values=metric
                    ).reindex(index=references, columns=conformers)
                    images[metric_index] = ax.imshow(
                        table,
                        aspect="auto",
                        cmap=palettes[metric_index],
                        norm=norms[metric_index],
                    )
                    for y in range(len(references)):
                        for x in range(len(conformers)):
                            value = table.iloc[y, x]
                            text = (
                                "—"
                                if not np.isfinite(value)
                                else f"{value:.2f}"
                                if metric_index == 1 and is_time
                                else f"{value:.3f}"
                            )
                            rgba = palettes[metric_index](norms[metric_index](value))
                            luminance = float(
                                np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
                            )
                            ax.text(
                                x,
                                y,
                                text,
                                ha="center",
                                va="center",
                                fontsize=5.5,
                                color="black"
                                if luminance > 0.5 or not np.isfinite(value)
                                else "white",
                            )
                    ax.set_xticks(range(len(conformers)), conformers, fontsize=6)
                    labels = [
                        f"all≤30 ({pool_size})"
                        if r == 0
                        else str(r)
                        if r <= pool_size
                        else f"{r}: n/a"
                        for r in references
                    ]
                    ax.set_yticks(range(len(references)), labels, fontsize=6)
                    if column == 0:
                        ax.set_ylabel(
                            f"{method['name']}\n{label}\nReference count", fontsize=7
                        )
                    if method_index == 0 and metric_index == 0:
                        ax.set_title(
                            f"Isomer cap: {'all≤30' if cap == 30 else cap}", fontsize=10
                        )
                    if method_index == len(methods) - 1 and metric_index == 1:
                        ax.set_xlabel("Candidate conformer attempts", fontsize=8)
        for i, label in enumerate((first_label, second_label)):
            bar = fig.add_axes([0.965, 0.56 if i == 0 else 0.12, 0.008, 0.30])
            fig.colorbar(images[i], cax=bar).set_label(label, fontsize=9)
        status = (
            f"PROVISIONAL {manifest['panel_size']}/{manifest['planned_panel_size']} completed | "
            if manifest.get("state") == "provisional"
            else ""
        )
        fig.suptitle(
            f"{status}{filename.upper()} | fixed query vs {manifest['panel_size']} different generated molecules\n"
            "Each tile: reference count × candidate conformer attempts; no RNN training",
            fontsize=14,
        )
        footer = (
            "Parent-molecule summaries after averaging embedding seeds; failures score zero. Grey = unavailable reference count. "
            "Time is decomposed wall seconds on concurrent one-thread workers; cached work is charged to every configuration. "
            "All is explicitly capped at 30. None = no force-field minimization or energy-window filtering. "
            "Top-5% overlap uses each method's largest tested budget, not activity labels or biological ground truth."
        )
        fig.text(0.5, 0.006, fill(footer, 200), ha="center", fontsize=9)
        fig.subplots_adjust(
            left=0.06, right=0.95, top=0.96, bottom=0.04, hspace=0.36, wspace=0.22
        )
        fig.savefig(output / f"mega_{filename}.png", dpi=200)
        plt.close(fig)


def _analysis_finish(fig, output, filename, metadata, title, note=""):
    status = "PROVISIONAL" if metadata["state"] == "provisional" else "COMPLETE PANEL"
    fig.suptitle(
        f"{status} · {metadata['completed_molecules']}/{metadata['planned_molecules']} molecules\n{title}",
        fontsize=15,
    )
    footer = (
        "ROCS = TanimotoCombo / 2; three seeds averaged per parent. "
        "Preparation failures are assigned score zero. "
        "Groups are RDKit isomer upper bounds. Missing slow cases can bias time downward. "
        "Higher scores do not establish biological accuracy. " + note
    )
    fig.text(
        0.5,
        0.006,
        fill(footer, int(fig.get_figwidth() * 9)),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(output / filename, dpi=160)
    plt.close(fig)


def _analysis_caps(stratum, caps):
    """Hide redundant cap aliases only when all molecules in a stratum share them."""
    upper = {"1": 1, "2": 2, "3-4": 4, "5-8": 8}.get(stratum, np.inf)
    selected, seen = [], set()
    for cap in caps:
        effective = min(cap, upper)
        if effective not in seen:
            selected.append(cap)
            seen.add(effective)
    return selected


def _analysis_heatmaps(summary, output, metadata, profile="mean"):
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    change = max(
        summary[["mean_isomer_gain", "median_isomer_gain"]].abs().max().max(), 0.001
    )
    profiles = {
        "mean": (
            ["mean_score", "mean_isomer_gain", "median_isomer_gain", "mean_seconds"],
            [
                "Mean ROCS (failures=0)",
                "Mean paired gain vs cap 1",
                "Median paired gain vs cap 1",
                "Mean seconds (log color)",
            ],
        ),
        "median": (
            ["median_score", "median_seconds"],
            ["Median ROCS (failures=0)", "Median seconds (log color)"],
        ),
        "tails": (
            ["q10_score", "p95_seconds"],
            ["10th-percentile ROCS", "95th-percentile seconds (log color)"],
        ),
        "reliability": (
            ["failure_fraction", "top5_overlap"],
            ["Preparation failure fraction", "Ranking overlap (not ROCS)"],
        ),
    }
    metrics, labels = profiles[profile]
    norms, palettes = [], []
    for metric in metrics:
        if metric.endswith("seconds"):
            positive = summary.loc[summary[metric] > 0, metric]
            norms.append(
                LogNorm(positive.min(), max(positive.max(), positive.min() * 1.01))
            )
            palette = "magma"
        elif "gain" in metric:
            norms.append(TwoSlopeNorm(0, -change, change))
            palette = "RdBu_r"
        else:
            norms.append(Normalize(0, 1))
            palette = (
                "Reds"
                if metric == "failure_fraction"
                else "Greens"
                if metric == "top5_overlap"
                else "viridis"
            )
        cmap = plt.get_cmap(palette).copy()
        cmap.set_bad("#dddddd")
        palettes.append(cmap)
    for stratum, count in metadata["strata"].items():
        if not count:
            continue
        caps = _analysis_caps(stratum, metadata["isomer_caps"])
        shown = (
            [0, 3]
            if profile == "mean" and stratum == "1"
            else list(range(len(metrics)))
        )
        fig, axes = plt.subplots(
            len(metadata["methods"]),
            len(shown),
            figsize=(12, 14) if len(shown) == 2 else (20, 21),
            squeeze=False,
        )
        for method, record in enumerate(metadata["methods"]):
            subset = summary[(summary.stratum == stratum) & (summary.method == method)]
            for column, metric_index in enumerate(shown):
                metric = metrics[metric_index]
                ax = axes[method, column]
                grid = subset.pivot(
                    index="isomer_cap", columns="conformers", values=metric
                ).reindex(index=caps, columns=metadata["conformers"])
                im = ax.imshow(
                    grid,
                    aspect="auto",
                    cmap=palettes[metric_index],
                    norm=norms[metric_index],
                )
                for y, x in np.ndindex(grid.shape):
                    value = grid.iloc[y, x]
                    rgba = palettes[metric_index](norms[metric_index](value))
                    color = (
                        "black"
                        if not np.isfinite(value)
                        or np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]) > 0.5
                        else "white"
                    )
                    label = (
                        "n/a"
                        if not np.isfinite(value)
                        else f"{value:.2f}"
                        if metric.endswith("seconds")
                        else f"{value:+.3f}"
                        if "gain" in metric
                        else f"{value:.3f}"
                    )
                    ax.text(
                        x, y, label, ha="center", va="center", fontsize=8, color=color
                    )
                ax.set_xticks(
                    range(len(metadata["conformers"])),
                    metadata["conformers"],
                    fontsize=8,
                )
                ax.set_yticks(
                    range(len(caps)),
                    [str(c) if c < 30 else "≤30" for c in caps],
                    fontsize=8,
                )
                method_label = record["name"].replace(" eps=", "\nε=")
                ax.set_ylabel(
                    f"{method_label}\nIsomer cap" if column == 0 else "Isomer cap",
                    fontsize=9,
                )
                if method == 0:
                    label = labels[metric_index]
                    if metric == "top5_overlap":
                        k = int(subset.top_k.iloc[0])
                        label = f"Top-{k} ranking overlap (not ROCS)"
                    ax.set_title(label, fontsize=11)
                if method == len(metadata["methods"]) - 1:
                    ax.set_xlabel("Conformer attempts per enumerated isomer")
                if method == 0:
                    bar = fig.add_axes(
                        [
                            0.067 + column * 0.96 / len(shown),
                            0.07,
                            0.72 / len(shown),
                            0.008,
                        ]
                    )
                    fig.colorbar(im, cax=bar, orientation="horizontal")
        fig.subplots_adjust(
            left=0.065,
            right=0.99,
            top=0.90 if len(shown) == 2 else 0.94,
            bottom=0.12,
            hspace=0.55,
            wspace=0.24,
        )
        slug = stratum.replace(">=", "ge").replace("-", "_")
        _analysis_finish(
            fig,
            output,
            f"isomers_{slug}{'' if profile == 'mean' else '_' + profile}.png",
            metadata,
            f"{profile.upper()} · Isomer-upper-bound range {stratum} · n={count} · all available references",
            "Shared scales; redundant cap aliases hidden. "
            + (
                "Ranking overlap compares each method with its C100/I30 anchor within this category. At that anchor it is 1 by construction. Small categories track only one molecule."
                if profile == "reliability"
                else "Overall gains include failure recovery; success-only gains are in sampling_diagnostics.png. Quantiles describe parent seed averages, not confidence intervals."
            ),
        )


def _analysis_tradeoffs(summary, output, metadata):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    colors = plt.get_cmap("tab10")
    for ax, (stratum, count) in zip(axes.flat, metadata["strata"].items()):
        subset = summary[
            (summary.stratum == stratum)
            & summary.isomer_cap.isin(_analysis_caps(stratum, metadata["isomer_caps"]))
        ]
        for method, record in enumerate(metadata["methods"]):
            points = subset[subset.method == method].sort_values("mean_seconds")
            ax.scatter(
                points.mean_seconds,
                points.mean_budget_gain,
                color=colors(method),
                s=23,
                alpha=0.45,
                label=record["name"],
            )
            best = -np.inf
            frontier = []
            for row in points.itertuples():
                if row.mean_budget_gain > best:
                    frontier.append(row)
                    best = row.mean_budget_gain
            if frontier:
                ax.plot(
                    [r.mean_seconds for r in frontier],
                    [r.mean_budget_gain for r in frontier],
                    color=colors(method),
                    lw=1,
                )
            if method == 1:
                labelled = {
                    r.conformers: r
                    for r in frontier
                    if r.conformers in [1, 10, 30, 100]
                }
                for row in labelled.values():
                    ax.plot(
                        [row.median_seconds, row.p95_seconds],
                        [row.mean_budget_gain, row.mean_budget_gain],
                        color=colors(method),
                        alpha=0.2,
                        lw=3,
                    )
                    ax.annotate(
                        f"{row.conformers}/{row.isomer_cap}",
                        (row.mean_seconds, row.mean_budget_gain),
                        xytext=(3, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=colors(method),
                    )
        ax.set(
            xscale="log",
            xlabel="Mean seconds per molecule",
            ylabel="Mean paired score gain vs C1 / I1",
            title=f"Isomer bound {stratum} · n={count}",
        )
        ax.axhline(0, color="grey", lw=0.7)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    panels = list(axes.flat)[: len(metadata["strata"])]
    limits = [ax.get_xlim() for ax in panels]
    gain_limit = max(float(summary.mean_budget_gain.max()), 0.001)
    for ax in panels:
        ax.set_xlim(min(x[0] for x in limits), max(x[1] for x in limits))
        ax.set_ylim(
            min(0, float(summary.mean_budget_gain.min())) - 0.005, gain_limit * 1.1
        )
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10)
    axes.flat[-1].text(
        0.05,
        0.08,
        "Each point: C / isomer-cap setting\nLines: within-method nondominated choices\nMMFF94 labels: C / isomer cap\nMMFF94 bars: median–95th percentile time\nAll available references",
        transform=axes.flat[-1].transAxes,
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.4, wspace=0.28)
    _analysis_finish(
        fig,
        output,
        "score_time.png",
        metadata,
        "Score gain versus computation cost",
        "Frontiers are descriptive within each method, not evidence for a biologically optimal force field.",
    )


def _analysis_distributions(parents, cohort_ids, output, metadata):
    all_refs = parents[parents.reference_cap.eq(0)]
    mmff = all_refs[all_refs.method.eq(1)]
    maximum_cap = max(metadata["isomer_caps"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for c, i in [(1, 1), (10, 3), (30, 10), (100, 30)]:
        values = (
            mmff[mmff.conformers.eq(c) & mmff.isomer_cap.eq(i)]
            .score.sort_values()
            .to_numpy()
        )
        axes[0, 0].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=f"C={c}, I={i}",
        )
    axes[0, 0].set(
        xlabel="Normalized ROCS",
        ylabel="Fraction of molecules ≤ score",
        xlim=(0, 1),
        title="MMFF94: complete score distributions",
    )
    axes[0, 0].legend()
    full_cap = mmff[mmff.isomer_cap.eq(maximum_cap)]
    for quantile in [0.5, 0.9, 0.95]:
        values = full_cap.groupby("conformers").score.quantile(quantile)
        axes[0, 1].plot(
            values.index, values, "o-", label=f"Percentile {100 * quantile:g}"
        )
    axes[0, 1].set(
        xlabel="Conformer attempts",
        ylabel="Normalized ROCS",
        ylim=(0, 1),
        title="MMFF94 / I≤30: upper-score response",
    )
    axes[0, 1].legend()
    cohort = all_refs[
        all_refs.molecule_id.isin(cohort_ids) & all_refs.isomer_cap.eq(maximum_cap)
    ]
    for method, record in enumerate(metadata["methods"]):
        values = cohort[cohort.method.eq(method)].groupby("conformers").score.mean()
        axes[1, 0].plot(values.index, values, "o-", label=record["name"], ms=3)
    axes[1, 0].set(
        xlabel="Conformer attempts",
        ylabel="Mean normalized ROCS",
        ylim=(0, 1),
        title=f"Same {len(cohort_ids)} high-score molecules in every curve · I≤30",
    )
    axes[1, 0].legend(fontsize=7)
    anchor = mmff[mmff.conformers.eq(100) & mmff.isomer_cap.eq(30)][
        ["molecule_id", "score", "stratum"]
    ].rename(columns={"score": "anchor"})
    smaller = mmff[mmff.conformers.eq(10) & mmff.isomer_cap.eq(3)][
        ["molecule_id", "score"]
    ]
    paired = anchor.merge(smaller, on="molecule_id", validate="one_to_one")
    for stratum, count in metadata["strata"].items():
        points = paired[paired.stratum.eq(stratum)]
        axes[1, 1].scatter(
            points.anchor, points.score, s=13, alpha=0.6, label=f"{stratum} (n={count})"
        )
    axes[1, 1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[1, 1].set(
        xlabel="MMFF94: C100 / I30 score",
        ylabel="MMFF94: C10 / I3 score",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Individual molecules: smaller versus larger budget",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.36, wspace=0.24)
    _analysis_finish(
        fig,
        output,
        "score_distributions.png",
        metadata,
        "Absolute scores and a fixed high-score cohort",
        "Cohort chosen once by MMFF94/C100/I30/all on these measurements; exploratory selection, not an independent validation set.",
    )


def _analysis_diagnostics(summary, output, metadata):
    largest = summary[
        summary.conformers.eq(max(metadata["conformers"]))
        & summary.isomer_cap.eq(max(metadata["isomer_caps"]))
    ]
    strata = list(metadata["strata"])
    names = [m["name"] for m in metadata["methods"]]
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    heatmaps = [
        (
            axes[0, 0],
            "failure_fraction",
            "Preparation failure fraction",
            "Reds",
            (0, 1),
        ),
        (
            axes[0, 1],
            "successful_pair_gain",
            "Mean isomer gain: successful pairs only",
            "RdBu_r",
            None,
        ),
        (
            axes[1, 0],
            "retained_fraction",
            "Mean retained / embedded (nonempty generations)",
            "viridis",
            (0, 1),
        ),
        (
            axes[1, 1],
            "seed_score_sd",
            "Mean within-parent score SD across seeds",
            "magma",
            None,
        ),
    ]
    for ax, metric, title, palette, limits in heatmaps:
        grid = largest.pivot(index="method", columns="stratum", values=metric).reindex(
            index=range(len(names)), columns=strata
        )
        lower, upper = limits or (
            (-max(grid.abs().max().max(), 0.001), max(grid.abs().max().max(), 0.001))
            if metric == "successful_pair_gain"
            else (0, max(grid.max().max(), 0.001))
        )
        im = ax.imshow(grid, aspect="auto", cmap=palette, vmin=lower, vmax=upper)
        ax.set_xticks(
            range(len(strata)),
            [f"{s}\nn={metadata['strata'][s]}" for s in strata],
            fontsize=8,
        )
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(title, fontsize=11)
        for y, x in np.ndindex(grid.shape):
            value = grid.iloc[y, x]
            rgba = im.cmap(im.norm(value))
            color = (
                "black" if np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]) > 0.5 else "white"
            )
            label = "n/a" if not np.isfinite(value) else f"{value:.3f}"
            if metric == "successful_pair_gain":
                n = largest[
                    (largest.method == y) & (largest.stratum == strata[x])
                ].successful_pairs
                label += f"\n(n={int(n.iloc[0])})" if len(n) else ""
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    generated = summary[summary.method.eq(0)]
    for stratum, count in metadata["strata"].items():
        selected = generated[
            (generated.stratum == stratum) & generated.conformers.eq(100)
        ]
        axes[0, 2].plot(
            selected.isomer_cap,
            selected.mean_isomers,
            "o-",
            label=f"{stratum}, n={count}",
        )
        selected = generated[
            (generated.stratum == stratum) & generated.isomer_cap.eq(30)
        ]
        axes[1, 2].plot(
            selected.conformers,
            selected.median_embedded,
            "o-",
            label=f"{stratum}, n={count}",
        )
    axes[0, 2].set(
        xlabel="Requested isomer cap",
        ylabel="Mean distinct SDF-readback identities",
        title="Recorded identity counts · C100 · may exceed cap",
    )
    axes[1, 2].set(
        xlabel="Conformer attempts per enumerated isomer",
        ylabel="Median total embedded conformers",
        title="Total generated across isomers · I≤30",
    )
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.4, wspace=0.65)
    _analysis_finish(
        fig,
        output,
        "sampling_diagnostics.png",
        metadata,
        "Failures, successful comparisons, sampling yield and seed variation",
        "Heatmaps: C100/I30/all. Recorded identity counts are not validated enumeration counts; see identity_count_warnings.csv. SD is not a confidence interval.",
    )


def _analysis_references(parents, output, metadata):
    fig, axes = plt.subplots(
        2, len(metadata["methods"]), figsize=(24, 8), squeeze=False
    )
    for method, record in enumerate(metadata["methods"]):
        selected = parents[parents.method.eq(method) & parents.isomer_cap.eq(30)]
        for c in [1, 10, 30, 100]:
            points = selected[selected.conformers.eq(c)]
            reference_one = points[points.reference_cap.eq(1)][
                ["molecule_id", "score"]
            ].rename(columns={"score": "reference_one"})
            paired = points.merge(
                reference_one, on="molecule_id", validate="many_to_one"
            )
            paired["reference_gain"] = paired.score - paired.reference_one
            # "All" may duplicate an explicit cap; each actual count is plotted once.
            paired = paired.sort_values("reference_cap").drop_duplicates(
                ["molecule_id", "actual_references"]
            )
            table = paired.groupby("actual_references")[
                ["reference_gain", "seconds"]
            ].mean()
            axes[0, method].plot(
                table.index, table.reference_gain, "o-", label=f"C={c}"
            )
            axes[1, method].plot(table.index, table.seconds, "o-")
        axes[0, method].set_title(record["name"], fontsize=10)
        for ax in axes[:, method]:
            ax.grid(alpha=0.2)
            ax.set_xticks(sorted(selected.actual_references.unique()))
        axes[1, method].set_xlabel("Actual reference count")
        axes[1, method].set_yscale("log")
    axes[0, 0].set_ylabel("Mean paired score gain vs reference 1")
    axes[1, 0].set_ylabel("Mean seconds (log)")
    axes[0, -1].legend(fontsize=8)
    for row in range(2):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(lim[0] for lim in limits), max(lim[1] for lim in limits))
    fig.subplots_adjust(top=0.84, bottom=0.16, hspace=0.22, wspace=0.32)
    _analysis_finish(
        fig,
        output,
        "reference_effects.png",
        metadata,
        "Reference-count effects · all completed molecules · isomer cap ≤30",
        "Only measured distinct reference counts appear; unavailable counts remain absent. Each method has its own query conformers.",
    )


def render_conformer_analysis(parents, summary, cohort_ids, output, metadata):
    """Render the selected stratified analysis from saved parent-level tables."""
    for profile in ("mean", "median", "tails", "reliability"):
        _analysis_heatmaps(summary, output, metadata, profile)
    _analysis_tradeoffs(summary, output, metadata)
    _analysis_distributions(parents, cohort_ids, output, metadata)
    _analysis_diagnostics(summary, output, metadata)
    _analysis_references(parents, output, metadata)


def render_sampling_plan(cases, output: Path):
    """Show heterogeneous planned effort without generating conformers or scores."""
    rows = [
        dict(
            molecule_id=c["molecule_id"],
            smiles=c["smiles"],
            category=c["category"],
            rotatable_bonds=c["rotatable_bonds"],
            oh_nh_count=c["oh_nh_count"],
            saturated_rings=c["saturated_rings"],
            isomer_upper_bound=c["isomer_upper_bound"],
            reference_isomers=len(c["reference_isomers"]),
            baseline_attempts=c["conformers_per_isomer"],
            half_attempts=c["attempts"][0],
            double_attempts=c["attempts"][2],
        )
        for c in cases
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "sampling_plans.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    categories = ["1", "2", "3-4", "5-8", "9-25"]
    groups = [frame[frame.category.eq(c)] for c in categories]
    axes[0].boxplot(
        [g.baseline_attempts for g in groups],
        tick_labels=[f"{c}\nn={len(g)}" for c, g in zip(categories, groups)],
        whis=(0, 100),
    )
    for count in [25, 50, 100]:
        axes[0].axhline(count, lw=1, ls="--", label=f"Fixed {count}")
    axes[0].set(
        xlabel="Enumerated reference stereoisomer category",
        ylabel="Planned conformer attempts per stereoisomer",
        title="Same adaptive recipe, different search effort",
    )
    axes[0].legend()
    points = axes[1].scatter(
        frame.rotatable_bonds,
        frame.baseline_attempts,
        c=frame.oh_nh_count + frame.saturated_rings,
        s=35,
        alpha=0.5,
        cmap="viridis",
    )
    axes[1].set(
        xlabel="Rotatable bonds",
        ylabel="Planned conformer attempts per stereoisomer",
        title="OH/NH and saturated rings also change the budget",
    )
    fig.colorbar(points, ax=axes[1]).set_label("NH/OH count + saturated rings")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"SAMPLING PLAN ONLY · {len(frame)} parents · no conformer generation or ROCS measurement"
    )
    fig.text(
        0.5,
        0.02,
        "AQME-style mid heuristic: min(250, 30 × (rotors + NH/OH + saturated rings)); featureless inputs receive 10.\nBox whiskers show the full planned range, not confidence intervals. Higher budgets are not established as equally accurate across molecules.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.23, wspace=0.3)
    fig.savefig(output / "sampling_budget_plan.png", dpi=180)
    plt.close(fig)
    return frame


def render_mini_recovery(frame: pd.DataFrame, output: Path, manifest: dict) -> None:
    """Fixed-cohort reference-count × candidate-count panels; one input per cell."""
    from matplotlib.colors import LogNorm, Normalize

    keys = [
        "group",
        "cutoff",
        "method",
        "isomer_cap",
        "reference_cap",
        "candidate_attempts",
    ]
    if frame.duplicated(keys + ["molecule_id"]).any():
        raise ValueError(
            "Mini-study summaries require one observation per input/configuration"
        )
    summary = (
        frame.groupby(keys, observed=True)
        .agg(
            inputs=("molecule_id", "size"),
            available=("available", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q10_score=("score", lambda x: x.quantile(0.1)),
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            p95_seconds=("seconds", lambda x: x.quantile(0.95)),
            mean_shape=("shape_tanimoto", "mean"),
            mean_color=("color_tanimoto", "mean"),
            candidate_failures=("candidate_failure", "sum"),
            reference_failures=("reference_failure", "sum"),
            mean_usable_conformers=("usable_conformers", "mean"),
            mean_actual_isomers=("actual_candidate_isomers", "mean"),
            mean_reference_count=("actual_references", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    profiles = {
        "mean": [
            ("mean_score", "Mean normalized ROCS"),
            ("mean_seconds", "Mean component seconds"),
        ],
        "median": [
            ("median_score", "Median normalized ROCS"),
            ("median_seconds", "Median component seconds"),
        ],
        "tails": [
            ("q10_score", "10th-percentile ROCS"),
            ("p95_seconds", "95th-percentile component seconds"),
        ],
        "components": [
            ("mean_shape", "Mean winning shape"),
            ("mean_color", "Mean winning effective color"),
        ],
    }
    for group, cutoff in (
        summary[["group", "cutoff"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        selected = summary[summary.group.eq(group) & summary.cutoff.eq(cutoff)]
        for profile, metrics in profiles.items():
            fig, axes = plt.subplots(
                6, 4, figsize=(23, 16), squeeze=False, layout="constrained"
            )
            for method, settings in enumerate(manifest["methods"]):
                for metric_index, (metric, label) in enumerate(metrics):
                    values = summary[metric].dropna()
                    is_time = metric.endswith("seconds")
                    positive = values[values > 0]
                    norm = (
                        LogNorm(
                            max(positive.min(), 1e-5),
                            max(positive.max(), positive.min() * 1.01),
                        )
                        if is_time and len(positive)
                        else Normalize(0, 1)
                    )
                    cmap = plt.get_cmap("magma" if is_time else "viridis").copy()
                    cmap.set_bad("#eeeeee")
                    for column, cap in enumerate(manifest["isomer_caps"]):
                        ax = axes[method * 2 + metric_index, column]
                        cells = selected[
                            selected.method.eq(method) & selected.isomer_cap.eq(cap)
                        ]
                        table = cells.pivot(
                            index="reference_cap",
                            columns="candidate_attempts",
                            values=metric,
                        ).reindex(
                            index=manifest["reference_caps"],
                            columns=manifest["candidate_attempts"],
                        )
                        im = ax.imshow(table, aspect="auto", cmap=cmap, norm=norm)
                        for y, x in np.ndindex(table.shape):
                            value = table.iloc[y, x]
                            cell = cells[
                                cells.reference_cap.eq(table.index[y])
                                & cells.candidate_attempts.eq(table.columns[x])
                            ]
                            coverage = (
                                f"{int(cell.available.iloc[0])}/{int(cell.inputs.iloc[0])}"
                                if len(cell)
                                else "0/0"
                            )
                            text = (
                                "n/a"
                                if not np.isfinite(value)
                                else f"{value:.2f}"
                                if is_time
                                else f"{value:.3f}"
                            )
                            ink = (
                                "black"
                                if not np.isfinite(value)
                                or np.dot(
                                    cmap(norm(value))[:3], [0.2126, 0.7152, 0.0722]
                                )
                                > 0.5
                                else "white"
                            )
                            ax.text(
                                x,
                                y,
                                f"{text}\n{coverage}",
                                ha="center",
                                va="center",
                                color=ink,
                                fontsize=7,
                            )
                        ax.set_xticks(range(len(table.columns)), table.columns)
                        ax.set_yticks(
                            range(len(table.index)), [r or "all" for r in table.index]
                        )
                        ax.set_title(
                            f"{settings['name']} · candidate isomer cap {cap}",
                            fontsize=9,
                        )
                        if column == 0:
                            ax.set_ylabel(f"{label}\nRetained references scored")
                        if method == 2 and metric_index == 1:
                            ax.set_xlabel(
                                "Conformer requests per candidate stereoisomer"
                            )
                    fig.colorbar(
                        im,
                        ax=axes[method * 2 + metric_index, :],
                        label=label,
                        shrink=0.8,
                        pad=0.01,
                    )
            count = manifest["group_counts"][group]
            fig.suptitle(
                f"Each input against its own references · retained-size group {group} · n={count} · RMSD {cutoff:g} Å\n{profile.capitalize()} · one best pair per input · stereo/candidate seed 42, reference geometry seed 43",
                fontsize=14,
            )
            fig.supxlabel(
                "Cell second line = available / selected inputs. Candidate failures score 0; unavailable reference budgets are grey. Times sum measured shared components, not independent end-to-end runs. Native ROCS; colorless inputs excluded before scoring.",
                fontsize=9,
            )
            fig.savefig(
                output / f"group_{group}_rmsd_{cutoff:g}_{profile}.png", dpi=140
            )
            plt.close(fig)


def render_self_recovery(frame: pd.DataFrame, output: Path, manifest: dict) -> None:
    """Self-recovery panels with explicit per-cell reference availability counts."""
    from matplotlib.colors import LogNorm, Normalize

    frame = frame.assign(category=frame.category.astype(str))

    keys = [
        "category",
        "method",
        "reference_scale",
        "candidate_scale",
        "isomer_cap",
        "reference_cap",
    ]
    parents = (
        frame.groupby(keys + ["molecule_id"], observed=True)
        .agg(
            score=("score", "mean"),
            seconds=("seconds", "mean"),
            available=("available", "first"),
            failure=("failure", "mean"),
            reference_seconds=("reference_seconds", "mean"),
            candidate_attempts=("candidate_attempts", "first"),
            reference_attempts=("reference_attempts", "first"),
            actual_references=("actual_references", "first"),
            actual_isomers=("actual_isomers", "mean"),
            candidate_conformers=("candidate_conformers", "mean"),
            shape=("shape_tanimoto", "mean"),
            color=("color_tanimoto", "mean"),
        )
        .reset_index()
    )
    parents["recovered"] = parents.score.ge(0.9).astype(float).where(parents.available)
    summary = (
        parents.groupby(keys, observed=True)
        .agg(
            attempted=("molecule_id", "size"),
            available=("available", "sum"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            q10_score=("score", lambda x: x.quantile(0.1)),
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            p95_seconds=("seconds", lambda x: x.quantile(0.95)),
            recovery_fraction=("recovered", "mean"),
            failure_fraction=("failure", "mean"),
            mean_reference_seconds=("reference_seconds", "mean"),
            mean_actual_references=("actual_references", "mean"),
            mean_actual_isomers=("actual_isomers", "mean"),
            mean_candidate_conformers=("candidate_conformers", "mean"),
        )
        .reset_index()
    )
    parents.to_csv(output / "parent_scores.csv.gz", index=False, compression="gzip")
    summary.to_csv(output / "summary.csv", index=False)
    profiles = {
        "mean": ("mean_score", "mean_seconds", "Mean normalized ROCS", "Mean seconds"),
        "median": (
            "median_score",
            "median_seconds",
            "Median normalized ROCS",
            "Median seconds",
        ),
        "tails": (
            "q10_score",
            "p95_seconds",
            "10th-percentile ROCS",
            "95th-percentile seconds",
        ),
        "reliability": (
            "recovery_fraction",
            "failure_fraction",
            "Fraction reaching ≥0.9",
            "Preparation failure fraction",
        ),
    }
    scales = manifest["scales"]
    reference_caps = manifest["reference_caps"]
    for category in ["1", "2", "3-4", "5-8", "9-25"]:
        selected = summary[summary.category.eq(category)]
        if selected.empty:
            continue
        n = parents[parents.category.eq(category)].molecule_id.nunique()
        bound = {"1": 1, "2": 2, "3-4": 4, "5-8": 8, "9-25": 25}[category]
        caps = []
        for cap in manifest["isomer_caps"]:
            if not caps or min(cap, bound) > min(caps[-1], bound):
                caps.append(cap)

        def finish(fig, name, title, category=category, n=n):
            fig.suptitle(
                f"Each molecule against its OWN reference · category {category} · n={n}\n{title}",
                fontsize=14,
            )
            note = (
                "Categories count reference stereoisomers enumerated before 3D filtering. "
                "Three seed maxima averaged per parent before summaries. n = available / attempted parents. "
                "Reference preparation time is separate from candidate generation/optimization/overlay time. "
                "ROCS = (shape + color Tanimoto) / 2. ≥0.9 is a self-recovery hypothesis, not biological accuracy."
            )
            fig.text(
                0.5,
                0.008,
                fill(note, int(fig.get_figwidth() * 9)),
                ha="center",
                fontsize=8,
            )
            fig.savefig(output / name, dpi=160)
            plt.close(fig)

        for profile, (first, second, first_label, second_label) in profiles.items():
            fig, axes = plt.subplots(
                2 * len(manifest["methods"]),
                len(caps),
                figsize=(5 * len(caps) + 2, 28),
                squeeze=False,
            )
            for method, record in enumerate(manifest["methods"]):
                group = selected[
                    selected.method.eq(method) & selected.reference_scale.eq(scales[1])
                ]
                for metric_index, (metric, label) in enumerate(
                    [(first, first_label), (second, second_label)]
                ):
                    is_time = metric.endswith("seconds")
                    values = summary[metric].dropna()
                    positive = values[values > 0]
                    norm = (
                        LogNorm(
                            max(positive.min(), 1e-5),
                            max(positive.max(), positive.min() * 1.01),
                        )
                        if is_time and len(positive)
                        else Normalize(0, 1)
                    )
                    palette = plt.get_cmap(
                        "magma"
                        if is_time
                        else "Reds"
                        if metric == "failure_fraction"
                        else "viridis"
                    ).copy()
                    palette.set_bad("#dddddd")
                    for column, cap in enumerate(caps):
                        ax = axes[method * 2 + metric_index, column]
                        cells = group[group.isomer_cap.eq(cap)]
                        table = cells.pivot(
                            index="reference_cap",
                            columns="candidate_scale",
                            values=metric,
                        ).reindex(index=reference_caps, columns=scales)
                        im = ax.imshow(table, aspect="auto", cmap=palette, norm=norm)
                        for y, x in np.ndindex(table.shape):
                            value = table.iloc[y, x]
                            cell = cells[
                                cells.reference_cap.eq(reference_caps[y])
                                & cells.candidate_scale.eq(scales[x])
                            ]
                            available, attempted = (
                                (
                                    int(cell.available.iloc[0]),
                                    int(cell.attempted.iloc[0]),
                                )
                                if len(cell)
                                else (0, n)
                            )
                            text = (
                                "n/a"
                                if not np.isfinite(value)
                                else f"{value:.2f}"
                                if is_time
                                else f"{value:.3f}"
                            )
                            color = (
                                "black"
                                if not np.isfinite(value)
                                or np.dot(
                                    palette(norm(value))[:3], [0.2126, 0.7152, 0.0722]
                                )
                                > 0.5
                                else "white"
                            )
                            ax.text(
                                x,
                                y,
                                f"{text}\nn={available}/{attempted}",
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                color=color,
                            )
                        ax.set_xticks(range(3), [f"{s:g}×" for s in scales], fontsize=8)
                        ax.set_yticks(
                            range(len(reference_caps)),
                            ["all≤25" if r == 0 else str(r) for r in reference_caps],
                            fontsize=8,
                        )
                        if column == 0:
                            ax.set_ylabel(
                                f"{record['name'].replace(' eps=', chr(10) + 'ε=')}\n{label}\nRetained reference cap",
                                fontsize=8,
                            )
                        if method == 0 and metric_index == 0:
                            ax.set_title(f"Candidate isomer cap {cap}")
                        if method == len(manifest["methods"]) - 1 and metric_index == 1:
                            ax.set_xlabel(
                                "Candidate conformer attempts / AQME-style baseline"
                            )
                    bar = (
                        fig.add_axes(
                            [0.93, 0.56 if metric_index == 0 else 0.14, 0.009, 0.28]
                        )
                        if method == 0
                        else None
                    )
                    if bar is not None:
                        fig.colorbar(im, cax=bar).set_label(label, fontsize=8)
            fig.subplots_adjust(
                left=0.14 if len(caps) == 1 else 0.08,
                right=0.91,
                top=0.94,
                bottom=0.065,
                hspace=0.65,
                wspace=0.3,
            )
            finish(
                fig,
                f"category_{category.replace('-', '_')}_{profile}.png",
                f"{profile.upper()} · reference initial attempts = {scales[1]:g}× its baseline",
            )

        fig, axes = plt.subplots(
            len(manifest["methods"]), 2, figsize=(12, 18), squeeze=False
        )
        for method, record in enumerate(manifest["methods"]):
            group = selected[
                selected.method.eq(method)
                & selected.isomer_cap.eq(max(manifest["isomer_caps"]))
                & selected.reference_cap.eq(0)
            ]
            for column, metric in enumerate(["mean_score", "mean_seconds"]):
                table = group.pivot(
                    index="reference_scale", columns="candidate_scale", values=metric
                ).reindex(index=scales, columns=scales)
                ax = axes[method, column]
                im = ax.imshow(
                    table,
                    aspect="auto",
                    cmap="viridis" if column == 0 else "magma",
                    vmin=0,
                    vmax=1
                    if column == 0
                    else max(float(selected.mean_seconds.max()), 1e-5),
                )
                for y, x in np.ndindex(table.shape):
                    ax.text(
                        x,
                        y,
                        f"{table.iloc[y, x]:.3f}"
                        if column == 0
                        else f"{table.iloc[y, x]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if im.norm(table.iloc[y, x]) < 0.6 else "black",
                    )
                ax.set_xticks(range(3), [f"{s:g}×" for s in scales])
                ax.set_yticks(range(3), [f"{s:g}×" for s in scales])
                ax.set_ylabel(
                    f"{record['name'].replace(' eps=', chr(10) + 'ε=')}\nReference initial attempts",
                    fontsize=8,
                )
                if method == 0:
                    ax.set_title(
                        "Mean normalized ROCS"
                        if column == 0
                        else "Mean candidate seconds"
                    )
                if method == len(manifest["methods"]) - 1:
                    ax.set_xlabel("Candidate initial attempts")
        fig.subplots_adjust(
            left=0.15, right=0.97, top=0.91, bottom=0.085, hspace=0.65, wspace=0.45
        )
        finish(
            fig,
            f"category_{category.replace('-', '_')}_initial_effort.png",
            "Reference versus candidate initial sampling · all≤25 references/isomers",
        )


def render_reward_heatmaps(
    summary: pd.DataFrame,
    output: Path,
    *,
    methods: dict[str, object],
    labels: dict[str, str],
    caps: tuple[int, ...] = (1, 2, 4, 8, 12, 0),
    candidates: tuple[int, ...] = (5, 10, 15, 20, 35, 50),
    warning: str = "",
) -> None:
    """Render 4x2 ROCS/latency heatmap sheets across force-field methods."""
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    sheets = [
        ("mean", "mean_seconds", "Mean ROCS / mean scoring time", "mean"),
        ("median", "median_seconds", "Median ROCS / median scoring time", "median"),
        ("p05", "p95_seconds", "Fifth-percentile ROCS / 95th-percentile scoring time", "p05"),
        (
            "fraction_ge_095",
            "p95_seconds",
            "Fraction reaching ROCS 0.95 / 95th-percentile scoring time",
            "success",
        ),
    ]
    for score_key, time_key, title, filename in sheets:
        fig, axes = plt.subplots(4, 2, figsize=(18, 23))
        fig.subplots_adjust(
            left=0.075, right=0.96, bottom=0.04, top=0.91, hspace=0.40, wspace=0.30
        )
        vmax = max(float(summary[time_key].max()), 0.001)
        vmin = max(float(summary.loc[summary[time_key].gt(0), time_key].min()), 0.001)
        for row, method in enumerate(methods):
            data = summary[summary.method.eq(method)].set_index(
                ["reference_cap", "candidate_attempts"]
            )
            all_bank = data.loc[(0, candidates[0])]
            bank_label = (
                f"all-bank median {all_bank.median_references:g}; max {all_bank.max_references:g}"
            )
            for col, key in enumerate((score_key, time_key)):
                ax = axes[row, col]
                values = np.array(
                    [
                        [data.loc[(cap, count), key] for cap in caps]
                        for count in candidates
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
                ax.set_xticks(range(len(caps)), ["all" if c == 0 else str(c) for c in caps])
                ax.set_yticks(range(len(candidates)), [f"C{count}" for count in candidates])
                ax.set_xlabel("Total reference-structure cap (at most)")
                ax.set_ylabel("Candidate requests per stereoisomer")
                ax.set_title(
                    f"{labels[method]} · {'ROCS' if col == 0 else 'seconds'}\n{bank_label}",
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
        fig.savefig(output / f"{filename}.png", dpi=170, bbox_inches="tight")
        fig.savefig(output / f"{filename}.svg", bbox_inches="tight")
        plt.close(fig)


FALLBACK_RL_REFERENCE: dict[str, Any] | None = dict(
    provenance=(
        "dichtator RL smoke 2026-10-29, MMFF94s eps78.4 K4/C20, 2 workers, "
        "17 rows (raw_data/rl_smoke); see 20261029_1155_reward_calibration"
    ),
    serial_equivalent_seconds_per_row=6.07,
    wall_seconds_per_row=3.04,
    baseline_mean_seconds=1.88,
    factor=3.2,
)
RL_PROJECTION_SAMPLES = 500
RL_PROJECTION_WORKERS = 4


def rl_reference_measurement(output: Path) -> dict[str, Any] | None:
    """Read the measured live-RL cost reference for this calibration, if saved.

    Looks for ``raw_data/rl_smoke`` then ``raw_data/rl_egcg`` (written by
    ``rl_cost_probe``) and converts its wall-clock ROCS seconds into a
    serial-equivalent per-row cost using the recorded alignment worker count.
    Returns ``None`` when no complete measurement is available.
    """
    import json

    for name in ("rl_smoke", "rl_egcg"):
        raw = output / "raw_data" / name
        manifest = raw / "manifest.json"
        timings = raw / "epoch_timings.csv"
        if not manifest.exists() or not timings.exists():
            continue
        try:
            meta = json.loads(manifest.read_text())
            if meta.get("state") != "complete":
                continue
            epochs = pd.read_csv(timings)
            rows = int(epochs.environment_rows.sum())
            workers = int(meta.get("actual_alignment_workers", 0))
            rocs_seconds = float(epochs.rocs_seconds.sum())
            if rows <= 0 or workers <= 0 or rocs_seconds <= 0:
                continue
            return dict(
                provenance=(
                    f"{meta.get('host', 'unknown host')} live RL probe "
                    f"({name}), {rows} scored rows, {workers} workers"
                ),
                serial_equivalent_seconds_per_row=rocs_seconds * workers / rows,
                wall_seconds_per_row=rocs_seconds / rows,
                baseline=meta.get("baseline"),
            )
        except (ValueError, KeyError, OSError):
            continue
    return None


def _component_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Resolve per-row generation and alignment/I/O timing column names."""
    generation = next(
        (
            name
            for name in (
                "generation_minimization_seconds",
                "generation_seconds",
            )
            if name in frame.columns
        ),
        None,
    )
    alignment = (
        "alignment_io_seconds" if "alignment_io_seconds" in frame.columns else None
    )
    return generation, alignment


def render_reward_pareto(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    *,
    labels: dict[str, str],
    keys: tuple[str, ...],
    warning: str = "",
) -> None:
    """Render Pareto score-vs-time frontier and signed anchor gaps.

    The x-axis uses the per-input **p95** scoring seconds (straggler-aware:
    an RL epoch's wall-clock is bounded by its slowest worker chunk), with the
    mean shown as faint companion markers. Per-input times are isolated serial
    single-molecule calls and systematically underestimate live RL cost; the
    measured live-RL reference and its calibration-vs-RL factor are printed in
    the banner and a top axis projects epoch minutes for a canonical sprint
    configuration (p95 x 500 samples / 4 workers).
    """
    # Merge per-row component timings (older runs name the column
    # generation_seconds) so the cost structure is visible even when the saved
    # summary predates those columns.
    generation, alignment = _component_columns(frame)
    group_keys = list(keys[1:])
    if generation and alignment:
        components = frame.groupby(group_keys, sort=True).agg(
            mean_generation_minimization_seconds=(generation, "mean"),
            mean_alignment_io_seconds=(alignment, "mean"),
        )
        for column in (
            "mean_generation_minimization_seconds",
            "mean_alignment_io_seconds",
        ):
            if column not in summary.columns:
                summary = summary.merge(
                    components[[column]], left_on=group_keys, right_index=True, how="left"
                )
    dominated = []
    for _, row in summary.iterrows():
        other = (
            summary.p95_seconds.le(row.p95_seconds)
            & summary["mean"].ge(row["mean"])
            & (
                summary.p95_seconds.lt(row.p95_seconds)
                | summary["mean"].gt(row["mean"])
            )
        )
        dominated.append(bool(other.any()))
    summary = summary.assign(pareto=~np.array(dominated))
    summary.to_csv(output / "raw_data/development/pareto.csv", index=False)
    anchor_attempts = int(frame.candidate_attempts.max())
    anchor = frame[
        frame.reference_cap.eq(0) & frame.candidate_attempts.eq(anchor_attempts)
    ][["molecule_id", "method", "score"]]
    paired = frame.merge(
        anchor.rename(columns={"score": "anchor_score"}),
        on=["molecule_id", "method"],
        validate="many_to_one",
    )
    paired["score_gap"] = paired.anchor_score - paired.score
    paired.to_csv(output / "raw_data/development/paired_anchor.csv.gz", index=False)
    gaps = (
        paired[~paired.color_excluded]
        .groupby(list(keys[1:]))
        .score_gap.agg(mean_gap="mean", p95_gap=lambda s: s.quantile(0.95))
        .reset_index()
    )
    summary = summary.merge(gaps, on=list(keys[1:]))

    reference = rl_reference_measurement(output) or FALLBACK_RL_REFERENCE
    baseline = reference.get("baseline")
    factor = reference.get("factor")
    if factor is None and baseline:
        cell = summary[
            summary.method.eq(baseline.get("method"))
            & summary.reference_cap.eq(baseline.get("reference_cap"))
            & summary.candidate_attempts.eq(baseline.get("candidate_attempts"))
        ]
        calibration_mean = float(cell.mean_seconds.iloc[0]) if len(cell) else None
        if calibration_mean:
            factor = reference["serial_equivalent_seconds_per_row"] / calibration_mean
    factor_text = (
        f"{factor:.1f}x" if factor else "unknown (no matching calibration cell)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    frontier, gap_axis, structure = axes
    method_colors = dict(zip(summary.method.unique(), plt.cm.tab10.colors))
    for method, group in summary.groupby("method"):
        color = method_colors[method]
        frontier.scatter(
            group.mean_seconds,
            group["mean"],
            s=18,
            alpha=0.30,
            color=color,
            marker="v",
        )
        points = frontier.scatter(
            group.p95_seconds, group["mean"], s=30, alpha=0.65, label=labels[method]
        )
        color = points.get_facecolors()[0]
        front = group[group.pareto].sort_values("p95_seconds")
        frontier.scatter(
            front.p95_seconds,
            front["mean"],
            s=100,
            facecolors="none",
            edgecolors=[color],
            linewidths=2,
        )
        for _, row in front.iterrows():
            cap = "all" if row.reference_cap == 0 else str(int(row.reference_cap))
            gen_share = ""
            if "mean_generation_minimization_seconds" in row and pd.notna(
                row.get("mean_generation_minimization_seconds")
            ):
                total = (
                    row.mean_generation_minimization_seconds
                    + row.mean_alignment_io_seconds
                )
                if total > 0:
                    gen_share = (
                        f" · {row.mean_generation_minimization_seconds / total:.0%} gen"
                    )
            frontier.annotate(
                f"K{cap}/C{int(row.candidate_attempts)}{gen_share}",
                (row.p95_seconds, row["mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        gap_axis.scatter(
            group.p95_seconds,
            group.p95_gap,
            s=30,
            color=method_colors[method],
            alpha=0.65,
            label=labels[method],
        )
    frontier.axhline(
        0.95,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Chosen mean target 0.95",
    )
    frontier.set_ylabel("Mean self-recovery ROCS")
    frontier.set_title(
        "Score-time frontier on p95 seconds (straggler-aware);\n"
        "faint triangles mark the mean, circles mark nondominated cells"
    )
    gap_axis.set_ylabel(
        f"95th-percentile signed score gap to same-method C{anchor_attempts}/all"
    )
    gap_axis.set_title("Large-score-loss tail relative to a finite sampled anchor")

    # Top axis: projected RL epoch minutes for a canonical sprint configuration.
    scale = RL_PROJECTION_SAMPLES / RL_PROJECTION_WORKERS / 60.0
    projected = frontier.secondary_xaxis(
        "top", functions=(lambda s: s * scale, lambda m: m / scale)
    )
    projected.set_xlabel(
        f"Projected RL epoch minutes (p95 x {RL_PROJECTION_SAMPLES} samples"
        f" / {RL_PROJECTION_WORKERS} workers; apply the {factor_text} factor above)"
    )
    for ax in (frontier, gap_axis):
        ax.set_xscale("log")
        ax.set_xlabel(
            "Per-input scoring seconds (p95) - isolated serial calls, NOT RL time"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    # Cost structure of the nondominated cells: candidate generation and
    # minimization versus ROCS alignment plus scorer I/O.
    front_cells = summary[summary.pareto].sort_values("mean_seconds")
    if {"mean_generation_minimization_seconds", "mean_alignment_io_seconds"} <= set(
        front_cells.columns
    ):
        labels_y = [
            f"{labels[row.method]} K"
            f"{'all' if row.reference_cap == 0 else int(row.reference_cap)}"
            f"/C{int(row.candidate_attempts)}"
            for row in front_cells.itertuples()
        ]
        y_pos = np.arange(len(front_cells))
        structure.barh(
            y_pos,
            front_cells.mean_generation_minimization_seconds,
            color="#31688e",
            label="Candidate generation + minimization",
        )
        structure.barh(
            y_pos,
            front_cells.mean_alignment_io_seconds,
            left=front_cells.mean_generation_minimization_seconds,
            color="#35b779",
            label="ROCS alignment + scorer I/O",
        )
        structure.set_yticks(y_pos, labels_y, fontsize=8)
        structure.set_xlabel("Mean per-input seconds (serial component sum)")
        structure.set_title("Cost structure of nondominated cells")
        structure.legend(fontsize=9)
        structure.grid(alpha=0.2, axis="x")
    fig.suptitle(
        f"{warning}Development-panel trade-offs · lower time and lower score gap preferred\n"
        f"Per-input times are isolated serial single-molecule calls, not RL epoch time · "
        f"measured live-RL reference: {reference['provenance']}, "
        f"{reference['serial_equivalent_seconds_per_row']:.2f} s/row serial-equivalent "
        f"-> calibration underestimates RL by ~{factor_text}\n"
        "Reference/candidate methods matched; this is not biological accuracy or RL improvement",
        fontsize=13,
    )
    fig.savefig(output / "pareto.png", dpi=170)
    fig.savefig(output / "pareto.svg")
    plt.close(fig)


def render_rl_timing(output: Path, *, labels: dict[str, str]) -> None:
    """Render the 5-epoch live RL cost attribution and fixed-batch replay comparison."""
    import json

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
    bar_labels = []
    for _, row in replay.iterrows():
        cap = "all" if row.reference_cap == 0 else str(int(row.reference_cap))
        bar_labels.append(
            f"{labels[row.method]}\nK{cap}/C{int(row.candidate_attempts)}\n{row.engine}, pass{int(row['repeat']) + 1}"
        )
    per_input = replay.seconds / replay.inputs
    bars = right.bar(range(len(replay)), per_input, color="#31688e")
    right.bar_label(
        bars,
        labels=[
            f"{t:.2f}s\nΔ {delta:+.3f}"
            for t, delta in zip(per_input, replay.mean_change)
        ],
        padding=4,
        fontsize=8,
    )
    right.set_xticks(range(len(replay)), bar_labels, rotation=35, ha="right", fontsize=8)
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

