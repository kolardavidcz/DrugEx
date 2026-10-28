"""Conformer-study figures; RL analysis remains in code.visualization."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
