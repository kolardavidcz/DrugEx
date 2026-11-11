"""The figure suite. One module so every plot shares a palette, a style and an arm order.

Numbering follows the earlier tree's suite so a reader who knows those figures finds the
same ones here. Gaps in the numbers are deliberate: they were retired, and renumbering
would break the correspondence.

Palette: the first four categorical slots of the reference design system, in fixed order.
Validated with the dataviz validator at the light surface on the *adjacent* pairlist --
lightness band, chroma floor, CVD separation (worst adjacent dE 9.1 protan) and the
normal-vision floor (22.9) all pass. The same four colours do NOT clear the floors on the
all-pairs list, which is what a four-series scatter would need, so the scatter is drawn as
small multiples instead of four overlaid series. Aqua and yellow fall below 3:1 contrast
against the surface, so every figure that uses them also carries direct labels, and
``report.md`` repeats the numbers as a table.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

#: Arms in reporting order: the baseline first, then what training produced.
ARMS = ("prior", "desired_ratio", "desired_ratio_at_stop", "gmean")

#: Fixed categorical order -- a colour belongs to an arm, never to its rank in a filter.
PALETTE = {
    "prior": "#2a78d6",
    "desired_ratio": "#eb6834",
    "gmean": "#1baf7a",
    "desired_ratio_at_stop": "#eda100",
}

#: Text and chrome. Series colours never carry text; ink does.
INK = "#1a1a19"
MUTED = "#6f6e69"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

#: Molecules drawn per arm in the structure gallery.
TOP_N = 12

#: Leads overlaid on the reference in the 3D figure.
TOP_3D = 4


def style() -> None:
    """Recessive chrome, ink text, no chartjunk."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _valid(arm: pd.DataFrame) -> pd.DataFrame:
    return arm[arm["Valid"] == 1]


def _unique(arm: pd.DataFrame) -> pd.DataFrame:
    """Valid rows, one per distinct molecule -- by structure, not by spelling."""
    valid = _valid(arm)
    return valid.drop_duplicates(subset="Canonical" if "Canonical" in valid else "SMILES")


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _short(objective: str) -> str:
    """Axis-friendly objective name, carrying its scale.

    An axis reading just "SA" on a plot bounded at 1.0 invites the reader to take it for
    raw synthetic accessibility on [1, 10], where lower is better. It is the ClippedScore
    desirability, on [0, 1], where higher is better -- the opposite direction.
    """
    if objective.startswith("RDKit_"):
        return f"{objective.split('_')[-1]} [0, 2]"
    if objective == "SA":
        return "SA desirability [0, 1], higher = easier"
    if objective == "QED":
        return "QED [0, 1]"
    return objective


def _mols(smiles: pd.Series) -> list[Chem.Mol]:
    return [m for m in (Chem.MolFromSmiles(s) for s in smiles if isinstance(s, str)) if m]


# --------------------------------------------------------------------------------------
# 01 / 01b -- what training changed about the output distribution
# --------------------------------------------------------------------------------------


def before_after_distributions(
    arms: dict[str, pd.DataFrame], objectives: list[str], thresholds: list[float], out: Path
) -> Path:
    """Per-objective distributions, every arm against the prior baseline."""
    style()
    fig, axes = plt.subplots(len(objectives), 1, figsize=(9, 3.1 * len(objectives)), squeeze=False)
    for axis, objective, threshold in zip(axes[:, 0], objectives, thresholds):
        # _unique, not _valid: every arm table and figures 05/06/07 deduplicate, and a
        # policy collapsed onto 20 molecules emitted 250x each would otherwise draw a
        # sharp peak indistinguishable from genuinely improved chemistry -- in the very
        # figure a reader uses to judge whether training worked.
        pool = [
            _unique(arms[name])[objective].to_numpy(dtype=float) for name in ARMS if name in arms
        ]
        pool = [values[np.isfinite(values)] for values in pool]
        finite = np.concatenate([v for v in pool if len(v)]) if any(len(v) for v in pool) else None
        if finite is None or not finite.size:
            axis.text(0.5, 0.5, "no valid molecules", ha="center", transform=axis.transAxes)
            continue
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        for name, values in zip([n for n in ARMS if n in arms], pool):
            if len(values):
                axis.hist(
                    values,
                    bins=40,
                    range=(low, high),
                    density=True,
                    histtype="step",
                    linewidth=2,
                    color=PALETTE[name],
                    label=name,
                )
        axis.axvline(threshold, color=MUTED, linestyle="--", linewidth=1)
        axis.annotate(
            f"threshold {threshold:.3g}",
            xy=(threshold, axis.get_ylim()[1]),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
        axis.set(xlabel=_short(objective), ylabel="density")
        axis.grid(axis="y")
        axis.legend(loc="upper left", fontsize=9)
    fig.suptitle("01 - objective distributions, prior vs trained arms", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "01_before_after_distributions.png")


def before_after_final_epoch(
    prior: pd.DataFrame, epochs: pd.DataFrame, objectives: list[str], out: Path
) -> Path | None:
    """The prior against the policy's own last training epoch, not a checkpoint."""
    if epochs is None or epochs.empty:
        return None
    style()
    last = int(epochs["Epoch"].max())
    tail = _valid(epochs[epochs["Epoch"] == last])
    fig, axes = plt.subplots(
        1, len(objectives), figsize=(5.0 * len(objectives), 3.6), squeeze=False
    )
    for axis, objective in zip(axes[0], objectives):
        for frame, name, colour in (
            (_valid(prior), "prior", PALETTE["prior"]),
            (tail, f"epoch {last}", PALETTE["desired_ratio"]),
        ):
            values = frame[objective].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                axis.hist(
                    values,
                    bins=25,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    color=colour,
                    label=f"{name} (n={len(values)})",
                )
        axis.set(xlabel=_short(objective), ylabel="density")
        axis.grid(axis="y")
        axis.legend(fontsize=9)
    fig.suptitle(
        f"01b - prior vs the last training epoch ({last}); the checkpoints are elsewhere",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save(fig, out, "01b_before_after_final_epoch.png")


# --------------------------------------------------------------------------------------
# 02 -- the objective trade-off, faceted
# --------------------------------------------------------------------------------------


def multiobjective_tradeoffs(
    arms: dict[str, pd.DataFrame], objectives: list[str], thresholds: list[float], out: Path
) -> Path | None:
    """One panel per arm. Four overlaid series would not clear the all-pairs colour floor."""
    if len(objectives) < 2:
        return None
    style()
    x, y = objectives[0], objectives[1]
    present = [name for name in ARMS if name in arms]
    fig, axes = plt.subplots(
        1, len(present), figsize=(3.7 * len(present), 3.8), squeeze=False, sharex=True, sharey=True
    )
    for axis, name in zip(axes[0], present):
        frame = _valid(arms[name])
        axis.scatter(
            frame[x], frame[y], s=14, alpha=0.5, linewidths=0, color=PALETTE[name], zorder=2
        )
        # Unique desired over all sampled molecules: the same rule report.md and DrugEx
        # use, so a panel title can never disagree with the table.
        unique = _unique(arms[name])
        desired = int(unique["Desired"].sum()) if "Desired" in unique else 0
        axis.set_title(f"{name}\nn={len(arms[name])}, desired={desired}", fontsize=10)
        axis.set_xlabel(_short(x))
        axis.grid(True)
    axes[0][0].set_ylabel(_short(y))

    # Shade AFTER plotting, bounded by the data limits. `Desired` needs every objective at
    # or above its threshold, so the region is the top-right rectangle -- not two crossing
    # bands, which would shade molecules that clear only one of them.
    left, right = axes[0][0].get_xlim()
    bottom, top = axes[0][0].get_ylim()
    for axis in axes[0]:
        axis.add_patch(
            Rectangle(
                (thresholds[0], thresholds[1]),
                max(right - thresholds[0], 0),
                max(top - thresholds[1], 0),
                facecolor=GRID,
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            )
        )
        axis.axvline(thresholds[0], color=MUTED, linestyle="--", linewidth=1, zorder=1)
        axis.axhline(thresholds[1], color=MUTED, linestyle="--", linewidth=1, zorder=1)
        axis.set_xlim(left, right)
        axis.set_ylim(bottom, top)
    # With a third objective the shaded corner shows only two of the conditions, so a
    # point inside it can still fail the one not drawn.
    caption = (
        "02 - objective trade-off; shaded corner is the desirability region (all objectives met)"
        if len(objectives) == 2
        else f"02 - objective trade-off; shaded corner shows {x} and {y} only, and "
        f"`Desired` also requires {', '.join(objectives[2:])}"
    )
    fig.suptitle(caption, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "02_multiobjective_tradeoffs.png")


# --------------------------------------------------------------------------------------
# 03 / 12 -- what happened during training
# --------------------------------------------------------------------------------------


def training_trajectories(fit: pd.DataFrame, summary: dict, out: Path) -> Path | None:
    """Sample quality per epoch, with the epoch each checkpoint was taken from marked."""
    if fit is None or fit.empty:
        return None
    style()
    split = [c for c in ("avg_ShapeTanimoto", "avg_ColorTanimoto") if c in fit]
    panels = 3 if split else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.0 * panels, 4.0))
    for metric, colour in (
        ("valid_ratio", PALETTE["prior"]),
        ("unique_ratio", PALETTE["gmean"]),
        ("desired_ratio", PALETTE["desired_ratio"]),
    ):
        if metric in fit:
            axes[0].plot(fit["Epoch"], fit[metric], linewidth=2, color=colour, label=metric)
    axes[0].set(xlabel="epoch", ylim=(-0.02, 1.02), title="sample quality")
    if "avg_gmean" in fit:
        axes[1].plot(
            fit["Epoch"], fit["avg_gmean"], linewidth=2, color=PALETTE["gmean"], label="avg_gmean"
        )
    axes[1].set(xlabel="epoch", title="geometric mean of the objectives")

    if split:
        # Panel D: the two halves of TanimotoCombo. A combo that climbs on shape alone
        # means the policy is matching EGCG's volume and ignoring its pharmacophore --
        # invisible in the combined score, which is why both are plotted.
        for column, colour in zip(split, (PALETTE["prior"], PALETTE["desired_ratio_at_stop"])):
            axes[2].plot(
                fit["Epoch"],
                fit[column],
                linewidth=2,
                color=colour,
                label=column.replace("avg_", ""),
            )
        axes[2].set(xlabel="epoch", title="shape and colour separately")

    # The freeze epoch is set even when the epoch cap arrived first and nothing froze.
    # Labelling that "freeze" would tell a reader the criterion converged when the run was
    # actually cut off while it was still improving.
    froze = bool(summary.get("desired_ratio_frozen"))
    marks = (
        ("best_desired_ratio_epoch", PALETTE["desired_ratio"], "desired_ratio checkpoint"),
        (
            "desired_ratio_freeze_epoch",
            PALETTE["desired_ratio_at_stop"],
            "freeze (at_stop)" if froze else "end of run (at_stop)",
        ),
        ("best_avg_gmean_epoch", PALETTE["gmean"], "gmean checkpoint"),
    )
    for axis in axes:
        for key, colour, label in marks:
            epoch = summary.get(key)
            if epoch:
                axis.axvline(epoch, color=colour, linestyle=":", linewidth=1.6, label=label)
        axis.grid(True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("03 - training trajectories", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "03_training_trajectories.png")


def training_diagnostics(fit: pd.DataFrame, epochs: pd.DataFrame, out: Path) -> Path | None:
    """Loss, and the collapse diagnostics a similarity score alone would hide."""
    if fit is None or fit.empty:
        return None
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    if "loss_train" in fit:
        axes[0].plot(fit["Epoch"], fit["loss_train"], linewidth=2, color=PALETTE["prior"])
    axes[0].set(xlabel="epoch", title="policy-gradient loss")

    if epochs is not None and not epochs.empty:
        grouped = epochs.groupby("Epoch")
        key = "Canonical" if "Canonical" in epochs.columns else "SMILES"
        duplicates = 1.0 - grouped[key].nunique() / grouped[key].size()
        axes[1].plot(
            duplicates.index, duplicates.to_numpy(), linewidth=2, color=PALETTE["desired_ratio"]
        )
        axes[1].set(xlabel="epoch", ylim=(-0.02, 1.02), title="duplicate fraction per epoch")
        invalid = 1.0 - grouped["Valid"].mean()
        axes[2].plot(invalid.index, invalid.to_numpy(), linewidth=2, color=PALETTE["gmean"])
        axes[2].set(xlabel="epoch", ylim=(-0.02, 1.02), title="invalid fraction per epoch")
    for axis in axes:
        axis.grid(True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(
        "12 - training diagnostics; a rising duplicate fraction is mode collapse", x=0.01, ha="left"
    )
    fig.tight_layout()
    return _save(fig, out, "12_training_diagnostics.png")


# --------------------------------------------------------------------------------------
# 05 -- what the molecules actually look like
# --------------------------------------------------------------------------------------


def _grid(frame: pd.DataFrame, objectives: list[str]) -> tuple[list, list[str]]:
    """The arm's best unique molecules as (structures, legends), ranked by objective one.

    Built together, never sliced apart: RDKit silently returns None for a SMILES it
    cannot parse, so taking the first ``len(mols)`` legends would shift every score after
    the gap onto the wrong structure.
    """
    mols, legends = [], []
    for _, row in _unique(frame).nlargest(TOP_N, objectives[0]).iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"]) if isinstance(row["SMILES"], str) else None
        if mol is None:
            continue
        mols.append(mol)
        legends.append(" ".join(f"{_short(o).split(' ')[0]}={row[o]:.2f}" for o in objectives))
    return mols, legends


def _captioned(image, caption: str):
    """The grid with a caption bar above it; the bare grid if text cannot be drawn.

    A gallery carries no axes, so without this the only statement of *which model drew
    these molecules* is the file name -- and a figure gets separated from its file name
    the moment it is pasted into a document.
    """
    try:
        from matplotlib import font_manager
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 19)
        bar = 34
        canvas = Image.new("RGB", (image.width, image.height + bar), SURFACE)
        canvas.paste(image, (0, bar))
        ImageDraw.Draw(canvas).text((10, 7), caption, fill=INK, font=font)
        return canvas
    except Exception as error:  # a caption must never cost the figure
        logger.warning("could not caption the gallery (%s); writing it bare", error)
        return image


def best_arm(arms: dict[str, pd.DataFrame], objectives: list[str]) -> str | None:
    """The trained checkpoint a reader would download: highest desired ratio, then median.

    "Best model" has to be *measured*, not named in advance. The three checkpoints are
    kept by different criteria, and which one wins is the question the run answers --
    hard-coding `gmean` here would quietly answer it in the figure instead.

    The ratio is computed exactly as the arm table computes it (unique molecules over
    every sampled row), so the figure and the table can never disagree about which model
    was best. `prior` is excluded: it is the untrained baseline, not a checkpoint.
    """
    best, ranked = None, (-1.0, -1.0)
    for name in ARMS:
        arm = arms.get(name)
        if name == "prior" or arm is None or not len(arm) or _unique(arm).empty:
            continue
        unique = _unique(arm)
        score = (
            float(unique["Desired"].sum()) / len(arm) if "Desired" in unique else 0.0,
            float(unique[objectives[0]].median()) if len(unique) else 0.0,
        )
        if score > ranked:
            best, ranked = name, score
    return best


def lead_structures_gallery(
    arms: dict[str, pd.DataFrame], objectives: list[str], out: Path
) -> list[Path]:
    """A structure grid per arm, ranked by the first objective."""
    written = []
    for name in ARMS:
        frame = arms.get(name)
        if frame is None:
            continue
        mols, legends = _grid(frame, objectives)
        if not mols:
            continue
        image = Draw.MolsToGridImage(
            mols, molsPerRow=4, subImgSize=(300, 230), legends=legends, returnPNG=False
        )
        path = out / f"05_lead_structures_gallery_{name}.png"
        image.save(path)
        written.append(path)
    return written


def best_model_leads(
    arms: dict[str, pd.DataFrame], objectives: list[str], out: Path
) -> Path | None:
    """05b -- the same leads, from the one model that could actually be shipped.

    05 draws four grids and leaves the comparison to the reader, with the untrained prior
    among them. This is the single grid to put in a results chapter: the best molecules
    of the best checkpoint, which is also the only set of molecules anyone can reproduce
    by loading one ``.pkg`` and sampling it. The arm it came from is in the caption and
    in the file name, because "the top leads" means nothing without the model that drew
    them.
    """
    name = best_arm(arms, objectives)
    if name is None:
        return None
    frame = arms[name]
    mols, legends = _grid(frame, objectives)
    if not mols:
        return None
    unique = _unique(frame)
    desired = float(unique["Desired"].sum()) / len(frame) if "Desired" in unique else 0.0
    image = Draw.MolsToGridImage(
        mols, molsPerRow=4, subImgSize=(300, 230), legends=legends, returnPNG=False
    )
    path = out / f"05b_top_leads_best_model_{name}.png"
    _captioned(
        image,
        f"05b - top {len(mols)} leads sampled from the best model: {name} "
        f"(desired_ratio {desired:.3f}, n={len(frame)}), ranked by {_short(objectives[0])}",
    ).save(path)
    return path


# --------------------------------------------------------------------------------------
# 06 -- do the objectives fight each other
# --------------------------------------------------------------------------------------


def objective_conflict_correlations(
    arms: dict[str, pd.DataFrame], objectives: list[str], out: Path
) -> Path | None:
    """Spearman correlation between objectives, per arm. Negative means they conflict."""
    if len(objectives) < 2:
        return None
    style()
    pairs = [(a, b) for i, a in enumerate(objectives) for b in objectives[i + 1 :]]
    present = [name for name in ARMS if name in arms]
    width = np.arange(len(pairs))
    span = 0.8 / max(len(present), 1)
    fig, axis = plt.subplots(figsize=(1.6 + 2.6 * len(pairs), 4.0))
    for index, name in enumerate(present):
        frame = _unique(arms[name])
        values, labels = [], []
        for a, b in pairs:
            # A saturated column has zero variance, so rho is undefined. Plotting 0.0
            # would read as "these objectives do not conflict" instead of "this cannot
            # be computed" -- which is exactly what a saturated SA column produces.
            if len(frame) > 2 and frame[a].std() > 0 and frame[b].std() > 0:
                rho = float(frame[a].corr(frame[b], method="spearman"))
                values.append(rho)
                labels.append(f"{rho:.2f}")
            else:
                values.append(0.0)
                labels.append("undefined")
        offset = width + index * span - 0.4 + span / 2
        bars = axis.bar(offset, values, span * 0.9, color=PALETTE[name], label=name)
        axis.bar_label(bars, labels=labels, fontsize=8, color=INK, padding=2)
    axis.axhline(0, color=MUTED, linewidth=1)
    axis.set_xticks(width, [f"{_short(a)}\nvs {_short(b)}" for a, b in pairs])
    axis.set(ylabel="Spearman rho", ylim=(-1.05, 1.05))
    axis.grid(axis="y")
    axis.legend(fontsize=9)
    fig.suptitle(
        "06 - objective conflict; rho < 0 means improving one costs the other", x=0.01, ha="left"
    )
    fig.tight_layout()
    return _save(fig, out, "06_objective_conflict_correlations.png")


# --------------------------------------------------------------------------------------
# 07 -- is the output diverse, or one scaffold repeated
# --------------------------------------------------------------------------------------


def scaffold_diversity(arms: dict[str, pd.DataFrame], out: Path) -> Path | None:
    """Bemis-Murcko scaffold counts. A collapsing run finds one scaffold and stays."""
    style()
    present = [name for name in ARMS if name in arms]
    rows = {}
    for name in present:
        frame = _unique(arms[name])
        scaffolds = []
        for mol in _mols(frame["SMILES"]):
            try:
                scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
            except Exception:
                continue
        if not scaffolds:
            continue
        series = pd.Series(scaffolds)
        rows[name] = {
            "molecules": len(scaffolds),
            "scaffolds": series.nunique(),
            "unique_fraction": series.nunique() / len(scaffolds),
            "top_share": series.value_counts().iloc[0] / len(scaffolds),
        }
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    names = list(rows)
    for axis, key, title in (
        (axes[0], "unique_fraction", "unique scaffolds / unique molecules"),
        (axes[1], "top_share", "share held by the single commonest scaffold"),
    ):
        bars = axis.bar(
            names, [rows[n][key] for n in names], color=[PALETTE[n] for n in names], width=0.6
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=9, color=INK, padding=2)
        axis.set(ylim=(0, 1.08), title=title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y")
    fig.suptitle(
        "07 - Bemis-Murcko scaffold diversity; low left bar or high right bar is collapse",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save(fig, out, "07_bemis_murcko_scaffold_diversity.png")


# --------------------------------------------------------------------------------------
# 10 -- do the leads actually sit on the reference in 3D
# --------------------------------------------------------------------------------------


def lead_to_seed_superposition(
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    reference: Path,
    out: Path,
    generator=None,
) -> Path | None:
    """Overlay the best leads on the reference and project the result onto three planes.

    The aligned poses are also written as an SDF, because a projection is a sanity check
    and a real inspection happens in PyMOL.
    """
    if generator is None or not reference.is_file():
        return None
    refs = [m for m in Chem.SDMolSupplier(str(reference), removeHs=False) if m is not None]
    if not refs:
        return None
    seed = refs[0]

    frame = arms.get("gmean", arms.get("prior"))
    if frame is None:
        return None
    best = _unique(frame).nlargest(TOP_3D, objectives[0])
    aligned: list[tuple[str, Chem.Mol, float]] = []
    import tempfile

    from rdkit.Chem import rdShapeAlign

    for smiles, score in zip(best["SMILES"], best[objectives[0]]):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = generator.genConformers([smiles], tmp)
                poses = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
            if not poses:
                continue
            probe = Chem.Mol(poses[0])
            rdShapeAlign.AlignMol(seed, probe, useColors=True)
            aligned.append((smiles, probe, float(score)))
        except Exception as error:  # a figure must never fail a run
            logger.warning("3D overlay failed for %s: %s", smiles, error)
    if not aligned:
        return None

    writer = Chem.SDWriter(str(out / "top_leads_aligned_to_seed.sdf"))
    for smiles, mol, score in aligned:
        mol.SetProp("_Name", smiles)
        # Named so it cannot be read as this pose's score: the pose here is a single
        # re-alignment, measured up to 0.38 TanimotoCombo below the value the pipeline's
        # full search found.
        mol.SetDoubleProp(f"pipeline_{objectives[0]}", score)
        mol.SetProp("pose_note", "illustrative re-alignment, not the scored pose")
        writer.write(mol)
    writer.close()

    style()
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    seed_xyz = Chem.RemoveHs(seed).GetConformer().GetPositions()
    for axis, (i, j, xi, yj) in zip(axes, planes):
        axis.scatter(
            seed_xyz[:, i],
            seed_xyz[:, j],
            s=70,
            color=MUTED,
            alpha=0.75,
            label="reference (EGCG)",
            zorder=3,
        )
        for index, (_, mol, score) in enumerate(aligned):
            xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
            axis.scatter(
                xyz[:, i],
                xyz[:, j],
                s=26,
                alpha=0.65,
                linewidths=0,
                color=list(PALETTE.values())[index % len(PALETTE)],
                label=f"lead {index + 1}",
            )
        axis.set(xlabel=f"{xi} (A)", ylabel=f"{yj} (A)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True)
    axes[0].legend(fontsize=8, loc="best")
    # The scores are deliberately NOT in the legend. This overlay is a single
    # AlignMol call on the first conformer against the first reference pose; the
    # pipeline's score is a maximum over every conformer pair, four axis-sign starts and
    # the whole bank. Printing that score next to this pose would attribute the best of
    # ~1000 alignments to one arbitrary one. The ranking is honest, the geometry is
    # illustrative, and top_leads_aligned_to_seed.sdf is there for real inspection.
    fig.suptitle(
        "10 - best leads, re-aligned onto the reference for illustration "
        "(not the scored pose); ranked by " + _short(objectives[0]),
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save(fig, out, "10_lead_to_seed_3d_superposition.png")


# --------------------------------------------------------------------------------------
# 11 -- property drift: the figure that catches a fragment-collapse "win"
# --------------------------------------------------------------------------------------


def epoch_property_trajectories(
    epochs: pd.DataFrame, prior: pd.DataFrame, out: Path
) -> Path | None:
    """Molecular properties per epoch against the prior's band.

    A shape score rises reliably when molecules shrink. Without this figure a collapse to
    small fragments reads as a similarity win.
    """
    if epochs is None or epochs.empty:
        return None
    style()
    properties = {
        "heavy atoms": lambda m: m.GetNumHeavyAtoms(),
        "molecular weight": Descriptors.MolWt,
        "cLogP": Descriptors.MolLogP,
        "TPSA": rdMolDescriptors.CalcTPSA,
    }
    rows = []
    for epoch, group in _valid(epochs).groupby("Epoch"):
        for mol in _mols(group["SMILES"]):
            row = {"Epoch": epoch}
            for label, fn in properties.items():
                try:
                    row[label] = float(fn(mol))
                except Exception:
                    row[label] = math.nan
            rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows)

    baseline = {}
    for mol in _mols(_valid(prior)["SMILES"]):
        for label, fn in properties.items():
            try:
                baseline.setdefault(label, []).append(float(fn(mol)))
            except Exception:
                continue

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, label in zip(axes.ravel(), properties):
        grouped = table.groupby("Epoch")[label]
        median, lo, hi = grouped.median(), grouped.quantile(0.1), grouped.quantile(0.9)
        axis.fill_between(
            median.index, lo, hi, color=PALETTE["desired_ratio"], alpha=0.18, linewidth=0
        )
        axis.plot(
            median.index,
            median,
            linewidth=2,
            color=PALETTE["desired_ratio"],
            label="training p10-p90",
        )
        if baseline.get(label):
            values = np.asarray(baseline[label], dtype=float)
            axis.axhline(
                float(np.median(values)),
                color=PALETTE["prior"],
                linestyle="--",
                linewidth=1.6,
                label="prior median",
            )
        axis.set(xlabel="epoch", ylabel=label, title=label)
        axis.grid(True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(fontsize=8)
    fig.suptitle(
        "11 - property drift; a falling heavy-atom count under a rising shape score is collapse",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    return _save(fig, out, "11_epoch_property_trajectories.png")


# --------------------------------------------------------------------------------------


def render_all(
    *,
    arms: dict[str, pd.DataFrame],
    epochs: pd.DataFrame | None,
    fit: pd.DataFrame | None,
    objectives: list[str],
    thresholds: list[float],
    summary: dict,
    reference: Path,
    out: Path,
    generator=None,
) -> list[Path]:
    """Every figure. A failure in one is logged and never stops the rest."""
    out.mkdir(parents=True, exist_ok=True)
    prior = arms.get("prior")
    jobs = [
        ("01", lambda: before_after_distributions(arms, objectives, thresholds, out)),
        ("01b", lambda: before_after_final_epoch(prior, epochs, objectives, out)),
        ("02", lambda: multiobjective_tradeoffs(arms, objectives, thresholds, out)),
        ("03", lambda: training_trajectories(fit, summary, out)),
        ("05", lambda: lead_structures_gallery(arms, objectives, out)),
        ("05b", lambda: best_model_leads(arms, objectives, out)),
        ("06", lambda: objective_conflict_correlations(arms, objectives, out)),
        ("07", lambda: scaffold_diversity(arms, out)),
        ("10", lambda: lead_to_seed_superposition(arms, objectives, reference, out, generator)),
        ("11", lambda: epoch_property_trajectories(epochs, prior, out)),
        ("12", lambda: training_diagnostics(fit, epochs, out)),
    ]
    written: list[Path] = []
    for number, job in jobs:
        try:
            result = job()
        except Exception:
            logger.exception("figure %s failed", number)
            continue
        if isinstance(result, list):
            written.extend(result)
        elif result is not None:
            written.append(result)
    return written
