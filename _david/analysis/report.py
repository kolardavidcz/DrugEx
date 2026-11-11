"""Figures and a summary table for a finished run.

``python -m _david.analysis <run_dir>`` reads only what the run wrote, so it can be
re-run on any old run directory and never touches training. Every number comes from a file
in that directory; nothing is rescored here, because rescoring with a setting other than
the one the run used is how two different numbers for one molecule happen.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from _david.analysis import figures as figures_module
from _david.analysis._helpers import rocs_scale

logger = logging.getLogger(__name__)

#: Arms in the order they are reported: the baseline first, then what training produced.
ARM_FILES = (
    ("prior", "prior.tsv"),
    ("desired_ratio", "final_desired_ratio.tsv"),
    ("desired_ratio_at_stop", "final_desired_ratio_at_stop.tsv"),
    ("gmean", "final_gmean.tsv"),
)

#: Molecules drawn per arm in the structure grid.
TOP_N = 12


def _configurable(config: dict) -> dict:
    """Only the keys ``Config`` actually declares, coerced back from the JSON manifest."""
    import dataclasses

    from _david.config import Config

    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for key, value in config.items():
        if key not in fields or value is None:
            continue
        if key in {"output_root"}:
            continue
        out[key] = tuple(value) if isinstance(value, list) else value
    return out


def _read_arm(path: Path) -> pd.DataFrame | None:
    """One scored arm, or None when the run did not write it."""
    if not path.is_file():
        logger.info("no %s; skipping it", path.name)
        return None
    return pd.read_csv(path, sep="\t")


def _dedup_key(frame: pd.DataFrame) -> str:
    """Prefer the canonical column; older runs only have the generated spelling."""
    return "Canonical" if "Canonical" in frame.columns else "SMILES"


def _valid(arm: pd.DataFrame) -> pd.DataFrame:
    return arm[arm["Valid"] == 1]


def _table(rows: dict[str, dict], first_column: str) -> str:
    """Markdown table, one row per key of ``rows``. Empty input says so rather than raising."""
    if not rows:
        return "_none_"
    columns = list(next(iter(rows.values())).keys())
    lines = [
        "| " + first_column + " | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def _pairs(mapping: dict, key_header: str, value_header: str = "value") -> str:
    """Markdown table of one key/value pair per row, for anything with many fields."""
    lines = [f"| {key_header} | {value_header} |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in mapping.items()]
    return "\n".join(lines)


#: Every table a run writes, and what it holds. Order is the order of the index.
TABLES = (
    ("prior.tsv", "the pretrained prior, scored -- the baseline every figure compares against"),
    ("final_desired_ratio.tsv", "the `desired_ratio` checkpoint's final arm, scored"),
    (
        "final_desired_ratio_at_stop.tsv",
        "the live policy at the freeze epoch, scored -- the same epochs, a different state",
    ),
    ("final_gmean.tsv", "the `avg_gmean` checkpoint's final arm, scored"),
    ("rl_fit.tsv", "one row per epoch: the metrics the stop condition and the checkpoints read"),
    ("rl_smiles.tsv", "the held-out evaluation sample of every epoch, scored"),
    (
        "training_samples.tsv",
        "every molecule the policy gradient was computed on; a training population, "
        "not a result, and never pooled into the arm statistics",
    ),
    ("reference.sdf", "the reference conformer bank, exactly as it was scored against"),
    (
        "figures/top_leads_aligned_to_seed.sdf",
        "figure 08's leads as 3D poses -- an illustrative re-alignment, not the scored pose",
    ),
    ("config.json", "every setting this run resolved, with its training summary"),
    ("manifest.json", "provenance: source digests, timings and the relaxation tallies"),
    ("rl_fit.log", "DrugEx's own per-epoch log, unedited"),
    ("run.log", "the run's log, from the first setting resolved to the last figure written"),
)


def _artifacts(run_dir: Path, written: list[Path]) -> str:
    """Link every file this run produced, each labelled by what is in it.

    Built from the figures that were actually written rather than from the list this
    module hopes for, so a figure that failed is absent from the index instead of being
    linked as a file that is not there.
    """
    lines = [f"- [{figures_module.caption(path)}](figures/{path.name})" for path in written]
    lines += [
        f"- [{name}]({name}) -- {what}" for name, what in TABLES if (run_dir / name).is_file()
    ]
    return "\n".join(lines) if lines else "_nothing was written_"


def top_molecules(arm: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """The best unique molecules of one arm by its first objective, for the report table.

    Drawing them is figure 05's job; this only selects, so the table and the gallery can
    never disagree about which molecules are "best".
    """
    return (
        _valid(arm)
        .drop_duplicates(subset=_dedup_key(_valid(arm)))
        .sort_values(objectives[0], ascending=False)
        .head(TOP_N)
    )


def _describe(arm: pd.DataFrame, objectives: list[str], scale: float = 1.0) -> dict:
    """The ratios a thesis table reports, plus each objective's median and p90."""
    valid = _valid(arm)
    unique = valid.drop_duplicates(subset=_dedup_key(valid))  # DrugEx's unique-compound denominator
    # An arm can legitimately have no rows -- run.py keeps going when one arm raises --
    # and dividing by its length would lose the whole report to a ZeroDivisionError that
    # run.py then swallows, leaving figures and no text.
    sampled = max(len(arm), 1)
    row = {
        "n": len(arm),
        "valid": round(len(valid) / sampled, 4),
        "unique": round(len(unique) / sampled, 4),
        "desired": round(float(unique["Desired"].sum()) / sampled, 4),
    }
    # Unique, not merely valid: the ratio columns above already deduplicate, and a row
    # whose ratios describe unique molecules while its median describes a sample would be
    # two statistics on two populations. Under mode collapse the sample median is the
    # score of whichever molecule was drawn most often.
    row["duplicate_fraction"] = round(1.0 - len(unique) / len(valid), 4) if len(valid) else 0.0
    # The ROCS column is stored in reward units; every number a reader sees is published
    # TanimotoCombo on [0, 2]. Only the first objective is ROCS -- SA and QED are already
    # on [0, 1] on both sides of the rescale.
    for index, objective in enumerate(objectives):
        column = pd.to_numeric(unique[objective], errors="coerce")
        if index == 0:
            column = column * scale
        row[f"{objective} median"] = round(float(column.median()), 4)
        row[f"{objective} p90"] = round(float(column.quantile(0.90)), 4)
    if "SA_raw" in unique.columns and not unique.empty:
        raw_sa = pd.to_numeric(unique["SA_raw"], errors="coerce")
        raw_sa = raw_sa[(raw_sa > 0.5) & (raw_sa < 9.99)]
        if len(raw_sa):
            row["SA_raw median"] = round(float(raw_sa.median()), 4)
    return row


def _threshold_source(objective: str, config: dict, from_reference: list[str]) -> str:
    """Where this objective's threshold came from, named in the table itself.

    Order matters and mirrors ``calibrate_thresholds``: fixed thresholds short-circuit
    before anything else is consulted, so testing ``from_reference`` first would label a
    hand-set threshold "reference" and the report would state a provenance the run never
    used.
    """
    if config.get("fixed_thresholds"):
        return "fixed"
    if objective in from_reference:
        return "reference"
    return f"p{config['threshold_percentile']:g}"


def _threshold_prose(
    config: dict, arms: dict, objectives: list[str], thresholds: list[float]
) -> str:
    """How the thresholds were obtained, and whether any of them is a hard gate.

    A percentile that lands on the maximum of its column is not a percentile: it means the
    column is saturated and `Desired` degenerates into "equal to the best value". Saying
    "p85" at that point is false, and it is exactly what the SA desirability does to a
    drug-like prior, so the report has to name it rather than leave it in a log line.
    """
    from_reference = [
        name for name in (config.get("threshold_from_reference") or []) if name in objectives
    ]
    if config.get("fixed_thresholds"):
        from_reference = []  # fixed thresholds short-circuit the reference path entirely
    if from_reference:
        lead = (
            f"Mixed sources. **{', '.join(f'`{n}`' for n in from_reference)}** use the "
            "reference molecule's own score, so `Desired` means 'at least as good as the "
            "reference on this objective'. The rest use "
            f"p{config['threshold_percentile']:g} of {config['n_prior']} molecules sampled "
            "from the pretrained prior."
        )
    elif config.get("fixed_thresholds"):
        lead = (
            "Fixed thresholds, set by hand in the configuration. `desired_ratio` therefore "
            "means 'clears these bars', not 'beats a percentile of the prior'."
        )
    else:
        lead = (
            f"Percentile p{config['threshold_percentile']:g} of {config['n_prior']} molecules "
            "sampled from the pretrained prior."
        )
    notes = []
    prior = arms.get("prior")
    if prior is not None and "SA_raw" in prior.columns:
        valid_raw = prior.loc[prior["Valid"] == 1, "SA_raw"]
        ceiling = (
            float((valid_raw <= config.get("sa_reward_one", 3.0)).mean()) if len(valid_raw) else 0.0
        )
        if ceiling > 0.5:
            sa_effect = (
                "the tied SA term cannot distinguish those molecules; their ranking comes "
                "from the other objectives, under the weights DrugEx computes"
                if config.get("scheme", "PRCD") == "WS"
                else "the Pareto reward is effectively the ROCS ranking"
            )
            notes.append(
                f"**SA barely shapes the reward.** {ceiling:.1%} of the valid prior sits "
                f"at raw SA <= {config.get('sa_reward_one', 3.0):g}, where `ClippedScore` "
                "returns exactly 1.0. Molecules tied at that ceiling cannot dominate one "
                "another on the SA axis and contribute no crowding distance along it, so "
                f"for most of the population {sa_effect}. This is a property of the "
                "*reward* and is not changed by where "
                "the threshold sits: widen the clip (`sa_reward_zero` / `sa_reward_one`) "
                "if SA should be a gradient rather than a near-constant. The `SA_raw` "
                "column shows what the modified column cannot."
            )
    if prior is not None:
        valid = _valid(prior)
        for objective, threshold in zip(objectives, thresholds):
            if objective not in valid or valid.empty:
                continue
            column = valid[objective]
            tied = int((column >= threshold).sum())
            if threshold >= column.max():
                notes.append(
                    f"**`{objective}` is a hard gate, not a percentile**: the threshold "
                    f"{threshold:.4g} equals the highest value in the prior, with "
                    f"{tied}/{len(column)} prior molecules tied there. A molecule clears it "
                    f"only by matching the best value, so this objective contributes no "
                    f"gradation to `Desired`."
                )
    scheme = config.get("scheme", "PRCD")
    if scheme == "WS":
        tail = (
            "**Under `WeightedSum` these thresholds are also part of the reward.** DrugEx "
            "builds each objective's weight from the fraction of the batch below versus "
            "above its own threshold, so unlike every Pareto run, here they do not only "
            "decide `Desired` and the stop condition. The weights are DrugEx's, not this "
            "project's, and they move every epoch with the batch. A weighted *sum* adds "
            "the columns, so range would be leverage -- which is why ROCS reaches the "
            "reward on [0, 1] like SA and QED, rather than on its published [0, 2]."
        )
    else:
        tail = (
            "They reach `Desired`, `desired_ratio` and the stop condition only: the reward "
            "is Pareto ranking, which ignores the thresholds it is handed, so nothing here "
            "weights the policy gradient."
        )
    return "\n\n".join([f"{lead} {tail}", *notes])


def _units(config: dict, objectives: list[str]) -> str:
    """State what the score columns are, because raw and modified look alike in a table."""
    notes = [
        f"- `{objectives[0]}`: raw ROCS TanimotoCombo (shape + colour), the best overlay "
        f"over up to {config.get('max_isomers', '?')} stereoisomers x "
        f"{config.get('max_conformers', '?')} conformers x the reference bank "
        f"({config.get('reference_conformers', '?')} conformers, "
        f"`alignment_shape_weight={config.get('alignment_shape_weight', 0.5)}`, "
        f"`minimize={config.get('minimize', True)}`) -- a "
        "generated SMILES usually leaves stereocentres unassigned, so this is a maximum "
        "over the isomers it could be, not the score of one molecule. Reported as shape "
        "+ colour Tanimoto on [0, 2]; the reward saw it halved onto [0, 1], so that "
        "every objective reaches the reward on one scale. Colour typed with "
        f"`{config.get('color_ff')}`, taken as the best overlay against any conformer in "
        "the reference bank.",
        f"- `SA`: the ClippedScore *desirability* on [0, 1], not the raw 1-10 score. 1.0 "
        f"means raw SA <= {config.get('sa_reward_one')}, 0.0 means raw SA >= "
        f"{config.get('sa_reward_zero')}, linear between. A molecule RDKit cannot score "
        "is reported as the worst value, never the best.",
    ]
    if "QED" in objectives:
        notes.append("- `QED`: native RDKit QED on [0, 1], no modifier.")
    return "\n".join(notes)


def analyse(run_dir: Path) -> Path:
    """Write every figure and ``report.md`` for one run directory. Returns the report path."""
    config = json.loads((run_dir / "config.json").read_text())
    objectives = config["objectives"]
    # Every ROCS number in this file is published TanimotoCombo on [0, 2]; the column on
    # disk is in the units that run's reward used. See `rocs_scale`.
    scale = rocs_scale(config)
    thresholds = config["thresholds"]
    summary = config.get("summary", {})
    from_reference = [
        name for name in (config.get("threshold_from_reference") or []) if name in objectives
    ]
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    arms = {}
    for name, filename in ARM_FILES:
        arm = _read_arm(run_dir / filename)
        if arm is not None:
            arms[name] = arm
    if not arms:
        raise FileNotFoundError(f"{run_dir} holds no scored arm to analyse")

    fit_path = run_dir / "rl_fit.tsv"
    fit = pd.read_csv(fit_path, sep="\t") if fit_path.is_file() else None
    epochs_path = run_dir / "rl_smiles.tsv"
    epochs = pd.read_csv(epochs_path, sep="\t") if epochs_path.is_file() else None

    generator = None
    reference = run_dir / "reference.sdf"
    if reference.is_file():
        try:
            from _david.config import Config
            from _david.scorers.conformers import RelaxingConformerGenerator

            generator = RelaxingConformerGenerator(Config(**_configurable(config)))
        except Exception:
            logger.exception("could not rebuild a conformer generator; skipping figure 08")

    written = figures_module.render_all(
        arms=arms,
        epochs=epochs,
        fit=fit,
        objectives=objectives,
        thresholds=thresholds,
        summary=summary,
        reference=reference,
        out=figures,
        generator=generator,
        config=config,
    )
    logger.info("wrote %d figures", len(written))

    tops = {name: top_molecules(arm, objectives) for name, arm in arms.items() if name != "prior"}

    settings = [
        "reference_smiles",
        "minimize",
        "force_field",
        "reward_scheme",
        "alignment_shape_weight",
        "n_eval",
        "threshold_percentile",
        "fixed_thresholds",
        "sa_reward_zero",
        "sa_reward_one",
        "reference_rmsd",
        "max_iterations",
        "dielectric",
        "dielectric_model",
        "max_conformers",
        "max_isomers",
        "reference_conformers",
        "reference_attempts",
        "color_ff",
        "include_qed",
        "epochs",
        "patience",
        "min_epochs",
        "n_samples",
        "batch_size",
        "epsilon",
        "beta",
        "n_prior",
        "n_final",
        "seed",
    ]
    report = [
        f"# {run_dir.name}",
        "",
        "## arms",
        "",
        "`prior` is the pretrained model before any reinforcement; the rest are the "
        "checkpoints this run kept. Every ratio divides by all sampled molecules, "
        "invalid ones included. `unique` and `desired` count a molecule once however "
        "often it was sampled; `valid` counts every row, so under mode collapse it can "
        "stay high while `unique` falls -- which is the pair worth reading together. "
        "Medians and percentiles are over valid *unique* molecules, the same population, "
        "so no two columns in a row describe different sets.",
        "",
        _table(
            {name: _describe(arm, objectives, scale) for name, arm in arms.items()}, "arm"
        ),
        "",
        "### what the score columns are",
        "",
        _units(config, objectives),
        "",
        "## thresholds",
        "",
        _threshold_prose(config, arms, objectives, thresholds),
        "",
        _pairs(
            {
                f"{objective} ({_threshold_source(objective, config, from_reference)})": round(
                    value, 4
                )
                for objective, value in zip(objectives, thresholds)
            },
            "objective",
            "threshold",
        ),
        "",
        "## training",
        "",
        f"Measured each epoch on a held-out arm of **n_eval = {config.get('n_eval', '?')}** "
        "molecules, which is a different denominator from the arm table above "
        f"(n_final = {config.get('n_final', '?')}). `desired_ratio` counts unique desired "
        "molecules over the sample size, and the unique count saturates while the "
        "denominator does not, so the same policy scores higher on the smaller arm. These "
        "numbers select checkpoints; the arm table is what the run reports.",
        "",
        _pairs(summary, "metric") if summary else "_no training summary recorded_",
        "",
        "## the non-dominated front",
        "",
        (
            "The reward is a weighted sum, whose weights DrugEx rebuilds from the batch "
            "every epoch, so no single weighted score describes the whole run. "
            if config.get("scheme", "PRCD") == "WS"
            else "The reward is Pareto ranking, so no weighted score says what a run improved. "
        )
        + "These two columns say it without weights: how many of an arm's molecules nothing "
        "else in that arm beats on every objective at once, and how much objective space "
        "the arm dominates, measured from the origin. Both grow with the sample, which is "
        "why `n scored` is in the table -- rows whose n differs are not comparable, and "
        "`n scored` is below `n` wherever a molecule was invalid, duplicated, or missing "
        "a score on one objective. Figure 02 draws these fronts.",
        "",
        _table(figures_module.pareto_rows(arms, objectives), "arm"),
        "",
        "## scaffolds",
        "",
        "Bemis-Murcko scaffolds of the valid unique molecules, dropping the acyclic ones, "
        "which have no scaffold. `unique_fraction` falling or `top_share` rising is a "
        "library collapsing onto one core; `de_novo` is the share of an arm's scaffolds "
        "the prior never produced, so a run that re-finds the prior's own chemistry "
        "scores well on the first two columns and poorly on the last. It is `n/a` for the "
        "prior, which is the baseline it is measured against. Figures 07 and 07b draw this table.",
        "",
        _table(figures_module.scaffold_rows(arms), "arm"),
        "",
        "## settings that define this run",
        "",
        _pairs({k: config[k] for k in settings if k in config}, "setting"),
        "",
        f"## best molecules, by {objectives[0]}",
        "",
        (
            f"Every figure describes the **{shown}** checkpoint unless its name says "
            "otherwise; the `b` figures (`01b`, `05b`, `07b`) repeat the same drawing for "
            "the `desired_ratio` checkpoint, which is frozen the first time its patience "
            "window expires. The two lead galleries are the exception: "
            "`05_lead_structures_gallery.png` draws the **final policy** "
            f"(`{figures_module.FINAL_ARM}`), because what a run would ship is the model "
            "it ended on and not the best epoch either criterion kept, and "
            "`05b_lead_structures_gallery_desired_ratio.png` draws `desired_ratio`. Both "
            "show one molecule per chemotype: a card marked *(analogue)* means the arm "
            "had no further distinct scaffold to show, which is itself the finding."
            if (shown := figures_module.shown_arm(arms, objectives, figures_module.PRIMARY_ARM))
            else "No trained arm was scored, so there is no checkpoint to draw."
        ),
        "",
    ]
    for name, best in tops.items():
        extra_cols = [c for c in ("ShapeTanimoto", "ColorTanimoto", "SA_raw") if c in best.columns]
        report += [
            f"### {name}",
            "",
            _table(
                {
                    row["SMILES"]: {
                        o: round(float(row[o]) * (scale if o == objectives[0] else 1.0), 4)
                        for o in [*objectives, *extra_cols]
                        if o in row
                    }
                    for _, row in best.iterrows()
                },
                "SMILES",
            ),
            "",
        ]

    report += [
        "## files",
        "",
        "Every artifact this run wrote. The figures are regenerated from these tables "
        "alone by `python -m _david.analysis <run_dir>`, which rescores nothing.",
        "",
        _artifacts(run_dir, written),
        "",
    ]

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n")
    logger.info("wrote %s and %d figures", report_path, len(list(figures.glob("*.png"))))
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run directory written by _david.run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    analyse(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
