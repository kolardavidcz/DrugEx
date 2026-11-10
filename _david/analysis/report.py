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

import matplotlib

matplotlib.use("Agg")  # headless: these also run on a cluster node with no display
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from _david.analysis import figures as figures_module
from rdkit import Chem
from rdkit.Chem import Draw

logger = logging.getLogger(__name__)

#: One colour per arm, used by every figure so a reader learns them once.
COLOURS = {
    "prior": "#9e9e9e",
    "desired_ratio": "#1f77b4",
    "desired_ratio_at_stop": "#7fb3e0",
    "gmean": "#d62728",
}

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


def _describe(arm: pd.DataFrame, objectives: list[str]) -> dict:
    """The ratios a thesis table reports, plus each objective's median and p90."""
    valid = _valid(arm)
    unique = valid.drop_duplicates(subset=_dedup_key(valid))  # DrugEx's unique-compound denominator
    row = {
        "n": len(arm),
        "valid": round(len(valid) / len(arm), 4),
        "unique": round(len(unique) / len(arm), 4),
        "desired": round(float(unique["Desired"].sum()) / len(arm), 4),
    }
    # Unique, not merely valid: the ratio columns above already deduplicate, and a row
    # whose ratios describe unique molecules while its median describes a sample would be
    # two statistics on two populations. Under mode collapse the sample median is the
    # score of whichever molecule was drawn most often.
    row["duplicate_fraction"] = round(1.0 - len(unique) / len(valid), 4) if len(valid) else 0.0
    for objective in objectives:
        row[f"{objective} median"] = round(float(unique[objective].median()), 4)
        row[f"{objective} p90"] = round(float(unique[objective].quantile(0.90)), 4)
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
            notes.append(
                f"**SA barely shapes the reward.** {ceiling:.1%} of the valid prior sits "
                f"at raw SA <= {config.get('sa_reward_one', 3.0):g}, where `ClippedScore` "
                "returns exactly 1.0. Molecules tied at that ceiling cannot dominate one "
                "another on the SA axis and contribute no crowding distance along it, so "
                "for most of the population the Pareto reward is effectively the ROCS "
                "ranking. This is a property of the *reward* and is not changed by where "
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
    # Scheme-aware, because the sentence is only true for the Pareto schemes. A run
    # whose report claims its thresholds stayed out of the reward while `WeightedSum`
    # was building weights from them is exactly the silent-weighting bug this project
    # was restarted to remove. Old runs predate the key and were all PRCD.
    scheme = config.get("scheme", "PRCD")
    if scheme == "WS":
        tail = (
            "**Under `WeightedSum` these thresholds are also part of the reward.** DrugEx "
            "builds each objective's weight from the fraction of the batch below versus "
            "above its own threshold, so unlike every Pareto run, here they do not only "
            "decide `Desired` and the stop condition. The weights are DrugEx's, not this "
            "project's, and they move every epoch with the batch. One consequence worth "
            "stating: a weighted *sum* adds raw columns, so range is leverage -- "
            "TanimotoCombo on [0, 2] moves the reward twice as far as SA or QED on [0, 1] "
            "at the same weight."
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
        f"{config.get('max_conformers', '?')} conformers x the reference bank -- a "
        "generated SMILES usually leaves stereocentres unassigned, so this is a maximum "
        "over the isomers it could be, not the score of one molecule. No "
        "modifier. Shape + colour Tanimoto on [0, 2], colour typed with "
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

    # The 3D overlay needs a generator; building one costs a conformer run, so it is only
    # made when the reference exists. A missing generator simply skips that one figure.
    generator = None
    reference = run_dir / "reference.sdf"
    if reference.is_file():
        try:
            from _david.config import Config
            from _david.scorers.conformers import RelaxingConformerGenerator

            generator = RelaxingConformerGenerator(Config(**_configurable(config)))
        except Exception:
            logger.exception("could not rebuild a conformer generator; skipping figure 10")

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
        "checkpoints this run kept. Ratios are over all sampled molecules, including "
        "invalid ones, and count a molecule once however often it was sampled. "
        "Medians and percentiles are over valid *unique* molecules, the same population, "
        "so no two columns in a row describe different sets.",
        "",
        _table({name: _describe(arm, objectives) for name, arm in arms.items()}, "arm"),
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
        "## settings that define this run",
        "",
        _pairs({k: config[k] for k in settings if k in config}, "setting"),
        "",
        f"## best molecules, by {objectives[0]}",
        "",
        # Which checkpoint won is the question the run exists to answer, so the report
        # answers it in one line instead of leaving a reader to compare three tables --
        # and it is the same function figure 05b uses, so the two cannot disagree.
        (
            f"Best model: **{best}** (highest desired ratio of the trained checkpoints). "
            f"Figure `05b_top_leads_best_model_{best}.png` draws this arm's leads."
            if (best := figures_module.best_arm(arms, objectives))
            else "No trained arm was scored, so there is no best model to name."
        ),
        "",
    ]
    for name, best in tops.items():
        report += [
            f"### {name}",
            "",
            _table(
                {
                    row["SMILES"]: {o: round(float(row[o]), 4) for o in objectives}
                    for _, row in best.iterrows()
                },
                "SMILES",
            ),
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
