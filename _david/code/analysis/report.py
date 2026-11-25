"""Figures and a summary table for a finished run.

``python -m _david.code.analysis <run_dir>`` reads only what the run wrote, so it can be
re-run on any old run directory and never touches training. Every number comes from a file
in that directory; nothing is rescored here, because rescoring with a setting other than
the one the run used is how two different numbers for one molecule happen.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from _david.code.analysis import figures as figures_module
from _david.code.analysis._helpers import (
    dedup_key,
    frame_rocs_scale,
    pareto_rows,
    scaffold_rows,
    threshold_rocs_scale,
    valid,
)
from _david.code.reconstruct.provenance import atomic_write_text

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

REPORT_CACHE_VERSION = 1
EXPECTED_FIGURE_SLOTS = ("01", "02", "03", "04", "05", "06", "07", "08", "09")
_RELAX_RE = re.compile(
    r"relaxation dropped (\d+)/(\d+) conformers \(([^)]+)\); (\d+) molecules left with none"
)
_FIG_FAIL_RE = re.compile(r"figure (\d+) failed")

#: Settings table order in ``report.md``.
REPORT_SETTINGS = (
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
)

#: Every table a run writes, and what it holds. Order is the order of the index.
TABLES = (
    ("prior.tsv", "the pretrained prior, scored -- the baseline every figure compares against"),
    (
        "final_desired_ratio_at_stop.tsv",
        "the live policy at the epoch training stopped, scored -- the arm a run delivers",
    ),
    ("final_desired_ratio.tsv", "the best `desired_ratio` epoch of a legacy run, scored"),
    ("final_gmean.tsv", "the best `avg_gmean` epoch of a legacy run, scored"),
    ("rl_fit.tsv", "one row per epoch: the metrics the stop condition reads"),
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


def _read_arm(path: Path) -> pd.DataFrame | None:
    """One scored arm, or ``None`` when the run did not write it."""
    if not path.is_file():
        logger.info("no %s; skipping it", path.name)
        return None
    return pd.read_csv(path, sep="\t")


def _table(rows: dict[str, dict[str, Any]], first_column: str) -> str:
    """Markdown table, one row per key of ``rows``. Empty input returns ``_none_``."""
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


def _pairs(mapping: dict[str, Any], key_header: str, value_header: str = "value") -> str:
    """Markdown table of one key/value pair per row."""
    lines = [f"| {key_header} | {value_header} |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in mapping.items()]
    return "\n".join(lines)


def _artifacts(run_dir: Path, written: list[Path]) -> str:
    """Link every artifact actually written in ``run_dir``."""
    lines = [f"- [{figures_module.caption(path)}](figures/{path.name})" for path in written]
    lines += [
        f"- [{name}]({name}) -- {what}" for name, what in TABLES if (run_dir / name).is_file()
    ]
    return "\n".join(lines) if lines else "_nothing was written_"


def top_molecules(arm: pd.DataFrame, objectives: list[str]) -> pd.DataFrame:
    """Select the top unique valid molecules of one arm by primary objective."""
    val = valid(arm)
    if val.empty:
        return val
    return (
        val.drop_duplicates(subset=dedup_key(val))
        .sort_values(objectives[0], ascending=False)
        .head(TOP_N)
    )


def _describe(arm: pd.DataFrame, objectives: list[str], scale: float = 1.0) -> dict[str, Any]:
    """Compute arm ratios plus per-objective median and p90 over unique valid molecules."""
    val = valid(arm)
    uniq = val.drop_duplicates(subset=dedup_key(val)) if not val.empty else val
    sampled = max(len(arm), 1)
    row: dict[str, Any] = {
        "n": len(arm),
        "valid": round(len(val) / sampled, 4),
        "unique": round(len(uniq) / sampled, 4),
        "desired": round(float(uniq["Desired"].sum()) / sampled, 4) if not uniq.empty else 0.0,
        "duplicate_fraction": round(1.0 - len(uniq) / len(val), 4) if len(val) else 0.0,
    }
    for index, objective in enumerate(objectives):
        column = pd.to_numeric(uniq[objective], errors="coerce") if not uniq.empty else pd.Series(dtype=float)
        if index == 0:
            column = column * scale
        row[f"{objective} median"] = round(float(column.median()), 4)
        row[f"{objective} p90"] = round(float(column.quantile(0.90)), 4)
    if "SA_raw" in uniq.columns and not uniq.empty:
        raw_sa = pd.to_numeric(uniq["SA_raw"], errors="coerce")
        raw_sa = raw_sa[(raw_sa > 0.5) & (raw_sa < 9.99)]
        if len(raw_sa):
            row["SA_raw median"] = round(float(raw_sa.median()), 4)
    return row


def _threshold_source(
    objective: str, config: dict[str, Any], from_reference: list[str]
) -> str:
    """Where this objective's threshold came from (mirrors ``calibrate_thresholds`` order)."""
    if config.get("fixed_thresholds"):
        return "fixed"
    if objective in from_reference:
        return "reference"
    return f"p{config['threshold_percentile']:g}"


def _threshold_prose(
    config: dict[str, Any],
    arms: dict[str, pd.DataFrame],
    objectives: list[str],
    thresholds: list[float],
) -> str:
    """Explain how thresholds were obtained, whether any saturated, and how the scheme uses them."""
    from_reference = [
        name for name in (config.get("threshold_from_reference") or []) if name in objectives
    ]
    if config.get("fixed_thresholds"):
        from_reference = []
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
    notes: list[str] = []
    prior = arms.get("prior")
    if prior is not None and "SA_raw" in prior.columns:
        valid_raw = prior.loc[prior["Valid"] == 1, "SA_raw"]
        ceiling = (
            float((valid_raw <= config.get("sa_reward_one", 3.0)).mean())
            if len(valid_raw)
            else 0.0
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
        val = valid(prior)
        for idx, (objective, threshold) in enumerate(zip(objectives, thresholds, strict=True)):
            if objective not in val or val.empty:
                continue
            column = val[objective]
            tied = int((column >= threshold).sum())
            if threshold >= column.max():
                shown_thr = (
                    threshold * threshold_rocs_scale(config, prior, objective)
                    if idx == 0
                    else threshold
                )
                notes.append(
                    f"**`{objective}` is a hard gate, not a percentile**: the threshold "
                    f"{shown_thr:.4g} equals the highest value in the prior, with "
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


def _units(config: dict[str, Any], objectives: list[str]) -> str:
    """State the units and modifier conventions of each objective column."""
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


def _source_fingerprints(run_dir: Path) -> dict[str, list[int]]:
    names = (
        "config.json",
        "manifest.json",
        "prior.tsv",
        "final_desired_ratio_at_stop.tsv",
        "final_desired_ratio.tsv",
        "final_gmean.tsv",
        "rl_fit.tsv",
        "run.log",
    )
    out: dict[str, list[int]] = {}
    for name in names:
        path = run_dir / name
        if path.is_file():
            st = path.stat()
            out[name] = [int(st.st_size), int(st.st_mtime_ns)]
        else:
            out[name] = [-1, -1]
    return out


def parse_run_log_diagnostics(log_path: Path) -> dict[str, Any]:
    """Extract relaxation tallies, timeouts, and error signals from ``run.log``."""
    if not log_path.is_file():
        return {"present": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    relax_events = 0
    conformers_total = 0
    conformers_dropped = 0
    zero_mol_events = 0
    detail_totals: dict[str, int] = {}
    scoring_failures = 0
    analysis_failures = 0
    figure_failures: list[str] = []
    error_lines: list[str] = []

    for line in text.splitlines():
        match = _RELAX_RE.search(line)
        if match:
            relax_events += 1
            dropped, total, details, zero_mols = match.groups()
            conformers_dropped += int(dropped)
            conformers_total += int(total)
            zero_mol_events += int(zero_mols)
            for part in details.split(","):
                k, _, v = part.strip().partition("=")
                if k and v.isdigit():
                    detail_totals[k] = detail_totals.get(k, 0) + int(v)
        if "scoring failed for one molecule" in line:
            scoring_failures += 1
        if "analysis failed" in line:
            analysis_failures += 1
        fig_m = _FIG_FAIL_RE.search(line)
        if fig_m:
            figure_failures.append(fig_m.group(1))
        if (
            " ERROR " in line or "CRITICAL " in line or line.startswith("Traceback")
        ) and len(error_lines) < 20:
            error_lines.append(line.strip())

    return {
        "present": True,
        "relaxation_events": relax_events,
        "conformers_total": conformers_total,
        "conformers_dropped": conformers_dropped,
        "zero_conformer_molecules": zero_mol_events,
        "relaxation_details": detail_totals,
        "scoring_failures": scoring_failures,
        "analysis_failures": analysis_failures,
        "figure_failures": figure_failures,
        "errors": error_lines,
    }


def _write_report_cache(
    run_dir: Path,
    *,
    arm_scales: dict[str, float],
    thr_scale: float,
    arms_table: dict[str, dict[str, Any]],
    pareto_table: dict[str, dict[str, Any]],
    scaffold_table: dict[str, dict[str, Any]],
    written: list[Path],
    summary: dict[str, Any],
    fit: pd.DataFrame | None,
) -> None:
    files = sorted(p.name for p in written)
    missing_slots = [
        slot for slot in EXPECTED_FIGURE_SLOTS if not any(name.startswith(slot) for name in files)
    ]
    payload = {
        "version": REPORT_CACHE_VERSION,
        "sources": _source_fingerprints(run_dir),
        "arm_scales": arm_scales,
        "threshold_rocs_scale": thr_scale,
        "arms": arms_table,
        "pareto": pareto_table,
        "scaffolds": scaffold_table,
        "figures": {
            "count": len(files),
            "expected": len(EXPECTED_FIGURE_SLOTS),
            "files": files,
            "missing_slots": missing_slots,
        },
        "epoch_summary": {
            **(summary or {}),
            "epochs_logged": len(fit) if fit is not None else 0,
        },
        "log_diagnostics": parse_run_log_diagnostics(run_dir / "run.log"),
    }
    atomic_write_text(
        run_dir / "cache_stats" / "report_stats.json", json.dumps(payload, indent=2) + "\n"
    )


def _render_markdown(
    run_dir: Path,
    *,
    config: dict[str, Any],
    objectives: list[str],
    thresholds: list[float],
    summary: dict[str, Any],
    from_reference: list[str],
    arms: dict[str, pd.DataFrame],
    arm_scales: dict[str, float],
    thr_scale: float,
    arms_table: dict[str, dict[str, Any]],
    pareto_table: dict[str, dict[str, Any]],
    scaffold_table: dict[str, dict[str, Any]],
    tops: dict[str, pd.DataFrame],
    written: list[Path],
) -> str:
    shown = figures_module.shown_arm(arms, objectives, figures_module.PRIMARY_ARM)
    shown_prose = (
        f"Every figure describes the **{shown}** arm unless its name says otherwise: the "
        "live policy at the epoch training stopped, which is the model a run would ship "
        "and the only checkpoint a new run writes. The lead gallery draws that same arm, "
        "because what a run delivers is the model it ended on, not its best epoch. It "
        "shows one molecule per chemotype: a card marked *(analogue)* means the arm "
        "had no further distinct scaffold to show, which is itself the finding."
        if shown
        else "No trained arm was scored, so there is no checkpoint to draw."
    )
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
        _table(arms_table, "arm"),
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
                    value * (thr_scale if idx == 0 else 1.0), 4
                )
                for idx, (objective, value) in enumerate(zip(objectives, thresholds, strict=True))
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
        "numbers decide when training stops; the arm table is what the run reports.",
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
        _table(pareto_table, "arm"),
        "",
        "## scaffolds",
        "",
        "Bemis-Murcko scaffolds of the valid unique molecules, dropping the acyclic ones, "
        "which have no scaffold. `unique_fraction` falling or `top_share` rising is a "
        "library collapsing onto one core; `de_novo` is the share of an arm's scaffolds "
        "the prior never produced, so a run that re-finds the prior's own chemistry "
        "scores well on the first two columns and poorly on the last. It is `n/a` for the "
        "prior, which is the baseline it is measured against. Figure 07 draws this table.",
        "",
        _table(scaffold_table, "arm"),
        "",
        "## settings that define this run",
        "",
        _pairs({k: config[k] for k in REPORT_SETTINGS if k in config}, "setting"),
        "",
        f"## best molecules, by {objectives[0]}",
        "",
        shown_prose,
        "",
    ]
    for name, best in tops.items():
        extra_cols = [c for c in ("ShapeTanimoto", "ColorTanimoto", "SA_raw") if c in best.columns]
        arm_scale = arm_scales[name]
        report += [
            f"### {name}",
            "",
            _table(
                {
                    row["SMILES"]: {
                        o: round(float(row[o]) * (arm_scale if o == objectives[0] else 1.0), 4)
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
        "alone by `python -m _david.code.analysis <run_dir>`, which rescores nothing.",
        "",
        _artifacts(run_dir, written),
        "",
    ]
    return "\n".join(report) + "\n"


def analyse(run_dir: Path) -> Path:
    """Write every figure, ``report.md``, and ``report_stats.json`` for one run directory."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
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

    arm_scales = {
        name: frame_rocs_scale(arm, objectives[0], config) for name, arm in arms.items()
    }
    thr_scale = threshold_rocs_scale(config, arms.get("prior"), objectives[0])

    fit_path = run_dir / "rl_fit.tsv"
    fit = pd.read_csv(fit_path, sep="\t") if fit_path.is_file() else None
    epochs_path = run_dir / "rl_smiles.tsv"
    epochs = pd.read_csv(epochs_path, sep="\t") if epochs_path.is_file() else None

    generator = None
    reference = run_dir / "reference.sdf"
    if reference.is_file():
        try:
            from _david.code.config import Config
            from _david.code.scorers.conformers import RelaxingConformerGenerator

            generator = RelaxingConformerGenerator(Config.from_dict(config))
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

    arms_table = {
        name: _describe(arm, objectives, arm_scales[name]) for name, arm in arms.items()
    }
    pareto_table = pareto_rows(arms, objectives, config)
    scaffold_table = scaffold_rows(arms)
    tops = {name: top_molecules(arm, objectives) for name, arm in arms.items() if name != "prior"}

    report_path = run_dir / "report.md"
    atomic_write_text(
        report_path,
        _render_markdown(
            run_dir,
            config=config,
            objectives=objectives,
            thresholds=thresholds,
            summary=summary,
            from_reference=from_reference,
            arms=arms,
            arm_scales=arm_scales,
            thr_scale=thr_scale,
            arms_table=arms_table,
            pareto_table=pareto_table,
            scaffold_table=scaffold_table,
            tops=tops,
            written=written,
        ),
    )
    try:
        _write_report_cache(
            run_dir,
            arm_scales=arm_scales,
            thr_scale=thr_scale,
            arms_table=arms_table,
            pareto_table=pareto_table,
            scaffold_table=scaffold_table,
            written=written,
            summary=summary,
            fit=fit,
        )
    except OSError:
        logger.warning("could not write report_stats.json cache for %s", run_dir)
    logger.info("wrote %s and %d figures", report_path, len(list(figures.glob("*.png"))))
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run directory written by _david.code.run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    analyse(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
