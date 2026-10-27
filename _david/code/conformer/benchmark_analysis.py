"""Analyze measured conformer trade-offs with molecule-level holdouts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class BenchmarkAnalysis:
    """Keep molecule-level uncertainty, held-out validation, and failures visible."""

    GROUPS: list[str] = [
        "protocol",
        "candidate_method",
        "matching",
        "reference_budget",
        "k_ref_requested",
        "k_candidate_requested",
    ]

    def __init__(self, output: Path) -> None:
        """Bind analysis to one measurement directory."""
        self.output: Path = output

    @staticmethod
    def confidence_interval(frame: pd.DataFrame, column: str) -> tuple[float, float]:
        """Bootstrap molecules after averaging repeated seeds; do not inflate sample size."""
        values = frame.groupby("molecule_id")[column].mean().dropna().to_numpy()
        if not len(values):
            return float("nan"), float("nan")
        sampled = (
            np.random.default_rng(20261027)
            .choice(values, size=(1000, len(values)), replace=True)
            .mean(axis=1)
        )
        return tuple(float(x) for x in np.quantile(sampled, [0.025, 0.975]))

    @classmethod
    def summarize(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Aggregate counts and losses against the best observed within-protocol score."""
        keys = ["protocol", "candidate_method", "matching", "molecule_id", "replicate"]
        frame = frame.copy()
        frame["best_observed"] = frame.groupby(keys).raw_combo.transform("max")
        frame["score_loss"] = frame.best_observed - frame.raw_combo
        rows: list[dict[str, Any]] = []
        for key, group in frame.groupby(cls.GROUPS, sort=True):
            by_molecule = (
                group.groupby("molecule_id")[["raw_combo", "best_observed"]]
                .mean()
                .dropna()
            )
            rho = float("nan")
            if (
                len(by_molecule) > 2
                and by_molecule.raw_combo.nunique() > 1
                and by_molecule.best_observed.nunique() > 1
            ):
                rho = float(
                    spearmanr(
                        by_molecule.raw_combo, by_molecule.best_observed
                    ).statistic
                )
            size_correlation = float("nan")
            if "heavy_atoms" in group:
                sizes = (
                    group.groupby("molecule_id")[["raw_combo", "heavy_atoms"]]
                    .mean()
                    .dropna()
                )
                if (
                    len(sizes) > 2
                    and sizes.raw_combo.nunique() > 1
                    and sizes.heavy_atoms.nunique() > 1
                ):
                    size_correlation = float(
                        spearmanr(sizes.raw_combo, sizes.heavy_atoms).statistic
                    )
            low, high = cls.confidence_interval(group, "score_loss")
            rows.append(
                {
                    **dict(zip(cls.GROUPS, key)),
                    "n_rows": len(group),
                    "n_molecules": group.molecule_id.nunique(),
                    "mean_raw_combo": group.raw_combo.mean(),
                    "mean_loss": group.score_loss.mean(),
                    "p95_loss": group.score_loss.quantile(0.95),
                    "loss_ci_low": low,
                    "loss_ci_high": high,
                    "fraction_within_0p02": (
                        group.score_loss.le(0.02) & group.raw_combo.notna()
                    ).mean(),
                    "median_candidate_seconds": group.candidate_total_seconds.median(),
                    "mean_candidate_seconds": group.candidate_total_seconds.mean(),
                    "max_candidate_seconds": group.candidate_total_seconds.max(),
                    "p95_candidate_seconds": group.candidate_total_seconds.quantile(
                        0.95
                    ),
                    "median_alignment_seconds": group.alignment_seconds.median(),
                    "mean_reference_seconds": group.reference_seconds.mean(),
                    "median_actual_reference": group.k_ref_actual.median(),
                    "median_actual_candidate": group.k_candidate_actual.median(),
                    "failure_fraction": group.status.ne("ok").mean(),
                    "rank_spearman": rho,
                    "heavy_atom_spearman": size_correlation,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def select(
        cls,
        screening: pd.DataFrame,
        holdout: pd.DataFrame,
        latency_column: str = "median_candidate_seconds",
    ) -> pd.DataFrame:
        """Select on screening only, then report the same setting on held-out molecules."""
        rows: list[dict[str, Any]] = []
        for key, group in screening.groupby(cls.GROUPS[:3]):
            acceptable = group[
                (group.mean_loss <= 0.02)
                & (group.p95_loss <= 0.05)
                & (group.failure_fraction <= 0.05)
            ]
            if acceptable.empty:
                rows.append(
                    {
                        **dict(zip(cls.GROUPS[:3], key)),
                        "selection_status": "no_acceptable_screening_setting",
                    }
                )
                continue
            selected = acceptable.sort_values(
                [
                    latency_column,
                    "mean_reference_seconds",
                    "k_ref_requested",
                    "k_candidate_requested",
                ]
            ).iloc[0]
            row = selected.to_dict()
            row["selection_latency_objective"] = latency_column
            row["selection_status"] = "screening_selected"
            matched = holdout.copy()
            for column in cls.GROUPS:
                matched = matched[matched[column] == selected[column]]
            if not matched.empty:
                validation = matched.iloc[0]
                for column in [
                    "mean_loss",
                    "p95_loss",
                    "loss_ci_low",
                    "loss_ci_high",
                    "failure_fraction",
                    "rank_spearman",
                    "n_molecules",
                ]:
                    row[f"holdout_{column}"] = validation[column]
                row["holdout_ci_confirms_mean_tolerance"] = (
                    validation.loss_ci_high <= 0.02
                )
                row["holdout_score_tolerance_pass"] = (
                    validation.mean_loss <= 0.02 and validation.p95_loss <= 0.05
                )
                row["holdout_preparation_tolerance_pass"] = (
                    validation.failure_fraction <= 0.05
                )
                row["selection_status"] = (
                    "holdout_pass"
                    if validation.mean_loss <= 0.02
                    and validation.p95_loss <= 0.05
                    and validation.failure_fraction <= 0.05
                    else "holdout_did_not_confirm"
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def diagnostics(
        self, frame: pd.DataFrame, reference: pd.DataFrame, candidates: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Expose seed noise, sampling stability, minimization effects, and long tails."""
        model = frame[frame.panel_split != "control"]
        seed_rows: list[dict[str, Any]] = []
        for key, group in model.groupby(self.GROUPS):
            per_molecule = group.groupby("molecule_id").raw_combo.agg(
                ["std", "min", "max"]
            )
            seed_rows.append(
                {
                    **dict(zip(self.GROUPS, key)),
                    "mean_seed_sd": per_molecule["std"].mean(),
                    "p95_seed_range": (
                        per_molecule["max"] - per_molecule["min"]
                    ).quantile(0.95),
                }
            )
        stability: list[dict[str, Any]] = []
        largest = model[
            (model.k_ref_requested == model.k_ref_requested.max())
            & (model.k_candidate_requested == model.k_candidate_requested.max())
        ]
        pair_keys = ["protocol", "candidate_method", "matching"]
        for key, group in largest.groupby(pair_keys):
            budgets = sorted(group.reference_budget.unique())
            for low, high in zip(budgets, budgets[1:]):
                paired = group[group.reference_budget == high].merge(
                    group[group.reference_budget == low],
                    on=["molecule_id", "replicate"],
                    suffixes=("_high", "_low"),
                    validate="one_to_one",
                )
                paired["change"] = paired.raw_combo_high - paired.raw_combo_low
                ci_low, ci_high = self.confidence_interval(paired, "change")
                stability.append(
                    {
                        **dict(zip(pair_keys, key)),
                        "budget_low": low,
                        "budget_high": high,
                        "n_molecules": paired.molecule_id.nunique(),
                        "mean_signed_change": paired.change.mean(),
                        "mean_absolute_change": paired.change.abs().mean(),
                        "p95_absolute_change": paired.change.abs().quantile(0.95),
                        "change_ci_low": ci_low,
                        "change_ci_high": ci_high,
                    }
                )
        effect_keys = [
            "protocol",
            "matching",
            "reference_budget",
            "k_ref_requested",
            "k_candidate_requested",
            "molecule_id",
            "replicate",
        ]
        paired_methods = model[model.candidate_method == "matched_minimized"].merge(
            model[model.candidate_method == "training_etkdg"],
            on=effect_keys,
            suffixes=("_minimized", "_training"),
            validate="one_to_one",
        )
        paired_methods["score_change"] = (
            paired_methods.raw_combo_minimized - paired_methods.raw_combo_training
        )
        paired_methods["extra_seconds"] = (
            paired_methods.candidate_total_seconds_minimized
            - paired_methods.candidate_total_seconds_training
        )
        effects: list[dict[str, Any]] = []
        for key, group in paired_methods.groupby(
            [k for k in self.GROUPS if k != "candidate_method"]
        ):
            low, high = self.confidence_interval(group, "score_change")
            valid = group[
                group.status_minimized.eq("ok") & group.status_training.eq("ok")
            ]
            valid_low, valid_high = self.confidence_interval(valid, "score_change")
            effects.append(
                {
                    **dict(
                        zip([k for k in self.GROUPS if k != "candidate_method"], key)
                    ),
                    "mean_score_change": group.score_change.mean(),
                    "change_ci_low": low,
                    "change_ci_high": high,
                    "n_pairs": len(group),
                    "n_valid_pairs": len(valid),
                    "minimized_failure_fraction": group.status_minimized.ne("ok").mean(),
                    "training_failure_fraction": group.status_training.ne("ok").mean(),
                    "mean_valid_pair_score_change": valid.score_change.mean(),
                    "valid_pair_change_ci_low": valid_low,
                    "valid_pair_change_ci_high": valid_high,
                    "fraction_scores_reduced": group.score_change.lt(-1e-8).mean(),
                    "median_extra_seconds": group.extra_seconds.median(),
                }
            )
        candidate_keys = ["method", "count"] + (
            ["protocol"] if "protocol" in candidates else []
        )
        candidate_stats = (
            candidates.groupby(candidate_keys, dropna=False)
            .agg(
                n_builds=("status", "size"),
                failure_fraction=("status", lambda x: x.ne("ok").mean()),
                median_actual=("actual_conformers", "median"),
                median_generation_seconds=("generation_seconds", "median"),
                p95_generation_seconds=(
                    "generation_seconds",
                    lambda x: x.quantile(0.95),
                ),
                max_generation_seconds=("generation_seconds", "max"),
            )
            .reset_index()
        )
        reference = reference.copy()
        reference["protocol_name"] = reference.protocol.map(
            lambda value: ast.literal_eval(value)["name"]
        )
        reference["iteration_limited"] = reference.optimizer_statuses.map(
            lambda value: (
                ast.literal_eval(value).get("1", 0) if isinstance(value, str) else 0
            )
        )
        reference_stats = (
            reference.groupby(["protocol_name", "budget"])
            .agg(
                n_builds=("status", "size"),
                min_available_basins=("available_basins", "min"),
                median_available_basins=("available_basins", "median"),
                max_available_basins=("available_basins", "max"),
                mean_embedded=("embedded", "mean"),
                mean_reference_seconds=("reference_seconds", "mean"),
                iteration_limited_conformers=("iteration_limited", "sum"),
            )
            .reset_index()
        )
        control_keys = self.GROUPS + ["molecule_id"]
        controls = (
            frame[frame.panel_split == "control"]
            .groupby(control_keys)
            .raw_combo.agg(["mean", "min", "max"])
            .reset_index()
        )
        results = {
            "seed_stability": pd.DataFrame(seed_rows),
            "reference_budget_stability": pd.DataFrame(stability),
            "minimization_effect": pd.DataFrame(effects),
            "candidate_diagnostics": candidate_stats,
            "reference_diagnostics": reference_stats,
            "control_scores": controls,
            "preparation_failures": candidates[candidates.status.ne("ok")].copy(),
        }
        for name, result in results.items():
            result.to_csv(self.output / f"{name}.csv", index=False)
        return results

    def run(self) -> None:
        """Export separate control, screening, holdout, and model summaries and heatmaps."""
        frame = pd.read_csv(self.output / "measurements.csv.gz")
        identity = self.GROUPS + ["molecule_id", "replicate"]
        manifest = json.loads((self.output / "manifest.json").read_text())
        if (
            frame.duplicated(identity).any()
            or len(frame) != manifest["measurement_rows"]
        ):
            raise ValueError(
                "Result rows disagree with the manifest or contain duplicates"
            )
        if not np.allclose(
            frame.raw_combo / 2, frame.modified_score, equal_nan=True, atol=1e-12
        ):
            raise ValueError("Raw and modified score scales disagree")
        finite_scores = frame.raw_combo.dropna()
        if not finite_scores.between(0, 2).all():
            raise ValueError("Raw combo scores are non-finite or outside [0,2]")
        order = [column for column in identity if column != "k_ref_requested"]
        differences = (
            frame.sort_values("k_ref_requested").groupby(order).raw_combo.diff()
        )
        if differences.lt(-1e-12).any():
            raise ValueError("A larger nested reference prefix lowered a maximum score")
        panel = pd.read_csv(self.output / "molecule_panel.csv")
        frame = frame.merge(
            panel[["molecule_id", "panel_split", "rotatable_bonds", "heavy_atoms"]],
            on="molecule_id",
            validate="many_to_one",
        )
        summaries: dict[str, pd.DataFrame] = {}
        for name in (
            "screening",
            "holdout",
            "holdout_in_domain",
            "control",
            "all_model",
        ):
            if name == "all_model":
                subset = frame[frame.panel_split != "control"]
            elif name == "holdout_in_domain":
                subset = frame[
                    (frame.panel_split == "holdout")
                    & (frame.heavy_atoms <= 45)
                    & (frame.rotatable_bonds <= 15)
                ]
            else:
                subset = frame[frame.panel_split == name]
            summaries[name] = self.summarize(subset)
            summaries[name].to_csv(self.output / f"summary_{name}.csv", index=False)
        selected = self.select(summaries["screening"], summaries["holdout"])
        for column in self.GROUPS:
            if column not in selected:
                selected[column] = np.nan
        conditional = summaries["holdout_in_domain"].rename(
            columns={
                column: f"holdout_in_domain_{column}"
                for column in summaries["holdout_in_domain"].columns
                if column not in self.GROUPS
            }
        )
        selected = selected.merge(
            conditional, on=self.GROUPS, how="left", validate="one_to_one"
        )
        selected.to_csv(self.output / "selected_settings.csv", index=False)
        mean_selected = self.select(
            summaries["screening"],
            summaries["holdout"],
            latency_column="mean_candidate_seconds",
        )
        for column in self.GROUPS:
            if column not in mean_selected:
                mean_selected[column] = np.nan
        mean_selected.merge(
            conditional, on=self.GROUPS, how="left", validate="one_to_one"
        ).to_csv(self.output / "selected_settings_mean_latency.csv", index=False)
        from _david.code.visualization import ConformerBenchmarkVisualizer

        plots = ConformerBenchmarkVisualizer.render(
            summaries["all_model"], self.output / "figures"
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        reference = pd.read_csv(self.output / "reference_builds.csv")
        candidates = pd.read_csv(self.output / "candidate_builds.csv")
        rejected_embeddings = int(
            candidates[candidates.method == "training_etkdg"]
            .get("rejected_embedding_results", pd.Series(dtype=float))
            .sum()
        )
        native_reproduction = (
            "This run also includes a native reproduction in negative_id_regression.json."
            if (self.output / "negative_id_regression.json").exists()
            else "The test suite includes a deterministic regression for the failed-isomer sentinel."
        )
        diagnostics = self.diagnostics(frame, reference, candidates)
        source_root = Path(__file__).resolve().parents[3]
        analysis_sources = [
            Path(__file__),
            source_root / "_david/code/visualization.py",
        ]
        (self.output / "analysis_provenance.json").write_text(
            json.dumps(
                {
                    "code_sha256": {
                        str(p.relative_to(source_root)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in analysis_sources
                    }
                },
                indent=2,
            )
            + "\n"
        )
        lines = [
            "# Conformer benchmark results",
            "",
            f"Revision `{manifest['git_revision']}`; exact source/input hashes, seeds, host and definitions: `manifest.json`.",
            f"{len(frame):,} measurements; {len(panel)} distinct molecules; {len(manifest['config']['reference_seeds'])} paired seed replicates.",
            "",
            "## Selection rule",
            "",
            "Choose the fastest setting with screening mean raw-combo loss <=0.02, p95 loss <=0.05, and preparation failures <=5%. These are predeclared engineering tolerances, not activity thresholds. Evaluate the selected setting on holdout molecules without retuning. Bootstrap intervals resample molecules after averaging seeds. A holdout_pass means observed losses meet these tolerances; the additional holdout_ci_confirms_mean_tolerance flag requires the upper bootstrap bound to meet the mean-loss tolerance too.",
            "Loss is relative to the best observed for the SAME molecule, seed replicate, matching function, reference protocol, and candidate method across this finite grid. This search envelope is not ground-truth pose accuracy or affinity. Higher scores between different matching functions cannot select a scientifically superior function.",
            "",
            "| Protocol | Candidate method | Matching | Ref attempts | Ref cap | Candidate request | Screen loss | Holdout loss | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in selected.to_dict("records"):
            lines.append(
                f"| {row['protocol']} | {row['candidate_method']} | {row['matching']} | {row.get('reference_budget', np.nan):g} | {row.get('k_ref_requested', np.nan):g} | {row.get('k_candidate_requested', np.nan):g} | {row.get('mean_loss', np.nan):.4f} | {row.get('holdout_mean_loss', np.nan):.4f} | {row['selection_status']} |"
            )
        lines += [
            "",
            "## Interpretation",
            "",
            "Heatmap columns vary independent reference embedding budgets; X varies retained-reference cap and Y candidate requests per stereoisomer. Rows show raw combo, loss, and measured candidate generation plus alignment latency. Actual retained reference counts are annotated. Counts beyond available basins reuse the same measured invocation, not independent repeated timings. Controls are separate from model-panel means.",
            "Candidate counts are generated independently at each requested budget, never sliced from a larger ensemble. Actual conformer counts include up to four unassigned stereoisomers. First candidate seed is the production seed. The generation path is the production RDKitConformerGenerator, and matching calls the production _score_single_reference kernel (canonical frame, four proper axis-sign starts, shape/color optimization=0.5). Raw combo [0,2], modified score=raw/2; no observed-max scaling.",
            "Reference protocols use production standardization/ETKDGv3, explicit mechanics, an 8 kcal/mol relative-energy window and Butina lowest-energy representatives. ETKDG-only has no force-field energy filter. Dielectric settings are mechanical sensitivity approximations, not explicit solvent, a validated protein pocket, or proof of bioactive geometry. Relative energies from different protocols are not compared. Finite iteration-limited geometries are retained like production, with optimizer statuses logged.",
            "The matched-minimization candidate arm is experimental and evaluated at reference budget 100; current training uses unminimized ETKDG candidate conformers. Reference preparation is a one-time cost, reported separately. Timing includes real per-molecule preparation/SDF and overlay work under a fixed concurrent CPU-worker budget; excludes neural sampling and pool IPC/startup. It does not predict complete RL epoch duration.",
            "Retained-reference caps are measured as nested prefixes: each pair calls the unchanged production kernel once, and the running score maximum and actual cumulative elapsed time are saved at each cap. This avoids repeating the same pair computations for every larger cap. It is measured prefix execution time rather than a time extrapolated from conformer count. Earlier whole-grid measurements superseded during baseline verification are retained separately and excluded from the final summaries.",
            "",
            "## Failures and limitations",
            "",
            f"Reference failures: {reference.status.ne('ok').sum()}/{len(reference)}. Candidate preparation/minimization failures: {candidates.status.ne('ok').sum()}/{len(candidates)}. Failures remain in denominators. Failed candidates score zero as in training; failed reference scores are undefined.",
            f"Rejected native embedding sentinel results: {rejected_embeddings}. RDKit can return [-1] after timeout; SDWriter previously treated confId=-1 as a request for a default conformer and wrote a 2D depiction. The corrected production generator rejects and counts negative IDs; it retains genuine successful 3D conformers. No coordinates are fabricated or relabeled. {native_reproduction} Any superseded candidate/matching artifacts remain separately archived.",
            "The panel is fingerprint-diverse rather than a random estimate of the full RNN distribution. Holdout labels are assigned to later max-min selections, not an independent random population sample; holdout size is limited. Self/analogue recovery does not prove activity; more references can inflate false-positive similarity. Recommendations apply within the tested grid. Unit tests verify full-scorer parity for all three metrics; an additional broader panel should be checked before deployment.",
            "Across-source canonical deduplication excludes every structure encountered in earlier source pools, not only their selected panel members. The reinforced-source panel therefore emphasizes novelty relative to the saved prior sample. Counts and uncertainty refer to this frozen stress panel, not an unbiased estimate of the full generation distribution.",
            "The primary holdout decision includes every held-out molecule. Separate holdout_in_domain columns provide a secondary chemistry-domain analysis restricted by the existing structural limits (<=45 heavy atoms and <=15 rotatable bonds), with its own explicit denominator. They do not replace or rescue the primary decision. Score-tolerance and preparation-tolerance flags identify whether a primary failure is due to insufficient conformer sampling or the scorer's existing input restrictions.",
            "The primary selection minimizes median candidate latency. Because the generation pilot exposed long tails, selected_settings_mean_latency.csv also applies the same screening-only loss tolerances while minimizing mean latency. This is a secondary throughput-oriented analysis added after observing the timing tail, not a replacement of the primary selection rule. Mean, median, p95 and maximum latency are all exported; neither median nor a fixed-worker per-molecule sum alone predicts a full RL epoch.",
            "Reference-budget stability compares paired seed labels at the largest retained-reference and candidate caps. These are independently built/clustered ensembles rather than nested sets, so larger embedding budgets can lower a particular score. Seed-stability tables distinguish repeatability from count effects. Minimization-effect tables compare the same candidate ensemble before and after mechanics, with molecule-bootstrap intervals; score increases do not demonstrate more accurate poses.",
            f"Additional diagnostic tables: {', '.join(diagnostics)}. Timing tails and iteration-limited optimizer counts are reported even when average scores appear acceptable.",
            "",
            "## Sources",
            "",
            "- https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation",
            "- https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html",
            "- https://www.rdkit.org/docs/source/rdkit.Chem.rdShapeAlign.html",
            "",
            "## Figures",
            "",
        ]
        lines.extend(f"- [{p.name}]({p.relative_to(self.output)})" for p in plots)
        audit = self.output / "environment_audit.json"
        if audit.exists():
            environment = json.loads(audit.read_text())
            if not environment["dimorphite_dl_available"]:
                lines += [
                    "",
                    "## Protonation availability",
                    "",
                    "Dimorphite-DL is absent in the measured environment. The existing protonation helper returns its input unchanged; the acetic-acid probe confirms this. These measurements reproduce that current training behavior and do not represent a validated pH-7.4 protomer distribution. Installing a protonation backend would change the chemistry and require a separate benchmark.",
                ]
        source_audit = self.output / "source_panel_audit.json"
        if source_audit.exists():
            lines += ["", "## Source sample denominators", ""]
            for source in json.loads(source_audit.read_text())["sources"]:
                lines.append(
                    f"- {Path(source['source']).name}: "
                    f"{source['n_invalid_smiles']}/{source['n_raw_rows']} invalid SMILES; "
                    f"{source['n_duplicate_valid_rows']} duplicate valid rows; "
                    f"{source['n_selected']} valid, canonical-unique panel selections."
                )
            lines.append(
                "Invalid source SMILES are excluded before conformer benchmarking; "
                "these source-generation failures are not successful scored molecules. "
                "The selected stress-panel denominator is reported separately above."
            )
        (self.output / "REPORT.md").write_text("\n".join(lines) + "\n")
