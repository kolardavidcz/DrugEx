"""Check selected counts through the complete production scorer after the grid."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from _david.code.conformer.benchmark import EnsembleIO
from _david.code.environment import create_rocs_scorer


class ProductionConfirmation:
    """Measure true batch scoring and compare threaded results with the grid."""

    def __init__(self, output: Path) -> None:
        """Use the frozen panel and existing reference ensembles from a completed run."""
        self.output: Path = output.resolve()

    def run(self) -> None:
        """Compare current counts and the screening-selected Tanimoto counts twice."""
        manifest = json.loads((self.output / "manifest.json").read_text())
        if manifest.get("state") != "complete":
            raise ValueError("Complete the grid before production confirmation")
        destination = self.output / "production_confirmation"
        destination.mkdir(exist_ok=False)
        panel = pd.read_csv(self.output / "molecule_panel.csv")
        measurements = pd.read_csv(self.output / "measurements.csv.gz")
        selected = pd.read_csv(self.output / "selected_settings.csv")
        selected = selected[
            (selected.protocol == "training_mmff94")
            & (selected.candidate_method == "training_etkdg")
            & (selected.matching == "tanimoto")
            & selected.reference_budget.notna()
        ]
        conditions: list[dict[str, Any]] = [
            {
                "name": "current_counts",
                "reference_budget": 100,
                "k_ref_requested": 3,
                "k_candidate_requested": 30,
            }
        ]
        if not selected.empty:
            row = selected.iloc[0]
            condition = {
                "name": "screening_selected",
                **{
                    key: int(row[key])
                    for key in (
                        "reference_budget",
                        "k_ref_requested",
                        "k_candidate_requested",
                    )
                },
            }
            if any(
                condition[key] != conditions[0][key]
                for key in condition
                if key != "name"
            ):
                conditions.append(condition)
        secondary = self.output / "selected_settings_mean_latency.csv"
        if secondary.exists():
            options = pd.read_csv(secondary)
            options = options[
                (options.protocol == "training_mmff94")
                & (options.candidate_method == "training_etkdg")
                & (options.matching == "tanimoto")
                & options.reference_budget.notna()
            ]
            if not options.empty:
                condition = {
                    "name": "mean_latency_selected",
                    **{
                        key: int(options.iloc[0][key])
                        for key in (
                            "reference_budget",
                            "k_ref_requested",
                            "k_candidate_requested",
                        )
                    },
                }
                if not any(
                    all(
                        condition[key] == existing[key]
                        for key in condition
                        if key != "name"
                    )
                    for existing in conditions
                ):
                    conditions.append(condition)
        os.environ["DRUGEX_SCORE_NJOBS"] = "8"
        os.environ["DRUGEX_MP_CONTEXT"] = "spawn"
        molecules = [Chem.MolFromSmiles(smile) for smile in panel.smiles]
        if any(mol is None for mol in molecules):
            raise ValueError("Frozen panel contains an invalid SMILES")
        jobs = [(condition, repeat) for repeat in range(2) for condition in conditions]
        np.random.default_rng(20261027).shuffle(jobs)
        rows: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        for condition, repeat in jobs:
            references = EnsembleIO.read(
                self.output
                / "references"
                / f"training_mmff94_{condition['reference_budget']}_0.sdf"
            )
            reference_file = destination / f"{condition['name']}_references.sdf"
            with Chem.SDWriter(str(reference_file)) as writer:
                for reference in references[: condition["k_ref_requested"]]:
                    writer.write(reference)
            scorer = create_rocs_scorer(
                reference_file,
                max_conformers=condition["k_candidate_requested"],
                score_variant="tanimoto",
                conformer_threads=0,
                n_jobs=8,
            )
            load = os.getloadavg()
            start = time.perf_counter()
            actual = scorer.getScores(molecules)
            elapsed = time.perf_counter() - start
            if actual.shape != (len(panel), 1) or not np.isfinite(actual).all():
                raise ValueError("Production scorer returned invalid batch scores")
            if actual.min() < 0 or actual.max() > 2:
                raise ValueError("Production raw-combo scores are outside [0,2]")
            expected = measurements[
                (measurements.protocol == "training_mmff94")
                & (measurements.candidate_method == "training_etkdg")
                & (measurements.matching == "tanimoto")
                & (measurements.replicate == 0)
            ]
            for key in ("reference_budget", "k_ref_requested", "k_candidate_requested"):
                expected = expected[expected[key] == condition[key]]
            expected = expected.set_index("molecule_id").loc[panel.molecule_id]
            delta = actual[:, 0] - expected.raw_combo.to_numpy()
            timings.append(
                {
                    **condition,
                    "repeat": repeat,
                    "n_molecules": len(panel),
                    "batch_seconds": elapsed,
                    "candidate_threads": 0,
                    "scorer_workers": 8,
                    "load_average_before": load,
                    "max_absolute_difference_from_grid": float(abs(delta).max()),
                    "n_differences_above_1e_minus6": int((abs(delta) > 1e-6).sum()),
                }
            )
            for index, molecule in enumerate(panel.itertuples()):
                rows.append(
                    {
                        **condition,
                        "repeat": repeat,
                        "molecule_id": molecule.molecule_id,
                        "panel_split": molecule.panel_split,
                        "raw_combo": float(actual[index, 0]),
                        "grid_raw_combo": float(expected.raw_combo.iloc[index]),
                        "difference_from_grid": float(delta[index]),
                    }
                )
            pd.DataFrame(rows).to_csv(destination / "scores.csv", index=False)
            (destination / "timings.json").write_text(
                json.dumps(timings, indent=2) + "\n"
            )
            print(
                f"production confirmation: {condition['name']} repeat {repeat}, {elapsed:.2f}s, max grid difference {abs(delta).max():.6g}",
                flush=True,
            )
        (destination / "README.md").write_text(
            "# Full production scorer confirmation\n\n"
            "The same frozen panel is scored through create_rocs_scorer and RDKitROCSScorer.getScores, including live generation, SDF I/O and pool startup/IPC. Candidate threads=0 matches the current training factory; scoring uses eight spawned workers. Counts are selected on screening only. Two timing repeats use the fixed production embedding seed and are not additional independent chemistry replicates. Grid preparation uses one candidate thread, so deviations are recorded rather than assumed absent. Times apply to this panel and resource setting, not a complete RL epoch. The current-count condition is B=100, R=3, C=30.\n"
        )
