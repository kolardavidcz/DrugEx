"""Reproducible reference/candidate conformer benchmark using the training scorer."""

from __future__ import annotations

import hashlib
import faulthandler
import json
import math
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolAlign,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import _score_single_reference

from _david.code.config import DEFAULT_MOLECULE
from _david.code.environment import PrepareMolecule


@dataclass(frozen=True)
class ReferenceProtocol:
    """A mechanical sensitivity condition, not a validated solvent model."""

    name: str
    force_field: str
    dielectric: float = 1.0
    dielectric_model: int = 1
    rmsd_cutoff: float = 1.25
    energy_window: float | None = 8.0
    max_iterations: int = 1000

    @staticmethod
    def suite() -> tuple[ReferenceProtocol, ...]:
        """Hold energy window and clustering fixed while comparing force fields."""
        return (
            ReferenceProtocol("training_mmff94", "MMFF94"),
            ReferenceProtocol("mmff94s_eps1", "MMFF94s"),
            ReferenceProtocol("uff", "UFF"),
            ReferenceProtocol("mmff94s_eps6p5", "MMFF94s", dielectric=6.5),
            ReferenceProtocol("mmff94s_eps78p4", "MMFF94s", dielectric=78.4),
            ReferenceProtocol("mmff94s_distance", "MMFF94s", dielectric_model=2),
            ReferenceProtocol("etkdg_only", "none", energy_window=None),
            ReferenceProtocol("mmff94s_tight_rmsd", "MMFF94s", rmsd_cutoff=0.75),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Predeclared grids, replicate seeds, and resource limits."""

    reference_budgets: tuple[int, ...] = (32, 100, 250)
    reference_counts: tuple[int, ...] = (1, 2, 3, 5, 8)
    candidate_counts: tuple[int, ...] = (1, 5, 10, 20, 30, 50)
    reference_seeds: tuple[int, ...] = (42, 43, 44)
    candidate_seeds: tuple[int, ...] = (0xC0FFEE, 0xC0FFEF, 0xC0FFF0)
    matching: tuple[str, ...] = ("tanimoto", "tversky_ref", "tversky_subshape")
    generated_per_source: int = 12
    workers: int = 8
    candidate_timeout: int = 20
    candidate_methods: tuple[str, ...] = ("training_etkdg", "matched_minimized")
    minimized_protocols: tuple[str, ...] = (
        "training_mmff94",
        "mmff94s_eps1",
        "mmff94s_eps78p4",
        "uff",
    )
    minimized_reference_budget: int = 100
    selection_seed: int = 20261027

    def __post_init__(self) -> None:
        """Reject invalid grids rather than silently changing the experiment."""
        for grid in (
            self.reference_budgets,
            self.reference_counts,
            self.candidate_counts,
        ):
            if not grid or min(grid) < 1 or tuple(sorted(set(grid))) != grid:
                raise ValueError("Conformer grids must be positive, sorted, and unique")
        if len(self.reference_seeds) != len(self.candidate_seeds):
            raise ValueError("Reference and candidate seed replicates must match")
        if self.workers < 1 or self.generated_per_source < 1:
            raise ValueError("Worker and panel sizes must be positive")


class MoleculePanel:
    """Freeze a scaffold-diverse model panel plus explicitly separated controls."""

    CONTROLS: tuple[tuple[str, str], ...] = (
        ("EGCG", DEFAULT_MOLECULE),
        ("epigallocatechin", "C1[C@H]([C@H](Oc2cc(O)cc(O)c21)c1cc(O)c(O)c(O)c1)O"),
        ("epicatechin", "C1[C@H]([C@H](Oc2cc(O)cc(O)c21)c1ccc(O)c(O)c1)O"),
        ("gallic_acid", "O=C(O)c1cc(O)c(O)c(O)c1"),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("benzene", "c1ccccc1"),
        ("flexible_amine", "CC(C)NCC(O)COc1cccc2ccccc12"),
    )

    @staticmethod
    def describe(identifier: str, smiles: str, source: str) -> dict[str, Any]:
        """Canonicalize and describe one molecule without using saved reward values."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid panel molecule: {identifier}")
        return {
            "molecule_id": identifier,
            "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            "source": source,
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "molecular_weight": Descriptors.MolWt(mol),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        }

    @classmethod
    def build(cls, source_files: Sequence[Path], count: int, seed: int) -> pd.DataFrame:
        """Select by fingerprint max-min diversity, independently of ROCS outcomes."""
        rows = [cls.describe(name, smi, "control") for name, smi in cls.CONTROLS]
        seen = {r["smiles"] for r in rows}
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        for source_index, source in enumerate(source_files):
            frame = pd.read_csv(source)
            pool: list[dict[str, Any]] = []
            for original_index, value in frame["SMILES"].items():
                if not isinstance(value, str) or Chem.MolFromSmiles(value) is None:
                    continue
                row = cls.describe(
                    f"source{source_index}_row{original_index}",
                    value,
                    f"model_{source_index}",
                )
                if row["smiles"] in seen:
                    continue
                seen.add(row["smiles"])
                pool.append(row)
            if len(pool) < count:
                raise ValueError(f"Not enough unique valid molecules in {source}")
            fps = [
                generator.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in pool
            ]
            first = int(np.random.default_rng(seed + source_index).integers(len(pool)))
            selected = [first]
            distances = np.ones(len(pool))
            while len(selected) < count:
                similarity = np.array(
                    DataStructs.BulkTanimotoSimilarity(fps[selected[-1]], fps)
                )
                distances = np.minimum(distances, 1 - similarity)
                distances[selected] = -1
                selected.append(int(distances.argmax()))
            held_out = max(1, count // 3)
            for position, index in enumerate(selected):
                row = pool[index]
                row["panel_split"] = (
                    "holdout" if position >= count - held_out else "screening"
                )
                rows.append(row)
        for row in rows:
            row.setdefault("panel_split", "control")
        return pd.DataFrame(rows)


class EnsembleIO:
    """Serialize conformers in the same one-record-per-conformer form as training."""

    @staticmethod
    def read(path: Path) -> list[Chem.Mol]:
        """Read valid conformers; malformed records are explicit benchmark errors."""
        result = list(Chem.SDMolSupplier(str(path), removeHs=False))
        if any(m is None for m in result):
            raise ValueError(f"Malformed conformer record in {path}")
        return result

    @staticmethod
    def write(path: Path, mol: Chem.Mol, identifiers: Sequence[int]) -> None:
        """Write the selected conformers in their deterministic selection order."""
        with Chem.SDWriter(str(path)) as writer:
            for identifier in identifiers:
                writer.write(mol, confId=int(identifier))


class MechanicalOptimizer:
    """Apply explicit mechanics and preserve convergence diagnostics."""

    @staticmethod
    def minimize(
        mol: Chem.Mol, protocol: ReferenceProtocol
    ) -> tuple[dict[int, float], dict[str, int]]:
        """Return finite energies and optimizer statuses without substituting another FF."""
        energies: dict[int, float] = {}
        statuses: dict[str, int] = {}
        properties = None
        if protocol.force_field.startswith("MMFF"):
            properties = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant=protocol.force_field
            )
            if properties is None:
                raise ValueError(f"Missing {protocol.force_field} parameters")
            properties.SetMMFFDielectricModel(protocol.dielectric_model)
            properties.SetMMFFDielectricConstant(protocol.dielectric)
        for conformer in mol.GetConformers():
            cid = conformer.GetId()
            if protocol.force_field == "none":
                energies[cid] = 0.0
                continue
            field = (
                AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if protocol.force_field == "UFF"
                else AllChem.MMFFGetMoleculeForceField(mol, properties, confId=cid)
            )
            if field is None:
                raise ValueError(f"Force field unavailable for conformer {cid}")
            status = int(field.Minimize(maxIts=protocol.max_iterations))
            statuses[str(status)] = statuses.get(str(status), 0) + 1
            energy = float(field.CalcEnergy())
            # Production reference preparation retains finite status-1 geometries too.
            if status >= 0 and math.isfinite(energy):
                energies[cid] = energy
        if not energies:
            raise ValueError("No conformers have finite energies")
        return energies, statuses


class ReferenceBuilder:
    """Reproduce production reference preparation with explicit sensitivity conditions."""

    @staticmethod
    def select(
        mol: Chem.Mol, energies: dict[int, float], protocol: ReferenceProtocol
    ) -> list[int]:
        """Pick the lowest-energy representative of each Butina heavy-atom RMSD basin."""
        minimum = min(energies.values())
        identifiers = [
            cid
            for cid, energy in energies.items()
            if protocol.energy_window is None
            or energy - minimum <= protocol.energy_window
        ]
        if len(identifiers) < 2:
            return identifiers
        heavy = Chem.RemoveHs(mol)
        distances = [
            rdMolAlign.GetBestRMS(heavy, heavy, identifiers[i], identifiers[j])
            for i in range(len(identifiers))
            for j in range(i)
        ]
        clusters = Butina.ClusterData(
            distances, len(identifiers), protocol.rmsd_cutoff, isDistData=True
        )
        representatives = [
            min((identifiers[i] for i in cluster), key=energies.get)
            for cluster in clusters
        ]
        return sorted(representatives, key=energies.get)

    @classmethod
    def run(cls, job: dict[str, Any]) -> dict[str, Any]:
        """Measure a real reference build at the requested embedding budget."""
        start = time.perf_counter()
        path = Path(job["path"])
        protocol = ReferenceProtocol(**job["protocol"])
        try:
            mol, _ = PrepareMolecule().load_molecule(DEFAULT_MOLECULE)
            mol = Chem.AddHs(mol)
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = job["seed"]
            parameters.numThreads = 1
            parameters.pruneRmsThresh = 0.35
            parameters.timeout = 20
            ids = list(
                AllChem.EmbedMultipleConfs(
                    mol, numConfs=job["budget"], params=parameters
                )
            )
            if not ids:
                raise ValueError("Reference embedding failed")
            embedding_s = time.perf_counter() - start
            energies, statuses = MechanicalOptimizer.minimize(mol, protocol)
            optimization_s = time.perf_counter() - start - embedding_s
            representatives = cls.select(mol, energies, protocol)
            EnsembleIO.write(path, mol, representatives)
            result = {
                **job,
                "status": "ok",
                "embedded": len(ids),
                "available_basins": len(representatives),
                "optimizer_statuses": statuses,
                "embedding_s": embedding_s,
                "optimization_s": optimization_s,
            }
        except (ValueError, RuntimeError) as error:
            result = {
                **job,
                "status": "failed",
                "error": str(error),
                "embedded": 0,
                "available_basins": 0,
            }
        result["reference_seconds"] = time.perf_counter() - start
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result


class SeededTrainingGenerator(RDKitConformerGenerator):
    """Use the production candidate pipeline, varying only its embedding seed."""

    benchmark_seed: int

    def _create_fresh_etkdg(self) -> Any:
        """Preserve production ETKDG settings and set the replicate seed explicitly."""
        parameters = super()._create_fresh_etkdg()
        parameters.randomSeed = self.benchmark_seed
        return parameters


class CandidateBuilder:
    """Measure true generation at each count, without slicing a larger cached ensemble."""

    @staticmethod
    def run(job: dict[str, Any]) -> dict[str, Any]:
        """Use the exact protonation, tautomer, stereo, and SDF path used in training."""
        import tempfile

        start = time.perf_counter()
        path = Path(job["path"])
        try:
            generator = SeededTrainingGenerator(
                max_conformers=job["count"],
                max_isomers=4,
                max_heavy_atoms=45,
                max_rotatable_bonds=15,
                num_threads=1,
                timeout=job["timeout"],
                show_progress=False,
            )
            generator.benchmark_seed = job["seed"]
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(generator.genConformers([job["smiles"]], temporary))
                path.write_bytes(source.read_bytes())
            molecules = EnsembleIO.read(path) if path.stat().st_size else []
            result = {
                **job,
                "status": "ok" if molecules else "no_conformers",
                "actual_conformers": len(molecules),
                "optimizer_statuses": {},
                "rejected_embedding_results": generator.rejected_embedding_results,
            }
        except (ValueError, RuntimeError) as error:
            result = {
                **job,
                "status": "failed",
                "error": str(error),
                "actual_conformers": 0,
            }
        result["generation_seconds"] = time.perf_counter() - start
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    @staticmethod
    def minimize(job: dict[str, Any]) -> dict[str, Any]:
        """Measure an additional, explicitly experimental candidate minimization arm."""
        start = time.perf_counter()
        source = job["parent"]
        result = {
            **source,
            "path": job["path"],
            "method": "matched_minimized",
            "protocol": job["protocol"]["name"],
        }
        path = Path(job["path"])
        statuses: dict[str, int] = {}
        try:
            if source["status"] != "ok":
                result["status"] = source["status"]
            else:
                protocol = ReferenceProtocol(**job["protocol"])
                molecules = EnsembleIO.read(Path(source["path"]))
                with Chem.SDWriter(str(path)) as writer:
                    for mol in molecules:
                        _, state = MechanicalOptimizer.minimize(mol, protocol)
                        for key, value in state.items():
                            statuses[key] = statuses.get(key, 0) + value
                        writer.write(mol)
        except (ValueError, RuntimeError) as error:
            result.update(status="failed", error=str(error), actual_conformers=0)
        result["optimizer_statuses"] = statuses
        result["minimization_seconds"] = time.perf_counter() - start
        result["generation_seconds"] = (
            source["generation_seconds"] + result["minimization_seconds"]
        )
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        return result


class TrainingKernel:
    """Match the production per-conformer maximum with measured scoring latency."""

    @staticmethod
    def score(
        queries: Sequence[Chem.Mol], references: Sequence[Chem.Mol], matching: str
    ) -> float:
        """Call the exact training overlay kernel and retain its raw combo scale [0,2]."""
        return max(
            (
                _score_single_reference(
                    query, reference, "TanimotoCombo", True, score_variant=matching
                )
                for query in queries
                for reference in references
            ),
            default=0.0,
        )

    @staticmethod
    def prefixes(
        queries: Sequence[Chem.Mol], references: Sequence[Chem.Mol], matching: str
    ) -> dict[int, tuple[float, float]]:
        """Measure nested reference prefixes in one pass through the exact pair kernel.

        Each prefix timer includes every pair needed to score that ensemble. Reusing
        its running maximum avoids evaluating those same pairs again for larger caps;
        durations are measured cumulative work, never extrapolated from conformer count.
        """
        start = time.perf_counter()
        best = 0.0
        result: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
        for count, reference in enumerate(references, 1):
            best = max(
                best,
                max(
                    (
                        _score_single_reference(
                            query,
                            reference,
                            "TanimotoCombo",
                            True,
                            score_variant=matching,
                        )
                        for query in queries
                    ),
                    default=0.0,
                ),
            )
            result[count] = (best, time.perf_counter() - start)
        return result

    @classmethod
    def run(cls, job: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate all reference caps and metrics for one paired candidate/reference job."""
        reference, candidate = job["reference"], job["candidate"]
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        faulthandler.dump_traceback_later(120)
        activity = Path(reference["path"]).parent.parent / "workers"
        activity.mkdir(exist_ok=True)
        worker_file = activity / f"{os.getpid()}.json"
        worker_file.write_text(
            json.dumps(
                {
                    "reference": reference["path"],
                    "candidate": candidate["path"],
                    "started_unix": time.time(),
                }
            )
        )
        references = (
            EnsembleIO.read(Path(reference["path"]))
            if reference["status"] == "ok"
            else []
        )
        queries = (
            EnsembleIO.read(Path(candidate["path"]))
            if candidate["status"] == "ok"
            else []
        )
        rows: list[dict[str, Any]] = []
        # Randomize metric order; each cap measures an actual cumulative scoring prefix.
        combinations = [
            (count, metric)
            for count in job["reference_counts"]
            for metric in job["matching"]
        ]
        rng = np.random.default_rng(job["seed"])
        rng.shuffle(combinations)
        measured: dict[tuple[int, str], tuple[float, float]] = {}
        metrics = list(job["matching"])
        rng.shuffle(metrics)
        for metric in metrics:
            prefixes = cls.prefixes(
                queries, references[: max(job["reference_counts"])], metric
            )
            measured.update(
                {(count, metric): value for count, value in prefixes.items()}
            )
        for count, metric in combinations:
            actual = min(count, len(references))
            key = (actual, metric)
            score, seconds = measured[key]
            status = (
                reference["status"]
                if reference["status"] != "ok"
                else candidate["status"]
            )
            rows.append(
                {
                    "protocol": reference["protocol"]["name"],
                    "reference_budget": reference["budget"],
                    "replicate": reference["replicate"],
                    "reference_seed": reference["seed"],
                    "candidate_seed": candidate["seed"],
                    "molecule_id": candidate["molecule_id"],
                    "source": candidate["source"],
                    "candidate_method": candidate["method"],
                    "k_ref_requested": count,
                    "k_ref_actual": actual,
                    "k_candidate_requested": candidate["count"],
                    "k_candidate_actual": candidate["actual_conformers"],
                    "matching": metric,
                    "raw_combo": score if reference["status"] == "ok" else np.nan,
                    "modified_score": score / 2
                    if reference["status"] == "ok"
                    else np.nan,
                    "generation_seconds": candidate["generation_seconds"],
                    "alignment_seconds": seconds,
                    "candidate_total_seconds": candidate["generation_seconds"]
                    + seconds,
                    "reference_seconds": reference["reference_seconds"],
                    "status": status,
                    "timing_mode": "measured_incremental_reference_prefix",
                }
            )
        faulthandler.cancel_dump_traceback_later()
        worker_file.unlink()
        return rows


class ConformerBenchmark:
    """Run bounded independent CPU tasks and persist the full reproducible experiment."""

    def __init__(
        self, output: Path, config: BenchmarkConfig, resume: bool = False
    ) -> None:
        """Create fresh results or explicitly resume a validated interrupted run."""
        self.output: Path = output.resolve()
        self.config: BenchmarkConfig = config
        self.resume: bool = resume
        self.output.mkdir(parents=True, exist_ok=resume)
        for folder in ("references", "candidates", "minimized"):
            (self.output / folder).mkdir(exist_ok=resume)

    def execute(
        self, phase: str, operation: Any, jobs: list[dict[str, Any]]
    ) -> list[Any]:
        """Run process-isolated tasks with explicit spawn and a fixed worker budget."""
        results: list[Any] = []
        start = time.monotonic()
        with ProcessPoolExecutor(
            max_workers=self.config.workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(operation, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 25 == 0 or len(results) == len(jobs):
                    print(
                        f"{phase}: {len(results)}/{len(jobs)} ({time.monotonic() - start:.1f}s)",
                        flush=True,
                    )
                    (self.output / "progress.json").write_text(
                        json.dumps(
                            {
                                "phase": phase,
                                "completed": len(results),
                                "total": len(jobs),
                                "elapsed_s": time.monotonic() - start,
                            }
                        )
                        + "\n"
                    )
        return results

    def build(
        self, phase: str, operation: Any, jobs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Reuse persisted build records only during an explicitly validated resume."""
        cached: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        for job in jobs:
            path = Path(job["path"]).with_suffix(".json")
            if self.resume and path.exists():
                record = json.loads(path.read_text())
                if record["status"] == "ok" and not Path(record["path"]).is_file():
                    raise ValueError(f"Missing cached ensemble: {record['path']}")
                cached.append(record)
            else:
                pending.append(job)
        return cached + (self.execute(phase, operation, pending) if pending else [])

    @staticmethod
    def measurement_key(row: dict[str, Any]) -> tuple[Any, ...]:
        """Identify one complete scoring task independently of cell evaluation order."""
        return tuple(
            row[name]
            for name in (
                "protocol",
                "reference_budget",
                "replicate",
                "molecule_id",
                "candidate_method",
                "k_candidate_requested",
            )
        )

    def run(self, sources: Sequence[Path]) -> None:
        """Execute reference, candidate, matching, and experimental minimization sweeps."""
        started = time.time()
        previous = (
            json.loads((self.output / "manifest.json").read_text())
            if self.resume
            else None
        )
        if previous is not None:
            if previous["config"] != json.loads(json.dumps(asdict(self.config))):
                raise ValueError(
                    "Resume configuration differs from the original experiment"
                )
            if previous["protocols"] != [asdict(p) for p in ReferenceProtocol.suite()]:
                raise ValueError(
                    "Resume reference protocols differ from the original experiment"
                )
            inputs = {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            }
            if previous["inputs"] != inputs:
                raise ValueError(
                    "Resume source files differ from the original experiment"
                )
        panel = (
            pd.read_csv(self.output / "molecule_panel.csv")
            if self.resume
            else MoleculePanel.build(
                sources, self.config.generated_per_source, self.config.selection_seed
            )
        )
        panel.to_csv(self.output / "molecule_panel.csv", index=False)
        root = Path(__file__).resolve().parents[3]
        tracked = [
            root / name
            for name in (
                "_david/code/conformer/benchmark.py",
                "_david/code/conformer/benchmark_cli.py",
                "_david/code/conformer/benchmark_analysis.py",
                "_david/code/visualization.py",
                "_david/code/config.py",
                "_david/code/environment.py",
                "drugex/training/scorers/rocs_rdkit.py",
                "drugex/training/scorers/conformer_generators.py",
            )
        ]
        manifest = {
            "config": asdict(self.config),
            "protocols": [asdict(p) for p in ReferenceProtocol.suite()],
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "code_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in tracked
            },
            "inputs": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "rdkit": rdBase.rdkitVersion,
            "python": platform.python_version(),
            "host": platform.node(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "load_average": os.getloadavg(),
            "start_unix": started,
            "timing_definition": "Measured per-molecule generation/SDF plus actual production overlay kernel; excludes amortized reference prep, neural sampling, and multiprocessing IPC. Concurrent fixed-worker conditions, not isolated CPU microbenchmarks.",
            "score_definition": "Raw combo [0,2], modified=raw/2, within-metric comparison only; no observed-max rescaling.",
        }
        if previous is not None:
            manifest = previous
            manifest.setdefault("resumptions", []).append(
                {
                    "start_unix": started,
                    "code_sha256": {
                        str(p.relative_to(root)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in tracked
                    },
                }
            )
        snapshot = (
            self.output
            / "source_snapshot"
            / f"segment_{len(manifest.get('resumptions', []))}"
        )
        for source in tracked:
            destination = snapshot / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        reference_jobs = [
            {
                "protocol": asdict(protocol),
                "budget": budget,
                "seed": seed,
                "replicate": replicate,
                "path": str(
                    self.output
                    / "references"
                    / f"{protocol.name}_{budget}_{replicate}.sdf"
                ),
            }
            for protocol in ReferenceProtocol.suite()
            for budget in self.config.reference_budgets
            for replicate, seed in enumerate(self.config.reference_seeds)
        ]
        references = self.build("reference", ReferenceBuilder.run, reference_jobs)
        pd.DataFrame(references).to_csv(
            self.output / "reference_builds.csv", index=False
        )
        candidate_jobs = [
            {
                "molecule_id": row.molecule_id,
                "smiles": row.smiles,
                "source": row.source,
                "seed": seed,
                "replicate": replicate,
                "count": count,
                "method": "training_etkdg",
                "timeout": self.config.candidate_timeout,
                "path": str(
                    self.output
                    / "candidates"
                    / f"{row.molecule_id}_{count}_{replicate}.sdf"
                ),
            }
            for row in panel.itertuples()
            for replicate, seed in enumerate(self.config.candidate_seeds)
            for count in self.config.candidate_counts
        ]
        candidates = self.build("candidate", CandidateBuilder.run, candidate_jobs)
        minimized_jobs = (
            [
                {
                    "parent": candidate,
                    "protocol": asdict(protocol),
                    "path": str(
                        self.output
                        / "minimized"
                        / f"{protocol.name}_{Path(candidate['path']).name}"
                    ),
                }
                for protocol in ReferenceProtocol.suite()
                if protocol.name in self.config.minimized_protocols
                for candidate in candidates
            ]
            if "matched_minimized" in self.config.candidate_methods
            else []
        )
        minimized = (
            self.build("minimized", CandidateBuilder.minimize, minimized_jobs)
            if minimized_jobs
            else []
        )
        pd.DataFrame(candidates + minimized).to_csv(
            self.output / "candidate_builds.csv", index=False
        )
        jobs: list[dict[str, Any]] = []
        for reference in references:
            for candidate in candidates + minimized:
                if candidate["replicate"] != reference["replicate"]:
                    continue
                if candidate["method"] == "matched_minimized" and (
                    candidate["protocol"] != reference["protocol"]["name"]
                    or reference["budget"] != self.config.minimized_reference_budget
                ):
                    continue
                jobs.append(
                    {
                        "reference": reference,
                        "candidate": candidate,
                        "reference_counts": self.config.reference_counts,
                        "matching": self.config.matching,
                        "seed": reference["seed"] + candidate["count"],
                    }
                )
        # A task is reusable only when every requested reference/metric cell was persisted.
        measurement_file = self.output / "measurements.jsonl"
        completed_keys: set[tuple[Any, ...]] = set()
        if self.resume and measurement_file.exists():
            existing = pd.read_json(measurement_file, lines=True)
            expected = len(self.config.reference_counts) * len(self.config.matching)
            for _, group in existing.groupby(
                [
                    "protocol",
                    "reference_budget",
                    "replicate",
                    "molecule_id",
                    "candidate_method",
                    "k_candidate_requested",
                ]
            ):
                if (
                    len(group) != expected
                    or group[["k_ref_requested", "matching"]].drop_duplicates().shape[0]
                    != expected
                ):
                    raise ValueError(
                        "Interrupted result contains an incomplete or duplicate scoring task"
                    )
                completed_keys.add(self.measurement_key(group.iloc[0].to_dict()))
        total = len(jobs)
        jobs = [
            job
            for job in jobs
            if (
                job["reference"]["protocol"]["name"],
                job["reference"]["budget"],
                job["reference"]["replicate"],
                job["candidate"]["molecule_id"],
                job["candidate"]["method"],
                job["candidate"]["count"],
            )
            not in completed_keys
        ]
        np.random.default_rng(self.config.selection_seed).shuffle(jobs)
        print(
            f"matching: {len(completed_keys)} cached tasks, {len(jobs)} pending",
            flush=True,
        )
        with measurement_file.open("a" if self.resume else "w") as handle:
            start = time.monotonic()
            with ProcessPoolExecutor(
                max_workers=self.config.workers, mp_context=mp.get_context("spawn")
            ) as executor:
                futures = [executor.submit(TrainingKernel.run, job) for job in jobs]
                for completed, future in enumerate(as_completed(futures), 1):
                    try:
                        rows = future.result()
                    except BaseException:
                        # Report immediately: executor shutdown otherwise waits for pending work.
                        traceback.print_exc()
                        for pending in futures:
                            pending.cancel()
                        raise
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if completed % 50 == 0 or completed == len(jobs):
                        progress = {
                            "phase": "matching",
                            "completed": completed + len(completed_keys),
                            "total": total,
                            "elapsed_s": time.monotonic() - start,
                        }
                        (self.output / "progress.json").write_text(
                            json.dumps(progress) + "\n"
                        )
                        print(json.dumps(progress), flush=True)
        frame = pd.read_json(self.output / "measurements.jsonl", lines=True)
        expected_rows = (
            total * len(self.config.reference_counts) * len(self.config.matching)
        )
        identity = [
            "protocol",
            "reference_budget",
            "replicate",
            "molecule_id",
            "candidate_method",
            "k_candidate_requested",
            "k_ref_requested",
            "matching",
        ]
        if len(frame) != expected_rows or frame.duplicated(identity).any():
            raise ValueError("Matching results are incomplete or contain duplicate cells")
        frame.to_csv(self.output / "measurements.csv.gz", index=False)
        manifest.update(
            state="complete",
            wall_seconds=time.time() - manifest["start_unix"],
            measurement_rows=len(frame),
        )
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
