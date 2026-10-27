"""Measure EGCG self-recovery with preserved identity and no wall-clock timeouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign
from rdkit.ML.Cluster import Butina

from drugex.molecules.converters.standardizers import DefaultStandardizer
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rocs_rdkit import _score_single_reference

from _david.code.config import DEFAULT_MOLECULE
from _david.code.environment import PrepareMolecule


@dataclass(frozen=True)
class StudySettings:
    """Explicit finite search budgets; neither embedding nor scoring has a time limit."""

    reference_attempts: int = 1000
    reference_caps: tuple[int, ...] = (1, 2, 3, 5, 10, 0)  # Zero means all basins.
    candidate_requests: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 250, 500)
    seeds: tuple[int, ...] = (42, 43, 44)
    metrics: tuple[str, ...] = ("tanimoto", "tversky_subshape")
    rmsd_cutoff: float = 1.25
    energy_window: float = 8.0
    max_iterations: int = 10000
    force_tolerance: float = 1e-5
    energy_tolerance: float = 1e-8


@dataclass(frozen=True)
class ReferenceMethod:
    """One mechanics condition; dielectric values do not implement explicit solvent."""

    name: str
    dielectric: float = 1.0
    dielectric_model: int = 1

    @staticmethod
    def suite() -> tuple[ReferenceMethod, ...]:
        """Use the five user-selected conditions with the same energy/RMSD filters."""
        return (
            ReferenceMethod("MMFF94s eps=1"),
            ReferenceMethod("UFF"),
            ReferenceMethod("MMFF94s eps=6.5", 6.5),
            ReferenceMethod("MMFF94s eps=78.4", 78.4),
            ReferenceMethod("MMFF94s distance", dielectric_model=2),
        )


@dataclass
class Ensemble:
    """Actual retained conformers and their measured preparation cost."""

    molecules: list[Chem.Mol]
    seconds: float
    embedded: int


@dataclass(frozen=True)
class Measurement:
    """One reference-cap/candidate-budget cell for one seed and scoring function."""

    method: str
    metric: str
    seed: int
    reference_cap: str
    reference_actual: int
    reference_available: int
    reference_embedded: int
    candidate_requested: int
    candidate_actual: int
    raw_combo: float
    rocs_fraction: float
    reference_seconds: float
    generation_seconds: float
    alignment_seconds: float
    time_seconds: float


@dataclass(frozen=True)
class Control:
    """A separately labeled identity, same-geometry, or matched-mechanics control."""

    method: str
    metric: str
    seed: int
    control: str
    raw_combo: float
    rocs_fraction: float
    reference_identity: str
    candidate_identity: str
    extra_preparation_seconds: float = 0.0


class SeededGenerator(RDKitConformerGenerator):
    """Use training chemistry with one CPU thread, a paired seed, and timeout zero."""

    def __init__(self, count: int, seed: int) -> None:
        """Vary only the requested budget and embedding seed from training settings."""
        super().__init__(max_conformers=count, num_threads=1, timeout=0)
        self.seed: int = seed

    def _create_fresh_etkdg(self) -> rdDistGeom.EmbedParameters:
        """Make the effective no-timeout setting explicit rather than relying on config."""
        parameters = super()._create_fresh_etkdg()
        parameters.randomSeed = self.seed
        parameters.timeout = 0
        return parameters


class EGCGSelfRecovery:
    """Run a small, auditable experiment without selecting training hyperparameters."""

    def __init__(self, output: Path, settings: StudySettings) -> None:
        """Create fresh output and freeze the expected isomeric identity."""
        self.output: Path = output.resolve()
        self.settings: StudySettings = settings
        self.expected: str = self.identity(Chem.MolFromSmiles(DEFAULT_MOLECULE))
        if self.output.exists() and any(
            p.name != ".gitignore" for p in self.output.iterdir()
        ):
            raise FileExistsError(
                "Use fresh output; existing study files are preserved"
            )
        self.output.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(mol: Chem.Mol) -> str:
        """Compare canonical isomeric SMILES, including charge and stereochemistry."""
        return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)

    def checked_record(self, mol: Chem.Mol, cid: int = 0) -> Chem.Mol:
        """Use production SDF precision and reject altered identity or invalid coordinates."""
        record = Chem.MolFromMolBlock(
            Chem.MolToMolBlock(mol, confId=cid), removeHs=False
        )
        if record is None or self.identity(record) != self.expected:
            raise ValueError("Reference/candidate identity differs from specified EGCG")
        conf = record.GetConformer()
        if not conf.Is3D() or not np.isfinite(conf.GetPositions()).all():
            raise ValueError("A conformer has no finite 3D geometry")
        return record

    def embed_reference_pool(self, seed: int) -> tuple[Chem.Mol, float]:
        """Give every mechanics condition the same stereochemically fixed ETKDG starts."""
        start = time.perf_counter()
        mol, _ = PrepareMolecule().load_molecule(DEFAULT_MOLECULE)
        if self.identity(mol) != self.expected:
            raise ValueError("Reference standardization removed stereochemistry")
        mol = Chem.AddHs(mol)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        parameters.numThreads = 1
        parameters.pruneRmsThresh = 0.35
        parameters.timeout = 0
        identifiers = list(
            AllChem.EmbedMultipleConfs(
                mol, numConfs=self.settings.reference_attempts, params=parameters
            )
        )
        if not identifiers or any(cid < 0 for cid in identifiers):
            raise RuntimeError(
                "Reference ETKDG failed; no fallback coordinates are used"
            )
        return mol, time.perf_counter() - start

    def minimize(self, mol: Chem.Mol, method: ReferenceMethod) -> dict[int, float]:
        """Require native optimizer convergence at tighter-than-default tolerances."""
        properties = None
        if method.name != "UFF":
            properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if properties is None:
                raise RuntimeError("MMFF94s parameters unavailable")
            properties.SetMMFFDielectricConstant(method.dielectric)
            properties.SetMMFFDielectricModel(method.dielectric_model)
        energies: dict[int, float] = {}
        for conf in mol.GetConformers():
            cid = conf.GetId()
            field = (
                AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if method.name == "UFF"
                else AllChem.MMFFGetMoleculeForceField(mol, properties, confId=cid)
            )
            if field is None:
                raise RuntimeError(f"Force-field construction failed: {method.name}")
            status = field.Minimize(
                maxIts=self.settings.max_iterations,
                forceTol=self.settings.force_tolerance,
                energyTol=self.settings.energy_tolerance,
            )
            energy = float(field.CalcEnergy())
            if status != 0 or not math.isfinite(energy):
                raise RuntimeError(
                    f"Nonconverged optimization: {method.name}, {cid}, {status}"
                )
            energies[cid] = energy
        return energies

    def reference_ensemble(
        self, pool: Chem.Mol, embedding_seconds: float, method: ReferenceMethod
    ) -> Ensemble:
        """Keep lowest-energy representatives of 1.25-A basins inside a common window."""
        start = time.perf_counter()
        mol = Chem.Mol(pool)
        energies = self.minimize(mol, method)
        minimum = min(energies.values())
        ids = [
            cid
            for cid, energy in energies.items()
            if energy - minimum <= self.settings.energy_window
        ]
        heavy = Chem.RemoveHs(mol)
        distances = [
            rdMolAlign.GetBestRMS(heavy, heavy, ids[i], ids[j])
            for i in range(len(ids))
            for j in range(i)
        ]
        clusters = Butina.ClusterData(
            distances, len(ids), self.settings.rmsd_cutoff, isDistData=True
        )
        representatives = sorted(
            [min((ids[i] for i in cluster), key=energies.get) for cluster in clusters],
            key=energies.get,
        )
        records = [self.checked_record(mol, cid) for cid in representatives]
        for record, cid in zip(records, representatives):
            record.SetDoubleProp("relative_energy_kcal", energies[cid] - minimum)
        return Ensemble(
            records,
            embedding_seconds + time.perf_counter() - start,
            pool.GetNumConformers(),
        )

    def candidates(self, count: int, seed: int) -> Ensemble:
        """Generate each requested budget independently through the training SDF path."""
        start = time.perf_counter()
        generator = SeededGenerator(count, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = generator.genConformers([DEFAULT_MOLECULE], directory)
            records = list(Chem.SDMolSupplier(path, removeHs=False))
        if not records or any(mol is None for mol in records):
            raise RuntimeError("Candidate generation failed")
        checked = [self.checked_record(mol) for mol in records]
        return Ensemble(checked, time.perf_counter() - start, len(checked))

    @staticmethod
    def pair_score(query: Chem.Mol, reference: Chem.Mol, metric: str) -> float:
        """Call the production shape/color optimizer and retain its raw [0,2] scale."""
        score = float(
            _score_single_reference(
                query, reference, "TanimotoCombo", True, score_variant=metric
            )
        )
        if not math.isfinite(score) or not 0 <= score <= 2:
            raise ValueError(f"Invalid native combo score: {score}")
        return score

    def measure(
        self,
        method: ReferenceMethod,
        references: Ensemble,
        candidates: Ensemble,
        seed: int,
        count: int,
        metric: str,
    ) -> list[Measurement]:
        """Record actual prefix time and maximum overlay score for every reference cap."""
        start = time.perf_counter()
        best = 0.0
        prefixes: dict[int, tuple[float, float]] = {}
        for number, reference in enumerate(references.molecules, 1):
            best = max(
                best,
                max(
                    self.pair_score(q, reference, metric) for q in candidates.molecules
                ),
            )
            prefixes[number] = best, time.perf_counter() - start
        rows: list[Measurement] = []
        for cap in self.settings.reference_caps:
            actual = min(cap, len(prefixes)) if cap else len(prefixes)
            score, seconds = prefixes[actual]
            rows.append(
                Measurement(
                    method=method.name,
                    metric=metric,
                    seed=seed,
                    reference_cap=str(cap) if cap else "all",
                    reference_actual=actual,
                    reference_available=len(prefixes),
                    reference_embedded=references.embedded,
                    candidate_requested=count,
                    candidate_actual=len(candidates.molecules),
                    raw_combo=score,
                    rocs_fraction=score / 2,
                    reference_seconds=references.seconds,
                    generation_seconds=candidates.seconds,
                    alignment_seconds=seconds,
                    time_seconds=candidates.seconds + seconds,
                )
            )
        return rows

    def control(
        self,
        method: str,
        metric: str,
        seed: int,
        name: str,
        queries: list[Chem.Mol],
        references: list[Chem.Mol],
        extra_preparation_seconds: float = 0.0,
    ) -> Control:
        """Keep diagnostic controls separate from the reference/candidate grid."""
        score = max(self.pair_score(q, r, metric) for q in queries for r in references)
        if name == "exact_geometry" and abs(score - 2) > 1e-6:
            raise ValueError("Exact-geometry positive control failed")
        return Control(
            method,
            metric,
            seed,
            name,
            score,
            score / 2,
            self.identity(references[0]),
            self.identity(queries[0]),
            extra_preparation_seconds,
        )

    @staticmethod
    def write_csv(path: Path, rows: list[Measurement] | list[Control]) -> None:
        """Export explicit record fields without hidden aggregation or score scaling."""
        dictionaries = [asdict(row) for row in rows]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
            writer.writeheader()
            writer.writerows(dictionaries)

    def run(self) -> None:
        """Build, measure, prove identity, and produce one compact report/figure bundle."""
        root = Path(__file__).resolve().parents[3]
        sources = [
            Path(__file__),
            root / "_david/code/visualization.py",
            root / "_david/code/environment.py",
            root / "_david/code/config.py",
            root / "drugex/molecules/converters/standardizers.py",
            root / "drugex/training/scorers/conformer_generators.py",
            root / "drugex/training/scorers/rocs_rdkit.py",
        ]
        manifest = {
            "state": "running",
            "settings": asdict(self.settings),
            "methods": [asdict(method) for method in ReferenceMethod.suite()],
            "identity": self.expected,
            "rdkit": rdBase.rdkitVersion,
            "python": platform.python_version(),
            "host": platform.node(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "code_root": str(root),
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "dimorphite_available": importlib.util.find_spec("dimorphite_dl")
            is not None,
            "embedding_timeout": 0,
            "scoring_timeout": None,
            "cpu_threads": 1,
            "candidate_seeds": [0xC0FFEE + i for i in range(len(self.settings.seeds))],
        }
        manifest_path = self.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        measurements: list[Measurement] = []
        controls: list[Control] = []
        with Chem.SDWriter(str(self.output / "conformers.sdf")) as writer:
            for replicate, seed in enumerate(self.settings.seeds):
                candidates = {
                    count: self.candidates(count, 0xC0FFEE + replicate)
                    for count in self.settings.candidate_requests
                }
                for count, ensemble in candidates.items():
                    for mol in ensemble.molecules:
                        mol.SetProp("role", "candidate")
                        mol.SetIntProp("seed", seed)
                        mol.SetIntProp("requested", count)
                        writer.write(mol)
                pool, embedding_seconds = self.embed_reference_pool(seed)
                for method in ReferenceMethod.suite():
                    references = self.reference_ensemble(
                        pool, embedding_seconds, method
                    )
                    print(
                        f"seed={seed}, {method.name}: {pool.GetNumConformers()} embedded, {len(references.molecules)} basins, {references.seconds:.2f}s",
                        flush=True,
                    )
                    for mol in references.molecules:
                        mol.SetProp("role", "reference")
                        mol.SetProp("method", method.name)
                        mol.SetIntProp("seed", seed)
                        writer.write(mol)
                    largest = candidates[max(candidates)]
                    relaxed: list[Chem.Mol] = []
                    relaxation_start = time.perf_counter()
                    for original in largest.molecules:
                        mol = Chem.Mol(original)
                        self.minimize(mol, method)
                        record = self.checked_record(mol)
                        record.SetProp("role", "matched_minimized")
                        record.SetProp("method", method.name)
                        record.SetIntProp("seed", seed)
                        record.SetIntProp("requested", max(candidates))
                        writer.write(record)
                        relaxed.append(record)
                    relaxation_seconds = time.perf_counter() - relaxation_start
                    for metric in self.settings.metrics:
                        controls.append(
                            self.control(
                                method.name,
                                metric,
                                seed,
                                "exact_geometry",
                                references.molecules[:1],
                                references.molecules[:1],
                            )
                        )
                        controls.append(
                            self.control(
                                method.name,
                                metric,
                                seed,
                                "matched_minimized",
                                relaxed,
                                references.molecules,
                                relaxation_seconds,
                            )
                        )
                        for count, ensemble in candidates.items():
                            measurements.extend(
                                self.measure(
                                    method, references, ensemble, seed, count, metric
                                )
                            )
                if replicate == 0:
                    with tempfile.TemporaryDirectory() as directory:
                        for name, smiles in (
                            (
                                "legacy_stereo_removed",
                                DefaultStandardizer()(DEFAULT_MOLECULE),
                            ),
                            ("corrected_production_reference", DEFAULT_MOLECULE),
                        ):
                            path = PrepareMolecule().generate_reference_conformers(
                                smiles,
                                Path(directory) / "reference.sdf",
                                num_conformers=100,
                            )
                            legacy = list(Chem.SDMolSupplier(str(path), removeHs=False))
                            for record in legacy:
                                record.SetProp("role", name)
                                writer.write(record)
                            for metric in self.settings.metrics:
                                controls.append(
                                    self.control(
                                        "production_audit",
                                        metric,
                                        seed,
                                        name,
                                        largest.molecules,
                                        legacy,
                                    )
                                )
        self.write_csv(self.output / "data.csv", measurements)
        self.write_csv(self.output / "controls.csv", controls)
        manifest.update(
            state="measured",
            measurement_rows=len(measurements),
            all_optimized_conformers_converged=True,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        from _david.code.visualization import EGCGSelfRecoveryVisualizer

        EGCGSelfRecoveryVisualizer.render(self.output)
        manifest["state"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


class SelfRecoveryCommand:
    """One command for measurement or regenerating the compact presentation."""

    @staticmethod
    def run() -> None:
        """Use --smoke for integration checks; --render-only never repeats chemistry."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--render-only", action="store_true")
        args = parser.parse_args()
        if args.render_only:
            from _david.code.visualization import EGCGSelfRecoveryVisualizer

            EGCGSelfRecoveryVisualizer.render(args.output)
        else:
            settings = (
                StudySettings(
                    reference_attempts=32, candidate_requests=(1, 5), seeds=(42,)
                )
                if args.smoke
                else StudySettings()
            )
            EGCGSelfRecovery(args.output, settings).run()


if __name__ == "__main__":
    SelfRecoveryCommand.run()
