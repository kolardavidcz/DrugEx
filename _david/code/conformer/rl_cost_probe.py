"""Five real EGCG RL epochs for cost attribution, followed by fixed-batch replays.

This is an integration/timing illustration. It does not select an RL learning
winner, fit training hyperparameters or overwrite the pretrained checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from drugex.training.monitors import FileMonitor

from _david.code.chemistry import standardize_smiles
from _david.code.config import DEFAULT_MOLECULE, PROJECT_ROOT, PipelineConfig
from _david.code.conformer.reference_bank import METHODS, prepare_banks, save_records
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.conformer.whole_call_scorer import WholeCallScorer
from _david.code.environment import create_environment
from _david.code.models import SequenceRNNOptimizer
from _david.code.schema import std


class TimingMonitor(FileMonitor):
    """Use the existing monitor, with logs/models/tables in their proper folders."""

    def __init__(self, root, raw, totals):
        super().__init__(root / "logs" / raw.name, save_smiles=True)
        self.raw = raw
        self.models = raw / "models"
        self.models.mkdir(exist_ok=True)
        self.outDF = raw / "performance.tsv"
        self.outSmiles = raw / "validation_smiles.tsv"
        self.totals = totals
        self.previous = dict(totals)
        self.previous_time = time.perf_counter()
        self.epochs = []

    def saveModel(self, model, identifier=None):
        self.currentState = model.getModel()
        suffix = f"_{identifier}" if identifier else ""
        torch.save(self.currentState, self.models / f"egcg{suffix}.pkg")

    def savePerformanceInfo(self, performance_dict, df_smiles=None):
        super().savePerformanceInfo(performance_dict, df_smiles)
        now = time.perf_counter()
        differences = {
            key: self.totals[key] - self.previous[key] for key in self.totals
        }
        self.epochs.append(
            dict(
                epoch=len(self.epochs) + 1,
                epoch_seconds=now - self.previous_time,
                **differences,
            )
        )
        pd.DataFrame(self.epochs).to_csv(self.raw / "epoch_timings.csv", index=False)
        self.previous, self.previous_time = dict(self.totals), now


def configured_environment(reference_path, setting, workers, epochs=5, samples=200):
    config = PipelineConfig(
        epochs=epochs,
        min_epochs=epochs,
        patience=6,
        n_samples=samples,
        training_seed=42,
        random_seed=42,
        cpu_ratio=0.8,
        max_conformers=int(setting["candidate_attempts"]),
        max_isomers=4,
        minimize_candidates=setting["method"] != "etkdg",
        optimization=METHODS[setting["method"]],
        timeout_seconds=10,
        conformer_threads=1,
        gpu=0,
    )
    environment = create_environment(reference_path, config=config)
    scorer = environment.scorers[0]
    scorer.n_jobs = workers
    scorer.conformer_generator.embedding_rmsd = None
    scorer.conformer_generator.protonate = False
    return environment, config


def run(output: Path, workers: int, smoke=False):
    raw = output / "raw_data" / ("rl_smoke" if smoke else "rl_egcg")
    raw.mkdir(parents=True, exist_ok=True)
    batches = raw / "batches"
    batches.mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    path = raw / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["state"] == "complete":
            print("RL timing illustration already complete")
            return
        raise RuntimeError(
            "A partial RL illustration exists; inspect it before restarting"
        )
    torch.set_num_threads(1)
    os.environ["DRUGEX_MP_CONTEXT"] = "forkserver"
    Chem.SetDefaultPickleProperties(
        Chem.GetDefaultPickleProperties() | Chem.PropertyPickleOptions.CoordsAsDouble
    )
    baseline = dict(
        method="mmff_high_dielectric", reference_cap=4, candidate_attempts=20
    )
    selected_path = output / "raw_data/recommended_settings.json"
    selected = (
        json.loads(selected_path.read_text()) if selected_path.exists() else baseline
    )
    keys = ("method", "reference_cap", "candidate_attempts")
    choices = (
        (baseline,)
        if smoke
        else (
            baseline,
            selected,
            dict(method="etkdg", reference_cap=4, candidate_attempts=20),
        )
    )
    presets = {
        tuple(item[key] for key in keys): {key: item[key] for key in keys}
        for item in choices
    }
    references = raw / "references"
    references.mkdir(exist_ok=True)
    pools, preparation = prepare_banks(
        standardize_smiles(DEFAULT_MOLECULE),
        references,
        tuple(dict.fromkeys(item["method"] for item in presets.values())),
    )
    ref_paths = {}
    for key, setting in presets.items():
        cap = setting["reference_cap"]
        bank = pools[setting["method"]]
        chosen = bank[:cap] if cap else bank
        if not chosen:
            raise RuntimeError("EGCG reference preparation failed")
        target = references / f"{setting['method']}_k{cap}.sdf"
        save_records(target, chosen)
        ref_paths[key] = target
    environment, config = configured_environment(
        ref_paths[tuple(baseline[key] for key in keys)],
        baseline,
        workers,
        epochs=1 if smoke else 5,
        samples=16 if smoke else 200,
    )
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    optimizer = SequenceRNNOptimizer(raw / "models", gpu=0, cpu_ratio=0.8)
    package, vocabulary = config.resolve_pretrained_paths()
    initial_hash = digest(package)
    source_files = [
        PROJECT_ROOT / f"_david/code/{name}.py"
        for name in (
            "chemistry",
            "config",
            "environment",
            "models",
            "reference",
            "rocs",
            "utils",
            "schema",
        )
    ] + [
        Path(__file__),
        PROJECT_ROOT / "_david/code/conformer/reference_bank.py",
        PROJECT_ROOT / "_david/code/conformer/whole_call_scorer.py",
        PROJECT_ROOT / "drugex/training/explorers/sequence_explorer.py",
        PROJECT_ROOT / "drugex/training/scorers/rocs_rdkit.py",
        PROJECT_ROOT / "drugex/training/scorers/conformer_generators.py",
    ]
    source_hashes = {str(file): digest(file) for file in source_files}
    model_started = time.perf_counter()
    agent, prior, _ = optimizer.load_models()
    load_seconds = time.perf_counter() - model_started
    totals = dict(
        rocs_seconds=0.0,
        generation_minimization_seconds=0.0,
        environment_seconds=0.0,
        environment_rows=0,
        scorer_calls=0,
    )
    scorer = environment.scorers[0]
    original_rocs = scorer.getScores
    original_generate = scorer.conformer_generator.genConformers
    original_environment = environment.getScores

    def timed_generation(smiles, directory):
        tick = time.perf_counter()
        result = original_generate(smiles, directory)
        totals["generation_minimization_seconds"] += time.perf_counter() - tick
        return result

    def timed_rocs(*args, **kwargs):
        tick = time.perf_counter()
        result = original_rocs(*args, **kwargs)
        totals["rocs_seconds"] += time.perf_counter() - tick
        totals["scorer_calls"] += 1
        return result

    def timed_environment(smiles, *args, **kwargs):
        tick = time.perf_counter()
        result = original_environment(smiles, *args, **kwargs)
        totals["environment_seconds"] += time.perf_counter() - tick
        totals["environment_rows"] += len(smiles)
        result.assign(SMILES=list(smiles)).to_csv(
            batches / f"batch_{totals['scorer_calls']:03d}.csv", index=False
        )
        return result

    scorer.conformer_generator.genConformers = timed_generation
    scorer.getScores = timed_rocs
    environment.getScores = timed_environment
    manifest = dict(
        state="training",
        purpose="Engineering smoke test"
        if smoke
        else "Five-epoch cost illustration; not an RL learning comparison",
        host=socket.gethostname(),
        config=asdict(config),
        actual_alignment_workers=workers,
        baseline=baseline,
        replay_presets=list(presets.values()),
        reference_preparation=preparation,
        model_load_seconds=load_seconds,
        source_sha256=source_hashes,
        actual_device=str(optimizer.device),
        torch_version=torch.__version__,
        base_checkpoint=str(package),
        base_checkpoint_sha256=initial_hash,
        vocabulary_sha256=digest(vocabulary),
        coordinate_pickle_flags=int(Chem.GetDefaultPickleProperties()),
        note="Native RL fit includes its 10%-size validation sample each epoch. Reference/model setup excluded from epoch timing; capture I/O included in residual epoch work. GPU kernels are not forced deterministic.",
        replay_selection="First 64 rows of the first recorded training batch, selected before inspecting scores; all epoch batches remain saved. Replays test scoring cost/score changes, not independent learning.",
        engine_note="Baseline native batching versus a persistent whole-call prototype. Singleton calls use per-graph RDKit defaults; native batches can inherit maxIterations from their first graph. Both timing and score changes are recorded; equivalence is not assumed.",
    )
    atomic_json(path, manifest)
    monitor = TimingMonitor(output, raw, totals)
    explorer = optimizer.build_explorer(agent, environment, prior, config)
    started = time.perf_counter()
    try:
        explorer.fit(
            monitor=monitor,
            epochs=config.epochs,
            patience=6,
            min_epochs=config.epochs,
            reload_interval=6,
            criteria=config.criteria,
        )
        manifest.update(
            state="replaying",
            fit_seconds=time.perf_counter() - started,
            epoch_count=len(monitor.epochs),
            totals=totals,
        )
        atomic_json(path, manifest)
        rows, molecule_rows = [], []
        for key, setting in presets.items():
            engines = (
                ("native_batch", "whole_call")
                if setting == baseline and not smoke
                else ("native_batch",)
            )
            for engine in engines:
                replay_env, _ = configured_environment(ref_paths[key], setting, workers)
                replay_scorer = replay_env.scorers[0]
                if engine == "whole_call":
                    replay_scorer = WholeCallScorer(
                        replay_scorer, ref_paths[key], setting, workers
                    )
                try:
                    # Second baseline pass tests warm reuse without caching scores.
                    for repeat in range(2 if setting == baseline and not smoke else 1):
                        batch = sorted(batches.glob("batch_*.csv"))[0]
                        recorded = pd.read_csv(batch, keep_default_na=False).head(64)
                        mols = [
                            Chem.MolFromSmiles(smi) if smi else None
                            for smi in recorded.SMILES
                        ]
                        tick = time.perf_counter()
                        scores = replay_scorer(mols)[:, 0]
                        elapsed = time.perf_counter() - tick
                        original_scores = recorded[std.ROCS].to_numpy()
                        rows.append(
                            dict(
                                **setting,
                                engine=engine,
                                repeat=repeat,
                                batch=batch.name,
                                inputs=len(recorded),
                                seconds=elapsed,
                                mean_score=float(scores.mean()),
                                mean_change=float((scores - original_scores).mean()),
                                max_absolute_change=float(
                                    np.abs(scores - original_scores).max()
                                ),
                            )
                        )
                        pd.DataFrame(rows).to_csv(
                            raw / "batch_replays.csv", index=False
                        )
                        molecule_rows.extend(
                            dict(
                                **setting,
                                engine=engine,
                                repeat=repeat,
                                batch=batch.name,
                                index=index,
                                smiles=smiles,
                                baseline_rocs=float(old),
                                replay_rocs=float(new),
                                change=float(new - old),
                            )
                            for index, (smiles, old, new) in enumerate(
                                zip(recorded.SMILES, original_scores, scores)
                            )
                        )
                        pd.DataFrame(molecule_rows).to_csv(
                            raw / "replay_scores.csv.gz", index=False
                        )
                finally:
                    if engine == "whole_call":
                        replay_scorer.close()
        if digest(package) != initial_hash:
            raise RuntimeError("Pretrained checkpoint changed during illustration")
        if source_hashes != {str(file): digest(file) for file in source_files}:
            raise RuntimeError("RL illustration source changed during execution")
        manifest.update(
            state="complete",
            base_checkpoint_unchanged=True,
            output_sha256={
                name: digest(raw / name)
                for name in (
                    "epoch_timings.csv",
                    "batch_replays.csv",
                    "replay_scores.csv.gz",
                )
            },
        )
    except Exception as exc:
        manifest.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        monitor.close()
        atomic_json(path, manifest)
        del agent, prior, explorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, args.smoke)
