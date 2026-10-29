"""Suggest a fixed per-query budget using measured cost order and self-checks.

This prototype does not establish reward equivalence for different generated
ligands. It proposes a setting for code/research review; it changes no defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from _david.code.chemistry import standardize_smiles
from _david.code.config import DEFAULT_MOLECULE
from _david.code.conformer.reference_bank import METHODS, REFERENCE_SEED, prepare_banks
from _david.code.conformer.self_recovery import atomic_json, digest
from _david.code.environment import create_rocs_scorer

CHECK_SEEDS = (42, 1_000_003)


def budget_order(summary, method, available):
    """Use empirical cost, clamp to actual references and skip equivalent caps."""
    selected = summary[summary.method.eq(method)].sort_values("mean_seconds")
    seen = set()
    for row in selected.to_dict("records"):
        requested = int(row["reference_cap"])
        actual = min(requested, available) if requested else available
        count = int(row["candidate_attempts"])
        if (actual, count) not in seen:
            seen.add((actual, count))
            yield dict(
                reference_cap=actual,
                candidate_attempts=count,
                panel_mean_seconds=row["mean_seconds"],
            )


def calibrate(study: Path, smiles: str, target: float = 0.95):
    if not 0 < target <= 1:
        raise ValueError("Target must be in (0, 1]")
    # Both candidate checks avoid reference seeds and each other's seed stream.
    streams = [{seed * i for i in range(1, 76)} for seed in CHECK_SEEDS]
    reference_stream = {REFERENCE_SEED * i for i in range(1, 51)}
    if any(reference_stream & stream for stream in streams) or streams[0] & streams[1]:
        raise ValueError("Query-calibration seed streams overlap")
    recommended = json.loads((study / "raw_data/recommended_settings.json").read_text())
    method = recommended["method"]
    cleaned = standardize_smiles(smiles)
    query_id = hashlib.sha256((method + ":" + cleaned).encode()).hexdigest()[:12]
    raw = study / "raw_data/query_calibration" / query_id
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "suggestion.json").exists():
        previous = json.loads((raw / "suggestion.json").read_text())
        if (
            previous["source_sha256"] == digest(__file__)
            and previous["target"] == target
        ):
            print(f"Existing query suggestion: {raw / 'suggestion.json'}")
            return previous
        raise RuntimeError(
            "Query calibration with different settings already exists; preserve it and use a new study output"
        )
    pools, preparation = prepare_banks(cleaned, raw, (method,))
    bank = pools[method]
    result = dict(
        status="no_passing_budget",
        query=cleaned,
        method=method,
        target=target,
        actual_reference_pool=len(bank),
        preparation=preparation,
        check_seeds=CHECK_SEEDS,
        reference_seed=REFERENCE_SEED,
        candidate_isomer_cap=4,
        source_sha256=digest(__file__),
        policy="First passing configuration in development-panel cost order, after clamping/deduplicating caps for this reference. Require target on each of two disjoint candidate-seed streams. A deterministic reference self-check, not a confidence interval.",
        limitation="Self-recovery cannot establish that a small bank preserves rewards/rankings for other ligands. Self-call latency is not an RL-candidate time guarantee. No runtime/default is changed; isomer cap4 remains provisional.",
    )
    if not bank or preparation["color_excluded"]:
        result["status"] = "unsupported_reference"
        atomic_json(raw / "suggestion.json", result)
        return result
    summary = pd.read_csv(study / "raw_data/development/summary.csv")
    checks = []
    for budget in budget_order(summary, method, len(bank)):
        refs = bank[: budget["reference_cap"]]
        scores = []
        for seed in CHECK_SEEDS:
            scorer = create_rocs_scorer(
                refs,
                max_conformers=budget["candidate_attempts"],
                max_isomers=4,
                random_seed=seed,
                minimize_candidates=method != "etkdg",
                optimization=METHODS[method],
                timeout=10,
                conformer_threads=1,
                n_jobs=1,
            )
            scorer.conformer_generator.embedding_rmsd = None
            scorer.conformer_generator.protonate = False
            score = float(scorer.getScores([Chem.MolFromSmiles(cleaned)])[0, 0]) / 2
            if not 0 <= score <= 1.0001:
                raise ValueError("Unexpected native self score")
            checks.append(dict(**budget, seed=seed, normalized_rocs=score))
            scores.append(score)
        pd.DataFrame(checks).to_csv(raw / "self_checks.csv", index=False)
        if min(scores) >= target:
            result.update(
                status="reference_self_check_passed",
                suggested=budget,
                minimum_self_score=min(scores),
            )
            break
    atomic_json(raw / "suggestion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--smiles", default=DEFAULT_MOLECULE)
    parser.add_argument("--target", type=float, default=0.95)
    args = parser.parse_args()
    calibrate(args.study, args.smiles, args.target)
