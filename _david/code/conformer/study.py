"""How well a molecule's conformer ensemble recovers its own shape, and at what cost.

    python -m _david.code.conformer --set smiles='CCO' --set name=ethanol
    python -m _david.code.conformer --set repeats=3 --set methods='ETKDG,MMFF94s_e78.4'

A cell is (method, K, C, seed): a bank of at most K conformers against C candidate
attempts, same method both sides, scored as the best shape+colour Tanimoto over all
pairs. Out comes one directory: ``measurements.tsv``, ``references.tsv``, ``config.json``
and the figures. Nothing is cached or read from a previous run.

Candidate generation and scoring are timed per cell, since that is what the RL loop pays
per molecule; bank construction is timed once per (method, seed) and reported apart.
Compare the CPU seconds -- cells run in parallel, so wall time is contaminated.

The embedding repeats native's ETKDG parameters instead of calling the pipeline's
generator, because native pins ``randomSeed = 0xC0FFEE`` and this study is about the
spread across ten seeds. ``tests.py`` pins the force field against the pipeline's
``single_point_energy`` so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from _david.code.config import parse_dataclass_overrides
from _david.code.conformer.config import METHODS, RANKING_FALLBACK, Method, StudyConfig
from _david.code.reconstruct.provenance import atomic_write_text

# The scoring seam the pipeline itself uses: best (shape, colour) over every
# conformer-reference pair, with native's per-molecule guard. Importing it rather than
# reimplementing it is the point -- the study must measure the pipeline's scorer.
from _david.code.scorers.rocs import PoseSearch, _annotate_color_features
from _david.code.scorers.utils import resolve_workers
from drugex.training.scorers.parallel import pool_context

logger = logging.getLogger(__name__)

#: Added to a reference seed to obtain its candidate seed. A large prime rather than ``+1``:
#: RDKit derives conformer *i* of seed *s* multiplicatively (matching conformer 0 of seed
#: ``s * (i + 1)``), so disjoint prime streams prevent candidate/reference seed collisions.
CANDIDATE_OFFSET = 15485863

#: Ten reference seeds, all prime above 10^5 and coprime with ``CANDIDATE_OFFSET``.
#: Against grids of 200 and 40 conformers every stream is provably disjoint.
SEEDS: tuple[int, ...] = (
    104729,
    224737,
    350377,
    479909,
    611953,
    746773,
    882377,
    1020379,
    1159523,
    1299721,
)


def _isomers(smiles: str, cap: int) -> list[Chem.Mol]:
    """Stereoisomers of one SMILES, capped. A generated SMILES often leaves centres open."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    options = StereoEnumerationOptions(maxIsomers=cap, onlyUnassigned=True, unique=True)
    out = [Chem.AddHs(iso) for iso in EnumerateStereoisomers(mol, options=options)]
    return out or [Chem.AddHs(mol)]


def _embed(smiles: str, attempts: int, seed: int, cfg: StudyConfig) -> list[Chem.Mol]:
    """``attempts`` ETKDGv3 conformers per stereoisomer, as one flat list.

    Native's parameters, with the seed exposed and threading off: the study parallelises
    over cells, so a conformer job that also threads would make CPU seconds unreadable.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = cfg.prune_rms
    out: list[Chem.Mol] = []
    for isomer in _isomers(smiles, cfg.max_isomers):
        ids = AllChem.EmbedMultipleConfs(isomer, numConfs=attempts, params=params)
        out.extend(Chem.Mol(isomer, confId=int(conf_id)) for conf_id in ids)
    return out


def force_field(mol: Chem.Mol, method: Method, cfg: StudyConfig) -> Any:
    """Return the RDKit force field requested by ``method``, or ``None`` if unparameterised."""
    _ = cfg
    if method.force_field is None:
        return None
    if method.force_field == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            return None
        return AllChem.UFFGetMoleculeForceField(mol)
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=method.force_field)
    if props is None:
        return None
    props.SetMMFFDielectricConstant(method.dielectric)
    props.SetMMFFDielectricModel(method.dielectric_model)
    return AllChem.MMFFGetMoleculeForceField(mol, props)


def _energy(mol: Chem.Mol, method: Method, cfg: StudyConfig) -> float:
    """Single-point energy, or ``inf`` when unscorable.

    Never returns 0.0: the bank sorts ascending, so a fabricated zero would seed it with an
    unparameterised or failed conformer.
    """
    field = force_field(mol, method, cfg)
    if field is None:
        return math.inf
    try:
        value = float(field.CalcEnergy())
    except RuntimeError:
        return math.inf
    return value if math.isfinite(value) else math.inf


def _relax(mols: list[Chem.Mol], method: Method, cfg: StudyConfig) -> list[Chem.Mol]:
    """Minimise copies of ``mols`` where ``method`` asks for it; ETKDG passes straight through.

    A conformer the force field cannot handle keeps its embedded geometry rather than
    being dropped: removing it would silently shrink the ensemble of exactly the methods
    with the worst parameter coverage, which is the comparison this study is making.
    """
    if method.force_field is None:
        return mols
    out: list[Chem.Mol] = []
    for mol in mols:
        # Minimised on a copy: Minimize() writes coordinates back as it iterates, so catching
        # an exception after mutating in place would keep a half-moved geometry.
        working = Chem.Mol(mol)
        field = force_field(working, method, cfg)
        if field is None:
            out.append(mol)
            continue
        try:
            field.Minimize(maxIts=cfg.max_iterations)
        except RuntimeError:
            out.append(mol)
            continue
        out.append(working)
    return out


def select_bank_ordered(
    mols: list[Chem.Mol],
    cap: int,
    method: Method,
    cfg: StudyConfig,
    *,
    caps_to_time: tuple[int, ...] = (),
    cap_times: dict[int, tuple[float, float]] | None = None,
) -> list[Chem.Mol]:
    """Select at most ``cap`` conformers in ascending energy order with RMSD > ``reference_rmsd``.

    Starts with the global minimum-energy conformer and walks through the remaining conformers
    in ascending energy order, keeping a conformer only if its heavy-atom RMSD to every
    already-kept reference conformer exceeds ``cfg.reference_rmsd``. Taking the first ``k``
    elements gives the exact same conformer set as calling :func:`select_bank` with ``cap=k``.
    """
    if not mols:
        if cap_times is not None:
            for k in caps_to_time:
                cap_times[k] = (0.0, 0.0)
        return []
    t0, c0 = time.perf_counter(), time.process_time()
    wanted = set(caps_to_time)
    # Match the pipeline: unminimised banks are still ranked by MMFF94s single-point energy.
    ranking = method if method.force_field else RANKING_FALLBACK
    ranked = sorted(((_energy(mol, ranking, cfg), mol) for mol in mols), key=lambda p: p[0])
    probes = [Chem.RemoveHs(Chem.Mol(mol)) for _, mol in ranked]
    kept = [0]
    if cap_times is not None and 1 in wanted:
        cap_times[1] = (time.perf_counter() - t0, time.process_time() - c0)
    for i in range(1, len(ranked)):
        if len(kept) >= cap:
            break
        if all(
            rdMolAlign.GetBestRMS(Chem.Mol(probes[i]), Chem.Mol(probes[k])) > cfg.reference_rmsd
            for k in kept
        ):
            kept.append(i)
            if cap_times is not None and len(kept) in wanted:
                cap_times[len(kept)] = (time.perf_counter() - t0, time.process_time() - c0)
    if cap_times is not None:
        end_t = (time.perf_counter() - t0, time.process_time() - c0)
        for k in caps_to_time:
            cap_times.setdefault(k, end_t)
    if len(kept) < cap:
        logger.debug(
            "bank asked for %d and kept %d: only that many poses are further than %.2f A "
            "apart in a pool of %d",
            cap,
            len(kept),
            cfg.reference_rmsd,
            len(ranked),
        )
    out: list[Chem.Mol] = []
    for i in kept:
        energy, raw_mol = ranked[i]
        mol = Chem.Mol(raw_mol)
        mol.SetDoubleProp("Energy", energy)
        out.append(mol)
    return out


def select_bank(mols: list[Chem.Mol], cap: int, method: Method, cfg: StudyConfig) -> list[Chem.Mol]:
    """Select at most ``cap`` conformers using the same energy-ranked RMSD rule as the pipeline."""
    selected = select_bank_ordered(mols, cap, method, cfg)
    return sorted(selected, key=lambda mol: mol.GetDoubleProp("Energy"))


def score(
    candidates: list[Chem.Mol], references: list[Chem.Mol], cfg: StudyConfig
) -> tuple[float, float, bool]:
    """Return ``(shape, colour, failed)`` for the best pair across ``candidates x references``."""
    if not candidates or not references:
        return 0.0, 0.0, True
    search = PoseSearch(
        references, [list(range(len(references)))], cfg.alignment_shape_weight, True, cfg.color_ff
    )
    shape, colour = search.best(candidates)[0]
    return float(shape), float(colour), shape + colour <= 0.0


def _one_stream(args: tuple[StudyConfig, str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run every ``(K, C)`` cell for one ``(method, seed)`` stream.

    Candidate generation does not depend on ``K``, so it runs once per ``C`` and its cost is
    reported against each ``K`` that reuses it.
    """
    cfg, method_key, seed = args
    method = cfg.method(method_key)

    started, cpu0 = time.perf_counter(), time.process_time()
    embedded = _embed(cfg.smiles, cfg.reference_attempts, seed, cfg)
    pool = _relax(embedded, method, cfg)
    reference_wall, reference_cpu = time.perf_counter() - started, time.process_time() - cpu0

    # Refuse to silently relabel an unparameterisable force field as the ETKDG control.
    parameterised = (
        sum(force_field(mol, method, cfg) is not None for mol in embedded)
        if method.force_field
        else 0
    )
    if method.force_field and not parameterised:
        raise SystemExit(
            f"{method.force_field} cannot parameterise {cfg.smiles!r}: every conformer "
            "would be returned unminimised and the results would be the ETKDG control "
            f"labelled {method_key!r}. Drop that method or change the molecule."
        )

    banks: dict[int, list[Chem.Mol]] = {}
    bank_wall: dict[int, float] = {}
    for cap in cfg.reference_caps:
        t0 = time.perf_counter()
        banks[cap] = select_bank(pool, cap, method, cfg)
        bank_wall[cap] = time.perf_counter() - t0

    # Annotate colour features once outside the per-cell stopwatch so the first K does not
    # pay typing setup cost for subsequent caps.
    if cfg.color_ff != "default":
        for mol in pool:
            _annotate_color_features(mol, color_ff=cfg.color_ff)

    rows: list[dict[str, Any]] = []
    for count in cfg.candidate_counts:
        t0, c0 = time.perf_counter(), time.process_time()
        candidates = _relax(_embed(cfg.smiles, count, seed + CANDIDATE_OFFSET, cfg), method, cfg)
        candidate_wall, candidate_cpu = time.perf_counter() - t0, time.process_time() - c0
        if cfg.color_ff != "default":
            for mol in candidates:
                _annotate_color_features(mol, color_ff=cfg.color_ff)
        for cap in cfg.reference_caps:
            t1, c1 = time.perf_counter(), time.process_time()
            shape, colour, failed = score(candidates, banks[cap], cfg)
            score_wall, score_cpu = time.perf_counter() - t1, time.process_time() - c1
            rows.append(
                {
                    "method": method_key,
                    "reference_cap": cap,
                    "candidate_count": count,
                    "seed": seed,
                    "bank_size": len(banks[cap]),
                    "candidates": len(candidates),
                    "shape": round(shape, 9),
                    "colour": round(colour, 9),
                    # Nine decimal places support the 1e-6 ceiling tolerance in figures.py.
                    "combo": round(shape + colour, 9),
                    "self_recovery": round((shape + colour) / 2.0, 9),
                    "score_failed": int(failed),
                    "candidate_seconds": round(candidate_wall, 4),
                    "candidate_cpu_seconds": round(candidate_cpu, 4),
                    "score_seconds": round(score_wall, 4),
                    "score_cpu_seconds": round(score_cpu, 4),
                    "bank_seconds": round(bank_wall[cap], 4),
                    "cell_seconds": round(candidate_wall + score_wall, 4),
                    "cell_cpu_seconds": round(candidate_cpu + score_cpu, 4),
                }
            )
    reference: dict[str, Any] = {
        "method": method_key,
        "seed": seed,
        "pool_size": len(pool),
        "parameterised": parameterised,
        "embedded": len(embedded),
        "reference_seconds": round(reference_wall, 4),
        "reference_cpu_seconds": round(reference_cpu, 4),
        "bank_sizes": json.dumps({str(k): len(v) for k, v in banks.items()}),
        "lowest_energy": round(min((_energy(m, method, cfg) for m in pool), default=math.inf), 4)
        if method.force_field
        else "",
    }
    return rows, reference


def run(cfg: StudyConfig, out_dir: Path) -> pd.DataFrame:
    """Run the full grid, write ``measurements.tsv`` and ``references.tsv``, and return measurements."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.repeats > len(SEEDS):
        raise ValueError(
            f"only {len(SEEDS)} independent seeds are defined, {cfg.repeats} asked for"
        )
    seeds = SEEDS[: cfg.repeats]
    tasks = [(cfg, method, seed) for method in cfg.methods for seed in seeds]
    worker_count = resolve_workers(cfg.n_jobs)
    logger.info(
        "%d cells: %d methods x %d caps x %d candidate counts x %d seeds, on %d workers",
        cfg.cells,
        len(cfg.methods),
        len(cfg.reference_caps),
        len(cfg.candidate_counts),
        cfg.repeats,
        worker_count,
    )

    rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    if worker_count <= 1 or len(tasks) < 2:
        results = [_one_stream(task) for task in tasks]
    else:
        with pool_context().Pool(min(worker_count, len(tasks))) as pool:
            results = pool.map(_one_stream, tasks)
    for cell_rows, reference in results:
        rows.extend(cell_rows)
        references.append(reference)

    measurements = pd.DataFrame(rows).sort_values(
        ["method", "reference_cap", "candidate_count", "seed"]
    )
    measurements.to_csv(out_dir / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame(references).sort_values(["method", "seed"]).to_csv(
        out_dir / "references.tsv", sep="\t", index=False
    )
    logger.info("wrote %d measurements to %s", len(measurements), out_dir / "measurements.tsv")
    return measurements


def config_from_args(overrides: Sequence[str]) -> StudyConfig:
    """Build a :class:`StudyConfig` from repeatable CLI ``name=value`` overrides."""
    return parse_dataclass_overrides(StudyConfig, overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one StudyConfig field; repeatable",
    )
    parser.add_argument("--tag", help="suffix for the study directory name")
    parser.add_argument("--no-figures", action="store_true", help="write the tables only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = config_from_args(args.overrides)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(cfg.output_root) / f"{stamp}_{args.tag or cfg.name}_self_similarity"

    resolved: dict[str, Any] = {
        k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()
    }
    resolved["methods_resolved"] = {k: dataclasses.asdict(METHODS[k]) for k in cfg.methods}
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_dir / "config.json", json.dumps(resolved, indent=2) + "\n")

    started = time.perf_counter()
    measurements = run(cfg, out_dir)
    logger.info("grid finished in %.1f s", time.perf_counter() - started)

    if not args.no_figures:
        from _david.code.conformer import figures

        figures.render_all(measurements, cfg, out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
