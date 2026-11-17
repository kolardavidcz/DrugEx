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
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from _david.code.conformer.config import METHODS, RANKING_FALLBACK, Method, StudyConfig

# The scoring seam the pipeline itself uses: best (shape, colour) over every
# conformer-reference pair, with native's per-molecule guard. Importing it rather than
# reimplementing it is the point -- the study must measure the pipeline's scorer.
from _david.code.scorers.rocs import PoseSearch, _annotate_color_features
from drugex.training.scorers.parallel import capped_n_jobs, pool_context

logger = logging.getLogger(__name__)

#: Ten seeds, far apart and pairwise non-multiples. RDKit derives each conformer's seed
#: from ``randomSeed`` by multiplication, so 42 and 84 share states and would make two
#: "independent" repeats correlated -- the collision the previous study hit and documented.
#: Primes above 10^5 keep every stream disjoint.
#: Added to a reference seed to get its candidate seed. A large prime, not ``+1``: RDKit
#: builds each conformer's seed from ``randomSeed`` multiplicatively -- conformer *i* of
#: seed *s* is bit-identical to conformer 0 of seed *s(i+1)* -- so two streams must not
#: share small factors, or the "independent" candidate set partly *is* the reference set.
CANDIDATE_OFFSET = 15485863

#: Ten reference seeds. All prime, and coprime with CANDIDATE_OFFSET, so by the rule above
#: a collision would need a quotient smaller than the grid; the smallest here is above
#: 10^5 against grids of 200 and 40, which makes every stream provably disjoint.
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
        for conf_id in ids:
            single = Chem.Mol(isomer, confId=int(conf_id))
            out.append(single)
    return out


def force_field(mol: Chem.Mol, method: Method, cfg: StudyConfig):
    """The force field a method asks for, or None when it cannot parameterise ``mol``."""
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
    """Single-point energy, or ``inf`` when unscorable. Never 0.0: the bank sorts ascending,
    so a fabricated zero would seed it with the worst conformer."""
    field = force_field(mol, method, cfg)
    if field is None:
        return math.inf
    try:
        value = float(field.CalcEnergy())
    except RuntimeError:
        return math.inf
    return value if math.isfinite(value) else math.inf


def _relax(mols: list[Chem.Mol], method: Method, cfg: StudyConfig) -> list[Chem.Mol]:
    """Minimise in place where the method asks for it; ETKDG-only passes straight through.

    A conformer the force field cannot handle keeps its embedded geometry rather than
    being dropped: removing it would silently shrink the ensemble of exactly the methods
    with the worst parameter coverage, which is the comparison this study is making.
    """
    if method.force_field is None:
        return mols
    out = []
    for mol in mols:
        # Minimised on a copy. Minimize() writes coordinates back as it iterates, so
        # minimising in place and then catching the exception kept a half-moved geometry
        # -- neither ETKDG's pose nor a minimum -- under the name of a minimised method.
        # The pipeline avoids it the same way (_david/code/scorers/conformers.py:201).
        working = Chem.Mol(mol)
        field = force_field(working, method, cfg)
        if field is None:
            out.append(mol)
            continue
        try:
            field.Minimize(maxIts=cfg.max_iterations)
        except RuntimeError:
            # RDKit's BFGS raises on pathological geometries. That is a property of this
            # conformer, not of the method; keep the pre-minimisation pose.
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
    # A method with no force field still gets an energy ranking, because the pipeline
    # gives one: its _force_field is not gated on cfg.minimize, so an unminimised bank is
    # MMFF94s-ranked all the same. Ranking ETKDG by nothing anchored its bank on
    # embedding order, which is not the control this row is meant to be.
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
        # Silence here made K=8 and K=16 the same measurement for every minimised method:
        # both returned the same six or seven poses, so the figures showed a "saturation
        # at K=8" that was the pool running out, not the metric settling.
        logger.debug(
            "bank asked for %d and kept %d: only that many poses are further than %.2f A "
            "apart in a pool of %d",
            cap,
            len(kept),
            cfg.reference_rmsd,
            len(ranked),
        )
    out = []
    for i in kept:
        energy, raw_mol = ranked[i]
        mol = Chem.Mol(raw_mol)
        mol.SetDoubleProp("Energy", energy)
        out.append(mol)
    return out


def select_bank(mols: list[Chem.Mol], cap: int, method: Method, cfg: StudyConfig) -> list[Chem.Mol]:
    """At most ``cap`` conformers, chosen the way the pipeline chooses its reference bank.

    Conformers are sorted by ascending energy; the lowest-energy conformer seeds the bank,
    and subsequent conformers are added in energy order only when their heavy-atom RMSD to
    every already-kept conformer exceeds ``cfg.reference_rmsd``. Identical rule to
    ``_david/code/scorers/conformers.build_reference``.
    """
    selected = select_bank_ordered(mols, cap, method, cfg)
    return sorted(selected, key=lambda mol: mol.GetDoubleProp("Energy"))


def score(
    candidates: list[Chem.Mol], references: list[Chem.Mol], cfg: StudyConfig
) -> tuple[float, float, bool]:
    """Best (shape, colour) over every pair, and whether the measurement is real.
    ``PoseSearch.best`` returns (0.0, 0.0) on a timeout, and two conformers of one molecule
    never overlay at zero."""
    if not candidates or not references:
        return 0.0, 0.0, True
    search = PoseSearch(
        references, [list(range(len(references)))], cfg.alignment_shape_weight, True, cfg.color_ff
    )
    shape, colour = search.best(candidates)[0]
    return float(shape), float(colour), shape + colour <= 0.0


def _one_stream(args: tuple[StudyConfig, str, int]) -> tuple[list[dict], dict]:
    """Every cell of one (method, seed): the bank once, the candidates once per C.

    Candidate generation does not depend on K, so it is done once per C and its cost is
    reported against each K that reuses it. Rebuilding it per K would multiply the run by
    the number of caps and measure the same thing five times.
    """
    cfg, method_key, seed = args
    method = cfg.method(method_key)

    started, cpu0 = time.perf_counter(), time.process_time()
    embedded = _embed(cfg.smiles, cfg.reference_attempts, seed, cfg)
    pool = _relax(embedded, method, cfg)
    reference_wall, reference_cpu = time.perf_counter() - started, time.process_time() - cpu0

    # A force field that cannot parameterise this molecule minimises nothing, so the row
    # would be the ETKDG control wearing another method's name -- a whole mislabelled
    # result set whose only trace was `lowest_energy=inf` in a file no figure reads.
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

    # Colour typing is annotated onto a molecule the first time it is scored and cached
    # on it, so without this the first cell to touch a molecule paid for every later one
    # and the K axis of the Pareto plot was tilted. Paid once, up front, outside the
    # stopwatch -- like the bank, it is setup rather than a per-cell cost.
    if cfg.color_ff != "default":
        for mol in pool:
            _annotate_color_features(mol, color_ff=cfg.color_ff)

    rows: list[dict] = []
    for count in cfg.candidate_counts:
        # A candidate stream disjoint from the reference stream: the same seed on both
        # sides would score an ensemble against conformers it had already produced.
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
                    # Nine places, not six: whether a cell sits at the ceiling is decided
                    # at a tolerance of 1e-6, and a column rounded to 1e-6 cannot support
                    # that test. The ceiling itself is a threshold the analysis chooses,
                    # so it is derived in figures.py rather than frozen in here.
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
    reference = {
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
    """Run the whole grid and write the tables. Returns the measurements."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: cfg.repeats]
    if cfg.repeats > len(SEEDS):
        raise ValueError(
            f"only {len(SEEDS)} independent seeds are defined, {cfg.repeats} asked for"
        )
    tasks = [(cfg, method, seed) for method in cfg.methods for seed in seeds]
    workers = capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (multiprocessing.cpu_count() or 1))
    logger.info(
        "%d cells: %d methods x %d caps x %d candidate counts x %d seeds, on %d workers",
        cfg.cells,
        len(cfg.methods),
        len(cfg.reference_caps),
        len(cfg.candidate_counts),
        cfg.repeats,
        workers,
    )

    rows: list[dict] = []
    references: list[dict] = []
    if workers <= 1 or len(tasks) < 2:
        results = [_one_stream(task) for task in tasks]
    else:
        with pool_context().Pool(min(workers, len(tasks))) as pool:
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


def _coerce(name: str, value: str):
    """One ``--set name=value`` override, typed from the dataclass field it names."""
    fields = {f.name: f for f in dataclasses.fields(StudyConfig)}
    if name not in fields:
        raise SystemExit(f"unknown setting {name!r}; known: {', '.join(sorted(fields))}")
    annotation = fields[name].type
    text = str(annotation)
    if "tuple[int" in text:
        return tuple(int(part) for part in value.replace(" ", "").split(",") if part)
    if "tuple[str" in text:
        return tuple(part for part in value.replace(" ", "").split(",") if part)
    if "Path" in text:
        return Path(value)
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value


def config_from_args(overrides: list[str]) -> StudyConfig:
    changes = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        changes[name.strip()] = _coerce(name.strip(), value)
    return dataclasses.replace(StudyConfig(), **changes)


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

    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
    resolved["methods_resolved"] = {k: dataclasses.asdict(METHODS[k]) for k in cfg.methods}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")

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
