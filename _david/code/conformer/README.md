# Conformer suite — code orientation

Package: `_david/code/conformer/` · Data: `_david/data/conformers/` ·
Deliverables: `_david/output/conformer/` (git-ignored, regenerable) ·
Decision record: [`results.md`](results.md) · Design rationale: [`DESIGN.md`](DESIGN.md)

Four components answer four questions. Everything else is plumbing for them.

| # | Component | Question | Output folder |
|---|---|---|---|
| 1 | Optimization methods | Which reference/candidate physics, `K`, `C` and energy window buys the most 3D self-recovery per second? | `01_optimization_methods/` |
| 2 | Color-shape scoring | Is symmetric TanimotoCombo the right default, and what breaks when a dial moves? | `02_color_shape_scoring/` |
| 3 | Clustering benchmark | Which rule reduces a 50–200 pose pool to a `K`-pose bank? | `03_clustering_benchmark/` |
| 4 | Molecule preparation | What is in the input set, and what survives each expensive stage? | `04_molecule_preparation/` |

The RL pipeline is a consumer of this work, not part of it. Training changes are
recorded in `results.md` under "scoring pin" and summarized in
[`../AGENTS.md`](../AGENTS.md).

---

## Where do I look?

| Task | File |
|---|---|
| Understand the model, defaults, validation | `core/contracts.py` |
| Understand the cache / why a rerun is safe | `core/cache.py` |
| Add a table or panel | `core/datasets.py`, then `cli.py:migrate` |
| Change how one molecule is standardized | `prepare/standardize.py` |
| Change embedding or minimization | `ensemble/embed.py`, `ensemble/minimize.py` (kernel: `../chemistry.py`) |
| Change reference-bank selection | `ensemble/select.py` (kernel: `../reference.py`) |
| Change the score | `scoring/rocs.py`, `scoring/tversky.py` (kernel: `../rocs.py`) |
| Add/change an experiment arm | `studies/optimization_methods.py`, `clustering.py`, `color_shape.py`, `preparation.py` |
| Change the one measurement every cell shares | `studies/measure.py` |
| Change how a study is executed / resumed | `studies/base.py` |
| Change a figure | `analysis/<component>.py`, wired in `analysis/reproduce.py` |
| Change figure style or provenance sidecar | `analysis/style.py` |
| Add a CLI command | `cli.py` |
| Trace historical data back to its run | `analysis/loaders.py`, `_david/data/conformers/canonical/migration_sources.json` |
| Audit real RL runs against the gold standard | `../audit_rl_runs.py`, report `../RL_RUN_AUDIT.md` |
| Verify a change | `_david/code/tests/test_conformer_{core,prepare,scoring,studies,analysis}.py` |

---

## Trace one cell

This is the whole package in one call chain. Read it top-to-bottom once and the
folder structure stops being arbitrary.

```
cli.py  study methods ...
  └─ studies/optimization_methods.py :: build_study     # MoleculeSpec × ArmSpec × repeats
      └─ studies/base.py :: run_study                   # serial or process pool
          └─ studies/base.py :: study_requests          # CellRequest(molecule, arm, repeat)
              └─ studies/measure.py :: MeasureExecutor.__call__
                  ├─ prepare/standardize.py :: prepare_molecule     # SMILES -> canonical identities
                  ├─ ensemble/pipeline.py :: build_reference       # embed -> minimize -> select
                  ├─ ensemble/pipeline.py :: build_candidate       # embed -> minimize
                  └─ scoring/rocs.py :: score_ensemble             # best candidate/reference pair
                      └─ results.csv + errors.csv + manifest.json
```

Workers receive **strings and frozen specs**, never RDKit objects: pickling
conformers is slow and RDKit molecules mutate. A cell is therefore a pure
function of `(spec, cache)` and serial/parallel runs must agree (a test pins it).

---

## Layer rules

```
cli.py
  └── studies/ ──► ensemble/ ──► prepare/ ──► core/
        └── analysis/ ─────────────────────► core/
                        scoring/ ───────────► core/
```

- `core/` imports nothing from the package. It owns contracts, cache, storage,
  provenance, paths, the table/panel registry and the isomer taxonomy.
- `prepare/` decides *what a molecule is*. Nothing above it may re-standardize.
- `ensemble/` consumes prepared molecules and returns conformer records.
- `scoring/` turns two conformer pools into a number. It never mutates a molecule.
- `studies/` declares matrices and measures cells. It contains no plotting.
- `analysis/` reads registered tables only. It must never import `ensemble/` or
  compute chemistry; if a figure needs a new number, the number belongs in a
  study table first.

A lower layer may not import a higher one. This is what makes
`cli figures --canonical-only` safe to run at any time (~40 s, no chemistry).

---

## Five invariants

1. **Expensive work is a pure function of a frozen spec.** `MoleculeSpec`,
   `EmbedSpec`, `MinimizeSpec`, `SelectSpec`, `ScoreSpec`, `ReferenceSpec`,
   `CandidateSpec`, `ArmSpec`, `StudySpec`, `SeedPlan` are frozen dataclasses that
   validate at construction and serialize to the manifest.
2. **The cache key includes the algorithm version.**
   `sha256(ALGORITHM_VERSION ‖ namespace ‖ canonical_json(spec))` under
   `_david/data/conformers/cache/<namespace>/`. Bump `ALGORITHM_VERSION` when a
   change alters a number; never silently delete a stale artifact.
3. **Study tables report the normalized score.** `STUDY_SCALE` is
   `ScoreScale.NORMALIZED`, i.e. `0.5 × (shape + color)` on `[0, 1]`, the RL reward
   axis. Physical TanimotoCombo is `2 × value`.
4. **Failures are data.** A failed cell is a row in `errors.csv` with molecule,
   arm, repeat, exception type and message. It is never swallowed and never
   silently removed from a denominator (`inputs` vs `scored_inputs`).
5. **No figure without a table.** Every PNG has a sibling `.manifest.json` with
   the input table paths, SHA-256 digests, git revision, dirty flag and package
   versions. A canonical figure refuses a legacy-fallback input.

---

## Commands

```bash
# once per host: import validated historical tables + panels into canonical data
.venv/bin/python -m _david.code.conformer.cli migrate

# rebuild every component figure/table from canonical data
.venv/bin/python -m _david.code.conformer.cli figures --canonical-only

# rebuild the molecule registry / preparation audit
.venv/bin/python -m _david.code.conformer.cli registry

# cache occupancy per namespace
.venv/bin/python -m _david.code.conformer.cli cache-stats

# a real study
.venv/bin/python -m _david.code.conformer.cli study methods \
    --panel egcg --workers 8 --output _david/output/conformer/runs/egcg_methods

# verification
.venv/bin/python -m unittest discover -s _david/code/tests -p 'test_*.py'
.venv/bin/ruff check _david
```

`python -m _david.code.audit_rl_runs --regenerate` is the separate RL-run audit:
it folds byte-identical runs, compares each recorded config with
`PipelineConfig()`, rebuilds missing figures, and writes `RL_RUN_AUDIT.md` plus
an untracked CSV.

---

## Traps already paid for

- **Blank cells are "not measured", not `""`.** `datasets.load_table` keeps
  pandas' default NA handling; `keep_default_na=False` once turned every numeric
  column of a table with one blank into `str`.
- **Closures cannot be pickled.** The parallel path must use a module-level
  `_GuardedCell`; a local closure silently made `--workers > 1` unusable.
- **Never key a selection on pool size.** Two different 4-conformer pools shared
  a bank; `pool_fingerprint` is part of the key.
- **Tversky cells need `(alpha, beta)`.** Locating a grid cell by `alpha` alone
  made Dice `(0.3, 0.7)` and Tanimoto `(0.3, 1.7)` the same entry.
- **`variant="tanimoto"` with `alpha != 1` must raise.** It was once accepted and
  silently ignored, reporting a metric nobody requested.
- **A cache hit costs zero seconds.** Reporting the original latency corrupts
  every cost/accuracy/Pareto table.
- **Reference and candidate seeds are disjoint** (`104729` vs `42`). The old
  `43 × 42 == 42 × 43` collision produced two "different" stereoisomers that
  were identical.

---

## Reading order

1. This file — orientation.
2. [`DESIGN.md`](DESIGN.md) — why the layers, cache and score scale look like this.
3. [`results.md`](results.md) — dated findings; the authoritative record, not
   repeated here because numbers rot.
4. `core/contracts.py` → `studies/measure.py` → one `studies/*.py` → one
   `analysis/*.py`.
5. `_david/code/conformer/core/README.md` for low-level serialization helpers
   (note: it predates the current module names; the code is authoritative).

Tests encode the invariants better than prose does. When in doubt, read the
test that covers the function you are about to change.
