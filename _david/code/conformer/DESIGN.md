# Design notes: why the conformer package looks like this

This document explains the *reasoning* behind the package structure. The
narrative overview is in `README.md`; the dated scientific findings are in
`results.md`. Nothing here restates numbers that live in tables.

---

## 1. What was wrong with the previous version

The first implementation grew by accretion: one script per experiment
(`mega.py`, `batch_probe.py`, `reward_calibration.py`, `rl_batch_calibration.py`,
`panel_figures.py`, `self_recovery.py`, …), each with its own CSV schema, its own
seeds, and its own copy of "load the panel, build the references, score the
candidates". Consequences that mattered scientifically:

1. **Two experiments could not be compared.** Different scripts normalized the
   score differently (raw `[0, 2]` TanimotoCombo in one, divided-by-two in
   another), so a "0.91" in one table and "1.82" in another were the same number.
2. **Nothing was reusable.** A 128-drug panel result could not be reused by a
   later figure without re-running it, because the intermediate conformers lived
   in a dated output folder with no identity.
3. **Caching was implicit and unsafe.** Some runs keyed on input paths, some on
   molecule counts; a changed force field could reuse a stale ensemble.
4. **Failure handling was ad hoc.** A molecule that failed preparation silently
   disappeared from a denominator, which quietly improved every mean.
5. **Grid searches dominated the tree.** Multi-thousand-cell "mega" sweeps
   answered questions nobody had asked again, while the four questions that
   actually decide the production scorer were spread across five files.

The rewrite keeps the physics and the scoring geometry — `_david/code/chemistry.py`,
`reference.py`, `rocs.py`, `config.py` remain the shared kernel that the RL
pipeline also consumes — and replaces everything above it with four components
that answer four questions. The kernel did get two surgical changes on the same
day, both recorded in `results.md`: `chemistry.embed_conformers()` became the
single ETKDGv3 entry point (so reference banks and candidate ensembles cannot
diverge), and production scoring was pinned to canonical TanimotoCombo — the
162-cell Tversky grid is a calibration result, not a runtime switch.

The previous tree is frozen, unimported, in `_david/code/archive/conformer_v1/`.
The multi-thousand-cell `mega` sweeps are intentionally **not** ported: they are
archived evidence, not a maintained code path.

---

## 2. Layer map and dependency direction

```
cli.py
  └── studies/  ──► ensemble/ ──► prepare/ ──► core/
        └── analysis/  ─────────────────────► core/
                        scoring/ ────────────► core/
```

- **`core/`** has no RDKit-dependent science: `contracts` (frozen specifications),
  `cache`, `storage` (atomic writes), `provenance`, `paths`, `datasets`
  (the canonical table/panel registry), `taxonomy` (isomer bins), `errors`.
  It may be imported by anything; it imports nothing from the package.
- **`prepare/`** owns standardization, stereoisomer enumeration and the molecule
  registry. It is the only place that decides *what a molecule is*.
- **`ensemble/`** owns ETKDG embedding, force-field minimization and
  reference-bank selection. It consumes prepared molecules and returns conformer
  records plus the statistics needed to justify a bank.
- **`scoring/`** owns the number: production ROCS wrapper and the Tversky grid
  mathematics. Scoring never mutates a molecule.
- **`studies/`** declares matrices (`MoleculeSpec × ArmSpec × repeat`) and the one
  measurement they share (`measure.py`). It contains no plotting.
- **`analysis/`** reads registered tables and draws figures. It never computes
  chemistry, which is why `cli figures` takes ~40 s and is always safe to run.

The rule that keeps this clean: **a lower layer may not import a higher one, and
`analysis/` may not import `ensemble/`.** If a figure needs a new number, the
number belongs in a study table first.

---

## 3. Contracts: expensive work is a pure function of a hashable spec

`core/contracts.py` defines the entire domain model as frozen dataclasses:
`MoleculeSpec`, `EmbedSpec`, `MinimizeSpec`, `SelectSpec`, `ScoreSpec`,
`ReferenceSpec`, `CandidateSpec`, `ArmSpec`, `StudySpec`, `SeedPlan`.

Why frozen dataclasses and not dicts or config files:

- **Hashability** gives the cache key for free and makes "did the spec change?"
  a one-line equality check instead of a diff of YAML.
- **Validation at construction** means an invalid study cannot exist in memory,
  let alone on disk. `EmbedSpec(attempts=0)`, `SelectSpec(cap=-1)`,
  `ScoreSpec(variant="tanimoto", alpha=0.6)` all raise immediately.
- **Serializability** (`jsonable`, `fingerprint`) makes the spec the manifest:
  what was run and what the run is keyed on are the same object.

Conventions worth knowing:

| Convention | Reason |
|---|---|
| `SelectSpec.cap == 0` means *retain the whole pool* (the historical `all` arm) | Avoids overloading `None` and keeps integer table columns integer |
| `EmbedSpec.isomer_cap` defaults to `1` | Default = "prepare exactly the identities the input specifies"; enumeration cost is paid only when explicitly requested |
| `ReferenceSpec.embed` defaults to `I=8`, 100 attempts, RMSD 0.35 Å, seed 104729 | One named factory (`_reference_embed_spec`) instead of a duplicated literal |
| `ScoreSpec(variant='tanimoto')` requires `alpha = beta = 1` | TanimotoCombo *is* symmetric Tversky; accepting an asymmetric request would report a metric different from the one asked for |
| All floats validated finite and non-negative | `NaN`/`inf` in a spec silently poisons every downstream cache key |

Seeds are split deliberately: references use `104729`, candidates use `42`.
Sharing a stream once produced a multiply collision in RDKit's enumeration
(`43*42 == 42*43`) and two "different" stereoisomers came out identical.

---

## 4. Score scale: one number, one meaning

`ScoreScale.PHYSICAL` is shape Tanimoto + color Tanimoto on `[0, 2]`.
`ScoreScale.NORMALIZED` is half of that on `[0, 1]`.

**Study tables always report `NORMALIZED`** (`studies/measure.py: STUDY_SCALE`),
because that is the axis the RL reward consumes; a table whose "0.91" needs a
mental doubling is a trap. `EnsembleScore` carries both the requested scale and
the variant, so a reader can never mistake one for the other, and
`PHYSICAL = 2 × NORMALIZED` recovers the physical value exactly.

The migrated historical tables already stored normalized scores, so figures
labelled "ROCS TanimotoCombo / 2" are directly comparable with new study output.
The one place where this matters most is the Pareto x-axis: seconds per molecule
are also normalized, and the calibration seconds are *serial single-molecule*
measurements — never RL epoch wall-clock (see `results.md`, 2026-11-01).

---

## 5. Configuration-independent caching

`core/cache.py` implements a content-addressed store:

```
key   = sha256(ALGORITHM_VERSION ‖ namespace ‖ canonical_json(spec_payload))
path  = <root>/<namespace>/<first-2-hex>/<key>.<ext>   (json | sdf | npz)
sidecar = <key>.meta.json                              (written after the payload)
```

Properties that were deliberate:

- **Version in the key.** `ALGORITHM_VERSION` is a literal string in
  `contracts.py`. Any change that alters the numbers an entry would produce
  requires bumping it; a stale artifact can then never be silently reused.
  Deleting cache files is never the fix.
- **One key, several representations.** The same artifact may be stored as SDF
  (conformers), NPZ (arrays) or JSON (statistics) without recomputation.
- **Payload before metadata.** A crash can leave an orphan payload, never a
  manifest pointing at a missing file.
- **Size-validated reads.** An empty or truncated file is a miss, not a failure.
- **`enabled=False` is a full no-op.** Even `put_*` writes nothing and returns
  the path it *would* have used, so debugging ("what would this compute without
  the cache?") cannot pollute the store.
- **Hit latency is zero.** A cache hit reports `seconds=0.0`; reporting the
  original cost would corrupt every cost/accuracy table and Pareto front.

Namespaces in use: `prepare`, `embed`, `minimize`, `select`, `score`.

Two key-design bugs are pinned by tests because they are easy to reintroduce:

- Keying a selection on the *pool size* meant two different 4-conformer pools
  shared a bank. `select_ensemble` now keys on `pool_fingerprint`.
- Keying a scoring call on anything less than the full `ScoreSpec` lets an
  asymmetric Tversky request hit a cached TanimotoCombo result.

---

## 6. Study execution model

A study is the Cartesian product `MoleculeSpec × ArmSpec × SeedPlan.repeats`.
`studies/base.py` expands that into `CellRequest`s and runs them either serially
or through a process pool.

```
cell = (molecule, arm, repeat)
worker: SMILES → prepare → embed → minimize → select → score → dict
```

Why workers rebuild chemistry from strings instead of receiving RDKit objects:
pickling conformers across a pipe is slow, and RDKit objects mutate. Because the
specs are hashable, a worker's result is a pure function of `(spec, cache)` — the
same code path runs in-process for a test and out-of-process for a 128-drug
panel.

Error handling is a design decision, not an accident:

- A failing cell is **a row in `errors.csv`**, with molecule, arm, repeat,
  exception type and message. It is never swallowed, and it never silently
  shrinks a denominator: `scored_inputs` and `inputs` are separate columns in
  the summary tables.
- The pool guard is a module-level `_GuardedCell` object, not a closure
  (closures cannot be pickled — that bug made the parallel path silently
  unusable).
- Every run writes `results.csv`, `errors.csv` and `manifest.json` with git
  revision, dirty flag, package versions, the full spec, cell/worker counts and
  cache statistics.

---

## 7. Taxonomy: stratify by truth, not by budget

`core/taxonomy.py` defines the only isomer bins in the package:

```
(0, 2] (2, 4] (4, 8] (8, 15] (15, 32] (32, ∞)
 1–2    3–4    5–8    9–15    16–32    33+
```

Two decisions:

1. **Bin by the true stereoisomer upper bound**, not by the number a study chose
   to enumerate. A molecule with 64 possible stereoisomers is hard because it
   has 64, not because the study sampled 8.
2. **Bin `0` and missing values as `NaN`.** `pd.cut(..., include_lowest=True)`
   would put `0` into `1–2` and make a broken input look like the easiest case.

The bins are right-closed and include an intentionally empty `9–15` bin so a gap
in the data is visible instead of silently dropped. Study tables
(`studies/optimization_methods.py: stratified_summary`) and figures import the
same definition; two copies with different edges would make two panels
incomparable while both claiming to show "isomer count".

Truncation and symmetry collapse are reported separately. `isomer_upper_bound`
is a naive RDKit count over unassigned centres; symmetric inputs legitimately
collapse (cis/trans ring equivalence), so `isomers_enumerated <
isomer_upper_bound` is not by itself truncation. The registry therefore reports
`hit_enumeration_cap` as the only assertable truncation flag, and the preparation
warnings distinguish `isomer-cap-N-truncated-M` from `symmetry-collapsed-M-to-N`.

---

## 8. Data layout and the provenance chain

```
_david/data/conformers/
├── cache/                     content-addressed artifacts (safe to delete, slow to rebuild)
└── canonical/
    ├── tables/                registered CSV/CSV.GZ, the source of every figure
    ├── panels/                name,smiles[,tags] molecule lists
    └── migration_sources.json which historical file each table came from

_david/output/conformer/
├── 01_optimization_methods/  figures + .manifest.json
├── 02_color_shape_scoring/
├── 03_clustering_benchmark/
└── 04_molecule_preparation/  registry / audit / reference-count tables
```

The chain a reader can follow without trusting anyone:

```
figure.png
  └── figure.manifest.json    input table path + SHA-256 + bytes, git rev, versions, notes
        └── canonical/tables/…csv
              └── migration_sources.json   legacy path + SHA-256 of the original run
```

`analysis/reproduce.py --canonical-only` refuses to draw a "canonical" figure
from a legacy fallback path. `datasets.load_table` still supports the fallback
for migration, but the resolved `TableSource.canonical` flag travels with the
data so the distinction cannot be lost.

---

## 9. Non-goals

- **No new chemistry in this package.** Standardization, embedding,
  minimization, conformer identity and `score_pair` stay in
  `_david/code/chemistry.py` / `reference.py` / `rocs.py`. This package composes
  them; duplicating them would create two truths about what a conformer is.
- **No RL code.** The reward function consumes the scorer; it is not measured
  from inside the training loop.
- **No mega grid searches.** Exploratory sweeps that are not tied to one of the
  four questions are archived evidence. Re-adding one means adding an arm to a
  component study, not a new script.
- **No figure without a table.** Every panel reads a registered table with a
  manifest; a hand-tuned plot from an ad-hoc CSV is exactly the failure mode this
  rewrite removes.

---

## 10. Invariants guarded by tests

`_david/code/tests/test_conformer_{core,prepare,scoring,studies,analysis}.py`
pin the properties above, in particular:

- contract validation and default values (`isomer_cap=1`, `I=8` reference bank),
- symmetric Tversky ≡ Tanimoto, off-grid rejection, asymmetry direction,
- cache namespacing, versioning, disabled no-op, and zero-second hits,
- numeric dtypes of canonical tables after loading,
- serial/parallel study parity and manifest contents,
- isomer bin edges, empty-bin preservation, and `NaN` for zero,
- figure sidecar manifests carrying input SHA-256 digests.
