# Conformer ensemble, shape/color scoring, and reproducible studies

Package: `_david/code/conformer/`
Data: `_david/data/conformers/` · Deliverables: `_david/output/conformer/`

This package answers four separate questions, and everything in it exists to
answer one of them:

| Component | Question | Deliverable folder |
|---|---|---|
| 1. Optimization methods | Which reference/candidate physics, reference cap `K`, candidate budget `C`, and energy window buys the most 3D self-recovery per second? | `01_optimization_methods/` |
| 2. Color-shape scoring | Is the production scorer (`K=4`, `C=20`, `I=4`, MMFF94s ε=78.4, symmetric TanimotoCombo, 0.5 shape + 0.5 color) the right default, and what breaks if it is moved? | `02_color_shape_scoring/` |
| 3. Clustering benchmark | Which reference-merging rule should reduce a 50–100 conformer bank to `K` poses? | `03_clustering_benchmark/` |
| 4. Molecule preparation | What exactly is in the input set, and how many conformers survive each expensive stage? | `04_molecule_preparation/` |

The RL pipeline's behaviour is untouched: this package *measures* the scorer and
the reference-bank policy that the reward function consumes. What it did change
outside the package is one thing — training can no longer be pointed at an
experimental similarity variant (see `results.md`, "scoring pin").

---

## 1. Design in one paragraph

Every expensive step is a pure function of an explicit, hashable specification
(`core/contracts.py`). A study is a Cartesian product of
`MoleculeSpec × ArmSpec × repeat`; a cell rebuilds chemistry from strings inside
its worker and shares one content-addressed cache on disk
(`sha256(algorithm_version, namespace, specification)`). That is why a study can
run serially or through a process pool, resume after a crash, and never silently
reuse an artifact produced by a different algorithm version.

Read `DESIGN.md` for the reasoning behind these choices, the score-scale
convention, and the cache-key scheme.

---

## 2. Layout

```
conformer/
├── core/         contracts, cache, storage, provenance, dataset registry, taxonomy
├── prepare/      standardization, stereoisomer enumeration, molecule registry, panels
├── ensemble/     ETKDG embedding, force-field minimization, reference-bank selection
├── scoring/      production ROCS wrapper + asymmetric Tversky grid mathematics
├── studies/      declarative study matrices (methods, clustering, color-shape, preparation)
├── analysis/     canonical table loaders, component figures, regeneration entry point
└── cli.py        migrate · registry · figures · study · cache-stats
```

The scientific kernel it builds on is read-only and lives one level up:
`_david/code/chemistry.py` (standardization, embedding, minimization, identity
checks), `_david/code/reference.py` (Butina / diverse ordering), and
`_david/code/rocs.py` (`score_pair`, physical TanimotoCombo on `[0, 2]`).

The previous implementation of this package is frozen in
`_david/code/archive/conformer_v1/` for provenance. It is not imported anywhere.

---

## 3. Usage

All commands are idempotent.

```bash
# Copy validated historical tables + panels into canonical storage (once).
python -m _david.code.conformer.cli migrate

# Rebuild the molecule registry / preparation audit from the canonical panels.
python -m _david.code.conformer.cli registry

# Regenerate every component figure and table from canonical data (~40 s).
python -m _david.code.conformer.cli figures --canonical-only

# Inspect cache occupancy per namespace.
python -m _david.code.conformer.cli cache-stats
```

Running a real study:

```bash
# Optimization-method arms on the EGCG stress panel, 8 worker processes.
python -m _david.code.conformer.cli study methods \
    --panel egcg --workers 8 \
    --output _david/output/conformer/runs/egcg_methods

# Clustering benchmark on the 128-drug development panel.
python -m _david.code.conformer.cli study clustering \
    --panel drug_panel_development --workers 8 \
    --selections diverse rmsd_0.35 rmsd_0.50 rmsd_1.25 \
    --caps 1 2 4 8 12 0 --candidate-attempts 20 \
    --output _david/output/conformer/runs/panel_clustering

# Full 162-cell asymmetric Tversky grid.
python -m _david.code.conformer.cli study color-shape \
    --panel-file my_panel.csv --workers 8 \
    --output _david/output/conformer/runs/tversky

# Preparation audit: does every input survive chemical preparation?
python -m _david.code.conformer.cli study preparation \
    --panel drug_panel_development \
    --output _david/output/conformer/runs/preparation_audit
```

Every run writes `results.csv`, `errors.csv` (child failures are recorded, never
swallowed), and a `manifest.json` with the git revision, package versions, the
full specification, cell counts, worker count, and cache statistics.

---

## 4. Conventions that matter scientifically

**Score scale.** Study tables report `ScoreScale.NORMALIZED`, i.e.
`0.5 × (shape + color)` on `[0, 1]`, the axis the RL reward uses. The physical
TanimotoCombo on `[0, 2]` is always recoverable as `2 × combo`. Figures built
from historical tables inherit the same convention because those pipelines
already divided by two.

**Isomer strata.** Molecules are grouped by the *true* number of stereoisomers
the input can have, not by what a study chose to sample:
`1–2, 3–4, 5–8, 9–15, 16–32, 33+` (`core/taxonomy.py`). The `9–15` bin is kept
even when empty so a gap in the data is visible instead of silently dropped.

**Enumeration is explicit.** `EmbedSpec.isomer_cap` defaults to 1, meaning
"prepare the identities the input already specifies". Any study that enumerates
unassigned stereocentres must raise the cap (`ReferenceSpec` defaults to `I=8`,
candidates to `I=4`). The registry audits at `I=32` and reports truncation
separately from symmetry collapse.

**Cache invalidation.** `ALGORITHM_VERSION` in `core/contracts.py` is part of
every key. Bump it whenever a change alters the numbers a cache entry would
produce; never "fix" a stale artifact by deleting it silently.

---

## 5. Tests

```bash
python -m unittest discover -s _david/code/tests -p 'test_*.py'
ruff check _david
```

`_david/code/tests/test_conformer_{core,prepare,scoring,studies,analysis}.py`
cover contract validation, cache semantics, dtype stability of canonical
tables, the Tversky/Tanimoto identity, selection modes, serial/parallel parity of
study execution, manifest contents, and figure provenance sidecars.

---

## 6. Scientific findings

`results.md` is the authoritative dated decision record. It is not duplicated
here on purpose: numbers change, and a README that restates them rots.
