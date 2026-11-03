# Output Organization Guidelines & Index

**Directory:** `_david/output/`  
**Purpose:** Standardized operational protocol for organizing, naming, and maintaining research experiment outputs.

---

## 1. Directory Structure Rules

```
output/
├── conformer/                            # TRACKED IN GIT: the thesis deliverable
│                                         #   (figures + manifests + tables, 3.4 MB)
│   ├── 01_optimization_methods/
│   ├── 02_color_shape_scoring/
│   ├── 03_clustering_benchmark/
│   └── 04_molecule_preparation/
├── yyyymmdd_<researchtopic>/             # Major research categories (date of 1st run)
│   ├── yyyymmdd_hhmm_<experiment_title>/ # Chronological experiment runs
│   └── README.md                         # High-density, no-yapping index of runs
├── yyyymmdd_hhmm_<active_run_title>/     # Currently active or standalone benchmark runs
├── _COOL_STUFF/                          # Index pointing at curated, publication-ready figures
├── archive/                              # Categorized retired, historical, and cautionary runs
└── CONFORMER_BUG_AND_ARTIFACTS_AUDIT.md  # Core mathematical & code audit reference
```

### What is tracked vs untracked

Everything under `_david/output/` is git-ignored by default (`.gitignore`:
`_david/output/*`). The single exception is `_david/output/conformer/`, whose
figures are re-included explicitly: the PNG, its `.manifest.json` input digests
and the CSV it was rendered from must be committed together so they cannot drift
apart, and so the deliverable is visible in any git-aware file explorer. Regenerate
with `cli figures --canonical-only` and commit the PNG + manifest + table in the
same change.

Raw run folders (`yyyymmdd_*`), caches and checkpoints stay untracked.

### Conformer deliverables (`conformer/`)

The conformer figures are no longer stored per-run. They are regenerated from
registered tables into four component folders:

```bash
python -m _david.code.conformer.cli figures --canonical-only
```

Each PNG has a `.manifest.json` with input table SHA-256 digests, git revision
and package versions. Do not copy these PNGs into run folders or `_COOL_STUFF/`
— link to them instead, so the published figures and the canonical data cannot
drift apart. Run folders keep the *raw* tables and per-run outputs; the
`conformer/` folders keep the cross-run answer.


### Naming Conventions
1. **Top-Level Research Categories:**  
   `yyyymmdd_<researchtopic>`  
   *(Timestamp = Date of the earliest run in that domain. If topics diverge significantly, branch into a separate specific `_researchtopic`).*
2. **Experiment Runs (RL & Calibration):**  
   `yyyymmdd_hhmm_<experiment_title>`  
   *(Strict sorting by launch time; lowercase alphanumeric with underscores; no spaces or parentheses).*

---

## 2. Research Topic README Specification

Every `yyyymmdd_<researchtopic>/` directory **MUST** maintain a compact `README.md`.  
Do **NOT** write long discursive summaries. Use strictly high-density, actionable entries:

```markdown
**[`yyyymmdd_hhmm_run_name`](relative_path/)**
- **Why:** Concrete scientific hypothesis or benchmark purpose tested.
- **Key Finding:** Quantitative result with exact numbers, metrics, and conclusion.
```

---

## 3. Archiving & Quarantine Policy

To keep the root clean and prevent erroneous citations:

| Status | Target Location | Naming Prefix / Folder |
| :--- | :--- | :--- |
| **Bugged / Unphysical Runs** | `archive/` | `WITH_CAUTION_<BUG_TYPE>_<folder_name>` *(Explicitly document the failure mechanism in a README)* |
| **Superseded / Intermediate Data** | `archive/` | `redundant_<folder_name>` |

A duplicate is only ever deleted when it is **byte-identical** to a surviving
path. Before deletion, the archive stub records every file's SHA-256 and the
surviving location (`archive/redundant_*/README.md`), so the removal is auditable
without keeping a second copy. Byte counts and hashes are the evidence; prose is
not.
