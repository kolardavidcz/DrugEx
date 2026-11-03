# Conformer core (`_david/code/conformer/core/`)

Plumbing shared by every component. This layer imports nothing from the rest of
the package; it may be imported by `prepare/`, `ensemble/`, `scoring/`,
`studies/` and `analysis/`.

| Module | Responsibility |
|---|---|
| `contracts.py` | Frozen specifications (`MoleculeSpec`, `EmbedSpec`, `MinimizeSpec`, `SelectSpec`, `ScoreSpec`, `ReferenceSpec`, `CandidateSpec`, `ArmSpec`, `StudySpec`, `SeedPlan`), validation, `ALGORITHM_VERSION`, `fingerprint`/`jsonable`. |
| `cache.py` | Content-addressed artifact cache: `sha256(ALGORITHM_VERSION ‖ namespace ‖ canonical_json(spec))`, size-validated reads, disabled no-op, zero-second hits, hit/miss statistics. |
| `storage.py` | Crash-safe atomic writes for text, JSON, CSV and SDF (temporary sibling + replace), plus file records with byte counts and SHA-256. |
| `provenance.py` | Git revision/dirty flag, runtime package versions, UTC timestamps and the study manifest builder. |
| `datasets.py` | Registry of the 10 canonical tables and 3 panels under `_david/data/conformers/canonical/`; dtype-aware loading, legacy fallback marking, and duplicate-name detection. |
| `paths.py` | Canonical locations: `DATA_ROOT`, `CACHE_ROOT`, `CANONICAL_ROOT`, `OUTPUT_ROOT`, the four component directories, and `PROJECT_ROOT`. |
| `taxonomy.py` | The only stereoisomer bins (`1–2, 3–4, 5–8, 9–15, 16–32, 33+`), right-closed, zero treated as missing, empty bins preserved. |
| `errors.py` | Typed recoverable failures: `PreparationError`, `EmbeddingError`, `MinimizationError`, `SelectionError`, `CacheError`, `StudyError`. |

The pre-refactor `utils.py` and `sampling.py` are frozen in
`_david/code/archive/conformer_v1/core/`; their serialization and sampling
helpers were superseded by `storage.py` and the ensemble layer.
