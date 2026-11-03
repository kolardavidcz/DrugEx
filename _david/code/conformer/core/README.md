# Conformer Core Utilities (`_david/code/conformer/core/`)

Core serialization primitives, atomic file system I/O, cryptographic provenance hashes, and stereoisomer/rotatable-bond sampling budgets.

---

### Key Modules

- **`utils.py`**:
  - `atomic_json(path, data)`: Two-phase atomic write via temporary file replacement; guarantees crash-safe telemetry.
  - `load_sdf(path)` / `save_sdf(records, path)`: Multi-conformer SDF serialization preserving float64 coordinates.
  - `digest(records)`: SHA-256 fingerprint over conformer coordinate arrays for tamper detection.
  - `source_hashes(paths)`: Tracks source code revision hashes inside every benchmark output manifest.
- **`sampling.py`**:
  - `plan_sampling(mol, ...)`: Rotatable bond-dependent conformer budget planning (AQME protocol).
  - `deduplicate_reference(...)`: Heavy-atom RMSD filtering for reference ensemble candidates.
