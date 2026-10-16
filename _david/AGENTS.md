# DrugEx Ligand-Based De Novo Drug Design: Architecture & Agent Guide

> **Audience**: AI Agents & Pair-Programmers  
> **Scope**: Universal 3D Ligand-Based De Novo Drug Design (LBDD) using Graph-Transformers in DrugEx  
> **Working Branch**: `main` / `my_personal` (strictly local commits; NEVER execute `git push`)

---

## 1. System Architecture Map

```
_david/
├── run.py                          # Primary IDE entrypoint & detached background daemon (--detach)
├── migrate_standards.py            # Automated meta-script migrating historical TSVs/CSVs to std schema
├── tests.py                        # 14-point regression & integration test suite (unit tests)
├── in_molecule.txt                 # Default reference query molecule (EGCG)
├── AGENTS.md                       # Single source of truth for AI agents (this document)
├── code/
│   ├── pipeline.py                 # MoleculePipeline: orchestrates Stage 0 -> Stage 1 -> Stage 2
│   ├── environment.py              # PrepareMolecule (3D conformers & scaffolds), create_environment, SafeSAModifier, create_rocs_scorer
│   ├── models.py                   # GraphTransformerOptimizer, AlgorithmConfig, prepare_seed_fragments
│   ├── schema.py                   # Canonical std schema: constants (ROCS, SA, SMILES, etc.) & duplicate pruning
│   ├── utils.py                    # HardwareManager (CPU/GPU/VRAM solver & environment isolation), FallbackMonitor
│   ├── analysis.py                 # ResultAnalyzer (canonical deduplication, QED tie-breaking) & ParetoFrontier
│   ├── visualization.py            # Publication renderers (Pareto, 2D leads, 9-panel distributions, training curves)
│   └── generate_gallery.py         # Self-contained HTML comparison gallery across all runs
├── output/
│   ├── 01_reference_representation/# Benchmark runs comparing reference conformer representations
│   ├── 02_reward_formulations_and_cheating/ # Reward scalarization ablations
│   ├── 03_hyperparameter_ablations/# Epsilon, tau, and exploration noise sweeps
│   ├── 04_flagship_lead_optimization/ # Native DrugEx 30-epoch production lead optimizations
│   ├── RUN_DIFFERENCES.md          # Comprehensive historical run differences and Git era matrix
│   └── archive/                    # Archived early prototypes and legacy test artifacts
└── legacy/                         # Archived experimental code (anti-cheating soft-gates, consensus pharmacophores)
```

---

## 2. Core Execution Stages

### Stage 0: 3D Reference Conformer Curation & Scaffold Extraction ([`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
- **Query Input**: Any valid SMILES string, RDKit `Mol`, or `.sdf` / `.smi` file (default proof molecule: Epigallocatechin gallate, EGCG).
- **Conformer Generation**: Generates 100 conformers with RDKit `ETKDGv3` (`pruneRmsThresh=0.35 Å`), minimizes via `MMFF94` with automatic graceful fallback to `UFF` for non-MMFF elements (e.g. Boron).
- **Energy Window Filtering**: Retains conformers within physiological energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
- **Hawkins (2017) Butina RMSD Clustering**: Heavy-atom sphere-exclusion clustering using `AllChem.GetBestRMS` (automorphism and topological symmetry invariant). Extracts lowest-energy cluster centroids.
- **Scaffold Framework Extraction**: Extracts canonical Bemis & Murcko (1996) molecular framework via `MurckoScaffold.GetScaffoldForMol()`, preserving all conjugated rings and linkers without fragile bond-cutting heuristics. Pure acyclic molecules fall back cleanly to sanitized SMILES.

### Stage 1: Graph-Transformer Lead Generation & Multi-Objective RL ([`models.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/models.py), [`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
- **Model**: `FragGraphExplorer` operating on pre-trained Papyrus prior (`Papyrus05.5_graph_trans_PT`).
- **Prompt Encoding**: Encodes seed Bemis-Murcko framework into graph DataLoader via `dummyMolsFromFragments()` and `FragmentCorpusEncoder(n_proc=1)`.
- **Environment & Scorers**:
  - **3D ROCS Scorer**: Native DrugEx `RDKitROCSScorer` with OpenEye Mills-Dean color SMARTS (`color_ff="oe_mills_dean"`), `score_type="TanimotoCombo"`, mapped to $[0, 1]$ via `ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)`.
  - **Synthesizability Feasibility Floor**: `Property(prop="SA")` governed by `SafeSAModifier(lower_x=6.0, upper_x=5.5, low_score=0.0, high_score=1.0)`. Clamps invalid/uncalculable fallback scores ($\le 0.0$) strictly to $0.0$ desirability.
  - **Multi-Objective Reward Scheme**: Native DrugEx `WeightedSum()` with dynamic threshold failure-ratio weighting, prioritizing objectives with lower passing rates.
- **Checkpointing & Peak Recovery**: `PeakRecoveryMonitor` tracks rolling desirability (`desired_ratio`), saving `_peak.pkg` on improvement and automatically restoring peak model weights onto the agent upon completion.

### Stage 2: Result Analysis & Publication Profiling ([`analysis.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/analysis.py), [`visualization.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/visualization.py))
- **Deduplication**: Identifies distinct chemical leads by canonical SMILES (`Chem.CanonSmiles`), eliminating duplicate conformers/isomers.
- **Pareto Ranking & Tie-Breaking**: Ranks candidates into non-dominated Pareto Fronts ($0, 1, 2, \dots$); Front 0 ties are broken deterministically by QED drug-likeness descending.
- **Visual Artifacts**:
  - `lead_structures_2d.png`: 2D lead grid with aligned core scaffolds and true metric captions.
  - `pareto_frontier.png`: Scatter plot of optimal Front 0 vs dominated candidates with clear axis labels (`Normalized ROCS Desirability (TanimotoCombo / 2.0)`).
  - `property_distributions.png`: 9-panel physicochemical profile (ROCS, QED, SA, MW, LogP, TPSA, HBD, RotBonds, 2D Tanimoto to seed).
  - `pareto_epoch_evolution.png` / `pareto_epoch_viewer.html`: Interactive micro-Pareto scrubber tracking epoch evolution.

---

## 3. Canonical Schema Standard ([`schema.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/schema.py))

All tabular outputs and visualizations must use canonical constants from `std` (`from _david.code.schema import std`):
- Objectives: `std.ROCS`, `std.SA`, `std.DESIRED`
- Identifiers: `std.SMILES`, `std.CANONICAL_SMILES`, `std.FRAGS`, `std.VALID`
- Telemetry: `std.EPOCH`, `std.LOSS_TRAIN`, `std.VALID_RATIO`, `std.UNIQUE_RATIO`, `std.DESIRED_RATIO`
- Physicochemical: `std.MW`, `std.LOGP`, `std.TPSA`, `std.QED`, `std.HBD`, `std.ROT_BONDS`, `std.PARETO_RANK`

`std.standardize(df)` automatically normalizes historical scorer names and eliminates redundant duplicate columns (`SAscore` when `SA` exists, `Desirability` when `Desired` exists).

---

## 4. Hardware Management & Multiprocessing Invariants ([`utils.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/utils.py))

| Component / Function | Invariant | Technical Rationale |
|---|---|---|
| `HardwareManager.init_runtime_environment()` | **Auto-called in `MoleculePipeline.__init__`** | Sets `DRUGEX_MP_CONTEXT="forkserver"` and `DRUGEX_SCORE_NJOBS` automatically upon pipeline creation. |
| Multiprocessing Context | **`DRUGEX_MP_CONTEXT="forkserver"`** | Forks worker pools from a clean server process that never inherits PyTorch CUDA thread states, preventing POSIX fork deadlocks. |
| CPU Worker Allocation | **60% Rule via `get_safe_parallel_jobs()`** | Allocates `max(1, int(cores * 0.60))`, leaving headroom for OS, PyTorch background workers, and I/O. |
| Batch Size Resolution | **Power-of-2 Tensor Core Sizing** | Dynamically scales batch size (32, 64, or 128) based on free VRAM, clamped to `n_samples`. |
| GPU Device Selection | **VRAM-Aware Routing via `select_best_gpu()`** | Queries `mem_get_info()`, auto-routes to least-loaded GPU meeting $\ge 4000\text{ MB}$ free threshold, or falls back to CPU (`()`). |

---

## 5. Development & Verification Workflows

### Running Locally (WSL):
```bash
# Run 14-point integration test suite
python -m unittest -v _david/tests.py

# Lint and check style compliance
ruff check _david/code _david/tests.py
```

### Running on Remote Hardware (`dichtator`):
- Probe connection (reuses 34ms `ControlMaster` socket via WSL):
  ```bash
  wsl ssh -o BatchMode=yes dichtator "nvidia-smi; free -h"
  ```
- Run tests remotely:
  ```bash
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python -m unittest -v _david/tests.py"
  ```
- Trigger detached background run:
  ```bash
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python _david/run.py --mode full --epochs 35 --n-samples 200 --detach"
  ```

---

## 6. Coding & Contribution Rules

1. **Brutally Honest Academic Rigor**: Challenge assumptions, verify empirical distributions, and report unpromising/counter-intuitive results honestly.
2. **Read-Only Mode for Questions**: Treat exploratory questions as read-only research requests without modifying project source files.
3. **Direct File Editing Tools Only**: Always use `write_to_file` and `replace_file_content`. Never execute ad-hoc scripts or one-liners to patch code.
4. **Type Annotations & Rich Docstrings**: All Python functions require full type hints on parameters and return values.
5. **Chemoinformatics Documentation**: External non-DrugEx library calls (RDKit, OpenEye) require `# [PACKAGE: NAME] WHY: ... WHAT: ... HOW: ...`.
6. **Git Safety**: Stage and commit locally with clear descriptive messages (`git commit -m "..."`). **NEVER execute `git push`**.
