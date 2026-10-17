# DrugEx Ligand-Based De Novo Drug Design: Architecture & Agent Guide

> **Audience**: AI Agents & Pair-Programmers  
> **Scope**: Universal 3D Ligand-Based De Novo Drug Design (LBDD) using Graph-Transformers in DrugEx  
> **Working Branch**: `main` (strictly local commits; NEVER execute `git push`)

---

## 1. System Architecture Map

```
_david/
├── tests.py                        # 13-point regression & integration test suite (unit tests)
├── in_molecule.txt                 # Default reference query molecule (EGCG)
├── AGENTS.md                       # Single source of truth for AI agents (this document)
├── code/
│   ├── config.py                   # Canonical PipelineConfig, RunPaths, and DEFAULT_MOLECULE
│   ├── pipeline.py                 # MoleculePipeline: orchestrates Stage 0 -> Stage 1 -> Stage 2 & CLI runner
│   ├── environment.py              # Modular loader, PrepareMolecule, create_environment, SafeSAModifier, create_rocs_scorer
│   ├── models.py                   # GraphTransformerOptimizer, prepare_seed_fragments
│   ├── schema.py                   # Canonical std schema constants (ROCS, SA, SMILES, etc.)
│   ├── utils.py                    # HardwareManager (CPU/GPU/VRAM solver & environment isolation), FallbackMonitor
│   └── visualization.py            # ResultAnalyzer, 2D leads, 2x5 diagnostic distributions, training curves, unscaling
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
- **Query Input**: Any valid SMILES string via `--molecule`, or structure file (`.sdf`, `.smi`, `.txt`) via `--input` (default proof molecule: Epigallocatechin gallate, EGCG).
- **Conformer Generation**: Generates 100 conformers with RDKit `ETKDGv3` (`pruneRmsThresh=0.35 Å`), minimizes via `MMFF94` with automatic graceful fallback to `UFF` for non-MMFF elements (e.g. Boron).
- **Energy Window Filtering**: Retains conformers within physiological energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
- **Hawkins (2017) Butina RMSD Clustering**: Heavy-atom sphere-exclusion clustering using `AllChem.GetBestRMS` (automorphism and topological symmetry invariant). Extracts lowest-energy cluster centroids.
- **Scaffold Framework Extraction**: Extracts canonical Bemis & Murcko (1996) molecular framework via `MurckoScaffold.GetScaffoldForMol()`, preserving all conjugated rings and linkers. Pure acyclic molecules fall back cleanly to sanitized SMILES.

### Stage 1: Graph-Transformer Lead Generation & Multi-Objective RL ([`models.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/models.py), [`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
- **Model**: `FragGraphExplorer` operating on pre-trained Papyrus prior (`Papyrus05.5_graph_trans_PT`).
- **Prompt Encoding**: Encodes seed Bemis-Murcko framework into graph DataLoader via `dummyMolsFromFragments()` and `FragmentCorpusEncoder(n_proc=1)`.
- **Environment & Scorers**:
  - **3D ROCS Scorer**: Native DrugEx `RDKitROCSScorer` with OpenEye Mills-Dean color SMARTS (`color_ff="oe_mills_dean"`), `score_type="TanimotoCombo"`, mapped to $[0, 1]$ via `ClippedScore(lower_x=0.0, upper_x=2.0, low_score=0.0, high_score=1.0)`.
  - **Synthesizability Feasibility Floor**: `Property(prop="SA")` governed by `SafeSAModifier(upper_x=3.5, lower_x=5.5, low_score=0.0, high_score=1.0)`. Clamps invalid/uncalculable fallback scores ($\le 0.0$) strictly to $0.0$ desirability.
  - **Multi-Objective Reward Scheme**: Native DrugEx `WeightedSum()` with dynamic threshold failure-ratio weighting.
- **Checkpointing & Peak Recovery**: `PeakRecoveryMonitor` tracks rolling desirability (`avg_amean`), saving `_peak.pkg` on improvement and automatically restoring peak model weights onto the agent upon completion.

### Stage 2: Result Analysis & Presentation Profiling ([`visualization.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/visualization.py))
- **Deduplication**: Identifies distinct chemical leads by canonical SMILES (`Chem.CanonSmiles`), eliminating duplicate conformers/isomers.
- **Prioritization**: Sorts candidates deterministically by ROCS (descending), SAscore (ascending / lower is easier), and QED (descending).
- **Physical Unscaling**: Maps normalized ROCS $[0, 1]$ to $[0, 2.0]$ TanimotoCombo and recalculates raw Ertl & Schuffenhauer (2009) SAscore in $[1.0, 10.0]$.
- **Visual Artifacts**:
  - `lead_structures_2d.png`: 2D lead grid with aligned core scaffolds and true physical metric captions.
  - `property_distributions.png`: 2x5 physicochemical profile (ROCS, SA, QED, MW, LogP, TPSA, HBD, HBA, RotBonds, 2D Tanimoto to seed).
  - `rl_training_curves.png`: Evolution of training loss, valid ratio, unique ratio, and desired ratio across epochs.
  - `top_prioritized_leads.tsv` & `summary.md`: Tabular reports saved directly into deliverables root.

---

## 3. Canonical Schema Standard ([`schema.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/schema.py))

All tabular outputs and visualizations must use canonical constants from `std` (`from _david.code.schema import std`):
- Objectives: `std.ROCS`, `std.SA`, `std.DESIRED`
- Identifiers: `std.SMILES`, `std.CANONICAL_SMILES`, `std.FRAGS`, `std.VALID`
- Telemetry: `std.EPOCH`, `std.LOSS_TRAIN`, `std.VALID_RATIO`, `std.UNIQUE_RATIO`, `std.DESIRED_RATIO`
- Physicochemical: `std.MW`, `std.LOGP`, `std.TPSA`, `std.QED`, `std.HBD`, `std.HBA`, `std.ROT_BONDS`

---

## 4. Hardware Management & Multiprocessing Invariants ([`utils.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/utils.py))

| Component / Function | Invariant | Technical Rationale |
|---|---|---|
| `HardwareManager.init_runtime_environment()` | **Auto-called in `MoleculePipeline.__init__`** | Sets `DRUGEX_MP_CONTEXT="forkserver"` and `DRUGEX_SCORE_NJOBS` automatically upon pipeline creation. |
| Multiprocessing Context | **`DRUGEX_MP_CONTEXT="forkserver"`** | Forks worker pools from a clean server process that never inherits PyTorch CUDA thread states, preventing POSIX fork deadlocks. |
| CPU Worker Allocation | **60% Rule via `get_safe_parallel_jobs()`** | Allocates `max(1, int(cores * 0.60))`, leaving headroom for OS, PyTorch background workers, and I/O. |
| Batch Size Resolution | **Power-of-2 Tensor Core Sizing** | Dynamically scales batch size (32, 64, 128, or 256) based on free VRAM, clamped to `n_samples`. |
| GPU Device Selection | **Single GPU or CPU via `get_safe_gpus()`** | Queries `mem_get_info()`, auto-routes to least-loaded single GPU meeting $\ge 4000\text{ MB}$ free threshold, or falls back to CPU (`()`). |

---

## 5. Development & Verification Workflows

### Running Locally (WSL):
```bash
# Run unit and integration test suite
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
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python _david/tests.py"
  ```
- Trigger detached background run:
  ```bash
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python _david/code/pipeline.py --epochs 35 --n-samples 200 --detach"
  ```

---

## 6. Coding & Contribution Rules

1. **Brutally Honest Academic Rigor**: Challenge assumptions, verify empirical distributions, and report unpromising/counter-intuitive results honestly.
2. **Literature Citations (`#~`)**: Non-trivial scientific/algorithmic decisions must include explicit `#~ [Author, Title/Journal, Year]` citations.
3. **Concise Docstrings**: Maximum 1 line useful for agents (or none where self-explanatory).
4. **Class-First Encapsulation**: Construct objects explicitly and bundle state with behavior; avoid singletons and mutable global states.
5. **Direct File Editing Tools Only**: Always use `write_to_file` and `replace_file_content`. Never execute ad-hoc scripts or one-liners to patch code.
6. **Git Safety**: Stage and commit locally with clear descriptive messages (`git commit -m "..."`). **NEVER execute `git push`**.
