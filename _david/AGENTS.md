# DrugEx Ligand-Based De Novo Drug Design: Architecture & Agent Guide

> **Audience**: AI Agents & Pair-Programmers  
> **Scope**: Universal 3D Ligand-Based De Novo Drug Design (LBDD) in DrugEx  
> **Working Branch**: `main` (Strictly local commits; push ONLY to private remote `dichtator`; NEVER push to GitHub `origin`)  
> **Strategic Context**: See [`_david/THESIS_PIVOT_REVIEW.md`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/THESIS_PIVOT_REVIEW.md) for complete consultation pivot post-mortem.

---

## 1. System Architecture Map

```
_david/
├── THESIS_PIVOT_REVIEW.md          # Master architectural post-mortem and thesis narrative
├── AGENTS.md                       # Single source of truth for AI agents (this document)
├── in_molecule.txt                 # Default reference query molecule (EGCG)
├── references/
│   └── my_molecule_3d.sdf          # Default 3D reference query coordinates
├── code/
│   ├── config.py                   # Canonical PipelineConfig, RunPaths, and DEFAULT_MOLECULE
│   ├── environment.py              # Continuous scoring environment (Harrington, Tversky, Hawkins)
│   ├── models.py                   # SequenceRNN & GraphTransformer agent loaders and optimizers
│   ├── pipeline.py                 # MoleculePipeline execution orchestrator
│   ├── schema.py                   # Canonical std schema constants (ROCS, SA, SMILES, etc.)
│   ├── utils.py                    # HardwareManager (CPU/GPU/VRAM solver & environment isolation)
│   └── visualization.py            # ResultAnalyzer, 2D leads, 2x5 diagnostic distributions, RL curves
├── output/                         # Pristine working directory for active SequenceRNN runs (.gitkeep)
├── legacy/                         # Consolidated archive of early research and graph runs
│   ├── runs/                       # Preserved canonical benchmark runs
│   │   ├── run01_baseline/         # Baseline exploration (scaffold-conditioned)
│   │   ├── run14_harrington_decay/ # Harrington Desirability + Cosine Epsilon decay
│   │   ├── run15_harrington_deep_sampling/ # Hawkins 10-conformer sampling (all-time ROCS 1.678)
│   │   └── RUN_DIFFERENCES.md      # Matrix of historical run evolution
│   ├── anti_cheating/              # Early soft-gate experiments
│   ├── consensus_alignment/        # Consensus pharmacophore experiments
│   ├── dynamic_envelopes/          # Dynamic bounding box prototypes
│   ├── experiments/                # Hypothesis runners and threshold ablations
│   ├── morl_scalarizations/        # Early scalarization implementations
│   ├── standardize_for_regeneration/# Early standardization tools
│   └── tests.py                    # Legacy regression test suite
└── validation/                     # Molecular dynamics and docking validation suite (SIRT6 / DNMT1)
```

---

## 2. Active Generative Paradigm: Pure De Novo SequenceRNN (DrugEx v2)

Following the supervisor consultation on October 25, 2026, the project pivoted from fragment-conditioned graph generation to **unprompted autoregressive SequenceRNN**:

### Conceptual Contrast
- **Legacy Graph-Transformer (Archived Baseline)**: Seeded with retrosynthetic fragments (chroman, flavan-3-ol, galloyl). Suffered from an inductive bias toward scaffold reconstruction ("trying to get the molecule back").
- **Active SequenceRNN (Primary Thesis Focus)**: Receives strictly the `<START>` token. Autoregressively generates SMILES strings character-by-character from the learned probability distribution of drug-like molecules (Papyrus pre-trained prior), guided exclusively by 3D multi-objective reinforcement learning.

### Execution Stages
1. **Stage 0: 3D Reference Conformer Curation & Scoring Setup** ([`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
   - Generate 100 conformers with RDKit `ETKDGv3`, minimize via `MMFF94` (with `UFF` fallback).
   - Filter to physiological energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
   - Cluster via **Hawkins (2017) Butina RMSD** to extract lowest-energy conformational basin centroids.
2. **Stage 1: Unprompted SequenceRNN Reinforcement Learning** ([`models.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/models.py), [`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
   - **Prior & Agent**: `SequenceRNN` initialized with pre-trained Papyrus prior (`data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/`).
   - **Explorer**: `SequenceExplorer` (or `DecayingSequenceExplorer` with cosine epsilon schedule).
   - **Environment**:
     - **3D ROCS Scorer**: `RDKitROCSScorer` (Mills-Dean color SMARTS) or `RDKitROCSTverskyScorer` (asymmetric subshape scoring).
     - **Synthetic Accessibility**: `Property(prop="SA")` mapped via continuous `SafeSAModifier`.
     - **Multi-Attribute Harrington Desirability**: Smooth Gompertz aggregation eliminating boolean step-function cliffs.
     - **Batch-Unique Zero-Masking**: Penalizes intra-batch duplicates and zero-reward invalid SMILES.
3. **Stage 2: Result Analysis & Presentation Profiling** ([`visualization.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/visualization.py))
   - Deduplicate SMILES by canonical form (`Chem.CanonSmiles`).
   - Extract Pareto-optimal leads across ROCS, SAscore, and QED.
   - Produce publication-ready visual artifacts: `lead_structures_2d.png`, `property_distributions.png`, `rl_training_curves.png`, and tabular reports.

---

## 3. Canonical Schema Standard ([`schema.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/schema.py))

All tabular outputs and visualizations must use canonical constants from `std` (`from _david.code.schema import std`):
- Objectives: `std.ROCS`, `std.SA`, `std.DESIRED`
- Identifiers: `std.SMILES`, `std.CANONICAL_SMILES`, `std.FRAGS`, `std.VALID`
- Telemetry: `std.EPOCH`, `std.LOSS_TRAIN`, `std.VALID_RATIO`, `std.UNIQUE_RATIO`, `std.DESIRED_RATIO`
- Physicochemical: `std.MW`, `std.LOGP`, `std.TPSA`, `std.QED`, `std.HBD`, `std.HBA`, `std.ROT_BONDS`

---

## 4. Hardware Management & Safety Invariants ([`utils.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/utils.py))

| Invariant | Setting | Rationale |
|---|---|---|
| Multiprocessing Context | `DRUGEX_MP_CONTEXT="forkserver"` | Prevents POSIX fork deadlocks with PyTorch CUDA thread pools. |
| CPU Worker Allocation | 60% Rule via `get_safe_parallel_jobs()` | Leaves headroom for OS, PyTorch background workers, and I/O. |
| Batch Size Resolution | Power-of-2 Sizing | Scaled dynamically (32, 64, 128, 256) based on available VRAM. |
| GPU Device Selection | Auto Single GPU / CPU Fallback | Auto-routes to the least-loaded GPU meeting $\ge 4000\text{ MB}$ free threshold. |

---

## 5. Security & Synchronization Protocols

- **GitHub Origin Push Locked**: The remote `origin` push URL is locked to `DISABLED_NO_PUSH_SECRET_DATA`. No training data, internal runs, or research commits may ever reach GitHub.
- **Private Machine Synchronization**: All synchronization between local WSL and remote lab machine `dichtator` occurs strictly point-to-point via private, encrypted SSH:
  ```bash
  # Push local WSL commits to lab machine
  git push dichtator main

  # Pull remote lab commits to local WSL
  git pull dichtator main
  ```
- **Pruning Heavy Files**: Never commit `.pkg` checkpoints, raw per-epoch image sequences, or gigabyte-sized log files. Store only summaries, TSV leads, and presentation graphics.

---

## 6. Coding & Academic Contribution Rules

1. **Brutally Honest Academic Rigor**: Challenge assumptions, verify empirical distributions, and report unpromising/counter-intuitive results honestly.
2. **Literature Citations (`#~`)**: Non-trivial scientific/algorithmic decisions must include explicit `#~ [Author, Title/Journal, Year]` citations.
3. **Concise Docstrings**: Maximum 1 line useful for agents (or none where self-explanatory).
4. **Class-First Encapsulation**: Construct objects explicitly and bundle state with behavior; avoid singletons and mutable global states.
5. **Direct File Editing Tools Only**: Always use `write_file` and `edit_file`. Never execute ad-hoc scripts or one-liners to patch code.
6. **Zero Broken Tests & Lints**: Always verify that `ruff check _david/code` passes with 0 errors before staging commits.
