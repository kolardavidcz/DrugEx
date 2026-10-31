# DrugEx Thesis Implementation & Technical Agent Guide

This document is the authoritative single source of truth for AI agents and human researchers working on the DrugEx bachelor's thesis codebase (`/home/kolar/DrugEx/_david/`).

---

## 1. Core Engineering & Execution Hygiene

1. **Isolation Boundary**:
   - All thesis code, experimental scripts, configurations, and test suites must strictly reside under `_david/`.
   - **DrugEx core remains strictly read-only.** Never modify files under `drugex/` or repository root.
2. **Dual-Host Development Workflow**:
   - **Local Workstation** (`/home/kolar/DrugEx/`): Used for editing, linting (`ruff`), unit test execution (`python -m unittest discover -s _david/tests -v`), and lightweight batch profiling.
   - **Remote Lab Server (`dichtator`)** (`kolarv@147.33.72.143`): Primary server for offline model evaluation, deep profiling, and sanity runs. Connect via SSH alias `dichtator`. Always check for user edits before synchronizing (`git status`, `git stash`).
   - **MetaCentrum HPC Cluster (`meta`)** (`kolardavidcz@tarkil.metacentrum.cz`): Primary cluster for multi-arm reinforcement learning training and GPU compute. Managed via `_david/code/metacentrum/grid.py`.
3. **Zero Broken Tests & Lints**:
   - Maintain zero tolerance for test failures or lint regressions.
   - Run tests: `/home/kolar/DrugEx/.venv/bin/python -m unittest discover -s _david/tests -v` (Currently 74/74 passing).
   - Run linter: `ruff check _david` (Must have 0 errors, 0 warnings).

---

## 2. Biological Target & Cheminformatics Model

- **Target Query Molecule**: (-)-Epigallocatechin gallate (EGCG)
  - SMILES: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`
  - Formula: $\text{C}_{22}\text{H}_{18}\text{O}_{11}$, MW: $458.37\text{ Da}$, Heavy Atoms: $33$, Rotatable Bonds: $3$, H-Bond Donors: $8$, H-Bond Acceptors: $11$.
  - Physicochemical Profile: Natural polyphenol; **$\text{QED} = 0.21$** (severely penalized by Lipinski / Veber drug-like rules).
- **Physics-Based 3D Scoring Engine**:
  - Implemented in `_david/code/rocs.py` via `ConfigurableRDKitROCSScorer` and `RDKitROCSScorer`.
  - Computes 3D Gaussian volume overlaps (Shape Tanimoto) and color feature alignments (Color Tanimoto using RDKit pharmacophore donor/acceptor/hydrophobe features).
  - Metric: $\text{TanimotoCombo} = \text{ShapeTanimoto} + \text{ColorTanimoto} \in [0, 2]$.

---

## 3. Scientific Discoveries & Autopsy of Prior Failures

### 3.1 Run 01 Autopsy (`rl01_opt1_mild_color_heavy`, 250 Epochs)
- **Results**: Completed 250 epochs in 21.4 hours on `dichtator`. However, median ROCS dropped to $0.000$, syntactic validity crashed to $39.0\%$, and median molecular weight contracted by $-21.3\%$ ($385.5 \to 303.3\text{ Da}$).
- **Root Cause 1: The SA Shrinkage Trap**:
  Used `SmoothClippedScore(5.0, 3.0)` on synthetic accessibility. This function continuously rewarded lower SA without an asymptotic plateau. Because small fragments naturally have lower SA scores, the agent maximized reward by dropping heavy atoms ($27 \to 20\text{ HAC}$).
- **Root Cause 2: The QED Objective Conflict**:
  Included QED with weight $0.25$. Because EGCG has $\text{QED} = 0.21$, optimizing QED provided a strong gradient to strip polyphenol rings into high-QED fragments ($0.85\text{--}0.92$), fighting the 3D shape objective.
- **Root Cause 3: Harrington Gompertz Anchor Compression**:
  Normalized ROCS by dividing by `rocs_clip_upper=2.0`. Under Gompertz anchors $(0.20, 0.05, 0.60, 0.80)$, baseline candidates (raw ROCS $<0.60$) landed in a flat zero-desirability basin, eliminating policy gradients.

### 3.2 Supervisor Breakthrough (`fulopjoz/DrugEx:egcg-rocs-tutorial`)
In `tutorial/advanced/rocs/egcg/final_out/eps0.1/executed_eps0.1.ipynb`, supervisor Jozef Fülöp achieved a massive positive shift:
- **Median ROCS Combo**: $+0.1425$ ($0.638 \to 0.781$), max hit $0.992$, $19.4\times$ enrichment at ROCS $\ge 0.85$.
- **Training Horizon**: Only **50 epochs** (completed in 20.7 hours on a 32-core node).
- **The Critical Fixes**:
  1. Changed SA modifier to `ClippedScore(lower_x=5.0, upper_x=3.0)`. Flat at $1.0$ below raw SA $3.0$, yielding zero marginal reward for fragment shrinkage. MW increased toward EGCG ($387.6 \to 404.0\text{ Da}$).
  2. Dropped QED from RL environment; optimized only ROCS and SA.
  3. Evaluated raw TanimotoCombo $[0, 2]$ directly via `ParetoCrowdingDistance`.
  4. Kept constant exploration noise $\epsilon = 0.10$.

### 3.3 The Epsilon Exploration Mechanism
- In `drugex/training/generators/sequence_rnn.py`, $\epsilon$ is evaluated **at every individual character token** (`torch.rand < epsilon`).
- At $\epsilon = 0.10$, a 40-token SMILES contains an average of $4.0$ prior-mutated tokens, and $98.5\%$ of molecules contain alien tokens.
- However, setting $\epsilon = 0.0$ leads to catastrophic mode collapse (uniqueness collapsed to $0.226$ by epoch 60 in supervisor's `v2` run) because SequenceRNN lacks entropy regularization ($\mathcal{H}=0$) or adaptive KL penalties.
- **The Golden Mean**: $\epsilon = 0.05$ with `use_epsilon_decay = False`. It protects against mode collapse while allowing $70\%\text{--}80\%$ of tokens to be generated purely by the trained policy.

### 3.4 Conformer Generation & Multi-Core Serialization Bottleneck
- In `CandidateConformerGenerator._embed_batch_to_sdf` ([`_david/code/chemistry.py`](code/chemistry.py)), conformer embedding previously ran in a serial Python loop in the main process, capping throughput at $\sim 0.34\text{ mol/s}$ and causing idle CPU cores.
- **The Fix**: Parallelized candidate embedding via chunked multiprocessing pools (`pool_context().Pool`) with `params.numThreads = 1` inside worker processes. Achieves exact bit-for-bit parity (`diff <= 1e-8`) and scales throughput with core count.
- **Pareto Trade-off**: Unminimized ETKDGv3 ($C=30, K=1$) has $\rho = 0.978$ rank correlation with MMFF94s-minimized candidates ($C=50, K=4$) on the 128-drug panel, while running $4\times\text{--}5\times$ faster (~3.8h vs ~16.5h for 50 epochs).

---

## 4. Current Codebase Architecture (`_david/code/`)

```
_david/
├── AGENTS.md                          # Authoritative project instructions & architectural decisions
├── code/
│   ├── config.py                      # PipelineConfig single source of truth for hyperparameters
│   ├── chemistry.py                   # Conformer generation, standardization, parallel embedding
│   ├── environment.py                 # Multi-objective environment, scorers, modifiers, rewards
│   ├── models.py                      # DecayingSequenceExplorer and model wrapper factories
│   ├── pipeline.py                    # 5-stage LBDD execution pipeline and CLI entry point
│   ├── rocs.py                        # ConfigurableRDKitROCSScorer and Gaussian volume alignment
│   ├── schema.py, utils.py            # Standard column names and hardware job allocation
│   ├── visualization.py               # 8 publication-grade diagnostics & trajectory figures
│   ├── conformer/                     # Conformer calibration, batch probes, and results.md
│   │   ├── batch_probe.py             # Realistic multi-core batch scoring benchmark
│   │   ├── drug_panel.py              # DrugCentral curation for self-recovery
│   │   ├── results.md                 # Conformer research findings & decision record
│   │   └── reward_calibration.py      # Historical singleton Pareto benchmark
│   └── metacentrum/                   # MetaCentrum grid submission and cluster orchestration
│       ├── grid.py                    # MetaCentrumGridManager, CLI, and grid generators
│       ├── job_template.sh            # PBS Pro cluster execution script
│       └── scripts/                   # Standalone submission scripts (h1..h4, submit_all)
├── tests/                             # 74 unit and regression tests (100% passing)
└── output/                            # Experiment deliverables, figures, and models
```

---

## 5. MetaCentrum HPC Cluster Execution Guide

- **Connection**: `ssh meta` (Configured in `~/.ssh/config` as `kolardavidcz@tarkil.metacentrum.cz`).
- **Persistent Storage**: `/storage/brno2/brno2/kolardavidcz/DrugEx/` (2.0 PB total, >370 TB free).
- **Codebase Releases**: Staged automatically by `MetaCentrumGridManager.sync_codebase_to_grid()` under `${storage_root}/releases/<commit_sha>/`.

### Resource Directive Guidelines
- **CPU Cores & Memory**: `ncpus=16, mem=32gb, scratch_local=15gb, walltime=24:00:00`.
- **SPEC Directive**: Use `:spec=8.0` for GPU jobs.
  - Setting `:spec=10.0` restricts jobs to only `fobos` (spec 12.0) and `bee` (spec 11.0), which can cause long queue wait times.
  - Setting `:spec=8.0` enables all modern high-performance GPU nodes (Ada Lovelace RTX 4090 on `alfrid` and `fobos`, Hopper H100 on `bee`, Ampere RTX 3090 on `galdor`), giving a pool of >60 GPU nodes while filtering out ancient nodes (`konos`, `glados`).

### Common CLI Commands
```bash
# Check PBS Pro job status
python -m _david.code.metacentrum.grid status
# (or directly via SSH: ssh meta "qstat -u kolardavidcz")

# Synchronize local codebase release to MetaCentrum storage
python -m _david.code.metacentrum.grid sync

# Preview the 4-hypothesis Daytime Sprint
python -m _david.code.metacentrum.grid sprint

# Submit the 4-hypothesis Daytime Sprint to PBS Pro (4 CPUs, gpu_long, 25h walltime, spec>=8.0)
python -m _david.code.metacentrum.grid submit-sprint --queue gpu_long --ncpus 4 --walltime 25:00:00 --spec 8.0

# Pull completed deliverables from MetaCentrum to local _david/output/
python -m _david.code.metacentrum.grid pull
```

---

## 6. Active Experiments (4-Hypothesis Sprint)

Submitted to PBS Pro queue `gpu_long` with `:spec=8.0`, `ncpus=4`, and `walltime=25:00:00` (all unminimized Regime 2: $K=1, C=30, \varepsilon=0.05$, 200 epochs max, 50 min/patience):
1. **`25170006` (`sprint_h1_without_qed_pareto`)**: Supervisor tutorial replication baseline (2-objective `[ROCS, SA]`, Pareto crowding).
2. **`25170007` (`sprint_h2_without_qed_harrington`)**: Harrington reward scheme isolation (2-objective `[ROCS, SA]`, Harrington desirability).
3. **`25170008` (`sprint_h3_with_qed_pareto`)**: QED shrinkage causal ablation (3-objective `[ROCS, SA, QED]`, Pareto crowding).
4. **`25170009` (`sprint_h4_with_qed_harrington`)**: Modernized Run 01 comparison (3-objective `[ROCS, SA, QED]`, Harrington desirability).
