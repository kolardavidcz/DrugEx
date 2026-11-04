# DrugEx Thesis Implementation & Technical Agent Guide

This document is the authoritative single source of truth for AI agents and human researchers working on the DrugEx bachelor's thesis codebase (`/home/kolar/DrugEx/_david/`).

---

## 1. Core Engineering & Execution Hygiene

1. **Isolation Boundary**:
   - All thesis code, experimental scripts, configurations, and test suites must strictly reside under `_david/`.
   - **DrugEx core remains strictly read-only.** Never modify files under `drugex/` or repository root.
2. **Dual-Host Development Workflow**:
   - **Local Workstation** (`/home/kolar/DrugEx/`): Used for editing, linting (`ruff`), unit test execution (`python -m unittest discover -s _david/code/tests -p "test_*.py" -v`), and lightweight batch profiling.
   - **Remote Lab Server (`dichtator`)** (`kolarv@147.33.72.143`): Primary server for offline model evaluation, deep profiling, and sanity runs. Connect via SSH alias `dichtator`. Always check for user edits before synchronizing (`git status`, `git stash`).
   - **MetaCentrum HPC Cluster (`meta`)** (`kolardavidcz@tarkil.metacentrum.cz`): **Ad-hoc** host for occasional single CPU-heavy jobs; not a mirror of the local tree and not the default production queue. Managed via `python -m _david.code.metacentrum.submit`; the agent-facing template is [`code/metacentrum/RUNBOOK.md`](code/metacentrum/RUNBOOK.md).
3. **Zero Broken Tests & Lints**:
   - Maintain zero tolerance for test failures or lint regressions.
   - Run tests: `/home/kolar/DrugEx/.venv/bin/python -m unittest discover -s _david/code/tests -p "test_*.py"` (Currently 145/145 passing; 10 canonical-data tests skip on a host without `_david/data/conformers/canonical/`).
   - Run linter: `ruff check _david` (Must have 0 errors, 0 warnings).

---

## 2. Biological Target & Cheminformatics Model

- **Target Query Molecule**: (-)-Epigallocatechin gallate (EGCG)
  - SMILES: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`
  - Formula: $\text{C}_{22}\text{H}_{18}\text{O}_{11}$, MW: $458.37\text{ Da}$, Heavy Atoms: $33$, Rotatable Bonds: $3$, H-Bond Donors: $8$, H-Bond Acceptors: $11$.
  - Physicochemical Profile: Natural polyphenol; **$\text{QED} = 0.21$** (severely penalized by Lipinski / Veber drug-like rules).
- **Physics-Based 3D Scoring Engine**:
  - Implemented in `_david/code/rocs.py` via `create_rocs_scorer` → `CanonicalROCSScorer` (`RDKitROCSScorer`), using `use_colors=True`.
  - Computes 3D Gaussian volume overlaps (Shape Tanimoto) and color feature alignments (Color Tanimoto). Color typing is `color_ff = "oe_mills_dean"` (the OpenEye ImplicitMillsDean SMARTS port in DrugEx core), so donor/acceptor/cation/anion/rings/hydrophobe features match OpenEye ROCS rather than RDKit's `BaseFeatures.fdef` fallback.
  - Metric: $\text{TanimotoCombo} = \text{ShapeTanimoto} + \text{ColorTanimoto} \in [0, 2]$, passed to the reward **raw** (`rocs_use_modifier=False`, no ClippedScore re-scaling). If the modifier is ever enabled, `rocs_clip_upper = 2.0` — the physical ceiling; the old 1.8 bound truncated the best-scoring decile.
- **Gold standards (2026-11-03)** — the defaults every new production run uses; historical runs differ and are audited in `_david/code/RL_RUN_AUDIT.md`:
  1. **Checkpoint criterion**: `criteria = avg_gmean` (geometric mean of per-objective desirability ratios), not `avg_amean`. `--criteria gmean` is the CLI spelling.
  2. **Similarity**: Surový (raw) ROCS TanimotoCombo on $[0, 2]$, `Shape + Color`, `use_colors = True`, `color_ff = "oe_mills_dean"`; no asymmetric Tversky, no clipping, no Gaussian re-weighting.
  3. **Final evaluation**: `n_eval_samples = 5000` molecules per arm (5× the historical 1,000), so distribution and Pareto statistics are not sampling-noise dominated.

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
- **Calibration-Seconds ≠ RL-Seconds (2026-11-01)**: reward-calibration per-input seconds are isolated serial single-molecule calls on DrugCentral drugs; the measured live-RL reference (dichtator smoke, MMFF94s ε78.4 K4/C20, 2 workers) ran **≈3.2× slower per row** (6.07 vs 1.88 s serial-equivalent) because RNN-prior molecules carry unassigned stereocenters (I≤4 enumeration) and straggler chunks bound epoch wall-clock. Never read the Pareto x-axis as RL epoch time; use the p95 axis and the printed RL factor (`visualization.rl_reference_measurement`).

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
│   ├── rocs.py                        # CanonicalROCSScorer (raw TanimotoCombo) and Gaussian volume alignment
│   ├── schema.py, utils.py            # Standard column names and hardware job allocation
│   ├── visualization.py               # 10 publication-grade diagnostics & trajectory figures
│   ├── audit_rl_runs.py               # Gold-standard conformance + figure-completeness audit
│   ├── RL_RUN_AUDIT.md                # Generated run table (config vs gold, artifact verdict)
│   ├── conformer/                     # Conformer suite: four components + results.md
│   │   ├── core/                      # Frozen contracts, content-addressed cache, storage,
│   │   │                              #   provenance, dataset registry, isomer taxonomy
│   │   ├── prepare/                   # Standardization, stereoisomer enumeration, registry
│   │   ├── ensemble/                  # ETKDG embedding, force-field minimization, bank selection
│   │   ├── scoring/                   # Production ROCS wrapper + asymmetric Tversky maths
│   │   ├── studies/                   # Declarative cell matrices per component
│   │   ├── analysis/                  # Canonical table loaders, figures, `reproduce.py`
│   │   ├── cli.py                     # migrate · registry · figures · study · cache-stats
│   │   ├── README.md / DESIGN.md      # Package overview / design rationale
│   │   └── results.md                 # Conformer findings & decision record
│   └── metacentrum/                   # Ad-hoc PBS jobs: one CLI, one launcher + job body, tutorials
│       ├── RUNBOOK.md                 # The agent-facing template: commands + recipes
│       ├── submit.py                  # check · sync · submit · status · logs · cancel · pull
│       ├── pbs_launcher.sh            # PBS entry: env guards + exec, no logic
│       ├── job.py                     # PBS body: run id, release copy, pipeline, copy-back
│       ├── help/                      # 01-submit … 06-storage + monitor_live_epochs.py
│       ├── scripts/runed/             # Frozen historical PBS runners (provenance only)
│       └── archive/                   # compare_output_trees.py + its usage contract
├── code/tests/                         # 145 unit and regression tests (100% passing)
└── output/                            # Experiment deliverables, figures, and models
```

---

## 5. MetaCentrum (ad-hoc PBS jobs)

MetaCentrum is used **ad hoc**: periodic single jobs for CPU-heavy ROCS / conformer work,
when local or `dichtator` throughput is insufficient. It is not a production mirror of the
local tree. Read [`code/metacentrum/RUNBOOK.md`](code/metacentrum/RUNBOOK.md) before
submitting; the tutorials in [`code/metacentrum/help/`](code/metacentrum/help/) cover
monitoring, transfer and failure triage.

- **Connection**: `ssh meta` (configured in `~/.ssh/config` as
  `kolardavidcz@tarkil.metacentrum.cz`). Kerberos tickets expire and need an interactive
  2FA `kinit`; `submit.py` detects the failure and prints the fix.
- **Storage**: `/storage/brno2/brno2/kolardavidcz/DrugEx/` — `releases/<sha>/`, `current`,
  `models/` (Papyrus `.pkg`/`.vocab`), `jobs/<name>/spec.json`, `_david/output/<run-id>/`,
  and a `.venv/` (5.7 GB, Python 3.11 + CUDA torch) that must never be deleted casually.
  `$HOME` is a different filesystem (`/storage/praha1/home/kolardavidcz`) and only receives
  PBS `.o<id>` logs.
- **Releases**: `submit.py sync` streams `git archive HEAD` to `releases/<sha>/`, repoints
  `current` and refreshes `models/`. A job copies its release into `$SCRATCHDIR/DrugEx` and
  runs from there, so later local edits cannot change a running or queued job. `sync`
  refuses a dirty tracked tree for the same reason.
- **Defaults**: pipeline hyperparameters come from `PipelineConfig`; a submission carries only
  the requested deltas. `--output-dir`, `--agent-path` and `--vocab-path` are owned by the
  template and rejected if passed.

```bash
# Submit with all PipelineConfig defaults, then watch and collect
.venv/bin/python -m _david.code.metacentrum.submit submit --name egcg_default
.venv/bin/python -m _david.code.metacentrum.submit status
.venv/bin/python -m _david.code.metacentrum.submit logs egcg_default -f
.venv/bin/python -m _david.code.metacentrum.submit pull
.venv/bin/python -m _david.code.audit_rl_runs

# Live epochs (reads worker scratch; stdlib-only, runs on the login node)
ssh meta 'python3 /storage/brno2/brno2/kolardavidcz/DrugEx/current/_david/code/metacentrum/help/monitor_live_epochs.py'
```

There is no preview/dry-run mode and no 1-epoch rehearsal: a submission queues the real job,
prints its release, queue, resources and pipeline deltas before `qsub`, and the job's own
startup guards plus its first epoch are the check.

### Resource defaults

- **CPU job**: `ncpus=16, mem=32gb, scratch_local=15gb, walltime=24:00:00`, `spec=8.0`,
  `ngpus=0`, and **no `-q`**. This is what completed the 200-epoch unminimised sprint arms
  inside 24 h.
- **Queues are routed, not chosen**: from `tarkil`, `-q p2e_1d` fails with `Access from host
  not allowed` (`acl_hosts`) and `q_2h`/`q_4h`/`q_1d`/`q_2d` fail with `Access to queue is
  denied` (`from_route_only`). The `default` router picks by walltime: ≤2 h → `q_2h`,
  ≤4 h → `q_4h`, ≤24 h → `q_1d`, ≤48 h → `q_2d`, ≤1 week → `q_1w`. **The default is the only
  lane: 24 h → `q_1d`.** Do not shorten the walltime to land in a shallower queue; request
  the budget the run needs (`48:00:00` → `q_2d` only for minimised sets that cannot fit in a
  day). `--queue` exists only for a route queue/administrator grant such as `--queue gpu_long`.
- **`spec=8.0`**: excludes ancient nodes without shrinking the pool to `fobos`/`bee` as
  `spec=10.0` does.
- **GPU**: only on explicit request (`--ngpus 1 --queue gpu_long`); the generic `gpu` queue
  currently holds 2000+ jobs and ROCS is CPU-bound.
- **Minimised candidates** cost roughly 4× more wall-clock than unminimised ETKDGv3; a
  200-epoch minimised run does not fit in 24 h.

### Collection policy

`pull` is MetaCentrum → WSL only. Before deleting anything remote, verify with
`compare_output_trees.py` (`code/metacentrum/archive/file_folder_tree_comparasion/`) and
pull the specs and PBS logs into `_david/output/_metacentrum_jobs/`. Historical runs exist
only locally since the 2026-11-04 cleanup; see [`code/metacentrum/help/06-storage.md`](code/metacentrum/help/06-storage.md).

---

## 6. Runs, Output Layout & Current State

Every run writes one self-describing directory:

```
_david/output/YYYYmmdd_HHMM[_tag]/
├── manifest.json      # run_id, state, full config, host, PBS job id, source revision
├── submission.json    # MetaCentrum only: exact args + release that produced the run
├── execution.log      # MetaCentrum only: live pipeline stdout/stderr
├── models/            # SequenceRNN checkpoints, monitor telemetry, per-epoch TSV
├── raw_data/          # baseline + reinforced evaluation arms, reference SDF
├── figures/           # publication figures
└── logs/              # execution log
```

- Local default: `_david/output/<minute>[_tag]`; `--run-tag` names it, `--output-dir` overrides the whole path.
- MetaCentrum: `$STORAGE/_david/output/<minute>_<experiment>`, written by `job.py`.
- A same-minute collision gets `_2`, `_3` … so two runs cannot share a manifest.
- Historical runs (`sprint_h1_*`, `egcg_k4_*`, …) keep their original names; `RL_RUN_AUDIT.md` and `manifests` are the authoritative record of what each one is.

No production batch is queued as of 2026-11-04. The gold-standard defaults are
`avg_gmean` / `n_eval_samples=5000`; run `python -m _david.code.audit_rl_runs`
after a pull to check which completed runs match them.
