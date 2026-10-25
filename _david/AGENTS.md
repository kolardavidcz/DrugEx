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
├── in_molecule.txt                 # Default reference query molecule
├── references/                     # 3D reference structures and coordinates
├── code/
│   ├── config.py                   # Canonical PipelineConfig, RunPaths, and DEFAULT_MOLECULE
│   ├── environment.py              # Multi-objective environment (ROCS, SA, QED, WeightedSum, Harrington)
│   ├── models.py                   # SequenceRNN optimizer and DecayingSequenceExplorer
│   ├── pipeline.py                 # MoleculePipeline execution orchestrator
│   ├── schema.py                   # Canonical std schema constants (ROCS, SA, QED, SMILES, etc.)
│   ├── utils.py                    # HardwareManager (CPU/GPU/VRAM solver & environment isolation)
│   └── visualization.py            # ResultAnalyzer, 5 publication-ready multi-panel figure generators
├── tests/
│   └── test_sequencer_pipeline.py  # Fast automated unit test suite (models, conformers, environment, schedules)
├── output/                         # Pristine working directory for active SequenceRNN runs (.gitkeep)
└── legacy/                         # Preserved benchmark runs and historical scalarization prototypes
    ├── stupid_graph_transformer/   # Preserved legacy GraphTransformer benchmark runs (01, 14, 15)
    ├── consensus_alignment/        # Consensus pharmacophore experiments
    └── dynamic_envelopes/          # Dynamic bounding box prototypes
```

---

## 2. Generative Paradigm: Pure De Novo SequenceRNN (DrugEx v2)

The primary thesis methodology is **unprompted autoregressive SequenceRNN** (DrugEx v2 SMILES RNN):
- **Input**: Strictly the `<START>` token (no fragment prompts, no scaffold bias, no retrosynthetic constraints).
- **Prior Checkpoint**: Pretrained Papyrus SMILES-RNN (`tutorial/data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/`).
- **Policy Gradient (REINFORCE)**: Evolves character-by-character SMILES generation guided by multi-objective 3D shape, synthetic accessibility, and drug-likeness rewards.

### Execution Stages
1. **Stage 0: 3D Reference Conformer Curation & Scoring Setup** ([`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
   - Generate conformers with RDKit `ETKDGv3`, minimize via `MMFF94` (with `UFF` fallback).
   - Filter to physiological energy window ($\Delta E \le 8.0\text{ kcal/mol}$).
   - Cluster via **Hawkins (2017) Butina RMSD** to extract 3 diverse basin conformers (`max_reference_conformers=3`).
2. **Stage 1: Pre-training Baseline Evaluation Arm** ([`pipeline.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/pipeline.py))
   - Sample $N=1000$ molecules from the untrained prior to establish ground-truth empirical baseline distributions.
3. **Stage 2: Reinforcement Learning Exploration** ([`models.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/models.py), [`environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py))
   - **Agent & Mutate Prior**: Trainable `SequenceRNN` updated against an independent, frozen `SequenceRNN` prior.
   - **Explorer**: `SequenceExplorer` (baseline constant $\epsilon=0.20$) or `DecayingSequenceExplorer` (cosine annealing).
   - **Environment**: Multi-objective `DrugExEnvironment` evaluating 3D ROCS (TanimotoCombo), SA (SmoothClippedScore), and QED.
   - **Reward Scheme**: `WeightedSum` (dynamic failure-ratio adaptive weighting). Pluggable `ParetoCrowdingDistance` and `HarringtonDesirabilityReward`.
4. **Stage 3: Post-Training Evaluation Arm** ([`pipeline.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/pipeline.py))
   - Sample $N=1000$ molecules from the reinforced agent across identical evaluation conditions.
5. **Stage 4: Publication Visualization & Comparative Reporting** ([`visualization.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/visualization.py))
   - Produce 5 publication-ready figures (distributions, trade-offs, trajectories, negative feature diagnostics, lead gallery) and bootstrap 95% confidence intervals on $\Delta\text{median}$.

---

## 3. General (Target-Independent) Methodological Discoveries & Invariants

These architectural laws were derived empirically from extensive profiling, mathematical analysis, and supervisor benchmarks:

### 3.1 The Mutate Prior Identity Trap in DrugEx `SequenceExplorer`
- **Mechanism**: In `SequenceExplorer.forward()`, the agent evolves batches via:
  ```python
  seq = self.agent.evolve(self.batchSize, epsilon=self.epsilon, crover=self.crover, mutate=self.mutate)
  ```
  With probability $\epsilon$, tokens are sampled from `mutate`. If `mutate` is identical to `agent` (or `mutate=None`), mutating samples from the agent's *own* policy. When the agent converges or drifts, $\epsilon$-greedy exploration becomes completely inert ($0\%$ effective exploration).
- **Consequence**: Catastrophic mode collapse (measured uniqueness collapsing from $0.955$ down to $0.226$).
- **Rule**: `mutate` **MUST** be an explicit, independent, frozen instance of `SequenceRNN`:
  ```python
  prior = SequenceRNN(voc, is_lstm=True, device=self.device)
  prior.loadStatesFromFile(str(pkg_path))
  for p in prior.parameters():
      p.requires_grad_(False)
  prior.eval()
  assert prior is not agent, "mutate must not be agent!"
  assert id(prior) != id(agent), "prior and agent share identical memory address!"
  ```

### 3.2 Sequential Conformer Generation vs. Alignment Scaling Laws
- **Mechanism**: In `RDKitROCSScorer.getScores()`, execution is strictly two-stage and sequential:
  1. `conformer_generator.genConformers(smiles)`: Runs in the main process, iterating serially over molecules (`for i, smi in enumerate(smiles_list)`). `num_threads=0` parallelizes conformers within a single molecule, but cannot parallelize across molecules. This stage accounts for $\approx 85\%$ of runtime and sets an empirical floor of $\approx 4.6\text{ s/molecule}$ on CPU.
  2. `Pool(n_jobs)`: Parallel worker pool handles shape/color overlay alignment on pre-generated conformers sequentially *after* stage 1 finishes.
- **Consequences**:
  - `num_threads` and `n_jobs` do **NOT** multiply or oversubscribe because they never run concurrently.
  - Setting `num_threads=1` as an "oversubscription guard" pins conformer generation to 1 CPU core, doubling runtime ($9.33\text{ s/mol}$ vs $4.58\text{ s/mol}$).
  - Hardware scaling saturates rapidly: 16–32 cores achieves optimal throughput. Allocating $>32$ cores yields only $1.01\times$ speedup while wasting cluster capacity. Always set `conformer_threads=0` and cap `n_jobs` at 32.

### 3.3 The Unbounded Single-Objective Exploitation Hazard (Molecule Shrinkage)
- **Mechanism**: Synthetic Accessibility (SAscore) mapped via `SmoothClippedScore(lower_x=5, upper_x=3)` is easy to optimize and unbounded in the rewarded direction. Generating smaller, simpler molecules trivially lowers SAscore. In contrast, 3D shape matching against a flexible drug-sized query is difficult.
- **The Pareto Flaw**: Under `ParetoCrowdingDistance`, non-dominated sorting rewards a molecule that significantly improves SA even if its 3D shape marginally worsens. Consequently, unconstrained RL exploits the cheap axis, causing molecules to shrink (measured heavy atoms dropping from 27 down to 18).
- **The Solution**:
  1. **Include QED**: Quantitative Estimate of Drug-likeness naturally penalizes tiny fragments, stopping the agent from cheating via shrinkage.
  2. **Failure-Rate WeightedSum**: Dynamic weighting increases gradient pressure on difficult objectives (ROCS) when they fail thresholds more often than easy objectives (SA).
  3. **Strict Non-Compensability**: `HarringtonDesirabilityReward` uses Gompertz geometric aggregation where $d_{\text{shape}} \to 0 \implies D(x) \to 0$.

### 3.4 Decoupling Reporting Thresholds from Explorer Weight Selection
- **Mechanism**: `SequenceExplorer.fit()` defaults to `criteria='desired_ratio'`, `reload_interval=50`, `patience=50`, `min_epochs=100`. Even when using a threshold-free reward scheme (like `ParetoCrowdingDistance`), thresholds silently select which model checkpoint survives via periodic resets to `bestState`.
- **Rule**: Fully decouple training thresholds by passing:
  ```python
  explorer.fit(
      epochs=epochs,
      patience=epochs + 1,
      reload_interval=epochs + 1,
      min_epochs=epochs,
  )
  ```
  Save and evaluate the actual final-epoch weights separately (`torch.save(agent.getModel(), ...)`).

### 3.5 POSIX Multiprocessing Context (`fork` vs `forkserver`)
- In Python 3.12 with PyTorch and RDKit, forcing `DRUGEX_MP_CONTEXT="forkserver"` causes worker pool deadlocks during background task spawning. Standard POSIX `fork` executes cleanly in parallel without hang. Set `DRUGEX_MP_CONTEXT="fork"`.

### 3.6 Generator Return Schema Invariants
- `agent.generate(evaluator=env)` does not reliably return `Valid` or `Desired` columns and routes to unclipped scores.
- **Safe Protocol**: Generate with `evaluator=None`, then call `env.getScores(smiles)` explicitly, and canonicalize SMILES with `Chem.CanonSmiles` for exact uniqueness metrics.

---

## 4. Hardware Management & Safety Invariants ([`utils.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/utils.py))

| Invariant | Setting | Rationale |
|---|---|---|
| Multiprocessing Context | `DRUGEX_MP_CONTEXT="fork"` | Avoids Python 3.12 forkserver deadlocks with PyTorch / RDKit. |
| Scorer Pool Worker Cap | Max 32 workers via `capped_n_jobs` | Beyond 32 cores, serial conformer generation dominates; extra workers yield 0 speedup. |
| Conformer Embedding Threads | `conformer_threads = 0` (all cores) | Maximizes multi-threading within single-molecule RDKit conformer embedding. |
| GPU Device Selection | Auto Single GPU / CPU Fallback | Auto-routes to the least-loaded GPU meeting $\ge 4000\text{ MB}$ free threshold. |
| Laptop Battery / CPU Protection | Local WSL for editing/tests only; ML on Dichtator | Laptop lacks CUDA / compute capacity; all training runs execute on remote `dichtator`. |

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
2. **Measure Unfavorable Features**: Always measure and visualize features that do not look promising (molecular shrinkage, repeat counts, rotatable bond drift) to diagnose algorithmic limitations.
3. **Class-First Encapsulation**: Construct objects explicitly and bundle state with behavior; avoid singletons and mutable global states.
4. **Direct File Editing Tools Only**: Always use native file tools (`write_file`, `edit_file`). Never execute ad-hoc scripts or one-liners to patch code.
5. **Zero Broken Tests & Lints**: Always verify that `ruff check _david/code _david/tests` and unit tests pass with 0 errors before staging commits.
