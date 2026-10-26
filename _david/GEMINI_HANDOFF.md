# Hand-Off Context Document: DrugEx Bachelor's Thesis

**To**: Codex  
**From**: Gemini (Antigravity Pairing Agent)  
**Date**: October 26, 2026  
**Repository**: `/home/kolar/DrugEx` (Branch: `main`, Commit: `9261af8`)  
**Target File**: `_david/GEMINI_HANDOFF.md`  

---

## 1. Precise Research Question, Objectives, Scope, and Supervisor Requirements

### Academic Context
* **Candidate**: David Kolar (VŠCHT Praha – University of Chemistry and Technology Prague; username: `kolarv`).
* **Thesis Topic**: Recurrent Neural Network (RNN) ligand-based *de novo* drug design (LBDD) implementation in DrugEx.
* **Target Query Ligand**: **Epigallocatechin gallate (EGCG)** — SMILES:  
  `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`  
  *Properties*: MW $458.37\text{ g/mol}$, 8 phenolic hydroxyl groups, 7 rotatable bonds. High conformational flexibility and propensity for internal hydrogen bonding collapse.

### Core Research Objectives
1. **Bioisosteric Shape & Pharmacophore Guidance**: Guide an autoregressive `SequenceRNN` agent (pre-trained on the Papyrus 05.5 database) using multi-objective reinforcement learning (RL) to generate novel molecules that match the 3D shape and chemical feature density (ROCS TanimotoCombo) of EGCG.
2. **Multi-Objective Lead Balancing**: Balance 3D ROCS against synthetic accessibility (SA score $\le 3.5$) and drug-likeness (QED $\ge 0.50$).
3. **Exploration vs. Exploitation Dynamics**: Empirically evaluate reinforcement learning exploration mechanisms:
   - Baseline $\epsilon$-greedy exploration ($\epsilon = 0.20$).
   - Pure REINFORCE without exploration noise ($\epsilon = 0.00$).
   - Exploration noise decay schedules ($\epsilon(t) = 0.20 \to 0.05$ via cosine and linear decay).
   - Baseline policy regularization / entropy penalization ($\beta = 0.05$).
4. **The Conformer Bottleneck & Dielectric Physics**:
   - Resolve the "single conformer trap" where candidate molecules are falsely penalized because their 3D pose fails to align to an unshielded vacuum reference.
   - Determine the mathematically justified hyperparameters for:
     - **Initial Reference Optimization** ($K_{\mathrm{ref}}$, ensemble size of target actives).
     - **RL Training Evaluation** ($K_{\mathrm{gen}}$, candidate conformer pool generated per agent sample).
   - Evaluate physical solvent screening: Vacuum MMFF94s ($\epsilon_r = 1.0$) vs. UFF vs. Protein Binding Pocket ($\epsilon_r = 6.5$) vs. Bulk Aqueous ($\epsilon_r = 78.4$).
5. **HPC Grid Scaling**: Transition execution from workstation `dichtator` to the MetaCentrum national computing grid (`metavo.metacentrum.cz`, e-INFRA CZ / CESiD) for large-scale 100-epoch hyperparameter exploration.

---

## 2. Research & Implementation Decisions, Rationale, and Alternatives Rejected

### Decisions Made & Their Rationales
1. **Architecture Decoupling in `_david/code/conformer/`**:
   - *Decision*: Decompose the conformer study into single-responsibility classes:
     - `physics.py`: `ForceFieldVariant` & `PhysicalVariantFactory`.
     - `generator.py`: `ConformerGenerator` (ETKDGv3 embedding + in-C++ RMSD pruning + MMFF/UFF minimization + Butina clustering).
     - `analyzer.py`: `ConformerConvergenceAnalyzer` (Radius of Gyration $R_g$ + pairwise ROCS alignment + grid scan).
     - `io.py`: `ConformerIOManager` (multi-model `.sdf` and `.pdb` export).
     - `visualization.py`: `ConformerVisualizer` (Matplotlib 3D surfaces + 2D annotated heatmaps + Plotly 3D HTML widget).
     - `orchestrator.py`: `ConformerStudyOrchestrator`.
   - *Rationale*: Eliminates monolithic script coupling and ensures direct reuse in both standalone analysis and RL training pipelines.
2. **Dynamic Empirical Baseline Quantile Calibration ($Q_3$)**:
   - *Decision*: Auto-calibrate desirability thresholds from the 75th percentile ($Q_3$) of the pre-training Stage 1 baseline prior arm.
   - *Rationale*: In Run 17, a static ROCS threshold of $0.70$ (requiring raw TanimotoCombo $\ge 1.40$) resulted in `desired_ratio = 0.0%` throughout 50 epochs because the untrained prior distribution maxed out at $0.496$ (median $0.348$, $Q_3 = 0.374$). Dynamic $Q_3$ thresholding provides a non-zero, moving reward gradient.
3. **Unlimited Conformer Timeout (`timeout_seconds = 0`)**:
   - *Decision*: Set `timeout_seconds = 0` (disabled) in `_david/code/config.py` and `_david/code/environment.py`.
   - *Rationale*: The previous 60-second wall-clock timeout aborted conformer generation for larger, flexible candidate molecules. The only constraint must be conformer count (`max_conformers`), never execution time.
4. **Lean HPC Resource Sizing for MetaCentrum**:
   - *Decision*: Sized PBS Pro batch jobs to `ncpus=16` (or 24), `mem=32gb`, `ngpus=1:gpu_cap=cuda80`, `scratch_local=15gb`.
   - *Rationale*: Profiling confirmed DrugEx is **95% CPU-bound** (RDKit ETKDG embedding, MMFF minimization, and ROCS Gaussian overlays take ~18 minutes per epoch on 44 cores, while PyTorch GPU sequence generation takes ~3 seconds). Requesting A100/H100 GPUs or 64+ GB RAM creates massive queue wait times on MetaCentrum without performance gain.

### Alternatives Rejected
* **Hardcoded Static Desirability Thresholds**: Rejected due to mathematical stalling of the desirability reward.
* **Tversky Subshape Scoring during Conformer Calibration**: Postponed. The user mandated focusing strictly on `tanimoto_combo` self-similarity before testing subshape fragment matching.
* **Local Laptop Training**: Strict policy to keep the local machine idle; all compute is executed on remote `dichtator` or MetaCentrum.

### Unresolved Debates & Scientific Questions
* **Candidate Conformer Count ($K_{\mathrm{gen}}$) in RL Training**:
  - *The Debate*: Our conformer self-similarity test on EGCG demonstrated that $K_{\mathrm{ref}}=3$ and $K_{\mathrm{gen}}=30$ recovers $98.6\%$ of the score in vacuum, but in pocket dielectric ($\epsilon_r=6.5$), $K_{\mathrm{gen}}=1\text{--}5$ already achieves $>99\%$ recovery.
  - *The Gap*: The benchmark has not yet measured **Wall-Clock Latency ($\text{ms/molecule}$)** across candidate pool sizes ($K_{\mathrm{gen}} \in [1, 30]$). Because candidate scoring is executed $128 \times \text{epochs}$ times, $K_{\mathrm{gen}}=30$ may impose an unviable $4\times$ computational penalty for negligible gain over $K_{\mathrm{gen}}=10$.

---

## 3. User Ideas vs. Agent Suggestions

| Domain | User Ideas (Approved / Directed) | Agent Suggestions (Adopted) | Speculative / Rejected / Deferred |
| :--- | :--- | :--- | :--- |
| **Target Molecule** | Epigallocatechin gallate (EGCG) selected as thesis focus. | Confirmed appropriate as a hard test case (high flexibility, 8 OH groups). | Fragment-only targets (deferred). |
| **Conformer Benchmark** | Map 3D surface: $K_{\mathrm{ref}}$ vs. $K_{\mathrm{gen}}$ vs. ROCS score; test MMFF, UFF, and water/human environments. | Set literature pocket dielectric to $\epsilon_r = 6.5$ (Warshel / Simonson consensus); added distance-dependent model $\epsilon(r)=r$. | Testing $K_{\mathrm{ref}} > 12$ (rejected as redundant). |
| **Visualization & UX** | Strict $[0.0, 2.0]$ ROCS scale; multi-model `.sdf` and `.pdb` export for VS Code 3D extensions; unlimited timeout. | 2D annotated heatmap matrix to eliminate 3D optical distortion; interactive Plotly 3D HTML widget. | Using PyMOL sidecar for headless batch rendering (deferred). |
| **Architecture** | Refactor `conformer_experiment.py` into clean modules (physics, generator, analyzer, I/O, visualizer). | Created `_david/code/conformer/` package with 10 unit tests and zero pyright errors. | Changing `config.py` user scratchpad comments (rejected by user). |
| **Grid Infrastructure** | Use MetaCentrum (`metavo.metacentrum.cz`, user `kolarv`) via e-INFRA CZ / CESiD. | Built class-first orchestrator `_david/metacentrum/grid.py` and lean PBS template. | Oversized multi-node GPU clusters (rejected). |

---

## 4. Completed Experiments, Measured Results, and Exact Artifacts

### Commit State
* **Local WSL & Remote Dichtator**: Clean, synchronized at commit **`9261af8`** (`refactor(code-review-prep): sanitize legacy imports, resolve pyright typing strictness, and format codebase for manual review`).

### Experiment 1: 50-Epoch Baseline RL (`run17_baseline_50ep`)
* **Configuration**: $\text{epochs}=50$, $\epsilon=0.20$, $\beta=0.0$, `reward_scheme=weighted_sum`, `score_variant=tanimoto_combo`, batch size 64, 500 samples/epoch.
* **Execution**: Workstation `dichtator` (44 CPU threads, GPU 0), duration: 2.88 hours.
* **Measured Metrics**:
  - Validity: $99.80\%$ | Uniqueness: $99.20\%$.
  - Mean Prior ROCS: $0.348$ ($Q_3 = 0.374$, Max $= 0.496$).
  - Mean Reinforced ROCS: $0.463$ (Top lead ROCS: $1.411$).
  - Desired Ratio: Stalled at $0.0\%$ due to legacy static ROCS threshold $0.70$.
* **Artifact Path**: `/home/kolar/DrugEx/_david/output/run17_baseline_50ep/`  
  - Figures: `figures/` (property shifts, Bemis-Murcko scaffolds, ROCS shape vs color).  
  - Leads: `top_prioritized_leads.tsv`.

### Experiment 2: 50-Epoch Pure REINFORCE (`run18_epsilon_zero_50ep`)
* **Configuration**: Identical to Run 17 except $\epsilon=0.00$ (no exploration noise).
* **Execution**: Workstation `dichtator` (GPU 0), duration: 2.85 hours.
* **Measured Metrics**:
  - Validity: $100.00\%$ | Uniqueness: $98.60\%$.
  - Mean Reinforced ROCS: $0.479$ (Top lead ROCS: $1.436$).
  - Desired Ratio: Stalled at $0.0\%$ (pre-dynamic threshold calibration).
* **Artifact Path**: `/home/kolar/DrugEx/_david/output/run18_epsilon_zero_50ep/`.

### Experiment 3: 3D Conformer Convergence & Dielectric Study
* **Script**: `python -u _david/code/conformer_experiment.py`
* **Execution**: Executed on `dichtator` (commit `b2b34f7`, verified again on `9261af8`).
* **Measured Results (Target: EGCG)**:

| Environment Variant | Physics Model | Basins ($>1.25\text{\AA}$) | Mean $R_g$ ($\text{\AA}$) | $K_{\mathrm{ref}}=1, K_{\mathrm{gen}}=1$ | $K_{\mathrm{ref}}=3, K_{\mathrm{gen}}=30$ | Max Recovery | Recovery Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MMFF94s Vacuum** | $\epsilon_r = 1.0, \Delta E \le 8.0$ | 2 | $4.339$ | **`0.7808`** | **`1.9712`** | `1.9712` | **`+152.5%`** |
| **UFF Vacuum** | Classical UFF | 2 | $4.298$ | **`1.9963`** | **`1.9988`** | `1.9988` | `+0.1%` |
| **MMFF94s Pocket** | $\epsilon_r = 6.5$, window=None | 3 | $4.314$ | **`1.9865`** | **`1.9865`** | `1.9865` | `0.0%` |
| **MMFF94s Water** | $\epsilon_r = 78.4$, window=None | 3 | $4.310$ | **`1.9988`** | **`1.9988`** | `1.9989` | `0.0%` |
| **MMFF94s Distance-Dep** | $\epsilon(r) = r$, window=None | 2 | $4.301$ | **`0.8056`** | **`1.8174`** | `1.9707` | **`+144.6%`** |

* **Key Deliverables (`_david/output/conformer_study/`)**:
  - Scorecard: `conformer_convergence_scorecard.csv`.
  - 2D Heatmaps: `01b_conformer_convergence_heatmaps.png`.
  - 3D Surfaces: `01_conformer_convergence_3d_surfaces.png`.
  - Interactive 3D HTML Widget: `01_conformer_convergence_interactive.html`.
  - Multi-Model 3D Models: `models_3d/egcg_conformers_{pocket,water,distancedep,vacuum,uff_vacuum}.{sdf,pdb}`.

---

## 5. Current Unfinished Work, Known Bugs, and Exact Stopping Point

### Exact Stopping Point
The codebase was prepared for manual review:
- Sanitized `_david/legacy/__init__.py`.
- Resolved all pyright typing errors (`0 errors, 0 warnings` in `_david/code/conformer/`).
- Standardized formatting with `ruff format`.
- Pushed commit `9261af8` to `dichtator`.
- The user triggered **"Grilling time for conformer"**: challenging how the benchmark translates to concrete hyperparameters ($K_{\mathrm{ref}}$ and $K_{\mathrm{gen}}$) and flagging that Wall-Clock Latency ($\text{ms/molecule}$) is still missing from the trade-off.

### Current Unfinished Items
1. **Conformer Latency Pareto Frontier**: We have not yet measured empirical wall-clock evaluation time ($\text{ms/molecule}$) as a function of $K_{\mathrm{gen}} \in [1, 30]$ on diverse candidate molecules.
2. **MetaCentrum Live Connection**: The user uploaded an SSH key to Perun (username `kolarv`), but active SSH authentication has not yet completed its first successful handshake (propagation delay). `_david/metacentrum/grid.py` has not yet submitted a live PBS Pro test job.
3. **100-Epoch Production Matrix**: Long runs (Runs 19, 20) were halted to prevent resource waste on `dichtator` until thresholds and conformer settings are calibrated.

### Known Quirks & Bugs Resolved
* *Device Mismatch on Multi-GPU*: `SequenceRNNOptimizer.load_models()` previously instantiated `SequenceRNN` without `use_gpus`, causing tensors to remain on `cuda:0` while explorer targeted `cuda:1`. Resolved by explicitly passing `use_gpus=self.use_gpus` to both agent and prior in `_david/code/models.py`.
* *SSH ControlMaster Hang*: Stale multiplexing sockets in `/home/kolar/.ssh/sockets/kolarv@147.33.72.143-22` caused SSH commands to hang indefinitely. Resolved by terminating stale PID processes and clearing socket files.

---

## 6. User Corrections, Recurring Misunderstandings, and Working Preferences

1. **Academic Debate Over Passive Agreement**:
   - The user does *not* want an assistant that passively agrees. The user expects brutal academic honesty, skepticism, and debate from first principles. Always challenge weak assumptions.
2. **Preserve User Config Comments**:
   - **Critical Rule**: Do NOT delete or rewrite informal user annotations in `_david/code/config.py` (e.g. `#^[3, all]`, `# ^ also try insted of vacume with 8.0 kcal, try water, humen-like envirument`). These are the user's active exploration markers.
3. **CPU-Bound Reality**:
   - Never treat DrugEx as a GPU-heavy deep learning workload. RDKit conformation embedding and shape alignment take **95% of wall-clock time**. Prioritize CPU core allocation (0.90 ratio) over high-end GPUs.
4. **Path Protocol (Anti-Black-Screen)**:
   - Always link files with forward slashes and the Windows mount scheme: `file:///C:/Ubuntu/home/kolar/...`. **Never** use `\\wsl.localhost` or `//wsl.localhost` (causes fatal React `[UriError]` IDE webview crashes).
5. **Direct Tool Editing Only**:
   - Never write Python scripts that parse and edit other Python files via regex or string substitution. Always use direct file editing tools (`write_file`, `edit_file`).

---

## 7. Relevant Files Codex Should Read First

1. **Configuration Single Source of Truth**:
   - [`_david/code/config.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/config.py): Contains all hyperparameters, dynamic threshold percentiles, and exploration schedules.
2. **Conformer Engine Package**:
   - [`_david/code/conformer/physics.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/conformer/physics.py): Dielectric specifications.
   - [`_david/code/conformer/generator.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/conformer/generator.py): ETKDGv3 embedding + Butina clustering.
   - [`_david/code/conformer/analyzer.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/conformer/analyzer.py): Pairwise ROCS and $R_g$ extraction.
   - [`_david/code/conformer/visualization.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/conformer/visualization.py): 3D surface and 2D annotated heatmaps.
3. **RL Pipeline & Environment**:
   - [`_david/code/environment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/environment.py): ROCS scorer wrapping and desirability schemes.
   - [`_david/code/pipeline.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/pipeline.py): 5-stage training orchestrator.
   - [`_david/code/models.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/models.py): Hardware allocation and explorer construction.
4. **MetaCentrum Grid Infrastructure**:
   - [`_david/metacentrum/grid.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/metacentrum/grid.py): PBS Pro job submission and remote rsync syncing.
   - [`_david/metacentrum/job_template.sh`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/metacentrum/job_template.sh): PBS execution shell script.
5. **Key Reports & Artifacts**:
   - Conformer Study Report: `C:\Users\kolar\.gemini\antigravity\brain\3c0732c1-95a4-4444-ba73-318653568f70\conformer_convergence_and_dielectric_study.md`
   - Run 17 Report: `C:\Users\kolar\.gemini\antigravity\brain\3c0732c1-95a4-4444-ba73-318653568f70\run17_50epoch_baseline_evaluation_report.md`
   - Run 18 Report: `C:\Users\kolar\.gemini\antigravity\brain\3c0732c1-95a4-4444-ba73-318653568f70\run18_50epoch_pure_reinforce_evaluation_report.md`

---

## 8. The Next Three Concrete Tasks

### Task 1: Conformer Latency Pareto Benchmark ($\text{ms/molecule}$ vs. $K_{\mathrm{gen}}$)
* **Goal**: Defend the exact choice of `max_conformers` in `config.py` with empirical wall-clock numbers.
* **Implementation**:
  - Sample 50 diverse candidate molecules from the Papyrus baseline prior (covering MW 200–500 and 2–10 rotatable bonds).
  - Measure scoring latency ($\text{ms/molecule}$) and ROCS recovery against the $K_{\mathrm{ref}}=3$ pocket reference for $K_{\mathrm{gen}} \in [1, 3, 5, 10, 15, 20, 30]$.
* **Completion Criteria**:
  - Generate a Pareto chart: Scoring Latency vs. ROCS Recovery.
  - Formally update `max_conformers` in `config.py` based on the knee of the curve (expected sweet spot: $K_{\mathrm{gen}} = 10$).

### Task 2: MetaCentrum Connectivity Verification & Smoke Test
* **Goal**: Validate Perun/LDAP SSH access and verify PBS Pro submission.
* **Implementation**:
  - Run `python -m _david.metacentrum.grid test`.
  - Once verified, sync the codebase and Papyrus models via `python -m _david.metacentrum.grid sync`.
  - Submit a 1-epoch test job: `python -m _david.metacentrum.grid submit --name test_smoke --epochs 1 --samples 100`.
* **Completion Criteria**:
  - `test_connection()` returns `True`.
  - The job completes in scratch, returns zero errors, and output figures sync back to `_david/output/`.

### Task 3: Launch Production 100-Epoch Matrix Runs
* **Goal**: Execute the 4 core exploration experiments to 100 epochs with dynamic $Q_3$ calibrated thresholds.
* **Experiments**:
  1. `run_01_baseline`: $\epsilon = 0.20, \beta = 0.0$.
  2. `run_02_pure_reinforce`: $\epsilon = 0.00, \beta = 0.0$.
  3. `run_03_cosine_decay`: $\epsilon = 0.20 \to 0.05$ (cosine, 100 epochs).
  4. `run_04_policy_penalty`: $\epsilon = 0.20, \beta = 0.05$.
* **Completion Criteria**:
  - All 4 runs complete 100 epochs without early stopping (`desired_ratio > 0.0`).
  - Comparative scorecard and Bemis-Murcko scaffold diversity figures generated for thesis chapter.
