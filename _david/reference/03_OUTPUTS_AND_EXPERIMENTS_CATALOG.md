# 03: Outputs & Experiments Catalog

> **Author**: Antigravity (Gemini 3.8 Flash High)  
> **Prepared for**: Gemini 4.0 Argon & David Kolár  
> **Date**: 2026-10-29  
> **Scope**: Exhaustive catalog of all experimental runs, benchmark datasets, scorecards, and figures in `_david/output/`.

---

## 1. Directory Overview & Size Profile

The `_david/output/` directory contains all generated experimental artifacts, raw CSV tables, PDB/SDF 3D ensembles, training trajectories, and diagnostic figures.

```
_david/output/
├── run16_sequencer_weighted_sum/     [48 MB]  50-epoch RL (WeightedSum reward)
├── run17_baseline_50ep/              [48 MB]  50-epoch baseline RL (Papyrus SequenceRNN)
├── run18_epsilon_zero_50ep/          [48 MB]  50-epoch pure REINFORCE (ε = 0)
├── conformer_study/                  [4 MB]   Initial EGCG dielectric conformer study
├── conformer_benchmark_20261027/     [36 MB]  Conformer benchmark archive & zip
├── 20261028_1405_conformer_analysis/ [26 MB]  Multi-molecule conformer analysis
├── 20261028_1520_self_recovery/      [4.8 MB] Initial self-recovery pilot
├── 20261028_1740_mini_self_recovery/ [142 MB] 16-molecule mini self-recovery panel
├── 20261028_1753_native_self_recovery/ [405 MB] Native RDKit ETKDGv3 self-recovery
├── 20261028_2145_joint_calibration/  [709 MB] Full 512-drug panel joint calibration
├── 20261028_rl_preparation/          [524 KB] RL preflight checks & timing profiles
├── 20261029_0208_reward_cost/        [1.5 MB] Per-molecule scoring latency breakdowns
├── 20261029_0250_reward_calibration/ [1.7 GB] SUPERSEDED (RDKit seed multiplication bug)
└── 20261029_1155_reward_calibration/ [1.9 GB] CORRECTED independent-stream calibration
```

---

## 2. Reinforcement Learning Training Runs (`run16` – `run18`)

| Run Identifier | Training Configuration | Primary Observations | Key Takeaway for Thesis |
| :--- | :--- | :--- | :--- |
| **`run16_sequencer_weighted_sum`** | 50 epochs, unprompted SequenceRNN, linear weighted sum: $0.5 \cdot \text{ROCS} + 0.25 \cdot \text{SA} + 0.25 \cdot \text{QED}$. $\epsilon = 0.20$. | Generated valid molecules with high QED (>0.80) and SA (>0.75). ROCS Combo scores rose from 0.45 to ~0.95. | Established that multi-objective RL can successfully drive shape alignment without prompt conditioning. |
| **`run17_baseline_50ep`** | 50 epochs, baseline prior comparison with raw data stored in `raw_data/baseline_raw.csv.gz` and `reinforced_raw.csv.gz`. | Serves as the primary frozen-prior distribution control for measuring property shifts. | Provides the baseline distribution against which all reinforced agents are benchmarked. |
| **`run18_epsilon_zero_50ep`** | 50 epochs, pure REINFORCE policy gradient with **zero exploration noise** ($\epsilon = 0.0$). | Severe mode collapse occurred around epoch 12. The generator collapsed into outputting a narrow cluster of repetitive, low-entropy fragments. | **Proved that exploration noise ($\epsilon > 0$) is mandatory** in sequence RL to prevent premature mode collapse. |

---

## 3. Conformer Calibration & The 512-Drug Panel Saga

A central challenge in ligand-based design is establishing the optimal hyperparameter balance between 3D shape recovery and computational throughput.

```mermaid
xychart-beta
    title "Throughput vs Shape Recovery Trade-off (Dielectric Pocket ε = 6.5)"
    x-axis ["K=1", "K=5", "K=10 (Recommended)", "K=15", "K=20", "K=30"]
    y-axis "Score Recovery (%)" 80 --> 100
    line [82.5, 94.2, 98.4, 98.9, 99.3, 99.8]
```

### 3.1 `20261028_2145_joint_calibration` (Full 512-Drug Panel)
- **Scope**: Scored a diverse panel of 512 commercial drugs across varied rotatable bonds ($N_{\mathrm{rot}} \in [1, 15]$) and molecular weights ($MW \in [150, 750]$).
- **Verified Metrics**:
  - 509 scored molecules (3 native colorless exclusions, 13 preparation failures retained as zeros).
  - 18,432 measurement rows, 92,160 retained-cap reductions, 68,347 saved-geometry identity checks.
- **Key Finding**:
  - Target Reference Ensemble: $K_{\mathrm{ref}} = 3$ is optimal. Increasing $K_{\mathrm{ref}} > 3$ yields $<0.01$ improvement in shape recovery while multiplying pairwise alignment overhead.

### 3.2 The `20261029_0250` vs. `20261029_1155` Calibration Saga
- **`20261029_0250_reward_calibration` (1.7 GB, SUPERSEDED)**:
  - Intended to benchmark high candidate conformer budgets ($C50$, $C75$).
  - **Fatal Artifact Uncovered**: RDKit's conformer seed formula $\text{seed}(i) = \text{base} \times i$ caused reference conformer 42 (base 43) and candidate conformer 43 (base 42) to share seed **1806**, leaking identical 3D coordinates.
  - **Status**: Formally invalidated and superseded.
- **`20261029_1155_reward_calibration` (1.9 GB, CORRECTED & MAINTAINED)**:
  - Repaired with independent streams: Reference Base Seed = **104,729** (prime), Candidate Base Seed = **42**.
  - Verified 18,432 unique cells across 128 development drugs and 184,015 saved-geometry identity checks.
  - Replays confirmed zero coordinate leakage (maximum public score difference $< 1.12 \times 10^{-16}$).

---

## 4. Empirical Hyperparameter Scorecard

Based on the validated runs, the following hyperparameters are empirically justified for thesis implementation:

| Parameter | Recommended Setting | Physical / Computational Justification |
| :--- | :--- | :--- |
| **Reference Conformers ($K_{\mathrm{ref}}$)** | **3 conformers** | One-time target prep cost. Provides 98.5% shape coverage of bioactive pocket ensemble; $K_{\mathrm{ref}}>3$ flatlines. |
| **Candidate Conformers ($K_{\mathrm{gen}}$)** | **10 conformers** | Evaluated millions of times during RL. $K=10$ achieves **>98% shape recovery** at **3× higher throughput** than $K=30$. |
| **Candidate Minimization** | **FALSE (Embedding Only)** | Force-field minimization adds 15–20× per-molecule latency penalty with zero statistical improvement in ROCS ranking. |
| **Dielectric Model** | **$\epsilon_r = 6.5$ (Pocket)** | Electrostatic screening flattens intramolecular H-bonding wells without requiring full explicit water simulation. |
| **Stereoisomer Capping** | **`max_isomers = 4`** | Prevents exponential latency explosion on molecules with multiple undefined stereocenters. |
| **Heavy Atom Cap** | **45 heavy atoms** | Excludes macromolecules that trigger RDKit distance geometry embedding timeouts. |
| **Rotatable Bond Cap** | **15 rotatable bonds** | Eliminates ultra-flexible peptides that cause conformer sampling combinatorial explosion. |
