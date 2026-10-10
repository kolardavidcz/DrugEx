# Legacy Multi-Objective Anti-Cheating Reward Archive

This directory archives historical reward formulations and experimental heuristics developed during early exploration of Ligand-Based De Novo Drug Design (LBDD) in DrugEx.

---

## 1. What was `AntiCollapseSoftGateReward`?

`AntiCollapseSoftGateReward` was a custom non-linear multi-objective reinforcement learning (MORL) reward scheme designed to prevent policy collapse in generative Graph-Transformer agents.

### The Problem It Attempted to Solve ("Pareto Cheating")
In early runs, when optimizing multi-objective DrugEx environments using linear scalarization:
$$R_{\text{linear}}(\mathbf{x}) = \sum_{j=1}^M w_j S_j(\mathbf{x})$$

The policy gradient optimizer frequently discovered "adversarial shortcuts":
1. **Dimensional Cheating**: Small, trivial fragments (e.g., simple benzenes, phenols) effortlessly achieved high scores on secondary 1D/2D physicochemical properties ($S_{\text{props}} \approx 1.0$) while completely failing 3D template alignment ($S_{3D} \approx 0$). The agent received a high enough average reward that it never learned 3D pharmacophore alignment.
2. **Synthetic Collapse**: In pure 3D optimization without constraints, the agent generated bizarre, hyper-fused polycyclic cages or strained bridgeheads that filled 3D space but had un-synthesizable complexity ($\text{SA} > 6.0$).

### Mathematical Architecture
To resolve these without the zero-gradient cliffs of hard threshold cutoffs ($R = 0 \text{ if } S_{3D} < \tau$), `AntiCollapseSoftGateReward` introduced continuous soft gating:

$$R(\mathbf{x}) = w_{3D} \cdot S_{3D}(\mathbf{x}) + w_{\text{props}} \cdot G(S_{3D}(\mathbf{x})) \cdot \sum_i \tilde{w}_i S_{\text{prop}, i}(\mathbf{x}) + S_{\text{SA\_floor}}(\mathbf{x})$$

Where:
- **Soft Gating Function**:
  $$G(S_{3D}) = \min\left(1.0, \max\left(0.0, \frac{S_{3D}}{\tau_{3D}}\right)\right) \quad (\tau_{3D} = 0.60)$$
  Secondary properties were throttled unless the molecule demonstrated meaningful 3D bioactive template overlap.
- **Strict Anti-Cheating Zero Masking**:
  $$\text{If } S_{3D}(\mathbf{x}) \le 10^{-6} \implies R(\mathbf{x}) = 0.0$$
- **Synthesizability Feasibility Floor**:
  $$S_{\text{SA\_floor}}(\text{SA}) = \begin{cases} 
  1.0 & \text{if } \text{SA} \le 5.5 \\
  \frac{6.0 - \text{SA}}{6.0 - 5.5} & \text{if } 5.5 < \text{SA} < 6.0 \\
  0.0 & \text{if } \text{SA} \ge 6.0 
  \end{cases}$$

---

## 2. Why Are We No Longer Using It?

### Root Cause Analysis: A Hyperparameter Artifact
Through systematic hyperparameter ablation and empirical testing, we discovered that **the early policy collapse was NOT caused by fundamental flaws in standard multi-objective reward scalarization**.

Instead, the collapse was an artifact of **severely misconfigured hyperparameters**:
1. **Zero Exploration ($\epsilon = 0.0$)**: The Prior mutation rate was inactive, trapping the agent in local mode collapse. Once $\epsilon = 0.20$ was enabled, regularized sampling from the frozen Prior naturally diversified the policy.
2. **Improper Early Stopping Patience**: Stagnant checkpoints were prematurely selected or frozen before the agent discovered the 3D alignment manifold.
3. **Miscalibrated Batch Sizes & Learning Rates**: Policy gradient variance in small mini-batches caused gradient instability.

### Scientific & Practical Justification for Deprecation:
- **Occam's Razor**: Once hyperparameters are calibrated ($\epsilon = 0.20$, patience $\ge 15$, 200 samples/epoch), DrugEx's native multi-objective reward schemes—specifically **`WeightedSum`** (dynamic failure-rate weighting) and **`ParetoCrowdingDistance`** (NSGA-II crowding distance)—operate stably without collapsing.
- **Elimination of Arbitrary Hyperparameters**: `AntiCollapseSoftGateReward` introduced multiple ad-hoc parameters ($\tau_{3D} = 0.60$, $\text{SA}_{\text{floor}} = 5.5$, $\text{SA}_{\text{ceil}} = 6.0$, $w_{3D} = 0.70$) that risked overfitting to specific reference molecules.
- **Thesis Defensibility**: Standard DrugEx benchmarks use native algorithms. Demonstrating that standard Pareto or WeightedSum formulations succeed with properly calibrated exploration rates is scientifically stronger than relying on an ad-hoc custom gating patch.

---

## 3. Archived Components

- [`anti_cheating_rewards.py`](file:///home/kolarv/DrugEx/_david/legacy/anti_cheating_rewards.py): Full standalone implementation of `AntiCollapseSoftGateReward` preserved for historical reference and backward compatibility.
