# Legacy MORL Non-Linear Scalarizations

This directory archives experimental non-linear Multi-Objective Reinforcement Learning (MORL) scalarization functions developed during early reward ablation studies.

---

## 1. Archived Formulations

### A. Augmented Chebyshev Scalarization (`AugmentedChebyshevReward`)
In Multi-Objective Optimization (MOO) literature (Steuer 1986, Miettinen 1999), weighted linear summation cannot access non-convex regions of the Pareto frontier. Chebyshev scalarization minimizes the weighted $L_\infty$ distance to an ideal utopian reference point $\mathbf{z}^* = (1, 1, \dots, 1)$:

$$\text{Chebyshev}(\mathbf{x}) = \max_{j=1}^M \left\{ w_j \cdot (1.0 - s_j(\mathbf{x})) \right\} + \rho \sum_{j=1}^M w_j \cdot (1.0 - s_j(\mathbf{x}))$$

Because the $\max_j \{\cdot\}$ operator penalizes the candidate according to its **worst single dimension**, an agent cannot cheat by sacrificing 3D shape for secondary properties. The augmentation term $\rho \sum_j d_j$ ($\rho = 0.05$) guarantees strict Pareto efficiency.

### B. Geometric Mean / Nash Bargaining Reward (`GeometricMeanReward`)
A multiplicative aggregation rooted in Cobb-Douglas economics and Nash bargaining game theory:

$$R_{\text{geom}}(\mathbf{x}) = \left( \prod_{j=1}^M \max(\epsilon, s_j(\mathbf{x})) \right)^{1/M}$$

If a generated molecule completely fails any single objective ($s_j \approx 0$), the entire reward collapses toward zero, creating an intrinsic multi-objective bottleneck.

---

## 2. Why Are We No Longer Using Them?

1. **Optimization Instability**: In policy gradient reinforcement learning, non-linear min-max operators ($\max_j$) and multiplicative products ($\prod_j$) create non-smooth, non-convex gradient surfaces. Early in training, when policies produce low-scoring candidates across multiple dimensions, gradients become extremely steep or vanish, destabilizing policy updates.
2. **Occam's Razor**: DrugEx's native **`WeightedSum`** (which adaptively weights objectives based on threshold failure rates) and **`ParetoCrowdingDistance`** (NSGA-II crowding distance) achieve balanced multi-objective optimization with superior numerical stability and zero additional hyperparameters ($\rho, \epsilon$).
3. **Thesis Scope**: For a Bachelor Thesis, introducing custom non-linear MOO scalarizations dilutes the core contribution (3D ligand-based de novo design). Standard DrugEx reward algorithms provide a validated, recognized baseline.
