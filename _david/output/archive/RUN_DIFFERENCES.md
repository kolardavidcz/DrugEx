# Historical Run Differences & Code Evolution Across Git Milestones

This document highlights the **underlying code differences** across runs in `_david/output/`. Because the codebase developed over time in Git, runs that look similar on the surface used fundamentally different reward schemes, conformer curation algorithms, and convergence rules.

For detailed run-by-run comparisons within each research topic, see the respective category READMEs:
- [`01_reference_representation/README.md`](file:///home/kolarv/DrugEx/_david/output/01_reference_representation/README.md)
- [`02_reward_formulations_and_cheating/README.md`](file:///home/kolarv/DrugEx/_david/output/02_reward_formulations_and_cheating/README.md)
- [`03_hyperparameter_ablations/README.md`](file:///home/kolarv/DrugEx/_david/output/03_hyperparameter_ablations/README.md)
- [`04_flagship_lead_optimization/README.md`](file:///home/kolarv/DrugEx/_david/output/04_flagship_lead_optimization/README.md)

---

## Master Code Evolution Matrix

| Feature | Era 1: Early Prototypes (Oct 4–7)<br>`0e33179` $\to$ `1a552ad` | Era 2: 3 Hypotheses (Oct 9)<br>`658486b` | Era 3: Ablations Suite (Oct 10)<br>`0385808` $\to$ `1a12ad9` | Era 4: Native DrugEx (Oct 11)<br>`11c9b28` $\to$ `5694605` |
| :--- | :--- | :--- | :--- | :--- |
| **Category** | `02_reward_...` & `archive/` | `01_reference_representation` | `03_hyperparameter_ablations` | `04_flagship_lead_optimization` |
| **Objectives** | 2 (ROCS, SA) | 7 (ROCS, HAC, LogP, TPSA, HBD, HBA, SA) | 2 to 4 (ROCS, SA $\pm$ HAC, LogP) | **Pure 2** (ROCS, SA) |
| **Reward Scheme** | `AntiCollapseSoftGateReward`<br>Soft-gating $G(S_{3D}) = \min(1, S_{3D}/0.6)$ | Gaussian `SmoothHump` sum with soft-gating | Controlled: Soft-gate vs. native `WeightedSum` | Native **`WeightedSum(0.5, 0.5)`** |
| **Anti-Cheating** | Hard zero-mask: $S_{3D} \le 10^{-6} \implies R=0$ | Hard zero-mask: $S_{3D} \le 10^{-6} \implies R=0$ | Systematically ablated (Study 1 ON vs Study 2 OFF) | **Zero heuristic penalty** |
| **Desirability** | Piecewise floor on SAscore ($\le 5.5$) | Boolean conjunction: $\prod_{i=1}^7 \mathbb{I}(S_i \ge 0.30)$ | Boolean conjunction + drop-off | Native Pareto ranking & dominance |
| **Exploration ($\epsilon$)** | $\epsilon = 0.10$ | $\epsilon = 0.10$ | Swept: $\epsilon \in \{0.00, 0.05, 0.10, 0.20\}$ | $\epsilon = 0.00$ (greedy exploitation) |
| **Threshold ($\tau$)** | $\tau = 0.30$ | $\tau = 0.30$ | Swept: $\tau \in \{0.20, 0.30\}$ | Default $\tau = 0.25$ |
| **3D Conformer Filter**| Radius of Gyration ($R_g$) median | $R_g$ filtering | $R_g$ filtering | **Hawkins Butina RMSD** ($1.25\text{ \AA}$) |
| **Conformer Count** | Variable | Variable | Variable | **8 Curated Basins** (scaffold-aligned) |
| **Early Stopping** | Inactive | Active (`patience = 15`) | Active (halted Studies 3–6 at epoch 17) | **Disabled** (full 30-epoch Pareto tracking) |
| **Key Takeaway** | Zero-mask caused gradient cliffs | 7-dim conjunction led to 0% desired ratio | Disproved that native DrugEx cheats | Clean, stable multi-objective Pareto front |
