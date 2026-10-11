# Category 02: Reward Formulations & Cheating

**Research Question**: *Do generative agents game the reward function ("Pareto cheating"), and do custom soft-gates help or hurt policy learning compared to native DrugEx WeightedSum?*

---

## Controlled Differences Across Runs

| Run Name | Reward Formulation | Anti-Cheating Guard | Epochs | Empirical Takeaway |
| :--- | :--- | :--- | :---: | :--- |
| `20261007_123924 - EXTRA - M4_Empirical AntiCollapse_SoftGate (20)` | `AntiCollapseSoftGateReward`<br>($w_{3D}=0.70, w_{props}=0.30$) | Hard zero-mask: $S_{3D} \le 10^{-6} \implies R=0$<br>Soft-gate: $G(S_{3D}) = \min(1, S_{3D}/0.60)$ | 20 | Zero-mask eliminated policy gradients for early exploratory molecules, causing high training stall rates. |
| `20261010_031545 - 3D_ROCS SA e0.1 tau0.3 (15)` | Soft-gated with anti-cheating penalty | **Active**: Punished unaligned intermediate molecules | 15 | Halted early at epoch 15 due to drop-off; penalty restricted smooth exploration of transitional chemotypes. |
| `20261010_031545 - 3D_ROCS SA NoAntiCheating e0.1 tau0.3 (25)` | Upstream native **`WeightedSum()`** | **Disabled**: Zero heuristic penalty | 25 | **Key Finding**: Reached 25 epochs without policy collapse (`valid=100%`, `desired=0.255`, `amean=0.742`). Disproved that native DrugEx inherently cheats. |
| `20261009_131429 - 3D_ROCS SA Envelope Single_Ref_EGCG e0.1 tau0.3 (13)` | 7-Objective Gaussian Envelope | Soft-gate + zero-mask | 13 | High-dimensional Boolean conjunction ($P(D=1) \approx 0$) caused `desired_ratio = 0.0%`, triggering drop-off early stopping at epoch 13. |

---

## Key Scientific Findings
1. **Ad-Hoc Anti-Cheating Penalties Are Counter-Productive**: Zero-masking creates non-differentiable reward cliffs. Native `WeightedSum` naturally maintains policy stability without artificial guards.
2. **Conjunctive Desirability Bottlenecks**: Requiring all 7 physical properties to simultaneously pass $\tau=0.30$ kills the exploration signal. A focused 2-objective landscape (ROCS + SA) provides smooth, unhindered optimization.
