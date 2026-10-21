# Category 03: Hyperparameter Ablations

**Research Question**: *How sensitive is the Graph-Transformer policy to prior mutation noise ($\epsilon$) and passing threshold stringency ($\tau$), and how does early stopping interact with training longevity?*

---

## Controlled Experimental Matrix

All 6 runs in this category isolated either the prior exploration rate $\epsilon$ or the desirability threshold $\tau$, with an early stopping drop-off active (`patience = 15`):

| Run Name | $\epsilon$ (Prior Noise) | $\tau$ (Threshold) | Actual Epochs | Key Difference & Outcome |
| :--- | :---: | :---: | :---: | :--- |
| `20261010_031545 - 3D_ROCS SA Envelope e0.0 tau0.3 (17)` | **0.00** | 0.30 | 17 | Pure greedy exploitation. Rapid initial convergence, early plateau (halted at epoch 17). |
| `20261010_031545 - 3D_ROCS SA Envelope e0.05 tau0.3 (17)` | **0.05** | 0.30 | 17 | Minimal mutation noise. Balanced exploration without mode collapse, but halted at epoch 17. |
| `20261010_031545 - 3D_ROCS SA Envelope e0.1 tau0.3 (17)` | **0.10** | 0.30 | 17 | Standard DrugEx baseline exploration. Halted at epoch 17. |
| `20261010_031545 - 3D_ROCS SA Envelope e0.1 tau0.2 (17)` | 0.10 | **0.20** | 17 | Permissive threshold test. Higher candidate acceptance rate, lower selection stringency. |
| `20261010_121416 - 3D_ROCS SA e0.2 tau0.3 (34)` | **0.20** | 0.30 | 34 | Elevated 20% prior mutation. Prevented premature plateau, doubling training lifespan to 34 epochs. |
| `20261010_121416 - 3D_ROCS SA Envelope e0.2 tau0.2 (22)` | **0.20** | **0.20** | 22 | Elevated noise + permissive threshold. Explored broad chemical space for 22 epochs before drop-off. |

---

## Key Scientific Findings
1. **The "Epoch 17" Artifact Explained**: In runs with $\epsilon \le 0.10$, peak desirability occurred at epoch 2. With `patience = 15`, early stopping triggered precisely at epoch $2 + 15 = 17$.
2. **Prior Exploration Noise Extends Effective Training**: Increasing $\epsilon$ to $0.20$ infused continuous topological variety from the pre-trained prior, preventing early plateau and extending productive policy learning to 34 epochs.
