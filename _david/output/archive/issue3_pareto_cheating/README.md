# Methodological Issue 3: Pareto Cheating & Conflicting Synthesizability Metrics

**Historical Run ID**: `20261001_130044`  
**Execution Mode**: Full 3-Stage Pipeline (Sequence-RNN -> Funnel -> Graph-Transformer)  
**Duration**: 1,833 seconds (~30.5 minutes)  
**Host Hardware**: `dichtator` (NVIDIA RTX PRO 4000 Blackwell GPU 0)

---

## 1. What This Experiment Attempted
This experiment attempted multi-objective lead generation by stacking multiple disparate property and synthesizability scorers together:
1. `RDKit_Aggregate_6refs_TanimotoCombo` (3D shape & color overlap)
2. `RAscore` (retrosynthetic accessibility AI model)
3. `QED` (quantitative estimate of drug-likeness)
4. `SA` (synthetic accessibility fragment complexity score)

Scored output from `stage3_graph_transformer/stage3_leads.tsv`:
```text
SMILES: CN(CC(=O)Nc1ccccc1)C(=O)Cc1c[nH]c2ccccc12
ROCS TanimotoCombo: 0.290  (Poor 3D match to EGCG)
RAscore:            0.987  (Extremely high)
QED:                0.930  (Extremely high)
SA:                 0.962  (Extremely high)
Desired:            0
```

---

## 2. Why It Failed Academically
1. **Pareto Front Cheating**:
   - In unconstrained multi-objective scalarization or standard Pareto ranking, the RL agent easily discovered that optimizing 2D descriptors (`QED` and `RAscore`) is far easier than generating complex 3D shape overlaps.
   - Small, flat, drug-like amide fragments (e.g. `CN(CC(=O)Nc1ccccc1)...`) scored $>0.95$ on all 2D metrics, dominating the scalar reward sum while completely ignoring 3D shape overlap ($ROCS \approx 0.29$).
2. **Conflicting Synthesizability Metrics**:
   - Stacking multiple synthesizability penalties (`RAscore` + `SAscore` + BRICS rules) created conflicting gradients. EGCG has a natural product complexity ($\text{SA} \approx 3.74$), but RAscore penalizes natural product stereocenters harshly ($\text{RA} \approx 0.15$). Stacking both scorers penalized legitimate bioisosteric scaffold hops.

---

## 3. Academic Resolution
In our current pipeline:
1. **Continuous Soft-Gated Multi-Objective Reward (`AntiCollapseSoftGateReward`)**:
   $$R(\mathbf{x}) = w_{\text{3D}} \cdot S_{\text{3D}}(\mathbf{x}) + w_{\text{props}} \cdot G(S_{\text{3D}}(\mathbf{x})) \cdot \sum_i \tilde{w}_i S_{\text{prop}, i}(\mathbf{x}) + S_{\text{SA\_floor}}(\mathbf{x})$$
   $$\text{where} \quad G(S_{\text{3D}}) = \min\left(1.0, \; \frac{S_{\text{3D}}}{\tau_{\text{3D}}}\right) \quad (\tau_{\text{3D}} = 0.60)$$
   - If $S_{\text{3D}} = 0$, the secondary property reward is strictly zero ($G(0) = 0$). The agent cannot harvest any reward for QED, MW, or LogP unless it satisfies the 3D shape gate.
2. **Single Synthesizability Floor**:
   - Strictly a **single** synthesizability scorer (SAscore) functioning as a feasibility floor: rewarding $1.0$ for $\text{SA} \le 5.5$, and penalizing only pathological nonsense ($\text{SA} > 6.0$). This eliminates conflicting penalties and accommodates natural product complexity without penalty.
