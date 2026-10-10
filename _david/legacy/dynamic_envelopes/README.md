# Legacy Dynamic Property Envelopes

This directory archives the dynamic physicochemical property envelope calculation framework (`DynamicPropertyEnvelope`, `compute_dynamic_envelope`).

---

## 1. What was the Dynamic Property Envelope?

The dynamic envelope system automatically scanned a bioactive query template (e.g. EGCG or target actives) and computed empirical tolerance intervals around its 1D/2D physicochemical descriptors:
- Molecular Weight ($\text{MW} \pm 50$)
- Lipophilicity ($\log P \pm 1.0$)
- Polar Surface Area ($\text{TPSA} \pm 25\,\text{Å}^2$)
- Heavy Atom Count ($\text{HAC} \pm 4$)
- Hydrogen Bond Donors/Acceptors ($\text{HBD}, \text{HBA}$)

Each descriptor was converted into a DrugEx `Property` scorer wrapped in a `ClippedScore` or `Gaussian` modifier, establishing an empirical bounding box.

---

## 2. Why Are We No Longer Using It?

1. **Over-Constraint Bottleneck (Agent Paralysis)**:
   In multi-objective RL, multiplying or combining 5+ strict thresholded objectives with primary 3D ROCS alignment severely shrinks the viable optimization manifold. The agent frequently generated molecules with excellent 3D pharmacophore alignment that were rejected simply because $\text{MW}$ or $\text{TPSA}$ fell slightly outside the narrow empirical envelope.
2. **Artificial Threshold Dropouts**:
   Setting arbitrary thresholds ($0.25$) on secondary descriptors created artificial failure cliffs.
3. **Better Alternative in Level 3 (Continuous Positive Feedback)**:
   Instead of rigid, thresholded property envelopes that penalize exploration, our thesis Level 3 architecture uses **continuous, positive-feedback scoring** (e.g., QED in $[0.0, 1.0]$ and smooth unconstrained descriptors): if a molecule exhibits good drug-likeness, it receives an additive reward boost; if not, it is not disqualified from learning 3D alignment.
