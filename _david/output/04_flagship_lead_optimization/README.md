# Category 04: Flagship Lead Optimization

**Research Question**: *What is the generative potential of native DrugEx when powered by an energy-curated multi-conformer 3D reference ensemble and full Pareto tracking without heuristic penalties or artificial drop-off?*

---

## Flagship Experimental Setup

| Feature | Configuration & Implementation |
| :--- | :--- |
| **Run Identifier** | `20261011_170620 - EXTRA - 3D_ROCS SA 8Refs e0.0 (30)` |
| **3D Reference Model** | **8 Curated Basins**: Hawkins Butina sphere-exclusion clustering ($1.25\text{ \AA}$ RMSD, $\Delta E \le 8.0\text{ kcal/mol}$), aligned on fused bicyclic ring anchor (`curated_references_8_aligned.sdf`). |
| **Optimization Objectives** | **Pure 2-Objective**: RDKit 3D ROCS (`oe_mills_dean`, `TanimotoCombo` on $[0, 2] \to [0, 1]$) + SAscore (`ClippedScore` $6.0 \to 5.5$). |
| **Scalarization Function** | Native upstream DrugEx **`WeightedSum(0.5, 0.5)`**; zero heuristic soft-gating or penalty zero-masking. |
| **Hyperparameters** | $\epsilon = 0.00$ (greedy exploitation), $\tau = 0.25$ (passing threshold), $\beta = 0.10$ (entropy regularization). |
| **Training Longevity** | **Full 30 Epochs**: Early stopping drop-off disabled to observe long-term policy dynamics across the complete Pareto landscape. |
| **Diagnostic Outputs** | • 30 per-epoch micro-Pareto distributions (`epoch_01.png` $\to$ `epoch_30.png`)<br>• Interactive scrubber widget (`pareto_epoch_viewer.html`)<br>• 6-panel progression grid (`pareto_epoch_evolution.png`)<br>• 9-panel property distribution (`property_distributions.png`)<br>• Lead structure cards & 2D depictions (`lead_structures_2d.png`, `top_prioritized_leads.tsv`) |

---

## Key Scientific Findings
1. **Convergence Without Premature Collapse**: Disabling early stopping drop-off permitted continuous policy refinement across all 30 epochs, reaching a high-density Pareto frontier ($ROCS > 0.85$, $SA \le 3.5$).
2. **Anchor-Aligned Ensembles Guide Realistic Morphologies**: Aligning conformer basins on the fused bicyclic ring anchor enforced consistent spatial orientation while allowing the generator to discover structurally diverse, synthetically accessible bioactive analogs.
