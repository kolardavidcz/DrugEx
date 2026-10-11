# Category 01: 3D Reference Representation

**Research Question**: *How should 3D bioactive reference templates be modeled for shape alignment, and what are the failure modes of point clouds vs. real conformer ensembles?*

---

## Controlled Differences Across Runs

| Run Name | Template Model | Code & Scorer Architecture | Epochs | Empirical Takeaway |
| :--- | :--- | :--- | :---: | :--- |
| `20261007_130035 - Benchmark_Only EGCG Conformer_Profile (0)` | Single EGCG Conformer | • Stage 0 conformer curation only (no RL)<br>• Initial ETKDGv3 + MMFF94 conformer baseline | 0 | Baseline coordinate reference and chemical property bounds for EGCG. |
| `20261009_025112 - EXTRA - 3D_ROCS SA Envelope Single_Ref_EGCG e0.1 tau0.3 (50)` | Single EGCG Conformer | • 3D ROCS against single rigid EGCG template<br>• 7-objective Gaussian envelope with boolean conjunction | 50 | High ROCS overlap ($0.88$), but constrained candidate diversity to single scaffold envelope. |
| `20261009_025112 - 3D_ROCS SA Envelope DNMT1_Ensemble e0.1 tau0.3 (50)` | Multi-Reference Ensemble | • Multi-reference ensemble across diverse DNMT1 active binders<br>• Aggregate ROCS alignment over multiple actives | 50 | Broadened shape tolerance ($ROCS=0.84$), enabling higher Murcko scaffold diversity than single EGCG. |
| `20261009_025112 - EXTRA - 3D_ROCS SA Envelope Consensus_Pharmacophore e0.1 tau0.3 (50)` | 23-Atom Point Cloud | • 23 spatial centroids derived from agglomerative clustering<br>• Virtual pseudo-molecule without covalent bonds | 50 | **Critical Negative Result**: ROCS shape alignment failed ($ROCS=0.47$). Point clouds lack van der Waals volumes, breaking Gaussian volume superposition. |

---

## Key Scientific Findings
1. **Whole-Molecule Conformer Ensembles Outperform Spatial Centroids**: RDKit ROCS requires continuous electron-density / van der Waals volume envelopes; point-cloud models without realistic molecular shape produce severe shape-matching degradation.
2. **Multi-Reference Ensembles Increase Diversity**: Aligning against an ensemble of diverse actives prevents the generator from overfitting to a single rigid chemical core.
