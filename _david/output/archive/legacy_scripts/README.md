# Archived Legacy Scripts & Historical Code Milestones

This directory archives the chronological scripts developed throughout the thesis project prior to the current unified Graph-Transformer architecture. They are preserved for academic transparency and historical traceability.

---

## 1. Timeline of Code Development

| Script | Date Introduced | Milestone Role | Key Innovations & Methodological Limitations |
|---|---|---|---|
| [`1st_code_david.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/1st_code_david.py) | 2026-09-14 | **Milestone 0: 1st Code** | Initial exploration using pretrained Papyrus Sequence RNN (`Papyrus05.5_smiles_rnn_PT`). Established early transfer-learning workflows and target deconvolution. |
| [`1st_code_receptor_similar.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/1st_code_receptor_similar.py) | 2026-09-14 | **Target Deconvolution** | Automated ChEMBL bioactivity querying across target families, InChIKey normalization, and Papyrus pAffinity cross-referencing. |
| [`SequenceRNN.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/SequenceRNN.py) | 2026-09-27 | **1D SMILES Generator** | Trained 1D SMILES RNN agent against EGCG query. *Limitation*: Suffered from SMILES syntax invalidity, high repetition, and inability to build around fixed 3D Murcko scaffold cores. |
| [`scaffold_funnel.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/scaffold_funnel.py) | 2026-09-27 | **Multi-Stage Funnel** | Two-stage candidate selection funneling SMILES RNN hits into Graph-Transformer elaboration. *Limitation*: Relied on discrete step filtering that caused steep candidate attrition. |
| [`usrcat_scorer.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/usrcat_scorer.py) | 2026-10-01 | **1D Shape Moments** | 60-D USRCAT and 75-D ElectroShape moment scorers. *Limitation*: Suffered from the "degenerate sphere" artifact (matching radial distance moments without true 3D spatial alignment to EGCG's tri-aromatic galloyl ester). |
| [`visualize_molecular_space.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/visualize_molecular_space.py) | 2026-09-27 | **Chemical Space Manifolds** | 1,235-line visualization module computing t-SNE, PCA, UMAP, property radars, and chemical space coverage plots. Subsumed by `ResultAnalyzer`. |
| [`thesis_benchmark.py`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/legacy_scripts/thesis_benchmark.py) | 2026-09-27 | **Multi-Model Benchmark** | 828-line comparative benchmark suite comparing Sequence RNN, Graph Transformer, and Funnel across VUN and property metrics. |

---

## 2. Why They Were Superseded

1. **Graph-Transformer Superiority**: Graph-Transformer operates directly on molecular graph topologies with 100% SMILES validity, naturally attaching functional groups to predefined 3D seed scaffolds (`dummyMolsFromFragments`).
2. **Authentic RDKit ROCS**: Replaced 1D USRCAT/ElectroShape moment approximations with true Gaussian volume overlap (`RDKitROCSScorer` with `oe_mills_dean` color forcefield and `TanimotoCombo` 0–2 scale).
3. **Continuous Soft Gating**: Replaced fragile multi-stage funnels with continuous anti-collapse soft gating ($G(S_{\text{3D}}) \times \sum w_i S_{\text{prop}, i}$), eliminating Pareto cheating without early RL stalls.
