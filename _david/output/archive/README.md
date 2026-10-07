# Historical Output Archive & Methodological Evolution

This directory preserves authentic empirical data from earlier iterations of our Ligand-Based De Novo Drug Design (LBDD) pipeline. Each subfolder documents a critical methodological lesson and demonstrates why earlier approximations were discarded in favor of our current mathematically rigorous Graph-Transformer + authentic RDKit ROCS architecture.

---

## Overview of Archived Experiments

| Directory | Core Methodological Issue | Why It Failed / Was Discarded | Replaced By |
|---|---|---|---|
| [`issue1_usrcat_electroshape/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/issue1_usrcat_electroshape/README.md) | **1D Moment Invariant Approximation** | Suffered from the *degenerate sphere problem*: compact molecules matched centroid distance moments despite having 0% spatial pharmacophore alignment. | Native RDKit ROCS Gaussian volume and Mills-Dean color overlap (`score_type="TanimotoCombo"`, 0–2 scale). |
| [`issue2_fixed_smarts_cheating/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/issue2_fixed_smarts_cheating/README.md) | **Fixed SMARTS / Hardcoded Rules** | Manually penalizing catechols (`c1c([OH])c([OH])ccc1`) was tailored only to EGCG and destroyed pipeline universality for arbitrary query ligands. | Universal dynamic physicochemical envelopes (HAC, CLogP, TPSA) calibrated relative to any query seed. |
| [`issue3_pareto_cheating/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/issue3_pareto_cheating/README.md) | **Pareto Front Cheating & Multi-Scorer Conflict** | The agent maximized 2D properties (RAscore=0.987, QED=0.930, SA=0.962) while generating poor 3D shapes (ROCS=0.290), hacking the scalar reward. Stacking conflicting synthesizability penalties (RA + SA) unfairly penalized natural products. | Continuous soft-gating ($G(S_{\text{3D}}) \times \sum w_i S_{\text{prop}, i}$) eliminating Pareto cheating, paired with a single SA feasibility floor ($\text{SA} \le 5.5$). |
| [`historic_docs/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/archive/historic_docs/) | **Historical Documentation** | Preserves original exploratory notes (`dodo_notes.md`), architecture designs (`old_AGENTS.md`), and comprehensive HTML guide (`guide.html`). | Modern, streamlined `_david/AGENTS.md` and active codebase documentation. |

---

## Active Benchmark Runs (Root Output)
- **[`m4_empirical/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/m4_empirical/README.md)**: Full-scale 20-epoch Graph-Transformer RL run demonstrating true convergence, soft-gating anti-collapse stability, and peak desirability checkpoint recovery (`patience=10`).
- **[`m4_sanity/`](file:///home/kolar/build_projects/drugex_edu/repo/_david/output/m4_sanity/README.md)**: Fast 2-epoch sanity verification confirming end-to-end execution without crashes.
