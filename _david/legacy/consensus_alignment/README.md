# Legacy Consensus Alignment & Actives Fetchers

This directory archives target-specific and multi-ligand consensus alignment utilities:
- `fetch_target_actives.py`: ChEMBL API querying tool to download active ligands for specific protein targets (e.g., DNMT1).
- `test_consensus_align.py`: Exploratory script for generating multi-conformer consensus alignments across multiple active ligands.

---

## Why Are They Archived?

1. **Scope Alignment**: Our Bachelor Thesis is centered on **ligand-based de novo drug design** seeded from a single bioactive query reference (e.g. EGCG) using Graph-Transformer reinforcement learning.
2. **Decoupling**: Target actives fetching and multi-ligand consensus pharmacophores represent tangential exploration that is not required for the primary Level 1–3 LBDD Ground Model Ladder.
