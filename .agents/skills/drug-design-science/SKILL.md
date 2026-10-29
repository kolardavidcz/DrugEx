---
name: drug-design-science
description: >-
  Unified scientific research and discovery meta-skill for RNN ligand-based de novo drug design (DrugEx).
  Orchestrates cheminformatics (ChEMBL, PubChem), structural target biology (PDB, UniProt, AlphaFold),
  3D molecular visualization (Mol*, PyMOL), 3D conformer generation and ML time-vs-accuracy calibration,
  and academic literature retrieval (Undermind MCP, Europe PMC, arXiv, bioRxiv, OpenAlex). Use whenever
  the user asks to profile ligands, mine target actives, fetch co-crystal structures, analyze 3D conformers,
  calibrate RL scoring budgets, or retrieve thesis-relevant machine learning and pharmacology publications.
---

# Computational Drug Design & Cheminformatics Meta-Skill

This meta-skill acts as the central orchestration layer for **RNN ligand-based de novo drug design** (DrugEx), focusing on polar reference systems (e.g. Epigallocatechin gallate / EGCG), epigenetic targets (e.g. DNMT1), and general polypharmacology benchmarks.

It unifies the active `/science` tools into four cohesive scientific pipelines while strictly filtering out non-relevant genomic and clinical noise.

---

## 1. Fast-Path CLI: `drug_design_suite.py`

A dedicated, universal Python CLI is provided in [`scripts/drug_design_suite.py`](./scripts/drug_design_suite.py) for rapid execution:

```bash
# 1. Resolve ligand name or SMILES and compute physicochemical envelope
python scripts/drug_design_suite.py profile --name "EGCG"
python scripts/drug_design_suite.py profile --smiles "c1ccccc1"

# 2. Fetch PDB metadata and download 3D coordinate files
python scripts/drug_design_suite.py pdb --id 3PTA --download ./pdb_structures

# 3. Mine target-specific bioactive chemical series from ChEMBL
python scripts/drug_design_suite.py chembl --target CHEMBL2016 --cutoff-nm 10000 --limit 50 --output dnmt1_actives.json

# 4. Search peer-reviewed literature and preprints via Europe PMC
python scripts/drug_design_suite.py literature --query "DrugEx de novo drug design" --limit 5
```

---

## 2. Core Scientific Pipelines

### Pipeline A: Target Protein & Binding Pocket Profiling
- **Tools**: `uniprot-database`, `pdb-database`, `biomolecular-structure-viewer`
- **When to Use**: When characterizing a biological target, identifying catalytic residues, or inspecting co-crystallized ligand interactions.
- **Runbook**: [Target Profiling & Structural Characterization](./references/target_profiling_runbook.md)
- **Sidecar Requirement**: Whenever loading or updating a 3D structure, always surface the one-click viewer pill:  
  `[Biomolecular Structure Viewer](sidecar://science/biomolecular-structure-viewer)`

### Pipeline B: Bioactive Ligand Mining & Prior Dataset Curation
- **Tools**: `chembl-database`, `pubchem-database`, RDKit
- **When to Use**: When mining known inhibitors ($IC_{50}, K_i \le 10\,\mu\text{M}$), generating training priors, or building benchmark test sets.
- **Runbook**: [Bioactive Ligand Mining & Prior Dataset Runbook](./references/active_ligand_mining_runbook.md)

### Pipeline C: 3D Conformer Generation & ML Time-vs-Accuracy Trade-Offs
- **Tools**: [`_david/code/chemistry.py`](file:///home/kolar/DrugEx/_david/code/chemistry.py), [`_david/code/reference.py`](file:///home/kolar/DrugEx/_david/code/reference.py), [`_david/code/rocs.py`](file:///home/kolar/DrugEx/_david/code/rocs.py), [`_david/code/conformer/`](file:///home/kolar/DrugEx/_david/code/conformer/), `pymol`
- **When to Use**: When preparing 3D reference conformers for ROCS shape/color alignment or calibrating candidate conformer budgets during DrugEx RL training.
- **Runbook**: [3D Conformer & Pharmacophore Alignment Runbook](./references/conformer_and_alignment_runbook.md)
- **ML Time-vs-Accuracy Principle (Matched Minimization & Basin Collapse)**:
  - **Reference vs. Candidate Asymmetry**: Reference ensemble preparation ($K_{\mathrm{ref}}$) is a **one-time setup cost** (~1–5 s before training), whereas candidate conformer generation ($K_{\mathrm{gen}}$) runs **tens of thousands of times** inside the RL loop ($N_{\mathrm{epochs}} \times N_{\mathrm{samples}}$).
  - **Never Mix Minimized References with Unminimized Candidates**: A force-field minimized reference (`MMFF94s`) paired with an unminimized `ETKDGv3` candidate suffers severe pharmacophoric (color) mismatch (EGCG `TanimotoCombo` drops to `0.765` at $C=20$ and plateaus at `0.833` even at $C=100$). Reference and candidate optimization states must be **matched**.
  - **Why Fewer Minimized Conformers Beat More Unminimized Conformers**: Under independent RNG streams (`104729` vs `42`), raw `ETKDGv3` draws scatter around local potential-energy basins, requiring huge candidate budgets ($C=50\text{–}75$) that multiply $\mathcal{O}(K_{\mathrm{ref}} \times C_{\mathrm{cand}})$ 3D overlay cost. Matched `MMFF94s` / `UFF` minimization collapses noisy draws into shared local minima: on the 128-drug panel, `MMFF94s` at $K=4, C=10$ (`0.889` mean, `1.000` median, `1.07 s`) and $K=4, C=20$ (`0.910` mean, `1.000` median, `1.88 s`) is **both more accurate and 2–3× faster** than unminimized `ETKDG` at $K=4, C=50$ (`0.882` mean, `2.86 s`) or $K=4, C=75$ (`0.892` mean, `4.04 s`), especially when parallelized across workers via `WholeCallScorer`.

### Pipeline D: Academic Literature, SOTA Discovery & Critical Triage
- **Engines**: **Undermind MCP** (Tier 1 AI-agentic deep research) + **Europe PMC / PubMed / arXiv / bioRxiv / OpenAlex** (Tier 2 unmetered fallback).
- **When to Use**: When researching frontier generative ML algorithms, benchmarking baseline models (DrugEx v1–v3, REINVENT), exporting BibTeX citations, or triaging published claims for ML training feasibility.
- **Runbooks**: [Academic Literature & SOTA Discovery Runbook](./references/literature_retrieval_runbook.md) | [Undermind MCP Guide](./references/undermind.md)

---

## 3. Exclusion Matrix (Noise Filtering)

To prevent context window bloat, do **NOT** invoke these unrelated `/science` skills for this thesis:
- `predictingthepast` (Ancient Latin/Greek epigraphy)
- `alphagenome-*` (Genomic variant effect, non-coding GWAS)
- `dbsnp-database`, `clinvar-database`, `gnomad-database` (Human SNP population genetics)
- `clinical-trials-database`, `openfda-database` (Clinical trial operations & FDA adverse events)
- `jaspar-database`, `unibind-database` (Transcription factor DNA binding motifs)
- `encode-ccres-database`, `gtex-database` (Tissue RNA expression)
