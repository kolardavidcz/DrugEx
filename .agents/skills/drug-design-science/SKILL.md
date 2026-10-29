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
- **ML Time-vs-Accuracy Principle (No Dogmatic Dielectric Rules)**:
  - **Reference vs. Candidate Asymmetry**: Reference ensemble preparation ($K_{\mathrm{ref}}$) is a **one-time setup cost** (~1–5 s before training), whereas candidate conformer generation ($K_{\mathrm{gen}}$) runs **tens of thousands of times** inside the RL loop ($N_{\mathrm{epochs}} \times N_{\mathrm{samples}}$).
  - **Force-Field Minimization Trade-Off**: Do not dogmatically mandate candidate MMFF94s minimization or specific dielectric constants ($\epsilon_r = 6.5$ or $78.4$) during RL without accounting for per-epoch latency. Unminimized ETKDGv3 (`minimize_candidates=False`) with $K_{\mathrm{ref}}=4, K_{\mathrm{gen}}=15$ achieves $\ge 0.955$ mean self-recovery on FDA drugs at ~3–5× higher throughput than candidate force-field minimization. Treat force-field variants and dielectric constants as empirical screening hyperparameters to be validated against wall-clock RL budget and downstream chemical quality.

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
