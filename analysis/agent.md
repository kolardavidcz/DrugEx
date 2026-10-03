# AGENT.md — Analysis Subsystem Architecture & Maintenance Guide

> **Directory**: `drugex_edu/analysis/`  
> **Mission**: Chemical space manifold visualization, conformer generation, and 3D ROCS shape/color similarity scoring for Bachelor's Thesis on *Ligand-Based De Novo Drug Design in DrugEx*.  
> **Target Audience**: 100% for AI agents and maintainers developing, debugging, and extending molecular space analyses.  
> **Execution Context**: Local WSL2 (Ubuntu) CPU execution (`~/build_projects/drugex_edu/.venv`), completely independent of remote SSH sessions.  
> **Git Tracking**: Managed by dedicated repository history agents. **Do not run ad-hoc `git commit` or `git add` from within this folder.**

---

## 1. Scientific & Academic Standards (Thesis Strictness)

This analysis subsystem generates benchmark figures and interactive inspection tools for a **Bachelor's Thesis on Ligand-Based De Novo Drug Design in DrugEx**. All work in this folder must adhere to bulletproof scientific rigor:

1. **Academic Scrutiny**: Every metric, distribution, alignment algorithm, and threshold must be mathematically verified and defensible before academic evaluators.
2. **Library Precedence**:
   - Check **DrugEx** built-in implementations first (`drugex.training.scorers.rocs_rdkit`).
   - Check standard cheminformatics standards second (**RDKit** `rdShapeAlign.AlignMol`).
   - For external packages, explicitly explain: `[PACKAGE NAME: WHY? WHAT?]`.
3. **Reproducibility**: All conformer embedding and manifold methods use fixed seeds (`RANDOM_STATE = 42`). Zero stochastic drift between runs.
4. **Root Cause Integrity**: Never mask errors with silent fallbacks, dummy zeros, or commented assertions. If conformer generation or alignment fails for a molecule, log the failure explicitly and triage the root chemical cause (e.g. invalid valence, undefined stereocenters, hypervalent sulfur/phosphorus).

---

## 2. Directory Layout & Artifact Inventory

```
drugex_edu/analysis/
├── agent.md                                                  # This maintenance and architecture guide
├── AGENTS.md                                                 # Symlink pointing to agent.md
├── visualize_tanimoto_combo.py                               # Production CPU alignment & rendering engine
├── v1_space_raw.csv                                          # Raw coordinates + metadata for Version 1 (Papyrus-Only)
├── v2_space_raw.csv                                          # Raw coordinates + metadata for Version 2 (Papyrus + ChEMBL)
├── molecular_space_v1_papyrus_tanimotocombo.csv              # Fully scored V1 dataset (2D Sim + 3D Shape + 3D Color + Combo)
├── molecular_space_v1_papyrus_tanimotocombo.png              # Publication-grade 300 DPI Matplotlib figure (V1)
├── molecular_space_v1_papyrus_tanimotocombo.html             # Interactive dual-filter Plotly dashboard (V1)
├── molecular_space_v2_papyrus_chembl_tanimotocombo.csv       # Fully scored V2 dataset (2D Sim + 3D Shape + 3D Color + Combo)
├── molecular_space_v2_papyrus_chembl_tanimotocombo.png       # Publication-grade 300 DPI Matplotlib figure (V2)
└── molecular_space_v2_papyrus_chembl_tanimotocombo.html      # Interactive dual-filter Plotly dashboard (V2)
```

---

## 3. Core Chemical & Mathematical Formulations

### Reference Query Molecule (EGCG / `my_molecule`)
- **Canonical SMILES**: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`
- **Role**: Reference bioactive polyphenol / allosteric ligand used as the single seed query.
- **Reference Self-Scores**: `ShapeTanimoto = 1.0000`, `ColorTanimoto = 1.0000`, `TanimotoCombo = 2.0000`, `2D Sim = 1.0000`.

### ROCS 3D Alignment & `TanimotoCombo`
Alignment uses RDKit's native Gaussian shape overlay implementation (`rdkit.Chem.rdShapeAlign.AlignMol`), mathematically equivalent to OpenEye ROCS and CDPKit:
- **Color Matching**: `useColors = True` (activates pharmacophore color features).
- **Optimization Parameter**: `opt_param = 0.5` (equal weighting between shape and pharmacophore color overlap during Nelder-Mead optimization, equivalent to OpenEye ROCS `-optchem` mode).
- **Metric Formulation**:
  $$\text{TanimotoCombo} = \text{ShapeTanimoto} + \text{ColorTanimoto} \in [0.0, 2.0]$$
  - $\text{ShapeTanimoto} = \frac{O_{AB}}{\text{Self}_A + \text{Self}_B - O_{AB}} \in [0.0, 1.0]$
  - $\text{ColorTanimoto} = \frac{C_{AB}}{\text{ColorSelf}_A + \text{ColorSelf}_B - C_{AB}} \in [0.0, 1.0]$
- **DrugEx RL Benchmark Threshold**: `ROCS_THRESHOLD = 0.80` in `drugex/training/scorers/rocs_rdkit.py` and `GraphTransformer.py`.

### Empirical Distribution Realities (Crucial for Presets & Filters)
Never set arbitrary cutoffs without verifying the empirical score quantiles of the chemical space:

| Metric | V1 (Papyrus-Only) | V2 (Papyrus + ChEMBL) | Rationale & Interpretation |
|---|---|---|---|
| **2D Bioactivity Sim (Mean)** | 0.2668 | 0.3280 | Soft profile similarity across shared targets |
| **2D Bioactivity Sim (90th %ile)** | **0.3143** | **0.3724** | *Note*: Setting cutoffs $> 0.35$ eliminates $\sim 95\%$ of candidates |
| **TanimotoCombo (Min / Max)** | 0.2998 / **0.9336** | 0.3820 / **0.8431** | Max candidate is $\sim 0.93$; only the reference reaches $2.000$ |
| **TanimotoCombo (Median / p50)** | 0.5732 | 0.5998 | Half of candidate library is $\ge 0.57$ |
| **TanimotoCombo (Top 25% / p75)** | 0.6350 | 0.6478 | Upper quartile 3D matches |
| **TanimotoCombo (Top 10% / p90)** | 0.6881 | 0.6961 | Decile elite 3D matches |
| **Count $\ge 0.80$ (DrugEx ROCS)** | 8 molecules | 7 molecules | Elite multi-feature matches |

---

## 4. Visual Encodings & UI Specifications

### Dual-Channel Information Encoding
1. **Marker Color (2D Bioactivity Profile Similarity)**:
   - Colormap: Vibrant Blue (`#FFFFFF` $\rightarrow$ `#E1EDFE` $\rightarrow$ `#85B6FF` $\rightarrow$ `#1B64F2` $\rightarrow$ `#001A66`).
   - Highest score ($1.00$) = Deep Sapphire Blue (`#001A66`); lowest = Pure White (`#FFFFFF`).
2. **Marker Size (3D TanimotoCombo Similarity)**:
   - Matplotlib: $s = 22.0 + 220.0 \times (s_{\text{combo}} / 1.0)^{1.6}$
   - Plotly: $\text{size} = 6.0 + 22.0 \times (s_{\text{combo}} / 1.0)^{1.6}$
   - Larger markers represent molecules with superior 3D shape and pharmacophore alignment.
3. **Background Trace (Chemical Universe)**:
   - 1,000 neutral gray points (`#A0AEC0`, alpha $0.45$, size $5.5$) representing unselected random molecules from Papyrus.
4. **Reference Query (EGCG)**:
   - Distinct Ruby Star marker (`#E63946`, size $450$ in PNG, size $26$ in Plotly, black border $2.2$ px) with offset callout arrow (`xytext=(-160, 22)`) avoiding colorbar collision.

### Interactive Dashboard Architecture (`.html`)
The interactive HTML dashboards contain client-side JavaScript filtering with instant response:
- **Dynamic `.btn-active` State Management**:
  ```javascript
  function setActivePreset(activeBtn) {
      document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('btn-active'));
      if (activeBtn) activeBtn.classList.add('btn-active');
  }
  ```
- **Slider Sync**: Moving sliders manually resets `.btn-active` so the user is never misled by outdated active badges.
- **Calibrated Presets**:
  - `Show All`: Reset cutoffs to $0.00$ ($1,000 / 1,000$, $100\%$).
  - `Combo ≥ p50 (Top 50%)`: Median split ($\sim 500$ molecules).
  - `Combo ≥ p75 (Top 25%)`: Upper quartile ($\sim 250$ molecules).
  - `Combo ≥ p90 (Top 10%)`: Decile elite ($\sim 100$ molecules).
  - `Top 100 3D Matches`: Dynamically derives 100th highest score, showing **exactly 100 molecules**.
  - `Dual 2D+3D Hits`: Filters by 75th percentile of both 2D bioactivity profile and 3D TanimotoCombo.
  - `Reset`: Returns to `Show All`.

---

## 5. Coding Conventions & Maintenance Rules

When modifying scripts in this folder:

1. **Parameter-Only Type Annotations**:
   - Annotate function parameters: `def func(param: Type):`
   - **Do NOT** include return types (`-> ReturnType`) or class attribute annotations in pipeline scripts (per `_david/AGENTS.md`).
2. **Zero In-Code Docstrings**:
   - Keep production scripts dense, direct, and uncluttered. Document architecture and math here in `agent.md`.
3. **Uppercase Constants at Top**:
   - Hyperparameters (`SEED_SMILES`, `ROCS_USE_COLORS`, `ROCS_OPT_PARAM`, `RANDOM_STATE`, `N_JOBS`) live at the top of the file.
4. **Module-First, Class-Second Imports**:
   - `import rdkit.Chem as Chem` / `from rdkit.Chem import AllChem, rdShapeAlign`.
5. **Direct File Editing Tools Only**:
   - Always edit via `write_to_file` or `replace_file_content`. Never run ad-hoc scripts to mutate files.
6. **No Git Pollution**:
   - Do **NOT** run `git add` or `git commit` from this workspace. Git history and staging are managed by the repository history agent.

---

## 6. Execution & Verification Runbook

### Environment Setup & Execution
Run entirely within the local WSL2 Python virtual environment:

```bash
# 1. Activate environment
source ~/build_projects/drugex_edu/.venv/bin/activate

# 2. Verify dependencies
python -c "import rdkit, plotly, matplotlib, pandas, numpy, scipy; print('Environment OK')"

# 3. Run TanimotoCombo calculation & rendering
python ~/build_projects/drugex_edu/analysis/visualize_tanimoto_combo.py
```

### Empirical Verification Checklist
Before concluding any task involving this analysis folder:
- [ ] **Data Integrity**: Verify `scores_combo` in output CSVs ranges between $0.30$ and $0.94$ for candidates and equals $2.000$ for EGCG query.
- [ ] **No Zero-Score Overwrites**: Ensure background points remain $0.0$ and candidates have non-zero, valid floats.
- [ ] **Visual Layout**: Inspect rendered PNGs at 300 DPI — verify query annotation box does not overlap the colorbar or plot border.
- [ ] **Dashboard Responsiveness**: Verify via Chrome DevTools MCP that clicking each preset dynamically updates visible molecule count, swaps `.btn-active`, and alters Plotly scatter points.
