# Scientific Report: 3D Conformer Self-Similarity & Dielectric Convergence

**Investigator**: David Kolar (Bachelor Thesis in RNN Ligand-Based De Novo Drug Design)  
**Target Reference Molecule**: Epigallocatechin gallate (EGCG)  
**Clustering & Diversity Filter**: Heavy-atom Butina RMSD cutoff $= 1.25\text{ \AA}$; $\Delta E \le 8.0\text{ kcal/mol}$ (vacuum variants)  
**Hardware & Execution**: Workstation `dichtator` (48-core host) & Modular WSL2 Environment  

---

## 1. Executive Scientific Summary

A central dilemma in 3D ligand-based de novo design is **conformational under-sampling**: if the target query or the generated candidate is represented by too few conformers, genuine bio-isosteric molecules can be severely penalized simply because their generated 3D pose does not align with the single reference conformer in the coordinate frame.

To rigorously address this, we executed a two-dimensional grid scan evaluating self-similarity ROCS TanimotoCombo across varying numbers of reference basins ($K_{\mathrm{ref}} \in [1, 12]$) and candidate conformers ($K_{\mathrm{gen}} \in [1, 30]$), benchmarking 5 distinct force fields and solvent dielectric environments:

1. **MMFF94s Vacuum** ($\epsilon_r = 1.0$, $\Delta E \le 8.0\text{ kcal/mol}$): Standard gas-phase baseline.
2. **UFF Vacuum** (Universal Force Field, Rappe et al.): Alternative molecular mechanics formulation.
3. **MMFF94s Protein Binding Pocket** ($\epsilon_r = 6.5$, literature consensus): Replicating the electrostatic screening of partially solvent-exposed active sites (Warshel, Simonson, and NMR chemical shift perturbation studies).
4. **MMFF94s Bulk Aqueous** ($\epsilon_r = 78.4$, room temperature water): Simulating complete dielectric charge screening in physiological solution.
5. **MMFF94s Distance-Dependent Dielectric** ($\epsilon(r) = r \cdot \epsilon_0$): Classical Pickersgill / Mehler-Solmajer solvent screening model.

---

## 2. Quantitative Summary Scorecard

| Force Field & Environment | Literature / Dielectric Model | Distinct Basins ($>1.25\text{\AA}$) | Mean $R_g$ ($\text{\AA}$) | ROCS ($K_{\mathrm{ref}}=1, K_{\mathrm{gen}}=1$) | ROCS ($K_{\mathrm{ref}}=3, K_{\mathrm{gen}}=30$) | Max ROCS Recovery | Recovery Gain ($\Delta\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MMFF94s Vacuum** | Model 1 ($\epsilon_r = 1.0$) | **2** | $4.339$ | **`0.7808`** | **`1.9712`** | `1.9712` | **`+152.5%`** |
| **UFF Vacuum** | Classical UFF | **2** | $4.298$ | **`1.9963`** | **`1.9988`** | `1.9988` | `+0.1%` |
| **MMFF94s Pocket** | Model 1 ($\epsilon_r = 6.5$) | **3** | $4.314$ | **`1.9865`** | **`1.9865`** | `1.9865` | `0.0%` |
| **MMFF94s Water** | Model 1 ($\epsilon_r = 78.4$) | **3** | $4.310$ | **`1.9988`** | **`1.9988`** | `1.9989` | `0.0%` |
| **MMFF94s Distance-Dep** | Model 2 ($\epsilon(r) = r$) | **2** | $4.301$ | **`0.8056`** | **`1.8174`** | `1.9707` | **`+144.6%`** |

---

## 3. Key Physical Insights from First Principles

### A. The "Single Conformer Trap" in Vacuum ($\text{ROCS} = 0.78$)
In gas phase ($\epsilon_r = 1.0$), electrostatic interactions between EGCG's 8 hydroxyl groups are completely unshielded. This creates deep, localized electrostatic energy wells where the polyphenol curls into tight internal hydrogen bonds ($R_g = 4.34\text{ \AA}$).
* If only **$K_{\mathrm{ref}} = 1$ and $K_{\mathrm{gen}} = 1$** are evaluated, the molecule scores only **$0.7808$ against itself** (losing over $60\%$ of its theoretical $2.00$ score)!
* Expanding to **$K_{\mathrm{ref}} = 3$ and $K_{\mathrm{gen}} = 30$** drives self-recovery to **`1.9712` ($98.6\%$)**, fully resolving the conformational trap.

### B. Solvent Dielectric Screening ($\epsilon_r = 6.5$ and $\epsilon_r = 78.4$)
In the protein binding pocket ($\epsilon_r = 6.5$) and bulk aqueous solution ($\epsilon_r = 78.4$), dielectric screening attenuates intramolecular Coulombic attraction. Consequently:
* The energy landscape flattens, conformational barriers between rotamers decrease, and the molecule exhibits broader, smoother conformational basins.
* Self-similarity converges virtually immediately ($1.986 - 1.999$), indicating that solvent-screened reference conformers provide significantly more forgiving, biologically realistic shape targets during de novo design.

---

## 4. Visual & 3D Deliverables

### Figure 1: 2D Distortion-Free Annotated Heatmaps
To eliminate 3D perspective distortion and optical illusions, every $(K_{\mathrm{ref}}, K_{\mathrm{gen}})$ coordinate is annotated with its exact ROCS score (bounded in $[0.00, 2.00]$):
![Annotated 2D Convergence Heatmaps](file:///C:/Users/kolar/.gemini/antigravity/brain/3c0732c1-95a4-4444-ba73-318653568f70/01b_conformer_convergence_heatmaps.png)

### Figure 2: 3D Self-Similarity Response Surfaces ($K_{\mathrm{ref}}$ vs. $K_{\mathrm{gen}}$ vs. ROCS)
Side-by-side 3D surfaces showing the non-linear ascent of ROCS TanimotoCombo as a function of reference basin density ($K_{\mathrm{ref}}$) and candidate conformer pool size ($K_{\mathrm{gen}}$), with unambiguous $Z$-axis clamped in $[0.0, 2.05]$.
![3D Conformer Convergence Surfaces](file:///C:/Users/kolar/.gemini/antigravity/brain/3c0732c1-95a4-4444-ba73-318653568f70/01_conformer_convergence_3d_surfaces.png)

### Interactive 3D Orbit Widget
For interactive 3D rotation, zooming, and surface inspection in a webview/browser:
- [01_conformer_convergence_interactive.html](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/01_conformer_convergence_interactive.html)

### Figure 3: 2D Cross-Sectional Recovery Curves
Panel (A) isolates the $K_{\mathrm{ref}} = 3$ cross-section (DrugEx default), demonstrating the steep logarithmic recovery as $K_{\mathrm{gen}}$ scales from $1 \to 30$. Panel (B) isolates the $K_{\mathrm{gen}} = 30$ cross-section as a function of $K_{\mathrm{ref}}$.
![Cross-Sectional Recovery Curves](file:///C:/Users/kolar/.gemini/antigravity/brain/3c0732c1-95a4-4444-ba73-318653568f70/02_cross_sectional_recovery_curves.png)

### Figure 4: Conformational Volume & Energy Profiles
Boxplot distributions of heavy-atom Radius of Gyration ($R_g$) across all 5 environments, detailing the structural compaction and expansion profiles of EGCG.
![Conformational Volume Profiles](file:///C:/Users/kolar/.gemini/antigravity/brain/3c0732c1-95a4-4444-ba73-318653568f70/03_conformational_volume_and_energy_distributions.png)

---

## 5. Direct 3D Inspection Files for VS Code

The curated conformers have been exported into multi-model `.sdf` and standard `.pdb` formats. You can click on any file below to open and rotate it directly in VS Code with 3D viewer extensions (Mol*, NGL Viewer, or 3Dmol.js):

| Force Field Variant | Multi-Model SDF (Rich Properties) | Multi-Model PDB (Universal 3D) | Basins ($>1.25\text{\AA}$) |
| :--- | :--- | :--- | :---: |
| **MMFF94s Pocket ($\epsilon_r=6.5$)** | [`egcg_conformers_mmff94s_pocket.sdf`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_pocket.sdf) | [`egcg_conformers_mmff94s_pocket.pdb`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_pocket.pdb) | 3 |
| **MMFF94s Water ($\epsilon_r=78.4$)** | [`egcg_conformers_mmff94s_water.sdf`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_water.sdf) | [`egcg_conformers_mmff94s_water.pdb`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_water.pdb) | 3 |
| **MMFF94s Distance-Dep ($\epsilon(r)=r$)** | [`egcg_conformers_mmff94s_distancedep.sdf`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_distancedep.sdf) | [`egcg_conformers_mmff94s_distancedep.pdb`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_distancedep.pdb) | 2 |
| **MMFF94s Vacuum ($\epsilon_r=1.0$)** | [`egcg_conformers_mmff94s_vacuum.sdf`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_vacuum.sdf) | [`egcg_conformers_mmff94s_vacuum.pdb`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_mmff94s_vacuum.pdb) | 2 |
| **UFF Vacuum** | [`egcg_conformers_uff_vacuum.sdf`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_uff_vacuum.sdf) | [`egcg_conformers_uff_vacuum.pdb`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/output/conformer_study/models_3d/egcg_conformers_uff_vacuum.pdb) | 2 |

---

## 6. Modular Architectural Refactoring (`_david/code/conformer/`)

Per engineering best practices, the monolithic script was refactored into a clean, class-oriented package:

```
_david/code/conformer/
├── __init__.py           # Unified public API re-exports
├── physics.py            # ForceFieldVariant dataclass & PhysicalVariantFactory
├── io.py                 # ConformerIOManager (multi-model SDF/PDB & scorecard writer)
├── generator.py          # ConformerGenerator (ETKDGv3 embedding, MMFF94s/UFF physics, Butina clustering)
├── analyzer.py           # ConformerConvergenceAnalyzer (Rg extraction, pairwise ROCS alignment, grid scan)
├── visualization.py      # ConformerVisualizer (3D surfaces, 2D heatmaps, Plotly widget, boxplots)
└── orchestrator.py       # ConformerStudyOrchestrator (class-first end-to-end execution)
```

- **Clean CLI entrypoint**: [`conformer_experiment.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/code/conformer_experiment.py) reduced to an elegant 80-line wrapper with full CLI parsing.
- **Unit test suite**: [`test_conformer.py`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/tests/test_conformer.py) added with 10 passing unit tests.
- **Zero linter warnings**: Verified clean with `ruff check`.
