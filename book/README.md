# DrugEx Hub — De Novo Drug Design & ROCS Shape-Matching Workbench

> **High-density, interactive educational platform and computational workbench tailored for the Bachelor's Thesis on *De Novo Drug Design Combined with 3D Shape Matching (ROCS) for Flexible Targets and Intrinsically Disordered Proteins (IDPs)*.**
>
> **Institution**: VŠCHT Praha / ÚOCHB AV ČR (UCT Prague / IOCB Prague)  
> **Repository Baseline**: [`fulopjoz/DrugEx` (branch: `feature/rocs-scoring`)](https://github.com/fulopjoz/DrugEx/tree/feature/rocs-scoring) & [`CDDLeiden/DrugEx`](https://github.com/CDDLeiden/DrugEx)  
> **Local Server**: `http://127.0.0.1:34100/app/index.html`

---

## Highlights & Features

### 1. Modern VS Code Glassmorphic Dark UI & Explorer
- **Developer-Centric Design**: Dark mode glassmorphic UI matching VS Code developer tooling with zero visual clutter.
- **Tree Navigation**: 6 structured modules, 18 comprehensive lectures, 6 interactive Dojos, and 6 active recall tests.
- **Multi-Dimensional Filters**: Filter lectures by tags (`[Core]`, `[WOW]`, `[Legendary]`, `[Tricky]`) and search bar.

### 2. 3D WebGL Molecular & Shape Alignment Studio (3Dmol.js)
- **Interactive 3D Overlays**: Visualize 3D conformers, Gaussian shape density envelopes, and pharmacophore colors.
- **Multi-Conformer Inspection**: Inspect stereoisomer variations, RMSD alignment, and stick/sphere representations.

### 3. Interactive Dojos & Simulators
- **Dojo 1**: SMILES Tokenizer & Temperature-Scaled Softmax Sampling Simulator.
- **Dojo 2**: 2D/3D Multi-Objective Pareto Frontier Scatter Plotter & Desirability Modifier Sandbox (`SmoothClippedScore`).
- **Dojo 3**: 3D Conformer & Stereoisomer Alignment Explorer + ETKDGv3 Parameter Calibrator.
- **Dojo 4**: ROCS Scorer Code Generator & Multi-Reference Benchmark Matrix (RDKit vs CDPKit vs OpenEye).
- **Dojo 5**: Interactive ROC / PR Curve & Youden's $J$ Threshold Sandbox + RL Training Trajectory Playback.
- **Dojo 6**: Production CLI Command Builder & Slurm HPC GPU Batch Script Generator.

### 4. DrugEx Reference Documentation Platform (cppreference style)
- **Authentic Format**: Header synopsis, parameter descriptions, member functions, member variables, practical edge case notes, executable Python examples with stdout, and see-also cross-links.
- **Complete Coverage (234 Symbols)**: Comprehensive documentation across DrugEx Core (186 symbols: environments, generators, explorers, scorers, modifiers, data pipelines, parallel workers, high-level CLI workflows, models, collectors, exceptions, neural layers, logging system, cryptographic hashing), Chemoinformatics Ecosystem (24 symbols: RDKit descriptors, 2D depiction coordinates, QED, rotatable bonds, QSPRPred variance filtering, CDPKit basic molecule & analytical shape overlap), and Experimental techniques (24 symbols: USRCAT, ElectroShape, ProLIF interaction fingerprints, Chemprop D-MPNN datasets & uncertainty, Scaffviz, RAScore, Optuna MOTPE samplers, MMFF94 strain, AutoDock Vina receptor grids & flexible docking, OpenFE alchemical networks & chemical systems).
- **Zero-Dependency React 18 SPA**: Instant search, category tabs, dark/paper themes, and deep linking (`pnpm docs` on DevPort `34150` or direct `file://` opening).

### 5. Bachelor Thesis Companion & Protocol Exporter
- **Literature Review Hub**: Structured summaries and citations of key papers in de novo drug design, ROCS, and IDP targeting.
- **Computational Protocol Exporter**: Single-click export of the exact LaTeX / Markdown computational methods section.
- **Unified A4 Print Engine (`Ctrl+P`)**: Clean 2-column printable study sheets for laboratory binders.

---

## Quick Start & Local Execution

DrugEx Hub runs with zero heavy dependencies—just Python 3:

```bash
# 1. Start the local server on dedicated DevPort 34100
python3 serve.py 34100

# 2. Open in your browser:
# → http://127.0.0.1:34100/app/index.html
```

---

## Verification & Audit Suite

```bash
# Run syntax, WCAG 2.1 contrast, and quiz option equilibrium audits
pnpm test
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+P` | Print / Generate PDF A4 Study Sheet |
| `Ctrl+K` | Open Search & Command Palette |
| `Ctrl+B` | Expand / Collapse Explorer Sidebar |

---

## License & Credits

Created for the **Bachelor's Thesis at VŠCHT Praha / ÚOCHB AV ČR**. Built upon the algorithms of **DrugEx** (CDDLeiden / József Fülöp) and the UI/UX architecture of **Python Hub**.
