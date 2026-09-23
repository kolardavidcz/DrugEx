# DrugEx Reference Platform — C++ & Chemoinformatics Documentation

> **High-performance, interactive documentation and API reference platform styled after [cppreference.com](https://cppreference.com), covering 234 symbols across DrugEx Core, Chemoinformatics Ecosystem (RDKit, CDPKit, QSPRPred), and Experimental Molecular Techniques.**
>
> **Institution**: VŠCHT Praha / ÚOCHB AV ČR (UCT Prague / IOCB Prague)  
> **Deployment**: Optimized for zero-configuration static deployment on **Vercel**  
> **Local Dev Server**: `http://localhost:34150` (DevPort assigned block)

---

## Highlights & Experience

### 1. Concentrated Multi-Column Library Index
- **cppreference.com Aesthetic**: High-density 3-column reference matrix categorizing 234 symbols across 10 core thematic libraries.
- **Immediate Visual Scannability**: Distinct typographic badges indicating hardware acceleration (`GPU`), algorithmic dimensionality (`60D`, `4D`), framework origin (`RDKit`, `CDPKit`), and release tiers (`v3.4`).

### 2. Instant Zero-Build React 18 Single-Page Architecture
- **Zero-Build Deployment**: Self-contained React 18 SPA with Babel standalone runtime. No bundler lock-in, zero build-step overhead, runs instantaneously in any modern browser or CDN.
- **Synchronized Deep-Linking**: Real-time URL hash routing (`#symbol`) with complete browser history support (Back/Forward navigation, bookmarking, and link sharing).
- **Sub-Millisecond Search**: Client-side full-text search across titles, module paths, synopsis signatures, and full docstring descriptions.
- **Dynamic Theming**: One-click toggling between deep dark developer mode and authentic high-contrast paper reading themes.

### 3. Comprehensive Reference Coverage (234 Symbols)
- **DrugEx Core (186 symbols)**: Environments, generators (RNN, SequenceTransformer, GraphTransformer), explorers, scorers, modifiers, data pipelines, parallel workers, high-level CLI workflows, models, collectors, exceptions, neural layers, logging system, cryptographic hashing.
- **Chemo Ecosystem (24 symbols)**: RDKit 2D/3D descriptors, coordinates, QED, rotatable bonds, QSPRPred variance filtering, and CDPKit basic molecule & analytical shape overlap.
- **Experimental Techniques (24 symbols)**: USRCAT 60D shape-moments, ElectroShape 4D electrostatics, ProLIF protein-ligand interaction fingerprints, Chemprop D-MPNN architectures, Scaffviz scaffold hopping, RAScore synthetic accessibility, MMFF94 force-field strain, AutoDock Vina receptor grids, and OpenFE alchemical free energy networks.

### 4. Interactive Pipeline Schemas & Architecture Visualizer
- **5 Core DrugEx Workflows (Visual Diagrams & Algebraic Chaining)**:
  1. **Multi-Objective RL Pipeline** (`#pipelines/morl`): `Scorer + Modifier(Threshold) -> Objective Score s_i in [0,1] + Environment -> Reward R + Agent(pi_theta) vs Prior(pi_0) -> Pareto Selection -> Policy Update`.
  2. **End-to-End Training Flow** (`#pipelines/training`): `Corpus (Papyrus) + Tokenizer -> Pre-training (MLE) -> Prior pi_0 + Actives -> Fine-Tuning -> Agent pi_init + Environment -> Explorer Loop -> Pareto Leads`.
  3. **3D Shape-Matching (ROCS) Scoring Flow** (`#pipelines/rocs`): `SMILES + Deduplication -> Conformer Generator (ETKDGv3) -> MMFF94 Strain Filter -> 3D Alignment -> Shape/Color Tanimoto -> TanimotoCombo -> SmoothClipped Reward`.
  4. **Fragment-Based Generative Pipeline** (`#pipelines/fragment`): `Active Hits + BRICS/Murcko -> Fragment Synthons [*] -> VocSmiles(encode_frags) -> FragExplorer + Valence Mask -> Combinatorial Assembly -> LE/LipE Feedback`.
  5. **Telemetry & Pathology Filter Cascade** (`#pipelines/telemetry`): `Generated Batch (N) -> [Q1 Validity] -> [Q2 Uniqueness] -> [Q3 SAScore/Ring Strain] -> [Q4 Desired Ratio Pareto] -> [Q5 Diversity] -> Telemetry Feedback`.
- **Deep Inspection & Interactivity**:
  - Pure SVG vector diagrams with responsive layout, custom stage color themes, and dynamic dark/paper theme compatibility.
  - Interactive Node Inspector displaying mathematical transformations, input/output data contracts, parameter tables, and diagnostics.
  - Clickable cross-references linking directly to cppreference symbol documentation.
  - Component Chaining Matrix providing complete tabular architectural breakdowns.
  - Dedicated standalone visual page at `/pipelines` and `/schemas` (`pipelines.html`).

---

## Quickstart & Local Development

### Prerequisites
- Node.js 18+
- `pnpm`

### Start Development Server
```bash
# Serves the platform on dedicated DevPort 34150
pnpm dev
# Open in browser: http://localhost:34150
```

### Validate Documentation Database
```bash
# Validates 234 symbols, schema conformity, badge mappings, and cross-link integrity
pnpm test
# or
pnpm check:docs
```

---

## Vercel Deployment

This branch is pre-configured for instant zero-hassle deployment on [Vercel](https://vercel.com):

1. **Root Static Entry**: `index.html` resides at repository root.
2. **Pre-Deploy Verification**: `vercel.json` runs `node tools/check_docs.mjs` as `buildCommand` to ensure zero broken links or schema regressions before traffic goes live.
3. **Route Rewriting**: Legacy paths (`/docs`, `/docs/cppreference`, `/cppreference`, `/reference`, `/api`) rewrite cleanly to `/index.html`.
4. **Security & Caching**: Pre-configured HTTP response headers for strict transport security, MIME sniffing prevention, frame isolation, and zero-stale HTML cache revalidation.
