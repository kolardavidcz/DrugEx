# DrugEx Education & Chemoinformatics Hub

[![Documentation](https://img.shields.io/badge/docs-live-emerald?style=flat-square&logo=vercel)](https://drugexdocs.vercel.app)
[![Coverage](https://img.shields.io/badge/symbols-234%20verified-blue?style=flat-square)](https://drugexdocs.vercel.app)
[![Monograph](https://img.shields.io/badge/book-Nature%20Typst-violet?style=flat-square)](DrugEx_Book_Master.pdf)
[![License](https://img.shields.io/badge/license-MIT-gray?style=flat-square)](LICENSE)

> Interactive documentation platform, chemoinformatics reference suite, and monograph onboarding curriculum for **DrugEx** (De Novo Drug Design & Multi-Objective Reinforcement Learning with 3D Shape-Matching / ROCS).

---

## Live Platform & Interactive Docs

The official reference platform is hosted on Vercel:
**[drugexdocs.vercel.app](https://drugexdocs.vercel.app)**

---

## Repository Structure

This dedicated `education` branch isolates all learning materials and reference documentation from the core DrugEx computational engine:

```
DrugEx (branch: education)
├── docs/                                  # Cppreference-style documentation SPA platform (v2)
│   ├── index.html                         # Master 3-column SPA entrypoint (zero-build React 18)
│   ├── docs/cppreference/index.html       # Exact mirror copy for deep-link parity
│   ├── tools/                             # Master verification & symbol audit suites (234 symbols)
│   ├── vercel.json                        # Static SPA routing & security headers
│   └── package.json                       # Preview scripts (pnpm dev, pnpm test)
├── book/                                  # Publication monograph & Typst compiler suite
│   ├── book/                              # Monograph chapters (ch01..ch07), appendices, figures, themes
│   ├── data/                              # Benchmark SDFs, active/decoy CSVs, curriculum, quizzes, thesis guide
│   └── package.json                       # Typst build scripts (pnpm build, pnpm test)
├── DrugEx_Book_Master.pdf                 # Complete compiled publication book PDF (Chapters 1–7 + Appendices)
├── DrugEx_Book_Advanced_Continuation.pdf  # Continuation monograph PDF (Deep Doping & Advanced ROCS)
├── README.md                              # This repository guide
├── AGENT.md                               # AI agent maintenance & execution guide
└── .gitignore                             # Ignores repo/, podcast/, node_modules/, .venv/
```

---

## Section 1: Documentation Platform (`docs/`)

The documentation platform provides a concentrated, high-density API reference modeled after **[cppreference.com](https://cppreference.com)**:
- **234 Symbols Indexed**:
  - **DrugEx Core (186 symbols)**: Generators (`SequenceRNN`, `GraphTransformer`), MORL Explorers (`SequenceExplorer`), Environments (`DrugExEnvironment`), Scorers (`RDKitROCSScorer`, `SAScorer`, `SmoothClippedScore`), Modifiers, and Decoders.
  - **Chemoinformatics Ecosystem (24 symbols)**: RDKit ETKDGv3 conformer generation, QSPRpred predictors, Scaffviz clustering, and Papyrus data loaders.
  - **Experimental Methods (24 symbols)**: Shape-matching descriptors, Pareto Crowding Distance, Youden's Index calibration, and BRICS fragmentation.
- **Interactive Pipeline & Method Schemas**:
  - Visual diagrams and mathematical dataflow schemas for 5 core DrugEx workflows: Multi-Objective RL, End-to-End Training, 3D Shape-Matching (ROCS), Fragment-Based Elaboration, and Telemetry / Pathology Filter Cascades.
  - Interactive SVG visual flowcards, component inspection, algebraic chaining expressions, and full cppreference-grade component chaining matrices.
  - Dedicated standalone page accessible at `/pipelines` and `/schemas` (`pipelines.html`).
- **Fast Client-Side Search**: Instant prefix matching, category filtering, and direct links to GitHub source implementations.

### Running Docs Locally
```bash
cd docs
pnpm install
pnpm dev      # Serves on DevPort 34150
node tools/check_docs.mjs  # Runs full 234-symbol database integrity test
```

---

## Section 2: Onboarding Book & Monographs (`book/`)

The monograph provides a scientific foundation and experimental protocol for Bachelor's Thesis research on:
> *De Novo Drug Design & 3D Shape Matching (ROCS) for Flexible Targets / IDPs*

- **Chapter 1**: Molecular Representations (SMILES, Molecular Graphs, Latent Vectors)
- **Chapter 2**: Multi-Objective Reinforcement Learning & Gated Networks
- **Chapter 3**: ROCS 3D Shape Matching & Bioactive Conformational Strain
- **Chapter 4**: Rosetta Stone: Cross-Engine ROCS Comparison (RDKit vs OpenEye vs CDPKit)
- **Chapter 5**: Ultrafast Shape Recognition (USR) & 3D Alignments
- **Chapter 6**: Scaffold Hopping via Extended Graph Clipping (EGC)
- **Chapter 7**: Diagnostics, Failure Modes & Experimental LEGO Assemblies
- **Appendices A–F**: Standard Operating Procedure (SOP), IDP Theory, Thesis Methods Template, Annotated Bibliography, Glossary, Cheatsheet.

### Precompiled Publication PDFs
- **[DrugEx_Book_Master.pdf](DrugEx_Book_Master.pdf)** — Complete Nature-style monograph (Master edition).
- **[DrugEx_Book_Advanced_Continuation.pdf](DrugEx_Book_Advanced_Continuation.pdf)** — Advanced continuation chapters.

### Compiling Monographs with Typst
```bash
cd book
pnpm build         # Compiles DrugEx_Book_Master.pdf via Typst
pnpm test          # Compiles all editions and verifies PDF generation
```

---

## Deployment Policy
- **Automated Vercel Deployment**: Vercel monitors `docs/` on this `education` branch.
- **Isolated Builds**: Modifying chapters in `book/` or updating root PDFs will not trigger or disrupt the documentation deployment.
