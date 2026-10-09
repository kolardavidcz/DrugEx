# AGENT.md — Technical Architecture & Maintenance Guide for AI Agents

> **Project**: DrugEx Reference Platform (`~/build_projects/drugex_edu/docs`)  
> **Purpose**: Complete, high-performance, interactive DrugEx API documentation and chemoinformatics reference platform styled after [cppreference.com](https://cppreference.com).  
> **Coverage**: 234 symbols across DrugEx Core (186), Chemoinformatics Ecosystem (24), and Experimental Molecular Techniques (24).  
> **Environment**: Windows 11 + WSL2 (Ubuntu) / Zero-Build React 18 SPA / DevPort `34150` / Vercel Production (`https://drugexdocs.vercel.app`).  

---

## Directory Layout & Single Source of Truth

The documentation platform operates under a strict Single Source of Truth architecture. Legacy nested mirror directories (`docs/cppreference/`) have been removed; root files are served directly by Vercel for all inbound routes via `vercel.json` rewrites.

```
~/build_projects/drugex_edu/docs/
├── index.html                   # Clean, zero-build Master Reference SPA HTML shell (links style.css, data/*.js, app.js)
├── app.js                       # React 18 UI application shell, search engine, and interactive MicroViz components
├── style.css                    # Definitive cppreference stylesheet (layout, syntax tokens, responsive drawer, micro-viz)
├── data/                        # Modular Reference Database (11 discrete JS data modules)
│   ├── sections.js              # 10 Thematic cppreference library sections (window.LIBRARY_SECTIONS)
│   ├── 01_morl.js               # MORL Environments, Pareto rewards, monitors (15 symbols)
│   ├── 02_neural.js             # Generator, RNN, Transformers, attention blocks (21 symbols)
│   ├── 03_explorers.js          # Explorers, Sequence/Graph exploration loops (6 symbols)
│   ├── 04_scoring.js            # Scorers, 3D ROCS, conformers, SAscore, fingerprints (31 symbols)
│   ├── 05_modifiers.js          # Desirability modifiers, Gaussian, SmoothHump (11 symbols)
│   ├── 06_data.js               # Vocabularies, standardization, corpora, suppliers, datasets (31 symbols)
│   ├── 07_molecules.js          # Molecule, DrExMol, suppliers, converters, exceptions (23 symbols)
│   ├── 08_workflows.js          # Parallel processors, evaluators, CLI pipelines, loggers (48 symbols)
│   ├── 09_ecosystem.js          # RDKit 3D/scaffolds, QSPRPred ML, CDPKit pharmacophores (24 symbols)
│   └── 10_experimental.js       # USRCAT 3D/4D, ProLIF, Chemprop D-MPNN, Vina, OpenFE, Optuna (24 symbols)
├── TEST_INFRA.md                # 4-tier testing infrastructure specification and test methodology
├── TEST_READY.md                # Test readiness checklist, test runner command, and coverage summary
├── tools/
│   ├── check_docs.mjs           # Master audit & verification suite (validates 234 symbols, links, emojis, schema)
│   ├── deep_audit.mjs           # Deep symbol analyzer across core DrugEx modules and codebase
│   └── audit_gh_pages.mjs       # Legacy comparison scanner against codebase
├── vercel.json                  # Clean static SPA rewrite & security header config
├── package.json                 # Node package configuration and npm scripts
└── pnpm-lock.yaml               # Lockfile for pnpm dev server dependencies
```

---

## Authoritative Files & Modular Architecture

- `index.html`: The lightweight (~40 lines) zero-build static HTML shell. Vercel rewrites routes (`/`, `/docs`, `/cppreference`, `/reference`, `/api`) to this entrypoint.
- `style.css`: Unified stylesheet eliminating redundant embedded CSS and obsolete pipeline diagram styles.
- `app.js`: Component shell containing search filtering, responsive navigation, and interactive SVG micro-visualizers.
- `data/*.js`: The 11 modular data files containing all 234 symbols and 10 library sections. Individual libraries can be inspected, updated, and diffed without context collapse.
- Obsolete pipeline visualizer files (`pipelines.html`, `PIPELINE_SCHEMAS`) have been eliminated in favor of clean CLI and Python workflows.
- Redundant mirror files in `docs/cppreference/` remain strictly deprecated and eliminated.

---

## Verification & Audit Commands

All agents modifying documentation must run:

```bash
# 1. Full database integrity check (234 symbols, 0 broken cross-links, 10 sections, schema parity, zero emoji)
node tools/check_docs.mjs
# or
pnpm test

# 2. Local preview server on DevPort 34150
pnpm dev
```

---

## Deployment Rules

- **Zero Build Overhead**: Static files are served directly from root (`index.html` and `pipelines.html`).
- **No Mirror Files**: Never maintain or generate redundant nested copies in subdirectories.
- **Strict Zero-Emoji Rule**: Unicode emojis are strictly forbidden across all HTML, JS, JSON, and Markdown documentation.
- **No Book Assets**: All Typst book chapters, PDFs, and data benchmarks live exclusively on the `education_book` branch under `~/build_projects/drugex_edu/book`.
