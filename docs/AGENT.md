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
├── index.html                   # Sole authoritative Master Reference SPA entrypoint (DATABASE across 234 symbols)
├── pipelines.html               # Sole authoritative Pipeline Visualizer page (5 PIPELINE_SCHEMAS)
├── TEST_INFRA.md                # 4-tier testing infrastructure specification and test methodology
├── TEST_READY.md                # Test readiness checklist, test runner command, and coverage summary
├── tools/
│   ├── check_docs.mjs           # Master audit & verification suite (validates 234 symbols, links, emojis, schemas)
│   ├── deep_audit.mjs           # Deep symbol analyzer across core DrugEx modules and codebase
│   └── audit_gh_pages.mjs       # Legacy comparison scanner against codebase
├── vercel.json                  # Clean static SPA rewrite & security header config
├── package.json                 # Node package configuration and npm scripts
└── pnpm-lock.yaml               # Lockfile for pnpm dev server dependencies
```

---

## Authoritative Files

- `index.html`: The definitive single source of truth for all 234 symbols in `DATABASE`, 10 `LIBRARY_SECTIONS`, and embedded `PIPELINE_SCHEMAS`. Vercel rewrites routes (`/`, `/docs`, `/cppreference`, `/reference`, `/api`) to this file.
- `pipelines.html`: The standalone visualizer for interactive workflow inspection. Vercel rewrites routes (`/pipelines`, `/schemas`) to this file.
- Redundant mirror files in `docs/cppreference/` are strictly deprecated and eliminated; do not recreate them.

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
