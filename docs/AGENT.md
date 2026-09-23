# 🤖 AGENT.md — Technical Architecture & Maintenance Guide for AI Agents

> **Project**: DrugEx Reference Platform (`~/build_projects/drugex_edu/docs`)  
> **Purpose**: Complete, high-performance, interactive DrugEx API documentation and chemoinformatics reference platform styled after [cppreference.com](https://cppreference.com).  
> **Coverage**: 234 symbols across DrugEx Core (186), Chemoinformatics Ecosystem (24), and Experimental Molecular Techniques (24).  
> **Environment**: Windows 11 + WSL2 (Ubuntu) / Zero-Build React 18 SPA / DevPort `34150` / Vercel Production (`https://drugexdocs.vercel.app`).  

---

## 🏛️ Directory Layout

This branch (`docs`) strictly houses only the documentation platform:

```
~/build_projects/drugex_edu/docs/
├── index.html                   # Primary concentrated 3-column cppreference-style SPA entrypoint
├── docs/
│   └── cppreference/
│       └── index.html           # Exact mirror copy ensuring parity on /docs/cppreference deep-links
├── tools/
│   ├── check_docs.mjs           # Master audit & verification suite (validates 234 symbols & links)
│   ├── deep_audit.mjs           # Deep symbol analyzer across core DrugEx modules
│   └── audit_gh_pages.mjs       # Legacy gh-pages comparison scanner
├── vercel.json                  # Clean static SPA rewrite & security header config
├── package.json                 # Node package configuration and npm scripts
└── pnpm-lock.yaml               # Lockfile for pnpm dev server dependencies
```

---

## 🧪 Verification & Audit Commands

All agents modifying documentation must run:

```bash
# 1. Full database integrity check (234 symbols, 0 broken cross-links, exact dual-file sync)
node tools/check_docs.mjs
# or
pnpm test

# 2. Local preview server on DevPort 34150
pnpm dev
```

---

## 🚀 Deployment Rules

- **Zero Build Overhead**: Static files are served directly from root (`index.html`).
- **No Book Assets**: All Typst book chapters, PDFs, and data benchmarks live exclusively on the `education_book` branch under `~/build_projects/drugex_edu/book`.
