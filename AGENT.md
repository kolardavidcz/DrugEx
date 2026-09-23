# AGENT.md — Education Branch Architecture & Maintenance Guide

> **Branch**: `education` (in `kolardavidcz/DrugEx`)  
> **Purpose**: Houses all educational materials, cppreference-style API docs, and Typst publication monographs.  
> **Environment**: Windows 11 + WSL2 (Ubuntu) / Node.js 22+ & pnpm / Typst 0.15+ / Vercel (`https://drugexdocs.vercel.app`).  

---

## Directory Layout

```
DrugEx (branch: education)
├── docs/                                  # Cppreference-style documentation SPA
│   ├── index.html                         # Primary 3-column SPA entrypoint
│   ├── docs/cppreference/index.html       # Exact mirror copy for deep-link parity
│   ├── tools/check_docs.mjs               # Master symbol audit suite (234 symbols)
│   ├── vercel.json                        # Vercel SPA rewrites and cache headers
│   └── package.json                       # Dev scripts
├── book/                                  # Monograph source suite
│   ├── book/                              # Monograph chapters (ch01..ch07), appendices, themes
│   ├── data/                              # Curricula, benchmarks, quizzes, thesis guide
│   └── package.json                       # Typst build scripts
├── DrugEx_Book_Master.pdf                 # Complete master monograph PDF at root
├── DrugEx_Book_Advanced_Continuation.pdf  # Continuation monograph PDF at root
├── README.md                              # Showcase README linking to https://drugexdocs.vercel.app
├── AGENT.md                               # This agent specification
└── .gitignore                             # Ignores repo/, podcast/, node_modules/, .venv/
```

---

## Verification Commands

Before reporting completion on any task in this branch, all agents MUST run the validation suites:

```bash
# 1. Audit documentation integrity (234 symbols, 0 broken cross-links)
cd docs && node tools/check_docs.mjs

# 2. Compile and verify Typst monographs
cd book && pnpm test
```

---

## Deployment Rules
- **Vercel Root Directory**: `docs` (configured in Vercel project settings).
- **Branch Target**: `education`.
- **Change Isolation**: Only modifications under `docs/` trigger deployments to `https://drugexdocs.vercel.app`.
