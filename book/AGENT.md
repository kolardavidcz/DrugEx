# AGENT.md — Technical Architecture & Maintenance Guide for AI Agents

> **Project**: DrugEx Hub (`~/learn_projects/drugex`)  
> **Purpose**: Interactive educational onboarding platform & computational workbench for Bachelor's Thesis on *De Novo Drug Design & 3D Shape Matching (ROCS) for Flexible Targets / IDPs*.  
> **Environment**: Windows 11 + WSL2 (Ubuntu) / Vanilla ES Modules & React (Modern Component SPA) / DevPort `34100`.  
> **UI Aesthetic**: cppreference-style structural density & clarity infused with DrugEx Chemoinformatics bio-accents (cyan, teal, emerald, nitrogen-blue, oxygen-red badges, 2D/3D molecular hooks). React components and modern libraries are fully supported as the documentation and interactive tools expand.

---

## Architecture Blueprint

DrugEx Hub is engineered as a zero-friction, client-side Single Page Application (SPA) pairing modern VS Code glassmorphic design with WebGL 3D molecular visualization:

```
~/learn_projects/drugex/
├── app/
│   ├── css/
│   │   ├── tokens.css         # Modern VS Code Dark/Light design tokens & bio accents
│   │   ├── shell.css          # Activitybar, collapsible explorer sidebar, tabbar & statusbar
│   │   ├── lecture.css        # Slide presentation decks, comparative grids & GitHub alerts
│   │   ├── dojo.css           # 3D WebGL viewer frames, SVG ROC curves & Pareto sandboxes
│   │   ├── quiz.css           # Active recall questions with immediate contract feedback
│   │   └── print.css          # @media print 2-column A4 study sheet & thesis protocol rules
│   ├── js/
│   │   ├── app.js             # SPA bootstrap, shortcut manager (Ctrl+B, Ctrl+K, Ctrl+P)
│   │   ├── state.js           # Reactive store, salted SHA-256 local auth & FNV-1a option shuffler
│   │   ├── router.js          # Hash router (#/lecture/*, #/dojo/*, #/quiz/*, #/thesis-guide)
│   │   ├── content.js         # Slide deck renderer, comparative panels & thesis methods
│   │   ├── lectures_content.js# Full 18-lecture course curriculum content database
│   │   ├── dojos.js           # Interactive Dojos 1-6 (Tokenizer, Pareto, 3D Confs, ROC, CLI)
│   │   ├── quiz.js            # FNV-1a balanced assessment engine with score persistence
│   │   ├── viewer3d.js        # 3Dmol.js WebGL molecular conformer & shape alignment engine
│   │   ├── format.js          # Syntax highlighter (Python, CLI) & chemical formula formatter
│   │   ├── tree.js            # Explorer sidebar tree rendering with tags and filters
│   │   └── ui.js              # Micro DOM helpers (el, clear, showToast, copyText)
│   └── index.html             # Main HTML5 entrypoint with 3Dmol.js CDN and theme script
├── data/
│   ├── curriculum.json        # Master 6-module curriculum tree manifest
│   ├── thesis_guide.json      # Bachelor thesis literature review, roadmap & protocol template
│   ├── benchmarks/            # CCR2 active/decoy CSVs, SDF references & .sq shape queries
│   └── quizzes/               # Balanced quiz datasets (m1.json ... m6.json)
├── tools/
│   ├── check_contrast.mjs     # WCAG 2.1 AAA color contrast test suite (49 test cases)
│   ├── quiz_stat.mjs          # 25% A/B/C/D option equilibrium statistical verifier
│   └── prepare_build.mjs      # Production static bundle synchronizer (copies to public/)
├── repo/                      # Cloned fulopjoz/DrugEx (branch: feature/rocs-scoring)
├── public/                    # Production bundle destination
├── serve.py                   # Local multi-threaded dev server (Port 34100, zero-cache headers)
└── package.json               # Development scripts and audits
```

---

## Development, Auditing & Verification Commands

All agents modifying code in this repository MUST run the verification suite before reporting task completion:

```bash
# 1. Full reference documentation platform audit (234 symbols, 0 broken cross-links)
node tools/check_docs.mjs

# 2. Serve documentation locally on DevPort 34150
pnpm docs

# 3. Compile Typst monographs and run all validation suites
pnpm test
```

---

## Preserved Workspace Structure
The project materials are cleanly organized and preserved under `~/build_projects/`:
- `~/build_projects/docs` -> `_solved/drugex_book/docs` (includes `cppreference/index.html` React SPA)
- `~/build_projects/book` -> `_solved/drugex_book/book` (monograph chapters, figures, build scripts)
- `~/build_projects/podcast` -> `_solved/drugex_podcast` (TTS generation scripts, mp3s, transcripts)
- `~/build_projects/drugex_help/` -> umbrella directory linking directly to `docs`, `book`, and `podcast`.

---

## Pedagogical & Scientific Rules for Agents

1. **Deterministic Quiz Equilibrium**: When adding or updating questions in `data/quizzes/*.json`, ensure that `ensureShuffledOptions` in `app/js/state.js` maintains an exact ~25% distribution across options A, B, C, D using FNV-1a hashing.
2. **Flexible Target & IDP Context**: Keep all references and educational callouts grounded in the student's thesis topic (using ROCS ligand-based shape matching to overcome the lack of rigid receptor pockets in flexible proteins / IDPs).
3. **Zero Broken Tests**: Maintain 100% test pass rate with 0 lint errors, 0 syntax faults, and 0 failing contrast assertions.
4. **WSL-First & DevPort Compliance**: Dev server strictly binds to port `34100` (`34000 - 38999` block).
