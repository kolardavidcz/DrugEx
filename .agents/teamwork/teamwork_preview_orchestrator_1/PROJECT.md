# Project: DrugEx Documentation Enhancement Initiative

## Architecture
- Documentation Platform Root: `/home/kolar/build_projects/drugex_edu/docs`
- Source Code Repository (Ground Truth): `/home/kolar/build_projects/drugex_edu/repo/drugex`
- Single Master Reference SPA: `/home/kolar/build_projects/drugex_edu/docs/index.html` (DATABASE across 234 symbols, LIBRARY_SECTIONS with 10 thematic sections, PIPELINE_SCHEMAS with 5 schemas)
- Single Pipeline Visualizer Page: `/home/kolar/build_projects/drugex_edu/docs/pipelines.html`
- Master Audit Suite: `tools/check_docs.mjs` (executed via `node tools/check_docs.mjs` or `pnpm test`)
- Legacy Mirror Clean-Up: Elimination of redundant nested mirror directory `/home/kolar/build_projects/drugex_edu/docs/cppreference/` in favor of root SPA files deployed directly by Vercel

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Mirror Cleanup & Audit Suite Parity | Remove redundant `docs/cppreference/` mirror files, update `check_docs.mjs` and `deep_audit.mjs` to target root files exclusively, maintain 100% audit pass | M_TEST | survey |
| 2 | Data Preprocessing & Molecular Graphs Rigor | Elevate all symbols in Sections 1 & 6 (`drugex.data`, `drugex.molecules`, `drugex.parallel`): exact Python types, tensor shapes, boundary conditions, authentic executable examples | M1 | survey |
| 3 | Deep Generative Architectures & Telemetry Rigor | Elevate all symbols in Sections 2 & 7 (`drugex.training.generators`, `drugex.logs`, `drugex.training.monitors`): layer dimensions, causal masks, dynamic valence tracking, telemetry | M2 | survey |
| 4 | MORL Policy Exploration & Rewards Rigor | Elevate all symbols in Section 3 (`drugex.training.explorers`, `environment`, `rewards`, `pareto`): Pareto ranking formulas, NSGA-II, Crowding Distance, REINFORCE with baseline, mutation/crossover | M3 | survey |
| 5 | Scoring Engines & 3D Conformer/ROCS Overlays | Elevate all symbols in Sections 4 & 5 (`drugex.training.scorers`, conformer generation, ROCS shape matching, modifiers): ETKDGv3, Tversky shape/color overlap, score modifier curves | M4 | survey |
| 6 | Ecosystem, Experimental & CLI Automation | Elevate all symbols in Sections 8, 9, 10 (`rdkit.*`, `qsprpred.*`, `cdpkit.*`, `experimental::*`, CLI scripts `train.py`, `dataset.py`, `generate.py`, and PIPELINE_SCHEMAS parity) | M5 | survey |
| 7 | Full Cross-Link Graph & Single Source Parity | Verify 0 broken cross-links in seeAlso arrays across all 234 symbols, 0 unicode emojis, complete LIBRARY_SECTIONS coverage with 0 duplicates | M_TEST | survey |
| 8 | E2E Verification & Adversarial Coverage Hardening | Achieve 100% pass on all test tiers, adversarial edge-case verification, and clean Forensic Integrity Audit | M_FINAL | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M_TEST | E2E Test Suite & Mirror Cleanup | Mirror cleanup, audit runner alignment, cross-link verification, publication of TEST_READY.md | none | DONE |
| M1 | Data Preprocessing & Molecular Graphs | Elevate Sections 1 & 6 (data, molecules, parallel) in DATABASE to publication grade | none | PLANNED |
| M2 | Deep Generative Architectures & Telemetry | Elevate Sections 2 & 7 (generators, logs, monitors) in DATABASE to publication grade | none | PLANNED |
| M3 | MORL Policy Exploration & Multi-Objective Rewards | Elevate Section 3 (explorers, environment, rewards, pareto) in DATABASE to publication grade | M1, M2 | PLANNED |
| M4 | Scoring Engines & 3D Conformer/ROCS Overlays | Elevate Sections 4 & 5 (scorers, conformer generators, ROCS overlays, modifiers) | M1 | PLANNED |
| M5 | Ecosystem Bridges, Experimental & CLI Pipelines | Elevate Sections 8, 9, 10 and PIPELINE_SCHEMAS to publication grade | M3, M4 | PLANNED |
| M_FINAL | Final Milestone: E2E Pass & Adversarial Hardening | Pass 100% of test suite across Tiers 1-4, execute Tier 5 adversarial hardening, Forensic Audit | M_TEST, M1-M5 | PLANNED |

## Interface Contracts
### DATABASE Symbol Schema (`docs/index.html`)
Each symbol in `DATABASE[symbol_key]` must strictly conform to:
```javascript
{
  synopsis: "def func(...) -> ReturnType:" or "class ClassName(...):",
  category: "core" | "ecosystem" | "experimental",
  description: "Rigorous academic description explaining both operational role and computational/mathematical mechanics...",
  parameters: [
    { name: "param_name", type: "PreciseType", default: "val", desc: "Detailed boundary condition, valid numerical interval, failure mode..." }
  ],
  returns: { type: "PreciseReturnType", desc: "Exact tensor shape, DataFrame schema, or object contract..." }, // for functions / callables
  members: [
    { name: "method_name", synopsis: "def method_name(...) -> ReturnType:", desc: "Method contract and mechanics..." }
  ], // for classes, regular methods strictly preceding magic methods
  variables: [
    { name: "attr_name", type: "PreciseType", desc: "Instance attribute description..." }
  ], // for classes
  notes: [
    "Preconditions, mathematical formulations, tensor dimensions, default behavior, edge cases..."
  ],
  example: {
    code: "# Idiomatic, executable DrugEx 3.4+ Python code\n...",
    output: "Authentic execution output string\n..."
  },
  seeAlso: ["valid_symbol_key_1", "valid_symbol_key_2"]
}
```

### Constraints & Invariants
1. Single Source of Truth: Root `docs/index.html` and `docs/pipelines.html` are the sole authoritative files.
2. Zero-Emoji Rule: No unicode emojis anywhere in docs or scripts.
3. Strict Cross-Link Graph: Every key in `seeAlso` arrays must resolve to an existing key in `DATABASE`.
4. LIBRARY_SECTIONS Parity: All 234 symbols must be indexed across the 10 thematic sections with 0 omissions and 0 duplicates.
5. Ground Truth Grounding: All signatures, parameters, and types must match `repo/drugex` runtime reality.

## Code Layout
- `docs/index.html`: Master Reference SPA containing `DATABASE`, `LIBRARY_SECTIONS`, and embedded `PIPELINE_SCHEMAS`.
- `docs/pipelines.html`: Standalone Pipeline Visualizer containing standalone `PIPELINE_SCHEMAS`.
- `docs/tools/check_docs.mjs`: Master Audit Suite verifying schema compliance, zero emojis, cross-link integrity, and schema parity.
- `docs/tools/deep_audit.mjs`: Deep audit tool verifying token and member alignments.
- `repo/drugex/`: Ground truth Python implementation files.
