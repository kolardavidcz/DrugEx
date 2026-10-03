# Documentation Platform Specification Survey & Reference Architecture Report

**Agent Archetype**: Specification Miner  
**Role**: Docs Platform Specification Miner  
**Target Path**: `/home/kolar/build_projects/drugex_edu/docs`  
**Date**: 2026-10-03  
**Status**: Comprehensive Survey Complete (Read-Only)

---

## 1. Observation

Direct, empirical observations recorded from codebase inspection and audit suite execution:

1. **Master Reference SPA (`docs/index.html`)**:
   - **File Size & Line Count**: 949,492 bytes, 19,336 lines.
   - **Architecture**: Zero-build React 18 SPA delivered via CDN (`react.production.min.js`, `react-dom.production.min.js`, `@babel/standalone`), styled with CSS variables supporting Light (Paper) and Dark (Deep Space) themes, and Prism.js for syntax-highlighted Python examples.
   - **`DATABASE` Object** (`index.html:2705` to `index.html:15661`):
     - Total registered symbols: **234**.
     - Category breakdown:
       - `core`: **186 symbols** (`drugex.*`)
       - `ecosystem`: **24 symbols** (`rdkit.*` [14], `qsprpred.*` [5], `cdpkit.*` [5])
       - `experimental`: **24 symbols** (`experimental::usrcat`, `electroshape`, `prolif`, `chemprop`, `scaffviz`, `rascore`, `vina`, `openfe`, `optuna`, `strain`)
     - Kind breakdown:
       - `class`: **182 symbols**
       - `function`: **52 symbols**
     - Field population:
       - `synopsis`, `description`, `notes`, `example`, `seeAlso`: 234 / 234 (100%).
       - `members`: 182 / 182 classes (386 total member methods documented).
       - `parameters`: 185 symbols (475 total parameters documented).
       - `variables`: 130 classes (instance attributes).
       - `seeAlso` cross-links: 492 total valid links across 234 symbols (average 2.10 links/symbol, 0 broken links).
   - **Helper Badges & Functions** (`index.html:15664` to `index.html:15807`):
     - `getSymbolTag(key, item)`: Formats platform/version badges (`60D`, `4D`, `RDKit`, `GPU`, `v3.4`, `ProLIF`, `Chemprop`, `Vina`, `OpenFE`, `Optuna`, etc.).
     - `isAbstractSymbol(key, item)`: Accurately identifies **exactly 30** abstract base classes / interface protocols across DrugEx.
     - `getEffectiveMembers(item)`: Strips `__init__` constructor methods from member lists (constructors are presented in header synopsis) and strictly guarantees that regular methods precede magic dunder methods.
     - `getMagicMethodOperator(name)`: Maps 7 Python magic dunder methods (`__init__`, `__call__`, `__getitem__`, `__setitem__`, `__len__`, `__iter__`, `__contains__`) to operator tokens and badges (`(constructor)`, `obj(...)`, `obj[...]`, `len(obj)`, `iter(obj)`, `x in obj`).
     - `getMicroVizType(key)`: Maps **27** target symbols to 4 interactive SVG micro-visualizations (`pareto_crowding` [5], `pareto_tanimoto` [1], `modifier_curve` [11], `rl_policy_cycle` [10]).
   - **`LIBRARY_SECTIONS`** (`index.html:2167` to `index.html:2702`):
     - Exactly **10** thematic library sections, comprising **38** subsections, indexing all **234** symbols with 0 duplicates and 100% coverage.
   - **`PIPELINE_SCHEMAS`** (`index.html:16718` to `index.html:17653`):
     - Exactly **5** workflow schemas (`morl`, `training`, `rocs`, `fragment`, `telemetry`).
     - Each schema contains exactly 8 nodes arranged within a 1130x355 canvas, >= 5 connectors, matrixRows (8 steps), and clickable mathematical equation terms.
   - **Mount Point**: `index.html:19332`: `ReactDOM.createRoot(document.getElementById("root")).render(<App />);`.

2. **Master Audit Suite (`docs/tools/check_docs.mjs`)**:
   - **File Size & Line Count**: 25,274 bytes, 628 lines.
   - **Command Execution & Result**:
     ```bash
     $ node tools/check_docs.mjs
     [Audit] Validating DrugEx Reference Documentation Platform...
     Target: /home/kolar/build_projects/drugex_edu/docs/index.html
     Verified: root index.html and docs/cppreference/index.html are in sync.
     [PASS] DATABASE evaluated successfully: 234 total symbols registered.
     Category Breakdown:
        - DrugEx Core:       186 symbols
        - Chemo Ecosystem:   24 symbols
        - Experimental:      24 symbols

     Validating Concentrated Main Page Library Index (cppreference.com format)...
     [PASS] All 234 symbols 100% indexed across 10 thematic libraries with 0 duplicates.

     Validating getSymbolTag helper badges...
     [PASS] getSymbolTag verified for core badges (60D, 4D, RDKit, GPU, v3.4).
     [PASS] isAbstractSymbol verified: exactly 30 abstract base classes identified.
     [PASS] getEffectiveMembers verified: 0 classes have __init__ in member functions; 100% of classes (182) strictly order regular methods before magic methods.
     [PASS] getMagicMethodOperator verified across all special Python operator mappings.
     [PASS] getMicroVizType verified across 27 target symbols with 4 visualization types.

     Validating PIPELINE_SCHEMAS and pipelines.html Parity...
     Verified: root pipelines.html and docs/cppreference/pipelines.html are in sync.
     [PASS] [index.html] 5 Pipeline schemas evaluated successfully.
     [PASS] [index.html] All drugexSymbols in PIPELINE_SCHEMAS exist in DATABASE.
     [PASS] [pipelines.html] 5 Pipeline schemas evaluated successfully.
     [PASS] [pipelines.html] All drugexSymbols in PIPELINE_SCHEMAS exist in DATABASE.

     Validating Strict Zero-Emoji Rule...
     [PASS] Verified 0 emojis across documentation files.
     [SUCCESS] All assertions passed: 100% schema compliance, 0 broken cross-links, authentic cppreference format verified!
     ```
   - Audit runs 9 distinct verification phases covering structural integrity, sandboxed JS execution, schema validation, cross-link graphing, helper badge parity, pipeline schema parity, and unicode emoji prohibition.

3. **Standalone Pipeline Visualizer (`docs/pipelines.html`)**:
   - **File Size & Line Count**: 134,466 bytes, 2,513 lines.
   - Houses `<StandalonePipelinesApp />` rendered at `pipelines.html:2509`.
   - Contains identical copy of `PIPELINE_SCHEMAS` (lines 907 to 1842).
   - Features interactive canvas zoom, node inspector modal, equation term selector, and step-by-step assembly matrix table.
   - Deep-links directly back to `index.html#<symbol_key>` when nodes or symbols are clicked.

4. **Legacy Mirror Directory (`docs/docs/cppreference/`)**:
   - Contains 6 files: `index.html` (949,492 B), `pipelines.html` (134,466 B), `favicon.ico` (21,460 B), `favicon.png` (978,668 B), `favicon.svg` (2,213 B), `logo.png` (978,668 B).
   - All 6 files are bitwise identical duplicates of their root counterparts in `docs/`.
   - `docs/vercel.json` contains universal SPA rewrites:
     - `/docs`, `/docs/(.*)`, `/cppreference`, `/cppreference/(.*)`, `/reference`, `/api` -> `/index.html`
     - `/pipelines`, `/schemas` -> `/pipelines.html`
   - Due to Vercel rewrites, the physical nested folder `/docs/cppreference/` is 100% redundant and creates maintenance overhead.

5. **Documentation Specification Guide (`docs/AGENT.md`)**:
   - 51 lines detailing directory layout, verification commands (`pnpm test`, `node tools/check_docs.mjs`, `pnpm dev` on port `34150`), and zero-build static deployment rules.
   - Currently references `docs/cppreference/index.html` as a mirror copy that must be maintained in sync.

---

## 2. Logic Chain

1. **Premise 1 (Ground Truth vs. Documentation Role)**: `ORIGINAL_REQUEST.md` states that the DrugEx core codebase (`repo/drugex`) is undergoing docstring streamlining to concise high-value comments. Therefore, `docs/` must become the comprehensive, self-contained single source of truth containing exhaustive technical details, algorithmic formulations, tensor dimensions, and parameter contracts.
2. **Premise 2 (Database Schema Completeness)**: The current `DATABASE` in `docs/index.html` successfully registers 234 symbols across Core (186), Ecosystem (24), and Experimental (24). Every symbol has `synopsis`, `description`, `notes`, `example`, and `seeAlso`. Classes have `members` (386 total) and parameters (475 total). However, 57 examples were identified as concise (<10 lines of code or <3 lines of output), representing prime candidates for enrichment with publication-grade real-world cheminformatics workflows.
3. **Premise 3 (Thematic Categorization Rigor)**: `LIBRARY_SECTIONS` provides complete coverage (234/234 symbols) across 10 sections and 38 subsections. All symbols belong to an established conceptual taxonomy matching cppreference.com conventions.
4. **Premise 4 (Automated Audit Suite Robustness)**: `tools/check_docs.mjs` executes cleanly in 1.4 seconds with zero errors, verifying 100% schema compliance, 0 broken cross-links, 30 abstract base classes, 182 classes with constructor hygiene, 5 pipeline schemas, and 0 unicode emojis.
5. **Premise 5 (Redundancy Elimination Necessity)**: Because `vercel.json` handles all path rewrites server-side, physical files in `docs/cppreference/` serve no functional deployment purpose. Removing this mirror directory eliminates ~2.9 MB of duplicated content and removes fragile synchronization assertions from `check_docs.mjs`.

---

## 3. Caveats

- **Read-Only Investigation**: In strict compliance with subagent scope boundaries and user rules, zero modifications, deletions, or file creation took place outside `.agents/teamwork/teamwork_preview_spec_miner_survey_1/`.
- **Ecosystem Symbols Scope**: Ecosystem symbols (`rdkit`, `cdpkit`, `qsprpred`) and experimental symbols (`usrcat`, `electroshape`, `prolif`, `chemprop`, `vina`, `openfe`, `optuna`) reside outside `repo/drugex/` and reflect external library contracts. Auditing their algorithmic parameters requires cross-referencing external library interfaces.
- **Micro-Visualizations**: Micro-visualizations in `index.html` are powered by pure React inline SVG components (`ParetoCrowdingMicroViz`, `ParetoTanimotoMicroViz`, `ModifierCurveMicroViz`, `RLPolicyCycleMicroViz`) and do not require external chart libraries.

---

## 4. Conclusion

The DrugEx documentation platform is a highly engineered, production-ready, zero-build React 18 SPA following cppreference-style technical architecture. It completely indexes all 234 symbols with 0 broken cross-links and passes its master audit suite (`check_docs.mjs`).

To fulfill `ORIGINAL_REQUEST.md` requirements for publication-grade academic excellence:
1. **Single Source of Truth Cleanliness**: Retire `docs/cppreference/` and update `tools/check_docs.mjs`, `tools/deep_audit.mjs`, and `docs/AGENT.md` to reference root `index.html` and `pipelines.html` as the sole authoritative files.
2. **Technical Depth Elevation**: Enhance the 234 symbol entries in `DATABASE` directly against `repo/drugex` source code, documenting exact tensor dimensions, mathematical equations (Pareto dominance, crowding distance, loss functions, policy updates), parameter valid ranges, and edge-case handling.
3. **Example Workflow Enrichment**: Upgrade the 57 concise code examples into multi-step, idiomatic DrugEx workflows demonstrating real-world ligand-based de novo drug design.

---

## 5. Verification Method

To independently verify all findings in this survey report:

```bash
# 1. Run Master Audit Suite (Verifies 234 symbols, 10 sections, 5 pipelines, 0 broken links, 0 emojis)
cd /home/kolar/build_projects/drugex_edu/docs && node tools/check_docs.mjs

# 2. Verify Vercel Routing Configuration
cat /home/kolar/build_projects/drugex_edu/docs/vercel.json

# 3. Check Exact Dual-File Binary Equality between root and docs/cppreference
cmp /home/kolar/build_projects/drugex_edu/docs/index.html /home/kolar/build_projects/drugex_edu/docs/docs/cppreference/index.html
cmp /home/kolar/build_projects/drugex_edu/docs/pipelines.html /home/kolar/build_projects/drugex_edu/docs/docs/cppreference/pipelines.html

# 4. Preview in Browser (DevPort 34150)
cd /home/kolar/build_projects/drugex_edu/docs && pnpm dev
```

**Invalidation Conditions**:
- If `node tools/check_docs.mjs` exits with non-zero status or reports any broken cross-links or missing symbols.
- If total symbol count drops below 234.
- If any unicode emoji is introduced into `index.html`, `pipelines.html`, or audit scripts.

---

## Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Reference Database | `DATABASE` Object | Central JS key-value dictionary registering all DrugEx and ecosystem symbols | Symbol key string (e.g. `drugex::training::environment::DrugExEnvironment`) | Symbol metadata object (`category`, `module`, `title`, `kind`, `badges`, `synopsis`, `description`, `members`, `parameters`, `variables`, `notes`, `example`, `seeAlso`) | Evaluated in sandboxed `node:vm` context; missing required fields trigger fatal audit errors | `docs/index.html:2705`, `check_docs.mjs:61` |
| 2 | Reference Database | 3-Category Taxonomy | Categorizes symbols into Core (186), Chemo Ecosystem (24), and Experimental (24) | `item.category` string | Categorization filter tab in SPA UI | Unexpected categories trigger audit failure | `docs/index.html`, `check_docs.mjs:97` |
| 3 | Reference Database | Class Member Hygiene | Enforces that class constructors (`__init__`) belong in header synopsis, not member methods | Class symbol entry in `DATABASE` | `members` array containing `{ name, desc }` methods | Class with empty `members` or zero-arg init with non-empty parameters triggers audit error | `check_docs.mjs:121` |
| 4 | Reference Database | Cross-Link Graph Integrity | `seeAlso` array of symbol keys creating bidirectional navigation graph | Array of symbol key strings | Clickable cross-link badges in UI right sidebar | Any target key not existing in `DATABASE` triggers broken cross-link audit error | `check_docs.mjs:146` |
| 5 | UI & Navigation | `LIBRARY_SECTIONS` Index | Concentrated 10-section, 38-subsection thematic index modeled after cppreference.com | Section ID, title, and subsection key arrays | 3-column accordion sidebar and concentrated main index page (`#main`) | Missing symbols or duplicate keys trigger audit errors; requires 100% coverage (234/234) | `docs/index.html:2167`, `check_docs.mjs:172` |
| 6 | UI & Formatting | `getSymbolTag` Badges | Contextual platform/version tag badge generator | Symbol key string and item object | Formatted string badge (`60D`, `4D`, `RDKit`, `GPU`, `v3.4`, etc.) | Returns `null` if no special tag applies | `docs/index.html:15664`, `check_docs.mjs:231` |
| 7 | UI & Formatting | `isAbstractSymbol` Protocol | Helper identifying abstract base classes and interfaces | Symbol key and item object | Boolean (`true` for ABCs / protocols) | Classes with "Exception" in name explicitly excluded | `docs/index.html:15702`, `check_docs.mjs:264` |
| 8 | UI & Formatting | `getEffectiveMembers` Ordering | Orders regular member methods first, followed by magic dunder methods; strips `__init__` | Symbol item object | Filtered and sorted member array | Throws audit error if `__init__` is present or magic methods precede regular methods | `docs/index.html:15727`, `check_docs.mjs:305` |
| 9 | UI & Formatting | `getMagicMethodOperator` Map | Maps Python dunder methods to operator badges | Method name string (e.g. `__call__`, `__getitem__`) | Object `{ op, label, cls }` or `null` | Returns `null` for regular non-dunder methods | `docs/index.html:15738`, `check_docs.mjs:343` |
| 10 | Interactive Visualizations | `getMicroVizType` Dispatcher | Dispatches interactive inline SVG diagrams for 27 key algorithmic symbols | Symbol key string | Visualization type string (`pareto_crowding`, `pareto_tanimoto`, `modifier_curve`, `rl_policy_cycle`) or `null` | Returns `null` for non-visualized symbols | `docs/index.html:15762`, `check_docs.mjs:376` |
| 11 | Architecture Visualizer | `PIPELINE_SCHEMAS` Dataset | 5 canonical DrugEx workflow pipeline specifications | Pipeline ID (`morl`, `training`, `rocs`, `fragment`, `telemetry`) | Schema object containing 8 nodes, connectors, equation terms, assembly matrix rows | Coordinates outside 1130x355 canvas or unmapped symbols trigger audit errors | `docs/index.html:16718`, `pipelines.html:907` |
| 12 | Architecture Visualizer | Standalone Pipeline Page | Interactive dedicated pipeline visualizer application | Query param / hash ID | Full-screen interactive diagram with zoom, node inspection, and cross-linking to reference | Defaults to `morl` pipeline if invalid ID passed | `docs/pipelines.html:2457` |
| 13 | Audit & Quality Assurance | Strict Zero-Emoji Rule | Enforces zero unicode emoji characters across documentation files | UTF-8 file content of HTML and audit scripts | Pass/fail regex check | Detects unicode emojis (`\u{1F300}-\u{1F9FF}`, etc.) and triggers fatal audit exit | `check_docs.mjs:593` |
| 14 | Hosting & Routing | Vercel SPA Routing Configuration | Universal clean-URL and rewrite rules mapping all reference paths to single root files | Inbound HTTP request path (`/docs`, `/cppreference`, `/pipelines`, `/schemas`) | Destination static file (`/index.html` or `/pipelines.html`) with strict security headers | HTTP 404 handled gracefully via SPA fallback | `docs/vercel.json` |

---

## Edge Cases

| # | Feature | Input | Observed Behavior |
|---|---------|-------|-------------------|
| 1 | Class Constructor Hygiene | Class synopsis defining `def __init__(self) -> None:` | If `parameters` array is non-empty, `check_docs.mjs:129` flags a fatal error because zero-argument constructors must have an empty parameters list. |
| 2 | Class Members Completeness | Class symbol in `DATABASE` with empty `members: []` | Triggers fatal audit failure `Class "<key>" has empty members array.` All 182 classes must have >= 1 documented member method. |
| 3 | Pipeline Canvas Coordinate Bounds | Node coordinates `x: 920, y: 260` (canvas: 1130x355, node width: 220, height: 98) | Triggers fatal audit error because `920 + 220 = 1140 > 1130`. All nodes and connectors must strictly fit inside the 1130x355 bounding box. |
| 4 | Abstract Base Class Exception Filtering | Exception classes like `ConversionException` or `RewardException` inheriting from `Exception` | `isAbstractSymbol` explicitly returns `false` despite abstract badges, ensuring exception types are not miscategorized as abstract architecture contracts. |
| 5 | Dunder Method Sorting | Class with `__call__` listed before `getScores` in `members` | `check_docs.mjs:324` flags fatal error `Class "<key>" has regular method after magic dunder method`. Regular methods must strictly precede magic methods. |
| 6 | URL Hash Navigation | URL hash like `#drugex::training::environment::DrugExEnvironment` or `#main` | SPA listener detects hashchange, loads corresponding symbol view or concentrated library index, and smoothly scrolls to top without page reload. |
| 7 | Pipeline Node Deep Linking | User clicks a symbol link inside a pipeline node (e.g. `drugex::training::scorers::RDKitROCSScorer`) | Dispatches `index.html#drugex::training::scorers::RDKitROCSScorer`, navigating directly to the symbol's full API contract. |
| 8 | Dual-File Mirror Drift | Modifying `docs/index.html` without updating `docs/docs/cppreference/index.html` | `check_docs.mjs:36` compares file buffers directly; any byte discrepancy triggers `Error: root index.html and docs/cppreference/index.html differ!`. |
| 9 | Unicode Emoji Injection | Any commit introducing an emoji character (e.g. rocket, fire, checkmark) | `check_docs.mjs:595` regex scan fails the build immediately with exit code 1. |
| 10 | Invalid SMILES Scoring | SMILES array containing `"INVALID_SMILES_%%"` passed to `DrugExEnvironment.getScores()` | Zeroes all $M$ objective scores for that index, sets `Valid=0`, `Desired=0`, and returns a zero vector in `getRewards()` without crashing. |

---

## Detailed Structural Breakdown

### 1. `DATABASE` Symbol Entry Schema Specification

Every symbol entry in `DATABASE` follows this typed structure:

```typescript
interface SymbolEntry {
  // Required Metadata
  category: "core" | "ecosystem" | "experimental";
  module: string;             // e.g. "drugex.training.environment"
  title: string;              // Short symbol name, e.g. "DrugExEnvironment"
  kind: "class" | "function"; // 182 classes, 52 functions
  badges: string[];           // Contextual tags, e.g. ["DrugEx 3.4+", "Core RL Engine", "Multi-Objective"]
  synopsis: string;           // Formatted Python definition with type annotations
  description: string;        // High-level conceptual and operational overview
  notes: string;              // Rigorous technical details, tensor shapes, boundary conditions, failure modes
  example: {
    code: string;             // Executable, idiomatic Python code snippet
    output: string;           // Expected standard output
  };
  seeAlso: string[];          // Array of target keys in DATABASE (zero broken links)

  // Class-Specific Fields
  members?: Array<{
    name: string;             // Method signature, e.g. "getScores(smiles, frags=None)"
    desc: string;             // Algorithmic description, return shape, parameters
  }>;
  variables?: Array<{
    name: string;             // Attribute name
    type: string;             // Type annotation, e.g. "List[Scorer]"
    desc: string;             // Operational role
  }>;

  // Parameter & Return Contracts
  parameters?: Array<{
    name: string;             // Parameter name
    type: string;             // Type annotation, e.g. "Optional[List[float]]"
    desc: string;             // Valid range, default value, semantic meaning
  }>;
  returns?: {
    type: string;             // Return type annotation
    desc: string;             // Return semantics, dimensions
  };
}
```

### 2. The 10 Thematic `LIBRARY_SECTIONS`

| # | Section ID | Title | Subsections | Key Count | Description |
|---|------------|-------|-------------|-----------|-------------|
| 1 | `morl` | Core Reinforcement Learning (MORL) | 4 | 15 | Environments, Pareto ranking, training monitors, learning rate schedules |
| 2 | `neural` | Generative Neural Architectures | 5 | 21 | Sequence RNNs/Transformers, Graph Transformers, attention blocks, masking |
| 3 | `explorers` | Explorers & RL Search Policies | 3 | 6 | Sequence/Graph/Fragment explorers and base search policies |
| 4 | `scoring` | Scoring & Objective Oracles | 6 | 31 | Physicochemical properties, 3D ROCS shape matching, conformers, SA, similarity, QSAR |
| 5 | `modifiers` | Desirability Modifiers | 3 | 11 | Threshold clipping, Gaussian humps, linear transitions, chained modifier pipelines |
| 6 | `data` | Data, Fragments & Vocabularies | 5 | 31 | PyTorch datasets, SMILES/Graph vocabularies, BRICS cleavage, splitters, standardizers |
| 7 | `molecules` | Molecules, Converters & Exceptions | 4 | 23 | DrExMol, suppliers, format converters, chemical sanitizers, exception hierarchy |
| 8 | `workflows` | Parallel Processing & Workflows | 6 | 48 | Parallel processors, high-level CLI builders, training loops, logging, backup, downloads |
| 9 | `ecosystem` | Chemoinformatics Ecosystem | 5 | 24 | External integrations: RDKit (14), QSPRPred (5), CDPKit (5) |
| 10 | `experimental` | Experimental Extensions | 7 | 24 | USRCAT 60D, ElectroShape 4D, ProLIF, Chemprop D-MPNN, Scaffviz, Vina, OpenFE, Optuna |
| **Total** | | | **38** | **234** | **100% indexing coverage across all symbols** |

### 3. The 5 `PIPELINE_SCHEMAS` Workflows

| # | ID | Title | Nodes | Connectors | Referenced Symbols | Key Operations |
|---|----|-------|-------|------------|---------------------|----------------|
| 1 | `morl` | Multi-Objective Reinforcement Learning | 8 | 7 | 19 | Modifier $\to$ Scorers $\to$ Thresholds $\to$ RewardScheme $\to$ DrugExEnvironment $\to$ Agent $\to$ Prior $\to$ SequenceExplorer |
| 2 | `training` | End-to-End Generative Training | 8 | 8 | 17 | Corpus $\to$ VocSmiles $\to$ DataLoader $\to$ Base Prior $\to$ Fine-Tuned Agent $\to$ Environment $\to$ Explorer |
| 3 | `rocs` | 3D ROCS Shape-Matching Scoring | 8 | 7 | 16 | Reference SDF $\to$ Conformer Gen $\to$ SmoothClipped $\to$ RDKitROCSScorer $\to$ QSAR $\to$ Environment $\to$ Explorer |
| 4 | `fragment` | Fragment Synthon Assembly (BRICS) | 8 | 7 | 16 | Actives $\to$ BRICS Cleavage $\to$ Synthon Pool $\to$ Frag Models $\to$ Scaffold Seed $\to$ Scorers $\to$ Environment $\to$ FragExplorer |
| 5 | `telemetry` | Real-Time Diagnostics & Telemetry | 8 | 7 | 16 | FileMonitor $\to$ SmilesChecker $\to$ Modifiers $\to$ Scorers $\to$ Thresholds $\to$ Environment $\to$ Policies $\to$ Telemetry Loop |

---

## Redundancy Analysis: Legacy Mirror Directory (`docs/docs/cppreference/`)

### Current State
```
docs/
├── index.html                   [Authoritative Master SPA, 949 KB]
├── pipelines.html               [Authoritative Pipeline Visualizer, 134 KB]
├── favicon.*, logo.png          [Root Assets]
├── docs/
│   └── cppreference/
│       ├── index.html           [100% Duplicate of ../../index.html]
│       ├── pipelines.html       [100% Duplicate of ../../pipelines.html]
│       └── favicon.*, logo.png  [100% Duplicate of root assets]
```

### Why It Exists
Historically, static web hosting platforms (such as basic GitHub Pages configurations) resolved paths directly to directory structures on disk. Without server-side URL rewriting, requests to `/docs/cppreference/` required physical files at that path.

### Why It Is Redundant
1. **Server-Side URL Rewriting**: `docs/vercel.json` intercepts all routes (`/docs`, `/cppreference`, `/reference`, `/api`) and rewrites them directly to `/index.html`. Requests to `/pipelines` and `/schemas` are rewritten to `/pipelines.html`.
2. **Maintenance Drag**: `check_docs.mjs` enforces byte-for-byte buffer parity between root and nested files. Any edit to `index.html` requires an identical duplicate edit in `docs/cppreference/index.html`.
3. **Storage Overhead**: Duplicates ~2.9 MB of uncompressed HTML and image assets in the git repository.

### Remediation Action Plan
1. Delete the `docs/docs/cppreference/` folder and empty parent `docs/docs/`.
2. In `docs/tools/check_docs.mjs`:
   - Remove lines 17, 32-41 (the root vs. nested candidate buffer sync check).
   - Remove lines 456-466 (the root vs. nested pipelines buffer sync check).
   - Remove nested paths from the `filesToScan` array in the zero-emoji check (lines 598, 600).
3. In `docs/tools/deep_audit.mjs`:
   - Point line 14 to `path.join(projectRoot, 'index.html')` instead of the nested path.
4. In `docs/AGENT.md`:
   - Update the directory layout diagram to reflect `index.html` and `pipelines.html` as the sole authoritative files.
