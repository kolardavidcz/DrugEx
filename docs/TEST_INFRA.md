# DrugEx Documentation Platform Test Infrastructure Specification

## Architecture Overview

The DrugEx Reference Platform (`drugex_edu/docs`) provides publication-grade API documentation, mathematical formulations, and interactive workflows for ligand-based de novo drug design. The platform is architected as a zero-build React 18 single-page application (SPA) styled after cppreference.com.

The platform operates under a strict Single Source of Truth architecture:
- Primary Master Reference SPA: `docs/index.html` (contains `DATABASE`, `LIBRARY_SECTIONS`, and embedded `PIPELINE_SCHEMAS`).
- Standalone Pipeline Visualizer: `docs/pipelines.html` (contains standalone `PIPELINE_SCHEMAS`).
- Legacy redundant mirror directories (`docs/docs/cppreference/`, `docs/cppreference/`) have been removed. Server-side path routing is handled directly via `docs/vercel.json`.

---

## 4-Tier Testing Methodology

The documentation platform is guarded by a comprehensive 4-tier automated test suite implemented in `docs/tools/check_docs.mjs`. The suite executes within Node.js without browser overhead, evaluating data structures in sandboxed execution environments.

### Tier 1: Document Structure & Sandboxed JS Execution

- **Objective**: Ensure structural validity of HTML templates and parseability of client-side React/Babel script payloads.
- **Verification Logic**:
  1. Validates standard HTML5 preamble (`<!DOCTYPE html>`) and root application mount element (`<div id="root"></div>`).
  2. Extracts the client-side `<script type="text/babel">` block from `docs/index.html` and `docs/pipelines.html`.
  3. Executes the script block inside an isolated `node:vm` context to extract data dictionaries (`DATABASE`, `LIBRARY_SECTIONS`, `PIPELINE_SCHEMAS`) without browser DOM dependencies.
- **Failure Conditions**: Missing HTML preamble, absent mount container, malformed script tags, or JavaScript syntax/evaluation errors.

### Tier 2: Reference Schema Conformance & Database Integrity

- **Objective**: Guarantee complete, rigorous API specifications across all registered symbols.
- **Verification Logic**:
  1. **Symbol Count & Taxonomy**: Asserts exactly 234 registered symbols in `DATABASE`:
     - DrugEx Core: 186 symbols (`drugex.*`)
     - Chemoinformatics Ecosystem: 24 symbols (`rdkit.*`, `qsprpred.*`, `cdpkit.*`)
     - Experimental Molecular Techniques: 24 symbols (`experimental::*`)
  2. **Symbol Kinds**: Asserts exactly 182 classes and 52 functions.
  3. **Mandatory Field Contracts**: Every symbol entry must provide:
     - `synopsis`: Complete Python signature with explicit type annotations.
     - `description`: Academic and operational description explaining underlying computational mechanics.
     - `notes`: Rigorous technical notes detailing tensor shapes, mathematical formulations, boundary conditions, and edge cases.
     - `example`: Dictionary with executable `code` and authentic `output`.
     - `seeAlso`: Array of valid cross-reference keys.
  4. **Class Constructor & Member Hygiene**:
     - Constructors (`__init__`) must reside exclusively in the header `synopsis`, never in the `members` array.
     - Every class must specify at least one documented member method (`members.length >= 1`).
     - Zero-argument constructors must specify an empty `parameters` array.
     - Class attributes must be explicitly documented in `variables`.
- **Failure Conditions**: Symbol count != 234, missing required fields, `__init__` in member functions, empty members on classes, or mismatched constructor parameters.

### Tier 3: Thematic Indexing & Helper Function Parity

- **Objective**: Validate navigation hierarchy and UI presentation helpers.
- **Verification Logic**:
  1. **10 Thematic `LIBRARY_SECTIONS`**:
     - Verifies 10 top-level libraries across 38 subsections (`morl`, `neural`, `explorers`, `scoring`, `modifiers`, `data`, `molecules`, `workflows`, `ecosystem`, `experimental`).
     - Asserts 100% indexing coverage: all 234 symbols indexed with exactly 0 duplicates and 0 omissions.
  2. **Helper Function Contracts**:
     - `getSymbolTag(key, item)`: Accurately formats platform badges (`60D`, `4D`, `RDKit`, `GPU`, `v3.4`).
     - `isAbstractSymbol(key, item)`: Identifies exactly 30 abstract base classes and interfaces, strictly excluding `Exception` classes.
     - `getEffectiveMembers(item)`: Filters constructor methods and guarantees that regular methods strictly precede magic dunder methods.
     - `getMagicMethodOperator(name)`: Maps 7 Python magic dunder methods (`__init__`, `__call__`, `__getitem__`, `__setitem__`, `__len__`, `__iter__`, `__contains__`) to operator tokens.
     - `getMicroVizType(key)`: Dispatches interactive SVG diagrams for 27 key algorithmic symbols across 4 visualization types (`pareto_crowding`, `pareto_tanimoto`, `modifier_curve`, `rl_policy_cycle`).
- **Failure Conditions**: Unindexed or duplicate symbols in `LIBRARY_SECTIONS`, abstract class count != 30, incorrect method ordering, or missing micro-visualization mappings.

### Tier 4: Cross-Link Graph Integrity, Pipeline Schema Parity & Zero-Emoji Prohibition

- **Objective**: Enforce graph referential integrity, workflow visualizer consistency, and clean publication aesthetics.
- **Verification Logic**:
  1. **Cross-Link Graph Integrity**:
     - Evaluates every target key inside `seeAlso` arrays across all 234 symbols.
     - Asserts that every target key resolves to an existing, registered entry in `DATABASE` (0 broken links across 490+ references).
  2. **Pipeline Schema Parity**:
     - Evaluates the 5 canonical workflow schemas (`morl`, `training`, `rocs`, `fragment`, `telemetry`).
     - Asserts exact parity between embedded schemas in `index.html` and standalone schemas in `pipelines.html`.
     - Validates that each schema contains exactly 8 nodes and >= 5 connectors.
     - Enforces canvas boundary coordinates: all nodes (`x + 220 <= 1130`, `y + 98 <= 355`) and connectors must strictly fit inside the 1130x355 canvas.
     - Verifies that all referenced symbols in `drugexSymbols` resolve to valid keys in `DATABASE`.
  3. **Strict Zero-Emoji Prohibition**:
     - Scans `index.html`, `pipelines.html`, and audit scripts against Unicode emoji regex ranges:
       `[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F600}-\u{1F64F}\u{1F1E6}-\u{1F1FF}]`
     - Asserts exactly 0 emojis across documentation files.
- **Failure Conditions**: Broken cross-links in `seeAlso`, out-of-bounds canvas nodes, unmapped pipeline symbols, schema divergence between pages, or any detected emoji character.

---

## Tier 5: Adversarial Hardening & Forensic Verification (M_FINAL)

In milestone M_FINAL, the test suite executes deep adversarial and forensic audits:
- Ground-truth AST comparison against `repo/drugex` using `docs/tools/deep_audit.mjs`.
- Verification of edge-case failure modes:
  - Unparseable SMILES handling (`Valid = 0`, zeroed score vectors).
  - Multifragment dot filtering (`no_multifrag_smiles = True`).
  - Empty candidate batches (`Environment.getRewards` returning zero vector without crashing).
  - Conformer generation 20-second timeout enforcement.
  - Out-of-domain applicability fallback for QSPR models.
- Independent forensic integrity audit to guarantee all documentation enhancements reflect genuine codebase logic without dummy facades.

---

## Audit Execution Commands

All tests must be run via the command line and pass with 0 errors before staging or committing changes:

```bash
# Execute Primary Master Audit Suite (Tiers 1-4)
cd /home/kolar/build_projects/drugex_edu/docs
node tools/check_docs.mjs

# Alternative npm/pnpm test script alias
pnpm test

# Run Deep Codebase AST Comparison Audit
node tools/deep_audit.mjs

# Launch Local Preview Server (DevPort 34150)
pnpm dev
```

---

## Metric & Coverage Targets

| Metric | Target | Current Status | Enforcement Tier |
|---|---|---|---|
| Total Registered Symbols | 234 | 234 | Tier 2 |
| Core DrugEx Symbols | 186 | 186 | Tier 2 |
| Chemoinformatics Ecosystem Symbols | 24 | 24 | Tier 2 |
| Experimental Molecular Techniques | 24 | 24 | Tier 2 |
| Classes with Constructor Hygiene | 182 / 182 (100%) | 182 / 182 (100%) | Tier 2 |
| Library Sections Indexing Coverage | 234 / 234 (100%) | 234 / 234 (100%) | Tier 3 |
| Library Indexing Duplicates | 0 | 0 | Tier 3 |
| Identified Abstract Base Classes | Exactly 30 | Exactly 30 | Tier 3 |
| SVG Micro-Visualized Symbols | 27 | 27 | Tier 3 |
| Broken Cross-Links in `seeAlso` | 0 | 0 | Tier 4 |
| Canonical Pipeline Schemas | 5 | 5 | Tier 4 |
| Pipeline Canvas Bounds Violations | 0 | 0 | Tier 4 |
| Pipeline Unmapped Symbol References | 0 | 0 | Tier 4 |
| Unicode Emojis in Documentation | 0 | 0 | Tier 4 |
