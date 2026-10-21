# Agent Guide: `_david/` Workspace

Definitive operational manual and architecture guide for AI agents working in `/home/kolar/build_projects/drugex_edu/repo/_david`.

---

## 1. Mission, Scope & Academic Rigor

- **Academic Context**: Bachelor Thesis — *Ligand-Based De Novo Drug Design (LBDD) implementation in DrugEx*.
- **Quality Mandate**: **Bulletproof engineering**. Zero broken tests, zero unhandled exceptions, zero lint errors, zero synthetic/mocked fallbacks, and zero unscientific heuristics.
- **Pure LBDD Principle**: Strictly 100% ligand-based drug design centered around an arbitrary seed query ligand (defaulting to EGCG: `MY_MOLECULE`).
  - **Strict Prohibition**: Zero receptor atomic coordinates, zero docking grids, zero crystal complexes, and zero structure-based interaction fingerprints (ProLIF / PLIF).
- **Core Scientific Objective**: Discover novel, synthetically accessible modulators that preserve the 3D electrostatic pharmacophore, steric envelope, and volume of the query seed while escaping toxicophoric liabilities and structural degradation.

---

## 2. Core Academic Thesis Architecture & Scientific Decisions

### 1. Two Decoupled Generative Regimes
- **The Information Destruction Paradox**: The legacy design forced an intermediate "Chemotype Funnel" gating step between Sequence-RNN and Graph-Transformer. Decomposing early hits into bare Murcko scaffolds discarded functional pharmacophoric groups and created a fragile cliff (0-survivor crashes on realistic distributions).
- **Our Solution**: Two decoupled, independent generative regimes:
  - **Regime 1 (Sequence-RNN Pure RL)**: Unconstrained global hit exploration in SMILES space optimizing multi-objective reward against the 3D reference query.
  - **Regime 2 (Graph-Transformer Scaffold RL)**: Fragment-conditioned lead optimization directly seeded by the active query ligand's Bemis-Murcko core scaffold (from Stage 0).
- **Academic Impact**: Both models explore chemical space concurrently without blocking each other. Neither regime is held hostage by intermediate filtering.

### 2. Elimination of the Binary Chemotype Funnel Gate
- **The Cliff Filter Vulnerability**: A cascading binary filter ($ROCS \ge 0.38 \land ElectroShape \ge 0.20 \land RA \ge 0.50 \dots$) creates artificial discontinuities and catastrophic attrition.
- **Our Solution**: Post-generation continuous geometric desirability ranking (`compute_composite_desirability`):
  $$D = \left( d_{\text{ES}}^{0.30} \times d_{\text{RA}}^{0.25} \times d_{\text{QED}}^{0.25} \times d_{\text{Size}}^{0.20} \right) \times \text{NoveltyPenalty}$$
- **Academic Impact**: Replaces fragile gating with smooth continuous ranking. Top-K elite leads are prioritized, evaluated with physical ROCS, and clustered via Butina sphere exclusion without arbitrary cliff crashes.

### 3. Tiered 3D Conformer Scoring & Ground-Truth Realities
- **75-D ElectroShape as Ultrafast Screening Pre-Filter**: 75-D ElectroShape moments are coarse 1D centroid projections ($<0.08\text{ ms}$ per conformer). They are reserved strictly for high-throughput Tier 1 RL rollouts to prevent uncharged lipophilic decoys from gaming volume metrics.
- **Physical Gaussian Overlay (ROCS) as Ground-Truth Evaluation**: Ground-truth evaluation and lead selection require authentic physical Gaussian volume overlap (CDPKit / RDKit ROCS) evaluated across multi-isomer ensembles (`max_isomers=4`, evaluating stereoisomers) with MMFF94 force-field relaxation within a thermodynamic Boltzmann window ($\Delta E \le 8.0\text{ kcal/mol}$).

### 4. Dynamic Size Envelope & Universal Multi-Seed Generalization
- **Dynamic Size Plateau**: Rather than hardcoding heavy atom bounds (e.g. 24–42 HAC), bounds are dynamically scaled relative to any query seed molecule:
  $$\text{min\_hac} = \max(6, \text{int}(\text{seed\_hac} \times 0.50)), \quad \text{max\_hac} = \text{int}(\text{seed\_hac} \times 1.50)$$
- **Deterministic Hashing**: Seed conformers and moments are resolved and cached via deterministic SHA-256 hashes (`get_seed_hash(smiles)`), ensuring bulletproof isolation across arbitrary active query templates.

---

## 3. DrugEx Native Library Parity & Audit

To prevent redundant reimplementation and maintain architectural purity, our code directly reuses native DrugEx v3 APIs:

| Native DrugEx API / Class | Usage in `_david/` | Audit Note / Justification |
|---|---|---|
| `drugex.training.explorers.SequenceExplorer` | Stage 1 policy gradient RL exploration | Direct instantiation; redundant subclass wrappers eliminated. |
| `drugex.training.explorers.FragGraphExplorer` | Stage 2 fragment-conditioned RL optimization | Direct instantiation conditioned on seed core scaffold. |
| `drugex.training.scorers.modifiers.SmoothHump` | Dynamic size envelope & property thresholds | Native Gaussian-bell plateau modifier replacing custom math. |
| `drugex.training.scorers.properties.Property` | Heavy atoms, QED, SA score evaluation | Native DrugEx property calculators. |
| `drugex.training.scorers.ra_score.RAScoreModel` | Synthesizability assessment | Native forward-synthesis predictor model. |
| `drugex.molecules.converters.dummy_molecules.dummyMolsFromFragments` | Murcko scaffold fragment conversion | Native DrugEx fragment encoder; constructs graph fragments from intact Murcko cores. |
| `drugex.training.scorers.interfaces.Scorer` | Base class for `USRCATScorer` | Cleanly implements `getScores()` and `getKey()`. |
| Upstream Bug Fix in `drugex/training/scorers/properties.py` | Fixed `mol.GerNumAtoms()` typo | Corrected to `mol.GetNumHeavyAtoms()` and guarded division by zero in `LigandEfficiency`. |

---

## 4. Directory Architecture & Decoupled Stages Map

```
_david/
├── run.py                          # 1-Click execution entrypoint (zero CLI args needed)
├── monitor.py                      # Live terminal telemetry dashboard (`--watch`)
├── tests.py                        # Empirical unit test suite (27 passing tests)
├── in_molecule.txt                 # Input query molecule SMILES / configuration
├── AGENTS.md                       # Definitive agent manual & architectural spec
├── dodo_notes.md                   # Academic research notes & failure mode audits
├── code/                           # Modular implementation package
│   ├── __init__.py                 # Re-exports public classes
│   ├── pipeline.py                 # Multi-stage orchestrator (Stages 0 -> 4)
│   ├── SequenceRNN.py              # Stage 1: Sequence-RNN global hit generator (Regime 1)
│   ├── GraphTransformer.py         # Stage 2: Fragment-conditioned lead optimizer (Regime 2)
│   ├── scaffold_funnel.py          # Stage 3: Top-K lead selection & diversity clustering
│   ├── thesis_benchmark.py         # Stage 4: Publication benchmark suite (ROCS, PCA, radar)
│   ├── fallback_monitor.py         # Observable fallback alert & audit registry
│   ├── monitor.py                  # Live telemetry, GPU stats, and convergence dashboard
│   ├── usrcat_scorer.py            # 75-D ElectroShape & 60-D USRCAT 3D moments scorer
│   ├── anti_cheating_rewards.py    # Augmented Chebyshev & Geometric Mean scalarizers
│   ├── prepare_query_conformer.py  # Stage 0: 3D reference conformer & moment cache generator
│   └── visualize_molecular_space.py# Chemical space manifold & property projectors
└── output/
    └── <YYYYMMDD_HHMMSS>/          # Timestamped run output directory
        ├── pipeline.log            # Complete execution log
        ├── pipeline_status.json    # Live stage progress & GPU telemetry
        ├── pipeline_summary.json   # Final execution summary & metrics
        ├── fallback_audit.json     # Comprehensive fallback audit record
        ├── references/             # 3D reference SDF & seed core scaffold TSV (Stage 0)
        ├── stage1_sequence_rnn/    # Stage 1 models & generated candidate TSV (Stage 1)
        ├── stage2_graph_transformer/# Stage 2 models & optimized lead TSV (Stage 2)
        ├── stage3_lead_selection/  # Prioritized elite leads & diversity clusters (Stage 3)
        └── benchmark/              # Publication radar charts, PCA plots, benchmark TSVs (Stage 4)
```

---

## 5. Telemetry, Dashboard & Fallback Monitoring Architecture

### Centralized Fallback Monitor (`_david/code/fallback_monitor.py`)
- **Design**: Thread-safe (`threading.RLock()`) central auditor recording every heuristic relaxation.
- **Observable Banners**: Emits high-visibility alert boxes to `stdout` and logs warnings via `DrugEx.FallbackMonitor`:
  ```
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  [FALLBACK MONITOR ALERT] Category: <fallback_category>
  Reason: <explanation>
  Context: <metadata>
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ```
- **Monitored Categories**:
  - `conformer_embedding_random_coords`: Standard ETKDGv3 failed; relaxed distance geometry with random coordinates.
  - `conformer_embedding_failure`: Molecule failed all embedding passes; zero-score assigned.
  - `rocs_cdpkit_failed`: CDPKit ROCS encountered an exception; fell back to RDKit ROCS.
  - `rocs_all_engines_failed`: Both CDPKit and RDKit ROCS engines failed; fell back to default 0.50.
- **Audit Export**: Persisted to `fallback_audit.json` in the run directory for thesis integrity validation.

### Telemetry Dashboard (`_david/code/monitor.py` & `_david/monitor.py`)
- **Execution**: Run `python _david/monitor.py` for one-shot check or `python _david/monitor.py --watch` for 2s live refresh.
- **Tracked Metrics**:
  - Run ID, execution mode, elapsed time, current active stage.
  - Hardware & GPU telemetry: CUDA availability, GPU device name, memory allocated (MB), memory reserved (MB), total memory (MB) on NVIDIA RTX PRO 4000 Blackwell.
  - Per-stage execution status (`SUCCESS`, `RUNNING`, `FAILED`).
  - RL Convergence: Latest epoch, `loss_train`, `valid_ratio`, `unique_ratio`, `desired_ratio` from `david_agent_fit.tsv`.
  - Fallback counts and recent log tail.

---

## 6. Conformer Generation & Bounded Distance Geometry

- **Deadlock Mitigation**: Strained polycyclic ring systems generated during early exploratory RL epochs can cause RDKit C++ triangle bounds smoothing routines to hang indefinitely.
- **Bounded ETKDGv3 Implementation** in `_david/code/usrcat_scorer.py`:
  - Primary pass: `AllChem.ETKDGv3()`, `maxIterations=200`, `randomSeed=42`.
  - Fallback pass: `useRandomCoords=True`, `maxIterations=50`, capped at 3 conformers.
  - Multi-isomer evaluation: `max_isomers=4` with MMFF94 force-field relaxation (UFF fallback).
  - Catastrophic fallback: If conformer generation fails completely across all isomers, assign score 0.0 and record event in `FallbackMonitor`. Never hang or block worker threads.
- **Reference Ensemble (Stage 0)**: Generates 50 energy-minimized conformers (MMFF94) for the seed molecule and precomputes 75-D ElectroShape moments cached as `query_molecule_3d_electroshape.npy` for sub-millisecond retrieval.

---

## 7. Unit Test Suite & Empirical Verification

- **Test Suite**: `_david/tests.py` containing **27 empirical unit tests**.
- **Execution**:
  - From repository root: `python _david/tests.py` or `.venv/bin/python _david/tests.py`.
  - Direct execution supported via `sys.path` bootstrap.
- **Key Test Coverage**:
  1. `test_smooth_hump_modifier`: Validates Gaussian plateau response.
  2. `test_scaffold_catechol_penalty`: Validates toxicophore screening.
  3. `test_augmented_chebyshev_anti_cheating`: Verifies severe penalty on 3D-collapsed cheaters.
  4. `test_geometric_mean_reward`: Verifies weighted geometric mean calculations.
  5. `test_dummy_mols_from_fragments`: Verifies native Murcko fragment encoding.
  6. `test_edge_cases`: Verifies empty arrays, NaNs, and 1D score matrices.
  7. `test_dynamic_size_scaling`: Verifies HAC envelope adaptation across different seeds.
  8. `test_universal_environment_defaults`: Verifies 5 default SmallMolecule objectives.
  9. `test_scaffold_funnel_gated_execution`: Verifies candidate pre-filtering before clustering.
  10. `test_fallback_monitor_recording_and_audit`: Verifies thread-safe alerts and JSON export.
  11. `test_backward_compatibility_imports`: Verifies aliased root imports (`from _david import ...`).
  12. `test_one_click_runner_interface`: Verifies `_david/run.py` main callable.
  13. `test_prepare_regime_dataframe_with_frags`: Verifies column safety (no FRAGS/RAscore collision).
  14. `test_ligand_efficiency_scorer_handles_strings_and_mols`: Verifies upstream LE bug fix.
  15. `test_scaffold_funnel_column_matching_isolation`: Verifies exact token matching (ROCS vs ElectroShape).
  16. `test_usrcat_scorer_companion_npy_loading`: Verifies precomputed `.npy` moment cache loading.
  17. `test_thesis_benchmark_ref_sdf_and_seed_isolation`: Verifies arbitrary query seed propagation.
  18. `test_pipeline_benchmark_only_creates_summaries`: Verifies end-to-end report generation in benchmark mode.
  19. `test_scaffold_funnel_attrition_tracking`: Verifies `funnel_attrition.json` creation and schema.
  20. `test_thesis_benchmark_regime_deduplication`: Verifies distinct decoupled regime loading and duplicate suppression.
  21. `test_seed_hash_and_core_scaffold_extraction`: Verifies deterministic seed hashing and core extraction.
  22. `test_composite_desirability_continuous_ranking`: Verifies smooth continuous desirability ranking.
  23. `test_monitor_dashboard_display`: Verifies dashboard rendering without exceptions.
  24. `test_usrcat_scorer_strained_molecule_fast_embedding`: Verifies rapid bounded ETKDGv3 on strained polycycles.
  25. `test_usrcat_scorer_batch`: Verifies batch evaluation of molecular lists.

---

## 8. Execution Workflows & Operational Runbook

### 1-Click Execution (`_david/run.py`)
- **Full Production Pipeline**:
  ```bash
  .venv/bin/python _david/run.py
  ```
- **Rapid Smoke Test Mode** (1 epoch, 50 samples):
  ```bash
  .venv/bin/python _david/run.py --test
  # or via environment variable:
  DRUGEX_RUN_MODE=test .venv/bin/python _david/run.py
  ```
- **Custom Query Molecule**:
  ```bash
  .venv/bin/python _david/run.py --molecules "c1ccccc1NC(=O)c1ccc(cn1)C1CC1"
  ```
- **Benchmark-Only Mode** (generates publication plots on existing outputs):
  ```bash
  .venv/bin/python _david/run.py --mode benchmark_only
  ```

### Remote Server Operations (`dichtator`)
- **Host**: `dichtator` (`kolarv@147.33.72.143`), project root `/home/kolarv/DrugEx`.
- **Python**: `/home/kolarv/DrugEx/.venv/bin/python`.
- **Zero Cisco VPN Disruption**:
  - Reuses WSL active `ControlMaster` socket via `wsl ssh -o BatchMode=yes dichtator ...`.
  - Always probe connection first: `wsl ssh -o BatchMode=yes dichtator "echo OK"`.
  - Never restart or disrupt the active Cisco VPN connection.
- **Syncing Code**:
  ```bash
  wsl rsync -avz --exclude "__pycache__" _david/ dichtator:/home/kolarv/DrugEx/_david/
  ```
- **Remote Test Execution**:
  ```bash
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python _david/tests.py"
  ```

---

## 9. Agent Coding Standards & Hard Constraints

1. **Python Type Annotations**: All parameters, return values, and attributes must have full, explicit type annotations (`def func(param: Type, ...) -> ReturnType:`).
2. **Rich IDE Docstrings**: Structured Google / NumPy docstrings with summary, `Args` / `Parameters`, and `Returns` for every class, method, and function.
3. **Compact `#? [Library]` Annotations**: Every external non-DrugEx library invocation (RDKit, NumPy, Pandas, Matplotlib, SciPy, Scikit-learn) must include a 1–3 line explanatory comment.
4. **No Ad-Hoc Modification Scripts**: Always modify and create files using native file tools (`write_to_file`, `replace_file_content`). Never execute one-liners or throwaway scripts to patch project code.
5. **Zero Broken Tests / Clean Compilation**: Always verify `python3 -m py_compile` and unit test passes before finishing tasks.
6. **Automatic Local Commits**: Stage and commit changes locally with a descriptive summary:
   ```bash
   git add <modified_files> && git commit -m "<descriptive summary>"
   ```
   **Never run `git push`**.
