# Conformer study: status and evidence

**2026-11-02 (1:1 Training-Interface Cost & Early-Accuracy Calibration — `rl_batch_calibration.py`):**
- **Why**: the reward-Pareto x-axis (isolated serial single-molecule calls) is *not* RL epoch time; the live-RL reference showed ≈3.2× underestimation. Decision: benchmark time × ROCS accuracy through the **exact production training interface** — `SequenceExplorer.fit → env.getRewards → env.getScores` scoring the **full epoch's unique batch (~500 molecules, not the 64-molecule RNN gradient mini-batch)** via `create_environment → ConfigurableRDKitROCSScorer` (native parallel embedding + alignment; `WholeCallScorer` is not in the production path). Reference-bank preparation runs untimed in the parent; per-arm reference loading happens before timing patches install.
- **Design**: per-method arms (method:K:C, cheapest-first `DEFAULT_ARMS`), each in a **fresh forkserver subprocess** on the **same frozen prior-RNN batch** (paired inputs); phase 1 = `--repeats` full-batch scoring calls with nested stage clocks (gen/minimization ⊂ ROCS ⊂ env ⊂ wall) and cold/warm separation; phase 2 = `--epochs`-epoch RL pilot through the real `explorer.fit` with sprint-H1 `PipelineConfig` parity (2-objective [ROCS, SA], pareto_crowding, eps=0.05, tanimoto_combo raw, cpu_ratio=0.90, seed 42); incremental per-arm `manifest.json` + re-render (a stopped run stays inspectable); resumable via per-arm `complete.json`.
- **Fairness guards (the "minimized arm must not look faster than unminimized" requirement)**: paired identical inputs, no cross-repeat caching, worker parity pinned to 0.90 in both phases (`HardwareManager.init_runtime_environment` before `create_environment`; `cpu_ratio=config.cpu_ratio` in the phase-2 optimizer), reference counts recorded per row (K=0 all-bank arms keep different bank sizes — visible, not hidden), and dropped-conformer counters (`embedding_failures`, `identity_rejections`, `filtered_molecules`, `minimization_attempts/failures`) per timing row so an arm that drops work cannot silently fake a speedup.
- **Peer review before launch** (user-mandated panel): Gemini 3.8 Flash — YES WITH FIXES (6 findings: incremental render/manifest, monitor-after-model-load, cpu_ratio parity, references column, accumulator-based stage split, stale-TSV unlink — all applied); GLM-5.3 — YES WITH FIXES (timeout 10→30 s sprint parity, the dropped-conformer counters above, resume-cache validation, manifest "complete"-only-when-all-arms, PBS scratch rsync-in for resume — all applied; walltime-splitting of the k=all arms on PBS deferred to the 10:00 checkpoint with the user). Qwen 3.8-27B — YES / YES WITH FIXES (both copies: no blockers; no-op `assertTrue`→`assertEqual` fixed, duplicate PBS resume-rsync block removed, "pools persist" docstring claim corrected to per-call pool re-creation, smoke-provenance caveat below; deferred: `last_embedding_failures` counts SDF records lacking conformers, not pre-record tautomer/stereo failures — fairness unaffected by paired inputs). DeepSeek (two independent copies) — both YES WITH FIXES for the local run: (i) fork-after-torch hazard in the parent (`sample_prior_batch` initializes torch, then `prepare_shared_references` opened a fork pool; PBS exported forkserver before Python so only the local run was exposed) — fixed by forcing `DRUGEX_MP_CONTEXT=forkserver` at the top of `main()` before any torch use (utils only sets it when unset, so the early assignment wins); (ii) C1 matched-pair gap — no arm pair held (K, C) fixed, so minimization cost was not attributable; adopted by adding `mmff_high_dielectric:1:30` as the third overnight arm (identical K/C to `etkdg:1:30`, minimization only); (iii) E1 paired-accuracy labeling — the paired n=500 accuracy comparison already exists via phase-1 `batch_scores_repeat*.csv` on the identical frozen batch; epoch-1..5 50-molecule validation rows are to be labeled **training-log artifacts**, not accuracy comparisons. Deferred (with user, 10:00 checkpoint): per-arm `--max-arm-seconds` watchdog, `run_key` resume-config guard, m0/m1 arm-spec extension, `setup_seconds` split, `cpu_seconds`→`parent_cpu_seconds` rename, `parse_arms` negative-cap validation.
- **Consolidated panel verdict (7 review copies: Gemini 3.8 Flash, GLM-5.3, 3× Qwen 3.8-27B, 2× DeepSeek)**: **all reviewers agree the local overnight run may proceed** (every verdict YES or YES WITH FIXES; all requested pre-launch fixes applied and validated). The single NO applies only to the PBS job *as originally written*: Qwen verified live over SSH that the pretrained Papyrus05.5 checkpoint is absent from MetaCentrum storage (untracked in git; the job would die in `sample_prior_batch` before any arm), and the 6-arm grid needs ~250–280 h, not 25 h. Both are fixed in the script (model staged from `$STORAGE_DIR/models` + 30-second `resolve_pretrained_paths()` probe; conservative 3-arm default) — **qsub still awaits explicit user confirmation at the 10:00 checkpoint**.
- **Smoke-provenance caveat (Qwen)**: the smoke-derived per-molecule costs (3.07/4.44 s/mol) came from revision `5293122` with `timeout_seconds=10` and n=48; the overnight run uses the timeout-30 / n=500 final config. Timeout only affects pathologically slow embeddings (>10 s), so the estimate carries over — but the smoke output must not be read as final-config data.
- **Reference-protocol disclosure (Qwen, mandatory for write-up)**: benchmark arms score against the R50/I8 reward-calibration banks, not the sprint's R100/I1 MMFF-minimized production preparer. The `etkdg:1:30` arm is therefore **candidate-side H1 parity only** — its K=1 reference is the first record of the unminimized ETKDG bank, not H1's lowest-energy MMFF conformer from 100 attempts. Absolute `mean_rocs` is not comparable to sprint H1 logs; timing is unaffected (K counts recorded per row). Disclosed per-arm in `complete.json:reference_bank_protocol`.
- **Notebook scope decision (from smoke timing ≈3.07 s/mol etkdg K1/C30, ≈4.44 s/mol mmff_high_dielectric K4/C20)**: overnight CPU run = **3 arms** in order `etkdg:1:30,mmff_high_dielectric:4:20,mmff_high_dielectric:1:30` — the two production-decision arms first, then DeepSeek's C1 matched-pair arm (mmff K1/C30 vs etkdg K1/C30; runs last so it is cancelable at the 10:00 checkpoint with zero loss of the primary data), samples=500, **repeats=2**, epochs=5 (arms 1–2 ≈ 7.5 h; arm 3 +~4–5 h, may spill past 10:00; incremental per-arm rendering keeps partial results usable). *Record correction*: commit `6777317`'s message says "repeats 2→3"; the final decision is **repeats=2** (Qwen's own condition for a third repeat was window headroom, which a 02:35 launch does not have — the third arm already consumes the slack). Full 6-arm grid deferred to MetaCentrum PBS (`scripts/rl_batch_calibration.sh`, 4 CPUs, gpu_long, 25 h, resumable, now with conservative 3-arm default + model staging). 5-epoch pilots are **early trajectories for engineering cost comparison**, not final learning comparisons — accuracy ordering of the top-5 methods needs ≥3 seeds or ≥20 epochs before any claim.
- **Pre-registered accuracy metric (DeepSeek E1/C2, fixed before launch)**: gold arm for post-hoc rank-agreement analysis = **`mmff_high_dielectric_k4_c20`**. From the per-molecule phase-1 CSVs (identical frozen 500-molecule batch per arm): Spearman ρ, top-10% overlap, and per-molecule ΔROCS of each cheaper arm against the gold. Mean/median ROCS *levels* are not accuracy (they rise mechanically with K); the epoch trajectory columns are training-log artifacts.
- **Deliberate deviations from sprint H1 (disclosed)**: (1) `training_seed=42` (production ran unseeded; makes arms paired — same epoch-1 molecules across arms); (2) `torch.set_num_threads(1)` (production used default threads; inflates `rnn_residual` slightly, so ROCS share of epoch seconds is a lower bound); (3) R50/I8 reference banks (see reference-protocol disclosure above — K=1 arm is candidate-side parity only); (4) `DRUGEX_SCORE_MOL_TIMEOUT=180` on the overnight run (cheap watchdog against alignment-worker spins; production ran unbounded); (5) 5 epochs/min_epochs=5/patience=6 (pilot truncation).
- **Also restored**: `NAMED_OPTIMIZATION_METHODS` (rocs.py) to the historical 4-method set {etkdg, uff, mmff_vacuum (ε=1.0), mmff_high_dielectric (ε=78.4)} — `mmff_pocket` (ε=6.5) had been accidentally swapped in for `mmff_vacuum`; `reference_bank.METHODS/LABELS` re-derived cleanly (no executable code depends on a 5-method set; `mega.py` has an independent positional tuple).
- **Validation**: 91/91 tests (14 in `_david/tests/test_rl_batch_calibration.py`: nested-clock purity, warm/cold split, stage sums, fairness counters, NaN guard, parity), `ruff check _david` clean, `bash -n` clean, `--render-only` verified on synthetic new-format data.

**2026-11-01 (Pareto RL-Timing Correction & Figure Cleanup):**
- **Measured calibration→RL timing gap (why the Pareto x-axis was off)**: the per-input seconds plotted on the reward-Pareto come from isolated serial single-molecule `getScores` calls (`n_jobs=1`, fresh factory per cell) on DrugCentral development drugs. The live-RL reference on the same host and setting (`dichtator` smoke, MMFF94s ε78.4 K4/C20, 2 workers, 17 rows) measured **6.07 s/row serial-equivalent** (3.15 s generation/minimization + 2.92 s alignment/I/O) against the calibration mean of **1.88 s/input** → the old axis underestimated real RL cost by **≈3.2×**. Root causes: RNN-prior molecules carry 0–2 unassigned stereocenters (I≤4 enumeration multiplies embedding work) and 16–34 heavy atoms, unlike the rigid stereo-specified development drugs; the plotted mean hides the straggler tail that bounds an RL epoch's wall-clock (calibration p95 = 5.28 s); and one randomly shuffled cell per molecule worker paid the per-process rdShapeAlign/MMFF warm-up.
- **Code fixes** (`reward_calibration.py`, `visualization.py`): (1) untimed warm-up scoring call per molecule worker before the timed grid (warm-up seconds recorded in the checkpoint detail); (2) Pareto front and x-axis now use **p95 seconds** with faint mean markers, since RL epochs are bounded by their slowest worker chunk; (3) per-cell candidate-generation vs alignment/I/O cost split merged from per-row timings (handles the legacy `generation_seconds` column name) and shown as a third panel; (4) top axis projects RL epoch minutes (p95 × 500 samples ÷ 4 workers) and the banner prints the measured live-RL reference and its calibration-vs-RL factor, read from `raw_data/rl_smoke` when present (`rl_reference_measurement`), with the dichtator smoke numbers as documented fallback; (5) axis relabelled "isolated serial calls, NOT RL time". `render_reward_calibration` now derives the caps/candidates grid from the saved summary, fixing a `KeyError` that made `--analyze` crash on runs whose grid differs from current module constants. `pareto.png/svg` and `pareto.csv` in `20261029_1155_reward_calibration/` were re-rendered from the unchanged saved tables (no measurements changed).
- **Figure cleanup (user decision 2026-11-01)**: `drug_panel_stratified_vs_trimmed.png` regenerated **without panel D** (fixed-K vs `all - m` retained-reference boxplot — not applicable to real training, where the reference bank is frozen once by the pipeline); panel A now draws per-input p95 latency whiskers and the measured-RL banner. Removed from `20261029_egcg_omnibus_diagnostic/`: `egcg_omnibus_diagnostic.png`, `egcg_rmsd_tversky_receptor_critique.png`, `drug128_definitive_synthesis.png` (sloppy / not applicable to real training; `tversky_color_162grid_selectivity.png` retained). The original ad-hoc SSH heredoc that generated the omnibus/drug-panel figures was recovered from the session archive and preserved as `tmp/plot_egcg_omnibus_and_drugpanel_recovered_20261029.py`; the maintained regeneration path is the new `_david/code/conformer/panel_figures.py` (CLI, provenance manifest `drug_panel_figure_manifest.json` with input/source hashes). **Pending**: mirror the deletions and the regenerated figure on `dichtator` (unreachable, Cisco VPN down); an RL-realistic timing arm that re-times saved prior-RNN batch SMILES through the shortlisted cells is deferred until proper hardware is available.
- **Validation**: 77/77 tests (3 new in `_david/tests/test_conformer_figures.py`), `ruff check _david` clean.

**2026-10-30 (128-Drug Definitive Synthesis & Complete 162-Combo Tversky $(\alpha, \beta) \times \text{Shape/Color } (w_S, w_C)$ Selectivity Audit Across 14,641 Drug Pairs = 2.37M Evaluations):**
- **Outputs**: `_david/output/20261029_egcg_omnibus_diagnostic/` (`drug128_synthesis_grid.csv`, `drug128_cross_decoy_selectivity.csv`, `drug121_tversky_color_162grid.csv`, `drug121_allpairs_162grid_cache.npz`, `egcg_hard_decoys_162grid.csv`, `drug128_definitive_synthesis.png`, `tversky_color_162grid_selectivity.png`).
- **Complete 162-Combo ($18\text{ Tversky} \times 9\text{ Shape/Color}$) Selectivity & Asymmetric "Scissors Trap" (`121 x 121 = 14,641` drug pairs)**:
  - Stratifying the `13,150` unrelated cross-drug decoy pairs ($\text{2D Morgan Tanimoto} < 0.15$) into **Small-Fragment Decoys** ($V_{\text{Cand}} \le 0.65\,V_{\text{Ref}}$, $N=3,185$), **Bloated-Monster Decoys** ($V_{\text{Cand}} \ge 1.35\,V_{\text{Ref}}$, $N=4,251$), and **Hard Shape-Matched "Trojan Horse" RL Decoys** ($0.85 \le V_{\text{Cand}}/V_{\text{Ref}} \le 1.15$, $\text{Shape} \ge 0.68$, $\text{Color} \le 0.22$, $N=568$) proves why extreme Tversky ratios fail in opposite ways:
    - **Extreme Subshape `(0.2, 1.8)` ($\alpha+\beta=2$) / `(0.1, 0.9)` ($\alpha+\beta=1$)**: Crushes bloated decoys (`P95 = 0.3312`), but causes **Small-Fragment Decoy `P95` to explode to `0.6024` (`0.6854` on $\alpha+\beta=1$)** (on EGCG, 1-ring `Acetaminophen = 0.6234`, `Gallic Acid = 0.5462`), inviting **Fragment Collapse** in RL.
    - **Extreme Supershape `(1.8, 0.2)` ($\alpha+\beta=2$) / `(0.9, 0.1)` ($\alpha+\beta=1$)**: Crushes small fragments (`P95 = 0.2990`), but causes **Bloated-Monster Decoy `P95` to explode to `0.6107` (`0.6939` on $\alpha+\beta=1$)** while crushing compact active sub-scaffolds (`(-)-Epicatechin` drops from `0.8079` to `0.4540`), inviting **Monster-Blob Polymer Inflation** in RL.
    - **Goldilocks Minimax Saddle Zone ($\alpha + \beta = 2$, $\alpha \in [0.6, 1.2], \beta \in [0.8, 1.4]$ with Color-Heavy $w_S \in [0.20, 0.30], w_C \in [0.70, 0.80]$)**: Simultaneously suppresses all three decoy classes: at `sum2 (1.0, 1.0) | 0.2S+0.8C` and `sum2 (0.8, 1.2) | 0.2S+0.8C`, **Worst-Regime Decoy `P95` drops to `0.3030–0.3060` (`0.61%–0.82%` false positives $\ge 0.35$)**, and on the EGCG 12-decoy stress panel every fragment, size-matched steric blob, and bloated monster decoy is pushed down into **`[0.11, 0.24]`** while `EGCG` scores `0.7695` and active `(-)-Epigallocatechin`/`(-)-Epicatechin` score `0.6120`/`0.5379`.
- **128-Drug Reference Merging (`r = 0.35 Å` & `diverse` vs. `0.50 Å` and `1.25 Å`)**:
  - Across all 128 development drugs (`126` color-eligible, `121` with valid candidate pools; `MMFF94s` $\epsilon_r = 78.4$ references vs. `C=20` unminimized `ETKDG` candidates), legacy Butina `r = 1.25 Å` collapses 52/126 drugs to 1–2 conformers (`median = 4.0` clusters, `2.84` mean retained at $K=4$), restricting $K=4$ self-recovery to `0.8052` (`0.8233` at $K=\text{all}$).
  - Tightening Butina RMSD to **`r = 0.35 Å`** (RDKit `ETKDGv3` default) retains a median of **16.0 clusters** (`3.70` mean retained at $K=4$) and raises $K=4$ self-recovery to **`0.8160`** (`0.8436` at $K=\text{all}\le 12$).
  - **`diverse` (farthest-first 3D shape+color)** achieves the highest small-cap recovery at $K=2$ (`0.7993`), $K=3$ (`0.8106`), and $K=4$ (**`0.8172`**, retaining `3.99` conformers on average).
- **128-Drug Speed-Accuracy Frontier at $K=4$**:
  - Curated `MMFF94s` ($\epsilon_r = 78.4$) references + unminimized `ETKDG` candidates achieve `0.8172` at `C=20` (`1.36s` mean per drug) and `0.8274` at `C=30` (`1.89s` mean per drug).
  - Minimizing candidates with `MMFF94s` (`C=20`, `1.95s` helper time; ~7–10x slower per RL epoch) reaches `0.9477` (`diverse`) / `0.9453` (`r = 0.35 Å`), while unminimized references + unminimized candidates (`ETKDG` vs `ETKDG`) reaches `0.9129` only by matching unminimized against unminimized strain.
- **Validating O'Hagan & Kell (2016) (*Front. Pharmacol.* 7:266) on 6,776 3D Cross-Drug Decoy Pairs**:
  - **`tversky_03_07_sum1` ($\alpha=0.3, \beta=0.7, \alpha+\beta=1$ Dice scale)**: Inflates random unrelated cross-drug decoy scores to `0.4371` mean (`P90 = 0.5341`), causing **86.36% of random unrelated drug pairs to pass the `0.35` RL desirability threshold** and shrinking selectivity margin (`self_mean - decoy_mean`) to `0.4461`.
  - **`tversky_06_14_sum2` ($\alpha=0.6, \beta=1.4, \alpha+\beta=2$ Tanimoto scale)**: Preserves the exact $3:7$ subshape asymmetry ratio on the true Tanimoto scale, reducing random decoy mean from `0.4371` to **`0.3241`** (`38.53%` >= `0.35`) and restoring selectivity margin to **`0.4885`**.
  - **`tanimoto_03S_07C` (`0.3 Shape + 0.7 Color`) & `tversky_03_07_color_heavy` ($\alpha=0.6, \beta=1.4$ with `0.35S + 0.65C`)**: Achieve the **highest selectivity margins in the 128-drug study (`0.5533` and `0.5320`)** and cut false-positive decoy leakage at `0.35` down to **`2.89%` and `7.85%`** (compared to `74.59%` decoy leakage under `Shape-Heavy 0.7S + 0.3C`).

**2026-10-29 (EGCG 12,600-Cell Omnibus Diagnostic, Joint `Butina RMSD × (all - m)`, Receptor vs. ADME Poses, & `Tversky (0.3, 0.7)` Critique):**
- **Outputs**: `_david/output/20261029_egcg_omnibus_diagnostic/` (`egcg_omnibus_grid.csv`, `egcg_rmsd_x_all_minus_m.csv`, `egcg_receptor_vs_adme_coverage.csv`, `egcg_tversky_vs_tanimoto_panel.csv`, `egcg_omnibus_diagnostic.png`, `drug_panel_stratified_vs_trimmed.png`, `egcg_rmsd_tversky_receptor_critique.png`).
- **Joint `Butina RMSD (Å) × (all - m)` Finding**: On EGCG (`MMFF94s` water $\epsilon_r = 78.4$, $\Delta E \le 8.0\text{ kcal/mol}$, $86$ conformers), Butina clustering compresses the pool to $N=7$ clusters at $r \in \{0.20, 0.35\}\text{ \AA}$ (`0.8715` vs unminimized ETKDG candidates), $N=6$ at $r = 0.50\text{ \AA}$ (`0.8029`), and $N=3$ at $r = 1.25\text{ \AA}$ (`0.7989`). Because $N_{\text{clusters}}$ is already small ($3\text{–}7$) after RMSD merging, subtracting `all - 3`, `all - 5`, or `all - 10` (`max(1, N - m)`) deletes the higher-energy open/extended conformers first and collapses the reference bank toward $K=1$ (`0.7555`). Fixed caps $K \in \{3, 5\}$ with `selection_mode="diverse"` (farthest-first 3D shape+color diversity) or `rmsd_cutoff=0.35` strictly dominate `all - m`.
- **ADME Solution ($\epsilon_r = 78.4$) vs. Receptor Pocket Settling ($\epsilon_r = 6.5$, strained $\Delta E \in [1.8, 5.7]\text{ kcal/mol}$)**: A single global minimum ($K=1, \Delta E = 0$) covers only `0.5471` mean (`0.3386` min) of the 7 distinct binding-pocket conformer clusters, and energy-sorted Butina `0.50 Å` ($K=3$) covers only `0.7835` (`0.4868` min). Farthest-first `diverse` selection ($K=5$) reaches **`0.9023` mean (`0.8862` on strained $\Delta E \ge 1\text{ kcal/mol}$ poses, `0.6948` min)**.
- **`TanimotoCombo` vs. `Tversky Subshape (0.3, 0.7)` vs. Color Weighting**: `tversky_subshape` ($\alpha=0.3$ Ref, $\beta=0.7$ Cand) increases active subshape recovery on `(-)-Epicatechin` ($\text{MW}=290\text{ Da}$) from `0.5910` to **`0.8140`**, enabling compact high-QED scaffold hopping, but also raises small unrelated decoys (`Ibuprofen`, $\text{MW}=206\text{ Da}$) from `0.2643` to `0.4290` — requiring a higher ROCS threshold (`0.50–0.55` instead of `0.35`) during RL to prevent fragment collapse.
- **Why Hill-Climbing `0.95` Self-Recovery Is a Goodhart Trap**: Crossing `0.89 -> 0.95` on EGCG self-recovery requires candidate force-field minimization (~10x slower in RL: `1,480 ms` vs `145 ms/mol`) while off-diagonal candidate ranking order is already preserved without candidate minimization (`Spearman rho = 0.967–1.000`).

**2026-10-29 (Argon Architectural Refactoring & Root-Cause Resolution):**
All 25 sprawling, partially duplicated conformer scripts and hardcoded shell wrappers (`conformer_experiment.py`, `study.py`, `repair_identities.py`, `seed_control.py`, `reference_caps.py`, `validate_calibration.py`, `whole_call_scorer.py`, `validate_reward_calibration.py`, `query_calibration.py`, `reward_analysis.py`, `resume_calibration.sh`, `run_cost_audit.sh`, `run_reward_study.sh`) have been consolidated into a clean 12-module package (`_david/code/conformer/`) and promoted into core `_david/code/`:
1. **Multiprocessing Float32 Coordinate Truncation Fixed (`chemistry.py`)**: Enabled `Chem.PropertyPickleOptions.CoordsAsDouble` globally via `enable_double_precision_mol_pickling()`. Previously, pickling `Chem.Mol` across worker queues truncated 3D coordinates from `float64` to `float32` (`~2.9e-5 Å` error), causing parallel (`n_jobs > 1`) vs. serial (`n_jobs = 1`) ROCS score divergence (e.g., ID 732 diverging by `0.069` when a secondary alignment seed shifted local optima).
2. **ETKDGv3 `maxIterations` In-Place Mutation Fixed (`chemistry.py`)**: `CandidateConformerGenerator.genConformers` now instantiates a fresh `AllChem.ETKDGv3()` parameter object per stereoisomer rather than once per batch, preventing RDKit's C++ embedder from mutating `params.maxIterations` (`0 -> 90` on small first molecules) and starving larger subsequent molecules in the batch.
3. **Disjoint Conformer RNG Streams (`chemistry.py`, `pipeline.py`, `reference_bank.py`)**: Promoted `DEFAULT_REFERENCE_SEED = 104729`, `DEFAULT_CANDIDATE_SEED = 42`, and `verify_disjoint_conformer_seeds()` into `chemistry.py` and `pipeline.py`, preventing `base_seed * (conf_idx + 1)` collisions (`43 * 42 = 1806 = 42 * 43`).
4. **Stereo-Balanced & Shape+Color Farthest-First Reference Selection (`reference.py`, `config.py`, `pipeline.py`)**: Promoted `balanced_reference_order`, `diverse_order`, `max_reference_isomers`, and `selection_mode: Literal["rmsd", "diverse"]` into `PrepareMolecule` and `PipelineConfig`, solving both polar-rotamer loss under heavy-atom Butina RMSD pruning and stereoisomer starvation under small reference caps $K$.
5. **Persistent Worker Pool (`rocs.py`)**: Promoted `WholeCallScorer` and `NAMED_OPTIMIZATION_METHODS` into `_david/code/rocs.py`.
6. **Explicit Physical ROCS Unscaling (`pipeline.py`, `visualization.py`)**: `MoleculePipeline` now records `Raw_ROCS` explicitly alongside desirability-transformed `ROCS`, and `unscale_to_physical_metrics` uses `Raw_ROCS` (or explicit `rocs_is_normalized`) rather than a fragile `max <= 1.0001` heuristic.

**2026-10-29: stereochemistry proposal under discussion; new runs on hold.**
The user proposed stereo-unspecified EGCG/drug-panel inputs. No chemistry policy
or source input was changed. A metadata-only RDKit audit of the frozen panel
found 239/512 inputs with specified stereo; masking it increases their enumeration
bounds, and 159 inputs yield eight distinct formal variants in capped enumeration.
The specified EGCG input yields one identity, versus four after masking. Masking
also merges the panel to 507 distinct stereo-unspecified structures, with two
collision groups spanning development/confirmation. Counts precede chemical-state
preparation and do not establish 3D feasibility or required conformer budgets.
The [report](../../output/20261029_1155_reward_calibration/REPORT.md) records the
evidence and recommendation to retain known reference stereo and consider a
separate masked-candidate comparison. That comparison is proposed, not selected.
The current calibration is preserved; the pending RL timing launch is held.

**2026-10-29: corrected development grid validated; confirmation continues.**
The [current report](../../output/20261029_1155_reward_calibration/REPORT.md)
now links all four development score/time sheets. Verified 18,432 unique cells,
184,015 saved-geometry identity checks and 12 public/component replays; maximum
public difference 1.11e−16. Among 126 colour-eligible inputs, with five preparation
failures retained as zero, MMFF94s ε78.4 K4/C20 averages 0.9101 at 1.88 s/input;
all/C50 averages 0.9511 at 19.81 s/input. These are development measurements,
not the final confirmed preset. The existing lab job is testing the frozen
six-setting shortlist on the other 384 inputs. T3 took over continuation after
the user paused the old Codex automation; no measurement source changed.

**2026-10-29 update: expanded C50/C75 study superseded by an RNG-seed control.**
RDKit's effective seeds multiply the base by the conformer ordinal: bases 43/42
share 1806 at reference request 42/candidate request 43. Exact saved raw coordinate
reuse was verified. Do not use that expanded study's high-budget recovery or
Pareto recommendation. See [the diagnostic](../../output/20261029_0250_reward_calibration/REPORT.md)
and [the corrected independent-stream study](../../output/20261029_1155_reward_calibration/REPORT.md).
The replacement keeps candidate seed 42 and uses reference seed 104729, with arithmetic and
empirical overlap controls. Its full grid is 128 development drugs × 144 cells;
384 confirmation drugs check a development-selected shortlist. Fresh public calls
measure latency; short RL pilots are cost illustrations, not learning comparisons.

The corrected pilot passed 40 tests, 72 cells, 1,131 saved-geometry identity checks and
8 public/component replays (maximum difference 1.12e−16). The final software controls
passed **42 tests in 4.342 seconds** on dichtator; the science measurement sources
remain fixed while that run computes. The parameter/batch
prototype adds a control against singleton public calls, including duplicate and
invalid rows. RDKit mutates default `maxIterations` on its parameter object
(observed 0→90 for ethanol); reusing it across differently sized graphs can couple
a native batch to its first graph. The per-input calibration is unaffected.
`whole_call_scorer.py` avoids that coupling and persists whole-call workers; it is
a reviewable prototype, not a deployed default. `query_calibration.py` suggests
query budgets from measured cost order and two disjoint self-check streams, with
explicit limits on what self-recovery proves.

**2026-10-29: full drug-panel calibration completed and verified.**
Read the [single maintained report](../../output/20261028_2145_joint_calibration/REPORT.md)
and [final retained-reference figure](../../output/20261028_2145_joint_calibration/reference_caps.png).
All 512 inputs are accounted for: 509 scored, three native-colourless exclusions,
13 preparation failures retained as zero. Verified 18,432 measurement rows,
92,160 retained-cap reductions, 68,347 saved-geometry identity checks, 12 public
replays and 48 complete factory calls; maximum replay difference 1.12e−16.
Main-checkout validation passed 36 tests, scoped Ruff and shell syntax checks.

**Main methodological finding:** 0.5 Å reference RMSD pruning can lose important
ROCS colour matches. Fosfomycin scores 0.7085 with one pruned reference versus
1.0000 with its three saved unpruned references, using the same candidates.
Increasing candidate requests 10→40 did not repair this example. Twenty-four
scored inputs at C10 lose more than 0.01 through this reference pruning.
Recommend evaluating reference-only shape/colour-aware selection before freezing
an RL preset; that method is proposed, not silently implemented.

Lowest-cost tested no-candidate-post-pruning baseline above 0.95: MMFF94s ε78.4, reference preparation
100 requests/stereoisomer and stereo cap8, all post-filter references at RMSD0.5;
candidate C10/I4, additional pruning off. Mean0.9551, 442/509 at least0.95;
component mean3.460s versus R50/C20/all mean0.9552 at4.780s. Among496 prepared
inputs its mean is0.9802; primary results still include13 zero-score failures.
No universal small fixed reference bank or optimal stereo cap4 is established.

User-selected validity policy is now shared by reference/RL paths: reject only
invalid conformers, preserve valid siblings and the input, count identity rejects
separately (−2). Seven reference and five candidate identity rejections occurred
across budgets. Eight affected inputs were recomputed;504 unchanged checkpoints
were retained with explicit source/provenance. Hydrogen normalization preserves
E/Z and isotope checks. Pentazocine's failed stereo embeddings caused the long
tail (41/85min reference preparation; C20 generation94s); it was not dropped.

All active code is in the main checkout's `_david/code/`. Run artifacts are under
`_david/output/20261028_2145_joint_calibration/`, with report/figure/progress at
root, data/provenance in `raw_data/`, logs in `logs/`. All17 cached worktrees were
removed. Native user edits were preserved. Candidate minimization remains OFF
by default; no RL training has been launched. Historical evidence below is not
an instruction to restore old methods or removed execution paths.

**2026-10-28 follow-up: RL launch on hold; pipeline refactoring under review.**
Read the dated [pipeline/reference plan](../../output/20261028_rl_preparation/PLAN.md)
for the scientific interpretation, exact preset proposal, adaptive-planner limits,
code-review order and MetaCentrum access blocker. No RL/PBS job has been submitted.
At MMFF94s ε78.4/C20/I4/0.5 Å/all retained references, 135/139 selected inputs
reach 0.95; four fall below it (minimum 0.777589). This is not a universal
guarantee or evidence for exactly three references. The user's latest steering
keeps epoch runs on hold. Defaults remain unchanged; the new explicit profile
configuration is optional. Historical measurements remain bound to 95d0d60.
Clean refactored source **0dca8cc** passed **27 tests** (3.906 s), scoped Ruff
and PBS shell syntax checks. Its production EGCG diagnostic retained three
references and seven candidates from 20 requests, scoring **0.99871485**
(shape 0.99979687, color 0.99763283), exactly matching component replay.
The adaptive-planner audit covers 507 saved prepared inputs: 151 hit the 250
per-isomer ceiling; nine exceed the stereo ceiling 25; total requests can reach
6,250. These are planning diagnostics, not validated coverage or new defaults.

**2026-10-28: native-scoring mini-study completed and verified.**

Read the compact [current report](../../output/20261028_1753_native_self_recovery/REPORT.md)
for selected settings, measured EGCG controls, limitations, reproduction and
code-review pointers. The old cross-molecule run and the later color-override
panel are cancelled/superseded; their saved data must not be mixed with this run.

Selected protocol: each input against its own independent reference ensemble;
no minimization, MMFF94 ε1 and MMFF94s ε78.4; reference RMSD selections 1.25/0.5 Å
from the same saved pools; reference requests 50 per stereoisomer, stereo cap 4;
candidate requests 1/5/10/15/20/ 25/30/50 and stereo caps 1/2/4/8. Stereo seed 42 on
both sides, candidate geometry seed 42 and independent reference geometry seed 43.
One maximum over permitted pairs per input, then summaries across inputs; no
AQME budgets, seed averaging, formal hypothesis test or RL training.

The native color/combo score is unchanged. Colorless prepared reference inputs
are excluded by feature presence **before any ROCS alignment**, with reasons
recorded in cohort.csv. This is an operational scope restriction, not proof of
drug-likeness. Candidate generation failures remain zero-score observations for
eligible inputs. Do not select by high scores or silently omit failures.

Groups count distinct retained reference structures (conformers and permitted
stereo identities), not merely stereoisomers. Use the maximum count across the
two optimized methods and BOTH cutoffs; exclude the unminimized profile from
this grouping calculation. Fixed ranges 1–4/5–8/9–12; more than 12 is outside this
mini-study and must be reported. Fixed membership applies across all figures.

Historical measurement source: Git revision **95d0d60**. Its cached execution checkout was removed during workspace consolidation; reproduce from Git, not an old execution path.
All **22** regression tests passed (3.644 s) and scoped Ruff passed there. The new
EGCG pilot completed 960 rows; six representative native public-scorer replays
matched within 1.2e-16. Exact settings, source hashes, test command and replay
values are in [its manifest](../../output/20261028_1753_native_self_recovery/egcg_pilot/manifest.json).
EGCG at 50 reference / 25 candidate requests retained 8 usable candidates and scored:
none 0.8838/0.9332, MMFF94 0.9571/0.9571, MMFF94s ε78.4 0.9989/0.9989 for cutoffs
1.25/0.5 Å. These are single-input controls, not panel-wide conclusions.

The panel finished successfully with **eight** one-thread workers; it was already
complete when the requested CPU increase was checked. No restart or allocation
change was needed. All 512 reference preparations were accounted for; selected
groups contain **61/40/38 inputs** (1–4/5–8/9–12), **139 total**. Exclusions:
345 above the selected reference-count range and 28 preparation failures
(14 embedding, five domain-filter, nine identity-check failures). No colorless
exclusions occurred in this panel; no candidate failures occurred in the selected
grid. The scope is 139/512 inputs, not a claim about the excluded inputs.

Output on both hosts: _david/output/20261028_1753_native_self_recovery/panel.
All **133,440** rows form complete unique 960-cell grids. Independent checks
passed reference masks/counts, same-pose component and timing reductions,
2,880 summary rows, source/output hashes and 1,454 reference SDF hashes.
Eighteen representative native PUBLIC-scorer replays spanning three groups,
three methods and two cutoffs matched within 1.12e-16. Evidence and inherited
22-test/pilot validation are appended to the existing panel manifest. Native
numeric overshoot is retained: combo max 1.00000361, shape max 1.00000899;
verification tolerance 1e-5. All 24 PNGs are mirrored locally with root data and
cohort; full reference SDFs remain on dichtator, with replay examples local.

At candidate C25/I4/all retained references, equal-input panel mean ROCS is
**0.9692/0.9890/0.9893** (none/MMFF94/MMFF94s ε78.4) at 1.25 Å, versus
**0.9872/0.9953/0.9961** at 0.5 Å. For MMFF94/0.5 Å, C5/10/25/50 means are
0.9925/0.9946/0.9953/0.9963; corresponding mean component seconds are
0.258/0.349/0.547/0.806. These are descriptive best-pair results for the restricted
cohort, not full ensemble coverage, an established optimal budget or RL benefit.
Read the report for all figures, denominators, caveats and review order.

Timing clarification: at the current C30/I4 budgets, candidate MMFF94 helper
time averages 0.0940 s/input (median 0.0532, p95 0.2189); MMFF94s ε78.4 averages
0.0933 s. Generation averages 0.3247 s. These are component measurements on
the selected 139 inputs, **not measured RL epoch slowdowns**. Total-time method
comparisons also change reference ensembles and cannot isolate the candidate
switch. See the report's RL timing section; no new experiment was launched.

The completion monitor can now be paused. The proposed existing-drug panel is
a later complementary study, not a replacement or extra experiment authorized
for this run. Production candidate minimization remains OFF by default. The
native user's environment.py argument-order edit must remain preserved.

## Superseded cross-molecule experiment — retained evidence

The user previously selected a
fixed generated-molecule scoring test, not RNN training. The new matrix compares
seven matched reference/candidate preparation methods on 512 randomly selected
valid, unique molecules from saved RNN evaluations, with three embedding seeds.
Candidate attempts: 1/5/10/15/20/30/50/100; isomer caps: 1/3/5/10/all≤30;
reference caps: 1/2/4/8/all≤30. Missing reference counts remain unavailable.
Four images will show mean, median, tail score/time, and failure/ranking stability.
The selected training switch remains OFF by default. The earlier minimized
in-memory trial failed a public-scorer parity check at C=15 (0.034 raw difference);
the shared helper now round-trips through SDF before either path scores it.

Evidence consulted for this extension: full-text Methods and Discussion of
[McNutt et al., 2023, DOI 10.1021/acs.jcim.3c01245](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647020/).
Their task-specific comparisons support measuring minimization, ensemble size
and diversity separately; larger ensembles can also introduce false positives.
Their docking/pharmacophore results do not establish DrugEx RL benefit or the
best parameters here. The new study keeps specified input stereo and labels
best-over-isomer scoring as a search over prepared forms, not activity evidence.

## Superseded execution checkpoint — do not resume

- **2026-10-28 provisional preview:** [mean score/time](../../output/conformer/mega_20261028/preview_508/mega_mean.png),
  [median](../../output/conformer/mega_20261028/preview_508/mega_median.png),
  [tails](../../output/conformer/mega_20261028/preview_508/mega_tails.png), and
  [failure/ranking stability](../../output/conformer/mega_20261028/preview_508/mega_reliability.png).
  These use 508 complete parent grids (2,133,600 rows), averaging three seeds
  per parent before aggregation. Missing parents: 51, 61, 226, 446. Their unusually
  slow generation makes this an incomplete subset; timing summaries may be biased
  downward. Do not treat the preview as the completed study or select production
  settings from it. Saved [summary](../../output/conformer/mega_20261028/preview_508/summary.csv)
  and [preview manifest](../../output/conformer/mega_20261028/preview_508/manifest.json)
  record included IDs, checkpoint hashes and rendering source hashes. Local copies
  match all preview artifact hashes. Preview rendering passed scoped Ruff checks
  on dichtator at `98afd83`, checked unique grid keys and 4,200 rows per parent,
  and left the running measurement source/data unchanged. Reproduce from existing
  checkpoints with `.venv/bin/python -B -m _david.code.conformer.mega --render-partial
  --output /home/kolarv/DrugEx/_david/output/conformer/mega_20261028`;
  this writes a separate `preview_<completed-count>/` directory. It does not run
  conformer generation or validate final public-scorer replay.
- Measurement source: clean committed `ff96921`, isolated at
  `/home/kolarv/.cache/drugex-mega-ff96921` on dichtator.
- Persistent job: tmux `drugex-mega-512`; 16 one-thread workers, no RL epochs.
- Output: `/home/kolarv/DrugEx/_david/output/conformer/mega_20261028`.
  Read its `manifest.json` and `exit_code` for the next bounded completion check.
  A successful full grid contains 2,150,400 parent/configuration/seed rows.
- Query preparation completed. Retained reference counts by method are:
  none=4; MMFF94=3; MMFF94s ε1=3; ε6.5=2; ε78.4=3; distance-dependent=3;
  UFF=2. Unsupported reference requests are unavailable, not clipped repeats.
- Eleven regression tests passed on the lab source (2.124 s), scoped Ruff
  passed, and all seven pilot public-scorer checks matched within 1e-10.
  Exact validation records are `/tmp/drugex-mega-ff96921-validation.json` on
  dichtator; copy them into the completed experiment's manifest verification.
- Measurements must finish in the frozen source above. The current renderer
  improves contrast and records rendering provenance; use `--render-only` after
  completion to generate final images without changing raw data or summaries.
- Remaining delivery: validate grid/denominators and representative public-scorer
  replay, render the four images, mirror evidence locally, replace this checkpoint
  with findings, and synchronize through private `dichtator` preserving user edits.
- The failed first minimized trial is preserved under
  `output/conformer/archive/mmff94_candidates_7039298_failed_parity` on dichtator,
  with its manifest explicitly marking the 0.034 raw-score parity failure.

## Selected stratified visualization — 2026-10-28

The user selected disjoint ranges **{1}, {2}, {3–4}, {5–8}, {9+}**, containing
293, 128, 55, 18 and 14 completed parents. These group the estimated isomer upper
bound after preparation, not physically realizable or successfully embedded
isomer counts; see [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html).
The 508-parent snapshot is still provisional; four unusually slow parents are
missing, so time summaries may be biased downward. No new conformers or scores
were generated for this analysis. The training switch remains OFF.

Artifacts on both hosts: `output/20261028_1405_conformer_analysis/`:
[group 1](../../output/20261028_1405_conformer_analysis/isomers_1.png),
[group 2](../../output/20261028_1405_conformer_analysis/isomers_2.png),
[group 3–4](../../output/20261028_1405_conformer_analysis/isomers_3_4.png),
[group 5–8](../../output/20261028_1405_conformer_analysis/isomers_5_8.png),
[group 9+](../../output/20261028_1405_conformer_analysis/isomers_ge9.png),
[score/time](../../output/20261028_1405_conformer_analysis/score_time.png),
[distributions/fixed cohort](../../output/20261028_1405_conformer_analysis/score_distributions.png),
[sampling diagnostics](../../output/20261028_1405_conformer_analysis/sampling_diagnostics.png),
[reference counts](../../output/20261028_1405_conformer_analysis/reference_effects.png).
The folder also contains `molecules.csv.gz` (seed-averaged parent measurements),
`stratified.csv` (plotted summaries), `panel.csv`, `fixed_top_cohort.csv`,
`identity_count_warnings.csv`, and the manifest with source/output hashes,
definitions, verification and separate rendering provenance. No extra report MD.

The user also selected median, tail and reliability panels for every range.
Use each `isomers_<range>_median.png`, `_tails.png`, and `_reliability.png` beside
the original mean/gain image (24 images total including four companion figures).
`additional_metrics.csv` holds the derived lower-tail and ranking values; the
original five data tables remain unchanged. Tails are 10th-percentile ROCS and
95th-percentile time across parent seed averages. Reliability compares each
method's ranking with its C100/I30 anchor within that range, with
k=ceil(0.05*n): 15/7/3/1/1 for ranges 1/2/3–4/5–8/9+. Anchor overlap is 1 by
construction. The last two figures measure top-1 stability, not ROCS similarity
or statistically established enrichment. Insufficient positive anchor scores
produce unavailable overlap, not a fabricated perfect score.

Heatmaps use all available references and shared scales. Paired isomer gains
compare the same parent/method/C against cap 1. Group 1 has zero such gain and
omits redundant gain panels. Score/time frontiers compare budgets within each
method; methods have different query geometries. MMFF94 time bars span median
to 95th percentile across parent seed-average times, not confidence intervals.
Molecule categories depend only on the recorded isomer bound. Conformer attempts
remain a varied budget (1/5/10/15/20/30/50/100) in every category, including group 1.
For each parent and seed, the score is the maximum over usable candidate
conformers/isomers and selected reference conformers, matching training's
best-match aggregation. It is never the mean of the candidate conformer scores.
Only afterward are the three seed maxima averaged for that parent, followed by
category mean/median/quantiles across parents. Requested attempts may yield fewer
conformers because of pruning or failures; realized counts remain separate data.
Preparation failures score zero in overall summaries. The success-only diagnostic
requires all three seeds to succeed at both caps and reports its smaller denominator.
At MMFF94/C100 in group 5–8, cap-30 minus cap-1 gain is 0.096466 mean, 0.024601
median, and about 0.020 among 12 fully successful pairs: overall gains partly
reflect failure recovery. Small strata remain descriptive, without significance claims.

Two inspectable examples (same prepared parent, MMFF94/C100/all references,
three-seed averages): parent 100 scores 0 at cap 1 with failure in every seed,
then 0.392378 at cap 30 with no failures. Parent 138 succeeds at both and scores
0.393194 vs 0.437888. Both inputs have unassigned stereo (bound 8), so changing
the cap can search different stereoisomers; these are not guaranteed comparisons
of one fully specified chemical identity. Different generated candidates versus
EGCG are not EGCG self-recovery. All 20 saved exact-geometry controls score 1.0;
ROCS here is `(shape Tanimoto + color Tanimoto) / 2`. Separate winning-pose
components were not saved, so this dataset alone cannot establish which component
limits an individual low score. Top-5% overlap is ranking stability, not similarity;
the fixed 26-parent MMFF94 anchor cohort has mean 0.421813 and maximum 0.441972.
It is selected on these measurements and is exploratory, not held-out validation.

**Diagnostic exception:** recorded `actual_isomers` exceeds its requested cap in
seven generation settings for parent 46. The benchmark counts distinct isomeric
SMILES after SDF readback; this is not a validated enumeration count. The cause
is unresolved. Preserve the measured values, flag them in the warnings CSV and
label that curve as SDF-readback identities; do not silently clip it to the cap.

Validation on dichtator: scoped Ruff and two targeted regression tests passed
(seed averaging, paired failure handling, range boundaries, lower-tail quantile,
fixed within-stratum ranking anchor and unavailable ranking cases). Every included
parent passed complete/unique grid, reference availability/count, score range,
failure/count and decomposed-time checks, with the identity-count exception above.
Independent Python statistics calculations checked all 1,400 summary cells and
12 raw-parent score/SD reductions for IDs 0/1/11/22/41/46. This is reduction replay,
not public molecular-scorer replay. Saved tables were unchanged during rendering.

Reproduce on dichtator from the repository root into a new directory:
```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m _david.code.conformer.analyze \
  --source /home/kolarv/DrugEx/_david/output/conformer/mega_20261028 \
  --output /home/kolarv/DrugEx/_david/output/YYYYMMDD_HHMM_conformer_analysis
```
Use `--render-only --output <existing-analysis-directory>` without `--source`
to regenerate figures from saved tables. Review [analyze.py](analyze.py) for
validation/aggregation and [visualization.py](../visualization.py) for the
`_analysis_*` renderers, then [the test](../../tests/test_conformer_analysis.py).

Evidence reused/refreshed from free Europe PMC full text: McNutt et al. 2023
(linked above), conformer retrieval and pharmacophore ensemble-size results.
They report heterogeneous responses hidden by average trends and effects of
ligand flexibility; their task does not select our ranges or budgets. Search:
ensemble size, rotatable bonds, flexibility, false positives. Measurements remain
bound to `ff96921`; this selected change only analyzes and renders saved data.

## One-isomer-category audit — 2026-10-28

Read-only analysis of the existing 293 upper-bound-one parents, with no new
generation/scoring. Every parent is compared with the shared EGCG reference;
**zero of the 293 has EGCG connectivity**. The 1,000-attempt deep preparation
created the EGCG ensemble, not an independent reference for each parent.
MMFF94 has three retained query conformers. All isomer-cap score aliases in this
category agree exactly; candidate conformers are still maximized before seed
averaging. The score is divided by two once, before storage, and plotted directly.

MMFF94/all references, means over all parents after averaging three seeds:

| Requested conformers | Mean score | Median score | Mean embedded count | Mean seconds |
| --- | --- | --- | --- | --- |
| 1 | 0.312815 | 0.317233 | 0.993 | 0.143 |
| 10 | 0.332728 | 0.337417 | 7.412 | 0.508 |
| 30 | 0.337885 | 0.341747 | 17.507 | 1.164 |
| 100 | 0.341244 | 0.344548 | 41.896 | 3.024 |

At C100 the 10th/90th percentiles are 0.297435/0.384890; maximum is 0.438530.
Only parents 8 and 72 score zero, in all seeds. Their input sizes are 46 and 50
heavy atoms, above the existing generator limit of 45: these are size-filter
exclusions before embedding, not evidence of minimization nonconvergence. Raw
rows record zero candidates and a subsequent empty-SDF file error. Excluding
these two changes the mean only from 0.341244 to 0.343590, so failures do not
explain the category's low average.

The paired C30→C100 mean gain is 0.003360 (median 0.000748); 33/293 parents gain
more than 0.01. This is diminishing average benefit for this different-molecule
versus-EGCG task, not a universal conformer-budget recommendation. For context,
the median parent has 25 heavy atoms and three oxygens, versus EGCG's 33/11.
Parent 473 is the 9-heavy-atom triamine `NCCCNCCCN` and scores 0.198657; the
median parent 10 scores 0.344548; parent 373 scores 0.438530. These are different
chemical structures, not independent preparations of EGCG. Morgan radius-2/2048
fingerprint Tanimoto to EGCG has median 0.090909 and maximum 0.169811; its
Spearman association with C100 ROCS is only 0.028, so do not use this 2D measure
as a quantitative explanation or validation of the 3D score.

Conclusion: neither averaging conformers nor a large failed-parent fraction
explains 0.34. This panel does not measure the same-identity self-recovery the
user is questioning. Its low means alone neither prove a scorer bug nor validate
the scorer. Separate winning-pose shape/color components were not saved, and the
parent-46 SDF identity-count exception remains unresolved.

## Filtering and self-recovery proposal — not yet selected

Keep the full attempted-parent dataset and failure ledger. Individual optimized
conformers already require native convergence status 0; this does not establish a
global minimum or biological plausibility. "None" has no force-field convergence
criterion. Report preparation availability separately from conditional score
quality. For a planned paired comparison, use the same successful-parent subset
for both settings and report its denominator plus failure rates on the full panel;
do not compare independently filtered method-specific populations as if matched.
Restricting to supported chemistry by declared rules is defensible, but its scope
and exclusions must be explicit. Selecting only high-scoring or universally easy
cases after inspection would answer a biased, narrower question.

A proposed next study addresses independent molecular self-recovery: a fixed,
diverse panel including EGCG, each with a specified identity and frozen reference
ensemble. Regenerate from isomeric SMILES with independent seeds and no reference
coordinates as candidate input. Use known stereo first; an intentionally masked
candidate-stereo challenge would be a separate labeled arm. Record reference
preparation failures, candidate failures, normalized shape/color combo, its two
components at the same winning pose, symmetry-aware heavy-atom RMSD and timing.
An exact-copy control should score 1; independently regenerated conformers need
not. Fixed targets isolate candidate preparation, while the existing matched
reference/candidate methods compare whole protocols. Computational references
measure recovery of those protocols; experimental conformers would support a
different physical-recovery claim. No new study or dataset filtering is authorized
or implemented by this proposal.

Evidence consulted 2026-10-28: free full text of [High-Quality Conformer Generation
with CONFORGE, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10498443/), sections
Dataset Preparation, Processing Time Measurement and RMSD Calculation, and
molecule-processing-failure tables. It converts reference structures to isomeric
SMILES to avoid supplying reference coordinates, assesses best symmetry-aware
heavy-atom RMSD, and separately reports failed molecules. Those benchmarks do not
prescribe our ROCS thresholds or panel size. [RDKit force-field documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)
defines status 0 as converged and status 1 as requiring more iterations; neither
is an activity label. Search scope: conformer benchmark failures, exclusion,
stereochemistry, independent reference RMSD. Current raw scores are unchanged.

## Historical single-molecule baseline

The following dated sections describe the earlier single-molecule experiment;
they are not the current mega-study result or a production-parameter selection.

Updated 2026-10-27. Current instructions: [AGENTS.md](../../AGENTS.md).
Source lives here; generated evidence belongs under `_david/output/conformer/`.

## Current implementation

The preparation/configuration fix is `3cb45dc`; the initial study implementation
is `51b5562`. The subsequent folder move follows the user's dichtator layout.
All implementation changes are under `_david/`. DrugEx core and the RNN model
implementation are unchanged by this work.

- `../chemistry.py` reproduces DrugEx's RDKit standardization sequence with
  isomeric serialization. Identity checks compare connectivity and specified
  stereo, reassigning stereo from 3D coordinates rather than trusting SDF tags.
- `../environment.py` uses that standardization for training reference
  preparation, rejects mixed-isomer SDFs and aligns atom mappings when combining
  records. It honors constructor seeds/attempts, the configured energy window
  and requested reference cap. Clustering uses an explicit 1.25 Å cutoff.
  Only converged, finite-energy reference conformers are retained; the actual
  force field and embedding/convergence counts are saved in the SDF.
- Scorer construction now forwards timeout and isomer limits. Omitted environment
  arguments defer to PipelineConfig instead of overriding it. The candidate
  generator is the existing core algorithm with a small seed-exposing subclass;
  its default embedding seed remains 0xC0FFEE.
- `study.py` compares training-path ETKDG candidates with the same candidates
  minimized using MMFF94. Candidate minimization is a diagnostic alternative;
  training still uses unminimized candidates. Only MMFF94 references are accepted
  in this condition; a UFF fallback is not silently mixed into its results.
- `conformer_experiment.py` delegates to the current study. The moved legacy
  launcher no longer imports the removed Gemini implementation.

A reference must specify its stereochemistry for this self-recovery study.
Candidate input can be supplied separately; unresolved centers can be enumerated
while preserving specified centers. Reference policy for unresolved stereo
remains a research decision. The study rejects a candidate preparation whose
connectivity or specified stereo differs from the stated input; this is not a
general tautomer/protonation equivalence checker.

## Controls and validation

The single module `_david/tests/test_sequencer_pipeline.py` contains six focused
tests: standardization chemistry/stereo, reference SDF round trips and energy
windows, rejection of a wrong isomer/mixed SDF, exact-geometry raw/modified score
scales and parameter wiring, candidate identity for the specified reference, and
independent storage plus freezing of the RNN mutation prior.

Local WSL, RDKit 2026.09.1: all six passed (3.216 s); Ruff passed.
Dichtator, RDKit 2026.03.6, revision `51b5562`: all six passed (0.984 s);
Ruff passed for the task files. A broader check also found an unrelated E741 in
the user's moved molecular-dynamics code; it was not changed.

Final validation and measurements used clean committed source `a09fe58` in
`/home/kolarv/.cache/drugex-mmff94-a09fe58`. All six tests passed (0.937 s), and
Ruff passed for that checkout's code/tests. Import origins and before/after source
hashes were verified. Full commands, logs, versions and validation records are in
the experiment manifest. Concurrent user reorganization, including deletion of
old molecular-dynamics ligands/output, was preserved in the active checkout.

The earlier weak smoke assertions were replaced, not treated as proof of
conformer correctness. Tests do not substitute for reviewing the saved structures
and measurements. The RNN guard does not constitute an audit of the earlier
SequenceRNN transfer or establish training convergence.

## Selected experiment

The user selected the default MMFF model (RDKit MMFF94) and RDKit Tanimoto
shape-plus-color as the initial baseline, with paired minimized/unminimized
candidates. The earlier five-method/two-score comparison is deferred.

Completed run: 1,000 reference embedding attempts, reference seed 42, MMFF94
minimization up to 1,000 iterations, 8 kcal/mol energy window and 1.25 Å heavy-atom
RMSD clustering. Candidate attempts per isomer: 1, 10, 30, 100, 200; isomer caps:
1, 4, 8; reference caps: 1, 2, 3, all; candidate seeds: 12648430–12648432.
These budgets define this comparison; they are not selected production settings.
No wall-clock embedding timeout; one CPU thread; no RNN training.

Each row records requested and actual counts, convergence failures, winning
isomer, raw combo [0,2], combo/2 [0,1], generation/minimization/overlay times.
Timings exclude one-time reference preparation and multiprocessing startup.
They include the study's identity audits and are not a training throughput claim.
Reference-prefix scores reuse measured pairwise overlays; times sum those
overlays and add candidate preparation. They are not full training latency.

Fully specified EGCG should yield one stereoisomer at every isomer cap. Flat
isomer panels are an expected negative result, not evidence for a universal cap.
A later broader-molecule study must evaluate whole distributions as well as a
predefined top-5% tail. The current lead renderer selects five molecules, not 5%.

## Measured EGCG self-recovery

One specified molecule, three independent candidate embedding seeds. The 360 CSV
rows are configuration/seed combinations, not 360 molecules. Isomer caps 1/4/8
each produced one isomer and identical scores for each otherwise matched case.
Reference preparation took 16.98 s: 1,000 requested attempts → 36 embedded →
36 converged → three retained representatives. All 375 generated candidate
conformers converged in the paired minimization; 750 original/minimized SDF
records were saved and their 3D identities checked.

The following uses isomer cap 4 and all three references. Scores are combo/2,
mean ± sample SD across the three seeds; time is the mean seconds per molecule
under the single-thread measurement definition above.

| Requested C | Actually embedded | ETKDG score | Minimized score | ETKDG seconds | With minimization seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.596 ± 0.018 | 0.877 ± 0.133 | 0.168 | 0.189 |
| 10 | 5–7 | 0.685 ± 0.047 | 0.988 ± 0.015 | 0.429 | 0.543 |
| 30 | 7–10 | 0.714 ± 0.050 | 0.988 ± 0.015 | 0.820 | 1.019 |
| 100 | 11–13 | 0.736 ± 0.015 | 0.998 ± 0.004 | 2.055 | 2.322 |
| 200 | 13–15 | 0.736 ± 0.015 | 0.998 ± 0.004 | 3.670 | 3.980 |

Minimization has a much larger effect here than increasing the unminimized search
from 100 to 200 attempts. This supports a geometry-preparation mismatch for this
specified EGCG comparison. It does not establish bioactive conformations, an
optimal force field, general conformer/isomer budgets or improved RNN training.
In particular, the flat isomer axis cannot select a cap for ambiguous molecules.

All three exact-geometry controls scored 2 raw. A deliberately inverted center
was rejected by the identity check (its raw overlay was 0.621431; no similarity
threshold was used as an identity test). Public-scorer replay at C=30/200,
default seed, cap 4 and all references reproduced both CSV scores exactly.
Replay from saved C=200 SDFs matched the unminimized score exactly and differed
by 1.554e-8 raw for the minimized score, consistent with SDF coordinate rounding.
These are the stated replay cases, not a replay of every measured cell.

![Score and time grids](../../output/conformer/mmff94_20261027/summary.png)

Generated evidence, mirrored from dichtator with matching hashes:
[measurements](../../output/conformer/mmff94_20261027/data.csv),
[controls](../../output/conformer/mmff94_20261027/controls.csv),
[reference structures](../../output/conformer/mmff94_20261027/reference.sdf),
[candidate structures](../../output/conformer/mmff94_20261027/candidates.sdf),
[manifest and verification](../../output/conformer/mmff94_20261027/manifest.json).
These ignored outputs are not included in a fresh Git clone.

From the repository root on dichtator, reproduce into a new output directory:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DRUGEX_SCORE_NJOBS=1 \
  .venv/bin/python -B -m _david.code.conformer.study \
  --output /home/kolarv/DrugEx/_david/output/conformer/mmff94_repeat
```

Use `--render-only --output <existing-run-directory>` to regenerate the PNG from
the CSV. Read [chemistry.py](../chemistry.py), [reference/scorer construction](../environment.py),
[study.py](study.py), [the renderer](../visualization.py) and the focused test
module for code review. No earlier unreviewed RNN implementation is certified by
this study.

## Evidence and limits

Consulted 2026-10-27: full-text Gypsum-DL methods, “Unspecified chiral centers”
and “Controlling the combinatorial explosion”, preserve specified centers and
distinguish variant dimensions. That docking study establishes neither a DrugEx
budget nor an RL benefit: [Ropp et al., 2019, DOI
10.1186/s13321-019-0358-3](https://link.springer.com/article/10.1186/s13321-019-0358-3).
Official [RDKit force-field documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)
defines MMFF94 as default and optimization status 0 as converged.
[RDKit stereo enumeration](https://www.rdkit.org/docs/source/rdkit.Chem.EnumerateStereoisomers.html)
documents preservation of assigned stereo and seeded subset enumeration.
Online documentation may differ from the installed version; live checks cover
both installed versions above. There is no established optimal isomer/conformer
ratio. Code-based deductions and proposals are not measured training effects.

Remaining limits: core clamps conformer requests above 200; native failed
embedding IDs and multiprocessing coordinate precision require separate
investigation. This study runs sequentially and checks 3D identity, so it does
not validate multiprocessing parity. Legacy training visualization still infers
score scales and can double ROCS twice; the new figure reads explicit CSV scales.

## Historical provenance

The review baseline before this work was `e7035d7`; runtime was restored to
`0f6a30d` in rollback `8a8386c`, preserving config annotations and timeout=0.
Previous conformer recommendations are withdrawn. Removed scripts, tests and
reports remain recoverable at `570fc77` on
`codex/recovery-before-conformer-rollback-20261027T185549Z`.
Patches, copied user work and documentation archives are under
`_david/output/rollback_checkpoints/20261027T185549Z/`. User reorganization and
deletions on dichtator are preserved.

The 2026-10-26 review of historical Run 17/18 found 50 recorded epochs and 1,000
evaluation rows per arm. Validity fell from 955 to 883 rows in Run 17 and 951 to
845 in Run 18; stored all-row ROCS means fell 0.328771→0.303615 and
0.326311→0.276163 respectively. Sources were the runs' baseline/reinforced raw
CSVs and training TSVs. These were not rerun here; scale/provenance issues prevent
using them as calibrated physical scores or causal hyperparameter evidence.
