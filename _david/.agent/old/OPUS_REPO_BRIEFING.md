# DrugEx Thesis — High-Yield Repo Briefing (for incoming agents, e.g. Opus 5.5)

> Written 2026-10-31 from: full repo inspection + `research_state_master_v2.md` (≡ `research_state_master_handoff.md`), `egcg_omnibus_and_panel_critique.md`, `rl01_evaluation_report.md`.
> Read this top-to-bottom before touching anything. Every number below is empirical, not opinion.
>
> **MetaCentrum update (2026-11-04):** `grid.py` and its `grid`/`sprint` verbs were replaced
> by `submit.py` and a single template. For any cluster work follow
> [`../code/metacentrum/RUNBOOK.md`](../code/metacentrum/RUNBOOK.md); the grid commands and
> the multi-arm sprint workflow described below are historical.

## 0. TL;DR

Bachelor thesis: **RNN ligand-based de novo drug design** (DrugEx v3 fork) with **3D ROCS Gaussian shape/color alignment as the RL reward** against a polar reference molecule (default: EGCG, `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`, MW 458.4, QED 0.21, 8 HBD/11 HBA, TPSA 197.4). Scientific goal is **scaffold hopping**: drug-like bioisosteres (QED > 0.60, TPA < 100 Å², stable heterocycles) matching EGCG's 3D envelope *without* catechol/pyrogallol PAINS reactivity. NOT regenerating polyphenols; NOT docking (docking is a future validation step, PDBs 3PTA/4WXX vs DNMT1 Cys1226/Glu1266/SAM cavity).

All thesis code lives in **`_david/`**. The DrugEx core (`drugex/`, repo root) is a **fork of CDDLeiden/DrugEx v3.4.8** and is **READ-ONLY**. Currently running: 4-hypothesis RL sprint on MetaCentrum (jobs `25170006`–`25170009`, queue `gpu_long`).

## 1. Non-negotiable rules

1. **Isolation**: everything thesis-related stays under `/home/kolar/DrugEx/_david/`. Core `drugex/` + root files: never modify, never force-push. (The fork *does* contain ROCS additions in core, e.g. `drugex/training/scorers/rocs_rdkit.py` — still read-only for you.)
2. **Molecule-agnostic**: EGCG is `DEFAULT_MOLECULE` in `config.py` (mirrored in `_david/in_molecule.txt`), an input — not a special case in any algorithm.
3. **Scientific ownership**: the user (David Kolář) owns research decisions. Historical reports are evidence, not instructions. Consult literature before changing methods.
4. **Hosts**:
   - `local` (WSL2, `/home/kolar/DrugEx`, `.venv/`): edit, lint (`ruff check _david`), unit tests, smoke runs.
   - `dichtator` (`kolarv@147.33.72.143`, 48 EPYC cores, 125 GB RAM, 2× RTX PRO 4000 Blackwell): offline eval, deep profiling, RL suites. **Never launch/reconnect Cisco VPN if active** (single-client lock breaks tunnels); probe first: `ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator "echo OK"`.
   - `meta` (MetaCentrum, `kolardavidcz@tarkil.metacentrum.cz`): multi-arm RL training via PBS Pro. Storage `/storage/brno2/brno2/kolardavidcz/DrugEx/`.
5. **Git**: local `main` tracks `dichtator/main` (never GitHub `origin`). Sync only via `git push dichtator main` / `git pull dichtator main`, after checking `git status` on both sides (user edits on both hosts). At time of writing local is **3 commits ahead** of `dichtator/main` (`7f88d26`, `60f58df`, `2f3e686` = HEAD).
6. **Zero broken tests/lints**: 74/74 tests pass; `ruff check _david` clean. Commands in §10.

## 2. Repo topology

```
/home/kolar/DrugEx/
├── drugex/            # core (READ-ONLY): data/, molecules/, training/{generators,explorers,scorers,rewards,monitors}, logs/, parallel/
├── data/ -> tutorial/data   # pretrained models incl. Papyrus05.5 SMILES-RNN .pkg/.vocab, A2AR QSAR models
├── _david/            # THE THESIS
│   ├── AGENTS.md      # authoritative technical rules (this file's superset)
│   ├── code/          # pipeline: config schema utils chemistry reference rocs models environment pipeline visualization
│   │   ├── conformer/ # calibration research package (see §9)
│   │   └── metacentrum/ # submit.py, pbs_launcher.sh, job.py, help/, scripts/runed/
│   ├── tests/         # 74 tests (see §9)
│   ├── reference/     # 5 architecture/context markdown docs (~56 KB)
│   ├── onboarding/    # model-harness-transfer/HANDOFF.md + this file
│   ├── output/        # 830 MB, 22 run dirs, timestamped
│   ├── legacy/        # pre-refactor exploratory code — do not build on
│   └── validation/    # early-Oct docking/MD (EGCG + 5 leads, TTR 3NG5)
```

## 3. DrugEx core — the 20 facts that prevent confusion

1. Version is dynamic (setuptools_scm, tag `v3.4.8` + 411 commits); `about.py` fallback "3.4.7" is stale. Python ≥ 3.10.
2. CLI: `python -m drugex.{dataset,train,generate,download}` (`scripts/drugex` is just a shim).
3. **Generator taxonomy**: plain `Generator` (e.g. `SequenceRNN` — samples ex nihilo, `sample(batch_size)`) vs `FragGenerator` (transformer/graph variants — *cannot* generate without input fragments). The thesis uses **SequenceRNN only**.
4. RL loop lives in `SequenceExplorer.fit` (`training/explorers/sequence_explorer.py`): forward → policy gradient → validate on 10% of nSamples → `desired_ratio`; **best-state reload every 50 epochs**; early stop via patience/min_epochs.
5. `SequenceRNN.evolve` mixes agent + frozen prior logits **per token**: `proba[is_mutate,:] = prior_logits` where `rand < epsilon` — ε applies at *every SMILES character* (see §7, failure 9).
6. `DrugExEnvironment` (`training/environment.py`): assembles scorers+thresholds+reward scheme; invalid SMILES → zero reward; `Desired = 1` only if **all** objectives ≥ thresholds.
7. Scorers return DataFrame columns keyed by `Scorer.getKey()`. Score modifiers: `ClippedScore`, `SmoothHump` (raw→[0,1] desirability).
8. Reward schemes: `WeightedSum`, `ParetoCrowdingDistance` (PRCD), `ParetoTanimotoDistance` (PRTD). Thesis adds its own **Harrington** scheme in `_david/code/training/environment.py`.
9. Monitoring artifacts (v3 has **no** `DiskLogger`/`CsvSink` — those are v4 names): `FileMonitor` writes `{path}_fit.log`, `{path}_fit.tsv` (epoch metrics), `{path}_smiles.tsv`, `{path}.pkg` torch checkpoints (`best|all|improvement`); fork adds `PeakRecoveryMonitor` (`{path}_peak.pkg`).
10. **Columns are UPPERCASE**: `SMILES`, `Frags`, `Valid`, `Desired`, `Accurate`. Input molecule column default `'SMILES'`.
11. `DataSet` = on-disk artifact (data file + `.vocab` pair managed together, `rewrite=True` resets). Vocab files sit next to data: `{prefix}.{txt|tsv}.vocab`.
12. `drugex/__init__.py` runs `torch.set_num_threads(1)` and silences RDKit logs on import — expect this perf surprise.
13. Fork-only core modules (present, read-only): `scorers/rocs_rdkit.py` (uses `rdShapeAlign.AlignMol(..., useColors=True, opt_param=0.5)` → (T_S, T_C); `PrepareConformer` gives Gaussian self-overlaps V, C — the trick behind exact 162-combo Tversky sweeps), `rocs_openeye.py`, `rocs_cdpkit.py`, `conformer_generators.py`, `parallel.py`.
14. Core ROCS scorer embeds candidates in a **serial** Python loop — the reason the 190-core HPC job stalled (§7, failure 3).
15. Pretrained prior: `data/models/pretrained/.../Papyrus05.5` SMILES-RNN `.pkg` + `.vocab` (resolved by `config.resolve_pretrained_paths()`).
16. `frag→mol` mechanics (`FragGenerator` path) use `dummyMolsFromFragments`; `SmilesChecker.checkSmiles(..., frags=...)` computes `Accurate` (frag ⊂ mol). Not used by the thesis RNN path, but appears in tutorials.
17. `qsprpred` extra is hard-imported by `train.py`/`download.py` (QSPRPred from git). Tutorial QSAR models live in `data/models/qsar/`.
18. `download.py` fetches Zenodo models (v2/v3 graph, Papyrus 05.5, A2AR QSAR).
19. `tutorial/Sequence-RNN.ipynb` = best core starting point; `tutorial/advanced/rocs/` = fork's ROCS RL tutorial.
20. Supervisor's successful run lives in `fulopjoz/DrugEx:egcg-rocs-tutorial` at `tutorial/advanced/rocs/egcg/` — see §6.1 (golden reference: executed successful run, ε-sweep, core-API traps).

## 4. `_david/code/` — data flow and modules

**Data flow** (one line): `reference.PrepareMolecule` (Stage 0: embed 100 ETKDGv3 confs, MMFF-minimize, ΔE ≤ 8 kcal/mol filter, `diverse_order` farthest-first selection → K refs → `raw_data/query_molecule_3d.sdf`) → Stage 1 baseline sampling from frozen prior → Stage 2 RL (`explorer.fit`, ROCS scorer generates candidate conformers *inside* the env) → Stage 3 reinforced sampling → Stage 4 `ResultAnalyzer` 8 figures + scorecard. Every run writes `manifest.json` (config asdict, host, PBS job id, source sha256s, versions, reference sha256).

| Module | Key symbols | Notes |
| :-- | :-- | :-- |
| `config.py` | `PipelineConfig` (frozen dataclass), `RunPaths.resolve()`, `resolve_pretrained_paths()` | Single source of truth. Output layout `<root>/{models,raw_data,logs,figures}` under `_david/output/<run_id>`. |
| `schema.py` | `std` | Canonical column names + score-variant + reward-scheme constants. |
| `utils.py` | `HardwareManager` | PBS-aware CPU allocation (`PBS_NODEFILE`, ratio 0.90), GPU pick (prefers GPU 1, ≥ 4 GB free), `DRUGEX_SCORE_NJOBS`/`DRUGEX_MP_CONTEXT`. |
| `chemistry.py` | `CandidateConformerGenerator`, `Optimization`, `check_identity`, `standardize_smiles`, `enable_double_precision_mol_pickling()`, `verify_disjoint_conformer_seeds()` | Parallel chunked ETKDGv3 embedding (serial/parallel bit-exact parity); per-isomer fresh ETKDG params; HAC ≤ 45, rotors ≤ 15 pre-filters; seeds **104729 (reference) / 42 (candidate)**. |
| `reference.py` | `PrepareMolecule`, `diverse_order`, `select_conformers`, `balanced_reference_order` | Farthest-first **3D shape+color** diversity (symmetrized directional ROCS overlaps); Butina RMSD alternative (0.35 Å); stereoisomer round-robin prevents isomer starvation at small K. |
| `rocs.py` | `ConfigurableRDKitROCSScorer`, `WholeCallScorer`, `score_pair`, `ROCS_SIMILARITY_PRESETS`, `NAMED_OPTIMIZATION_METHODS` | Arbitrary Tversky (α,β) + shape/color weights; delegates to native path for standard weights; raw [0,2] scale with 4-sign pose guard. `WholeCallScorer` = persistent spawn `ProcessPoolExecutor`, each worker runs conformers→minimization→ROCS end-to-end. Named FFs: `etkdg`, `uff`, `mmff_pocket` (ε 6.5), `mmff_high_dielectric` (ε 78.4). |
| `models.py` | `SequenceRNNOptimizer`, `DecayingSequenceExplorer` | Loads agent + **independent frozen prior** (`assert prior is not agent`); cosine/linear ε decay support. |
| `environment.py` | `create_rocs_scorer`, `HarringtonDesirabilityReward`, `resolve_reward_scheme`, `create_environment` | ROCS raw [0,2] with `setModifier(None)` default; SA **`ClippedScore(5.0, 3.0)`**; optional QED `ClippedScore(0,1)`. Thresholds (0.35, 0.50[, 0.50]); weights 2-obj (0.75, 0.25), 3-obj (0.50, 0.25, 0.25). Harrington raw-scale ROCS anchors (0.40, 0.05, 1.20, 0.80), Gompertz per objective + weighted geometric mean, degenerate-SMILES penalty. |
| `pipeline.py` | `MoleculePipeline`, CLI `python -m _david.code.training.pipeline` | 5 stages; explicit `Raw_ROCS` column recorded; full argparse mirror of PipelineConfig. |
| `visualization.py` | `ResultAnalyzer`, `AlignedDepicter`, `unscale_to_physical_metrics`, `bootstrap_ci` | 8 publication figures (see §7 numbering in rl01 report) + `comparison_scorecard.csv`, `summary.md`, `top_prioritized_leads.tsv`. |
| `launch_rl_suite.sh` | 6-arm dichtator tmux suite | Legacy vs sprint: uses ε=0.0, K=4, C=20, MMFF ε78.4 (predates ε calibration; superseded by MetaCentrum sprint). `--status/--attach/--restart/--run/--pull/--stop`; time gate 12:00–20:00; completion = manifest `state=complete` + `reinforced_raw.csv.gz` + final `.pkg`. |

## 5. Calibrated constants — do not change without new evidence

| Constant | Value | Why (one line) |
| :-- | :-- | :-- |
| Reference seed / candidate seed | 104729 / 42 | Disjoint primes; `eff_seed = base*(i+1)` collision at 43×42=1806 previously faked self-recovery 1.000. |
| Reference selection | `diverse` (farthest-first), K=4 (sprint trains at K=1) | 128-drug panel: diverse K=4 self-recovery 0.8172 (best small-bank); K=1 global-min covers only 54.7% of EGCG pocket poses vs 90.2% at diverse K=5. |
| Butina RMSD (if used) | **0.35 Å** — never 1.25 Å | 1.25 Å collapses 52/126 drugs to 1–2 conformers, caps recovery 0.8052; 0.35 Å retains median 16 true basins. |
| Energy window ΔE | ≤ 8 kcal/mol (3.0 works, zero score loss, smaller pool) | Keeps strained receptor-settling poses (>60% of PDB ligands bind at ΔE ∈ [1.5, 5.0]). |
| Candidate conformers C | 20 minimized, or 30 unminimized (Regime 2) | Matched MMFF94s K=4/C=20 = 0.9101 mean at 1.88 s/mol — dominates unminimized C=75 (0.8915 at 4.04 s). |
| Candidate minimization | **Must match reference state** | Minimized ref + unminimized ETKDG cand collapses EGCG to 0.7645 (Color 0.602), plateau 0.8325. |
| Force field | MMFF94s, ε_r = 78.4 (ADME) or 6.5 (pocket) | Dielectric is an empirical geometry filter, NOT solvation thermodynamics. |
| SA modifier | `ClippedScore(lower=5.0, upper=3.0)` — **never `SmoothClippedScore`** | Flat 1.0 below raw SA 3.0 removes the shrink-to-fragment gradient (Run 01 root cause A). |
| ε (exploration) | 0.05, `use_epsilon_decay=False` | ε=0 → mode collapse (uniqueness 0.226, no entropy/KL in SequenceRNN); ε=0.10 → 98.5% of 40-token molecules carry alien prior tokens. |
| Score variant | `tanimoto_combo` raw [0,2], `rocs_use_modifier=False` | Modifier-free raw scale + Pareto/Harrington on raw values. If Tversky: **α+β=2 scale only**, w_C ≥ 0.65 (see §8). |
| RL | epochs 200, patience/min_epochs 50, n_samples 500, batch 64, β=0.0, training_seed 42 | β=0.0 unpenalized baseline eliminated `[N-]NNN…` collapse. |
| Parallelism | `DRUGEX_MP_CONTEXT=forkserver`, numThreads=1 per worker, cpu_ratio 0.90 | Serial-loop scaling died at 1.01× beyond 16–32 cores before chunked pools. |

## 6. Current execution state (as of 2026-10-31)

- **MetaCentrum 4-hypothesis sprint** (`generate_4hypothesis_grid`, all arms locked to Regime 2: unminimized, C=30, K=1, ε=0.05, 200 epochs, patience 50, tanimoto_combo, ~0.68 s/mol):
  - `25170006` `sprint_h1_without_qed_pareto` — supervisor-replication champion (2-obj [ROCS, SA], Pareto crowding)
  - `25170007` `sprint_h2_without_qed_harrington` — reward-scheme isolation
  - `25170008` `sprint_h3_with_qed_pareto` — QED-shrinkage causal ablation
  - `25170009` `sprint_h4_with_qed_harrington` — modernized Run 01
  - PBS: `-l select=1:ncpus=4:mem=32gb:scratch_local=15gb:spec=8.0:ngpus=1 -q gpu_long -l walltime=25:00:00`; staged release commit `7f88d26`; `spec=8.0` keeps >60 modern GPU nodes eligible (spec 10.0 → only 2). Results land in `/storage/brno2/.../DrugEx/_david/output/<name>/` via job cleanup trap.
- **dichtator**: `_david/output/rl_suite_egcg_6arm/` legacy suite, only `rl01_tanimoto_combo_default` active.
- **Git**: local HEAD `2f3e686`, 3 ahead of `dichtator/main`; dichtator unreachable (Cisco VPN down) at time of writing — push pending.
- **Open questions** (from handoff): (1) H1–H4 comparative analysis on arrival; (2) does a K=1-trained agent generalize to K=4 scoring; (3) deploy `tversky_mild_color_heavy` as Sprint 2; (4) DNMT1 co-crystal docking cross-validation.

### 6.1 Golden reference: supervisor's SUCCESSFUL EGCG run (read this before designing any new arm)

Repo: https://github.com/fulopjoz/DrugEx/tree/egcg-rocs-tutorial/tutorial/advanced/rocs/egcg (`README.md` + `RESULTS.md` = full write-up; `*_executed.html` = executed notebooks readable without running). **The successful execution**: [`final_out/eps0.1/executed_eps0.1.ipynb`](https://github.com/fulopjoz/DrugEx/blob/egcg-rocs-tutorial/tutorial/advanced/rocs/egcg/final_out/eps0.1/executed_eps0.1.ipynb) — driven by `egcg_final.ipynb` ("best configuration, every parameter set from a measurement"). Sprint H1 is the replication of exactly this champion.

- **Setup**: Papyrus pretrained SMILES-RNN, reward = 3D shape+colour vs a **single EGCG reference conformer** + SA, RDKit backend, `ParetoCrowdingDistance`, no fine-tuning, no scaffold constraint, 50×500 budget, ε=0.1, SA capped via `ClippedScore` (flat below raw 3.0).
- **Result (ε=0.1 arm, matched seed)**: median TanimotoCombo **0.638 → 0.781** (Δ+0.1425, 95% CI [+0.132, +0.152]), p90 0.877, enrichment @0.85 = **19.4×**, uniqueness 0.982, heavy atoms **27 → 29** (MW 388 → 404, moving *toward* EGCG size). ε sweep is monotonic: 0.1 > 0.2 (+0.062) > 0.4 (+0.036); uniqueness ≥ 0.98 in every arm — the expected diversity cost of low ε never materialized (ε=0 still collapses, see §7.2; optimum lies in (0, 0.1]).
- **README findings that matter downstream**:
  - `SmoothClippedScore` (asymptotic logistic: 0.881 at raw SA 3.0, 0.998 at 1.0) pays without limit for ever-lower SA → shrinkage; `ClippedScore` reversed the drift (27→29 HAC vs v3's 27→20).
  - ε was **silently inert in v1/v2** (`mutate=agent`, same object): on v2's final checkpoint uniqueness stayed 0.53 at *every* ε ∈ [0, 0.6]; with a separate frozen prior the same sweep gives 0.53 → 0.93.
  - **TanimotoCombo's theoretical 2.0 is unreachable in practice**: self-score in the same pose = 2.000, but regenerate conformers independently and it falls to **1.260** (shape 1.00→0.87, colour 1.00→**0.39** — colour is the weak channel; query conformers are built independently of the reference pose). Scores there are also reported as fraction of the 1.26 ceiling.
  - **No winner's curse**: top-20 re-scored with a different conformer seed + 60 conformers: 0.934 → 0.938.
  - Limitations (their own): single seed per config; one reference conformer (matches *that pose*, not EGCG in general); reward and evaluation share a scorer — shape gain is not independent validation; SA-cap vs ε attribution approximate (both changed v3→final).
- **Three core-API traps documented there (verified on 190 cores)**:
  1. `num_threads` and `n_jobs` do **not** multiply — `genConformers` completes before the alignment pool exists; conformer generation runs serially across molecules (~86% of runtime), only alignment uses the pool. 16/64/190 cores scored 64 molecules in 172.5/170.6/171.7 s. Use 16–32 cores.
  2. `generate(evaluator=env)` returns **neither `Valid` nor `Desired`** — only scorer keys. Generate with `evaluator=None`, then call `env.getScores()`.
  3. `fit()` defaults leak the threshold into training via `criteria='desired_ratio'` + `reload_interval` + `patience` — pin all three and evaluate **final-epoch weights**, not the monitor's checkpoint.

## 7. Autopsy — failures already paid for (do not re-learn)

1. **Run 01** (`rl01_opt1_mild_color_heavy`, 250 ep, 21.4 h): median ROCS → 0.000, MW −21.3% (458→303), HAC −25.9%, grammar validity 39%. Root causes: `SmoothClippedScore(5,3)` shrinkage gradient; QED w=0.25 fighting shape (EGCG QED=0.21); Harrington anchors (0.20, 0.05, 0.60, 0.80) + `rocs_clip_upper=2.0` normalized raw ROCS < 0.60 into a zero-gradient basin; serial conformer loop. Fixes: ClippedScore SA, default QED off, raw ROCS anchors (0.40, 0.05, 1.20, 0.80) on [0,2], parallel `WholeCallScorer`.
2. **Supervisor evolution** (fulopjoz; full record: [`egcg/`](https://github.com/fulopjoz/DrugEx/tree/egcg-rocs-tutorial/tutorial/advanced/rocs/egcg), successful execution = §6.1): v1 (20 ep) shrinkage again; v2 (ε=0.0, mutate accidentally aliased to agent → ε inert) catastrophic mode collapse, uniqueness 0.226; v3 (ε=0.2, mutate=prior) +0.026; **`final` (ε=0.10, `ClippedScore(5,3)`, raw [ROCS, SA] Pareto) = optimum**: median ROCS +0.1425 (0.638→0.781), max 0.992, ≥0.85 hit rate 19.4×, 50 ep / 20.7 h.
3. **190-core HPC stall** (job 14052, load avg 1.00): serial candidate embedding in core scorer. Fix: chunked multiprocessing in `chemistry.py` (`params.numThreads = 1` inside workers).
4. **PBS rejection** (jobs 25157837–40, `Insufficient amount of resource: queue_list`): `gpu` queue + ncpus=16 + 24 h walltime. Fix: `--queue gpu_long`, ncpus=4, 25 h (commit `60f58df`).
5. **RNG seed collisions**: shared seed 42/43 → identical coordinates → fake self-recovery. Fix: `verify_disjoint_conformer_seeds()`, primes 104729/42.
6. **float32 coordinate truncation** across multiprocessing queues (~2.9e-5 Å error → up to 0.069 ROCS divergence serial vs parallel). Fix: `enable_double_precision_mol_pickling()` (`Chem.PropertyPickleOptions.CoordsAsDouble`) at import time.
7. **`ETKDGv3` in-place mutation**: `params.maxIterations` mutated by first molecule (0→90) when 0; reused object starved larger molecules. Fix: fresh params per stereoisomer.
8. **Butina 1.25 Å collapse + `all - m` fallacy**: 1.25 Å collapsed EGCG's 100 conformers to 1–3 clusters (recovery cap 0.7989); `all - m` trims the *tail* = already-redundant poses, or after energy-sorted Butina it deletes exactly the high-ΔE receptor-settling conformers. Fix: `diverse` / 0.35 Å.
9. **ε dilemma**: per-token ε mixing (see §3 fact 5) injects credit-assignment noise; ε=0 collapses. Sweet spot 0.05, frozen prior.

## 8. Scoring physics — methodological landmines

- **Matched minimization is mandatory**: reference and candidate optimization states must match (numbers in §5). Matched MMFF94s K=4/C=10 already gives 0.8889 mean / 1.07 s; `WholeCallScorer` parallel minimization costs only ~+25% wall-clock vs unminimized.
- **0.95 self-recovery is a Goodhart trap**: self-recovery is the *diagonal* of the similarity matrix; RL lives on the off-diagonal (novel scaffolds score 0.30–0.75). Unminimized ETKDG ranks candidates identically to minimized (Spearman ρ = 0.967–1.000) at ~10× less cost. Better calibration target: **selectivity margin per millisecond** within ≤ 300 ms/mol.
- **Tversky scissors trap** (2.37M evals on 14,641 drug pairs): extreme subshape (0.2, 1.8) lets tiny fragments nest in the reference (decoy P95 0.6024; acetaminophen 0.6234); extreme supershape (1.8, 0.2) lets blobs engulf it (P95 0.6107); the **Dice scale α+β=1 (legacy 0.3, 0.7) leaks 86.36% of random decoy pairs above the 0.35 threshold**. Use α+β=2 (α ∈ [0.6, 1.2]) or raw TanimotoCombo, with Color weight ≥ 0.65. Best strict configs: `Tan(1.0,1.0) | 0.2S+0.8C` (worst-regime decoy P95 0.303, FP@0.35 = 0.61%) and `(1.0,1.0) | 0.1S+0.9C` (P95 0.2547, FP = 0.24%); best margins on 128-drug panel: Color-Heavy Tanimoto 0.3S+0.7C (margin 0.5533, FP 2.89%).
- **Trojan-horse shape decoys**: shape-only overlap scores Sildenafil 0.4827 vs EGCG. Never up-weight Shape over Color for polyphenol/H-bond targets. Only w_C ∈ [0.70, 0.80] crushes steric blobs.
- **ADME vs receptor settling**: water-curated (ε 78.4) diverse banks cover strained pocket poses; single global minimum does not (54.7% vs 90.2% coverage, §5).
- **Matched-protocol detail**: EGCG omnibus (`sym_mmff94s_vac_spec`, ε=1.0) reaches 1.0000 at C=15; ε=78.4 reaches 0.9990 at K≥1 with C≥3; UFF is a viable cheap alternative (0.9993 at C=3).

## 9. Where things live

**`_david/code/conformer/`** (calibration research; `results.md` = dated decision record, read it first):
`analyze.py` (stratify saved measurements) · `batch_probe.py` (batch scoring throughput) · `calibrate.py` (36-cell self-recovery grid) · `decoy_benchmark.py` (ROC-AUC/BEDROC, 16 polyphenol actives + 64 decoys) · `diagnostics.py` (reference counts + scorer cost) · `drug_panel.py` (512-drug DrugCentral panel curation) · `mega.py` (matched-preparation benchmark, seed-averaged) · `reference_bank.py` (energy-filtered, score-blind diversity ordering) · `reward_calibration.py` (4-method self-recovery + latency; 128-dev + 384-confirmation streams) · `rl_cost_probe.py` (5-epoch cost attribution) · `sampling.py` (budget plans, symmetry-aware dedup) · `self_recovery.py` · `utils.py` (sha256 provenance) · `visualization.py` (conformer-study figures).

**`_david/tests/`** (74, all passing): `test_candidate_embedding_parallel.py` (15) · `test_conformer_analysis.py` (2) · `test_joint_calibration.py` (21) · `test_pipeline_configuration.py` (15) · `test_rocs.py` (5) · `test_self_recovery.py` (4) · `test_sequencer_pipeline.py` (12).

**`_david/code/metacentrum/`**: `submit.py` (CLI verbs: `check sync submit status logs cancel pull`; stages `git archive` of a **committed** revision to `/storage/.../releases/<sha>/`; Kerberos-aware), `pbs_launcher.sh` (six-line PBS entry that `exec`s `job.py` with the stored venv), `job.py` (stdlib-only job body: spec validation, run id, release copy, pipeline invocation, log mirroring, copy-back and 128+signal exit codes), `help/` (chapters 01–06 + live-epoch monitor), `scripts/runed/` (frozen historical PBS runners). The old grid/sprint machinery and `job_template.sh` are gone; `_david/metacentrum/` (top level) is stale — only a `.pyc`.

**`_david/output/`** (key dirs): `20261028_2145_joint_calibration/`, `20261029_egcg_omnibus_diagnostic/` (12,600-cell grid + CSVs + `drug121_*` caches), `20261029_1155_reward_calibration/` (dichtator), `rl_suite_egcg_6arm/`, `rollback_checkpoints/`, `run16_sequencer_weighted_sum`, `run17_baseline_50ep`, `run18_epsilon_zero_50ep`.

**Run artifacts** (per run): `manifest.json`, `models/SequenceRNN_reinforced.pkg` (+ `_final_epoch.pkg`, `_fit.tsv`, `_smiles.tsv`), `raw_data/{baseline,reinforced}_raw.csv.gz` (with `Raw_ROCS`), `query_molecule_3d.sdf`, 9 figures, `comparison_scorecard.csv`, `summary.md`, `top_prioritized_leads.tsv`.

## 10. Operating quick reference

```bash
# Tests + lint (local, must be green before any commit)
cd /home/kolar/DrugEx && .venv/bin/python -m unittest discover -s _david/tests -v
ruff check _david

# Probe lab server (never touch VPN if this succeeds)
ssh -o BatchMode=yes -o ConnectTimeout=15 dichtator "echo OK"

# MetaCentrum sprint monitoring / results
.venv/bin/python -m _david.code.metacentrum.grid status    # or: ssh meta "qstat -u kolardavidcz"
.venv/bin/python -m _david.code.metacentrum.grid pull

# dichtator legacy suite
ssh dichtator ... launch_rl_suite.sh --status               # tmux: drugex_rl_suite

# Full pipeline (local or remote)
.venv/bin/python -m _david.code.training.pipeline --output-dir _david/output/<run_id> ...

# Sync to lab server (after commit; check git status both sides first)
git push dichtator main
```

**Obligations per completed RL run**: generate the 8 publication figures via `ResultAnalyzer`; record negative/unfavorable results (size drift, collapse) — diagnostics are first-class research output, not embarrassment.
