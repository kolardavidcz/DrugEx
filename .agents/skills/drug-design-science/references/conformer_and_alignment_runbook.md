# 3D Conformer & Pharmacophore Alignment Runbook

This runbook documents the architecture and empirical **time-vs-accuracy trade-offs** for 3D reference ensemble preparation and candidate conformer scoring in DrugEx ligand-based reinforcement learning.

---

## 1. The Machine Learning Time-vs-Accuracy Asymmetry

In 3D ROCS-guided reinforcement learning (`RDKitROCSScorer`), reference preparation and candidate generation operate on fundamentally different computational scales:

1. **Reference Ensemble Preparation ($K_{\mathrm{ref}}$) — One-Time Setup Cost**:
   - Executed **once** before RL training begins (~1–5 seconds wall time).
   - Because reference preparation cost is amortized over hundreds of epochs, we can afford generous initial sampling (e.g. 50–100 `ETKDGv3` attempts per stereoisomer), force-field minimization with an $8\text{ kcal/mol}$ per-stereoisomer energy window, and score-blind pharmacophoric diversity selection (`selection_mode="diverse"` or Butina RMSD clustering).

2. **Candidate Conformer Generation ($K_{\mathrm{gen}}$) — Per-Epoch Multiplier**:
   - Executed **tens of thousands of times** during RL training ($N_{\mathrm{epochs}} \times N_{\mathrm{samples}} \times I_{\mathrm{isomers}}$).
   - Every millisecond spent per candidate multiplies across the entire RL run.
   - **Why `minimize_candidates=False` is the RL Default**: Force-field minimization (`MMFF94s` or `UFF`) on every generated candidate increases batch scoring latency by **3–5×** (and runs serially in `RDKitROCSScorer` unless wrapped in `WholeCallScorer`). Empirical calibration across 512 FDA-approved drugs (`_david/code/conformer/results.md`) shows that unminimized `ETKDGv3` (`minimize_candidates=False`) with $K_{\mathrm{ref}}=4, K_{\mathrm{gen}}=15, I_{\mathrm{cand}}=4$ achieves **$0.955$ mean self-recovery** (`0.985` median) at **$0.225\text{ s/molecule}$**, outperforming force-field minimized candidates on the speed–accuracy Pareto frontier.

---

## 2. Dielectric Screening Is an Empirical Hypothesis, Not a Dogma

RDKit's `MMFFMolProperties` defines dielectric model `1` as constant ($\epsilon_r$) and model `2` as distance-dependent, with default $\epsilon_r = 1.0$.
- Scaling Coulombic terms via $\epsilon_r = 6.5$ or $\epsilon_r = 78.4$ in `MMFF94s` does **not** implement explicit solvent or continuum solvation free energy (GBSA/PBSA).
- Treat force-field variants (`Optimization("none")`, `Optimization("MMFF94")`, `Optimization("MMFF94s", 6.5)`, `Optimization("MMFF94s", 78.4)`, `Optimization("UFF")`) as empirical screening protocols whose utility must be justified by wall-clock cost and downstream chemical validity.

---

## 3. Mandatory Engineering & Scientific Invariants

Whenever modifying or invoking `_david/code/chemistry.py`, `_david/code/reference.py`, or `_david/code/rocs.py`:

1. **Disjoint Conformer RNG Seed Streams (`verify_disjoint_conformer_seeds`)**:
   - With `enableSequentialRandomSeeds = False`, RDKit seeds conformer index $i$ ($0 \le i < N$) using $\text{effective\_seed} = \text{base\_seed} \times (i + 1)$.
   - Never share the same seed (`42` vs `42`) or small adjacent seeds (`43` vs `42`, which collide at $43 \times 42 = 1806 = 42 \times 43$) between reference and candidate conformers in self-recovery or calibration studies.
   - Production defaults use `DEFAULT_REFERENCE_SEED = 104729` (prime) and `DEFAULT_CANDIDATE_SEED = 42`.

2. **Per-Graph `ETKDGv3` Parameter Isolation**:
   - When `params.maxIterations == 0` (RDKit's auto-scaling default), `AllChem.EmbedMultipleConfs` mutates `params.maxIterations` **in-place** based on the first molecule's atom count.
   - `CandidateConformerGenerator.genConformers` must instantiate a fresh `ETKDGv3` object (`_create_fresh_etkdg()`) inside the per-isomer loop so batch ordering never alters cyclic-ring embedding budgets.

3. **Double-Precision Coordinate Transport (`CoordsAsDouble`)**:
   - RDKit's default `Chem.Mol` pickle rounds 3D coordinates to `float32` ($\sim 2.9 \times 10^{-5}\text{ \AA}$ error), which can flip local Gaussian shape/color overlay optima across multiprocessing workers (`n_jobs > 1`).
   - `_david/code/chemistry.py` enables `Chem.PropertyPickleOptions.CoordsAsDouble` at import time.

4. **Pharmacophoric Diversity vs. Pure Geometric RMSD (`PrepareMolecule`)**:
   - Heavy-atom Butina RMSD pruning (`select_conformers`) is blind to polar hydrogen rotamers (e.g. hydroxyl/amine orientations) that alter ROCS color Tanimoto, and global energy slicing can starve higher-energy stereoisomers.
   - `PrepareMolecule` supports `selection_mode="diverse"` (`diverse_order`) and `max_reference_isomers`, which seed each stereoisomer with its lowest-energy conformer and greedily select remaining references by farthest-first shape+color dissimilarity.
