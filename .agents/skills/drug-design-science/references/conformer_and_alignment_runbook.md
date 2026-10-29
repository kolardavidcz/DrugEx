# 3D Conformer & Pharmacophore Alignment Runbook

This runbook documents the architecture and empirical **time-vs-accuracy trade-offs** for 3D reference ensemble preparation and candidate conformer scoring in DrugEx ligand-based reinforcement learning.

---

## 1. The Machine Learning Time-vs-Accuracy Asymmetry

In 3D ROCS-guided reinforcement learning (`RDKitROCSScorer`), reference preparation and candidate generation operate on fundamentally different computational scales:

1. **Reference Ensemble Preparation ($K_{\mathrm{ref}}$) — One-Time Setup Cost**:
   - Executed **once** before RL training begins (~1–5 seconds wall time).
   - Because reference preparation cost is amortized over hundreds of epochs, we can afford generous initial sampling (e.g. 50–100 `ETKDGv3` attempts per stereoisomer), force-field minimization with an $8\text{ kcal/mol}$ per-stereoisomer energy window, and score-blind pharmacophoric diversity selection (`selection_mode="diverse"` or Butina RMSD clustering).

2. **Candidate Conformer Generation ($K_{\mathrm{gen}}$) — Per-Epoch Multiplier & Basin Collapse**:
   - Executed **tens of thousands of times** during RL training ($N_{\mathrm{epochs}} \times N_{\mathrm{samples}} \times I_{\mathrm{isomers}}$).
   - **Never Mix Minimized References with Unminimized Candidates**: Pairing an `MMFF94s`-minimized reference with unminimized `ETKDGv3` candidates degrades EGCG color Tanimoto to `0.602` (`Combo = 0.765` at $C=20$, plateauing at `0.833` at $C=100$) because force-field relaxation shifts phenolic/ester torsion angles and H-bond donor/acceptor vectors away from raw distance-geometry templates. Reference and candidate optimization states must be **matched**.
   - **Why Fewer Minimized Conformers Beat More Unminimized Conformers**: Under truly disjoint RNG streams (`104729` vs `42`), unminimized `ETKDGv3` conformers scatter around local energy basins; compensating by increasing $C$ from $10\text{–}20$ to $50\text{–}75$ inflates the $\mathcal{O}(K_{\mathrm{ref}} \times C_{\mathrm{cand}})$ 3D Gaussian overlay cost by $5\times\text{–}7.5\times$. By contrast, matched `MMFF94s` or `UFF` minimization relaxes noisy draws into identical local minima: across the 128-drug panel (`20261029_1155_reward_calibration`), `MMFF94s` ($\epsilon_r=78.4$ or $1.0$) at $K=4, C=10$ (`0.889` mean, `1.000` median, `1.07 s`) and $K=4, C=20$ (`0.910` mean, `1.000` median, `1.88 s`) outperforms unminimized `ETKDGv3` at $K=4, C=50$ (`0.882` mean, `2.86 s`) and $K=4, C=75$ (`0.892` mean, `4.04 s`) in **both** accuracy and wall-clock speed. On EGCG ($K=4$), matched `MMFF94s` ($\epsilon_r=6.5$ or $78.4$) reaches `0.996–0.998` already at $C=3$, whereas unminimized `ETKDGv3` with masked candidate stereochemistry ($I=4$) requires $C \ge 15$ just to reach `0.961`. Use `WholeCallScorer` to parallelize candidate generation + minimization + ROCS across worker cores.

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
