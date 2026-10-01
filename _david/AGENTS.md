# Agent Guide: `_david/` Workspace

Guide for AI agents working in `/home/kolar/build_projects/drugex_edu/repo/_david`.

---

## 1. Mission & Goal
- **Core Task**: Pure Universal Ligand-Based *De Novo* Drug Design (LBDD) centered around an arbitrary seed query ligand (defaulting to EGCG: `MY_MOLECULE`).
- **Strict Constraint**: 100% Pure LBDD with zero receptor coordinates, crystal complexes, or docking grids.
- **Target Objective**: Generate novel, synthetically accessible modulators preserving the 3D shape, volume, and electrostatic pharmacophore features of the query ligand without PAINS liabilities or structural degradation.

---

## 2. Skeptical Academic Architecture & Method Decisions

1. **Rejection of ProLIF / PLIF**:
   - ProLIF is designed for Protein-Ligand Interaction Fingerprints (Structure-Based Drug Design, SBDD) requiring 3D receptor atomic coordinates (e.g., CCR2 5T1A).
   - In pure LBDD, no receptor structure is available. Using docked complexes introduces severe docking function bias, force-field errors, and violates the pure LBDD premise. ProLIF is strictly out of scope.
2. **Rejection of 2D Pharmacophore Filters**:
   - Flatten 3D spatial conformation into 2D topological graph distance pairs.
   - Ignore out-of-plane orientations, chiral arrangements, and steric volume, leading to high false-positive rates.
3. **Adoption of 75-D ElectroShape Moments**:
   - Superior to 60-D USRCAT by evaluating both 3D steric envelope and Gasteiger partial charge distributions across 5 spatial centroids (geometric centroid, positive/negative centroids, extreme charge positions).
   - Prevents uncharged, non-polar lipophilic decoys from gaming shape metrics in $<0.08\text{ ms}$ per conformer.
4. **Gated Two-Tier 3D Shape Strategy**:
   - **Tier 1 (RL & Pre-Filters)**: Ultrafast rotation/translation-invariant 75-D ElectroShape moments in RL inner loops (Stages 1 and 3) provide ~100x speedup.
   - **Tier 2 (Gated Physical Gaussian Overlay)**: Physical Gaussian volume overlay (RDKit / CDPKit ROCS) in Stage 2 Chemotype Funnel is **strictly gated to run ONLY on candidate molecules that pass all Tier 1 checks** (valid graph, dynamic size bounds, ElectroShape >= 0.45, RAscore >= 0.50, QED >= 0.40, PAINS absence).
5. **Universal LBDD Architecture (Root Improvements)**:
   - **Dynamic Size Envelope**: Size modifier is dynamically centered on the query seed's heavy atom count (`SmoothHump(lower_x=max(6.0, seed_hac-5), upper_x=seed_hac+5, sigma=4)`).
   - **Universal 5-Objective RL Environment**: Defaults to 5 universal small-molecule drug design objectives: `[ElectroShape, Heavy, RAscore, QED, SA]`.
   - **Optional Target Toxicophores**: Target-specific liabilities are decoupled via optional parameter `toxicophore_smarts: str | Sequence[str] | None = None`.
   - **Universal PAINS Screening**: `ScaffoldFunnel` applies RDKit's standard 480-rule `FilterCatalog` PAINS library alongside user-specified toxicophores.
6. **Resolution of Previous RL Failure Modes**:
   - Conformer under-sampling resolved: Increased reference conformers to 50 with ETKDGv3 and MMFF94 minimization.
   - Pareto reward hacking resolved: Replaced vulnerable Pareto ranking with `AugmentedChebyshevReward` scalarization. Extreme single-objective cheats (e.g. 0.98 QED but 0.05 3D shape) are severely penalized.
   - Native scaffold encoding: Replaced BRICS/RECAP cleavage in Graph-Transformer with DrugEx native `dummyMolsFromFragments()`, ensuring intact Bemis-Murcko frameworks are directly elaborated.
   - Mock simulation eliminated: All synthetic random-number generation in `thesis_benchmark.py` has been removed.

---

## 3. Directory Map & File Roles

| File / Folder | Role | Status |
|---|---|---|
| `_david/prepare_query_conformer.py` | Stage 0: 3D reference conformer ensemble (50 confs, MMFF94) & ElectroShape/USRCAT moment arrays. | **Active** |
| `_david/SequenceRNN.py` | Stage 1: Global hit discovery via Sequence-RNN policy gradient RL (75-D ElectroShape, dynamic SmoothHump). | **Active** |
| `_david/scaffold_funnel.py` | Stage 2: Bemis-Murcko framework extraction, dynamic size envelope, PAINS purge, gated ROCS, Butina clustering. | **Active** |
| `_david/GraphTransformer.py` | Stage 3: Fragment-conditioned lead optimization on discovered funnel scaffolds using `dummyMolsFromFragments()`. | **Active** |
| `_david/anti_cheating_rewards.py` | Augmented Chebyshev & Geometric Mean scalarization schemes preventing Pareto reward hacking. | **Active Tool** |
| `_david/usrcat_scorer.py` | Pure LBDD 3D shape and electrostatic moment scorer (`drugex.training.scorers.interfaces.Scorer`). | **Active Tool** |
| `_david/pipeline.py` | Unified end-to-end multi-stage pipeline orchestrator (Stages 0 -> 1 -> 2 -> 3 -> 4). | **Active Production** |
| `_david/thesis_benchmark.py` | Stage 4: Publication benchmark suite (CDPKit ROCS, radar, PCA, boxplots, grids) on real empirical data. | **Active Production** |
| `_david/tests.py` | Empirical unit test suite verifying anti-cheating rewards, dynamic size bounds, gated ROCS, and API parity. | **Active Test** |
| `_david/dodo_notes.md` | Academic evaluation notes, query molecule SMILES, and failure mode documentation. | **Reference** |
| `_david/outputs/` | Pipeline runs, benchmark tables, and generated molecule figures. | **Output Storage** |

---

## 4. Code Standards & Conventions

1. **Python Type Annotations**: Full explicit type annotations on function/method parameters and return types: `def func(param: Type, ...) -> ReturnType:`.
2. **Structured Google / NumPy Docstrings**: High-value docstrings with concise summaries, `Args` / `Parameters`, and `Returns` for every class, method, and function.
3. **Compact `#? [Library]` Annotations**: Every external non-DrugEx library call (RDKit, NumPy, Pandas, Matplotlib, SciPy, Scikit-learn) must have a 1-3 line note explaining its role.
4. **No Synthetic Mock Fallbacks**: Code must process authentic empirical candidate data and never fabricate random distributions.
5. **Zero Compilation / Lint Errors**: Clean compilation verified with `python3 -m py_compile` across all files in `_david/`.

---

## 5. Execution & Operations
- **Remote Host**: `dichtator` (`kolarv@147.33.72.143`), branch `my_personal`.
- **Python Env**: `/home/kolarv/DrugEx/.venv/bin/python`.
- **VPN Safety**: Never disrupt active Cisco VPN. Probe with `ssh -o BatchMode=yes dichtator "echo OK"`.
- **Editing Protocol**: Use direct file tools (`write_to_file`, `replace_file_content`).
- **Git**: Automatically stage and commit locally (`git add ... && git commit -m "..."`). **Never run `git push`**.
