# Agent Guide: `_david/` Workspace

Guide for AI agents working in `/home/kolar/build_projects/drugex_edu/repo/_david`.

---

## 1. Mission & Goal
- **Core Task**: Pure Ligand-Based *De Novo* Drug Design (LBDD) centered around **exactly 1 seed query ligand**:
  EGCG derivative (`my_molecule` / SMILES: `C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O`).
- **Strict Constraint**: 100% Pure LBDD with zero receptor coordinates, crystal complexes, or docking grids.
- **Target Objective**: Generate novel, synthetically accessible modulators preserving the 3D shape, volume, and pharmacophore features of the query ligand without PAINS liabilities (e.g. catechol/polyphenol groups).

---

## 2. Skeptical Academic Architecture & Method Decisions

1. **Rejection of ProLIF / PLIF**:
   - ProLIF is designed for Protein-Ligand Interaction Fingerprints (Structure-Based Drug Design, SBDD) requiring 3D receptor atomic coordinates (e.g., CCR2 5T1A).
   - In pure LBDD, no receptor structure is available. Using docked complexes introduces severe docking function bias, force-field errors, and violates the pure LBDD premise. ProLIF is strictly out of scope.
2. **Rejection of 2D Pharmacophore Filters**:
   - Flatten 3D spatial conformation into 2D topological graph distance pairs.
   - Ignore out-of-plane orientations, chiral arrangements, and steric volume, leading to high false-positive rates.
3. **Adoption of USRCAT & ElectroShape**:
   - Pure 3D ligand-based methods implemented in RDKit (`rdkit.Chem.rdMolDescriptors.GetUSRCAT` and `GetUSR`).
   - 60-dimensional rotationally/translationally invariant moment vectors across 5 pharmacophore atom types.
   - Exact $O(1)$ inverse Manhattan distance ($S = \frac{1}{1 + \frac{1}{60} \sum |u_i - v_i|}$) executing in microseconds without alignment search traps.
4. **Resolution of Previous RL Failure Modes**:
   - Conformer under-sampling resolved: Increased reference conformers to 50 with ETKDGv3 and MMFF94 minimization.
   - Decoupled 3D shape tiers: Ultrafast rotation-invariant USRCAT moments in RL inner loops (Stages 1 and 3) provide ~100x speedup; physical Gaussian volume overlay (CDPKit ROCS: `CDPKitROCSScorer` + `RDKitConformerGenerator`) is reserved for Stage 2 scaffold funnel and Stage 4 lead validation.
   - Pareto reward hacking resolved: Replaced vulnerable Pareto ranking with `AugmentedChebyshevReward` and `GeometricMeanReward` scalarization. Extreme single-objective cheats (e.g. 0.98 QED but 0.05 3D shape) are severely penalized.
   - Fragment collapse resolved: Upgraded to native `SmoothHump(lower_x=28, upper_x=38, sigma=4)` on heavy atom count.
   - Toxicophore suppression: Integrated native `Scaffold(smart="c1c([OH])c([OH])ccc1", is_match=False)` with threshold 0.99 into RL environment to actively suppress catechol PAINS liabilities.
   - Native scaffold encoding: Replaced BRICS/RECAP cleavage in Graph-Transformer with DrugEx native `dummyMolsFromFragments()`, ensuring intact Bemis-Murcko frameworks are directly elaborated.
   - Mock simulation eliminated: All synthetic random-number generation in `thesis_benchmark.py` has been removed.

---

## 3. Directory Map & File Roles

| File / Folder | Role | Status |
|---|---|---|
| `_david/prepare_query_conformer.py` | Stage 0: 3D reference conformer ensemble (50 confs, MMFF94) & USRCAT moment array generation. | **Active** |
| `_david/SequenceRNN.py` | Stage 1: Global hit discovery via Sequence-RNN policy gradient RL (USRCAT, SmoothHump, Scaffold). | **Active** |
| `_david/scaffold_funnel.py` | Stage 2: Bemis-Murcko framework extraction, heavy-atom volume filter, PAINS purge, CDPKit ROCS, Butina clustering. | **Active** |
| `_david/GraphTransformer.py` | Stage 3: Fragment-conditioned lead optimization on discovered funnel scaffolds using `dummyMolsFromFragments()`. | **Active** |
| `_david/anti_cheating_rewards.py` | Augmented Chebyshev & Geometric Mean scalarization schemes preventing Pareto reward hacking. | **Active Tool** |
| `_david/usrcat_scorer.py` | Pure LBDD 3D shape and pharmacophore moment scorer (`drugex.training.scorers.interfaces.Scorer`). | **Active Tool** |
| `_david/pipeline.py` | Unified end-to-end multi-stage pipeline orchestrator (Stages 0 -> 1 -> 2 -> 3 -> 4). | **Active Production** |
| `_david/thesis_benchmark.py` | Stage 4: Publication benchmark suite (CDPKit ROCS, radar, PCA, boxplots, grids) on real empirical data. | **Active Production** |
| `_david/test_anti_cheating_and_api.py` | Unit tests for anti-cheating rewards, SmoothHump, toxicophore Scaffold, and dummyMolsFromFragments. | **Active Test** |
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
