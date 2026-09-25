# Agent Guide: `_david/` Workspace

Guide for AI agents working in `/home/kolarv/DrugEx/_david`.

---

## 1. Mission & Goal
- **Core Task**: Ligand-based *de novo* drug design starting from **exactly 1 seed query ligand** (e.g. allosteric modulator from `dodo_notes.md`).
- **Target Objective**: Generate novel, synthetically accessible modulators preserving 3D shape and pharmacophore features of the seed ligand (e.g., CCR2 5T1A allosteric pocket interactions).
- **Core Architecture**: **Graph-Transformer** (`drugex.models.graph_transformer`). The 2D Sequence-RNN pipeline is retired.

---

## 2. The Data Reality & Strategy
Official DrugEx tutorials assume 1,000–4,000 clean ligands with known crystal structures (`rocs_rl_tutorial.ipynb`). We have **1 seed ligand** and ~150–200 noisy chemogenomic neighbors harvested from Papyrus.

### The 4-Step Pipeline:
1. **Bioactivity Space Discovery** (`receptor_similar/`):
   - Query Papyrus / ChEMBL using seed SMILES to retrieve ~150–200 active neighbor compounds and identify candidate receptor targets.
2. **Warm-Up Transfer Learning (TL) — 3 to 5 Epochs Only**:
   - Deconstruct harvested compounds into BRICS fragments (`FragmentCorpusEncoder`).
   - Train Graph-Transformer for **strictly 3–5 epochs** on pretrained `Papyrus05.5_graph_trans_PT`.
   - *Rationale*: A short warm-up biases attention heads toward the relevant chemical subspace and avoids a cold-start gradient desert ($R \approx 0$), while preventing catastrophic memorization / mode collapse.
3. **Multi-Objective Reinforcement Learning (RL)** (`3D_bro.py`):
   - Explore chemical space using Graph-Transformer with multi-objective rewards:
     - **3D Shape & Pharmacophore Matching**: Tanimoto shape + color similarity against seed ligand conformers.
     - **Synthetic Accessibility**: SA score filter / reward.
     - **Uniqueness & Novelty**: Diversity penalty against mode collapse.
4. **Hybrid 2D-to-3D Screening (Performance Defense)**:
   - Full 3D conformer generation (50–200 conformers) and 3D alignment during every RL rollout is a CPU bottleneck.
   - Use fast 2D graph / fingerprint similarity during rollout to filter out poor candidates; run full 3D shape and color alignment only on top-percentile candidates.

---

## 3. Directory Map & File Roles

| File / Folder | Role | Status |
|---|---|---|
| `_david/3D_bro.py` | **Primary Production Pipeline**. Graph-Transformer fine-tuning & 3D RL. | **Active** |
| `_david/not_ment_to_be.py` | Legacy 2D Sequence-RNN script. Preserved for reference (metrics, logging, visualizer patterns). | **Archived (Read-Only)** |
| `_david/receptor_similar/` | Papyrus/ChEMBL bioactivity retrieval & candidate clustering pipeline. | **Active Tool** |
| `_david/visualize_molecular_space.py` | UMAP / PCA chemical space visualization & distribution checks. | **Active Tool** |
| `_david/dodo_notes.md` | User context, seed ligand SMILES, presentation goals, Discord notes. | **Reference** |
| `_david/outputs/` | Timestamped run outputs, checkpoints, and generated molecule grids. | **Output Storage** |
| `_david/agent.md` | Symlink pointing to `AGENTS.md`. | **Config** |

---

## 4. Strict Code Conventions in `_david`
Agents editing code in `_david` must follow these rules:

1. **Parameter-Only Type Annotations**:
   - Annotate function/method parameters: `def func(param: Type):`
   - **Do NOT** add return types (`-> ReturnType`) or class attribute annotations in `_david` scripts.
2. **Zero Docstrings**:
   - Do **NOT** add docstrings inside functions or classes in `_david`. Keep files dense and uncluttered. Use brief inline comments only for non-obvious logic.
3. **Uppercase Constants at Top**:
   - Keep all hyperparameters, paths, batch sizes, thresholds, and seeds as uppercase constants at the top of the file.
4. **Module-First, Class-Second Imports**:
   - Import module level first (`import drugex.data.datasets as datasets`).
   - Import specific classes second (`from drugex.data.datasets import GraphFragDataSet`).
5. **Fast Code Reading**:
   - Inspect `~/build_projects/drugex_edu/repo` on local WSL for instant, zero-latency inspection of DrugEx source code and tutorials.

---

## 5. Execution & Operations
- **Remote Host**: `dichtator` (`kolarv@147.33.72.143`), branch `my_personal`.
- **Python Env**: `/home/kolarv/DrugEx/.venv/bin/python`.
- **VPN Safety**: Never disrupt active Cisco VPN. Probe with `ssh -o BatchMode=yes dichtator "echo OK"`.
- **Editing Protocol**: Write locally to scratch, SCP to `dichtator:/home/kolarv/DrugEx/...`.
- **Git**: Automatically stage and commit locally (`git add ... && git commit -m "..."`). **Never run `git push`**.
