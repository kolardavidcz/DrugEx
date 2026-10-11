# DrugEx Ligand-Based De Novo Drug Design: Architecture & Agent Guide

> **Audience**: AI Agents & Pair-Programmers  
> **Scope**: Universal 3D Ligand-Based De Novo Drug Design (LBDD) using Graph-Transformers in DrugEx  
> **Working Branch**: `my_personal` (strictly local commits; NEVER execute `git push`)

---

## 1. System Architecture Map

```
_david/
├── run.py                          # Primary IDE entrypoint (F5 / 1-click) & detached background daemon
├── code/
│   ├── pipeline.py                 # MoleculePipeline: orchestrates Stage 0 -> Stage 1 -> Stage 2
│   ├── prepare_query_conformer.py  # QueryConformerPreparer, get_safe_parallel_jobs, create_rocs_scorer
│   ├── GraphTransformer.py        # GraphTransformerOptimizer: RL environment & fragment prompts
│   ├── anti_cheating_rewards.py    # AntiCollapseSoftGateReward: continuous anti-collapse soft gating
│   ├── dynamic_envelope.py         # Dynamic property scorers (HAC, CLogP, TPSA with SmoothHump)
│   ├── result_analyzer.py          # ResultAnalyzer: 50-conf ROCS, VUN metrics, 9-panel distributions
│   ├── fallback_monitor.py         # FallbackMonitor: centralized tracking of all fallbacks
│   ├── monitor.py                  # Custom monitor hooks & telemetry
│   ├── visualize_molecular_space.py# t-SNE chemical manifold visualizer
│   └── legacy/                     # Superseded exploratory modules (SequenceRNN, usrcat, scaffold_funnel)
├── output/
│   ├── m4_empirical/               # Full 20-epoch empirical Graph-Transformer RL benchmark
│   ├── m4_sanity/                  # Fast 2-epoch end-to-end integration verification
│   └── archive/                    # Historical methodological issue archive & old notes
└── tests.py                        # Unit test suite verifying all invariants, scorers, and pipelines
```

---

## 2. Core Execution Stages

### Stage 0: 3D Query Reference & Scaffold Extraction (`prepare_query_conformer.py`)
- **Input**: Any valid SMILES string or `.sdf` / `.smi` file (default proof molecule: Epigallocatechin gallate, EGCG).
- **Conformer Ensemble**: Generates 50 conformers with RDKit `ETKDGv3` (`pruneRmsThresh=0.20Å`), relaxes with `MMFF94` ($\Delta E \le 8.0\text{ kcal/mol}$), and curates by Radius of Gyration ($R_g \ge R_{g,\text{median}}$) to preserve extended outward-facing pharmacophores.
- **Murcko Framework Extraction**: Extracts the core scaffold framework (e.g. flavan-3-ol for EGCG) via RDKit `MurckoScaffold` as the generative seed prompt.
- **Dynamic Property Envelopes**: Dynamically calibrates heavy atom count, CLogP, and TPSA bounds relative to the query molecule ($[0.70 \times X, 1.30 \times X]$) using native DrugEx `SmoothHump` modifiers. **Zero hardcoded bounds or SMARTS.**

### Stage 1: Graph-Transformer Lead Generation & RL (`GraphTransformer.py`)
- **Model**: `FragGraphExplorer` operating on pre-trained Papyrus prior (`Papyrus05.5_graph_trans_PT`) directly without noisy 2D transfer learning.
- **Fragment Conditioning**: Converts seed Murcko framework into graph DataLoader via `dummyMolsFromFragments()` and `FragmentCorpusEncoder(n_proc=1)`.
- **Active Scorer**: Native DrugEx `RDKitROCSScorer` with OpenEye Mills-Dean color SMARTS (`color_ff="oe_mills_dean"`), relaxed conformer parameters (`max_conformers=22`, `max_heavy_atoms=55`, `max_rotatable_bonds=25`, `timeout=30s`), and `score_type="TanimotoCombo"` (0–2 scale). All 1D moment approximations (USRCAT, ElectroShape) are strictly excluded.
- **Continuous Soft-Gate Reward**:
  $$R(\mathbf{x}) = w_{\text{3D}} \cdot S_{\text{3D}}(\mathbf{x}) + w_{\text{props}} \cdot G(S_{\text{3D}}(\mathbf{x})) \cdot \sum_i \tilde{w}_i S_{\text{prop}, i}(\mathbf{x}) + S_{\text{SA\_floor}}(\mathbf{x})$$
  $$\text{where} \quad G(S_{\text{3D}}) = \min\left(1.0, \; \frac{S_{\text{3D}}}{\tau_{\text{3D}}}\right) \quad (\tau_{\text{3D}} = 0.60), \quad w_{\text{3D}} = 0.70, \quad w_{\text{props}} = 0.30$$
  - Eliminates Pareto cheating: candidates with $S_{\text{3D}} = 0$ receive 0 reward for secondary properties ($G(0) = 0$).
  - Gradient remains non-zero for moderate $S_{\text{3D}}$, preventing early RL stall.
- **Single Synthesizability Floor**: SAscore rewarding 1.0 for $\text{SA} \le 5.5$ and penalizing only $\text{SA} > 6.0$, accommodating natural product complexity without penalty. Conflicting multi-scorer penalties (RAscore, BRICS) are purged.
- **Peak Recovery**: Custom `PeakRecoveryMonitor` tracks rolling desirability at every epoch and restores the peak historical model weights upon completion, safeguarding against late-epoch RL policy degradation.

### Stage 2: Result Analysis & Publication Profiling (`result_analyzer.py`)
- Unbiased sampling (`drop_invalid=False`, `drop_duplicates=False`) from restored peak model.
- High-fidelity 50-conformer RDKit ROCS evaluation.
- Standardized VUN metrics (Validity $\ge 80\%$, internal Uniqueness $\ge 80\%$, Novelty against Papyrus).
- Bemis-Murcko scaffold diversity ratio calculation.
- 9-panel property distribution histograms benchmarked against query envelopes.
- Publication artifacts: `lead_structures_2d.png`, `property_distributions.png`, `result_analysis_summary.json`.

---

## 3. Multiprocessing & Hardware Invariants

| Component / Function | Invariant | Technical Reason |
|---|---|---|
| `FragmentCorpusEncoder` | **Strictly `n_proc = 1`** | Multiprocessing fork after PyTorch CUDA tensor initialization causes allocator mutex deadlock. |
| `Standardization` | **Strictly `n_proc = 1`** | Process-spawning overhead exceeds sequential execution; prevents nested worker hangs. |
| `FragGraphExplorer.loaderFromFrags` | **Strictly `n_proc = 1`** | Calls `FragmentCorpusEncoder`; must remain single-process. |
| `RDKitConformerGenerator.genConformers` | **Strictly `n_proc = 1`** (and `num_threads = 0`) | Coordinates write sequentially to a single destination SDF via `Chem.SDWriter`; parallel forks corrupt file. |
| `RDKitROCSScorer.getScores` | **Capped at max 24 CPU cores** | Gaussian volume integrals safely parallelize across molecules via `pool_context().Pool()`. Capped at half host CPUs to avoid memory exhaustion (cell-615 hang). |
| Multiprocessing context | **`DRUGEX_MP_CONTEXT="forkserver"`** | Forks workers from a pristine server process that never inherited PyTorch CUDA threads, eliminating deadlocks. |
| GPU Execution | **Target GPU 0 on `dichtator`** | `dichtator` has NVIDIA RTX PRO 4000 Blackwell on GPU 0 (GPU 1 is in a driver error state). |

---

## 4. Development, Testing & Verification Workflows

### Running Locally (WSL):
```bash
# Run unit tests verifying anti-cheating rewards and API compliance
uv run pytest _david/tests.py -v

# Run 60-point LBDD requirements verification suite
uv run pytest testing/test_lbdd_requirements.py -q

# Run 14-point adversarial verification suite
uv run pytest testing/adversarial_m4_empirical_verification.py -q
```

### Running on Remote Hardware (`dichtator`):
- Connect via WSL multiplexed SSH (`~/.ssh/config` reuses `ControlMaster` socket in ~34ms):
  ```bash
  wsl ssh -o BatchMode=yes dichtator "nvidia-smi; free -h"
  ```
- Trigger detached background run (survives closing IDE / VPN):
  ```bash
  wsl ssh -o BatchMode=yes dichtator "cd ~/DrugEx && .venv/bin/python _david/run.py --mode full --epochs 20 --n-samples 200"
  ```
- Monitor live execution:
  ```bash
  wsl ssh -o BatchMode=yes dichtator "tail -f ~/DrugEx/_david/output/<RUN_ID>/pipeline.log"
  ```

---

## 5. Coding & Contribution Rules

1. **Academic Honesty**: No synthetic simulation data. All numbers must come from authentic RDKit/DrugEx calculations.
2. **Type Annotations**: All Python functions require 100% explicit type hints on parameters and return types.
3. **Targeted Comments**: Cheminformatics library (outside snatdart one like os, numpy, ...) functions follow `# [PACKAGE: NAME] ...` WHY: / WHAT: / HOW: . Avoid line-by-line noise.
4. **Direct File Tools Only**: Use `write_to_file` and `replace_file_content`. Never run ad-hoc scripts to patch project code.
5. **Git Safety**: Stage and commit locally with clear commit messages (`git add . && git commit -m "..."`). **NEVER execute `git push`**.
