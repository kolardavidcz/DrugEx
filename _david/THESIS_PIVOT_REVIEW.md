# Bachelor Thesis Architectural Review: From Fragment-Conditioned Graph Reconstruction to Pure De Novo SequenceRNN

**Author**: David Kolář  
**Thesis Topic**: Universal 3D Ligand-Based *De Novo* Drug Design in DrugEx  
**Date**: October 25, 2026  
**Document Version**: 2.0 (Post-Consultation Strategic Pivot)

---

## 1. Executive Summary: The Consultation Feedback & Strategic Pivot

During the supervisor consultation on October 25, 2026, a fundamental conceptual critique was raised regarding our initial implementation:

> *"We went down a problematic path: by supplying fragment-based generative models with substructures derived from the query molecule (EGCG), the generator was largely tasked with scaffold reconstruction ('trying to get the molecule back') rather than true de novo design."*

### First-Principles Reality of the Fragment Conditioning Trap
In our Graph-Transformer experiments (Runs 01–15, using DrugEx v3 `FragGraphExplorer`), the generator was seeded with retrosynthetically extracted scaffolds (such as the chroman ring, flavan-3-ol core, and galloyl ester fragments). While this framework allowed us to systematically calibrate reward functions and successfully replace the labile ester linker with non-ester bioisosteres (ketone, pyrrolidine amide, alkane, and benzyl ether in Run 15, reaching ROCS 1.678), it possessed a severe **inductive bias**:
1. The model was constrained to grow atoms around a predetermined core structure.
2. Even with high shape and pharmacophore alignment, the resulting chemical diversity was largely confined to decorated analogues of the query molecule.
3. True *de novo* discovery requires the algorithm to autonomously explore the vast chemical space of drug-like molecules without prior knowledge or structural scaffolding of the target ligand.

### The Strategic Decision
1. **Abandon Graph-Transformer Fragment Growing**: Terminate fragment-conditioned generation.
2. **Purge Intermediate Debris & Bloated Artifacts**: Remove 90%+ of intermediate, abandoned, or collapsed experiment outputs. Delete all model checkpoints (`*.pkg`, reclaiming >9.7 GB of disk space on `dichtator`).
3. **Preserve Only Canonical Benchmark Runs**: Retain only 3 milestone runs in [`_david/legacy/runs/`](file:///C:/Ubuntu/home/kolar/DrugEx/_david/legacy/runs/) (`run01_baseline`, `run14_harrington_decay`, `run15_harrington_deep_sampling`) as historical reference points for Chapter 4 of the thesis.
4. **Pivot to Pure De Novo SequenceRNN (DrugEx v2)**: Generate molecules character-by-character from `<START>` tokens using autoregressive SMILES recurrent neural networks, guided exclusively by our validated 3D multi-objective scoring environment.

---

## 2. Post-Mortem of the Graph-Transformer Era (Runs 01–15)

Between October 4 and October 22, 2026, fifteen major overnight runs and dozens of hyperparameter sweeps were conducted. A brutal, honest academic retrospective shows why most of these runs were intermediate steps:

| Era | Focus & Hypothesis | Result / Flaw | Fate |
| :--- | :--- | :--- | :--- |
| **Era 1 (Runs 01–03)** | Hard thresholding & boolean desirability conjunctions ($\prod \mathbb{I}(S_i \ge \tau_i)$). | Severe collapse; 0% desired candidate ratio due to step-function reward cliffs. | Pruned as obsolete. |
| **Era 2 (Runs 04–06)** | WeightedSum scalarization & empirical soft-gating ($G(S_{3D})$). | Pareto front distortion: 2D property maximization dominated 3D shape alignment; gradient cliffs at zero-mask boundaries. | Pruned as obsolete. |
| **Era 3 (Runs 07–10)** | Multi-scale scaffold portfolios (Option B) & Tversky subshape volume scoring. | Addressed global volume penalties, but exposed that larger fragments merely memorized 80% of EGCG's heavy atoms. | Pruned as obsolete. |
| **Era 4 (Runs 11–13)** | Augmented Chebyshev & Continuous Harrington Desirability. | Harrington Desirability eliminated reward collapse, smoothly balancing 3D ROCS and SAscore. | Superseded by Run 14/15. |
| **Apex (Runs 14 & 15)** | Cosine Epsilon Decay + Hawkins 10-conformer deep sampling. | **Milestone reached**: ROCS = 1.678, 10 unique leads $\ge 1.60$, discovery of non-ester bioisosteres (ketone, pyrrolidine amide). | **Preserved in `_david/legacy/runs/`** as the definitive benchmark of fragment-based generation. |

### Why Retaining 90% of Results is Meaningless
The intermediate runs were tuning steps for an inductive setup that was fundamentally flawed. Storing gigabytes of `.pkg` PyTorch weights for models that merely grew fragments onto EGCG cores represents scientific and storage debt. All model weights have been expunged. Only lightweight, fully transparent summary metrics (`summary.md`, `top_prioritized_leads.tsv`, `david_agent_fit.tsv`, and 3 presentation visuals per canonical run) remain.

---

## 3. What Was Genuinely Developed: Reusable Algorithmic Innovations

The true intellectual contribution of the work to date is **not** the graph model weights, but our mathematical and architectural modifications to the DrugEx platform. Every single one of these innovations directly transfers to the pure *de novo* SequenceRNN pipeline:

```
+-----------------------------------------------------------------------------------+
|                        OUR CORE ALGORITHMIC INNOVATIONS                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. Continuous Harrington Desirability (ContinuousHarringtonDesirability)         |
|     - Replaces step-function Pareto cliffs with smooth exponential mapping.       |
|     - D = exp( - exp( - (beta_0 + beta_1 * S) ) ), strictly monotonic in (0, 1).  |
|                                                                                   |
|  2. Tversky 3D Subshape & Color Alignment (RDKitROCSTverskyScorer)               |
|     - Replaces rigid Tanimoto volume matching with asymmetric Tversky scoring.    |
|     - Allows generated molecules to match core 3D pharmacophores without          |
|       being penalized for differing overall molecular volumes.                    |
|                                                                                   |
|  3. Scheduled Cosine Exploration Decay (DecayingSequenceExplorer)                 |
|     - Anneals epsilon exploration noise via cosine schedule:                      |
|       eps(t) = eps_min + 0.5 * (eps_max - eps_min) * (1 + cos(pi * t / T)).       |
|     - Enables broad early scaffold exploration and deep late-stage exploitation.  |
|                                                                                   |
|  4. Hawkins (2017) Ensemble Conformer Curation (HawkinsConformerGenerator)        |
|     - Generates 100 ETKDG conformers, minimises via MMFF/UFF, filters by          |
|       Delta E <= 8.0 kcal/mol, and extracts Butina RMSD basin minima.             |
|     - Evaluates generated candidates against physiological conformational ensembles|
|                                                                                   |
|  5. Batch-Unique Zero-Masking & Conformer Fallback Guards                         |
|     - Dynamically penalizes identical duplicate molecules in generation batches.  |
|     - Bulletproofs MMFF minimization against atypical coordination geometries.    |
+-----------------------------------------------------------------------------------+
```

---

## 4. The Transition: Pure De Novo SequenceRNN (DrugEx v2)

### Why SequenceRNN Solves the Consultation Critique
In the SequenceRNN paradigm:
1. **Unprompted Generation**: The generative agent receives strictly `<START>` as input. It is provided **no** seed fragments, **no** scaffold constraints, and **no** initial tokens.
2. **True De Novo Chemical Exploration**: Every bond, ring, and functional group is sampled autoregressively character-by-character from the learned probability distribution of drug-like SMILES (Papyrus prior).
3. **Objective-Guided Evolution**: The agent discovers 3D pharmacophoric arrangements through reinforcement learning pressure alone, guided by our continuous Harrington Desirability environment. If an aromatic ring or hydrogen-bonding group emerges, it is because it satisfies the 3D electrostatic and steric shape field—not because we handed it to the model.

### Architecture Comparison

```
+------------------------------------+------------------------------------+
|   LEGACY GRAPH-TRANSFORMER (v3)    |     PURE DE NOVO SEQUENCERNN (v2)  |
+------------------------------------+------------------------------------+
| • Input: Seed fragments (EGCG core)| • Input: Unprompted <START> token  |
| • Mechanism: Graph atom addition   | • Mechanism: Autoregressive SMILES |
| • Risk: Scaffold reconstruction    | • Risk: High initial invalidity    |
| • Bias: Constrained to EGCG core   | • Bias: Unconstrained exploration  |
| • Scientific Role: Comparative Ref | • Scientific Role: Primary Thesis  |
+------------------------------------+------------------------------------+
```

---

## 5. Thesis Chapter Outline & Research Narrative

This pivot transforms a potential flaw into a rigorous, publication-grade academic narrative:

- **Chapter 1: Introduction & Principles of 3D Ligand-Based Drug Design (LBDD)**
  - Molecular representations: SMILES vs. Molecular Graphs.
  - 3D Shape and Electrostatic Color Theory (ROCS, Mills-Dean pharmacophores).
  - Multi-Objective Reinforcement Learning (MORL) in chemical space.
- **Chapter 2: Mathematical & Algorithmic Formulations**
  - Continuous Multi-Attribute Harrington Desirability vs. Weighted Sum and Pareto dominance.
  - Asymmetric Tversky subshape volume formulation.
  - Multi-conformer Hawkins energy basin curation and alignment.
  - Cosine-annealed stochastic policy exploration.
- **Chapter 3: The Scaffold-Conditioning Trap (Comparative Baseline)**
  - Analysis of fragment-based Graph-Transformer generation (Runs 01 vs. 14 vs. 15).
  - Demonstration of scaffold reconstruction bias and the limits of fragment growing.
  - Milestone analysis: non-ester bioisosteric replacements (ROCS = 1.678).
- **Chapter 4: Pure De Novo Discovery via Autoregressive SequenceRNN**
  - Unprompted character-by-character generation from `<START>`.
  - RL training dynamics under Harrington Desirability.
  - Novelty, diversity, and scaffold-hopping metrics across generated populations.
- **Chapter 5: Biophysical & Molecular Dynamics Validation**
  - Induced-fit docking and molecular dynamics (MD) simulation against target macromolecules (SIRT6 / DNMT1).
  - Binding pose stability and MM-PBSA energetic comparison with native EGCG.
- **Chapter 6: Conclusion & Outlook**
  - Critical comparison of graph-based vs. sequence-based generative architectures in ligand-based drug design.

---

## 6. Repository Organization (Current State)

```
DrugEx/
├── _david/
│   ├── THESIS_PIVOT_REVIEW.md      # This document (master post-mortem & pivot rationale)
│   ├── AGENTS.md                   # Agent technical instructions & active pipeline conventions
│   ├── in_molecule.txt             # Target reference query molecule (EGCG SMILES)
│   ├── references/                 # Reference 3D coordinates (my_molecule_3d.sdf)
│   ├── code/                       # Active implementation modules
│   │   ├── config.py               # Pipeline configuration, paths, and hyperparameters
│   │   ├── environment.py          # Continuous scoring environment (Harrington, Tversky, Hawkins)
│   │   ├── models.py               # Generator abstractions & optimizers
│   │   ├── pipeline.py             # Pipeline execution orchestration
│   │   ├── schema.py               # Canonical data schema definitions
│   │   ├── utils.py                # Hardware management & environment safety
│   │   └── visualization.py        # Lead depiction, distribution radars, and RL curve visualizers
│   ├── output/                     # Pristine working directory for SequenceRNN outputs (.gitkeep)
│   ├── legacy/                     # Consolidated legacy archive
│   │   ├── runs/                   # Preserved benchmark runs (run01, run14, run15, RUN_DIFFERENCES.md)
│   │   ├── anti_cheating/          # Archived early soft-gate experiments
│   │   ├── consensus_alignment/    # Archived consensus pharmacophore tools
│   │   ├── dynamic_envelopes/      # Archived dynamic bounding box code
│   │   ├── experiments/            # Archived hypothesis runner scripts
│   │   ├── morl_scalarizations/    # Archived scalarization functions
│   │   ├── standardize_for_regeneration/ # Legacy standardization routines
│   │   └── tests.py                # Legacy regression test suite
│   └── validation/                 # Molecular dynamics and docking validation suite
```
