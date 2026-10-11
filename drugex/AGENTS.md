# DrugEx Core Architecture Guide & Agent Specification

> **Audience**: AI Agents & Computational Chemistry Pair-Programmers  
> **Scope**: The authoritative architecture of the `drugex/` core engine (Data Flow, Model Graphs, RL Exploration, Scoring Backends, and Parallelization)  
> **Documentation Layer**: PEP 484 `.pyi` Type Stubs (strictly decoupled from executable `.py` files)  
> **Working Rules**: Zero broken tests/lints; local commits only (`git commit`); NEVER `git push`.

---

## 1. System Architecture & Package Layout

`drugex/` is a modular, high-performance deep learning platform for ligand-based *de novo* drug design (LBDD). The codebase strictly decouples clean runtime execution in `.py` from rich developer typing and docstrings in `.pyi`.

```
drugex/
├── __init__.py (.pyi)          # Top-level exports and package metadata
├── dataset.py (.pyi)           # CLI entrypoint for dataset preprocessing and vocabulary compilation
├── train.py (.pyi)             # CLI entrypoint for pretraining and RL fine-tuning
├── generate.py (.pyi)          # CLI entrypoint for molecular generation and sampling
├── data/                       # Data ingestion, vocabularies, splitting, and tensor loaders
│   ├── corpus/                 # Corpus tokenizers, vocabulary encoders (SMILES & Graph)
│   ├── datasets.py (.pyi)      # PyTorch dataset abstractions (SmilesDataSet, SmilesFragDataSet, GraphFragDataSet)
│   ├── fragments.py (.pyi)     # Fragment extraction, combinations, and prompt encoding
│   ├── interfaces.py (.pyi)    # Abstract contracts for datasets and data splitters
│   └── processing.py (.pyi)    # Parallel standardization and corpus transformation pipelines
├── molecules/                  # Low-level cheminformatics and RDKit molecule adapters
│   ├── converters/             # Standardizers, fragmenters (BRICS, RECAP), dummy molecule injectors
│   ├── suppliers.py (.pyi)     # Chunked lazy molecular suppliers for parallel pipelines
│   └── mol.py (.pyi)           # Lightweight molecular wrapper classes
├── training/                   # Neural network training, RL agents, and scoring environments
│   ├── generators/             # Deep generative network architectures (SequenceRNN, SequenceTransformer, GraphTransformer)
│   ├── explorers/              # Reinforcement learning policy explorers (SequenceExplorer, FragGraphExplorer, etc.)
│   ├── scorers/                # 3D ROCS shape/color overlap, QSPR models, SAscore, modifiers, conformers
│   ├── environment.py (.pyi)   # Multi-objective reward aggregation and scoring environments
│   ├── rewards.py (.pyi)       # Reward schemes (Pareto rankings, crowding distance, weighted sums)
│   ├── monitors.py (.pyi)      # Training hooks, early stopping, and peak model recovery
│   └── interfaces.py (.pyi)    # Core abstract base classes for models, generators, and explorers
├── parallel/                   # Safe multiprocessing pool workers and chunked result collectors
└── utils/                      # Fingerprint dispatchers, Pareto sorting, downloads, and optimization math
```

---

## 2. Documentation Architecture: PEP 484 `.pyi` Stub Layer

To balance **runtime token efficiency** for AI agents with **comprehensive IDE developer tooling** (autocompletion, parameter docstrings, tensor shape annotations, KaTeX formulas), DrugEx utilizes a dual-file architecture:

1. **Executable `.py` Source**:
   - Kept lean, direct, and unpolluted by multi-page docstrings.
   - Contains execution logic, PyTorch forward loops, RDKit calls, and CLI bindings.
2. **PEP 484 `.pyi` Type Stubs**:
   - Reside alongside every `.py` file (e.g., `drugex/data/corpus/corpus.pyi`).
   - Houses exhaustive NumPy/Google-style docstrings, parameter types, mathematical formulations, and structural edge cases.
   - Provides instant hover tooltips, type hints, and parameter suggestions in VS Code and PyCharm.
3. **Deterministic AST Extraction & Verification**:
   - Generated deterministically from git history via `scripts/extract_git_stubs.py` using Python's `ast` module.
   - Parameter names and counts match current AST signatures with 100% parity (zero signature drift).
   - Verified clean with Mypy (`mypy --ignore-missing-imports drugex`) with zero errors and zero warnings.

---

## 3. Data Pipeline & Vocabulary Encodings (`drugex.data`)

```
   Raw Ligands (SMILES / SDF)
               │
               ▼
   Standardization / CleanSMILES (ParallelProcessor)
               │
               ├───────────────────────────────────────────────┐
               ▼                                               ▼
       SMILES Sequence Route                           Molecular Graph Route
               │                                               │
    Fragmenter (BRICS / RECAP)                         Fragmenter (BRICS / RECAP)
               │                                               │
       SmilesFragDataSet                                GraphFragDataSet
               │                                               │
        VocSmiles Encoder                               VocGraph Encoder
   (Token Indices: C, N, =, #, [NH+])             (Nodes: Atom Types, Charges, Hybridizations;
               │                                   Edges: Adjacency, Bond Orders)
               ▼                                               ▼
   LongTensor [Batch, 2*MaxLen]                    GraphTuple (Node Matrix, Adj Matrix)
```

### 3.1 Vocabulary Architectures
- **`SequenceVocabulary` (`drugex.data.corpus.vocabulary.VocSmiles`)**:
  - Encodes molecular strings into discrete integer tokens using regular expression tokenization matching organic subsets, brackets, ions, and ring indices.
  - Special tokens: `_` (pad), `GO` (sequence start), `EOS` (end-of-sequence).
  - Handles fragment conditioning by concatenating prefix prompt tokens with generated target tokens separated by delimiters.
- **`VocGraph` (`drugex.data.corpus.vocabulary.VocGraph`)**:
  - Graph-based vocabulary encoding chemical atom-node vectors and discrete bond-edge matrices.
  - Atoms are represented by feature vectors: atomic number, formal charge, implicit hydrogen count, and hybridization state ($sp, sp^2, sp^3$).
  - Bonds are encoded as discrete multi-channel adjacency tensors (Single, Double, Triple, Aromatic, No-Bond).

### 3.2 Dataset Classes & Splitters
- **`SmilesDataSet` & `SmilesFragDataSet`**:
  - Wraps tabular TSV datasets for sequence models. Encodes input fragments and output targets into fixed-length padded integer tensors.
- **`GraphFragDataSet`**:
  - Encodes graph pairs (seed scaffold fragment $\to$ complete molecule). Produces graph matrices formatted for `GraphTransformer` attention heads.
- **`DataSplitter` & `RandomSplitter`**:
  - Deterministic splitting of chemical corpuses into train, validation, and test splits with fixed random seeds.

---

## 4. Deep Generative Network Graphs (`drugex.training.generators`)

DrugEx features three generative backends implementing the `drugex.training.interfaces.Model` and `drugex.training.generators.interfaces.Generator` contracts:

### 4.1 SequenceRNN (`drugex.training.generators.sequence_rnn.SequenceRNN`)
- **Architecture**: Multi-layer autoregressive Recurrent Neural Network (LSTM / GRU).
- **Forward Flow**:
  1. Input token index sequence $\mathbf{x}_{1:t-1}$ embedded via `nn.Embedding`.
  2. Sequential processing through $L$ stacked recurrent layers with optional dropout.
  3. Linear projection to vocabulary logit space $\mathbb{R}^{|\mathcal{V}|}$.
  4. Softmax probability sampling or greedy argmax token emission.
- **Best Suited For**: Fast linear *de novo* generation, unconstrained chemical library expansion.

### 4.2 SequenceTransformer (`drugex.training.generators.sequence_transformer.SequenceTransformer`)
- **Architecture**: Causal autoregressive Transformer decoder with multi-head self-attention.
- **Attention Masking**: Triangular causal mask prevents attending to future tokens ($t' > t$).
- **Fragment-to-Lead Conditioning**: Scaffold prompt tokens attend bidirectionally, while generated molecule extension tokens attend causally.
- **Positional Encoding**: Fixed sinusoidal positional encodings:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 4.3 GraphTransformer (`drugex.training.generators.graph_transformer.GraphTransformer`)
- **Architecture**: Non-autoregressive or step-wise molecular graph generation network.
- **Node & Edge Representations**:
  - Atom embeddings: node features projected through linear layers.
  - Spatial edge bias: Shortest Path Distance (SPD) and bond type features injected directly into attention logits:
    $$A_{ij} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}} + b_{\text{bond}}(i, j) + b_{\text{spd}}(i, j)$$
- **Multi-Task Prediction Heads**:
  - `out_atom`: Predicts the identity and properties of the next added atom.
  - `out_bond`: Predicts bond orders connecting newly generated atoms to existing subgraphs.
  - `out_curr` & `out_prev`: Identifies anchor nodes participating in cycle closures.
- **Physical Invariants**: Guarantees valence validity and avoids non-physical topological artifacts inherent to linear string tokenization.

---

## 5. Reinforcement Learning, Environments & Multi-Objective Rewards

### 5.1 Policy Gradient Formulation (REINFORCE)
During reinforcement learning, generative models are tuned toward bioactivity and 3D shape targets:
$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\mathbf{x} \sim p_\theta(\mathbf{x})} \left[ \nabla_\theta \log p_\theta(\mathbf{x}) \cdot \left( R(\mathbf{x}) - b \right) \right]$$
where $R(\mathbf{x}) \in [0, 1]$ is the aggregated reward from `drugex.training.environment.Environment`, and $b$ is a moving baseline (or critic value) mitigating gradient variance.

### 5.2 Environment & Scorer Orchestration (`drugex.training.environment.Environment`)
- Orchestrates multiple parallel scoring components:
  1. Primary 3D shape & pharmacophore matching ($S_{\text{3D}}$).
  2. Secondary target QSPR models ($S_{\text{bio}}$).
  3. Physicochemical envelopes (MW, CLogP, TPSA).
  4. Synthesizability floor ($S_{\text{SA}}$).
- Applies modifier functions (`ScoreModifier`) to raw values before reward calculation.

### 5.3 Multi-Objective Reward Schemes (`drugex.training.rewards`)
- **`WeightedSum`**: Linear combination of individual normalized scores: $R(\mathbf{x}) = \sum_i w_i S_i(\mathbf{x})$.
- **`ParetoRewardScheme` / `ParetoCrowdingDistance` / `ParetoTanimotoDistance`**:
  - Non-dominated sorting partitions generated candidates into Pareto tiers $\mathcal{F}_1, \mathcal{F}_2, \dots, \mathcal{F}_k$.
  - Crowding distance or chemical Tanimoto diversity distance breaks ties within tiers to maximize chemical novelty.
- **Continuous Soft-Gating (Anti-Cheating Principle)**:
  - In 3D ligand-based designs, secondary property maximization must be gated on non-zero 3D overlap:
    $$R(\mathbf{x}) = w_{\text{3D}} \cdot S_{\text{3D}}(\mathbf{x}) + w_{\text{props}} \cdot G(S_{\text{3D}}(\mathbf{x})) \cdot \sum_i \tilde{w}_i S_{\text{prop}, i}(\mathbf{x})$$
    $$\text{with} \quad G(S_{\text{3D}}) = \min\left(1.0, \; \frac{S_{\text{3D}}}{\tau_{\text{3D}}}\right)$$
  - Prevents the policy from exploiting easy-to-optimize 2D metrics while generating conformations with zero binding site overlap.

---

## 6. Scoring Backends & Conformer Generation (`drugex.training.scorers`)

### 6.1 3D Shape & Color Overlap (ROCS)
- **`RDKitROCSScorer` (`drugex.training.scorers.rocs_rdkit`)**:
  - Open-source, production-grade 3D Gaussian volume overlap calculator.
  - Implements Grant & Pickup analytic Gaussian shape overlap:
    $$V_{AB} = \int \rho_A(\mathbf{r}) \rho_B(\mathbf{r}) d\mathbf{r}, \quad \text{ShapeTanimoto} = \frac{V_{AB}}{V_{AA} + V_{BB} - V_{AB}}$$
  - **Color Force Field**: Uses OpenEye Mills-Dean color definitions (`oe_mills_dean`) mapped to 6 chemical features (H-bond donor, H-bond acceptor, positive charge, negative charge, hydrophobe, aromatic ring).
  - Calculates `ColorTanimoto` and returns `TanimotoCombo = ShapeTanimoto + ColorTanimoto` (range $[0, 2]$).
- **`CDPKitROCSScorer` (`drugex.training.scorers.rocs_cdpkit`)**:
  - High-performance C++ backend via CDPKit bindings for ultra-fast pharmacophore overlay.
- **`OpenEyeROCSScorer` (`drugex.training.scorers.rocs_openeye`)**:
  - Official OpenEye OEChem/OEROCS backend where enterprise licenses are available.

### 6.2 Conformer Generators (`drugex.training.scorers.conformer_generators`)
- **`RDKitConformerGenerator`**:
  - Generates 3D ensembles using RDKit `ETKDGv3` with distance geometry.
  - Energy minimizations via `MMFF94` force field.
  - Enforces strict single-process file writing to avoid corrupting output SDF streams.

### 6.3 QSPR Models & Synthesizability
- **`QSPRScorer` (`drugex.training.scorers.qsprpred`)**: Evaluates affinity/activity predictions using serialized QSPRpred pipelines.
- **`SAScorer` (`drugex.training.scorers.sascorer`)**: Ertl & Schuffenhauer synthetic accessibility score based on fragment contributions and complexity penalties.
- **`RAScorer` (`drugex.training.scorers.ra_scorer`)**: Machine learning retrosynthetic accessibility score predicting step count and synthetic feasibility.

### 6.4 Score Modifiers (`drugex.training.scorers.modifiers`)
Transform raw physicochemical parameters into normalized $[0, 1]$ desirability functions:
- **`ClippedScore` / `ThresholdedLinear`**: Hard-bounded piecewise linear ramps.
- **`Gaussian` / `SmoothHump`**: Smooth bell-shaped curves optimal for property windows (e.g., CLogP within $2.0 \pm 1.0$):
  $$f(x) = \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

---

## 7. Parallelization & Hardware Invariants (`drugex.parallel`)

| Component | Multiprocessing Invariant | Hardware Rationale |
|---|---|---|
| `RDKitROCSScorer.getScores` | **Capped at max 24 CPU cores** | Gaussian volume integrals safely parallelize across molecules via `pool_context().Pool()`. Capped at half host CPUs to avoid memory exhaustion. |
| `FragmentCorpusEncoder` | **Strictly `n_proc = 1`** | Multiprocessing fork after PyTorch CUDA tensor initialization causes allocator mutex deadlock. |
| `Standardization` | **Strictly `n_proc = 1`** | Process-spawning overhead exceeds sequential execution; prevents nested worker hangs. |
| `RDKitConformerGenerator.genConformers` | **Strictly `n_proc = 1`** | Coordinates write sequentially to a single destination SDF via `Chem.SDWriter`; parallel forks corrupt file. |
| Multiprocessing Context | **`DRUGEX_MP_CONTEXT="forkserver"`** | Pristine server process never inherits PyTorch CUDA threads, eliminating deadlocks. |
| Remote Lab GPU | **NVIDIA RTX PRO 4000 Blackwell (GPU 0)** | Targeted on `dichtator` (GPU 1 is in a driver error state). |

---

## 8. Verification & Development Workflow

### 8.1 Verifying Type Stubs & Lints
```bash
# In WSL or lab environment:
cd ~/DrugEx

# 1. Regenerate any updated stubs deterministically:
python3 scripts/extract_git_stubs.py

# 2. Verify with Mypy with zero errors:
mypy --ignore-missing-imports drugex
```

### 8.2 Testing Suite
```bash
# Run core DrugEx unit tests:
pytest drugex/training/tests.py -v
pytest drugex/training/scorers/tests.py -v
```

### 8.3 Git Commit Rules
Upon completing changes, stage and commit locally:
```bash
git add drugex/*.pyi drugex/**.pyi drugex/AGENTS.md scripts/
git commit -m "feat(docs): add PEP 484 .pyi stub documentation layer and AGENTS.md architecture guide"
```
**CRITICAL**: NEVER execute `git push`.
