# Cheminformatics & Algorithmic Foundations Survey Report

**Author**: Teamwork Preview Explorer (Survey Agent 3 — Cheminformatics Algorithmic Analyst)  
**Date**: 2026-10-03  
**Target Milestone**: DrugEx 3.4+ Documentation Platform (`docs/index.html` DATABASE & `docs/pipelines.html`)  
**Scope**: Publication-Grade Algorithmic, Mathematical, and Cheminformatics Rigor for Ligand-Based De Novo Drug Design  
**Working Directory**: `/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_explorer_survey_3`  
**Ground-Truth Source Code**: `/home/kolar/build_projects/drugex_edu/repo/drugex`

---

## 1. Observation

### 1.1 Multi-Objective Reinforcement Learning (MORL) Core Implementations

#### 1.1.1 Non-Dominated Sorting & Pareto Dominance
In `repo/drugex/utils/pareto.py`, lines 4–44, the Pareto frontier identification is implemented as an iterative peeling non-dominated sorting algorithm:
```python
def get_Pareto_fronts(scores):
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    all_fronts = []
    while population_size > 0:
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_front[i] = 0
                    break
        current_front_ids = population_ids[pareto_front]
        all_fronts.append(current_front_ids)
        scores = scores[~pareto_front]
        population_ids = population_ids[~pareto_front]
        population_size = scores.shape[0]
    return all_fronts
```
- **Mathematical Definition of Pareto Dominance**: Candidate $j$ dominates candidate $i$ ($j \succ i$) in an $M$-objective maximization space iff:
  $$\forall m \in \{1, \dots, M\}: s_{j, m} \ge s_{i, m} \quad \land \quad \exists m \in \{1, \dots, M\}: s_{j, m} > s_{i, m}$$
- **Front Hierarchy**: The first extracted tier $F_0$ is the empirical non-dominated Pareto frontier of the candidate batch. Subsequent tiers $F_1, F_2, \dots, F_{K-1}$ are formed by recursively removing dominated solutions until the population is exhausted.

#### 1.1.2 NSGA-II Crowding Distance Calculation
In `repo/drugex/training/rewards.py`, lines 81–146 (`ParetoCrowdingDistance`):
- For each Pareto front $F_k \subseteq \{0, \dots, N-1\}$ with $|F_k| = n_k$:
  - Solutions are initialized with crowding distance $d_j = 0.0$ for all $j \in F_k$.
  - For each objective dimension $m \in \{0, \dots, M-1\}$:
    1. Sort front solutions in ascending order of score along dimension $m$: $\text{order} = \text{argsort}(s_{:, m})$.
    2. Assign boundary solutions infinite distance: $d_{\text{order}[0]} \leftarrow \infty$ and $d_{\text{order}[n_k - 1]} \leftarrow \infty$.
    3. If the objective range $\Delta s_m = s_{\max, m} - s_{\min, m} == 0$, skip dimension $m$.
    4. For all interior solutions $j \in \{1, \dots, n_k - 2\}$:
       $$d_{\text{order}[j]} \leftarrow d_{\text{order}[j]} + \frac{s_{\text{order}[j+1], m} - s_{\text{order}[j-1], m}}{\Delta s_m}$$
  - Front solutions are sorted in descending order of crowding distance: `sorted_indices = np.argsort(-crowding_distance)`.
  - Ranked fronts are combined and reversed from worst to best: `rank = np.concatenate(ranked_fronts)[::-1]`.

#### 1.1.3 Normalized Rank Scalarization
In `ParetoRewardScheme.__call__` (`repo/drugex/training/rewards.py`, lines 53–79):
```python
fronts = self.getParetoFronts(scores)
ranks = self.getMoleculeRank(fronts, smiles=smiles, scores=scores)
rewards = np.zeros((len(smiles), 1))
rewards[ranks, 0] = np.arange(len(scores)) / len(scores)
return rewards
```
- Because `ranks` lists indices from worst (rank index 0) to best (rank index $N-1$), the reward $R_i \in [0, 1)$ assigned to the molecule at rank index $r$ is:
  $$R(S_{\text{ranks}[r]}) = \frac{r}{N}$$
- This normalizes rewards strictly to $[0.0, \frac{N-1}{N}]$, making the policy gradient scale-invariant and immune to numerical scale differences between disparate objective functions (e.g., nM affinity vs. molecular weight in Daltons).

#### 1.1.4 Fingerprint Tanimoto Distance Ranking
In `repo/drugex/training/rewards.py`, lines 147–279 (`ParetoTanimotoDistance`):
- Measures molecular diversity in chemical fingerprint space (ECFP6) instead of objective score space.
- Supports three distance aggregation strategies for molecules in the same front:
  1. `min`: Minimizes distance to the nearest neighbor within the front.
  2. `mean`: Computes average Tanimoto distance to all other members in the front: $1 - \frac{1}{|F_k|-1} \sum_{l \neq j} T(\text{fp}_j, \text{fp}_l)$.
  3. `mutual`: Emulates crowding distance directly in fingerprint space using cumulative sorted distance increments:
     $$\text{dist}[\text{order}[k]] \leftarrow \text{dist}[\text{order}[k]] + (T_{\text{dist}}[\text{order}[k+1]] - T_{\text{dist}}[\text{order}[k-1]])$$

#### 1.1.5 Dynamic Adaptive Weighted Sum Scalarization
In `repo/drugex/training/rewards.py`, lines 280–307 (`WeightedSum`):
```python
weight = ((scores < thresholds).mean(axis=0, keepdims=True) + 0.01) / \
         ((scores >= thresholds).mean(axis=0, keepdims=True) + 0.01)
weight = weight / weight.sum()
return scores.dot(weight.T)
```
- Objective weight $w_m$ dynamically scales with the proportion of compounds failing desirability threshold $\theta_m$:
  $$w_m \propto \frac{\mathbb{P}(s_m < \theta_m) + 0.01}{\mathbb{P}(s_m \ge \theta_m) + 0.01}$$
  If an objective has few candidates achieving the threshold (a "hard" target), its failure rate is high, automatically increasing $w_m$ to focus exploration on that bottleneck.

#### 1.1.6 Environment Scoring & Candidate Qualification
In `repo/drugex/training/environment.py`, lines 8–51 (`DrugExEnvironment`) and `interfaces.py`, lines 183–200:
- `SmilesChecker.checkSmiles` evaluates:
  1. RDKit parseability: `Chem.MolFromSmiles(s)` $\to$ `Valid = 1` or `0`.
  2. Disconnected component filtering: if `no_multifrag_smiles=True`, `s.count('.') == 0` is enforced.
  3. Fragment substructure preservation: if `frags` supplied, `mol.HasSubstructMatch(Chem.MolFromSmiles(sub))` for all sub-components $\to$ `Accurate = 1` or `0`.
- All scorer values are zeroed for invalid candidates: `scores.loc[scores['Valid'] == 0, keys] = 0.0`.
- Desirability flag: `scores['Desired'] = ( (scores[keys] < thresholds).sum(axis=1) == 0 ).astype(int)`.
- In `Environment.getRewards`: Only valid compounds are passed to `rewardScheme`; invalid molecules maintain reward $0.0$. If zero valid molecules exist in a batch, rewards are a zero vector: `rewards = np.zeros((len(smiles), 1))`.

#### 1.1.7 Policy Gradient Formulation & Optimization
In `SequenceExplorer.policy_gradient` (`sequence_explorer.py`, lines 112–133) and `FragExplorer.policy_gradient` (`interfaces.py`, lines 244–270):
```python
loss = self.agent.likelihood(seq) # shape (batch_size,)
loss = loss * (reward - self.beta)
loss = -loss.mean()
loss.backward()
self.optim.step()
```
- Expected reward objective: $J(\theta) = \mathbb{E}_{S \sim P_\theta}[R(S)]$.
- REINFORCE policy gradient with baseline $\beta$:
  $$\nabla_\theta J(\theta) = \mathbb{E}_{S \sim P_\theta} \left[ \nabla_\theta \log P_\theta(S) \cdot (R(S) - \beta) \right]$$
- PyTorch surrogate loss minimized by gradient descent:
  $$\mathcal{L}(\theta) = - \frac{1}{B} \sum_{i=1}^B \log P_\theta(S_i) \cdot (R(S_i) - \beta)$$
- Baseline parameter `self.beta` (default $0.0$) provides variance reduction without introducing policy gradient bias.

#### 1.1.8 Genetic-RL Exploration: Crossover & Mutation Operators
In `SequenceRNN.evolve` (`repo/drugex/training/generators/sequence_rnn.py`, lines 236–292):
- At each token step $t$, the next-token probability distribution combines:
  1. Base agent policy: $P_A(x_t \mid x_{<t}) = \text{softmax}(\text{logit}_A)$.
  2. Crossover network (`crover`): with random ratio $\alpha \sim \mathcal{U}(0, 1)$ per sequence:
     $$P(x_t) = \alpha \cdot P_A(x_t) + (1 - \alpha) \cdot P_C(x_t)$$
  3. Mutation network (`mutate`, frozen pre-trained model): with probability $\epsilon$ (`is_mutate = rand < epsilon`):
     $$P(x_t) \leftarrow P_M(x_t)$$
- DrugEx uses REINFORCE with baseline and this genetic-RL token interpolation mechanism rather than PPO or entropy regularization.

---

### 1.2 Generative Deep Learning Architectures

| Architecture | Source Location | Model Class | Key Structural Specifications |
|---|---|---|---|
| **Sequence RNN** | `training/generators/sequence_rnn.py` | `SequenceRNN` | 3-layer LSTM or GRU (`num_layers=3`), `embed_size=128`, `hidden_size=512`. Token embedding $\to$ recurrent propagation $\to$ linear projection $\to$ logits `(B, voc.size)`. Autoregressive sampling via `torch.multinomial`. |
| **Sequence Transformer** | `training/generators/sequence_transformer.py` | `SequenceTransformer` | 12-layer GPT-2 causal decoder (`d_emb=512`, `d_model=512`, `n_head=8`, `d_inner=1024`, `n_layer=12`). Sinusoidal positional embeddings + causal triangular mask (`tri_mask`) + padding key mask (`pad_mask`). Conditions on concatenated input `[src, trg]`. |
| **Graph Transformer** | `training/generators/graph_transformer.py` | `GraphTransformer` | 12-layer AtomLayer self-attention (`d_model=512`, `n_head=8`, `d_inner=1024`, `n_layer=12`) coupled with `nn.GRUCell(512, 512)` recurrent transition cells. Dual embeddings (`emb_word`, `emb_site`). Predicts 4-tuple graph actions: `(atom, curr_locus, prev_locus, bond)`. Dynamic valence tracking masks. |
| **Fragment Decomposition** | `molecules/converters/fragmenters.py` | `Fragmenter` | RECAP (`RecapDecompose`) or BRICS (`BRICSDecompose`) fragmentation. Filters leaf fragments, selects top `n_frags` by heavy atom count, enumerates combinations up to `n_combs`, validates substructure match with `HasSubstructMatch`. |

#### Graph Transformer Valence Tracking & Action Masks
In `GraphTransformer.forward` (`graph_transformer.py`, lines 324–370):
1. Intrinsic valence vector `voc_mask` (from `VocGraph.masks`) defines the maximum allowed valence per atom type.
2. At each step, remaining valence is tracked as:
   $$\text{vals\_rom} = \text{vals\_max} - \sum_{k} \text{exists}[:, :, k]$$
   where `exists` is the $(B, n_{\text{grows}}, n_{\text{grows}})$ adjacency tensor of existing bond orders.
3. Forbidden transitions are masked to probability 0 before multinomial sampling:
   - Atoms with zero remaining valence cannot form further bonds (`vals_rom[curr] == 0`).
   - Bonds with order $b > \text{vals\_rom}$ are masked out.
   - Ring closures to atoms with insufficient remaining valences or already connected are masked out.

---

### 1.3 Data Representation, Tensors & Coordinate Frames

#### 1.3.1 Exact Tensor Dimensions

| Model / Pipeline Stage | Tensor Name | Shape | Data Type | Semantic Description |
|---|---|---|---|---|
| **SequenceRNN** | `input` | `(B,)` or `(B, 1)` | `torch.long` | Input token indices |
| **SequenceRNN** | `h` (GRU) | `(3, B, 512)` | `torch.float32` | Hidden state tensor |
| **SequenceRNN** | `(h, c)` (LSTM) | `2 x (3, B, 512)` | `torch.float32` | Hidden and cell state tensors |
| **SequenceRNN** | `target` | `(B, T)` | `torch.long` | Target token sequences (max length $T$) |
| **SequenceRNN** | `scores` | `(B,)` | `torch.float32` | Summed sequence log-likelihood $\sum_t \log P(x_t \mid x_{<t})$ |
| **SequenceTransformer** | `src` | `(B, \text{src\_len})` | `torch.long` | Encoded scaffold fragment tokens |
| **SequenceTransformer** | `trg` | `(B, \text{trg\_len})` | `torch.long` | Encoded candidate molecule tokens |
| **SequenceTransformer** | `input_seq` | `(B, \text{src\_len} + \text{trg\_len})` | `torch.long` | Concatenated context-target sequence |
| **SequenceTransformer** | `key_mask` | `(B, \text{seq\_len})` | `torch.bool` | Padding key mask (True for padding index 0) |
| **SequenceTransformer** | `atn_mask` | `(\text{seq\_len}, \text{seq\_len})` | `torch.bool` | Causal triangular attention mask |
| **SequenceTransformer** | `out` (training) | `(B, \text{trg\_len})` | `torch.float32` | Gathered target log-probabilities |
| **GraphTransformer** | `src` | `(B, \text{seq\_len}, 5)` | `torch.long` | 5 action channels: `[atom, curr_locus, prev_locus, bond, grow_flag]` |
| **GraphTransformer** | `emb_word` | `(B, \text{seq\_len}, 512)` | `torch.float32` | Combined atom-bond embedding: $\text{atom} \times 4 + \text{bond}$ |
| **GraphTransformer** | `emb_site` | `(B, \text{seq\_len}, 512)` | `torch.float32` | Connection locus pair embedding: $\text{curr} \times n_{\text{grows}} + \text{prev}$ |
| **GraphTransformer** | `exists` | `(B, n_{\text{grows}}, n_{\text{grows}})` | `torch.long` | Current molecular adjacency matrix |
| **GraphTransformer** | `vals_max` | `(B, n_{\text{grows}})` | `torch.long` | Intrinsic maximum valence of active atoms |
| **GraphTransformer** | `vals_rom` | `(B, n_{\text{grows}})` | `torch.long` | Remaining free valence per active atom |
| **Environment** | `scores` | `(N, M)` | `np.float64` | Matrix of $M$ objective scores for $N$ molecules |
| **Environment** | `rewards` | `(N, 1)` | `np.float64` | Scalarized policy gradient rewards in $[0.0, 1.0]$ |

#### 1.3.2 SMILES & Graph Vocabularies
- **`VocSmiles`** (`data/corpus/vocabulary.py:29–160`):
  - Token splitting handles brackets, rings, two-character elements. Multi-character halogens are mapped to unique single characters: `'Cl'` $\to$ `'L'`, `'Br'` $\to$ `'R'`.
  - Special tokens: `'EOS'` = 0 (end of sequence), `'GO'` = 1 (start of generation), padding = 0, fragment delimiter = `'_'` or `'.'`.
- **`VocGraph`** (`data/corpus/vocabulary.py:388–460`):
  - Words represent atom valence states (e.g. `'4C'`, `'2O'`, `'3N'`, `'3O+'`, `'1Cl'`, `'*'`).
  - Metadata parsed via `parseWord`: `(element_with_charge, valence, charge_number, atomic_number, word)`.
  - Intrinsic valence vector `self.masks` stored for valence-rule enforcement.

#### 1.3.3 3D Conformers & Coordinate Frame Invariance
In `rocs_rdkit.py`, lines 164–195:
- **Pose Dependence Issue (RDKit #8513 / PR #8999)**: `rdShapeAlign.AlignMol` optimizes alignment starting from the probe's input coordinate orientation. Different arbitrary rotations or translations of the same 3D conformer can converge to different local optima (yielding up to 0.13 score spread).
- **Principal Inertial Axis Alignment**:
  1. `rdMolTransforms.CanonicalizeConformer(conf)` translates the center of mass to $(0, 0, 0)$ and aligns axes to the principal moments of inertia (eigenvectors of the moment of inertia tensor).
  2. Resolves sign ambiguity of principal axes by testing the 4 proper rotations ($\det = +1$) with 180° flips:
     $$\mathcal{R} \in \left\{ \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \begin{pmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix}, \begin{pmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{pmatrix}, \begin{pmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \right\}$$
  3. Evaluates overlays across all 4 sign-starts and retains the maximum score. This drives cross-pose score variance to exactly 0.000.

#### 1.3.4 ROCS Analytic Gaussian Overlap & Tversky Index Recovery
In `rocs_rdkit.py`, lines 134–156:
- Shape similarity is calculated via first-order Gaussian volume overlap:
  $$O_{ab} = \sum_{i \in A} \sum_{j \in B} p_{ij} \exp(-\alpha_{ij} r_{ij}^2)$$
  where $r_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$ is inter-atomic distance and $\alpha_{ij}$ is Gaussian width.
- `rdShapeAlign` natively reports Tanimoto $T = \frac{O_{ab}}{S_a + S_b - O_{ab}}$ and self-overlaps $S_a = O_{aa}, S_b = O_{bb}$.
- Cross-overlap $O_{ab}$ is analytically recovered as:
  $$O_{ab} = \frac{T \cdot (S_a + S_b)}{1 + T}$$
- The reference-weighted Tversky index ($\alpha=0.95, \beta=0.05$) is then calculated as:
  $$T_{\text{versky}} = \frac{O_{ab}}{O_{ab} + \alpha(S_a - O_{ab}) + \beta(S_b - O_{ab})}$$

#### 1.3.5 Pharmacophore Feature Schemes
- **6-type Scheme**: Donor, Acceptor, Cation (`PosIonizable`), Anion (`NegIonizable`), Rings (`Aromatic`), Hydrophobe.
- Injecting color via `PUBCHEM_PHARMACOPHORE_FEATURES` property into `rdShapeAlign`:
  - `default` / `fdef`: RDKit `BaseFeatures.fdef`.
  - `oe_mills_dean`: OpenEye `ImplicitMillsDean` SMARTS (`oe_color_smarts.py`), typing saturated rings as `rings` and alkyl chains as `hydrophobe`.

#### 1.3.6 Boundary Conditions & Failure Modes
1. **Unparseable SMILES**: `Chem.MolFromSmiles` returns `None`. `SmilesChecker` marks `Valid=0`. `DrugExEnvironment` zeros all scorer outputs (`0.0`). `Environment.getRewards` sets scalar reward to `0.0`.
2. **Multifragment Compounds**: If `no_multifrag_smiles=True`, SMILES with `.` are considered invalid and receive reward `0.0`.
3. **Fragment Accuracy Failure**: If generated molecule does not preserve all input scaffold fragments (`HasSubstructMatch` is False), flagged with `Accurate=0`.
4. **Empty Batch / Zero Valid Molecules**: `Environment.getRewards` catches `len(valid_idx) == 0`, logs `"No valid molecules generated. All rewards are 0."`, and safely returns an all-zero reward vector without raising an unhandled exception.
5. **Conformer Generation Timeout**: Pathological torsion-driving or stereoisomer explosions are bounded by strict 20s timeouts (`timeout=20`), preventing deadlocks in RL training loops.
6. **Out-of-Domain Compounds**: For `QSPRPredScorer`, setting `app_domain='invalid'` assigns fallback penalty `invalids_score` (default `0.0`) when molecular descriptors fall outside the training domain.

---

## 2. Logic Chain

1. **Premise 1 (R1 & R2 Requirements)**: Academic rigor for a bachelor thesis in ligand-based de novo drug design requires mathematical precision: exact definitions of dominance relations, crowding distance algorithms, policy gradient update derivations, tensor dimensions, and coordinate frames.
2. **Premise 2 (Empirical Code Verification)**: Direct inspection of `repo/drugex/` reveals that DrugEx 3.4 implements NSGA-II non-dominated sorting (`pareto.py`), objective cuboid crowding distance and ECFP6 Tanimoto distance ranking (`rewards.py`), REINFORCE policy gradients with baseline $\beta$ (`sequence_explorer.py`, `interfaces.py`), genetic crossover and mutation operators (`sequence_rnn.py`), GPT-2 fragment-conditioned autoregressive sequence generation (`sequence_transformer.py`), and 4-tuple graph action generation with dynamic valence checking (`graph_transformer.py`).
3. **Premise 3 (Coordinate & 3D Rigor)**: Inspection of `conformer_generators.py` and `rocs_rdkit.py` reveals that 3D shape matching requires pose invariance via Principal Inertial Axis alignment and 4 proper rotation sign-flips, native Gaussian self-overlap recovery for Tversky calculations, and 6-feature pharmacophore color matching.
4. **Premise 4 (Documentation Platform Gap)**: Many entries in documentation references describe concepts generically (e.g., "calculates rewards using machine learning" or omitting tensor shapes). Upgrading all 234 symbols in `docs/index.html` to reflect the exact verified signatures, mathematical formulas, tensor shapes `(B, T)`, `(B, seq_len, 5)`, and edge-case contracts will establish `docs/` as the definitive, publication-grade reference platform.
5. **Deduction**: The documentation platform can achieve 100% mathematical and technical accuracy by directly mapping each DATABASE symbol to its corresponding verified source implementation, mathematical formulation, and executable workflow.

---

## 3. Caveats

1. **PPO / Entropy Regularization**: While PPO and explicit entropy regularization terms are common in general reinforcement learning, DrugEx's codebase relies on the REINFORCE algorithm with baseline subtraction and maintains exploration diversity through an $\epsilon$-greedy genetic mutation network (`mutate`) and crossover network (`crover`), rather than PPO clipping or an entropy loss bonus. Documentation should accurately describe this hybrid genetic-RL mechanism rather than claiming PPO is implemented.
2. **Multi-GPU Parallelism**: `SequenceRNN` explicitly binds only to a single primary GPU (`self.gpus = (gpus[0],)`), whereas `FragSequenceExplorer` and `GraphTransformer` wrap models in `torch.nn.DataParallel` across multiple GPUs.
3. **Optional Ecosystem Dependencies**: Packages such as OpenEye Toolkits (`openeye`), CDPKit (`CDPL`), RAscore (`RAscore_XGB`), and Dimorphite-DL are optional runtime dependencies guarded by `try...except ImportError`. Fallbacks and prerequisites must be documented for each ecosystem symbol.

---

## 4. Conclusion

The DrugEx 3.4 codebase embodies a rigorous, mathematically sound, and cheminformatically grounded framework for ligand-based de novo drug design:
1. **MORL**: Combines NSGA-II non-dominated sorting, cuboid crowding distance, normalized rank scalarization, dynamic adaptive weighted sums, and REINFORCE with baseline.
2. **Generative Modeling**: Spans 3-layer LSTM/GRU SMILES autoregression, 12-layer GPT-2 fragment-conditioned sequence generation, and 12-layer AtomLayer graph generation with step-wise GRU transitions and dynamic chemical valence masks.
3. **Cheminformatics & 3D Coordinates**: Integrates ETKDGv3 conformers, Principal Inertial Axis pose canonicalization with 4 proper rotation sign-flips, analytic Gaussian volume overlaps, native Tversky recovery, and 6-type pharmacophore feature alignment.
4. **Actionable Standard for `docs/index.html`**: Every symbol in DATABASE should document:
   - Exact Python signature and type annotations matching runtime reality.
   - Mathematical formula and algorithmic mechanism (equations, normalization bounds).
   - Exact tensor and array dimensions (e.g., `(B, seq_len, 5)`, `(N, M)`, `(N, 1)`).
   - Preconditions, valid numerical ranges, and edge-case failure modes (`Valid=0`, `0.0` fallbacks, timeout caps).
   - Realistic, executable DrugEx 3.4+ code examples with authentic outputs.

---

## 5. Verification Method

To independently verify the algorithmic findings and test suite compliance:

1. **Automated Documentation Audit**:
   ```bash
   wsl bash -c "cd /home/kolar/build_projects/drugex_edu/docs && node tools/check_docs.mjs"
   ```
   *Expected Result*: Passes all assertions (234 symbols, 10 library sections, 0 broken links, 0 emojis).

2. **Source Code Inspection Points**:
   - NSGA-II sorting: `repo/drugex/utils/pareto.py:4–44`
   - Crowding distance & rank scalarization: `repo/drugex/training/rewards.py:81–146`
   - Dynamic weighted sum: `repo/drugex/training/rewards.py:280–307`
   - Environment qualification: `repo/drugex/training/environment.py:8–51`
   - REINFORCE policy gradient: `repo/drugex/training/explorers/sequence_explorer.py:121–127`
   - Genetic-RL evolve routine: `repo/drugex/training/generators/sequence_rnn.py:236–292`
   - Sequence Transformer GPT-2 decoder: `repo/drugex/training/generators/sequence_transformer.py:248–299`
   - Graph Transformer action decoding & valence masks: `repo/drugex/training/generators/graph_transformer.py:293–370`
   - Conformer generation caps & timeouts: `repo/drugex/training/scorers/conformer_generators.py:413–520`
   - ROCS pose invariance & Tversky recovery: `repo/drugex/training/scorers/rocs_rdkit.py:134–195`

3. **Invalidation Conditions**:
   - If `ParetoCrowdingDistance` is found to use raw unnormalized scores rather than normalized rank scalarization.
   - If `GraphTransformer` generates bonds without checking remaining valence `vals_rom`.
   - If `rocs_rdkit.py` fails pose invariance across rigid translations/rotations.
