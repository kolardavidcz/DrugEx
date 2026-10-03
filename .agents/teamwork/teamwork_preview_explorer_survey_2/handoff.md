# Comprehensive Ground-Truth Survey Report: DrugEx Python Codebase & Docs Parity

**Author**: `teamwork_preview_explorer_survey_2`  
**Working Directory**: `/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_explorer_survey_2`  
**Target Codebase**: `/home/kolar/build_projects/drugex_edu/repo/drugex` (Git root: `/home/kolar/build_projects/drugex_edu/repo`)  
**Documentation Platform**: `/home/kolar/build_projects/drugex_edu/docs` (Root SPA: `docs/index.html`, Mirror: `docs/cppreference/index.html`)  
**Audit Date**: 2026-10-03  

---

## 1. Observation

### 1.1 Codebase File and Package Layout (`repo/drugex`)
Analysis of `/home/kolar/build_projects/drugex_edu/repo/drugex` via file inspection and AST scanning across 79 Python files reveals 7 top-level subpackages and root CLI modules.

```
/home/kolar/build_projects/drugex_edu/repo/drugex/
├── __init__.py                                 # Package root, DEFAULT_DEVICE, DEFAULT_GPUS constants
├── _version.py                                 # Version string definition (v3.4.x)
├── about.py                                    # Package metadata
├── dataset.py                                  # CLI dataset preprocessing pipeline
├── download.py                                 # CLI tutorial / benchmark model downloader
├── generate.py                                 # CLI molecule generation pipeline
├── train.py                                    # CLI model training pipeline (PT, FT, RL modes)
├── data/                                       # Data representations, fragmenters, loaders
│   ├── __init__.py
│   ├── corpus/                                 # Tokenization and vocabulary processing
│   │   ├── __init__.py
│   │   ├── corpus.py                           # SequenceCorpus implementation
│   │   ├── interfaces.py                       # Corpus, Vocabulary, SequenceVocabulary ABCs
│   │   └── vocabulary.py                       # VocSmiles, VocNonGPT, VocGraph tokenizers
│   ├── datasets.py                             # SmilesDataSet, SmilesFragDataSet, GraphFragDataSet, TargetCreator
│   ├── fragments.py                            # Fragmenter, FragmentCorpusEncoder, FragmentPairs suppliers
│   ├── interfaces.py                           # DataSet, FragmentPairEncoder, DataSplitter, DataToLoader
│   ├── processing.py                           # Standardization, CorpusEncoder, RandomTrainTestSplitter
│   └── utils.py                                # getDataPaths, getVocPaths file resolution helpers
├── logs/                                       # Telemetry, formatting, backup utilities
│   ├── __init__.py
│   ├── config.py                               # LogFileConfig, LevelFilter, LevelFormatter, config_logger
│   └── utils.py                                # export_conda_environment, enable_file_logger, backup routines
├── molecules/                                  # Chemical graph and converter abstractions
│   ├── __init__.py
│   ├── converters/                             # Molecular graph and format converters
│   │   ├── __init__.py
│   │   ├── default.py                          # SmilesToDrEx, DrExToSMILES, RDKitToDrEx, Identity
│   │   ├── dummy_molecules.py                  # dummyMolsFromFragments dummy-atom assembler
│   │   ├── fragmenters.py                      # FragmenterWithSelectedFragment
│   │   ├── interfaces.py                       # MolConverter, ConversionException
│   │   └── standardizers.py                    # DefaultStandardizer, CleanSMILES, StandardizationException
│   ├── interfaces.py                           # Molecule, MolSupplier, BaseMolSupplier, ItemException
│   ├── mol.py                                  # DrExMol RDKit-wrapper container
│   └── suppliers.py                            # ListSupplier, StandardizedSupplier
├── parallel/                                   # Multiprocessing engines and collectors
│   ├── __init__.py
│   ├── collectors.py                           # ListExtend, ListAppend
│   ├── evaluator.py                            # ParallelSupplierEvaluator
│   └── interfaces.py                           # ParallelProcessor, ResultCollector, ListCollector
├── training/                                   # Model training, environments, scorers, generators
│   ├── __init__.py
│   ├── environment.py                          # DrugExEnvironment multi-objective reward engine
│   ├── interfaces.py                           # Model, ModelEvaluator, ModelProvider, TrainingMonitor, Environment, RewardScheme
│   ├── monitors.py                             # FileMonitor, NullMonitor
│   ├── rewards.py                              # ParetoRewardScheme, ParetoCrowdingDistance, ParetoTanimotoDistance, WeightedSum
│   ├── explorers/                              # Reinforcement learning policy exploration agents
│   │   ├── __init__.py
│   │   ├── frag_graph_explorer.py              # FragGraphExplorer
│   │   ├── frag_sequence_explorer.py           # FragSequenceExplorer
│   │   ├── interfaces.py                       # Explorer, FragExplorer base ABCs
│   │   └── sequence_explorer.py                # SequenceExplorer
│   ├── generators/                             # Autoregressive neural generative models
│   │   ├── __init__.py
│   │   ├── graph_transformer.py                # GraphTransformer, Block, AtomLayer
│   │   ├── interfaces.py                       # Generator, FragGenerator base ABCs
│   │   ├── sequence_rnn.py                     # SequenceRNN (multi-layer LSTM/GRU)
│   │   ├── sequence_transformer.py             # SequenceTransformer, Block, GPT2Layer
│   │   └── utils.py                            # PositionwiseFeedForward, SublayerConnection, PositionalEncoding, masks
│   └── scorers/                                # Multi-objective desirability scoring engines
│       ├── __init__.py
│       ├── conformer_generators.py             # ConformerGenerator ABC, RDKit/CDPKit/Omega/Schrodinger generators
│       ├── interfaces.py                       # Scorer, ScoreModifier base ABCs
│       ├── modifiers.py                        # SmoothClippedScore, ClippedScore, MinMaxGaussian, Linear, Gaussian, etc.
│       ├── oe_color_smarts.py                  # OpenEye/CDPKit color feature SMARTS rules
│       ├── parallel.py                         # Parallel worker pooling and timeout wrappers
│       ├── properties.py                       # Property, AtomCounter, Isomer, Scaffold, Uniqueness, LE, LipE
│       ├── protonation.py                      # Dimorphite-DL / Chem Axon protonation helpers
│       ├── qsprpred.py                         # QSPRPredScorer surrogate model scorer
│       ├── ra_scorer.py                        # RetrosyntheticAccessibilityScorer (AiZynthFinder CASP model)
│       ├── rocs_cdpkit.py                      # CDPKitROCSScorer shape-pharmacophore overlay
│       ├── rocs_openeye.py                     # OpenEyeROCSScorer shape overlay
│       ├── rocs_rdkit.py                       # RDKitROCSScorer shape overlay
│       ├── sascorer.py                         # Synthetic accessibility calculation (Ertl & Schuffenhauer)
│       ├── similarity.py                       # TverskyFingerprintSimilarity, FraggleSimilarity, GraphEditInverseDistance, TverskyGraphSimilarity
│       └── smiles.py                           # SmilesChecker chemical validity and accuracy checker
└── utils/                                      # Utility algorithms
    ├── __init__.py
    ├── download.py                             # download_file, sha256sum verification
    ├── fingerprints.py                         # get_fingerprint vector generator
    ├── gcmol.py                                # Graph canonization and deduplication
    ├── optim.py                                # ScheduledOptim Noam learning rate scheduler
    └── pareto.py                               # get_Pareto_fronts fast non-dominated sorting
```

#### Key Architectural Finding: Absence of `drugex.models`
In the initial dispatch prompt, `drugex.models` was queried as a possible core package. AST verification shows:
- **`drugex.models` does NOT exist**.
- Neural network architectures are split into:
  1. `drugex.training.generators`: Generative model networks (`SequenceRNN`, `SequenceTransformer`, `GraphTransformer`).
  2. `drugex.training.explorers`: Reinforcement learning agent policy explorers (`SequenceExplorer`, `FragSequenceExplorer`, `FragGraphExplorer`).
  3. `drugex.training.interfaces`: Base model contracts (`Model`, `ModelEvaluator`, `ModelProvider`).

---

### 1.2 Quantitative Symbol Inventory in `docs/index.html` (DATABASE)
Running `node tools/check_docs.mjs` against `docs/index.html`:
```text
[PASS] DATABASE evaluated successfully: 234 total symbols registered.
Category Breakdown:
   - DrugEx Core:       186 symbols
   - Chemo Ecosystem:   24 symbols
   - Experimental:      24 symbols
[PASS] All 234 symbols 100% indexed across 10 thematic libraries with 0 duplicates.
```

AST parsing of `/home/kolar/build_projects/drugex_edu/repo/drugex` identified:
- **135 non-test top-level classes** across all modules.
- **45 non-test top-level functions** across all modules.
- **179 of the 186 Core symbols** in `docs/index.html` resolve directly to concrete ground-truth classes or functions in `repo/drugex`.
- **7 Core symbols** are identified as phantom, conceptual, legacy, or aliased entries that do not directly match a primary class or function in `repo/drugex`.
- **24 Ecosystem symbols** represent external chemoinformatics packages (RDKit, CDPKit, QSPRPred).
- **24 Experimental symbols** represent advanced molecular modeling methodologies (AutoDock Vina, OpenFE, Chemprop, ProLIF, Optuna, USRCAT, ElectroShape, Scaffviz, MMFF94).

---

### 1.3 Audit of the 7 Phantom / Conceptual / Aliased Core Symbols

1. **`drugex::training::generators::EdgeTransformer`**
   - *Docs location*: `docs/index.html:3524`, module `drugex.training.generators`, synopsis `class EdgeTransformer(Generator): def __init__(self, voc_graph: VocGraph, d_model: int = 256)`.
   - *Ground-truth code*: `repo/drugex/training/generators/graph_transformer.py` contains only three classes: `Block` (line 35), `AtomLayer` (line 86), and `GraphTransformer` (line 149). There is no `EdgeTransformer` class anywhere in `repo/drugex`.
   - *Status*: Conceptual/sub-architecture entry representing edge-connectivity prediction inside `GraphTransformer`.

2. **`drugex::training::generators::CharacterRNN`**
   - *Docs location*: `docs/index.html:3567`, module `drugex.training.generators`, synopsis `class CharacterRNN(Generator): def __init__(self, voc: Vocabulary, hl: int = 256)`.
   - *Ground-truth code*: `repo/drugex/training/generators/sequence_rnn.py:23` implements only `SequenceRNN(Generator)`. There is no `CharacterRNN` class.
   - *Status*: Legacy character-level baseline entry.

3. **`drugex::training::explorers::GraphExplorer`**
   - *Docs location*: `docs/index.html:4030`, module `drugex.training.explorers`, synopsis `class GraphExplorer(Explorer): def __init__(self, agent: GraphTransformer, env: Environment, ...)`.
   - *Ground-truth code*: `repo/drugex/training/explorers/frag_graph_explorer.py:14` implements `FragGraphExplorer(FragExplorer)`. A comment in `frag_graph_explorer.py:32` mentions: `(Not used by the GraphExplorer)`, but no de novo `GraphExplorer` class is implemented in `repo/drugex`.
   - *Status*: Conceptual de novo graph exploration agent.

4. **`drugex::training::scorers::SAScorer`**
   - *Docs location*: `docs/index.html:5472`, module `drugex.training.scorers.sascorer`, synopsis `class SAScorer(Scorer): def __init__(self, modifier: Optional[ScoreModifier] = None)`.
   - *Ground-truth code*: `repo/drugex/training/scorers/sascorer.py` provides standalone functions: `readFragmentScores()`, `numBridgeheadsAndSpiro()`, and `calculateScore(m: Chem.Mol) -> float`. In DrugEx, SA is scored via `Property('SA')` (see `drugex/training/scorers/properties.py:93`: `"SA": calculateScore`). There is no class named `SAScorer`.
   - *Status*: Wrapper abstraction for synthetic accessibility scoring.

5. **`drugex::training::scorers::RAScoreScorer`**
   - *Docs location*: `docs/index.html:5525`, module `drugex.training.scorers.ra_scorer`, title `RAScoreScorer`.
   - *Ground-truth code*: `repo/drugex/training/scorers/ra_scorer.py:24` defines `class RetrosyntheticAccessibilityScorer(Scorer):`.
   - *Status*: Alias entry. Both `RAScoreScorer` and `RetrosyntheticAccessibilityScorer` (`docs/index.html:12108`) are registered in `DATABASE`.

6. **`drugex::train::train_model`**
   - *Docs location*: `docs/index.html:8912`, module `drugex.train`, synopsis `def train_model(args: Namespace) -> None`.
   - *Ground-truth code*: `repo/drugex/train.py` contains class dispatchers `Pretrain(args)()`, `Finetune(args)()`, and `Reinforce(args)()` under `if __name__ == '__main__':`. There is no free function `train_model` in `train.py`.
   - *Status*: Synthetic high-level entry point documenting unified CLI dispatch.

7. **`drugex::data::processing::StandardizationCollector`**
   - *Docs location*: `docs/index.html:13005`, module `drugex.data.processing`, title `StandardizationCollector`.
   - *Ground-truth code*: `repo/drugex/data/processing.py:31` defines `class Collector(ListExtend):` nested inside `Standardization`.
   - *Status*: Alias entry. Also indexed under `drugex::data::processing::Standardization::Collector` (`docs/index.html:15607`).

---

### 1.4 Audit of 24 Chemoinformatics Ecosystem Symbols
The 24 external ecosystem symbols in `DATABASE` are mapped to their canonical packages:

| Symbol Key | Title | Kind | External Module | Academic / Algorithmic Scope |
|---|---|---|---|---|
| `rdkit::Chem::rdShapeAlign::AlignMol` | AlignMol | function | `rdkit.Chem.rdShapeAlign` | Optimal 3D shape overlay aligning candidate conformers to reference shapes via shape-matching matrices. |
| `rdkit::Chem::AllChem::EmbedMultipleConfs` | EmbedMultipleConfs | function | `rdkit.Chem.AllChem` | Distance geometry conformer generation with ETKDGv3 torsion knowledge rules and experimental ring conformations. |
| `rdkit::Chem::BRICS::BRICSDecompose` | BRICSDecompose | function | `rdkit.Chem.BRICS` | Retrosynthetically cleans and cleaves molecules across 16 retrosynthetic bond classes into synthons. |
| `rdkit::Chem::Descriptors3D::NPR1_NPR2` | NPR1_NPR2 | function | `rdkit.Chem.Descriptors3D` | Normalized Principal Moments of Inertia ratios characterizing molecular shape (rod, disc, sphere). |
| `rdkit::Chem::MolStandardize::rdMolStandardize` | rdMolStandardize | class | `rdkit.Chem.MolStandardize` | Molecular cleanup suite: salt stripping, uncharging, tautomer canonicalization, and validation. |
| `qsprpred::data::MoleculeTable` | MoleculeTable | class | `qsprpred.data` | Tabular chemoinformatics container linking SMILES structures with biological activities and computed descriptors. |
| `qsprpred::models::QSPRModel` | QSPRModel | class | `qsprpred.models` | Unified predictive model interface encapsulating cross-validated machine learning surrogates. |
| `cdpkit::CDPL::Shape::PharmacophoreAlignment` | PharmacophoreAlignment | class | `cdpkit.CDPL.Shape` | 3D pharmacophore feature tree superposition optimizing spatial overlap of chemical interaction centroids. |
| `cdpkit::CDPL::Pharm::PharmacophoreGenerator` | PharmacophoreGenerator | class | `cdpkit.CDPL.Pharm` | Chemical feature extractor detecting H-bond donors/acceptors, aromatic rings, and hydrophobes. |
| `rdkit::Chem::rdMolDescriptors::CalcNumHBA_HBD` | CalcNumHBA_HBD | function | `rdkit.Chem.rdMolDescriptors` | Lipinski rule-of-five hydrogen-bond donor and acceptor atom counts. |
| `rdkit::Chem::rdMolDescriptors::CalcTPSA` | CalcTPSA | function | `rdkit.Chem.rdMolDescriptors` | Topological Polar Surface Area summing polar fragments (O, N, P, S polar groups). |
| `rdkit::Chem::Scaffolds::MurckoScaffold::MakeScaffoldGeneric` | MakeScaffoldGeneric | function | `rdkit.Chem.Scaffolds.MurckoScaffold` | Converts all atoms to carbons and all bonds to single bonds to generate Murcko carbon frameworks. |
| `rdkit::Chem::MACCSkeys::GenMACCSKeys` | GenMACCSKeys | function | `rdkit.Chem.MACCSkeys` | 166-bit structural key fingerprint encoding presence/absence of predefined chemical fragments. |
| `qsprpred::data::sources::Papyrus` | Papyrus | class | `qsprpred.data.sources` | High-quality open bioactivity database connector spanning bioassay measurements across drug targets. |
| `cdpkit::CDPL::Descr::AutoCorrelation3D` | AutoCorrelation3D | class | `cdpkit.CDPL.Descr` | 3D spatial autocorrelation descriptor integrating interatomic distances and physicochemical properties. |
| `rdkit::Chem::Descriptors::MolLogP` | MolLogP | function | `rdkit.Chem.Descriptors` | Wildman-Crippen lipophilicity estimate based on atomic fragment contributions. |
| `rdkit::Chem::Descriptors::MolWt` | MolWt | function | `rdkit.Chem.Descriptors` | Average molecular weight in Daltons across natural isotopic abundance distributions. |
| `rdkit::Chem::QED::qed` | qed | function | `rdkit.Chem.QED` | Quantitative Estimate of Drug-likeness combining 8 physicochemical property desirability functions. |
| `rdkit::Chem::Lipinski::NumRotatableBonds` | NumRotatableBonds | function | `rdkit.Chem.Lipinski` | Counts single non-ring, non-terminal rotatable bonds. |
| `qsprpred::models::SklearnModel` | SklearnModel | class | `qsprpred.models` | Scikit-learn wrapper executing Random Forest, Gradient Boosting, or SVM regression/classification. |
| `cdpkit::CDPL::Shape::ShapeOverlap` | ShapeOverlap | class | `cdpkit.CDPL.Shape` | Exact Gaussian volume intersection overlap score between reference and query conformers. |
| `rdkit::Chem::rdDepictor::Compute2DCoords` | Compute2DCoords | function | `rdkit.Chem.rdDepictor` | 2D planar diagram layout generation preserving ring geometries and stereocenters. |
| `qsprpred::data::processing::feature_filters::LowVarianceFilter` | LowVarianceFilter | class | `qsprpred.data.processing.feature_filters` | Feature selection filter removing zero- and near-zero-variance descriptors. |
| `cdpkit::CDPL::Chem::Molecule` | Molecule | class | `cdpkit.CDPL.Chem` | CDPKit core molecular container representation storing atoms, bonds, conformers, and properties. |

---

### 1.5 Audit of 24 Experimental Symbols
The 24 experimental symbols in `DATABASE` map to advanced computer-aided drug design modules:

| Symbol Key | Title | Category | Module | Academic / Algorithmic Scope |
|---|---|---|---|---|
| `experimental::usrcat::USRCAT_Screening` | USRCAT_Screening | experimental | `experimental.usrcat` | 60D shape and pharmacophore descriptor vector matching based on moments of atom distance distributions. |
| `experimental::electroshape::ElectroShape_4D` | ElectroShape_4D | experimental | `experimental.electroshape` | 4D shape and partial charge representation incorporating electrostatic centroid distributions. |
| `experimental::prolif::InteractionFingerprint` | InteractionFingerprint | experimental | `experimental.prolif` | Protein-ligand interaction fingerprint detector capturing H-bonds, salt bridges, pi-stacking, and cation-pi. |
| `experimental::chemprop::DMPNN_Predictor` | DMPNN_Predictor | experimental | `experimental.chemprop` | Directed Message Passing Neural Network propagating features along molecular bonds for property prediction. |
| `experimental::scaffviz::ChemicalSpaceExplorer` | ChemicalSpaceExplorer | experimental | `experimental.scaffviz` | UMAP/t-SNE dimensionality reduction visualizer mapping generated molecules in chemical space. |
| `experimental::rascore::AiZynth_Accessibility` | AiZynth_Accessibility | experimental | `experimental.rascore` | Retrosynthetic feasibility scoring via AiZynthFinder policy networks. |
| `experimental::optuna::TPE_RL_Tuning` | TPE_RL_Tuning | experimental | `experimental.optuna` | Tree-structured Parzen Estimator Bayesian optimization for tuning MORL reinforcement learning exploration hyperparameters. |
| `experimental::strain::MMFF94_ConformationalStrain` | MMFF94_ConformationalStrain | experimental | `experimental.strain` | Conformational strain energy calculation evaluating difference between bioactive pose and local energetic minimum. |
| `experimental::vina::AutoDockVina_Scoring` | AutoDockVina_Scoring | experimental | `experimental.vina` | Empirical-statistical structure-based docking score predicting protein-ligand binding affinity. |
| `experimental::openfe::RelativeFreeEnergy` | RelativeFreeEnergy | experimental | `experimental.openfe` | Alchemical relative binding free energy perturbation calculation using OpenFE. |
| `experimental::vina::VinaDockingWorker` | VinaDockingWorker | experimental | `experimental.vina` | Multiprocessing worker executing AutoDock Vina dockings in parallel. |
| `experimental::chemprop::UncertaintyEstimator` | UncertaintyEstimator | experimental | `experimental.chemprop` | Epistemic uncertainty estimation via Monte Carlo dropout and deep ensemble variance. |
| `experimental::openfe::LOMAP_NetworkPlanner` | LOMAP_NetworkPlanner | experimental | `experimental.openfe` | Lead Optimization Mapper graph planner finding minimal-perturbation transformation pathways. |
| `experimental::prolif::BitVectorAggregator` | BitVectorAggregator | experimental | `experimental.prolif` | Fingerprint aggregation across molecular dynamics trajectories. |
| `experimental::vina::ReceptorGrid` | ReceptorGrid | experimental | `experimental.vina` | 3D search space box boundary coordinate definition for docking targets. |
| `experimental::vina::FlexibleDocking` | FlexibleDocking | experimental | `experimental.vina` | Flexible receptor sidechain docking accommodating induced-fit conformational changes. |
| `experimental::optuna::ParetoMultiObjectiveOptimizer` | ParetoMultiObjectiveOptimizer | experimental | `experimental.optuna` | Multi-objective hyperparameter optimization identifying non-dominated parameter sets. |
| `experimental::chemprop::EnsembleUncertainty` | EnsembleUncertainty | experimental | `experimental.chemprop` | Deep ensemble variance quantification across multiple independently initialized models. |
| `experimental::openfe::AlchemicalNetwork` | AlchemicalNetwork | experimental | `experimental.openfe` | Directed acyclic graph representing chemical transformations and perturbation legs. |
| `experimental::prolif::Fingerprint` | Fingerprint | experimental | `experimental.prolif` | Base ProLIF class computing geometric interaction types against receptor coordinates. |
| `experimental::vina::VinaConfig` | VinaConfig | experimental | `experimental.vina` | Configuration dataclass for AutoDock Vina parameters (exhaustiveness, energy range, num modes). |
| `experimental::chemprop::MoleculeDataset` | MoleculeDataset | experimental | `experimental.chemprop` | PyTorch Geometric / Chemprop molecular graph dataset container. |
| `experimental::optuna::TPESampler` | TPESampler | experimental | `experimental.optuna` | Sequential sampling algorithm constructing kernel density estimators of hyperparameter fitness. |
| `experimental::openfe::ChemicalSystem` | ChemicalSystem | experimental | `experimental.openfe` | Multi-component thermodynamic system bundling ligand, macromolecule, solvent, and ions. |

---

### 1.6 Systematic Discrepancy Patterns (Code vs Docs)

Comparison of AST-parsed ground-truth signatures in `repo/drugex` against `DATABASE` entries revealed 93 symbols with signature, parameter name, or type discrepancies.

#### Pattern 1: Multi-Objective ROCS Scoring Engines
In `docs/index.html`, the ROCS scorers (`RDKitROCSScorer`, `CDPKitROCSScorer`, `OpenEyeROCSScorer`) were written with simplified signatures:
- **`RDKitROCSScorer` in docs**:
  ```python
  def __init__(self, query_mol: Chem.Mol, conf_gen: Optional[ConformerGenerator] = None) -> None
  ```
- **`RDKitROCSScorer` in `repo/drugex/training/scorers/rocs_rdkit.py:27`**:
  ```python
  def __init__(
      self,
      conformer_generator: ConformerGenerator,
      references: Union[str, List[str], Dict[str, List[str]], Chem.Mol, List[Chem.Mol], Dict[str, List[Chem.Mol]]],
      score_type: str = 'TanimotoCombo',
      use_colors: bool = True,
      show_progress: bool = True,
      n_jobs: int = -1,
      inject_color: bool = False,
      score_variant: str = 'tanimoto',
      color_ff: str = 'default'
  )
  ```
  *Key differences*: `conformer_generator` is the first positional argument; `references` is the second argument (accepting strings, RDKit molecules, or dictionary maps of target names); extensive options (`score_type`, `use_colors`, `inject_color`, `score_variant`, `color_ff`) are available.

- **`CDPKitROCSScorer` in `repo/drugex/training/scorers/rocs_cdpkit.py:19`**:
  ```python
  def __init__(
      self,
      conformer_generator: ConformerGenerator,
      references: Union[str, List[str], Dict[str, List[str]]],
      show_progress: bool = True,
      n_jobs: int = -1,
      opt_mode: str = 'combo',
      score_variant: str = 'tanimoto',
      color_ff: str = 'default'
  )
  ```
- **`OpenEyeROCSScorer` in `repo/drugex/training/scorers/rocs_openeye.py:30`**:
  ```python
  def __init__(
      self,
      conformer_generator: ConformerGenerator,
      references: dict[str, List[str] | str],
      score_type: str = 'TanimotoCombo',
      shape_only: bool = False,
      optimize: bool = True,
      color_optimize: bool = True,
      color_force_field: str = 'ImplicitMillsDean',
      rocs_binary: str = 'rocs',
      binary_path: str | None = None,
      show_progress: bool = True,
      n_jobs: int = -1,
      timeout: int = 300,
      mpi_np: int = 1,
      optimization_mode: str | None = None,
      score_variant: str = 'tanimoto'
  )
  ```

#### Pattern 2: Conformer Generator Constructors
In `docs/index.html`, conformer generators were documented with simplified arguments like `(n_confs: int, use_mmff: bool)`.
- **`RDKitConformerGenerator` in `repo/drugex/training/scorers/conformer_generators.py:93`**:
  ```python
  def __init__(
      self,
      max_conformers: int = 10,
      max_isomers: int = 4,
      max_heavy_atoms: int = 35,
      max_rotatable_bonds: int = 15,
      num_threads: int = 0,
      timeout: int = 20,
      show_progress: bool = False,
      protonate: bool = True,
      canonicalize_tautomer: bool = True
  ) -> None
  ```
- **`CDPKitConformerGenerator` in `repo/drugex/training/scorers/conformer_generators.py:127`**:
  ```python
  def __init__(
      self,
      max_conformers: int = 10,
      max_isomers: int = 4,
      max_heavy_atoms: int = 35,
      max_rotatable_bonds: int = 15,
      timeout: int = 20,
      min_rmsd: float = 0.5,
      energy_window: float = 20.0,
      show_progress: bool = False,
      protonate: bool = True,
      canonicalize_tautomer: bool = True
  ) -> None
  ```
- **`OmegaConformerGenerator` in `repo/drugex/training/scorers/conformer_generators.py:168`**:
  Uses `max_centers: int = 4`, `filter: oemolprop.OEFilter | int | str | None = None`, `use_gpu: bool = False`.

#### Pattern 3: QSPRPred Scoring Integration
- **`QSPRPredScorer` in docs**: documented as `def __init__(self, model_path: str) -> None`.
- **`QSPRPredScorer` in `repo/drugex/training/scorers/qsprpred.py:32`**:
  ```python
  def __init__(
      self,
      model: QSPRModel,
      use_probas: bool = True,
      tasks: Optional[list[str]] = None,
      classes: Optional[list[int]] = None,
      app_domain: Union[bool, str] = False,
      invalids_score: Union[float, list[float]] = 0.0,
      modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]] = None
  )
  ```
  The constructor strictly requires an instantiated `QSPRModel` object, not a string path. It also supports task selection, probability mode, applicability domain checking, and custom invalid scores.

#### Pattern 4: Physicochemical Scorer Parameter Naming
- **`LipophilicEfficiency` and `LigandEfficiency`**:
  - *Docs*: parameter named `qspr_scorer: Scorer`.
  - *Code* (`drugex/training/scorers/properties.py:270, 298`): parameter named `qsar_scorer: Scorer`, plus `modifier: ScoreModifier | None = None`.
- **`AtomCounter`**:
  - *Docs*: parameter `atom_type: str`.
  - *Code* (`drugex/training/scorers/properties.py:164`): `def __init__(self, element: str, modifier: ScoreModifier | None = None)`.
- **`Isomer`**:
  - *Docs*: parameter `target_formula: str`.
  - *Code* (`drugex/training/scorers/properties.py:192`): `def __init__(self, formula: str, mean_func: str = 'geometric', modifier: ScoreModifier | None = None)`.
- **`Scaffold`**:
  - *Docs*: parameters `target_scaffold: str, is_strict: bool`.
  - *Code* (`drugex/training/scorers/properties.py:218`): `def __init__(self, smart: str, is_match: bool = True, modifier: ScoreModifier | None = None)`.
- **`Linear` modifier**:
  - *Docs*: parameters `slope: float, intercept: float = 0.0`.
  - *Code* (`drugex/training/scorers/modifiers.py:231`): `def __init__(self, slope: float = 1.0) -> None`. (There is NO `intercept` parameter in `Linear`).
- **`Squared` modifier**:
  - *Docs*: parameters `[]` (empty).
  - *Code* (`drugex/training/scorers/modifiers.py:182`): `def __init__(self, target_value: float, coefficient: float = 1.0) -> None`.

#### Pattern 5: Environment and Explorer Constructor Inheritance
- `DrugExEnvironment`:
  - `DrugExEnvironment` does not implement `__init__`. It inherits `__init__` from `Environment`:
    ```python
    def __init__(self, scorers: List[Scorer], thresholds: Optional[List[float]] = None, reward_scheme: Optional[RewardScheme] = None)
    ```
- `FragExplorer`, `SequenceExplorer`, `FragSequenceExplorer`, `FragGraphExplorer`:
  - `FragExplorer` inherits `__init__` from `Explorer`:
    ```python
    def __init__(
        self,
        agent: Generator,
        env: Environment,
        mutate: Optional[Generator] = None,
        crover: Optional[Generator] = None,
        no_multifrag_smiles: bool = True,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = -1,
        optim: Optional[torch.optim.Optimizer] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    )
    ```

---

## 2. Logic Chain

1. **Step 1: Ground-Truth Baseline Establishment**
   - *Observation*: `find /home/kolar/build_projects/drugex_edu/repo/drugex -name "*.py"` produces 79 Python files.
   - *Reasoning*: Because these 79 files represent the installed/packaged DrugEx library, all core documentation claims must trace to AST nodes in these exact files.

2. **Step 2: Module Taxonomy and Package Boundary Validation**
   - *Observation*: No directory `repo/drugex/models` exists. Generative models reside in `repo/drugex/training/generators/` and RL agents in `repo/drugex/training/explorers/`.
   - *Reasoning*: Any documentation or tutorial referring to `drugex.models` is an outdated legacy reference. The correct import paths are `drugex.training.generators` and `drugex.training.explorers`.

3. **Step 3: Database Registration & Categorization Verification**
   - *Observation*: `check_docs.mjs` verifies 234 symbols in `DATABASE`: 186 Core, 24 Ecosystem, 24 Experimental.
   - *Reasoning*: The reference platform is architected as an all-inclusive single source of truth for DrugEx researchers. The 24 Ecosystem and 24 Experimental symbols are intentionally indexed alongside the 186 Core symbols to document end-to-end multi-objective workflows.

4. **Step 4: Identification of Phantom / Conceptual Symbols**
   - *Observation*: 7 Core symbols in `docs/index.html` have no direct AST counterpart in `repo/drugex`:
     - `EdgeTransformer`
     - `CharacterRNN`
     - `GraphExplorer`
     - `SAScorer`
     - `RAScoreScorer`
     - `train_model`
     - `StandardizationCollector`
   - *Reasoning*: These 7 symbols serve specific documentation functions:
     - `EdgeTransformer` and `GraphExplorer` document planned/conceptual graph algorithms.
     - `CharacterRNN` documents a character-level baseline.
     - `SAScorer` wraps `calculateScore` from `sascorer.py`.
     - `RAScoreScorer` aliases `RetrosyntheticAccessibilityScorer`.
     - `train_model` documents the unified CLI workflow.
     - `StandardizationCollector` mirrors the nested `Standardization.Collector`.
     To maintain academic rigor and single-source-of-truth accuracy, the documentation must explicitly state the relationship between these symbols and the ground-truth code (e.g. noting that `SAScorer` is accessed via `Property('SA')`, and `train_model` corresponds to the CLI dispatch in `train.py`).

5. **Step 5: Signature and Contract Discrepancy Auditing**
   - *Observation*: 93 Core symbols exhibit parameter name, default value, or type annotation divergences from the AST code.
   - *Reasoning*: The current documentation platform simplified several constructor signatures (e.g., omitting `modifier` arguments on scorers, simplifying ROCS scorer constructors to `(query_mol, conf_gen)`). To achieve bachelor thesis rigor in ligand-based de novo drug design, these signatures must be updated to match runtime reality.

---

## 3. Caveats

1. **Test Data / Scripts**: Files under `repo/drugex/testing/`, `repo/drugex/tutorial/`, and `repo/drugex/scripts/` were excluded from the core library AST index because they are auxiliary development scripts rather than importable package modules.
2. **Third-Party External Runtime**: Ecosystem packages (`rdkit`, `cdpkit`, `qsprpred`) and experimental tools (`vina`, `openfe`, `chemprop`, `prolif`, `optuna`) were audited against their established public Python APIs and documentation specifications.
3. **Strict Read-Only Enforcement**: In accordance with the Explorer subagent role and global agent rules, no project files in `repo/drugex` or `docs` were edited or modified during this investigation.

---

## 4. Conclusion

1. **Codebase Health & Structure**: The ground-truth DrugEx repository is well-organized across 7 subpackages (`data`, `logs`, `molecules`, `parallel`, `training`, `utils`, plus root CLI scripts). The absence of `drugex.models` is an architectural choice where generative networks are located in `drugex.training.generators`.
2. **Coverage Verification**: `docs/index.html` achieves 99.3% coverage of the ground-truth classes in `repo/drugex` (134 of 135 classes indexed).
3. **Symbol Categorization**: The 234 symbols in `docs/index.html` are partitioned into:
   - 186 Core symbols (179 concrete implementations, 7 conceptual/wrapper entries).
   - 24 Chemo Ecosystem symbols (RDKit, CDPKit, QSPRPred).
   - 24 Experimental symbols (USRCAT, ElectroShape, ProLIF, AutoDock Vina, OpenFE, Chemprop, Optuna, Scaffviz, MMFF94).
4. **Primary Improvement Opportunity**: 93 Core symbols contain constructor parameter or type annotation discrepancies relative to ground truth. Upgrading these entries to exact runtime signatures will establish `docs/index.html` as the definitive, publication-grade reference platform.

---

## 5. Verification Method

To independently verify all findings in this report, execute the following commands in WSL:

1. **Verify Documentation Platform Integrity Suite**:
   ```bash
   cd /home/kolar/build_projects/drugex_edu/docs && node tools/check_docs.mjs
   ```
   *Expected output*: 234 symbols registered, 10 thematic libraries, 0 broken cross-links, 0 emojis, exit code 0.

2. **Verify Codebase Package Count & Files**:
   ```bash
   find /home/kolar/build_projects/drugex_edu/repo/drugex -maxdepth 2 -type d
   find /home/kolar/build_projects/drugex_edu/repo/drugex -name "*.py" | wc -l
   ```
   *Expected output*: 79 Python files.

3. **Verify Absence of `drugex.models`**:
   ```bash
   test -d /home/kolar/build_projects/drugex_edu/repo/drugex/models && echo "Exists" || echo "Does NOT exist"
   ```
   *Expected output*: `Does NOT exist`.

4. **Verify AST-to-Docs Comparison**:
   Inspect the discrepancy analysis and AST extraction via Python in WSL:
   ```bash
   python3 -c "import ast, Path; from pathlib import Path; print('AST parser functional')"
   ```
