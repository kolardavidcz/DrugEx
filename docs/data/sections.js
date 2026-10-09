/**
 * DrugEx Documentation Platform - Library Sections Hierarchy
 * 10 Thematic cppreference-style library sections indexing all 234 symbols.
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.LIBRARY_SECTIONS = [
      {
        id: "morl",
        title: "Core Reinforcement Learning (MORL)",
        subsections: [
          {
            subtitle: "Environments & Markov Decision Process",
            keys: [
              "drugex::training::environment::DrugExEnvironment",
              "drugex::training::environment::Environment"
            ]
          },
          {
            subtitle: "Multi-Objective Rewards & Pareto Ranking",
            keys: [
              "drugex::training::rewards::RewardScheme",
              "drugex::training::rewards::ParetoRewardScheme",
              "drugex::training::rewards::ParetoCrowdingDistance",
              "drugex::training::rewards::ParetoTanimotoDistance",
              "drugex::training::rewards::WeightedSum",
              "drugex::utils::pareto::get_Pareto_fronts"
            ]
          },
          {
            subtitle: "Training Monitors & Convergence",
            keys: [
              "drugex::training::interfaces::TrainingMonitor",
              "drugex::training::monitors::FileMonitor",
              "drugex::training::monitors::NullMonitor"
            ]
          },
          {
            subtitle: "Model Providers & Learning Schedules",
            keys: [
              "drugex::training::interfaces::Model",
              "drugex::training::interfaces::ModelEvaluator",
              "drugex::training::interfaces::ModelProvider",
              "drugex::utils::optim::ScheduledOptim"
            ]
          }
        ]
      },
      {
        id: "neural",
        title: "Generative Neural Architectures",
        subsections: [
          {
            subtitle: "Sequence-Based Generators (SMILES)",
            keys: [
              "drugex::training::generators::SequenceRNN",
              "drugex::training::generators::SequenceTransformer",
              "drugex::training::generators::CharacterRNN"
            ]
          },
          {
            subtitle: "Graph-Based Generators (Molecular Graphs)",
            keys: [
              "drugex::training::generators::GraphTransformer",
              "drugex::training::generators::EdgeTransformer"
            ]
          },
          {
            subtitle: "Generator Base Classes & Interfaces",
            keys: [
              "drugex::training::generators::Generator",
              "drugex::training::generators::FragGenerator"
            ]
          },
          {
            subtitle: "Transformer Attention Blocks & Layers",
            keys: [
              "drugex::training::generators::graph_transformer::Block",
              "drugex::training::generators::graph_transformer::AtomLayer",
              "drugex::training::generators::sequence_transformer::Block",
              "drugex::training::generators::sequence_transformer::GPT2Layer",
              "drugex::training::generators::utils::PositionwiseFeedForward",
              "drugex::training::generators::utils::SublayerConnection",
              "drugex::training::generators::utils::PositionalEmbedding",
              "drugex::training::generators::utils::PositionalEncoding"
            ]
          },
          {
            subtitle: "Attention Masks & Graph Featurization",
            keys: [
              "drugex::training::generators::utils::pad_mask",
              "drugex::training::generators::utils::tri_mask",
              "drugex::training::generators::utils::unique",
              "drugex::utils::gcmol::canonicalize",
              "drugex::utils::gcmol::canonicalize_list",
              "drugex::utils::gcmol::remove_duplicates"
            ]
          }
        ]
      },
      {
        id: "explorers",
        title: "Explorers & RL Search Policies",
        subsections: [
          {
            subtitle: "Sequence RL Explorers",
            keys: [
              "drugex::training::explorers::SequenceExplorer",
              "drugex::training::explorers::FragSequenceExplorer"
            ]
          },
          {
            subtitle: "Graph & Fragment RL Explorers",
            keys: [
              "drugex::training::explorers::GraphExplorer",
              "drugex::training::explorers::FragGraphExplorer"
            ]
          },
          {
            subtitle: "Explorer Base Architecture",
            keys: [
              "drugex::training::explorers::Explorer",
              "drugex::training::explorers::FragExplorer"
            ]
          }
        ]
      },
      {
        id: "scoring",
        title: "Scoring & Objective Oracles",
        subsections: [
          {
            subtitle: "Oracle Base Interfaces & Properties",
            keys: [
              "drugex::training::scorers::Scorer",
              "drugex::training::scorers::Property",
              "drugex::training::scorers::AtomCounter",
              "drugex::training::scorers::Isomer",
              "drugex::training::scorers::Scaffold",
              "drugex::training::scorers::Uniqueness"
            ]
          },
          {
            subtitle: "3D ROCS & Shape-Matching Oracles",
            keys: [
              "drugex::training::scorers::RDKitROCSScorer",
              "drugex::training::scorers::CDPKitROCSScorer",
              "drugex::training::scorers::OpenEyeROCSScorer",
              "drugex::training::scorers::rocs_cdpkit::CDPKitWorkerContext",
              "drugex::training::scorers::rocs_cdpkit::CDPKitScoringWorker"
            ]
          },
          {
            subtitle: "Conformer Generation Engines",
            keys: [
              "drugex::training::scorers::conformer_generators::ConformerGenerator",
              "drugex::training::scorers::conformer_generators::RDKitConformerGenerator",
              "drugex::training::scorers::conformer_generators::CDPKitConformerGenerator",
              "drugex::training::scorers::conformer_generators::OmegaConformerGenerator",
              "drugex::training::scorers::conformer_generators::SchrodingerConformerGenerator"
            ]
          },
          {
            subtitle: "Synthetic Accessibility & Retrosynthesis",
            keys: [
              "drugex::training::scorers::SmilesChecker",
              "drugex::training::scorers::SAScorer",
              "drugex::training::scorers::sascorer::calculateScore",
              "drugex::training::scorers::sascorer::readFragmentScores",
              "drugex::training::scorers::sascorer::numBridgeheadsAndSpiro",
              "drugex::training::scorers::RAScoreScorer",
              "drugex::training::scorers::ra_scorer::RetrosyntheticAccessibilityScorer"
            ]
          },
          {
            subtitle: "Molecular Similarity & Efficiency Oracles",
            keys: [
              "drugex::training::scorers::LipophilicEfficiency",
              "drugex::training::scorers::LigandEfficiency",
              "drugex::training::scorers::TverskyFingerprintSimilarity",
              "drugex::training::scorers::similarity::FraggleSimilarity",
              "drugex::training::scorers::similarity::GraphEditInverseDistance",
              "drugex::training::scorers::similarity::TverskyGraphSimilarity",
              "drugex::utils::fingerprints::get_fingerprint"
            ]
          },
          {
            subtitle: "Machine Learning Surrogates",
            keys: [
              "drugex::training::scorers::qsprpred::QSPRPredScorer"
            ]
          }
        ]
      },
      {
        id: "modifiers",
        title: "Desirability Modifiers",
        subsections: [
          {
            subtitle: "Threshold & Boundary Modifiers",
            keys: [
              "drugex::training::scorers::ScoreModifier",
              "drugex::training::scorers::SmoothClippedScore",
              "drugex::training::scorers::ClippedScore",
              "drugex::training::scorers::modifiers::ThresholdedLinear"
            ]
          },
          {
            subtitle: "Gaussian, Polynomial & Non-Linear Functions",
            keys: [
              "drugex::training::scorers::Gaussian",
              "drugex::training::scorers::MinMaxGaussian",
              "drugex::training::scorers::modifiers::SmoothHump",
              "drugex::training::scorers::modifiers::Squared",
              "drugex::training::scorers::modifiers::AbsoluteScore",
              "drugex::training::scorers::Linear"
            ]
          },
          {
            subtitle: "Composite Pipelines",
            keys: [
              "drugex::training::scorers::modifiers::Chained"
            ]
          }
        ]
      },
      {
        id: "data",
        title: "Data, Fragments & Vocabularies",
        subsections: [
          {
            subtitle: "PyTorch Datasets & Data Containers",
            keys: [
              "drugex::data::interfaces::DataSet",
              "drugex::data::datasets::SmilesDataSet",
              "drugex::data::datasets::SmilesFragDataSet",
              "drugex::data::datasets::GraphFragDataSet",
              "drugex::data::datasets::TargetCreator",
              "drugex::data::datasets::TgtData"
            ]
          },
          {
            subtitle: "Corpus & Vocabularies",
            keys: [
              "drugex::data::corpus::interfaces::Corpus",
              "drugex::data::corpus::interfaces::Vocabulary",
              "drugex::data::corpus::interfaces::SequenceVocabulary",
              "drugex::data::corpus::vocabulary::VocSmiles",
              "drugex::data::corpus::vocabulary::VocNonGPT",
              "drugex::data::corpus::vocabulary::VocGraph",
              "drugex::data::corpus::SequenceCorpus"
            ]
          },
          {
            subtitle: "Fragment Encoders & Graph Cleavage",
            keys: [
              "drugex::data::fragments::Fragmenter",
              "drugex::data::fragments::SequenceFragmentEncoder",
              "drugex::data::fragments::GraphFragmentEncoder",
              "drugex::data::fragments::FragmentCorpusEncoder",
              "drugex::data::interfaces::FragmentPairEncoder"
            ]
          },
          {
            subtitle: "Suppliers, Splitters & Loaders",
            keys: [
              "drugex::data::fragments::FragmentPairsSupplier",
              "drugex::data::fragments::FragmentPairsEncodedSupplier",
              "drugex::data::fragments::FragmentPairsSplitter",
              "drugex::data::fragments::FragmentPairsCollector",
              "drugex::data::interfaces::DataSplitter",
              "drugex::data::interfaces::DataToLoader",
              "drugex::data::processing::RandomTrainTestSplitter"
            ]
          },
          {
            subtitle: "Standardization & Path Helpers",
            keys: [
              "drugex::data::processing::Standardization",
              "drugex::data::processing::CorpusEncoder",
              "drugex::data::processing::StandardizationCollector",
              "drugex::data::processing::Standardization::Collector",
              "drugex::data::utils::getDataPaths",
              "drugex::data::utils::getVocPaths"
            ]
          }
        ]
      },
      {
        id: "molecules",
        title: "Molecules, Converters & Exceptions",
        subsections: [
          {
            subtitle: "Molecule Data Structures & Suppliers",
            keys: [
              "drugex::molecules::mol::DrExMol",
              "drugex::molecules::interfaces::Molecule",
              "drugex::molecules::interfaces::MolSupplier",
              "drugex::molecules::interfaces::BaseMolSupplier",
              "drugex::molecules::suppliers::StandardizedSupplier",
              "drugex::molecules::suppliers::ListSupplier"
            ]
          },
          {
            subtitle: "Molecule Converters & Transformers",
            keys: [
              "drugex::molecules::converters::MolConverter",
              "drugex::molecules::converters::default::SmilesToDrEx",
              "drugex::molecules::converters::default::DrExToSMILES",
              "drugex::molecules::converters::default::RDKitToDrEx",
              "drugex::molecules::converters::default::Identity",
              "drugex::molecules::converters::dummyMolsFromFragments",
              "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment"
            ]
          },
          {
            subtitle: "Standardizers & Chemical Sanitizers",
            keys: [
              "drugex::molecules::converters::DefaultStandardizer",
              "drugex::molecules::converters::CleanSMILES"
            ]
          },
          {
            subtitle: "Exception Hierarchy",
            keys: [
              "drugex::molecules::converters::ConversionException",
              "drugex::molecules::converters::StandardizationException",
              "drugex::molecules::interfaces::ItemException",
              "drugex::molecules::mol::InitializationException",
              "drugex::training::interfaces::RewardScheme::RewardException",
              "drugex::parallel::interfaces::ParallelException",
              "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException",
              "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException"
            ]
          }
        ]
      },
      {
        id: "workflows",
        title: "Parallel Processing & Workflows",
        subsections: [
          {
            subtitle: "Parallel Execution & Map-Reduce Collectors",
            keys: [
              "drugex::parallel::ParallelProcessor",
              "drugex::parallel::ParallelSupplierEvaluator",
              "drugex::parallel::interfaces::ResultCollector",
              "drugex::parallel::interfaces::ListCollector",
              "drugex::parallel::collectors::ListExtend",
              "drugex::parallel::collectors::ListAppend"
            ]
          },
          {
            subtitle: "High-Level Dataset CLI & Builders",
            keys: [
              "drugex::dataset::Dataset",
              "drugex::dataset::FragmentDataset",
              "drugex::dataset::SequenceDataset",
              "drugex::dataset::FragSequenceDataset",
              "drugex::dataset::FragGraphDataset",
              "drugex::dataset::DatasetArgParser",
              "drugex::dataset::load_molecules"
            ]
          },
          {
            subtitle: "Training Pipelines & CLI Entrypoints",
            keys: [
              "drugex::train::train_model",
              "drugex::train::Pretrain",
              "drugex::train::Finetune",
              "drugex::train::Reinforce",
              "drugex::train::CreateEnvironment",
              "drugex::train::DataPreparation",
              "drugex::train::SmilesDataPreparation",
              "drugex::train::FragSmilesDataPreparation",
              "drugex::train::FragGraphDataPreparation",
              "drugex::train::SetUpGenerator",
              "drugex::train::getModifiers",
              "drugex::train::GeneratorArgParser"
            ]
          },
          {
            subtitle: "De Novo Generation Workflows",
            keys: [
              "drugex::generate::Design",
              "drugex::generate::DesignerFragsDataPreparation",
              "drugex::generate::DesignArgParser"
            ]
          },
          {
            subtitle: "Logging, Provenance & Checkpoint Backup",
            keys: [
              "drugex::logs::setLogger",
              "drugex::logs::config::LogFileConfig",
              "drugex::logs::config::LevelFilter",
              "drugex::logs::config::LevelFormatter",
              "drugex::logs::config::config_logger",
              "drugex::logs::config::init_logfile",
              "drugex::logs::config::get_git_info",
              "drugex::logs::config::get_runid",
              "drugex::logs::utils::enable_file_logger",
              "drugex::logs::utils::generate_backup_runID",
              "drugex::logs::utils::generateBackupDir",
              "drugex::logs::utils::backUpFiles",
              "drugex::logs::utils::backUpFilesInFolder",
              "drugex::logs::utils::callwarning",
              "drugex::logs::utils::export_conda_environment"
            ]
          },
          {
            subtitle: "Data Downloads & Cryptographic Verification",
            keys: [
              "drugex::download::DownloadTutorial",
              "drugex::download::DownloadArgParser",
              "drugex::utils::download::download_file",
              "drugex::utils::download::sha256sum",
              "drugex::utils::download::check_sha256sum"
            ]
          }
        ]
      },
      {
        id: "ecosystem",
        title: "Chemoinformatics Ecosystem (RDKit, QSPRPred, CDPKit)",
        subsections: [
          {
            subtitle: "RDKit: 3D Conformations, Alignment & PMI Shapes",
            keys: [
              "rdkit::Chem::rdShapeAlign::AlignMol",
              "rdkit::Chem::AllChem::EmbedMultipleConfs",
              "rdkit::Chem::Descriptors3D::NPR1_NPR2"
            ]
          },
          {
            subtitle: "RDKit: Retrosynthetic Fragmentation & Scaffold Hopping",
            keys: [
              "rdkit::Chem::BRICS::BRICSDecompose",
              "rdkit::Chem::Scaffolds::MurckoScaffold::MakeScaffoldGeneric"
            ]
          },
          {
            subtitle: "RDKit: Standardization & Descriptors",
            keys: [
              "rdkit::Chem::MolStandardize::rdMolStandardize",
              "rdkit::Chem::rdDepictor::Compute2DCoords",
              "rdkit::Chem::Descriptors::MolLogP",
              "rdkit::Chem::Descriptors::MolWt",
              "rdkit::Chem::QED::qed",
              "rdkit::Chem::Lipinski::NumRotatableBonds",
              "rdkit::Chem::rdMolDescriptors::CalcNumHBA_HBD",
              "rdkit::Chem::rdMolDescriptors::CalcTPSA",
              "rdkit::Chem::MACCSkeys::GenMACCSKeys"
            ]
          },
          {
            subtitle: "QSPRPred: Automated QSAR / QSPR Machine Learning",
            keys: [
              "qsprpred::data::MoleculeTable",
              "qsprpred::models::QSPRModel",
              "qsprpred::models::SklearnModel",
              "qsprpred::data::sources::Papyrus",
              "qsprpred::data::processing::feature_filters::LowVarianceFilter"
            ]
          },
          {
            subtitle: "CDPKit: Pharmacophores & Gaussian Shape Overlaps",
            keys: [
              "cdpkit::CDPL::Shape::PharmacophoreAlignment",
              "cdpkit::CDPL::Pharm::PharmacophoreGenerator",
              "cdpkit::CDPL::Shape::ShapeOverlap",
              "cdpkit::CDPL::Descr::AutoCorrelation3D",
              "cdpkit::CDPL::Chem::Molecule"
            ]
          }
        ]
      },
      {
        id: "experimental",
        title: "Experimental Extensions",
        subsections: [
          {
            subtitle: "3D & 4D Ultrafast Shape Screening",
            keys: [
              "experimental::usrcat::USRCAT_Screening",
              "experimental::electroshape::ElectroShape_4D"
            ]
          },
          {
            subtitle: "Protein-Ligand Interaction Fingerprints (ProLIF)",
            keys: [
              "experimental::prolif::InteractionFingerprint",
              "experimental::prolif::BitVectorAggregator",
              "experimental::prolif::Fingerprint"
            ]
          },
          {
            subtitle: "Deep Graph QSAR & Uncertainty (Chemprop D-MPNN)",
            keys: [
              "experimental::chemprop::DMPNN_Predictor",
              "experimental::chemprop::UncertaintyEstimator",
              "experimental::chemprop::EnsembleUncertainty",
              "experimental::chemprop::MoleculeDataset"
            ]
          },
          {
            subtitle: "Chemical Space & Retrosynthesis",
            keys: [
              "experimental::scaffviz::ChemicalSpaceExplorer",
              "experimental::rascore::AiZynth_Accessibility"
            ]
          },
          {
            subtitle: "Structure-Based Docking & Conformational Strain",
            keys: [
              "experimental::vina::AutoDockVina_Scoring",
              "experimental::vina::VinaDockingWorker",
              "experimental::vina::ReceptorGrid",
              "experimental::vina::FlexibleDocking",
              "experimental::vina::VinaConfig",
              "experimental::strain::MMFF94_ConformationalStrain"
            ]
          },
          {
            subtitle: "Alchemical Free Energy Perturbation (OpenFE)",
            keys: [
              "experimental::openfe::RelativeFreeEnergy",
              "experimental::openfe::LOMAP_NetworkPlanner",
              "experimental::openfe::AlchemicalNetwork",
              "experimental::openfe::ChemicalSystem"
            ]
          },
          {
            subtitle: "Bayesian Multi-Objective Hyperparameter Tuning (Optuna)",
            keys: [
              "experimental::optuna::TPE_RL_Tuning",
              "experimental::optuna::ParetoMultiObjectiveOptimizer",
              "experimental::optuna::TPESampler"
            ]
          }
        ]
      }
  ];
})();
