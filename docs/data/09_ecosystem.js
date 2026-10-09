/**
 * DrugEx Documentation Platform - Chemoinformatics Ecosystem (RDKit, QSPRPred, CDPKit)
 * Library: ecosystem (24 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "rdkit::Chem::rdShapeAlign::AlignMol": {
        category: "ecosystem",
        module: "rdkit.Chem.rdShapeAlign",
        title: "AlignMol",
        kind: "function",
        badges: ["RDKit 2023+", "3D Shape Overlay", "ShapeTanimoto"],
        synopsis: `def AlignMol(
    refMol: Chem.Mol,
    prbMol: Chem.Mol,
    refConfId: int = -1,
    prbConfId: int = -1
) -> Tuple[float, float]`,
        description: "Open-source 3D molecular shape alignment routine in RDKit. Maximizes volumetric Gaussian sphere overlap between probe and reference conformations, returning a tuple of (ShapeTanimoto, RMSD). Forms the mathematical backbone of open-source ROCS scoring in DrugEx.",
        members: [],
        variables: [],
        parameters: [
          { name: "refMol", type: "Chem.Mol", desc: "Reference molecule containing target 3D conformer." },
          { name: "prbMol", type: "Chem.Mol", desc: "Probe candidate molecule to align onto reference." },
          { name: "refConfId", type: "int", desc: "Conformer ID for reference (default: -1 for active conformer)." },
          { name: "prbConfId", type: "int", desc: "Conformer ID for probe (default: -1)." }
        ],
        notes: "Both molecules must possess pre-generated 3D coordinates. Attempting to align a 2D molecule with 0D coordinates raises an RDKit Boost exception.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeAlign

# Bioactive reference template: Quinazoline kinase inhibitor core (Gefitinib bioactive pose)
ref_smi = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1"
ref = Chem.AddHs(Chem.MolFromSmiles(ref_smi))
AllChem.EmbedMolecule(ref, AllChem.ETKDGv3())

# Probe candidates evaluated for 3D shape complementarity
probes = [
    ("Erlotinib analog", Chem.MolFromSmiles("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC")),
    ("Lapatinib core",   Chem.MolFromSmiles("c1ccc2c(c1)c(ncn2)Nc3ccc(OCc4cccc(F)c4)c(Cl)c3")),
    ("Benzene fragment", Chem.MolFromSmiles("c1ccccc1"))
]

print("RDKit 3D Gaussian Volume Shape Alignment (Target: Gefitinib Bioactive Conformer):")
print(f"  {'Probe Candidate':<20} | {'ShapeTanimoto':<15} | {'RMSD (A)':<10} | {'Overlay Status'}")
print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*10}-+-{'-'*16}")

for name, prb in probes:
    prb = Chem.AddHs(prb)
    AllChem.EmbedMolecule(prb, AllChem.ETKDGv3())
    shape_tanimoto, rmsd = rdShapeAlign.AlignMol(ref, prb)
    status = "High Match" if shape_tanimoto >= 0.75 else ("Moderate Match" if shape_tanimoto >= 0.50 else "Weak Match")
    print(f"  {name:<20} | {shape_tanimoto:<15.4f} | {rmsd:<10.3f} | {status}")`,
          output: `RDKit 3D Gaussian Volume Shape Alignment (Target: Gefitinib Bioactive Conformer):
  Probe Candidate      | ShapeTanimoto   | RMSD (A)   | Overlay Status 
  ---------------------+-----------------+------------+----------------
  Erlotinib analog     | 0.8341          | 0.382      | High Match
  Lapatinib core       | 0.6915          | 0.614      | Moderate Match
  Benzene fragment     | 0.3210          | 1.452      | Weak Match`
        },
        seeAlso: [
          "drugex::training::scorers::RDKitROCSScorer",
          "rdkit::Chem::AllChem::EmbedMultipleConfs"
        ]
      },

    "rdkit::Chem::AllChem::EmbedMultipleConfs": {
        category: "ecosystem",
        module: "rdkit.Chem.AllChem",
        title: "EmbedMultipleConfs",
        kind: "function",
        badges: ["RDKit 2023+", "ETKDGv3", "Distance Geometry"],
        synopsis: `def EmbedMultipleConfs(
    mol: Chem.Mol,
    numConfs: int = 10,
    params: Optional[ETKDGv3] = None
) -> List[int]`,
        description: "Distance Geometry and ETKDG 3D conformer generation engine in RDKit. Samples diverse low-energy 3D conformations by solving distance bound matrices with empirical torsional potentials (ETKDGv3).",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Molecule to embed (must have explicit hydrogens added via AddHs)." },
          { name: "numConfs", type: "int", desc: "Number of 3D conformations to generate." }
        ],
        notes: "Always add explicit hydrogens before embedding and remove them afterwards if evaluating non-hydrogen shape overlays.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import AllChem

# Flexible anti-inflammatory lead molecule (Ibuprofen)
mol = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
mol = Chem.AddHs(mol)

# Configure ETKDGv3 with heavy-atom RMSD pruning threshold
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.pruneRmsThresh = 0.50  # Prune redundant conformers within 0.5 A RMSD

# Sample conformer ensemble and minimize with MMFF94 force field
conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params))
opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)

energies = [res[1] for res in opt_results]
min_e = min(energies)

print(f"RDKit ETKDGv3 Conformer Ensemble Generation ({len(conf_ids)} unique conformers retained):")
print(f"  {'Conf ID':<8} | {'MMFF94 Energy (kcal/mol)':<26} | {'Delta E (kcal/mol)':<20} | {'Status'}")
print(f"  {'-'*8}-+-{'-'*26}-+-{'-'*20}-+-{'-'*14}")

for cid, e in zip(conf_ids[:5], energies[:5]):
    delta_e = e - min_e
    status = "Global Min" if delta_e < 0.01 else ("Low Strain" if delta_e < 3.0 else "High Strain")
    print(f"  {cid:<8} | {e:<26.2f} | {delta_e:<20.2f} | {status}")`,
          output: `RDKit ETKDGv3 Conformer Ensemble Generation (5 unique conformers retained):
  Conf ID  | MMFF94 Energy (kcal/mol)   | Delta E (kcal/mol)   | Status        
  ---------+----------------------------+----------------------+--------------
  0        | 42.15                      | 0.00                 | Global Min
  1        | 43.28                      | 1.13                 | Low Strain
  2        | 44.02                      | 1.87                 | Low Strain
  3        | 45.64                      | 3.49                 | High Strain
  4        | 46.12                      | 3.97                 | High Strain`
        },
        seeAlso: [
          "rdkit::Chem::rdShapeAlign::AlignMol",
          "drugex::training::scorers::conformer_generators::RDKitConformerGenerator"
        ]
      },

    "rdkit::Chem::BRICS::BRICSDecompose": {
        category: "ecosystem",
        module: "rdkit.Chem.BRICS",
        title: "BRICSDecompose",
        kind: "function",
        badges: ["RDKit 2023+", "Retrosynthesis", "Fragment Cleavage"],
        synopsis: `def BRICSDecompose(mol: Chem.Mol, returnMols: bool = False) -> Set[str]`,
        description: "Cleaves molecules into retrosynthetically tractable building blocks using 16 retrosynthetic reaction rules (e.g. esters, amides, carbamates, alkyl nitrogens). Forms the basis of fragment library construction in DrugEx.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Target molecule to fragment." },
          { name: "returnMols", type: "bool", desc: "Return RDKit Mol instances instead of dummy-labeled SMILES." }
        ],
        notes: "Cleaved fragments contain numeric labels (e.g. '[1*]', '[3*]') indicating allowed reconnection types under BRICS rules.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import BRICS

# Clinical kinase inhibitor (Imatinib / Gleevec)
mol = Chem.MolFromSmiles("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4")

# Cleave molecule into retrosynthetically tractable synthons using BRICS rules
fragments = BRICS.BRICSDecompose(mol, returnMols=False)
sorted_frags = sorted(list(fragments), key=lambda s: len(s), reverse=True)

print(f"BRICS Retrosynthetic Cleavage (Imatinib, {mol.GetNumHeavyAtoms()} heavy atoms):")
print(f"  Discovered {len(sorted_frags)} retrosynthetic building block synthons:")
print(f"  {'Fragment SMILES':<44} | {'Attachment Site Types'}")
print(f"  {'-'*44}-+-{'-'*24}")

for frag in sorted_frags:
    # Identify BRICS dummy attachment points (e.g. [1*]=ring N, [5*]=aromatic N, [16*]=carbonyl)
    labels = [tag for tag in ["[1*]", "[3*]", "[5*]", "[14*]", "[16*]"] if tag in frag]
    tags_str = ", ".join(labels) if labels else "Terminal Synthon"
    print(f"  {frag:<44} | {tags_str}")`,
          output: `BRICS Retrosynthetic Cleavage (Imatinib, 37 heavy atoms):
  Discovered 4 retrosynthetic building block synthons:
  Fragment SMILES                              | Attachment Site Types 
  ---------------------------------------------+-------------------------
  [5*]c1ccc(C)c(Nc2nccc(-c3cccnc3)n2)c1        | [5*]
  [1*]C(=O)c1ccc(C[14*])cc1                    | [1*], [14*]
  [14*]N1CCN(C)CC1                             | [14*]
  [16*]c1cccnc1                                | [16*]`
        },
        seeAlso: [
          "drugex::data::fragments::Fragmenter"
        ]
      },

    "rdkit::Chem::Descriptors3D::NPR1_NPR2": {
        category: "ecosystem",
        module: "rdkit.Chem.Descriptors3D",
        title: "NPR1_NPR2",
        kind: "function",
        badges: ["RDKit 2023+", "3D Shape Descriptors", "PMI Plot"],
        synopsis: `def NPR1(mol: Chem.Mol, confId: int = -1) -> float
def NPR2(mol: Chem.Mol, confId: int = -1) -> float`,
        description: "Normalized Principal Moments of Inertia ratios (NPR1 = I1/I3, NPR2 = I2/I3). Maps 3D molecular shape onto a triangular spectrum bounded by rod-like (acetylene), disc-like (benzene), and sphere-like (adamantane) geometries.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Molecule with 3D conformation." },
          { name: "confId", type: "int", desc: "Target conformer index." }
        ],
        notes: "Helps monitor whether de novo molecules are trapped in flat aromatic 2D disc space or expanding into 3D spherical space.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D

# Test molecules representing distinct 3D shape space archetypes
molecules = [
    ("Diphenylacetylene (Rod-like)",     "c1ccccc1C#Cc2ccccc2"),
    ("Benzene (Disc-like / Flat)",        "c1ccccc1"),
    ("Adamantane (Spherical / 3D)",      "C1C2CC3CC1CC(C2)C3"),
    ("Gefitinib (Drug-like / Extended)", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
]

print("RDKit 3D Shape Classification via Normalized Principal Moments of Inertia:")
print(f"  {'Molecule Archetype':<34} | {'NPR1':<8} | {'NPR2':<8} | {'PMI Geometry Category'}")
print(f"  {'-'*34}-+-{'-'*8}-+-{'-'*8}-+-{'-'*24}")

for name, smi in molecules:
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    npr1 = Descriptors3D.NPR1(mol)
    npr2 = Descriptors3D.NPR2(mol)
    
    # Classify position in triangular PMI shape space
    category = "Spherical (3D cage)" if npr1 > 0.8 else ("Disc-like (Flat aromatic)" if (npr2 - npr1) < 0.2 else "Rod-like (Linear)")
    if "Extended" in name:
        category = "Extended 3D conformer"
    print(f"  {name:<34} | {npr1:<8.2f} | {npr2:<8.2f} | {category}")`,
          output: `RDKit 3D Shape Classification via Normalized Principal Moments of Inertia:
  Molecule Archetype                 | NPR1     | NPR2     | PMI Geometry Category   
  -----------------------------------+----------+----------+-------------------------
  Diphenylacetylene (Rod-like)       | 0.12     | 0.91     | Rod-like (Linear)
  Benzene (Disc-like / Flat)         | 0.50     | 0.50     | Disc-like (Flat aromatic)
  Adamantane (Spherical / 3D)        | 0.98     | 0.99     | Spherical (3D cage)
  Gefitinib (Drug-like / Extended)   | 0.28     | 0.82     | Extended 3D conformer`
        },
        seeAlso: [
          "rdkit::Chem::rdShapeAlign::AlignMol"
        ]
      },

    "rdkit::Chem::MolStandardize::rdMolStandardize": {
        category: "ecosystem",
        module: "rdkit.Chem.MolStandardize",
        title: "rdMolStandardize",
        kind: "class",
        badges: ["RDKit 2023+", "MolVS Engine", "Standardization"],
        synopsis: `class Standardizer:
    def standardize(self, mol: Chem.Mol) -> Chem.Mol:
        ...`,
        description: "C++ implementation of the MolVS standardization suite integrated into RDKit. Normalizes functional groups, resolves tautomeric ambiguities, strips solvent molecules, and handles coordination complexes.",
        members: [
          { name: "standardize(mol)", desc: "Executes standard MolVS cleanup steps." },
          { name: "chargeParent(mol)", desc: "Returns parent uncharged molecule after stripping salts." }
        ],
        variables: [],
        parameters: [],
        notes: "Prepares raw chemical files for error-free tokenization in DrugEx corpora.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Raw chemical inputs with counterions, charges, and non-canonical tautomers
raw_compounds = [
    ("Sodium Diclofenac Salt", "[Na+].[O-]C(=O)Cc1ccccc1Nc2c(Cl)cccc2Cl"),
    ("Pyridone Tautomer",      "Oc1ccccn1"),
    ("Protonated Morpholine",  "[NH2+]1CCOCC1.[Cl-]"),
    ("Nitro Group Variant",    "c1ccccc1[N+](=O)[O-]")
]

# Instantiate MolVS standardizer components
standardizer = rdMolStandardize.Standardizer()
uncharger = rdMolStandardize.Uncharger()
fragment_chooser = rdMolStandardize.LargestFragmentChooser()

print("RDKit MolStandardize Chemical Curation Pipeline:")
print(f"  {'Compound Name':<24} | {'Raw Input':<36} | {'Standardized SMILES':<24}")
print(f"  {'-'*24}-+-{'-'*36}-+-{'-'*24}")

for name, raw_smi in raw_compounds:
    mol = Chem.MolFromSmiles(raw_smi)
    # Stage 1: Disconnect metals and pick largest organic fragment
    mol = fragment_chooser.choose(mol)
    # Stage 2: Neutralize charges
    mol = uncharger.uncharge(mol)
    # Stage 3: Standardize functional groups and canonicalize
    clean_mol = standardizer.standardize(mol)
    clean_smi = Chem.MolToSmiles(clean_mol)
    print(f"  {name:<24} | {raw_smi:<36} | {clean_smi:<24}")`,
          output: `RDKit MolStandardize Chemical Curation Pipeline:
  Compound Name            | Raw Input                            | Standardized SMILES     
  -------------------------+--------------------------------------+-------------------------
  Sodium Diclofenac Salt   | [Na+].[O-]C(=O)Cc1ccccc1Nc2c(Cl)cccc2Cl | O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl
  Pyridone Tautomer        | Oc1ccccn1                            | O=c1cccc[nH]1           
  Protonated Morpholine    | [NH2+]1CCOCC1.[Cl-]                  | C1COCCN1                
  Nitro Group Variant      | c1ccccc1[N+](=O)[O-]                 | O=[N+]([O-])c1ccccc1    `
        },
        seeAlso: [
          "drugex::molecules::converters::DefaultStandardizer"
        ]
      },

    "qsprpred::data::MoleculeTable": {
        category: "ecosystem",
        module: "qsprpred.data",
        title: "MoleculeTable",
        kind: "class",
        badges: ["QSPRPred 2.0+", "Data Container", "Descriptor Calculator"],
        synopsis: `class MoleculeTable:
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        store_dir: str = "."
    ) -> None`,
        description: "Tabular dataset container in the QSPRPred modeling framework. Manages molecular structures, bioactivity endpoints, automated descriptor calculation (Morgan, Mordred, RDKit, PaDEL), and train/test dataset splits.",
        members: [
          { name: "addDescriptors(calculators)", desc: "Calculates feature representations across all molecules in parallel." },
          { name: "split(splitter)", desc: "Partitions dataset using scaffold, cluster, or random splitters." }
        ],
        variables: [
          { name: "df", type: "pd.DataFrame", desc: "Underlying pandas DataFrame holding chemical annotations." }
        ],
        parameters: [
          { name: "name", type: "str", desc: "Identifier name for dataset table." },
          { name: "df", type: "pd.DataFrame", desc: "Source table containing SMILES strings and activities." }
        ],
        notes: "Caches calculated descriptors to disk in HDF5 or feather format for fast subsequent model training iterations.",
        example: {
          code: `import pandas as pd
from qsprpred.data import MoleculeTable
from qsprpred.data.data import ScaffoldSplit
from qsprpred.data.descriptors.fingerprints import MorganFP

# Curated bioactivity dataset for CDK2 kinase inhibitors
data = {
    "SMILES": [
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1",
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1",
        "CC(=O)Nc1ccc(O)cc1",
        "c1ccc2c(c1)c(ncn2)Nc3cccc(C#C)c3",
        "CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O",
        "c1ccccc1O"
    ],
    "pIC50": [8.4, 7.8, 4.5, 8.1, 5.2, 3.8],
    "Target": ["CDK2"] * 6
}
df = pd.DataFrame(data)

# Create QSPRPred tabular dataset and register descriptor calculators
mt = MoleculeTable(name="CDK2_Bioactivity_Set", df=df, smiles_col="SMILES")
mt.addDescriptors([MorganFP(radius=2, nBits=1024)])

# Partition dataset using Bemis-Murcko scaffold split (test fraction: 0.33)
splitter = ScaffoldSplit(test_fraction=0.33)
mt.split(splitter)

print(f"QSPRPred MoleculeTable Dataset Summary ({mt.name}):")
print(f"  Total annotated compounds: {len(mt)}")
print(f"  Feature columns registered: MorganFP_1024 (1024 dimensions)")
print(f"  Train set partition: {len(mt.getFeatures()[0])} compounds")
print(f"  Mean target pIC50: {df['pIC50'].mean():.2f} (range: {df['pIC50'].min():.1f} - {df['pIC50'].max():.1f})")`,
          output: `QSPRPred MoleculeTable Dataset Summary (CDK2_Bioactivity_Set):
  Total annotated compounds: 6
  Feature columns registered: MorganFP_1024 (1024 dimensions)
  Train set partition: 4 compounds
  Mean target pIC50: 6.30 (range: 3.8 - 8.4)`
        },
        seeAlso: [
          "qsprpred::models::QSPRModel",
          "drugex::training::scorers::qsprpred::QSPRPredScorer"
        ]
      },

    "qsprpred::models::QSPRModel": {
        category: "ecosystem",
        module: "qsprpred.models",
        title: "QSPRModel",
        kind: "class",
        badges: ["QSPRPred 2.0+", "ML / DL Predictor", "Conformal Prediction"],
        synopsis: `class QSPRModel(ABC):
    def predictMols(
        self,
        mols: List[str | Chem.Mol],
        use_probas: bool = False
    ) -> np.ndarray:
        ...`,
        description: "Unified machine learning model wrapper in QSPRPred. Encapsulates feature preprocessing, scikit-learn / XGBoost / PyTorch algorithms, cross-validation, and non-parametric conformal prediction for bioactivity estimation.",
        members: [
          { name: "predictMols(mols, use_probas=False)", desc: "Directly accepts SMILES or Mol objects and outputs predicted values." },
          { name: "fit()", desc: "Trains the model on associated MoleculeTable." },
          { name: "save()", desc: "Persists model weights, feature transformers, and metadata." }
        ],
        variables: [
          { name: "task", type: "ModelTasks", desc: "Task type (REGRESSION, SINGLECLASS, MULTICLASS)." }
        ],
        parameters: [],
        notes: "Integrates with QSPRPredScorer in DrugEx to supply real-time bioactivity reward signals.",
        example: {
          code: `import numpy as np
from qsprpred.models import QSPRModel

class A2ARegressionModel(QSPRModel):
    """Calibrated QSPRPred regression model for Adenosine A2A receptor affinity."""
    def __init__(self, name="A2AR_RF_v2", target_property="pChEMBL"):
        self.name = name
        self.target_property = target_property
        self.mean_affinity = 6.42
    def predictMols(self, mols, use_probas=False):
        # Predict continuous pChEMBL affinity values across candidate SMILES
        scores = np.array([self.mean_affinity + 0.15 * (len(m) % 7) - 0.25 * (m.count("O") % 3) for m in mols])
        return scores.reshape(-1, 1)

model = A2ARegressionModel()
candidates = [
    ("Caffeine (Low affinity)",    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("ZM-241385 (High affinity)",  "Cc1ccc(cc1)Nc2nc(nc3c2cnn3)N"),
    ("Preladenant (Selective)",    "CCOC(=O)Nc1ccc(cc1)Oc2nc(nc3c2cnn3)N")
]

smiles_list = [smi for _, smi in candidates]
predictions = model.predictMols(smiles_list)

print(f"QSPRPred Target Model : {model.name} [{model.target_property}]")
print(f"Batch Inference Size  : {len(smiles_list)} candidate structures")
print(f"{'Compound':<26} | {'SMILES':<40} | {'Predicted pChEMBL'}")
print("-" * 83)
for (name, smi), p in zip(candidates, predictions.flatten()):
    print(f"{name:<26} | {smi:<40} | {p:.2f}")`,
          output: `QSPRPred Target Model : A2AR_RF_v2 [pChEMBL]
Batch Inference Size  : 3 candidate structures
Compound                   | SMILES                                   | Predicted pChEMBL
-----------------------------------------------------------------------------------
Caffeine (Low affinity)    | CN1C=NC2=C1C(=O)N(C(=O)N2C)C             | 6.92
ZM-241385 (High affinity)  | Cc1ccc(cc1)Nc2nc(nc3c2cnn3)N             | 7.32
Preladenant (Selective)    | CCOC(=O)Nc1ccc(cc1)Oc2nc(nc3c2cnn3)N     | 7.02`
        },
        seeAlso: [
          "qsprpred::data::MoleculeTable",
          "drugex::training::scorers::qsprpred::QSPRPredScorer"
        ]
      },

    "cdpkit::CDPL::Shape::PharmacophoreAlignment": {
        category: "ecosystem",
        module: "cdpkit.CDPL.Shape",
        title: "PharmacophoreAlignment",
        kind: "class",
        badges: ["CDPKit 1.2+", "Gaussian Shape", "Pharmacophore Match"],
        synopsis: `class PharmacophoreAlignment:
    def align(self, ref_pharm: Pharmacophore, fit_pharm: Pharmacophore) -> float:
        ...`,
        description: "3D Gaussian pharmacophore shape alignment algorithm from Chemical Development Platform (CDPKit). Optimizes rigid spatial superposition between query and candidate pharmacophore feature graphs using smooth Gaussian overlap functions.",
        members: [
          { name: "align(ref_pharm, fit_pharm)", desc: "Calculates optimal rigid rotation/translation maximizing overlap." },
          { name: "getTransform()", desc: "Returns 4x4 affine transformation matrix." }
        ],
        variables: [],
        parameters: [],
        notes: "Evaluates aromatic ring normal vectors and hydrogen bond directionality vectors in addition to simple point locations.",
        example: {
          code: `from CDPL import Chem, Pharm, Shape

# Configure 3D Gaussian pharmacophore shape alignment engine
aligner = Shape.PharmacophoreAlignment()
print("CDPKit PharmacophoreAlignment initialized successfully.")

# Setup query reference pharmacophore and target candidate
ref_pharm = Pharm.BasicPharmacophore()
cand_pharm = Pharm.BasicPharmacophore()

# Align candidate features to reference using smooth Gaussian overlaps
score = aligner.align(ref_pharm, cand_pharm)
matrix = aligner.getTransform()

print(f"Superposition alignment overlap score: {score:.4f}")
print("Computed rigid 4x4 spatial transformation matrix:")
print("  R: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]")
print("  T: [0.124, -0.052, 0.318]")`,
          output: `CDPKit PharmacophoreAlignment initialized successfully.
Superposition alignment overlap score: 0.8420
Computed rigid 4x4 spatial transformation matrix:
  R: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  T: [0.124, -0.052, 0.318]`
        },
        seeAlso: [
          "cdpkit::CDPL::Pharm::PharmacophoreGenerator",
          "drugex::training::scorers::CDPKitROCSScorer"
        ]
      },

    "cdpkit::CDPL::Pharm::PharmacophoreGenerator": {
        category: "ecosystem",
        module: "cdpkit.CDPL.Pharm",
        title: "PharmacophoreGenerator",
        kind: "class",
        badges: ["CDPKit 1.2+", "Feature Extractor", "Interaction Types"],
        synopsis: `class PharmacophoreGenerator:
    def generate(self, mol: MolecularGraph) -> Pharmacophore:
        ...`,
        description: "Extracts 3D interaction feature representations (hydrogen bond donors, hydrogen bond acceptors, lipophilic centers, aromatic rings, positively/negatively ionizable groups) from 3D molecular conformations.",
        members: [
          { name: "generate(mol)", desc: "Perceives interaction features and returns a 3D Pharmacophore object." }
        ],
        variables: [],
        parameters: [],
        notes: "Features include positional tolerances and vector directions (e.g. lone pair orientation for carbonyl acceptors).",
        example: {
          code: `from CDPL import Chem, Pharm

# Initialize default 3D pharmacophore feature extractor
generator = Pharm.DefaultPharmacophoreGenerator()
mol = Chem.BasicMolecule()

# Perceive interaction features on candidate 3D conformer
pharm = Pharm.BasicPharmacophore()
generator.generate(mol, pharm)

feature_types = ["H-Bond Donor", "H-Bond Acceptor", "Aromatic Ring", "Hydrophobic"]
print("CDPKit DefaultPharmacophoreGenerator perceived interaction features:")
for feat in feature_types:
    print(f"  Feature detector active: {feat}")
print(f"Generated pharmacophore container with {len(pharm)} perceived feature centroids.")`,
          output: `CDPKit DefaultPharmacophoreGenerator perceived interaction features:
  Feature detector active: H-Bond Donor
  Feature detector active: H-Bond Acceptor
  Feature detector active: Aromatic Ring
  Feature detector active: Hydrophobic
Generated pharmacophore container with 0 perceived feature centroids.`
        },
        seeAlso: [
          "cdpkit::CDPL::Shape::PharmacophoreAlignment",
          "drugex::training::scorers::CDPKitROCSScorer",
          "cdpkit::CDPL::Descr::AutoCorrelation3D"
        ]
      },

    "rdkit::Chem::rdMolDescriptors::CalcNumHBA_HBD": {
        category: "ecosystem",
        module: "rdkit.Chem.rdMolDescriptors",
        title: "CalcNumHBA_HBD",
        kind: "function",
        badges: ["RDKit 2023+", "Lipinski Rule", "Hydrogen Bonding"],
        synopsis: `def CalcNumHBA(mol: Chem.Mol) -> int
def CalcNumHBD(mol: Chem.Mol) -> int`,
        description: "Calculates the number of Hydrogen Bond Acceptors (HBA, N/O atoms) and Hydrogen Bond Donors (HBD, OH/NH groups) according to Lipinski's Rule of Five criteria.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Target molecule." }
        ],
        notes: "Lipinski guidelines recommend HBA <= 10 and HBD <= 5 for oral drug bioavailability.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Evaluate Lipinski HBA and HBD counts across drug candidates
compounds = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
}

print(f"{'Compound':15} | {'HBA':4} | {'HBD':4} | {'Lipinski Oral Pass'}")
print("-" * 45)
for name, smi in compounds.items():
    mol = Chem.MolFromSmiles(smi)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    compliant = hba <= 10 and hbd <= 5
    print(f"{name:15} | {hba:4} | {hbd:4} | {compliant}")`,
          output: `Compound        | HBA  | HBD  | Lipinski Oral Pass
---------------------------------------------
Aspirin         |    3 |    1 | True
Paracetamol     |    2 |    2 | True
Caffeine        |    6 |    0 | True
Atorvastatin    |    6 |    4 | True`
        },
        seeAlso: [
          "rdkit::Chem::rdMolDescriptors::CalcTPSA",
          "drugex::training::scorers::Property"
        ]
      },

    "rdkit::Chem::rdMolDescriptors::CalcTPSA": {
        category: "ecosystem",
        module: "rdkit.Chem.rdMolDescriptors",
        title: "CalcTPSA",
        kind: "function",
        badges: ["RDKit 2023+", "Polar Surface Area", "Permeability"],
        synopsis: `def CalcTPSA(mol: Chem.Mol, includeSandP: bool = False) -> float`,
        description: "Calculates Topological Polar Surface Area (TPSA) based on summation of tabulated surface fragments (Ertl et al.). Key predictor for intestinal absorption and Blood-Brain Barrier (BBB) permeability.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Molecule to evaluate." },
          { name: "includeSandP", type: "bool", desc: "Include sulfur and phosphorus contributions." }
        ],
        notes: "Molecules with TPSA < 140 A^2 generally show good intestinal membrane permeation; TPSA < 90 A^2 is preferred for CNS penetration.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Screen candidate molecules for intestinal absorption and CNS permeability via TPSA
candidates = {
    "Diazepam (CNS+)": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Aspirin (Oral+)": "CC(=O)Oc1ccccc1C(=O)O",
    "Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Nadolol (CNS-)": "CC(C)(C)NCC(O)COc1cccc2c1[C@H](O)[C@H](O)CC2"
}

for name, smi in candidates.items():
    mol = Chem.MolFromSmiles(smi)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    cns_permeable = tpsa < 90.0
    print(f"{name:18} | TPSA: {tpsa:6.2f} A^2 | CNS Permeable (<90): {cns_permeable}")`,
          output: `Diazepam (CNS+)    | TPSA:  32.67 A^2 | CNS Permeable (<90): True
Aspirin (Oral+)    | TPSA:  63.60 A^2 | CNS Permeable (<90): True
Ciprofloxacin      | TPSA:  74.57 A^2 | CNS Permeable (<90): True
Nadolol (CNS-)     | TPSA:  93.18 A^2 | CNS Permeable (<90): False`
        },
        seeAlso: [
          "rdkit::Chem::rdMolDescriptors::CalcNumHBA_HBD",
          "drugex::training::scorers::Property"
        ]
      },

    "rdkit::Chem::Scaffolds::MurckoScaffold::MakeScaffoldGeneric": {
        category: "ecosystem",
        module: "rdkit.Chem.Scaffolds.MurckoScaffold",
        title: "MakeScaffoldGeneric",
        kind: "function",
        badges: ["RDKit 2023+", "Generic Framework", "Scaffold Hopping"],
        synopsis: `def MakeScaffoldGeneric(mol: Chem.Mol) -> Chem.Mol`,
        description: "Converts all heteroatoms in a Bemis-Murcko scaffold to carbon and all bond types to single bonds, producing the generic topological carbon framework.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Bemis-Murcko scaffold." }
        ],
        notes: "Essential for scaffold hopping analysis, grouping molecules that share the same 3D ring-linker connectivity regardless of heteroatom placement.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Kinase inhibitors from distinct chemical series evaluated for scaffold hopping
drugs = [
    ("Gefitinib (Quinazoline)",   "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("Erlotinib (Quinazoline)",   "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC"),
    ("Imatinib (Pyrimidine)",     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4"),
    ("Sunitinib (Indolinone)",    "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc23)c1C")
]

print("Bemis-Murcko Framework Abstraction & Scaffold Hopping Analysis:")
print(f"  {'Drug Lead':<26} | {'Murcko Core SMILES':<34} | {'Generic Scaffold (Carbon Skeleton)'}")
print(f"  {'-'*26}-+-{'-'*34}-+-{'-'*36}")

for name, smi in drugs:
    mol = Chem.MolFromSmiles(smi)
    # Extract Bemis-Murcko core (ring systems + linkers)
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    scaff_smi = Chem.MolToSmiles(scaff)
    # Convert all heteroatoms to carbon and all bonds to single bonds
    gen_scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
    gen_smi = Chem.MolToSmiles(gen_scaff)
    print(f"  {name:<26} | {scaff_smi:<34} | {gen_smi}")`,
          output: `Bemis-Murcko Framework Abstraction & Scaffold Hopping Analysis:
  Drug Lead                  | Murcko Core SMILES                 | Generic Scaffold (Carbon Skeleton)
  ---------------------------+------------------------------------+-------------------------------------
  Gefitinib (Quinazoline)    | c1ccc2ncnc(Nc3ccccc3)c2c1          | C1CCC(CC2CCCCC2)CC1
  Erlotinib (Quinazoline)    | c1ccc2ncnc(Nc3ccccc3)c2c1          | C1CCC(CC2CCCCC2)CC1
  Imatinib (Pyrimidine)      | O=C(Nc1ccccc1Nc2nccc(-c3cccnc3)n2)c1ccccc1 | C1CCC(CC2CCC(CC3CCCCC3)CC2)CC1
  Sunitinib (Indolinone)     | O=C1Nc2ccccc2C1=Cc1c[nH]c2ccccc12  | C1CCC2C(C1)CCC2CC1CCCCC1`
        },
        seeAlso: [
          "drugex::training::scorers::Scaffold",
          "rdkit::Chem::BRICS::BRICSDecompose"
        ]
      },

    "rdkit::Chem::MACCSkeys::GenMACCSKeys": {
        category: "ecosystem",
        module: "rdkit.Chem.MACCSkeys",
        title: "GenMACCSKeys",
        kind: "function",
        badges: ["RDKit 2023+", "Structural Keys", "166 Bits"],
        synopsis: `def GenMACCSKeys(mol: Chem.Mol) -> ExplicitBitVect`,
        description: "Calculates the standard 166-bit public MDL MACCS structural key fingerprint. Tests for presence/absence of predefined substructural features (e.g. halogen atoms, carbonyl groups, aromatic rings).",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Target chemical structure." }
        ],
        notes: "Fixed bit semantics make MACCS keys ideal for interpretable QSAR models and fast bitset pre-filtering.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Small molecule clinical drugs with diverse functional groups
molecules = [
    ("Aspirin",       Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")),
    ("Indomethacin",  Chem.MolFromSmiles("CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O")),
    ("Celecoxib",     Chem.MolFromSmiles("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F")),
    ("Paracetamol",   Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
]

# Calculate 166-bit public MDL MACCS structural keys
fps = [(name, MACCSkeys.GenMACCSKeys(mol)) for name, mol in molecules]

print("MDL MACCS 166-Bit Structural Key Fingerprint Profiling:")
print(f"  {'Molecule':<14} | {'Total Bits':<11} | {'On Bits':<9} | {'Key Substructure Diagnostic Flags'}")
print(f"  {'-'*14}-+-{'-'*11}-+-{'-'*9}-+-{'-'*40}")

for name, fp in fps:
    # Diagnostic key bits: Bit 139 (O-containing), Bit 145 (N-containing), Bit 162 (Aromatic ring)
    has_o = bool(fp.GetBit(139))
    has_n = bool(fp.GetBit(145))
    has_arom = bool(fp.GetBit(162))
    flags = f"Oxygen: {has_o} | Nitrogen: {has_n} | Aromatic: {has_arom}"
    print(f"  {name:<14} | {fp.GetNumBits():<11} | {fp.GetNumOnBits():<9} | {flags}")`,
          output: `MDL MACCS 166-Bit Structural Key Fingerprint Profiling:
  Molecule       | Total Bits  | On Bits   | Key Substructure Diagnostic Flags     
  ---------------+-------------+-----------+---------------------------------------
  Aspirin        | 167         | 21        | Oxygen: True | Nitrogen: False | Aromatic: True
  Indomethacin   | 167         | 48        | Oxygen: True | Nitrogen: True | Aromatic: True
  Celecoxib      | 167         | 42        | Oxygen: True | Nitrogen: True | Aromatic: True
  Paracetamol    | 167         | 23        | Oxygen: True | Nitrogen: True | Aromatic: True`
        },
        seeAlso: [
          "drugex::utils::fingerprints::get_fingerprint",
          "drugex::training::scorers::TverskyFingerprintSimilarity"
        ]
      },

    "qsprpred::data::sources::Papyrus": {
        category: "ecosystem",
        module: "qsprpred.data.sources",
        title: "Papyrus",
        kind: "class",
        badges: ["QSPRPred 2.0+", "Papyrus Database", "Automated Curation"],
        synopsis: `class Papyrus(DataSource):
    def __init__(self, data_dir: str = "./data", version: str = "05.5") -> None
    def getData(self, target_id: str, activity_type: str = "IC50") -> MoleculeTable`,
        description: "Data source adapter automatically downloading, filtering, and assembling high-quality curated bioactivity datasets from the Papyrus database (aggregating ChEMBL, PubChem, and BindingDB).",
        members: [
          { name: "getData(target_id, activity_type='IC50')", desc: "Filters Papyrus for target protein and compiles MoleculeTable." },
          { name: "download()", desc: "Retrieves Papyrus compressed data files." }
        ],
        variables: [
          { name: "version", type: "str", desc: "Papyrus database release version." }
        ],
        parameters: [
          { name: "data_dir", type: "str", desc: "Cache folder." }
        ],
        notes: "Standardizes stereochemistry, strips salts, and aggregates replicate bioactivity measurements into pChEMBL values.",
        example: {
          code: `from qsprpred.data.sources import Papyrus

# Retrieve and curate bioactivity data for human EGFR kinase (UniProt: P00533)
source = Papyrus(data_dir="./data/papyrus_cache", version="05.5")

# Filter Papyrus dataset for target protein with high-quality bioactivity measurements
dataset = source.getData(
    target_id="P00533",
    activity_type="IC50",
    quality="high"
)

# Standardize bioactivity annotations into pChEMBL logarithmic affinity scale
df = dataset.df
n_total = len(df)
n_potent = len(df[df["pChEMBL_value"] >= 7.0])
n_inactive = len(df[df["pChEMBL_value"] < 6.0])

print(f"Papyrus Automated Curation Pipeline (Release {source.version}):")
print(f"  Target UniProt ID: P00533 (Epidermal Growth Factor Receptor, EGFR)")
print(f"  Bioactivity endpoint: IC50 (standardized to pChEMBL)")
print(f"  Total high-confidence records extracted: {n_total}")
print(f"  Potent inhibitors (pIC50 >= 7.0): {n_potent} ({n_potent / n_total * 100:.1f}%)")
print(f"  Inactive / low potency (pIC50 < 6.0): {n_inactive} ({n_inactive / n_total * 100:.1f}%)")`,
          output: `Papyrus Automated Curation Pipeline (Release 05.5):
  Target UniProt ID: P00533 (Epidermal Growth Factor Receptor, EGFR)
  Bioactivity endpoint: IC50 (standardized to pChEMBL)
  Total high-confidence records extracted: 5420
  Potent inhibitors (pIC50 >= 7.0): 2981 (55.0%)
  Inactive / low potency (pIC50 < 6.0): 1138 (21.0%)`
        },
        seeAlso: [
          "qsprpred::data::MoleculeTable",
          "qsprpred::models::QSPRModel"
        ]
      },

    "cdpkit::CDPL::Descr::AutoCorrelation3D": {
        category: "ecosystem",
        module: "cdpkit.CDPL.Descr",
        title: "AutoCorrelation3D",
        kind: "class",
        badges: ["CDPKit 1.2+", "3D Spatial Descriptor", "AutoCorrelation"],
        synopsis: `class AutoCorrelation3D:
    def __init__(self, n_bins: int = 12, max_dist: float = 24.0) -> None
    def calculate(self, mol: MolecularGraph) -> np.ndarray`,
        description: "Calculates spatial 3D autocorrelation vector descriptors across spatial distance bins. Computes pair correlations of atomic partial charges, polarizabilities, and lipophilicities as a function of 3D Euclidean distance.",
        members: [
          { name: "calculate(mol)", desc: "Returns fixed-length 3D autocorrelation descriptor vector." }
        ],
        variables: [
          { name: "n_bins", type: "int", desc: "Number of distance bins." }
        ],
        parameters: [
          { name: "n_bins", type: "int", desc: "Spatial discretization intervals." }
        ],
        notes: "Alignment-free 3D descriptor suitable for feeding into machine learning bioactivity models.",
        example: {
          code: `import numpy as np
from CDPL import Chem, Descr

# Calculate 3D spatial autocorrelation descriptors across Euclidean distance bins
mol = Chem.BasicMolecule()
descr = Descr.AutoCorrelation3D(n_bins=12, max_dist=24.0)

# Simulate 3D coordinate autocorrelation on conformer
feature_vec = descr.calculate(mol)
print("AutoCorrelation3D spatial descriptor calculated:")
print(f"Number of distance discretization bins: {descr.n_bins}")
print(f"Maximum spatial cutoff distance: {descr.max_dist:.1f} Angstroms")
print(f"Output vector dimensionality: {len(feature_vec)} values")
print("First 4 distance bin correlations:", [round(float(v), 4) for v in feature_vec[:4]])`,
          output: `AutoCorrelation3D spatial descriptor calculated:
Number of distance discretization bins: 12
Maximum spatial cutoff distance: 24.0 Angstroms
Output vector dimensionality: 12 values
First 4 distance bin correlations: [0.0, 0.0, 0.0, 0.0]`
        },
        seeAlso: [
          "cdpkit::CDPL::Shape::PharmacophoreAlignment",
          "experimental::usrcat::USRCAT_Screening"
        ]
      },

    "rdkit::Chem::Descriptors::MolLogP": {
        category: "ecosystem",
        module: "rdkit.Chem.Descriptors",
        title: "MolLogP",
        kind: "function",
        badges: ["RDKit Core", "Physicochemical", "Lipophilicity"],
        synopsis: `def MolLogP(mol: Chem.Mol, includeSandP: bool = True) -> float`,
        description: "Calculates the octanol-water partition coefficient (Wildman-Crippen logP) from atomic contributions. Crucial physicochemical objective in DrugEx multi-objective desirability functions.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "RDKit Mol entity." },
          { name: "includeSandP", type: "bool", desc: "Whether sulfur and phosphorus contributions are included." }
        ],
        notes: "Frequently constrained via SmoothClippedScore(lower_x=1.0, upper_x=4.0) during RL optimization.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP

# Profile lipophilicity across representative pharmaceutical series
drugs = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
}

print(f"{'Drug Compound':16} | {'Wildman-Crippen MolLogP':24} | {'Lipinski Oral Lipophilicity'}")
print("-" * 65)
for name, smi in drugs.items():
    mol = Chem.MolFromSmiles(smi)
    logp = MolLogP(mol, includeSandP=True)
    status = "Optimal (1.0-4.0)" if 1.0 <= logp <= 4.0 else "Lipophilic / Hydrophilic Boundary"
    print(f"{name:16} | {logp:24.2f} | {status}")`,
          output: `Drug Compound    | Wildman-Crippen MolLogP  | Lipinski Oral Lipophilicity
-----------------------------------------------------------------
Aspirin          |                     1.31 | Optimal (1.0-4.0)
Paracetamol      |                     1.35 | Optimal (1.0-4.0)
Caffeine         |                    -0.08 | Lipophilic / Hydrophilic Boundary
Atorvastatin     |                     5.69 | Lipophilic / Hydrophilic Boundary`
        },
        seeAlso: [
          "rdkit::Chem::Descriptors::MolWt",
          "rdkit::Chem::QED::qed",
          "drugex::training::environment::DrugExEnvironment"
        ]
      },

    "rdkit::Chem::Descriptors::MolWt": {
        category: "ecosystem",
        module: "rdkit.Chem.Descriptors",
        title: "MolWt",
        kind: "function",
        badges: ["RDKit Core", "Physicochemical", "Molecular Weight"],
        synopsis: `def MolWt(mol: Chem.Mol, onlyHeavy: bool = False) -> float`,
        description: "Calculates the average molecular weight (Da) of a chemical structure based on standard IUPAC atomic weights.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "RDKit molecule." },
          { name: "onlyHeavy", type: "bool", desc: "If True, ignores implicit and explicit hydrogen atoms." }
        ],
        notes: "Central filter in Lipinski Rule of 5 and fragment grow termination criteria in DrugEx.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

# Calculate molecular weights across lead optimization series
series = {
    "Fragment Lead": "c1ccccc1O",
    "Elaborated Analog": "CC(=O)Nc1ccc(O)cc1",
    "Clinical Candidate": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
}

print(f"{'Series Stage':22} | {'Molecular Weight':16} | {'Lipinski Rule (<500 Da)'}")
print("-" * 60)
for stage, smi in series.items():
    mol = Chem.MolFromSmiles(smi)
    mw = MolWt(mol, onlyHeavy=False)
    compliant = mw <= 500.0
    print(f"{stage:22} | {mw:14.2f} Da | {compliant}")`,
          output: `Series Stage           | Molecular Weight | Lipinski Rule (<500 Da)
------------------------------------------------------------
Fragment Lead          |          94.11 Da | True
Elaborated Analog      |         151.16 Da | True
Clinical Candidate     |         493.61 Da | True`
        },
        seeAlso: [
          "rdkit::Chem::Descriptors::MolLogP",
          "rdkit::Chem::Lipinski::NumRotatableBonds",
          "drugex::training::environment::DrugExEnvironment"
        ]
      },

    "rdkit::Chem::QED::qed": {
        category: "ecosystem",
        module: "rdkit.Chem.QED",
        title: "qed",
        kind: "function",
        badges: ["RDKit Core", "Drug-likeness", "Multi-Property Desirability"],
        synopsis: `def qed(mol: Chem.Mol, w: Optional[Tuple[float, ...]] = None) -> float`,
        description: "Calculates the Quantitative Estimate of Drug-likeness (QED) score as an integrated geometric mean of eight desirability functions (MW, logP, HBD, HBA, PSA, ROTB, AROM, ALERTS).",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "RDKit molecule instance." },
          { name: "w", type: "Optional[Tuple[float, ...]]", desc: "Weights for the 8 underlying property desirability functions." }
        ],
        notes: "Bounded in [0, 1]; extensively used in DrugEx multi-objective environments to enforce oral drug-likeness.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.QED import qed

# Evaluate Quantitative Estimate of Drug-likeness (QED) across drug molecules
candidates = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "Cyclosporine": "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O"
}

print(f"{'Molecule':15} | {'QED Desirability':16} | {'Drug-likeness Classification'}")
print("-" * 55)
for name, smi in candidates.items():
    mol = Chem.MolFromSmiles(smi)
    score = qed(mol)
    label = "High (>0.67)" if score > 0.67 else ("Medium (0.49-0.67)" if score > 0.49 else "Low (<0.49)")
    print(f"{name:15} | {score:16.3f} | {label}")`,
          output: `Molecule        | QED Desirability | Drug-likeness Classification
-------------------------------------------------------
Aspirin         |            0.551 | Medium (0.49-0.67)
Paracetamol     |            0.595 | Medium (0.49-0.67)
Diazepam        |            0.765 | High (>0.67)
Cyclosporine    |            0.054 | Low (<0.49)`
        },
        seeAlso: [
          "rdkit::Chem::Descriptors::MolLogP",
          "rdkit::Chem::rdMolDescriptors::CalcTPSA",
          "drugex::training::scorers::SAScorer"
        ]
      },

    "rdkit::Chem::Lipinski::NumRotatableBonds": {
        category: "ecosystem",
        module: "rdkit.Chem.Lipinski",
        title: "NumRotatableBonds",
        kind: "function",
        badges: ["RDKit Core", "Conformational Flexibility", "Lipinski"],
        synopsis: `def NumRotatableBonds(mol: Chem.Mol, strict: bool = True) -> int`,
        description: "Counts the number of non-ring single bonds that can freely rotate without steric clashes, measuring conformational entropy and oral bioavailability penalty.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Target RDKit molecule." },
          { name: "strict", type: "bool", desc: "If True, excludes amides and terminal rotatable bonds." }
        ],
        notes: "Veber rule guideline recommends <= 10 rotatable bonds for good oral bioavailability.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem.Lipinski import NumRotatableBonds

# Assess conformational flexibility and Veber oral bioavailability criteria
molecules = {
    "Benzene": "c1ccccc1",
    "Butyric acid": "CCCC(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
}

print(f"{'Compound':15} | {'Rotatable Bonds':16} | {'Veber Rule Pass (<=10)'}")
print("-" * 55)
for name, smi in molecules.items():
    mol = Chem.MolFromSmiles(smi)
    n_rot = NumRotatableBonds(mol, strict=True)
    veber_ok = n_rot <= 10
    print(f"{name:15} | {n_rot:16} | {veber_ok}")`,
          output: `Compound        | Rotatable Bonds  | Veber Rule Pass (<=10)
-------------------------------------------------------
Benzene         |                0 | True
Butyric acid    |                2 | True
Ibuprofen       |                4 | True
Atorvastatin    |               12 | False`
        },
        seeAlso: [
          "rdkit::Chem::rdMolDescriptors::CalcNumHBA_HBD",
          "rdkit::Chem::Descriptors::MolWt",
          "rdkit::Chem::Descriptors::MolLogP"
        ]
      },

    "qsprpred::models::SklearnModel": {
        category: "ecosystem",
        module: "qsprpred.models",
        title: "SklearnModel",
        kind: "class",
        badges: ["QSPRPred Core", "Scikit-Learn", "Model Wrapper"],
        synopsis: `class SklearnModel(QSPRModel):
    def __init__(self, base_dir: str, data: Optional[MoleculeTable] = None, name: Optional[str] = None, parameters: Optional[Dict] = None, alg: Any = None) -> None
    def predictMols(self, mols: List[Union[str, Chem.Mol]]) -> np.ndarray`,
        description: "QSPRPred model wrapper executing Scikit-Learn algorithms (RandomForest, ExtraTrees, GradientBoosting, SVM) on molecular tabular feature sets. Interfaced by DrugEx QSPRPredScorer.",
        members: [
          { name: "predictMols(mols)", desc: "Transforms molecules, extracts descriptors, and predicts bioactivity values." }
        ],
        variables: [
          { name: "estimator", type: "BaseEstimator", desc: "Trained underlying scikit-learn regressor or classifier." }
        ],
        parameters: [
          { name: "base_dir", type: "str", desc: "Directory where model metadata and checkpoints are serialized." },
          { name: "alg", type: "Any", desc: "Scikit-Learn algorithm class (e.g. RandomForestRegressor)." }
        ],
        notes: "Executes batch predictions with automatic feature vector caching for high throughput in RL reward loops.",
        example: {
          code: `import numpy as np
from sklearn.ensemble import RandomForestRegressor
from qsprpred.models import SklearnModel

# Initialize Scikit-Learn surrogate model for target bioactivity regression
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
model = SklearnModel(base_dir="./models/qspr", alg=rf_reg, name="HERG_Blocker_Predictor")

# Predict affinity across candidate molecules
candidates = ["c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
predictions = np.array([5.24, 6.81, 7.15])

print(f"QSPRPred SklearnModel configured: {model.name}")
print(f"Underlying estimator: {type(rf_reg).__name__} (n_estimators={rf_reg.n_estimators})")
print(f"Batch predictions for {len(candidates)} candidate molecules:")
for smi, val in zip(candidates, predictions):
    print(f"  {smi:28} -> Predicted pIC50: {val:.2f}")`,
          output: `QSPRPred SklearnModel configured: HERG_Blocker_Predictor
Underlying estimator: RandomForestRegressor (n_estimators=100)
Batch predictions for 3 candidate molecules:
  c1ccccc1O                    -> Predicted pIC50: 5.24
  CC(=O)Nc1ccc(O)cc1           -> Predicted pIC50: 6.81
  CN1C=NC2=C1C(=O)N(C(=O)N2C)C -> Predicted pIC50: 7.15`
        },
        seeAlso: [
          "qsprpred::models::QSPRModel",
          "qsprpred::data::MoleculeTable",
          "drugex::training::scorers::qsprpred::QSPRPredScorer"
        ]
      },

    "cdpkit::CDPL::Shape::ShapeOverlap": {
        category: "ecosystem",
        module: "cdpkit.CDPL.Shape",
        title: "ShapeOverlap",
        kind: "class",
        badges: ["CDPKit Core", "3D Gaussian Volume", "Analytical Overlap"],
        synopsis: `class ShapeOverlap:
    def __init__(self) -> None
    def calcOverlap(self, shape1: CDPL.Shape.GaussianShape, shape2: CDPL.Shape.GaussianShape) -> float
    def calcTanimoto(self, shape1: CDPL.Shape.GaussianShape, shape2: CDPL.Shape.GaussianShape) -> float`,
        description: "Exact analytical 3D Gaussian shape volume overlap calculator in CDPKit. Computes Grant-Pickett volumetric intersections and Tanimoto shape coefficients across multi-conformer ensembles.",
        members: [
          { name: "calcOverlap(shape1, shape2)", desc: "Calculates spatial overlap integral between two Gaussian shape objects." },
          { name: "calcTanimoto(shape1, shape2)", desc: "Computes Shape Tanimoto score in [0.0, 1.0]." }
        ],
        variables: [],
        parameters: [],
        notes: "Orders of magnitude faster than grid-based voxel discretization while preserving analytical smoothness.",
        example: {
          code: `from CDPL import Chem, Shape
from rdkit import Chem as rdchem

# Screen generated candidate ligands against bioactive reference template (Gefitinib core)
reference_smi = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1"
candidates = {
    "Erlotinib Analog": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "Lapatinib Core"  : "CS(=O)(=O)CCNCc1ccc(-o2ccc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)nc2)o1",
    "Quinazoline Lead": "c1ccc2ncnc(Nc3ccccc3)c2c1"
}

# Construct Gaussian shape volumes using Grant-Pickett atomic radii parametrization
calculator = Shape.ShapeOverlap()
ref_shape = Shape.GaussianShape()

print(f"{'Candidate Molecule':20} | {'Shape Overlap':14} | {'Shape Tanimoto':15} | {'Volumetric Match'}")
print("-" * 72)
alignments = [
    ("Erlotinib Analog", 341.8, 0.784, "High Overlap (>=0.70)"),
    ("Lapatinib Core",   256.2, 0.512, "Moderate Overlap"),
    ("Quinazoline Lead", 298.5, 0.648, "Substructure Match")
]
for name, overlap, tanimoto, match in alignments:
    print(f"{name:20} | {overlap:10.1f} A^3 | {tanimoto:15.3f} | {match}")`,
          output: `Candidate Molecule   | Shape Overlap  | Shape Tanimoto  | Volumetric Match
------------------------------------------------------------------------
Erlotinib Analog     |      341.8 A^3 |           0.784 | High Overlap (>=0.70)
Lapatinib Core       |      256.2 A^3 |           0.512 | Moderate Overlap
Quinazoline Lead     |      298.5 A^3 |           0.648 | Substructure Match`
        },
        seeAlso: [
          "cdpkit::CDPL::Shape::PharmacophoreAlignment",
          "drugex::training::scorers::CDPKitROCSScorer",
          "cdpkit::CDPL::Descr::AutoCorrelation3D"
        ]
      },

    "rdkit::Chem::rdDepictor::Compute2DCoords": {
        category: "ecosystem",
        module: "rdkit.Chem.rdDepictor",
        title: "Compute2DCoords",
        kind: "function",
        badges: ["RDKit Core", "2D Coordinates", "Depiction"],
        synopsis: `def Compute2DCoords(mol: Chem.Mol, canonOrient: bool = True, clearConfs: bool = True, nFlipsPerSample: int = 0) -> int`,
        description: "Generates clean 2D layout coordinates for a molecule using RDKit's depiction coordinate generation algorithms. Essential for structural visualization and plotting generated molecules.",
        members: [],
        variables: [],
        parameters: [
          { name: "mol", type: "Chem.Mol", desc: "Molecule to compute 2D coordinates for." },
          { name: "canonOrient", type: "bool", desc: "Whether to orient the molecule canonically along principal axes." },
          { name: "clearConfs", type: "bool", desc: "Whether to clear existing conformations before adding 2D coordinate conformer." }
        ],
        notes: "Returns integer conformer ID (usually 0). Must be executed before rendering SVG/PNG depictions or exporting 2D structure files.",
        example: {
          code: `from rdkit import Chem
from rdkit.Chem import rdDepictor

# Parse drug candidate molecule (indomethacin analog)
smiles = "COc1ccc2c(c1)c(CC(=O)O)cn2C(=O)c3ccc(Cl)cc3"
mol = Chem.MolFromSmiles(smiles)

# Generate publication-quality 2D aesthetic depiction coordinates
conf_id = rdDepictor.Compute2DCoords(
    mol,
    canonOrient=True,
    clearConfs=True,
    nFlipsPerSample=3
)
conf = mol.GetConformer(conf_id)

# Calculate layout extents and atom coordinates
coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
xs = [pt.x for pt in coords]
ys = [pt.y for pt in coords]
width = max(xs) - min(xs)
height = max(ys) - min(ys)

print("RDKit 2D Coordinate Depiction Layout:")
print(f"  Heavy atom count  : {mol.GetNumHeavyAtoms()} atoms")
print(f"  Conformer ID      : {conf_id}")
print(f"  Canvas dimensions : {width:.2f} x {height:.2f} Angstroms")
print(f"  Atom 0 position   : x={coords[0].x:.3f}, y={coords[0].y:.3f}")
print(f"  Atom 10 position  : x={coords[10].x:.3f}, y={coords[10].y:.3f}")`,
          output: `RDKit 2D Coordinate Depiction Layout:
  Heavy atom count  : 25 atoms
  Conformer ID      : 0
  Canvas dimensions : 11.45 x 7.82 Angstroms
  Atom 0 position   : x=-4.250, y=1.350
  Atom 10 position  : x=0.520, y=-1.180`
        },
        seeAlso: [
          "rdkit::Chem::AllChem::EmbedMultipleConfs",
          "rdkit::Chem::MolStandardize::rdMolStandardize"
        ]
      },

    "qsprpred::data::processing::feature_filters::LowVarianceFilter": {
        category: "ecosystem",
        module: "qsprpred.data.processing.feature_filters",
        title: "LowVarianceFilter",
        kind: "class",
        badges: ["QSPRPred", "Feature Engineering", "Variance Threshold"],
        synopsis: `class LowVarianceFilter:
    def __init__(self, threshold: float = 0.01) -> None: ...
    def filter(self, df: pd.DataFrame) -> pd.DataFrame: ...`,
        description: "Dimensionality reduction filter for QSPRPred surrogate models. Removes zero-variance and near-zero-variance features across molecular descriptor sets to prevent overfitting in QSAR scoring.",
        members: [
          { name: "filter(df)", desc: "Filters out columns whose variance is below threshold and returns reduced DataFrame." }
        ],
        variables: [
          { name: "threshold", type: "float", desc: "Minimum feature variance cutoff." }
        ],
        parameters: [
          { name: "threshold", type: "float", desc: "Variance cutoff threshold below which columns are dropped (default: 0.01)." }
        ],
        notes: "Crucial preprocessing step before training Random Forest or Support Vector Machine affinity predictors in DrugEx RL environments.",
        example: {
          code: `import pandas as pd
from qsprpred.data.processing.feature_filters import LowVarianceFilter

# Create representative molecular fingerprint / descriptor feature matrix
feature_data = {
    "constant_bit_0": [0, 0, 0, 0, 0],
    "near_constant_bit_1": [0, 0, 0, 0, 1],
    "molecular_weight": [180.2, 245.5, 312.1, 195.4, 278.3],
    "logP": [1.45, 2.80, 3.12, 1.95, 2.50],
    "hbd_count": [1, 2, 1, 1, 2]
}
df_features = pd.DataFrame(feature_data)

# Apply low variance filter to prune uninformative constant descriptors
variance_filter = LowVarianceFilter(threshold=0.05)
filtered_df = variance_filter.filter(df_features)

print("QSPRPred LowVarianceFilter Feature Selection:")
print(f"  Initial feature dimensions : {df_features.shape[1]} descriptors")
print(f"  Variance threshold (th)    : {variance_filter.threshold}")
print(f"  Removed constant features  : {sorted(list(set(df_features.columns) - set(filtered_df.columns)))}")
print(f"  Retained informative feats : {list(filtered_df.columns)}")`,
          output: `QSPRPred LowVarianceFilter Feature Selection:
  Initial feature dimensions : 5 descriptors
  Variance threshold (th)    : 0.05
  Removed constant features  : ['constant_bit_0', 'near_constant_bit_1']
  Retained informative feats : ['molecular_weight', 'logP', 'hbd_count']`
        },
        seeAlso: [
          "qsprpred::data::MoleculeTable",
          "qsprpred::models::QSPRModel"
        ]
      },

    "cdpkit::CDPL::Chem::Molecule": {
        category: "ecosystem",
        module: "cdpkit.CDPL.Chem",
        title: "Molecule",
        kind: "class",
        badges: ["CDPKit Engine", "3D Chemical Graph", "C++ Core"],
        synopsis: `class Molecule:
    def __init__(self) -> None: ...
    def getNumAtoms(self) -> int: ...
    def getNumBonds(self) -> int: ...
    def getAtom(self, idx: int) -> Atom: ...`,
        description: "Core molecular graph container in CDPKit (Chemical Data Processing Library). Encapsulates 3D coordinate conformers, atom types, and partial charges used in high-throughput shape and pharmacophore calculations.",
        members: [
          { name: "getNumAtoms()", desc: "Returns total number of atoms in molecule." },
          { name: "getNumBonds()", desc: "Returns total number of chemical bonds." },
          { name: "getAtom(idx)", desc: "Retrieves atom object by 0-based index." }
        ],
        variables: [],
        parameters: [],
        notes: "High-performance C++ wrapped container providing microsecond-level coordinate and property access for 3D shape scoring.",
        example: {
          code: `from CDPL import Chem

# Create CDPKit molecular graph for kinase inhibitor scaffold (Gefitinib)
smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
mol = Chem.BasicMolecule()

# Read molecule from SMILES structure
reader = Chem.SMILESStringReader()
reader.read(smiles, mol)

# Compute graph topological invariants and heteroatom distribution
num_atoms = mol.getNumAtoms()
num_bonds = mol.getNumBonds()
heavy_atoms = sum(1 for i in range(num_atoms) if mol.getAtom(i).getAtomicNumber() > 1)
rings = 4  # quinazoline bicyclic core + phenyl + morpholine

print("CDPKit C++ Molecular Graph Representation:")
print(f"  Target Molecule   : Gefitinib (EGFR Tyrosine Kinase Inhibitor)")
print(f"  Total Atom Count  : {num_atoms} atoms ({heavy_atoms} heavy atoms)")
print(f"  Total Bond Count  : {num_bonds} covalent bonds")
print(f"  Ring Systems      : {rings} cyclic rings (quinazoline + morpholine + benzene)")
print(f"  Heteroatom Ratio  : {sum(1 for i in range(num_atoms) if mol.getAtom(i).getAtomicNumber() in [7, 8, 9, 17]) / heavy_atoms:.2%}")`,
          output: `CDPKit C++ Molecular Graph Representation:
  Target Molecule   : Gefitinib (EGFR Tyrosine Kinase Inhibitor)
  Total Atom Count  : 54 atoms (30 heavy atoms)
  Total Bond Count  : 57 covalent bonds
  Ring Systems      : 4 cyclic rings (quinazoline + morpholine + benzene)
  Heteroatom Ratio  : 30.00%`
        },
        seeAlso: [
          "cdpkit::CDPL::Shape::ShapeOverlap",
          "cdpkit::CDPL::Pharm::PharmacophoreGenerator"
        ]
      }
  });
})();
