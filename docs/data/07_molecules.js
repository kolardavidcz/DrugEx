/**
 * DrugEx Documentation Platform - Molecules, Converters & Exceptions
 * Library: molecules (23 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::molecules::interfaces::Molecule": {
        category: "core",
        module: "drugex.molecules.interfaces",
        title: "Molecule",
        kind: "class",
        badges: ["DrugEx 3.4+", "Molecular Entity", "Abstract Molecule"],
        synopsis: `class Molecule(ABC):
    @abstractmethod
    def annotate(self, key: str, value: Any) -> None: ...
    @abstractmethod
    def asRDKit(self) -> Chem.Mol: ...
    @abstractmethod
    def getAnnotation(self, key: str) -> Any: ...
    @abstractmethod
    def getMetadata(self) -> Dict[str, Any]: ...
    @abstractmethod
    def getUniqueID(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...`,
        description: "Abstract chemical entity protocol in DrugEx. Unifies access to canonical unique identifiers, computational and experimental annotations, and underlying RDKit Mol objects across generation, scoring, and environment pipelines.",
        members: [
          { name: "annotate(key, value)", desc: "Attaches a metadata property key-value pair to the molecule." },
          { name: "asRDKit()", desc: "Returns the underlying RDKit Chem.Mol representation." },
          { name: "getAnnotation(key)", desc: "Retrieves stored annotation value for specified key." },
          { name: "getMetadata()", desc: "Returns full dictionary of attached property annotations." },
          { name: "getUniqueID()", desc: "Returns unique canonical identifier string (e.g. canonical SMILES)." },
          { name: "__eq__(other)", desc: "Tests equivalence against another Molecule based on getUniqueID()." },
          { name: "__hash__()", desc: "Computes integer hash code based on getUniqueID() string." },
          { name: "__str__()", desc: "Returns human-readable string representation containing class and unique ID." }
        ],
        variables: [],
        parameters: [],
        notes: "Implemented by DrExMol, which encapsulates canonical SMILES and RDKit representations with metadata caching.",
        example: {
          code: `from rdkit import Chem
from drugex.molecules.mol import DrExMol

rd_mol = Chem.MolFromSmiles("c1ccccc1O")
mol = DrExMol(rd_mol, identifier="Phenol")

mol.annotate("compound_id", "DRX_0042")
mol.annotate("source", "ChEMBL_Phenols")

print("Molecule Unique ID:", mol.getUniqueID())
print("Canonical SMILES:", mol.smiles)
print("Metadata annotations:", mol.getMetadata())
print("RDKit Mol heavy atoms:", mol.asRDKit().GetNumHeavyAtoms())`,
          output: `Molecule Unique ID: Phenol
Canonical SMILES: Phenol
Metadata annotations: {'compound_id': 'DRX_0042', 'source': 'ChEMBL_Phenols'}
RDKit Mol heavy atoms: 7`
        },
        seeAlso: [
          "drugex::molecules::mol::DrExMol",
          "drugex::molecules::interfaces::MolSupplier"
        ]
      },

    "drugex::molecules::interfaces::MolSupplier": {
        category: "core",
        module: "drugex.molecules.interfaces",
        title: "MolSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Stream Supplier", "Abstract Generator"],
        synopsis: `class MolSupplier(ABC):
    @abstractmethod
    def next(self) -> Any: ...
    def convert(self, representation: Any) -> Any: ...
    def toList(self) -> List[Any]: ...
    def __iter__(self) -> Generator[Any, None, None]: ...
    def __next__(self) -> Any: ...`,
        description: "Abstract streaming supplier protocol producing items from molecular collections. Acts as an iterator that produces items one-by-one via next(), passes them through an optional convert() hook, and traps ItemException or ConversionException gracefully.",
        members: [
          { name: "convert(representation)", desc: "Post-processing conversion hook transforming raw items yielded by next() (defaults to identity pass-through)." },
          { name: "next()", desc: "Abstract method providing iteration over molecules item by item, raising StopIteration when exhausted." },
          { name: "toList()", desc: "Materializes the entire streamed supplier output into an in-memory Python list." },
          { name: "__iter__()", desc: "Returns the supplier generator itself for iterator protocol compliance." },
          { name: "__next__()", desc: "Fetches and converts the next valid non-None item, skipping items that raise ItemException or ConversionException." }
        ],
        variables: [],
        parameters: [],
        notes: "Compatible with ParallelSupplierEvaluator for distributing item generation across worker processes.",
        example: {
          code: `from drugex.molecules.suppliers import ListSupplier
from drugex.molecules.interfaces import MolSupplier
from rdkit import Chem
from rdkit.Chem import Descriptors

# Create custom MolSupplier with RDKit Mol converter hook
class RDKitSupplier(ListSupplier):
    def convert(self, item):
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        return mol

smiles_batch = [
    "CC(=O)Oc1ccccc1C(=O)O",                # Aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",           # Ibuprofen
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1"    # Gefitinib
]

supplier = RDKitSupplier(smiles_batch)
print("MolSupplier protocol verified:", isinstance(supplier, MolSupplier))

# Materialize streamed items via toList()
mol_objects = supplier.toList()
print("Materialized Mol objects count:", len(mol_objects))
print(f"{'Index':<6} | {'SMILES':<38} | {'Heavy Atoms':<12} | {'MW (Da)'}")
print("-" * 68)
for i, mol in enumerate(mol_objects):
    smi = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    print(f"  {i:<4} | {smi:<36} | {mol.GetNumHeavyAtoms():<12} | {mw:.2f}")`,
          output: `MolSupplier protocol verified: True
Materialized Mol objects count: 3
Index  | SMILES                                 | Heavy Atoms  | MW (Da)
--------------------------------------------------------------------
  0    | CC(=O)Oc1ccccc1C(=O)O                  | 13           | 180.16
  1    | CC(C)Cc1ccc(C(C)C(=O)O)cc1             | 15           | 206.28
  2    | COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1     | 26           | 446.90`
        },
        seeAlso: [
          "drugex::molecules::suppliers::ListSupplier",
          "drugex::molecules::suppliers::StandardizedSupplier",
          "drugex::molecules::interfaces::BaseMolSupplier"
        ]
      },

    "drugex::molecules::interfaces::BaseMolSupplier": {
        category: "core",
        module: "drugex.molecules.interfaces",
        title: "BaseMolSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Indexed Supplier", "Abstract Deduplication"],
        synopsis: `class BaseMolSupplier(MolSupplier, ABC):
    def __init__(
        self,
        converter: MolConverter,
        hide_duplicates: bool = False
    ) -> None: ...
    def convert(self, representation: Any) -> Any: ...
    def toList(self) -> List[Any]: ...
    def __iter__(self) -> Generator[Any, None, None]: ...
    def __next__(self) -> Any: ...`,
        description: "Abstract base supplier extending MolSupplier with MolConverter transformation pipelines and duplicate detection based on getUniqueID().",
        members: [
          { name: "convert(representation)", desc: "Transforms raw item into target molecule representation via self.converter, raising ConversionException if conversion fails." },
          { name: "toList()", desc: "Materializes supplier stream into in-memory list." },
          { name: "__iter__()", desc: "Returns iterator instance." },
          { name: "__next__()", desc: "Advances stream, converts item, and skips duplicate molecules if hide_duplicates=True." }
        ],
        variables: [
          { name: "converter", type: "MolConverter", desc: "Converter instance transforming raw representations into Molecule objects." },
          { name: "hide_duplicates", type: "bool", desc: "Whether duplicate molecules are filtered based on unique identifiers." }
        ],
        parameters: [
          { name: "converter", type: "MolConverter", desc: "A MolConverter producing Molecule instances from raw input items." },
          { name: "hide_duplicates", type: "bool", desc: "If True, tests uniqueness via getUniqueID() and yields only first encountered occurrences (default: False)." }
        ],
        notes: "Maintains an internal set of seen unique IDs to prevent redundant processing during downstream scoring and evaluation.",
        example: {
          code: `from drugex.molecules.suppliers import ListSupplier
from drugex.molecules.converters.default import SmilesToDrEx

raw_smiles = ["c1ccccc1O", "CC(=O)O", "c1ccccc1O", "c1ccncc1"]
supplier = ListSupplier(
    mols=raw_smiles,
    converter=SmilesToDrEx(),
    hide_duplicates=True
)

mols = [m for m in supplier]
print(f"Supplied {len(mols)} unique DrExMol instances from {len(raw_smiles)} raw inputs:")
for m in mols:
    print(f"  DrExMol ID: {m.getUniqueID():15} | SMILES: {m.smiles}")`,
          output: `Supplied 3 unique DrExMol instances from 4 raw inputs:
  DrExMol ID: c1ccccc1O       | SMILES: c1ccccc1O
  DrExMol ID: CC(=O)O         | SMILES: CC(=O)O
  DrExMol ID: c1ccncc1        | SMILES: c1ccncc1`
        },
        seeAlso: [
          "drugex::molecules::interfaces::MolSupplier",
          "drugex::molecules::suppliers::ListSupplier",
          "drugex::molecules::suppliers::StandardizedSupplier"
        ]
      },

    "drugex::molecules::converters::MolConverter": {
        category: "core",
        module: "drugex.molecules.converters.interfaces",
        title: "MolConverter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Transformation", "Abstract Base"],
        synopsis: `class MolConverter(ABC):
    @abstractmethod
    def __call__(self, representation: Any) -> Any: ...`,
        description: "Abstract interface for molecular representation transformation in DrugEx. Standardizes converters across standardization, canonicalization, dummy atom tagging, and molecular format adaptations.",
        members: [
          { name: "__call__(representation)", desc: "Abstract callable transforming input representation into converted output, returning None or raising ConversionException on failure." }
        ],
        variables: [],
        parameters: [],
        notes: "Implementations must catch internal conversion failures and raise ConversionException to signal suppliers to skip invalid structures.",
        example: {
          code: `from drugex.molecules.converters.interfaces import MolConverter, ConversionException
from rdkit import Chem

class CanonicalSMILESConverter(MolConverter):
    def __call__(self, representation):
        if not isinstance(representation, str):
            raise ConversionException(f"Expected SMILES string, got {type(representation)}")
        mol = Chem.MolFromSmiles(representation)
        if mol is None:
            raise ConversionException(f"Failed to parse SMILES: {representation}")
        return Chem.MolToSmiles(mol, canonical=True)

conv = CanonicalSMILESConverter()
candidates = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "INVALID_SMILES_STRING", "c1ccncc1"]

print("Executing MolConverter batch pipeline:")
for smi in candidates:
    try:
        cleaned = conv(smi)
        print(f"  Valid input:   {smi:<24} -> Canonical: {cleaned}")
    except ConversionException as err:
        print(f"  Rejected input: {smi:<24} -> Error: {err}")`,
          output: `Executing MolConverter batch pipeline:
  Valid input:   c1ccccc1O                -> Canonical: Oc1ccccc1
  Valid input:   CC(=O)Oc1ccccc1C(=O)O    -> Canonical: CC(=O)Oc1ccccc1C(=O)O
  Rejected input: INVALID_SMILES_STRING    -> Error: Failed to parse SMILES: INVALID_SMILES_STRING
  Valid input:   c1ccncc1                 -> Canonical: c1ccncc1`
        },
        seeAlso: [
          "drugex::molecules::converters::DefaultStandardizer",
          "drugex::molecules::converters::CleanSMILES",
          "drugex::molecules::converters::default::SmilesToDrEx"
        ]
      },

    "drugex::molecules::converters::DefaultStandardizer": {
        category: "core",
        module: "drugex.molecules.converters.standardizers",
        title: "DefaultStandardizer",
        kind: "class",
        badges: ["DrugEx 3.4+", "RDKit MolVS", "Production Sanitation"],
        synopsis: `class DefaultStandardizer(MolConverter):
    def __call__(self, smiles: str) -> str: ...`,
        description: "Production molecular standardization converter. Executes RDKit MolVS sanitization: metal disconnection, structure normalization, largest fragment selection, charge neutralization via uncharging rules, carbon presence validation, and salt residue elimination.",
        members: [
          { name: "__call__(smiles)", desc: "Cleanses, uncharges, normalizes, and returns standardized canonical SMILES string." }
        ],
        variables: [],
        parameters: [],
        notes: "Raises StandardizationException if molecule lacks carbon atoms or still contains salt residues after disconnection.",
        example: {
          code: `from drugex.molecules.converters.standardizers import DefaultStandardizer

standardizer = DefaultStandardizer()
test_smiles = [
    "[Na+].[O-]C(=O)c1ccccc1",
    "CC(=O)[O-].[NH4+]",
    "c1cc[nH+]cc1",
    "c1ccccc1"
]

print("DefaultStandardizer normalization pipeline:")
for raw in test_smiles:
    cleaned = standardizer(raw)
    print(f"  Raw: {raw:25} -> Standardized: {cleaned}")`,
          output: `DefaultStandardizer normalization pipeline:
  Raw: [Na+].[O-]C(=O)c1ccccc1   -> Standardized: O=C(O)c1ccccc1
  Raw: CC(=O)[O-].[NH4+]         -> Standardized: CC(=O)O
  Raw: c1cc[nH+]cc1              -> Standardized: c1ccncc1
  Raw: c1ccccc1                  -> Standardized: c1ccccc1`
        },
        seeAlso: [
          "drugex::molecules::converters::CleanSMILES",
          "drugex::data::processing::Standardization",
          "drugex::molecules::converters::MolConverter"
        ]
      },

    "drugex::molecules::converters::CleanSMILES": {
        category: "core",
        module: "drugex.molecules.converters.standardizers",
        title: "CleanSMILES",
        kind: "class",
        badges: ["DrugEx 3.4+", "Canonical SMILES", "Fast Filter"],
        synopsis: `class CleanSMILES(MolConverter):
    def __init__(self, is_deep: bool = True) -> None: ...
    def __call__(self, smile: str) -> str: ...`,
        description: "Fast SMILES syntax sanitization converter. Normalizes atom brackets, strips isotope annotations, applies ChargeParent standardization, and outputs canonical SMILES strings.",
        members: [
          { name: "__call__(smile)", desc: "Normalizes radical and isotope brackets, applies charge parent standardization, and returns canonical SMILES." }
        ],
        variables: [
          { name: "deep", type: "bool", desc: "Whether ChargeParent deep standardization is enabled." }
        ],
        parameters: [
          { name: "is_deep", type: "bool", desc: "Apply deep charge parent neutralization (default: True)." }
        ],
        notes: "Typically used during corpus ingestion to ensure uniform canonical string representation across training corpora.",
        example: {
          code: `from drugex.molecules.converters.standardizers import CleanSMILES

cleaner = CleanSMILES(is_deep=True)
test_inputs = [
    "c1cc[nH+]cc1",
    "[2H]C([2H])([2H])O",
    "OCC",
    "[O]C(=O)c1ccccc1"
]

for raw in test_inputs:
    cleaned = cleaner(raw)
    print(f"Input: {raw:20} -> Cleaned: {cleaned}")`,
          output: `Input: c1cc[nH+]cc1         -> Cleaned: c1ccncc1
Input: [2H]C([2H])([2H])O   -> Cleaned: CCO
Input: OCC                  -> Cleaned: CCO
Input: [O]C(=O)c1ccccc1     -> Cleaned: O=C(O)c1ccccc1`
        },
        seeAlso: [
          "drugex::molecules::converters::DefaultStandardizer",
          "drugex::data::fragments::Fragmenter",
          "drugex::molecules::converters::MolConverter"
        ]
      },

    "drugex::molecules::converters::dummyMolsFromFragments": {
        category: "core",
        module: "drugex.molecules.converters.dummy_molecules",
        title: "dummyMolsFromFragments",
        kind: "class",
        badges: ["DrugEx 3.4+", "Attachment Points", "Dummy Atoms"],
        synopsis: `class dummyMolsFromFragments(MolConverter):
    @staticmethod
    def addCCToFragments(frag: str) -> Optional[str]: ...
    @staticmethod
    def bridgeFragments(frags: str) -> Optional[str]: ...
    def __call__(self, frag: str) -> Optional[List[Tuple[str, str]]]: ...`,
        description: "Converter that synthesizes dummy molecules from fragment strings. Connects CC groups to single fragments or bridges multiple fragment components together across attachment vectors.",
        members: [
          { name: "addCCToFragments(frag)", desc: "Static helper adding an ethyl CC bridge to a single fragment attachment vector." },
          { name: "bridgeFragments(frags)", desc: "Static helper bridging multiple disconnected leaf fragments together into a unified dummy molecule." },
          { name: "__call__(frag)", desc: "Creates dummy molecules by adding CC bridges or connecting multiple fragments together, returning [(fragment, dummy_smiles)]." }
        ],
        variables: [],
        parameters: [],
        notes: "Crucial for guiding fragment elaboration models so they connect new atoms strictly at designed exit vectors.",
        example: {
          code: `from dummyMolsFromFragments import dummyMolsFromFragments
from drugex.molecules.converters.dummy_molecules import dummyMolsFromFragments

converter = dummyMolsFromFragments()

# 1. Single leaf fragment elaboration via ethyl (CC) addition
frag_single = "c1ccccc1"
pair_single = converter(frag_single)

# 2. Multi-fragment bridging connecting two disconnected scaffold components
frag_multi = "c1ccccc1.c1ccncc1"
pair_multi = converter(frag_multi)

# 3. Direct inspection of helper static methods
ethyl_added = dummyMolsFromFragments.addCCToFragments("c1ccncc1")
bridged = dummyMolsFromFragments.bridgeFragments("c1ccccc1.c1ccncc1")

print("Fragment Dummy Molecule Synthesis:")
print(f"  Single scaffold input : {frag_single} -> Dummy pair: {pair_single}")
print(f"  Multi-fragment input  : {frag_multi} -> Dummy pair: {pair_multi}")
print(f"  Direct ethyl addition : c1ccncc1 -> {ethyl_added}")
print(f"  Direct bond bridging  : c1ccccc1.c1ccncc1 -> {bridged}")`,
          output: `Fragment Dummy Molecule Synthesis:
  Single scaffold input : c1ccccc1 -> Dummy pair: [('c1ccccc1', 'CCc1ccccc1')]
  Multi-fragment input  : c1ccccc1.c1ccncc1 -> Dummy pair: [('c1ccccc1.c1ccncc1', 'c1ccc(-c2ccncc2)cc1')]
  Direct ethyl addition : c1ccncc1 -> CCc1ccncc1
  Direct bond bridging  : c1ccccc1.c1ccncc1 -> c1ccc(-c2ccncc2)cc1`
        },
        seeAlso: [
          "drugex::data::fragments::Fragmenter",
          "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment"
        ]
      },

    "drugex::molecules::converters::default::SmilesToDrEx": {
        category: "core",
        module: "drugex.molecules.converters.default",
        title: "SmilesToDrEx",
        kind: "class",
        badges: ["DrugEx 3.4+", "Type Converter", "Fast Ingestion"],
        synopsis: `class SmilesToDrEx(MolConverter):
    def __call__(self, smiles: str) -> DrExMol: ...`,
        description: "Converter parsing SMILES strings into unified DrExMol wrapper objects.",
        members: [
          { name: "__call__(smiles)", desc: "Constructs and validates DrExMol from SMILES string, raising ConversionException if invalid." }
        ],
        variables: [],
        parameters: [],
        notes: "Raises ConversionException if SMILES parsing fails, allowing suppliers to catch errors gracefully.",
        example: {
          code: `from drugex.molecules.converters.default import SmilesToDrEx

converter = SmilesToDrEx()
smiles_batch = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "c1ccncc1"]
drex_mols = [converter(smi) for smi in smiles_batch]

print(f"Converted {len(drex_mols)} SMILES to DrExMol objects:")
for m in drex_mols:
    print(f"  ID: {m.getUniqueID():25} | Heavy atoms: {m.asRDKit().GetNumHeavyAtoms()}")

# Verify round-trip conversion and RDKit binding
first_mol = drex_mols[0]
print(f"First DrExMol smiles: {first_mol.smiles}")
print(f"First DrExMol conforms: {first_mol.asRDKit() is not None}")`,
          output: `Converted 3 SMILES to DrExMol objects:
  ID: c1ccccc1O                 | Heavy atoms: 7
  ID: CC(=O)Oc1ccccc1C(=O)O     | Heavy atoms: 13
  ID: c1ccncc1                  | Heavy atoms: 6
First DrExMol smiles: c1ccccc1O
First DrExMol conforms: True`
        },
        seeAlso: [
          "drugex::molecules::mol::DrExMol",
          "drugex::molecules::converters::default::DrExToSMILES",
          "drugex::molecules::suppliers::ListSupplier"
        ]
      },

    "drugex::molecules::converters::default::DrExToSMILES": {
        category: "core",
        module: "drugex.molecules.converters.default",
        title: "DrExToSMILES",
        kind: "class",
        badges: ["DrugEx 3.4+", "Serialization", "Canonical Output"],
        synopsis: `class DrExToSMILES(MolConverter):
    def __call__(self, drugex_mol: DrExMol) -> str: ...`,
        description: "Converter extracting the canonical SMILES string from a DrExMol instance.",
        members: [
          { name: "__call__(drugex_mol)", desc: "Returns canonical SMILES string from DrExMol.smiles property." }
        ],
        variables: [],
        parameters: [],
        notes: "Returns the cached canonical SMILES identifier property directly.",
        example: {
          code: `from drugex.molecules.converters.default import SmilesToDrEx, DrExToSMILES

to_drex = SmilesToDrEx()
to_smiles = DrExToSMILES()

dmol = to_drex("CC(=O)Oc1ccccc1C(=O)O")
extracted_smi = to_smiles(dmol)

print("Source DrExMol unique ID:", dmol.getUniqueID())
print("Extracted canonical SMILES:", extracted_smi)
print("Round-trip verified:", extracted_smi == dmol.smiles)

# Batch extraction across multiple DrExMol instances
batch_mols = [to_drex("c1ccccc1"), to_drex("CCO")]
batch_smiles = [to_smiles(m) for m in batch_mols]
print(f"Batch extracted SMILES count: {len(batch_smiles)}")`,
          output: `Source DrExMol unique ID: CC(=O)Oc1ccccc1C(=O)O
Extracted canonical SMILES: CC(=O)Oc1ccccc1C(=O)O
Round-trip verified: True
Batch extracted SMILES count: 2`
        },
        seeAlso: [
          "drugex::molecules::converters::default::SmilesToDrEx",
          "drugex::molecules::mol::DrExMol"
        ]
      },

    "drugex::molecules::converters::default::RDKitToDrEx": {
        category: "core",
        module: "drugex.molecules.converters.default",
        title: "RDKitToDrEx",
        kind: "class",
        badges: ["DrugEx 3.4+", "RDKit Bridge", "Mol Ingestion"],
        synopsis: `class RDKitToDrEx(MolConverter):
    def __call__(self, rd_mol: Chem.Mol) -> DrExMol: ...`,
        description: "Converter bridging native RDKit Mol instances into DrExMol representations.",
        members: [
          { name: "__call__(rd_mol)", desc: "Wraps RDKit Mol into DrExMol object, raising ConversionException if molecule is empty." }
        ],
        variables: [],
        parameters: [],
        notes: "Caches RDKit Mol reference directly in DrExMol to avoid re-parsing overhead.",
        example: {
          code: `from rdkit import Chem
from drugex.molecules.converters.default import RDKitToDrEx

converter = RDKitToDrEx()
rd_mols = [
    Chem.MolFromSmiles("c1ccccc1O"),
    Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
]

for mol in rd_mols:
    drex_mol = converter(mol)
    print(f"Converted {Chem.MolToSmiles(mol)} -> DrExMol(ID='{drex_mol.getUniqueID()}')")

print(f"Total converted instances: {len(rd_mols)}")`,
          output: `Converted Oc1ccccc1 -> DrExMol(ID='Oc1ccccc1')
Converted Cn1cnc2c1c(=O)n(C)c(=O)n2C -> DrExMol(ID='Cn1cnc2c1c(=O)n(C)c(=O)n2C')
Total converted instances: 2`
        },
        seeAlso: [
          "drugex::molecules::mol::DrExMol",
          "drugex::molecules::converters::default::SmilesToDrEx"
        ]
      },

    "drugex::molecules::converters::default::Identity": {
        category: "core",
        module: "drugex.molecules.converters.default",
        title: "Identity",
        kind: "class",
        badges: ["DrugEx 3.4+", "Pass Through", "No-Op Converter"],
        synopsis: `class Identity(MolConverter):
    def __call__(self, item: Any) -> Any: ...`,
        description: "No-op pass-through converter returning input item unchanged. Used as default converter in suppliers when data is already in target format.",
        members: [
          { name: "__call__(item)", desc: "Returns item unchanged." }
        ],
        variables: [],
        parameters: [],
        notes: "Zero performance overhead.",
        example: {
          code: `from drugex.molecules.converters.default import Identity

ident = Identity()
test_items = [
    "c1ccccc1O",
    {"task": "classification", "scheme": "WS"},
    [1, 2, 3, 4]
]

for item in test_items:
    result = ident(item)
    print(f"Input: {repr(item):35} -> Identity Output: {repr(result)}")`,
          output: `Input: 'c1ccccc1O'                         -> Identity Output: 'c1ccccc1O'
Input: {'task': 'classification', 'scheme': 'WS'} -> Identity Output: {'task': 'classification', 'scheme': 'WS'}
Input: [1, 2, 3, 4]                        -> Identity Output: [1, 2, 3, 4]`
        },
        seeAlso: [
          "drugex::molecules::converters::MolConverter",
          "drugex::molecules::suppliers::ListSupplier"
        ]
      },

    "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment": {
        category: "core",
        module: "drugex.molecules.converters.fragmenters",
        title: "FragmenterWithSelectedFragment",
        kind: "class",
        badges: ["DrugEx 3.4+", "Scaffold Focus", "Conditional Cleavage"],
        synopsis: `class FragmenterWithSelectedFragment(Fragmenter):
    def __init__(
        self,
        fragment: str,
        n_frags: int,
        n_combs: int,
        method: str = "recap",
        deep_clean: bool = True,
        max_bonds: int = 75,
        allow_single: bool = False,
        exclusive: bool = False
    ) -> None: ...
    def __call__(self, smiles: str) -> Optional[List[Tuple[str, str]]]: ...`,
        description: "Specialized fragmenter filtering cleavage pairs to only retain combinations where the input conditioning fragment contains the specified scaffold.",
        members: [
          { name: "getFragments(mol)", desc: "Cleaves molecular bonds using configured fragmentation method (recap or brics) and returns list of leaf fragments." },
          { name: "filterFragments(frags)", desc: "Filters fragments against size limits, bond counts, and exclusivity criteria." },
          { name: "__call__(smiles)", desc: "Generates fragment-molecule pairs for given SMILES string and returns only combinations matching selected fragment." }
        ],
        variables: [
          { name: "fragment", type: "str", desc: "Core scaffold required in generated fragments." },
          { name: "exclusive", type: "bool", desc: "Whether to strictly require exact scaffold match without extra fragments." }
        ],
        parameters: [
          { name: "fragment", type: "str", desc: "Fragment to search for and retain in cleavage combinations." },
          { name: "n_frags", type: "int", desc: "Number of fragments to generate per compound." },
          { name: "n_combs", type: "int", desc: "Maximum combinations of found leaf fragments." },
          { name: "method", type: "str", desc: "Fragmentation scheme ('recap' or 'brics', default: 'recap')." },
          { name: "deep_clean", type: "bool", desc: "Deep clean SMILES before fragmentation (default: True)." },
          { name: "max_bonds", type: "int", desc: "Maximum bond count threshold (default: 75)." },
          { name: "allow_single", type: "bool", desc: "Return fragments for single unfragmented molecules (default: False)." },
          { name: "exclusive", type: "bool", desc: "If True, only return fragments that match specified fragment and nothing else (default: False)." }
        ],
        notes: "Key driver for scaffold decoration and R-group optimization campaigns around known intellectual property cores.",
        example: {
          code: `from drugex.molecules.converters.fragmenters import FragmenterWithSelectedFragment

scaffold_core = "c1ccc2ncnc(Nc3ccccc3)c2c1"
fragmenter = FragmenterWithSelectedFragment(
    fragment=scaffold_core,
    n_frags=4,
    n_combs=2,
    method="brics"
)

# Test molecule containing target quinazoline scaffold
mol_match = "COc1cc2ncnc(Nc3ccccc3)c2cc1OCC"
pairs_match = fragmenter(mol_match)

# Test molecule without target scaffold
mol_nomatch = "CC(=O)Oc1ccccc1C(=O)O"
pairs_nomatch = fragmenter(mol_nomatch)

print("FragmenterWithSelectedFragment Scaffolding Pipeline:")
print(f"  Target core fragment  : {scaffold_core}")
print(f"  Matched compound      : {mol_match}")
print(f"  Retained pairs count  : {len(pairs_match) if pairs_match else 0}")
if pairs_match:
    for i, (frag, smi) in enumerate(pairs_match):
        print(f"    Pair {i+1}: Fragment='{frag}' | Full='{smi}'")
print(f"  Non-matching compound : {mol_nomatch} -> Result: {pairs_nomatch}")`,
          output: `FragmenterWithSelectedFragment Scaffolding Pipeline:
  Target core fragment  : c1ccc2ncnc(Nc3ccccc3)c2c1
  Matched compound      : COc1cc2ncnc(Nc3ccccc3)c2cc1OCC
  Retained pairs count  : 1
    Pair 1: Fragment='c1ccc2ncnc(Nc3ccccc3)c2c1' | Full='COc1cc2ncnc(Nc3ccccc3)c2cc1OCC'
  Non-matching compound : CC(=O)Oc1ccccc1C(=O)O -> Result: None`
        },
        seeAlso: [
          "drugex::data::fragments::Fragmenter",
          "drugex::molecules::converters::dummyMolsFromFragments"
        ]
      },

    "drugex::molecules::suppliers::StandardizedSupplier": {
        category: "core",
        module: "drugex.molecules.suppliers",
        title: "StandardizedSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Lazy Supplier", "On-the-fly Cleanse"],
        synopsis: `class StandardizedSupplier(BaseMolSupplier):
    def __init__(
        self,
        mols: Iterable[Any],
        standardizer: MolConverter
    ) -> None: ...
    def next(self) -> Any: ...`,
        description: "Lazy wrapper supplier applying a molecular standardizer converter pipeline on-the-fly as molecules are yielded by an underlying iterable or supplier.",
        members: [
          { name: "next()", desc: "Defines access to the next item to be processed from input molecules iterator." }
        ],
        variables: [
          { name: "mols", type: "Iterator[Any]", desc: "Underlying iterator over input molecules." }
        ],
        parameters: [
          { name: "mols", type: "Iterable[Any]", desc: "Input collection or iterator of molecules." },
          { name: "standardizer", type: "MolConverter", desc: "Converter instance used to transform molecules into standardized form." }
        ],
        notes: "Invalid or unparseable molecules in the stream are silently skipped or returned as None depending on configuration.",
        example: {
          code: `from drugex.molecules.suppliers import ListSupplier, StandardizedSupplier
from drugex.molecules.converters.standardizers import CleanSMILES

raw_smiles = ["c1ccccc1", "CCO", "CC(=O)O"]
base_supplier = ListSupplier(raw_smiles)
cleaner = CleanSMILES()
std_supplier = StandardizedSupplier(base_supplier, standardizer=cleaner)
cleaned_mols = list(std_supplier)

print(f"Standardized stream count: {len(cleaned_mols)}")
for idx, mol in enumerate(cleaned_mols, 1):
    print(f"Standardized item {idx}: {mol}")`,
          output: `Standardized stream count: 3
Standardized item 1: c1ccccc1
Standardized item 2: CCO
Standardized item 3: CC(=O)O`
        },
        seeAlso: [
          "drugex::molecules::suppliers::ListSupplier",
          "drugex::molecules::converters::DefaultStandardizer",
          "drugex::molecules::interfaces::BaseMolSupplier"
        ]
      },

    "drugex::molecules::suppliers::ListSupplier": {
        category: "core",
        module: "drugex.molecules.suppliers",
        title: "ListSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "In-Memory", "List Adapter"],
        synopsis: `class ListSupplier(BaseMolSupplier):
    def __init__(
        self,
        mols: Iterable[Any],
        converter: MolConverter = SmilesToDrEx(),
        hide_duplicates: bool = False
    ) -> None: ...
    def next(self) -> Any: ...`,
        description: "Memory-efficient iterable molecule supplier backed by in-memory Python collections. Converts raw molecules to target representations (e.g. SMILES to DrExMol) and optionally filters duplicates.",
        members: [
          { name: "next()", desc: "Yields next molecule from internal iterator." }
        ],
        variables: [
          { name: "mols", type: "Iterator[Any]", desc: "Internal iterator over input molecules." }
        ],
        parameters: [
          { name: "mols", type: "Iterable[Any]", desc: "Collection of input molecules (e.g. SMILES strings)." },
          { name: "converter", type: "MolConverter", desc: "Converter transforming items to desired representation (default: SmilesToDrEx())." },
          { name: "hide_duplicates", type: "bool", desc: "Skip duplicate molecules based on unique ID (default: False)." }
        ],
        notes: "Recommended for feeding batch outputs of generator models directly into parallel evaluation pipelines.",
        example: {
          code: `from drugex.molecules.suppliers import ListSupplier
from drugex.molecules.converters.default import SmilesToDrEx

smiles_data = ["c1ccccc1", "CCO", "c1ccccc1", "CC(=O)O"]
supplier = ListSupplier(
    mols=smiles_data,
    converter=SmilesToDrEx(),
    hide_duplicates=True
)

mols = [m for m in supplier]
print(f"Supplier processed {len(smiles_data)} items into {len(mols)} unique DrExMols:")
for m in mols:
    print(f"  SMILES: {m.smiles}")`,
          output: `Supplier processed 4 items into 3 unique DrExMols:
  SMILES: c1ccccc1
  SMILES: CCO
  SMILES: CC(=O)O`
        },
        seeAlso: [
          "drugex::molecules::suppliers::StandardizedSupplier",
          "drugex::molecules::interfaces::BaseMolSupplier",
          "drugex::molecules::converters::default::SmilesToDrEx"
        ]
      },

    "drugex::molecules::mol::DrExMol": {
        category: "core",
        module: "drugex.molecules.mol",
        title: "DrExMol",
        kind: "class",
        badges: ["DrugEx 3.4+", "Core Molecule", "Unified Wrapper"],
        synopsis: `class DrExMol(Molecule):
    def __init__(
        self,
        rd_mol: Chem.Mol,
        identifier: Optional[str] = None
    ) -> None: ...
    @property
    def smiles(self) -> str: ...
    def annotate(self, key: str, value: Any) -> None: ...
    def asRDKit(self) -> Chem.Mol: ...
    def getAnnotation(self, key: str) -> Any: ...
    def getMetadata(self) -> Dict[str, Any]: ...
    def getUniqueID(self) -> str: ...`,
        description: "Unified DrugEx molecular wrapper. Encapsulates 2D graph topology, canonical SMILES, metadata annotations, and underlying RDKit Chem.Mol instance in a single standardized object.",
        members: [
          { name: "annotate(key, value)", desc: "Attaches computational annotation key-value pair to molecule." },
          { name: "asRDKit()", desc: "Returns underlying RDKit Chem.Mol instance." },
          { name: "getAnnotation(key)", desc: "Retrieves annotation property value for specified key." },
          { name: "getMetadata()", desc: "Returns dictionary of attached property annotations." },
          { name: "getUniqueID()", desc: "Returns unique canonical identifier (canonical SMILES)." }
        ],
        variables: [
          { name: "canonical_smiles", type: "str", desc: "Canonical SMILES representation." },
          { name: "annotations", type: "Dict[str, Any]", desc: "Property annotation dictionary." },
          { name: "rd_mol", type: "Chem.Mol", desc: "Wrapped RDKit Mol object." }
        ],
        parameters: [
          { name: "rd_mol", type: "Chem.Mol", desc: "Input RDKit molecule." },
          { name: "identifier", type: "Optional[str]", desc: "Custom identifier (defaults to canonical SMILES)." }
        ],
        notes: "Provides lazy canonical SMILES calculation if identifier is not explicitly provided.",
        example: {
          code: `from rdkit import Chem
from drugex.molecules.mol import DrExMol

rdkit_mol = Chem.MolFromSmiles("c1ccccc1O")
dmol = DrExMol(rdkit_mol, identifier="Phenol_01")

dmol.annotate("MW", 94.11)
dmol.annotate("LogP", 1.46)

print(f"DrExMol Identifier: {dmol.getUniqueID()}")
print(f"Canonical SMILES: {dmol.smiles}")
print(f"Annotations: {dmol.getMetadata()}")
print(f"RDKit Mol atoms count: {dmol.asRDKit().GetNumAtoms()}")`,
          output: `DrExMol Identifier: Phenol_01
Canonical SMILES: Phenol_01
Annotations: {'MW': 94.11, 'LogP': 1.46}
RDKit Mol atoms count: 7`
        },
        seeAlso: [
          "drugex::molecules::interfaces::Molecule",
          "drugex::molecules::converters::default::SmilesToDrEx",
          "drugex::molecules::converters::default::DrExToSMILES"
        ]
      },

    "drugex::molecules::converters::ConversionException": {
        category: "core",
        module: "drugex.molecules.converters.interfaces",
        title: "ConversionException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Converters"],
        synopsis: `class ConversionException(Exception):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Fundamental exception signaling molecular representation conversion failure in DrugEx pipelines. Raised when chemical parsers or format converters encounter malformed SMILES strings, invalid chemical valences, unresolvable Kekulization errors, or stereochemical ambiguities. Serves as the primary error boundary caught by MolSupplier and parallel supplier evaluators to discard corrupted or unparseable structures gracefully without terminating training or preprocessing pipelines.",
        members: [
          { name: "args", desc: "Tuple of diagnostic arguments and error strings passed to the exception constructor." },
          { name: "with_traceback(tb)", desc: "Attaches a traceback object to the exception instance and returns self." },
          { name: "__str__()", desc: "Formats human-readable diagnostic error string for logging and debugging." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Positional diagnostic error message strings or cause objects detailing the conversion failure." }
        ],
        notes: "Serves as the root exception caught by parallel suppliers and evaluators to safely discard malformed molecules without aborting training.",
        example: {
          code: `from drugex.molecules.converters.interfaces import ConversionException
from rdkit import Chem

raw_smiles_batch = [
    "c1ccccc1O",               # Phenol (valid)
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin (valid)
    "c1cccc1=C=C=C",           # Invalid strained valence (malformed)
    "CCN(CC)CC"                # Triethylamine (valid)
]

parsed_molecules = []
dropped_records = []

for idx, smi in enumerate(raw_smiles_batch):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ConversionException(f"Failed to parse SMILES syntax at index {idx}: '{smi}'")
        parsed_molecules.append(mol)
    except ConversionException as err:
        dropped_records.append((idx, smi, str(err)))

print(f"Batch conversion processed: {len(raw_smiles_batch)} candidates")
print(f"Successfully converted     : {len(parsed_molecules)} valid Mol structures")
print(f"Caught ConversionException : {len(dropped_records)} dropped record")
for row in dropped_records:
    print(f"  Record [{row[0]}]: {row[1]} -> {row[2]}")`,
          output: `Batch conversion processed: 4 candidates
Successfully converted     : 3 valid Mol structures
Caught ConversionException : 1 dropped record
  Record [2]: c1cccc1=C=C=C -> Failed to parse SMILES syntax at index 2: 'c1cccc1=C=C=C'`
        },
        seeAlso: [
          "drugex::molecules::converters::StandardizationException",
          "drugex::molecules::interfaces::ItemException",
          "drugex::molecules::converters::MolConverter",
          "drugex::molecules::interfaces::MolSupplier"
        ]
      },

    "drugex::molecules::converters::StandardizationException": {
        category: "core",
        module: "drugex.molecules.converters.standardizers",
        title: "StandardizationException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Standardizer"],
        synopsis: `class StandardizationException(ConversionException):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Domain-specific exception raised when automated chemical standardization fails. Triggered by DefaultStandardizer or CleanSMILES when a chemical structure contains unresolvable hypervalent heteroatoms, persistent radical centers, failed salt-stripping disconnections, or tautomeric canonicalization failures.",
        members: [
          { name: "args", desc: "Tuple of diagnostic arguments passed during exception initialization." },
          { name: "with_traceback(tb)", desc: "Attaches a traceback object to the exception instance and returns self." },
          { name: "__str__()", desc: "Formats human-readable standardization failure error message." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Diagnostic error message indicating why molecular standardization failed." }
        ],
        notes: "Inherits from ConversionException. Trapped by DefaultStandardizer when running across multi-threaded and multi-process pipelines.",
        example: {
          code: `from drugex.molecules.converters.standardizers import StandardizationException
from rdkit import Chem

candidates = [
    ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
    ("Hypervalent Nitrogen", "[N+5](=O)(=O)(=O)(=O)[O-]"),
    ("Ibuprofen Sodium", "CC(C)Cc1ccc(cc1)C(C)C(=O)[O-].[Na+]")
]

standardized = []
standardization_errors = []

for name, smi in candidates:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or any(atom.GetExplicitValence() > 4 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'):
            raise StandardizationException(f"Valence ceiling exceeded during standardization of '{name}'")
        standardized.append(name)
    except StandardizationException as e:
        standardization_errors.append((name, str(e)))

print("Batch molecular standardization results:")
print(f"  Successfully standardized: {len(standardized)} compounds {standardized}")
print(f"  Standardization failures : {len(standardization_errors)}")
for name, msg in standardization_errors:
    print(f"  - [{name}]: {msg}")`,
          output: `Batch molecular standardization results:
  Successfully standardized: 2 compounds ['Acetaminophen', 'Ibuprofen Sodium']
  Standardization failures : 1
  - [Hypervalent Nitrogen]: Valence ceiling exceeded during standardization of 'Hypervalent Nitrogen'`
        },
        seeAlso: [
          "drugex::molecules::converters::ConversionException",
          "drugex::molecules::converters::DefaultStandardizer",
          "drugex::molecules::converters::CleanSMILES"
        ]
      },

    "drugex::molecules::interfaces::ItemException": {
        category: "core",
        module: "drugex.molecules.interfaces",
        title: "ItemException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Suppliers"],
        synopsis: `class ItemException(Exception):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Supplier-level exception raised when accessing or generating an item in a MolSupplier fails due to missing indices, corrupt storage records, or supplier iteration faults. Caught inside MolSupplier.__next__() to log diagnostic warnings and proceed to the next item without aborting iteration.",
        members: [
          { name: "args", desc: "Tuple of arguments containing access failure details." },
          { name: "with_traceback(tb)", desc: "Attaches a traceback to the exception instance and returns self." },
          { name: "__str__()", desc: "Formats supplier item retrieval failure description." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Supplier index, file path, or record details where item access failed." }
        ],
        notes: "Allows MolSupplier subclasses to log indexed item read errors while continuing stream ingestion.",
        example: {
          code: `from drugex.molecules.interfaces import ItemException

class RobustMolSupplier:
    def __init__(self, records):
        self.records = records

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.records):
            raise ItemException(f"Index {idx} out of supplier bounds [0, {len(self.records)-1}]")
        rec = self.records[idx]
        if rec is None or not rec.strip():
            raise ItemException(f"Supplier record at index {idx} is empty or corrupted")
        return rec

supplier = RobustMolSupplier(["c1ccccc1", "", "c1ccncc1", "CC(=O)O"])
retrieved, skipped = [], []

for i in range(5):
    try:
        item = supplier[i]
        retrieved.append((i, item))
    except ItemException as err:
        skipped.append((i, str(err)))

print(f"Supplier streaming completed: {len(retrieved)} valid, {len(skipped)} exceptions")
print(f"Retrieved items: {retrieved}")
print("Caught ItemException events:")
for idx, msg in skipped:
    print(f"  Index {idx}: {msg}")`,
          output: `Supplier streaming completed: 3 valid, 2 exceptions
Retrieved items: [(0, 'c1ccccc1'), (2, 'c1ccncc1'), (3, 'CC(=O)O')]
Caught ItemException events:
  Index 1: Supplier record at index 1 is empty or corrupted
  Index 4: Index 4 out of supplier bounds [0, 3]`
        },
        seeAlso: [
          "drugex::molecules::interfaces::MolSupplier",
          "drugex::molecules::interfaces::BaseMolSupplier",
          "drugex::molecules::mol::InitializationException",
          "drugex::molecules::converters::ConversionException"
        ]
      },

    "drugex::molecules::mol::InitializationException": {
        category: "core",
        module: "drugex.molecules.mol",
        title: "InitializationException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Molecule Entity"],
        synopsis: `class InitializationException(Exception):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Core domain exception raised when instantiating a DrExMol wrapper around an invalid, empty, or unparseable chemical representation. Enforces strict chemical entity invariants at the boundary of DrugEx's molecular graph and descriptor calculators, preventing None-valued or corrupt RDKit Mol instances from propagating into neural networks.",
        members: [
          { name: "args", desc: "Tuple of arguments detailing molecule initialization error." },
          { name: "with_traceback(tb)", desc: "Attaches exception traceback and returns self." },
          { name: "__str__()", desc: "Formats molecule construction failure description." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Details of failed molecule initialization, including compound identifier." }
        ],
        notes: "Ensures that all downstream transformers and encoders receive strictly valid Molecule wrappers.",
        example: {
          code: `from drugex.molecules.mol import InitializationException
from rdkit import Chem

def init_molecule_wrapper(raw_input):
    if not isinstance(raw_input, (str, Chem.Mol)) or (isinstance(raw_input, str) and not raw_input.strip()):
        raise InitializationException(f"Invalid representation '{raw_input}': Expected non-empty SMILES string or RDKit Mol")
    mol = Chem.MolFromSmiles(raw_input) if isinstance(raw_input, str) else raw_input
    if mol is None:
        raise InitializationException(f"Cannot initialize Molecule from unparseable SMILES: '{raw_input}'")
    return mol

test_inputs = ["c1ccccc1", "", 12345, "invalid_smiles_string"]
valid_mols, init_errors = [], []

for item in test_inputs:
    try:
        mol = init_molecule_wrapper(item)
        valid_mols.append(Chem.MolToSmiles(mol))
    except InitializationException as err:
        init_errors.append(str(err))

print(f"Molecule entity initialization: {len(valid_mols)} valid, {len(init_errors)} failures")
print(f"Initialized valid SMILES: {valid_mols}")
print("InitializationException diagnostics:")
for err in init_errors:
    print(f"  - {err}")`,
          output: `Molecule entity initialization: 1 valid, 3 failures
Initialized valid SMILES: ['c1ccccc1']
InitializationException diagnostics:
  - Invalid representation '': Expected non-empty SMILES string or RDKit Mol
  - Invalid representation '12345': Expected non-empty SMILES string or RDKit Mol
  - Cannot initialize Molecule from unparseable SMILES: 'invalid_smiles_string'`
        },
        seeAlso: [
          "drugex::molecules::mol::DrExMol",
          "drugex::molecules::interfaces::Molecule",
          "drugex::molecules::interfaces::ItemException"
        ]
      },

    "drugex::parallel::interfaces::ParallelException": {
        category: "core",
        module: "drugex.parallel.interfaces",
        title: "ParallelException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Concurrency"],
        synopsis: `class ParallelException(Exception):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Execution-level exception encapsulating failures encountered during multi-process high-throughput molecular processing. Raised when parallel worker subprocesses terminate unexpectedly, inter-process communication (IPC) queues fail, or worker execution exceeds timeout limits. Preserves worker tracebacks across multiprocessing boundaries for robust pipeline diagnostics.",
        members: [
          { name: "args", desc: "Tuple of arguments passed to ParallelException." },
          { name: "with_traceback(tb)", desc: "Attaches traceback across multiprocessing pool boundary." },
          { name: "__str__()", desc: "Formats parallel worker error description." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Multiprocessing worker pool error diagnostic message or exception cause." }
        ],
        notes: "Provides diagnostic traceback across multiprocessing pool boundaries for robust training workflows.",
        example: {
          code: `from drugex.parallel.interfaces import ParallelException

def evaluate_shard(shard_id, payload):
    # Simulate worker shard execution where shard 3 encounters memory limit
    if shard_id == 3:
        raise ParallelException(f"Worker PID 24891 on shard {shard_id} terminated unexpectedly (OOM)")
    return {"shard_id": shard_id, "processed": len(payload)}

shards = {1: ["mol_1", "mol_2"], 2: ["mol_3", "mol_4"], 3: ["mol_5", "mol_6"], 4: ["mol_7", "mol_8"]}
completed_shards = []
failed_shards = []

for sid, batch in shards.items():
    try:
        res = evaluate_shard(sid, batch)
        completed_shards.append(res)
    except ParallelException as p_err:
        failed_shards.append((sid, str(p_err)))

print("Multiprocessing worker supervisor status:")
print(f"  Completed shards : {len(completed_shards)} / {len(shards)}")
print(f"  Failed shards    : {len(failed_shards)}")
for sid, err in failed_shards:
    print(f"  Caught ParallelException on shard {sid}: {err}")
print("Pipeline safely degraded: failed shard rescheduled to recovery queue.")`,
          output: `Multiprocessing worker supervisor status:
  Completed shards : 3 / 4
  Failed shards    : 1
  Caught ParallelException on shard 3: Worker PID 24891 on shard 3 terminated unexpectedly (OOM)
Pipeline safely degraded: failed shard rescheduled to recovery queue.`
        },
        seeAlso: [
          "drugex::parallel::ParallelProcessor",
          "drugex::parallel::ParallelSupplierEvaluator",
          "drugex::parallel::collectors::ListExtend"
        ]
      },

    "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsEncodedSupplier.FragmentEncodingException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Fragments"],
        synopsis: `class FragmentEncodingException(ConversionException):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Specialized conversion exception raised within FragmentPairsEncodedSupplier.next() when an input fragment scaffold cannot be encoded into integer token sequences by the configured FragmentPairEncoder. Indicates missing sub-vocabulary tokens, invalid dummy attachment point syntax, or unmatchable substructure patterns.",
        members: [
          { name: "args", desc: "Tuple of arguments passed during exception instantiation." },
          { name: "with_traceback(tb)", desc: "Standard traceback assignment method returning self." },
          { name: "__str__()", desc: "Formats fragment vocabulary encoding error message." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Fragment representation or tokenization mismatch details." }
        ],
        notes: "Allows fragment dataset preparation to identify corrupted or out-of-vocabulary building blocks without dropping the entire batch.",
        example: {
          code: `from drugex.data.fragments import FragmentPairsEncodedSupplier

# Vocabulary with standard single attachment points [1*], [2*]
vocab_tokens = {"[1*]", "[2*]", "c", "n", "o", "C", "(", ")", "1", "2"}

def encode_fragment(frag_smi):
    # Tokens required by fragment
    if "[99*]" in frag_smi:
        raise FragmentPairsEncodedSupplier.FragmentEncodingException(
            f"Unsupported dummy attachment point '[99*]' in fragment '{frag_smi}'"
        )
    return [c for c in frag_smi]

fragments = ["[1*]c1ccccc1", "[2*]c1ncn(C)c1", "[99*]c1ccc(O)cc1"]
encoded_frags, encoding_failures = [], []

for frag in fragments:
    try:
        toks = encode_fragment(frag)
        encoded_frags.append(frag)
    except FragmentPairsEncodedSupplier.FragmentEncodingException as e:
        encoding_failures.append((frag, str(e)))

print("Fragment vocabulary encoding summary:")
print(f"  Successfully encoded fragments : {len(encoded_frags)}")
print(f"  Fragment encoding exceptions   : {len(encoding_failures)}")
for frag, msg in encoding_failures:
    print(f"  - Failed fragment: {frag} -> {msg}")`,
          output: `Fragment vocabulary encoding summary:
  Successfully encoded fragments : 2
  Fragment encoding exceptions   : 1
  - Failed fragment: [99*]c1ccc(O)cc1 -> Unsupported dummy attachment point '[99*]' in fragment '[99*]c1ccc(O)cc1'`
        },
        seeAlso: [
          "drugex::data::fragments::FragmentPairsEncodedSupplier",
          "drugex::molecules::converters::ConversionException",
          "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException"
        ]
      },

    "drugex::data::fragments::FragmentPairsEncodedSupplier::MoleculeEncodingException": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsEncodedSupplier.MoleculeEncodingException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Error Handling", "Fragments"],
        synopsis: `class MoleculeEncodingException(ConversionException):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Specialized conversion exception raised within FragmentPairsEncodedSupplier.next() when a parent molecule cannot be encoded into integer token sequences by the configured FragmentPairEncoder. Indicates out-of-vocabulary atoms, invalid token sequences, or structural anomalies, enabling graceful skipping of unencodable parent molecules.",
        members: [
          { name: "args", desc: "Tuple of arguments detailing molecule encoding failure." },
          { name: "with_traceback(tb)", desc: "Standard traceback assignment method returning self." },
          { name: "__str__()", desc: "Formats molecule token encoding error message." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Molecule token or SMILES sequence that failed vocabulary encoding." }
        ],
        notes: "Trapped during fragment-molecule pair corpus encoding to gracefully skip unencodable pairs.",
        example: {
          code: `from drugex.data.fragments import FragmentPairsEncodedSupplier

supported_atoms = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}

def encode_molecule_tokens(smiles):
    if "[Se]" in smiles:
        raise FragmentPairsEncodedSupplier.MoleculeEncodingException(
            f"Unrecognized atom token '[Se]' in candidate SMILES: '{smiles}'"
        )
    return len(smiles)

candidates = ["CC(=O)Nc1ccc(O)cc1", "c1ccc([Se]C)cc1", "c1cnc2ccccc2n1"]
encoded, skipped = [], []

for smi in candidates:
    try:
        token_len = encode_molecule_tokens(smi)
        encoded.append((smi, token_len))
    except FragmentPairsEncodedSupplier.MoleculeEncodingException as err:
        skipped.append((smi, str(err)))

print("Batch molecule corpus token encoding:")
print(f"  Successfully encoded molecules: {len(encoded)}")
print(f"  Encoding exception count      : {len(skipped)}")
for smi, reason in skipped:
    print(f"  Caught MoleculeEncodingException: {reason}")`,
          output: `Batch molecule corpus token encoding:
  Successfully encoded molecules: 2
  Encoding exception count      : 1
  Caught MoleculeEncodingException: Unrecognized atom token '[Se]' in candidate SMILES: 'c1ccc([Se]C)cc1'`
        },
        seeAlso: [
          "drugex::data::fragments::FragmentPairsEncodedSupplier",
          "drugex::molecules::converters::ConversionException",
          "drugex::data::fragments::FragmentPairsEncodedSupplier::FragmentEncodingException"
        ]
      },

    "drugex::training::interfaces::RewardScheme::RewardException": {
        category: "core",
        module: "drugex.training.interfaces",
        title: "RewardScheme.RewardException",
        kind: "class",
        badges: ["DrugEx 3.4+", "Sphinx gh-pages", "Error Handling"],
        synopsis: `class RewardException(Exception):
    def __init__(self, *args: Any) -> None: ...`,
        description: "Exception raised when an unrecoverable failure occurs during multi-objective Pareto front computation, crowding distance calculation, or reward scalarization in reinforcement learning.",
        members: [
          { name: "args", desc: "Tuple of arguments passed to RewardException." },
          { name: "with_traceback(tb)", desc: "Attaches traceback to reward exception instance." },
          { name: "__str__()", desc: "Formats reward scalarization error diagnostic." }
        ],
        variables: [],
        parameters: [
          { name: "*args", type: "Any", desc: "Error message describing the failure in reward calculation." }
        ],
        notes: "Prevents numerical NaN or Inf objective score arrays from silently corrupting policy gradient tensor updates in generator training.",
        example: {
          code: `import numpy as np
from drugex.training.interfaces import RewardScheme

def validate_and_compute_reward(objective_matrix):
    if np.isnan(objective_matrix).any():
        nan_cols = np.where(np.isnan(objective_matrix).any(axis=0))[0].tolist()
        raise RewardScheme.RewardException(
            f"Objective score matrix contains NaN at column indices {nan_cols}; gradient update aborted"
        )
    if objective_matrix.shape[1] < 2:
        raise RewardScheme.RewardException(
            f"Multi-objective scheme requires >= 2 objectives, got {objective_matrix.shape[1]}"
        )
    return objective_matrix.mean(axis=1)

# Batch of 3 candidates across 3 objectives [Affinity, QED, SA]
valid_batch = np.array([[0.82, 0.74, 0.65], [0.65, 0.81, 0.70]])
corrupt_batch = np.array([[0.85, np.nan, 0.60], [0.72, 0.68, 0.75]])
single_obj_batch = np.array([[0.82], [0.65]])

for batch_id, matrix in [("Batch_1", valid_batch), ("Batch_2", corrupt_batch), ("Batch_3", single_obj_batch)]:
    try:
        rewards = validate_and_compute_reward(matrix)
        print(f"[{batch_id}] Reward calculation succeeded: mean rewards = {np.round(rewards, 3).tolist()}")
    except RewardScheme.RewardException as r_err:
        print(f"[{batch_id}] Caught RewardException: {r_err}")`,
          output: `[Batch_1] Reward calculation succeeded: mean rewards = [0.737, 0.72]
[Batch_2] Caught RewardException: Objective score matrix contains NaN at column indices [1]; gradient update aborted
[Batch_3] Caught RewardException: Multi-objective scheme requires >= 2 objectives, got 1`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoRewardScheme",
          "drugex::training::environment::DrugExEnvironment"
        ]
      }
  });
})();
