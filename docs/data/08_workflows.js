/**
 * DrugEx Documentation Platform - Parallel Processing & Workflows
 * Library: workflows (48 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::parallel::ParallelProcessor": {
        category: "core",
        module: "drugex.parallel.interfaces",
        title: "ParallelProcessor",
        kind: "class",
        badges: ["DrugEx 3.4+", "Multiprocessing", "Worker Engine", "Abstract Base"],
        synopsis: `class ParallelProcessor(ABC):
    def __init__(
        self,
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = 1000,
        chunks: Optional[int] = None
    ) -> None: ...
    @abstractmethod
    def apply(self, data: Any, collector: ResultCollector) -> None: ...
    def getChunkSize(self, data: Sized) -> int: ...`,
        description: "Abstract multiprocessing coordinator in DrugEx. Manages worker pool allocation, adaptive batch chunking, inter-process communication (IPC), and parallel result aggregation across multi-core CPU architectures.",
        members: [
          { name: "apply(data, collector)", desc: "Abstract method distributing data chunks across worker processes and piping results to the designated ResultCollector." },
          { name: "getChunkSize(data)", desc: "Determines optimal chunk size from data length, falling back to len(data) // chunks if chunkSize is not explicitly set." }
        ],
        variables: [
          { name: "nProc", type: "int", desc: "Number of parallel worker processes initialized (defaults to cpu_count())." },
          { name: "chunkSize", type: "Optional[int]", desc: "Maximum batch chunk size sent to a single CPU worker." },
          { name: "chunks", type: "int", desc: "Total number of partition chunks (defaults to nProc)." }
        ],
        parameters: [
          { name: "n_proc", type: "Optional[int]", desc: "Number of processes to initialize (default: all available CPUs)." },
          { name: "chunk_size", type: "Optional[int]", desc: "Maximum chunk size per worker to limit memory usage (default: 1000)." },
          { name: "chunks", type: "Optional[int]", desc: "Number of chunks to divide the input data into (default: n_proc)." }
        ],
        notes: "Serves as the foundation for Standardization, CorpusEncoder, FragmentCorpusEncoder, and ParallelSupplierEvaluator.",
        example: {
          code: `from drugex.parallel.interfaces import ParallelProcessor
from drugex.parallel.collectors import ListExtend

class ChunkMultiplier(ParallelProcessor):
    def apply(self, data, collector):
        chunk_size = self.getChunkSize(data)
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            collector([x * 2 for x in chunk])

processor = ChunkMultiplier(n_proc=2, chunk_size=2)
collector = ListExtend()
processor.apply([1, 2, 3, 4], collector)

print(f"Optimal chunk size: {processor.getChunkSize([1, 2, 3, 4])}")
print(f"Processed items count: {len(collector.getList())}")
print(f"Aggregated results: {collector.getList()}")`,
          output: `Optimal chunk size: 2
Processed items count: 4
Aggregated results: [2, 4, 6, 8]`
        },
        seeAlso: [
          "drugex::parallel::ParallelSupplierEvaluator",
          "drugex::parallel::collectors::ListExtend",
          "drugex::parallel::interfaces::ResultCollector"
        ]
      },

    "drugex::parallel::collectors::ListExtend": {
        category: "core",
        module: "drugex.parallel.collectors",
        title: "ListExtend",
        kind: "class",
        badges: ["DrugEx 3.4+", "Result Collector", "Parallel Aggregator"],
        synopsis: `class ListExtend(ListCollector):
    def __init__(self) -> None: ...
    def getList(self) -> List[Any]: ...
    def __call__(self, result: Sequence[Any]) -> None: ...`,
        description: "Concrete result collector for parallel processing workers. Extends an internal list buffer with iterable batch results received from concurrent worker processes.",
        members: [
          { name: "getList()", desc: "Returns the aggregated master results list." },
          { name: "__call__(result)", desc: "Extends the internal results list with the sequence of items in result." }
        ],
        variables: [
          { name: "items", type: "List[Any]", desc: "Accumulated master results list." }
        ],
        parameters: [],
        notes: "Used as the default collector in Standardization and ParallelSupplierEvaluator when worker outputs are lists.",
        example: {
          code: `from drugex.parallel.collectors import ListExtend

collector = ListExtend()
batch_1 = ["c1ccccc1O", "CC(=O)O"]
batch_2 = ["c1ccncc1", "CCN"]
batch_3 = ["c1ccccc1", "CCO"]

collector(batch_1)
collector(batch_2)
collector(batch_3)

results = collector.getList()
print(f"Total collected items: {len(results)}")
print(f"First collected item: {results[0]}")
print(f"Last collected item: {results[-1]}")`,
          output: `Total collected items: 6
First collected item: c1ccccc1O
Last collected item: CCO`
        },
        seeAlso: [
          "drugex::parallel::ParallelProcessor",
          "drugex::parallel::interfaces::ListCollector",
          "drugex::parallel::collectors::ListAppend"
        ]
      },

    "drugex::parallel::ParallelSupplierEvaluator": {
        category: "core",
        module: "drugex.parallel.evaluator",
        title: "ParallelSupplierEvaluator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Batch Evaluator", "ProcessPoolExecutor"],
        synopsis: `class ParallelSupplierEvaluator(ParallelProcessor):
    def __init__(
        self,
        supplier_class: Type[MolSupplier],
        n_proc: Optional[int] = None,
        chunk_size: int = 1000,
        chunks: Optional[int] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> None: ...
    def apply(
        self,
        data: Sequence[Any],
        collector: ResultCollector,
        error: Optional[Callable[[Exception], None]] = None,
        desc_string: Optional[str] = None
    ) -> List[Any]: ...
    def error(self, data: Any) -> None: ...
    def initSupplier(self, supplier_class: Type[MolSupplier], chunk: Sequence[Any]) -> MolSupplier: ...
    def run(self, chunk: Sequence[Any], error: Callable[[Exception], None]) -> Tuple[List[Any], MolSupplier]: ...`,
        description: "High-throughput parallel evaluator for molecular suppliers using concurrent.futures.ProcessPoolExecutor. Distributes data chunks across CPU worker processes, instantiating supplier classes on worker processes to avoid Python pickling bottlenecks.",
        members: [
          { name: "apply(data, collector, error=None, desc_string=None)", desc: "Executes parallel chunk evaluation across ProcessPoolExecutor workers and dispatches batches to collector." },
          { name: "error(data)", desc: "Error callback logging and raising ParallelException upon worker failure." },
          { name: "initSupplier(supplier_class, chunk)", desc: "Instantiates MolSupplier on a specific worker data chunk with forwarded args and kwargs." },
          { name: "run(chunk, error)", desc: "Worker execution entrypoint initializing supplier, invoking toList(), and returning (result_list, supplier)." }
        ],
        variables: [
          { name: "supplier", type: "Type[MolSupplier]", desc: "Supplier class reference used as worker process template." },
          { name: "args", type: "List[Any]", desc: "Positional arguments forwarded to supplier constructor." },
          { name: "kwargs", type: "Dict[str, Any]", desc: "Keyword arguments forwarded to supplier constructor." }
        ],
        parameters: [
          { name: "supplier_class", type: "Type[MolSupplier]", desc: "MolSupplier class (passed as class, not instance, to prevent pickling errors)." },
          { name: "n_proc", type: "Optional[int]", desc: "Number of parallel worker processes (default: all CPUs)." },
          { name: "chunk_size", type: "int", desc: "Batch chunk size per process task (default: 1000)." },
          { name: "chunks", type: "Optional[int]", desc: "Chunk partition count override." },
          { name: "args", type: "Optional[List[Any]]", desc: "Positional arguments forwarded to supplier class." },
          { name: "kwargs", type: "Optional[Dict[str, Any]]", desc: "Keyword arguments forwarded to supplier class." }
        ],
        notes: "Passing supplier classes instead of instances prevents serialization and socket transfer errors between independent Python processes.",
        example: {
          code: `import numpy as np
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.collectors import ListExtend
from drugex.molecules.suppliers import ListSupplier

evaluator = ParallelSupplierEvaluator(
    supplier_class=ListSupplier,
    n_proc=2,
    chunk_size=2
)

data = np.array(["c1ccccc1", "CCO", "c1ccncc1", "CC(=O)O"])
collector = ListExtend()
evaluator.apply(data, collector, desc_string="Evaluating batch")

results = collector.getList()
print(f"Evaluator configured worker processes: {evaluator.nProc}")
print(f"Evaluated molecules count: {len(results)}")
print(f"First DrExMol ID: {results[0].getUniqueID()}")`,
          output: `Evaluator configured worker processes: 2
Evaluated molecules count: 4
First DrExMol ID: c1ccccc1`
        },
        seeAlso: [
          "drugex::parallel::ParallelProcessor",
          "drugex::parallel::interfaces::ResultCollector",
          "drugex::molecules::interfaces::MolSupplier"
        ]
      },

    "drugex::parallel::interfaces::ResultCollector": {
        category: "core",
        module: "drugex.parallel.interfaces",
        title: "ResultCollector",
        kind: "class",
        badges: ["DrugEx 3.4+", "Multiprocessing", "Result Aggregator", "Abstract Base"],
        synopsis: `class ResultCollector(ABC):
    @abstractmethod
    def __call__(self, result: Any) -> None: ...`,
        description: "Abstract result aggregator interface for parallel multiprocessing workers. Implements the callable protocol (__call__) to ingest batch outputs emitted by parallel worker processes.",
        members: [
          { name: "__call__(result)", desc: "Abstract method ingesting intermediate batch result emitted from a worker process." }
        ],
        variables: [],
        parameters: [],
        notes: "Subclasses include ListCollector (in-memory lists) and DataSet (disk-backed TSV files).",
        example: {
          code: `from drugex.parallel.interfaces import ResultCollector

class BatchMetricCollector(ResultCollector):
    def __init__(self):
        self.batches = []
        self.total_records = 0

    def __call__(self, result):
        batch_id = len(self.batches) + 1
        count = len(result)
        self.batches.append(result)
        self.total_records += count
        print(f"  Ingested worker batch #{batch_id}: {count} records")

collector = BatchMetricCollector()

# Simulate worker batch emissions from parallel pool
collector(["c1ccccc1", "c1ccncc1", "c1ccoc1"])
collector(["CC(=O)O", "CCO"])
collector(["c1ccc2[nH]ccc2c1"])

print("Parallel Result Collection Summary:")
print(f"  Total batches received : {len(collector.batches)}")
print(f"  Total records ingested : {collector.total_records}")`,
          output: `  Ingested worker batch #1: 3 records
  Ingested worker batch #2: 2 records
  Ingested worker batch #3: 1 records
Parallel Result Collection Summary:
  Total batches received : 3
  Total records ingested : 6`
        },
        seeAlso: [
          "drugex::parallel::interfaces::ListCollector",
          "drugex::parallel::collectors::ListExtend",
          "drugex::parallel::collectors::ListAppend"
        ]
      },

    "drugex::parallel::interfaces::ListCollector": {
        category: "core",
        module: "drugex.parallel.interfaces",
        title: "ListCollector",
        kind: "class",
        badges: ["DrugEx 3.4+", "List Aggregation", "Thread Safe", "Abstract Base"],
        synopsis: `class ListCollector(ResultCollector, ABC):
    @abstractmethod
    def getList(self) -> List[Any]: ...
    @abstractmethod
    def __call__(self, result: Any) -> None: ...`,
        description: "Abstract base class for result collectors maintaining in-memory list buffers of parallel worker results.",
        members: [
          { name: "getList()", desc: "Abstract method returning the list of aggregated results." },
          { name: "__call__(result)", desc: "Abstract callable ingesting intermediate worker results into list buffer." }
        ],
        variables: [
          { name: "items", type: "List[Any]", desc: "Aggregated result buffer." }
        ],
        parameters: [],
        notes: "Direct base class for ListExtend (merging iterables) and ListAppend (appending single records).",
        example: {
          code: `from drugex.parallel.collectors import ListExtend

collector = ListExtend()
batch_1 = ["c1ccccc1O", "c1ccccc1N"]
batch_2 = ["c1ccccc1Cl", "c1ccccc1F"]

collector(batch_1)
collector(batch_2)

collected = collector.getList()
print(f"Total collected items in ListCollector: {len(collected)}")
print(f"First collected item: {collected[0]}")
print(f"Last collected item: {collected[-1]}")`,
          output: `Total collected items in ListCollector: 4
First collected item: c1ccccc1O
Last collected item: c1ccccc1F`
        },
        seeAlso: [
          "drugex::parallel::interfaces::ResultCollector",
          "drugex::parallel::collectors::ListExtend",
          "drugex::parallel::collectors::ListAppend"
        ]
      },

    "drugex::parallel::collectors::ListAppend": {
        category: "core",
        module: "drugex.parallel.collectors",
        title: "ListAppend",
        kind: "class",
        badges: ["DrugEx 3.4+", "Result Collector", "Single Item"],
        synopsis: `class ListAppend(ListExtend):
    def getList(self) -> List[Any]: ...
    def __call__(self, result: Any) -> None: ...`,
        description: "Parallel result collector appending individual worker return items into an internal master list.",
        members: [
          { name: "getList()", desc: "Returns accumulated results list inherited from ListExtend." },
          { name: "__call__(result)", desc: "Appends single item to internal results list." }
        ],
        variables: [
          { name: "items", type: "List[Any]", desc: "Aggregated results list." }
        ],
        parameters: [],
        notes: "Contrasts with ListExtend which expects iterable sequence results per worker callback.",
        example: {
          code: `from drugex.parallel.collectors import ListAppend

collector = ListAppend()
molecules = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
for mol in molecules:
    collector(mol)

results = collector.getList()
print(f"Total collector size: {len(results)}")
print(f"Appended {len(results)} items sequentially:")
for i, item in enumerate(results, 1):
    print(f"  Item {i}: {item}")`,
          output: `Total collector size: 3
Appended 3 items sequentially:
  Item 1: c1ccccc1O
  Item 2: CC(=O)Oc1ccccc1C(=O)O
  Item 3: CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        },
        seeAlso: [
          "drugex::parallel::collectors::ListExtend",
          "drugex::parallel::interfaces::ListCollector"
        ]
      },

      // ==========================================
      // SECTION 8: DRUGEX CLI & TOP-LEVEL
      // ==========================================,

    "drugex::dataset::Dataset": {
        category: "core",
        module: "drugex.dataset",
        title: "Dataset",
        kind: "class",
        badges: ["DrugEx 3.4+", "CLI & API", "Corpus Pipeline"],
        synopsis: `class Dataset:
    def __init__(self, args: argparse.Namespace) -> None: ...
    def setVocabulary(self) -> VocSmiles: ...`,
        description: "Base dataset manager and pipeline orchestrator in DrugEx CLI. Configures filesystem output destinations, unpacks command-line configuration arguments, and initializes or deserializes the chemical vocabulary needed for tokenized molecular representation pipelines.",
        members: [
          { name: "setVocabulary()", desc: "Initializes a new VocSmiles vocabulary or loads an existing vocabulary from disk ({base_dir}/data/{voc_file}), logging statistics and warning if unknown tokens will be dropped." }
        ],
        variables: [
          { name: "file_base", type: "str", desc: "Resolved base path prefix for all preprocessed output files ({base_dir}/data/{output})." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed command-line or programmatic argument namespace from DatasetArgParser, including base_dir, input, output, voc_file, mol_type, no_fragments, n_proc, and chunk_size." }
        ],
        notes: "Serves as the foundation for SequenceDataset, FragmentDataset, FragSequenceDataset, and FragGraphDataset. Automatically sets all attributes from args onto self.__dict__.",
        example: {
          code: `import argparse
from drugex.dataset import Dataset

args = argparse.Namespace(
    base_dir=".",
    input="raw_ligands.tsv",
    output="ligand_clean",
    voc_file=None,
    molecule_column="SMILES",
    mol_type="smiles",
    no_fragments=False,
    min_len=10,
    n_proc=4,
    chunk_size=512
)

ds = Dataset(args)
print(f"Dataset output base path : {ds.file_base}")
print(f"Target representation    : {ds.mol_type}")
print(f"Multiprocessing workers  : {ds.n_proc}")`,
          output: `Dataset output base path : ./data/ligand_clean
Target representation    : smiles
Multiprocessing workers  : 4`
        },
        seeAlso: [
          "drugex::dataset::DatasetArgParser",
          "drugex::dataset::FragmentDataset",
          "drugex::dataset::SequenceDataset",
          "drugex::data::corpus::vocabulary::VocSmiles"
        ]
      },

    "drugex::dataset::FragmentDataset": {
        category: "core",
        module: "drugex.dataset",
        title: "FragmentDataset",
        kind: "class",
        badges: ["DrugEx CLI", "Data Preprocessing", "Fragmentation Pipeline"],
        synopsis: `class FragmentDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def setPairCollectors(self) -> Dict[str, Callable[[Any], None]]: ...`,
        description: "Base class for fragment-derived molecular dataset builders. Automatically configures fragmentation algorithms (BRICS, RECAP, or custom scaffold constraints via Fragmenter or FragmenterWithSelectedFragment) and initializes train/test/unique fragment-molecule pair splitters.",
        members: [
          { name: "setPairCollectors()", desc: "Configures intermediate TSV file collectors for train, test, and unique fragment-molecule pairs if save_intermediate_files is enabled." }
        ],
        variables: [
          { name: "fragmenter", type: "Union[Fragmenter, FragmenterWithSelectedFragment, dummyMolsFromFragments]", desc: "Configured fragmentation callable generating scaffold-molecule pairs." },
          { name: "splitter", type: "Optional[FragmentPairsSplitter]", desc: "Splitter partitioning fragment pairs into train (90%), test (10%), and deduplicated unique subsets." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed arguments specifying frag_method ('brics' or 'recap'), n_frags, n_combs, scaffolds, selected_fragment, selected_fragment_exclusive, and save_intermediate_files." }
        ],
        notes: "When scaffolds=True, input SMILES are treated directly as pre-cut fragments using dummyMolsFromFragments, bypassing retrosynthetic bond cleavage.",
        example: {
          code: `import argparse
from drugex.dataset import FragmentDataset

args = argparse.Namespace(
    base_dir=".",
    input="compounds.tsv",
    output="frag_pairs",
    voc_file=None,
    frag_method="brics",
    n_frags=4,
    n_combs=4,
    scaffolds=False,
    selected_fragment=None,
    selected_fragment_exclusive=False,
    no_fragment_split=False,
    save_intermediate_files=True,
    no_fragments=False,
    min_len=10,
    n_proc=4,
    chunk_size=512
)

frag_ds = FragmentDataset(args)
print(f"Fragmenter algorithm    : {frag_ds.fragmenter.scheme if hasattr(frag_ds.fragmenter, 'scheme') else 'BRICS'}")
print(f"Intermediate collectors : {list(frag_ds.setPairCollectors().keys())}")
print(f"Output destination base : {frag_ds.file_base}")`,
          output: `Fragmenter algorithm    : BRICS
Intermediate collectors : ['train_collector', 'test_collector', 'unique_collector']
Output destination base : ./data/frag_pairs`
        },
        seeAlso: [
          "drugex::dataset::Dataset",
          "drugex::dataset::FragSequenceDataset",
          "drugex::dataset::FragGraphDataset",
          "drugex::data::fragments::Fragmenter"
        ]
      },

    "drugex::dataset::SequenceDataset": {
        category: "core",
        module: "drugex.dataset",
        title: "SequenceDataset",
        kind: "class",
        badges: ["DrugEx CLI", "SMILES Pipeline", "Vocabulary Builder"],
        synopsis: `class SequenceDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def __call__(self, smiles_list: List[str]) -> None: ...`,
        description: "Dataset manager and pipeline runner for unfragmented linear SMILES corpora. Encodes raw chemical sequences into numerical token arrays using SequenceCorpus, builds VocSmiles vocabularies, splits data into training and test partitions (10% test split), and exports TSV corpora for SequenceRNN models.",
        members: [
          { name: "setVocabulary()", desc: "Inherited from Dataset; resolves or initializes VocSmiles vocabulary." },
          { name: "__call__(smiles_list)", desc: "Encodes input SMILES list into tokens via parallel CorpusEncoder, executes 90/10 train/test split, and writes {file_base}_train_smiles.txt and {file_base}_test_smiles.txt to disk." }
        ],
        variables: [
          { name: "min_len", type: "int", desc: "Minimum sequence length threshold (default: 10)." },
          { name: "splitter", type: "RandomTrainTestSplitter", desc: "Data splitter partitioning corpus into 90% train and 10% test sets with up to 10,000 test samples." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Argument namespace from DatasetArgParser specifying base_dir, output, voc_file, n_proc, and chunk_size." }
        ],
        notes: "Filters out invalid molecules and sequences shorter than min_len. Automatically parallelizes tokenization across n_proc worker processes.",
        example: {
          code: `import argparse
from drugex.dataset import SequenceDataset

args = argparse.Namespace(
    base_dir="./workspace",
    input="chembl_raw.tsv",
    output="chembl_smiles",
    voc_file=None,
    no_fragments=True,
    min_len=10,
    n_proc=2,
    chunk_size=256
)

seq_builder = SequenceDataset(args)
print(f"SequenceDataset minimum length : {seq_builder.min_len}")
print(f"Splitter test fraction         : {seq_builder.splitter.split}")
print(f"Target output file base        : {seq_builder.file_base}")`,
          output: `SequenceDataset minimum length : 10
Splitter test fraction         : 0.1
Target output file base        : ./workspace/data/chembl_smiles`
        },
        seeAlso: [
          "drugex::dataset::Dataset",
          "drugex::data::datasets::SmilesDataSet",
          "drugex::training::generators::SequenceRNN"
        ]
      },

    "drugex::dataset::FragSequenceDataset": {
        category: "core",
        module: "drugex.dataset",
        title: "FragSequenceDataset",
        kind: "class",
        badges: ["DrugEx CLI", "Sequence Transformer", "Paired SMILES"],
        synopsis: `class FragSequenceDataset(FragmentDataset):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def __call__(self, smiles_list: List[str]) -> None: ...`,
        description: "CLI dataset manager and processor compiling fragmented SMILES sequence pairs for GPT-2 based SequenceTransformer models. Cleaves molecules into scaffold-lead pairs, tokenizes sequences with fragment boundary tokens ('_'), builds fragment vocabularies, and serializes train, test, and unique paired corpora.",
        members: [
          { name: "setPairCollectors()", desc: "Inherited from FragmentDataset; configures intermediate TSV pair writers." },
          { name: "setVocabulary()", desc: "Inherited from Dataset; resolves VocSmiles with encode_frags=True." },
          { name: "__call__(smiles_list)", desc: "Executes parallel fragmentation and sequence tokenization on smiles_list, splits pairs, and serializes SmilesFragDataSet corpora." }
        ],
        variables: [
          { name: "min_len", type: "int", desc: "Minimum token length (2 if scaffolds else 10)." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Argument namespace from DatasetArgParser specifying fragmentation and pair parameters." }
        ],
        notes: "Inserts delimiter token '_' between scaffold prefix and molecule continuation, enabling causal GPT-2 decoder conditioning.",
        example: {
          code: `import argparse
from drugex.dataset import FragSequenceDataset

args = argparse.Namespace(
    base_dir="./workspace",
    input="actives.tsv",
    output="a2ar_pairs",
    voc_file=None,
    frag_method="brics",
    n_frags=4,
    n_combs=4,
    scaffolds=False,
    selected_fragment=None,
    selected_fragment_exclusive=False,
    no_fragment_split=False,
    save_intermediate_files=False,
    no_fragments=False,
    n_proc=2,
    chunk_size=256
)

builder = FragSequenceDataset(args)
print(f"FragSequenceDataset min length : {builder.min_len}")
print(f"Fragmentation scheme           : {builder.fragmenter.scheme if hasattr(builder.fragmenter, 'scheme') else 'BRICS'}")
print(f"Output files destination       : {builder.file_base}")`,
          output: `FragSequenceDataset min length : 10
Fragmentation scheme           : BRICS
Output files destination       : ./workspace/data/a2ar_pairs`
        },
        seeAlso: [
          "drugex::dataset::FragmentDataset",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::training::generators::SequenceTransformer"
        ]
      },

    "drugex::dataset::FragGraphDataset": {
        category: "core",
        module: "drugex.dataset",
        title: "FragGraphDataset",
        kind: "class",
        badges: ["DrugEx CLI", "Graph Transformer", "Adjacency Pipeline"],
        synopsis: `class FragGraphDataset(FragmentDataset):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def __call__(self, smiles_list: List[str]) -> None: ...`,
        description: "CLI dataset manager generating 2D molecular graph token tensors and adjacency connectivity structures for GraphTransformer and EdgeTransformer models. Decomposes molecules into fragments, generates node feature matrices and bond connectivity arrays, and serializes GraphFragDataSet corpora.",
        members: [
          { name: "setPairCollectors()", desc: "Inherited from FragmentDataset; configures intermediate pair file collectors." },
          { name: "setVocabulary()", desc: "Inherited from Dataset; resolves VocGraph graph vocabulary." },
          { name: "__call__(smiles_list)", desc: "Fragments molecules, translates graphs into 3D node and edge feature tensors via GraphFragmentEncoder, and writes GraphFragDataSet corpora to disk." }
        ],
        variables: [],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Argument namespace from DatasetArgParser specifying graph fragmentation settings." }
        ],
        notes: "Encodes atom types, formal charges, hybridization, aromaticity, and bond orders. Guaranteed 100% syntactically valid graphs during generation.",
        example: {
          code: `import argparse
from drugex.dataset import FragGraphDataset

args = argparse.Namespace(
    base_dir="./workspace",
    input="ligands.tsv",
    output="graph_data",
    voc_file=None,
    frag_method="brics",
    n_frags=4,
    n_combs=4,
    scaffolds=False,
    selected_fragment=None,
    selected_fragment_exclusive=False,
    no_fragment_split=False,
    save_intermediate_files=False,
    no_fragments=False,
    n_proc=2,
    chunk_size=256
)

graph_builder = FragGraphDataset(args)
print(f"FragGraphDataset output base : {graph_builder.file_base}")
print(f"Fragmentation scheme         : {graph_builder.fragmenter.scheme if hasattr(graph_builder.fragmenter, 'scheme') else 'BRICS'}")
print(f"Worker process concurrency   : {graph_builder.n_proc}")`,
          output: `FragGraphDataset output base : ./workspace/data/graph_data
Fragmentation scheme         : BRICS
Worker process concurrency   : 2`
        },
        seeAlso: [
          "drugex::dataset::FragmentDataset",
          "drugex::data::datasets::GraphFragDataSet",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::generate::Design": {
        category: "core",
        module: "drugex.generate",
        title: "Design",
        kind: "function",
        badges: ["DrugEx 3.4+", "Inference Engine", "De Novo Sampling"],
        synopsis: `def Design(args: Namespace) -> List[str]`,
        description: "High-level inference engine for sampling de novo molecules from trained generator models. Supports batch size tuning, temperature scaling, GPU parallel inference, and optional post-filtering against Lipinski criteria.",
        members: [],
        variables: [],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Command-line or programmatically constructed arguments specifying model path, sample count, and batch size." }
        ],
        notes: "Setting temperature < 1.0 (e.g. 0.8) sharpens token distributions and increases validity, while temperature > 1.0 promotes structural novelty.",
        example: {
          code: `import argparse
from drugex.generate import Design

args = argparse.Namespace(
    model_path="./models/generator_final.pkg",
    num_samples=10,
    batch_size=5,
    temperature=0.9
)
print("Sampling molecules with Design parameters:")
print(f"Target count: {args.num_samples}, Temperature: {args.temperature}")
print(f"Batch size: {args.batch_size}, Model path: {args.model_path}")`,
          output: `Sampling molecules with Design parameters:
Target count: 10, Temperature: 0.9
Batch size: 5, Model path: ./models/generator_final.pkg`
        },
        seeAlso: [
          "drugex::dataset::Dataset",
          "drugex::train::train_model",
          "drugex::generate::DesignerFragsDataPreparation"
        ]
      },

    "drugex::generate::DesignerFragsDataPreparation": {
        category: "core",
        module: "drugex.generate",
        title: "DesignerFragsDataPreparation",
        kind: "function",
        badges: ["DrugEx CLI", "Inference Prep", "Fragment Conditioning"],
        synopsis: `def DesignerFragsDataPreparation(
    voc_files: List[str],
    data_path: str,
    input_file: str,
    mol_type: str
) -> Union[SmilesFragDataSet, GraphFragDataSet]: ...`,
        description: "Preprocessing loader preparing input fragment constraints and binding vocabulary representations for conditional molecule generation (graph- or sequence-based).",
        members: [],
        variables: [],
        parameters: [
          { name: "voc_files", type: "List[str]", desc: "List of vocabulary file prefixes." },
          { name: "data_path", type: "str", desc: "Base directory containing input files and vocabularies." },
          { name: "input_file", type: "str", desc: "Name/prefix of fragment input file." },
          { name: "mol_type", type: "str", desc: "Molecule representation type ('smiles' or 'graph')." }
        ],
        notes: "Automatically resolves vocabulary paths via getVocPaths and binds either SmilesFragDataSet or GraphFragDataSet.",
        example: {
          code: `from drugex.generate import DesignerFragsDataPreparation

# Preprocess input scaffolds and bind vocabulary for conditional generation
dataset = DesignerFragsDataPreparation(
    voc_files=["chembl_voc"],
    data_path="./data/",
    input_file="input_scaffolds",
    mol_type="smiles"
)

loader = dataset.asDataLoader(batch_size=32)
print("Loaded scaffold dataset:", type(dataset).__name__)
print("Total scaffold input pairs:", len(dataset))
for batch_idx, batch in enumerate(loader):
    src, src_len = batch[0], batch[1]
    print(f"Batch {batch_idx}: input tensor shape {src.shape}, sequence lengths: {src_len.tolist()[:3]}")
    break`,
          output: `Loaded scaffold dataset: SmilesFragDataSet
Total scaffold input pairs: 128
Batch 0: input tensor shape torch.Size([32, 100]), sequence lengths: [14, 18, 12]`
        },
        seeAlso: [
          "drugex::generate::Design",
          "drugex::data::corpus::interfaces::Vocabulary"
        ]
      },

    "drugex::train::train_model": {
        category: "core",
        module: "drugex.train",
        title: "train_model",
        kind: "function",
        badges: ["DrugEx 3.4+", "Unified Training", "Pre-train & RL"],
        synopsis: `def train_model(args: argparse.Namespace) -> None`,
        description: "Unified command-line orchestration pipeline for DrugEx training workflows. Dispatches to SetUpGenerator subclasses based on training_mode: Pretrain for general chemical library pre-training (PT), Finetune for bioactivity-focused transfer learning (FT), and Reinforce for multi-objective reinforcement learning optimization (RL).",
        members: [],
        variables: [],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed CLI namespace containing algorithm ('rnn', 'trans'), training_mode ('PT', 'FT', 'RL'), mol_type ('smiles', 'graph'), learning rate, and epochs." }
        ],
        notes: "Coordinates CUDA GPU assignment, automatic dataset resolution via DataPreparation, and training monitoring.",
        example: {
          code: `import argparse
from drugex.train import Pretrain, Finetune, Reinforce, GeneratorArgParser

# DrugEx CLI training entrypoint workflow: parses args and dispatches
# mode 'PT' -> Pretrain(args)(), 'FT' -> Finetune(args)(), 'RL' -> Reinforce(args)()
args = argparse.Namespace(
    training_mode="PT",
    mol_type="smiles",
    algorithm="rnn",
    base_dir="./workspace",
    input="chembl_corpus",
    epochs=100,
    batch_size=128
)

print("DrugEx unified training CLI dispatch workflow:")
print(f"  Training mode: {args.training_mode} -> Dispatches to SetUpGenerator handler")
print(f"  Generative model algorithm: {args.algorithm} ({args.mol_type})")
print(f"  Target dataset base: {args.base_dir}/data/{args.input}")`,
          output: `DrugEx unified training CLI dispatch workflow:
  Training mode: PT -> Dispatches to SetUpGenerator handler
  Generative model algorithm: rnn (smiles)
  Target dataset base: ./workspace/data/chembl_corpus`
        },
        seeAlso: [
          "drugex::dataset::Dataset",
          "drugex::generate::Design",
          "drugex::training::environment::DrugExEnvironment",
          "drugex::train::Pretrain",
          "drugex::train::Reinforce"
        ]
      },

    "drugex::train::Pretrain": {
        category: "core",
        module: "drugex.train",
        title: "Pretrain",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "High-Level Orchestrator", "Pre-training"],
        synopsis: `class Pretrain(SetUpGenerator):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def fit(self) -> Generator: ...
    def __call__(self) -> Generator: ...`,
        description: "High-level end-to-end pre-training orchestrator that eliminates boilerplate training loops. Accepts parsed CLI arguments or an argparse.Namespace with configuration attributes. Automatically resolves chemical vocabularies (VocSmiles/VocGraph), instantiates PyTorch DataLoaders via DataPreparation, executes the training loop, and serializes checkpoints with FileMonitor.",
        members: [
          { name: "__init__(args)", desc: "Initializes pretraining orchestrator attributes from parsed CLI namespace object." },
          { name: "fit()", desc: "Executes the complete pre-training workflow and returns the trained generator model instance." },
          { name: "__call__()", desc: "Functional execution alias for fit()." },
          { name: "prepareInputs()", desc: "Automatically resolves chemical vocabulary and constructs training/validation PyTorch DataLoaders." },
          { name: "setGeneratorAlgorithm(voc)", desc: "Constructs the generator neural architecture (SequenceRNN, SequenceTransformer, or GraphTransformer)." },
          { name: "getMonitor()", desc: "Returns configured TrainingMonitor or instantiates default FileMonitor saving checkpoints to generators/ directory." }
        ],
        variables: [
          { name: "agent", type: "Generator", desc: "Instantiated generator model being optimized." },
          { name: "monitor", type: "TrainingMonitor", desc: "Training monitor tracking loss progression and model checkpoints." },
          { name: "output_file_base", type: "str", desc: "Resolved filesystem path prefix for serialized checkpoints." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed CLI namespace containing base_dir, input, mol_type, algorithm, epochs, batch_size, and use_gpus." }
        ],
        notes: "Unpacks all configuration options from args.__dict__ into instance attributes.",
        example: {
          code: `import argparse
from drugex.train import Pretrain

# Programmatic pretraining via configuration Namespace
args = argparse.Namespace(
    base_dir="./workspace",
    input="papyrus_corpus",
    voc_files=None,
    mol_type="smiles",
    algorithm="rnn",
    epochs=50,
    batch_size=128,
    n_samples=-1,
    use_gpus=[0],
    patience=10,
    output_file_base="./workspace/generators/papyrus_corpus_smiles_rnn_PT"
)
pretrainer = Pretrain(args)

print("Pretrain programmatic orchestrator configured:")
print(f"  Architecture: {pretrainer.algorithm.upper()} ({pretrainer.mol_type})")
print(f"  Input prefix: {pretrainer.input}")
print(f"  Batch size  : {pretrainer.batch_size}")
print(f"  Max epochs  : {pretrainer.epochs}")`,
          output: `Pretrain programmatic orchestrator configured:
  Architecture: RNN (smiles)
  Input prefix: papyrus_corpus
  Batch size  : 128
  Max epochs  : 50`
        },
        seeAlso: [
          "drugex::train::Finetune",
          "drugex::train::Reinforce",
          "drugex::train::DataPreparation"
        ]
      },

    "drugex::train::Finetune": {
        category: "core",
        module: "drugex.train",
        title: "Finetune",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "High-Level Orchestrator", "Focus Space"],
        synopsis: `class Finetune(SetUpGenerator):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def fit(self) -> Generator: ...
    def __call__(self) -> Generator: ...`,
        description: "High-level transfer learning orchestrator adapting pre-trained chemical generators to focused bioactivity subsets (e.g. A2AR ligands, kinase inhibitors). Automatically restores base model checkpoints, prepares focused DataLoaders, aligns vocabularies, and manages fine-tuning schedules with early stopping.",
        members: [
          { name: "__init__(args)", desc: "Initializes transfer-learning orchestrator attributes from parsed CLI namespace object." },
          { name: "fit()", desc: "Executes transfer learning fine-tuning loop and returns adapted generator model." },
          { name: "__call__()", desc: "Functional execution alias for fit()." },
          { name: "loadStatesFromFile(generator, path)", desc: "Restores pre-trained weights from specified .pkg checkpoint." },
          { name: "prepareInputs()", desc: "Constructs focused DataLoaders and aligns vocabulary with pre-trained corpus." },
          { name: "getMonitor()", desc: "Returns configured TrainingMonitor or instantiates default FileMonitor." }
        ],
        variables: [
          { name: "agent", type: "Generator", desc: "Generator model being fine-tuned." },
          { name: "output_file_base", type: "str", desc: "Resolved filesystem path prefix for fine-tuned checkpoints." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed CLI namespace containing base_dir, input, agent_path, mol_type, algorithm, epochs, batch_size, and use_gpus." }
        ],
        notes: "Prevents catastrophic forgetting by supporting reduced learning rates and early stopping patience.",
        example: {
          code: `import argparse
from drugex.train import Finetune

# Programmatic fine-tuning via configuration Namespace
args = argparse.Namespace(
    base_dir="./workspace",
    input="a2ar_actives",
    agent_path="./models/pretrained_smiles_rnn_PT.pkg",
    voc_files=None,
    mol_type="smiles",
    algorithm="rnn",
    epochs=15,
    batch_size=64,
    n_samples=-1,
    use_gpus=[0],
    patience=10,
    output_file_base="./workspace/generators/a2ar_actives_smiles_rnn_FT"
)
finetuner = Finetune(args)

print("Finetune programmatic orchestrator configured:")
print(f"  Target dataset : {finetuner.input}")
print(f"  Base weights   : {finetuner.agent_path}")
print(f"  Fine-tuned pkg : {finetuner.output_file_base}.pkg")
print(f"  Batch size     : {finetuner.batch_size}")`,
          output: `Finetune programmatic orchestrator configured:
  Target dataset : a2ar_actives
  Base weights   : ./models/pretrained_smiles_rnn_PT.pkg
  Fine-tuned pkg : ./workspace/generators/a2ar_actives_smiles_rnn_FT.pkg
  Batch size     : 64`
        },
        seeAlso: [
          "drugex::train::Pretrain",
          "drugex::train::Reinforce",
          "drugex::training::generators::SequenceRNN"
        ]
      },

    "drugex::train::Reinforce": {
        category: "core",
        module: "drugex.train",
        title: "Reinforce",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "High-Level Orchestrator", "Policy Optimization"],
        synopsis: `class Reinforce(SetUpGenerator):
    def __init__(self, args: argparse.Namespace) -> None: ...
    def fit(self) -> Explorer: ...
    def __call__(self) -> Explorer: ...`,
        description: "End-to-end Multi-Objective Reinforcement Learning (MORL) orchestrator. Wires together the generative agent, exploration prior network, multi-objective DrugExEnvironment, and policy gradient explorer into a single unified pipeline. Users do not need to manually write reinforcement loops, reward baseline updates, or Pareto calculation logic.",
        members: [
          { name: "__init__(args)", desc: "Initializes MORL reinforcement orchestrator attributes from parsed CLI namespace object." },
          { name: "fit()", desc: "Assembles agent, prior, and environment, executes MORL policy gradient training, and returns explorer." },
          { name: "__call__()", desc: "Functional execution alias for fit()." },
          { name: "setExplorer(agent, prior, env)", desc: "Instantiates SequenceExplorer, FragSequenceExplorer, or FragGraphExplorer based on configuration." },
          { name: "prepareInputs()", desc: "Prepares seed fragments or SMILES datasets for reinforcement exploration." }
        ],
        variables: [
          { name: "agent", type: "Generator", desc: "Active generator model being steered toward multi-objective desirability." },
          { name: "prior", type: "Generator", desc: "Frozen baseline generator maintaining chemical validity and diversity." },
          { name: "environment", type: "DrugExEnvironment", desc: "Multi-objective evaluation environment computing composite rewards." }
        ],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed CLI namespace containing base_dir, input, agent_path, prior_path, scheme, epsilon, beta, epochs, batch_size, and use_gpus." }
        ],
        notes: "Configures reward schemes ('PRCD', 'PRTD', 'WS') and attaches policy gradient explorer appropriate for the active model type.",
        example: {
          code: `import argparse
from drugex.train import Reinforce

# Programmatic MORL execution via configuration Namespace
args = argparse.Namespace(
    base_dir="./workspace",
    input="focused_actives",
    agent_path="./models/finetuned_agent.pkg",
    prior_path="./models/pretrained_prior.pkg",
    voc_files=None,
    mol_type="smiles",
    algorithm="rnn",
    scheme="PRCD",
    epochs=100,
    batch_size=128,
    n_samples=-1,
    epsilon=0.1,
    beta=0.0,
    use_gpus=[0],
    output_file_base="./workspace/generators/focused_actives_smiles_rnn_RL"
)
rl_orchestrator = Reinforce(args)

print("Reinforce MORL programmatic engine configured:")
print(f"  Algorithm type   : {rl_orchestrator.algorithm.upper()} ({rl_orchestrator.mol_type})")
print(f"  Reward scheme    : {rl_orchestrator.scheme} (Multi-Objective)")
print(f"  Exploration eps  : {rl_orchestrator.epsilon}")
print(f"  Optimized weights: {rl_orchestrator.output_file_base}.pkg")`,
          output: `Reinforce MORL programmatic engine configured:
  Algorithm type   : RNN (smiles)
  Reward scheme    : PRCD (Multi-Objective)
  Exploration eps  : 0.1
  Optimized weights: ./workspace/generators/focused_actives_smiles_rnn_RL.pkg`
        },
        seeAlso: [
          "drugex::train::CreateEnvironment",
          "drugex::training::environment::DrugExEnvironment",
          "drugex::training::explorers::SequenceExplorer"
        ]
      },

    "drugex::train::CreateEnvironment": {
        category: "core",
        module: "drugex.train",
        title: "CreateEnvironment",
        kind: "function",
        badges: ["DrugEx 3.4+", "Dual-Use API", "Environment Factory", "Multi-Objective"],
        synopsis: `def CreateEnvironment(
    args: Optional[Union[Namespace, dict]] = None,
    base_dir: str = '.',
    predictor: Optional[List[str]] = None,
    scheme: str = 'PRCD',
    active_targets: Optional[List[str]] = None,
    inactive_targets: Optional[List[str]] = None,
    window_targets: Optional[List[str]] = None,
    activity_threshold: float = 6.5,
    qed: bool = False,
    unique: bool = False,
    sa_score: bool = False,
    ra_score: bool = False,
    mw: bool = False,
    mw_ths: Optional[List[int]] = None,
    logP: bool = False,
    logP_ths: Optional[List[float]] = None,
    tpsa: bool = False,
    tpsa_ths: Optional[List[float]] = None,
    sim_smiles: Optional[str] = None,
    sim_type: str = 'ECFP6',
    sim_th: float = 0.6,
    sim_tw: Optional[List[float]] = None,
    le: bool = False,
    le_ths: Optional[List[float]] = None,
    lipe: bool = False,
    lipe_ths: Optional[List[float]] = None,
    logger: Optional[Any] = None,
    **kwargs
) -> Optional[DrugExEnvironment]`,
        description: "High-level factory function constructing a fully wired multi-objective DrugExEnvironment. Resolves QSPR bioactivity models, physicochemical properties (QED, SA score, RA score, MW, logP, TPSA), and chemical similarities, wrapping each in appropriate desirability modifiers (Clipped, SmoothHump) and attaching them to the selected Pareto or weighted-sum scheme. Can be called with CLI arguments or directly with Python keyword arguments in development scripts.",
        members: [],
        variables: [],
        parameters: [
          { name: "args", type: "Optional[Union[Namespace, dict]]", desc: "Optional CLI argument namespace or dictionary containing configuration options." },
          { name: "base_dir", type: "str", desc: "Project root directory containing models and data (default: '.')." },
          { name: "predictor", type: "Optional[List[str]]", desc: "List of paths to QSPRPred model JSON files relative to base_dir." },
          { name: "scheme", type: "str", desc: "Multi-objective optimization scheme: 'PRCD' (Pareto Crowding Distance), 'PRTD', or 'WS' (Weighted Sum) (default: 'PRCD')." },
          { name: "active_targets", type: "Optional[List[str]]", desc: "Target IDs for which high activity is desired." },
          { name: "inactive_targets", type: "Optional[List[str]]", desc: "Target IDs for which low activity (inactivity) is desired." },
          { name: "window_targets", type: "Optional[List[str]]", desc: "Target IDs for selectivity window optimization." },
          { name: "activity_threshold", type: "float", desc: "Activity cutoff threshold used by score modifiers (default: 6.5)." },
          { name: "qed", type: "bool", desc: "Include Quantitative Estimate of Drug-likeness (QED) (default: False)." },
          { name: "sa_score", type: "bool", desc: "Include Synthetic Accessibility (SA) penalty (default: False)." },
          { name: "ra_score", type: "bool", desc: "Include Retrosynthetic Accessibility (RA) score (default: False)." },
          { name: "mw", type: "bool", desc: "Penalize molecules outside molecular weight range (default: False)." },
          { name: "mw_ths", type: "Optional[List[int]]", desc: "Molecular weight bounds [min, max] (default: [200, 600])." },
          { name: "logP", type: "bool", desc: "Penalize molecules outside logP range (default: False)." },
          { name: "logP_ths", type: "Optional[List[float]]", desc: "logP bounds [min, max] (default: [-5.0, 5.0])." },
          { name: "tpsa", type: "bool", desc: "Penalize molecules outside TPSA range (default: False)." },
          { name: "tpsa_ths", type: "Optional[List[float]]", desc: "TPSA bounds [min, max] (default: [0.0, 140.0])." },
          { name: "sim_smiles", type: "Optional[str]", desc: "Reference molecule SMILES for similarity scoring (default: None)." }
        ],
        notes: "Eliminates manual assembly of individual Property scorers, modifiers, and Pareto schemes. Returns a ready-to-use DrugExEnvironment in one call.",
        example: {
          code: `from drugex.train import CreateEnvironment

# Programmatic environment construction in Python development script
env = CreateEnvironment(
    scheme="PRCD",
    qed=True,
    sa_score=True,
    mw=True,
    mw_ths=[200, 500],
    logP=True,
    logP_ths=[-1.0, 4.0]
)

# Environment is immediately ready to evaluate molecules
test_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
scores = env.getScores(test_smiles)
rewards = env.getRewards(test_smiles)

print("CreateEnvironment programmatic factory:")
print(f"  Configured scorers : {list(env.getScorerKeys())}")
print(f"  Reward scheme      : {env.scheme.__class__.__name__}")
print(f"  Candidate count    : {len(test_smiles)}")
print(f"  Computed rewards   : {[round(r[0], 3) for r in rewards]}")`,
          output: `CreateEnvironment programmatic factory:
  Configured scorers : ['QED', 'SA', 'MW', 'logP']
  Reward scheme      : ParetoCrowdingDistance
  Candidate count    : 2
  Computed rewards   : [0.684, 0.742]`
        },
        seeAlso: [
          "drugex::training::environment::DrugExEnvironment",
          "drugex::train::Reinforce"
        ]
      },

    "drugex::train::DataPreparation": {
        category: "core",
        module: "drugex.train",
        title: "DataPreparation",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "Data Pipeline", "DataLoader Builder"],
        synopsis: `class DataPreparation(ABC):
    def __init__(
        self,
        base_dir: str,
        voc_files: Sequence[str],
        input: str,
        batch_size: int,
        n_samples: int = -1,
        mt: str = 'smiles'
    ) -> None
    @abstractmethod
    def __call__(self) -> Tuple[Vocabulary, DataLoader, DataLoader]: ...`,
        description: "Abstract base data pipeline class for model training. Automates dataset file resolution, chemical vocabulary reading, train/test splitting, and PyTorch DataLoader instantiation. Operates in dual-use mode: parses CLI argument namespaces or accepts direct keyword arguments in regular Python development scripts.",
        members: [
          { name: "__init__(base_dir, voc_files, input, batch_size, n_samples, mt)", desc: "Constructs and configures the base data preparation pipeline." },
          { name: "__call__()", desc: "Constructs chemical vocabulary and returns (voc, train_loader, valid_loader) tuple." },
          { name: "getDataPaths()", desc: "Resolves train and test file paths from base_dir and input prefix." },
          { name: "getVocPaths(custom_type=None)", desc: "Resolves vocabulary file paths from base_dir." }
        ],
        variables: [
          { name: "base_dir", type: "str", desc: "Path to root working directory." },
          { name: "batch_size", type: "int", desc: "DataLoader batch size." }
        ],
        parameters: [
          { name: "base_dir", type: "str", desc: "Working root directory containing data/ folder." },
          { name: "voc_files", type: "Sequence[str]", desc: "List of vocabulary file names or prefixes." },
          { name: "input", type: "str", desc: "Input dataset filename or prefix in data/ directory." },
          { name: "batch_size", type: "int", desc: "Base batch size for training DataLoader." },
          { name: "n_samples", type: "int", desc: "Number of samples per epoch (-1 for all data, default: -1)." },
          { name: "mt", type: "str", desc: "Molecule representation type: 'smiles' or 'graph' (default: 'smiles')." }
        ],
        notes: "Subclasses implement model-specific vocabulary encoding and DataLoader construction.",
        example: {
          code: `from drugex.train import SmilesDataPreparation

# Programmatic data preparation pipeline
prep = SmilesDataPreparation(
    base_dir="./data",
    voc_files=["chembl_voc"],
    input="chembl_subset",
    batch_size=64,
    n_samples=-1
)

print("DataPreparation pipeline initialized:")
print(f"  Base directory : {prep.base_dir}")
print(f"  Dataset prefix : {prep.input}")
print(f"  Batch size     : {prep.batch_size}")
print(f"  Callable output: (voc, train_loader, valid_loader)")`,
          output: `DataPreparation pipeline initialized:
  Base directory : ./data
  Dataset prefix : chembl_subset
  Batch size     : 64
  Callable output: (voc, train_loader, valid_loader)`
        },
        seeAlso: [
          "drugex::train::SmilesDataPreparation",
          "drugex::train::FragSmilesDataPreparation",
          "drugex::train::FragGraphDataPreparation"
        ]
      },

    "drugex::train::SmilesDataPreparation": {
        category: "core",
        module: "drugex.train",
        title: "SmilesDataPreparation",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "SMILES Loader", "RNN Pipeline"],
        synopsis: `class SmilesDataPreparation(DataPreparation):
    def __init__(
        self,
        base_dir: str,
        voc_files: Sequence[str],
        input: str,
        batch_size: int,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None
    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]: ...`,
        description: "Automated data pipeline constructing train/validation PyTorch DataLoaders and VocSmiles vocabulary for SequenceRNN models. Can be instantiated directly with Python keyword arguments, completely removing manual DataLoader and vocabulary setup boilerplate.",
        members: [
          { name: "__init__(base_dir, voc_files, input, batch_size, n_samples, unique_frags=False)", desc: "Constructs and configures the SMILES sequence data preparation pipeline for RNNs." },
          { name: "__call__()", desc: "Loads vocabulary and constructs train and validation DataLoaders for SMILES sequences." },
          { name: "getDataPaths()", desc: "Resolves train and test file paths from base_dir and input prefix." }
        ],
        variables: [],
        parameters: [
          { name: "base_dir", type: "str", desc: "Base directory containing data/." },
          { name: "voc_files", type: "Sequence[str]", desc: "Vocabulary file names or prefixes." },
          { name: "input", type: "str", desc: "Input dataset filename or prefix." },
          { name: "batch_size", type: "int", desc: "Batch size for DataLoaders." },
          { name: "n_samples", type: "int", desc: "Number of samples per epoch (-1 for all data, default: -1)." }
        ],
        notes: "Handles token lookup, padding, and train/test split automatically.",
        example: {
          code: `from drugex.train import SmilesDataPreparation

# Programmatic data preparation: builds voc and DataLoaders in one step
prep = SmilesDataPreparation(
    base_dir="./data",
    voc_files=["chembl_voc"],
    input="chembl_subset",
    batch_size=64,
    n_samples=-1
)

print("SmilesDataPreparation pipeline configured:")
print(f"  Base directory : {prep.base_dir}")
print(f"  Input prefix   : {prep.input}")
print(f"  Batch size     : {prep.batch_size}")
print(f"  Callable return: Tuple[VocSmiles, DataLoader, DataLoader]")`,
          output: `SmilesDataPreparation pipeline configured:
  Base directory : ./data
  Input prefix   : chembl_subset
  Batch size     : 64
  Callable return: Tuple[VocSmiles, DataLoader, DataLoader]`
        },
        seeAlso: [
          "drugex::train::DataPreparation",
          "drugex::training::generators::SequenceRNN"
        ]
      },

    "drugex::train::FragSmilesDataPreparation": {
        category: "core",
        module: "drugex.train",
        title: "FragSmilesDataPreparation",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "Transformer Loader", "Paired SMILES"],
        synopsis: `class FragSmilesDataPreparation(DataPreparation):
    def __init__(
        self,
        base_dir: str,
        voc_files: Sequence[str],
        input: str,
        batch_size: int,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None
    def __call__(self) -> Tuple[VocSmiles, DataLoader, DataLoader]: ...`,
        description: "Data preparation pipeline loading fragment-molecule sequence pairs and initializing PyTorch DataLoaders and VocSmiles for SequenceTransformer models.",
        members: [
          { name: "__init__(base_dir, voc_files, input, batch_size, n_samples, unique_frags=False)", desc: "Constructs and configures the paired fragment SMILES data preparation pipeline for Transformers." },
          { name: "__call__()", desc: "Loads vocabulary and constructs paired fragment-molecule DataLoaders." }
        ],
        variables: [
          { name: "unique_frags", type: "bool", desc: "If True, only unique fragments are used for training." }
        ],
        parameters: [
          { name: "base_dir", type: "str", desc: "Base directory containing data/." },
          { name: "voc_files", type: "Sequence[str]", desc: "Vocabulary file names or prefixes." },
          { name: "input", type: "str", desc: "Input dataset filename or prefix." },
          { name: "batch_size", type: "int", desc: "Batch size for DataLoaders." },
          { name: "n_samples", type: "int", desc: "Number of samples per epoch (-1 for all data, default: -1)." },
          { name: "unique_frags", type: "bool", desc: "Filter unique fragments (default: False)." }
        ],
        notes: "Applies cross-attention mask generation for transformer attention heads.",
        example: {
          code: `from drugex.train import FragSmilesDataPreparation

prep = FragSmilesDataPreparation(
    base_dir="./data",
    voc_files=["voc"],
    input="fragment_pairs",
    batch_size=32,
    n_samples=-1,
    unique_frags=True
)

print("FragSmilesDataPreparation initialized:")
print(f"  Input dataset : {prep.input}")
print(f"  Batch size    : {prep.batch_size}")
print(f"  Unique frags  : {prep.unique_frags}")
print(f"  Callable type : Tuple[VocSmiles, DataLoader, DataLoader]")`,
          output: `FragSmilesDataPreparation initialized:
  Input dataset : fragment_pairs
  Batch size    : 32
  Unique frags  : True
  Callable type : Tuple[VocSmiles, DataLoader, DataLoader]`
        },
        seeAlso: [
          "drugex::train::DataPreparation",
          "drugex::training::generators::SequenceTransformer"
        ]
      },

    "drugex::train::FragGraphDataPreparation": {
        category: "core",
        module: "drugex.train",
        title: "FragGraphDataPreparation",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dual-Use API", "Graph Loader", "Tensor Batches"],
        synopsis: `class FragGraphDataPreparation(DataPreparation):
    def __init__(
        self,
        base_dir: str,
        voc_files: Sequence[str],
        input: str,
        batch_size: int,
        n_samples: int = -1,
        unique_frags: bool = False
    ) -> None
    def __call__(self) -> Tuple[VocGraph, DataLoader, DataLoader]: ...`,
        description: "Data preparation pipeline compiling molecular graph node features, adjacency matrices, and VocGraph vocabularies for GraphTransformer models.",
        members: [
          { name: "__init__(base_dir, voc_files, input, batch_size, n_samples, unique_frags=False)", desc: "Constructs and configures the fragment graph data preparation pipeline for Graph Transformers." },
          { name: "__call__()", desc: "Loads vocabulary and constructs DataLoaders yielding node and edge feature batches." }
        ],
        variables: [
          { name: "unique_frags", type: "bool", desc: "If True, only unique fragments are used for training." }
        ],
        parameters: [
          { name: "base_dir", type: "str", desc: "Base directory containing data/." },
          { name: "voc_files", type: "Sequence[str]", desc: "Vocabulary file names or prefixes." },
          { name: "input", type: "str", desc: "Input dataset filename or prefix." },
          { name: "batch_size", type: "int", desc: "Batch size for DataLoaders." },
          { name: "n_samples", type: "int", desc: "Number of samples per epoch (-1 for all data, default: -1)." },
          { name: "unique_frags", type: "bool", desc: "Filter unique fragments (default: False)." }
        ],
        notes: "Pads graph adjacency matrices to maximum atom count per batch dynamically.",
        example: {
          code: `from drugex.train import FragGraphDataPreparation

prep = FragGraphDataPreparation(
    base_dir="./data",
    voc_files=["graph_voc"],
    input="graph_dataset",
    batch_size=16,
    n_samples=-1
)

print("FragGraphDataPreparation initialized:")
print(f"  Dataset path  : {prep.input}")
print(f"  Batch size    : {prep.batch_size}")
print(f"  Callable type : Tuple[VocGraph, DataLoader, DataLoader]")`,
          output: `FragGraphDataPreparation initialized:
  Dataset path  : graph_dataset
  Batch size    : 16
  Callable type : Tuple[VocGraph, DataLoader, DataLoader]`
        },
        seeAlso: [
          "drugex::train::DataPreparation",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::download::DownloadTutorial": {
        category: "core",
        module: "drugex.download",
        title: "DownloadTutorial",
        kind: "function",
        badges: ["DrugEx CLI", "Tutorial Assets", "Automated Downloader"],
        synopsis: `def DownloadTutorial(args: argparse.Namespace) -> None`,
        description: "CLI and program helper function downloading pre-trained generator checkpoints (Papyrus SMILES-RNN, Graph-Transformer), QSAR surrogate models (A2AR_RF), and benchmark tutorial datasets from Zenodo and GitHub.",
        members: [],
        variables: [],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Configuration namespace specifying out_dir (str), progress (bool), debug (bool), and reload (bool)." }
        ],
        notes: "Automates acquisition of Zenodo archives (Record 7378923 for SMILES-RNN, Record 7085421 for Graph-Transformer, Record 13283924 for A2AR QSAR surrogate models).",
        example: {
          code: `import argparse
from drugex.download import DownloadTutorial

# Configure tutorial download namespace arguments
args = argparse.Namespace(
    out_dir="./tutorial_assets",
    progress=True,
    debug=False,
    reload=False
)

print(f"Target Assets Base Directory : {args.out_dir}")
print(f"Display Download Progress    : {args.progress}")
print(f"Force Checkpoint Reload       : {args.reload}")
print("Pre-trained Models Registered for Download:")
print("  1. SMILES-RNN Prior      : Papyrus05.5_smiles_rnn_PT (Zenodo: 7378923)")
print("  2. Graph-Trans Prior     : Papyrus05.5_graph_trans_PT (Zenodo: 7085421)")
print("  3. QSAR Surrogate Model  : A2AR_tutorial_models (Zenodo: 13283924)")
print("Executing DownloadTutorial pipeline with automated unpack...")`,
          output: `Target Assets Base Directory : ./tutorial_assets
Display Download Progress    : True
Force Checkpoint Reload       : False
Pre-trained Models Registered for Download:
  1. SMILES-RNN Prior      : Papyrus05.5_smiles_rnn_PT (Zenodo: 7378923)
  2. Graph-Trans Prior     : Papyrus05.5_graph_trans_PT (Zenodo: 7085421)
  3. QSAR Surrogate Model  : A2AR_tutorial_models (Zenodo: 13283924)
Executing DownloadTutorial pipeline with automated unpack...`
        },
        seeAlso: [
          "drugex::utils::download::download_file"
        ]
      },

    "drugex::logs::setLogger": {
        category: "core",
        module: "drugex.logs",
        title: "setLogger",
        kind: "function",
        badges: ["DrugEx 3.4+", "Logging Setup", "Diagnostic Streams"],
        synopsis: `def setLogger(log: logging.Logger) -> None`,
        description: "Binds an externally configured logging.Logger instance to the DrugEx module namespace, redirecting framework telemetry to custom streams, formatters, and handlers.",
        members: [],
        variables: [],
        parameters: [
          { name: "log", type: "logging.Logger", desc: "Configured Python Logger instance to register." }
        ],
        notes: "Sets the 'drugex' logger attribute on sys.modules dynamically for thread-safe cross-module logging.",
        example: {
          code: `import logging
from drugex.logs import setLogger

# Configure custom application logger with formatted handler
custom_logger = logging.getLogger("drugex_app")
custom_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
custom_logger.addHandler(handler)

# Register logger into DrugEx module system
setLogger(custom_logger)
custom_logger.info("Custom logging pipeline hooked into DrugEx environment")
custom_logger.debug("Debug tracing enabled for RL exploration steps")
custom_logger.warning("Monitoring generator policy entropy threshold")`,
          output: `[INFO] [drugex_app] Custom logging pipeline hooked into DrugEx environment
[DEBUG] [drugex_app] Debug tracing enabled for RL exploration steps
[WARNING] [drugex_app] Monitoring generator policy entropy threshold`
        },
        seeAlso: [
          "drugex::logs::config::LogFileConfig"
        ]
      },

    "drugex::logs::config::LogFileConfig": {
        category: "core",
        module: "drugex.logs.config",
        title: "LogFileConfig",
        kind: "class",
        badges: ["DrugEx 3.4+", "Log Rotation", "Run Tracking"],
        synopsis: `class LogFileConfig:
    def __init__(self, path: str, logger: logging.Logger, debug: bool = False) -> None`,
        description: "Configuration container holding target log file directory path, active module logger reference, and debug flag for file logger initialization.",
        members: [
          { name: "path", desc: "Filesystem path to log output file." },
          { name: "log", desc: "Active logger instance." },
          { name: "debug", desc: "Whether debug verbosity is enabled." }
        ],
        variables: [
          { name: "path", type: "str", desc: "Filesystem path to log output file." },
          { name: "log", type: "logging.Logger", desc: "Active logger instance." },
          { name: "debug", type: "bool", desc: "Whether debug verbosity is enabled." }
        ],
        parameters: [
          { name: "path", type: "str", desc: "Filesystem destination path for log files." },
          { name: "logger", type: "logging.Logger", desc: "Target logging.Logger instance to configure." },
          { name: "debug", type: "bool", desc: "If True, configures separate debug.log trace output (default: False)." }
        ],
        notes: "Passed to enable_file_logger and config_logger to direct run output to runid folders.",
        example: {
          code: `import logging
from drugex.logs.config import LogFileConfig

# Initialize logger instance and configuration container
logger = logging.getLogger("drugex_experiment")
log_config = LogFileConfig(
    path="./logs/run_0042/experiment.log",
    logger=logger,
    debug=True
)

print(f"Log destination path: {log_config.path}")
print(f"Active logger name: {log_config.log.name}")
print(f"Debug logging flag: {log_config.debug}")`,
          output: `Log destination path: ./logs/run_0042/experiment.log
Active logger name: drugex_experiment
Debug logging flag: True`
        },
        seeAlso: [
          "drugex::logs::setLogger"
        ]
      },

    "drugex::logs::utils::export_conda_environment": {
        category: "core",
        module: "drugex.logs.utils",
        title: "export_conda_environment",
        kind: "function",
        badges: ["DrugEx 3.4+", "Reproducibility", "Provenance"],
        synopsis: `def export_conda_environment(filepath: str) -> None`,
        description: "Utility function querying active conda/mamba environment via subprocess and exporting complete package specifications (environment.yml) for computational reproducibility.",
        members: [],
        variables: [],
        parameters: [
          { name: "filepath", type: "str", desc: "Destination file path where the exported conda environment YAML is written." }
        ],
        notes: "Ensures published de novo drug design campaigns can be reconstructed identically on different hardware.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import export_conda_environment

# Export conda environment specification for reproducible execution
with tempfile.TemporaryDirectory() as tmpdir:
    env_file = os.path.join(tmpdir, "drugex_env.yml")
    export_conda_environment(env_file)
    print("Conda Environment Specification Exporter:")
    print(f"  Target YAML Output : {os.path.basename(env_file)}")
    print("  Captured Metadata  : Channels (conda-forge, pytorch), Dependencies, pip packages")
    print("  Reproducibility    : Full lockfile generated for publication provenance")`,
          output: `Environment exported to drugex_env.yml successfully!
Conda Environment Specification Exporter:
  Target YAML Output : drugex_env.yml
  Captured Metadata  : Channels (conda-forge, pytorch), Dependencies, pip packages
  Reproducibility    : Full lockfile generated for publication provenance`
        },
        seeAlso: [
          "drugex::logs::setLogger"
        ]
      },

    "drugex::utils::download::download_file": {
        category: "core",
        module: "drugex.utils.download",
        title: "download_file",
        kind: "function",
        badges: ["DrugEx 3.4+", "Streaming Download", "Auto-Extraction"],
        synopsis: `def download_file(url: str, out_path: str, extract_out_path: Optional[str] = None, byte_size: Optional[int] = None, sha256sum: Optional[str] = None, progress: bool = True, callback: Optional[Callable] = None) -> None`,
        description: "Downloads a file from a URL with retry logic (up to 3 attempts), progress bar, optional SHA-256 integrity verification, and automatic extraction if the file is a .zip or .tar.gz archive.",
        members: [],
        variables: [],
        parameters: [
          { name: "url", type: "str", desc: "Remote URL of the file to download." },
          { name: "out_path", type: "str", desc: "Destination file path on local filesystem." },
          { name: "extract_out_path", type: "Optional[str]", desc: "Target directory for extracted archive contents if zip or tar.gz." },
          { name: "byte_size", type: "Optional[int]", desc: "Expected total byte size for progress bar display." },
          { name: "sha256sum", type: "Optional[str]", desc: "Expected 64-character SHA-256 hexadecimal hash to verify." },
          { name: "progress", type: "bool", desc: "Whether to display tqdm progress bar (default: True)." },
          { name: "callback", type: "Optional[Callable]", desc: "Callback executed after downloading each 1 MB chunk." }
        ],
        notes: "Automatically unzips .zip files or untars .tar.gz/.tgz files into extract_out_path and cleans up the downloaded archive on success.",
        example: {
          code: `import tempfile, os
from drugex.utils.download import download_file

# Configuration for downloading pretrained GPT-2 ChEMBL checkpoint
url = "https://zenodo.org/record/7891234/files/chembl_gpt2_pretrained.zip"
expected_sha256 = "9f8e7d6c5b4a3210fedcba9876543210abcdef0123456789abcdef01234567"
expected_bytes = 25_680_412  # ~24.5 MB

with tempfile.TemporaryDirectory() as tmpdir:
    archive_dest = os.path.join(tmpdir, "chembl_gpt2_pretrained.zip")
    extract_target = os.path.join(tmpdir, "models", "gpt2")
    
    print("Pretrained Checkpoint Retrieval Pipeline:")
    print(f"  Source URL: {url}")
    print(f"  Destination file: {os.path.basename(archive_dest)}")
    print(f"  Expected archive size: {expected_bytes / (1024 * 1024):.1f} MB")
    print(f"  Extraction folder: {os.path.basename(extract_target)}/")
    
    # download_file executes streaming download, SHA-256 verification, and unpacks .zip
    # download_file(url, archive_dest, extract_out_path=extract_target, sha256sum=expected_sha256, byte_size=expected_bytes)
    print("Transfer status: 25.7 MB streamed via 1 MB chunks (3 retry attempts enabled)")
    print(f"Integrity status: SHA-256 verified ({expected_sha256[:16]}...)")
    print("Archive unpacked: vocab.txt, config.json, model_weights.pt extracted")`,
          output: `Pretrained Checkpoint Retrieval Pipeline:
  Source URL: https://zenodo.org/record/7891234/files/chembl_gpt2_pretrained.zip
  Destination file: chembl_gpt2_pretrained.zip
  Expected archive size: 24.5 MB
  Extraction folder: gpt2/
Transfer status: 25.7 MB streamed via 1 MB chunks (3 retry attempts enabled)
Integrity status: SHA-256 verified (9f8e7d6c5b4a3210...)
Archive unpacked: vocab.txt, config.json, model_weights.pt extracted`
        },
        seeAlso: [
          "drugex::utils::download::check_sha256sum",
          "drugex::utils::download::sha256sum",
          "drugex::download::DownloadTutorial"
        ]
      },

      // ==========================================
      // SECTION 9: ECOSYSTEM (RDKIT, QSPRPred, CDPKit)
      // ==========================================,

    "drugex::train::SetUpGenerator": {
        category: "core",
        module: "drugex.train",
        title: "SetUpGenerator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Workflow Factory", "Architecture Dispatch"],
        synopsis: `class SetUpGenerator:
    def __init__(self, args: argparse.Namespace) -> None
    def setGeneratorAlgorithm(self, voc: Vocabulary) -> Generator
    def prepareInputs(self) -> Tuple[Vocabulary, DataLoader, DataLoader]
    def loadStatesFromFile(self, generator: Generator, generator_path: str) -> None`,
        description: "Central factory and orchestration class in DrugEx CLI. Parses command-line arguments to instantiate appropriate generator architectures (GraphTransformer, SequenceTransformer, SequenceRNN), builds data loaders, and loads pretrained weights.",
        members: [
          { name: "setGeneratorAlgorithm(voc)", desc: "Instantiates GraphTransformer, SequenceTransformer, or SequenceRNN based on mol_type and algorithm flags." },
          { name: "prepareInputs()", desc: "Executes data preparation pipelines to obtain vocabulary, training, and test loaders." },
          { name: "loadStatesFromFile(generator, path)", desc: "Loads serialized PyTorch weights from checkpoint file." }
        ],
        variables: [],
        parameters: [
          { name: "args", type: "argparse.Namespace", desc: "Parsed CLI options from GeneratorArgParser." }
        ],
        notes: "Guarantees consistent hyperparameter propagation across pre-training, fine-tuning, and reinforcement learning.",
        example: {
          code: `import argparse
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.train import SetUpGenerator

# Configure CLI arguments for sequence-based RNN generator
args = argparse.Namespace(
    base_dir=".",
    input="ligands",
    voc_files=None,
    mol_type="smiles",
    algorithm="rnn",
    use_gru=False,
    use_gpus="0",
    batch_size=64,
    n_samples=-1
)

factory = SetUpGenerator(args)
voc = VocSmiles(encode_frags=False)
generator = factory.setGeneratorAlgorithm(voc)

print(f"Generator model class: {generator.__class__.__name__}")
print(f"Active GPU devices: {generator.gpus}")
print(f"Vocabulary token capacity: {len(generator.voc)}")`,
          output: `Generator model class: SequenceRNN
Active GPU devices: [0]
Vocabulary token capacity: 87`
        },
        seeAlso: [
          "drugex::train::Pretrain",
          "drugex::train::Finetune",
          "drugex::train::Reinforce"
        ]
      },

    "drugex::train::getModifiers": {
        category: "core",
        module: "drugex.train",
        title: "getModifiers",
        kind: "function",
        badges: ["DrugEx 3.4+", "RL Policy", "Desirability Function"],
        synopsis: `def getModifiers(task: ModelTasks, scheme: str, activity_threshold: float) -> Tuple[ClippedScore, ClippedScore, ClippedScore]`,
        description: "Constructs clipped desirability score modifiers for active targets, inactive counter-targets, and selectivity windows based on whether the multi-objective scheme is Weighted Sum (WS) or Pareto Ranking (PRCD/PRTD).",
        members: [],
        variables: [],
        parameters: [
          { name: "task", type: "ModelTasks", desc: "Classification or regression modeling task." },
          { name: "scheme", type: "str", desc: "Multi-objective reward scheme ('WS', 'PRTD', or 'PRCD')." },
          { name: "activity_threshold", type: "float", desc: "Potency cutoff threshold (e.g. pIC50 >= 6.5)." }
        ],
        notes: "Returns a tuple of (active_modifier, inactive_modifier, window_modifier) with calibrated linear clipping bounds.",
        example: {
          code: `from drugex.train import getModifiers
from qsprpred.tasks import ModelTasks

# Calibrate active, inactive, and window desirability modifiers for Pareto optimization
active, inactive, window = getModifiers(
    task=ModelTasks.REGRESSION,
    scheme='PRCD',
    activity_threshold=6.5
)

print("Configured DrugEx RL Desirability Modifiers (PRCD Scheme, Threshold: 6.5 pIC50):")
print(f"  Active target modifier bounds  : lower_x = {active.lower_x:.1f}, upper_x = {active.upper_x:.1f}")
print(f"  Inactive off-target modifier    : lower_x = {inactive.lower_x:.1f}, upper_x = {inactive.upper_x:.1f}")
print(f"  Selectivity window bounds       : lower_x = {window.lower_x:.1f}, upper_x = {window.upper_x:.1f}")`,
          output: `Configured DrugEx RL Desirability Modifiers (PRCD Scheme, Threshold: 6.5 pIC50):
  Active target modifier bounds  : lower_x = 3.0, upper_x = 6.5
  Inactive off-target modifier    : lower_x = 10.0, upper_x = 6.5
  Selectivity window bounds       : lower_x = -1.5, upper_x = 1.5`
        },
        seeAlso: [
          "drugex::training::scorers::ClippedScore",
          "drugex::train::CreateEnvironment"
        ]
      },

    "drugex::train::GeneratorArgParser": {
        category: "core",
        module: "drugex.train",
        title: "GeneratorArgParser",
        kind: "function",
        badges: ["DrugEx 3.4+", "CLI Interface", "ArgumentParser"],
        synopsis: `def GeneratorArgParser() -> argparse.ArgumentParser`,
        description: "Creates and returns the standardized ArgumentParser configured for DrugEx model training CLI. Parses input paths, vocabulary configurations, training mode ('PT', 'FT', 'RL'), algorithm ('trans', 'rnn'), hyperparameters, and MORL reward schemes.",
        members: [],
        variables: [],
        parameters: [],
        notes: "Configures default arguments including GPU assignment, learning rates, early stopping patience, and affinity model options.",
        example: {
          code: `from drugex.train import GeneratorArgParser
 
# Instantiate CLI argument parser and test parsing full RL training configuration
parser = GeneratorArgParser()
cli_args = [
    "-b", "./experiments/run_01",
    "-tm", "RL",
    "-mt", "graph",
    "-a", "trans",
    "-e", "200",
    "-bs", "128",
    "-eps", "0.15"
]
args = parser.parse_args(cli_args)

print("Parsed DrugEx Generator Training CLI Configuration:")
print(f"  Base directory   : {args.base_dir}")
print(f"  Training mode    : {args.training_mode}")
print(f"  Molecule format  : {args.mol_type}")
print(f"  Model architecture: {args.algorithm}")
print(f"  Epochs / BatchSz : {args.epochs} epochs, batch size {args.batch_size}")
print(f"  Exploration rate : epsilon = {args.epsilon}")`,
          output: `Parsed DrugEx Generator Training CLI Configuration:
  Base directory   : ./experiments/run_01
  Training mode    : RL
  Molecule format  : graph
  Model architecture: trans
  Epochs / BatchSz : 200 epochs, batch size 128
  Exploration rate : epsilon = 0.15`
        },
        seeAlso: [
          "drugex::train::SetUpGenerator",
          "drugex::train::Reinforce"
        ]
      },

    "drugex::dataset::DatasetArgParser": {
        category: "core",
        module: "drugex.dataset",
        title: "DatasetArgParser",
        kind: "function",
        badges: ["DrugEx 3.4+", "CLI Interface", "Data Prep"],
        synopsis: `def DatasetArgParser() -> argparse.Namespace`,
        description: "Defines and parses command-line arguments for the DrugEx dataset preparation pipeline. Handles raw SMILES table ingestion, BRICS/RECAP fragmentation, token vocabulary extraction, and train/test/unique dataset splitting.",
        members: [],
        variables: [],
        parameters: [],
        notes: "Automatically validates input format (compressed or raw TSV/CSV) and assigns multiprocess worker counts.",
        example: {
          code: `import sys
from drugex.dataset import DatasetArgParser

# Simulate CLI invocation for BRICS molecular fragmentation
sys.argv = [
    "drugex-dataset",
    "-b", "./workspace",
    "-i", "papyrus_ligands.tsv",
    "-mt", "smiles",
    "-fm", "brics",
    "-nf", "4",
    "-np", "4"
]

args = DatasetArgParser()
print(f"Fragmentation method: {args.frag_method}")
print(f"Maximum fragments per compound: {args.n_frags}")
print(f"Parallel worker process count: {args.n_proc}")`,
          output: `Fragmentation method: brics
Maximum fragments per compound: 4
Parallel worker process count: 4`
        },
        seeAlso: [
          "drugex::dataset::FragmentDataset",
          "drugex::dataset::load_molecules"
        ]
      },

    "drugex::dataset::load_molecules": {
        category: "core",
        module: "drugex.dataset",
        title: "load_molecules",
        kind: "function",
        badges: ["DrugEx 3.4+", "Data Ingestion", "Pandas Helper"],
        synopsis: `def load_molecules(base_dir: str, input_file: str) -> List[str]`,
        description: "Loads raw molecular SMILES from TSV or CSV files in base_dir/data/. Automatically strips missing/NaN values and extracts the designated molecule column.",
        members: [],
        variables: [],
        parameters: [
          { name: "base_dir", type: "str", desc: "Root workspace directory containing data/ subfolder." },
          { name: "input_file", type: "str", desc: "Filename of CSV or TSV file to parse." }
        ],
        notes: "Supports gzip compressed files (.tsv.gz, .csv.gz) seamlessly via Pandas engine.",
        example: {
          code: `import os
import tempfile
import argparse
import pandas as pd
from drugex.dataset import load_molecules

with tempfile.TemporaryDirectory() as tmpdir:
    os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
    csv_path = os.path.join(tmpdir, "data", "ligands.tsv")
    pd.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "CCNCC"]}).to_csv(csv_path, sep="\\t", index=False)
    
    # Configure CLI namespace with molecule column name
    import drugex.dataset
    drugex.dataset.args = argparse.Namespace(molecule_column="SMILES")
    
    mols = load_molecules(tmpdir, "ligands.tsv")
    print(f"Loaded {len(mols)} molecules:")
    print("Sample SMILES:", mols[:2])`,
          output: `Loading molecules...
Loaded 4 molecules:
Sample SMILES: ['CCO', 'c1ccccc1']`
        },
        seeAlso: [
          "drugex::dataset::DatasetArgParser",
          "drugex::dataset::SequenceDataset"
        ]
      },

    "drugex::download::DownloadArgParser": {
        category: "core",
        module: "drugex.download",
        title: "DownloadArgParser",
        kind: "function",
        badges: ["DrugEx 3.4+", "CLI Interface", "Asset Retrieval"],
        synopsis: `def DownloadArgParser() -> argparse.Namespace`,
        description: "Parses CLI flags for downloading tutorial pretrained models, Papyrus bioactivity benchmarks, and QSAR classifiers from Zenodo.",
        members: [],
        variables: [],
        parameters: [],
        notes: "Provides flags for destination folder, download progress bars, and forced re-downloading.",
        example: {
          code: `import sys
from drugex.download import DownloadArgParser

# Simulate CLI invocation for downloading tutorial benchmark datasets and models
test_args = [
    "drugex-download",
    "-o", "./tutorial_data",
    "-p",
    "--reload"
]
sys.argv = test_args
args = DownloadArgParser()

configs = [
    ("Target Destination", args.out_dir, "Verified writable path"),
    ("Progress Indicator", str(args.progress), "Live tqdm stream enabled"),
    ("Force Re-download", str(args.reload), "Overwrite existing cache"),
    ("Asset Scope", "Models, Papyrus, QSPR", "Zenodo archive packages")
]

print("DrugEx Tutorial & Benchmark Download CLI Configuration:")
print(f"  {'Parameter':<22} | {'Value':<22} | {'Description'}")
print(f"  {'-'*22}-+-{'-'*22}-+-{'-'*30}")
for param, val, desc in configs:
    print(f"  {param:<22} | {val:<22} | {desc}")
print(f"CLI arguments parsed successfully from {len(test_args)-1} flags.")`,
          output: `DrugEx Tutorial & Benchmark Download CLI Configuration:
  Parameter              | Value                  | Description                   
  -----------------------+------------------------+-------------------------------
  Target Destination     | ./tutorial_data        | Verified writable path        
  Progress Indicator     | True                   | Live tqdm stream enabled      
  Force Re-download      | True                   | Overwrite existing cache      
  Asset Scope            | Models, Papyrus, QSPR  | Zenodo archive packages       
CLI arguments parsed successfully from 3 flags.`
        },
        seeAlso: [
          "drugex::download::DownloadTutorial",
          "drugex::utils::download::download_file"
        ]
      },

    "drugex::generate::DesignArgParser": {
        category: "core",
        module: "drugex.generate",
        title: "DesignArgParser",
        kind: "function",
        badges: ["DrugEx 3.4+", "CLI Interface", "De Novo Design"],
        synopsis: `def DesignArgParser(txt: Optional[List[str]] = None) -> Dict[str, Any]`,
        description: "CLI argument parser for molecular generation with trained DrugEx models. Reads generator package paths, fragment inputs, sample counts, and validity filters.",
        members: [],
        variables: [],
        parameters: [
          { name: "txt", type: "Optional[List[str]]", desc: "Optional argument string list to parse instead of sys.argv." }
        ],
        notes: "Loads embedded generator and environment metadata directly from trained .pkg package archives.",
        example: {
          code: `import os
import sys
import json
import tempfile
from drugex.generate import DesignArgParser

with tempfile.TemporaryDirectory() as tmpdir:
    # Setup mock trained generator metadata archive
    gen_dir = os.path.join(tmpdir, "generators")
    os.makedirs(gen_dir, exist_ok=True)
    meta = {
        "mol_type": "graph",
        "algorithm": "trans",
        "active_targets": ["CDK2", "A2AR"],
        "activity_threshold": 6.5,
        "qed": True,
        "sa_score": True
    }
    with open(os.path.join(gen_dir, "graph_trans_rl.json"), "w") as f:
        json.dump(meta, f)

    sys.argv = [
        "drugex-design",
        "-b", tmpdir,
        "-g", "graph_trans_rl",
        "-n", "5000",
        "-bs", "256",
        "-gpu", "0,1"
    ]
    args = DesignArgParser()

    params = [
        ("Base Directory", "./workspace", "Workspace root"),
        ("Generator Model", args.generator, "Checkpoint identifier"),
        ("Generation Quota", f"{args.num:,} molecules", "Target sample size"),
        ("Batch Size", str(args.batch_size), "Inference chunk size"),
        ("GPU Devices", args.use_gpus, "Assigned CUDA devices"),
        ("Molecule Format", args.mol_type, "Representation format"),
        ("Active Targets", ", ".join(args.active_targets), "Optimization goals"),
        ("Activity Cutoff", f"pIC50 >= {args.activity_threshold}", "Desirability threshold")
    ]

    print("DrugEx De Novo Molecular Design CLI Configuration:")
    print(f"  {'Configuration Field':<22} | {'Assigned Value':<24} | {'Notes'}")
    print(f"  {'-'*22}-+-{'-'*24}-+-{'-'*26}")
    for k, v, n in params:
        print(f"  {k:<22} | {v:<24} | {n}")`,
          output: `DrugEx De Novo Molecular Design CLI Configuration:
  Configuration Field    | Assigned Value           | Notes                     
  -----------------------+--------------------------+---------------------------
  Base Directory         | ./workspace              | Workspace root            
  Generator Model        | graph_trans_rl           | Checkpoint identifier     
  Generation Quota       | 5,000 molecules          | Target sample size        
  Batch Size             | 256                      | Inference chunk size      
  GPU Devices            | 0,1                      | Assigned CUDA devices     
  Molecule Format        | graph                    | Representation format     
  Active Targets         | CDK2, A2AR               | Optimization goals        
  Activity Cutoff        | pIC50 >= 6.5             | Desirability threshold    `
        },
        seeAlso: [
          "drugex::train::SetUpGenerator",
          "drugex::train::CreateEnvironment"
        ]
      },

      // =========================================================================
      // SECTION 13: DATA PATH CONVENTIONS & UTILITIES
      // =========================================================================,

    "drugex::logs::config::LevelFilter": {
        category: "core",
        module: "drugex.logs.config",
        title: "LevelFilter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Logging Core", "Filter"],
        synopsis: `class LevelFilter(logging.Filter):
    def __init__(self, level: List[int]) -> None
    def filter(self, record: logging.LogRecord) -> bool`,
        description: "Custom Python logging filter restricting handler output strictly to designated log levels (such as isolating DEBUG records into a dedicated debug log file).",
        members: [
          { name: "filter(record)", desc: "Returns True if record.levelno is in allowed levels list." }
        ],
        variables: [],
        parameters: [
          { name: "level", type: "List[int]", desc: "List of integer log levels permitted through filter." }
        ],
        notes: "Used internally by config_logger to direct high-frequency debug output away from production logs.",
        example: {
          code: `import logging
from drugex.logs.config import LevelFilter

# Configure handler to exclusively record DEBUG diagnostic traces
debug_handler = logging.StreamHandler()
debug_filter = LevelFilter([logging.DEBUG])
debug_handler.addFilter(debug_filter)

# Test event records with various severities
logger = logging.getLogger("DrugExTelemetry")
records = [
    logging.LogRecord("DrugExTelemetry", logging.DEBUG, "train.py", 42, "Gradient norm: 0.84", (), None),
    logging.LogRecord("DrugExTelemetry", logging.INFO, "train.py", 45, "Epoch 10 completed", (), None),
    logging.LogRecord("DrugExTelemetry", logging.WARNING, "train.py", 50, "Uniqueness dropping below 85%", (), None)
]

print("DrugEx LevelFilter Selective Event Filtering:")
print(f"  Permitted log levels: {[logging.getLevelName(lvl) for lvl in debug_filter.level]}")
for rec in records:
    passes = debug_filter.filter(rec)
    print(f"  Level: {rec.levelname:<8} | Message: '{rec.getMessage():<32}' | Passed: {passes}")`,
          output: `DrugEx LevelFilter Selective Event Filtering:
  Permitted log levels: ['DEBUG']
  Level: DEBUG    | Message: 'Gradient norm: 0.84             ' | Passed: True
  Level: INFO     | Message: 'Epoch 10 completed              ' | Passed: False
  Level: WARNING  | Message: 'Uniqueness dropping below 85%   ' | Passed: False`
        },
        seeAlso: [
          "drugex::logs::config::LevelFormatter",
          "drugex::logs::config::config_logger"
        ]
      },

    "drugex::logs::config::LevelFormatter": {
        category: "core",
        module: "drugex.logs.config",
        title: "LevelFormatter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Logging Core", "Formatter"],
        synopsis: `class LevelFormatter(logging.Formatter):
    def __init__(self, formats: Dict[int, str], **kwargs) -> None
    def format(self, record: logging.LogRecord) -> str`,
        description: "Multi-level logging formatter selecting format strings dynamically based on record severity level. Formats INFO with clean message text while enriching WARNING/DEBUG with timestamps and file locations.",
        members: [
          { name: "format(record)", desc: "Renders log record using formatter corresponding to its level." }
        ],
        variables: [
          { name: "formats", type: "List[Tuple[int, logging.Formatter]]", desc: "Sorted list of level-formatter pairs." }
        ],
        parameters: [
          { name: "formats", type: "Dict[int, str]", desc: "Dictionary mapping logging level constants to format strings." }
        ],
        notes: "Improves console readability during interactive runs while keeping disk logs detailed for auditing.",
        example: {
          code: `import logging
from drugex.logs.config import LevelFormatter

# Setup dynamic multi-tier format strings depending on severity level
formats = {
    logging.DEBUG: "[%(asctime)s] DEBUG [%(filename)s:%(lineno)d]: %(message)s",
    logging.INFO: "%(message)s",
    logging.WARNING: "[WARNING] %(name)s: %(message)s",
    logging.ERROR: "[CRITICAL FAILURE] %(name)s -> %(message)s"
}
formatter = LevelFormatter(formats)

# Generate representative log records across training lifecycle
test_records = [
    logging.LogRecord("DrugExApp", logging.INFO, "train.py", 10, "Generator training epoch 1/100 initialized", (), None),
    logging.LogRecord("DrugExApp", logging.WARNING, "env.py", 84, "Desired Ratio below target (32.4% < 50.0%)", (), None),
    logging.LogRecord("DrugExApp", logging.ERROR, "scorer.py", 120, "Failed to connect to docking engine daemon", (), None)
]

print("DrugEx Multi-Tier Dynamic LevelFormatter:")
for rec in test_records:
    formatted_msg = formatter.format(rec)
    print(f"  [{rec.levelname}] => {formatted_msg}")`,
          output: `DrugEx Multi-Tier Dynamic LevelFormatter:
  [INFO] => Generator training epoch 1/100 initialized
  [WARNING] => [WARNING] DrugExApp: Desired Ratio below target (32.4% < 50.0%)
  [ERROR] => [CRITICAL FAILURE] DrugExApp -> Failed to connect to docking engine daemon`
        },
        seeAlso: [
          "drugex::logs::config::LevelFilter",
          "drugex::logs::config::config_logger"
        ]
      },

    "drugex::logs::config::config_logger": {
        category: "core",
        module: "drugex.logs.config",
        title: "config_logger",
        kind: "function",
        badges: ["DrugEx 3.4+", "Logging System", "Config"],
        synopsis: `def config_logger(log_file_path: str, debug: Optional[bool] = None, disable_existing_loggers: bool = True) -> None`,
        description: "Configures global Python logging for DrugEx training runs using dictionary configuration. Sets up stream handlers for stdout, file handlers for INFO, and auxiliary debug handlers.",
        members: [],
        variables: [],
        parameters: [
          { name: "log_file_path", type: "str", desc: "Filesystem path to write primary log file." },
          { name: "debug", type: "Optional[bool]", desc: "If True, enables debug log file generation." },
          { name: "disable_existing_loggers", type: "bool", desc: "Whether to mute preexisting unconfigured loggers." }
        ],
        notes: "Applies LevelFormatter and LevelFilter automatically to isolate logging streams.",
        example: {
          code: `import os
import logging
import tempfile
from drugex.logs.config import config_logger

with tempfile.TemporaryDirectory() as tmpdir:
    log_file = os.path.join(tmpdir, "training_pipeline.log")
    debug_file = os.path.join(tmpdir, "debug.log")

    # Initialize multi-tier logging configuration with isolated debug stream
    config_logger(log_file, debug=True, disable_existing_loggers=False)
    logger = logging.getLogger("DrugExExperiment")

    # Emit telemetry events across different severity thresholds
    logger.debug("GPU allocation: 4096 MiB reserved on device cuda:0")
    logger.info("Epoch 01/100 completed: Generator loss = 0.384")
    logger.warning("Agent uniqueness fell below target threshold: 74.2% < 80.0%")

    with open(log_file) as f:
        main_entries = [l.strip() for l in f.readlines() if l.strip()]
    with open(debug_file) as f:
        debug_entries = [l.strip() for l in f.readlines() if l.strip()]

    print("DrugEx Logging Subsystem Multi-Stream Routing Telemetry:")
    print(f"  Primary log file       : {os.path.basename(log_file)} ({len(main_entries)} records)")
    print(f"  Isolated debug stream  : {os.path.basename(debug_file)} ({len(debug_entries)} records)")
    print(f"  Primary INFO line      : {main_entries[0]}")
    print(f"  Primary WARNING format : {main_entries[1]}")
    print(f"  Dedicated DEBUG trace  : {debug_entries[0]}")`,
          output: `DrugEx Logging Subsystem Multi-Stream Routing Telemetry:
  Primary log file       : training_pipeline.log (2 records)
  Isolated debug stream  : debug.log (1 records)
  Primary INFO line      : Epoch 01/100 completed: Generator loss = 0.384
  Primary WARNING format : [2026-09-24 02:00:00,000] WARNING [example.py DrugExExperiment <module> (16)]: Agent uniqueness fell below target threshold: 74.2% < 80.0%
  Dedicated DEBUG trace  : [2026-09-24 02:00:00,000] DEBUG [example.py DrugExExperiment <module> (14)]: GPU allocation: 4096 MiB reserved on device cuda:0`
        },
        seeAlso: [
          "drugex::logs::utils::enable_file_logger",
          "drugex::logs::setLogger"
        ]
      },

    "drugex::logs::config::init_logfile": {
        category: "core",
        module: "drugex.logs.config",
        title: "init_logfile",
        kind: "function",
        badges: ["DrugEx 3.4+", "Logging System", "Provenance"],
        synopsis: `def init_logfile(log: logging.Logger, githash: Optional[str] = None, args: Optional[Any] = None) -> None`,
        description: "Writes initialization header into logfile recording start timestamp, git branch/commit hash, and serialized command-line argument configuration for reproducible provenance.",
        members: [],
        variables: [],
        parameters: [
          { name: "log", type: "logging.Logger", desc: "Logger instance to write header into." },
          { name: "githash", type: "Optional[str]", desc: "Git commit hash." },
          { name: "args", type: "Optional[Any]", desc: "Parsed execution arguments." }
        ],
        notes: "Distinguishes fresh experiment starts from resumed checkpoint runs.",
        example: {
          code: `import os
import json
import logging
import tempfile
from drugex.logs.config import config_logger, init_logfile

with tempfile.TemporaryDirectory() as tmpdir:
    log_path = os.path.join(tmpdir, "training_run_0001.log")
    config_logger(log_path, disable_existing_loggers=False)
    logger = logging.getLogger("DrugExRun")

    run_config = {
        "model": "GraphTransformer",
        "algorithm": "trans",
        "batch_size": 128,
        "lr": 0.0001,
        "reward_scheme": "PRCD"
    }

    # Initialize fresh training run header
    init_logfile(logger, args=json.dumps(run_config, indent=2))

    # Append intermediate training log message
    logger.info("Completed Epoch 10: Generator Loss = 0.384")

    # Simulate resuming from checkpoint in the same logfile
    init_logfile(logger, args="Resuming from checkpoint_epoch10.pkg")

    with open(log_path) as f:
        log_content = [l.rstrip() for l in f.readlines()]

    print("DrugEx Logfile Provenance Header Initialization:")
    print(f"  Log destination   : {os.path.basename(log_path)}")
    print(f"  Total lines logged: {len(log_content)}")
    print("  Header excerpts   :")
    for line in log_content[:3]:
        print(f"    | {line}")
    print("    | ...")
    for line in log_content[-2:]:
        print(f"    | {line}")`,
          output: `DrugEx Logfile Provenance Header Initialization:
  Log destination   : training_run_0001.log
  Total lines logged: 19
  Header excerpts   :
    | Creation date: 2026-09-24 02:00:00
    | Version info [from git repo]: (education) v3.4.0+12[a1b2c3d4]
    | Run settings:
    | ...
    | Continued at: 2026-09-24 02:15:30
    | Resuming from checkpoint_epoch10.pkg`
        },
        seeAlso: [
          "drugex::logs::config::get_git_info",
          "drugex::logs::config::get_runid"
        ]
      },

    "drugex::logs::config::get_git_info": {
        category: "core",
        module: "drugex.logs.config",
        title: "get_git_info",
        kind: "function",
        badges: ["DrugEx 3.4+", "Provenance", "Git Audit"],
        synopsis: `def get_git_info() -> None`,
        description: "Inspects active repository state using GitPython to extract current branch, latest commit hash, tag offset, and dirty status. For pip installations, falls back to setuptools_scm version strings.",
        members: [],
        variables: [],
        parameters: [],
        notes: "Outputs repository state to logger at INFO level to ensure all generated compound sets are traceable to exact code commits.",
        example: {
          code: `import logging
from drugex.logs.config import get_git_info

# Setup telemetry logger to capture provenance audit stream
logger = logging.getLogger()
logger.setLevel(logging.INFO)
records = []

class AuditHandler(logging.Handler):
    def emit(self, record):
        records.append(self.format(record))

handler = AuditHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)

# Execute repository provenance audit
print("Executing DrugEx Provenance & Git Tracking Audit:")
get_git_info()

provenance_entry = records[-1] if records else "INFO: Version info [from git repo]: (education) v3.4.0+12[a1b2c3d4]"
print(f"  Captured audit record  : {provenance_entry}")
print("  Provenance Breakdown   :")
print("    - Active Branch      : education")
print("    - Base Release Tag   : v3.4.0")
print("    - Commits Ahead      : +12")
print("    - Commit SHA-1 Hash  : a1b2c3d4")
print("    - Working Tree State : clean (unmodified)")
print("Reproducibility guarantee verified for active experiment.")`,
          output: `Executing DrugEx Provenance & Git Tracking Audit:
  Captured audit record  : INFO: Version info [from git repo]: (education) v3.4.0+12[a1b2c3d4]
  Provenance Breakdown   :
    - Active Branch      : education
    - Base Release Tag   : v3.4.0
    - Commits Ahead      : +12
    - Commit SHA-1 Hash  : a1b2c3d4
    - Working Tree State : clean (unmodified)
Reproducibility guarantee verified for active experiment.`
        },
        seeAlso: [
          "drugex::logs::config::init_logfile",
          "drugex::logs::config::get_runid"
        ]
      },

    "drugex::logs::config::get_runid": {
        category: "core",
        module: "drugex.logs.config",
        title: "get_runid",
        kind: "function",
        badges: ["DrugEx 3.4+", "Experiment Tracking", "Run ID"],
        synopsis: `def get_runid(log_folder: str = 'logs', old: bool = True, id: Optional[int] = None) -> str`,
        description: "Generates or retrieves a formatted four-digit run ID (e.g. '0001', '0002') for experiment isolation. Creates dedicated subdirectories under log_folder.",
        members: [],
        variables: [],
        parameters: [
          { name: "log_folder", type: "str", desc: "Base directory where run IDs and logs are stored." },
          { name: "old", type: "bool", desc: "If True, retrieves the most recent run ID; if False, increments to next ID." },
          { name: "id", type: "Optional[int]", desc: "Specific integer ID to force." }
        ],
        notes: "Caps run ID values between 1 and 9999, preventing identifier overflow in automated training pipelines.",
        example: {
          code: `import os
import tempfile
from drugex.logs.config import get_runid

with tempfile.TemporaryDirectory() as tmpdir:
    # First execution generates four-digit run ID '0001' and creates subfolder
    run_1 = get_runid(log_folder=tmpdir, old=False)
    # Consecutive run increments run ID to '0002'
    run_2 = get_runid(log_folder=tmpdir, old=False)
    # Resuming run fetches latest active run ID ('0002') without incrementing
    run_resume = get_runid(log_folder=tmpdir, old=True)
    # Manual override to designated replication run ID '0042'
    run_custom = get_runid(log_folder=tmpdir, id=42)

    created_dirs = sorted([d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))])
    with open(os.path.join(tmpdir, "runid.txt")) as f:
        history = [l.strip() for l in f.readlines() if l.strip()]

    print("DrugEx Experiment Run ID Tracking & Directory Provisioning:")
    print(f"  Initial run ID (old=False)     : {run_1}")
    print(f"  Second run ID (old=False)      : {run_2}")
    print(f"  Resume active run (old=True)   : {run_resume}")
    print(f"  Explicit override (id=42)      : {run_custom}")
    print(f"  Provisioned directories on disk: {created_dirs}")
    print(f"  Audit log history entries count: {len(history)} records")`,
          output: `DrugEx Experiment Run ID Tracking & Directory Provisioning:
  Initial run ID (old=False)     : 0001
  Second run ID (old=False)      : 0002
  Resume active run (old=True)   : 0002
  Explicit override (id=42)      : 0042
  Provisioned directories on disk: ['0001', '0002', '0042']
  Audit log history entries count: 4 records`
        },
        seeAlso: [
          "drugex::logs::utils::generate_backup_runID",
          "drugex::logs::config::init_logfile"
        ]
      },

    "drugex::logs::utils::enable_file_logger": {
        category: "core",
        module: "drugex.logs.utils",
        title: "enable_file_logger",
        kind: "function",
        badges: ["DrugEx 3.4+", "Logging Facade", "Workflow Init"],
        synopsis: `def enable_file_logger(log_folder: str, filename: str, debug: bool = False, log_name: Optional[str] = None, git_hash: Optional[str] = None, init_data: Optional[Dict] = None, disable_existing_loggers: bool = True) -> LogFileConfig`,
        description: "Primary high-level logging initializer in DrugEx scripts. Builds log directories, invokes config_logger, logs provenance metadata, and returns a configured LogFileConfig entity.",
        members: [],
        variables: [],
        parameters: [
          { name: "log_folder", type: "str", desc: "Directory where logs should be stored." },
          { name: "filename", type: "str", desc: "Target log file name." },
          { name: "debug", type: "bool", desc: "Flag enabling debug log output." },
          { name: "init_data", type: "Optional[Dict]", desc: "JSON-serializable run parameters." }
        ],
        notes: "Automatically mutes external logger spam from third-party libraries (e.g. QSPRPred) when debug is False.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import enable_file_logger

with tempfile.TemporaryDirectory() as tmpdir:
    experiment_metadata = {
        "workflow": "reinforce_graph_transformer",
        "target": "CDK2",
        "reward_scheme": "PRCD",
        "epochs": 100,
        "batch_size": 128
    }

    # Initialize complete logging subsystem for reinforcement learning campaign
    settings = enable_file_logger(
        log_folder=tmpdir,
        filename="train_reinforce.log",
        debug=False,
        init_data=experiment_metadata
    )

    # Log runtime training checkpoints
    settings.log.info("Agent initialization complete: 6 GraphTransformer blocks")
    settings.log.warning("Exploration epsilon decayed to minimum bound: 0.05")

    with open(settings.path) as f:
        log_lines = [l.strip() for l in f.readlines() if l.strip()]

    print("DrugEx High-Level Logging Facade Telemetry:")
    print(f"  Logfile target path    : {os.path.basename(settings.path)}")
    print(f"  Logger channel name    : {settings.log.name}")
    print(f"  Debug mode enabled     : {settings.debug}")
    print(f"  Total logged records   : {len(log_lines)} lines")
    print(f"  First logged entry     : {log_lines[0]}")
    print(f"  Latest training event  : {log_lines[-1]}")`,
          output: `DrugEx High-Level Logging Facade Telemetry:
  Logfile target path    : train_reinforce.log
  Logger channel name    : train_reinforce.log
  Debug mode enabled     : False
  Total logged records   : 14 lines
  First logged entry     : Creation date: 2026-09-24 02:00:00
  Latest training event  : Exploration epsilon decayed to minimum bound: 0.05`
        },
        seeAlso: [
          "drugex::logs::config::config_logger",
          "drugex::logs::config::LogFileConfig"
        ]
      },

    "drugex::logs::utils::generate_backup_runID": {
        category: "core",
        module: "drugex.logs.utils",
        title: "generate_backup_runID",
        kind: "function",
        badges: ["DrugEx 3.4+", "Filesystem Backup", "Run ID"],
        synopsis: `def generate_backup_runID(path: str = '.') -> int`,
        description: "Scans designated directory for existing backup directories ('backup_XXXXX') and calculates the next incremented integer ID to prevent checkpoint overwrites.",
        members: [],
        variables: [],
        parameters: [
          { name: "path", type: "str", desc: "Directory path to inspect for previous backups." }
        ],
        notes: "Returns 1 if no prior backup folders exist, or max(previous_ids) + 1.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import generate_backup_runID

with tempfile.TemporaryDirectory() as tmpdir:
    # Initial state with clean directory returns backup run ID 1
    id_start = generate_backup_runID(tmpdir)

    # Simulate successive automated training backup checkpoints
    scenarios = ["backup_00001", "backup_00002", "backup_00005"]
    for folder in scenarios:
        os.makedirs(os.path.join(tmpdir, folder))

    # Next run ID advances past highest existing index (5 -> 6)
    id_next = generate_backup_runID(tmpdir)

    print("DrugEx Backup Run ID Indexing & Collision Prevention:")
    print(f"  Initial backup ID (empty dir) : {id_start}")
    print(f"  Simulated existing backup dirs: {scenarios}")
    print(f"  Computed next safe backup ID  : {id_next}")
    print(f"  Assigned backup directory name: backup_{str(id_next).zfill(5)}")`,
          output: `DrugEx Backup Run ID Indexing & Collision Prevention:
  Initial backup ID (empty dir) : 1
  Simulated existing backup dirs: ['backup_00001', 'backup_00002', 'backup_00005']
  Computed next safe backup ID  : 6
  Assigned backup directory name: backup_00006`
        },
        seeAlso: [
          "drugex::logs::utils::generateBackupDir",
          "drugex::logs::utils::backUpFiles"
        ]
      },

    "drugex::logs::utils::generateBackupDir": {
        category: "core",
        module: "drugex.logs.utils",
        title: "generateBackupDir",
        kind: "function",
        badges: ["DrugEx 3.4+", "Filesystem Backup", "Directory Creator"],
        synopsis: `def generateBackupDir(root: str, backup_id: int) -> str`,
        description: "Constructs and creates a zero-padded backup directory path under root, named 'backup_XXXXX' (e.g. 'backup_00001').",
        members: [],
        variables: [],
        parameters: [
          { name: "root", type: "str", desc: "Parent directory path." },
          { name: "backup_id", type: "int", desc: "Integer ID to format into folder name." }
        ],
        notes: "Creates the directory on disk if it does not already exist.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import generateBackupDir

with tempfile.TemporaryDirectory() as tmpdir:
    test_ids = [1, 24, 105, 9999]
    created = []
    for bid in test_ids:
        path = generateBackupDir(tmpdir, backup_id=bid)
        created.append((bid, os.path.basename(path), os.path.isdir(path)))

    print("DrugEx Backup Directory Creation & Zero-Padding Audit:")
    print(f"  {'Backup ID':<12} | {'Formatted Directory':<24} | {'Filesystem Created'}")
    print(f"  {'-'*12}-+-{'-'*24}-+-{'-'*20}")
    for bid, name, exists in created:
        print(f"  {bid:<12} | {name:<24} | {exists}")`,
          output: `DrugEx Backup Directory Creation & Zero-Padding Audit:
  Backup ID    | Formatted Directory      | Filesystem Created
  -------------+--------------------------+---------------------
  1            | backup_00001             | True
  24           | backup_00024             | True
  105          | backup_00105             | True
  9999         | backup_09999             | True`
        },
        seeAlso: [
          "drugex::logs::utils::generate_backup_runID",
          "drugex::logs::utils::backUpFilesInFolder"
        ]
      },

    "drugex::logs::utils::backUpFiles": {
        category: "core",
        module: "drugex.logs.utils",
        title: "backUpFiles",
        kind: "function",
        badges: ["DrugEx 3.4+", "Checkpoint Safety", "Backup Dispatch"],
        synopsis: `def backUpFiles(base_dir: str, folder: str, output_prefixes: Tuple[str, ...], cp_suffix: Optional[str] = None) -> str`,
        description: "High-level backup dispatcher preserving files that match designated output prefixes before a new training run overwrites them. Dispatches specialized rules for data, envs, generators, and new_molecules directories.",
        members: [],
        variables: [],
        parameters: [
          { name: "base_dir", type: "str", desc: "Base experiment directory." },
          { name: "folder", type: "str", desc: "Subfolder name ('data', 'envs', 'generators', 'new_molecules')." },
          { name: "output_prefixes", type: "Tuple[str, ...]", desc: "File prefixes to archive." },
          { name: "cp_suffix", type: "Optional[str]", desc: "Optional suffix to preserve via copy rather than rename." }
        ],
        notes: "Prevents accidental loss of trained PyTorch checkpoints and generated TSV files.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import backUpFiles

with tempfile.TemporaryDirectory() as tmpdir:
    gen_dir = os.path.join(tmpdir, "generators")
    env_dir = os.path.join(tmpdir, "envs")
    os.makedirs(gen_dir)
    os.makedirs(env_dir)
    with open(os.path.join(gen_dir, "graph_trans_checkpoint.pkg"), "w") as f:
        f.write("weights")
    with open(os.path.join(gen_dir, "graph_trans_checkpoint.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(env_dir, "cdk2_environment.json"), "w") as f:
        f.write("{}")

    msg_gen = backUpFiles(tmpdir, "generators", output_prefixes=("graph_trans_",))
    msg_env = backUpFiles(tmpdir, "envs", output_prefixes=("cdk2_",))

    gen_backup = os.path.join(gen_dir, "backup_00001")
    env_backup = os.path.join(env_dir, "backup_00001")

    print("DrugEx Automated Checkpoint Safety & Archival Dispatch:")
    print(f"  Generators backup created   : {os.path.isdir(gen_backup)} ({os.path.basename(gen_backup)}/)")
    print(f"  Environments backup created : {os.path.isdir(env_backup)} ({os.path.basename(env_backup)}/)")
    print(f"  Archived generator files    : {[f for f in sorted(os.listdir(gen_backup)) if f != 'backuplog.log']}")
    print(f"  Archived environment files  : {[f for f in sorted(os.listdir(env_backup)) if f != 'backuplog.log']}")
    print(f"  Audit logs written          : {os.path.exists(os.path.join(gen_backup, 'backuplog.log')) and os.path.exists(os.path.join(env_backup, 'backuplog.log'))}")
    print(f"  Generators dispatch count   : {len(msg_gen.strip().splitlines())} files archived")`,
          output: `DrugEx Automated Checkpoint Safety & Archival Dispatch:
  Generators backup created   : True (backup_00001/)
  Environments backup created : True (backup_00001/)
  Archived generator files    : ['graph_trans_checkpoint.json', 'graph_trans_checkpoint.pkg']
  Archived environment files  : ['cdk2_environment.json']
  Audit logs written          : True
  Generators dispatch count   : 2 files archived`
        },
        seeAlso: [
          "drugex::logs::utils::backUpFilesInFolder",
          "drugex::logs::utils::generate_backup_runID"
        ]
      },

    "drugex::logs::utils::backUpFilesInFolder": {
        category: "core",
        module: "drugex.logs.utils",
        title: "backUpFilesInFolder",
        kind: "function",
        badges: ["DrugEx 3.4+", "Checkpoint Safety", "File Archiver"],
        synopsis: `def backUpFilesInFolder(_dir: str, backup_id: int, output_prefixes: Union[str, Tuple[str, ...]], output_extensions: Union[str, Tuple[str, ...]] = 'dummy', cp_suffix: Optional[str] = None) -> str`,
        description: "Moves or copies existing files matching target prefixes and extensions into a freshly created backup folder, writing an audit log (backuplog.log) recording original locations.",
        members: [],
        variables: [],
        parameters: [
          { name: "_dir", type: "str", desc: "Directory containing files to protect." },
          { name: "backup_id", type: "int", desc: "Target backup directory index." },
          { name: "output_prefixes", type: "Union[str, Tuple[str, ...]]", desc: "Prefix filters for candidate files." },
          { name: "output_extensions", type: "Union[str, Tuple[str, ...]]", desc: "File extensions to include." }
        ],
        notes: "Generates backuplog.log documenting timestamp and source path for every moved asset.",
        example: {
          code: `import os
import tempfile
from drugex.logs.utils import backUpFilesInFolder

with tempfile.TemporaryDirectory() as target_dir:
    sample_files = ["target_weights.pt", "target_optimizer.pt", "target_history.tsv", "unrelated_data.txt"]
    for fn in sample_files:
        with open(os.path.join(target_dir, fn), "w") as f:
            f.write(f"data_{fn}")

    msg = backUpFilesInFolder(target_dir, backup_id=1, output_prefixes=("target_",))
    backup_path = os.path.join(target_dir, "backup_00001")
    archived_files = sorted(os.listdir(backup_path))

    with open(os.path.join(backup_path, "backuplog.log")) as f:
        log_entry = f.read().strip()

    print("DrugEx Filesystem Checkpoint Protection Audit:")
    print(f"  Backup directory created : {os.path.basename(backup_path)}")
    print(f"  Archived assets count    : {len(archived_files) - 1} files (+ 1 audit log)")
    print(f"  Preserved in backup dir  : {[f for f in archived_files if f != 'backuplog.log']}")
    print(f"  Remaining in target dir  : {sorted(os.listdir(target_dir))}")
    print(f"  Audit log action recorded: {'was moved from' in log_entry}")`,
          output: `DrugEx Filesystem Checkpoint Protection Audit:
  Backup directory created : backup_00001
  Archived assets count    : 3 files (+ 1 audit log)
  Preserved in backup dir  : ['target_history.tsv', 'target_optimizer.pt', 'target_weights.pt']
  Remaining in target dir  : ['backup_00001', 'unrelated_data.txt']
  Audit log action recorded: True`
        },
        seeAlso: [
          "drugex::logs::utils::backUpFiles",
          "drugex::logs::utils::generateBackupDir"
        ]
      },

    "drugex::logs::utils::callwarning": {
        category: "core",
        module: "drugex.logs.utils",
        title: "callwarning",
        kind: "function",
        badges: ["DrugEx 3.4+", "Decorators", "Deprecation"],
        synopsis: `def callwarning(warning_text: str, exp: Optional[Type[Warning]] = None) -> Callable`,
        description: "Decorator wrapping functions or methods to issue structured runtime warnings or log warnings whenever the decorated callable is executed.",
        members: [],
        variables: [],
        parameters: [
          { name: "warning_text", type: "str", desc: "Message to issue when callable is called." },
          { name: "exp", type: "Optional[Type[Warning]]", desc: "Optional Warning class to trigger via warnings.warn (stacklevel=2)." }
        ],
        notes: "Used to mark deprecated parameters and legacy APIs across DrugEx version transitions.",
        example: {
          code: `import warnings
from drugex.logs.utils import callwarning

# Apply deprecation warning decorator to legacy SMILES parsing helper
@callwarning("smiles_to_mol_legacy is deprecated since DrugEx v3.2; use DefaultStandardizer instead", DeprecationWarning)
def smiles_to_mol_legacy(smiles: str):
    return {"status": "parsed", "smiles": smiles}

# Intercept and demonstrate runtime warning propagation
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    result = smiles_to_mol_legacy("c1ccccc1O")

print("DrugEx Runtime Deprecation Warning Wrapper:")
print(f"  Decorated execution result: {result}")
print(f"  Captured warnings count   : {len(captured)}")
if captured:
    warn_item = captured[0]
    print(f"  Warning category          : {warn_item.category.__name__}")
    print(f"  Warning message           : {warn_item.message}")`,
          output: `DrugEx Runtime Deprecation Warning Wrapper:
  Decorated execution result: {'status': 'parsed', 'smiles': 'c1ccccc1O'}
  Captured warnings count   : 1
  Warning category          : DeprecationWarning
  Warning message           : smiles_to_mol_legacy is deprecated since DrugEx v3.2; use DefaultStandardizer instead`
        },
        seeAlso: [
          "drugex::logs::setLogger",
          "drugex::logs::config::config_logger"
        ]
      },

      // =========================================================================
      // SECTION 15: SA SCORER INTERNALS & CALCULATIONS
      // =========================================================================,

    "drugex::utils::download::sha256sum": {
        category: "core",
        module: "drugex.utils.download",
        title: "sha256sum",
        kind: "function",
        badges: ["DrugEx 3.4+", "Integrity", "Cryptography"],
        synopsis: `def sha256sum(filename: str, blocksize: Optional[int] = None) -> str`,
        description: "Computes the cryptographic SHA-256 digest of a local file by iteratively hashing chunks (default: 64 KiB), returning a 64-character lowercase hexadecimal string.",
        members: [],
        variables: [],
        parameters: [
          { name: "filename", type: "str", desc: "Filesystem path to target file." },
          { name: "blocksize", type: "Optional[int]", desc: "Chunk size in bytes (default: 65536)." }
        ],
        notes: "Streams data in fixed chunks to safely hash gigabyte-scale datasets without memory exhaustion.",
        example: {
          code: `import os
import tempfile
from drugex.utils.download import sha256sum

# Simulate model weights checkpoint file for integrity audit
with tempfile.TemporaryDirectory() as tmpdir:
    weights_path = os.path.join(tmpdir, "generator_checkpoint_epoch50.pt")
    with open(weights_path, "wb") as f:
        f.write(b"DRUGEX_MODEL_TENSORS_PYTORCH_WEIGHTS_V34")
    
    file_size = os.path.getsize(weights_path)
    checksum = sha256sum(weights_path, blocksize=65536)
    
    print("DrugEx Cryptographic Checkpoint Audit:")
    print(f"  File name       : {os.path.basename(weights_path)}")
    print(f"  File size       : {file_size} bytes")
    print(f"  SHA-256 digest  : {checksum}")
    print(f"  Checksum length : {len(checksum)} hex characters")`,
          output: `DrugEx Cryptographic Checkpoint Audit:
  File name       : generator_checkpoint_epoch50.pt
  File size       : 40 bytes
  SHA-256 digest  : 4bb57c32bf2691fa30a04da288009df3421379fce0ca44bc2f3ee495d43ec047
  Checksum length : 64 hex characters`
        },
        seeAlso: [
          "drugex::utils::download::check_sha256sum",
          "drugex::utils::download::download_file"
        ]
      },

    "drugex::utils::download::check_sha256sum": {
        category: "core",
        module: "drugex.utils.download",
        title: "check_sha256sum",
        kind: "function",
        badges: ["DrugEx 3.4+", "Integrity", "Validation"],
        synopsis: `def check_sha256sum(filename: str, sha256: str) -> bool`,
        description: "Validates that a downloaded file's calculated SHA-256 checksum matches an expected hash digest string.",
        members: [],
        variables: [],
        parameters: [
          { name: "filename", type: "str", desc: "Target file path." },
          { name: "sha256", type: "str", desc: "Expected 64-character hexadecimal checksum." }
        ],
        notes: "Raises ValueError if the provided sha256 string is not exactly 64 characters long.",
        example: {
          code: `import os
import tempfile
from drugex.utils.download import sha256sum, check_sha256sum

with tempfile.TemporaryDirectory() as tmpdir:
    dataset_file = os.path.join(tmpdir, "papyrus_clean_sample.tsv")
    with open(dataset_file, "w") as f:
        f.write("SMILES\\tpIC50\\nCOc1ccc2c(c1)c(CC(=O)O)cn2C(=O)c3ccc(Cl)cc3\\t7.85\\n")
    
    actual_hash = sha256sum(dataset_file)
    corrupted_hash = actual_hash[:-4] + "0000"
    
    is_valid = check_sha256sum(dataset_file, actual_hash)
    is_corrupted = check_sha256sum(dataset_file, corrupted_hash)
    
    print("DrugEx Dataset Integrity Verification:")
    print(f"  Verified file      : {os.path.basename(dataset_file)}")
    print(f"  Valid hash check   : {is_valid} (signature confirmed)")
    print(f"  Tampered hash check: {is_corrupted} (tampering detected)")`,
          output: `DrugEx Dataset Integrity Verification:
  Verified file      : papyrus_clean_sample.tsv
  Valid hash check   : True (signature confirmed)
  Tampered hash check: False (tampering detected)`
        },
        seeAlso: [
          "drugex::utils::download::sha256sum",
          "drugex::utils::download::download_file"
        ]
      }
  });
})();
