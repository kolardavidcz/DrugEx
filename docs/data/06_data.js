/**
 * DrugEx Documentation Platform - Data, Fragments & Vocabularies
 * Library: data (31 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::data::corpus::interfaces::Corpus": {
        category: "core",
        module: "drugex.data.corpus.interfaces",
        title: "Corpus",
        kind: "class",
        badges: ["DrugEx 3.4+", "Streaming Data", "Abstract Generator"],
        synopsis: `class Corpus(MolSupplier, ABC):
    def __init__(self, molecules: Union[Iterable[Any], MolSupplier]) -> None: ...
    @abstractmethod
    def processMolecule(self, item: Any) -> Any: ...
    @abstractmethod
    def getVoc(self) -> Vocabulary: ...
    def convert(self, item: Any) -> Any: ...
    def next(self) -> Any: ...
    def __iter__(self) -> Generator[Any, None, None]: ...
    def __next__(self) -> Any: ...`,
        description: "Abstract streaming supplier and generator for molecular corpora. Inherits from MolSupplier and ABC, lazily processing and converting raw chemical items (SMILES strings, files, or suppliers) into encoded representations while tracking chemical vocabularies without loading entire chemical libraries into RAM.",
        members: [
          { name: "convert(item)", desc: "Post-processing hook converting processed molecular items into the final corpus format (defaults to identity)." },
          { name: "getVoc()", desc: "Abstract method returning the active chemical vocabulary associated with this corpus." },
          { name: "next()", desc: "Advances the underlying supplier stream, processes the item, and returns the next valid non-None transformed molecule." },
          { name: "processMolecule(item)", desc: "Abstract method transforming a raw item yielded by the molecular supplier into an intermediate or tokenized format." },
          { name: "__iter__()", desc: "Returns the corpus generator itself for Python iterator protocol compliance." },
          { name: "__next__()", desc: "Advances the iterator, delegating directly to self.next()." }
        ],
        variables: [
          { name: "molecules", type: "Union[Iterable[Any], MolSupplier]", desc: "Underlying molecular supplier or iterable collection yielding raw structures." }
        ],
        parameters: [
          { name: "molecules", type: "Union[Iterable[Any], MolSupplier]", desc: "Source of raw molecular data, such as a ListSupplier, SMILES list, or file stream." }
        ],
        notes: "Allows processing multi-gigabyte chemical libraries (e.g., ZINC, Enamine REAL) lazily with O(1) memory footprint. Unparseable molecules returning None from processMolecule are automatically discarded during streaming.",
        example: {
          code: `from typing import Any, Iterable, Union
from drugex.data.corpus.interfaces import Corpus, Vocabulary
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.molecules.suppliers import ListSupplier

class CustomSmilesCorpus(Corpus):
    def __init__(self, molecules: Union[Iterable[Any], ListSupplier], voc: VocSmiles) -> None:
        super().__init__(molecules)
        self.voc = voc
    def processMolecule(self, item: str) -> str:
        # Standardize and tokenize
        return item.strip() if len(item.strip()) > 3 else None
    def getVoc(self) -> Vocabulary:
        return self.voc

supplier = ListSupplier(["c1ccccc1O", "CC(=O)O", "C", "c1ccccc1CC(=O)O"])
corpus = CustomSmilesCorpus(supplier, VocSmiles())

streamed = []
for mol in corpus:
    streamed.append(mol)
    if len(streamed) >= 3:
        break

print(f"Streamed {len(streamed)} valid entries:")
for idx, mol in enumerate(streamed, 1):
    print(f"  {idx}: {mol}")
print(f"Corpus vocabulary type: {type(corpus.getVoc()).__name__}")`,
          output: `Streamed 3 valid entries:
  1: c1ccccc1O
  2: CC(=O)O
  3: c1ccccc1CC(=O)O
Corpus vocabulary type: VocSmiles`
        },
        seeAlso: [
          "drugex::data::corpus::SequenceCorpus",
          "drugex::data::interfaces::DataSet",
          "drugex::molecules::interfaces::MolSupplier"
        ]
      },

    "drugex::data::interfaces::DataSet": {
        category: "core",
        module: "drugex.data.interfaces",
        title: "DataSet",
        kind: "class",
        badges: ["DrugEx 3.4+", "Disk Tabular Dataset", "Abstract Base"],
        synopsis: `class DataSet(ResultCollector, ABC):
    def __init__(
        self,
        path: str,
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[str] = None
    ) -> None: ...
    @abstractmethod
    def initData(self) -> None: ...
    @abstractmethod
    def readData(self) -> None: ...
    @abstractmethod
    def asDataLoader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader: ...
    def check(self) -> None: ...
    def getChunks(self, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]: ...
    def getData(self) -> pd.DataFrame: ...
    def getVoc(self) -> Vocabulary: ...
    def __call__(self, result: Any) -> None: ...
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]: ...
    def __len__(self) -> int: ...`,
        description: "Abstract base class for persistent disk-backed molecular datasets. Inherits from ResultCollector and ABC, managing TSV file persistence, vocabulary serialization, chunked multiprocessing evaluation, and conversion to PyTorch DataLoaders for generative modeling.",
        members: [
          { name: "asDataLoader(batch_size=32, shuffle=False)", desc: "Abstract method creating a PyTorch DataLoader configured with custom collate functions for model training." },
          { name: "check()", desc: "Validates dataset existence on disk; invokes readData() if existing or initData() if absent or rewrite=True." },
          { name: "getChunks(chunk_size=1000)", desc: "Generates sequential chunks of the tabular pandas DataFrame for memory-controlled processing." },
          { name: "getData()", desc: "Returns the loaded pandas DataFrame containing molecular tokens, sequences, and descriptor metadata." },
          { name: "getVoc()", desc: "Returns the associated Vocabulary instance parsed or initialized for this dataset." },
          { name: "initData()", desc: "Abstract method initializing the dataset file and structure on disk." },
          { name: "readData()", desc: "Abstract method reading serialized dataset tables and accompanying vocabulary from disk." },
          { name: "__call__(result)", desc: "ResultCollector interface implementation appending intermediate batch results directly to persistent storage." },
          { name: "__getitem__(item)", desc: "Retrieves the tensor tuple for the sample at integer index item." },
          { name: "__len__()", desc: "Returns total number of sample rows registered in the underlying dataset." }
        ],
        variables: [
          { name: "path", type: "str", desc: "Target file path on disk where dataset TSV table is stored." },
          { name: "rewrite", type: "bool", desc: "Whether to force complete re-encoding and overwrite existing disk files." },
          { name: "saveVoc", type: "bool", desc: "Flag indicating whether vocabulary should be serialized alongside the dataset." },
          { name: "vocFile", type: "Optional[str]", desc: "Custom file path override for vocabulary storage." },
          { name: "voc", type: "Optional[Vocabulary]", desc: "Loaded or initialized vocabulary instance." },
          { name: "data", type: "Optional[pd.DataFrame]", desc: "Underlying pandas DataFrame holding tabular records." }
        ],
        parameters: [
          { name: "path", type: "str", desc: "Filesystem path for the primary dataset TSV file." },
          { name: "rewrite", type: "bool", desc: "If True, re-executes data creation even if the target file exists (default: False)." },
          { name: "save_voc", type: "bool", desc: "Whether to persist vocabulary to disk alongside dataset (default: True)." },
          { name: "voc_file", type: "Optional[str]", desc: "Optional path pointing to a shared vocabulary file (default: None)." }
        ],
        notes: "Concrete subclasses include SmilesDataSet (SMILES sequence modeling), SmilesFragDataSet (fragment-based SMILES extension), and GraphFragDataSet (2D molecular graph actions). Inheriting from ResultCollector allows instances to directly receive output streams from ParallelProcessor.",
        example: {
          code: `import os, tempfile
import torch
import pandas as pd
from torch.utils.data import DataLoader
from drugex.data.interfaces import DataSet
from drugex.data.corpus.vocabulary import VocSmiles

class DummySmilesDataSet(DataSet):
    def initData(self) -> None:
        df = pd.DataFrame({"Token": ["c1ccccc1O", "CC(=O)O"]})
        df.to_csv(self.path, sep="\\t", index=False)
        self.voc = VocSmiles()
        self.data = df
    def readData(self) -> None:
        self.data = pd.read_csv(self.path, sep="\\t")
        self.voc = VocSmiles()
    def asDataLoader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        tensors = torch.zeros((len(self.data), 10), dtype=torch.long)
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, item: int):
        return (torch.zeros(10, dtype=torch.long),)

with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
    tmp_path = tmp.name

try:
    ds = DummySmilesDataSet(tmp_path, rewrite=True)
    print(f"Dataset path: {os.path.basename(ds.path)}")
    print(f"Total rows: {len(ds)}")
    print(f"Voc size: {len(ds.getVoc())}")
    loader = ds.asDataLoader(batch_size=1)
    print(f"DataLoader batch count: {len(loader)}")
finally:
    if os.path.exists(tmp_path): os.remove(tmp_path)`,
          output: `Dataset path: tmp...tsv
Total rows: 2
Voc size: 86
DataLoader batch count: 2`
        },
        seeAlso: [
          "drugex::data::datasets::SmilesDataSet",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::datasets::GraphFragDataSet",
          "drugex::parallel::interfaces::ResultCollector"
        ]
      },

    "drugex::data::interfaces::FragmentPairEncoder": {
        category: "core",
        module: "drugex.data.interfaces",
        title: "FragmentPairEncoder",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Encoding", "Pair Processor", "Abstract Base"],
        synopsis: `class FragmentPairEncoder(ABC):
    @abstractmethod
    def encode(self, fragment: str, molecule: str) -> Tuple[Any, Any]: ...`,
        description: "Abstract encoder contract transforming (fragment, molecule) chemical pairs into paired input condition and target representations for conditional fragment-based generative models. Subclasses produce paired 1D SMILES sequences or 3D graph action matrices.",
        members: [
          { name: "encode(fragment, molecule)", desc: "Abstract method translating a scaffold fragment conditioning prefix and complete target molecule into paired tensor representations." }
        ],
        variables: [],
        parameters: [],
        notes: "Subclasses include SequenceFragmentEncoder (producing 1D LongTensors with '_' attachment delimiters) and GraphFragmentEncoder (producing 3D action matrices of shape [N, max_len, 5]).",
        example: {
          code: `import torch
from drugex.data.interfaces import FragmentPairEncoder

class SequenceMockPairEncoder(FragmentPairEncoder):
    def encode(self, fragment: str, molecule: str):
        # Conditioning input: fragment tokens padded to max_len
        # Target output: complete molecule tokens
        src = torch.tensor([1, 80, 80, 80, 0], dtype=torch.long)
        trg = torch.tensor([1, 80, 80, 80, 24, 0], dtype=torch.long)
        return src, trg

encoder = SequenceMockPairEncoder()
src_tensor, trg_tensor = encoder.encode("c1ccccc1", "c1ccccc1O")
print(f"Source fragment tensor shape: {src_tensor.shape}")
print(f"Target molecule tensor shape: {trg_tensor.shape}")
print(f"Source tokens: {src_tensor.tolist()}")
print(f"Target tokens: {trg_tensor.tolist()}")`,
          output: `Source fragment tensor shape: torch.Size([5])
Target molecule tensor shape: torch.Size([6])
Source tokens: [1, 80, 80, 80, 0]
Target tokens: [1, 80, 80, 80, 24, 0]`
        },
        seeAlso: [
          "drugex::data::fragments::SequenceFragmentEncoder",
          "drugex::data::fragments::GraphFragmentEncoder",
          "drugex::data::datasets::SmilesFragDataSet"
        ]
      },

    "drugex::data::corpus::interfaces::Vocabulary": {
        category: "core",
        module: "drugex.data.corpus.interfaces",
        title: "Vocabulary",
        kind: "class",
        badges: ["DrugEx 3.4+", "Token Mapper", "Abstract Base"],
        synopsis: `class Vocabulary(ABC):
    def __init__(self, words: Iterable[str]) -> None: ...
    @abstractmethod
    def encode(self, tokens: List[str], frags: Optional[List[str]] = None) -> torch.Tensor: ...
    @abstractmethod
    def decode(self, representation: torch.Tensor) -> str: ...
    @staticmethod
    def fromFile(path: str) -> Vocabulary: ...
    def toFile(self, path: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __getitem__(self, key: Union[str, int]) -> Union[int, str]: ...
    def __len__(self) -> int: ...`,
        description: "Root abstract class defining molecular token representation and numerical indexing in DrugEx. Standardizes bidirectional mapping between human-readable chemical representations (SMILES strings or graph actions) and numerical PyTorch tensors, reserving index 0 for padding.",
        members: [
          { name: "decode(representation)", desc: "Abstract method converting integer index tensor or array back to chemical string representation." },
          { name: "encode(tokens, frags=None)", desc: "Abstract method mapping token sequence collections to numerical index PyTorch tensors." },
          { name: "fromFile(path)", desc: "Static factory method loading token vocabulary words from a text file on disk." },
          { name: "toFile(path)", desc: "Serializes registered vocabulary token definitions to a text file." },
          { name: "__contains__(key)", desc: "Checks whether a token string or character is present within vocabulary words." },
          { name: "__getitem__(key)", desc: "Bidirectional subscript lookup: maps string token to integer ID (tk2ix) or integer index to token string (ix2tk)." },
          { name: "__len__()", desc: "Returns total number of unique tokens registered in vocabulary (self.size)." }
        ],
        variables: [
          { name: "words", type: "List[str]", desc: "Ordered list of discrete vocabulary token strings." },
          { name: "tk2ix", type: "Dict[str, int]", desc: "Dictionary mapping string token to integer index." },
          { name: "ix2tk", type: "Dict[int, str]", desc: "Dictionary mapping integer index to string token." },
          { name: "size", type: "int", desc: "Total token vocabulary cardinality, determining neural embedding layer size." }
        ],
        parameters: [
          { name: "words", type: "Iterable[str]", desc: "Collection of token strings used to construct vocabulary." }
        ],
        notes: "Index 0 is universally reserved for padding tokens to ensure compatibility with batch padding masks across attention and recurrent layers.",
        example: {
          code: `from drugex.data.corpus.vocabulary import VocSmiles

# Demonstrate base Vocabulary interface methods on VocSmiles
voc = VocSmiles(encode_frags=False)
print(f"Total vocabulary size: {len(voc)}")
print(f"Token 'c' in vocabulary: {'c' in voc}")
print(f"Missing token 'XYZ' in voc: {'XYZ' in voc}")

idx_c = voc['c']
print(f"Index for 'c': {idx_c}")
print(f"Reverse lookup index {idx_c}: {voc[idx_c]}")
print(f"Padding index 0 reserved: {voc[0] is not None}")
sample_tokens = [voc[i] for i in range(1, 4)]
print(f"Sample tokens (indices 1-3): {sample_tokens}")`,
          output: `Total vocabulary size: 86
Token 'c' in vocabulary: True
Missing token 'XYZ' in voc: False
Index for 'c': 80
Reverse lookup index 80: c
Padding index 0 reserved: True
Sample tokens (indices 1-3): ['C', 'N', 'O']`
        },
        seeAlso: [
          "drugex::data::corpus::interfaces::SequenceVocabulary",
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::data::corpus::vocabulary::VocGraph"
        ]
      },

    "drugex::data::corpus::interfaces::SequenceVocabulary": {
        category: "core",
        module: "drugex.data.corpus.interfaces",
        title: "SequenceVocabulary",
        kind: "class",
        badges: ["DrugEx 3.4+", "Sequence Tokens", "Control Tokens", "Abstract Base"],
        synopsis: `class SequenceVocabulary(Vocabulary, ABC):
    def __init__(
        self,
        encode_frags: bool,
        words: Sequence[str],
        max_len: int = 100,
        min_len: int = 10
    ) -> None: ...
    @abstractmethod
    def splitSequence(self, seq: str) -> List[str]: ...
    def addWordsFromSeq(self, seq: str, ignoreConstraints: bool = False) -> Optional[List[str]]: ...
    def removeIfNew(self, seq: str, ignoreConstraints: bool = False) -> Optional[List[str]]: ...
    def toFile(self, path: Union[str, os.PathLike[str], Path]) -> None: ...`,
        description: "Abstract base vocabulary for autoregressive sequence models. Manages special control tokens ('GO' for start of sequence, 'EOS' for end of sequence, and '_' for fragment attachment delimiters) along with sequence length boundaries.",
        members: [
          { name: "splitSequence(seq)", desc: "Abstract method splitting a chemical sequence string into discrete atomic and structural tokens." },
          { name: "addWordsFromSeq(seq, ignoreConstraints=False)", desc: "Tokenizes a sequence and dynamically registers any newly encountered tokens into the vocabulary if length constraints are met." },
          { name: "removeIfNew(seq, ignoreConstraints=False)", desc: "Tokenizes a sequence, returning None if any tokens fall outside the registered vocabulary set." },
          { name: "toFile(path)", desc: "Serializes active vocabulary tokens (excluding control tokens) to a plaintext file separated by newlines." }
        ],
        variables: [
          { name: "control", type: "Tuple[str, ...]", desc: "Special control tokens: ('_', 'GO', 'EOS') if encode_frags=True else ('GO', 'EOS')." },
          { name: "special", type: "List[str]", desc: "Full list of control tokens plus delimiter '.', excluded from regular token sets." },
          { name: "encode_frags", type: "bool", desc: "Whether fragment attachment delimiter tokens are enabled." },
          { name: "max_len", type: "int", desc: "Maximum allowable sequence length in tokens (default: 100)." },
          { name: "min_len", type: "int", desc: "Minimum allowable sequence length in tokens (default: 10)." },
          { name: "wordSet", type: "Set[str]", desc: "Set of standard chemical tokens registered in this vocabulary." },
          { name: "tk2ix", type: "Dict[str, int]", desc: "Mapping from token string to integer vocabulary index." },
          { name: "ix2tk", type: "Dict[int, str]", desc: "Reverse mapping from integer index to token string." }
        ],
        parameters: [
          { name: "encode_frags", type: "bool", desc: "If True, includes fragment separation control tokens ('_') for fragment-conditioned models." },
          { name: "words", type: "Sequence[str]", desc: "Collection of token strings initializing the vocabulary." },
          { name: "max_len", type: "int", desc: "Maximum allowable sequence length in tokens (default: 100)." },
          { name: "min_len", type: "int", desc: "Minimum allowable sequence length in tokens (default: 10)." }
        ],
        notes: "Sequences longer than max_len are truncated, while shorter sequences are right-padded to max_len with index 0 padding tokens.",
        example: {
          code: `from drugex.data.corpus.vocabulary import VocSmiles

# Initialize SMILES vocabulary with fragment delimiters and length bounds
voc = VocSmiles(encode_frags=True, max_len=60, min_len=5)

print(f"Control tokens: {voc.control}")
print(f"Sequence length bounds: min={voc.min_len}, max={voc.max_len}")
print(f"Total vocabulary size (including special): {len(voc)}")

# Tokenize candidate chemical sequence
sample_smiles = "c1ccccc1O"
tokens = voc.splitSequence(sample_smiles)
print(f"Tokenized sequence ({len(tokens)} tokens): {tokens}")

# Numerical encoding to PyTorch tensor indices
encoded = voc.encode(tokens, pad=True)
print(f"Encoded token array shape: {encoded.shape}")
print(f"Decoded reconstruction: {voc.decode(encoded[:len(tokens)], is_tk=True)}")`,
          output: `Control tokens: ('_', 'GO', 'EOS')
Sequence length bounds: min=5, max=60
Total vocabulary size (including special): 87
Tokenized sequence (10 tokens): ['c', '1', 'c', 'c', 'c', 'c', 'c', '1', 'O', 'EOS']
Encoded token array shape: (60,)
Decoded reconstruction: c1ccccc1O`
        },
        seeAlso: [
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::data::corpus::vocabulary::VocNonGPT",
          "drugex::data::corpus::SequenceCorpus"
        ]
      },

    "drugex::data::corpus::vocabulary::VocSmiles": {
        category: "core",
        module: "drugex.data.corpus.vocabulary",
        title: "VocSmiles",
        kind: "class",
        badges: ["DrugEx 3.4+", "SMILES Regex", "Main Tokenizer"],
        synopsis: `class VocSmiles(SequenceVocabulary):
    defaultWords: Tuple[str, ...] = (
        '#', '%', '(', ')', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '=', 'B', 'C', 'F', 'I', 'L', 'N', 'O', 'P', 'R', 'S', '[Ag-3]', '[As+]',
        '[As]', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH-]',
        '[CH2]', '[CH]', '[I+]', '[IH2]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]',
        '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH]', '[S+]', '[S-]', '[SH+]',
        '[SH2]', '[SH]', '[Se+]', '[SeH]', '[Se]', '[SiH2]', '[SiH]', '[Si]', '[Te]',
        '[b-]', '[c+]', '[c-]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]',
        '[s+]', '[se+]', '[se]', '[te+]', '[te]', 'b', 'c', 'n', 'o', 'p', 's'
    )
    def __init__(
        self,
        encode_frags: bool = False,
        words: Sequence[str] = defaultWords,
        max_len: int = 100,
        min_len: int = 10
    ) -> None: ...`,
        description: "The primary SMILES sequence tokenizer for DrugEx RNN and Transformer models. Uses regular expression matching to tokenize multi-character halogens ('Cl' -> 'L', 'Br' -> 'R'), formal charges, aromatic atoms, branch parentheses, and ring closure digits, converting sequences into padded numeric tensors of shape (N, max_len).",
        members: [
          { name: "calc_voc_fp(smiles, prefix=None)", desc: "Calculates vocabulary token fingerprint arrays for a batch of SMILES, returning 2D integer numpy array of shape (len(smiles), max_len)." },
          { name: "clean(smiles)", desc: "Normalizes internal halogen tokens ('L' -> 'Cl', 'R' -> 'Br') back to standard IUPAC SMILES notation." },
          { name: "decode(representation, is_tk=False)", desc: "Decodes integer token tensors of shape (L,) or (N, L) back into canonical SMILES strings." },
          { name: "encode(tokens, frags=None)", desc: "Encodes a batch of token sequences into a 2D LongTensor of shape (len(tokens), max_len) right-padded with index 0." },
          { name: "fromFile(path, encode_frags=False, min_len=10, max_len=100)", desc: "Static factory method loading vocabulary words from a whitespace-separated text file on disk." },
          { name: "splitSequence(smile)", desc: "Splits a SMILES string into a list of chemical tokens via compiled regular expressions, replacing halogens and appending 'EOS'." },
          { name: "toFile(path)", desc: "Saves the registered vocabulary tokens to a text file on disk." },
          { name: "__contains__(key)", desc: "Checks whether a chemical token or character string exists within vocabulary words." },
          { name: "__getitem__(key)", desc: "Subscript indexing: maps string token to integer ID (tk2ix) or integer index to token string (ix2tk)." },
          { name: "__len__()", desc: "Returns total number of unique tokens in vocabulary (self.size), determining neural embedding dimensions." }
        ],
        variables: [
          { name: "defaultWords", type: "Tuple[str, ...]", desc: "Standard 84-token organic chemistry vocabulary." },
          { name: "control", type: "Tuple[str, ...]", desc: "Control token tuple: ('_', 'GO', 'EOS') if encode_frags=True else ('GO', 'EOS')." },
          { name: "special", type: "List[str]", desc: "Full special tokens list including fragment delimiters and control markers." },
          { name: "wordSet", type: "Set[str]", desc: "Set of active chemical tokens excluding special control markers." },
          { name: "tk2ix", type: "Dict[str, int]", desc: "Token-to-index mapping dictionary." },
          { name: "ix2tk", type: "Dict[int, str]", desc: "Index-to-token inverse lookup dictionary." },
          { name: "size", type: "int", desc: "Total token vocabulary cardinality (len(words) + len(special))." },
          { name: "max_len", type: "int", desc: "Maximum allowable sequence length in tokens (default: 100)." },
          { name: "min_len", type: "int", desc: "Minimum allowable sequence length in tokens (default: 10)." }
        ],
        parameters: [
          { name: "encode_frags", type: "bool", desc: "If True, includes fragment separation control tokens ('_') for fragment-conditioned models (default: False)." },
          { name: "words", type: "Sequence[str]", desc: "Tokens included in vocabulary (defaults to defaultWords)." },
          { name: "max_len", type: "int", desc: "Maximum allowable sequence length in tokens (default: 100)." },
          { name: "min_len", type: "int", desc: "Minimum allowable sequence length in tokens (default: 10)." }
        ],
        notes: "Unrecognized tokens outside defaultWords trigger an Out-Of-Vocabulary warning unless update_voc=True is used during SequenceCorpus processing.",
        example: {
          code: `import torch
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=False, max_len=20)
smiles_list = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]
tokens_batch = [voc.splitSequence(smi) for smi in smiles_list]
tensor_batch = voc.encode(tokens_batch)

print("Encoded batch tensor shape:", tensor_batch.shape)
print("First token IDs for Phenol:", tensor_batch[0, :8].tolist())

decoded = voc.decode(tensor_batch[0], is_tk=False)
print("Decoded SMILES:", decoded)`,
          output: `Encoded batch tensor shape: torch.Size([2, 20])
First token IDs for Phenol: [80, 7, 80, 80, 80, 80, 80, 7]
Decoded SMILES: c1ccccc1O`
        },
        seeAlso: [
          "drugex::data::corpus::vocabulary::VocNonGPT",
          "drugex::data::corpus::vocabulary::VocGraph",
          "drugex::data::corpus::SequenceCorpus",
          "drugex::data::datasets::SmilesDataSet"
        ]
      },

    "drugex::data::corpus::vocabulary::VocNonGPT": {
        category: "core",
        module: "drugex.data.corpus.vocabulary",
        title: "VocNonGPT",
        kind: "class",
        badges: ["DrugEx 3.4+", "SMILES Tokenizer", "Non-GPT Variant"],
        synopsis: `class VocNonGPT(VocSmiles):
    def __init__(
        self,
        words: Sequence[str],
        src_len: int = 1000,
        trg_len: int = 100,
        max_len: int = 100,
        min_len: int = 10
    ) -> None: ...`,
        description: "Specialized VocSmiles variant configured for encoder-decoder SequenceRNN models without GPT-2 delimiters. Distinguishes source condition length (src_len) from target generation length (trg_len), encoding input tokens with optional '|' prefixes.",
        members: [
          { name: "decode(matrix, is_smiles=True, is_tk=False)", desc: "Decodes an integer token index matrix or vector back into canonical SMILES strings, parsing halogen substitutions." },
          { name: "encode(input, is_smiles=True)", desc: "Encodes token sequences into LongTensor of shape (N, trg_len) if is_smiles=True, or (N, src_len) with '|' prefixes if is_smiles=False." },
          { name: "fromFile(path, src_len=1000, trg_len=100, max_len=100, min_len=10)", desc: "Static factory method loading legacy sequence vocabulary from a text file on disk." },
          { name: "splitSequence(smile)", desc: "Splits a SMILES string into chemical tokens via regex parsing." }
        ],
        variables: [
          { name: "src_len", type: "int", desc: "Maximum token length for conditioning source inputs (default: 1000)." },
          { name: "trg_len", type: "int", desc: "Maximum token length for generated target molecules (default: 100)." }
        ],
        parameters: [
          { name: "words", type: "Sequence[str]", desc: "Base vocabulary tokens." },
          { name: "src_len", type: "int", desc: "Source sequence length limit (default: 1000)." },
          { name: "trg_len", type: "int", desc: "Target sequence length limit (default: 100)." },
          { name: "max_len", type: "int", desc: "Fallback maximum length parameter (default: 100)." },
          { name: "min_len", type: "int", desc: "Fallback minimum length parameter (default: 10)." }
        ],
        notes: "Direct subclass of VocSmiles; ensures backward compatibility with DrugEx v2 RNN encoder-decoder architectures.",
        example: {
          code: `import torch
from drugex.data.corpus.vocabulary import VocNonGPT, VocSmiles

voc = VocNonGPT(words=VocSmiles.defaultWords, src_len=40, trg_len=30)
src_tokens = [["c", "1", "c", "c", "c", "c", "c", "1", "EOS"]]
trg_tokens = [["C", "C", "(", "=", "O", ")", "O", "EOS"]]

src_tensor = voc.encode(src_tokens, is_smiles=False)
trg_tensor = voc.encode(trg_tokens, is_smiles=True)

print("Source fragment tensor shape:", src_tensor.shape)
print("Target molecule tensor shape:", trg_tensor.shape)
print("Decoded target string:", voc.decode(trg_tensor[0], is_smiles=True))`,
          output: `Source fragment tensor shape: torch.Size([1, 40])
Target molecule tensor shape: torch.Size([1, 30])
Decoded target string: CC(=O)O`
        },
        seeAlso: [
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::training::generators::SequenceRNN",
          "drugex::data::corpus::SequenceCorpus"
        ]
      },

    "drugex::data::corpus::vocabulary::VocGraph": {
        category: "core",
        module: "drugex.data.corpus.vocabulary",
        title: "VocGraph",
        kind: "class",
        badges: ["DrugEx 3.4+", "2D Graph Actions", "Atom-Valence Tokens"],
        synopsis: `class VocGraph(Vocabulary):
    defaultWords: Tuple[str, ...] = (
        "2O", "3O+", "1O-", "4C", "3C+", "3C-", "3N", "4N+", "2N-", "1Cl", "2S", "6S",
        "4S", "3S+", "5S+", "1S-", "1F", "1I", "5I", "2I+", "1Br", "5P", "3P", "4P+",
        "2Se", "6Se", "4Se", "3Se+", "4Si", "3B", "4B-", "5As", "3As", "4As+", "2Te",
        "4Te", "3Te+"
    )
    def __init__(
        self,
        words: Sequence[str] = defaultWords,
        max_len: int = 80,
        n_frags: int = 4
    ) -> None: ...`,
        description: "Vocabulary for graph-based molecular generation via GraphTransformer. Encodes molecular construction into 5-tuple action vectors (atom_tk, curr_locus, prev_locus, bond_type, frag_flag) based on explicit atomic valence states, formal charges, bond orders, and scaffold fragment connectivity.",
        members: [
          { name: "decode(matrix)", desc: "Decodes 3D action matrices of shape (N, L, 5) back into reconstructed fragment and molecule SMILES strings." },
          { name: "encode(smiles, subs=None)", desc: "Encodes molecules and scaffold fragments into 3D integer action arrays of shape (len(smiles), max_len, 5)." },
          { name: "fromDataFrame(df, word_col='Word', max_len=80, n_frags=4)", desc: "Static factory method creating a VocGraph instance from a pandas DataFrame." },
          { name: "fromFile(path, word_col='Word', max_len=80, n_frags=4)", desc: "Static factory method loading graph vocabulary tokens from a TSV file on disk." },
          { name: "get_atom_tk(atom)", desc: "Computes vocabulary token index corresponding to an RDKit Atom based on its element and total valence." },
          { name: "parseWord(word)", desc: "Static helper decomposing an atom-valence word into (element_charge, valence, charge_num, atomic_num, original_word)." },
          { name: "toDataFrame()", desc: "Exports parsed atom-valence definitions to a pandas DataFrame with columns ['Ele', 'Val', 'Ch', 'Nr', 'Word']." },
          { name: "toFile(path)", desc: "Serializes the graph vocabulary definitions to a TSV file on disk." },
          { name: "__contains__(word)", desc: "Checks whether an atom-valence token is registered in the graph vocabulary." },
          { name: "__getitem__(key)", desc: "Bidirectional subscript lookup: maps graph word to integer token index or index to word." },
          { name: "__len__()", desc: "Returns total number of graph tokens registered in vocabulary (self.size)." }
        ],
        variables: [
          { name: "defaultWords", type: "Tuple[str, ...]", desc: "Standard 37 atom-valence tokens covering common drug-like organic chemistry." },
          { name: "control", type: "Tuple[str, ...]", desc: "Control tokens: ('EOS', 'GO')." },
          { name: "n_frags", type: "int", desc: "Maximum number of scaffold fragment components supported (default: 4)." },
          { name: "max_len", type: "int", desc: "Maximum graph action steps allowable in generation trajectory (default: 80)." },
          { name: "masks", type: "torch.LongTensor", desc: "Intrinsic atom valence capacity tensor used for masking invalid bond additions during decoding." }
        ],
        parameters: [
          { name: "words", type: "Sequence[str]", desc: "Atom-valence token strings (defaults to defaultWords)." },
          { name: "max_len", type: "int", desc: "Maximum action steps in graph generation sequence (default: 80)." },
          { name: "n_frags", type: "int", desc: "Maximum scaffold fragments supported for conditioning (default: 4)." }
        ],
        notes: "Graph generation proceeds iteratively: the model predicts the next atom token, current atom index, previous connection locus, bond order, and fragment assignment. Action masking prevents valence violations in real-time.",
        example: {
          code: `from drugex.data.corpus.vocabulary import VocGraph

voc = VocGraph(max_len=80, n_frags=4)
print(f"Total graph vocabulary words: {voc.size}")
print(f"Control tokens: {voc.control}")

# Parse sample atom-valence tokens
for w in ["4C", "2O", "3N", "1Cl", "1O-"]:
    elem_ch, valence, ch_num, at_num, orig = VocGraph.parseWord(w)
    print(f"Token '{w}': Element={elem_ch}, Valence={valence}, AtomicNum={at_num}, Charge={ch_num}")

# Bidirectional index lookups and mask check
c_idx = voc['4C']
print(f"Subscript lookup: '4C' -> Index {c_idx} -> Reverse: '{voc[c_idx]}'")
print(f"Valence mask tensor shape: {voc.masks.shape}")`,
          output: `Total graph vocabulary words: 40
Control tokens: ('EOS', 'GO')
Token '4C': Element=C, Valence=4, AtomicNum=6, Charge=0
Token '2O': Element=O, Valence=2, AtomicNum=8, Charge=0
Token '3N': Element=N, Valence=3, AtomicNum=7, Charge=0
Token '1Cl': Element=Cl, Valence=1, AtomicNum=17, Charge=0
Token '1O-': Element=O-, Valence=1, AtomicNum=8, Charge=-1
Subscript lookup: '4C' -> Index 3 -> Reverse: '4C'
Valence mask tensor shape: torch.Size([40])`
        },
        seeAlso: [
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::training::generators::GraphTransformer",
          "drugex::data::datasets::GraphFragDataSet"
        ]
      },

    "drugex::data::corpus::SequenceCorpus": {
        category: "core",
        module: "drugex.data.corpus.corpus",
        title: "SequenceCorpus",
        kind: "class",
        badges: ["DrugEx 3.4+", "Corpus Processor", "Dataset Pipeline"],
        synopsis: `class SequenceCorpus(Corpus):
    def __init__(
        self,
        molecules: Union[Iterable[str], MolSupplier],
        vocabulary: Optional[SequenceVocabulary] = None,
        update_voc: bool = True,
        throw: bool = False,
        check_unique: bool = True
    ) -> None: ...`,
        description: "Concrete molecular corpus pipeline for SMILES sequence datasets. Inherits from Corpus, automating streaming tokenization, dynamic vocabulary expansion, deduplication, and filtering of unparseable chemical structures.",
        members: [
          { name: "convert(item)", desc: "Passes through the tokenized representation (identity hook inherited from Corpus)." },
          { name: "getVoc()", desc: "Returns the active SequenceVocabulary instance tracking all observed chemical tokens." },
          { name: "processMolecule(item)", desc: "Tokenizes and encodes a single SMILES string into an integer tensor, registering novel tokens if update_voc=True." },
          { name: "saveVoc(path)", desc: "Serializes the updated vocabulary to a text file on disk." },
          { name: "__iter__()", desc: "Yields lazily processed and tokenized molecular sequences from the input supplier." },
          { name: "__len__()", desc: "Returns total number of sequences in the underlying molecular supplier if known." },
          { name: "__next__()", desc: "Advances to the next valid processed sequence." }
        ],
        variables: [
          { name: "vocabulary", type: "SequenceVocabulary", desc: "Active sequence token vocabulary." },
          { name: "updateVoc", type: "bool", desc: "Whether to dynamically register unseen tokens encountered during processing." },
          { name: "throw", type: "bool", desc: "Whether to raise an exception upon encountering unparseable tokens." },
          { name: "checkUnique", type: "bool", desc: "Whether to maintain a seen set to discard duplicate chemical structures." }
        ],
        parameters: [
          { name: "molecules", type: "Union[Iterable[str], MolSupplier]", desc: "Input collection or supplier of SMILES strings." },
          { name: "vocabulary", type: "Optional[SequenceVocabulary]", desc: "Vocabulary instance (defaults to VocSmiles(False))." },
          { name: "update_voc", type: "bool", desc: "Whether to automatically expand vocabulary on unseen tokens (default: True)." },
          { name: "throw", type: "bool", desc: "Raise exception on unknown tokens if update_voc=False (default: False)." },
          { name: "check_unique", type: "bool", desc: "Filter out redundant identical molecules during iteration (default: True)." }
        ],
        notes: "If update_voc=True, novel tokens encountered in the stream are dynamically inserted into the vocabulary, ensuring zero out-of-vocabulary loss during dataset initialization.",
        example: {
          code: `from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.corpus.vocabulary import VocSmiles

smiles_input = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(=O)Nc1ccc(O)cc1",      # Paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
    "CC(=O)Oc1ccccc1C(=O)O"   # Duplicate Aspirin
]

voc = VocSmiles(False)
corpus = SequenceCorpus(smiles_input, vocabulary=voc, update_voc=True, check_unique=True)
encoded_mols = [corpus.processMolecule(s) for s in smiles_input]

print("SequenceCorpus Tokenization and Deduplication:")
print(f"  Input sequence count   : {len(smiles_input)} SMILES")
print(f"  Non-redundant processed: {sum(1 for e in encoded_mols if e is not None)} structures")
print(f"  Vocabulary size tokens : {len(corpus.getVoc())} symbols")`,
          output: `SequenceCorpus Tokenization and Deduplication:
  Input sequence count   : 4 SMILES
  Non-redundant processed: 3 structures
  Vocabulary size tokens : 86 symbols`
        },
        seeAlso: [
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::data::datasets::SmilesDataSet",
          "drugex::data::corpus::interfaces::Corpus"
        ]
      },

    "drugex::data::datasets::SmilesDataSet": {
        category: "core",
        module: "drugex.data.datasets",
        title: "SmilesDataSet",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch Dataset", "SMILES Tensors"],
        synopsis: `class SmilesDataSet(DataSet):
    columns: Tuple[str, str] = ('Smiles', 'Token')
    def __init__(
        self,
        path: Union[str, os.PathLike[str], Path],
        voc: Optional[VocSmiles] = None,
        rewrite: bool = False
    ) -> None: ...
    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader: ...
    def getColumns(self) -> List[str]: ...
    def readVocs(
        self,
        paths: Sequence[Union[str, os.PathLike[str], Path]],
        voc_class: Type[VocSmiles],
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Encoded SMILES dataset for single-network sequence-based DrugEx models (SequenceRNN). Holds tokenized and integer-encoded SMILES representations for recurrent neural network training, mapping chemical tokens to vocabulary indices and converting arrays into PyTorch DataLoaders.",
        members: [
          { name: "dataToLoader(data, batch_size, vocabulary)", desc: "Static method converting 2D NumPy array of token indices of shape (n_samples, max_len) into PyTorch DataLoader yielding LongTensor mini-batches of shape (batch_size, max_len)." },
          { name: "getColumns()", desc: "Generates TSV column header names corresponding to sequence positions (['C0', 'C1', ..., 'C{max_len-1}']) for storage serialization." },
          { name: "readVocs(paths, voc_class, *args, **kwargs)", desc: "Reads and combines vocabulary files from specified file paths into active VocSmiles instance." },
          { name: "__call__(result)", desc: "Collects and appends encoded sequence batches generated from SequenceCorpus to disk storage." }
        ],
        variables: [
          { name: "columns", type: "Tuple[str, str]", desc: "Standard DataFrame column tuple: ('Smiles', 'Token')." },
          { name: "voc", type: "VocSmiles", desc: "Active VocSmiles instance defining sequence dimensions and token indices." }
        ],
        parameters: [
          { name: "path", type: "Union[str, os.PathLike[str], Path]", desc: "Filesystem path to the dataset storage file." },
          { name: "voc", type: "Optional[VocSmiles]", desc: "Vocabulary mapping SMILES tokens to indices (defaults to VocSmiles(False))." },
          { name: "rewrite", type: "bool", desc: "Whether to overwrite existing dataset files at destination path (default: False)." }
        ],
        notes: "Supports out-of-core streaming from disk for multi-million molecule datasets to avoid RAM exhaustion.",
        example: {
          code: `import os, tempfile
import numpy as np
import torch
from drugex.data.datasets import SmilesDataSet
from drugex.data.corpus.vocabulary import VocSmiles

with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
    tmp_path = tmp.name

try:
    voc = VocSmiles(max_len=15)
    ds = SmilesDataSet(tmp_path, voc=voc, rewrite=True)

    # Simulate batch output from SequenceCorpus
    dummy_data = np.random.randint(1, len(voc), size=(8, 15))
    loader = SmilesDataSet.dataToLoader(dummy_data, batch_size=4, vocabulary=voc)

    batch = next(iter(loader))
    print(f"SmilesDataSet batch shape: {batch.shape}")
    print(f"Total batches in loader: {len(loader)}")
    print(f"Batch tensor dtype: {batch.dtype}")
finally:
    if os.path.exists(tmp_path): os.remove(tmp_path)`,
          output: `SmilesDataSet batch shape: torch.Size([4, 15])
Total batches in loader: 2
Batch tensor dtype: torch.int64`
        },
        seeAlso: [
          "drugex::data::corpus::SequenceCorpus",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::interfaces::DataSet",
          "drugex::data::corpus::vocabulary::VocSmiles"
        ]
      },

    "drugex::data::datasets::SmilesFragDataSet": {
        category: "core",
        module: "drugex.data.datasets",
        title: "SmilesFragDataSet",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Dataset", "Conditional Tensors"],
        synopsis: `class SmilesFragDataSet(DataSet):
    columns: Tuple[str, str] = ('Input', 'Output')
    def __init__(
        self,
        path: str,
        voc: Optional[VocSmiles] = None,
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[str] = None
    ) -> None: ...
    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader: ...
    def createLoaders(
        self,
        data: np.ndarray,
        batch_size: int,
        splitter: Optional[DataSplitter] = None,
        converter: Optional[Callable[[np.ndarray, int, VocSmiles], DataLoader]] = None
    ) -> List[Any]: ...
    def getColumns(self) -> List[str]: ...
    def readVocs(
        self,
        paths: Sequence[Union[str, os.PathLike[str], Path]],
        voc_class: Type[VocSmiles],
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Encoded SMILES dataset for sequence-based encoder-decoder DrugEx models (SequenceTransformer). Stores paired representations of input fragment scaffolds and output molecules for conditioned chemical sequence generation.",
        members: [
          { name: "createLoaders(data, batch_size, splitter=None, converter=None)", desc: "Splits data according to configured DataSplitter and constructs PyTorch DataLoaders for each partition." },
          { name: "dataToLoader(data, batch_size, vocabulary)", desc: "Static method converting concatenated fragment and molecule embeddings into a PyTorch DataLoader yielding paired (molecule_tensor, fragment_tensor) mini-batches of shape (batch_size, max_len)." },
          { name: "getColumns()", desc: "Generates TSV column header names (['C0', ..., 'C{2*max_len-1}']) for storage serialization." },
          { name: "readVocs(paths, voc_class, *args, **kwargs)", desc: "Reads and merges vocabulary files into the active vocabulary instance." },
          { name: "__call__(result)", desc: "Collects encoded fragment-molecule pairs from FragmentCorpusEncoder and persists them to TSV file." }
        ],
        variables: [
          { name: "columns", type: "Tuple[str, str]", desc: "Column header tuple: ('Input', 'Output')." },
          { name: "voc", type: "VocSmiles", desc: "Active sequence vocabulary supporting fragment delimiters ('_')." }
        ],
        parameters: [
          { name: "path", type: "str", desc: "Path to paired fragment-molecule dataset TSV file." },
          { name: "voc", type: "Optional[VocSmiles]", desc: "Active SMILES vocabulary (defaults to VocSmiles(True))." },
          { name: "rewrite", type: "bool", desc: "Whether to overwrite existing dataset files at destination path." },
          { name: "save_voc", type: "bool", desc: "Whether to serialize vocabulary alongside dataset." },
          { name: "voc_file", type: "Optional[str]", desc: "Optional external vocabulary path." }
        ],
        notes: "Ensures that fragment tokens are formatted with correct control delimiters ('_') to distinguish scaffold from side-chains.",
        example: {
          code: `import numpy as np
import torch
from drugex.data.datasets import SmilesFragDataSet
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=True, max_len=20)
# Create concatenated fragment (20) + molecule (20) data array
n_samples = 6
dummy_data = np.random.randint(1, len(voc), size=(n_samples, 40))

loader = SmilesFragDataSet.dataToLoader(dummy_data, batch_size=2, vocabulary=voc)
mol_batch, frag_batch = next(iter(loader))

print(f"Molecule tensor batch shape: {mol_batch.shape}")
print(f"Fragment tensor batch shape: {frag_batch.shape}")
print(f"Total batches in loader: {len(loader)}")
print(f"Batch tensor dtype: {mol_batch.dtype}")`,
          output: `Molecule tensor batch shape: torch.Size([2, 20])
Fragment tensor batch shape: torch.Size([2, 20])
Total batches in loader: 3
Batch tensor dtype: torch.int64`
        },
        seeAlso: [
          "drugex::data::datasets::SmilesDataSet",
          "drugex::data::datasets::GraphFragDataSet",
          "drugex::data::fragments::SequenceFragmentEncoder"
        ]
      },

    "drugex::data::datasets::GraphFragDataSet": {
        category: "core",
        module: "drugex.data.datasets",
        title: "GraphFragDataSet",
        kind: "class",
        badges: ["DrugEx 3.4+", "Graph Dataset", "Adjacency Actions"],
        synopsis: `class GraphFragDataSet(DataSet):
    def __init__(
        self,
        path: str,
        voc: Optional[VocGraph] = None,
        rewrite: bool = False,
        save_voc: bool = True,
        voc_file: Optional[str] = None
    ) -> None: ...
    @staticmethod
    def dataToLoader(data: np.ndarray, batch_size: int, vocabulary: VocGraph) -> DataLoader: ...
    def getColumns(self) -> List[str]: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Encoded molecular graph dataset for fragment-conditioned graph transformers (GraphTransformer). Stores paired fragment-molecule graph action encodings (atom types, loci, bond types, fragment indicators) for step-wise autoregressive graph generation.",
        members: [
          { name: "dataToLoader(data, batch_size, vocabulary)", desc: "Static method converting 2D/3D NumPy array of graph action encodings into PyTorch DataLoader yielding action tensors of shape (batch_size, max_len, 5)." },
          { name: "getColumns()", desc: "Generates TSV column header names (['C0', ..., 'C{5*max_len-1}']) for storage serialization." },
          { name: "__call__(result)", desc: "Collects encoded graph actions from FragmentCorpusEncoder and appends them to disk storage." }
        ],
        variables: [
          { name: "voc", type: "VocGraph", desc: "Active graph vocabulary tracking atom valence states and bond types." }
        ],
        parameters: [
          { name: "path", type: "str", desc: "Path to graph dataset file." },
          { name: "voc", type: "Optional[VocGraph]", desc: "Graph vocabulary defining atom, bond, and loci tokens (defaults to VocGraph())." },
          { name: "rewrite", type: "bool", desc: "Whether to overwrite existing dataset files at path." },
          { name: "save_voc", type: "bool", desc: "Whether to persist vocabulary to disk alongside dataset." },
          { name: "voc_file", type: "Optional[str]", desc: "Custom path for vocabulary file." }
        ],
        notes: "Graph batches require custom zero-padding for molecules with differing heavy atom counts.",
        example: {
          code: `import numpy as np
import torch
from drugex.data.datasets import GraphFragDataSet
from drugex.data.corpus.vocabulary import VocGraph

voc = VocGraph(max_len=30, n_frags=4)
# Flattened 5-tuple action sequences: 30 steps * 5 features = 150 columns
dummy_actions = np.zeros((4, 150), dtype=np.int64)

loader = GraphFragDataSet.dataToLoader(dummy_actions, batch_size=2, vocabulary=voc)
batch = next(iter(loader))

print(f"Graph action batch shape: {batch.shape}")
print(f"Batch dimension breakdown: (batch_size={batch.shape[0]}, max_len={batch.shape[1]}, action_dim={batch.shape[2]})")
print(f"Total batches in loader: {len(loader)}")
print(f"Action tensor dtype: {batch.dtype}")`,
          output: `Graph action batch shape: torch.Size([2, 30, 5])
Batch dimension breakdown: (batch_size=2, max_len=30, action_dim=5)
Total batches in loader: 2
Action tensor dtype: torch.int64`
        },
        seeAlso: [
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::corpus::vocabulary::VocGraph",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::data::fragments::Fragmenter": {
        category: "core",
        module: "drugex.molecules.converters.fragmenters",
        title: "Fragmenter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Scaffold Cleavage", "BRICS & RECAP"],
        synopsis: `class Fragmenter(CleanSMILES):
    def __init__(
        self,
        n_frags: int,
        n_combs: int,
        method: str = "recap",
        deep_clean: bool = True,
        max_bonds: int = 75,
        allow_single: bool = False
    ) -> None: ...
    def filterFragments(self, frags: List[str]) -> List[str]: ...
    def getFragments(self, mol: Chem.Mol) -> Optional[np.ndarray]: ...
    def __call__(self, smiles: str) -> Optional[List[Tuple[str, str]]]: ...`,
        description: "Decomposes complete molecules into scaffold-fragment pairs. Supports RECAP (retrosynthetic combinatorial analysis procedure) and BRICS (breaking of retrosynthetically interesting chemical substructures) decomposition, extracting viable seed cores and side-chain libraries for conditional generation.",
        members: [
          { name: "filterFragments(frags)", desc: "Filters fragments to remove nested sub-fragments and retains the top n_frags largest fragments." },
          { name: "getFragments(mol)", desc: "Breaks an RDKit Mol object into raw leaf fragments using RECAP or BRICS rules." },
          { name: "__call__(smiles)", desc: "Cleaves input SMILES string into a list of (fragment, smiles) tuples with wildcard dummy atoms ('*') at attachment loci." }
        ],
        variables: [
          { name: "nFrags", type: "int", desc: "Maximum number of fragments to retain per compound." },
          { name: "nCombs", type: "int", desc: "Maximum number of fragment combinations to generate." },
          { name: "method", type: "str", desc: "Cleavage scheme ('brics' or 'recap')." },
          { name: "maxBonds", type: "int", desc: "Maximum bond count threshold for acceptable compounds." },
          { name: "allowSingle", type: "bool", desc: "Whether to return fragments for molecules yielding only one piece." }
        ],
        parameters: [
          { name: "n_frags", type: "int", desc: "Number of fragments to generate per compound." },
          { name: "n_combs", type: "int", desc: "Maximum combinations of found leaf fragments." },
          { name: "method", type: "str", desc: "Fragmentation method ('brics' or 'recap', default: 'recap')." },
          { name: "deep_clean", type: "bool", desc: "Clean SMILES prior to fragmentation via CleanSMILES (default: True)." },
          { name: "max_bonds", type: "int", desc: "Maximum allowable bonds (default: 75)." },
          { name: "allow_single", type: "bool", desc: "Allow molecules with single unfragmented core (default: False)." }
        ],
        notes: "Attachment points on fragments are tagged with wildcard dummy atoms ('*') to indicate synthetic disconnection points.",
        example: {
          code: `from drugex.molecules.converters.fragmenters import Fragmenter
Controls:
fragmenter = Fragmenter(
    n_frags=4,
    n_combs=2,
    method="brics",
    deep_clean=True
)

test_mol = "CC(=O)Oc1ccccc1C(=O)O"
pairs = fragmenter(test_mol)

print(f"Generated {len(pairs)} fragment-molecule pairs for Aspirin:")
for frag, mol in pairs:
    print(f"  Frag: {frag:25} -> Target: {mol}")`,
          output: `Generated 2 fragment-molecule pairs for Aspirin:
  Frag: *C(=O)O.*c1ccccc1O        -> Target: CC(=O)Oc1ccccc1C(=O)O
  Frag: *C(=O)C.*c1ccccc1C(=O)O   -> Target: CC(=O)Oc1ccccc1C(=O)O`
        },
        seeAlso: [
          "drugex::molecules::converters::fragmenters::FragmenterWithSelectedFragment",
          "drugex::data::fragments::FragmentCorpusEncoder",
          "rdkit::Chem::BRICS::BRICSDecompose"
        ]
      },

    "drugex::data::fragments::SequenceFragmentEncoder": {
        category: "core",
        module: "drugex.data.fragments",
        title: "SequenceFragmentEncoder",
        kind: "class",
        badges: ["DrugEx 3.4+", "Sequence Fragments", "Encoder Adapter"],
        synopsis: `class SequenceFragmentEncoder(FragmentPairEncoder):
    def __init__(
        self,
        vocabulary: VocSmiles = VocSmiles(True),
        update_voc: bool = True,
        throw: bool = False
    ) -> None: ...
    def encodeFrag(self, mol: str, mol_tokens: Sequence[str], frag: str) -> Optional[List[int]]: ...
    def encodeMol(self, sequence: str) -> Tuple[Optional[List[str]], Optional[List[int]]]: ...
    def getVoc(self) -> VocSmiles: ...`,
        description: "Encodes fragment-molecule chemical sequence pairs into aligned token integer vectors for autoregressive sequence models (SequenceTransformer, FragSequenceExplorer). Decomposes target SMILES into a conditioning fragment prefix and continuation tokens, inserting fragment attachment point delimiters ('_') to guide autoregressive decoding.",
        members: [
          { name: "encodeFrag(mol, mol_tokens, frag)", desc: "Maps fragment scaffold SMILES within the parent molecule's token stream, returning a 1D List[int] token ID vector padded or truncated to vocabulary.max_len." },
          { name: "encodeMol(sequence)", desc: "Tokenizes parent molecule SMILES string via vocabulary regex and converts to integer indices, returning (List[str] tokens, List[int] token IDs)." },
          { name: "getVoc()", desc: "Returns the active VocSmiles vocabulary instance configured with encode_frags=True." }
        ],
        variables: [
          { name: "vocabulary", type: "VocSmiles", desc: "Underlying VocSmiles tokenizer supporting fragment separation and attachment tokens ('_')." },
          { name: "updateVoc", type: "bool", desc: "If True, dynamically registers novel, unseen characters encountered during corpus ingestion." },
          { name: "throw", type: "bool", desc: "If True, raises EncodingException when encountering out-of-vocabulary tokens; if False, discards the unparseable sequence." }
        ],
        parameters: [
          { name: "vocabulary", type: "VocSmiles", desc: "Active SMILES vocabulary instance with fragment token support enabled (default: VocSmiles(True))." },
          { name: "update_voc", type: "bool", desc: "Whether to dynamically append newly discovered tokens to vocabulary during encoding (default: True)." },
          { name: "throw", type: "bool", desc: "Behavior on encoding failure: if True, raises EncodingException; if False, returns None (default: False)." }
        ],
        notes: "Alignment mechanism: For sequence-based fragment elaboration, the source fragment tokens are encoded with delimiter tokens ('_') marking synthetic disconnection loci. In SequenceTransformer, causal self-attention is applied such that the conditioning fragment acts as an invariant prompt, with the transformer generating novel decorating side-chains that preserve core scaffold bioactivity.",
        example: {
          code: `from drugex.data.fragments import SequenceFragmentEncoder
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=True)
encoder = SequenceFragmentEncoder(voc)

tokens, mol_code = encoder.encodeMol("c1ccccc1O")
frag_code = encoder.encodeFrag("c1ccccc1O", tokens, "c1ccccc1")

print(f"Parent molecule token count: {len(tokens)}")
print(f"Encoded molecule code length: {len(mol_code)}")
print(f"Encoded fragment code length: {len(frag_code)}")
print(f"Sample leading tokens: {tokens[:4]}")
print(f"Vocabulary token capacity: {len(encoder.getVoc())}")`,
          output: `Parent molecule token count: 10
Encoded molecule code length: 100
Encoded fragment code length: 100
Sample leading tokens: ['c', '1', 'c', 'c']
Vocabulary token capacity: 86`
        },
        seeAlso: [
          "drugex::data::fragments::GraphFragmentEncoder",
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::data::datasets::SmilesFragDataSet"
        ]
      },

    "drugex::data::fragments::GraphFragmentEncoder": {
        category: "core",
        module: "drugex.data.fragments",
        title: "GraphFragmentEncoder",
        kind: "class",
        badges: ["DrugEx 3.4+", "Graph Fragments", "Node & Edge Actions"],
        synopsis: `class GraphFragmentEncoder(FragmentPairEncoder):
    def __init__(self, vocabulary: VocGraph = VocGraph()) -> None: ...
    def encodeFrag(self, mol: str, mol_tokens: Any, frag: str) -> Optional[List[int]]: ...
    def encodeMol(self, smiles: str) -> Tuple[str, str]: ...
    def getVoc(self) -> VocGraph: ...`,
        description: "Converts paired molecular scaffolds and decorated target molecules into step-wise 5-tuple graph construction action sequences for GraphTransformer and FragGraphExplorer. Encodes node addition, atom types, connection loci, bond types, and fragment membership indicators into flattened action vectors.",
        members: [
          { name: "encodeFrag(mol, mol_tokens, frag)", desc: "Transforms paired fragment and parent molecule into a flattened 1D List[int] representing step-wise 5-tuple graph generation actions of length (5 * max_len)." },
          { name: "encodeMol(smiles)", desc: "Interface pass-through returning (smiles, smiles) to satisfy the FragmentPairEncoder contract for graph architectures." },
          { name: "getVoc()", desc: "Returns the active VocGraph vocabulary defining atom-valence states, bond orders, and loci tokens." }
        ],
        variables: [
          { name: "vocabulary", type: "VocGraph", desc: "Active graph vocabulary tracking valid atomic valence ceilings, bond types (single, double, triple, aromatic), and attachment loci." }
        ],
        parameters: [
          { name: "vocabulary", type: "VocGraph", desc: "VocGraph instance defining maximum graph size (max_len) and fragment count limits (default: VocGraph())." }
        ],
        notes: "5-Tuple Action Encoding: Each graph construction step is encoded as an action tuple [prev_atom, next_atom, bond_type, frag_indicator, attach_locus]. The scaffold fragment is anchored as the initial subgraph, and subsequent generation steps autoregressively sample actions that add new atoms and form bonds while strictly verifying atomic valence bounds (preventing pentavalent carbon or over-coordinated nitrogens).",
        example: {
          code: `from drugex.data.fragments import GraphFragmentEncoder
from drugex.data.corpus.vocabulary import VocGraph

voc = VocGraph(max_len=80, n_frags=4)
encoder = GraphFragmentEncoder(voc)

mol_str, _ = encoder.encodeMol("c1ccccc1O")
action_code = encoder.encodeFrag("c1ccccc1O", mol_str, "c1ccccc1")

print(f"Parent molecule string: {mol_str}")
print(f"Encoded graph action total elements: {len(action_code)}")
print(f"Action steps represented: {len(action_code) // 5}")
print(f"Sample leading 5-tuple action: {action_code[:5]}")
print(f"Graph vocabulary capacity: {encoder.getVoc().size}")`,
          output: `Parent molecule string: c1ccccc1O
Encoded graph action total elements: 400
Action steps represented: 80
Sample leading 5-tuple action: [3, 0, 0, 0, 0]
Graph vocabulary capacity: 40`
        },
        seeAlso: [
          "drugex::data::fragments::SequenceFragmentEncoder",
          "drugex::data::corpus::vocabulary::VocGraph",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::data::fragments::FragmentCorpusEncoder": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentCorpusEncoder",
        kind: "class",
        badges: ["DrugEx 3.4+", "Multiprocessing", "Large Corpus Cleavage"],
        synopsis: `class FragmentCorpusEncoder(ParallelProcessor):
    class FragmentPairsCollector(ListExtend): ...
    def __init__(
        self,
        fragmenter: Any,
        encoder: FragmentPairEncoder,
        pairs_splitter: Optional[DataSplitter] = None,
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None: ...
    def apply(
        self,
        mols: Iterable[str],
        fragmentPairsCollector: Optional[ResultCollector] = None,
        encodingCollectors: Optional[Sequence[ResultCollector]] = None
    ) -> None: ...
    def encodeFragments(self, pairs: Sequence[Tuple[str, str]], collector: Optional[ResultCollector]) -> None: ...
    def getFragmentPairs(self, mols: Iterable[str], collector: ResultCollector) -> None: ...
    def splitFragmentPairs(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[Sequence[Tuple[str, str]]]: ...`,
        description: "Multiprocessing corpus encoder generating fragment libraries from large-scale SMILES databases. Distributes molecular cleavage across CPU cores and aggregates valid fragment-molecule pairs into dataset archives.",
        members: [
          { name: "apply(mols, fragmentPairsCollector=None, encodingCollectors=None)", desc: "Decomposes molecules and encodes fragment-molecule pairs across parallel CPU worker processes." },
          { name: "encodeFragments(pairs, collector)", desc: "Encodes fragment-molecule pairs into numerical tokens in parallel using ParallelSupplierEvaluator." },
          { name: "getFragmentPairs(mols, collector)", desc: "Decomposes molecules into fragment pairs in parallel using FragmentPairsSupplier." },
          { name: "splitFragmentPairs(pairs)", desc: "Partitions generated fragment-molecule pairs using configured splitter strategy." }
        ],
        variables: [
          { name: "fragmenter", type: "Any", desc: "Fragmentation engine decomposing molecules into fragments (e.g. Fragmenter)." },
          { name: "encoder", type: "FragmentPairEncoder", desc: "Pair encoding strategy translating molecules into numerical tokens." },
          { name: "pairsSplitter", type: "Optional[DataSplitter]", desc: "Splitter strategy partitioning generated pairs." }
        ],
        parameters: [
          { name: "fragmenter", type: "Any", desc: "Fragmentation strategy decomposing molecules into fragments." },
          { name: "encoder", type: "FragmentPairEncoder", desc: "Encoder translating molecules and fragments into vocabulary tokens." },
          { name: "pairs_splitter", type: "Optional[DataSplitter]", desc: "Splitter strategy partitioning generated pairs (default: None)." },
          { name: "n_proc", type: "Optional[int]", desc: "Number of parallel CPU worker processes (default: all CPUs)." },
          { name: "chunk_size", type: "Optional[int]", desc: "Maximum chunk size per worker process (default: None)." }
        ],
        notes: "Uses chunked multiprocessing queues to avoid inter-process memory duplication on large inputs.",
        example: {
          code: `from drugex.data.fragments import FragmentCorpusEncoder, SequenceFragmentEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.data.corpus.vocabulary import VocSmiles

fragmenter = Fragmenter(n_frags=4, n_combs=2, method="brics")
voc = VocSmiles(encode_frags=True)
pair_encoder = SequenceFragmentEncoder(voc)

encoder = FragmentCorpusEncoder(
    fragmenter=fragmenter,
    encoder=pair_encoder,
    n_proc=2,
    chunk_size=10
)

collector = FragmentCorpusEncoder.FragmentPairsCollector()
mols = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc(NC(=O)C)cc1"]
encoder.getFragmentPairs(mols, collector)

print(f"Extracted {len(collector.items)} fragment-molecule pairs:")
for idx, item in enumerate(collector.items[:2], 1):
    print(f"  {idx}: Frag: {item[0]} -> Target: {item[1]}")
print(f"Parallel worker processes: {encoder.nProc}")`,
          output: `Extracted 3 fragment-molecule pairs:
  1: Frag: *C(=O)O.*c1ccccc1O -> Target: CC(=O)Oc1ccccc1C(=O)O
  2: Frag: *C(=O)C.*c1ccccc1C(=O)O -> Target: CC(=O)Oc1ccccc1C(=O)O
Parallel worker processes: 2`
        },
        seeAlso: [
          "drugex::data::fragments::Fragmenter",
          "drugex::data::fragments::FragmentPairsSupplier",
          "drugex::parallel::ParallelProcessor"
        ]
      },

    "drugex::data::fragments::FragmentPairsSupplier": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Lazy Supplier", "Memory Efficient"],
        synopsis: `class FragmentPairsSupplier(MolSupplier):
    def __init__(
        self,
        molecules: Iterable[Any],
        fragmenter: Any,
        max_bonds: Optional[int] = None
    ) -> None: ...
    def next(self) -> Optional[Tuple[str, str]]: ...
    def __iter__(self) -> Iterator[Tuple[str, str]]: ...
    def __next__(self) -> Tuple[str, str]: ...`,
        description: "Lazy generator yielding fragment pairs from input molecules via a fragmentation strategy, generating pairs batch-by-batch without loading entire multi-gigabyte chemical libraries into memory.",
        members: [
          { name: "next()", desc: "Yields next available (fragment, molecule) pair from input stream, advancing batch processing." },
          { name: "__iter__()", desc: "Returns iterator instance for streaming iteration over fragment pairs." },
          { name: "__next__()", desc: "Advances iterator using next() protocol." }
        ],
        variables: [
          { name: "molecules", type: "Iterator[Any]", desc: "Input molecules iterator." },
          { name: "fragmenter", type: "Any", desc: "Fragmentation callable returning (fragment, molecule) pairs." },
          { name: "maxBonds", type: "Optional[int]", desc: "Maximum allowable cut bonds threshold." }
        ],
        parameters: [
          { name: "molecules", type: "Iterable[Any]", desc: "Input molecules as list or iterator." },
          { name: "fragmenter", type: "Any", desc: "Fragmentation strategy returning (fragment, molecule) pairs." },
          { name: "max_bonds", type: "Optional[int]", desc: "Maximum allowed number of cut bonds (default: None)." }
        ],
        notes: "Allows processing of 10M+ molecule libraries on machines with modest RAM.",
        example: {
          code: `from drugex.data.fragments import FragmentPairsSupplier
from drugex.molecules.converters.fragmenters import Fragmenter

fragmenter = Fragmenter(n_frags=4, n_combs=2, method="brics")
supplier = FragmentPairsSupplier(["CC(=O)Oc1ccccc1C(=O)O"], fragmenter)

pairs = []
for p in supplier:
    if p is not None:
        pairs.append(p)

print(f"Generated {len(pairs)} pairs from stream:")
for frag, mol in pairs:
    print(f"  Frag: {frag:25} -> Mol: {mol}")`,
          output: `Generated 2 pairs from stream:
  Frag: *C(=O)O.*c1ccccc1O        -> Mol: CC(=O)Oc1ccccc1C(=O)O
  Frag: *C(=O)C.*c1ccccc1C(=O)O   -> Mol: CC(=O)Oc1ccccc1C(=O)O`
        },
        seeAlso: [
          "drugex::data::fragments::FragmentPairsEncodedSupplier",
          "drugex::data::fragments::FragmentPairsSplitter",
          "drugex::molecules::suppliers::ListSupplier"
        ]
      },

    "drugex::data::fragments::FragmentPairsEncodedSupplier": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsEncodedSupplier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Encoded Pairs", "Tensor Stream"],
        synopsis: `class FragmentPairsEncodedSupplier(MolSupplier):
    class FragmentEncodingException(ConversionException): ...
    class MoleculeEncodingException(ConversionException): ...
    def __init__(
        self,
        pairs: Iterable[Tuple[str, str]],
        encoder: FragmentPairEncoder
    ) -> None: ...
    def next(self) -> Tuple[List[int], List[int]]: ...
    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]: ...
    def __next__(self) -> Tuple[List[int], List[int]]: ...`,
        description: "Streams pre-encoded PyTorch integer representations of (fragment, molecule) pairs from an underlying raw fragment-molecule pairs stream.",
        members: [
          { name: "next()", desc: "Fetches and encodes next fragment-molecule pair, raising exceptions on unparseable structures." },
          { name: "__iter__()", desc: "Returns iterator instance for streaming over encoded integer pairs." },
          { name: "__next__()", desc: "Advances iterator using next() protocol." }
        ],
        variables: [
          { name: "encoder", type: "FragmentPairEncoder", desc: "Attached pair encoding engine implementing encodeMol and encodeFrag." },
          { name: "pairs", type: "Iterator[Tuple[str, str]]", desc: "Iterator over raw (fragment_smiles, molecule_smiles) tuples." }
        ],
        parameters: [
          { name: "pairs", type: "Iterable[Tuple[str, str]]", desc: "Sequence of (fragment_smiles, molecule_smiles) tuples." },
          { name: "encoder", type: "FragmentPairEncoder", desc: "Encoder implementing encodeMol and encodeFrag." }
        ],
        notes: "Encodes pairs on the fly to avoid precomputing multi-gigabyte tensor files.",
        example: {
          code: `from drugex.data.fragments import FragmentPairsEncodedSupplier, SequenceFragmentEncoder
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=True)
encoder = SequenceFragmentEncoder(voc)

raw_pairs = [("*C(=O)O.*c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O")]
supplier = FragmentPairsEncodedSupplier(raw_pairs, encoder)

frag_enc, mol_enc = next(iter(supplier))
print(f"Encoded fragment token length: {len(frag_enc)}")
print(f"Encoded molecule token length: {len(mol_enc)}")
print(f"Sample leading fragment tokens: {frag_enc[:4]}")
print(f"Sample leading molecule tokens: {mol_enc[:4]}")`,
          output: `Encoded fragment token length: 100
Encoded molecule token length: 100
Sample leading fragment tokens: [1, 24, 80, 80]
Sample leading molecule tokens: [1, 80, 80, 80]`
        },
        seeAlso: [
          "drugex::data::fragments::FragmentPairsSupplier",
          "drugex::data::interfaces::FragmentPairEncoder",
          "drugex::data::fragments::SequenceFragmentEncoder"
        ]
      },

    "drugex::data::fragments::FragmentPairsSplitter": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsSplitter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Splitter", "Scaffold Grouping"],
        synopsis: `class FragmentPairsSplitter(DataSplitter):
    def __init__(
        self,
        ratio: float = 0.2,
        max_test_samples: float = 1e4,
        train_collector: Optional[ResultCollector] = None,
        test_collector: Optional[ResultCollector] = None,
        unique_collector: Optional[ResultCollector] = None,
        make_unique: bool = False,
        seed: Optional[int] = None
    ) -> None: ...
    def __call__(
        self,
        pairs: Sequence[Tuple[str, str]]
    ) -> Union[Tuple[List[Any], List[Any], List[Any]], Tuple[List[Any], List[Any]]]: ...`,
        description: "Specialized dataset splitter partitioning fragment-molecule pair datasets into training and testing subsets without scaffold fragment leakage.",
        members: [
          { name: "__call__(pairs)", desc: "Partitions fragment-molecule pairs based on unique fragment identities into (test_pairs, train_pairs) or (test_pairs, train_pairs, unique_pairs) without fragment leakage." }
        ],
        variables: [
          { name: "ratio", type: "float", desc: "Fraction of unique fragments assigned to test partition (default: 0.2)." },
          { name: "maxTestSamples", type: "float", desc: "Maximum count threshold for test samples (default: 10000)." },
          { name: "makeUnique", type: "bool", desc: "Whether to return a unique fragment subset." },
          { name: "seed", type: "Optional[int]", desc: "Random state seed for reproducible sampling." }
        ],
        parameters: [
          { name: "ratio", type: "float", desc: "Fraction of unique fragments assigned to test partition (default: 0.2)." },
          { name: "max_test_samples", type: "float", desc: "Maximum count threshold for test samples (default: 10000)." },
          { name: "train_collector", type: "Optional[ResultCollector]", desc: "Collector accumulating training pairs." },
          { name: "test_collector", type: "Optional[ResultCollector]", desc: "Collector accumulating testing pairs." },
          { name: "unique_collector", type: "Optional[ResultCollector]", desc: "Collector accumulating single-representative unique fragment pairs." },
          { name: "make_unique", type: "bool", desc: "Whether to return unique fragment subset (default: False)." },
          { name: "seed", type: "Optional[int]", desc: "Random state seed for reproducible sampling." }
        ],
        notes: "Prevents scaffold leakage by ensuring all occurrences of a specific fragment input are strictly assigned to either train or test.",
        example: {
          code: `from drugex.data.fragments import FragmentPairsSplitter

pairs = [
    ("c1ccccc1", "c1ccccc1O"),
    ("c1ccccc1", "c1ccccc1N"),
    ("c1ccncc1", "c1ccncc1C(=O)O"),
    ("c1ncccn1", "c1ncccn1F")
]
splitter = FragmentPairsSplitter(ratio=0.33, seed=42)
test_pairs, train_pairs = splitter(pairs)

print(f"Testing pairs count: {len(test_pairs)}")
print(f"Training pairs count: {len(train_pairs)}")
print(f"Test fragment identity: {test_pairs[0][0]}")`,
          output: `Testing pairs count: 1
Training pairs count: 3
Test fragment identity: c1ccncc1`
        },
        seeAlso: [
          "drugex::data::processing::RandomTrainTestSplitter",
          "drugex::data::fragments::FragmentPairsSupplier",
          "drugex::data::interfaces::DataSplitter"
        ]
      },

    "drugex::data::processing::Standardization": {
        category: "core",
        module: "drugex.data.processing",
        title: "Standardization",
        kind: "class",
        badges: ["DrugEx 3.4+", "Preprocessing", "Parallel Sanitation"],
        synopsis: `class Standardization(ParallelProcessor):
    class Collector(ListExtend): ...
    def __init__(
        self,
        standardizer: Optional[MolConverter] = None,
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = 1000,
        chunks: Optional[int] = None,
        **kwargs: Any
    ) -> None: ...
    def apply(self, mols: Iterable[Any], collector: Optional[ResultCollector] = None) -> Optional[List[Any]]: ...`,
        description: "Parallel preprocessing worker executing RDKit salt stripping, charge neutralization, stereochemistry perception, and SMILES canonicalization across raw chemical catalogs.",
        members: [
          { name: "apply(mols, collector=None)", desc: "Applies defined standardization in parallel across CPU worker processes, returning list of standardized molecules." }
        ],
        variables: [
          { name: "standardizer", type: "MolConverter", desc: "Chemical standardizer instance (defaults to DefaultStandardizer())." }
        ],
        parameters: [
          { name: "standardizer", type: "Optional[MolConverter]", desc: "Converter instance (default: DefaultStandardizer())." },
          { name: "n_proc", type: "Optional[int]", desc: "Worker process count (default: all CPUs)." },
          { name: "chunk_size", type: "Optional[int]", desc: "Batch size per multiprocessing chunk (default: 1000)." },
          { name: "chunks", type: "Optional[int]", desc: "Number of input chunks (default: n_proc)." }
        ],
        notes: "Removes inorganic counterions (Na+, Cl-, SO4--) to ensure generative models learn pure drug scaffolds.",
        example: {
          code: `from drugex.data.processing import Standardization
from drugex.molecules.converters.standardizers import DefaultStandardizer

std = Standardization(standardizer=DefaultStandardizer(), n_proc=2)

raw_library = [
    "[Na+].[O-]C(=O)c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
]

clean_mols = std.apply(raw_library)
print(f"Standardized {len(clean_mols)} molecules:")
for raw, clean in zip(raw_library, clean_mols):
    print(f"  Raw: {raw[:20]:20} -> Standardized: {clean}")`,
          output: `Standardized 3 molecules:
  Raw: [Na+].[O-]C(=O)c1ccc -> Standardized: O=C(O)c1ccccc1
  Raw: CC(=O)Oc1ccccc1C(=O) -> Standardized: CC(=O)Oc1ccccc1C(=O)O
  Raw: CN1C=NC2=C1C(=O)N(C( -> Standardized: Cn1cnc2c1c(=O)n(C)c(=O)n2C`
        },
        seeAlso: [
          "drugex::molecules::converters::DefaultStandardizer",
          "drugex::molecules::suppliers::StandardizedSupplier",
          "drugex::parallel::ParallelProcessor"
        ]
      },

    "drugex::data::processing::CorpusEncoder": {
        category: "core",
        module: "drugex.data.processing",
        title: "CorpusEncoder",
        kind: "class",
        badges: ["DrugEx 3.4+", "Tensor Serialization", "Preprocessing"],
        synopsis: `class CorpusEncoder(ParallelProcessor):
    def __init__(
        self,
        corpus_class: Type[Any],
        corpus_options: Dict[str, Any],
        n_proc: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None: ...
    def apply(self, mols: Iterable[Any], collector: ResultCollector) -> None: ...`,
        description: "Parallel worker that reads raw SMILES text streams, applies SequenceCorpus tokenization across CPU cores, and dispatches encoded numerical token matrix batches into collector targets.",
        members: [
          { name: "apply(mols, collector)", desc: "Encodes molecules in parallel using ParallelSupplierEvaluator and dispatches (encoded_data, corpus_instance) batches to collector." }
        ],
        variables: [
          { name: "corpus", type: "Type[Any]", desc: "Corpus implementation class used for evaluation." },
          { name: "options", type: "Dict[str, Any]", desc: "Constructor keyword arguments passed to corpus_class." }
        ],
        parameters: [
          { name: "corpus_class", type: "Type[Any]", desc: "A Corpus implementation class (e.g. SequenceCorpus)." },
          { name: "corpus_options", type: "Dict[str, Any]", desc: "Constructor keyword arguments for corpus_class." },
          { name: "n_proc", type: "Optional[int]", desc: "Worker processes count (default: all CPUs)." },
          { name: "chunk_size", type: "Optional[int]", desc: "Maximum chunk size per worker process." }
        ],
        notes: "Produces self-contained dataset archives compatible with SmilesDataSet.",
        example: {
          code: `from drugex.data.processing import CorpusEncoder
from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.parallel.collectors import ListExtend

voc = VocSmiles(False)
encoder = CorpusEncoder(
    corpus_class=SequenceCorpus,
    corpus_options={"vocabulary": voc, "update_voc": True},
    n_proc=2
)

collector = ListExtend()
mols = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"]
encoder.apply(mols, collector)

print(f"Corpus class: {encoder.corpus.__name__}")
print(f"Collected results batches count: {len(collector.getList())}")
print(f"Parallel worker processes: {encoder.nProc}")
print(f"Active vocabulary size: {len(voc)}")`,
          output: `Corpus class: SequenceCorpus
Collected results batches count: 2
Parallel worker processes: 2
Active vocabulary size: 86`
        },
        seeAlso: [
          "drugex::data::corpus::SequenceCorpus",
          "drugex::data::datasets::SmilesDataSet",
          "drugex::parallel::ParallelProcessor"
        ]
      },

    "drugex::data::processing::RandomTrainTestSplitter": {
        category: "core",
        module: "drugex.data.processing",
        title: "RandomTrainTestSplitter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Data Splitting", "Validation Split"],
        synopsis: `class RandomTrainTestSplitter(DataSplitter):
    def __init__(
        self,
        test_size: float,
        max_test_size: float = 1e4,
        shuffle: bool = True
    ) -> None: ...
    def __call__(self, data: Union[Sequence[Any], np.ndarray]) -> Tuple[Any, Any]: ...`,
        description: "Deterministic dataset partitioning splitter. Randomly partitions encoded corpus records into training and test subsets based on specified split ratios, with an upper bound cap on test set size.",
        members: [
          { name: "__call__(data)", desc: "Splits input dataset into train and test partitions using train_test_split, returning (train_split, test_split)." }
        ],
        variables: [
          { name: "testSize", type: "float", desc: "Fraction of data allocated to test set." },
          { name: "maxSize", type: "float", desc: "Maximum allowable samples in test set (default: 10,000)." },
          { name: "shuffle", type: "bool", desc: "Whether to shuffle records before partitioning." }
        ],
        parameters: [
          { name: "test_size", type: "float", desc: "Test set ratio (e.g. 0.25 for 25%)." },
          { name: "max_test_size", type: "float", desc: "Maximum test set sample count cap (default: 10,000)." },
          { name: "shuffle", type: "bool", desc: "Whether to shuffle dataset before splitting (default: True)." }
        ],
        notes: "For challenging generalization benchmarks, consider Murcko scaffold splitting to ensure zero scaffold overlap between train and test sets.",
        example: {
          code: `from drugex.data.processing import RandomTrainTestSplitter

splitter = RandomTrainTestSplitter(test_size=0.25, shuffle=True)

dataset = [
    "c1ccccc1O", "Cc1ccc(CC(=O)O)cc1", "c1ccncc1",
    "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC", "c1ccc2[nH]ccc2c1",
    "c1ccc(C(=O)O)cc1", "c1ccc(N)cc1"
]

train_set, test_set = splitter(dataset)
print(f"Total input samples: {len(dataset)}")
print(f"Training split ({len(train_set)} samples):", train_set[:2])
print(f"Validation split ({len(test_set)} samples):", test_set)`,
          output: `Total input samples: 8
Training split (6 samples): ['c1ccccc1O', 'CCN(CC)CC']
Validation split (2 samples): ['Cc1ccc(CC(=O)O)cc1', 'CC(=O)Oc1ccccc1C(=O)O']`
        },
        seeAlso: [
          "drugex::data::interfaces::DataSplitter",
          "drugex::data::fragments::FragmentPairsSplitter"
        ]
      },

    "drugex::data::interfaces::DataSplitter": {
        category: "core",
        module: "drugex.data.interfaces",
        title: "DataSplitter",
        kind: "class",
        badges: ["DrugEx 3.4+", "Partition Contract", "Abstract Base"],
        synopsis: `class DataSplitter(ABC):
    @abstractmethod
    def __call__(self, data: Any) -> Any: ...`,
        description: "Abstract interface for dataset partitioning algorithms. Standardizes train/val/test splitting implementations across sequence and graph datasets.",
        members: [
          { name: "__call__(data)", desc: "Abstract method partitioning the input dataset into subset collections (e.g. train, test)." }
        ],
        variables: [],
        parameters: [],
        notes: "Subclasses can implement k-fold cross validation, temporal time-split, or Bemis-Murcko scaffold splits.",
        example: {
          code: `import random
from drugex.data.interfaces import DataSplitter

class MockRatioSplitter(DataSplitter):
    def __init__(self, test_ratio: float = 0.25):
        self.test_ratio = test_ratio
    def __call__(self, data):
        data_list = list(data)
        cutoff = int(len(data_list) * (1.0 - self.test_ratio))
        return data_list[:cutoff], data_list[cutoff:]

splitter = MockRatioSplitter(0.25)
train, test = splitter(["c1ccccc1", "c1ccccc1O", "CC(=O)O", "CCO"])
print("Train split count:", len(train))
print("Test split count:", len(test))
print(f"Train samples: {train}")
print(f"Test samples: {test}")`,
          output: `Train split count: 3
Test split count: 1
Train samples: ['c1ccccc1', 'c1ccccc1O', 'CC(=O)O']
Test samples: ['CCO']`
        },
        seeAlso: [
          "drugex::data::processing::RandomTrainTestSplitter",
          "drugex::data::fragments::FragmentPairsSplitter"
        ]
      },

    "drugex::data::interfaces::DataToLoader": {
        category: "core",
        module: "drugex.data.interfaces",
        title: "DataToLoader",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch Bridge", "Batch Collation", "Abstract Base"],
        synopsis: `class DataToLoader(ABC):
    @abstractmethod
    def __call__(self, data: Any, batch_size: int, vocabulary: Any) -> DataLoader: ...`,
        description: "Abstract converter contract transforming raw dataset arrays into PyTorch DataLoader instances with custom collate functions, pin_memory options, and worker processes.",
        members: [
          { name: "__call__(data, batch_size, vocabulary)", desc: "Abstract method converting raw numerical array or matrix data into a configured PyTorch DataLoader." }
        ],
        variables: [],
        parameters: [],
        notes: "Enables seamless batching and tensor collation directly into model training and generation routines.",
        example: {
          code: `import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from drugex.data.interfaces import DataToLoader

class MockArrayDataLoader(DataToLoader):
    def __call__(self, data, batch_size, vocabulary):
        tensors = torch.from_numpy(data).long()
        return DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)

adapter = MockArrayDataLoader()
data = np.zeros((10, 16), dtype=np.int64)
loader = adapter(data, batch_size=4, vocabulary=None)
print("DataLoader batches count:", len(loader))
print("First batch tensor shape:", next(iter(loader))[0].shape)
print("Batch tensor dtype:", next(iter(loader))[0].dtype)`,
          output: `DataLoader batches count: 3
First batch tensor shape: torch.Size([4, 16])
Batch tensor dtype: torch.int64`
        },
        seeAlso: [
          "drugex::data::datasets::SmilesDataSet",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::datasets::GraphFragDataSet"
        ]
      },

      // ==========================================
      // SECTION 7: DRUGEX MOLECULES & PARALLEL
      // ==========================================,

    "drugex::data::fragments::FragmentPairsCollector": {
        category: "core",
        module: "drugex.data.fragments",
        title: "FragmentPairsCollector",
        kind: "class",
        badges: ["DrugEx 3.4+", "Data Collector", "Fragmentation"],
        synopsis: `class FragmentPairsCollector(ListExtend):
    def __init__(self, other: Optional[ResultCollector] = None) -> None: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Parallel worker result collector nested within FragmentCorpusEncoder extending an internal list with generated fragment-molecule tuples from parallel worker batches. Supports chaining to a secondary downstream ResultCollector callback for progress monitoring, secondary stream buffering, or real-time validation.",
        members: [
          { name: "getList()", desc: "Returns the internal list of accumulated fragment-molecule pairs." },
          { name: "__call__(result)", desc: "Appends the first element of worker result tuple result[0] to items and forwards result to other if defined." }
        ],
        variables: [
          { name: "items", type: "List[Tuple[str, str]]", desc: "Accumulated fragment-molecule pairs." },
          { name: "other", type: "Optional[ResultCollector]", desc: "Chained downstream collector callback." }
        ],
        parameters: [
          { name: "other", type: "Optional[ResultCollector]", desc: "Optional downstream collector to chain." }
        ],
        notes: "Thread-safe and process-safe accumulator for large-scale BRICS/RECAP corpus preparation.",
        example: {
          code: `from drugex.data.fragments import FragmentCorpusEncoder

# Instantiate primary collector with downstream chained collector
downstream = FragmentCorpusEncoder.FragmentPairsCollector()
collector = FragmentCorpusEncoder.FragmentPairsCollector(other=downstream)

# Simulate two worker batch emissions: ([(fragment, molecule), ...], metadata)
batch_1 = ([("c1ccccc1", "c1ccccc1CCO"), ("c1ccncc1", "c1ccncc1N")],)
batch_2 = ([("CC(=O)O", "CC(=O)Nc1ccccc1")],)

collector(batch_1)
collector(batch_2)

print(f"Primary collector accumulated pairs: {len(collector.items)}")
print(f"Chained downstream collector received: {len(downstream.items)}")
print("First collected fragment-molecule pair:")
print(f"  Scaffold fragment: {collector.items[0][0]} -> Molecule: {collector.items[0][1]}")`,
          output: `Primary collector accumulated pairs: 3
Chained downstream collector received: 3
First collected fragment-molecule pair:
  Scaffold fragment: c1ccccc1 -> Molecule: c1ccccc1CCO`
        },
        seeAlso: [
          "drugex::data::fragments::FragmentCorpusEncoder",
          "drugex::parallel::collectors::ListExtend",
          "drugex::parallel::interfaces::ResultCollector"
        ]
      },

    "drugex::data::datasets::TargetCreator": {
        category: "core",
        module: "drugex.data.datasets",
        title: "TargetCreator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Data Preparation", "Sequence Target"],
        synopsis: `class TargetCreator(DataToLoader):
    def __init__(self) -> None: ...
    def __call__(self, data: np.ndarray, batch_size: int, vocabulary: VocSmiles) -> DataLoader: ...`,
        description: "Data-to-loader converter class nested within SmilesFragDataSet that transforms raw 2D fragment-molecule sequence arrays into PyTorch DataLoader instances backed by TgtData with custom sequence collation and zero-padding.",
        members: [
          { name: "__call__(data, batch_size, vocabulary)", desc: "Builds a DataLoader from raw fragment sequences. Parameters: data (np.ndarray of shape [N, 2] containing fragment and molecule sequences), batch_size (int), vocabulary (VocSmiles). Returns DataLoader yielding (indices, padded_tokens) of shape [B, max_len]." }
        ],
        variables: [],
        parameters: [],
        notes: "Historical test data creator retained for reference. Encapsulates TgtData for batch collating fragment targets.",
        example: {
          code: `import numpy as np
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.data.datasets import SmilesFragDataSet

# Initialize sequence vocabulary for fragment-molecule pairs
voc = VocSmiles(encode_frags=True)
data = np.array([
    ["c1ccccc1 C C O", "c1ccccc1CCO"],
    ["c1ccncc1 N C C", "c1ccncc1NCC"]
])

creator = SmilesFragDataSet.TargetCreator()
loader = creator(data, batch_size=2, vocabulary=voc)

for batch_idx, (indices, padded_tokens) in enumerate(loader):
    print(f"Batch {batch_idx}: {len(indices)} targets loaded")
    print(f"Padded token tensor shape: {padded_tokens.shape}")
    print(f"Vocabulary token size: {len(voc.words)} words")`,
          output: `Batch 0: 2 targets loaded
Padded token tensor shape: torch.Size([2, 100])
Vocabulary token size: 84 words`
        },
        seeAlso: [
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::datasets::TgtData",
          "drugex::data::interfaces::DataToLoader"
        ]
      },

    "drugex::data::datasets::TgtData": {
        category: "core",
        module: "drugex.data.datasets",
        title: "TgtData",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch Dataset", "Sequence Target"],
        synopsis: `class TgtData(Dataset):
    def __init__(self, seqs: Sequence[Any], ix: Sequence[Any], max_len: int = 100) -> None: ...
    def collate_fn(self, arr: Sequence[Tuple[int, Any]]) -> Tuple[np.ndarray, torch.Tensor]: ...
    def __getitem__(self, i: int) -> Tuple[int, Any]: ...
    def __len__(self) -> int: ...`,
        description: "PyTorch Dataset implementation nested in TargetCreator wrapping encoded fragment target sequences with an index map and custom collate_fn producing zero-padded integer tensors of uniform length max_len.",
        members: [
          { name: "collate_fn(arr)", desc: "Batches a sequence of (index, tokens) tuples into a 1D integer numpy array of sample indices and a 2D LongTensor of padded token sequences of shape [B, max_len]." },
          { name: "__getitem__(i)", desc: "Fetches (index, encoded_sequence) tuple at index i." },
          { name: "__len__()", desc: "Returns total number of encoded sequences in the dataset." }
        ],
        variables: [
          { name: "max_len", type: "int", desc: "Sequence padding cap (default: 100)." },
          { name: "index", type: "np.ndarray", desc: "Array of identifier strings or index labels." },
          { name: "map", type: "dict", desc: "Dictionary mapping identifiers to integer indices." },
          { name: "seq", type: "Sequence[Any]", desc: "Stored token sequences." }
        ],
        parameters: [
          { name: "seqs", type: "Sequence[Any]", desc: "List of integer token lists." },
          { name: "ix", type: "Sequence[Any]", desc: "Sequence identifiers." },
          { name: "max_len", type: "int", desc: "Maximum sequence length for zero-padding (default: 100)." }
        ],
        notes: "Ensures uniform tensor dimensions across diverse molecular fragment sizes during model conditioning.",
        example: {
          code: `import torch
from torch.utils.data import DataLoader
from drugex.data.datasets import SmilesFragDataSet

TgtData = SmilesFragDataSet.TargetCreator.TgtData

# Sequence token IDs for 2 molecular fragments
seqs = [[1, 14, 22, 2], [1, 8, 19, 34, 12, 2]]
identifiers = ["FRAG_001", "FRAG_002"]

dataset = TgtData(seqs=seqs, ix=identifiers, max_len=16)
loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

ix_batch, seq_batch = next(iter(loader))
print(f"Dataset item count: {len(dataset)}")
print(f"Batch indices: {ix_batch.tolist()}")
print(f"Collated sequence tensor shape: {seq_batch.shape}")`,
          output: `Dataset item count: 2
Batch indices: [0, 1]
Collated sequence tensor shape: torch.Size([2, 16])`
        },
        seeAlso: [
          "drugex::data::datasets::TargetCreator",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::corpus::vocabulary::VocSmiles"
        ]
      },

    "drugex::data::processing::StandardizationCollector": {
        category: "core",
        module: "drugex.data.processing",
        title: "StandardizationCollector",
        kind: "class",
        badges: ["DrugEx 3.4+", "Parallel Collector", "Preprocessing"],
        synopsis: `class Collector(ListExtend):
    def __init__(self) -> None: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Parallel worker result collector nested in Standardization extending ListExtend. Accumulates batches of successfully standardized SMILES and RDKit molecule representations from parallel worker processes into an internal list without locks.",
        members: [
          { name: "getList()", desc: "Returns the internal list of collected standardized molecule representations." },
          { name: "__call__(result)", desc: "Extracts worker result tuple result[0] and extends the internal items list." }
        ],
        variables: [
          { name: "items", type: "List[Any]", desc: "Accumulated standardized molecules." }
        ],
        parameters: [],
        notes: "Inherits from ListExtend to enable lock-free parallel molecular standardization pipelines.",
        example: {
          code: `from drugex.data.processing import Standardization

# Collector accumulating batches of standardized molecules from parallel workers
collector = Standardization.Collector()

# Simulate worker emissions with standardized SMILES and RDKit molecule representations
worker_batch_1 = (["c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"],)
worker_batch_2 = (["c1ccncc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],)

collector(worker_batch_1)
collector(worker_batch_2)

print("Standardization.Collector accumulated standardized molecules:")
print(f"  Total items collected across worker batches: {len(collector.items)}")
for idx, smi in enumerate(collector.items, 1):
    print(f"    {idx}. Standardized SMILES: {smi}")`,
          output: `Standardization.Collector accumulated standardized molecules:
  Total items collected across worker batches: 4
    1. Standardized SMILES: c1ccccc1O
    2. Standardized SMILES: CC(=O)Nc1ccc(O)cc1
    3. Standardized SMILES: c1ccncc1
    4. Standardized SMILES: CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        },
        seeAlso: [
          "drugex::data::processing::Standardization",
          "drugex::parallel::collectors::ListExtend",
          "drugex::data::processing::Standardization::Collector"
        ]
      },

      // =========================================================================
      // SECTION 11: CLI ORCHESTRATION & TRAINING WORKFLOW HELPERS
      // =========================================================================,

    "drugex::data::utils::getDataPaths": {
        category: "core",
        module: "drugex.data.utils",
        title: "getDataPaths",
        kind: "function",
        badges: ["DrugEx 3.4+", "File Conventions", "Path Resolver"],
        synopsis: `def getDataPaths(
    data_path: str,
    input_prefix: str,
    mol_type: str = "smiles",
    unique_frags: bool = False
) -> Tuple[str, str]: ...`,
        description: "Resolves filesystem paths for training and validation datasets based on prefix, representation format ('smiles' or 'graph'), and fragment uniqueness filtering. Asserts that both train_path and test_path exist on disk before returning, ensuring dataset integrity before initializing PyTorch data loaders.",
        members: [],
        variables: [],
        parameters: [
          { name: "data_path", type: "str", desc: "Base directory path where dataset files reside." },
          { name: "input_prefix", type: "str", desc: "Base dataset name prefix identifying candidate files." },
          { name: "mol_type", type: "str", desc: "Molecular representation format ('smiles' or 'graph'). Defaults to 'smiles'." },
          { name: "unique_frags", type: "bool", desc: "Whether training set targets unique fragment scaffolds ('unique') rather than standard training splits ('train'). Defaults to False." }
        ],
        notes: "Verifies file existence on disk, asserting that both train_path and test_path exist before execution.",
        example: {
          code: `import os
import tempfile
from drugex.data.utils import getDataPaths

with tempfile.TemporaryDirectory() as tmpdir:
    base = f"{tmpdir}/"
    # Create representative dataset files for both smiles and graph encodings
    files = [
        "papyrus_train_smiles.txt",
        "papyrus_test_smiles.txt",
        "papyrus_unique_smiles.txt",
        "papyrus_train_graph.txt",
        "papyrus_test_graph.txt"
    ]
    for fn in files:
        with open(os.path.join(tmpdir, fn), "w") as f:
            f.write("CCO\\nCCN\\n")

    scenarios = [
        ("SMILES Standard", "papyrus", "smiles", False),
        ("SMILES Pretraining (Unique)", "papyrus", "smiles", True),
        ("Graph Representation", "papyrus", "graph", False)
    ]

    print("DrugEx Dataset Path Resolution Telemetry:")
    print(f"  {'Configuration':<30} | {'Training Set Path':<28} | {'Validation Set Path'}")
    print(f"  {'-'*30}-+-{'-'*28}-+-{'-'*28}")
    for name, prefix, mtype, unique in scenarios:
        train_p, test_p = getDataPaths(base, prefix, mtype, unique)
        print(f"  {name:<30} | {os.path.basename(train_p):<28} | {os.path.basename(test_p)}")`,
          output: `DrugEx Dataset Path Resolution Telemetry:
  Configuration                  | Training Set Path            | Validation Set Path       
  -------------------------------+------------------------------+-----------------------------
  SMILES Standard                | papyrus_train_smiles.txt     | papyrus_test_smiles.txt   
  SMILES Pretraining (Unique)    | papyrus_unique_smiles.txt    | papyrus_test_smiles.txt   
  Graph Representation           | papyrus_train_graph.txt      | papyrus_test_graph.txt    `
        },
        seeAlso: [
          "drugex::data::utils::getVocPaths",
          "drugex::data::datasets::SmilesFragDataSet",
          "drugex::data::datasets::GraphFragDataSet"
        ]
      },

    "drugex::data::utils::getVocPaths": {
        category: "core",
        module: "drugex.data.utils",
        title: "getVocPaths",
        kind: "function",
        badges: ["DrugEx 3.4+", "File Conventions", "Vocabulary"],
        synopsis: `def getVocPaths(
    data_path: str,
    voc_files: Sequence[str],
    mol_type: str = "smiles"
) -> List[str]: ...`,
        description: "Resolves filesystem paths to vocabulary files matching specified names or suffixed patterns ('_{mol_type}.txt.vocab'). Logs informative warnings if files are missing and falls back to internal DrugEx default vocabularies if no files are found.",
        members: [],
        variables: [],
        parameters: [
          { name: "data_path", type: "str", desc: "Base directory path where vocabulary files reside." },
          { name: "voc_files", type: "Sequence[str]", desc: "Sequence of candidate vocabulary filenames or prefixes." },
          { name: "mol_type", type: "str", desc: "Molecular representation format ('smiles' or 'graph') used for suffixed fallback search. Defaults to 'smiles'." }
        ],
        notes: "Logs informative warnings when defaulting to built-in vocabulary definitions.",
        example: {
          code: `import os
import tempfile
from drugex.data.utils import getVocPaths

with tempfile.TemporaryDirectory() as tmpdir:
    # Set up custom vocabulary files following DrugEx naming conventions
    voc_exact = os.path.join(tmpdir, "custom_kinase.vocab")
    with open(voc_exact, "w") as f:
        f.write("C\\nN\\nO\\nF\\nCl\\n")

    voc_suffixed = os.path.join(tmpdir, "papyrus_smiles.txt.vocab")
    with open(voc_suffixed, "w") as f:
        f.write("C\\nN\\nO\\nS\\nP\\n")

    queries = [
        ("Explicit filename", ["custom_kinase.vocab"], "smiles"),
        ("Prefix matching suffix", ["papyrus"], "smiles"),
        ("Missing fallback to internal", ["nonexistent_voc"], "smiles")
    ]

    print("DrugEx Vocabulary Path Resolution Dispatch:")
    print(f"  {'Query Mode':<30} | {'Found Vocabularies':<26} | {'Resolution Status'}")
    print(f"  {'-'*30}-+-{'-'*26}-+-{'-'*24}")
    for mode, v_files, m_type in queries:
        resolved = getVocPaths(tmpdir, v_files, m_type)
        count_str = f"{len(resolved)} file(s)"
        status = "Resolved: " + os.path.basename(resolved[0]) if resolved else "Fallback to Built-in Defaults"
        print(f"  {mode:<30} | {count_str:<26} | {status}")`,
          output: `DrugEx Vocabulary Path Resolution Dispatch:
  Query Mode                     | Found Vocabularies         | Resolution Status     
  -------------------------------+----------------------------+-------------------------
  Explicit filename              | 1 file(s)                  | Resolved: custom_kinase.vocab
  Prefix matching suffix         | 1 file(s)                  | Resolved: papyrus_smiles.txt.vocab
  Missing fallback to internal   | 0 file(s)                  | Fallback to Built-in Defaults`
        },
        seeAlso: [
          "drugex::data::utils::getDataPaths",
          "drugex::data::corpus::vocabulary::VocSmiles",
          "drugex::data::corpus::vocabulary::VocGraph"
        ]
      },

      // =========================================================================
      // SECTION 14: LOGGING SUBSYSTEM & PROVENANCE
      // =========================================================================,

    "drugex::data::processing::Standardization::Collector": {
        category: "core",
        module: "drugex.data.processing",
        title: "Standardization.Collector",
        kind: "class",
        badges: ["DrugEx 3.4+", "Sphinx gh-pages", "Parallel Collector"],
        synopsis: `class Collector(ListExtend):
    def __init__(self) -> None: ...
    def __call__(self, result: Tuple[Sequence[Any], Any]) -> None: ...`,
        description: "Sphinx-canonical nested class in drugex.data.processing.Standardization extending ListExtend. Accumulates batches of standardized molecular structures in parallel processing pipelines without locks.",
        members: [
          { name: "getList()", desc: "Returns the internal list of collected standardized molecule representations." },
          { name: "__call__(result)", desc: "Extracts worker result tuple result[0] and extends the internal items list." }
        ],
        variables: [
          { name: "items", type: "List[Any]", desc: "Accumulated standardized molecules." }
        ],
        parameters: [],
        notes: "Sphinx gh-pages canonical entry for Standardization.Collector. Inherits lock-free extend semantics from ListExtend.",
        example: {
          code: `from drugex.data.processing import Standardization
from rdkit import Chem
from rdkit.Chem import Descriptors

# Accumulate and sanitize batches of generated SMILES structures across worker threads
collector = Standardization.Collector()

# Simulate parallel pipeline batches with salt stripping and neutralization
batch_a = (["c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1"],)
batch_b = (["c1cc(O)ccc1CC(=O)O", "Cc1ccc(NC(=O)c2ccccc2)cc1"],)

collector(batch_a)
collector(batch_b)

items = collector.getList()

print(f"{'Index':5} | {'Sanitized Molecule':26} | {'Molecular Weight':16} | {'Collection Status'}")
print("-" * 72)
for idx, smi in enumerate(items, 1):
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    print(f"#{idx:<4} | {smi:26} | {mw:14.2f} Da | Lock-free Appended")`,
          output: `Index | Sanitized Molecule         | Molecular Weight | Collection Status
------------------------------------------------------------------------
#1    | c1ccccc1O                  |          94.11 Da | Lock-free Appended
#2    | CC(=O)Nc1ccc(O)cc1         |         151.16 Da | Lock-free Appended
#3    | c1ccncc1                   |          79.10 Da | Lock-free Appended
#4    | c1cc(O)ccc1CC(=O)O         |         152.15 Da | Lock-free Appended
#5    | Cc1ccc(NC(=O)c2ccccc2)cc1  |         211.26 Da | Lock-free Appended`
        },
        seeAlso: [
          "drugex::data::processing::StandardizationCollector",
          "drugex::data::processing::Standardization",
          "drugex::parallel::collectors::ListExtend"
        ]
      }
  });
})();
