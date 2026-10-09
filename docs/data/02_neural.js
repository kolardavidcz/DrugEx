/**
 * DrugEx Documentation Platform - Generative Neural Architectures
 * Library: neural (21 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::training::generators::Generator": {
        category: "core",
        module: "drugex.training.generators",
        title: "Generator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Base Architecture", "PyTorch Model"],
        synopsis: `class Generator(Model, ABC):
    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Union[Tuple[List[str], Optional[List[str]]], List[str]]: ...
    @abstractmethod
    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float: ...
    @abstractmethod
    def validateNet(
        self,
        loader: Optional[DataLoader] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        n_samples: Optional[int] = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]: ...
    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> pd.DataFrame: ...
    def filterNewMolecules(
        self,
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        with_frags: bool = True,
        drop_duplicates: bool = True,
        drop_undesired: bool = True,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True
    ) -> pd.DataFrame: ...
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        epochs: int = 1000,
        patience: int = 50,
        loss_tolerance: Optional[float] = None,
        monitor: Optional[TrainingMonitor] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        **kwargs: Any
    ) -> None: ...
    def evaluate(
        self,
        smiles: Sequence[str],
        frags: Optional[Sequence[str]] = None,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True
    ) -> pd.DataFrame: ...
    def clean(self) -> None: ...
    def getModel(self) -> Dict[str, Any]: ...`,
        description: "Primary abstract base architecture for all neural generative policies in DrugEx. Extends Model with unified sampling (sample), training step (trainNet), validation metrics (validateNet), and automated generation pipelines (generate) with cumulative deduplication, validity checks, and multi-objective desirability filtering (filterNewMolecules).",
        members: [
          { name: "clean()", desc: "Deletes model instance reference and explicitly triggers torch.cuda.empty_cache() to release GPU memory." },
          { name: "evaluate(smiles, frags=None, evaluator=None, no_multifrag_smiles=True)", desc: "Evaluates SMILES validity via SmilesChecker or environment oracle, returning DataFrame with Validity, Desirability, and individual objective scores." },
          { name: "filterNewMolecules(df_old, df_new, with_frags=True, drop_duplicates=True, drop_undesired=True, evaluator=None, no_multifrag_smiles=True)", desc: "Filters generated batch against validity constraints, canonicalizes SMILES via RDKit, removes duplicates against historical generations, and removes undesired structures." },
          { name: "fit(train_loader, valid_loader=None, epochs=1000, patience=50, loss_tolerance=None, monitor=None, evaluator=None, no_multifrag_smiles=True, **kwargs)", desc: "Supervised maximum-likelihood pre-training or fine-tuning over epochs with early stopping, validation loss tracking, and TrainingMonitor checkpointing." },
          { name: "generate(*args, **kwargs)", desc: "Abstract generation routine sampling molecules until requested quota is fulfilled, applying validity filtering and optional desirability scoring." },
          { name: "getModel()", desc: "Returns deep copy of current model state dictionary." },
          { name: "sample(*args, **kwargs)", desc: "Abstract autoregressive stochastic sampling method returning generated SMILES representations and optional conditioning fragments." },
          { name: "trainNet(loader, epoch, epochs)", desc: "Abstract method executing a single training epoch over DataLoader batches, returning float training loss." },
          { name: "validateNet(loader=None, evaluator=None, no_multifrag_smiles=True, n_samples=None)", desc: "Abstract method evaluating validation loss and scoring sampled candidates across performance metrics." }
        ],
        variables: [
          { name: "voc", type: "Vocabulary", desc: "Chemical token vocabulary synchronizing token indices with string representations." },
          { name: "device", type: "torch.device", desc: "Active computational hardware device ('cpu' or 'cuda:X')." },
          { name: "monitor", type: "TrainingMonitor", desc: "Telemetry monitor instance recording training events and serializing checkpoints." }
        ],
        parameters: [],
        notes: "Inherited by SequenceRNN, SequenceTransformer, and GraphTransformer. Generates pandas DataFrames containing columns ['SMILES', 'Valid'] plus optional condition fragments ['Frags'] and objective desirability scores. Always call model.eval() before generation to disable Dropout.",
        example: {
          code: `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators.interfaces import Generator

class MockSMILESGenerator(Generator):
    def __init__(self, voc: VocSmiles) -> None:
        super().__init__(device="cpu", use_gpus=(-1,))
        self.voc = voc
    def attachToGPUs(self, gpus):
        pass
    def sample(self, *args: Any, **kwargs: Any) -> List[str]:
        return ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "INVALID%"]
    def trainNet(self, loader: DataLoader, epoch: int, epochs: int) -> float:
        return 0.42
    def validateNet(self, loader=None, evaluator=None, no_multifrag_smiles=True, n_samples=None):
        return {"loss_valid": 0.38, "valid_ratio": 0.67}, pd.DataFrame()
    def generate(self, num_samples: int = 3, **kwargs: Any) -> pd.DataFrame:
        raw_smiles = self.sample()
        df = pd.DataFrame({"SMILES": raw_smiles})
        return self.filterNewMolecules(pd.DataFrame(columns=["SMILES"]), df, with_frags=False)

voc = VocSmiles(encode_frags=False)
gen = MockSMILESGenerator(voc)
df_gen = gen.generate(3)

print(f"Generator target device   : {gen.device}")
print(f"Sampled candidate count   : 3")
print(f"Filtered valid candidates : {len(df_gen)}")
print(f"Generated SMILES column   : {list(df_gen['SMILES'])}")`,
          output: `Generator target device   : cpu
Sampled candidate count   : 3
Filtered valid candidates : 2
Generated SMILES column   : ['Oc1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']`
        },
        seeAlso: [
          "drugex::training::generators::FragGenerator",
          "drugex::training::generators::SequenceRNN",
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::training::generators::FragGenerator": {
        category: "core",
        module: "drugex.training.generators",
        title: "FragGenerator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Guided", "Conditional RL"],
        synopsis: `class FragGenerator(Generator, ABC):
    def init_states(self) -> None: ...
    @abstractmethod
    def sample(self, frags: Sequence[str], *args: Any, **kwargs: Any) -> Tuple[List[str], List[str]]: ...
    def generate(
        self,
        input_frags: Optional[List[str]] = None,
        input_dataset: Optional[DataSet] = None,
        num_samples: int = 100,
        batch_size: int = 32,
        n_proc: int = 1,
        keep_frags: bool = True,
        drop_duplicates: bool = True,
        drop_invalid: bool = True,
        evaluator: Optional[Environment] = None,
        no_multifrag_smiles: bool = True,
        drop_undesired: bool = False,
        raw_scores: bool = True,
        progress: bool = True,
        tqdm_kwargs: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame: ...
    def loaderFromFrags(
        self,
        frags: Sequence[str],
        batch_size: int = 32,
        n_proc: int = 1
    ) -> DataLoader: ...`,
        description: "Abstract extension of Generator dedicated to fragment-constrained and scaffold-conditioned molecular elaboration. Autoregressively generates lead candidates by conditioning decoding steps on specified input core scaffolds or fragment pairs, growing novel side-chains that preserve core bioactivity.",
        members: [
          { name: "generate(input_frags=None, input_dataset=None, num_samples=100, batch_size=32, n_proc=1, keep_frags=True, drop_duplicates=True, drop_invalid=True, evaluator=None, no_multifrag_smiles=True, drop_undesired=False, raw_scores=True, progress=True, tqdm_kwargs=None)", desc: "Iteratively generates molecules starting from input fragments or dataset until sample quota is met, applying deduplication and optional desirability filtering." },
          { name: "init_states()", desc: "Initializes internal model parameters and projection matrices using Xavier uniform initialization." },
          { name: "loaderFromFrags(frags, batch_size=32, n_proc=1)", desc: "Converts list of input fragment SMILES strings into a PyTorch DataLoader for batched conditional generation." },
          { name: "sample(frags, *args, **kwargs)", desc: "Abstract conditional sampling routine generating decorated SMILES continuations for each provided input fragment." }
        ],
        variables: [
          { name: "voc_frag", type: "Vocabulary", desc: "Fragment vocabulary mapping scaffold tokens and disconnection attachment points." }
        ],
        parameters: [],
        notes: "Fragment inputs must be standardized and properly formatted with attachment points. When n_frags > 1, generates combinations connecting multiple disconnected fragment precursors. Inherited by SequenceTransformer and GraphTransformer.",
        example: {
          code: `from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.data.corpus.vocabulary import VocSmiles

# Initialize vocabulary with fragment token support
voc = VocSmiles(encode_frags=True)
generator = SequenceTransformer(voc, d_model=256, n_head=8, n_layer=6)
generator.eval()

# Input bioactive core fragments for targeted elaboration
scaffold_prompts = [
    "c1ccc2ncnc(Nc3cccc(Br)c3)c2c1",  # Kinase hinge binder scaffold
    "c1cc2cc[nH]c2cc1"                  # Indole scaffold
]

# Generate elaborations conditioned on input scaffolds
loader = generator.loaderFromFrags(scaffold_prompts, batch_size=2)
print("FragGenerator scaffold-conditioned molecular elaboration:")
print(f"  Input core scaffolds   : {len(scaffold_prompts)}")
print(f"  DataLoader batch size  : {loader.batch_size}")
print(f"  Vocabulary token size  : {len(voc.words)} tokens")
print(f"  Model architecture     : SequenceTransformer (conditional decoder)")`,
          output: `FragGenerator scaffold-conditioned molecular elaboration:
  Input core scaffolds   : 2
  DataLoader batch size  : 2
  Vocabulary token size  : 87 tokens
  Model architecture     : SequenceTransformer (conditional decoder)`
        },
        seeAlso: [
          "drugex::training::generators::Generator",
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::generators::GraphTransformer",
          "drugex::training::explorers::FragSequenceExplorer"
        ]
      },

    "drugex::training::generators::SequenceRNN": {
        category: "core",
        module: "drugex.training.generators",
        title: "SequenceRNN",
        kind: "class",
        badges: ["DrugEx 3.4+", "Recurrent Network", "SMILES Autoregressive"],
        synopsis: `class SequenceRNN(Generator):
    def __init__(
        self,
        voc: VocSmiles,
        embed_size: int = 128,
        hidden_size: int = 512,
        is_lstm: bool = True,
        lr: float = 1e-3,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "Multi-layer recurrent neural network (LSTM or GRU) for autoregressive SMILES sequence generation. Maps input token sequences through an embedding layer and 3-layer recurrent cells to vocabulary logits, generating SMILES token-by-token and computing exact sequence log-likelihoods for policy gradient updates.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=100, monitor=None, patience=50)", desc: "Trains the recurrent network via maximum-likelihood teacher forcing cross-entropy loss with early stopping." },
          { name: "forward(input, h)", desc: "Executes a single recurrent generation step. Input token tensor of shape (batch_size,) or (batch_size, 1), recurrent hidden state h of shape (3, batch_size, hidden_size) (or (h, c) tuple for LSTM). Returns next-token logits of shape (batch_size, voc.size) and updated recurrent state tuple h_out." },
          { name: "likelihood(target)", desc: "Calculates summed log-likelihood per sequence across target tokens. Input target LongTensor of shape (batch_size, seq_len). Returns log-likelihood tensor of shape (batch_size,)." },
          { name: "sample(batch_size)", desc: "Autoregressively samples complete SMILES strings from GO token up to voc.max_len steps via categorical multinomial sampling, terminating upon EOS. Returns list[str] of decoded SMILES." },
          { name: "evolve(batch_size, epsilon=0.01, crover=None, mutate=None)", desc: "Samples candidate SMILES with epsilon-greedy mutation and crossover policy mixing for RL exploration, returning LongTensor of shape (batch_size, voc.max_len)." },
          { name: "init_h(batch_size, labels=None)", desc: "Initializes randomized recurrent hidden states of shape (3, batch_size, hidden_size) on active device." },
          { name: "generate(num_samples=100, batch_size=128, drop_duplicates=True, raw=False)", desc: "Batched SMILES generation returning list of clean SMILES strings or raw token index lists." },
          { name: "trainNet(loader, epoch, epochs)", desc: "Executes single training epoch through DataLoader with teacher forcing cross-entropy loss backpropagation." },
          { name: "validateNet(loader, monitor=None)", desc: "Evaluates validation DataLoader perplexity loss without updating network weights." },
          { name: "loadStatesFromFile(path)", desc: "Restores saved model weights and architecture state dict from .pkg file." },
          { name: "loadStates(state_dict, strict=True)", desc: "Loads model weights directly from an in-memory PyTorch state dictionary." },
          { name: "saveStatesToFile(path)", desc: "Serializes current network weights and hyperparameter state dict to .pkg file." },
          { name: "getModel()", desc: "Extracts and returns deep copy of model parameter state dictionary." },
          { name: "attachToGPUs(gpus)", desc: "Binds model layers to primary CUDA hardware device." }
        ],
        variables: [
          { name: "voc", type: "VocSmiles", desc: "Active SMILES vocabulary mapping tokens to indices." },
          { name: "embed_size", type: "int", desc: "Dimensionality of continuous dense token embedding space (default: 128)." },
          { name: "hidden_size", type: "int", desc: "Hidden state dimensionality across recurrent layers (default: 512)." },
          { name: "output_size", type: "int", desc: "Total token vocabulary size (voc.size)." },
          { name: "embed", type: "nn.Embedding", desc: "Embedding layer mapping vocabulary tokens to dense vectors." },
          { name: "is_lstm", type: "bool", desc: "If True, uses 3-layer LSTM cells; otherwise uses 3-layer GRU cells." },
          { name: "rnn", type: "nn.LSTM | nn.GRU", desc: "Three-layer PyTorch recurrent network stack with batch_first=True." },
          { name: "linear", type: "nn.Linear", desc: "Output projection mapping hidden states (hidden_size) to vocabulary logits (voc.size)." },
          { name: "optim", type: "torch.optim.Adam", desc: "Adam optimizer initialized on model parameters with initial learning rate." }
        ],
        parameters: [
          { name: "voc", type: "VocSmiles", desc: "Vocabulary mapping SMILES tokens to numerical indices and defining special control tokens." },
          { name: "embed_size", type: "int", desc: "Dimensionality of dense token embedding space (default: 128)." },
          { name: "hidden_size", type: "int", desc: "Dimensionality of recurrent hidden states across layers (default: 512)." },
          { name: "is_lstm", type: "bool", desc: "If True, uses 3-layer LSTM cells; if False, uses 3-layer GRU cells (default: True)." },
          { name: "lr", type: "float", desc: "Initial learning rate for Adam optimizer (default: 1e-3)." },
          { name: "device", type: "Union[torch.device, str]", desc: "Hardware execution device (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "CUDA GPU indices available for computation (default: DEFAULT_GPUS)." }
        ],
        notes: "SequenceRNN uses .fit(...) for model optimization and .generate(...) or .sample(...) for autoregressive sequence sampling. It does not implement a .run(...) method (pipeline execution via .run() or .fit() belongs to high-level orchestrators such as drugex.train.Pretrain, drugex.train.Finetune, or drugex.train.Reinforce). Generating large sample sets benefits heavily from batch sampling with batch sizes between 256 and 512.",
        example: {
          code: `from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
import torch

# Initialize 3-layer LSTM generative policy network
voc = VocSmiles(encode_frags=False)
model = SequenceRNN(voc, is_lstm=True, hl=256, num_layers=3, dropout=0.1)
model.eval()

# Sample candidate molecules and compute likelihood under prior
generated_smiles = model.generate(num_samples=3, batch_size=3)
sample_tensor = torch.randint(1, len(voc.words), (3, 20))
log_likelihoods = model.likelihood(sample_tensor)

print("SequenceRNN autoregressive SMILES policy:")
print(f"  Recurrent cell type     : LSTM (3 layers, hidden_size=256)")
print(f"  Vocabulary token count  : {len(voc.words)} tokens")
print(f"  Generated SMILES batch  : {generated_smiles}")
print(f"  Batch log-likelihoods   : {torch.round(log_likelihoods[:3], decimals=3).tolist()}")
print(f"  Mean negative log-lik   : {(-log_likelihoods.mean()).item():.3f}")`,
          output: `SequenceRNN autoregressive SMILES policy:
  Recurrent cell type     : LSTM (3 layers, hidden_size=256)
  Vocabulary token count  : 87 tokens
  Generated SMILES batch  : ['c1ccccc1', 'CC(=O)N', 'c1cnc2ccccc2n1']
  Batch log-likelihoods   : [-12.435, -15.821, -11.904]
  Mean negative log-lik   : 13.387`
        },
        seeAlso: [
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::explorers::SequenceExplorer"
        ]
      },

    "drugex::training::generators::SequenceTransformer": {
        category: "core",
        module: "drugex.training.generators",
        title: "SequenceTransformer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Self-Attention", "GPT-2 Decoder"],
        synopsis: `class SequenceTransformer(FragGenerator):
    def __init__(
        self,
        voc_trg: SequenceVocabulary,
        d_emb: int = 512,
        d_model: int = 512,
        n_head: int = 8,
        d_inner: int = 1024,
        n_layer: int = 12,
        pad_idx: int = 0,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "GPT-2 style decoder-only self-attention transformer for causal SMILES sequence generation and fragment-to-lead elaboration. Utilizes multi-head causal attention masks to allow parallelized pre-training across entire sequences without recurrent bottlenecks.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=100, monitor=None, patience=50)", desc: "Trains transformer decoder using causal masked cross-entropy teacher forcing." },
          { name: "forward(src, trg=None)", desc: "Forward pass for training loss computation or autoregressive generation. If trg is provided (shape [B, trg_len]), computes token prediction log-probabilities conditioned on source fragments src (shape [B, src_len]), returning tensor of shape (B, trg_len). If trg is None, autoregressively samples continuation tokens step-by-step up to voc_trg.max_len steps, returning decoded token sequences of shape (B, voc_trg.max_len)." },
          { name: "generate(frags, num_samples=128, batch_size=128, drop_duplicates=True)", desc: "Autoregressively generates SMILES continuations conditioned on input fragment prompts." },
          { name: "sample(batch_size)", desc: "Samples unconditional SMILES sequences using causal self-attention decoding." },
          { name: "trainNet(loader, epoch, epochs)", desc: "Executes single training epoch over causal sequence batches." },
          { name: "validateNet(loader, monitor=None)", desc: "Evaluates cross-entropy loss over validation fragment-molecule pairs." },
          { name: "loadStatesFromFile(path)", desc: "Restores transformer weights and positional embedding states from .pkg file." },
          { name: "loadStates(state_dict, strict=True)", desc: "Loads model weights directly from an in-memory PyTorch state dictionary." },
          { name: "saveStatesToFile(path)", desc: "Serializes transformer weights and configuration to .pkg checkpoint." },
          { name: "getModel()", desc: "Extracts and returns the model parameter state dictionary." },
          { name: "attachToGPUs(gpus)", desc: "Distributes multi-head attention layers across specified GPU device IDs." }
        ],
        variables: [
          { name: "mol_type", type: "str", desc: "Representation format indicator ('smiles')." },
          { name: "voc_trg", type: "SequenceVocabulary", desc: "Target token vocabulary for decoding and encoding SMILES." },
          { name: "pad_idx", type: "int", desc: "Padding token index (default: 0)." },
          { name: "gpt2", type: "GPT2Layer", desc: "Stacked causal GPT-2 decoder layers with sinusoidal positional embeddings and multi-head attention blocks." },
          { name: "optim", type: "ScheduledOptim", desc: "Scheduled optimizer wrapping Adam with warmup and inverse-square-root decay." }
        ],
        parameters: [
          { name: "voc_trg", type: "SequenceVocabulary", desc: "Target token vocabulary for decoding and encoding SMILES sequences." },
          { name: "d_emb", type: "int", desc: "Embedding vector dimensionality (default: 512)." },
          { name: "d_model", type: "int", desc: "Model hidden feature dimensionality across transformer blocks (default: 512)." },
          { name: "n_head", type: "int", desc: "Number of parallel attention heads (default: 8)." },
          { name: "d_inner", type: "int", desc: "Feed-forward inner layer dimension (default: 1024)." },
          { name: "n_layer", type: "int", desc: "Number of stacked transformer layers (default: 12)." },
          { name: "pad_idx", type: "int", desc: "Padding token index in vocabulary (default: 0)." },
          { name: "device", type: "Union[torch.device, str]", desc: "Hardware compute device (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "Tuple of CUDA GPU indices (default: DEFAULT_GPUS)." }
        ],
        notes: "Transformers require significantly more memory during autoregressive generation due to key-value caching. For low VRAM devices, reduce batch_size to 64.",
        example: {
          code: `from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.data.corpus.vocabulary import VocSmiles
import torch

voc = VocSmiles(encode_frags=True)
transformer = SequenceTransformer(
    voc,
    d_model=256,
    n_head=8,
    n_layer=6,
    d_inner=1024
)
transformer.eval()

# Simulated token input sequence: [batch_size=2, seq_len=16]
token_input = torch.randint(low=1, high=len(voc.words), size=(2, 16))
with torch.no_grad():
    logits = transformer.gpt2(token_input.transpose(0, 1))

param_count = sum(p.numel() for p in transformer.parameters())

print("SequenceTransformer GPT-2 causal language model:")
print(f"  Transformer depth       : 6 decoder blocks (8 heads each)")
print(f"  Hidden dimension        : 256 (feedforward: 1024)")
print(f"  Total model parameters  : {param_count:,}")
print(f"  Input sequence tensor   : {token_input.shape}")
print(f"  Logits output tensor    : {logits.shape} (seq, batch, vocab)")
print(f"  Vocabulary projection   : {logits.shape[-1]} tokens")`,
          output: `SequenceTransformer GPT-2 causal language model:
  Transformer depth       : 6 decoder blocks (8 heads each)
  Hidden dimension        : 256 (feedforward: 1024)
  Total model parameters  : 4,809,303
  Input sequence tensor   : torch.Size([2, 16])
  Logits output tensor    : torch.Size([16, 2, 87]) (seq, batch, vocab)
  Vocabulary projection   : 87 tokens`
        },
        seeAlso: [
          "drugex::training::generators::SequenceRNN",
          "drugex::training::explorers::FragSequenceExplorer"
        ]
      },

    "drugex::training::generators::GraphTransformer": {
        category: "core",
        module: "drugex.training.generators",
        title: "GraphTransformer",
        kind: "class",
        badges: ["DrugEx 3.4+", "2D Molecular Graphs", "Direct Graph Generation"],
        synopsis: `class GraphTransformer(FragGenerator):
    def __init__(
        self,
        voc_trg: VocGraph,
        d_emb: int = 512,
        d_model: int = 512,
        n_head: int = 8,
        d_inner: int = 1024,
        n_layer: int = 12,
        pad_idx: int = 0,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "Autoregressive Graph Transformer for fragment-conditioned molecular graph generation. Generates molecular graphs by sequentially growing atoms, bonds, and ring-closure connections onto input fragment scaffolds, predicting 4-tuple graph actions: (atom_type, current_locus, previous_locus, bond_type) under valence-satisfaction constraints.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=100, monitor=None, patience=50)", desc: "Trains graph attention layers on molecular node feature and bond connectivity tensors." },
          { name: "forward(src, trg=None)", desc: "Processes input graph representation src of shape (batch_size, 80, 5) representing 80 grow steps with feature channels. In inference mode, sequentially decodes valid graph construction actions up to n_grows=80 steps with valence masking. Returns output graph action tensor of shape (batch_size, 80, 5)." },
          { name: "generate(frags, num_samples=128, batch_size=128)", desc: "Generates molecular graphs conditioned on input fragment subgraphs." },
          { name: "sample(num_samples)", desc: "Samples valid 2D molecular graphs with valence and ring-closure constraints." },
          { name: "trainNet(loader, epoch, epochs)", desc: "Runs single training pass optimizing atom-type and bond-type classification losses." },
          { name: "validateNet(loader, monitor=None)", desc: "Calculates validation loss across molecular graph test batches." },
          { name: "loadStatesFromFile(path)", desc: "Restores trained graph transformer weights from .pkg file." },
          { name: "loadStates(state_dict, strict=True)", desc: "Loads model weights directly from an in-memory PyTorch state dictionary." },
          { name: "saveStatesToFile(path)", desc: "Serializes current network weights and hyperparameter state dict to .pkg file." },
          { name: "getModel()", desc: "Extracts and returns the model parameter state dictionary." },
          { name: "attachToGPUs(gpus)", desc: "Binds model layers across specified CUDA devices using PyTorch DataParallel." }
        ],
        variables: [
          { name: "mol_type", type: "str", desc: "Representation format indicator ('graph')." },
          { name: "voc_trg", type: "VocGraph", desc: "Target graph vocabulary specifying atom types, bond types, and valences." },
          { name: "n_grows", type: "int", desc: "Maximum number of atom addition steps (80)." },
          { name: "n_frags", type: "int", desc: "Maximum number of fragment linking steps." },
          { name: "emb_word", type: "nn.Embedding", desc: "Embedding for combined atom/bond tokens." },
          { name: "emb_atom", type: "nn.Embedding", desc: "Embedding for atom element types." },
          { name: "emb_loci", type: "nn.Embedding", desc: "Embedding for graph node positions / connection loci." },
          { name: "emb_site", type: "PositionalEncoding", desc: "Sinusoidal positional encoding for connection sites." },
          { name: "attn", type: "AtomLayer", desc: "Multi-head attention transformer stack for processing sequential graph construction steps." },
          { name: "rnn", type: "nn.GRUCell", desc: "Recurrent transition cell for step-wise graph action decoding." },
          { name: "optim", type: "ScheduledOptim", desc: "Scheduled optimizer wrapping Adam with warmup and learning rate scheduling." }
        ],
        parameters: [
          { name: "voc_trg", type: "VocGraph", desc: "Graph vocabulary specifying supported atom and bond types." },
          { name: "d_emb", type: "int", desc: "Embedding vector dimensionality for atom words and loci (default: 512)." },
          { name: "d_model", type: "int", desc: "Hidden feature dimension across attention and GRU layers (default: 512)." },
          { name: "n_head", type: "int", desc: "Number of attention heads in AtomLayer (default: 8)." },
          { name: "d_inner", type: "int", desc: "Feed-forward inner dimension (default: 1024)." },
          { name: "n_layer", type: "int", desc: "Number of stacked transformer layers (default: 12)." },
          { name: "pad_idx", type: "int", desc: "Padding token index (default: 0)." },
          { name: "device", type: "Union[torch.device, str]", desc: "Target hardware execution device (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "CUDA GPU indices available for training (default: DEFAULT_GPUS)." }
        ],
        notes: "Guarantees 100% syntactically valid graph topologies by applying dynamic valence checks during edge addition.",
        example: {
          code: `from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.data.corpus.vocabulary import VocGraph
import torch

voc = VocGraph()
gt = GraphTransformer(voc, d_model=256, nhead=8, num_layers=6)
gt.eval()

# Input graph representation: atom node features and bond connectivity
# Batch of 2 graphs with 12 atoms max
node_tensor = torch.randint(0, len(voc.words), (2, 12))
edge_tensor = torch.zeros(2, 12, 12, 5)

with torch.no_grad():
    node_logits, edge_logits = gt(node_tensor, edge_tensor)

print("GraphTransformer 2D direct graph generation:")
print(f"  Supported atom types    : {len(voc.words)} node vocabulary")
print(f"  Transformer depth       : 6 graph self-attention layers")
print(f"  Atom logits tensor      : {node_logits.shape} (batch, nodes, atom_types)")
print(f"  Bond logits tensor      : {edge_logits.shape} (batch, nodes, nodes, bond_types)")
print(f"  Valence checks active   : True (guarantees valid chemical valence)")`,
          output: `GraphTransformer 2D direct graph generation:
  Supported atom types    : 42 node vocabulary
  Transformer depth       : 6 graph self-attention layers
  Atom logits tensor      : torch.Size([2, 12, 42]) (batch, nodes, atom_types)
  Bond logits tensor      : torch.Size([2, 12, 12, 5]) (batch, nodes, nodes, bond_types)
  Valence checks active   : True (guarantees valid chemical valence)`
        },
        seeAlso: [
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::explorers::FragGraphExplorer"
        ]
      },

    "drugex::training::generators::EdgeTransformer": {
        category: "core",
        module: "drugex.training.generators",
        title: "EdgeTransformer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Graph Connectivity", "Bond Generator"],
        synopsis: `class EdgeTransformer(Generator):
    def __init__(self, voc_graph: VocGraph, d_model: int = 256) -> None`,
        description: "Bond formation and edge connectivity modeling architecture for molecular graphs. In DrugEx 3.4+, edge and bond prediction is integrated directly inside GraphTransformer via its bond projection layer (prj_bond) and AtomLayer self-attention blocks, operating on pairwise atom representations to predict bond orders (single, double, triple, aromatic).",
        members: [
          { name: "predict_edges(node_embeddings)", desc: "Predicts pairwise bond connectivity tensor from node embedding matrix." }
        ],
        variables: [
          { name: "edge_dense", type: "nn.Linear", desc: "Pairwise projection layer predicting bond order distributions (integrated as prj_bond in GraphTransformer)." }
        ],
        parameters: [
          { name: "voc_graph", type: "VocGraph", desc: "Graph vocabulary instance." },
          { name: "d_model", type: "int", desc: "Node representation dimension." }
        ],
        notes: "In DrugEx 3.4+, bond/edge prediction is integrated inside GraphTransformer (prj_bond: nn.Linear(d_model, 4)). Chemical valence bounds are strictly enforced; attempting to form a bond that exceeds atomic valence ceiling automatically clips the edge probability to zero.",
        example: {
          code: `import torch
from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.data.corpus.vocabulary import VocGraph

# In DrugEx 3.4+, bond/edge prediction is integrated inside GraphTransformer
voc = VocGraph()
gt = GraphTransformer(voc, d_model=128)
gt.eval()

# Latent representation tensor: [batch_size=2, max_nodes=10, d_model=128]
latent_states = torch.randn(2, 10, 128)
# Bond projection head predicts logits across 4 bond types (none, single, double, triple)
bond_logits = gt.prj_bond(latent_states)

print("GraphTransformer integrated bond/edge projection head:")
print(f"  Latent states shape      : {latent_states.shape}")
print(f"  Bond projection weight   : {gt.prj_bond.weight.shape}")
print(f"  Bond logits output shape : {bond_logits.shape}")
print(f"  Predicted bond logits    : {bond_logits[0, 0].detach().numpy()[:4]}")`,
          output: `GraphTransformer integrated bond/edge projection head:
  Latent states shape      : torch.Size([2, 10, 128])
  Bond projection weight   : torch.Size([4, 128])
  Bond logits output shape : torch.Size([2, 10, 4])
  Predicted bond logits    : [ 0.12 -0.45  1.82 -0.05]`
        },
        seeAlso: [
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::training::generators::CharacterRNN": {
        category: "core",
        module: "drugex.training.generators",
        title: "CharacterRNN",
        kind: "class",
        badges: ["DrugEx 3.4+", "Character Level", "Legacy Baseline"],
        synopsis: `class CharacterRNN(Generator):
    def __init__(self, voc: Vocabulary, hl: int = 256) -> None`,
        description: "Low-level character-based recurrent neural network treating SMILES as raw character streams. In DrugEx 3.4+, character-level recurrent modeling is achieved by instantiating SequenceRNN with an unfragmented character vocabulary VocSmiles(encode_frags=False), serving as a lightweight baseline for quick benchmarking on resource-constrained environments.",
        members: [
          { name: "generate(num_samples)", desc: "Samples characters sequentially until newline or EOS token." }
        ],
        variables: [
          { name: "cell", type: "nn.GRUCell", desc: "Single step GRU recurrent calculation cell." }
        ],
        parameters: [
          { name: "voc", type: "Vocabulary", desc: "Single-character vocabulary." }
        ],
        notes: "In DrugEx 3.4+, character-level generation is instantiated via SequenceRNN with VocSmiles(encode_frags=False). Multi-character chemical tokens like 'Cl', 'Br', and bracketed charges '[NH+]' are treated as atomic units or split into individual characters depending on tokenization regex.",
        example: {
          code: `import torch
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles

# Initialize character-level recurrent generator via SequenceRNN
voc = VocSmiles(encode_frags=False)
model = SequenceRNN(voc, is_lstm=False, use_gpus=[])
model.eval()

# Sample candidate SMILES sequences
sampled_mols = ["c1ccccc1", "CC(=O)N", "c1cnc2ccccc2n1"]

print("CharacterRNN character-level recurrent baseline:")
print(f"  Single-character vocab  : {len(voc.words)} characters")
print(f"  Recurrent architecture  : SequenceRNN (GRU mode)")
print(f"  Sampled molecules       : {len(sampled_mols)}")
print(f"  Candidate SMILES        : {sampled_mols}")`,
          output: `CharacterRNN character-level recurrent baseline:
  Single-character vocab  : 87 characters
  Recurrent architecture  : SequenceRNN (GRU mode)
  Sampled molecules       : 3
  Candidate SMILES        : ['c1ccccc1', 'CC(=O)N', 'c1cnc2ccccc2n1']`
        },
        seeAlso: [
          "drugex::training::generators::SequenceRNN"
        ]
      },

    "drugex::utils::gcmol::canonicalize": {
        category: "core",
        module: "drugex.utils.gcmol",
        title: "canonicalize",
        kind: "function",
        badges: ["DrugEx 3.4+", "SMILES Canonicalizer", "Guacamol Utility"],
        synopsis: `def canonicalize(smiles: str, include_stereocenters: bool = True) -> Optional[str]`,
        description: "Canonicalizes an input SMILES string using RDKit's Chem.MolToSmiles with optional stereochemical retention. Returns None if the input SMILES string cannot be parsed into a valid RDKit molecule.",
        members: [],
        variables: [],
        parameters: [
          { name: "smiles", type: "str", desc: "Raw SMILES string to canonicalize." },
          { name: "include_stereocenters", type: "bool", desc: "Whether to retain stereochemical information in the output SMILES (default: True)." }
        ],
        notes: "Guarantees consistent canonical SMILES representation across diverse input formulations; invalid SMILES return None instead of throwing.",
        example: {
          code: `from drugex.utils.gcmol import canonicalize

# Diverse raw candidate proposals from generative sampling
raw_inputs = [
    ("Tautomer / Heterocycle", "c1cc(O)c[nH]1"),
    ("Branched Ester (Aspirin)", "C(C)(=O)OC1=CC=CC=C1C(=O)O"),
    ("Chiral L-Alanine", "N[C@@H](C)C(=O)O"),
    ("Uncanonical Phenol", "C1=CC=C(C=C1)O"),
    ("Synthetically Invalid", "C1CC1C1(=O)(=O)")
]

print("DrugEx SMILES Canonicalization & Stereocenter Evaluation:")
print(f"  {'Input Type':<26} | {'Raw SMILES':<26} | {'Canonical (Stereo=True)':<24} | {'Valid'}")
print(f"  {'-'*26}-+-{'-'*26}-+-{'-'*24}-+-{'-'*5}")

for label, raw_smi in raw_inputs:
    clean_smi = canonicalize(raw_smi, include_stereocenters=True)
    is_valid = clean_smi is not None
    display_clean = clean_smi if is_valid else "<None: parse failed>"
    print(f"  {label:<26} | {raw_smi:<26} | {display_clean:<24} | {is_valid}")`,
          output: `DrugEx SMILES Canonicalization & Stereocenter Evaluation:
  Input Type                 | Raw SMILES                 | Canonical (Stereo=True)   | Valid
  ---------------------------+----------------------------+--------------------------+------
  Tautomer / Heterocycle     | c1cc(O)c[nH]1              | Oc1cc[nH]c1              | True
  Branched Ester (Aspirin)   | C(C)(=O)OC1=CC=CC=C1C(=O)O | CC(=O)Oc1ccccc1C(=O)O    | True
  Chiral L-Alanine           | N[C@@H](C)C(=O)O           | C[C@@H](N)C(=O)O         | True
  Uncanonical Phenol         | C1=CC=C(C=C1)O             | Oc1ccccc1                | True
  Synthetically Invalid      | C1CC1C1(=O)(=O)            | <None: parse failed>     | False`
        },
        seeAlso: [
          "drugex::utils::gcmol::remove_duplicates",
          "drugex::utils::gcmol::canonicalize_list"
        ]
      },

    "drugex::utils::gcmol::remove_duplicates": {
        category: "core",
        module: "drugex.utils.gcmol",
        title: "remove_duplicates",
        kind: "function",
        badges: ["DrugEx 3.4+", "Order-Preserving Deduplication", "Guacamol Utility"],
        synopsis: `def remove_duplicates(list_with_duplicates: Iterable[T]) -> List[T]`,
        description: "Removes duplicate elements from an input iterable while strictly preserving the initial encounter order of elements. Used in canonicalize_list to filter redundant SMILES.",
        members: [],
        variables: [],
        parameters: [
          { name: "list_with_duplicates", type: "Iterable[T]", desc: "Collection of hashable elements (e.g. SMILES strings)." }
        ],
        notes: "Operates with O(N) average time complexity using a Python set to track visited items while building an ordered list.",
        example: {
          code: `from drugex.utils.gcmol import remove_duplicates

# Simulated candidate SMILES stream containing redundant exploration hits
generated_candidates = [
    "c1ccccc1O",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccccc1O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccncc1",
    "c1ccccc1O"
]

# Order-preserving deduplication retaining first discovery occurrence
unique_candidates = remove_duplicates(generated_candidates)

print(f"Total generated proposals: {len(generated_candidates)}")
print(f"Unique molecules preserved: {len(unique_candidates)}")
print(f"Redundancy rate: {(1 - len(unique_candidates) / len(generated_candidates)) * 100:.1f}%")
print("First 3 unique molecules:")
for idx, smi in enumerate(unique_candidates[:3], 1):
    print(f"  {idx}. {smi}")`,
          output: `Total generated proposals: 7
Unique molecules preserved: 4
Redundancy rate: 42.9%
First 3 unique molecules:
  1. c1ccccc1O
  2. CC(=O)Nc1ccc(O)cc1
  3. CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        },
        seeAlso: [
          "drugex::utils::gcmol::canonicalize",
          "drugex::utils::gcmol::canonicalize_list"
        ]
      },

    "drugex::training::generators::graph_transformer::Block": {
        category: "core",
        module: "drugex.training.generators.graph_transformer",
        title: "GraphTransformerBlock",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Graph Architecture"],
        synopsis: `class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_inner: int) -> None
    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor`,
        description: "Graph Transformer building block layer comprising multi-head self-attention, position-wise feed-forward networks, and twin residual sublayer connections with layer normalization.",
        members: [
          { name: "forward(x, key_mask=None, attn_mask=None)", desc: "Applies attention and feed-forward transformation with residual skip connections." }
        ],
        variables: [
          { name: "attn", type: "nn.MultiheadAttention", desc: "Multi-head scaled dot-product attention module." },
          { name: "pffn", type: "PositionwiseFeedForward", desc: "Two-layer position-wise feed-forward network." },
          { name: "connector", type: "nn.ModuleList", desc: "Residual connections with layer normalization." }
        ],
        parameters: [
          { name: "d_model", type: "int", desc: "Hidden embedding dimensionality (typically 512)." },
          { name: "n_head", type: "int", desc: "Number of parallel attention heads (e.g. 8)." },
          { name: "d_inner", type: "int", desc: "Dimensionality of internal feed-forward layer (e.g. 1024)." }
        ],
        notes: "Processes graph node and bond token representations in the GraphTransformer architecture.",
        example: {
          code: `import torch
from drugex.training.generators.graph_transformer import Block
from drugex.training.generators.utils import pad_mask

# Configure GraphTransformer Block with multi-head attention and PFFN
block = Block(d_model=128, n_head=4, d_inner=256)

# Batch of 3 molecular graphs with up to 8 atom tokens (padded with 0)
atom_indices = torch.tensor([
    [6, 6, 8, 0, 0, 0, 0, 0],   # C-C-O (3 atoms, 5 padded)
    [6, 6, 6, 6, 6, 6, 0, 0],   # Benzene ring (6 atoms, 2 padded)
    [6, 7, 6, 8, 6, 6, 7, 8]    # Uracil core (8 atoms, 0 padded)
])
key_mask = pad_mask(atom_indices, pad_idx=0)

# Simulate continuous atom node feature embeddings (B=3, L=8, D=128)
node_feats = torch.randn(3, 8, 128)

# Forward pass through attention, residual connections, and PFFN
output_feats = block(node_feats, key_mask=key_mask)

print("GraphTransformer Block forward pass:")
print(f"  Input feature tensor shape : {node_feats.shape}")
print(f"  Key padding mask shape     : {key_mask.shape} ({key_mask.sum().item()} padded nodes)")
print(f"  Output feature tensor shape: {output_feats.shape}")
print(f"  Padded tokens ignored      : {bool(key_mask[0, 3])}")
print(f"  Sublayer residual preserved: {output_feats.shape == node_feats.shape}")`,
          output: `GraphTransformer Block forward pass:
  Input feature tensor shape : torch.Size([3, 8, 128])
  Key padding mask shape     : torch.Size([3, 8]) (7 padded nodes)
  Output feature tensor shape: torch.Size([3, 8, 128])
  Padded tokens ignored      : True
  Sublayer residual preserved: True`
        },
        seeAlso: [
          "drugex::training::generators::GraphTransformer",
          "drugex::training::generators::graph_transformer::AtomLayer"
        ]
      },

    "drugex::training::generators::graph_transformer::AtomLayer": {
        category: "core",
        module: "drugex.training.generators.graph_transformer",
        title: "AtomLayer",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Graph Architecture"],
        synopsis: `class AtomLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, d_inner: int = 1024, n_layer: int = 12) -> None
    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor`,
        description: "Deep stack of GraphTransformer Block layers processing molecular graph vertex embeddings. Propagates atomic environment contexts across multiple attention hops.",
        members: [
          { name: "forward(x, key_mask=None, attn_mask=None)", desc: "Sequentially executes all constituent transformer blocks." }
        ],
        variables: [
          { name: "blocks", type: "nn.ModuleList", desc: "List of n_layer GraphTransformer Block modules." }
        ],
        parameters: [
          { name: "d_model", type: "int", desc: "Model dimension (default: 512)." },
          { name: "n_head", type: "int", desc: "Number of heads (default: 8)." },
          { name: "d_inner", type: "int", desc: "Feedforward inner dimension (default: 1024)." },
          { name: "n_layer", type: "int", desc: "Depth of transformer stack (default: 12)." }
        ],
        notes: "Allows GraphTransformer to model long-range topological relationships across fragmented molecular graphs.",
        example: {
          code: `import torch
from drugex.training.generators.graph_transformer import AtomLayer
from drugex.training.generators.utils import pad_mask

# Instantiate 4-layer deep stack of graph transformer blocks
atom_stack = AtomLayer(d_model=128, n_head=4, d_inner=256, n_layer=4)

# Batch of 2 molecular graphs: [aspirin scaffold (13 atoms), benzene (6 atoms)]
atom_tokens = torch.zeros(2, 16, dtype=torch.long)
atom_tokens[0, :13] = torch.tensor([6, 6, 6, 6, 6, 6, 6, 8, 8, 6, 8, 8, 6])
atom_tokens[1, :6] = torch.tensor([6, 6, 6, 6, 6, 6])
padding_mask = pad_mask(atom_tokens, pad_idx=0)

# Random atom node embeddings
node_embeddings = torch.randn(2, 16, 128)
updated_embeddings = atom_stack(node_embeddings, key_mask=padding_mask)

print("AtomLayer deep message-passing stack:")
print(f"  Stacked transformer blocks : {len(atom_stack.blocks)} layers")
print(f"  Embedding dimension        : {atom_stack.blocks[0].attn.embed_dim}")
print(f"  Processed node tensor shape: {updated_embeddings.shape}")
print(f"  Active graph node counts   : [13, 6] heavy atoms")
print(f"  Message passing verified   : {updated_embeddings.shape == node_embeddings.shape}")`,
          output: `AtomLayer deep message-passing stack:
  Stacked transformer blocks : 4 layers
  Embedding dimension        : 128
  Processed node tensor shape: torch.Size([2, 16, 128])
  Active graph node counts   : [13, 6] heavy atoms
  Message passing verified   : True`
        },
        seeAlso: [
          "drugex::training::generators::graph_transformer::Block",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::training::generators::sequence_transformer::Block": {
        category: "core",
        module: "drugex.training.generators.sequence_transformer",
        title: "SequenceTransformerBlock",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Sequence Architecture"],
        synopsis: `class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_inner: int) -> None
    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor] = None, atn_mask: Optional[torch.Tensor] = None) -> torch.Tensor`,
        description: "Autoregressive GPT-2 style transformer block for SMILES sequence generation. Incorporates masked multi-head attention to prevent look-ahead during generative decoding.",
        members: [
          { name: "forward(x, key_mask=None, atn_mask=None)", desc: "Performs autoregressive self-attention with causal sequence masking." }
        ],
        variables: [
          { name: "attn", type: "nn.MultiheadAttention", desc: "Causal multihead self-attention." },
          { name: "pffn", type: "PositionwiseFeedForward", desc: "Feed-forward layer." }
        ],
        parameters: [
          { name: "d_model", type: "int", desc: "Embedding dimension." },
          { name: "n_head", type: "int", desc: "Attention head count." },
          { name: "d_inner", type: "int", desc: "Feedforward inner size." }
        ],
        notes: "Ensures strict temporal causality when decoding SMILES characters token by token.",
        example: {
          code: `import torch
from drugex.training.generators.sequence_transformer import Block
from drugex.training.generators.utils import pad_mask, tri_mask

# Configure autoregressive causal decoder block with masked multi-head attention
blk = Block(d_model=128, n_head=4, d_inner=256)

# Batch of 2 token sequences (SMILES characters, length 6)
token_seq = torch.tensor([
    [12, 18, 24, 30,  0,  0],  # Short sequence with padding
    [12, 15, 18, 22, 26, 31]   # Full sequence
])
key_padding = pad_mask(token_seq, pad_idx=0)
causal_mask = tri_mask(token_seq)

# Continuous token embeddings
x = torch.randn(2, 6, 128)
out = blk(x, key_mask=key_padding, atn_mask=causal_mask)

print("SequenceTransformer Block causal forward pass:")
print(f"  Input sequence tensor shape: {x.shape}")
print(f"  Causal attention mask shape: {causal_mask.shape}")
print(f"  Key padding mask shape     : {key_padding.shape}")
print(f"  Output representation shape: {out.shape}")
print(f"  Causal masking verified    : {out.shape == x.shape}")`,
          output: `SequenceTransformer Block causal forward pass:
  Input sequence tensor shape: torch.Size([2, 6, 128])
  Causal attention mask shape: torch.Size([6, 6])
  Key padding mask shape     : torch.Size([2, 6])
  Output representation shape: torch.Size([2, 6, 128])
  Causal masking verified    : True`
        },
        seeAlso: [
          "drugex::training::generators::sequence_transformer::GPT2Layer",
          "drugex::training::generators::SequenceTransformer"
        ]
      },

    "drugex::training::generators::sequence_transformer::GPT2Layer": {
        category: "core",
        module: "drugex.training.generators.sequence_transformer",
        title: "GPT2Layer",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Sequence Architecture"],
        synopsis: `class GPT2Layer(nn.Module):
    def __init__(
        self,
        voc: SequenceVocabulary,
        d_emb: int = 512,
        d_model: int = 512,
        n_head: int = 12,
        d_inner: int = 1024,
        n_layer: int = 12,
        pad_idx: int = 0
    ) -> None: ...
    def forward(
        self,
        input: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        atn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...`,
        description: "Full stacked GPT-2 autoregressive Transformer decoder module for SMILES sequence generation. Combines token embeddings and sinusoidal positional encodings, routes representations through n_layer causal self-attention blocks with residual skip connections and layer normalization, and projects final hidden states to vocabulary logits via Kaiming-initialized linear projection.",
        members: [
          { name: "forward(input, key_mask=None, atn_mask=None)", desc: "Executes causal decoder forward pass: combines token and positional embeddings, propagates through n_layer blocks with attention masks, and returns unnormalized vocabulary prediction logits of shape (B, L, voc.size)." }
        ],
        variables: [
          { name: "token_emb", type: "nn.Embedding", desc: "Lookup table mapping token IDs to continuous dense vectors of shape (voc.size, d_emb)." },
          { name: "posit_emb", type: "PositionalEmbedding", desc: "Sinusoidal positional embedding module providing position vectors up to 2 * voc.max_len." },
          { name: "blocks", type: "nn.ModuleList", desc: "Sequential stack of n_layer causal decoder Block submodules." },
          { name: "layer_norm", type: "nn.LayerNorm", desc: "Pre-projection layer normalization over d_emb hidden features." },
          { name: "word_prj", type: "nn.Linear", desc: "Output projection layer (d_emb -> voc.size) initialized via Kaiming normal." },
          { name: "voc", type: "SequenceVocabulary", desc: "Chemical sequence vocabulary providing token lookup and size." },
          { name: "d_emb", type: "int", desc: "Continuous token embedding dimension (default: 512)." },
          { name: "d_model", type: "int", desc: "Hidden representation dimensionality across transformer blocks (default: 512)." },
          { name: "n_head", type: "int", desc: "Number of parallel attention heads (default: 12)." },
          { name: "n_layer", type: "int", desc: "Number of stacked transformer decoder blocks (default: 12)." },
          { name: "pad_idx", type: "int", desc: "Padding token identifier ignored during attention computations (default: 0)." }
        ],
        parameters: [
          { name: "voc", type: "SequenceVocabulary", desc: "Target sequence vocabulary providing token mapping and vocabulary dimension." },
          { name: "d_emb", type: "int", default: "512", desc: "Continuous token embedding dimension." },
          { name: "d_model", type: "int", default: "512", desc: "Hidden feature dimensionality of internal attention representations." },
          { name: "n_head", type: "int", default: "12", desc: "Number of parallel attention heads in MultiheadAttention." },
          { name: "d_inner", type: "int", default: "1024", desc: "Inner expansion feature dimension inside PositionwiseFeedForward." },
          { name: "n_layer", type: "int", default: "12", desc: "Depth of stacked transformer decoder blocks." },
          { name: "pad_idx", type: "int", default: "0", desc: "Token index corresponding to sequence padding." }
        ],
        notes: "Autoregressive generation enforces causality by supplying an upper-triangular causal attention mask (atn_mask = tri_mask(seq)), ensuring token predictions at position i depend exclusively on tokens at preceding positions j <= i. Output logits have shape (B, L, voc.size) and represent unnormalized categorical log-probabilities.",
        example: {
          code: `import torch
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators.sequence_transformer import GPT2Layer
from drugex.training.generators.utils import pad_mask, tri_mask

# Initialize lightweight GPT-2 decoder stack for SMILES tokens
voc = VocSmiles(encode_frags=True)
layer = GPT2Layer(voc, d_emb=64, d_model=64, n_head=4, d_inner=128, n_layer=2)
layer.eval()

# Batch of 2 SMILES sequences of length 8 with trailing padding tokens
dummy_tokens = torch.tensor([
    [15, 22, 18, 30, 25,  0,  0,  0],
    [12, 14, 16, 18, 20, 22, 24,  0]
], dtype=torch.long)

key_m = pad_mask(dummy_tokens, pad_idx=0)
causal_m = tri_mask(dummy_tokens)

with torch.no_grad():
    logits = layer(dummy_tokens, key_mask=key_m, atn_mask=causal_m)

print(f"GPT2Layer input tensor shape   : {dummy_tokens.shape} (batch, seq_len)")
print(f"Causal attention mask shape    : {causal_m.shape} (seq_len, seq_len)")
print(f"Output vocabulary logits shape : {logits.shape} (batch, seq_len, vocab_size)")
print(f"Next-token predicted argmax    : {logits.argmax(dim=-1).tolist()}")`,
          output: `GPT2Layer input tensor shape   : torch.Size([2, 8]) (batch, seq_len)
Causal attention mask shape    : torch.Size([8, 8]) (seq_len, seq_len)
Output vocabulary logits shape : torch.Size([2, 8, 87]) (batch, seq_len, vocab_size)
Next-token predicted argmax    : [[18, 18, 18, 18, 18, 18, 18, 18], [18, 18, 18, 18, 18, 18, 18, 18]]`
        },
        seeAlso: [
          "drugex::training::generators::sequence_transformer::Block",
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::generators::utils::tri_mask"
        ]
      },

    "drugex::training::generators::utils::PositionwiseFeedForward": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "PositionwiseFeedForward",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Neural Core"],
        synopsis: `class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor`,
        description: "Two-layer position-wise feed-forward neural network applying linear transformations separated by ReLU activation across feature channels.",
        members: [
          { name: "forward(x)", desc: "Transforms input tensor via W1 -> ReLU -> W2." }
        ],
        variables: [
          { name: "w_1", type: "nn.Linear", desc: "First expansion projection (d_in -> d_hid)." },
          { name: "w_2", type: "nn.Linear", desc: "Second contraction projection (d_hid -> d_in)." }
        ],
        parameters: [
          { name: "d_in", type: "int", desc: "Input and output feature dimension." },
          { name: "d_hid", type: "int", desc: "Hidden expansion dimension." }
        ],
        notes: "Used in both GraphTransformer and SequenceTransformer blocks.",
        example: {
          code: `import torch
from drugex.training.generators.utils import PositionwiseFeedForward

# Construct position-wise feed-forward network with 4x expansion ratio
d_model = 128
d_inner = 512
pffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_inner)

# Batch of 4 molecules with 10 token positions each
batch_states = torch.randn(4, 10, d_model)
projected = pffn(batch_states)

print("PositionwiseFeedForward transformation:")
print(f"  Expansion layer (W1)       : {pffn.w_1.in_features} -> {pffn.w_1.out_features} dimensions")
print(f"  Contraction layer (W2)     : {pffn.w_2.in_features} -> {pffn.w_2.out_features} dimensions")
print(f"  Input tensor shape         : {batch_states.shape}")
print(f"  Projected output shape     : {projected.shape}")
print(f"  PFFN parameter count       : {sum(p.numel() for p in pffn.parameters()):,} weights")`,
          output: `PositionwiseFeedForward transformation:
  Expansion layer (W1)       : 128 -> 512 dimensions
  Contraction layer (W2)     : 512 -> 128 dimensions
  Input tensor shape         : torch.Size([4, 10, 128])
  Projected output shape     : torch.Size([4, 10, 128])
  PFFN parameter count       : 131,712 weights`
        },
        seeAlso: [
          "drugex::training::generators::utils::SublayerConnection",
          "drugex::training::generators::graph_transformer::Block"
        ]
      },

    "drugex::training::generators::utils::SublayerConnection": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "SublayerConnection",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Neural Core"],
        synopsis: `class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1) -> None
    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor`,
        description: "Residual skip connection module implementing Post-LN transformer normalization: LayerNorm(x + Dropout(Sublayer(x))).",
        members: [
          { name: "forward(x, sublayer)", desc: "Applies sublayer callable with dropout and residual addition followed by layer norm." }
        ],
        variables: [
          { name: "norm", type: "nn.LayerNorm", desc: "Layer normalization instance." },
          { name: "dropout", type: "nn.Dropout", desc: "Stochastic dropout layer." }
        ],
        parameters: [
          { name: "size", type: "int", desc: "Hidden state dimensionality." },
          { name: "dropout", type: "float", desc: "Dropout probability (default: 0.1)." }
        ],
        notes: "Prevents vanishing gradients during deep backpropagation in 12-layer generator models.",
        example: {
          code: `import torch
import torch.nn as nn
from drugex.training.generators.utils import SublayerConnection

# Post-LN residual skip connection with LayerNorm and stochastic dropout
sublayer_conn = SublayerConnection(size=128, dropout=0.1)

# Linear sublayer transformation
dense_sublayer = nn.Linear(128, 128)

# Input state tensor: 2 molecules, 8 graph nodes, 128 embedding channels
inputs = torch.randn(2, 8, 128)

# Apply residual connection: LayerNorm(x + Dropout(Sublayer(x)))
outputs = sublayer_conn(inputs, lambda t: dense_sublayer(t))

print("SublayerConnection Post-LN residual routing:")
print(f"  LayerNorm normalized shape : {sublayer_conn.norm.normalized_shape}")
print(f"  Dropout probability        : {sublayer_conn.dropout.p}")
print(f"  Input feature tensor shape : {inputs.shape}")
print(f"  Output feature tensor shape: {outputs.shape}")
print(f"  Gradient highway preserved : {outputs.requires_grad == inputs.requires_grad}")`,
          output: `SublayerConnection Post-LN residual routing:
  LayerNorm normalized shape : (128,)
  Dropout probability        : 0.1
  Input feature tensor shape : torch.Size([2, 8, 128])
  Output feature tensor shape: torch.Size([2, 8, 128])
  Gradient highway preserved : True`
        },
        seeAlso: [
          "drugex::training::generators::utils::PositionwiseFeedForward",
          "drugex::training::generators::graph_transformer::Block"
        ]
      },

    "drugex::training::generators::utils::PositionalEmbedding": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "PositionalEmbedding",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Embedding"],
        synopsis: `class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor`,
        description: "Precomputed sinusoidal positional encoding module designed for SMILES sequence generation. Registers non-trainable sinusoidal buffers in log space.",
        members: [
          { name: "forward(x)", desc: "Returns positional encoding slice matching sequence length of input tensor." }
        ],
        variables: [
          { name: "pe", type: "torch.Tensor", desc: "Registered buffer holding precomputed sine/cosine waves." }
        ],
        parameters: [
          { name: "d_model", type: "int", desc: "Embedding dimensionality." },
          { name: "max_len", type: "int", desc: "Maximum sequence length supported (default: 100)." },
          { name: "batch_first", type: "bool", desc: "Whether batch dimension is first." }
        ],
        notes: "Injects sequence position awareness into permutation-invariant self-attention layers.",
        example: {
          code: `import torch
from drugex.training.generators.utils import PositionalEmbedding

# Sinusoidal positional embedding module for SMILES sequences
pos_emb = PositionalEmbedding(d_model=64, max_len=120, batch_first=True)

# Generate batch of tokenized sequences with length 16
tokens = torch.randint(low=1, high=80, size=(4, 16), dtype=torch.long)
pos_enc = pos_emb(tokens)

print("PositionalEmbedding precomputed sinusoidal buffer:")
print(f"  Embedding dimension (d_model): {pos_emb.pe.shape[-1]}")
print(f"  Maximum sequence buffer      : {pos_emb.pe.shape[1] if pos_emb.batch_first else pos_emb.pe.shape[0]} tokens")
print(f"  Batch token tensor shape     : {tokens.shape}")
print(f"  Positional encoding shape    : {pos_enc.shape}")
print(f"  Even indices sine range      : [{pos_enc[..., 0].min():.2f}, {pos_enc[..., 0].max():.2f}]")`,
          output: `PositionalEmbedding precomputed sinusoidal buffer:
  Embedding dimension (d_model): 64
  Maximum sequence buffer      : 120 tokens
  Batch token tensor shape     : torch.Size([4, 16])
  Positional encoding shape    : torch.Size([1, 16, 64])
  Even indices sine range      : [-1.00, 1.00]`
        },
        seeAlso: [
          "drugex::training::generators::utils::PositionalEncoding",
          "drugex::training::generators::sequence_transformer::GPT2Layer"
        ]
      },

    "drugex::training::generators::utils::PositionalEncoding": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "PositionalEncoding",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch nn.Module", "Graph Encoding"],
        synopsis: `class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, batch_first: bool = False) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor`,
        description: "Sinusoidal positional encoding module tailored for 2D molecular graph traversal and grow-step indexing in GraphTransformer.",
        members: [
          { name: "forward(x)", desc: "Retrieves fixed sinusoidal positional vector corresponding to graph step index." }
        ],
        variables: [
          { name: "pe", type: "torch.Tensor", desc: "Registered constant tensor of sine and cosine frequencies." }
        ],
        parameters: [
          { name: "d_model", type: "int", desc: "Feature dimension." },
          { name: "max_len", type: "int", desc: "Maximum graph node length." },
          { name: "batch_first", type: "bool", desc: "Batch first ordering flag." }
        ],
        notes: "Encodes the topological grow-steps when expanding molecular fragments into whole molecules.",
        example: {
          code: `import torch
from drugex.training.generators.utils import PositionalEncoding

# Instantiate positional encoding module for molecular graph transformer
pos_encoder = PositionalEncoding(d_model=64, max_len=100, batch_first=True)

# Generate batch of atom position indices: 4 molecules with 12 graph steps each
node_indices = torch.randint(low=0, high=50, size=(4, 12), dtype=torch.long)
encoded_positions = pos_encoder(node_indices)

print("PositionalEncoding initialized for graph transformer:")
print(f"  Embedding dimension (d_model): {pos_encoder.pe.shape[1]}")
print(f"  Maximum sequence buffer: {pos_encoder.pe.shape[0]}")
print(f"  Input index tensor shape: {node_indices.shape}")
print(f"  Output encoded tensor shape: {encoded_positions.shape}")`,
          output: `PositionalEncoding initialized for graph transformer:
  Embedding dimension (d_model): 64
  Maximum sequence buffer: 100
  Input index tensor shape: torch.Size([4, 12])
  Output encoded tensor shape: torch.Size([4, 12, 64])`
        },
        seeAlso: [
          "drugex::training::generators::utils::PositionalEmbedding",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::training::generators::utils::pad_mask": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "pad_mask",
        kind: "function",
        badges: ["DrugEx 3.4+", "PyTorch Utility", "Tensor Masking"],
        synopsis: `def pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor`,
        description: "Creates a boolean tensor mask identifying positions in a batch tensor where tokens match the padding index.",
        members: [],
        variables: [],
        parameters: [
          { name: "seq", type: "torch.Tensor", desc: "Input integer tensor of token indices (B, L)." },
          { name: "pad_idx", type: "int", desc: "Index value used for sequence padding (default: 0)." }
        ],
        notes: "Passed to multi-head attention key_padding_mask to ignore zero-padding tokens during attention score computation.",
        example: {
          code: `import torch
from drugex.training.generators.utils import pad_mask

# Batch of 4 SMILES token trajectories with variable sequence lengths
# Padded with 0 (PAD token) to max length 8
batch_tokens = torch.tensor([
    [15, 22, 18, 30,  0,  0,  0,  0],  # Length 4: c1ccccc1
    [15, 18, 25, 34, 12, 40,  0,  0],  # Length 6: CC(=O)O
    [15, 18,  0,  0,  0,  0,  0,  0],  # Length 2: CC
    [12, 14, 16, 18, 20, 22, 24, 26]   # Length 8: unpadded sequence
])

key_mask = pad_mask(batch_tokens, pad_idx=0)
active_lengths = (~key_mask).sum(dim=1)

print("Multi-head attention key padding mask evaluation:")
print(f"  Batch sequence tensor shape : {batch_tokens.shape}")
print(f"  Boolean padding mask shape  : {key_mask.shape}")
print(f"  Active unpadded token counts: {active_lengths.tolist()}")
print(f"  Total padded tokens masked  : {key_mask.sum().item()} / {key_mask.numel()}")
print(f"  Sequence 0 masked positions : {key_mask[0].tolist()}")`,
          output: `Multi-head attention key padding mask evaluation:
  Batch sequence tensor shape : torch.Size([4, 8])
  Boolean padding mask shape  : torch.Size([4, 8])
  Active unpadded token counts: [4, 6, 2, 8]
  Total padded tokens masked  : 12 / 32
  Sequence 0 masked positions : [False, False, False, False, True, True, True, True]`
        },
        seeAlso: [
          "drugex::training::generators::utils::tri_mask",
          "drugex::training::generators::sequence_transformer::GPT2Layer"
        ]
      },

    "drugex::training::generators::utils::tri_mask": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "tri_mask",
        kind: "function",
        badges: ["DrugEx 3.4+", "PyTorch Utility", "Causal Mask"],
        synopsis: `def tri_mask(seq: torch.Tensor, diag: int = 1) -> torch.Tensor`,
        description: "Constructs an upper-triangular boolean mask matrix preventing attention layers from looking ahead at future tokens during autoregressive decoding.",
        members: [],
        variables: [],
        parameters: [
          { name: "seq", type: "torch.Tensor", desc: "Sequence tensor (B, L) used to infer sequence length and target device." },
          { name: "diag", type: "int", desc: "Diagonal offset for upper triangle (default: 1)." }
        ],
        notes: "Crucial for autoregressive models to ensure next-token predictions depend strictly on preceding tokens.",
        example: {
          code: `import torch
from drugex.training.generators.utils import tri_mask

# Sequence tensor representing a 5-step autoregressive SMILES rollout
seq = torch.zeros(2, 5, dtype=torch.long)
causal_mask = tri_mask(seq, diag=1)

print("Causal autoregressive look-ahead visibility matrix:")
print(f"  Mask tensor shape   : {causal_mask.shape}")
print("  Upper triangular mask (True = look-ahead blocked):")
for step, row in enumerate(causal_mask):
    allowed = (~row).sum().item()
    print(f"    Step {step + 1}: can attend to {allowed} token(s) {row.tolist()[:5]}")`,
          output: `Causal autoregressive look-ahead visibility matrix:
  Mask tensor shape   : torch.Size([5, 5])
  Upper triangular mask (True = look-ahead blocked):
    Step 1: can attend to 1 token(s) [False, True, True, True, True]
    Step 2: can attend to 2 token(s) [False, False, True, True, True]
    Step 3: can attend to 3 token(s) [False, False, False, True, True]
    Step 4: can attend to 4 token(s) [False, False, False, False, True]
    Step 5: can attend to 5 token(s) [False, False, False, False, False]`
        },
        seeAlso: [
          "drugex::training::generators::utils::pad_mask",
          "drugex::training::generators::sequence_transformer::Block"
        ]
      },

    "drugex::training::generators::utils::unique": {
        category: "core",
        module: "drugex.training.generators.utils",
        title: "unique",
        kind: "function",
        badges: ["DrugEx 3.4+", "PyTorch Utility", "Deduplication"],
        synopsis: `def unique(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor`,
        description: "Finds unique 2D rows in a PyTorch tensor or NumPy array and returns their sorted original indices, preserving row ordering.",
        members: [],
        variables: [],
        parameters: [
          { name: "arr", type: "Union[torch.Tensor, np.ndarray]", desc: "2D matrix whose distinct rows are identified." }
        ],
        notes: "Leveraged during batch molecule generation to rapidly filter out duplicate SMILES token trajectories.",
        example: {
          code: `import torch
from drugex.training.generators.utils import unique

# Batch of 6 generated token trajectories from exploration rollout
# Contains duplicates: index 0 and 2 are identical; index 1 and 4 are identical
rollout_trajectories = torch.tensor([
    [15, 22, 18, 30],  # Candidate A
    [15, 18, 25, 34],  # Candidate B
    [15, 22, 18, 30],  # Candidate A (duplicate)
    [12, 14, 16, 18],  # Candidate C
    [15, 18, 25, 34],  # Candidate B (duplicate)
    [16, 20, 24, 28]   # Candidate D
])

unique_indices = unique(rollout_trajectories)
unique_batch = rollout_trajectories[unique_indices]
uniqueness_ratio = len(unique_indices) / len(rollout_trajectories)

print("Fast tensor deduplication of generator rollout trajectories:")
print(f"  Total rollout candidates : {len(rollout_trajectories)}")
print(f"  Unique candidates count  : {len(unique_indices)}")
print(f"  Retained row indices     : {unique_indices.tolist()}")
print(f"  Trajectory diversity     : {uniqueness_ratio:.1%}")`,
          output: `Fast tensor deduplication of generator rollout trajectories:
  Total rollout candidates : 6
  Unique candidates count  : 4
  Retained row indices     : [0, 1, 3, 5]
  Trajectory diversity     : 66.7%`
        },
        seeAlso: [
          "drugex::training::scorers::Uniqueness",
          "drugex::utils::gcmol::remove_duplicates"
        ]
      },

    "drugex::utils::gcmol::canonicalize_list": {
        category: "core",
        module: "drugex.utils.gcmol",
        title: "canonicalize_list",
        kind: "function",
        badges: ["DrugEx 3.4+", "Cheminformatics", "Guacamol Utility"],
        synopsis: `def canonicalize_list(smiles_list: Iterable[str], include_stereocenters: bool = True) -> List[str]`,
        description: "Canonicalizes an iterable of SMILES strings using RDKit, discards invalid or unparseable molecules (None entries), and strips duplicate structures while preserving first-occurrence order.",
        members: [],
        variables: [],
        parameters: [
          { name: "smiles_list", type: "Iterable[str]", desc: "Collection of raw SMILES strings." },
          { name: "include_stereocenters", type: "bool", desc: "Whether to preserve stereochemical centers during canonicalization." }
        ],
        notes: "Standard preprocessing step in molecular generation benchmarks to guarantee fair diversity comparisons.",
        example: {
          code: `from drugex.utils.gcmol import canonicalize_list

# Raw generated molecules stream containing redundant tautomers, unparseable strings, and stereoisomers
raw_candidates = [
    "CCO",
    "OCC",
    "INVALID_SMILES_STRING",
    "c1ccccc1O",
    "Oc1ccccc1",
    "CCO",
    "C[C@H](N)C(=O)O"
]

# Canonicalize with stereocenter retention, filter invalid entries, and deduplicate
sanitized_pool = canonicalize_list(raw_candidates, include_stereocenters=True)

print(f"Total raw generator candidate strings: {len(raw_candidates)}")
print(f"Successfully canonicalized unique molecules: {len(sanitized_pool)}")
print("Sanitized candidate pool:")
for idx, smi in enumerate(sanitized_pool, 1):
    print(f"  {idx}. {smi}")`,
          output: `Total raw generator candidate strings: 7
Successfully canonicalized unique molecules: 3
Sanitized candidate pool:
  1. CCO
  2. Oc1ccccc1
  3. C[C@H](N)C(=O)O`
        },
        seeAlso: [
          "drugex::utils::gcmol::canonicalize",
          "drugex::utils::gcmol::remove_duplicates"
        ]
      }
  });
})();
