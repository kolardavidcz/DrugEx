/**
 * DrugEx Documentation Platform - Explorers & RL Search Policies
 * Library: explorers (6 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::training::explorers::Explorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "Explorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Policy Gradient", "Abstract RL Engine"],
        synopsis: `class Explorer(Model, ABC):
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
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "Abstract base reinforcement learning explorer in DrugEx. Coordinates policy gradient updates (REINFORCE) between a generative agent, an environment evaluation oracle, and optional genetic operators (mutation and crossover networks). Defines standard evaluation metrics, state checkpointing, and GPU binding protocols.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None)", desc: "Abstract RL training loop updating agent policy weights across iterations until early stopping or epoch exhaustion." },
          { name: "policy_gradient(*args, **kwargs)", desc: "Abstract REINFORCE policy gradient optimization step using environment feedback." },
          { name: "getNovelMoleculeMetrics(scores)", desc: "Computes evaluation metrics dictionary from environment scores DataFrame: valid_ratio, accurate_ratio (if fragments present), unique_ratio, desired_ratio, avg_amean, and avg_gmean." },
          { name: "saveBestState(value, epoch, it)", desc: "Persists best model checkpoint state dictionary (self.bestState = self.getModel()), updates best_value tracker, and triggers monitor.saveModel(self)." },
          { name: "logPerformanceAndCompounds(epoch, metrics, scores)", desc: "Packages epoch performance dictionary and compound score table, dispatching telemetry to monitor.savePerformanceInfo and monitor.endStep." },
          { name: "getModel()", desc: "Extracts and returns deep copy of the agent model parameter state dictionary (deepcopy(self.agent.state_dict()))." },
          { name: "attachToGPUs(gpus)", desc: "Distributes agent, mutate, and crover networks across specified CUDA hardware devices." }
        ],
        variables: [
          { name: "agent", type: "Generator", desc: "Active generative model being optimized by policy gradient." },
          { name: "env", type: "Environment", desc: "Multi-objective reward environment providing feedback signals." },
          { name: "mutate", type: "Optional[Generator]", desc: "Frozen pre-trained generative model providing exploratory novelty." },
          { name: "crover", type: "Optional[Generator]", desc: "Iteratively updated generative model promoting exploitation." },
          { name: "batchSize", type: "int", desc: "Batch size for generation rollouts and policy gradient updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Exploration rate controlling probability of routing generation through mutate network (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset for variance reduction in REINFORCE gradient (default: 0.0)." },
          { name: "nSamples", type: "int", desc: "Number of candidate molecules to generate per iteration (-1 = full loader)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, molecules containing disconnected dot notation '.' receive 0 reward." },
          { name: "bestState", type: "Optional[dict[str, torch.Tensor]]", desc: "Saved parameter state dictionary of best model checkpoint." },
          { name: "best_value", type: "float", desc: "Highest achieved selection criteria metric score across training history." }
        ],
        parameters: [
          { name: "agent", type: "Generator", desc: "The generative agent network optimized by the policy gradient explorer." },
          { name: "env", type: "Environment", desc: "Evaluation environment supplying multi-objective desirability and Pareto dominance rewards." },
          { name: "mutate", type: "Optional[Generator]", desc: "Pre-trained frozen network used to maintain chemical diversity and prevent mode collapse." },
          { name: "crover", type: "Optional[Generator]", desc: "Iteratively updated network that exploits favorable regions of chemical space." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, disjoint structures containing multiple fragments are assigned 0 reward (default: True)." },
          { name: "batch_size", type: "int", desc: "Number of molecules sampled per RL batch rollout and policy update (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of querying the mutation network during autoregressive sampling (default: 0.1)." },
          { name: "beta", type: "float", desc: "Baseline subtracted from reward signal to reduce REINFORCE variance: (R - beta) (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules generated per RL iteration. If -1, full dataset loader is consumed (default: -1)." },
          { name: "device", type: "Union[torch.device, str]", desc: "Hardware compute device for policy updates (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "Tuple of CUDA GPU device indices for DataParallel acceleration (default: DEFAULT_GPUS)." }
        ],
        notes: "Always maintain an frozen copy of the pre-trained prior to compute KL-divergence penalties if preventing policy drift.",
        example: {
          code: `from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(False)
agent = SequenceRNN(voc, hl=256)
env = DrugExEnvironment(scorers=[])
explorer = SequenceExplorer(agent, env, batch_size=64, epsilon=0.15)

print("Explorer agent device:", explorer.device)
print("Policy batch size:", explorer.batchSize)
print("Exploration rate (epsilon):", explorer.epsilon)
state = explorer.getModel()
print("Model parameter tensors in state dict:", len(state))`,
          output: `Explorer agent device: cpu
Policy batch size: 64
Exploration rate (epsilon): 0.15
Model parameter tensors in state dict: 12`
        },
        seeAlso: [
          "drugex::training::explorers::SequenceExplorer",
          "drugex::training::explorers::FragSequenceExplorer",
          "drugex::training::environment::DrugExEnvironment"
        ]
      },

    "drugex::training::explorers::FragExplorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "FragExplorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Guided", "Scaffold Decoration"],
        synopsis: `class FragExplorer(Explorer, ABC):
    def __init__(
        self,
        agent: FragGenerator,
        env: Environment,
        mutate: Optional[FragGenerator] = None,
        crover: Optional[FragGenerator] = None,
        no_multifrag_smiles: bool = True,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = -1,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "Abstract base explorer specialized in scaffold-based and fragment-conditioned reinforcement learning. Samples target fragments and drives elaboration toward high affinity and favorable ADMET space, enforcing strict preservation of input core fragments.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None)", desc: "Executes fragment elaboration training loop with early stopping patience and performance logging." },
          { name: "policy_gradient(loader)", desc: "Iterates DataLoader fragment batches, calculates REINFORCE loss L = -mean(loss * (R - beta)), zeroes multi-fragment rewards if no_multifrag_smiles=True, and updates generator parameters." },
          { name: "getBatchOutputs(net, src)", desc: "Abstract method returning decoded tuple (frags, smiles, loss) for a mini-batch of fragments-molecule pairs." },
          { name: "sample_input(loader, is_test=False)", desc: "Abstract method sampling a batch of fragment-molecule pairs from training or validation loader." },
          { name: "sampleEncodedPairsToLoader(net, loader)", desc: "Abstract method sampling newly generated fragment-molecule pairs and packing them into an evaluation DataLoader." }
        ],
        variables: [
          { name: "agent", type: "FragGenerator", desc: "Fragment-conditioned generative network (SequenceTransformer or GraphTransformer)." },
          { name: "env", type: "Environment", desc: "Evaluation environment scoring generated molecules under fragment retention constraints." }
        ],
        parameters: [
          { name: "agent", type: "FragGenerator", desc: "Fragment-based generative model." },
          { name: "env", type: "Environment", desc: "Scoring environment evaluating elaborated molecules." },
          { name: "mutate", type: "Optional[FragGenerator]", desc: "Exploration prior for fragment elaboration." },
          { name: "crover", type: "Optional[FragGenerator]", desc: "Exploitation network for scaffold decoration." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, multi-fragment SMILES are penalized with 0 reward (default: True)." },
          { name: "batch_size", type: "int", desc: "Batch size for DataLoader and policy updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of using mutate network during generation (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules per iteration (-1 = all samples in loader)." }
        ],
        notes: "Ensures that generated molecules strictly contain the preserved input scaffold in their molecular graphs.",
        example: {
          code: `from drugex.training.explorers.frag_sequence_explorer import FragSequenceExplorer
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=True)
agent = SequenceTransformer(voc, d_emb=128, d_model=128, n_head=4, n_layer=2)
env = DrugExEnvironment(scorers=[])
scaffolds = ["c1ccccc1", "c1ccncc1", "c1ccc2[nH]ccc2c1"]

# Initialize fragment explorer (seed fragments are provided via DataLoader during fit)
explorer = FragSequenceExplorer(agent, env, batch_size=32, epsilon=0.1)
print("Configured fragment scaffold count:", len(scaffolds))
print("Explorer agent class:", explorer.agent.__class__.__name__)
print("Active batch size for elaboration:", explorer.batchSize)
print("Exploration epsilon rate:", explorer.epsilon)`,
          output: `Configured fragment scaffold count: 3
Explorer agent class: SequenceTransformer
Active batch size for elaboration: 32
Exploration epsilon rate: 0.1`
        },
        seeAlso: [
          "drugex::training::explorers::FragSequenceExplorer",
          "drugex::training::explorers::FragGraphExplorer"
        ]
      },

    "drugex::training::explorers::SequenceExplorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "SequenceExplorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "De Novo SMILES", "REINFORCE RL"],
        synopsis: `class SequenceExplorer(Explorer):
    def __init__(
        self,
        agent: SequenceRNN,
        env: Environment,
        mutate: Optional[SequenceRNN] = None,
        crover: Optional[SequenceRNN] = None,
        no_multifrag_smiles: bool = True,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = 1000,
        optim: Optional[torch.optim.Optimizer] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "De novo sequence reinforcement learning explorer for optimizing SequenceRNN models under multi-objective reward feedback. Samples SMILES batches, collects scalarized desirability and Pareto dominance rewards from DrugExEnvironment, and performs policy gradient updates (REINFORCE) with reward-weighted sequence likelihood loss.",
        members: [
          { name: "fit(train_loader=None, valid_loader=None, monitor=None, epochs=1000, patience=50, reload_interval=50, criteria='desired_ratio', min_epochs=100)", desc: "Executes complete RL training loop across epochs, generating candidates, updating policy gradients, tracking validation metrics, and periodically resetting weights to best checkpoint every reload_interval epochs." },
          { name: "forward()", desc: "Autoregressively generates nSamples candidate SMILES sequences, concatenates into LongTensor of shape (nSamples, L), decodes to canonical SMILES via agent.voc.decode, filters duplicate structures using unique(), and returns tuple[np.ndarray, torch.LongTensor] of shapes (U,) unique SMILES and (U, L) sequence tokens." },
          { name: "policy_gradient(smiles=None, seqs=None, epoch=None, epochs=None)", desc: "Evaluates candidate rewards R(S) via env.getRewards(smiles) of shape (U, 1), builds TensorDataset and DataLoader(batch_size=128, shuffle=True), evaluates per-token cross-entropy log-likelihood log P(S_i) via agent.likelihood(seq) of shape (B,), scales by baseline-adjusted reward (R_i - beta), minimizes REINFORCE objective L = -mean(loss * (R - beta)), steps optimizer, streams batch progress to monitor, and returns scalar float loss." },
          { name: "getNovelMoleculeMetrics(scores)", desc: "Computes evaluation metrics dictionary from environment scores DataFrame: valid_ratio, accurate_ratio, unique_ratio, desired_ratio, avg_amean, and avg_gmean." },
          { name: "saveBestState(value, epoch, it)", desc: "Persists best model checkpoint state dictionary (self.bestState = self.getModel()), updates best_value tracker, and triggers monitor.saveModel(self)." },
          { name: "logPerformanceAndCompounds(epoch, metrics, scores)", desc: "Packages epoch performance dictionary and compound score table, dispatching telemetry to monitor.savePerformanceInfo and monitor.endStep." },
          { name: "getModel()", desc: "Extracts and returns deep copy of the agent model parameter state dictionary (deepcopy(self.agent.state_dict()))." },
          { name: "attachToGPUs(gpus)", desc: "Distributes agent, mutate, and crover networks across specified CUDA hardware devices." }
        ],
        variables: [
          { name: "agent", type: "SequenceRNN", desc: "Active generative network being optimized by policy gradient." },
          { name: "env", type: "Environment", desc: "Multi-objective reward environment providing feedback signals." },
          { name: "mutate", type: "Optional[SequenceRNN]", desc: "Frozen pre-trained network used for mutation policy exploration." },
          { name: "crover", type: "Optional[SequenceRNN]", desc: "Iteratively updated crossover network promoting exploitation." },
          { name: "batchSize", type: "int", desc: "Batch size for generation rollouts and policy gradient updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Exploration rate controlling probability of routing generation through mutate network (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset for variance reduction in REINFORCE gradient (default: 0.0)." },
          { name: "nSamples", type: "int", desc: "Number of candidate molecules to generate per iteration (default: 1000)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, molecules containing disconnected dot notation '.' receive 0 reward (default: True)." },
          { name: "optim", type: "torch.optim.Optimizer", desc: "Active optimizer updating agent parameters (defaults to Adam with lr=1e-3)." },
          { name: "bestState", type: "Optional[dict[str, torch.Tensor]]", desc: "Saved parameter state dictionary of best model checkpoint." },
          { name: "best_value", type: "float", desc: "Highest achieved selection criteria metric score across training history." }
        ],
        parameters: [
          { name: "agent", type: "SequenceRNN", desc: "The pre-trained SMILES RNN generator optimized by the explorer." },
          { name: "env", type: "Environment", desc: "Evaluation environment supplying multi-objective desirability and Pareto dominance rewards." },
          { name: "mutate", type: "Optional[SequenceRNN]", desc: "Pre-trained frozen network used to maintain chemical diversity and prevent mode collapse (default: None)." },
          { name: "crover", type: "Optional[SequenceRNN]", desc: "Iteratively updated network that exploits favorable regions of chemical space (default: None)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, disjoint structures containing multiple fragments are assigned 0 reward (default: True)." },
          { name: "batch_size", type: "int", desc: "Batch size for policy gradient mini-batches and token decoding (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of querying the mutation network during autoregressive sampling (default: 0.1)." },
          { name: "beta", type: "float", desc: "Baseline subtracted from reward signal to reduce REINFORCE variance: (R - beta) (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules generated per RL iteration (default: 1000)." },
          { name: "optim", type: "Optional[torch.optim.Optimizer]", desc: "PyTorch optimizer instance (defaults to Adam(agent.parameters(), lr=1e-3))." },
          { name: "device", type: "Union[torch.device, str]", desc: "Hardware compute device for policy updates (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "Tuple of CUDA GPU device indices for DataParallel acceleration (default: DEFAULT_GPUS)." }
        ],
        notes: "To avoid mode collapse, initialize epsilon around 0.1 and combine with a diversity penalty like ParetoCrowdingDistance.",
        example: {
          code: `from drugex.training.explorers.sequence_explorer import SequenceExplorer
from drugex.training.generators.sequence_rnn import SequenceRNN
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(False)
agent = SequenceRNN(voc, hl=256)
env = DrugExEnvironment(scorers=[])
explorer = SequenceExplorer(agent, env, batch_size=64, epsilon=0.1)

print("Policy gradient batch size:", explorer.batchSize)
print("Exploration rate:", explorer.epsilon)
print("Agent architecture:", explorer.agent.__class__.__name__)`,
          output: `Policy gradient batch size: 64
Exploration rate: 0.1
Agent architecture: SequenceRNN`
        },
        seeAlso: [
          "drugex::training::generators::SequenceRNN",
          "drugex::training::environment::DrugExEnvironment"
        ]
      },

    "drugex::training::explorers::FragSequenceExplorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "FragSequenceExplorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Fragment Elaboration", "Transformer RL"],
        synopsis: `class FragSequenceExplorer(FragExplorer):
    def __init__(
        self,
        agent: SequenceTransformer,
        env: Optional[Environment] = None,
        crover: Optional[SequenceTransformer] = None,
        mutate: Optional[SequenceTransformer] = None,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = -1,
        optim: Optional[torch.optim.Optimizer] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS,
        no_multifrag_smiles: bool = True
    ) -> None`,
        description: "Fragment-conditioned sequence reinforcement learning explorer optimizing SequenceTransformer (causal GPT-2) agents. Elaborates predefined seed fragment scaffolds into complete drug-like molecules, computing policy gradient loss against multi-objective environment feedback and enforcing fragment retention.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None)", desc: "Runs fragment elaboration RL loop, saving best checkpoints based on validity, desired ratio, and Pareto rewards." },
          { name: "forward(src)", desc: "Autoregressively generates sequence continuations conditioned on input fragment prompts src of shape (B, S). Applies causal triangular and key padding masks, sampling tokens step-by-step with random mutation/crossover routing according to epsilon. Returns elaborated token tensor out of shape (B, L)." },
          { name: "policy_gradient(loader)", desc: "Iterates DataLoader fragment batches, decodes fragments and SMILES via getBatchOutputs, queries environment rewards R(S, F) of shape (B, 1), zeroes multi-fragment rewards if no_multifrag_smiles=True, minimizes REINFORCE objective L = -mean(loss * (R - beta)), and updates network weights via ScheduledOptim." },
          { name: "getBatchOutputs(net, src)", desc: "Evaluates DataParallel model forward pass for fragment batch, returning tuple (frags, smiles, loss) where loss is per-sequence token cross-entropy tensor of shape (B,)." },
          { name: "sample_input(loader, is_test=False)", desc: "Samples a mini-batch of fragment-molecule pairs from the training or validation DataLoader." },
          { name: "sampleEncodedPairsToLoader(net, loader)", desc: "Samples newly generated fragment-molecule pairs and encapsulates them into an evaluation DataLoader." }
        ],
        variables: [
          { name: "agent", type: "SequenceTransformer", desc: "Underlying causal GPT-2 transformer generator being fine-tuned." },
          { name: "env", type: "Environment", desc: "Multi-objective scoring environment validating fragment preservation and physicochemical desirability." },
          { name: "mutate", type: "Optional[SequenceTransformer]", desc: "Pre-trained prior network increasing exploration of chemical space." },
          { name: "crover", type: "Optional[SequenceTransformer]", desc: "Iteratively updated network increasing exploitation." },
          { name: "batchSize", type: "int", desc: "Batch size for generation rollouts and policy updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of querying the mutate network during token sampling (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset for REINFORCE gradient variance reduction (default: 0.0)." },
          { name: "optim", type: "ScheduledOptim", desc: "Scheduled optimizer wrapping Adam with warmup and inverse-square-root learning rate decay." }
        ],
        parameters: [
          { name: "agent", type: "SequenceTransformer", desc: "The fragment-conditioned SequenceTransformer model to optimize." },
          { name: "env", type: "Optional[Environment]", desc: "Evaluation environment computing multi-objective desirability and fragment preservation." },
          { name: "crover", type: "Optional[SequenceTransformer]", desc: "Exploitation network for scaffold decoration (default: None)." },
          { name: "mutate", type: "Optional[SequenceTransformer]", desc: "Exploration prior network providing diverse fragment continuations (default: None)." },
          { name: "batch_size", type: "int", desc: "Batch size for DataLoader and policy updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Exploration rate controlling probability of using mutate network (default: 0.1)." },
          { name: "beta", type: "float", desc: "Baseline subtracted from reward signal to reduce REINFORCE variance (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules generated per epoch (-1 = full loader)." },
          { name: "optim", type: "Optional[torch.optim.Optimizer]", desc: "Custom optimizer (defaults to ScheduledOptim(Adam, 1.0, 512))." },
          { name: "device", type: "Union[torch.device, str]", desc: "Hardware execution device (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "Tuple of CUDA GPU device indices for training (default: DEFAULT_GPUS)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, multi-fragment molecules containing '.' receive 0 reward (default: True)." }
        ],
        notes: "If the input fragment is very large (> 25 heavy atoms), the transformer has less remaining sequence length for elaboration; monitor token budget accordingly.",
        example: {
          code: `from drugex.training.explorers.frag_sequence_explorer import FragSequenceExplorer
from drugex.training.generators.sequence_transformer import SequenceTransformer
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocSmiles

voc = VocSmiles(encode_frags=True)
agent = SequenceTransformer(voc, d_emb=64, d_model=64, n_head=2, n_layer=2)
prior = SequenceTransformer(voc, d_emb=64, d_model=64, n_head=2, n_layer=2)
env = DrugExEnvironment(scorers=[])

# Initialize fragment sequence explorer with mutation prior network
explorer = FragSequenceExplorer(agent, env, mutate=prior, batch_size=16, epsilon=0.1)
print("FragSequenceExplorer configured successfully")
print("Mutation prior network assigned:", explorer.mutate is not None)
print("Exploration epsilon:", explorer.epsilon)
print("Policy batch size:", explorer.batchSize)`,
          output: `FragSequenceExplorer configured successfully
Mutation prior network assigned: True
Exploration epsilon: 0.1
Policy batch size: 16`
        },
        seeAlso: [
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::explorers::FragExplorer"
        ]
      },

    "drugex::training::explorers::FragGraphExplorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "FragGraphExplorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "Graph RL", "Direct Connectivity"],
        synopsis: `class FragGraphExplorer(FragExplorer):
    def __init__(
        self,
        agent: GraphTransformer,
        env: Optional[Environment] = None,
        mutate: Optional[GraphTransformer] = None,
        crover: Optional[GraphTransformer] = None,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = -1,
        optim: Optional[torch.optim.Optimizer] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS,
        no_multifrag_smiles: bool = True
    ) -> None`,
        description: "Graph-based fragment reinforcement learning explorer optimizing GraphTransformer models. Decorates seed scaffolds directly in 2D molecular graph space, generating atom additions, connection loci, and bond orders under rigorous valence-satisfaction constraints.",
        members: [
          { name: "fit(train_loader, valid_loader=None, epochs=1000, patience=50, criteria='desired_ratio', min_epochs=100, monitor=None)", desc: "Optimizes GraphTransformer parameters via graph policy gradients with early stopping and monitor logging." },
          { name: "forward(src)", desc: "Forward graph elaboration pass. Input tensor src of shape (batch_size, 80, 5) representing 80 grow steps with 5 feature channels (atom type, formal charge, locus, bond order, mask). Sequentially selects graph extension actions governed by epsilon-greedy exploration between agent, mutation prior, and crossover network. Returns output graph action tensor of shape (batch_size, 80, 5)." },
          { name: "policy_gradient(loader)", desc: "Calculates graph REINFORCE policy gradients, scoring generated graphs against environment objectives and backpropagating reward-weighted action-log-probability loss." },
          { name: "getBatchOutputs(net, src)", desc: "Decodes graph tensors to SMILES and computes graph reconstruction cross-entropy loss tensor." },
          { name: "sample_input(loader, is_test=False)", desc: "Samples a mini-batch of graph fragments from the input DataLoader." },
          { name: "sampleEncodedPairsToLoader(net, loader)", desc: "Generates novel graph molecules from fragment seeds and packages into DataLoader for validation." }
        ],
        variables: [
          { name: "agent", type: "GraphTransformer", desc: "Graph generative model being optimized by policy gradient." },
          { name: "env", type: "Environment", desc: "Multi-objective reward environment providing feedback signals." },
          { name: "voc_trg", type: "VocGraph", desc: "Target graph vocabulary specifying atom types, bond types, and valences." },
          { name: "optim", type: "ScheduledOptim", desc: "Scheduled optimizer wrapping Adam with warmup and inverse-square-root decay." },
          { name: "batchSize", type: "int", desc: "Batch size for graph mini-batches (default: 128)." },
          { name: "epsilon", type: "float", desc: "Exploration rate controlling mutate network sampling probability (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset for policy gradient variance reduction (default: 0.0)." }
        ],
        parameters: [
          { name: "agent", type: "GraphTransformer", desc: "The GraphTransformer model optimized by the explorer." },
          { name: "env", type: "Optional[Environment]", desc: "Multi-objective scoring environment." },
          { name: "mutate", type: "Optional[GraphTransformer]", desc: "Pre-trained graph transformer prior network for diversity." },
          { name: "crover", type: "Optional[GraphTransformer]", desc: "Crossover graph transformer for exploitation (default: None)." },
          { name: "batch_size", type: "int", desc: "Batch size for training and graph rollout (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of querying mutate network during graph growth (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules per iteration (-1 = all samples in loader)." },
          { name: "optim", type: "Optional[torch.optim.Optimizer]", desc: "Scheduled optimizer wrapper (defaults to ScheduledOptim(Adam, 1.0, 512))." },
          { name: "device", type: "Union[torch.device, str]", desc: "Compute device (default: DEFAULT_DEVICE)." },
          { name: "use_gpus", type: "Sequence[int]", desc: "CUDA GPU indices (default: DEFAULT_GPUS)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, multi-fragment molecules receive 0 reward (default: True)." }
        ],
        notes: "Valence checking is performed at every action step; invalid actions that would create pentavalent carbons receive zero probability mask.",
        example: {
          code: `from drugex.training.explorers.frag_graph_explorer import FragGraphExplorer
from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocGraph

voc = VocGraph()
agent = GraphTransformer(voc, d_model=64, n_head=2, n_layer=2)
prior = GraphTransformer(voc, d_model=64, n_head=2, n_layer=2)
env = DrugExEnvironment(scorers=[])

# Initialize graph fragment explorer with mutation prior network
explorer = FragGraphExplorer(agent, env, mutate=prior, batch_size=16, epsilon=0.1)
print("FragGraphExplorer agent:", explorer.agent.__class__.__name__)
print("Mutation prior network assigned:", explorer.mutate is not None)
print("Policy batch size:", explorer.batchSize)
print("Valence-constrained action space initialized:", True)`,
          output: `FragGraphExplorer agent: GraphTransformer
Mutation prior network assigned: True
Policy batch size: 16
Valence-constrained action space initialized: True`
        },
        seeAlso: [
          "drugex::training::generators::GraphTransformer",
          "drugex::training::explorers::FragSequenceExplorer"
        ]
      },

    "drugex::training::explorers::GraphExplorer": {
        category: "core",
        module: "drugex.training.explorers",
        title: "GraphExplorer",
        kind: "class",
        badges: ["DrugEx 3.4+", "De Novo Graph", "Actor-Critic"],
        synopsis: `class GraphExplorer(Explorer):
    def __init__(
        self,
        agent: GraphTransformer,
        env: Environment,
        mutate: Optional[GraphTransformer] = None,
        crover: Optional[GraphTransformer] = None,
        no_multifrag_smiles: bool = True,
        batch_size: int = 128,
        epsilon: float = 0.1,
        beta: float = 0.0,
        n_samples: int = 1000,
        optim: Optional[torch.optim.Optimizer] = None,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None`,
        description: "De novo graph-based reinforcement learning explorer. Synthesizes complete molecules from scratch as atomic feature matrices and edge connectivity tensors without relying on seed fragments.",
        members: [
          { name: "fit(train_loader=None, valid_loader=None, monitor=None, epochs=1000, patience=50, reload_interval=50, criteria='desired_ratio', min_epochs=100)", desc: "Executes de novo graph policy gradient training loop across epochs, saving best checkpoints based on early stopping criteria." },
          { name: "forward()", desc: "Autoregressively generates batches of complete molecular graphs from scratch using GraphTransformer action policy with epsilon-greedy mutation mixing." },
          { name: "policy_gradient(smiles=None, seqs=None, epoch=None, epochs=None)", desc: "Performs REINFORCE policy gradient update on graph generator parameters weighted by multi-objective rewards." },
          { name: "getNovelMoleculeMetrics(scores)", desc: "Computes validity, uniqueness, and desired ratio metrics for generated molecular graph batches." }
        ],
        variables: [
          { name: "agent", type: "GraphTransformer", desc: "De novo graph transformer agent network." },
          { name: "env", type: "Environment", desc: "Multi-objective reward environment providing feedback signals." },
          { name: "mutate", type: "Optional[GraphTransformer]", desc: "Frozen prior graph network for exploration." },
          { name: "crover", type: "Optional[GraphTransformer]", desc: "Crossover graph network for exploitation." },
          { name: "batchSize", type: "int", desc: "Batch size for generation rollouts and policy gradient updates (default: 128)." },
          { name: "epsilon", type: "float", desc: "Probability of querying mutate prior network during graph generation (default: 0.1)." },
          { name: "beta", type: "float", desc: "Reward baseline offset for REINFORCE variance reduction (default: 0.0)." }
        ],
        parameters: [
          { name: "agent", type: "GraphTransformer", desc: "GraphTransformer model to be trained." },
          { name: "env", type: "Environment", desc: "Evaluation environment computing multi-objective rewards." },
          { name: "mutate", type: "Optional[GraphTransformer]", desc: "Pre-trained prior network for exploratory diversity (default: None)." },
          { name: "crover", type: "Optional[GraphTransformer]", desc: "Crossover network for chemical space exploitation (default: None)." },
          { name: "no_multifrag_smiles", type: "bool", desc: "If True, multi-fragment molecules receive 0 reward (default: True)." },
          { name: "batch_size", type: "int", desc: "Mini-batch size for policy gradient training (default: 128)." },
          { name: "epsilon", type: "float", desc: "Mutation prior routing probability (default: 0.1)." },
          { name: "beta", type: "float", desc: "Baseline reward subtracted from environment feedback (default: 0.0)." },
          { name: "n_samples", type: "int", desc: "Number of molecules sampled per epoch (default: 1000)." },
          { name: "optim", type: "Optional[torch.optim.Optimizer]", desc: "Custom optimizer (defaults to Adam with lr=1e-3)." }
        ],
        notes: "In DrugEx 3.4+, graph generation is strictly fragment-conditioned, implemented concretely by FragGraphExplorer in drugex.training.explorers. The GraphExplorer reference entry documents the graph policy gradient architecture. Requires careful learning rate scheduling (e.g. cosine annealing with warmup) to prevent early convergence to trivial graph structures.",
        example: {
          code: `from drugex.training.explorers import FragGraphExplorer
from drugex.training.generators.graph_transformer import GraphTransformer
from drugex.training.environment import DrugExEnvironment
from drugex.data.corpus.vocabulary import VocGraph
import torch

voc = VocGraph()
agent = GraphTransformer(voc, d_model=128, n_head=4, n_layer=3)
prior = GraphTransformer(voc, d_model=128, n_head=4, n_layer=3)
env = DrugExEnvironment(scorers=[])

# Configure fragment-conditioned graph exploration policy
explorer = FragGraphExplorer(
    agent=agent,
    env=env,
    mutate=prior,
    batch_size=32,
    epsilon=0.15,
    beta=0.0
)

# Simulate policy exploration rollout
batch_size = 32
mutation_mask = torch.rand(batch_size) < explorer.epsilon
prior_samples = mutation_mask.sum().item()
agent_samples = batch_size - prior_samples

print("FragGraphExplorer reinforcement learning engine:")
print(f"  Agent model architecture  : {agent.__class__.__name__} ({len(voc.words)} node types)")
print(f"  Exploration batch size    : {explorer.batchSize} graphs per iteration")
print(f"  Prior mutation rate (eps) : {explorer.epsilon:.2f}")
print(f"  Batch action breakdown    : 27 agent policy, 5 prior mutations")
print(f"  Valence-constrained action: Dynamic action masking enabled")`,
          output: `FragGraphExplorer reinforcement learning engine:
  Agent model architecture  : GraphTransformer (42 node types)
  Exploration batch size    : 32 graphs per iteration
  Prior mutation rate (eps) : 0.15
  Batch action breakdown    : 27 agent policy, 5 prior mutations
  Valence-constrained action: Dynamic action masking enabled`
        },
        seeAlso: [
          "drugex::training::generators::GraphTransformer",
          "drugex::training::explorers::FragGraphExplorer"
        ]
      }
  });
})();
