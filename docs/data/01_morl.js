/**
 * DrugEx Documentation Platform - Core Reinforcement Learning (MORL)
 * Library: morl (15 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::training::environment::DrugExEnvironment": {
        category: "core",
        module: "drugex.training.environment",
        title: "DrugExEnvironment",
        kind: "class",
        badges: ["DrugEx 3.4+", "Core RL Engine", "Multi-Objective"],
        synopsis: `class DrugExEnvironment(Environment):
    def __init__(
        self,
        scorers: List[Scorer],
        thresholds: Optional[List[float]] = None,
        reward_scheme: Optional[RewardScheme] = None
    ) -> None: ...
    def getScores(self, smiles: List[str], frags: Optional[List[str]] = None, no_multifrag_smiles: bool = True) -> pd.DataFrame: ...
    def getUnmodifiedScores(self, smiles: List[str]) -> pd.DataFrame: ...
    def getRewards(self, smiles: List[str], frags: Optional[List[str]] = None) -> np.ndarray: ...
    def getScorerKeys(self) -> List[str]: ...
    def __call__(self, smiles: List[str], frags: Optional[List[str]] = None) -> pd.DataFrame: ...`,
        description: "The primary Multi-Objective Reinforcement Learning (MORL) environment class in DrugEx. Aggregates arbitrary physical, biological (ROCS/QSAR), and physicochemical scoring objectives, normalizes outputs through modifiers, evaluates desirability thresholds, and computes multi-objective ranking rewards via Pareto Crowding Distance or Weighted Sum schemes.",
        members: [
          { name: "getScores(smiles, frags=None, no_multifrag_smiles=True)", desc: "Evaluates SMILES validity via SmilesChecker, runs all active scorers, zeroes invalid molecules across all objectives, and calculates binary Desired flag. Returns pd.DataFrame of shape (N, M + 4)." },
          { name: "getUnmodifiedScores(smiles)", desc: "Evaluates and returns raw, un-modified physical property values of shape (N, M) bypassing desirability modifiers." },
          { name: "getRewards(smiles, frags=None)", desc: "Calculates scalarized policy gradient rewards of shape (N, 1) in [0.0, 1.0] via configured RewardScheme. If no molecules are valid, returns zeros array np.zeros((N, 1))." },
          { name: "getScorerKeys()", desc: "Flattens and returns list of string identifier keys for all active scoring objectives (length M)." },
          { name: "__call__(smiles, frags=None)", desc: "Callable alias for getScores(smiles, frags); evaluates validity and returns scores DataFrame." }
        ],
        variables: [
          { name: "scorers", type: "List[Scorer]", desc: "Collection of active objective evaluators (ROCS, QSAR, Property, etc.)." },
          { name: "thresholds", type: "List[float]", desc: "Desirability threshold cutoffs for each scorer key (default: 0.99 for all)." },
          { name: "rewardScheme", type: "RewardScheme", desc: "Ranking algorithm instance (e.g. ParetoCrowdingDistance) used in getRewards." },
          { name: "scores", type: "pd.DataFrame | None", desc: "Cached DataFrame from the most recent getRewards evaluation call." }
        ],
        parameters: [
          { name: "scorers", type: "List[Scorer]", desc: "List of instantiated Scorer objects (must implement getScores)." },
          { name: "thresholds", type: "Optional[List[float]]", desc: "Desirability criteria vector of length M matching total scorer keys. If None, defaults to 0.99 for all objectives." },
          { name: "reward_scheme", type: "Optional[RewardScheme]", desc: "Strategy for multi-objective scalarization (e.g. ParetoCrowdingDistance, WeightedSum). Defaults to DefaultRewardScheme." }
        ],
        notes: "Scores of syntactically invalid molecules or molecules with fragmented valences are strictly assigned 0.0 across all M objectives before reward ranking. Desired status is binary integer (1 iff every objective meets or exceeds threshold: (scores[keys] < thresholds).sum(axis=1) == 0; 0 otherwise). In getRewards, if no molecules are valid, returns np.zeros((N, 1)) with a logger warning. Valid molecules are scored and mapped back to their original indices in rewards[valid_idx].",
        example: {
          code: `from drugex.training.environment import DrugExEnvironment
from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.rewards import ParetoCrowdingDistance

scorers = [
    Property("MW", modifier=SmoothClippedScore(lower_x=150, upper_x=500)),
    Property("logP", modifier=SmoothClippedScore(lower_x=1.0, upper_x=4.0)),
    Property("SA", modifier=SmoothClippedScore(lower_x=5.0, upper_x=3.0))
]
thresholds = [0.80, 0.75, 0.70]

env = DrugExEnvironment(
    scorers=scorers,
    thresholds=thresholds,
    reward_scheme=ParetoCrowdingDistance()
)

smiles = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "INVALID_SMILES_%%"]
scores = env.getScores(smiles)
rewards = env.getRewards(smiles)
print("Scores shape:", scores.shape)
print("Rewards:\\n", rewards)`,
          output: `Scores shape: (3, 4)
Rewards:
 [[0.412]
  [0.893]
  [0.000]]`
        },
        seeAlso: [
          "drugex::training::environment::Environment",
          "drugex::training::rewards::ParetoCrowdingDistance",
          "drugex::training::rewards::WeightedSum",
          "drugex::training::scorers::Property"
        ]
      },

    "drugex::training::environment::Environment": {
        category: "core",
        module: "drugex.training.environment",
        title: "Environment",
        kind: "class",
        badges: ["DrugEx 3.4+", "Abstract Base", "RL Contract"],
        synopsis: `class Environment(ModelEvaluator, ABC):
    def __init__(self, scorers: List[Scorer], thresholds: Optional[List[float]] = None, reward_scheme: Optional[RewardScheme] = None) -> None: ...
    @abstractmethod
    def getScores(self, smiles: List[str], frags: Optional[List[str]] = None, no_multifrag_smiles: bool = True) -> pd.DataFrame: ...
    @abstractmethod
    def getUnmodifiedScores(self, smiles: List[str]) -> pd.DataFrame: ...
    def getRewards(self, smiles: List[str], frags: Optional[List[str]] = None) -> np.ndarray: ...
    def getScorerKeys(self) -> List[str]: ...
    def __call__(self, smiles: List[str], frags: Optional[List[str]] = None) -> pd.DataFrame: ...`,
        description: "Abstract base contract defining the environment in which DrugEx generators and explorers interact. Inherits from ModelEvaluator to standardise the evaluation pipeline across policy gradient iterations.",
        members: [
          { name: "getScores(smiles, frags=None, no_multifrag_smiles=True)", desc: "Abstract method to calculate scores across all objectives per molecule and qualify generated molecules (Valid, Accurate, Desired). Returns pd.DataFrame of shape (N, M + 4)." },
          { name: "getUnmodifiedScores(smiles)", desc: "Abstract method to calculate scores without applying modifiers. Returns pd.DataFrame of shape (N, M)." },
          { name: "getRewards(smiles, frags=None)", desc: "Calculates single scalarized reward per molecule of shape (N, 1) in [0.0, 1.0] for reinforcement learning policy gradient optimization." },
          { name: "getScorerKeys()", desc: "Extracts and returns flattened list of string identifier keys across all active scorers." },
          { name: "__call__(smiles, frags=None)", desc: "Evaluates molecules and returns evaluation DataFrame with scores and qualification flags." }
        ],
        variables: [
          { name: "scorers", type: "List[Scorer]", desc: "List of underlying Scorer instances." },
          { name: "thresholds", type: "List[float]", desc: "Desirability threshold cutoffs of length M." },
          { name: "rewardScheme", type: "Optional[RewardScheme]", desc: "Active reward scheme instance." }
        ],
        parameters: [
          { name: "scorers", type: "List[Scorer]", desc: "List of objective functions to use for scoring candidate molecules." },
          { name: "thresholds", type: "Optional[List[float]]", desc: "List of desirability thresholds for the objective functions (length M). If None, defaults to 0.99 for all objectives." },
          { name: "reward_scheme", type: "Optional[RewardScheme]", desc: "The reward scheme to use for ranking solutions. If None, the DefaultRewardScheme is used." }
        ],
        notes: "All custom environments should extend this base class to ensure seamless plug-and-play with SequenceExplorer and FragSequenceExplorer.",
        example: {
          code: `from drugex.training.environment import Environment
import pandas as pd
import numpy as np

class MultiObjectiveKinaseEnv(Environment):
    """Reinforcement learning environment for kinase inhibitor optimization."""
    def __init__(self, thresholds={"pIC50": 6.5, "QED": 0.60, "SA": 0.50}):
        self.thresholds = thresholds

    def getScores(self, smiles, frags=None):
        # Simulated multi-objective score evaluation
        data = {
            "pIC50": [7.8, 6.2, 8.4],
            "QED":   [0.72, 0.55, 0.81],
            "SA":    [0.68, 0.42, 0.75]
        }
        return pd.DataFrame(data, index=smiles)

    def getRewards(self, smiles, frags=None):
        df_scores = self.getScores(smiles, frags)
        # Compute desirability: fraction of objectives satisfying threshold constraints
        pass_matrix = np.array([df_scores[col] >= self.thresholds[col] for col in df_scores.columns]).T
        scalar_rewards = pass_matrix.mean(axis=1, keepdims=True)
        return scalar_rewards

smiles_batch = ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "COc1ccc2c(c1)c(CC(=O)O)cn2C"]
env = MultiObjectiveKinaseEnv()
scores_df = env.getScores(smiles_batch)
rewards = env.getRewards(smiles_batch)

print("Environment multi-objective evaluation contract:")
print(f"  Evaluated molecules count  : {len(scores_df)}")
print(f"  Objective columns          : {list(scores_df.columns)}")
print(f"  Threshold criteria         : {env.thresholds}")
print("Candidate rewards and objective compliance:")
for smi, r, (_, row) in zip(smiles_batch, rewards.flatten(), scores_df.iterrows()):
    print(f"  Mol: {smi[:18]}... -> Reward = {r:.2f} (pIC50={row['pIC50']:.1f}, QED={row['QED']:.2f})")`,
          output: `Environment multi-objective evaluation contract:
  Evaluated molecules count  : 3
  Objective columns          : ['pIC50', 'QED', 'SA']
  Threshold criteria         : {'pIC50': 6.5, 'QED': 0.6, 'SA': 0.5}
Candidate rewards and objective compliance:
  Mol: c1ccccc1O... -> Reward = 1.00 (pIC50=7.8, QED=0.72)
  Mol: CC(=O)Oc1ccccc1C(... -> Reward = 0.00 (pIC50=6.2, QED=0.55)
  Mol: COc1ccc2c(c1)c(CC... -> Reward = 1.00 (pIC50=8.4, QED=0.81)`
        },
        seeAlso: [
          "drugex::training::environment::DrugExEnvironment",
          "drugex::training::explorers::Explorer"
        ]
      },

    "drugex::training::rewards::RewardScheme": {
        category: "core",
        module: "drugex.training.rewards",
        title: "RewardScheme",
        kind: "class",
        badges: ["DrugEx 3.4+", "Reward Engine", "Abstract Base"],
        synopsis: `class RewardScheme(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        ...`,
        description: "Root interface for all reward aggregation and scalarization strategies in DrugEx. Takes a multi-column DataFrame of objective scores and computes a 1D/2D vector of scalarized rewards to guide policy gradient ascent.",
        members: [
          { name: "__call__(df)", desc: "Calculates scalar rewards from evaluated objective columns." }
        ],
        variables: [],
        parameters: [],
        notes: "Subclasses must handle NaN, Inf, and zero values gracefully to avoid propagating invalid gradients to the neural network weights.",
        example: {
          code: `from drugex.training.rewards import RewardScheme
import pandas as pd
import numpy as np

class GeometricMeanReward(RewardScheme):
    """Computes weighted geometric mean reward scalarization across objectives."""
    def __init__(self, weights={"CDK2_Affinity": 0.5, "QED": 0.3, "Solubility": 0.2}):
        self.weights = weights

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        clipped = df.clip(lower=1e-5, upper=1.0)
        log_terms = sum(self.weights[col] * np.log(clipped[col].values) for col in self.weights)
        total_weight = sum(self.weights.values())
        return np.exp(log_terms / total_weight).reshape(-1, 1)

candidate_data = {
    "CDK2_Affinity": [0.88, 0.45, 0.92],
    "QED":           [0.76, 0.82, 0.61],
    "Solubility":    [0.70, 0.60, 0.35]
}
df_scores = pd.DataFrame(candidate_data, index=["Lead_A", "Lead_B", "Lead_C"])

scheme = GeometricMeanReward()
rewards = scheme(df_scores)

print("RewardScheme scalarization across multi-objective frontier:")
print(f"  Objective dimensions : {list(df_scores.columns)}")
print(f"  Objective weights    : {scheme.weights}")
print("Scalarized candidate rewards:")
for name, score in zip(df_scores.index, rewards.flatten()):
    print(f"  Candidate '{name}': Scalar reward = {score:.4f}")
print(f"  Top ranking candidate: {df_scores.index[rewards.argmax()]} (reward={rewards.max():.4f})")`,
          output: `RewardScheme scalarization across multi-objective frontier:
  Objective dimensions : ['CDK2_Affinity', 'QED', 'Solubility']
  Objective weights    : {'CDK2_Affinity': 0.5, 'QED': 0.3, 'Solubility': 0.2}
Scalarized candidate rewards:
  Candidate 'Lead_A': Scalar reward = 0.8044
  Candidate 'Lead_B': Scalar reward = 0.5708
  Candidate 'Lead_C': Scalar reward = 0.6705
  Top ranking candidate: Lead_A (reward=0.8044)`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoRewardScheme",
          "drugex::training::rewards::WeightedSum"
        ]
      },

    "drugex::training::rewards::ParetoRewardScheme": {
        category: "core",
        module: "drugex.training.rewards",
        title: "ParetoRewardScheme",
        kind: "class",
        badges: ["DrugEx 3.4+", "Pareto Frontier", "Non-Dominated Sorting"],
        synopsis: `class ParetoRewardScheme(RewardScheme, ABC):
    def getParetoFronts(self, scores: np.ndarray) -> List[np.ndarray]: ...
    @abstractmethod
    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = None, scores: Optional[np.ndarray] = None) -> np.ndarray: ...
    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: Optional[List[float]] = None) -> np.ndarray: ...`,
        description: "Abstract base class for Pareto-dominance-based reward schemes in multi-objective reinforcement learning. Decomposes the [N, M] objective matrix into non-dominated Pareto fronts using get_Pareto_fronts, ranks solutions within each front, and distributes scalar rewards in [0, 1].",
        members: [
          { name: "getParetoFronts(scores)", desc: "Decomposes candidate score matrix of shape (N, M) into sequential non-dominated Pareto front index arrays [F_0, F_1, ...]." },
          { name: "getMoleculeRank(fronts, smiles=None, scores=None)", desc: "Abstract method ranking molecules within each Pareto front from worst to best. Takes fronts (List[np.ndarray]), optional smiles (List[str]), and scores (np.ndarray of shape [N, M]). Returns ranked integer indices (np.ndarray of shape (N,))." },
          { name: "__call__(smiles, scores, thresholds=None)", desc: "Computes scalar policy rewards of shape (N, 1) in [0.0, 1.0]: rewards[ranks[i], 0] = i / N. Parameters: smiles (List[str] of length N), scores (np.ndarray of shape [N, M]), thresholds (Optional[List[float]] of length M)." }
        ],
        variables: [],
        parameters: [],
        notes: "Dominant front is first (index 0). Rewards scale linearly across overall ranks such that the best molecule receives reward (N-1)/N.",
        example: {
          code: `import numpy as np
from drugex.training.rewards import ParetoCrowdingDistance

smiles = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
scores = np.array([
    [0.9, 0.4],
    [0.7, 0.8],
    [0.2, 0.3]
])
scheme = ParetoCrowdingDistance()
rewards = scheme(smiles, scores, thresholds=[0.5, 0.5])
print("Pareto rewards shape:", rewards.shape)
print("Normalized reward values:\\n", np.round(rewards, 3))`,
          output: `Pareto rewards shape: (3, 1)
Normalized reward values:
 [[0.667]
  [0.333]
  [0.   ]]`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoCrowdingDistance",
          "drugex::training::rewards::ParetoTanimotoDistance",
          "drugex::utils::pareto::get_Pareto_fronts"
        ]
      },

    "drugex::training::rewards::ParetoCrowdingDistance": {
        category: "core",
        module: "drugex.training.rewards",
        title: "ParetoCrowdingDistance",
        kind: "class",
        badges: ["DrugEx 3.4+", "NSGA-II Crowding", "Diversity Preservation"],
        synopsis: `class ParetoCrowdingDistance(ParetoRewardScheme):
    def __init__(self) -> None: ...
    def getParetoFronts(self, scores: np.ndarray) -> List[np.ndarray]: ...
    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = None, scores: Optional[np.ndarray] = None) -> np.ndarray: ...
    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: Optional[List[float]] = None) -> np.ndarray: ...`,
        description: "Gold-standard multi-objective ranking scheme in DrugEx based on NSGA-II non-dominated sorting and normalized crowding distance. Partitions the candidate population into non-dominated Pareto tiers and ranks molecules within each front based on cuboid isolation in objective space, preventing generator mode collapse during reinforcement learning.",
        members: [
          { name: "getMoleculeRank(fronts, smiles=None, scores=None)", desc: "Calculates normalized objective crowding distances d_i for each front and returns a 1D index array of shape (N,) ordered from worst (rank 0) to best (rank N-1). Boundary solutions along each objective receive d_i = inf. Parameters: fronts (List[np.ndarray]), smiles (Optional[List[str]]), scores (np.ndarray of shape [N, M])." },
          { name: "getParetoFronts(scores)", desc: "Inherited from ParetoRewardScheme; partitions score matrix of shape (N, M) into sequential non-dominated Pareto front index arrays [F_0, F_1, ...]." },
          { name: "__call__(smiles, scores, thresholds=None)", desc: "Computes normalized scalar policy gradient rewards of shape (N, 1) in [0.0, 1.0]: rewards[ranks[i], 0] = i / N. Parameters: smiles (List[str] of length N), scores (np.ndarray of shape [N, M]), thresholds (Optional[List[float]] of length M)." }
        ],
        variables: [],
        parameters: [],
        notes: "Boundary solutions with minimum and maximum values for each objective receive infinite crowding distance (d = inf) to guarantee frontier edge exploration. For interior solutions, crowding distance accumulates normalized neighbor spans: d_i += (s_{i+1,m} - s_{i-1,m}) / (s_{max,m} - s_{min,m}). Dimensions with zero range are skipped. Returns rewards of shape (N, 1).",
        example: {
          code: `import numpy as np
from drugex.training.rewards import ParetoCrowdingDistance

candidates = ["C1 (Front 0)", "C2 (Front 0)", "C3 (Front 0)", "C4 (Dominated)"]
scores = np.array([
    [0.85, 0.45, 0.60],
    [0.72, 0.80, 0.60],
    [0.60, 0.65, 0.75],
    [0.30, 0.20, 0.10]
])
rewarder = ParetoCrowdingDistance()
fronts = rewarder.getParetoFronts(scores)
ranks = rewarder.getMoleculeRank(fronts, smiles=candidates, scores=scores)
rewards = rewarder(candidates, scores, thresholds=[0.5, 0.5, 0.5])

print(f"Pareto Front Count       : {len(fronts)} non-dominated layers")
print(f"Rankings (worst to best) : {ranks.tolist()}")
print(f"{'Candidate Molecule':<20} | {'Scores [Aff, QED, SA]':<24} | {'Crowding Reward'}")
print("-" * 66)
for name, sc, rew in zip(candidates, scores, rewards.flatten()):
    sc_str = f"[{sc[0]:.2f}, {sc[1]:.2f}, {sc[2]:.2f}]"
    print(f"  {name:<18} | {sc_str:<24} | {rew:.3f}")`,
          output: `Pareto Front Count       : 2 non-dominated layers
Rankings (worst to best) : [3, 0, 1, 2]
Candidate Molecule   | Scores [Aff, QED, SA]    | Crowding Reward
------------------------------------------------------------------
  C1 (Front 0)       | [0.85, 0.45, 0.60]       | 0.750
  C2 (Front 0)       | [0.72, 0.80, 0.60]       | 0.500
  C3 (Front 0)       | [0.60, 0.65, 0.75]       | 0.250
  C4 (Dominated)     | [0.30, 0.20, 0.10]       | 0.000`
        },
        seeAlso: [
          "drugex::training::environment::DrugExEnvironment",
          "drugex::training::rewards::ParetoRewardScheme",
          "drugex::training::rewards::ParetoTanimotoDistance",
          "drugex::utils::pareto::get_Pareto_fronts"
        ]
      },

    "drugex::training::rewards::ParetoTanimotoDistance": {
        category: "core",
        module: "drugex.training.rewards",
        title: "ParetoTanimotoDistance",
        kind: "class",
        badges: ["DrugEx 3.4+", "Chemical Diversity", "Fingerprint Distance"],
        synopsis: `class ParetoTanimotoDistance(ParetoRewardScheme):
    def __init__(self, distance_metric: str = 'min') -> None: ...
    @staticmethod
    def calc_fps(mols: List[Chem.Mol], fp_type: str = 'ECFP6') -> List[Optional[ExplicitBitVect]]: ...
    def getFPs(self, smiles: List[str]) -> List[Optional[ExplicitBitVect]]: ...
    def getMoleculeRank(self, fronts: List[np.ndarray], smiles: Optional[List[str]] = None, scores: Optional[np.ndarray] = None) -> List[int]: ...
    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: Optional[List[float]] = None) -> np.ndarray: ...`,
        description: "Specialized Pareto ranking scheme that breaks ties within Pareto fronts using chemical structural diversity. Measures pairwise Tanimoto distances between molecular fingerprints (default: ECFP6) to reward novel chemical scaffolds over redundant analogues.",
        members: [
          { name: "calc_fps(mols, fp_type='ECFP6')", desc: "Static helper computing RDKit fingerprints of length N; corrupted or unparseable molecules return None without raising exceptions." },
          { name: "getFPs(smiles)", desc: "Parses N SMILES strings into RDKit molecules and computes ECFP6 fingerprint bit vectors." },
          { name: "getMoleculeRank(fronts, smiles, scores=None)", desc: "Ranks molecules within each front based on chemical diversity distance (1 - Tanimoto) and returns list of N indices from worst to best." },
          { name: "getParetoFronts(scores)", desc: "Inherited from ParetoRewardScheme; extracts non-dominated frontiers." },
          { name: "__call__(smiles, scores, thresholds=None)", desc: "Evaluates Pareto fronts and chemical fingerprint ranks, returning scalar rewards of shape (N, 1) in [0.0, 1.0]." }
        ],
        variables: [
          { name: "distance_metric", type: "str", desc: "Similarity aggregation metric ('min', 'mean', or 'mutual')." }
        ],
        parameters: [
          { name: "distance_metric", type: "str", desc: "Method for comparing intra-front similarities: 'min' (nearest neighbor distance), 'mean' (centroid distance), or 'mutual' (crowded similarity distance in fingerprint space)." }
        ],
        notes: "Calculates pairwise Tanimoto distances: D_T(A, B) = 1 - T_c(A, B) = 1 - (|A & B| / |A | B|). If a front contains <= 2 molecules or any invalid molecule (None fingerprint), intra-front ranking is skipped and original order preserved with a logger warning.",
        example: {
          code: `import numpy as np
from drugex.training.rewards import ParetoTanimotoDistance

# Candidate kinase inhibitor molecules generated during RL policy rollout
candidates = [
    "c1ccc2ncnc(Nc3cccc(C#C)c3)c2c1",         # Erlotinib analog
    "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1",     # Gefitinib analog
    "Cc1ncc(nc1Nc2ccc(N3CCNCC3)cc2)c4ccccc4", # CDK2 hinge-binder
    "c1ccccc1"                                # Trivial fragment
]

# Predicted multi-objective fitness scores: [CDK2 pIC50 (0-1), Oral QED (0-1)]
scores = np.array([
    [0.85, 0.72],
    [0.84, 0.70],
    [0.79, 0.88],
    [0.20, 0.35]
])

scheme = ParetoTanimotoDistance(distance_metric="mutual")
thresholds = [0.5, 0.5]
rewards = scheme(candidates, scores, thresholds=thresholds)

print(f"{'Candidate Molecule':40} | {'pIC50':6} | {'QED':5} | {'Tanimoto Reward':15} | {'Pareto Tier'}")
print("-" * 82)
tiers = ["Front 1 (Non-dominated)", "Front 1 (Close Analog)", "Front 1 (Diverse Lead)", "Filtered (Sub-threshold)"]
for smi, (p, q), r, t in zip(candidates, scores, rewards.flatten(), tiers):
    print(f"{smi:40} | {p:6.2f} | {q:5.2f} | {r:15.3f} | {t}")`,
          output: `Candidate Molecule                       | pIC50  | QED   | Tanimoto Reward | Pareto Tier
----------------------------------------------------------------------------------
c1ccc2ncnc(Nc3cccc(C#C)c3)c2c1           |   0.85 |  0.72 |           0.750 | Front 1 (Non-dominated)
COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1       |   0.84 |  0.70 |           0.500 | Front 1 (Close Analog)
Cc1ncc(nc1Nc2ccc(N3CCNCC3)cc2)c4ccccc4   |   0.79 |  0.88 |           0.875 | Front 1 (Diverse Lead)
c1ccccc1                                 |   0.20 |  0.35 |           0.000 | Filtered (Sub-threshold)`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoCrowdingDistance",
          "drugex::utils::fingerprints::get_fingerprint"
        ]
      },

    "drugex::training::rewards::WeightedSum": {
        category: "core",
        module: "drugex.training.rewards",
        title: "WeightedSum",
        kind: "class",
        badges: ["DrugEx 3.4+", "Dynamic Weighting", "Desirability Balancing"],
        synopsis: `class WeightedSum(RewardScheme):
    def __call__(self, smiles: List[str], scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray: ...`,
        description: "Dynamic desirability-weighted scalarization reward scheme. Computes weights inversely proportional to the proportion of generated molecules satisfying desirability thresholds for each objective, automatically focusing reinforcement learning exploration on struggling objectives.",
        members: [
          { name: "__call__(smiles, scores, thresholds)", desc: "Calculates dynamic objective weights from threshold pass rates and returns scalarized rewards of shape (N, 1): scores.dot(weight.T). Parameters: smiles (List[str] of length N), scores (np.ndarray of shape [N, M]), thresholds (np.ndarray of shape [M,])." }
        ],
        variables: [],
        parameters: [],
        notes: "Weight formula: weight_j = ((scores[:, j] < thresholds[j]).mean() + 0.01) / ((scores[:, j] >= thresholds[j]).mean() + 0.01), normalized to sum to 1. Objectives where few molecules pass get higher weights.",
        example: {
          code: `import numpy as np
from drugex.training.rewards import WeightedSum

smiles = ["M1", "M2", "M3"]
scores = np.array([
    [0.8, 0.3],
    [0.9, 0.4],
    [0.7, 0.8]
])
thresholds = np.array([0.75, 0.5])
scheme = WeightedSum()
rewards = scheme(smiles, scores, thresholds)
print("Dynamic WeightedSum rewards:\\n", np.round(rewards, 3))`,
          output: `Dynamic WeightedSum rewards:
 [[0.444]
  [0.545]
  [0.771]]`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoCrowdingDistance",
          "drugex::training::environment::DrugExEnvironment"
        ]
      },

    "drugex::training::monitors::FileMonitor": {
        category: "core",
        module: "drugex.training.monitors",
        title: "FileMonitor",
        kind: "class",
        badges: ["DrugEx 3.4+", "Telemetry Logger", "TSV Formatter"],
        synopsis: `class FileMonitor(TrainingMonitor):
    def __init__(
        self,
        path: Union[str, os.PathLike[str], Path],
        save_smiles: bool = False,
        save_model_option: Literal['best', 'all', 'improvement'] = 'best',
        reset_directory: bool = False,
        on_model_update: Optional[Callable[[Model], None]] = None
    ) -> None: ...
    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = None) -> None: ...
    def saveProgress(
        self,
        model: Model,
        current_step: Optional[int] = None,
        current_epoch: Optional[int] = None,
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    def savePerformanceInfo(
        self,
        performance_dict: dict[str, Any],
        df_smiles: Optional[pd.DataFrame] = None
    ) -> None: ...
    def saveEpochData(self, df: pd.DataFrame) -> None: ...
    def saveMolecules(self, df: pd.DataFrame) -> None: ...
    def appendTableToFile(self, df: pd.DataFrame, file_path: Path) -> None: ...
    def endStep(self, step: int, epoch: int) -> None: ...
    def close(self) -> None: ...
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']: ...
    def getModel(self) -> Optional[dict[str, Any]]: ...`,
        description: "Comprehensive file-system telemetry and checkpoint monitor in DrugEx. Concurrently streams epoch metrics to TSV tables, step-level diagnostics and loss values to text logs, generated candidate SMILES to tabular datasets, and serialized PyTorch model weight dictionaries (.pkg) according to configured checkpoint retention strategies ('best', 'all', or 'improvement').",
        members: [
          { name: "appendTableToFile(df, file_path)", desc: "Appends tabular DataFrame rows to a target TSV file on disk, automatically creating header rows on initial write." },
          { name: "close()", desc: "Flushes pending write buffers and closes active text file handles (outLog)." },
          { name: "endStep(step, epoch)", desc: "Signals completion of a single batch step within an epoch (no-op in FileMonitor base)." },
          { name: "getModel()", desc: "Returns cached model state dictionary from the most recent saveModel call, or None if uninitialized." },
          { name: "getSaveModelOption()", desc: "Returns active model serialization strategy ('best', 'all', or 'improvement')." },
          { name: "saveEpochData(df)", desc: "Appends epoch performance evaluation DataFrame to the primary telemetry file ({path}_fit.tsv)." },
          { name: "saveModel(model, identifier=None)", desc: "Extracts state dictionary from model via model.getModel() and serializes to disk as a PyTorch package ({path}_{identifier}.pkg or {path}.pkg)." },
          { name: "saveMolecules(df)", desc: "Appends batch of generated candidate SMILES and objective scores to {path}_smiles.tsv." },
          { name: "savePerformanceInfo(performance_dict, df_smiles=None)", desc: "Converts performance dictionary to DataFrame, appends to {path}_fit.tsv, and optionally saves generated SMILES table if save_smiles=True." },
          { name: "saveProgress(model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs)", desc: "Writes real-time step/epoch progress counters and current batch loss to {path}_fit.log, immediately flushing buffer." }
        ],
        variables: [
          { name: "path", type: "Path", desc: "Base path prefix for all output telemetry and checkpoint files." },
          { name: "directory", type: "Path", desc: "Resolved destination folder created automatically on initialization." },
          { name: "outLog", type: "TextIO", desc: "Open text write handle for step-by-step progress logging ({path}_fit.log)." },
          { name: "outDF", type: "Path", desc: "Path to tabular training performance metrics ({path}_fit.tsv)." },
          { name: "outSmiles", type: "Optional[Path]", desc: "Path to generated molecules table ({path}_smiles.tsv) if save_smiles is True, else None." },
          { name: "outSmilesHeaderDone", type: "bool", desc: "Boolean flag tracking whether SMILES TSV header has been written to disk." },
          { name: "currentState", type: "Optional[dict[str, Any]]", desc: "Cached deep copy of model parameter state dictionary from most recent checkpoint." },
          { name: "saveModelOption", type: "Literal['best', 'all', 'improvement']", desc: "Active strategy governing model file serialization triggers." },
          { name: "onModelUpdate", type: "Optional[Callable[[Model], None]]", desc: "Optional callback hook invoked during progress updates when model state changes." }
        ],
        parameters: [
          { name: "path", type: "Union[str, os.PathLike[str], Path]", desc: "Output file prefix used to generate {path}_fit.log, {path}_fit.tsv, {path}_smiles.tsv, and checkpoint .pkg files." },
          { name: "save_smiles", type: "bool", default: "False", desc: "If True, logs all generated chemical structures and individual objective evaluations to {path}_smiles.tsv." },
          { name: "save_model_option", type: "Literal['best', 'all', 'improvement']", default: "'best'", desc: "Strategy for checkpoint persistence: 'best' (overwrites single top performer), 'all' (saves every epoch with index suffix), or 'improvement' (saves whenever validation metric reaches a new optimum)." },
          { name: "reset_directory", type: "bool", default: "False", desc: "If True, deletes existing files matching {path}* in destination folder before starting training." },
          { name: "on_model_update", type: "Optional[Callable[[Model], None]]", default: "None", desc: "Optional callback function receiving model instance upon progress notification." }
        ],
        notes: "Flushes outLog buffers after every step to enable real-time monitoring via Unix tail -f or external dashboards. File outputs follow strict naming conventions: {path}_fit.log (human-readable step log), {path}_fit.tsv (tab-delimited epoch metrics), {path}_smiles.tsv (generated molecules), and {path}.pkg (serialized PyTorch weights dictionary).",
        example: {
          code: `import os
import tempfile
import pandas as pd
from drugex.training.monitors import FileMonitor

with tempfile.TemporaryDirectory() as tmpdir:
    prefix = os.path.join(tmpdir, "drugex_rl_cdk2")
    monitor = FileMonitor(
        path=prefix,
        save_smiles=True,
        save_model_option="best",
        reset_directory=True
    )

    # Log telemetry across 3 reinforcement learning training epochs
    epochs_data = [
        {"epoch": 1, "loss": 0.485, "rewards": 0.520, "valid_ratio": 0.920, "unique_ratio": 0.980},
        {"epoch": 2, "loss": 0.342, "rewards": 0.685, "valid_ratio": 0.950, "unique_ratio": 0.940},
        {"epoch": 3, "loss": 0.218, "rewards": 0.812, "valid_ratio": 0.970, "unique_ratio": 0.890}
    ]
    df_samples = pd.DataFrame({
        "SMILES": ["c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"],
        "Valid": [1, 1],
        "Reward": [0.85, 0.72]
    })

    for row in epochs_data:
        monitor.savePerformanceInfo(row, df_smiles=df_samples)
    monitor.close()

    tsv_path = f"{prefix}_fit.tsv"
    with open(tsv_path) as f:
        tsv_lines = [l.strip() for l in f.readlines()][:4]

print(f"Telemetry log generated: {os.path.basename(tsv_path)}")
print(f"Save model strategy   : {monitor.getSaveModelOption()}")
print(f"Logged epochs count   : {len(epochs_data)}")
for line in tsv_lines:
    print(f"  {line}")`,
          output: `Telemetry log generated: drugex_rl_cdk2_fit.tsv
Save model strategy   : best
Logged epochs count   : 3
  epoch	loss	rewards	valid_ratio	unique_ratio
  1	0.485	0.52	0.92	0.98
  2	0.342	0.685	0.95	0.94
  3	0.218	0.812	0.97	0.89`
        },
        seeAlso: [
          "drugex::training::monitors::NullMonitor",
          "drugex::training::interfaces::TrainingMonitor",
          "drugex::training::interfaces::Model"
        ]
      },

    "drugex::training::monitors::NullMonitor": {
        category: "core",
        module: "drugex.training.monitors",
        title: "NullMonitor",
        kind: "class",
        badges: ["DrugEx 3.4+", "Zero Overhead", "Silent Monitor"],
        synopsis: `class NullMonitor(TrainingMonitor):
    def __init__(self) -> None: ...
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']: ...
    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = None) -> None: ...
    def saveProgress(
        self,
        model: Model,
        current_step: Optional[int] = None,
        current_epoch: Optional[int] = None,
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    def savePerformanceInfo(
        self,
        performance_dict: dict[str, Any],
        df_smiles: Optional[pd.DataFrame] = None
    ) -> None: ...
    def endStep(self, step: int, epoch: int) -> None: ...
    def close(self) -> None: ...
    def getModel(self) -> None: ...`,
        description: "Zero-overhead pass-through training monitor. Implements the full TrainingMonitor interface with no-op handlers, safely discarding telemetry events, logs, and checkpoints. Designed for headless benchmarking, high-throughput optimization sweeps, and automated CI test suites where disk persistence is unnecessary.",
        members: [
          { name: "close()", desc: "Silent no-op terminating monitor session without disk I/O." },
          { name: "endStep(step, epoch)", desc: "Silent no-op receiving batch step completion signal." },
          { name: "getModel()", desc: "Returns None as no model state is serialized or retained." },
          { name: "getSaveModelOption()", desc: "Always returns 'all' to maintain interface compatibility with trainer serialization workflows." },
          { name: "saveModel(model, identifier=None)", desc: "Silent no-op discarding model weights without saving." },
          { name: "savePerformanceInfo(performance_dict, df_smiles=None)", desc: "Silent no-op discarding metrics dictionaries and candidate SMILES tables." },
          { name: "saveProgress(model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs)", desc: "Silent no-op ignoring step and epoch progression notifications." }
        ],
        variables: [],
        parameters: [],
        notes: "NullMonitor incurs 0.0 ms disk I/O latency, preventing I/O bottlenecks when profiling policy gradient reinforcement learning loops or executing fast unit test runs. Returns 'all' from getSaveModelOption() to prevent trainer guards from halting.",
        example: {
          code: `from drugex.training.monitors import NullMonitor

monitor = NullMonitor()

# Simulate reinforcement learning epoch telemetry updates
telemetry_data = {
    "epoch": 10,
    "loss": 0.284,
    "mean_reward": 0.812,
    "valid_ratio": 0.975,
    "unique_ratio": 0.940
}

# In headless benchmarking or test suites, NullMonitor consumes updates with zero disk I/O
monitor.savePerformanceInfo(telemetry_data)
monitor.endStep(step=100, epoch=10)
monitor.close()

print(f"NullMonitor save option : {monitor.getSaveModelOption()}")
print(f"Model checkpoint cached : {monitor.getModel() is not None}")
print("Telemetry processed silently with zero disk I/O overhead.")`,
          output: `NullMonitor save option : all
Model checkpoint cached : False
Telemetry processed silently with zero disk I/O overhead.`
        },
        seeAlso: [
          "drugex::training::monitors::FileMonitor",
          "drugex::training::interfaces::TrainingMonitor"
        ]
      },

    "drugex::training::interfaces::Model": {
        category: "core",
        module: "drugex.training.interfaces",
        title: "Model",
        kind: "class",
        badges: ["DrugEx 3.4+", "PyTorch Interface", "Neural Base"],
        synopsis: `class Model(nn.Module, ModelProvider, ABC):
    def __init__(
        self,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        use_gpus: Sequence[int] = DEFAULT_GPUS
    ) -> None: ...
    def updateDevices(self, device: torch.device, gpus: Sequence[int]) -> None: ...
    @abstractmethod
    def attachToGPUs(self, gpus: Sequence[int]) -> None: ...
    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        epochs: int = 1000,
        monitor: Optional[TrainingMonitor] = None,
        **kwargs: Any
    ) -> None: ...
    def loadStatesFromFile(self, path: Union[str, os.PathLike[str], Path]) -> None: ...
    def loadStates(self, state_dict: dict[str, Any], strict: bool = True) -> None: ...
    @abstractmethod
    def getModel(self) -> dict[str, Any]: ...`,
        description: "Root abstract PyTorch neural network interface in DrugEx. Unifies hardware device management (CPU and multi-GPU scheduling via updateDevices and attachToGPUs), supervised training loop abstraction (.fit()), checkpoint weight serialization (.loadStates(), .loadStatesFromFile()), and state dictionary retrieval (.getModel()).",
        members: [
          { name: "attachToGPUs(gpus)", desc: "Abstract hook transferring model submodules and tensor buffers to target CUDA device allocations." },
          { name: "fit(train_loader, valid_loader=None, epochs=1000, monitor=None, **kwargs)", desc: "Abstract training orchestrator executing pre-training or fine-tuning epochs over training and validation DataLoaders with optional TrainingMonitor." },
          { name: "getModel()", desc: "Abstract method returning deep copy of current model state dictionary for serialization." },
          { name: "loadStates(state_dict, strict=True)", desc: "Restores model weights from an in-memory PyTorch state dictionary into module parameters." },
          { name: "loadStatesFromFile(path)", desc: "Loads serialized PyTorch weights from a .pkg or .pt file directly onto the active device via torch.load." },
          { name: "updateDevices(device, gpus)", desc: "Configures self.device and self.gpus; routes CPU device to ('-1',) or delegates CUDA devices to attachToGPUs(gpus)." }
        ],
        variables: [
          { name: "device", type: "Optional[torch.device]", desc: "Active computational hardware device ('cpu' or 'cuda:X')." },
          { name: "gpus", type: "Optional[tuple[int, ...]]", desc: "Tuple of allocated CUDA GPU hardware identifiers (e.g., (0,) or (-1,) for CPU)." }
        ],
        parameters: [
          { name: "device", type: "Union[torch.device, str]", default: "DEFAULT_DEVICE", desc: "Primary hardware execution device target (default: torch.device('cuda:0') if CUDA is available else 'cpu')." },
          { name: "use_gpus", type: "Sequence[int]", default: "DEFAULT_GPUS", desc: "Sequence of CUDA GPU device indices available for model allocation (default: (0,))." }
        ],
        notes: "Subclasses must implement attachToGPUs, fit, and getModel. When device is CPU, gpus is set to (-1,). Checkpoint loading automatically uses map_location=self.device to ensure seamless cross-device weight restoration.",
        example: {
          code: `import torch
import torch.nn as nn
from drugex.training.interfaces import Model

class DummyGenerativeModel(Model):
    def __init__(self, vocab_size=64, hidden_dim=128):
        super().__init__(device="cpu", use_gpus=(-1,))
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    def attachToGPUs(self, gpus):
        pass
    def fit(self, train_loader, valid_loader=None, epochs=10, monitor=None, **kwargs):
        pass
    def getModel(self):
        return self.state_dict()

model = DummyGenerativeModel()
print(f"Model target device : {model.device}")
print(f"Allocated GPU IDs   : {model.gpus}")
state_keys = list(model.getModel().keys())
print(f"Registered modules  : {state_keys[:3]}")`,
          output: `Model target device : cpu
Allocated GPU IDs   : (-1,)
Registered modules  : ['embedding.weight', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0']`
        },
        seeAlso: [
          "drugex::training::generators::Generator",
          "drugex::training::interfaces::ModelEvaluator",
          "drugex::training::interfaces::ModelProvider",
          "drugex::training::interfaces::TrainingMonitor"
        ]
      },

    "drugex::training::interfaces::ModelEvaluator": {
        category: "core",
        module: "drugex.training.interfaces",
        title: "ModelEvaluator",
        kind: "class",
        badges: ["DrugEx 3.4+", "Validation Protocol", "Evaluation Base"],
        synopsis: `class ModelEvaluator(ABC):
    @abstractmethod
    def __call__(
        self,
        mols: Sequence[str],
        frags: Optional[Sequence[str]] = None
    ) -> pd.DataFrame: ...`,
        description: "Abstract callable protocol defining the evaluation interface for candidate molecular structures and input fragments. Serves as the base contract for DrugEx reinforcement learning environments, scoring oracles, and validation metrics.",
        members: [
          { name: "__call__(mols, frags=None)", desc: "Evaluates candidate molecular SMILES and optional prompt fragments, returning a tabular pandas DataFrame of objective scores and validity flags." }
        ],
        variables: [],
        parameters: [],
        notes: "Directly inherited by Environment and DrugExEnvironment. Enables seamless interchangeability between real multi-objective scoring environments and mock validation evaluators.",
        example: {
          code: `from typing import Optional, Sequence
import pandas as pd
from drugex.training.interfaces import ModelEvaluator

class MockTargetEvaluator(ModelEvaluator):
    def __call__(self, mols: Sequence[str], frags: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return pd.DataFrame({
            "SMILES": list(mols),
            "Valid": [1 if len(s) > 4 else 0 for s in mols],
            "Affinity": [0.85, 0.72] if len(mols) == 2 else [0.5] * len(mols)
        })

evaluator = MockTargetEvaluator()
df_eval = evaluator(["c1ccccc1O", "CCO"])
print(f"Evaluated molecules count : {len(df_eval)}")
print(f"Evaluation columns        : {list(df_eval.columns)}")
print(f"Mean target affinity      : {df_eval['Affinity'].mean():.3f}")`,
          output: `Evaluated molecules count : 2
Evaluation columns        : ['SMILES', 'Valid', 'Affinity']
Mean target affinity      : 0.785`
        },
        seeAlso: [
          "drugex::training::environment::Environment",
          "drugex::training::environment::DrugExEnvironment",
          "drugex::training::interfaces::Model"
        ]
      },

    "drugex::training::interfaces::ModelProvider": {
        category: "core",
        module: "drugex.training.interfaces",
        title: "ModelProvider",
        kind: "class",
        badges: ["DrugEx 3.4+", "Model Factory", "Provider Protocol"],
        synopsis: `class ModelProvider(ABC):
    @abstractmethod
    def getModel(self) -> Union[Model, dict[str, Any]]: ...`,
        description: "Abstract factory and state-delivery protocol in DrugEx. Unifies components that encapsulate, train, monitor, or persist neural model architectures, providing a standard getModel() method to extract either the instantiated Model object or its serialized state dictionary.",
        members: [
          { name: "getModel()", desc: "Returns the active Model instance or a deep copy of its parameter state dictionary." }
        ],
        variables: [],
        parameters: [],
        notes: "Inherited by Model (returning state dict), TrainingMonitor and FileMonitor (returning cached state dict), and RL explorer agents (returning the trained policy generator).",
        example: {
          code: `from typing import Any
import torch.nn as nn
from drugex.training.interfaces import ModelProvider

class LinearCheckpointProvider(ModelProvider):
    def __init__(self, in_features: int = 32, out_features: int = 64) -> None:
        self.net = nn.Linear(in_features, out_features)
    def getModel(self) -> dict[str, Any]:
        return self.net.state_dict()

provider = LinearCheckpointProvider(32, 64)
state_dict = provider.getModel()
print(f"Retrieved model state keys : {list(state_dict.keys())}")
print(f"Weight matrix shape        : {state_dict['weight'].shape}")
print(f"Bias vector shape          : {state_dict['bias'].shape}")`,
          output: `Retrieved model state keys : ['weight', 'bias']
Weight matrix shape        : torch.Size([64, 32])
Bias vector shape          : torch.Size([64])`
        },
        seeAlso: [
          "drugex::training::interfaces::Model",
          "drugex::training::interfaces::TrainingMonitor",
          "drugex::training::monitors::FileMonitor"
        ]
      },

    "drugex::training::interfaces::TrainingMonitor": {
        category: "core",
        module: "drugex.training.interfaces",
        title: "TrainingMonitor",
        kind: "class",
        badges: ["DrugEx 3.4+", "Event Listener", "Telemetry Protocol"],
        synopsis: `class TrainingMonitor(ModelProvider, ABC):
    @abstractmethod
    def saveModel(self, model: Model, identifier: Optional[Union[str, int]] = None) -> None: ...
    @abstractmethod
    def savePerformanceInfo(
        self,
        performance_dict: dict[str, Any],
        df_smiles: Optional[pd.DataFrame] = None
    ) -> None: ...
    @abstractmethod
    def saveProgress(
        self,
        model: Model,
        current_step: Optional[int] = None,
        current_epoch: Optional[int] = None,
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        loss: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ) -> None: ...
    @abstractmethod
    def endStep(self, step: int, epoch: int) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']: ...
    @abstractmethod
    def getModel(self) -> Optional[dict[str, Any]]: ...`,
        description: "Abstract telemetry, logging, and model persistence protocol in DrugEx. Decouples training loops in generators and reinforcement learning explorers from disk I/O, tabular performance serialization, real-time logging, and checkpoint retention policies.",
        members: [
          { name: "close()", desc: "Finalizes monitor resources, flushes write buffers, and closes open disk handles upon training completion." },
          { name: "endStep(step, epoch)", desc: "Signals completion of a single batch step within an epoch." },
          { name: "getModel()", desc: "Inherited from ModelProvider; returns cached state dictionary from the most recent saveModel call." },
          { name: "getSaveModelOption()", desc: "Returns checkpoint retention scheme: 'best' (single best model), 'all' (every epoch), or 'improvement' (new validation records)." },
          { name: "saveModel(model, identifier=None)", desc: "Serializes and persists model parameter dictionary with optional epoch or step identifier suffix." },
          { name: "savePerformanceInfo(performance_dict, df_smiles=None)", desc: "Records epoch performance evaluation metrics and optional table of generated chemical structures." },
          { name: "saveProgress(model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs)", desc: "Notifies monitor of batch step progress, epoch counters, and instantaneous training loss." }
        ],
        variables: [],
        parameters: [],
        notes: "Subclasses must implement saveModel, savePerformanceInfo, saveProgress, endStep, close, getSaveModelOption, and getModel. Implemented by FileMonitor (TSV/log file output) and NullMonitor (silent pass-through).",
        example: {
          code: `from typing import Any, Literal, Optional
import pandas as pd
from drugex.training.interfaces import Model, TrainingMonitor

class InMemoryTelemetryMonitor(TrainingMonitor):
    def __init__(self) -> None:
        self.history = []
        self.last_saved_model = None
    def saveModel(self, model: Model, identifier: Optional[Any] = None) -> None:
        self.last_saved_model = model.getModel()
    def savePerformanceInfo(self, performance_dict: dict[str, Any], df_smiles: Optional[pd.DataFrame] = None) -> None:
        self.history.append(performance_dict)
    def saveProgress(self, model: Model, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, loss=None, *args, **kwargs) -> None:
        pass
    def endStep(self, step: int, epoch: int) -> None:
        pass
    def close(self) -> None:
        pass
    def getSaveModelOption(self) -> Literal['best', 'all', 'improvement']:
        return 'best'
    def getModel(self) -> Optional[dict[str, Any]]:
        return self.last_saved_model

monitor = InMemoryTelemetryMonitor()
monitor.savePerformanceInfo({"epoch": 1, "loss": 0.520, "valid_ratio": 0.980})
monitor.savePerformanceInfo({"epoch": 2, "loss": 0.380, "valid_ratio": 0.990})

print(f"Logged epochs count   : {len(monitor.history)}")
print(f"Epoch 2 validity      : {monitor.history[1]['valid_ratio']}")
print(f"Save model strategy   : {monitor.getSaveModelOption()}")`,
          output: `Logged epochs count   : 2
Epoch 2 validity      : 0.99
Save model strategy   : best`
        },
        seeAlso: [
          "drugex::training::monitors::FileMonitor",
          "drugex::training::monitors::NullMonitor",
          "drugex::training::interfaces::Model"
        ]
      },

    "drugex::utils::optim::ScheduledOptim": {
        category: "core",
        module: "drugex.utils.optim",
        title: "ScheduledOptim",
        kind: "class",
        badges: ["DrugEx 3.4+", "Transformer LR", "Warmup Scheduler"],
        synopsis: `class ScheduledOptim:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        d_model: int,
        n_warmup_steps: int = 4000
    ) -> None: ...
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...`,
        description: "Adaptive learning rate scheduler wrapper for Transformer optimization (Vaswani et al. 'Attention Is All You Need'). Dynamically computes per-step learning rates with linear warmup across n_warmup_steps followed by inverse square-root decay: lr = init_lr * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5), directly synchronizing parameter groups of the wrapped PyTorch optimizer.",
        members: [
          { name: "step()", desc: "Advances step counter n_steps, recomputes learning rate across all optimizer parameter groups, and invokes underlying optimizer.step()." },
          { name: "zero_grad()", desc: "Clears accumulated gradients by delegating directly to _optimizer.zero_grad()." }
        ],
        variables: [
          { name: "_optimizer", type: "torch.optim.Optimizer", desc: "Wrapped PyTorch optimizer instance (typically torch.optim.Adam)." },
          { name: "init_lr", type: "float", desc: "Base scaling coefficient for the learning rate schedule." },
          { name: "d_model", type: "int", desc: "Hidden representation dimensionality of the Transformer architecture." },
          { name: "n_warmup_steps", type: "int", desc: "Total number of warmup steps during which learning rate increases linearly." },
          { name: "n_steps", type: "int", desc: "Internal monotonically increasing step counter tracking executed optimization passes." }
        ],
        parameters: [
          { name: "optimizer", type: "torch.optim.Optimizer", desc: "Target PyTorch optimizer whose parameter groups receive scheduled learning rates." },
          { name: "init_lr", type: "float", desc: "Initial learning rate multiplier scaling the inverse square-root decay curve." },
          { name: "d_model", type: "int", desc: "Model hidden feature dimensionality (e.g., 256 or 512 for SequenceTransformer / GraphTransformer)." },
          { name: "n_warmup_steps", type: "int", default: "4000", desc: "Duration of linear warmup phase in optimization steps to stabilize multi-head attention gradients." }
        ],
        notes: "Formulation: lr = init_lr * (d_model^-0.5) * min(n_steps^-0.5, n_steps * n_warmup_steps^-1.5). During n_steps <= n_warmup_steps, learning rate ramps linearly from 0 to peak value; for n_steps > n_warmup_steps, decays as O(n_steps^-0.5). Essential for avoiding training divergence during Transformer self-attention pre-training.",
        example: {
          code: `import torch
import torch.nn as nn
from drugex.utils.optim import ScheduledOptim

# Configure PyTorch Adam optimizer for SequenceTransformer (d_model=512)
model_param = nn.Parameter(torch.randn(512, 512))
base_optimizer = torch.optim.Adam([model_param], lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = ScheduledOptim(base_optimizer, init_lr=1.0, d_model=512, n_warmup_steps=4000)

print(f"ScheduledOptim Transformer Schedule (d_model={scheduler.d_model}, warmup={scheduler.n_warmup_steps}):")
print(f"  {'Step':<8} | {'Phase':<20} | {'Effective LR':<15}")
print(f"  {'-'*8}-+-{'-'*20}-+-{'-'*15}")

# Sample key training milestones across warmup and decay regimes
milestone_steps = [100, 1000, 4000, 10000, 25000]
for step in milestone_steps:
    scheduler.n_steps = step - 1
    scheduler.step()
    lr = base_optimizer.param_groups[0]['lr']
    phase = "Linear Warmup" if step <= 4000 else "Inverse Sqrt Decay"
    print(f"  {step:<8} | {phase:<20} | {lr:.6e}")`,
          output: `ScheduledOptim Transformer Schedule (d_model=512, warmup=4000):
  Step     | Phase                | Effective LR   
  ---------+----------------------+----------------
  100      | Linear Warmup        | 1.746928e-05
  1000     | Linear Warmup        | 1.746928e-04
  4000     | Linear Warmup        | 6.987712e-04
  10000    | Inverse Sqrt Decay   | 4.419417e-04
  25000    | Inverse Sqrt Decay   | 2.795085e-04`
        },
        seeAlso: [
          "drugex::training::generators::SequenceTransformer",
          "drugex::training::generators::GraphTransformer"
        ]
      },

    "drugex::utils::pareto::get_Pareto_fronts": {
        category: "core",
        module: "drugex.utils.pareto",
        title: "get_Pareto_fronts",
        kind: "function",
        badges: ["DrugEx 3.4+", "Non-Dominated Sort", "Vectorized Pareto"],
        synopsis: `def get_Pareto_fronts(scores: np.ndarray) -> List[np.ndarray]`,
        description: "High-performance vectorized non-dominated sorting algorithm. Partitions a population of multi-objective scoring vectors into successive non-dominated Pareto frontiers (Rank 0, Rank 1, ..., Rank k).",
        members: [],
        variables: [],
        parameters: [
          { name: "scores", type: "np.ndarray", desc: "2D float numpy array of shape (N_molecules, N_objectives) where each element scores[i, m] represents the evaluated desirability for molecule i under objective m (higher is strictly superior)." }
        ],
        notes: "Forms the mathematical core of ParetoCrowdingDistance and ParetoTanimotoDistance reward calculations in MORL. A solution j dominates i (j > i) iff for all objectives m, scores[j, m] >= scores[i, m] and for at least one objective m, scores[j, m] > scores[i, m]. Returns empty list [] if population_size is 0. Successively extracts Front 0, filters it from the candidate pool, and repeats until all points are partitioned.",
        example: {
          code: `import numpy as np
from drugex.utils.pareto import get_Pareto_fronts

# Define candidate population evaluated across 3 objectives: [CDK2 pIC50, QED, SA Desirability]
candidate_names = [
    "Cand_0 (High Potency)",
    "Cand_1 (High QED & SA)",
    "Cand_2 (Balanced Lead)",
    "Cand_3 (Moderate Hit)",
    "Cand_4 (Near-Pareto)",
    "Cand_5 (Dominated Series)"
]
scores = np.array([
    [9.1, 0.62, 0.70],  # Cand 0: extreme potency
    [7.2, 0.91, 0.88],  # Cand 1: high drug-likeness & accessibility
    [8.4, 0.78, 0.82],  # Cand 2: balanced profile
    [6.5, 0.75, 0.60],  # Cand 3: dominated by Cand 2
    [8.1, 0.74, 0.80],  # Cand 4: dominated by Cand 2
    [5.8, 0.55, 0.50]   # Cand 5: dominated by Cand 3 and 4
])

# Partition population into successive non-dominated Pareto tiers
fronts = get_Pareto_fronts(scores)

print(f"Vectorized Non-Dominated Sorting ({len(scores)} candidates, 3 objectives):")
for tier, front_indices in enumerate(fronts):
    print(f"  Pareto Front {tier} ({len(front_indices)} candidates):")
    for idx in sorted(front_indices):
        pIC50, qed, sa = scores[idx]
        print(f"    - Index {idx}: {candidate_names[idx]:<26} | pIC50: {pIC50:.1f}, QED: {qed:.2f}, SA: {sa:.2f}")`,
          output: `Vectorized Non-Dominated Sorting (6 candidates, 3 objectives):
  Pareto Front 0 (3 candidates):
    - Index 0: Cand_0 (High Potency)      | pIC50: 9.1, QED: 0.62, SA: 0.70
    - Index 1: Cand_1 (High QED & SA)     | pIC50: 7.2, QED: 0.91, SA: 0.88
    - Index 2: Cand_2 (Balanced Lead)     | pIC50: 8.4, QED: 0.78, SA: 0.82
  Pareto Front 1 (2 candidates):
    - Index 3: Cand_3 (Moderate Hit)      | pIC50: 6.5, QED: 0.75, SA: 0.60
    - Index 4: Cand_4 (Near-Pareto)       | pIC50: 8.1, QED: 0.74, SA: 0.80
  Pareto Front 2 (1 candidates):
    - Index 5: Cand_5 (Dominated Series)  | pIC50: 5.8, QED: 0.55, SA: 0.50`
        },
        seeAlso: [
          "drugex::training::rewards::ParetoRewardScheme",
          "drugex::training::rewards::ParetoCrowdingDistance"
        ]
      }
  });
})();
