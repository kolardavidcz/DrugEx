/**
 * DrugEx Documentation Platform - Desirability Modifiers
 * Library: modifiers (11 Symbols)
 */

(function () {
  const _root = typeof window !== "undefined" ? window : globalThis;
  _root.DATABASE = _root.DATABASE || {};

  Object.assign(_root.DATABASE, {
    "drugex::training::scorers::ScoreModifier": {
        category: "core",
        module: "drugex.training.scorers",
        title: "ScoreModifier",
        kind: "class",
        badges: ["DrugEx 3.4+", "Desirability Function", "Normalization"],
        synopsis: `class ScoreModifier(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        ...`,
        description: "Abstract base class for desirability transformation functions. Converts raw metrics (e.g. MW in Daltons, IC50 in nM, Shape Tanimoto in [0, 1]) into normalized desirability values strictly bounded in [0, 1].",
        members: [
          { name: "__call__(x)", desc: "Transforms raw numerical array x into normalized desirability scores." }
        ],
        variables: [],
        parameters: [],
        notes: "Modifiers should be continuous and differentiable wherever possible to provide smooth gradient surfaces for optimization algorithms.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.interfaces import ScoreModifier

class ExponentialWindowModifier(ScoreModifier):
    """Custom modifier penalizing values outside [low, high] optimal window."""
    def __init__(self, low: float = 3.0, high: float = 6.0):
        self.low = low
        self.high = high
    def __call__(self, x):
        arr = np.asarray(x, dtype=float)
        # 1.0 inside target range, decaying exponentially outside
        dev = np.maximum(0.0, np.maximum(self.low - arr, arr - self.high))
        return np.exp(-dev)

mod = ExponentialWindowModifier(low=3.0, high=6.0)
test_vals = np.array([1.5, 3.0, 4.5, 6.0, 8.0])
scores = mod(test_vals)

print("Target optimal window: [3.0, 6.0]")
for val, s in zip(test_vals, scores):
    print(f"  Raw value: {val:4.1f} -> Desirability: {s:.4f}")`,
          output: `Target optimal window: [3.0, 6.0]
  Raw value:  1.5 -> Desirability: 0.2231
  Raw value:  3.0 -> Desirability: 1.0000
  Raw value:  4.5 -> Desirability: 1.0000
  Raw value:  6.0 -> Desirability: 1.0000
  Raw value:  8.0 -> Desirability: 0.1353`
        },
        seeAlso: [
          "drugex::training::scorers::SmoothClippedScore",
          "drugex::training::scorers::ClippedScore",
          "drugex::training::scorers::MinMaxGaussian"
        ]
      },

    "drugex::training::scorers::SmoothClippedScore": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "SmoothClippedScore",
        kind: "class",
        badges: ["DrugEx 3.4+", "Sigmoid Desirability", "Smooth Gradient"],
        synopsis: `class SmoothClippedScore(ScoreModifier):
    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0
    ) -> None`,
        description: "Smooth sigmoidal modifier mapping values between lower_x and upper_x using a logistic curve. Ensures non-zero gradients across the entire real number line (-inf, +inf), eliminating vanishing gradients during reinforcement learning.",
        members: [
          { name: "__call__(x)", desc: "Applies logistic transformation f(x) = low_score + (high_score - low_score) / (1 + exp(-k * (x - middle_x))) where k = 4 / (upper_x - lower_x). Evaluates element-wise across float scalars or np.ndarray tensors of shape (N,) or (N, M)." }
        ],
        variables: [
          { name: "upper_x", type: "float", desc: "Input x-value where score approaches high_score." },
          { name: "lower_x", type: "float", desc: "Input x-value where score approaches low_score (default: 0.0)." },
          { name: "high_score", type: "float", desc: "Upper asymptotic desirability bound (default: 1.0)." },
          { name: "low_score", type: "float", desc: "Lower asymptotic desirability bound (default: 0.0)." },
          { name: "k", type: "float", desc: "Logistic steepness coefficient: k = 4.0 / (upper_x - lower_x)." },
          { name: "middle_x", type: "float", desc: "Logistic inflection midpoint: middle_x = (upper_x + lower_x) / 2.0." },
          { name: "L", type: "float", desc: "Dynamic range amplitude: L = high_score - low_score." }
        ],
        parameters: [
          { name: "upper_x", type: "float", desc: "Transition boundary where score reaches approximately high_score (or inflection end)." },
          { name: "lower_x", type: "float", desc: "Transition boundary where score reaches approximately low_score (default: 0.0)." },
          { name: "high_score", type: "float", desc: "Maximum asymptote of the sigmoid curve (default: 1.0)." },
          { name: "low_score", type: "float", desc: "Minimum asymptote of the sigmoid curve (default: 0.0)." }
        ],
        notes: "Formula: f(x) = low_score + L / (1 + exp(-k * (x - middle_x))), where k = 4 / (upper_x - lower_x), middle_x = (upper_x + lower_x) / 2, and L = high_score - low_score. The first derivative df/dx = L * k * sigma(z) * (1 - sigma(z)) is strictly positive everywhere for upper_x > lower_x, ensuring continuous policy gradient backpropagation without dead zones. If upper_x < lower_x, k is negative and the curve acts as a descending desirability penalty.",
        example: {
          code: `from drugex.training.scorers.modifiers import SmoothClippedScore
import numpy as np

# Ascending sigmoidal desirability for target kinase bioactivity (pIC50)
# Maps pIC50 from 4.0 (lower anchor) to 8.0 (upper anchor)
pic50_modifier = SmoothClippedScore(upper_x=8.0, lower_x=4.0)

candidate_pic50 = {
    "Compound_A (Weak)": 3.5,
    "Compound_B (Low)": 4.0,
    "Compound_C (Mid)": 6.0,
    "Compound_D (High)": 8.0,
    "Compound_E (Potent)": 9.5
}

raw_pic50 = np.array(list(candidate_pic50.values()))
desirabilities = pic50_modifier(raw_pic50)

print("SmoothClippedScore sigmoidal bioactivity desirability:")
print(f"  Anchor transition bounds : lower_x={pic50_modifier.lower_x:.1f}, upper_x={pic50_modifier.upper_x:.1f}")
print(f"  Logistic steepness (k)   : {pic50_modifier.k:.4f} (midpoint x0={pic50_modifier.middle_x:.1f})")
print("Candidate evaluations:")
for (name, val), des in zip(candidate_pic50.items(), desirabilities):
    print(f"  {name:<22} pIC50: {val:4.1f} -> Desirability: {des:.4f}")
print(f"  Non-vanishing gradient   : Min score = {desirabilities.min():.4f} > 0.0")`,
          output: `SmoothClippedScore sigmoidal bioactivity desirability:
  Anchor transition bounds : lower_x=4.0, upper_x=8.0
  Logistic steepness (k)   : 1.0000 (midpoint x0=6.0)
Candidate evaluations:
  Compound_A (Weak)      pIC50:  3.5 -> Desirability: 0.0759
  Compound_B (Low)       pIC50:  4.0 -> Desirability: 0.1192
  Compound_C (Mid)       pIC50:  6.0 -> Desirability: 0.5000
  Compound_D (High)      pIC50:  8.0 -> Desirability: 0.8808
  Compound_E (Potent)    pIC50:  9.5 -> Desirability: 0.9707
  Non-vanishing gradient   : Min score = 0.0759 > 0.0`
        },
        seeAlso: [
          "drugex::training::scorers::ClippedScore",
          "drugex::training::scorers::MinMaxGaussian"
        ]
      },

    "drugex::training::scorers::ClippedScore": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "ClippedScore",
        kind: "class",
        badges: ["DrugEx 3.4+", "Piecewise Linear", "Hard Threshold"],
        synopsis: `class ClippedScore(ScoreModifier):
    def __init__(
        self,
        upper_x: float,
        lower_x: float = 0.0,
        high_score: float = 1.0,
        low_score: float = 0.0
    ) -> None`,
        description: "Piecewise linear ramp score modifier with hard clamping bounds. Linearly interpolates values between lower_x and upper_x, clipping values below lower_x to low_score and above upper_x to high_score.",
        members: [
          { name: "__call__(x)", desc: "Linearly maps inputs via y = slope * x + intercept and clamps output to [low_score, high_score] using np.clip. Evaluates element-wise on float scalars or np.ndarray of shape (N,)." }
        ],
        variables: [
          { name: "upper_x", type: "float", desc: "High threshold for linear ramp." },
          { name: "lower_x", type: "float", desc: "Low threshold for linear ramp (default: 0.0)." },
          { name: "high_score", type: "float", desc: "Ceiling score (default: 1.0)." },
          { name: "low_score", type: "float", desc: "Floor score (default: 0.0)." },
          { name: "slope", type: "float", desc: "Linear slope: (high_score - low_score) / (upper_x - lower_x)." },
          { name: "intercept", type: "float", desc: "Linear intercept: high_score - slope * upper_x." }
        ],
        parameters: [
          { name: "upper_x", type: "float", desc: "High threshold where output reaches high_score." },
          { name: "lower_x", type: "float", desc: "Low threshold where output reaches low_score (default: 0.0)." },
          { name: "high_score", type: "float", desc: "Maximum clipped score (default: 1.0)." },
          { name: "low_score", type: "float", desc: "Minimum clipped score (default: 0.0)." }
        ],
        notes: "Values strictly outside [min(lower_x, upper_x), max(lower_x, upper_x)] have zero gradient (df/dx = 0). In RL policy gradient exploration, this creates gradient dead zones where the agent receives zero directional learning signal. Use SmoothClippedScore when fine-tuning neural networks via REINFORCE.",
        example: {
          code: `from drugex.training.scorers.modifiers import ClippedScore
import numpy as np

# Piecewise linear ramp desirability for oral bioactivity LogP window
# Linearly maps LogP from 1.0 (0.0 desirability) to 4.0 (1.0 desirability)
logp_ramp = ClippedScore(upper_x=4.0, lower_x=1.0)

candidate_logp = {
    "Glycine": -3.2,
    "Ethanol": -0.3,
    "Benzene": 2.1,
    "Ibuprofen": 3.5,
    "Atorvastatin": 5.7
}

raw_vals = np.array(list(candidate_logp.values()))
scores = logp_ramp(raw_vals)

print("ClippedScore piecewise linear desirability:")
print(f"  Linear ramp interval   : [{logp_ramp.lower_x:.1f}, {logp_ramp.upper_x:.1f}]")
print(f"  Slope & intercept      : slope={logp_ramp.slope:.4f}, intercept={logp_ramp.intercept:.4f}")
print("Candidate evaluations:")
for (name, val), score in zip(candidate_logp.items(), scores):
    status = "Saturated (1.0)" if score == 1.0 else ("Clipped Floor (0.0)" if score == 0.0 else "Linear Ramp")
    print(f"  {name:<15} LogP: {val:4.1f} -> Desirability: {score:.3f} ({status})")`,
          output: `ClippedScore piecewise linear desirability:
  Linear ramp interval   : [1.0, 4.0]
  Slope & intercept      : slope=0.3333, intercept=-0.3333
Candidate evaluations:
  Glycine         LogP: -3.2 -> Desirability: 0.000 (Clipped Floor (0.0))
  Ethanol         LogP: -0.3 -> Desirability: 0.000 (Clipped Floor (0.0))
  Benzene         LogP:  2.1 -> Desirability: 0.367 (Linear Ramp)
  Ibuprofen       LogP:  3.5 -> Desirability: 0.833 (Linear Ramp)
  Atorvastatin    LogP:  5.7 -> Desirability: 1.000 (Saturated (1.0))`
        },
        seeAlso: [
          "drugex::training::scorers::SmoothClippedScore"
        ]
      },

    "drugex::training::scorers::MinMaxGaussian": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "MinMaxGaussian",
        kind: "class",
        badges: ["DrugEx 3.4+", "Half-Gaussian", "Directional Threshold"],
        synopsis: `class MinMaxGaussian(ScoreModifier):
    def __init__(
        self,
        mu: float,
        sigma: float,
        minimize: bool = False
    ) -> None`,
        description: "One-sided half-Gaussian score modifier for directional optimization. Evaluates to 1.0 on the optimal side of the boundary mu, and decays exponentially as a half-Gaussian curve on the penalized side.",
        members: [
          { name: "__call__(x)", desc: "Evaluates half-Gaussian decay on input scalar or np.ndarray array of shape (N,). If minimize=True, scores x <= mu as 1.0 and penalizes x > mu; if minimize=False, scores x >= mu as 1.0 and penalizes x < mu." }
        ],
        variables: [
          { name: "mu", type: "float", desc: "Threshold boundary where full desirability (1.0) begins or ends." },
          { name: "sigma", type: "float", desc: "Gaussian standard deviation controlling decay bandwidth on the penalized side." },
          { name: "minimize", type: "bool", desc: "If True, penalizes values above mu (minimization); if False, penalizes values below mu (maximization)." }
        ],
        parameters: [
          { name: "mu", type: "float", desc: "Threshold cutoff where half-Gaussian decay initiates." },
          { name: "sigma", type: "float", desc: "Standard deviation width controlling penalty decay slope." },
          { name: "minimize", type: "bool", desc: "Whether to penalize larger values (True) or smaller values (False, default)." }
        ],
        notes: "Formula: For minimize=True, f(x) = exp(-0.5 * (max(x, mu) - mu)^2 / sigma^2). For minimize=False, f(x) = exp(-0.5 * (min(x, mu) - mu)^2 / sigma^2). DrugEx provides MinGaussian (minimize=True) and MaxGaussian (minimize=False) convenience aliases.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import MinMaxGaussian

# Penalize molecular weights exceeding 450 Da with sigma=50 Da decay tail
mod = MinMaxGaussian(mu=450.0, sigma=50.0, minimize=True)
candidate_mws = np.array([320.0, 420.0, 450.0, 480.0, 520.0, 600.0])
scores = mod(candidate_mws)

# Contrast with maximization mode (e.g. demanding pIC50 >= 7.0)
mod_max = MinMaxGaussian(mu=7.0, sigma=1.0, minimize=False)
pic50_sample = 6.0
pic50_score = float(mod_max(pic50_sample))

print(f"MinMaxGaussian threshold: mu={mod.mu} Da, sigma={mod.sigma} Da (minimize={mod.minimize})")
for val, score in zip(candidate_mws, scores):
    status = "Optimal Plateau (1.0)" if val <= 450.0 else "Decaying Tail"
    print(f"  MW: {val:5.1f} Da -> Desirability: {score:.4f} ({status})")
print(f"Maximization test: pIC50 6.0 -> Desirability: {pic50_score:.4f}")`,
          output: `MinMaxGaussian threshold: mu=450.0 Da, sigma=50.0 Da (minimize=True)
  MW: 320.0 Da -> Desirability: 1.0000 (Optimal Plateau (1.0))
  MW: 420.0 Da -> Desirability: 1.0000 (Optimal Plateau (1.0))
  MW: 450.0 Da -> Desirability: 1.0000 (Optimal Plateau (1.0))
  MW: 480.0 Da -> Desirability: 0.8353 (Decaying Tail)
  MW: 520.0 Da -> Desirability: 0.3753 (Decaying Tail)
  MW: 600.0 Da -> Desirability: 0.0111 (Decaying Tail)
Maximization test: pIC50 6.0 -> Desirability: 0.6065`
        },
        seeAlso: [
          "drugex::training::scorers::Gaussian",
          "drugex::training::scorers::SmoothClippedScore"
        ]
      },

    "drugex::training::scorers::Linear": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "Linear",
        kind: "class",
        badges: ["DrugEx 3.4+", "Direct Scaling", "Simple Modifier"],
        synopsis: `class Linear(ScoreModifier):
    def __init__(self, slope: float = 1.0) -> None
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]`,
        description: "Direct linear transformation score modifier that scales input objective scores by a constant slope factor: y = slope * x. Leaves input values unshifted while scaling relative magnitude for multi-objective weighting or sign inversion.",
        members: [
          { name: "__call__(x)", desc: "Multiplies input score scalar or np.ndarray array by the constant slope factor: self.slope * x." }
        ],
        variables: [
          { name: "slope", type: "float", desc: "Linear slope multiplier factor." }
        ],
        parameters: [
          { name: "slope", type: "float", desc: "Multiplier coefficient applied to scores (default: 1.0)." }
        ],
        notes: "Formula: y = slope * x. If an intercept shift or bound clipping is required, use ClippedScore or chain with another modifier via Chained.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import Linear

# Scale normalized property scores by a slope multiplier (e.g. 0.5x scaling)
scaler = Linear(slope=0.5)
raw_scores = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
scaled_scores = scaler(raw_scores)

# Negative slope for cost minimization inversion
inverter = Linear(slope=-1.0)
inverted_scores = inverter(raw_scores)

print(f"Linear scaling multiplier: slope={scaler.slope}")
for raw, sc, inv in zip(raw_scores, scaled_scores, inverted_scores):
    print(f"  Raw: {raw:4.2f} -> Scaled (0.5x): {sc:4.2f} | Inverted (-1.0x): {inv:5.2f}")`,
          output: `Linear scaling multiplier: slope=0.5
  Raw: 0.20 -> Scaled (0.5x): 0.10 | Inverted (-1.0x): -0.20
  Raw: 0.40 -> Scaled (0.5x): 0.20 | Inverted (-1.0x): -0.40
  Raw: 0.60 -> Scaled (0.5x): 0.30 | Inverted (-1.0x): -0.60
  Raw: 0.80 -> Scaled (0.5x): 0.40 | Inverted (-1.0x): -0.80
  Raw: 1.00 -> Scaled (0.5x): 0.50 | Inverted (-1.0x): -1.00`
        },
        seeAlso: [
          "drugex::training::scorers::ClippedScore",
          "drugex::training::scorers::modifiers::Chained"
        ]
      },

    "drugex::training::scorers::Gaussian": {
        category: "core",
        module: "drugex.training.scorers",
        title: "Gaussian",
        kind: "class",
        badges: ["DrugEx 3.4+", "Bell Curve", "Single Optimum"],
        synopsis: `class Gaussian(ScoreModifier):
    def __init__(self, mu: float, sigma: float) -> None`,
        description: "Symmetric Gaussian bell-curve desirability modifier that maps property values into a normalized bell curve: y = exp(-0.5 * ((x - mu) / sigma)^2). Evaluates to exactly 1.0 at x == mu and decays continuously towards 0.0 with increasing distance from the target optimum.",
        members: [
          { name: "__call__(x)", desc: "Evaluates symmetric Gaussian desirability for a scalar float or element-wise across a 1D/2D NumPy array of shape (N,)." }
        ],
        variables: [
          { name: "mu", type: "float", desc: "Target property optimum value where desirability reaches its peak (f(mu) = 1.0)." },
          { name: "sigma", type: "float", desc: "Standard deviation width parameter controlling Gaussian tolerance bandwidth." }
        ],
        parameters: [
          { name: "mu", type: "float", desc: "Focal optimal target property mean (e.g. ideal logP, MW, or heavy atom count)." },
          { name: "sigma", type: "float", desc: "Gaussian standard deviation (sigma > 0); controls width of the reward window. Deviations of 1*sigma evaluate to exp(-0.5) ≈ 0.6065, while 2*sigma deviations evaluate to exp(-2.0) ≈ 0.1353." }
        ],
        notes: "Mathematical formula: f(x) = exp(-0.5 * ((x - mu) / sigma)^2). Unlike MinMaxGaussian which creates a one-sided plateau, Gaussian enforces a two-sided unimodal reward envelope. It provides continuous non-vanishing gradients across property boundaries, preventing RL policy gradient collapse caused by discontinuous step functions. For array inputs, operations are fully vectorized in NumPy.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import Gaussian

# Target molecular weight of 350 Da with tolerance sigma=40 Da
g = Gaussian(mu=350.0, sigma=40.0)
candidate_weights = np.array([250.0, 310.0, 350.0, 390.0, 450.0])
desirability = g(candidate_weights)

# Evaluate scalar call at exact target center and 2-sigma deviation
peak_val = float(g(350.0))
edge_val = float(g(430.0))

print(f"Gaussian target mu: {g.mu} Da, sigma: {g.sigma} Da (peak: {peak_val:.1f})")
for mw, d in zip(candidate_weights, desirability):
    print(f"  MW: {mw:5.1f} Da -> Desirability: {d:.4f}")
print(f"  Boundary 2-sigma check (430.0 Da): {edge_val:.4f}")`,
          output: `Gaussian target mu: 350.0 Da, sigma: 40.0 Da (peak: 1.0)
  MW: 250.0 Da -> Desirability: 0.0439
  MW: 310.0 Da -> Desirability: 0.6065
  MW: 350.0 Da -> Desirability: 1.0000
  MW: 390.0 Da -> Desirability: 0.6065
  MW: 450.0 Da -> Desirability: 0.0439
  Boundary 2-sigma check (430.0 Da): 0.1353`
        },
        seeAlso: [
          "drugex::training::scorers::MinMaxGaussian",
          "drugex::training::scorers::modifiers::SmoothHump"
        ]
      },

    "drugex::training::scorers::modifiers::SmoothHump": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "SmoothHump",
        kind: "class",
        badges: ["DrugEx 3.4+", "Desirability Function", "Bimodal Window"],
        synopsis: `class SmoothHump(ScoreModifier):
    def __init__(
        self,
        lower_x: float,
        upper_x: float,
        sigma: float = 1.0
    ) -> None`,
        description: "Smooth bell-curved desirability modifier defining a trapezoidal response plateau between lower_x and upper_x flanked by half-Gaussian wings. Inside [lower_x, upper_x], desirability evaluates to 1.0; outside this window, scores decay continuously without sharp discontinuities, providing stable non-vanishing gradients for policy gradient reinforcement learning.",
        members: [
          { name: "__call__(x)", desc: "Evaluates piecewise trapezoidal-Gaussian desirability across input float scalar or 1D/2D NumPy array of shape (N,)." }
        ],
        variables: [
          { name: "lower_x", type: "float", desc: "Lower threshold cutoff where full desirability plateau (1.0) begins." },
          { name: "upper_x", type: "float", desc: "Upper threshold cutoff where full desirability plateau (1.0) ends." },
          { name: "sigma", type: "float", desc: "Gaussian decay bandwidth parameter controlling roll-off steepness." }
        ],
        parameters: [
          { name: "lower_x", type: "float", desc: "Lower boundary of the optimal property plateau (must satisfy lower_x <= upper_x)." },
          { name: "upper_x", type: "float", desc: "Upper boundary of the optimal property plateau." },
          { name: "sigma", type: "float", desc: "Decay curvature bandwidth (default: 1.0). Smaller sigma values produce sharp, steep dropoffs approaching binary boxcar steps; larger sigma values provide permissive gradients extending deep into suboptimal property regimes." }
        ],
        notes: "Mathematical definition: Returns 1.0 for lower_x <= x <= upper_x; exp(-0.5 * ((x - lower_x) / sigma)^2) for x < lower_x; and exp(-0.5 * ((x - upper_x) / sigma)^2) for x > upper_x. Essential in multi-objective optimization (such as Lipinski property filtering or heavy atom count bounds) where hard step-function cuts cause vanishing gradients and policy collapse.",
        example: {
          code: `from drugex.training.scorers.modifiers import SmoothHump
import numpy as np

# Bounded desirability plateau for lead-like LogP interval [2.0, 4.0] with sigma=0.5 roll-offs
hump_modifier = SmoothHump(lower_x=2.0, upper_x=4.0, sigma=0.5)

candidate_logp = {
    "Compound_1 (Hydrophilic)": 1.0,
    "Compound_2 (Lower Bound)": 2.0,
    "Compound_3 (Optimal Mid)": 3.0,
    "Compound_4 (Upper Bound)": 4.0,
    "Compound_5 (Lipophilic)":  5.0
}

raw_vals = np.array(list(candidate_logp.values()))
scores = hump_modifier(raw_vals)

print("SmoothHump trapezoidal plateau desirability:")
print(f"  Optimal plateau window : [{hump_modifier.lower_x:.1f}, {hump_modifier.upper_x:.1f}]")
print(f"  Gaussian wing sigma    : {hump_modifier.sigma:.2f}")
print("Candidate evaluations:")
for (name, val), score in zip(candidate_logp.items(), scores):
    status = "Plateau (1.0)" if score == 1.0 else "Smooth Roll-off"
    print(f"  {name:<26} LogP: {val:4.1f} -> Desirability: {score:.4f} ({status})")`,
          output: `SmoothHump trapezoidal plateau desirability:
  Optimal plateau window : [2.0, 4.0]
  Gaussian wing sigma    : 0.50
Candidate evaluations:
  Compound_1 (Hydrophilic)   LogP:  1.0 -> Desirability: 0.1353 (Smooth Roll-off)
  Compound_2 (Lower Bound)   LogP:  2.0 -> Desirability: 1.0000 (Plateau (1.0))
  Compound_3 (Optimal Mid)   LogP:  3.0 -> Desirability: 1.0000 (Plateau (1.0))
  Compound_4 (Upper Bound)   LogP:  4.0 -> Desirability: 1.0000 (Plateau (1.0))
  Compound_5 (Lipophilic)    LogP:  5.0 -> Desirability: 0.1353 (Smooth Roll-off)`
        },
        seeAlso: [
          "drugex::training::scorers::SmoothClippedScore",
          "drugex::training::scorers::MinMaxGaussian",
          "drugex::training::scorers::Gaussian"
        ]
      },

    "drugex::training::scorers::modifiers::ThresholdedLinear": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "ThresholdedLinear",
        kind: "class",
        badges: ["DrugEx 3.4+", "Linear Ramp", "Piecewise Desirability"],
        synopsis: `class ThresholdedLinear(ScoreModifier):
    def __init__(self, threshold: float) -> None
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]`,
        description: "Linear score modifier with an upper saturation cap at threshold. Maps inputs linearly from 0.0 up to threshold (where score reaches 1.0), and saturates at 1.0 for all values exceeding threshold: f(x) = min(x, threshold) / threshold.",
        members: [
          { name: "__call__(x)", desc: "Calculates piecewise thresholded linear score: min(x, self.threshold) / self.threshold. Evaluates element-wise across float scalars or np.ndarray tensors of shape (N,)." }
        ],
        variables: [
          { name: "threshold", type: "float", desc: "Saturation threshold boundary where score reaches 1.0." }
        ],
        parameters: [
          { name: "threshold", type: "float", desc: "Saturation boundary value where desirability achieves 1.0." }
        ],
        notes: "Formula: f(x) = np.minimum(x, self.threshold) / self.threshold. Ideal for desirable continuous metrics that provide diminishing marginal returns once a threshold standard is satisfied (e.g. QED, target bioactivity, or novel scaffold diversity).",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import ThresholdedLinear

# Target bioactivity saturation at pIC50 = 8.0: linear ramp up to 8.0, saturated at 1.0 above
mod = ThresholdedLinear(threshold=8.0)
pic50_values = np.array([4.0, 6.0, 7.0, 8.0, 9.5])
scores = mod(pic50_values)

# Evaluate scalar edge cases: below zero and far above threshold
zero_test = float(mod(0.0))
over_test = float(mod(12.0))

print(f"ThresholdedLinear saturation boundary: threshold={mod.threshold}")
for val, s in zip(pic50_values, scores):
    status = "Saturated Plateau (1.0)" if val >= 8.0 else f"Linear Ramp ({s*100:.0f}%)"
    print(f"  pIC50: {val:4.1f} -> Desirability: {s:.4f} ({status})")
print(f"Edge cases: pIC50 0.0 -> {zero_test:.4f} | pIC50 12.0 -> {over_test:.4f}")`,
          output: `ThresholdedLinear saturation boundary: threshold=8.0
  pIC50:  4.0 -> Desirability: 0.5000 (Linear Ramp (50%))
  pIC50:  6.0 -> Desirability: 0.7500 (Linear Ramp (75%))
  pIC50:  7.0 -> Desirability: 0.8750 (Linear Ramp (88%))
  pIC50:  8.0 -> Desirability: 1.0000 (Saturated Plateau (1.0))
  pIC50:  9.5 -> Desirability: 1.0000 (Saturated Plateau (1.0))
Edge cases: pIC50 0.0 -> 0.0000 | pIC50 12.0 -> 1.0000`
        },
        seeAlso: [
          "drugex::training::scorers::Linear",
          "drugex::training::scorers::ClippedScore"
        ]
      },

    "drugex::training::scorers::modifiers::Squared": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "Squared",
        kind: "class",
        badges: ["DrugEx 3.4+", "Quadratic Penalty", "Power Modifier"],
        synopsis: `class Squared(ScoreModifier):
    def __init__(
        self,
        target_value: float,
        coefficient: float = 1.0
    ) -> None
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]`,
        description: "Quadratic penalty modifier achieving a peak desirability of 1.0 at target_value and penalizing deviations quadratically: f(x) = 1.0 - coefficient * (target_value - x)^2. Provides steep paraboloid penalties around an optimal property target.",
        members: [
          { name: "__call__(x)", desc: "Computes quadratic deviation penalty: 1.0 - self.coefficient * np.square(self.target_value - x)." }
        ],
        variables: [
          { name: "target_value", type: "float", desc: "Optimal property value where score achieves maximum of 1.0." },
          { name: "coefficient", type: "float", desc: "Curvature coefficient scaling quadratic distance penalty (default: 1.0)." }
        ],
        parameters: [
          { name: "target_value", type: "float", desc: "Target optimal property value." },
          { name: "coefficient", type: "float", desc: "Curvature steepness multiplier (default: 1.0)." }
        ],
        notes: "Formula: f(x) = 1.0 - coefficient * (target_value - x)^2. Unbounded below zero; pair with np.clip or ClippedScore if non-negative rewards are required.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import Squared

# Target logP = 3.0 with curvature coefficient 0.25: peak 1.0 at 3.0
mod = Squared(target_value=3.0, coefficient=0.25)
candidate_logp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
penalties = mod(candidate_logp)

# Evaluate scalar optimum check and severe out-of-range penalty
opt_val = float(mod(3.0))
out_val = float(mod(6.0))

print(f"Squared modifier: target={mod.target_value}, coefficient={mod.coefficient}")
for val, s in zip(candidate_logp, penalties):
    status = "Exact Optimum (1.0)" if val == 3.0 else f"Deviation Δ={abs(val - 3.0):.1f}"
    print(f"  LogP: {val:4.1f} -> Reward: {s:6.3f} ({status})")
print(f"Boundary evaluations: LogP 3.0 -> {opt_val:.3f} | LogP 6.0 -> {out_val:.3f}")`,
          output: `Squared modifier: target=3.0, coefficient=0.25
  LogP:  1.0 -> Reward:  0.000 (Deviation Δ=2.0)
  LogP:  2.0 -> Reward:  0.750 (Deviation Δ=1.0)
  LogP:  3.0 -> Reward:  1.000 (Exact Optimum (1.0))
  LogP:  4.0 -> Reward:  0.750 (Deviation Δ=1.0)
  LogP:  5.0 -> Reward:  0.000 (Deviation Δ=2.0)
Boundary evaluations: LogP 3.0 -> 1.000 | LogP 6.0 -> -1.250`
        },
        seeAlso: [
          "drugex::training::scorers::Linear",
          "drugex::training::scorers::modifiers::AbsoluteScore"
        ]
      },

    "drugex::training::scorers::modifiers::AbsoluteScore": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "AbsoluteScore",
        kind: "class",
        badges: ["DrugEx 3.4+", "Absolute Deviation", "Distance Penalty"],
        synopsis: `class AbsoluteScore(ScoreModifier):
    def __init__(self, target_value: float) -> None
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]`,
        description: "Linear absolute error distance modifier: y = 1.0 - |target_value - x|. Peaking at exactly 1.0 when property equals target_value and decreasing with slope -1.0 as distance from target increases.",
        members: [
          { name: "__call__(x)", desc: "Computes linear absolute error distance desirability: 1.0 - |target_value - x|." }
        ],
        variables: [
          { name: "target_value", type: "float", desc: "Optimal target property value." }
        ],
        parameters: [
          { name: "target_value", type: "float", desc: "Target focal value yielding maximum desirability (1.0)." }
        ],
        notes: "Linear alternative to Gaussian modifier without exponential calculation overhead; used to target ideal oral logP or molecular weight ranges.",
        example: {
          code: `from drugex.training.scorers.modifiers import AbsoluteScore
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP

# Define target lipophilicity (optimal oral bioavailability window centered at LogP = 3.0)
target_logp = 3.0
modifier = AbsoluteScore(target_value=target_logp)

drugs = {
    "Ethanol": "CCO",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Indomethacin": "COc1ccc2c(c1)c(CC(=O)O)cn2C(=O)c3ccc(Cl)cc3",
    "Celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "Amiodarone": "CCCCc1c(-c2cc(I)c(OCCN(CC)CC)c(I)c2)oc2ccccc12"
}

print(f"{'Drug Name':14} | {'Wildman-Crippen LogP':22} | {'|Target - LogP|':16} | {'Desirability Score'}")
print("-" * 72)
for name, smi in drugs.items():
    mol = Chem.MolFromSmiles(smi)
    logp = MolLogP(mol)
    diff = abs(target_logp - logp)
    score = modifier(logp)
    print(f"{name:14} | {logp:22.2f} | {diff:16.2f} | {score:18.3f}")`,
          output: `Drug Name      | Wildman-Crippen LogP   | |Target - LogP|  | Desirability Score
------------------------------------------------------------------------
Ethanol        |                  -0.00 |             3.00 |             -2.001
Paracetamol    |                   1.35 |             1.65 |             -0.651
Indomethacin   |                   3.10 |             0.10 |              0.902
Celecoxib      |                   3.99 |             0.99 |              0.010
Amiodarone     |                   7.24 |             4.24 |             -3.243`
        },
        seeAlso: [
          "drugex::training::scorers::Gaussian",
          "drugex::training::scorers::MinMaxGaussian"
        ]
      },

    "drugex::training::scorers::modifiers::Chained": {
        category: "core",
        module: "drugex.training.scorers.modifiers",
        title: "Chained",
        kind: "class",
        badges: ["DrugEx 3.4+", "Composite Modifier", "Pipeline Function"],
        synopsis: `class Chained(ScoreModifier):
    def __init__(self, modifiers: Sequence[ScoreModifier]) -> None
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]`,
        description: "Sequential composition of multiple score modifiers: y = f_n(...(f_2(f_1(x)))). Evaluates a pipeline of transformations sequentially, feeding intermediate outputs to subsequent modifiers.",
        members: [
          { name: "__call__(x)", desc: "Iteratively evaluates chained modifiers in forward sequence order." }
        ],
        variables: [
          { name: "modifiers", type: "List[ScoreModifier]", desc: "Ordered sequence of ScoreModifier instances executed in pipeline order." }
        ],
        parameters: [
          { name: "modifiers", type: "Sequence[ScoreModifier]", desc: "Ordered collection of ScoreModifier instances to pipeline." }
        ],
        notes: "Enables multi-stage reward shaping, such as scaling by Linear followed by thresholding via ThresholdedLinear or capping with ClippedScore.",
        example: {
          code: `import numpy as np
from drugex.training.scorers.modifiers import Chained, Linear, ThresholdedLinear

# Pipeline: 1) Scale raw property by 0.1x multiplier, 2) Saturate at 0.8 threshold
pipeline = Chained([
    Linear(slope=0.1),
    ThresholdedLinear(threshold=0.8)
])
test_vals = np.array([2.0, 5.0, 8.0, 10.0, 12.0])
pipeline_scores = pipeline(test_vals)

print("Chained Modifier Pipeline (Linear Scaling + Threshold Saturation):")
for raw, s in zip(test_vals, pipeline_scores):
    scaled = raw * 0.1
    print(f"  Raw: {raw:4.1f} -> Scaled: {scaled:4.2f} -> Pipeline Desirability: {s:.3f}")`,
          output: `Chained Modifier Pipeline (Linear Scaling + Threshold Saturation):
  Raw:  2.0 -> Scaled: 0.20 -> Pipeline Desirability: 0.250
  Raw:  5.0 -> Scaled: 0.50 -> Pipeline Desirability: 0.625
  Raw:  8.0 -> Scaled: 0.80 -> Pipeline Desirability: 1.000
  Raw: 10.0 -> Scaled: 1.00 -> Pipeline Desirability: 1.000
  Raw: 12.0 -> Scaled: 1.20 -> Pipeline Desirability: 1.000`
        },
        seeAlso: [
          "drugex::training::scorers::Linear",
          "drugex::training::scorers::ClippedScore",
          "drugex::training::scorers::ScoreModifier"
        ]
      }
  });
})();
