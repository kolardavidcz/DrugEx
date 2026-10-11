"""Multi-Objective Reinforcement Learning (MORL) Anti-Cheating Reward Framework.

====================================================================================================
THEORETICAL FOUNDATION: MORL REWARD FORMULATION IN LIGAND-BASED DE NOVO DRUG DESIGN
====================================================================================================

1. THE MULTI-OBJECTIVE DE NOVO OPTIMIZATION PROBLEM
----------------------------------------------------
In Ligand-Based De Novo Drug Design (LBDD) using deep generative models (e.g., Graph-Transformer
in DrugEx), the generative policy \\pi_\\theta(\\mathbf{x}) generates molecular graphs \\mathbf{x}
guided by a policy gradient objective:

    \\nabla_\\theta \\mathcal{J}(\\theta) = \\mathbb{E}_{\\mathbf{x} \\sim \\pi_\\theta} \\left[
        \\nabla_\\theta \\log \\pi_\\theta(\\mathbf{x}) \\cdot R(\\mathbf{x})
    \\right]

where R(\\mathbf{x}) \\in [0.0, 1.0] (or [0.0, 2.0]) is a scalarized reward synthesized from M
distinct objective evaluations \\mathbf{S}(\\mathbf{x}) = [S_1(\\mathbf{x}), \\dots, S_M(\\mathbf{x})]^T.

In our authentic 3D-LBDD framework, the objective vector typically encompasses:
  - S_{3D}(\\mathbf{x}): Primary 3D Bioactive Template Overlap (RDKit ROCS Gaussian shape +
    OpenEye Mills-Dean color overlap against a 3D bioactive reference such as Epigallocatechin
    gallate, EGCG).
  - S_{prop, j}(\\mathbf{x}): Secondary Physicochemical Descriptors calibrated to the bioactive
    template's dynamic envelope (Heavy Atom Count, CLogP, TPSA, QED).
  - S_{SA}(\\mathbf{x}): Synthesizability Feasibility Floor (Synthetic Accessibility score).


2. THE PARETO CHEATING PATHOLOGY (ALIGNMENT EXPLOITATION)
---------------------------------------------------------
Why does standard linear additive scalarization fail in 3D molecular design?
Under a standard weighted sum:

    R_{\\text{linear}}(\\mathbf{x}) = \\sum_{j=1}^M w_j S_j(\\mathbf{x}), \\quad \\sum_j w_j = 1.0

The optimization landscape between objectives is highly asymmetric:
  a) S_{3D}(\\mathbf{x}) is high-dimensional, non-convex, rugged, and computationally demanding.
     Generating a conformer ensemble that overlays extended 3D pharmacophores is a difficult task
     with sparse rewards in early exploration.
  b) Secondary 1D/2D physicochemical descriptors (MW, LogP, TPSA) form smooth, easily satisfied
     convex manifolds. Trivial small fragments (e.g., simple benzenes, phenols, or alkyl chains)
     effortlessly achieve near-perfect scores S_{prop, j} \\approx 1.0.

Because the policy gradient optimizer follows the path of steepest ascent, it discovers an
"adversarial shortcut" known as PARETO CHEATING:
  - The model generates trivial fragments with S_{3D}(\\mathbf{x}) = 0.0 and S_{prop}(\\mathbf{x}) = 1.0.
  - If w_{3D} = 0.50 and \\sum w_{props} = 0.50, this cheater candidate receives:
        R_{\\text{linear}} = 0.50 \\times 0.0 + 0.50 \\times 1.0 = 0.50
  - In early RL epochs, genuine candidates with modest 3D overlap (e.g., S_{3D} = 0.20) and moderate
    properties (S_{prop} = 0.50) receive only:
        R = 0.50 \\times 0.20 + 0.50 \\times 0.50 = 0.35
  - The optimizer concludes that generating 3D-inactive fragments yields higher reward than striving
    for 3D shape complementarity!
  - Result: Complete policy collapse into biologically inert, 3D-degenerate fragments that
    masquerade as optimal solutions under aggregate linear metrics.


3. THE ZERO-GRADIENT TRAP (NAIVE HARD THRESHOLDING COLLAPSE)
------------------------------------------------------------
A common naive reaction to Pareto cheating is imposing a hard step filter (Pareto gating):

    R_{\\text{hard}}(\\mathbf{x}) = \\begin{cases}
        \\sum_j w_j S_j(\\mathbf{x}), & \\text{if } S_{3D}(\\mathbf{x}) \\ge \\tau_{3D} \\\\
        0.0, & \\text{if } S_{3D}(\\mathbf{x}) < \\tau_{3D}
    \\end{cases}

If the threshold \\tau_{3D} is set to a biologically meaningful value (e.g., \\tau_{3D} = 0.50 or 0.60):
  - Pre-trained generative priors sample molecules from an unbiased chemical distribution.
  - Early in exploration, virtually 100% of sampled candidates possess S_{3D} < 0.50.
  - Consequently, EVERY sampled candidate in the mini-batch receives R(\\mathbf{x}) = 0.0.
  - The expected policy gradient vanishes completely:
        \\nabla_\\theta \\mathcal{J}(\\theta) = \\frac{1}{N} \\sum_{i=1}^N \\nabla_\\theta \\log \\pi_\\theta(\\mathbf{x}_i) \\cdot 0.0 = \\mathbf{0}
  - The agent receives zero feedback differentiating between a promising near-miss (S_{3D} = 0.45)
    and an unaligned failure (S_{3D} = 0.01).
  - Result: Complete exploration paralysis and policy stagnation.


4. THE SYNTHESIZABILITY PENALTY FALLACY (NATURAL PRODUCTS VS FEASIBILITY FLOORS)
--------------------------------------------------------------------------------
Many molecular RL pipelines stack multiple synthetic accessibility penalties (BRICS fragmentation,
ring strain, raw SAscore) and heavily penalize any candidate with SA > 3.0.
However:
  - Bioactive natural products and clinical leads frequently feature stereocenter complexity and
    polycyclic frameworks (e.g., Epigallocatechin gallate has SA \\approx 4.2).
  - Continuous SA penalties artificially repel generative models away from the natural-product
    chemical space, forcing them into flat, sp2-heavy synthetic trivialities.
  - SOLUTION: Treat synthesizability not as an open-ended optimization target, but as a LOOSE
    FEASIBILITY FLOOR S_{SA\\_floor}(\\mathbf{x}):
        * SA <= 5.5: Full reward (1.0), imposing zero penalty on natural-product complexity.
        * 5.5 < SA < 6.0: Linear transition zone.
        * SA >= 6.0: Hard zero reward (0.0), filtering out synthetically intractable monstrosities.


====================================================================================================
THREE MATHEMATICAL REWARD PARADIGMS IMPLEMENTED IN THIS MODULE
====================================================================================================

1. AntiCollapseSoftGateReward (Thesis Primary Scheme):
   Implements continuous soft-gating G(S_{3D}) = min(1.0, max(0.0, S_{3D} / \\tau_{3D})):
       R(\\mathbf{x}) = w_{3D} S_{3D}(\\mathbf{x}) + w_{props} G(S_{3D}(\\mathbf{x})) \\sum_i \\tilde{w}_i S_{prop, i}(\\mathbf{x}) + S_{SA\\_floor}(\\mathbf{x})
   - Eliminates Pareto cheating: When S_{3D} <= 1e-6, R(\\mathbf{x}) = 0.0 strictly.
   - Non-vanishing gradient: For 0 < S_{3D} < \\tau_{3D}, dR / dS_{3D} \\in [0.70, 1.20], providing
     a strong, smooth learning signal from the very first RL epoch.

2. AugmentedChebyshevReward (Operations Research / Tchebycheff Scalarization):
   Grounded in multi-criteria optimization literature (Steuer 1986, Miettinen 1999).
   Minimizes the weighted L_\\infty distance to utopian ideal point z* = (1, 1, ..., 1):
       Cheb(\\mathbf{x}) = \\max_j \\{ w_j (1 - s_j(\\mathbf{x})) \\} + \\rho \\sum_j w_j (1 - s_j(\\mathbf{x}))
   - Penalizes the weakest objective: If S_{3D} = 0, the max deviation is maximized, crushing
     overall reward regardless of secondary property scores.
   - Augmentation parameter \\rho = 0.02 strictly eliminates weakly Pareto-optimal solutions.

3. GeometricMeanReward (Nash Bargaining / Product Scalarization):
   Grounded in Nash bargaining game theory (Nash 1950) and Cobb-Douglas utility formulations:
       R(\\mathbf{x}) = \\exp\\left( \\frac{\\sum_j w_j \\ln s_j(\\mathbf{x})}{\\sum_j w_j} \\right)
   - Multiplicative zero-tolerance: If any single score s_j -> 0, R(\\mathbf{x}) -> 0.
   - Vectorized log-sum-exp with \\epsilon = 1e-4 flooring ensures numerical stability and finite
     gradients without floating-point underflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union


import numpy as np
from drugex.training.interfaces import RewardScheme

__all__: list[str] = [
    "AntiCollapseSoftGateReward",
    "AugmentedChebyshevReward",
    "GeometricMeanReward",
]


# [LEGACY ARCHIVE] AntiCollapseSoftGateReward is archived under _david/legacy/
# Re-exported here for full backward compatibility across test suites and previous scripts.
from _david.legacy.anti_cheating_rewards import AntiCollapseSoftGateReward


class AugmentedChebyshevReward(RewardScheme):
    """Augmented Chebyshev (Tchebycheff) scalarization reward scheme.

    Theoretical Architecture & Mathematical Derivation:
    --------------------------------------------------
    In Multi-Objective Optimization (MOO) literature (Steuer 1986, Miettinen 1999), weighted linear
    summation suffers from two fatal limitations:
      1. Inability to access non-convex regions of the true Pareto frontier.
      2. Vulnerability to dimensional cheating: a high score on one trivial objective completely
         masks severe failure on difficult objectives.

    Chebyshev scalarization overcomes this by minimizing the weighted L_\\infty distance between the
    candidate's score vector \\mathbf{s} and an ideal utopian reference point \\mathbf{z}^* = (1, 1, ..., 1):
        \\min_{\\mathbf{x}} \\max_{j=1}^M \\left\\{ w_j \\cdot (z^*_j - s_j(\\mathbf{x})) \\right\\}

    Because the max operator \\max_j \\{ \\cdot \\} penalizes the candidate according to its WORST single
    dimension, an agent cannot cheat: if S_3D = 0, the deviation (1 - 0) = 1.0 dominates the metric,
    severely penalizing the solution regardless of how optimal the secondary properties are.

    Augmentation Term (\\rho):
    -------------------------
    Pure Chebyshev scalarization produces weak Pareto optimality (an objective may be improved
    without altering the maximum component). To guarantee STRICT Pareto efficiency, an augmentation
    term \\rho \\sum_j d_j is incorporated (typically \\rho = 0.02):
        \\text{Cheb}(\\mathbf{x}) = \\max_{j=1}^M \\left\\{ w_j \\cdot (1.0 - s_j(\\mathbf{x})) \\right\\}
                                  + \\rho \\sum_{j=1}^M w_j \\cdot (1.0 - s_j(\\mathbf{x}))

    Reward Inversion and Normalization:
    -----------------------------------
    Because reinforcement learning algorithms maximize rewards, the Chebyshev distance is inverted
    and scaled into [0.0, 1.0]:
        R(\\mathbf{x}) = 1.0 - \\frac{\\text{Cheb}(\\mathbf{x})}{\\text{Cheb}_{\\max}}
    where \\text{Cheb}_{\\max} = \\max_j(w_j) + \\rho \\sum_j w_j.
    """

    def __init__(
        self, weights: Sequence[float] | None = None, rho: float = 0.02
    ) -> None:
        """Initialize AugmentedChebyshevReward with optional objective weights and rho.

        Args:
            weights: Optional per-objective importance weights. Defaults to equal weighting.
            rho: Augmentation coefficient ensuring strict Pareto optimality (default 0.02).
        """
        super().__init__()
        self.weights: np.ndarray | None = (
            np.array(weights, dtype=np.float64) if weights is not None else None
        )
        self.rho: float = float(rho)

    def __call__(
        self,
        smiles: Union[Sequence[str], np.ndarray],
        scores: Optional[np.ndarray] = None,
        thresholds: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """Calculate scalarized anti-cheating rewards via Augmented Chebyshev scalarization.

        Args:
            smiles: Sequence of SMILES strings or score matrix if called directly.
            scores: Objective evaluation score matrix of shape (N, M).
            thresholds: Desirability thresholds (optional).

        Returns:
            Column vector of calculated rewards in [0.0, 1.0] of shape (N, 1).
        """
        # Standalone signature support: scheme(scores)
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            scores_arr = np.asarray(smiles, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = (
                scores_arr.reshape(1, -1)
                if len(scores_arr) > 0
                else scores_arr.reshape(0, 1)
            )

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        # ? [NumPy] Clean NaN/Inf, clip scores into [0.0, 1.0], and ensure 2D array shape.
        scores_raw = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)
        scores_clean = np.clip(scores_raw, 0.0, 1.0)
        n_objs = scores_clean.shape[1]

        if (
            self.weights is None
            or len(self.weights) != n_objs
            or np.sum(self.weights) == 0
        ):
            w = np.ones(n_objs, dtype=np.float64) / float(n_objs)
        else:
            w = self.weights / np.sum(self.weights)

        # Deviations from ideal point z* = 1.0: d_ij = w_j * (1.0 - s_ij)
        deviations = w * (1.0 - scores_clean)  # Shape (N, M)

        # Max deviation per candidate: penalizes the weakest dimension
        max_dev = np.max(deviations, axis=1)  # Shape (N,)

        # Sum deviation: augmentation term ensuring strict Pareto efficiency
        sum_dev = np.sum(deviations, axis=1)  # Shape (N,)

        raw_cheb = max_dev + self.rho * sum_dev  # Shape (N,)
        max_possible = np.max(w) + self.rho * np.sum(w)

        # Invert distance so higher scores yield higher rewards in [0.0, 1.0]
        rewards = (
            1.0 - (raw_cheb / max_possible) if max_possible > 0 else 1.0 - raw_cheb
        )
        rewards = np.clip(rewards, 0.0, 1.0)

        return rewards.reshape(-1, 1)


class GeometricMeanReward(RewardScheme):
    """Weighted Geometric Mean scalarization reward scheme.

    Theoretical Architecture & Mathematical Derivation:
    --------------------------------------------------
    The Geometric Mean scalarization is mathematically equivalent to the Nash Bargaining Solution
    (Nash 1950) and Cobb-Douglas utility formulations. In contrast to linear summation, geometric
    aggregation enforces strict multiplicative coupling between all objectives:

        R(\\mathbf{x}) = \\left( \\prod_{j=1}^M s_j(\\mathbf{x})^{w_j} \\right)^{\\frac{1}{\\sum_j w_j}}
                       = \\exp\\left( \\frac{\\sum_{j=1}^M w_j \\ln s_j(\\mathbf{x})}{\\sum_j w_j} \\right)

    Anti-Cheating Property:
    -----------------------
    If any single objective drops to zero (e.g., S_3D = 0.0):
        \\lim_{s_j \\to 0^+} \\ln s_j = -\\infty  ==>  R(\\mathbf{x}) \\to 0.0
    The entire reward collapses to zero. It is mathematically impossible for an agent to compensate
    for an inactive binding/shape score by exaggerating secondary physicochemical properties.

    Numerical Stability & Gradient Dynamics (\\epsilon-Flooring):
    -----------------------------------------------------------
    Computing \\ln(0.0) yields -\\inf, which induces NaN gradients and numerical underflow in PyTorch
    backward passes. To maintain well-behaved gradient flow while preserving zero-tolerance
    penalization:
      1. Scores are floored at \\epsilon = 1e-4:
             \\tilde{s}_j = \\max(\\epsilon, s_j)
      2. The log-sum-exp is evaluated in stable floating-point space:
             \\ln R(\\mathbf{x}) = \\sum_j \\tilde{w}_j \\ln \\tilde{s}_j
      3. An objective with s_j = 0.0 receives \\ln(10^{-4}) \\approx -9.21, driving the overall
         geometric reward exponentially toward zero without inducing NaN numerical failures.
    """

    def __init__(
        self, weights: Sequence[float] | None = None, eps: float = 1e-4
    ) -> None:
        """Initialize GeometricMeanReward with optional objective weights and epsilon floor.

        Args:
            weights: Optional per-objective importance weights. Defaults to equal weighting.
            eps: Numerical floor preventing log(0) singularities and underflow (default 1e-4).
        """
        super().__init__()
        self.weights: np.ndarray | None = (
            np.array(weights, dtype=np.float64) if weights is not None else None
        )
        self.eps: float = float(eps)

    def __call__(
        self,
        smiles: Union[Sequence[str], np.ndarray],
        scores: Optional[np.ndarray] = None,
        thresholds: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """Calculate geometric mean reward across all objectives.

        Args:
            smiles: Sequence of SMILES strings or score matrix if called directly.
            scores: Objective evaluation score matrix of shape (N, M).
            thresholds: Desirability thresholds (optional).

        Returns:
            Column vector of calculated geometric rewards of shape (N, 1).
        """
        # Standalone signature support: scheme(scores)
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            scores_arr = np.asarray(smiles, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = (
                scores_arr.reshape(1, -1)
                if len(scores_arr) > 0
                else scores_arr.reshape(0, 1)
            )

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        # ? [NumPy] Clean NaN/Inf, floor scores at eps to protect log evaluation from underflow, and ensure 2D shape.
        scores_raw = np.nan_to_num(
            scores_arr, nan=self.eps, posinf=1.0, neginf=self.eps
        )
        scores_clean = np.clip(scores_raw, self.eps, 1.0)
        n_objs = scores_clean.shape[1]

        if (
            self.weights is None
            or len(self.weights) != n_objs
            or np.sum(self.weights) == 0
        ):
            w = np.ones(n_objs, dtype=np.float64) / float(n_objs)
        else:
            w = self.weights / np.sum(self.weights)

        # ? [NumPy] Vectorized log-sum-exp computation for geometric mean.
        log_scores = np.log(scores_clean)
        weighted_log_mean = np.sum(log_scores * w, axis=1) / np.sum(w)
        rewards = np.exp(weighted_log_mean)

        return rewards.reshape(-1, 1)
