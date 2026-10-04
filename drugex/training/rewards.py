# [PACKAGE: numpy] WHY: Matrix calculations, array sorting, and vector operations. WHAT: Numerical computing library.
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

from drugex.logs import logger
from drugex.training.interfaces import RewardScheme
from drugex.utils.fingerprints import get_fingerprint
from drugex.utils import get_Pareto_fronts

# [PACKAGE: rdkit] WHY: Molecular fingerprint generation and bulk Tanimoto similarity calculations. WHAT: Cheminformatics toolkit.
from rdkit import Chem, DataStructs

class ParetoRewardScheme(RewardScheme, ABC):
    """Abstract reward scheme prioritizing Pareto frontiers with secondary diversity ranking.

    Multi-objective scalarization prevents reward collapse on trivial 2D descriptors (unlike naive
    linear scalarization) by ranking candidates across non-dominated Pareto frontiers (or Augmented
    Chebyshev non-linear scalarization) before intra-front crowd diversification.
    """

    def getParetoFronts(self, scores):
        """Extract non-dominated Pareto frontier fronts from multi-objective evaluation matrix."""
        return get_Pareto_fronts(scores)

    @abstractmethod
    def getMoleculeRank(self, fronts, smiles=None, scores=None):
        """Rank molecules within Pareto fronts and return solution indices in sorted order."""
        pass


    def __call__(self, smiles, scores, thresholds) -> np.ndarray:
        """Compute scalar rewards normalized across Pareto fronts and intra-front rankings."""
        fronts = self.getParetoFronts(scores)
        ranks = self.getMoleculeRank(fronts, smiles=smiles, scores=scores)
        rewards = np.zeros((len(smiles), 1))
        rewards[ranks, 0] = np.arange(len(scores)) / len(scores)
        return rewards

class ParetoCrowdingDistance(ParetoRewardScheme):
    """Reward scheme ranking Pareto frontiers using NSGA-II crowding distance to maintain objective diversity."""

    def getMoleculeRank(self, fronts, smiles=None, scores=None):
        """Rank Pareto front molecules via NSGA-II crowding distance from boundary solutions inward."""

        # Rank molecules within each Pareto front based on crowding distance
        ranked_fronts = []
        for front in fronts:
            front_scores = scores[front]
            front_size = front_scores.shape[0]
            crowding_distance = np.zeros(front_size)

            # Calculate crowding distance for each score dimension
            for i in range(front_scores.shape[1]):
                sorted_indices = np.argsort(front_scores[:, i])

                # Set crowding distance of boundary solutions to infinity
                crowding_distance[sorted_indices[0]] = np.inf
                crowding_distance[sorted_indices[-1]] = np.inf

                # Get range of scores for current objective
                score_range = front_scores[:, i].max() - front_scores[:, i].min()
                if score_range == 0:
                    continue

                # Calculate crowding distance for all other solutions
                for j in range(1, front_size - 1):
                    crowding_distance[sorted_indices[j]] += (
                        front_scores[sorted_indices[j + 1], i]
                        - front_scores[sorted_indices[j - 1], i]
                    ) / score_range

            # Sort front indices based on crowding distance
            sorted_indices = np.argsort(-crowding_distance)
            ranked_front = front[sorted_indices] # First element is the best
            ranked_fronts.append(ranked_front)

        # Combine all ranked fronts
        rank = np.concatenate(ranked_fronts)
        rank = rank[::-1] # From worst to best

        return rank

class ParetoTanimotoDistance(ParetoRewardScheme):
    """Reward scheme ranking Pareto frontiers using chemical fingerprint Tanimoto distance to foster structural diversity."""

    def __init__(self, distance_metric : str ='min'):
        """Initialize Tanimoto distance scheme with similarity aggregation metric ('min', 'mean', or 'mutual')."""
        super().__init__()
        self.distance_metric = distance_metric

    @staticmethod
    def calc_fps(mols, fp_type='ECFP6'):
        """Calculate molecular fingerprints for candidate RDKit molecules."""   
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except BaseException:
                fps.append(None)
        return fps

    def getFPs(self, smiles):
        """Convert SMILES strings to RDKit molecules and generate their fingerprints."""
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        return self.calc_fps(mols)

    def _min_mean_similarity_ranking(self, fronts, fps):

        rank = []
        for i, front in enumerate(fronts):
            front_fps = [fps[f] for f in front]
            if len(front) > 2 and None not in front_fps:
                # find the min/average tanimoto distance for each molecule to all other molecules in the front
                dist = np.array(
                    [self.func(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, list(np.delete(front_fps, idx)))))
                     for idx, fp in enumerate(front_fps)])
                # sort front (molecule with the lowest min/average distance to the others is first)
                fronts[i] = front[dist.argsort()] 
            elif None in front_fps:
                logger.warning("Invalid molecule in front. Front not ranked.")
            rank.extend(fronts[i].tolist())
        
        # from worst to best (most similar in worst front to most dissimilar in best front)
        return rank 

    def _mutual_similarity_ranking(self, fronts, fps):
        """Emulate crowding distance using mutual fingerprint Tanimoto similarities within fronts."""

        rank = []
        for i, front in enumerate(fronts):
            fp = [fps[f] for f in front]
            if len(front) > 2 and None not in fp:
                dist = np.zeros(len(front)) # array of cumulative crowded similarity distance scores for each molecule in the front -> the higher the score, the more diverse the 'crowd' around the molecule -> the better the reward in the end
                for j in range(len(front)): # for each molecule in the front
                    tanimoto = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[j], fp)) # calculate tanimoto distance to all other molecules in the front
                    order = tanimoto.argsort() # get the order from lowest to highest tanimoto distance (the molecule j itself is at index 0)
                    dist[order[0]] += 0 # the molecule itself is scored 0 because we want a diverse set and this is the baseline lowest score
                    dist[order[-1]] += 10 ** 4 # ensure the farthest molecule always gets the highest score
                    for k in range(1, len(order) - 1): # for all other molecules in the front
                        # save the sum of the differences in distances between molecule j and the other molecules in the front
                        # in other words: measure how the other molecules 'crowd' together around molecule j
                        dist[order[k]] += tanimoto[order[k + 1]] - tanimoto[order[k - 1]]
                fronts[i] = front[dist.argsort()]
            rank.extend(fronts[i].tolist())
        
        return rank


    def getMoleculeRank(self, fronts, smiles=None, scores=None):
        """Rank molecules across reversed Pareto fronts using fingerprint Tanimoto distance."""

        fronts  = fronts[::-1] # From worst to best
        fps = self.getFPs(smiles)

        if self.distance_metric == 'mean':
            self.func = np.mean
            return self._min_mean_similarity_ranking(fronts, fps)
        elif self.distance_metric == 'min':
            self.func = np.min
            return self._min_mean_similarity_ranking(fronts, fps)
        if self.distance_metric == 'mutual':
            return self._mutual_similarity_ranking(fronts, fps)



class WeightedSum(RewardScheme):
    """Reward scheme combining multi-objective scores via dynamic failure-rate weighting relative to thresholds.

    Calculates adaptive failure-ratio weights so objectives that frequently fall below user thresholds
    receive higher gradient pressure, preventing the agent from collapsing onto trivial 2D descriptors.
    """

    def __call__(self, smiles, scores, thresholds):
        """Compute weighted sum scalar rewards using threshold failure-ratio weights."""
        
        weight = ((scores < thresholds).mean(axis=0, keepdims=True) + 0.01) / \
            ((scores >= thresholds).mean(axis=0, keepdims=True) + 0.01)
        weight = weight / weight.sum()
        return scores.dot(weight.T)


class AntiCollapseSoftGateReward(RewardScheme):
    """Continuous soft-gated multi-objective reward scheme eliminating Pareto cheating.

    Mathematically guarantees that candidates with S_3D = 0 receive strictly 0.0 reward
    while maintaining non-vanishing positive learning gradients (in [0.70, 1.20]) in early RL:
        R(x) = w_3D * S_3D(x) + w_props * G(S_3D(x)) * sum_i(w_tilde_i * S_prop,i(x)) + S_SA_floor(x)
    where G(S_3D) = min(1.0, S_3D / tau_3D), w_3D = 0.70, w_props = 0.30, tau_3D = 0.60.
    """

    def __init__(
        self,
        rocs_col_idx: int = 0,
        sa_col_idx: Optional[int] = None,
        property_weights: Optional[Sequence[float]] = None,
        w_3d: float = 0.70,
        w_props: float = 0.30,
        tau_3d: float = 0.60,
        normalize_rocs: bool = False,
        sa_floor_val: float = 5.5,
        sa_ceil_val: float = 6.0,
    ) -> None:
        """Initialize soft-gated reward scheme with objective weights and gating threshold.

        Parameters:
            rocs_col_idx: Column index of 3D ROCS alignment score in score matrix (default 0).
            sa_col_idx: Column index of synthesizability score (if present in score matrix).
            property_weights: Optional weights for secondary physicochemical property objectives.
            w_3d: Weight coefficient for primary 3D shape/color alignment objective (default 0.70).
            w_props: Weight coefficient for gated secondary property objectives (default 0.30).
            tau_3d: Soft-gating saturation threshold for 3D alignment score (default 0.60).
            normalize_rocs: Whether to divide raw TanimotoCombo scores in [0, 2] by 2.0 (default False).
            sa_floor_val: Upper bound of unpenalized synthesizability complexity (default 5.5).
            sa_ceil_val: Lower bound of penalized synthesizability inaccessibility (default 6.0).
        """
        super().__init__()
        self.rocs_col_idx: int = int(rocs_col_idx)
        self.sa_col_idx: Optional[int] = int(sa_col_idx) if sa_col_idx is not None else None
        self.w_3d: float = float(w_3d)
        self.w_props: float = float(w_props)
        self.tau_3d: float = float(tau_3d)
        self.normalize_rocs: bool = bool(normalize_rocs)
        self.sa_floor_val: float = float(sa_floor_val)
        self.sa_ceil_val: float = float(sa_ceil_val)
        self.property_weights: Optional[np.ndarray] = (
            np.asarray(property_weights, dtype=np.float64) if property_weights is not None else None
        )

    def gate(self, s_3d: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute continuous soft-gate multiplier G(S_3D) = min(1.0, max(0.0, S_3D / tau_3D))."""
        # [PACKAGE: numpy] WHY: Elementwise minimum/maximum vector clamping. WHAT: Numerical computing library. HOW: np.minimum, np.maximum.
        s_arr = np.asarray(s_3d, dtype=np.float64)
        g_val = np.minimum(1.0, np.maximum(0.0, s_arr / self.tau_3d))
        return float(g_val) if np.isscalar(s_3d) else g_val

    def sa_floor(self, sa_raw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute single synthesizability feasibility floor S_SA_floor(x).

        Rewards 1.0 for SA <= 5.5, linearly declines between 5.5 and 6.0, and penalizes
        SA >= 6.0 (yielding 0.0), accommodating natural product complexity without penalty.
        """
        # [PACKAGE: numpy] WHY: Piecewise conditional masking for feasibility floor. WHAT: Numerical computing library. HOW: np.where.
        sa_arr = np.asarray(sa_raw, dtype=np.float64)
        if np.all(sa_arr <= 1.0) and np.all(sa_arr >= 0.0):
            return float(sa_arr) if np.isscalar(sa_raw) else sa_arr
        denom = max(1e-9, self.sa_ceil_val - self.sa_floor_val)
        floor_score = np.where(
            sa_arr <= self.sa_floor_val,
            1.0,
            np.where(
                sa_arr >= self.sa_ceil_val,
                0.0,
                (self.sa_ceil_val - sa_arr) / denom,
            ),
        )
        return float(floor_score) if np.isscalar(sa_raw) else floor_score

    def __call__(
        self,
        smiles: Union[Sequence[str], np.ndarray],
        scores: Optional[np.ndarray] = None,
        thresholds: Optional[Union[Sequence[float], np.ndarray]] = None,
        sa_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate scalarized anti-collapse soft-gated rewards for candidate molecules.

        Supports both DrugEx Environment signature (smiles, scores, thresholds) and standalone
        evaluation signature (scores, sa_scores=None).
        """
        # [PACKAGE: numpy] WHY: Array conversions, nan sanitization, and matrix scalarization. WHAT: Numerical computing library. HOW: np.asarray, np.nan_to_num, np.clip.
        if isinstance(smiles, np.ndarray) or (
            not isinstance(smiles, (list, tuple))
            or (len(smiles) > 0 and not isinstance(smiles[0], str))
        ):
            # Standalone call: scheme(scores, sa_scores=...)
            scores_arr = np.asarray(smiles, dtype=np.float64)
            if scores is not None and sa_scores is None:
                if isinstance(scores, (np.ndarray, list, tuple)):
                    sa_scores = np.asarray(scores, dtype=np.float64)
        else:
            scores_arr = (
                np.asarray(scores, dtype=np.float64)
                if scores is not None
                else np.zeros((len(smiles), 0), dtype=np.float64)
            )

        if scores_arr.ndim == 1:
            scores_arr = scores_arr.reshape(1, -1) if len(scores_arr) > 0 else scores_arr.reshape(0, 1)

        n_samples = scores_arr.shape[0]
        if n_samples == 0:
            return np.zeros((0, 1), dtype=np.float64)

        scores_clean = np.nan_to_num(scores_arr, nan=0.0, posinf=1.0, neginf=0.0)

        # 1. Primary 3D ROCS alignment score
        col_3d = min(self.rocs_col_idx, scores_clean.shape[1] - 1)
        s_3d = scores_clean[:, col_3d].copy()
        if self.normalize_rocs:
            s_3d = np.clip(s_3d / 2.0, 0.0, 1.0)
        else:
            s_3d = np.clip(s_3d, 0.0, 1.0)

        # 2. Continuous soft-gate G(S_3D)
        g_3d = self.gate(s_3d)

        # 3. Secondary property objectives
        prop_indices = [
            i for i in range(scores_clean.shape[1])
            if i != col_3d and (self.sa_col_idx is None or i != self.sa_col_idx)
        ]
        if prop_indices:
            props = scores_clean[:, prop_indices]
            n_props = props.shape[1]
            if self.property_weights is not None and len(self.property_weights) == n_props:
                w_norm = self.property_weights / np.sum(self.property_weights)
            else:
                w_norm = np.ones(n_props, dtype=np.float64) / float(n_props)
            props_weighted_sum = np.sum(props * w_norm, axis=1)
        else:
            props_weighted_sum = np.zeros(n_samples, dtype=np.float64)

        # Strict Pareto anti-cheating guard: candidates with S_3D <= 1e-6 receive 0.0
        zero_mask = s_3d <= 1e-6
        base_reward = self.w_3d * s_3d + self.w_props * g_3d * props_weighted_sum
        base_reward[zero_mask] = 0.0

        # 4. Synthesizability Feasibility Floor
        if sa_scores is not None:
            sa_vals = np.asarray(sa_scores, dtype=np.float64).flatten()
            if np.any(sa_vals > 1.0):
                sa_term = self.sa_floor(sa_vals)
            else:
                sa_term = sa_vals
            sa_term_copy = np.asarray(sa_term, dtype=np.float64).copy()
            sa_term_copy[zero_mask] = 0.0
            total_reward = base_reward + sa_term_copy
        elif self.sa_col_idx is not None and self.sa_col_idx < scores_clean.shape[1]:
            sa_vals = scores_clean[:, self.sa_col_idx]
            if np.any(sa_vals > 1.0):
                sa_term = self.sa_floor(sa_vals)
            else:
                sa_term = sa_vals
            sa_term_copy = np.asarray(sa_term, dtype=np.float64).copy()
            sa_term_copy[zero_mask] = 0.0
            total_reward = base_reward + sa_term_copy
        else:
            total_reward = base_reward

        total_reward[zero_mask] = 0.0
        return total_reward.reshape(-1, 1)