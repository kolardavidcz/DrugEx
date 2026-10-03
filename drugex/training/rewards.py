# [PACKAGE: numpy] WHY: Matrix calculations, array sorting, and vector operations. WHAT: Numerical computing library.
import numpy as np
from abc import ABC, abstractmethod

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