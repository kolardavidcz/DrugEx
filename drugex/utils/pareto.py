# [PACKAGE: numpy] WHY: Array operations WHAT: Vectorized indexing and boolean mask operations for non-dominated sorting
import numpy as np
# [PACKAGE: torch] WHY: Tensor compatibility WHAT: PyTorch library in module namespace
import torch

def get_Pareto_fronts(scores):
    """Identify non-dominated Pareto fronts iteratively by index from an (n_points, n_scores) score array."""
    # Initialize
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    all_fronts = []

    # Identify Pareto fronts
    while population_size > 0:
        # Identify the current Pareto front
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                # Strictly j better than i in all scores (i dominated by j) 
                # -> i not in Pareto front
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_front[i] = 0
                    break

        # Add the current Pareto front to the list of all fronts
        current_front_ids = population_ids[pareto_front]
        all_fronts.append(current_front_ids)

        # Remove the current Pareto front from consideration in future iterations
        scores = scores[~pareto_front]
        population_ids = population_ids[~pareto_front]
        population_size = scores.shape[0]

    return all_fronts