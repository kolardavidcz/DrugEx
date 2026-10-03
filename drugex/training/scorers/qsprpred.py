"""
Scorer integration wrapping QSPRpred surrogate predictive models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

# [PACKAGE: numpy] WHY: Array manipulation and score tensor formatting. WHAT: Scientific computing library.
import numpy as np
# [PACKAGE: rdkit] WHY: Molecule conversion and structure validation. WHAT: Cheminformatics toolkit.
from rdkit import Chem

from drugex.logs import logger
from drugex.training.scorers.interfaces import ScoreModifier, Scorer
# [PACKAGE: qsprpred] WHY: Quantitative structure-property relationship surrogate modeling. WHAT: Cheminformatics QSAR library.
from qsprpred.tasks import TargetProperty, TargetTasks

if TYPE_CHECKING:
    from qsprpred.models.model import QSPRModel


class QSPRPredScorer(Scorer):
    """Surrogate predictive model scorer wrapping QSPRpred regression, classification, and applicability domain models."""

    def __init__(
        self,
        model: QSPRModel,
        use_probas: bool = True,
        tasks: Optional[list[str]] = None,
        classes: Optional[list[int]] = None,
        app_domain: Union[bool, str] = False,
        invalids_score: Union[float, list[float]] = 0.0,
        modifier: Optional[Union[ScoreModifier, list[ScoreModifier]]] = None,
        **kwargs: Any
    ) -> None:
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.use_probas = use_probas
        self.tasks = tasks
        self.app_domain = app_domain
        assert self.app_domain in [True, False, 'invalid'], "app_domain must be a boolean or 'invalid'"
        if tasks is not None:
            assert all(task in TargetProperty.getNames(model.targetProperties) for task in tasks), \
                f"Tasks {tasks} not found in model tasks {model.targetProperties}"
        self.classes = classes
        self.invalidsScore = invalids_score
        if isinstance(invalids_score, list):
            assert len(invalids_score) == self.nTasks, \
                "Invalids score list must have the same length as the number of tasks"
            assert all(isinstance(score, float) for score in invalids_score), \
                "Invalids score list must contain only floats"
        else:
            assert isinstance(invalids_score, float), "Invalids score must be a float"
        self.kwargs = kwargs

    def getScores(
        self,
        mols: Sequence[Union[str, Chem.Mol]],
        frags: Optional[Sequence[Union[str, Chem.Mol]]] = None
    ) -> np.ndarray:
        """Calculate surrogate model predictions or probabilities for input molecules."""
        if len(mols) == 0:
            logger.warning("No molecules to score. Returning empty list...")
            return np.array([])

        valid_mols, valid_indices = self._get_valid_molecules(mols)

        if isinstance(self.invalidsScore, list):
            scores = np.tile(self.invalidsScore, (len(mols), 1))
        else:
            scores = np.full((len(mols), self.nTasks), self.invalidsScore)

        if len(valid_mols) == 0:
            logger.warning("No valid molecules to score. Returning all invalidsScore...")
            if self.nTasks == 1:
                return scores.flatten()
            return scores

        valid_scores = self._get_predictions(valid_mols)

        nan_mask = np.isnan(valid_scores.astype(float)).any(axis=1)
        if np.any(nan_mask):
            logger.warning("Some scores are NaN. Dropping these scores...")
            valid_scores = valid_scores[~nan_mask]
            valid_indices = valid_indices[~nan_mask]

        scores[valid_indices, :] = valid_scores

        if scores.shape[1] == 1:
            return scores.flatten()

        return scores

    def _get_valid_molecules(
        self,
        mols: Sequence[Union[str, Chem.Mol]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert input to RDKit Mol objects and filter valid entries."""
        if any(isinstance(mol, str) for mol in mols):
            mols_converted = [Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols]
        else:
            mols_converted = list(mols)

        mols_array = np.array(mols_converted, dtype=object)
        valid_mask = mols_array != None
        valid_mols = mols_array[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        return valid_mols, valid_indices

    def _get_predictions(self, mols: np.ndarray) -> np.ndarray:
        """Query QSPRpred model predictions and applicability domain."""
        include_app = self.app_domain in [True, 'invalid']
        res = self.model.predictMols(
            mols,
            use_probas=self.use_probas,
            use_applicability_domain=self.app_domain if include_app else False,
            **self.kwargs
        )
        if include_app:
            scores, app = res
        else:
            scores = res

        if self.model.task.isRegression() or not self.use_probas:
            scores = self.handle_regression_task(scores)
        else:
            scores = self.handle_classification_task(scores)

        if include_app:
            scores = self.handle_app_domain(scores, app)

        return scores

    def handle_regression_task(self, scores: np.ndarray) -> np.ndarray:
        """Filter selected columns for multi-task regression."""
        if self.model.isMultiTask and self.tasks is not None:
            target_props = TargetProperty.getNames(self.model.targetProperties)
            column_idx = [target_props.index(task) for task in self.tasks]
            scores = scores[:, column_idx]
        return scores

    def handle_classification_task(self, scores: Union[list[np.ndarray], np.ndarray]) -> np.ndarray:
        """Extract positive class or multi-class probability columns."""
        if isinstance(scores, list):
            scores_list = list(scores)
        else:
            scores_list = [scores]

        for i, scores_per_task in enumerate(scores_list):
            if scores_per_task.shape[1] == 2:
                scores_list[i] = scores_per_task[:, 1].reshape(-1, 1)
            elif self.classes is not None:
                scores_list[i] = scores_per_task[:, self.classes]

        if self.model.isMultiTask and self.tasks is not None:
            target_props = TargetProperty.getNames(self.model.targetProperties)
            scores_list = [scores_list[target_props.index(task)] for task in self.tasks]

        return np.concatenate(scores_list, axis=1)

    def handle_app_domain(self, scores: np.ndarray, app: Any) -> np.ndarray:
        """Apply applicability domain gating or column concatenation."""
        if self.app_domain == 'invalid':
            app_arr = np.array(app).flatten().astype(bool)
            scores[~app_arr] = self.invalidsScore
        else:
            scores = np.concatenate([scores, np.array(app).reshape(-1, 1)], axis=1)
        return scores

    def getKey(self) -> Union[str, list[str]]:
        """Return the objective column key(s) generated by this QSPRpred model."""
        base_key = f"QSPRpred_{self.model.name}"
        keys = []
        for target_prop in self.model.targetProperties:
            if self.model.isMultiTask:
                if (self.tasks is not None and target_prop.name in self.tasks) or self.tasks is None:
                    task_key = f"{base_key}_{target_prop.name}"
                else:
                    continue
            else:
                task_key = f"{base_key}"

            if target_prop.task == TargetTasks.MULTICLASS and self.use_probas:
                idx = self.classes if self.classes is not None else range(target_prop.nClasses)
                for i in idx:
                    keys.append(f"{task_key}_{i}")
            else:
                keys.append(task_key)

        if self.app_domain is True:
            keys.append(f"{base_key}_app_domain")

        if len(keys) == 1:
            return keys[0]
        return keys

    @property
    def nTasks(self) -> int:
        """Number of distinct task score outputs."""
        tasks = self.getKey()
        return len(tasks) if isinstance(tasks, list) else 1