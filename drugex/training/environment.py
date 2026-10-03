# [PACKAGE: pandas] WHY: Tabular evaluation DataFrame construction. WHAT: Data manipulation library.
import pandas as pd
# [PACKAGE: rdkit] WHY: Chemical SMILES parsing into Mol objects. WHAT: Cheminformatics toolkit.
from rdkit import Chem

from drugex.training.interfaces import Environment
from drugex.training.scorers.smiles import SmilesChecker


class DrugExEnvironment(Environment):
    """Reinforcement learning environment coordinating objective scorers, validity checks, and desirability evaluation."""

    def getScores(self, smiles, frags=None, no_multifrag_smiles=True):

        """Evaluate molecule validity, fragment constraints, and multi-objective desirability across SMILES."""
        mols = [Chem.MolFromSmiles(s) if s != '' else None for s in smiles]
        
        # Check molecule validity and accuracy
        scores = SmilesChecker.checkSmiles(smiles, frags=frags, no_multifrag_smiles=no_multifrag_smiles)

        # Get scores per objective from the scorers
        for scorer in self.scorers:
            scores.loc[:, scorer.getKey()] = scorer(mols)

        # Set all scores to 0.0 for invalid molecules
        scores.loc[scores['Valid'] == 0, self.getScorerKeys()] = 0.0

        # Check if the molecule is desirable
        undesire = (scores[self.getScorerKeys()] < self.thresholds)
        scores['Desired'] = (undesire.sum(axis=1) == 0).astype(int)

        # TODO: Maybe smiles and frags should be added to the dataframe as well?
        
        return scores

    def getUnmodifiedScores(self, smiles):

        """Compute raw, unthresholded multi-objective scores directly from underlying scorers."""
        
        preds = {}
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        for scorer in self.scorers:
            score = scorer.getScores(mols)
            if isinstance(scorer.getKey(), list):
                for i, key in enumerate(scorer.getKey()):
                    preds[key] = score[:, i]
            else:
                preds[scorer.getKey()] = score
        preds = pd.DataFrame(preds)

        return preds