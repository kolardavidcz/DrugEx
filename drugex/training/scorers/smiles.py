"""SMILES validity and fragment accuracy verification scorers."""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from rdkit import Chem


class SmilesChecker:
    """Utility class to validate SMILES strings and verify fragment substructure constraints.

    Provides chemical validity checking through RDKit parsing and validates that
    generated molecules preserve required fragment substructures (e.g. for scaffold-based
    or fragment-based molecule generation).
    """

    @staticmethod
    def checkSmiles(
        smiles: Sequence[str | None],
        frags: Sequence[str | None] | None = None,
        no_multifrag_smiles: bool = True,
    ) -> pd.DataFrame:
        """Check chemical validity of SMILES strings and verify fragment preservation.

        Parameters
        ----------
        smiles : Sequence[str | None]
            List or sequence of SMILES strings to evaluate.
        frags : Sequence[str | None] | None, optional
            Corresponding list of fragment SMILES (dot-separated if multiple fragments)
            that each generated molecule must contain, by default None.
        no_multifrag_smiles : bool, optional
            If True, SMILES strings containing disconnected components ('.') are considered
            invalid and marked with 0, by default True.

        Returns
        -------
        pd.DataFrame
            DataFrame with index matching input SMILES and boolean integer columns:
            - 'Valid': 1 if SMILES is a valid, single-fragment molecule (if no_multifrag_smiles=True),
              0 otherwise.
            - 'Accurate': (if frags is provided) 1 if all fragment substructures are present
              in the molecule, 0 otherwise.
        """
        scores = pd.DataFrame()

        if no_multifrag_smiles:
            # Check if SMILES is not fragmented
            smiles = [smi if smi.count('.') == 0 else None for smi in smiles]

        for j, smile in enumerate(smiles):
            # 1. Check if SMILES can be parsed by rdkit
            try:
                mol = Chem.MolFromSmiles(smile)
                if not smile or smiles == '':
                    mol = None
                scores.loc[j, 'Valid'] = 0 if mol is None else 1
            except:
                scores.loc[j, 'Valid'] = 0

            if frags is not None:
                # 2. Check if SMILES contain given fragments
                try:
                    if mol is None:
                        scores.loc[j, 'Accurate'] = 0
                    else:
                        frag_str = frags[j]
                        if frag_str is None:
                            scores.loc[j, 'Accurate'] = 0
                        else:
                            subs = frag_str.split('.')
                            sub_mols = [Chem.MolFromSmiles(sub) for sub in subs]
                            scores.loc[j, 'Accurate'] = (
                                1 if np.all([mol.HasSubstructMatch(sub) for sub in sub_mols if sub is not None]) else 0
                            )
                except Exception:
                    scores.loc[j, 'Accurate'] = 0

        return scores
