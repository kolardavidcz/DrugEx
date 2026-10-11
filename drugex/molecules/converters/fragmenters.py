import re
from itertools import combinations
from typing import List, Tuple, Union


import numpy as np

# [PACKAGE: rdkit] WHY: Rule-based retrosynthetic decomposition (RECAP, BRICS) and substructure query matching. WHAT: Chem, Recap, and BRICS decomposition APIs.
from rdkit import Chem
from rdkit.Chem import Recap, BRICS

from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.converters.standardizers import CleanSMILES


class Fragmenter(CleanSMILES):
    """Decomposes molecules into leaf fragments using RECAP or BRICS and generates fragment-molecule combinations."""

    def __init__(
        self,
        n_frags: int,
        n_combs: int,
        method: str = "recap",
        deep_clean: bool = True,
        max_bonds: int = 75,
        allow_single: bool = False,
    ):
        """Initialize fragmenter with budget thresholds, decomposition method, and filtering criteria."""
        super().__init__(deep_clean)
        self.nFrags = n_frags
        self.nCombs = n_combs
        self.method = method
        self.maxBonds = max_bonds
        self.allowSingle = allow_single
        if self.method not in ("recap", "brics"):
            raise ConversionException(f"Unknown fragmentation method: {self.method}")

    def getFragments(self, mol: Chem.Mol) -> Union[np.array, None]:
        """Decompose an RDKit molecule into unique leaf fragments using the configured method."""
        # break molecule into leaf fragments
        if self.method == "recap":
            frags = np.array(sorted(Recap.RecapDecompose(mol).GetLeaves().keys()))
        else:
            frags = BRICS.BRICSDecompose(mol)
            frags = np.array(sorted({re.sub(r"\[\d+\*\]", "*", f) for f in frags}))

        if len(frags) == 1 and not self.allowSingle:
            logger.warning(f"Only one retrieved fragment for molecule: {Chem.MolToSmiles(mol)}. Skipping...")
            return None

        return frags

    def filterFragments(self, frags: List[str]) -> List[str]:
        """Filter leaf fragments to eliminate sub-fragments and retain the largest components within budget."""
        # replace connection tokens with [H]
        du, hy = Chem.MolFromSmiles("*"), Chem.MolFromSmiles("[H]")
        subs = np.array([Chem.MolFromSmiles(f) for f in frags])
        subs = np.array([Chem.RemoveHs(Chem.ReplaceSubstructs(f, du, hy, replaceAll=True)[0]) for f in subs])
        subs = np.array([m for m in subs if m.GetNumAtoms() > 1])
        # remove fragments that contain other fragments (or are contained in other fragments?)
        match = np.array([[m.HasSubstructMatch(f) for f in subs] for m in subs])
        frags = subs[match.sum(axis=0) == 1]
        # sort the fragments and only keep n_frag largest ones
        frags = sorted(frags, key=lambda x: -x.GetNumAtoms())[: self.nFrags]
        frags = [Chem.MolToSmiles(Chem.RemoveHs(f)) for f in frags]

        return frags

    def __call__(self, smiles: str) -> Union[List[Tuple[str, str]], None]:
        """Generate valid (fragment, smiles) pairs from an input molecule SMILES string."""
        ret_frags = []
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.warning(f"Molecule skipped due to invalid SMILES: {smiles}")
            return None

        if self.maxBonds and mol.GetNumBonds() >= self.maxBonds:
            logger.warning(f"Molecule skipped due to threshold on maximum bond count ({self.maxBonds}): {smiles}")
            return None

        frags = self.getFragments(mol)
        if frags is None:
            return None

        frags = self.filterFragments(frags)

        max_comb = min(self.nCombs, len(frags))
        for ix in range(1, max_comb + 1):
            # combine leaf fragments into larger fragments
            combs = combinations(frags, ix)
            for comb in combs:
                comb_frags = ".".join(comb)
                # remove pair of fragment combinations if longer than original SMILES
                if len(comb_frags) > len(smiles):
                    continue
                # check if substructure is in original molecule
                if mol.HasSubstructMatch(Chem.MolFromSmarts(comb_frags)):
                    ret_frags.append((comb_frags, smiles))

        return ret_frags


class FragmenterWithSelectedFragment(Fragmenter):
    """Fragmenter constraining generated pairs to those containing a specific target substructure fragment."""

    def __init__(
        self,
        fragment: str,
        n_frags: int,
        n_combs: int,
        method: str = "recap",
        deep_clean: bool = True,
        max_bonds: int = 75,
        allow_single: bool = False,
        exclusive: bool = False,
    ):
        """Initialize target-constrained fragmenter with required fragment query and exclusivity constraint."""
        super().__init__(n_frags, n_combs, method, deep_clean, max_bonds, allow_single)
        self.fragment = fragment
        self.exclusive = exclusive

    def __call__(self, smiles: str) -> Union[List[Tuple[str, str]], None]:
        """Generate fragment-molecule pairs requiring matches to the specified target substructure."""
        ret_frags = []
        mol = Chem.MolFromSmiles(smiles)

        if not mol.HasSubstructMatch(Chem.MolFromSmarts(self.fragment)):
            logger.warning(f"Molecule skipped due to missing the `{self.fragment}` fragment: {smiles}")
            return None

        if self.maxBonds and mol.GetNumBonds() >= self.maxBonds:
            logger.warning(f"Molecule skipped due to threshold on maximum bond count ({self.maxBonds}): {smiles}")
            return None

        frags = self.getFragments(mol)
        if frags is None:
            return None

        frags = self.filterFragments(frags)

        if self.exclusive:
            # only return fragments that contain the specified fragment and nothing else
            frags = [f for f in frags if Chem.CanonSmiles(f) == Chem.CanonSmiles(self.fragment)]
            if len(frags) == 0:
                logger.warning(f"Molecule skipped due to missing the `{self.fragment}` fragment: {smiles}")
                return None

        max_comb = min(self.nCombs, len(frags))
        for ix in range(1, max_comb + 1):
            # combine leaf fragments into larger fragments
            combs = combinations(frags, ix)
            for comb in combs:
                comb_frags = ".".join(comb)
                # remove combination that do not contain the selected fragment
                if not Chem.MolFromSmiles(comb_frags).HasSubstructMatch(Chem.MolFromSmarts(self.fragment)):
                    continue
                # remove pair of fragment combinations if longer than original SMILES
                if len(comb_frags) > len(smiles):
                    continue
                # check if substructure is in original molecule
                if mol.HasSubstructMatch(Chem.MolFromSmarts(comb_frags)):
                    ret_frags.append((comb_frags, smiles))

        return ret_frags
