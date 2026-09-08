#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

"""Synthetic Accessibility (SA) score calculation based on Ertl and Schuffenhauer (2009)."""

from __future__ import annotations

import gzip
import math
import os.path as op
import pickle
from typing import TYPE_CHECKING, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import RingInfo

_fscores: Dict[int, float] | None = None


def readFragmentScores(name: str = "fpscores") -> None:
    """Read precomputed fragment contribution scores from gzipped pickle file.

    Parameters
    ----------
    name : str, optional
        Base filename or path to the fragment score file (without extension),
        by default 'fpscores' located in the same directory.
    """
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    with gzip.open(f"{name}.pkl.gz", "rb") as f:
        data = pickle.load(f)
    outDict: Dict[int, float] = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol: Chem.Mol, ri: RingInfo | None = None) -> Tuple[int, int]:
    """Calculate the number of bridgehead and spiro atoms in a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule instance.
    ri : RingInfo | None, optional
        Precomputed ring info (unused, kept for API compatibility), by default None.

    Returns
    -------
    Tuple[int, int]
        (nBridgeheads, nSpiro) atom counts.
    """
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m: Chem.Mol) -> float:
    """Calculate the Synthetic Accessibility (SA) score for an RDKit molecule.

    The score ranges from 1.0 (very easy to synthesize) to 10.0 (very difficult to synthesize),
    combining historical fragment frequency contributions with molecular complexity penalties
    (rings, stereocenters, macrocycles, bridgeheads, spiro systems).

    Parameters
    ----------
    m : Chem.Mol
        RDKit molecule instance to score.

    Returns
    -------
    float
        Synthetic accessibility score normalized between 1.0 and 10.0.
    """
    if _fscores is None:
        readFragmentScores()
    assert _fscores is not None

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4.0) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min_val = -4.0
    max_val = 2.5
    sascore = 11.0 - (sascore - min_val + 1.0) / (max_val - min_val) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore
