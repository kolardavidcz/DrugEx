"""Native ROCS components and color-feature eligibility for the conformer study.

Colorless inputs are excluded from self-recovery; the RL score stays native.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from drugex.training.scorers import rocs_rdkit as native


def has_color(mol, *, inject_color=False, color_ff="default"):
    """Inspect the same native color representation that alignment will use."""
    probe = Chem.Mol(mol)
    if (inject_color or color_ff != "default") and not probe.HasProp(
        "PUBCHEM_PHARMACOPHORE_FEATURES"
    ):
        native._annotate_color_features(probe, color_ff=color_ff)
    options = rdShapeAlign.ShapeInputOptions()
    options.useColors = True
    return rdShapeAlign.PrepareConformer(probe, -1, options).sof > 0


def score_pair(candidate, reference):
    """Normalized combo and both components at the same winning alignment."""
    ref, probe = Chem.Mol(reference), Chem.Mol(candidate)
    native._canonicalize_conformer_inplace(ref, -1)
    native._canonicalize_conformer_inplace(probe, -1)
    best = (0.0, 0.0, 0.0)
    for signs in native._AXIS_SIGN_FLIPS:
        aligned = native._flip_conformer(probe, -1, signs)
        try:
            shape, color = rdShapeAlign.AlignMol(
                ref, aligned, useColors=True, opt_param=0.5
            )
        except (RuntimeError, ValueError):
            continue
        combo = (shape + color) / 2
        if combo > best[0]:
            best = (combo, shape, color)
    return best
