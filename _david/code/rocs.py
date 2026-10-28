"""Thesis ROCS convention: two empty color sets have similarity one.

Native RDKit returns zero for this undefined empty/empty case. Exactly one
empty set remains zero. Nonempty color scores and DrugEx core stay unchanged.
"""

from __future__ import annotations

import numpy as np
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
    both_empty = not has_color(candidate) and not has_color(reference)
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
        if both_empty:
            color = 1.0
        combo = (shape + color) / 2
        if combo > best[0]:
            best = (combo, shape, color)
    return best


class ThesisROCSScorer(native.RDKitROCSScorer):
    """Reuse native generation/parallel scoring; correct only empty/empty pairs.

    Re-maximize corrected pairs after native scoring. Adding one to an already
    aggregated score would be wrong when a reference group mixes colorless and
    colored structures. Capturing conformers is needed only for colorless refs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._empty_reference_indices = set()
        if self.use_colors and self.score_type != "shape":
            self._empty_reference_indices = {
                i
                for i, mol in enumerate(self.reference_mols)
                if not has_color(
                    mol, inject_color=self.inject_color, color_ff=self.color_ff
                )
            }
        self.conformer_generator.capture_records = bool(self._empty_reference_indices)

    def getScores(self, mols, frags=None):
        scores = super().getScores(mols, frags)
        if not self._empty_reference_indices or not mols:
            return scores
        _, originals = self._deduplicate_smiles(self._convert_to_smiles(mols))
        try:
            for candidate in self.conformer_generator.last_records:
                if has_color(
                    candidate, inject_color=self.inject_color, color_ff=self.color_ff
                ):
                    continue
                unique_id = int(candidate.GetProp("_Name").split("+")[0].split("_")[1])
                for group, indices in enumerate(self.group_to_indices):
                    for index in self._empty_reference_indices.intersection(indices):
                        native_value = native._score_single_reference(
                            candidate,
                            self.reference_mols[index],
                            "TanimotoCombo",
                            self.use_colors,
                            self.inject_color,
                            self.score_variant,
                            self.color_ff,
                        )
                        if native_value <= 0:
                            continue  # A failed overlay is not an empty-color match.
                        value = (
                            1.0 if self.score_type == "color" else native_value + 1.0
                        )
                        for original in originals[unique_id]:
                            scores[original, group] = max(
                                scores[original, group], value
                            )
        finally:
            self.conformer_generator.last_records = []
        return np.asarray(scores)
