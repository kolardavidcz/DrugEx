"""Curated molecule panels that ship with the package.

These are *inputs*, not algorithmic special cases.  EGCG is the thesis target,
but the same preparation and scoring path applies to every molecule in a panel;
the only thing the panel provides is structures and grouping tags.

Tags use the ``group:<name>`` convention so a study can stratify decoys without
knowing any specific molecule name.
"""

from __future__ import annotations

from pathlib import Path

from _david.code.conformer.core.contracts import MoleculeSpec
from _david.code.conformer.core.paths import DAVID_ROOT

#: Canonical EGCG structure used by the thesis (SMILES round-trip differs from
#: the legacy pipeline string but is the same molecule; the registry records
#: both the input and the standardized form).
EGCG_SMILES = (
    "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1)"
    "c1cc(O)c(O)c(O)c1"
)

#: Historical default query string, kept verbatim for continuity with old runs.
EGCG_LEGACY_SMILES = (
    "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)"
    "OC(=O)C4=CC(=C(C(=C4)O)O)O"
)

#: EGCG stress panel: positive controls, active sub-scaffolds, and four decoy
#: regimes that probe the two asymmetric-Tversky failure modes.
EGCG_STRESS_PANEL: tuple[tuple[str, str, str, str], ...] = (
    ("egcg", "self", EGCG_SMILES, "1_Self_Recovery"),
    (
        "epigallocatechin",
        "active_core",
        "Oc1cc(O)c2c(c1)O[C@H](c1cc(O)c(O)c(O)c1)[C@H](O)C2",
        "2_Active_Subscaffold",
    ),
    (
        "epicatechin",
        "active_core",
        "Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@H](O)C2",
        "2_Active_Subscaffold",
    ),
    (
        "myricetin",
        "active_flavonol",
        "O=c1c(O)c(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12",
        "2_Active_Subscaffold",
    ),
    ("gallic_acid", "fragment", "O=C(O)c1cc(O)c(O)c(O)c1", "3_Small_Fragment_Decoy"),
    ("pyrogallol", "fragment", "Oc1cccc(O)c1O", "3_Small_Fragment_Decoy"),
    ("catechol", "fragment", "Oc1ccccc1O", "3_Small_Fragment_Decoy"),
    (
        "ibuprofen",
        "small_drug",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "3_Small_Fragment_Decoy",
    ),
    (
        "acetaminophen",
        "small_drug",
        "CC(=O)Nc1ccc(O)cc1",
        "3_Small_Fragment_Decoy",
    ),
    (
        "verapamil",
        "shape_matched",
        "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "4_Shape_Matched_Steric_Decoy",
    ),
    (
        "glibenclamide",
        "shape_matched",
        "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",
        "4_Shape_Matched_Steric_Decoy",
    ),
    (
        "fexofenadine",
        "shape_matched",
        "CC(C)(C(=O)O)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "4_Shape_Matched_Steric_Decoy",
    ),
    (
        "tricyclohexyl_trimesate",
        "shape_matched",
        "O=C(OCC1CCCCC1)c1cc(C(=O)OCC2CCCCC2)cc(C(=O)OCC3CCCCC3)c1",
        "4_Shape_Matched_Steric_Decoy",
    ),
    (
        "atorvastatin",
        "bloated",
        "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
        "5_Bloated_Monster_Decoy",
    ),
    (
        "lopinavir",
        "bloated",
        "Cc1cccc(C)c1OCC(=O)N[C@@H](Cc1ccccc1)[C@@H](O)C[C@H](Cc1ccccc1)NC(=O)[C@H](C(C)C)N1CCCNC1=O",
        "5_Bloated_Monster_Decoy",
    ),
    (
        "tris_trimethoxybenzoyl_phloroglucinol",
        "bloated",
        "COc1cc(C(=O)Oc2cc(OC(=O)c3cc(OC)c(OC)c(OC)c3)cc(OC(=O)c3cc(OC)c(OC)c(OC)c3)c2)cc(OC)c1OC",
        "5_Bloated_Monster_Decoy",
    ),
)

#: Legacy output root referenced by the migration command. The tree was moved to
#: ``output/conformer/self_similarity`` by the output normalizer; hosts that were not
#: reorganized yet still carry the flat directory, so resolve instead of pinning one.
def _legacy_output_root() -> Path:
    categorized = DAVID_ROOT / "output" / "conformer" / "self_similarity"
    flat = DAVID_ROOT / "output" / "20261028_conformer_self_similarity"
    return categorized if categorized.is_dir() else flat


LEGACY_OUTPUT_ROOT = _legacy_output_root()


def egcg_panel() -> tuple[MoleculeSpec, ...]:
    """Return the EGCG stress panel as immutable molecule specs."""
    return tuple(
        MoleculeSpec(
            name=name,
            smiles=smiles,
            source="egcg_stress_panel",
            tags=(f"group:{category}", role),
        )
        for name, role, smiles, category in EGCG_STRESS_PANEL
    )


__all__ = [
    "EGCG_LEGACY_SMILES",
    "EGCG_SMILES",
    "EGCG_STRESS_PANEL",
    "LEGACY_OUTPUT_ROOT",
    "egcg_panel",
]
