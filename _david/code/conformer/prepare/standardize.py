"""Input standardization and the canonical prepared-molecule contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount

from _david.code.chemistry import (
    check_identity,
    enumerate_stereoisomers,
    prepare_candidate_molecule,
    standardize_smiles,
)
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import EmbedSpec, MoleculeSpec, fingerprint
from _david.code.conformer.core.errors import PreparationError
from _david.code.conformer.core.provenance import utc_now
from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score


@dataclass(frozen=True)
class PreparedMolecule:
    """A chemically validated input plus its enumerated stereoisomers.

    The object is immutable from the caller's perspective.  ``mol`` and
    ``isomers`` are RDKit objects without conformers; embedding is a separate,
    explicitly budgeted step.
    """

    spec: MoleculeSpec
    standard_smiles: str
    canonical_smiles: str
    mol: Chem.Mol
    isomers: tuple[Chem.Mol, ...]
    properties: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    cache_key: str | None = None

    @property
    def identity(self) -> str:
        """Canonical isomeric identity used for caching and de-duplication."""
        return self.canonical_smiles

    @property
    def n_isomers(self) -> int:
        """Number of enumerated stereoisomers retained by the cap."""
        return len(self.isomers)

    @property
    def isomer_smiles(self) -> tuple[str, ...]:
        """Canonical SMILES for each retained stereoisomer."""
        return tuple(
            molecule.GetProp("EnumeratedIsomerSMILES")
            if molecule.HasProp("EnumeratedIsomerSMILES")
            else Chem.MolToSmiles(molecule, isomericSmiles=True)
            for molecule in self.isomers
        )


def molecule_properties(mol: Chem.Mol) -> dict[str, Any]:
    """Return the physicochemical descriptors used in audit tables and figures."""
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    return {
        "heavy_atoms": int(heavy.GetNumHeavyAtoms()),
        "molecular_weight": round(float(Descriptors.MolWt(heavy)), 4),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(heavy)),
        "saturated_rings": int(Lipinski.NumSaturatedRings(heavy)),
        "aromatic_rings": int(Lipinski.NumAromaticRings(heavy)),
        "h_bond_donors": int(Lipinski.NHOHCount(heavy)),
        "h_bond_acceptors": int(Lipinski.NOCount(heavy)),
        "stereo_centers": int(rdMolDescriptors.CalcNumAtomStereoCenters(heavy)),
        "unspecified_stereo_centers": int(
            rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(heavy)
        ),
        "isomer_upper_bound": int(GetStereoisomerCount(heavy)),
        "qed": round(float(QED.qed(heavy)), 6),
        "sa_score": round(float(calculate_sa_score(heavy)), 6),
        "formal_charge": int(Chem.GetFormalCharge(heavy)),
        "fraction_csp3": round(float(rdMolDescriptors.CalcFractionCSP3(heavy)), 6),
    }


def _apply_caps(mol: Chem.Mol, spec: EmbedSpec) -> None:
    """Reject molecules that exceed the explicit preparation caps."""
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    if spec.heavy_atom_cap is not None and heavy_atoms > spec.heavy_atom_cap:
        raise PreparationError(
            f"Heavy-atom count {heavy_atoms} exceeds cap {spec.heavy_atom_cap}"
        )
    rotors = int(Lipinski.NumRotatableBonds(mol))
    if spec.rotatable_bond_cap is not None and rotors > spec.rotatable_bond_cap:
        raise PreparationError(
            f"Rotatable-bond count {rotors} exceeds cap {spec.rotatable_bond_cap}"
        )


def _build_prepared(
    spec: MoleculeSpec,
    standard: str,
    canonical: str,
    isomer_smiles: Iterable[str],
    properties: Mapping[str, Any],
    warnings: tuple[str, ...],
    cache_key: str | None,
) -> PreparedMolecule:
    """Reconstruct a :class:`PreparedMolecule` from cached SMILES.

    Reconstructing from SMILES is intentionally preferred over pickling RDKit
    objects: SMILES are stable across RDKit patch releases, human-readable in
    the cache manifest, and cheap to parse.
    """
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise PreparationError(f"Cached canonical SMILES is unparsable: {canonical!r}")
    isomers: list[Chem.Mol] = []
    for index, smiles in enumerate(isomer_smiles):
        isomer = Chem.MolFromSmiles(smiles)
        if isomer is None:
            raise PreparationError(f"Cached isomer SMILES is unparsable: {smiles!r}")
        isomer.SetProp("EnumeratedIsomerSMILES", smiles)
        isomer.SetIntProp("EnumeratedIsomerIndex", index)
        isomers.append(isomer)
    if not isomers:
        raise PreparationError("Preparation retained zero stereoisomers")
    return PreparedMolecule(
        spec=spec,
        standard_smiles=standard,
        canonical_smiles=canonical,
        mol=mol,
        isomers=tuple(isomers),
        properties=dict(properties),
        warnings=warnings,
        cache_key=cache_key,
    )


def prepare_molecule(
    spec: MoleculeSpec,
    embed: EmbedSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
) -> PreparedMolecule:
    """Standardize, validate, and enumerate one molecular input.

    The function is cacheable at the ``(input SMILES, preparation spec)`` level.
    Chemical transformations are recorded in the returned object so downstream
    tables can disclose exactly which tautomer/protonation state was scored.
    """
    embed = embed or EmbedSpec()
    payload = {
        "input_smiles": spec.smiles,
        "canonicalize_tautomer": embed.canonicalize_tautomer,
        "protonate": embed.protonate,
        "isomer_cap": embed.isomer_cap,
        "stereo_seed": embed.stereo_seed,
        "heavy_atom_cap": embed.heavy_atom_cap,
        "rotatable_bond_cap": embed.rotatable_bond_cap,
    }
    key = cache.key("prepare", payload) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get_json("prepare", key)
        if cached is not None:
            return _build_prepared(
                spec=spec,
                standard=str(cached["standard_smiles"]),
                canonical=str(cached["canonical_smiles"]),
                isomer_smiles=[str(value) for value in cached["isomer_smiles"]],
                properties=dict(cached["properties"]),
                warnings=tuple(str(value) for value in cached.get("warnings", [])),
                cache_key=key,
            )

    warnings: list[str] = []
    try:
        standard = standardize_smiles(spec.smiles)
    except ValueError as exc:
        raise PreparationError(f"Standardization failed for {spec.name!r}: {exc}") from exc

    try:
        mol = prepare_candidate_molecule(
            standard,
            canonicalize_tautomer=embed.canonicalize_tautomer,
            protonate=embed.protonate,
        )
    except (ValueError, RuntimeError) as exc:
        # Chemical-state normalization is a convenience, stereochemistry is a
        # fact about the input.  When the normalized tautomer/protomer would
        # destroy an assigned centre we keep the input state and disclose it,
        # rather than discarding an otherwise scoreable molecule.
        if not (embed.canonicalize_tautomer or embed.protonate):
            raise PreparationError(f"Chemical-state preparation failed: {exc}") from exc
        try:
            mol = prepare_candidate_molecule(
                standard, canonicalize_tautomer=False, protonate=False
            )
        except (ValueError, RuntimeError) as fallback_exc:
            raise PreparationError(
                f"Chemical-state preparation failed: {fallback_exc}"
            ) from fallback_exc
        warnings.append(f"chemical-state-fallback: {exc}")

    _apply_caps(mol, embed)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    if Chem.MolToSmiles(mol, isomericSmiles=False) != Chem.MolToSmiles(
        Chem.MolFromSmiles(standard), isomericSmiles=False
    ):
        warnings.append("tautomer-or-protonation-state-changed")

    try:
        isomers = list(
            enumerate_stereoisomers(mol, limit=embed.isomer_cap, seed=embed.stereo_seed)
        )
    except (ValueError, RuntimeError) as exc:
        raise PreparationError(f"Stereoisomer enumeration failed: {exc}") from exc
    if not isomers:
        raise PreparationError("Stereoisomer enumeration produced no identities")

    for isomer in isomers:
        try:
            check_identity(isomer, mol)
        except ValueError as exc:
            raise PreparationError(f"Enumerated isomer failed identity check: {exc}") from exc

    properties = molecule_properties(mol)
    properties["enumerated_isomers"] = len(isomers)
    properties["prepared_utc"] = utc_now()
    if len(isomers) < properties["isomer_upper_bound"]:
        # Two very different causes: the cap actually bit, or the naive RDKit
        # count ignored symmetry (e.g. a cis/trans ring pair collapsing to one
        # identity).  Only the first is truncation and must be labelled so.
        if len(isomers) >= embed.isomer_cap:
            warnings.append(
                f"isomer-cap-{embed.isomer_cap}-truncated-{properties['isomer_upper_bound']}"
            )
        else:
            warnings.append(
                f"symmetry-collapsed-{properties['isomer_upper_bound']}-to-{len(isomers)}"
            )
    if not any(
        atom.GetAtomicNum() in (7, 8) for atom in Chem.RemoveHs(Chem.Mol(mol)).GetAtoms()
    ):
        warnings.append("no-shape-color-features")

    prepared = PreparedMolecule(
        spec=spec,
        standard_smiles=standard,
        canonical_smiles=canonical,
        mol=mol,
        isomers=tuple(isomers),
        properties=properties,
        warnings=tuple(warnings),
        cache_key=key,
    )
    if cache is not None and key is not None:
        cache.put_json(
            "prepare",
            key,
            {
                "standard_smiles": standard,
                "canonical_smiles": canonical,
                "isomer_smiles": list(prepared.isomer_smiles),
                "properties": properties,
                "warnings": list(warnings),
            },
            metadata={"molecule": spec.name},
        )
    return prepared


def prepare_many(
    specs: Iterable[MoleculeSpec],
    embed: EmbedSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
) -> tuple[list[PreparedMolecule], list[dict[str, str]]]:
    """Prepare a batch, returning successes and per-molecule error records."""
    prepared: list[PreparedMolecule] = []
    errors: list[dict[str, str]] = []
    for spec in specs:
        try:
            prepared.append(prepare_molecule(spec, embed, cache=cache))
        except PreparationError as exc:
            errors.append({"molecule": spec.name, "smiles": spec.smiles, "error": str(exc)})
    return prepared, errors


__all__ = [
    "PreparedMolecule",
    "molecule_properties",
    "prepare_many",
    "prepare_molecule",
]
