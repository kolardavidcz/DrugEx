"""Reference loading, geometry preparation, Butina RMSD pruning, and diversity selection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from pathlib import Path
from typing import Literal

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from _david.code.chemistry import (
    IDENTITY_REJECTED,
    Optimization,
    check_identity,
    enumerate_stereoisomers,
    optimize_conformers,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE
from _david.code.rocs import score_pair


def _record_identity_key(mol: Chem.Mol) -> str:
    """Return the stereoisomer grouping key for a conformer record."""
    if mol.HasProp("EnumeratedIsomerSMILES"):
        return mol.GetProp("EnumeratedIsomerSMILES")
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def _record_energy(mol: Chem.Mol) -> float:
    """Return the stored ``Energy_kcal_mol`` property or ``0.0`` when absent."""
    return (
        float(mol.GetDoubleProp("Energy_kcal_mol"))
        if mol.HasProp("Energy_kcal_mol")
        else 0.0
    )


def _group_indices_by_identity(records: Sequence[Chem.Mol]) -> dict[str, list[int]]:
    """Group record indices by their canonical stereoisomer SMILES identity."""
    groups: dict[str, list[int]] = {}
    for index, mol in enumerate(records):
        groups.setdefault(_record_identity_key(mol), []).append(index)
    return groups


def balanced_reference_order(
    records: Sequence[Chem.Mol],
    indices: Sequence[int],
) -> list[int]:
    """Order selected conformer indices round-robin across stereoisomer identities.

    Preserves the within-isomer priority order (e.g. ascending force-field energy)
    while ensuring that truncating to a small reference cap ``K`` does not starve
    valid stereoisomers whose global minimum energy is slightly above another isomer's.
    """
    groups: dict[str, list[int]] = {}
    for index in indices:
        groups.setdefault(_record_identity_key(records[index]), []).append(index)
    ordered: list[int] = []
    round_index = 0
    while True:
        added = False
        for group in groups.values():
            if round_index < len(group):
                ordered.append(group[round_index])
                added = True
        if not added:
            return ordered
        round_index += 1


def diverse_order(
    records: Sequence[Chem.Mol],
    prefix: int = 12,
    similarity: Callable[[Chem.Mol, Chem.Mol], tuple[float, float, float]] = score_pair,
) -> list[int]:
    """Nested caps, stereo round-robin, energy seeds, and farthest-first shape+color diversity.

    Symmetrizes the two directional alignments by their maximum. The first pose for
    each stereo identity is its lowest-energy survivor; subsequent poses maximize
    dissimilarity to that identity's already-selected poses up to ``prefix``, then
    continue in stereo round-robin order.
    """
    groups = _group_indices_by_identity(records)
    for group in groups.values():
        group.sort(key=lambda i: (_record_energy(records[i]), i))

    selected: list[int] = []
    nearest: dict[int, float] = {}
    target_prefix = min(prefix, len(records))
    while len(selected) < target_prefix:
        for group in groups.values():
            remaining = [i for i in group if i not in selected]
            if not remaining:
                continue
            prior = [i for i in group if i in selected]
            chosen = (
                remaining[0]
                if not prior
                else min(
                    remaining,
                    key=lambda i: (nearest[i], _record_energy(records[i]), i),
                )
            )
            selected.append(chosen)
            if len(selected) == target_prefix:
                break
            for index in remaining:
                if index != chosen:
                    overlap = max(
                        similarity(records[index], records[chosen])[0],
                        similarity(records[chosen], records[index])[0],
                    )
                    nearest[index] = max(nearest.get(index, 0.0), overlap)

    tails = [[i for i in group if i not in selected] for group in groups.values()]
    while any(tails):
        for tail in tails:
            if tail:
                selected.append(tail.pop(0))
    return selected


def select_conformers(
    records: Sequence[Chem.Mol], cutoff: float | None
) -> list[int]:
    """Butina representatives within each stereo identity, without moving coordinates.

    Choose the lowest-energy member when energies are present. Return indices so
    studies can reuse saved geometries and measure pruning independently.
    """
    groups = _group_indices_by_identity(records)
    selected: list[int] = []
    for indices in groups.values():
        have_energy = all(records[i].HasProp("Energy_kcal_mol") for i in indices)
        if cutoff is None or len(indices) < 2:
            kept = indices.copy()
        else:
            heavy = [Chem.RemoveHs(records[i]) for i in indices]
            distances = [
                float(AllChem.GetBestRMS(Chem.Mol(heavy[i]), heavy[j]))
                for i in range(len(indices))
                for j in range(i)
            ]
            clusters = Butina.ClusterData(
                distances, len(indices), distThresh=cutoff, isDistData=True
            )
            kept = [
                min((indices[i] for i in cluster), key=lambda idx: _record_energy(records[idx]))
                if have_energy
                else indices[cluster[0]]
                for cluster in clusters
            ]
        if have_energy:
            kept.sort(key=lambda idx: _record_energy(records[idx]))
        selected.extend(kept)
    return selected


class PrepareMolecule:
    """Standardizes input chemical queries and generates 3D reference conformer ensembles."""

    def __init__(
        self,
        num_conformers: int = 100,
        max_reference_conformers: int = 3,
        max_reference_isomers: int = 1,
        random_seed: int = 42,
        stereo_seed: int = 42,
        rmsd_cutoff: float | None = 1.25,
        energy_window_kcal: float = 8.0,
        num_threads: int = 0,
        optimization: Optimization | None = None,
        embedding_rmsd: float | None = 0.35,
        embedding_timeout_seconds: int = 0,
        selection_mode: Literal["rmsd", "diverse"] = "rmsd",
    ) -> None:
        if max_reference_isomers < 1:
            raise ValueError("max_reference_isomers must be >= 1")
        if selection_mode not in {"rmsd", "diverse"}:
            raise ValueError("selection_mode must be 'rmsd' or 'diverse'")
        if embedding_rmsd is not None and embedding_rmsd < 0:
            raise ValueError("Embedding RMSD must be nonnegative or None")
        if embedding_timeout_seconds < 0:
            raise ValueError("Embedding timeout must be nonnegative")
        self.num_conformers: int = num_conformers
        self.max_reference_conformers: int = max_reference_conformers
        self.max_reference_isomers: int = max_reference_isomers
        self.random_seed: int = random_seed
        self.stereo_seed: int = stereo_seed
        self.rmsd_cutoff: float | None = rmsd_cutoff
        self.energy_window_kcal: float = energy_window_kcal
        self.num_threads: int = num_threads
        self.optimization: Optimization | None = optimization
        self.embedding_rmsd: float | None = embedding_rmsd
        self.embedding_timeout_seconds: int = embedding_timeout_seconds
        self.selection_mode: Literal["rmsd", "diverse"] = selection_mode
        self.last_optimization_statuses: list[int] = []
        self.last_identity_rejections: list[str] = []

    def standardize(self, smiles: str) -> str:
        """Standardize input chemical structure into canonical isomeric SMILES representation."""
        return standardize_smiles(smiles)

    def _load_sdf_records(self, path: Path) -> tuple[Chem.Mol, bool]:
        """Load and align multi-record SDF conformers onto a single base RDKit Mol."""
        mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False)]
        if not mols or any(m is None for m in mols):
            raise ValueError(f"Empty or invalid molecule record in SDF: {path}")
        base = mols[0]
        use_chirality = self.max_reference_isomers <= 1
        for extra in mols[1:]:
            if use_chirality:
                check_identity(extra, base)
            order = extra.GetSubstructMatch(base, useChirality=use_chirality)
            if len(order) != base.GetNumAtoms():
                raise ValueError("SDF records have incompatible atom mappings")
            aligned = Chem.RenumberAtoms(extra, order)
            base.AddConformer(aligned.GetConformer(), assignId=True)
        if base.GetNumConformers() > 0 and base.GetConformer().Is3D():
            return base, True
        return Chem.RemoveHs(base), False

    def load_molecule(self, query: str | Path | Chem.Mol) -> tuple[Chem.Mol, bool]:
        """Load and standardize molecule across RDKit Mols, SMILES strings, and structure files."""
        if isinstance(query, Chem.Mol):
            if query.GetNumConformers() > 0 and query.GetConformer().Is3D():
                return query, True
            if query.GetNumAtoms() > query.GetNumHeavyAtoms():
                return Chem.RemoveHs(query), False
            return query, False

        p = (
            Path(query)
            if isinstance(query, (Path, str))
            and (
                isinstance(query, Path)
                or query.endswith((".sdf", ".sd", ".txt", ".smi"))
            )
            else None
        )
        if p and p.is_file():
            if p.suffix.lower() in {".sdf", ".sd"}:
                return self._load_sdf_records(p)
            tokens = p.read_text(encoding="utf-8").strip().split()
            if not tokens:
                raise ValueError(f"Empty molecule file: {p}")
            query = tokens[0]

        if isinstance(query, str):
            std_smi = self.standardize(query)
            mol = Chem.MolFromSmiles(std_smi)
            if mol is None:
                raise ValueError(f"Could not parse molecule query: {query!r}")
            return Chem.RemoveHs(mol), False

        raise ValueError(f"Could not parse molecule query: {query!r}")

    @classmethod
    def calculate_conformer_energies(
        cls,
        mol_h: Chem.Mol,
        query_desc: str = "",
        num_threads: int = 0,
    ) -> dict[int, float]:
        """Minimize conformers in-place using batch MMFF94 with UFF fallback and return valid energies.

        Note: Unlike ``chemistry.optimize_conformers`` (which enforces strict single-force-field
        physics without UFF fallback and round-trips each conformer through SDF), this legacy
        method runs when ``optimization=None`` to provide fast multithreaded C++ batch minimization
        with automatic UFF fallback.
        """
        results = []
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=num_threads, mmffVariant="MMFF94"
            )
            mol_h.SetProp("ForceField", "MMFF94")
            if results and all(status == -1 for status, _ in results):
                results = []

        if not results and AllChem.UFFHasAllMoleculeParams(mol_h):
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol_h, maxIters=1000, numThreads=num_threads
            )
            mol_h.SetProp("ForceField", "UFF")

        if not results:
            raise RuntimeError(f"Force field optimization failed for: {query_desc}")

        energies: dict[int, float] = {}
        for conf, (status, e) in zip(mol_h.GetConformers(), results):
            cid = conf.GetId()
            e_val = float(e)
            if status == 0 and math.isfinite(e_val):
                energies[cid] = e_val
            else:
                mol_h.RemoveConformer(cid)

        if not energies:
            raise RuntimeError(
                f"Force field optimization produced zero valid conformers for: {query_desc}"
            )
        return energies

    def _build_etkdg_params(self, seed: int) -> AllChem.EmbedParameters:
        """Create configured ETKDGv3 embedding parameters for reference generation."""
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.enableSequentialRandomSeeds = False
        params.pruneRmsThresh = (
            -1.0 if self.embedding_rmsd is None else self.embedding_rmsd
        )
        params.numThreads = self.num_threads
        params.timeout = self.embedding_timeout_seconds
        return params

    def _embed_conformers_with_fallback(
        self, mol_h: Chem.Mol, attempts: int, seed: int
    ) -> list[int]:
        """Embed conformers with ETKDGv3, falling back to ``useRandomCoords=True`` if needed."""
        params = self._build_etkdg_params(seed)
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
        )
        if not conf_ids:
            params.useRandomCoords = True
            conf_ids = list(
                AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
            )
        return conf_ids

    def _minimize_embedded_conformers(
        self,
        mol_h: Chem.Mol,
        conf_ids: Sequence[int],
        query_desc: str,
        *,
        strict: bool,
    ) -> tuple[Chem.Mol, dict[int, float] | None]:
        """Apply the configured force-field policy to ``mol_h`` and return ``(mol_h, energies)``."""
        if self.optimization is None:
            energies = self.calculate_conformer_energies(
                mol_h, query_desc=query_desc, num_threads=self.num_threads
            )
            return mol_h, energies

        if self.optimization.method == "none":
            mol_h.SetProp("ForceField", "none")
            return mol_h, None

        raw_records = [Chem.Mol(mol_h, confId=cid) for cid in conf_ids]
        optimized, statuses = optimize_conformers(raw_records, self.optimization)
        self.last_optimization_statuses.extend(statuses)
        if not optimized:
            if strict:
                raise RuntimeError(
                    f"No reference converged with {self.optimization.name}"
                )
            mol_h.RemoveAllConformers()
            return mol_h, {}

        rebuilt = Chem.Mol(optimized[0])
        rebuilt.RemoveAllConformers()
        energies = {}
        for record in optimized:
            cid = rebuilt.AddConformer(record.GetConformer(), assignId=True)
            energies[cid] = record.GetDoubleProp("Energy_kcal_mol")
        rebuilt.SetIntProp("EmbeddedConformers", len(conf_ids))
        return rebuilt, energies

    def _filter_identity_and_energy_window(
        self,
        mol_h: Chem.Mol,
        expected_mol: Chem.Mol,
        energies: dict[int, float] | None,
    ) -> tuple[int, float | None]:
        """Remove conformers that fail 3D stereochemical identity or exceed the energy window."""
        for conf in list(mol_h.GetConformers()):
            cid = conf.GetId()
            try:
                check_identity(Chem.Mol(mol_h, confId=cid), expected_mol)
            except ValueError as exc:
                self.last_identity_rejections.append(str(exc))
                mol_h.RemoveConformer(cid)
                if energies is not None:
                    energies.pop(cid, None)

        if energies is None or not energies:
            return 0, None

        converged_count = len(energies)
        min_e = min(energies.values())
        for cid, e_val in list(energies.items()):
            if (e_val - min_e) > self.energy_window_kcal:
                mol_h.RemoveConformer(cid)
                del energies[cid]
        return converged_count, min_e

    def _select_reference_order(
        self, records: Sequence[Chem.Mol], *, balance_isomers: bool
    ) -> tuple[list[int], int]:
        """Select and order reference conformer indices via ``diverse`` or ``rmsd`` strategy."""
        if self.selection_mode == "diverse":
            order = diverse_order(records, prefix=self.max_reference_conformers)
            return order, len(records)
        clustered = select_conformers(records, self.rmsd_cutoff)
        order = (
            balanced_reference_order(records, clustered)
            if balance_isomers
            else clustered
        )
        return order, len(clustered)

    def _write_preloaded_3d_sdf(self, mol: Chem.Mol, target_path: Path) -> Path:
        """Validate and write pre-existing 3D conformers directly to ``target_path``."""
        written = 0
        with Chem.SDWriter(str(target_path)) as writer:
            for conf in mol.GetConformers():
                try:
                    check_identity(Chem.Mol(mol, confId=conf.GetId()), mol)
                except ValueError as exc:
                    self.last_identity_rejections.append(str(exc))
                    continue
                writer.write(mol, confId=conf.GetId())
                written += 1
        if not written:
            raise RuntimeError("No identity-valid reference conformers survived")
        return target_path

    def _generate_multi_isomer_reference_conformers(
        self,
        mol: Chem.Mol,
        query: str | Path | Chem.Mol,
        target_path: Path,
        attempts: int,
        seed: int,
    ) -> Path:
        """Embed, optimize, and select conformers across multiple enumerated stereoisomers."""
        isomers = list(
            enumerate_stereoisomers(
                mol, limit=self.max_reference_isomers, seed=self.stereo_seed
            )
        )
        all_records: list[Chem.Mol] = []
        total_embedded = 0
        total_converged = 0
        energy_window_applied = False

        for iso in isomers:
            iso_identity = iso.GetProp("EnumeratedIsomerSMILES")
            mol_h = Chem.AddHs(Chem.Mol(iso))
            conf_ids = self._embed_conformers_with_fallback(mol_h, attempts, seed)
            if not conf_ids:
                continue
            total_embedded += len(conf_ids)
            mol_h, energies = self._minimize_embedded_conformers(
                mol_h, conf_ids, str(query), strict=False
            )
            if not mol_h.GetNumConformers():
                continue
            converged, _ = self._filter_identity_and_energy_window(
                mol_h, iso, energies
            )
            if converged > 0:
                energy_window_applied = True
                total_converged += converged
            for conf in mol_h.GetConformers():
                cid = conf.GetId()
                single = Chem.Mol(mol_h, confId=cid)
                single.SetProp("EnumeratedIsomerSMILES", iso_identity)
                if energies is not None and cid in energies:
                    single.SetDoubleProp("Energy_kcal_mol", energies[cid])
                all_records.append(single)

        if not all_records:
            raise RuntimeError(
                f"No identity-valid reference conformers survived for: {query}"
            )

        order, cluster_count = self._select_reference_order(
            all_records, balance_isomers=True
        )
        curated = [all_records[i] for i in order[: self.max_reference_conformers]]
        global_min_e = (
            min(rec.GetDoubleProp("Energy_kcal_mol") for rec in all_records)
            if energy_window_applied
            else None
        )
        identity_rejected_total = self.last_optimization_statuses.count(
            IDENTITY_REJECTED
        ) + len(self.last_identity_rejections)

        with Chem.SDWriter(str(target_path)) as writer:
            for i, rec in enumerate(curated):
                rec.SetProp("_Name", f"Conformer_{i + 1}")
                rec.SetIntProp("EmbeddedConformers", total_embedded)
                rec.SetIntProp("IdentityRejectedConformers", identity_rejected_total)
                rec.SetIntProp("ConvergedConformers", total_converged)
                rec.SetBoolProp("EnergyWindowApplied", energy_window_applied)
                rec.SetIntProp("EnergyWindowConformers", len(all_records))
                rec.SetIntProp("ReferenceClusters", cluster_count)
                if global_min_e is not None and rec.HasProp("Energy_kcal_mol"):
                    e_val = rec.GetDoubleProp("Energy_kcal_mol")
                    rec.SetDoubleProp("DeltaE_kcal_mol", e_val - global_min_e)
                writer.write(rec)
        return target_path

    def generate_reference_conformers(
        self,
        query: str | Path | Chem.Mol = DEFAULT_MOLECULE,
        output_path: Path | str | None = None,
        num_conformers: int | None = None,
        random_seed: int | None = None,
    ) -> Path:
        """Embed, minimize, energy-filter, and select a curated 3D reference ensemble."""
        target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_optimization_statuses = []
        self.last_identity_rejections = []

        mol, is_3d = self.load_molecule(query)
        if is_3d:
            return self._write_preloaded_3d_sdf(mol, target_path)

        attempts = self.num_conformers if num_conformers is None else num_conformers
        seed = self.random_seed if random_seed is None else random_seed
        if self.max_reference_isomers > 1:
            return self._generate_multi_isomer_reference_conformers(
                mol, query, target_path, attempts, seed
            )

        mol_h = Chem.AddHs(mol)
        conf_ids = self._embed_conformers_with_fallback(mol_h, attempts, seed)
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {query}")

        mol_h.SetIntProp("EmbeddedConformers", len(conf_ids))
        mol_h, energies = self._minimize_embedded_conformers(
            mol_h, conf_ids, str(query), strict=True
        )
        converged_count, min_e = self._filter_identity_and_energy_window(
            mol_h, mol, energies
        )
        if not mol_h.GetNumConformers():
            raise RuntimeError("No identity-valid reference conformers survived")

        remaining_cids = [c.GetId() for c in mol_h.GetConformers()]
        records = [Chem.Mol(mol_h, confId=cid) for cid in remaining_cids]
        if energies is not None:
            for record, cid in zip(records, remaining_cids):
                record.SetDoubleProp("Energy_kcal_mol", energies[cid])

        selection, cluster_count = self._select_reference_order(
            records, balance_isomers=False
        )
        mol_h.SetIntProp(
            "IdentityRejectedConformers",
            self.last_optimization_statuses.count(IDENTITY_REJECTED)
            + len(self.last_identity_rejections),
        )
        mol_h.SetIntProp("ConvergedConformers", converged_count)
        mol_h.SetBoolProp("EnergyWindowApplied", energies is not None)
        mol_h.SetIntProp("EnergyWindowConformers", len(remaining_cids))
        mol_h.SetIntProp("ReferenceClusters", cluster_count)

        curated = [
            remaining_cids[i] for i in selection[: self.max_reference_conformers]
        ]
        with Chem.SDWriter(str(target_path)) as writer:
            for i, cid in enumerate(curated):
                check_identity(Chem.Mol(mol_h, confId=cid), mol)
                mol_h.SetProp("_Name", f"Conformer_{i + 1}")
                if energies is not None and min_e is not None:
                    mol_h.SetDoubleProp("Energy_kcal_mol", energies[cid])
                    mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
                writer.write(mol_h, confId=cid)
        return target_path
