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
        mol = records[index]
        key = _record_identity_key(mol)
        groups.setdefault(key, []).append(index)
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
    groups: dict[str, list[int]] = {}
    for index, mol in enumerate(records):
        identity = _record_identity_key(mol)
        groups.setdefault(identity, []).append(index)

    def energy(index: int) -> float:
        mol = records[index]
        return (
            float(mol.GetDoubleProp("Energy_kcal_mol"))
            if mol.HasProp("Energy_kcal_mol")
            else 0.0
        )

    for group in groups.values():
        group.sort(key=lambda i: (energy(i), i))
    selected: list[int] = []
    nearest: dict[int, float] = {}
    while len(selected) < min(prefix, len(records)):
        for group in groups.values():
            remaining = [i for i in group if i not in selected]
            if not remaining:
                continue
            prior = [i for i in group if i in selected]
            if not prior:
                chosen = remaining[0]
            else:
                chosen = min(remaining, key=lambda i: (nearest[i], energy(i), i))
            selected.append(chosen)
            if len(selected) == min(prefix, len(records)):
                break
            for index in remaining:
                if index != chosen:
                    overlap = max(
                        similarity(records[index], records[chosen])[0],
                        similarity(records[chosen], records[index])[0],
                    )
                    nearest[index] = max(nearest.get(index, 0.0), overlap)
    # Continue in stereo round-robin order, retaining every remaining identity.
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
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        identity = _record_identity_key(record)
        groups.setdefault(identity, []).append(index)
    selected: list[int] = []
    for indices in groups.values():

        def energy(i: int) -> float:
            return float(records[i].GetDoubleProp("Energy_kcal_mol"))

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
                min((indices[i] for i in cluster), key=energy)
                if have_energy
                else indices[cluster[0]]
                for cluster in clusters
            ]
        if have_energy:
            kept.sort(key=energy)
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
        self.num_conformers: int = num_conformers
        self.max_reference_conformers: int = max_reference_conformers
        self.max_reference_isomers: int = max_reference_isomers
        self.random_seed: int = random_seed
        self.stereo_seed: int = stereo_seed
        self.rmsd_cutoff: float | None = rmsd_cutoff
        self.energy_window_kcal: float = energy_window_kcal
        self.num_threads: int = num_threads
        self.optimization: Optimization | None = optimization
        if embedding_rmsd is not None and embedding_rmsd < 0:
            raise ValueError("Embedding RMSD must be nonnegative or None")
        self.embedding_rmsd: float | None = embedding_rmsd
        if embedding_timeout_seconds < 0:
            raise ValueError("Embedding timeout must be nonnegative")
        self.embedding_timeout_seconds: int = embedding_timeout_seconds
        self.selection_mode: Literal["rmsd", "diverse"] = selection_mode
        self.last_optimization_statuses: list[int] = []
        self.last_identity_rejections: list[str] = []

    def standardize(self, smiles: str) -> str:
        """Standardize input chemical structure into canonical SMILES representation."""
        return standardize_smiles(smiles)

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
                mols = [m for m in Chem.SDMolSupplier(str(p), removeHs=False)]
                if not mols or any(m is None for m in mols):
                    raise ValueError(f"Empty or invalid molecule record in SDF: {p}")
                base = mols[0]
                for extra in mols[1:]:
                    if self.max_reference_isomers <= 1:
                        check_identity(extra, base)
                        order = extra.GetSubstructMatch(base, useChirality=True)
                    else:
                        order = extra.GetSubstructMatch(base, useChirality=False)
                    if len(order) != base.GetNumAtoms():
                        raise ValueError("SDF records have incompatible atom mappings")
                    aligned = Chem.RenumberAtoms(extra, order)
                    base.AddConformer(aligned.GetConformer(), assignId=True)
                if base.GetNumConformers() > 0 and base.GetConformer().Is3D():
                    return base, True
                return Chem.RemoveHs(base), False

            text = p.read_text(encoding="utf-8").strip()
            tokens = text.split()
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
        """Minimize conformers using MMFF94 with UFF fallback and return valid conformer energies in kcal/mol."""
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
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.enableSequentialRandomSeeds = False
            params.pruneRmsThresh = (
                -1.0 if self.embedding_rmsd is None else self.embedding_rmsd
            )
            params.numThreads = self.num_threads
            params.timeout = self.embedding_timeout_seconds
            conf_ids = list(
                AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
            )
            if not conf_ids:
                params.useRandomCoords = True
                conf_ids = list(
                    AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
                )
            if not conf_ids:
                continue
            total_embedded += len(conf_ids)
            energies: dict[int, float] | None = None
            if self.optimization is None:
                energies = self.calculate_conformer_energies(
                    mol_h, query_desc=str(query), num_threads=self.num_threads
                )
            elif self.optimization.method != "none":
                raw_records = [Chem.Mol(mol_h, confId=cid) for cid in conf_ids]
                optimized, statuses = optimize_conformers(
                    raw_records, self.optimization
                )
                self.last_optimization_statuses.extend(statuses)
                if not optimized:
                    continue
                mol_h = Chem.Mol(optimized[0])
                mol_h.RemoveAllConformers()
                energies = {}
                for record in optimized:
                    cid = mol_h.AddConformer(record.GetConformer(), assignId=True)
                    energies[cid] = record.GetDoubleProp("Energy_kcal_mol")
            else:
                mol_h.SetProp("ForceField", "none")

            for conf in list(mol_h.GetConformers()):
                cid = conf.GetId()
                try:
                    check_identity(Chem.Mol(mol_h, confId=cid), iso)
                except ValueError as exc:
                    self.last_identity_rejections.append(str(exc))
                    mol_h.RemoveConformer(cid)
                    if energies is not None:
                        energies.pop(cid, None)

            if energies is not None and energies:
                energy_window_applied = True
                total_converged += len(energies)
                min_iso_e = min(energies.values())
                for cid, e in list(energies.items()):
                    if (e - min_iso_e) > self.energy_window_kcal:
                        mol_h.RemoveConformer(cid)
                        del energies[cid]

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

        if self.selection_mode == "diverse":
            order = diverse_order(all_records, prefix=self.max_reference_conformers)
            cluster_count = len(all_records)
        else:
            clustered = select_conformers(all_records, self.rmsd_cutoff)
            cluster_count = len(clustered)
            order = balanced_reference_order(all_records, clustered)

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
        """Embed, minimize, energy-filter and cluster a reference ensemble."""
        target_path = Path(output_path) if output_path else Path("reference_3d.sdf")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_optimization_statuses = []
        self.last_identity_rejections = []

        mol, is_3d = self.load_molecule(query)
        if is_3d:
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

        attempts = self.num_conformers if num_conformers is None else num_conformers
        seed = self.random_seed if random_seed is None else random_seed

        if self.max_reference_isomers > 1:
            return self._generate_multi_isomer_reference_conformers(
                mol, query, target_path, attempts, seed
            )

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.enableSequentialRandomSeeds = False
        params.pruneRmsThresh = (
            -1.0 if self.embedding_rmsd is None else self.embedding_rmsd
        )
        params.numThreads = self.num_threads
        params.timeout = self.embedding_timeout_seconds

        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
        )
        if not conf_ids:
            params.useRandomCoords = True
            conf_ids = list(
                AllChem.EmbedMultipleConfs(mol_h, numConfs=attempts, params=params)
            )
        if not conf_ids:
            raise RuntimeError(f"ETKDGv3 conformer embedding failed for: {query}")

        mol_h.SetIntProp("EmbeddedConformers", len(conf_ids))
        energies: dict[int, float] | None = None
        if self.optimization is None:
            energies = self.calculate_conformer_energies(
                mol_h, query_desc=str(query), num_threads=self.num_threads
            )
        elif self.optimization.method != "none":
            records = [Chem.Mol(mol_h, confId=cid) for cid in conf_ids]
            optimized, self.last_optimization_statuses = optimize_conformers(
                records, self.optimization
            )
            if not optimized:
                raise RuntimeError(
                    f"No reference converged with {self.optimization.name}"
                )
            mol_h = Chem.Mol(optimized[0])
            mol_h.RemoveAllConformers()
            energies = {}
            for record in optimized:
                cid = mol_h.AddConformer(record.GetConformer(), assignId=True)
                energies[cid] = record.GetDoubleProp("Energy_kcal_mol")
            mol_h.SetIntProp("EmbeddedConformers", len(conf_ids))
        else:
            mol_h.SetProp("ForceField", "none")
        for conf in list(mol_h.GetConformers()):
            cid = conf.GetId()
            try:
                check_identity(Chem.Mol(mol_h, confId=cid), mol)
            except ValueError as exc:
                self.last_identity_rejections.append(str(exc))
                mol_h.RemoveConformer(cid)
                if energies is not None:
                    energies.pop(cid, None)
        if not mol_h.GetNumConformers():
            raise RuntimeError("No identity-valid reference conformers survived")
        mol_h.SetIntProp(
            "IdentityRejectedConformers",
            self.last_optimization_statuses.count(IDENTITY_REJECTED)
            + len(self.last_identity_rejections),
        )
        mol_h.SetIntProp(
            "ConvergedConformers", len(energies) if energies is not None else 0
        )
        mol_h.SetBoolProp("EnergyWindowApplied", energies is not None)
        min_e = min(energies.values()) if energies is not None else None
        if energies is not None and min_e is not None:
            for cid, e in list(energies.items()):
                if (e - min_e) > self.energy_window_kcal:
                    mol_h.RemoveConformer(cid)
                    del energies[cid]
        remaining_cids = [c.GetId() for c in mol_h.GetConformers()]
        mol_h.SetIntProp("EnergyWindowConformers", len(remaining_cids))
        records = [Chem.Mol(mol_h, confId=cid) for cid in remaining_cids]
        if energies is not None:
            for record, cid in zip(records, remaining_cids):
                record.SetDoubleProp("Energy_kcal_mol", energies[cid])
        if self.selection_mode == "diverse":
            selection = diverse_order(records, prefix=self.max_reference_conformers)
        else:
            selection = select_conformers(records, self.rmsd_cutoff)
        mol_h.SetIntProp("ReferenceClusters", len(selection))
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
