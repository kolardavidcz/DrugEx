"""Reference loading, geometry preparation and retained-ensemble selection."""

from __future__ import annotations

import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from _david.code.chemistry import (
    IDENTITY_REJECTED,
    Optimization,
    check_identity,
    optimize_conformers,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE


def select_conformers(records: list[Chem.Mol], cutoff: float | None) -> list[int]:
    """Butina representatives within each stereo identity, without moving coordinates.

    Choose the lowest-energy member when energies are present. Return indices so
    studies can reuse saved geometries and measure pruning independently.
    """
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        identity = Chem.MolToSmiles(Chem.RemoveHs(record), isomericSmiles=True)
        groups.setdefault(identity, []).append(index)
    selected = []
    for indices in groups.values():

        def energy(i):
            return records[i].GetDoubleProp("Energy_kcal_mol")

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
        random_seed: int = 42,
        rmsd_cutoff: float | None = 1.25,
        energy_window_kcal: float = 8.0,
        num_threads: int = 0,
        optimization: Optimization | None = None,
        embedding_rmsd: float | None = 0.35,
        embedding_timeout_seconds: int = 0,
    ) -> None:
        self.num_conformers = num_conformers
        self.max_reference_conformers = max_reference_conformers
        self.random_seed = random_seed
        self.rmsd_cutoff = rmsd_cutoff
        self.energy_window_kcal = energy_window_kcal
        self.num_threads = num_threads
        self.optimization = optimization
        if embedding_rmsd is not None and embedding_rmsd < 0:
            raise ValueError("Embedding RMSD must be nonnegative or None")
        self.embedding_rmsd = embedding_rmsd
        if embedding_timeout_seconds < 0:
            raise ValueError("Embedding timeout must be nonnegative")
        self.embedding_timeout_seconds = embedding_timeout_seconds
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
                    check_identity(extra, base)
                    order = extra.GetSubstructMatch(base, useChirality=True)
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

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed if random_seed is None else random_seed
        params.pruneRmsThresh = -1.0 if self.embedding_rmsd is None else self.embedding_rmsd
        params.numThreads = self.num_threads
        params.timeout = self.embedding_timeout_seconds
        attempts = self.num_conformers if num_conformers is None else num_conformers

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
        energies = None
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
        if energies is not None:
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
        selection = select_conformers(records, self.rmsd_cutoff)
        mol_h.SetIntProp("ReferenceClusters", len(selection))
        curated = [
            remaining_cids[i] for i in selection[: self.max_reference_conformers]
        ]

        with Chem.SDWriter(str(target_path)) as writer:
            for i, cid in enumerate(curated):
                check_identity(Chem.Mol(mol_h, confId=cid), mol)
                mol_h.SetProp("_Name", f"Conformer_{i + 1}")
                if energies is not None:
                    mol_h.SetDoubleProp("Energy_kcal_mol", energies[cid])
                    mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
                writer.write(mol_h, confId=cid)

        return target_path
