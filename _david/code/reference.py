"""Reference loading, geometry preparation and retained-ensemble selection."""

from __future__ import annotations

import math
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from _david.code.chemistry import (
    Optimization,
    check_identity,
    optimize_conformers,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE


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
    ) -> None:
        self.num_conformers = num_conformers
        self.max_reference_conformers = max_reference_conformers
        self.random_seed = random_seed
        self.rmsd_cutoff = rmsd_cutoff
        self.energy_window_kcal = energy_window_kcal
        self.num_threads = num_threads
        self.optimization = optimization

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

        mol, is_3d = self.load_molecule(query)
        if is_3d:
            with Chem.SDWriter(str(target_path)) as writer:
                for conf in mol.GetConformers():
                    check_identity(Chem.Mol(mol, confId=conf.GetId()), mol)
                    writer.write(mol, confId=conf.GetId())
            return target_path

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed if random_seed is None else random_seed
        params.pruneRmsThresh = 0.35
        params.numThreads = self.num_threads
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
            optimized, _ = optimize_conformers(records, self.optimization)
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
        max_ensemble_size = self.max_reference_conformers
        if self.rmsd_cutoff is None:
            curated = (
                sorted(remaining_cids, key=energies.get)
                if energies is not None
                else remaining_cids
            )
            mol_h.SetIntProp("ReferenceClusters", len(curated))
            curated = curated[:max_ensemble_size]
        elif len(remaining_cids) > 1:
            mol_no_h = Chem.RemoveHs(mol_h)
            n_confs = len(remaining_cids)
            rms_matrix: list[float] = [
                float(
                    AllChem.GetBestRMS(
                        mol_no_h, mol_no_h, remaining_cids[i], remaining_cids[j]
                    )
                )
                for i in range(n_confs)
                for j in range(i)
            ]
            clusters = Butina.ClusterData(
                rms_matrix, n_confs, distThresh=self.rmsd_cutoff, isDistData=True
            )
            mol_h.SetIntProp("ReferenceClusters", len(clusters))
            curated = [
                min([remaining_cids[idx] for idx in cl], key=energies.get)
                if energies is not None
                else remaining_cids[cl[0]]
                for cl in clusters
            ]
            if energies is not None:
                curated.sort(key=energies.get)
            curated = curated[:max_ensemble_size]
        else:
            mol_h.SetIntProp("ReferenceClusters", len(remaining_cids))
            curated = remaining_cids[:max_ensemble_size]

        with Chem.SDWriter(str(target_path)) as writer:
            for i, cid in enumerate(curated):
                check_identity(Chem.Mol(mol_h, confId=cid), mol)
                mol_h.SetProp("_Name", f"Conformer_{i + 1}")
                if energies is not None:
                    mol_h.SetDoubleProp("Energy_kcal_mol", energies[cid])
                    mol_h.SetDoubleProp("DeltaE_kcal_mol", energies[cid] - min_e)
                writer.write(mol_h, confId=cid)

        return target_path
