from __future__ import annotations

from typing import Mapping

from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers, rdMolAlign
from rdkit.ML.Cluster import Butina

from _david.code.conformer.physics import ForceFieldVariant


class ConformerGenerator:
    """Generates, minimizes, and clusters non-degenerate 3D conformer ensembles.

    Implements distance geometry embedding (ETKDGv3) followed by molecular mechanics
    energy minimization under explicit force field and dielectric shielding specifications.
    Final ensemble selection uses Butina RMSD clustering and energy window gating.
    """

    def __init__(self, molecule_smiles: str) -> None:
        """Initialize generator with parent target molecule.

        Args:
            molecule_smiles: SMILES string of the target ligand.

        Raises:
            ValueError: If molecule_smiles cannot be parsed by RDKit.
        """
        self.smiles = molecule_smiles
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string provided: '{molecule_smiles}'")
        self.mol_h = Chem.AddHs(mol)

    def generate_raw_conformers(
        self,
        n_conformers: int = 250,
        seed: int = 42,
        prune_rms: float = 0.50,
    ) -> Chem.Mol:
        """Embed unminimized 3D conformers using RDKit ETKDGv3 with early in-C++ RMSD pruning.

        Args:
            n_conformers: Maximum number of conformers to attempt embedding.
            seed: RNG seed for reproducible coordinate embedding.
            prune_rms: Pre-minimization heavy-atom RMSD distance threshold in Angstroms.

        Returns:
            RDKit molecule with raw embedded 3D conformers.
        """
        mol = Chem.Mol(self.mol_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = prune_rms
        AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        return mol

    def minimize_and_score(
        self,
        mol: Chem.Mol,
        variant: ForceFieldVariant,
        max_iterations: int = 500,
    ) -> tuple[Chem.Mol, dict[int, float]]:
        """Minimize each conformer in place according to the variant's force field and dielectric.

        Supports MMFF94s with constant or distance-dependent dielectric screening, as well as UFF.

        Args:
            mol: Molecule with conformers to minimize.
            variant: Physical environment configuration.
            max_iterations: Maximum optimization cycles per conformer.

        Returns:
            Tuple of (minimized_mol, energy_dict_by_conf_id).
        """
        energies: dict[int, float] = {}
        num_confs = mol.GetNumConformers()

        for cid in range(num_confs):
            if variant.ff_type.lower() == "uff":
                ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            else:
                mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(
                    mol, mmffVariant="MMFF94s"
                )
                mp.SetMMFFDielectricModel(variant.dielectric_model)
                mp.SetMMFFDielectricConstant(variant.dielectric_constant)
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=cid)

            if ff is not None:
                ff.Initialize()
                ff.Minimize(maxIts=max_iterations)
                energies[cid] = float(ff.CalcEnergy())
            else:
                energies[cid] = 1e6

        return mol, energies

    def cluster_and_filter(
        self,
        mol: Chem.Mol,
        energies: Mapping[int, float],
        variant: ForceFieldVariant,
    ) -> list[int]:
        """Apply Delta E window filtering and Butina RMSD clustering to select basin centroids.

        Args:
            mol: Molecule containing minimized conformers.
            energies: Mapping of conf_id to calculated potential energy.
            variant: Force field configuration holding RMSD cutoff and energy window.

        Returns:
            List of representative conformer IDs (centroids), sorted by increasing energy.
        """
        if not energies:
            return []

        min_e = min(energies.values())

        # Step 1: Energy window gating
        if variant.energy_window_kcal is not None:
            valid_cids = [
                cid
                for cid, e in energies.items()
                if (e - min_e) <= variant.energy_window_kcal
            ]
        else:
            valid_cids = list(energies.keys())

        if not valid_cids:
            return [min(energies, key=energies.get)]

        valid_cids.sort(key=energies.get)
        n_cids = len(valid_cids)

        if n_cids <= 1:
            return valid_cids

        # Step 2: Butina RMSD clustering on heavy atoms
        heavy_mol = Chem.RemoveHs(mol)
        dists: list[float] = []
        for i in range(1, n_cids):
            for j in range(i):
                rmsd = float(
                    rdMolAlign.GetBestRMS(
                        heavy_mol, heavy_mol, valid_cids[i], valid_cids[j]
                    )
                )
                dists.append(rmsd)

        clusters = Butina.ClusterData(
            dists, n_cids, distThresh=variant.rmsd_cutoff, isDistData=True
        )

        centroids = [valid_cids[c[0]] for c in clusters]
        centroids.sort(key=energies.get)
        return centroids

    def build_ensemble(
        self,
        variant: ForceFieldVariant,
        n_raw: int = 250,
        seed: int = 42,
    ) -> tuple[Chem.Mol, list[int], dict[int, float]]:
        """Execute end-to-end ensemble pipeline: embed -> minimize -> filter & cluster.

        Args:
            variant: Physical environment specification.
            n_raw: Number of initial distance geometry conformers.
            seed: RNG seed for reproducible generation.

        Returns:
            Tuple of (minimized_mol, selected_centroid_cids, energies_dict).
        """
        raw_mol = self.generate_raw_conformers(n_conformers=n_raw, seed=seed)
        min_mol, energies = self.minimize_and_score(raw_mol, variant)
        curated_cids = self.cluster_and_filter(min_mol, energies, variant)
        return min_mol, curated_cids, energies
