from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdShapeAlign


class ConformerConvergenceAnalyzer:
    """Computes conformational metric properties and ROCS TanimotoCombo self-similarity surfaces."""

    @staticmethod
    def compute_radius_of_gyration(mol: Chem.Mol, conf_id: int) -> float:
        """Compute the heavy-atom radius of gyration (Rg in Angstroms).

        Quantifies overall molecular compactness versus spatial expansion:
            R_g = sqrt( (1 / N) * sum_i ||r_i - r_mean||^2 )

        Args:
            mol: Molecule holding the conformer.
            conf_id: Target conformer identifier.

        Returns:
            Radius of gyration in Angstroms.
        """
        conf = mol.GetConformer(conf_id)
        heavy_atom_indices = [
            i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1
        ]
        if not heavy_atom_indices:
            heavy_atom_indices = list(range(mol.GetNumAtoms()))

        positions = [np.array(conf.GetAtomPosition(i)) for i in heavy_atom_indices]
        centroid = np.mean(positions, axis=0)
        squared_distances = [np.sum((pos - centroid) ** 2) for pos in positions]
        return float(math.sqrt(np.mean(squared_distances)))

    @staticmethod
    def evaluate_pairwise_rocs(
        ref_mol: Chem.Mol,
        ref_cids: Sequence[int],
        query_mol: Chem.Mol,
        query_cids: Sequence[int],
    ) -> tuple[float, float, float]:
        """Align query conformers against reference conformers and identify maximal ROCS score.

        Evaluates pairwise shape density (Shape Tanimoto in [0, 1]) and chemical pharmacophore
        color density (Color Tanimoto in [0, 1]), summing to TanimotoCombo in [0, 2].

        Args:
            ref_mol: Molecule holding reference conformers.
            ref_cids: List of active reference conformer IDs.
            query_mol: Molecule holding candidate query conformers.
            query_cids: List of active query conformer IDs.

        Returns:
            Tuple of (max_tanimoto_combo, corresponding_shape_tanimoto, corresponding_color_tanimoto).
        """
        best_tc = 0.0
        best_shape = 0.0
        best_color = 0.0

        for q_cid in query_cids:
            for r_cid in ref_cids:
                try:
                    shape_t, color_t = rdShapeAlign.AlignMol(
                        ref_mol, query_mol, probeConfId=q_cid, refConfId=r_cid
                    )
                    tc = float(shape_t + color_t)
                    if tc > best_tc:
                        best_tc = tc
                        best_shape = float(shape_t)
                        best_color = float(color_t)
                except Exception:
                    continue

        return best_tc, best_shape, best_color

    def run_surface_scan(
        self,
        ref_mol: Chem.Mol,
        ref_cids: Sequence[int],
        query_mol: Chem.Mol,
        query_cids: Sequence[int],
        k_ref_grid: Sequence[int],
        k_gen_grid: Sequence[int],
    ) -> pd.DataFrame:
        """Scan the 2D grid of ensemble sizes and build the full convergence response surface.

        Args:
            ref_mol: Reference conformer molecule.
            ref_cids: Curated reference conformer IDs.
            query_mol: Candidate probe conformer molecule.
            query_cids: Curated candidate probe conformer IDs.
            k_ref_grid: Sequence of requested reference basin counts (K_ref).
            k_gen_grid: Sequence of requested candidate conformer counts (K_gen).

        Returns:
            Pandas DataFrame containing grid coordinates and resulting ROCS metrics.
        """
        records: list[dict[str, object]] = []
        max_ref = len(ref_cids)
        max_query = len(query_cids)

        for kr in k_ref_grid:
            actual_kr = min(kr, max_ref)
            active_r_cids = ref_cids[:actual_kr]

            for kq in k_gen_grid:
                actual_kq = min(kq, max_query)
                active_q_cids = query_cids[:actual_kq]

                tc, shape_t, color_t = self.evaluate_pairwise_rocs(
                    ref_mol=ref_mol,
                    ref_cids=active_r_cids,
                    query_mol=query_mol,
                    query_cids=active_q_cids,
                )

                records.append(
                    {
                        "k_ref_requested": kr,
                        "k_gen_requested": kq,
                        "k_ref_actual": actual_kr,
                        "k_gen_actual": actual_kq,
                        "tanimoto_combo": tc,
                        "shape_tanimoto": shape_t,
                        "color_tanimoto": color_t,
                    }
                )

        return pd.DataFrame(records)
