"""Temporary ROCS scorer subclass tracking per-molecule conformer energy & RMSD collapse basins.

Extends :class:`ComboROCSScorer` without altering ROCS or SA/QED reward values:
1. Evaluates the exact same ROCS alignment scores as ``ComboROCSScorer``.
2. Reads MMFF94s ``Energy`` / ``MinimizeStatus`` (or computes single-point MMFF94s
   energies on H-containing conformers when ``minimize=False`` or fallback applies).
3. Groups conformers by ``(isomer_id, stereo_key)`` so differing stereoisomers never trigger
   ``RuntimeError`` in ``rdMolAlign.GetBestRMS``, packs heavy-atom conformers into one
   multi-conformer ``Chem.Mol`` with precomputed symmetry automorphism maps, and performs
   energy-sorted closest-leader basin clustering at 0.10, 0.25, and 0.50 A.
4. Streams per-isomer summaries and per-basin details to append-only ``.tsv.gz`` logs so
   parent memory remains bounded across full 70-epoch RL runs.
"""

from __future__ import annotations

import csv
import gc
import gzip
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from _david.code.config import Config
from _david.code.scorers.conformers import _stereo_key, single_point_energy
from _david.code.scorers.rocs import (
    NO_OVERLAY,
    ComboROCSScorer,
    lookup_by_smiles,
    pair_components,
)
from _david.code.scorers.utils import SCORE_CHUNK, pool_can_run, resolve_workers
from drugex.training.scorers.parallel import (
    ScoreTimeout,
    mol_timeout,
    molecule_time_limit,
    pool_context,
)

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = (
    "call_id",
    "stage",
    "smiles",
    "isomer_id",
    "stereo_key",
    "n_embedded",
    "n_converged",
    "n_unconverged",
    "n_basins_010",
    "n_basins_025",
    "n_basins_050",
    "collapse_ratio_025",
    "top1_basin_count",
    "basin_entropy",
    "rmsd_min",
    "rmsd_median",
    "energy_min",
    "energy_median",
    "energy_span",
    "rocs_combo_max",
    "rocs_combo_median",
    "rocs_combo_at_emin",
    "emin_rank_by_rocs",
    "best_rocs_delta_e",
)

BASIN_COLUMNS = (
    "call_id",
    "stage",
    "smiles",
    "isomer_id",
    "basin_id",
    "count",
    "n_converged",
    "energy_min",
    "energy_max",
    "delta_e_from_global",
    "intra_rmsd_max",
    "rocs_shape",
    "rocs_color",
    "rocs_combo",
    "rocs_combo_spread",
)

BASIN_DIAGNOSTIC_KEYS = (
    "n_conformers",
    "n_converged",
    "n_basins_025",
    "collapse_ratio_025",
    "top1_basin_count",
    "energy_min",
    "energy_span",
    "rmsd_median",
    "best_rocs_delta_e",
)

_EMPTY_BASIN_DIAGNOSTICS: dict[str, float] = {
    "n_conformers": 0.0,
    "n_converged": 0.0,
    "n_basins_025": 0.0,
    "collapse_ratio_025": 0.0,
    "top1_basin_count": 0.0,
    "energy_min": math.nan,
    "energy_span": math.nan,
    "rmsd_median": math.nan,
    "best_rocs_delta_e": math.nan,
}


def _parse_isomer_id(conf: Chem.Mol) -> int:
    name = conf.GetProp("_Name") if conf.HasProp("_Name") else ""
    _, sep, tail = name.partition("+")
    return int(tail) if sep and tail.isdigit() else 0


def _conformer_energy_status(conf: Chem.Mol, cfg: Config) -> tuple[float, int]:
    """Return ``(energy_kcal_mol, status)`` without mutating ``conf``.

    ``status`` conventions:
      - ``0``: MMFF94s minimization converged (``MinimizeStatus == 0``)
      - ``> 0``: MMFF94s minimization hit iteration cap (unconverged)
      - ``-1``: unminimized run (``cfg.minimize=False``) single-point MMFF94s energy
      - ``-2``: unparameterized / non-finite MMFF94s energy
      - ``-3``: ``cfg.minimize=True`` fallback single-point energy (e.g. ``stereo_changed``)
    """
    if conf.HasProp("Energy"):
        energy = float(conf.GetDoubleProp("Energy"))
        status = int(conf.GetIntProp("MinimizeStatus")) if conf.HasProp("MinimizeStatus") else 0
        if not math.isfinite(energy):
            return math.inf, -2
        return energy, status
    energy = single_point_energy(conf, cfg)
    if not math.isfinite(energy):
        return math.inf, -2
    return energy, (-3 if cfg.minimize else -1)


def _heavy_rmsd_matrix(group_confs: list[Chem.Mol]) -> np.ndarray:
    """Compute exact pairwise heavy-atom symmetry-corrected RMSDs inside one stereoisomer."""
    m = len(group_confs)
    mat = np.zeros((m, m), dtype=float)
    if m <= 1:
        return mat

    heavy_confs = [Chem.RemoveHs(Chem.Mol(conf)) for conf in group_confs]
    if heavy_confs[0].GetNumAtoms() <= 1:
        return mat

    heavy_mol = Chem.Mol(heavy_confs[0])
    for other in heavy_confs[1:]:
        heavy_mol.AddConformer(other.GetConformer(0), assignId=True)
    matches = heavy_mol.GetSubstructMatches(heavy_mol, uniquify=False, maxMatches=256)
    atom_maps = [list(enumerate(match)) for match in matches] if matches else None
    for i in range(m):
        for j in range(i + 1, m):
            try:
                rms = float(
                    rdMolAlign.GetBestRMS(heavy_mol, heavy_mol, prbId=i, refId=j, map=atom_maps)
                )
            except RuntimeError:
                rms = math.inf
            mat[i, j] = mat[j, i] = rms
    return mat


def _greedy_leader_basins(
    order: list[int], rmsd_mat: np.ndarray, tau: float
) -> tuple[list[int], list[list[int]]]:
    """Energy-sorted closest-leader clustering at heavy-atom RMSD threshold ``tau``."""
    leaders: list[int] = []
    members: list[list[int]] = []
    for idx in order:
        best_r = -1
        best_d = math.inf
        for r, leader in enumerate(leaders):
            d = float(rmsd_mat[idx, leader])
            if d <= tau and d < best_d:
                best_d = d
                best_r = r
        if best_r >= 0:
            members[best_r].append(idx)
        else:
            leaders.append(idx)
            members.append([idx])
    return leaders, members


def _round_finite(value: float, digits: int) -> float:
    return round(value, digits) if math.isfinite(value) else math.nan


def cluster_isomer_conformers(
    group_confs: list[Chem.Mol],
    conf_rocs: list[tuple[float, float]],
    cfg: Config,
    *,
    isomer_id: int,
    stereo_key: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Cluster one stereoisomer's conformers into RMSD/energy basins and return summaries."""
    m = len(group_confs)
    energies: list[float] = []
    statuses: list[int] = []
    for conf in group_confs:
        energy, status = _conformer_energy_status(conf, cfg)
        energies.append(energy)
        statuses.append(status)

    shapes = [float(pair[0]) for pair in conf_rocs]
    colors = [float(pair[1]) for pair in conf_rocs]
    combos = [s + c for s, c in zip(shapes, colors, strict=True)]

    rmsd_mat = _heavy_rmsd_matrix(group_confs)
    order = sorted(
        range(m),
        key=lambda idx: (not math.isfinite(energies[idx]), energies[idx], -combos[idx], idx),
    )

    _, members_010 = _greedy_leader_basins(order, rmsd_mat, 0.10)
    leaders_025, members_025 = _greedy_leader_basins(order, rmsd_mat, 0.25)
    _, members_050 = _greedy_leader_basins(order, rmsd_mat, 0.50)

    n_basins_010 = len(members_010)
    n_basins_025 = len(members_025)
    n_basins_050 = len(members_050)
    collapse_ratio_025 = 1.0 - (n_basins_025 / m) if m > 0 else 0.0
    top1_basin_count = max((len(mem) for mem in members_025), default=0)

    if m <= 1:
        basin_entropy = 0.0
        rmsd_min, rmsd_median = 0.0, 0.0
    else:
        probs = [len(mem) / m for mem in members_025]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        basin_entropy = float(h / math.log(m))
        tri_i, tri_j = np.triu_indices(m, k=1)
        pair_vals = rmsd_mat[tri_i, tri_j]
        finite_pairs = pair_vals[np.isfinite(pair_vals)]
        if finite_pairs.size > 0:
            rmsd_min = float(np.min(finite_pairs))
            rmsd_median = float(np.median(finite_pairs))
        else:
            rmsd_min, rmsd_median = math.nan, math.nan

    finite_energies = [e for e in energies if math.isfinite(e)]
    if finite_energies:
        energy_min = float(min(finite_energies))
        energy_median = float(np.median(finite_energies))
        energy_span = float(max(finite_energies) - energy_min)
    else:
        energy_min, energy_median, energy_span = math.nan, math.nan, math.nan

    emin_idx = order[0]
    rocs_combo_max = float(max(combos))
    rocs_combo_median = float(np.median(combos))
    rocs_combo_at_emin = float(combos[emin_idx])

    best_rocs_idx = max(
        range(m),
        key=lambda idx: (
            combos[idx],
            -energies[idx] if math.isfinite(energies[idx]) else -math.inf,
            -idx,
        ),
    )
    best_rocs_delta_e = (
        float(energies[best_rocs_idx] - energy_min)
        if math.isfinite(energies[best_rocs_idx]) and math.isfinite(energy_min)
        else math.nan
    )

    basin_best_combos = [max(combos[idx] for idx in mem) for mem in members_025]
    # Rank of basin 0 (the lowest-energy basin) by best ROCS combo, 1-indexed.
    emin_basin_combo = basin_best_combos[0]
    emin_rank_by_rocs = 1 + sum(1 for val in basin_best_combos[1:] if val > emin_basin_combo)

    n_converged = sum(1 for st in statuses if st == 0)
    n_unconverged = sum(1 for st in statuses if st > 0)

    summary_row: dict[str, object] = {
        "isomer_id": isomer_id,
        "stereo_key": stereo_key,
        "n_embedded": m,
        "n_converged": n_converged,
        "n_unconverged": n_unconverged,
        "n_basins_010": n_basins_010,
        "n_basins_025": n_basins_025,
        "n_basins_050": n_basins_050,
        "collapse_ratio_025": round(collapse_ratio_025, 6),
        "top1_basin_count": top1_basin_count,
        "basin_entropy": round(basin_entropy, 6),
        "rmsd_min": _round_finite(rmsd_min, 5),
        "rmsd_median": _round_finite(rmsd_median, 5),
        "energy_min": _round_finite(energy_min, 4),
        "energy_median": _round_finite(energy_median, 4),
        "energy_span": _round_finite(energy_span, 4),
        "rocs_combo_max": round(rocs_combo_max, 6),
        "rocs_combo_median": round(rocs_combo_median, 6),
        "rocs_combo_at_emin": round(rocs_combo_at_emin, 6),
        "emin_rank_by_rocs": int(emin_rank_by_rocs),
        "best_rocs_delta_e": _round_finite(best_rocs_delta_e, 4),
    }

    basin_rows: list[dict[str, object]] = []
    for basin_id, (leader_idx, mem) in enumerate(zip(leaders_025, members_025, strict=True)):
        mem_finite_e = [energies[idx] for idx in mem if math.isfinite(energies[idx])]
        b_emin = float(min(mem_finite_e)) if mem_finite_e else math.nan
        b_emax = float(max(mem_finite_e)) if mem_finite_e else math.nan
        delta_e = (
            float(b_emin - energy_min)
            if (math.isfinite(b_emin) and math.isfinite(energy_min))
            else math.nan
        )
        intra_vals = [
            float(rmsd_mat[idx, leader_idx])
            for idx in mem
            if math.isfinite(rmsd_mat[idx, leader_idx])
        ]
        intra_max = max(intra_vals, default=0.0)
        best_idx = max(mem, key=lambda idx: (combos[idx], -idx))
        mem_combos = [combos[idx] for idx in mem]
        basin_rows.append(
            {
                "isomer_id": isomer_id,
                "basin_id": basin_id,
                "count": len(mem),
                "n_converged": sum(1 for idx in mem if statuses[idx] == 0),
                "energy_min": _round_finite(b_emin, 4),
                "energy_max": _round_finite(b_emax, 4),
                "delta_e_from_global": _round_finite(delta_e, 4),
                "intra_rmsd_max": round(intra_max, 5),
                "rocs_shape": round(shapes[best_idx], 6),
                "rocs_color": round(colors[best_idx], 6),
                "rocs_combo": round(combos[best_idx], 6),
                "rocs_combo_spread": round(max(mem_combos) - min(mem_combos), 6),
            }
        )
    return summary_row, basin_rows


@dataclass(frozen=True)
class BasinPoseSearch:
    """Stateless worker payload that scores ROCS and clusters conformers into basins."""

    references: list[Chem.Mol]
    groups: list[list[int]]
    opt_param: float
    cfg: Config
    use_colors: bool = True
    color_ff: str = "fdef"
    rmsd_thresholds: tuple[float, float, float] = (0.10, 0.25, 0.50)

    def score_and_cluster(
        self, mol_id: int, conformers: list[Chem.Mol]
    ) -> tuple[int, list[tuple[float, float]], list[dict[str, object]], list[dict[str, object]]]:
        """Score ``conformers`` against ``groups`` and return per-isomer and per-basin rows."""
        if not conformers:
            return mol_id, [NO_OVERLAY for _ in self.groups], [], []

        pair_cache: dict[tuple[int, int], tuple[float, float]] = {}
        try:
            with molecule_time_limit(mol_timeout()):
                groups_result: list[tuple[float, float]] = []
                conf_rocs_group0: list[tuple[float, float]] = [NO_OVERLAY] * len(conformers)
                for group_idx, indices in enumerate(self.groups):
                    best_group = NO_OVERLAY
                    for c_idx, conformer in enumerate(conformers):
                        conf_best = NO_OVERLAY
                        for ref_idx in indices:
                            key = (c_idx, ref_idx)
                            found = pair_cache.get(key)
                            if found is None:
                                found = pair_components(
                                    conformer,
                                    self.references[ref_idx],
                                    opt_param=self.opt_param,
                                    use_colors=self.use_colors,
                                    color_ff=self.color_ff,
                                )
                                pair_cache[key] = found
                            if sum(found) > sum(conf_best):
                                conf_best = found
                            if sum(found) > sum(best_group):
                                best_group = found
                        if group_idx == 0:
                            conf_rocs_group0[c_idx] = conf_best
                    groups_result.append(best_group)
        except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError) as error:
            logger.warning("scoring failed for one molecule (%s); scored 0.0", error)
            return mol_id, [NO_OVERLAY for _ in self.groups], [], []

        grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
        for c_idx, conf in enumerate(conformers):
            iso_id = _parse_isomer_id(conf)
            skey = _stereo_key(conf) or "__none__"
            grouped[(iso_id, skey)].append(c_idx)

        isomer_summaries: list[dict[str, object]] = []
        basin_rows: list[dict[str, object]] = []
        for (iso_id, skey), idx_list in sorted(grouped.items()):
            sub_confs = [conformers[i] for i in idx_list]
            sub_rocs = [conf_rocs_group0[i] for i in idx_list]
            summary_row, sub_basins = cluster_isomer_conformers(
                sub_confs,
                sub_rocs,
                self.cfg,
                isomer_id=iso_id,
                stereo_key=skey,
            )
            isomer_summaries.append(summary_row)
            basin_rows.extend(sub_basins)

        return mol_id, groups_result, isomer_summaries, basin_rows


_BASIN_SEARCH: BasinPoseSearch | None = None


def _basin_worker_init(search: BasinPoseSearch) -> None:
    global _BASIN_SEARCH
    _BASIN_SEARCH = search


def _basin_worker_score(
    task: tuple[int, list[Chem.Mol]],
) -> tuple[int, list[tuple[float, float]], list[dict[str, object]], list[dict[str, object]]]:
    if _BASIN_SEARCH is None:
        raise RuntimeError("basin ROCS worker pool was not initialized")
    mol_id, conformers = task
    return _BASIN_SEARCH.score_and_cluster(mol_id, conformers)


def _finite_column(rows: list[dict[str, object]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))
    ]


def _aggregate_molecule_diagnostics(
    isomer_summaries: list[dict[str, object]],
) -> dict[str, float]:
    """Roll up per-isomer basin summaries into one diagnostic dict per molecule."""
    if not isomer_summaries:
        return dict(_EMPTY_BASIN_DIAGNOSTICS)

    n_conf = sum(int(row["n_embedded"]) for row in isomer_summaries)
    n_conv = sum(int(row["n_converged"]) for row in isomer_summaries)
    n_basins = sum(int(row["n_basins_025"]) for row in isomer_summaries)
    collapse_ratio = (1.0 - n_basins / n_conf) if n_conf > 0 else 0.0
    top1 = max(int(row["top1_basin_count"]) for row in isomer_summaries)

    emins = _finite_column(isomer_summaries, "energy_min")
    espans = _finite_column(isomer_summaries, "energy_span")
    rmsds = _finite_column(isomer_summaries, "rmsd_median")
    best_iso = max(isomer_summaries, key=lambda r: float(r["rocs_combo_max"]))
    delta_e = float(best_iso["best_rocs_delta_e"])
    best_rmsd = best_iso["rmsd_median"]
    rmsd_val = (
        round(float(best_rmsd), 5)
        if isinstance(best_rmsd, (int, float)) and math.isfinite(float(best_rmsd))
        else (round(float(np.median(rmsds)), 5) if rmsds else math.nan)
    )
    return {
        "n_conformers": float(n_conf),
        "n_converged": float(n_conv),
        "n_basins_025": float(n_basins),
        "collapse_ratio_025": round(collapse_ratio, 6),
        "top1_basin_count": float(top1),
        "energy_min": round(min(emins), 4) if emins else math.nan,
        "energy_span": round(max(espans), 4) if espans else math.nan,
        "rmsd_median": rmsd_val,
        "best_rocs_delta_e": _round_finite(delta_e, 4),
    }


def _append_tsv_gz(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.is_file() or path.stat().st_size == 0
    with gzip.open(path, "at", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), delimiter="\t", lineterminator="\n"
        )
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


class BasinTrackingROCSScorer(ComboROCSScorer):
    """Temporary ``ComboROCSScorer`` subclass that tracks conformer collapse basins."""

    def __init__(
        self,
        *args: Any,
        cfg: Config,
        basin_log_dir: Path | None = None,
        stage: str = "init",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.basin_log_dir = Path(basin_log_dir) if basin_log_dir is not None else None
        self.stage = str(stage)
        self._call_id = 0
        self._active_unique_smiles: list[str] = []
        self._mol_basin_by_id: dict[int, dict[str, float]] = {}
        self.mol_basin_components: dict[str, dict[str, float]] = {}

    def set_stage(self, stage: str) -> None:
        """Tag subsequent logged basin rows with ``stage`` (e.g. ``prior``, ``train``, ``final``)."""
        self.stage = str(stage)

    def _score_sequential(
        self,
        unique_count: int,
        conformers_by_mol: dict[int, list[Chem.Mol]],
        num_groups: int,
    ) -> np.ndarray:
        self._by_mol_id = {}
        self._mol_basin_by_id = {}
        search = BasinPoseSearch(
            references=self.reference_mols,
            groups=self.group_to_indices,
            opt_param=self.alignment_shape_weight,
            cfg=self.cfg,
            use_colors=self.use_colors,
            color_ff=self.color_ff,
        )
        tasks = [(mol_id, conformers_by_mol.get(mol_id, [])) for mol_id in range(unique_count)]
        worker_count = resolve_workers(self.workers)

        if pool_can_run(worker_count, unique_count, min_items=8):
            try:
                with pool_context().Pool(worker_count, _basin_worker_init, (search,)) as pool:
                    results = pool.map(_basin_worker_score, tasks)
            except Exception as error:
                logger.error("parallel basin ROCS failed (%s); aborting", error)
                raise RuntimeError(f"parallel basin ROCS failed: {error}") from error
        else:
            results = [search.score_and_cluster(mol_id, confs) for mol_id, confs in tasks]

        scores = np.zeros((unique_count, num_groups))
        summary_batch: list[dict[str, object]] = []
        basin_batch: list[dict[str, object]] = []

        for mol_id, groups, isomer_summaries, basin_rows in results:
            for group_index, (shape, colour) in enumerate(groups):
                scores[mol_id, group_index] = shape + colour
            self._by_mol_id[mol_id] = groups[0] if groups else NO_OVERLAY
            self._mol_basin_by_id[mol_id] = _aggregate_molecule_diagnostics(isomer_summaries)

            smi = (
                self._active_unique_smiles[mol_id]
                if mol_id < len(self._active_unique_smiles)
                else f"mol_{mol_id}"
            )
            summary_batch.extend(
                {"call_id": self._call_id, "stage": self.stage, "smiles": smi, **row}
                for row in isomer_summaries
            )
            basin_batch.extend(
                {"call_id": self._call_id, "stage": self.stage, "smiles": smi, **row}
                for row in basin_rows
            )

        if self.basin_log_dir is not None:
            _append_tsv_gz(
                self.basin_log_dir / "molecule_basins_summary.tsv.gz",
                SUMMARY_COLUMNS,
                summary_batch,
            )
            _append_tsv_gz(
                self.basin_log_dir / "conformer_basins.tsv.gz",
                BASIN_COLUMNS,
                basin_batch,
            )
        return scores

    def getScores(self, mols: Any, frags: Any = None) -> np.ndarray:
        """Score ``mols`` in memory-bounded chunks while tracking ROCS splits and conformer basins."""
        self._call_id += 1
        self._by_mol_id = {}
        self._mol_basin_by_id = {}
        self.components = {}
        self.mol_basin_components = {}

        smiles_list = self._convert_to_smiles(mols)
        unique, unique_to_original = self._deduplicate_smiles(smiles_list)
        scores = np.zeros((len(mols), len(self.group_to_indices)))
        if not unique:
            return scores

        all_components: dict[str, tuple[float, float]] = {}
        all_basin_components: dict[str, dict[str, float]] = {}
        for start in range(0, len(unique), SCORE_CHUNK):
            piece = unique[start : start + SCORE_CHUNK]
            self._by_mol_id = {}
            self._mol_basin_by_id = {}
            self._active_unique_smiles = piece
            try:
                piece_scores = super(ComboROCSScorer, self).getScores(piece, frags)
            finally:
                self._active_unique_smiles = []
            for local_id, smi in enumerate(piece):
                all_components[smi] = self._by_mol_id.get(local_id, NO_OVERLAY)
                all_basin_components[smi] = self._mol_basin_by_id.get(
                    local_id, dict(_EMPTY_BASIN_DIAGNOSTICS)
                )
                for orig_idx in unique_to_original[start + local_id]:
                    scores[orig_idx] = piece_scores[local_id]
            if len(unique) > SCORE_CHUNK:
                gc.collect()

        self.components = all_components
        self.mol_basin_components = all_basin_components
        return scores

    def component_frame(self, smiles: list[str]) -> dict[str, list[float]]:
        """Return ROCS shape/colour and per-molecule conformer basin diagnostics for ``smiles``."""
        base = super().component_frame(smiles)
        for key in BASIN_DIAGNOSTIC_KEYS:
            base[key] = []
        for value in smiles:
            entry = lookup_by_smiles(self.mol_basin_components, value) or _EMPTY_BASIN_DIAGNOSTICS
            for key in BASIN_DIAGNOSTIC_KEYS:
                base[key].append(float(entry.get(key, _EMPTY_BASIN_DIAGNOSTICS[key])))
        return base
