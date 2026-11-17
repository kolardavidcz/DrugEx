"""Multi-run & full-factorial (C x K x minimize) conformer-basin study and plotting.

Uses a slice-once evaluator: embeds and relaxes/evaluates ``C = 30`` conformers against
``K = 4`` diversity-ranked references once per ``(molecule, minimize)``, then slices exact
prefixes ``C in {20, 30}`` x ``K in {1, 4}`` from the shared conformer pool, energies,
heavy-atom RMSD matrix, and ROCS pair scores.

CLI::
    python -m _david.code.analysis.basin_study sample --out-dir ... --per-run 10
    python -m _david.code.analysis.basin_study evaluate --out-dir ... --minimize true
    python -m _david.code.analysis.basin_study plot --out-dir ...
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import gzip
import json
import logging
import math
import multiprocessing
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from _david.code.config import PROJECT_ROOT, Config
from _david.code.scorers.conformers import (
    RelaxingConformerGenerator,
    _stereo_key,
    build_reference,
)
from _david.code.scorers.rocs import pair_components
from _david.code.scorers.rocs_basin_tmp import (
    _conformer_energy_status,
    _greedy_leader_basins,
    _heavy_rmsd_matrix,
    _parse_isomer_id,
)
from _david.code.scorers.utils import (
    check_pool,
    pool_is_healthy,
    read_sdf,
    use_double_precision_pickling,
)
from drugex.training.scorers.parallel import (
    ScoreTimeout,
    capped_n_jobs,
    mol_timeout,
    molecule_time_limit,
    pool_context,
)

logger = logging.getLogger(__name__)

GRID_CELLS: tuple[tuple[int, int], ...] = ((20, 1), (20, 4), (30, 1), (30, 4))


def _murcko_scaffold(can_smiles: str) -> str:
    mol = Chem.MolFromSmiles(can_smiles)
    if mol is None:
        return can_smiles
    try:
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaff or can_smiles
    except Exception:
        return can_smiles


def discover_default_e05_runs(output_root: Path | None = None) -> list[Path]:
    """Return deduplicated completed runs matching color_ff=default, epsilon=0.05, w=0.5."""
    root = output_root or (PROJECT_ROOT / "_david" / "output")
    found: dict[str, Path] = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        if "conformer" in manifest_path.parts or "svk_validation" in manifest_path.parts:
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if manifest.get("state") != "complete":
            continue
        cfg = manifest.get("config") or {}
        if (
            cfg.get("color_ff") == "default"
            and math.isclose(float(cfg.get("epsilon", -1.0)), 0.05, abs_tol=1e-6)
            and math.isclose(float(cfg.get("alignment_shape_weight", -1.0)), 0.5, abs_tol=1e-6)
        ):
            run_dir = manifest_path.parent
            found.setdefault(run_dir.name, run_dir)
    return sorted(found.values(), key=lambda p: p.name)


def sample_runs(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    per_run: int = 10,
    n_prior: int = 30,
    max_per_scaffold: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Draw Murcko-scaffold-capped desired molecules per run + a shared prior arm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, object]] = []
    prior_added = False

    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        cfg = manifest.get("config") or {}
        arm_tsv = run_dir / "final_desired_ratio_at_stop.tsv"
        if not arm_tsv.is_file():
            arm_tsv = run_dir / "final_desired_ratio.tsv"
        if not arm_tsv.is_file():
            continue

        if not prior_added and (run_dir / "prior.tsv").is_file():
            p_df = pd.read_csv(run_dir / "prior.tsv", sep="\t")
            p_df = p_df[p_df["Valid"] == 1].dropna(subset=["Canonical"]).drop_duplicates("Canonical")
            p_df["scaffold"] = [_murcko_scaffold(s) for s in p_df["Canonical"].tolist()]
            p_df = p_df.sample(frac=1.0, random_state=seed)
            p_capped = p_df.groupby("scaffold", sort=False).head(max_per_scaffold)
            p_pick = p_capped.head(min(n_prior, len(p_capped)))
            rows.extend(
                {
                    "run_id": "Prior",
                    "regime": "Prior",
                    "scheme": "Prior",
                    "train_minimize": "Prior",
                    "train_epsilon": float("nan"),
                    "train_K": 0,
                    "train_C": 0,
                    "include_qed": False,
                    "color_ff": str(cfg.get("color_ff", "default")),
                    "can_smiles": str(rec["Canonical"]),
                    "scaffold": str(rec["scaffold"]),
                }
                for rec in p_pick.to_dict("records")
            )
            prior_added = True

        df = pd.read_csv(arm_tsv, sep="\t")
        valid = df[df["Valid"] == 1].dropna(subset=["Canonical"]).copy()
        desired = valid[valid["Desired"] == 1] if "Desired" in valid.columns else valid
        pool = desired if len(desired) >= max(5, per_run // 2) else valid
        pool = pool.drop_duplicates("Canonical").copy()
        pool["scaffold"] = [_murcko_scaffold(s) for s in pool["Canonical"].tolist()]

        run_seed = int(rng.integers(0, 2**31 - 1))
        pool = pool.sample(frac=1.0, random_state=run_seed)
        capped = pool.groupby("scaffold", sort=False).head(max_per_scaffold)
        picked = capped.head(min(per_run, len(capped)))

        train_min = bool(cfg.get("minimize", True))
        regime = "RL-m1 (opt-trained)" if train_min else "RL-m0 (unopt-trained)"
        rows.extend(
            {
                "run_id": run_dir.name,
                "regime": regime,
                "scheme": str(cfg.get("scheme", "?")),
                "train_minimize": "m1" if train_min else "m0",
                "train_epsilon": float(cfg.get("epsilon", math.nan)),
                "train_K": int(cfg.get("reference_conformers", 0)),
                "train_C": int(cfg.get("max_conformers", 0)),
                "include_qed": bool(cfg.get("include_qed", False)),
                "color_ff": str(cfg.get("color_ff", "?")),
                "can_smiles": str(rec["Canonical"]),
                "scaffold": str(rec["scaffold"]),
            }
            for rec in picked.to_dict("records")
        )

    membership = pd.DataFrame(rows)
    membership.to_csv(out_dir / "membership.tsv", sep="\t", index=False)
    unique_smiles = list(dict.fromkeys(membership["can_smiles"].tolist()))
    (out_dir / "unique_molecules.txt").write_text("\n".join(unique_smiles) + "\n")
    logger.info(
        "sampled %d rows across %d runs (%d unique canonical SMILES)",
        len(membership),
        membership["run_id"].nunique(),
        len(unique_smiles),
    )
    return membership, unique_smiles


def _cluster_slice_metrics(
    sub_confs: list[Chem.Mol],
    energies: list[float],
    statuses: list[int],
    rmsd_mat: np.ndarray,
    pair_scores: list[list[tuple[float, float]]],
    c_limit: int,
    k_limit: int,
) -> dict[str, float]:
    """Compute summary basin + ROCS metrics on the prefix ``[:c_limit]`` and ``refs[:k_limit]``."""
    m = min(len(sub_confs), c_limit)
    if m == 0:
        return {}

    e_sub = energies[:m]
    st_sub = statuses[:m]
    r_sub = rmsd_mat[:m, :m]

    shapes: list[float] = []
    colors: list[float] = []
    combos: list[float] = []
    for i in range(m):
        best_s, best_c = max(pair_scores[i][:k_limit], key=lambda sc: sc[0] + sc[1])
        shapes.append(float(best_s))
        colors.append(float(best_c))
        combos.append(float(best_s + best_c))

    order = sorted(
        range(m),
        key=lambda idx: (not math.isfinite(e_sub[idx]), e_sub[idx], -combos[idx], idx),
    )
    _, members_010 = _greedy_leader_basins(order, r_sub, 0.10)
    _, members_025 = _greedy_leader_basins(order, r_sub, 0.25)
    _, members_050 = _greedy_leader_basins(order, r_sub, 0.50)

    n_b010 = len(members_010)
    n_b025 = len(members_025)
    n_b050 = len(members_050)
    collapse_025 = 1.0 - (n_b025 / m)
    collapse_050 = 1.0 - (n_b050 / m)
    top1_count = max((len(mem) for mem in members_025), default=0)

    if m <= 1:
        entropy = 0.0
        rmsd_median = 0.0
    else:
        probs = [len(mem) / m for mem in members_025]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        entropy = float(h / math.log(m))
        tri_i, tri_j = np.triu_indices(m, k=1)
        finite_pairs = r_sub[tri_i, tri_j]
        finite_pairs = finite_pairs[np.isfinite(finite_pairs)]
        rmsd_median = float(np.median(finite_pairs)) if finite_pairs.size else math.nan

    finite_e = [e for e in e_sub if math.isfinite(e)]
    if finite_e:
        e_min = float(min(finite_e))
        e_span = float(max(finite_e) - e_min)
    else:
        e_min, e_span = math.nan, math.nan

    best_rocs_idx = max(
        range(m),
        key=lambda idx: (combos[idx], -e_sub[idx] if math.isfinite(e_sub[idx]) else -math.inf, -idx),
    )
    best_delta_e = (
        float(e_sub[best_rocs_idx] - e_min)
        if (math.isfinite(e_sub[best_rocs_idx]) and math.isfinite(e_min))
        else math.nan
    )
    return {
        "n_embedded": float(m),
        "n_converged": float(sum(1 for st in st_sub if st == 0)),
        "n_basins_010": float(n_b010),
        "n_basins_025": float(n_b025),
        "n_basins_050": float(n_b050),
        "collapse_ratio_025": collapse_025,
        "collapse_ratio_050": collapse_050,
        "top1_basin_count": float(top1_count),
        "basin_entropy": entropy,
        "rmsd_median": rmsd_median,
        "energy_min": e_min,
        "energy_span": e_span,
        "rocs_combo_max": float(max(combos)),
        "rocs_combo_at_emin": float(combos[order[0]]),
        "best_rocs_delta_e": best_delta_e,
    }


@dataclasses.dataclass(frozen=True)
class SliceGridSearch:
    references: list[Chem.Mol]
    opt_param: float
    cfg: Config
    color_ff: str
    grid_cells: tuple[tuple[int, int], ...] = GRID_CELLS

    def evaluate_molecule(
        self, smiles: str, conformers: list[Chem.Mol]
    ) -> list[dict[str, object]]:
        if not conformers:
            return []

        k_max = len(self.references)
        pair_scores: list[list[tuple[float, float]]] = []
        try:
            with molecule_time_limit(mol_timeout()):
                pair_scores = [
                    [
                        pair_components(
                            conf,
                            self.references[ref_idx],
                            opt_param=self.opt_param,
                            use_colors=True,
                            color_ff=self.color_ff,
                        )
                        for ref_idx in range(k_max)
                    ]
                    for conf in conformers
                ]
        except (ScoreTimeout, RuntimeError, ValueError, AttributeError, TypeError) as error:
            logger.warning("slice ROCS failed for %s (%s)", smiles, error)
            return []

        grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
        for c_idx, conf in enumerate(conformers):
            grouped[(_parse_isomer_id(conf), _stereo_key(conf) or "__none__")].append(c_idx)

        group_data: list[tuple[list[Chem.Mol], list[float], list[int], np.ndarray, list[list[tuple[float, float]]]]] = []
        for (_, _), idx_list in sorted(grouped.items()):
            sub_confs = [conformers[i] for i in idx_list]
            energies: list[float] = []
            statuses: list[int] = []
            for conf in sub_confs:
                e, st = _conformer_energy_status(conf, self.cfg)
                energies.append(e)
                statuses.append(st)
            rmsd_mat = _heavy_rmsd_matrix(sub_confs)
            sub_pairs = [pair_scores[i] for i in idx_list]
            group_data.append((sub_confs, energies, statuses, rmsd_mat, sub_pairs))

        out_rows: list[dict[str, object]] = []
        for c_lim, k_lim in self.grid_cells:
            iso_rows = [
                _cluster_slice_metrics(sub_c, e, st, rmat, p_sc, c_lim, k_lim)
                for sub_c, e, st, rmat, p_sc in group_data
            ]
            iso_rows = [row for row in iso_rows if row]
            if not iso_rows:
                continue

            n_conf = sum(int(r["n_embedded"]) for r in iso_rows)
            n_conv = sum(int(r["n_converged"]) for r in iso_rows)
            n_b010 = sum(int(r["n_basins_010"]) for r in iso_rows)
            n_b025 = sum(int(r["n_basins_025"]) for r in iso_rows)
            n_b050 = sum(int(r["n_basins_050"]) for r in iso_rows)
            emins = [r["energy_min"] for r in iso_rows if math.isfinite(r["energy_min"])]
            espans = [r["energy_span"] for r in iso_rows if math.isfinite(r["energy_span"])]
            rmsds = [r["rmsd_median"] for r in iso_rows if math.isfinite(r["rmsd_median"])]
            best_iso = max(iso_rows, key=lambda r: r["rocs_combo_max"])
            delta_e = best_iso["best_rocs_delta_e"]
            out_rows.append(
                {
                    "can_smiles": smiles,
                    "minimize": bool(self.cfg.minimize),
                    "C": int(c_lim),
                    "K": int(k_lim),
                    "n_isomers": len(iso_rows),
                    "n_embedded": n_conf,
                    "n_converged": n_conv,
                    "n_basins_010": n_b010,
                    "n_basins_025": n_b025,
                    "n_basins_050": n_b050,
                    "collapse_ratio_025": round(1.0 - n_b025 / n_conf, 6) if n_conf else 0.0,
                    "collapse_ratio_050": round(1.0 - n_b050 / n_conf, 6) if n_conf else 0.0,
                    "top1_basin_count": max(int(r["top1_basin_count"]) for r in iso_rows),
                    "basin_entropy": round(float(np.mean([r["basin_entropy"] for r in iso_rows])), 6),
                    "rmsd_median": round(float(best_iso["rmsd_median"]), 5)
                    if math.isfinite(best_iso["rmsd_median"])
                    else (round(float(np.median(rmsds)), 5) if rmsds else math.nan),
                    "energy_min": round(float(min(emins)), 4) if emins else math.nan,
                    "energy_span": round(float(max(espans)), 4) if espans else math.nan,
                    "rocs_combo_max": round(float(best_iso["rocs_combo_max"]), 6),
                    "rocs_combo_at_emin": round(float(best_iso["rocs_combo_at_emin"]), 6),
                    "best_rocs_delta_e": round(float(delta_e), 4) if math.isfinite(delta_e) else math.nan,
                }
            )
        return out_rows


_SLICE_SEARCH: SliceGridSearch | None = None


def _slice_worker_init(search: SliceGridSearch) -> None:
    global _SLICE_SEARCH
    _SLICE_SEARCH = search


def _slice_worker_eval(task: tuple[str, list[Chem.Mol]]) -> list[dict[str, object]]:
    smiles, confs = task
    assert _SLICE_SEARCH is not None
    return _SLICE_SEARCH.evaluate_molecule(smiles, confs)


def evaluate_molecules(
    out_dir: Path,
    *,
    minimize: bool,
    color_ff: str = "default",
    n_jobs: int = -1,
    chunk_size: int = 36,
    max_chunks: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Embed C=30 conformers once per molecule and evaluate the (C={20,30} x K={1,4}) grid."""
    check_pool()
    use_double_precision_pickling()

    smiles_file = out_dir / "unique_molecules.txt"
    all_smiles = [s.strip() for s in smiles_file.read_text().splitlines() if s.strip()]
    if num_shards > 1:
        all_smiles = [s for idx, s in enumerate(all_smiles) if idx % num_shards == shard_index]

    tag = "opt" if minimize else "unopt"
    out_tsv = out_dir / (
        f"eval_{tag}.tsv.gz" if num_shards == 1 else f"eval_{tag}_shard{shard_index}.tsv.gz"
    )

    done_smiles: set[str] = set()
    existing_df = None
    if out_tsv.is_file() and out_tsv.stat().st_size > 0:
        existing_df = pd.read_csv(out_tsv, sep="\t")
        done_smiles = set(existing_df["can_smiles"].astype(str).tolist())

    todo = [s for s in all_smiles if s not in done_smiles]
    if not todo:
        logger.info("[%s] all %d molecules already evaluated in %s", tag, len(all_smiles), out_tsv)
        return out_tsv

    cfg = Config(
        minimize=minimize,
        color_ff=color_ff,
        reference_conformers=4,
        max_conformers=30,
        max_isomers=4,
        n_jobs=n_jobs,
    )
    ref_path = out_dir / f"reference_{tag}_K4.sdf"
    refs = read_sdf(ref_path) if ref_path.is_file() else build_reference(cfg, ref_path)
    generator = RelaxingConformerGenerator(cfg)
    search = SliceGridSearch(
        references=refs,
        opt_param=cfg.alignment_shape_weight,
        cfg=cfg,
        color_ff=cfg.color_ff,
    )
    workers = capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (os.cpu_count() or 1))

    for chunks_done, start in enumerate(range(0, len(todo), chunk_size), start=1):
        if max_chunks > 0 and chunks_done > max_chunks:
            break
        batch = todo[start : start + chunk_size]
        with tempfile.TemporaryDirectory() as tmp:
            sdf_path = Path(generator.genConformers(batch, tmp))
            confs = read_sdf(sdf_path) if sdf_path.is_file() else []

        by_mol: dict[int, list[Chem.Mol]] = defaultdict(list)
        for mol in confs:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "mol_0+0"
            head = name.split("+")[0]
            if head.startswith("mol_") and head[4:].isdigit():
                by_mol[int(head[4:])].append(mol)

        tasks = [(batch[i], by_mol.get(i, [])) for i in range(len(batch))]
        if workers > 1 and len(tasks) >= 4 and not multiprocessing.current_process().daemon and pool_is_healthy():
            with pool_context().Pool(workers, _slice_worker_init, (search,)) as pool:
                batch_lists = pool.map(_slice_worker_eval, tasks)
        else:
            batch_lists = [search.evaluate_molecule(smi, c) for smi, c in tasks]

        new_rows = [row for sub in batch_lists for row in sub]
        if new_rows:
            chunk_df = pd.DataFrame(new_rows)
            if out_tsv.is_file() and out_tsv.stat().st_size > 0:
                prev_df = pd.read_csv(out_tsv, sep="\t")
                merged = pd.concat([prev_df, chunk_df], ignore_index=True)
            else:
                merged = chunk_df
            with gzip.open(out_tsv, "wt", encoding="utf-8") as handle:
                merged.to_csv(handle, sep="\t", index=False)
        gc.collect()
        logger.info("[%s] saved chunk %d (%d/%d todo done)", tag, chunks_done, min(start + len(batch), len(todo)), len(todo))
    return out_tsv


def plot_multi_run_study(out_dir: Path) -> Path:
    """Render Argon 4's 4-panel hierarchical Run-Median + Distribution comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    membership = pd.read_csv(out_dir / "membership.tsv", sep="\t")
    opt_frames = [pd.read_csv(p, sep="\t") for p in sorted(out_dir.glob("eval_opt*.tsv.gz"))]
    unopt_frames = [pd.read_csv(p, sep="\t") for p in sorted(out_dir.glob("eval_unopt*.tsv.gz"))]
    if not opt_frames or not unopt_frames:
        raise FileNotFoundError(f"Missing eval_opt*.tsv.gz or eval_unopt*.tsv.gz in {out_dir}")

    opt_df = pd.concat(opt_frames, ignore_index=True).drop_duplicates(["can_smiles", "C", "K"])
    unopt_df = pd.concat(unopt_frames, ignore_index=True).drop_duplicates(["can_smiles", "C", "K"])

    paired = opt_df.merge(
        unopt_df,
        on=["can_smiles", "C", "K"],
        suffixes=("_opt", "_unopt"),
    )
    paired["delta_e_relax"] = paired["energy_min_unopt"] - paired["energy_min_opt"]
    joined = membership.merge(paired, on="can_smiles", how="inner")
    joined.to_csv(out_dir / "joined_multi_run.tsv.gz", sep="\t", index=False, compression="gzip")

    regime_colors = {
        "Prior": "#64748b",
        "RL-m1 (opt-trained)": "#2563eb",
        "RL-m0 (unopt-trained)": "#dc2626",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), dpi=160)

    # Panel A: Relaxation energy drop across training regimes (K-independent; show C=20 vs C=30 at K=1)
    ax = axes[0, 0]
    a_df = joined[joined["K"] == 1].copy()
    regimes = [r for r in ("Prior", "RL-m1 (opt-trained)", "RL-m0 (unopt-trained)") if r in set(a_df["regime"])]
    positions = []
    labels_a = []
    pos_idx = 0
    for c_val in (20, 30):
        for reg in regimes:
            sub = a_df[(a_df["C"] == c_val) & (a_df["regime"] == reg)]
            vals = sub["delta_e_relax"].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[pos_idx],
                widths=0.45,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#111827", "lw": 2.0},
            )
            bp["boxes"][0].set_facecolor(regime_colors[reg])
            bp["boxes"][0].set_alpha(0.28)
            bp["boxes"][0].set_edgecolor(regime_colors[reg])
            run_meds = sub.groupby("run_id")["delta_e_relax"].median()
            rng = np.random.default_rng(100 + len(positions))
            ax.scatter(
                np.full(len(run_meds), pos_idx) + rng.uniform(-0.12, 0.12, len(run_meds)),
                run_meds.to_numpy(dtype=float),
                color=regime_colors[reg],
                edgecolor="white",
                linewidth=0.7,
                s=46,
                zorder=4,
            )
            positions.append(pos_idx)
            short_reg = "Prior" if reg == "Prior" else ("RL-m1" if "m1" in reg else "RL-m0")
            labels_a.append(f"{short_reg}\n(C={c_val})")
            pos_idx += 1
        pos_idx += 0.5
    ax.set_xticks(positions, labels_a, fontsize=8.5)
    ax.set_ylabel("Relaxation ΔE = E_min(unopt) − E_min(opt) [kcal/mol]")
    ax.set_title("A. Relaxation energy drop across runs (dots = run medians)")
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Conformer collapse (%) at 0.25 A: C=20 vs C=30 paired unopt -> opt run medians
    ax = axes[0, 1]
    b_df = joined[joined["K"] == 1].copy()
    x_map_b = {20: (0.0, 1.0), 30: (2.2, 3.2)}
    for c_val, (xu, xo) in x_map_b.items():
        sub_c = b_df[b_df["C"] == c_val]
        u_all = 100.0 * sub_c["collapse_ratio_025_unopt"].to_numpy(dtype=float)
        o_all = 100.0 * sub_c["collapse_ratio_025_opt"].to_numpy(dtype=float)
        ax.boxplot(
            [u_all, o_all],
            positions=[xu, xo],
            widths=0.38,
            showfliers=False,
            medianprops={"color": "#111827", "lw": 2.0},
            boxprops={"color": "#475569", "lw": 1.1},
        )
        for _run_id, rsub in sub_c.groupby("run_id"):
            reg = str(rsub["regime"].iloc[0])
            col = regime_colors.get(reg, "#475569")
            ym_u = 100.0 * float(rsub["collapse_ratio_025_unopt"].median())
            ym_o = 100.0 * float(rsub["collapse_ratio_025_opt"].median())
            ls = "--" if (not bool(rsub["include_qed"].iloc[0]) and reg != "Prior") else "-"
            ax.plot([xu, xo], [ym_u, ym_o], color=col, ls=ls, lw=1.6, alpha=0.8)
            ax.scatter([xu, xo], [ym_u, ym_o], color=col, s=36, zorder=4)
    ax.set_xticks(
        [0.0, 1.0, 2.2, 3.2],
        ["C=20\nunopt", "C=20\nopt", "C=30\nunopt", "C=30\nopt"],
    )
    ax.set_ylabel("Collapsed conformers at 0.25 A (%)\n[100 × (1 − N_basins / N_conf)]")
    ax.set_title("B. Conformer basin collapse (lines = run medians)")
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Strain exploitation best_rocs_delta_e for (C=20, K=1) vs (C=30, K=4)
    ax = axes[1, 0]
    x_map_c = {(20, 1): (0.0, 1.0), (30, 4): (2.2, 3.2)}
    for (c_val, k_val), (xu, xo) in x_map_c.items():
        sub_ck = joined[(joined["C"] == c_val) & (joined["K"] == k_val)]
        u_all = sub_ck["best_rocs_delta_e_unopt"].dropna().to_numpy(dtype=float)
        o_all = sub_ck["best_rocs_delta_e_opt"].dropna().to_numpy(dtype=float)
        ax.boxplot(
            [u_all, o_all],
            positions=[xu, xo],
            widths=0.38,
            showfliers=False,
            medianprops={"color": "#111827", "lw": 2.0},
            boxprops={"color": "#475569", "lw": 1.1},
        )
        for _run_id, rsub in sub_ck.groupby("run_id"):
            reg = str(rsub["regime"].iloc[0])
            col = regime_colors.get(reg, "#475569")
            ym_u = float(rsub["best_rocs_delta_e_unopt"].median())
            ym_o = float(rsub["best_rocs_delta_e_opt"].median())
            ls = "--" if (not bool(rsub["include_qed"].iloc[0]) and reg != "Prior") else "-"
            ax.plot([xu, xo], [ym_u, ym_o], color=col, ls=ls, lw=1.6, alpha=0.8)
            ax.scatter([xu, xo], [ym_u, ym_o], color=col, s=36, zorder=4)
    ax.axhline(3.0, color="#059669", ls=":", lw=1.1, alpha=0.8, label="3 kcal/mol bioactive bound")
    ax.axhline(5.0, color="#d97706", ls=":", lw=1.1, alpha=0.8, label="5 kcal/mol strain bound")
    ax.set_xticks(
        [0.0, 1.0, 2.2, 3.2],
        ["C=20, K=1\nunopt", "C=20, K=1\nopt", "C=30, K=4\nunopt", "C=30, K=4\nopt"],
    )
    ax.set_ylabel("Strain ΔE = E(best ROCS) − E_min [kcal/mol]")
    ax.set_title("C. Strain of winning ROCS pose: (C=20,K=1) vs (C=30,K=4)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Score vs Strain arrow vector plot (opt -> unopt) per run for (C=20,K=1) & (C=30,K=4)
    ax = axes[1, 1]
    markers = {(20, 1): "o", (30, 4): "^"}
    for (c_val, k_val), marker in markers.items():
        sub_ck = joined[(joined["C"] == c_val) & (joined["K"] == k_val)]
        for _run_id, rsub in sub_ck.groupby("run_id"):
            reg = str(rsub["regime"].iloc[0])
            col = regime_colors.get(reg, "#475569")
            x_opt = float(rsub["best_rocs_delta_e_opt"].median())
            y_opt = float(rsub["rocs_combo_max_opt"].median())
            x_un = float(rsub["best_rocs_delta_e_unopt"].median())
            y_un = float(rsub["rocs_combo_max_unopt"].median())
            ax.annotate(
                "",
                xy=(x_un, y_un),
                xytext=(x_opt, y_opt),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3, "alpha": 0.65},
            )
            ax.scatter([x_opt], [y_opt], color=col, marker=marker, s=42, zorder=4)
            ax.scatter([x_un], [y_un], color=col, marker=marker, facecolors="none", s=46, lw=1.3, zorder=4)

    for reg, col in regime_colors.items():
        ax.scatter([], [], color=col, label=reg, s=40)
    ax.scatter([], [], color="#111827", marker="o", label="C=20, K=1 (filled=opt, hollow=unopt)", s=40)
    ax.scatter([], [], color="#111827", marker="^", label="C=30, K=4 (arrow: opt → unopt)", s=42)
    ax.set_xlabel("Run median strain ΔE = E(best ROCS) − E_min [kcal/mol]")
    ax.set_ylabel("Run median TanimotoCombo (max)")
    ax.set_title("D. Score–strain vector: opt (filled) → unopt (hollow)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    n_runs = membership["run_id"].nunique()
    n_mols = paired["can_smiles"].nunique()
    fig.suptitle(
        f"Multi-Run Conformer Collapse & Strain Study (color_ff=default, ε=0.05; {n_runs} runs, {n_mols} unique molecules)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = out_dir / "multi_run_basin_study.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # Save concise run-level summary table
    summary_rows = []
    for (run_id, reg, scheme, tmin, tk, tc, qed, c_val, k_val), grp in joined.groupby(
        ["run_id", "regime", "scheme", "train_minimize", "train_K", "train_C", "include_qed", "C", "K"],
        dropna=False,
    ):
        summary_rows.append(
            {
                "run_id": run_id,
                "regime": reg,
                "scheme": scheme,
                "train_minimize": tmin,
                "train_K": tk,
                "train_C": tc,
                "include_qed": qed,
                "eval_C": c_val,
                "eval_K": k_val,
                "n": len(grp),
                "emin_opt_med": round(float(grp["energy_min_opt"].median()), 3),
                "emin_unopt_med": round(float(grp["energy_min_unopt"].median()), 3),
                "collapse025_opt_pct": round(100.0 * float(grp["collapse_ratio_025_opt"].median()), 2),
                "collapse025_unopt_pct": round(100.0 * float(grp["collapse_ratio_025_unopt"].median()), 2),
                "delta_e_rocs_opt_med": round(float(grp["best_rocs_delta_e_opt"].median()), 3),
                "delta_e_rocs_unopt_med": round(float(grp["best_rocs_delta_e_unopt"].median()), 3),
                "rocs_opt_med": round(float(grp["rocs_combo_max_opt"].median()), 4),
                "rocs_unopt_med": round(float(grp["rocs_combo_max_unopt"].median()), 4),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "run_summary.tsv", sep="\t", index=False)
    return fig_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_samp = sub.add_parser("sample", help="Sample Murcko-capped molecules across matching runs.")
    p_samp.add_argument("--out-dir", type=Path, required=True)
    p_samp.add_argument("--per-run", type=int, default=10)
    p_samp.add_argument("--n-prior", type=int, default=30)
    p_samp.add_argument("--max-per-scaffold", type=int, default=2)
    p_samp.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("evaluate", help="Evaluate sampled unique molecules on the C x K grid.")
    p_eval.add_argument("--out-dir", type=Path, required=True)
    p_eval.add_argument("--minimize", choices=("true", "false"), required=True)
    p_eval.add_argument("--color-ff", default="default")
    p_eval.add_argument("--n-jobs", type=int, default=-1)
    p_eval.add_argument("--chunk-size", type=int, default=36)
    p_eval.add_argument("--max-chunks", type=int, default=0)
    p_eval.add_argument("--shard-index", type=int, default=0)
    p_eval.add_argument("--num-shards", type=int, default=1)

    p_plot = sub.add_parser("plot", help="Render the multi-run 4-panel study figure.")
    p_plot.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "sample":
        runs = discover_default_e05_runs()
        sample_runs(
            runs,
            args.out_dir,
            per_run=args.per_run,
            n_prior=args.n_prior,
            max_per_scaffold=args.max_per_scaffold,
            seed=args.seed,
        )
        return 0
    if args.cmd == "evaluate":
        evaluate_molecules(
            args.out_dir,
            minimize=(args.minimize == "true"),
            color_ff=args.color_ff,
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        return 0
    if args.cmd == "plot":
        fig = plot_multi_run_study(args.out_dir)
        print(fig)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
