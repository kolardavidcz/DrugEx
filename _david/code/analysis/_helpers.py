"""Shared chemical descriptors, Morgan fingerprints, Pareto dominance and 3D alignment."""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    QED,
    Descriptors,
    Lipinski,
    rdDepictor,
    rdFMCS,
    rdMolDescriptors,
    rdShapeAlign,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

from drugex.training.scorers.sascorer import calculateScore as calculate_sa_score

logger = logging.getLogger(__name__)

#: Arms in reporting order: the baseline first, then what training produced.
ARMS = ("prior", "desired_ratio", "desired_ratio_at_stop", "gmean")

#: 2D descriptors of one molecule, keyed by (SMILES, reference SMILES). They are a pure
#: function of those two, and the figure suite asks for the same molecules seventeen
#: times over -- parsing, Murcko scaffold, Morgan fingerprint and raw SA each time was
#: the whole cost of a render. One process analyses one run, so this holds at most
#: n_final x arms entries.
_DESCRIPTORS: dict[tuple[str, str], tuple] = {}

_NON_RUN_DIRS = frozenset(
    {"conformer", "rebuild_logs", "figures", "models", "cache_stats", "raw_data"}
)


def iter_run_dirs(root: Path) -> list[Path]:
    """Every run directory (holding ``manifest.json``) anywhere under ``root``."""
    found: list[Path] = []
    if not root.is_dir():
        return found

    def _walk(directory: Path) -> None:
        if (directory / "manifest.json").is_file():
            found.append(directory)
            return
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name in _NON_RUN_DIRS:
                continue
            _walk(child)

    _walk(root)
    return sorted(found, key=lambda p: (p.name, str(p)))


def valid(arm: pd.DataFrame | None) -> pd.DataFrame:
    """Valid rows only; invalid rows have every objective zeroed and are not real molecules."""
    if arm is None or arm.empty:
        return pd.DataFrame()
    if "Valid" not in arm.columns:
        return arm.copy()
    return arm[pd.to_numeric(arm["Valid"], errors="coerce").fillna(0) == 1].copy()


def dedup_key(frame: pd.DataFrame) -> str:
    """Always deduplicate on Canonical SMILES so two spellings of one molecule count once."""
    return "Canonical" if "Canonical" in frame.columns else "SMILES"


def unique_per_epoch(epochs: pd.DataFrame | None) -> pd.DataFrame:
    """Valid rows, one per distinct canonical structure *within each epoch*. Without the dedup
    a collapsed epoch's median is the score of whichever molecule was drawn most often."""
    rows = valid(epochs)
    if rows.empty or "Epoch" not in rows.columns:
        return rows
    return rows.drop_duplicates(subset=["Epoch", dedup_key(rows)]).copy()


def unique(arm: pd.DataFrame | None) -> pd.DataFrame:
    """Valid rows, one per distinct canonical structure."""
    v = valid(arm)
    if v.empty:
        return v
    return v.drop_duplicates(subset=dedup_key(v)).copy()


def parse_mols(smiles: pd.Series) -> list[Chem.Mol]:
    return [m for m in (Chem.MolFromSmiles(s) for s in smiles if isinstance(s, str)) if m]


def provenance_tag(config: dict | None, reference: Path | None = None) -> str:
    """Compact one-line provenance string stating every scoring and conformer axis.

    Figures from two colour typings, alignment weights or relaxation regimes carry
    incomparable scores; stating these from config.json on every figure makes each PNG
    self-describing even when pasted out of its directory.
    """
    if not config:
        return ""
    color_ff = config.get("color_ff", "fdef")
    weight = config.get("alignment_shape_weight", 0.5)
    n_refs = config.get("reference_conformers", "?")
    if reference is not None and reference.is_file():
        try:
            refs = [m for m in Chem.SDMolSupplier(str(reference), removeHs=False) if m is not None]
            if refs:
                n_refs = len(refs)
        except Exception:
            pass
    minim = config.get("minimize", True)
    ff_str = (
        f"{config.get('force_field', 'MMFF94s')}(eps={config.get('dielectric', 78.4):g})"
        if minim
        else "unrelaxed"
    )
    max_c = config.get("max_conformers", "?")
    max_iso = config.get("max_isomers", "?")
    scheme = config.get("reward_scheme") or config.get("scheme", "PRCD")
    eps = config.get("epsilon", 0.0)
    return (
        f"color_ff={color_ff} · alignment_shape_weight={weight:g} · ref_bank={n_refs} · "
        f"conformers={ff_str} ({max_iso} iso × {max_c} conf) · reward={scheme} (ε={eps:g})"
    )


def reference_properties(config: dict | None, reference: Path | None) -> dict[str, float]:
    """Properties of the reference ligand (EGCG) computed from config/reference.sdf, never hardcoded."""
    smi = (config or {}).get("reference_smiles")
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None and reference is not None and reference.is_file():
        try:
            suppl = Chem.SDMolSupplier(str(reference), removeHs=True)
            mol = next((m for m in suppl if m is not None), None)
        except Exception:
            mol = None
    if mol is None:
        return {}
    sa_raw = float(calculate_sa_score(mol))
    lo = float((config or {}).get("sa_reward_zero", 5.0))
    hi = float((config or {}).get("sa_reward_one", 3.0))
    sa_mod = float(np.clip((lo - sa_raw) / (lo - hi), 0.0, 1.0)) if lo != hi else 0.0
    return {
        "ROCS": 2.00,
        "MW": float(Descriptors.MolWt(mol)),
        "HAC": float(mol.GetNumHeavyAtoms()),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "cLogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "Fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "Rings": float(Lipinski.RingCount(mol)),
        "QED": float(QED.qed(mol)),
        "SA_raw": sa_raw,
        "SA": sa_mod,
    }


def enrich_descriptors(
    frame: pd.DataFrame,
    rocs_col: str | None = None,
    ref_smiles: str | None = None,
    config: dict | None = None,
    prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach 2D medicinal-chemistry descriptors, raw SA, Murcko scaffold and ECFP4 seed similarity.

    Returns ROCS as published TanimotoCombo on [0, 2] whatever units the frame stored it
    in (see ``frame_rocs_scale``), and replaces
    unscorable SA failure sentinels (10.0) with NaN for continuous distributions so a
    failed score can never sort, colour or rank as good.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    scale = frame_rocs_scale(df, rocs_col, config, prior=prior)
    if rocs_col and rocs_col in df.columns:
        df["ROCS"] = pd.to_numeric(df[rocs_col], errors="coerce") * scale
    elif "ROCS" not in df.columns:
        rocs_cands = [c for c in df.columns if c.startswith("RDKit_")]
        if rocs_cands:
            df["ROCS"] = pd.to_numeric(df[rocs_cands[0]], errors="coerce") * scale

    ref_fp = None
    if isinstance(ref_smiles, str) and ref_smiles:
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is not None:
            ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

    ref_key = ref_smiles if isinstance(ref_smiles, str) else ""
    (
        canon_list,
        mw_l,
        hac_l,
        rotb_l,
        logp_l,
        tpsa_l,
        fsp3_l,
        rings_l,
        qed_l,
        sa_raw_l,
        sim_l,
        scaf_l,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    has_sa_raw = "SA_raw" in df.columns
    has_qed = "QED" in df.columns

    for _, row in df.iterrows():
        smi = str(row.get("Canonical") or row.get("SMILES") or "")
        is_val = int(row.get("Valid", 1)) == 1
        if not is_val or not smi:
            canon_list.append("")
            mw_l.append(math.nan)
            hac_l.append(math.nan)
            rotb_l.append(math.nan)
            logp_l.append(math.nan)
            tpsa_l.append(math.nan)
            fsp3_l.append(math.nan)
            rings_l.append(math.nan)
            qed_l.append(math.nan)
            sa_raw_l.append(math.nan)
            sim_l.append(math.nan)
            scaf_l.append("")
            continue
        key = (smi, ref_key)
        if key not in _DESCRIPTORS:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                _DESCRIPTORS[key] = (
                    "",
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    "",
                )
            else:
                can = Chem.MolToSmiles(mol)
                try:
                    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except Exception:
                    scaf = ""
                try:
                    raw_s = float(calculate_sa_score(mol))
                except Exception:
                    raw_s = math.nan
                if not math.isfinite(raw_s) or raw_s >= 9.99 or raw_s <= 0.5:
                    raw_s = math.nan
                sim = (
                    float(
                        DataStructs.TanimotoSimilarity(
                            ref_fp,
                            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                        )
                    )
                    if ref_fp is not None
                    else math.nan
                )
                _DESCRIPTORS[key] = (
                    can,
                    float(Descriptors.MolWt(mol)),
                    float(mol.GetNumHeavyAtoms()),
                    float(Lipinski.NumRotatableBonds(mol)),
                    float(Descriptors.MolLogP(mol)),
                    float(rdMolDescriptors.CalcTPSA(mol)),
                    float(rdMolDescriptors.CalcFractionCSP3(mol)),
                    float(Lipinski.RingCount(mol)),
                    float(QED.qed(mol)),
                    raw_s,
                    sim,
                    scaf or "",
                )
        can, mw, hac, rotb, logp, tpsa, fsp3, rings, qed_v, raw_s, sim, scaf = _DESCRIPTORS[key]
        canon_list.append(can)
        mw_l.append(mw)
        hac_l.append(hac)
        rotb_l.append(rotb)
        logp_l.append(logp)
        tpsa_l.append(tpsa)
        fsp3_l.append(fsp3)
        rings_l.append(rings)
        qed_l.append(qed_v)
        sa_raw_l.append(raw_s)
        sim_l.append(sim)
        scaf_l.append(scaf)

    if "Canonical" not in df.columns:
        df["Canonical"] = canon_list
    df["MW"] = mw_l
    df["HAC"] = hac_l
    df["RotBonds"] = rotb_l
    df["LogP"] = logp_l
    df["TPSA"] = tpsa_l
    df["Fsp3"] = fsp3_l
    df["Rings"] = rings_l
    if not has_qed:
        df["QED"] = qed_l
    else:
        df["QED"] = pd.to_numeric(df["QED"], errors="coerce")

    if has_sa_raw:
        raw_col = pd.to_numeric(df["SA_raw"], errors="coerce")
        # Never let failure sentinel 10.0 or invalid 0.0 enter a distribution or rank as good
        df["Raw_SA"] = raw_col.where((df.get("Valid", 1) == 1) & (raw_col > 0.5) & (raw_col < 9.99))
    else:
        df["Raw_SA"] = sa_raw_l
        df["SA_raw"] = sa_raw_l

    if "SA" not in df.columns:
        lo = float((config or {}).get("sa_reward_zero", 5.0))
        hi = float((config or {}).get("sa_reward_one", 3.0))
        raw_arr = pd.to_numeric(df["Raw_SA"], errors="coerce").to_numpy(dtype=float)
        df["SA"] = np.where(
            np.isfinite(raw_arr) & (lo != hi),
            np.clip((lo - raw_arr) / (lo - hi), 0.0, 1.0),
            0.0,
        )
    df["SimToSeed"] = sim_l
    df["Scaffold"] = scaf_l
    return df


# --------------------------------------------------------------------------------------
# Non-dominated Pareto front and exact dominated hypervolume
# --------------------------------------------------------------------------------------


def scored_points(
    frame: pd.DataFrame,
    objectives: list[str],
    config: dict | None = None,
    prior: pd.DataFrame | None = None,
) -> np.ndarray:
    """Unique valid molecules as a point cloud, ROCS on published [0, 2]. Fronts are affine-
    invariant and would not care; ``dominated_volume`` is a volume and does."""
    u = unique(frame)
    if u.empty or any(o not in u.columns for o in objectives):
        return np.empty((0, len(objectives)), dtype=float)
    # copy=True: pandas hands back a read-only view of a single-dtype block, and the ROCS
    # column is rescaled in place below.
    pts = u[objectives].to_numpy(dtype=float, copy=True)
    if objectives:
        pts[:, 0] = pts[:, 0] * frame_rocs_scale(u, objectives[0], config, prior=prior)
    return pts[np.all(np.isfinite(pts), axis=1)]


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (all columns maximized; ties kept)."""
    points = np.asarray(points, dtype=float)
    if not points.size:
        return np.zeros(len(points), dtype=bool)
    mask = np.ones(len(points), dtype=bool)
    for i, pt in enumerate(points):
        better = np.all(points >= pt, axis=1) & np.any(points > pt, axis=1)
        mask[i] = not bool(better.any())
    return mask


def dominated_volume(points: np.ndarray) -> float:
    """Exact volume dominated by `points` above the origin in objective space."""
    points = np.asarray(points, dtype=float)
    if not points.size:
        return 0.0
    points = points[np.all(np.isfinite(points), axis=1)]
    if not points.size:
        return 0.0
    return _volume(np.clip(points, 0.0, None))


def _volume(points: np.ndarray) -> float:
    if points.shape[1] == 1:
        return float(points[:, 0].max())
    if points.shape[1] == 2:
        return _area(points)
    points = points[np.argsort(-points[:, -1])]
    vol, prev = 0.0, 0.0
    for idx in range(len(points)):
        level = float(points[idx, -1])
        if idx and level < prev:
            vol += (prev - level) * _volume(points[:idx, :-1])
        prev = level
    return vol + prev * _volume(points[:, :-1])


def _area(points: np.ndarray) -> float:
    ordered = points[np.argsort(-points[:, 0])]
    area, highest = 0.0, 0.0
    for idx in range(len(ordered)):
        x = float(ordered[idx, 0])
        right = float(ordered[idx + 1, 0]) if idx + 1 < len(ordered) else 0.0
        highest = max(highest, float(ordered[idx, 1]))
        area += max(x - right, 0.0) * highest
    return area


def compute_2d_skyline(
    x: np.ndarray,
    y: np.ndarray,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical non-dominated 2D skyline (step-wise frontier)."""
    pts = np.column_stack([x, y])
    ok = np.all(np.isfinite(pts), axis=1)
    pts = pts[ok]
    if len(pts) == 0:
        return np.array([]), np.array([])
    sx = pts[:, 0] if maximize_x else -pts[:, 0]
    sy = pts[:, 1] if maximize_y else -pts[:, 1]
    order = np.lexsort((-sy, -sx))
    sorted_pts = pts[order]
    sorted_sy = sy[order]
    sky = []
    max_sy = -math.inf
    for pt, cur_sy in zip(sorted_pts, sorted_sy):
        if cur_sy > max_sy:
            sky.append(pt)
            max_sy = cur_sy
    sky_arr = np.asarray(sky, dtype=float)
    if not len(sky_arr):
        return np.array([]), np.array([])
    sky_arr = sky_arr[np.argsort(sky_arr[:, 0])]
    return sky_arr[:, 0], sky_arr[:, 1]


#: Murcko scaffold SMILES -> its graph framework. Every arm is counted twice (once for
#: report.md's table, once for figure 07's legend) and 07b counts them again, while many
#: molecules share one scaffold -- on a collapsed arm, 1905 molecules over 84 scaffolds.
#: Keying the cache on the scaffold rather than the molecule collapses all of that.
_GENERIC_FRAMEWORKS: dict[str, str] = {}


def _graph_framework(scaffold_smiles: str, core: Chem.Mol) -> str:
    """Bemis and Murcko's graph framework: every atom carbon, every bond single."""
    cached = _GENERIC_FRAMEWORKS.get(scaffold_smiles)
    if cached is None:
        try:
            cached = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
        except Exception:
            cached = ""
        _GENERIC_FRAMEWORKS[scaffold_smiles] = cached
    return cached


def rocs_scale(config: dict | None) -> float:
    """Factor from a run's stored ROCS units to published TanimotoCombo on [0, 2]. Absent
    means the run predates the reward rescale and stored it raw."""
    try:
        return float((config or {}).get("rocs_reward_scale", 1.0)) or 1.0
    except (TypeError, ValueError):
        return 1.0


def _resolve_rocs_column(frame: pd.DataFrame, rocs_col: str | None) -> str | None:
    if rocs_col and rocs_col in frame.columns:
        return rocs_col
    for candidate in frame.columns:
        if candidate.startswith("RDKit_") or "TanimotoCombo" in candidate:
            return candidate
    return "ROCS" if "ROCS" in frame.columns else None


def frame_rocs_scale(
    frame: pd.DataFrame | None,
    rocs_col: str | None = None,
    config: dict | None = None,
    prior: pd.DataFrame | None = None,
) -> float:
    """Scale factor mapping ``frame``'s stored ROCS column onto published [0, 2].

    Arms written with ``ShapeTanimoto`` and ``ColorTanimoto`` carry their ground-truth [0, 2]
    sum directly, which catches runs whose stop arm was rebuilt after the reward rescale
    while ``config.json`` lacked ``rocs_reward_scale`` (or runs between the rescale and the
    manifest-field commit). Per-epoch tables lack those components and inherit from ``config``
    or ``prior`` (written in the same training run).
    """
    if frame is not None and not frame.empty:
        col = _resolve_rocs_column(frame, rocs_col)
        if col and "ShapeTanimoto" in frame.columns and "ColorTanimoto" in frame.columns:
            stored = pd.to_numeric(frame[col], errors="coerce")
            combo = pd.to_numeric(frame["ShapeTanimoto"], errors="coerce") + pd.to_numeric(
                frame["ColorTanimoto"], errors="coerce"
            )
            mask = (stored > 1e-6) & combo.notna()
            if mask.any():
                ratio = float((combo[mask] / stored[mask]).median())
                return 2.0 if ratio >= 1.5 else 1.0
    if (config or {}).get("rocs_reward_scale") is not None:
        return rocs_scale(config)
    if prior is not None and prior is not frame:
        return frame_rocs_scale(prior, rocs_col, config)
    return rocs_scale(config)


def threshold_rocs_scale(
    config: dict | None = None,
    prior: pd.DataFrame | None = None,
    rocs_col: str | None = None,
) -> float:
    """Scale factor mapping ``config['thresholds'][0]`` (calibrated on the prior) onto [0, 2]."""
    if (config or {}).get("rocs_reward_scale") is not None:
        return rocs_scale(config)
    if prior is not None and not prior.empty:
        return frame_rocs_scale(prior, rocs_col, config)
    return rocs_scale(config)


def scaffold_counts(frame: pd.DataFrame, generic: bool = False) -> pd.Series:
    """Bemis-Murcko scaffold counts of unique valid molecules. ``generic`` returns the graph
    framework instead -- every atom carbon, every bond single -- which a heteroatom swap does
    not move: 84 against 51 on this campaign's worst arm."""
    scaffolds = []
    for mol in parse_mols(unique(frame)["SMILES"]):
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(core)
        except Exception:
            continue
        if smi and generic:
            smi = _graph_framework(smi, core)
        if smi:
            scaffolds.append(smi)
    return pd.Series(scaffolds, dtype=str).value_counts()


def scaffold_rows(arms: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Scaffold concentration and novelty vs the prior for figure 07 and report.md."""
    counts = {name: scaffold_counts(arms[name]) for name in ARMS if name in arms}
    graphs = {name: scaffold_counts(arms[name], generic=True) for name in counts}
    prior = set(counts["prior"].index) if "prior" in counts else set()
    rows: dict[str, dict] = {}
    for name, series in counts.items():
        if series.empty:
            continue
        molecules = int(series.sum())
        novel = [s for s in series.index if s not in prior]
        graph = graphs.get(name, pd.Series(dtype=int))
        rows[name] = {
            "molecules": molecules,
            "scaffolds": int(series.size),
            "graph_frameworks": int(graph.size),
            "unique_fraction": round(series.size / molecules, 4),
            "top_share": round(float(series.iloc[0]) / molecules, 4),
            "de_novo_scaffolds": "n/a" if name == "prior" or not prior else len(novel),
            "de_novo_fraction": (
                "n/a" if name == "prior" or not prior else round(len(novel) / series.size, 4)
            ),
        }
    return rows


def pareto_rows(
    arms: dict[str, pd.DataFrame], objectives: list[str], config: dict | None = None
) -> dict[str, dict]:
    """Non-dominated front counts and dominated volume per arm for figure 02 and report.md."""
    rows: dict[str, dict] = {}
    prior = arms.get("prior")
    for name in ARMS:
        if name not in arms:
            continue
        pts = scored_points(arms[name], objectives, config, prior=prior)
        if not len(pts):
            continue
        front = int(pareto_mask(pts).sum())
        rows[name] = {
            "n scored": len(pts),
            "non-dominated": front,
            "front_fraction": round(front / len(pts), 4),
            "dominated_volume": round(dominated_volume(pts), 4),
        }
    return rows


# --------------------------------------------------------------------------------------
# 2D MCS-aligned depiction and 3D inertial-frame alignment
# --------------------------------------------------------------------------------------


class AlignedDepicter:
    """Renders 2D depictions aligned onto one shared core template (fast MCS, bounded timeout)."""

    def __init__(self, reference: str | Chem.Mol | None) -> None:
        if isinstance(reference, str):
            self.ref_mol = Chem.MolFromSmiles(reference) if reference else None
        else:
            self.ref_mol = reference
        if self.ref_mol is not None and self.ref_mol.GetNumConformers() == 0:
            rdDepictor.Compute2DCoords(self.ref_mol)
        self.ref_core = MurckoScaffold.GetScaffoldForMol(self.ref_mol) if self.ref_mol else None
        if self.ref_core and self.ref_core.GetNumHeavyAtoms() > 0:
            rdDepictor.Compute2DCoords(self.ref_core)
        elif self.ref_mol:
            self.ref_core = self.ref_mol
        self.template_mol: Chem.Mol | None = None

    def prepare_template(self, molecules: Sequence[Chem.Mol | None]) -> None:
        mols = [Chem.RemoveHs(Chem.Mol(m)) for m in molecules if m is not None]
        if not mols:
            self.template_mol = None
            return
        core = self.ref_core
        if (
            core is not None
            and core.GetNumHeavyAtoms() >= 4
            and all(m.HasSubstructMatch(core) for m in mols)
        ):
            template = Chem.Mol(core)
            rdDepictor.Compute2DCoords(template)
            self.template_mol = template
            return
        self.template_mol = None

    def align_molecule(self, mol: Chem.Mol) -> Chem.Mol:
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)
        for template in (self.template_mol, self.ref_core):
            if template is None or template.GetNumHeavyAtoms() == 0:
                continue
            if not mol_copy.HasSubstructMatch(template):
                continue
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol_copy, template)
                return mol_copy
            except (RuntimeError, ValueError):
                continue
        if self.ref_mol is not None and self.ref_mol.GetNumHeavyAtoms() >= 6:
            try:
                mcs_res = rdFMCS.FindMCS(
                    [self.ref_mol, mol_copy],
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                    timeout=1,
                )
                if mcs_res and mcs_res.numAtoms >= 6:
                    mcs_core = Chem.MolFromSmarts(mcs_res.smartsString)
                    if (
                        mcs_core is not None
                        and self.ref_mol.HasSubstructMatch(mcs_core)
                        and mol_copy.HasSubstructMatch(mcs_core)
                    ):
                        rdDepictor.GenerateDepictionMatching2DStructure(
                            mol_copy, self.ref_mol, refPatt=mcs_core
                        )
                        return mol_copy
            except (RuntimeError, ValueError):
                pass
        return mol_copy


def align_lead_to_bank(
    smiles: str,
    refs: list[Chem.Mol],
    generator,
    color_ff: str = "fdef",
    alignment_shape_weight: float = 0.5,
) -> tuple[Chem.Mol, Chem.Mol, float, float, float, int] | None:
    """Generate conformers with the run's generator and find the best overlay across the bank."""
    from drugex.training.scorers.rocs_rdkit import (
        _AXIS_SIGN_FLIPS,
        _annotate_color_features,
        _canonicalize_conformer_inplace,
        _flip_conformer,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = generator.genConformers([smiles], tmp)
        poses = [m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None]
    if not poses or not refs:
        return None

    prep_refs = []
    for ref in refs:
        rc = Chem.Mol(ref)
        for cid in [c.GetId() for c in rc.GetConformers()]:
            _canonicalize_conformer_inplace(rc, cid)
        if color_ff != "default" and not rc.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(rc, color_ff=color_ff)
        prep_refs.append(rc)

    best_combo = -1.0
    best_result = None
    for pose in poses:
        probe_base = Chem.Mol(pose)
        for cid in [c.GetId() for c in probe_base.GetConformers()]:
            _canonicalize_conformer_inplace(probe_base, cid)
        if color_ff != "default" and not probe_base.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(probe_base, color_ff=color_ff)

        for ref_idx, ref_mol in enumerate(prep_refs):
            ref_cid = ref_mol.GetConformer().GetId()
            for probe_conf in probe_base.GetConformers():
                probe_cid = probe_conf.GetId()
                for signs in _AXIS_SIGN_FLIPS:
                    trial = _flip_conformer(probe_base, probe_cid, signs)
                    try:
                        res = rdShapeAlign.AlignMol(
                            ref_mol,
                            trial,
                            refConfId=ref_cid,
                            probeConfId=probe_cid,
                            useColors=True,
                            opt_param=float(alignment_shape_weight),
                        )
                    except Exception:
                        continue
                    if not isinstance(res, (tuple, list)) or len(res) < 2:
                        continue
                    sh, col = float(res[0]), float(res[1])
                    if not (math.isfinite(sh) and math.isfinite(col)):
                        continue
                    combo = sh + col
                    if combo > best_combo:
                        best_combo = combo
                        best_result = (Chem.Mol(ref_mol), trial, combo, sh, col, ref_idx)
    return best_result
