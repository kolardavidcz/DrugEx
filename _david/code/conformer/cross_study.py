"""Large-sample 3D cross-molecule & self-recovery ROCS evaluation on prior molecules.

    python -m _david.code.conformer.cross_study
    python -m _david.code.conformer.cross_study --set methods='ETKDG,MMFF94s_e78.4,UFF'

Samples 1,000 unique valid pipeline-eligible molecules from the pretrained DrugEx prior
(``Papyrus05.5_smiles_rnn_PT``), draws 50 random reference molecules, and evaluates all
``50 x 1,000 = 50,000`` pairs across methods (ETKDG unminimised, MMFF94s eps=78.4, UFF) and
a ``(K reference conformers x C candidate conformers)`` grid using RDKit ``rdShapeAlign``
with ``color_ff="default"`` and ``alignment_shape_weight=0.5``.

Exploits two exact mathematical invariants to evaluate the entire ``(K, C)`` grid in a single
alignment sweep without approximation:

1. ``study.select_bank_ordered(pool, max_K)`` returns a nested sequence: slicing the first ``K``
   entries equals the exact set chosen by ``study.select_bank(pool, K)``.
2. RDKit ``EmbedMultipleConfs(..., numConfs=max_C)`` with fixed ``randomSeed`` produces
   per-isomer conformer indices ``0 .. max_C - 1`` as exact prefixes of smaller ``C`` calls.

Results are written to one self-contained directory under ``_david/output/conformer/``:
``config.json``, ``sampling_summary.json``, ``molecules.tsv``, ``cross_grid_summary.tsv``,
``timing_summary.tsv``, ``self_recovery.tsv``, ``default_cell_pairs.tsv``, ``pair_scores.npz``,
and the five diagnostic figures rendered by :mod:`_david.code.conformer.cross_figures`.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import dataclasses
import json
import logging
import multiprocessing
import queue as queue_mod
import random
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, rdFingerprintGenerator, rdMolDescriptors, rdShapeAlign
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from _david.code.config import PROJECT_ROOT
from _david.code.config import Config as PipelineConfig
from _david.code.conformer.config import METHODS, Method, StudyConfig
from _david.code.conformer.study import (
    CANDIDATE_OFFSET,
    _energy,
    _relax,
    force_field,
    select_bank_ordered,
)
from _david.code.scorers.rocs import _annotate_color_features
from drugex.training.scorers.conformer_generators import _RDKIT_TAUTOMER_ENUMERATOR
from drugex.training.scorers.parallel import capped_n_jobs, pool_context
from drugex.training.scorers.rocs_rdkit import (
    _AXIS_SIGN_FLIPS,
    _canonicalize_conformer_inplace,
    _flip_conformer,
)
from drugex.training.scorers.sascorer import calculateScore as calculate_sa

logger = logging.getLogger(__name__)

FLEXIBILITY_ORDER = ("rigid", "intermediate", "flexible")


@dataclasses.dataclass(frozen=True)
class CrossStudyConfig:
    name: str = "prior1000_cross"  # directory suffix label
    n_molecules: int = 1000  # unique valid in-bounds molecules sampled from prior
    n_references: int = 50  # random references selected from the 1,000-molecule pool
    methods: tuple[str, ...] = ("ETKDG", "MMFF94s_e78.4", "UFF")  # ^ keys of METHODS
    reference_caps: tuple[int, ...] = (1, 2, 4, 8)  # K: conformers kept in reference bank
    candidate_counts: tuple[int, ...] = (5, 10, 20, 30)  # C: candidate embed attempts/isomer

    # seeding
    sample_seed: int = 42  # RNG seed when sampling SMILES from the prior model
    ref_select_seed: int = 42  # RNG seed choosing the 50 references from the 1,000 molecules
    reference_seed: int = 104729  # ETKDG seed for reference pools; cands add CANDIDATE_OFFSET

    # conformer & pipeline bounds, matched to Config / StudyConfig
    reference_attempts: int = 200
    reference_rmsd: float = 0.5
    max_isomers: int = 4
    max_iterations: int = 500
    prune_rms: float = 0.5
    max_heavy_atoms: int = 45
    max_rotatable_bonds: int = 15
    embed_timeout: int = 30  # per-isomer ETKDGv3 cap (s), matching Config.timeout; 0 = unlimited

    # ROCS scoring (explicitly default colouring and equal shape/colour weight 0.5)
    alignment_shape_weight: float = 0.5
    color_ff: str = "default"

    # flexibility taxonomy thresholds on RDKit rotatable bond count
    rigid_max_rot: int = 2  # RotB <= 2 -> rigid; 3..6 -> intermediate; >= 7 -> flexible
    flexible_min_rot: int = 7

    # representative cell dumped in full (50,000 pair rows)
    representative_k: int = 4
    representative_c: int = 20

    # execution
    n_jobs: int = -1
    output_root: Path = PROJECT_ROOT / "_david" / "output" / "conformer"

    def method(self, key: str) -> Method:
        try:
            return METHODS[key]
        except KeyError:
            raise ValueError(
                f"unknown method {key!r}; known methods are {', '.join(sorted(METHODS))}"
            ) from None

    def as_study_config(self, method_key: str = "ETKDG") -> StudyConfig:
        return StudyConfig(
            name=self.name,
            methods=(method_key,),
            reference_caps=self.reference_caps,
            candidate_counts=self.candidate_counts,
            repeats=1,
            reference_attempts=self.reference_attempts,
            reference_rmsd=self.reference_rmsd,
            max_isomers=self.max_isomers,
            max_iterations=self.max_iterations,
            prune_rms=self.prune_rms,
            alignment_shape_weight=self.alignment_shape_weight,
            color_ff=self.color_ff,
            n_jobs=self.n_jobs,
            output_root=self.output_root,
        )

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("at least one method is needed")
        for key in self.methods:
            self.method(key)
        if self.n_molecules < 2:
            raise ValueError(f"n_molecules must be >= 2, got {self.n_molecules!r}")
        if not (1 <= self.n_references <= self.n_molecules):
            raise ValueError(
                f"n_references must lie in [1, {self.n_molecules}], got {self.n_references!r}"
            )
        if any(k < 1 for k in self.reference_caps):
            raise ValueError(f"reference_caps must be >= 1, got {self.reference_caps!r}")
        if any(c < 1 for c in self.candidate_counts):
            raise ValueError(f"candidate_counts must be >= 1, got {self.candidate_counts!r}")
        if max(self.reference_caps) > self.reference_attempts:
            raise ValueError(
                f"reference_caps asks for {max(self.reference_caps)} > {self.reference_attempts}"
            )
        if self.embed_timeout < 0:
            raise ValueError(f"embed_timeout must be >= 0, got {self.embed_timeout!r}")
        if self.rigid_max_rot >= self.flexible_min_rot:
            raise ValueError("rigid_max_rot must be < flexible_min_rot")
        if not 0.0 <= self.alignment_shape_weight <= 1.0:
            raise ValueError("alignment_shape_weight must lie in [0, 1]")


def classify_flexibility(rot_bonds: int, cfg: CrossStudyConfig) -> str:
    if rot_bonds <= cfg.rigid_max_rot:
        return "rigid"
    if rot_bonds >= cfg.flexible_min_rot:
        return "flexible"
    return "intermediate"


def _cross_isomers(smiles: str, cap: int) -> list[Chem.Mol]:
    """Tautomer-canonicalise then enumerate unassigned stereoisomers, matching pipeline."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    with contextlib.suppress(Exception):
        mol = _RDKIT_TAUTOMER_ENUMERATOR.Canonicalize(mol)
    options = StereoEnumerationOptions(maxIsomers=cap, onlyUnassigned=True, unique=True)
    if hasattr(options, "rand"):
        options.rand = 0xC0FFEE
    out = [Chem.AddHs(iso) for iso in EnumerateStereoisomers(mol, options=options)]
    return out or [Chem.AddHs(mol)]


def _etkdg_params(seed: int, cfg: CrossStudyConfig):
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.pruneRmsThresh = cfg.prune_rms
    if cfg.embed_timeout > 0:
        params.timeout = int(cfg.embed_timeout)
    return params


def _embed_ids(isomer: Chem.Mol, num_confs: int, params) -> tuple[list[int], bool]:
    """Embed conformers and filter RDKit's `-1` timeout sentinel."""
    raw = list(AllChem.EmbedMultipleConfs(isomer, numConfs=num_confs, params=params))
    valid = [int(c) for c in raw if int(c) >= 0]
    timed_out = any(int(c) < 0 for c in raw)
    return valid, timed_out


def _embed_tagged(smiles: str, attempts: int, seed: int, cfg: CrossStudyConfig) -> list[Chem.Mol]:
    """Embed ``attempts`` per stereoisomer, tagging each single-conformer Mol with ``LocalConfId``."""
    params = _etkdg_params(seed, cfg)
    out: list[Chem.Mol] = []
    for iso_idx, isomer in enumerate(_cross_isomers(smiles, cfg.max_isomers)):
        ids, _ = _embed_ids(isomer, attempts, params)
        for local_idx, conf_id in enumerate(ids):
            single = Chem.Mol(isomer, confId=int(conf_id))
            single.SetIntProp("IsomerIdx", iso_idx)
            single.SetIntProp("LocalConfId", local_idx)
            out.append(single)
    return out


def _annotate_if_needed(mols: list[Chem.Mol], color_ff: str) -> None:
    if color_ff == "default":
        return
    for mol in mols:
        if not mol.HasProp("PUBCHEM_PHARMACOPHORE_FEATURES"):
            _annotate_color_features(mol, color_ff=color_ff)


def _sample_prior_child(queue, target_n: int, seed: int, max_hac: int, max_rot: int) -> None:
    """Run prior sampling inside an isolated child process so PyTorch/CUDA never touches parent."""
    try:
        from _david.code.run import load_model, sample, seed_everything
        from drugex.data.corpus.vocabulary import VocSmiles

        seed_everything(seed)
        pipe_cfg = PipelineConfig()
        ckpt_path, voc_path = pipe_cfg.pretrained()
        voc = VocSmiles.fromFile(str(voc_path), False)
        prior = load_model(voc, ckpt_path)

        seen: set[str] = set()
        accepted: list[str] = []
        counts = Counter()
        batch_size = max(500, target_n)
        max_batches = 40

        for _ in range(max_batches):
            if len(accepted) >= target_n:
                break
            draws = sample(prior, batch_size, batch=500)
            for smi in draws:
                counts["raw_draws"] += 1
                mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
                if mol is None:
                    counts["invalid_smiles"] += 1
                    continue
                counts["valid_smiles"] += 1
                can = Chem.MolToSmiles(mol, canonical=True)
                if can in seen:
                    counts["duplicate_canonical"] += 1
                    continue
                seen.add(can)
                counts["unique_canonical"] += 1
                hac = mol.GetNumHeavyAtoms()
                rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
                if hac > max_hac or rot > max_rot:
                    counts["out_of_pipeline_bounds"] += 1
                    continue
                counts["accepted"] += 1
                accepted.append(can)
                if len(accepted) >= target_n:
                    break

        queue.put(("ok", accepted, dict(counts)))
    except Exception:
        queue.put(("err", traceback.format_exc(), {}))


def _filter_smiles_list(
    raw_smiles: list[str], target_n: int, max_hac: int, max_rot: int
) -> tuple[list[str], dict[str, int]]:
    seen: set[str] = set()
    accepted: list[str] = []
    counts = Counter()
    for smi in raw_smiles:
        counts["raw_draws"] += 1
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            counts["invalid_smiles"] += 1
            continue
        counts["valid_smiles"] += 1
        can = Chem.MolToSmiles(mol, canonical=True)
        if can in seen:
            counts["duplicate_canonical"] += 1
            continue
        seen.add(can)
        counts["unique_canonical"] += 1
        if mol.GetNumHeavyAtoms() > max_hac or rdMolDescriptors.CalcNumRotatableBonds(mol) > max_rot:
            counts["out_of_pipeline_bounds"] += 1
            continue
        counts["accepted"] += 1
        accepted.append(can)
        if len(accepted) >= target_n:
            break
    return accepted, dict(counts)


def acquire_cohort(
    cfg: CrossStudyConfig, smiles_file: Path | None = None
) -> tuple[pd.DataFrame, list[int], np.ndarray, dict[str, int]]:
    """Sample or load ``n_molecules``, compute 2D descriptors & ECFP4 matrix, pick 50 refs."""
    if smiles_file is not None:
        raw = [line.strip().split()[0] for line in smiles_file.read_text().splitlines() if line.strip()]
        smiles_list, funnel = _filter_smiles_list(
            raw, cfg.n_molecules, cfg.max_heavy_atoms, cfg.max_rotatable_bonds
        )
    else:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_sample_prior_child,
            args=(
                queue,
                cfg.n_molecules,
                cfg.sample_seed,
                cfg.max_heavy_atoms,
                cfg.max_rotatable_bonds,
            ),
        )
        proc.start()
        try:
            status, payload, funnel = queue.get(timeout=300)
        except queue_mod.Empty:
            proc.terminate()
            proc.join(timeout=5)
            raise RuntimeError("prior sampling child timed out after 300s without returning") from None
        proc.join(timeout=30)
        if status == "err":
            raise RuntimeError(f"prior sampling child failed:\n{payload}")
        if proc.exitcode != 0:
            raise RuntimeError(f"prior sampling child exited with code {proc.exitcode}")
        smiles_list = payload

    if len(smiles_list) < cfg.n_molecules:
        raise RuntimeError(
            f"only obtained {len(smiles_list)} eligible molecules, needed {cfg.n_molecules}"
        )

    rng = random.Random(cfg.ref_select_seed)
    ref_mol_indices = sorted(rng.sample(range(len(smiles_list)), cfg.n_references))
    ref_lookup = {mol_idx: ref_idx for ref_idx, mol_idx in enumerate(ref_mol_indices)}

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    rows = []
    for idx, can in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(can)
        rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
        hac = int(mol.GetNumHeavyAtoms())
        mw = float(rdMolDescriptors.CalcExactMolWt(mol))
        rings = int(rdMolDescriptors.CalcNumRings(mol))
        arom_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        hba = int(rdMolDescriptors.CalcNumHBA(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
        qed_val = float(QED.qed(mol))
        sa_val = float(calculate_sa(mol))
        n_iso = len(_cross_isomers(can, cfg.max_isomers))
        fps.append(morgan_gen.GetFingerprint(mol))
        rows.append(
            {
                "mol_idx": idx,
                "canonical_smiles": can,
                "is_reference": int(idx in ref_lookup),
                "ref_idx": ref_lookup.get(idx, -1),
                "rot_bonds": rot,
                "flexibility_class": classify_flexibility(rot, cfg),
                "hac": hac,
                "mw": round(mw, 3),
                "ring_count": rings,
                "aromatic_rings": arom_rings,
                "hbd": hbd,
                "hba": hba,
                "tpsa": round(tpsa, 2),
                "fsp3": round(fsp3, 4),
                "qed": round(qed_val, 4),
                "sa_raw": round(sa_val, 4),
                "n_isomers": n_iso,
            }
        )

    # Exact 50 x 1000 2D ECFP4 Tanimoto similarity matrix
    ecfp4 = np.zeros((len(ref_mol_indices), len(fps)), dtype=np.float32)
    for r_idx, mol_idx in enumerate(ref_mol_indices):
        sims = DataStructs.BulkTanimotoSimilarity(fps[mol_idx], fps)
        ecfp4[r_idx, :] = np.asarray(sims, dtype=np.float32)

    return pd.DataFrame(rows), ref_mol_indices, ecfp4, funnel


def _serialize_confs(mols: list[Chem.Mol]) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Pack conformers with exact float64 coordinates (``Mol.ToBinary`` rounds by ~1e-7)."""
    out = []
    for mol in mols:
        cf = mol.GetConformer()
        iso_idx = mol.GetIntProp("IsomerIdx") if mol.HasProp("IsomerIdx") else 0
        local_id = mol.GetIntProp("LocalConfId") if mol.HasProp("LocalConfId") else 0
        c_tier = mol.GetIntProp("CTier") if mol.HasProp("CTier") else 0
        out.append(
            (iso_idx, local_id, c_tier, int(cf.GetId()), np.array(cf.GetPositions(), dtype=np.float64))
        )
    return out


def _deserialize_confs(
    smiles: str, packed: list[tuple[int, int, int, int, np.ndarray]], max_isomers: int, color_ff: str
) -> list[Chem.Mol]:
    bases = _cross_isomers(smiles, max_isomers)
    _annotate_if_needed(bases, color_ff)
    out = []
    for iso_idx, local_id, c_tier, conf_id, pos in packed:
        base = Chem.Mol(bases[iso_idx])
        base.RemoveAllConformers()
        conf = Chem.Conformer(base.GetNumAtoms())
        conf.Set3D(True)
        conf.SetId(conf_id)
        for atom_i in range(pos.shape[0]):
            conf.SetAtomPosition(atom_i, (float(pos[atom_i, 0]), float(pos[atom_i, 1]), float(pos[atom_i, 2])))
        base.AddConformer(conf, assignId=False)
        base.SetIntProp("IsomerIdx", iso_idx)
        base.SetIntProp("LocalConfId", local_id)
        base.SetIntProp("CTier", c_tier)
        out.append(base)
    return out


def _prepare_reference_task(args: tuple[int, str, str, CrossStudyConfig]) -> dict:
    ref_idx, smiles, method_key, cfg = args
    method = cfg.method(method_key)
    study_cfg = cfg.as_study_config(method_key)
    caps = sorted(cfg.reference_caps)
    max_k = caps[-1]

    t0, c0 = time.perf_counter(), time.process_time()
    embedded = _embed_tagged(smiles, cfg.reference_attempts, cfg.reference_seed, cfg)
    embed_wall, embed_cpu = time.perf_counter() - t0, time.process_time() - c0

    t1, c1 = time.perf_counter(), time.process_time()
    pool = _relax(embedded, method, study_cfg)
    relax_wall, relax_cpu = time.perf_counter() - t1, time.process_time() - c1

    parameterised = (
        sum(force_field(mol, method, study_cfg) is not None for mol in embedded)
        if method.force_field
        else len(embedded)
    )

    bank_step_times: dict[int, tuple[float, float]] = {}
    ordered = select_bank_ordered(
        pool, max_k, method, study_cfg, caps_to_time=tuple(caps), cap_times=bank_step_times
    )
    bank_wall = {k: bank_step_times[k][0] for k in caps}
    bank_cpu = {k: bank_step_times[k][1] for k in caps}
    bank_sizes = {k: min(len(ordered), k) for k in caps}

    return {
        "ref_idx": ref_idx,
        "smiles": smiles,
        "pool_confs": len(pool),
        "embedded": len(embedded),
        "parameterised": parameterised,
        "bank_confs": _serialize_confs(ordered),
        "bank_sizes": bank_sizes,
        "embed_wall": embed_wall,
        "embed_cpu": embed_cpu,
        "relax_wall": relax_wall,
        "relax_cpu": relax_cpu,
        "bank_wall": bank_wall,
        "bank_cpu": bank_cpu,
        "lowest_energy": round(min((_energy(m, method, study_cfg) for m in pool), default=float("inf")), 4)
        if method.force_field and pool
        else None,
    }


def _prepare_candidate_task(args: tuple[int, str, str, CrossStudyConfig]) -> dict:
    mol_idx, smiles, method_key, cfg = args
    method = cfg.method(method_key)
    study_cfg = cfg.as_study_config(method_key)
    counts = sorted(cfg.candidate_counts)
    max_c = counts[-1]
    cand_seed = cfg.reference_seed + CANDIDATE_OFFSET

    params = _etkdg_params(cand_seed, cfg)
    isomers = _cross_isomers(smiles, cfg.max_isomers)

    # Determine each surviving conformer's true earliest C-tier even when pruneRmsThresh
    # prunes attempts (so M(C) <= C), and time each embedding budget directly while only
    # running expensive force-field relaxation once on the max-C pool.
    embed_wall: dict[int, float] = {}
    embed_cpu: dict[int, float] = {}
    seen_cutoff: dict[tuple[int, int], int] = {}
    failed_iso_time: dict[int, tuple[float, float]] = {}
    iso_timeout_hits = 0
    embedded_max: list[Chem.Mol] = []

    for ci, count in enumerate(counts):
        w_acc, c_acc = 0.0, 0.0
        for iso_idx, isomer in enumerate(isomers):
            if iso_idx in failed_iso_time:
                dw, dc = failed_iso_time[iso_idx]
                w_acc += dw
                c_acc += dc
                continue
            working = isomer if count == max_c else Chem.Mol(isomer)
            t0, c0 = time.perf_counter(), time.process_time()
            ids, timed_out = _embed_ids(working, count, params)
            dw, dc = time.perf_counter() - t0, time.process_time() - c0
            w_acc += dw
            c_acc += dc
            if timed_out and not ids:
                failed_iso_time[iso_idx] = (dw, dc)
                iso_timeout_hits += 1
            for local_idx, conf_id in enumerate(ids):
                tier = seen_cutoff.setdefault((iso_idx, local_idx), ci)
                if count == max_c:
                    single = Chem.Mol(working, confId=int(conf_id))
                    single.SetIntProp("IsomerIdx", iso_idx)
                    single.SetIntProp("LocalConfId", local_idx)
                    single.SetIntProp("CTier", tier)
                    embedded_max.append(single)
        embed_wall[count] = w_acc
        embed_cpu[count] = c_acc

    tier_relax_wall = [0.0] * len(counts)
    tier_relax_cpu = [0.0] * len(counts)
    relaxed_max: list[Chem.Mol] = []
    for mol in embedded_max:
        ct = mol.GetIntProp("CTier")
        t1, c1 = time.perf_counter(), time.process_time()
        rel = _relax([mol], method, study_cfg)[0]
        tier_relax_wall[ct] += time.perf_counter() - t1
        tier_relax_cpu[ct] += time.process_time() - c1
        relaxed_max.append(rel)

    relax_wall: dict[int, float] = {}
    relax_cpu: dict[int, float] = {}
    rw_acc = 0.0
    rc_acc = 0.0
    for ci, count in enumerate(counts):
        rw_acc += tier_relax_wall[ci]
        rc_acc += tier_relax_cpu[ci]
        relax_wall[count] = rw_acc
        relax_cpu[count] = rc_acc

    param_max = (
        sum(force_field(mol, method, study_cfg) is not None for mol in embedded_max)
        if method.force_field
        else len(embedded_max)
    )

    return {
        "mol_idx": mol_idx,
        "smiles": smiles,
        "confs_max": len(relaxed_max),
        "param_max": param_max,
        "embedded_max": len(embedded_max),
        "cand_confs": _serialize_confs(relaxed_max),
        "embed_wall": embed_wall,
        "embed_cpu": embed_cpu,
        "relax_wall": relax_wall,
        "relax_cpu": relax_cpu,
        "iso_timeouts": iso_timeout_hits,
        "actual_wall_total": sum(embed_wall.values()) + rw_acc,
    }


# Worker globals populated once per pool for ultra-low IPC ROCS sweep
_W_REF_TIERS: list[list[list]] = []  # [ref_idx][k_tier] -> list[ShapeInput]
_W_CAPS: tuple[int, ...] = ()
_W_COUNTS: tuple[int, ...] = ()
_W_OPT_PARAM: float = 0.5
_W_COLOR_FF: str = "default"
_W_MAX_ISOMERS: int = 4
_W_OPTS = None


def _tier_index(value: int, sorted_thresholds: tuple[int, ...]) -> int:
    """Map 0-indexed rank/id into tier idx `t` where `sorted_thresholds[t]` is first inclusive bound."""
    return bisect.bisect_right(sorted_thresholds, value)


def _score_worker_init(
    ref_banks_packed: list[tuple[str, list[tuple[int, int, int, int, np.ndarray]]]],
    caps: tuple[int, ...],
    counts: tuple[int, ...],
    opt_param: float,
    color_ff: str,
    max_isomers: int,
) -> None:
    global _W_REF_TIERS, _W_CAPS, _W_COUNTS, _W_OPT_PARAM, _W_COLOR_FF, _W_MAX_ISOMERS, _W_OPTS
    _W_CAPS = tuple(sorted(caps))
    _W_COUNTS = tuple(sorted(counts))
    _W_OPT_PARAM = float(opt_param)
    _W_COLOR_FF = color_ff
    _W_MAX_ISOMERS = int(max_isomers)
    _W_OPTS = rdShapeAlign.ShapeInputOptions()

    tiers_all: list[list[list]] = []
    n_kt = len(_W_CAPS)
    for ref_smiles, packed in ref_banks_packed:
        mols = _deserialize_confs(ref_smiles, packed, _W_MAX_ISOMERS, color_ff)
        by_tier: list[list] = [[] for _ in range(n_kt)]
        for rank, mol in enumerate(mols):
            kt = _tier_index(rank, _W_CAPS)
            if kt >= n_kt:
                continue
            cid = mol.GetConformer().GetId()
            _canonicalize_conformer_inplace(mol, cid)
            by_tier[kt].append(rdShapeAlign.PrepareConformer(mol, cid, _W_OPTS))
        tiers_all.append(by_tier)
    _W_REF_TIERS = tiers_all


def _score_candidate_worker(task: tuple[int, str, list[tuple[int, int, int, int, np.ndarray]]]) -> dict:
    """Align one candidate's conformers against all 50 reference banks across (k_tier, c_tier)."""
    mol_idx, cand_smiles, packed = task
    n_refs = len(_W_REF_TIERS)
    n_kt = len(_W_CAPS)
    n_ct = len(_W_COUNTS)

    best_shape = np.zeros((n_refs, n_kt, n_ct), dtype=np.float32)
    best_colour = np.zeros((n_refs, n_kt, n_ct), dtype=np.float32)
    best_total = np.full((n_refs, n_kt, n_ct), -1.0, dtype=np.float64)
    bucket_wall = np.zeros((n_kt, n_ct), dtype=np.float64)
    bucket_cpu = np.zeros((n_kt, n_ct), dtype=np.float64)
    align_calls = np.zeros((n_kt, n_ct), dtype=np.int64)

    # Group flipped candidate poses by C-tier once, accounting for deserialization & flip cost
    flipped_by_ct: list[list[tuple[Chem.Mol, int]]] = [[] for _ in range(n_ct)]
    td0, cd0 = time.perf_counter(), time.process_time()
    cand_mols = _deserialize_confs(cand_smiles, packed, _W_MAX_ISOMERS, _W_COLOR_FF)
    deser_w, deser_c = time.perf_counter() - td0, time.process_time() - cd0
    n_cands = max(len(cand_mols), 1)
    for mol in cand_mols:
        ct = mol.GetIntProp("CTier") if mol.HasProp("CTier") else 0
        if not (0 <= ct < n_ct):
            continue
        tp0, cp0 = time.perf_counter(), time.process_time()
        cid = mol.GetConformer().GetId()
        _canonicalize_conformer_inplace(mol, cid)
        for signs in _AXIS_SIGN_FLIPS:
            flipped_by_ct[ct].append((_flip_conformer(mol, cid, signs), cid))
        bucket_wall[0, ct] += (time.perf_counter() - tp0) + deser_w / n_cands
        bucket_cpu[0, ct] += (time.process_time() - cp0) + deser_c / n_cands

    opts = _W_OPTS
    opt_param = _W_OPT_PARAM
    for ct in range(n_ct):
        poses = flipped_by_ct[ct]
        if not poses:
            continue
        for kt in range(n_kt):
            w0, c0 = time.perf_counter(), time.process_time()
            calls_here = 0
            for r_idx in range(n_refs):
                r_sis = _W_REF_TIERS[r_idx][kt]
                if not r_sis:
                    continue
                b_s, b_c, b_tot = 0.0, 0.0, -1.0
                for fp, cid in poses:
                    for r_si in r_sis:
                        # PrepareConformer right before AlignShapes: AlignShapes mutates probeShape.
                        p_si = rdShapeAlign.PrepareConformer(fp, cid, opts)
                        try:
                            res = rdShapeAlign.AlignShapes(r_si, p_si, opt_param=opt_param)
                        except (RuntimeError, ValueError):
                            continue
                        calls_here += 1
                        s, c = float(res[0]), float(res[1])
                        tot = s + c
                        if tot > b_tot:
                            b_s, b_c, b_tot = s, c, tot
                if b_tot >= 0.0:
                    best_shape[r_idx, kt, ct] = b_s
                    best_colour[r_idx, kt, ct] = b_c
                    best_total[r_idx, kt, ct] = b_tot
            bucket_wall[kt, ct] += time.perf_counter() - w0
            bucket_cpu[kt, ct] += time.process_time() - c0
            align_calls[kt, ct] += calls_here

    # Cumulative 2D prefix max over (kt, ct) so cell (ki, ci) is exact max over <= ki, <= ci
    cum_shape = np.zeros((n_refs, n_kt, n_ct), dtype=np.float32)
    cum_colour = np.zeros((n_refs, n_kt, n_ct), dtype=np.float32)
    for r_idx in range(n_refs):
        for ki in range(n_kt):
            for ci in range(n_ct):
                # Best across all buckets (a <= ki, b <= ci)
                sub_tot = best_total[r_idx, : ki + 1, : ci + 1]
                flat_idx = int(np.argmax(sub_tot))
                if sub_tot.flat[flat_idx] >= 0.0:
                    a, b = divmod(flat_idx, ci + 1)
                    cum_shape[r_idx, ki, ci] = best_shape[r_idx, a, b]
                    cum_colour[r_idx, ki, ci] = best_colour[r_idx, a, b]

    return {
        "mol_idx": mol_idx,
        "cum_shape": cum_shape,
        "cum_colour": cum_colour,
        "bucket_wall": bucket_wall,
        "bucket_cpu": bucket_cpu,
        "align_calls": align_calls,
    }


def _band_stats(
    combo: np.ndarray, shape: np.ndarray, colour: np.ndarray, low_val: float, high_val: float
) -> dict[str, float]:
    n = int(combo.size)
    if n == 0:
        return {
            "n": 0,
            "cutoff_low": float("nan"),
            "cutoff_high": float("nan"),
            "median_combo": float("nan"),
            "mean_combo": float("nan"),
            "std_combo": float("nan"),
            "iqr_combo": float("nan"),
            "q25_combo": float("nan"),
            "q75_combo": float("nan"),
            "min_combo": float("nan"),
            "max_combo": float("nan"),
            "median_shape": float("nan"),
            "std_shape": float("nan"),
            "median_colour": float("nan"),
            "std_colour": float("nan"),
        }
    q25, med, q75 = np.percentile(combo, [25.0, 50.0, 75.0])
    return {
        "n": n,
        "cutoff_low": round(float(low_val), 6),
        "cutoff_high": round(float(high_val), 6),
        "median_combo": round(float(med), 6),
        "mean_combo": round(float(np.mean(combo)), 6),
        "std_combo": round(float(np.std(combo, ddof=1 if n > 1 else 0)), 6),
        "iqr_combo": round(float(q75 - q25), 6),
        "q25_combo": round(float(q25), 6),
        "q75_combo": round(float(q75), 6),
        "min_combo": round(float(np.min(combo)), 6),
        "max_combo": round(float(np.max(combo)), 6),
        "median_shape": round(float(np.median(shape)), 6),
        "std_shape": round(float(np.std(shape, ddof=1 if n > 1 else 0)), 6),
        "median_colour": round(float(np.median(colour)), 6),
        "std_colour": round(float(np.std(colour, ddof=1 if n > 1 else 0)), 6),
    }


def _summarize_cell_cross_bands(
    combo_mat: np.ndarray,  # [n_refs, n_mols]
    shape_mat: np.ndarray,
    colour_mat: np.ndarray,
    ref_mol_indices: list[int],
    cand_flex: np.ndarray,
    ref_flex: np.ndarray,
    method: str,
    cap: int,
    count: int,
) -> list[dict]:
    """Compute global cross bands, fraction sensitivity, per-ref bands, flex cohorts & self stats."""
    n_refs, n_mols = combo_mat.shape
    off_mask = np.ones((n_refs, n_mols), dtype=bool)
    self_combos, self_shapes, self_colours = [], [], []
    for r_idx, mol_idx in enumerate(ref_mol_indices):
        off_mask[r_idx, mol_idx] = False
        self_combos.append(float(combo_mat[r_idx, mol_idx]))
        self_shapes.append(float(shape_mat[r_idx, mol_idx]))
        self_colours.append(float(colour_mat[r_idx, mol_idx]))

    c_cross = combo_mat[off_mask]
    s_cross = shape_mat[off_mask]
    k_cross = colour_mat[off_mask]

    p50, p75, p85, p90, p95 = np.percentile(c_cross, [50.0, 75.0, 85.0, 90.0, 95.0])
    bands: list[tuple[str, np.ndarray, float, float]] = [
        ("ge_p85", c_cross >= p85, p85, float(c_cross.max())),
        ("p50_to_p85", (c_cross >= p50) & (c_cross < p85), p50, p85),
        ("lt_p50_noise", c_cross < p50, float(c_cross.min()), p50),
        ("ge_p95", c_cross >= p95, p95, float(c_cross.max())),
        ("ge_p90", c_cross >= p90, p90, float(c_cross.max())),
        ("ge_p75", c_cross >= p75, p75, float(c_cross.max())),
        ("ge_p50", c_cross >= p50, p50, float(c_cross.max())),
        ("p50_to_p90", (c_cross >= p50) & (c_cross < p90), p50, p90),
        ("p75_to_p95", (c_cross >= p75) & (c_cross < p95), p75, p95),
        ("all_cross", np.ones_like(c_cross, dtype=bool), float(c_cross.min()), float(c_cross.max())),
    ]

    rows: list[dict] = []
    base = {"method": method, "reference_cap": cap, "candidate_count": count}
    for b_name, b_mask, lo, hi in bands:
        rows.append(
            {
                **base,
                "scope": "global_cross",
                "band": b_name,
                **_band_stats(c_cross[b_mask], s_cross[b_mask], k_cross[b_mask], lo, hi),
            }
        )

    # Per-reference p85 and p50-85 (matches RL prior-threshold definition, per reference)
    per_ref_ge85_meds, per_ref_mid_meds = [], []
    pooled_ref_ge85_c, pooled_ref_ge85_s, pooled_ref_ge85_k = [], [], []
    pooled_ref_mid_c, pooled_ref_mid_s, pooled_ref_mid_k = [], [], []
    for r_idx in range(n_refs):
        row_mask = off_mask[r_idx]
        rc = combo_mat[r_idx, row_mask]
        rs = shape_mat[r_idx, row_mask]
        rk = colour_mat[r_idx, row_mask]
        rp50, rp85 = np.percentile(rc, [50.0, 85.0])
        m85 = rc >= rp85
        mmid = (rc >= rp50) & (rc < rp85)
        if m85.any():
            per_ref_ge85_meds.append(float(np.median(rc[m85])))
            pooled_ref_ge85_c.append(rc[m85])
            pooled_ref_ge85_s.append(rs[m85])
            pooled_ref_ge85_k.append(rk[m85])
        if mmid.any():
            per_ref_mid_meds.append(float(np.median(rc[mmid])))
            pooled_ref_mid_c.append(rc[mmid])
            pooled_ref_mid_s.append(rs[mmid])
            pooled_ref_mid_k.append(rk[mmid])

    for b_name, c_parts, s_parts, k_parts, ref_meds in (
        ("ge_p85", pooled_ref_ge85_c, pooled_ref_ge85_s, pooled_ref_ge85_k, per_ref_ge85_meds),
        ("p50_to_p85", pooled_ref_mid_c, pooled_ref_mid_s, pooled_ref_mid_k, per_ref_mid_meds),
    ):
        pc = np.concatenate(c_parts) if c_parts else np.empty(0)
        ps = np.concatenate(s_parts) if s_parts else np.empty(0)
        pk = np.concatenate(k_parts) if k_parts else np.empty(0)
        stats = _band_stats(pc, ps, pk, float(pc.min()) if pc.size else float("nan"), float(pc.max()) if pc.size else float("nan"))
        stats["ref_level_median_of_medians"] = round(float(np.median(ref_meds)), 6) if ref_meds else float("nan")
        stats["ref_level_std_of_medians"] = round(float(np.std(ref_meds, ddof=1)), 6) if len(ref_meds) > 1 else float("nan")
        rows.append({**base, "scope": "per_ref_cross", "band": b_name, **stats})

    # Breakdown by candidate flexibility tier AND by reference flexibility tier
    for tier in FLEXIBILITY_ORDER:
        cmask = off_mask & (cand_flex[None, :] == tier)
        if cmask.any():
            vals = combo_mat[cmask]
            sh = shape_mat[cmask]
            co = colour_mat[cmask]
            tp50, tp85 = np.percentile(vals, [50.0, 85.0])
            for b_name, m, lo, hi in (
                ("ge_p85", vals >= tp85, tp85, float(vals.max())),
                ("p50_to_p85", (vals >= tp50) & (vals < tp85), tp50, tp85),
                ("all_cross", np.ones_like(vals, dtype=bool), float(vals.min()), float(vals.max())),
            ):
                rows.append(
                    {
                        **base,
                        "scope": f"cand_{tier}_cross",
                        "band": b_name,
                        **_band_stats(vals[m], sh[m], co[m], lo, hi),
                    }
                )
        rmask = off_mask & (ref_flex[:, None] == tier)
        if rmask.any():
            vals = combo_mat[rmask]
            sh = shape_mat[rmask]
            co = colour_mat[rmask]
            tp50, tp85 = np.percentile(vals, [50.0, 85.0])
            for b_name, m, lo, hi in (
                ("ge_p85", vals >= tp85, tp85, float(vals.max())),
                ("p50_to_p85", (vals >= tp50) & (vals < tp85), tp50, tp85),
                ("all_cross", np.ones_like(vals, dtype=bool), float(vals.min()), float(vals.max())),
            ):
                rows.append(
                    {
                        **base,
                        "scope": f"ref_{tier}_cross",
                        "band": b_name,
                        **_band_stats(vals[m], sh[m], co[m], lo, hi),
                    }
                )

    sc_arr = np.asarray(self_combos, dtype=float)
    ss_arr = np.asarray(self_shapes, dtype=float)
    sk_arr = np.asarray(self_colours, dtype=float)
    self_stats = _band_stats(sc_arr, ss_arr, sk_arr, float(sc_arr.min()), float(sc_arr.max()))
    self_stats["ceiling_rate"] = round(float(np.mean(sc_arr >= 2.0 - 1e-6)), 6)
    self_stats["p10_self_recovery"] = round(float(np.percentile(sc_arr, 10.0) / 2.0), 6)
    rows.append({**base, "scope": "self_50", "band": "all_self", **self_stats})

    for tier in FLEXIBILITY_ORDER:
        tmask = ref_flex == tier
        if tmask.any():
            t_stats = _band_stats(
                sc_arr[tmask], ss_arr[tmask], sk_arr[tmask], float(sc_arr[tmask].min()), float(sc_arr[tmask].max())
            )
            t_stats["ceiling_rate"] = round(float(np.mean(sc_arr[tmask] >= 2.0 - 1e-6)), 6)
            t_stats["p10_self_recovery"] = round(float(np.percentile(sc_arr[tmask], 10.0) / 2.0), 6)
            rows.append({**base, "scope": f"self_{tier}", "band": "all_self", **t_stats})

    return rows


def run(
    cfg: CrossStudyConfig, out_dir: Path, smiles_file: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the full cross-molecule & self-recovery study and write tables + `.npz`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = tuple(sorted(cfg.reference_caps))
    counts = tuple(sorted(cfg.candidate_counts))
    methods = tuple(cfg.methods)

    logger.info("Acquiring %d prior molecules and selecting %d random references...", cfg.n_molecules, cfg.n_references)
    molecules_df, ref_mol_indices, ecfp4_mat, sampling_funnel = acquire_cohort(cfg, smiles_file=smiles_file)
    smiles_list = molecules_df["canonical_smiles"].tolist()
    cand_flex = molecules_df["flexibility_class"].to_numpy()
    ref_flex = molecules_df.iloc[ref_mol_indices]["flexibility_class"].to_numpy()

    workers = capped_n_jobs(cfg.n_jobs if cfg.n_jobs > 0 else (multiprocessing.cpu_count() or 1))
    logger.info("Running across %d CPU workers", workers)

    n_m = len(methods)
    n_r = len(ref_mol_indices)
    n_q = len(smiles_list)
    n_k = len(caps)
    n_c = len(counts)

    shape_all = np.zeros((n_m, n_r, n_q, n_k, n_c), dtype=np.float32)
    colour_all = np.zeros((n_m, n_r, n_q, n_k, n_c), dtype=np.float32)

    grid_rows: list[dict] = []
    timing_rows: list[dict] = []
    self_rows: list[dict] = []
    method_telemetry: dict[str, dict] = {}

    # Track representative cell (default K=4, C=20, or closest present)
    rep_k = min(caps, key=lambda k: abs(k - cfg.representative_k))
    rep_c = min(counts, key=lambda c: abs(c - cfg.representative_c))
    rep_ki = caps.index(rep_k)
    rep_ci = counts.index(rep_c)

    mp_ctx = pool_context("forkserver")

    def _lpt_rank(mol_idx: int) -> tuple[int, int, int, int]:
        m = Chem.MolFromSmiles(smiles_list[mol_idx])
        bh = int(rdMolDescriptors.CalcNumBridgeheadAtoms(m)) if m is not None else 0
        sp = int(rdMolDescriptors.CalcNumSpiroAtoms(m)) if m is not None else 0
        row = molecules_df.iloc[mol_idx]
        return (bh + sp, int(row["n_isomers"]), int(row["ring_count"]), int(row["hac"]))

    lpt_ranks = [_lpt_rank(i) for i in range(n_q)]

    for m_idx, method_key in enumerate(methods):
        logger.info("=== Method %s (%d/%d) ===", method_key, m_idx + 1, n_m)

        # 1. Reference pools & banks (LPT + chunksize=1 avoids slow-isomer tail)
        ref_tasks = sorted(
            [
                (r_idx, smiles_list[mol_idx], method_key, cfg)
                for r_idx, mol_idx in enumerate(ref_mol_indices)
            ],
            key=lambda t: lpt_ranks[ref_mol_indices[t[0]]],
            reverse=True,
        )
        t_ref_start = time.perf_counter()
        if workers <= 1 or len(ref_tasks) < 2:
            ref_results = [_prepare_reference_task(t) for t in ref_tasks]
        else:
            with mp_ctx.Pool(min(workers, len(ref_tasks))) as pool:
                ref_results = list(pool.imap_unordered(_prepare_reference_task, ref_tasks, chunksize=1))
        ref_phase_wall = time.perf_counter() - t_ref_start
        ref_results.sort(key=lambda r: r["ref_idx"])
        logger.info(
            "Refs %s: %d/%d ready (%.1fs, zero_bank=%d)",
            method_key,
            len(ref_results),
            len(ref_tasks),
            ref_phase_wall,
            sum(1 for r in ref_results if r["pool_confs"] == 0),
        )

        ref_banks_packed = [(r["smiles"], r["bank_confs"]) for r in ref_results]
        ref_prep_cpu = sum(r["embed_cpu"] + r["relax_cpu"] for r in ref_results)
        bank_cpu_by_k = {k: sum(r["bank_cpu"][k] for r in ref_results) for k in caps}
        total_ref_raw_wall = max(sum(r["embed_wall"] + r["relax_wall"] for r in ref_results), 1e-9)
        actual_bank_raw_wall = sum(r["bank_wall"][caps[-1]] for r in ref_results)
        prep_share = total_ref_raw_wall / max(total_ref_raw_wall + actual_bank_raw_wall, 1e-9)
        ref_prep_wall = ref_phase_wall * prep_share
        bank_wall_by_k = {
            k: ref_phase_wall * (1.0 - prep_share) * (sum(r["bank_wall"][k] for r in ref_results) / max(actual_bank_raw_wall, 1e-9))
            for k in caps
        }

        # Record per-molecule ref metrics on molecules_df
        ref_pool_col = np.full(n_q, -1, dtype=int)
        ref_bank_col = np.full(n_q, -1, dtype=int)
        for r in ref_results:
            mol_idx = ref_mol_indices[r["ref_idx"]]
            ref_pool_col[mol_idx] = r["pool_confs"]
            ref_bank_col[mol_idx] = r["bank_sizes"][caps[-1]]
        molecules_df[f"{method_key}_ref_pool_confs"] = ref_pool_col
        molecules_df[f"{method_key}_ref_bank_k{caps[-1]}"] = ref_bank_col

        # 2. Candidate conformer pools + genuine per-C timing (LPT puts bridged cages first)
        cand_tasks = sorted(
            [(mol_idx, smiles_list[mol_idx], method_key, cfg) for mol_idx in range(n_q)],
            key=lambda t: lpt_ranks[t[0]],
            reverse=True,
        )
        t_cand_start = time.perf_counter()
        cand_log_step = max(1, len(cand_tasks) // 5)
        if workers <= 1 or len(cand_tasks) < 2:
            cand_results = [_prepare_candidate_task(t) for t in cand_tasks]
        else:
            cand_results = []
            with mp_ctx.Pool(min(workers, len(cand_tasks))) as pool:
                for done_i, res in enumerate(
                    pool.imap_unordered(_prepare_candidate_task, cand_tasks, chunksize=1), 1
                ):
                    cand_results.append(res)
                    if done_i % cand_log_step == 0 or done_i == len(cand_tasks):
                        logger.info(
                            "Cand %s: %d/%d ready (%.1fs)",
                            method_key,
                            done_i,
                            len(cand_tasks),
                            time.perf_counter() - t_cand_start,
                        )
        cand_phase_wall = time.perf_counter() - t_cand_start
        cand_results.sort(key=lambda r: r["mol_idx"])

        cand_embed_cpu = {c: sum(r["embed_cpu"][c] for r in cand_results) for c in counts}
        cand_relax_cpu = {c: sum(r["relax_cpu"][c] for r in cand_results) for c in counts}
        actual_cand_raw_wall = max(sum(r["actual_wall_total"] for r in cand_results), 1e-9)
        cand_embed_wall = {
            c: cand_phase_wall * (sum(r["embed_wall"][c] for r in cand_results) / actual_cand_raw_wall)
            for c in counts
        }
        cand_relax_wall = {
            c: cand_phase_wall * (sum(r["relax_wall"][c] for r in cand_results) / actual_cand_raw_wall)
            for c in counts
        }

        molecules_df[f"{method_key}_cand_confs_c{counts[-1]}"] = [r["confs_max"] for r in cand_results]
        molecules_df[f"{method_key}_cand_param_frac"] = [
            round(r["param_max"] / max(r["embedded_max"], 1), 4) for r in cand_results
        ]
        cand_zero_confs = sum(1 for r in cand_results if r["confs_max"] == 0)
        cand_iso_timeouts = sum(int(r["iso_timeouts"]) for r in cand_results)

        # 3. 50 x 1000 ROCS incremental scoring sweep (LPT order + chunksize=1 removes straggler tail)
        score_tasks = [(r["mol_idx"], r["smiles"], r["cand_confs"]) for r in cand_results]
        del cand_results
        score_tasks.sort(key=lambda t: len(t[2]), reverse=True)
        bucket_wall = np.zeros((n_k, n_c), dtype=np.float64)
        bucket_cpu = np.zeros((n_k, n_c), dtype=np.float64)
        align_calls = np.zeros((n_k, n_c), dtype=np.int64)
        t_rocs_start = time.perf_counter()
        log_step = max(1, len(score_tasks) // 10)
        if workers <= 1 or len(score_tasks) < 2:
            _score_worker_init(
                ref_banks_packed,
                caps,
                counts,
                cfg.alignment_shape_weight,
                cfg.color_ff,
                cfg.max_isomers,
            )
            for done_i, t in enumerate(score_tasks, 1):
                res = _score_candidate_worker(t)
                q_idx = res["mol_idx"]
                shape_all[m_idx, :, q_idx, :, :] = res["cum_shape"]
                colour_all[m_idx, :, q_idx, :, :] = res["cum_colour"]
                bucket_wall += res["bucket_wall"]
                bucket_cpu += res["bucket_cpu"]
                align_calls += res["align_calls"]
                if done_i % log_step == 0 or done_i == len(score_tasks):
                    logger.info(
                        "ROCS %s: %d/%d candidates (%.1fs)",
                        method_key,
                        done_i,
                        len(score_tasks),
                        time.perf_counter() - t_rocs_start,
                    )
        else:
            with mp_ctx.Pool(
                min(workers, len(score_tasks)),
                initializer=_score_worker_init,
                initargs=(
                    ref_banks_packed,
                    caps,
                    counts,
                    cfg.alignment_shape_weight,
                    cfg.color_ff,
                    cfg.max_isomers,
                ),
            ) as pool:
                for done_i, res in enumerate(
                    pool.imap_unordered(_score_candidate_worker, score_tasks, chunksize=1), 1
                ):
                    q_idx = res["mol_idx"]
                    shape_all[m_idx, :, q_idx, :, :] = res["cum_shape"]
                    colour_all[m_idx, :, q_idx, :, :] = res["cum_colour"]
                    bucket_wall += res["bucket_wall"]
                    bucket_cpu += res["bucket_cpu"]
                    align_calls += res["align_calls"]
                    if done_i % log_step == 0 or done_i == len(score_tasks):
                        logger.info(
                            "ROCS %s: %d/%d candidates (%.1fs)",
                            method_key,
                            done_i,
                            len(score_tasks),
                            time.perf_counter() - t_rocs_start,
                        )
        rocs_phase_wall = time.perf_counter() - t_rocs_start

        total_bucket_wall = max(float(bucket_wall.sum()), 1e-9)
        cum_rocs_wall = np.cumsum(np.cumsum(bucket_wall, axis=0), axis=1) / total_bucket_wall * rocs_phase_wall
        cum_rocs_cpu = np.cumsum(np.cumsum(bucket_cpu, axis=0), axis=1)
        cum_calls = np.cumsum(np.cumsum(align_calls, axis=0), axis=1)

        method_telemetry[method_key] = {
            "ref_phase_wall_s": round(ref_phase_wall, 3),
            "cand_phase_wall_s": round(cand_phase_wall, 3),
            "rocs_phase_wall_s": round(rocs_phase_wall, 3),
            "ref_parameterised_confs": int(sum(r["parameterised"] for r in ref_results)),
            "ref_embedded_confs": int(sum(r["embedded"] for r in ref_results)),
            "zero_bank_refs": int(sum(1 for r in ref_results if r["pool_confs"] == 0)),
            "cand_zero_confs": cand_zero_confs,
            "cand_iso_timeouts": cand_iso_timeouts,
        }

        # 4. Aggregate summaries per (K, C)
        combo_method = shape_all[m_idx] + colour_all[m_idx]
        n_pairs = n_r * n_q
        for ki, cap in enumerate(caps):
            for ci, count in enumerate(counts):
                rocs_cpu = float(cum_rocs_cpu[ki, ci])
                rocs_wall = float(cum_rocs_wall[ki, ci])
                cand_cpu_here = float(cand_embed_cpu[count] + cand_relax_cpu[count])
                cand_wall_here = float(cand_embed_wall[count] + cand_relax_wall[count])
                bank_cpu_here = float(bank_cpu_by_k[cap])
                bank_wall_here = float(bank_wall_by_k[cap])
                total_cand_rocs_cpu = cand_cpu_here + rocs_cpu
                total_all_cpu = ref_prep_cpu + bank_cpu_here + total_cand_rocs_cpu
                total_wall = ref_prep_wall + bank_wall_here + cand_wall_here + rocs_wall
                timing_rows.append(
                    {
                        "method": method_key,
                        "reference_cap": cap,
                        "candidate_count": count,
                        "n_pairs": n_pairs,
                        "align_calls": int(cum_calls[ki, ci]),
                        "ref_prep_wall_s": round(ref_prep_wall, 3),
                        "ref_prep_cpu_s": round(ref_prep_cpu, 3),
                        "bank_select_wall_s": round(bank_wall_here, 3),
                        "bank_select_cpu_s": round(bank_cpu_here, 3),
                        "cand_embed_wall_s": round(cand_embed_wall[count], 3),
                        "cand_embed_cpu_s": round(cand_embed_cpu[count], 3),
                        "cand_relax_wall_s": round(cand_relax_wall[count], 3),
                        "cand_relax_cpu_s": round(cand_relax_cpu[count], 3),
                        "rocs_wall_s": round(rocs_wall, 3),
                        "rocs_cpu_s": round(rocs_cpu, 3),
                        "total_cand_plus_rocs_cpu_s": round(total_cand_rocs_cpu, 3),
                        "total_all_cpu_s": round(total_all_cpu, 3),
                        "total_wall_s": round(total_wall, 3),
                        "rocs_cpu_ms_per_pair": round(1000.0 * rocs_cpu / max(n_pairs, 1), 4),
                        "total_cpu_ms_per_pair": round(
                            1000.0 * total_cand_rocs_cpu / max(n_pairs, 1), 4
                        ),
                    }
                )

                grid_rows.extend(
                    _summarize_cell_cross_bands(
                        combo_method[:, :, ki, ci],
                        shape_all[m_idx, :, :, ki, ci],
                        colour_all[m_idx, :, :, ki, ci],
                        ref_mol_indices,
                        cand_flex,
                        ref_flex,
                        method_key,
                        cap,
                        count,
                    )
                )

                # Per-reference self-recovery rows
                for r_idx, mol_idx in enumerate(ref_mol_indices):
                    sh = float(shape_all[m_idx, r_idx, mol_idx, ki, ci])
                    co = float(colour_all[m_idx, r_idx, mol_idx, ki, ci])
                    cb = sh + co
                    self_rows.append(
                        {
                            "ref_idx": r_idx,
                            "mol_idx": mol_idx,
                            "canonical_smiles": smiles_list[mol_idx],
                            "rot_bonds": int(molecules_df.loc[mol_idx, "rot_bonds"]),
                            "flexibility_class": str(molecules_df.loc[mol_idx, "flexibility_class"]),
                            "method": method_key,
                            "reference_cap": cap,
                            "candidate_count": count,
                            "bank_size": int(ref_results[r_idx]["bank_sizes"][cap]),
                            "shape": round(sh, 6),
                            "colour": round(co, 6),
                            "combo": round(cb, 6),
                            "self_recovery": round(cb / 2.0, 6),
                            "at_ceiling": int(cb >= 2.0 - 1e-6),
                        }
                    )

    # Write outputs
    molecules_df.to_csv(out_dir / "molecules.tsv", sep="\t", index=False)
    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["method", "reference_cap", "candidate_count", "scope", "band"]
    )
    grid_df.to_csv(out_dir / "cross_grid_summary.tsv", sep="\t", index=False)

    timing_df = pd.DataFrame(timing_rows).sort_values(["method", "reference_cap", "candidate_count"])
    timing_df.to_csv(out_dir / "timing_summary.tsv", sep="\t", index=False)

    self_df = pd.DataFrame(self_rows).sort_values(
        ["method", "reference_cap", "candidate_count", "ref_idx"]
    )
    self_df.to_csv(out_dir / "self_recovery.tsv", sep="\t", index=False)

    # Wide table at representative (K=rep_k, C=rep_c) across all 50 x 1000 pairs
    rep_rows = []
    for r_idx, r_mol_idx in enumerate(ref_mol_indices):
        for q_idx in range(n_q):
            row = {
                "ref_idx": r_idx,
                "ref_mol_idx": r_mol_idx,
                "cand_idx": q_idx,
                "is_self": int(r_mol_idx == q_idx),
                "ref_flex": str(ref_flex[r_idx]),
                "cand_flex": str(cand_flex[q_idx]),
                "ecfp4_tanimoto": round(float(ecfp4_mat[r_idx, q_idx]), 5),
            }
            for m_idx, method_key in enumerate(methods):
                sh = float(shape_all[m_idx, r_idx, q_idx, rep_ki, rep_ci])
                co = float(colour_all[m_idx, r_idx, q_idx, rep_ki, rep_ci])
                row[f"{method_key}_shape"] = round(sh, 5)
                row[f"{method_key}_colour"] = round(co, 5)
                row[f"{method_key}_combo"] = round(sh + co, 5)
            rep_rows.append(row)
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(out_dir / "default_cell_pairs.tsv", sep="\t", index=False)

    np.savez_compressed(
        out_dir / "pair_scores.npz",
        methods=np.asarray(methods),
        reference_caps=np.asarray(caps, dtype=np.int32),
        candidate_counts=np.asarray(counts, dtype=np.int32),
        ref_mol_indices=np.asarray(ref_mol_indices, dtype=np.int32),
        ecfp4_tanimoto=ecfp4_mat,
        shape=shape_all,
        colour=colour_all,
    )

    flex_counts = molecules_df["flexibility_class"].value_counts().to_dict()
    ref_flex_counts = molecules_df.iloc[ref_mol_indices]["flexibility_class"].value_counts().to_dict()
    summary = {
        "sampling_funnel": sampling_funnel,
        "flexibility_counts_1000": {k: int(flex_counts.get(k, 0)) for k in FLEXIBILITY_ORDER},
        "flexibility_counts_refs": {k: int(ref_flex_counts.get(k, 0)) for k in FLEXIBILITY_ORDER},
        "representative_cell": {"K": rep_k, "C": rep_c},
        "workers": workers,
        "methods": method_telemetry,
    }
    (out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Finished cross study in %s", out_dir)
    return molecules_df, grid_df, timing_df, self_df


def _coerce(name: str, value: str):
    fields = {f.name: f for f in dataclasses.fields(CrossStudyConfig)}
    if name not in fields:
        raise SystemExit(f"unknown setting {name!r}; known: {', '.join(sorted(fields))}")
    text = str(fields[name].type)
    if "tuple[int" in text:
        return tuple(int(p) for p in value.replace(" ", "").split(",") if p)
    if "tuple[str" in text:
        return tuple(p for p in value.replace(" ", "").split(",") if p)
    if "Path" in text:
        return Path(value)
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value


def config_from_args(overrides: list[str]) -> CrossStudyConfig:
    changes = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        changes[name.strip()] = _coerce(name.strip(), value)
    return dataclasses.replace(CrossStudyConfig(), **changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one CrossStudyConfig field; repeatable",
    )
    parser.add_argument("--tag", help="suffix for output directory")
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="optional SMILES file instead of sampling the pretrained prior",
    )
    parser.add_argument("--no-figures", action="store_true", help="write tables only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = config_from_args(args.overrides)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(cfg.output_root) / f"{stamp}_{args.tag or cfg.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
    resolved["methods_resolved"] = {k: dataclasses.asdict(METHODS[k]) for k in cfg.methods}
    (out_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    started = time.perf_counter()
    _molecules_df, _grid_df, _timing_df, _self_df = run(cfg, out_dir, smiles_file=args.smiles_file)
    logger.info("Cross study finished in %.1f s", time.perf_counter() - started)

    if not args.no_figures:
        from _david.code.conformer import cross_figures

        cross_figures.render_all(out_dir, cfg=cfg)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
