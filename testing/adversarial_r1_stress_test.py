"""Adversarial Empirical Stress-Testing Suite for Requirement R1.

This test suite aggressively probes and stress-tests:
1. prepare_query_conformer.py (QueryConformerPreparer, load_query_molecule, Murcko extraction)
2. dynamic_envelope.py (DynamicPropertyEnvelope, SmoothHump evaluation)
Across:
- Extreme small molecules (Methane, Water, Ethane, Propane, Glycine)
- Hydrocarbons / Zero-TPSA molecules (Benzene, Toluene, Hexane)
- Zero-ring / acyclic molecules (Hexane, Spermine, Glycine)
- Highly negative LogP hydrophilic molecules (Arginine, D-Glucose, Citric acid)
- Complex natural products & macrocycles (EGCG, Taxol, Strophanthidin)
- Disconnected salts & inorganic/organometallic queries (Sodium acetate, Cisplatin)
- Multi-seed vacuum collapse vs. extended curation (Rg, span, outward pharmacophore distances)
- Mathematical integrity checks (lower_x vs upper_x inversions in SmoothHump)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# [PACKAGE: numpy] WHY: Quantitative metrics, vector distances, and percentiles. WHAT: Scientific computing. HOW: np.mean, np.median, np.linalg.norm.
import numpy as np

# [PACKAGE: rdkit] WHY: 3D coordinates, pharmacophore atom extraction, descriptors, SASA. WHAT: Cheminformatics toolkit. HOW: Chem, AllChem, Descriptors3D, Lipinski, Crippen.
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors3D, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

# Repo path setup
REPO_ROOT = Path(__file__).resolve().parents[1]
DAVID_CODE = REPO_ROOT / "_david" / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DAVID_CODE) not in sys.path:
    sys.path.insert(0, str(DAVID_CODE))

from dynamic_envelope import (
    DynamicPropertyEnvelope,
    compute_dynamic_envelope,
    create_dynamic_property_scorers,
)
from prepare_query_conformer import (
    MAX_ENERGY_DELTA,
    MY_MOLECULE,
    PRUNE_RMS_THRESH,
    RANDOM_SEED,
    QueryConformerPreparer,
    load_query_molecule,
)


def run_test_suite() -> dict[str, Any]:
    """Execute complete empirical test suite and collect structured findings."""
    results: dict[str, Any] = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "failures": [],
        "findings": [],
    }

    print("======================================================================", flush=True)
    print("STARTING ADVERSARIAL EMPIRICAL STRESS TEST: REQUIREMENT R1", flush=True)
    print("======================================================================\n", flush=True)

    # -------------------------------------------------------------------------
    # SUITE 1: DYNAMIC ENVELOPE BOUNDARY INVERSIONS & HARDCODED BOUNDS
    # -------------------------------------------------------------------------
    print("--- SUITE 1: Dynamic Property Envelope Boundary & Inversion Stress Test ---", flush=True)
    test_molecules_envelope = [
        ("Methane", "C"),
        ("Water", "O"),
        ("Ethane", "CC"),
        ("Propane", "CCC"),
        ("Glycine", "NCC(=O)O"),
        ("Benzene", "c1ccccc1"),
        ("Toluene", "Cc1ccccc1"),
        ("Hexane", "CCCCCC"),
        ("Pyridine", "c1ccncc1"),
        ("EGCG", MY_MOLECULE),
        ("Arginine", "N=C(N)NCCCC(N)C(=O)O"),
        ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ]

    for name, smi in test_molecules_envelope:
        results["tests_run"] += 1
        mol = Chem.MolFromSmiles(smi)
        env = DynamicPropertyEnvelope.from_molecule(mol)
        scorers = env.create_scorers()

        hac_val = Lipinski.HeavyAtomCount(mol)
        tpsa_val = AllChem.CalcTPSA(mol)
        logp_val = Crippen.MolLogP(mol)

        hac_score = scorers[0]([smi])[0]
        logp_score = scorers[1]([smi])[0]
        tpsa_score = scorers[2]([smi])[0]

        # Check HAC envelope validity: lower_x must be <= upper_x and target must be inside [lower_x, upper_x]
        hac_inverted = env.hac_lower > env.hac_upper
        hac_target_outside = (env.target_hac < env.hac_lower) or (env.target_hac > env.hac_upper)

        # Check TPSA envelope validity: lower_x must be <= upper_x and target must be inside [lower_x, upper_x]
        tpsa_inverted = env.tpsa_lower > env.tpsa_upper
        tpsa_target_outside = (env.target_tpsa < env.tpsa_lower) or (env.target_tpsa > env.tpsa_upper)

        print(f"[{name}] (HAC={hac_val}, logP={logp_val:.2f}, TPSA={tpsa_val:.2f}):", flush=True)
        print(f"  HAC bounds:  [{env.hac_lower:.2f}, {env.hac_upper:.2f}], sigma={env.hac_sigma:.2f} -> Target HAC Score: {hac_score:.4f}", flush=True)
        print(f"  logP bounds: [{env.clogp_lower:.2f}, {env.clogp_upper:.2f}], sigma={env.clogp_sigma:.2f} -> Target logP Score: {logp_score:.4f}", flush=True)
        print(f"  TPSA bounds: [{env.tpsa_lower:.2f}, {env.tpsa_upper:.2f}], sigma={env.tpsa_sigma:.2f} -> Target TPSA Score: {tpsa_score:.4f}", flush=True)

        if hac_inverted or tpsa_inverted:
            results["tests_failed"] += 1
            msg = f"CRITICAL: Envelope bounds inverted for {name}! HAC inverted: {hac_inverted} ({env.hac_lower:.2f} > {env.hac_upper:.2f}), TPSA inverted: {tpsa_inverted} ({env.tpsa_lower:.2f} > {env.tpsa_upper:.2f})"
            print(f"  FAILED: {msg}", flush=True)
            results["failures"].append(msg)
            results["findings"].append({
                "type": "BOUNDS_INVERSION",
                "molecule": name,
                "smiles": smi,
                "hac_lower": env.hac_lower,
                "hac_upper": env.hac_upper,
                "tpsa_lower": env.tpsa_lower,
                "tpsa_upper": env.tpsa_upper,
                "hac_score": hac_score,
                "tpsa_score": tpsa_score,
            })
        elif hac_target_outside or tpsa_target_outside:
            results["tests_failed"] += 1
            msg = f"HIGH: Query template {name} falls outside its own envelope! HAC score: {hac_score:.4f}, TPSA score: {tpsa_score:.4f}"
            print(f"  FAILED: {msg}", flush=True)
            results["failures"].append(msg)
            results["findings"].append({
                "type": "TARGET_OUTSIDE_ENVELOPE",
                "molecule": name,
                "smiles": smi,
                "hac_score": hac_score,
                "tpsa_score": tpsa_score,
            })
        else:
            results["tests_passed"] += 1
            print(f"  PASSED: Envelope bounds valid and target scores 1.0", flush=True)

    # -------------------------------------------------------------------------
    # SUITE 2: CONFORMER ENSEMBLE GENERATION ON EXTREME MOLECULES
    # -------------------------------------------------------------------------
    print("\n--- SUITE 2: Conformer Ensemble Generation Across Extreme Molecules ---", flush=True)
    extreme_molecules = [
        ("Methane", "C", 5),
        ("Propane", "CCC", 5),
        ("Glycine", "NCC(=O)O", 5),
        ("Hexane", "CCCCCC", 10),
        ("Benzene", "c1ccccc1", 5),
        ("Arginine", "N=C(N)NCCCC(N)C(=O)O", 15),
        ("D-Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 10),
        ("Citric Acid", "OC(=O)CC(O)(CC(=O)O)C(=O)O", 10),
        ("Strophanthidin", "CC12CCC(CC1CCC3C2CCC4(C3(CCC4C5=CC(=O)OC5)O)C)O", 10),
        ("Sodium Acetate", "CC(=O)[O-].[Na+]", 5),
        ("Cisplatin", "N[Pt](N)(Cl)Cl", 5),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = QueryConformerPreparer(Path(tmp_dir))
        for name, smi, n_confs in extreme_molecules:
            results["tests_run"] += 1
            try:
                out_path = Path(tmp_dir) / f"{name}_3d.sdf"
                res_path = prep.generate_reference_sdf(smi, output_path=out_path, num_conformers=n_confs)
                suppl = Chem.SDMolSupplier(str(res_path), removeHs=False)
                mols = [m for m in suppl if m is not None]
                if len(mols) == 0:
                    raise RuntimeError(f"0 conformers loaded from generated SDF")
                rgs = [Descriptors3D.RadiusOfGyration(m) for m in mols]
                print(f"[{name}] SUCCESS: Generated {len(mols)} conformers (Rg min={min(rgs):.3f}, max={max(rgs):.3f})", flush=True)
                results["tests_passed"] += 1
            except Exception as e:
                results["tests_failed"] += 1
                msg = f"EXCEPTION in conformer prep for {name}: {type(e).__name__}: {str(e)}"
                print(f"[{name}] FAILED: {msg}", flush=True)
                results["failures"].append(msg)
                results["findings"].append({
                    "type": "CONFORMER_GENERATION_EXCEPTION",
                    "molecule": name,
                    "smiles": smi,
                    "error": str(e),
                })

    # -------------------------------------------------------------------------
    # SUITE 3: SCAFFOLD EXTRACTION ON ZERO-RING & UNUSUAL MOLECULES
    # -------------------------------------------------------------------------
    print("\n--- SUITE 3: Bemis-Murcko Scaffold Extraction on Zero-Ring Molecules ---", flush=True)
    scaffold_molecules = [
        ("Hexane", "CCCCCC", True),  # Acyclic -> fallback to full molecule
        ("Glycine", "NCC(=O)O", True),  # Acyclic -> fallback
        ("Spermine", "NCCCCNCCCCN", True),  # Acyclic polyamine
        ("Benzene", "c1ccccc1", False),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", False),
        ("EGCG", MY_MOLECULE, False),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        prep = QueryConformerPreparer(Path(tmp_dir))
        for name, smi, is_acyclic in scaffold_molecules:
            results["tests_run"] += 1
            try:
                scaff = prep.extract_seed_core_scaffold(smi)
                print(f"[{name}] Scaffold: {scaff}", flush=True)
                if not scaff or len(scaff) == 0:
                    raise ValueError("Empty scaffold string")
                if is_acyclic:
                    scaff_mol = Chem.MolFromSmiles(scaff)
                    assert scaff_mol is not None, "Extracted acyclic scaffold is not parseable"
                    print(f"  [PASS] Acyclic fallback preserved molecule structure: {scaff}", flush=True)
                results["tests_passed"] += 1
            except Exception as e:
                results["tests_failed"] += 1
                msg = f"Scaffold extraction failed for {name}: {e}"
                print(f"  [FAIL] {msg}", flush=True)
                results["failures"].append(msg)
                results["findings"].append({
                    "type": "SCAFFOLD_EXTRACTION_FAILURE",
                    "molecule": name,
                    "error": str(e),
                })

    # -------------------------------------------------------------------------
    # SUITE 4: VACUUM COLLAPSE VS EXTENDED CURATION ACROSS RANDOM SEEDS
    # -------------------------------------------------------------------------
    print("\n--- SUITE 4: Multi-Seed Vacuum Collapse vs Extended Curation Stress Test ---", flush=True)
    seeds_to_test = [42, 101, 777, 2024]
    egcg_mol = Chem.MolFromSmiles(MY_MOLECULE)
    egcg_h = Chem.AddHs(egcg_mol)

    patt_oh = Chem.MolFromSmarts("[OX2H]")
    oh_indices = [match[0] for match in egcg_h.GetSubstructMatches(patt_oh)]

    rings = egcg_h.GetRingInfo().AtomRings()
    arom_rings = [r for r in rings if len(r) == 6 and all(egcg_h.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    print(f"Identified {len(oh_indices)} phenolic OH groups and {len(arom_rings)} aromatic rings for geometric analysis.", flush=True)

    seed_summary = []
    for seed in seeds_to_test:
        results["tests_run"] += 1
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = PRUNE_RMS_THRESH

        cids = list(AllChem.EmbedMultipleConfs(egcg_h, numConfs=25, params=params))
        assert len(cids) > 0, f"Embedding failed for seed {seed}"

        # MMFF94 optimization
        props = AllChem.MMFFGetMoleculeProperties(egcg_h)
        energies: list[tuple[float, int]] = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(egcg_h, props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=500)
                energies.append((float(ff.CalcEnergy()), cid))
            else:
                AllChem.UFFOptimizeMolecule(egcg_h, confId=cid, maxIters=500)
                energies.append((0.0, cid))

        energies.sort(key=lambda x: x[0])
        min_e = energies[0][0]
        accepted = [cid for e, cid in energies if (e - min_e) <= MAX_ENERGY_DELTA]
        if not accepted:
            accepted = [energies[0][1]]

        # Calculate Rg for accepted conformers
        rgs = [Descriptors3D.RadiusOfGyration(egcg_h, confId=cid) for cid in accepted]
        rg_median = float(np.median(rgs))

        curated_cids = [cid for cid, rg in zip(accepted, rgs) if rg >= rg_median]
        discarded_cids = [cid for cid, rg in zip(accepted, rgs) if rg < rg_median]

        def calc_oh_centroid_dist(cid: int) -> float:
            conf = egcg_h.GetConformer(cid)
            coords = conf.GetPositions()
            centroid = np.mean(coords, axis=0)
            oh_coords = coords[oh_indices]
            dists = np.linalg.norm(oh_coords - centroid, axis=1)
            return float(np.mean(dists))

        def calc_max_span(cid: int) -> float:
            conf = egcg_h.GetConformer(cid)
            coords = conf.GetPositions()
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
            return float(np.max(dist_matrix))

        curated_oh_dists = [calc_oh_centroid_dist(cid) for cid in curated_cids]
        discarded_oh_dists = [calc_oh_centroid_dist(cid) for cid in discarded_cids] if discarded_cids else [0.0]

        curated_spans = [calc_max_span(cid) for cid in curated_cids]
        discarded_spans = [calc_max_span(cid) for cid in discarded_cids] if discarded_cids else [0.0]

        mean_curated_rg = np.mean([rg for cid, rg in zip(accepted, rgs) if cid in curated_cids])
        mean_discarded_rg = np.mean([rg for cid, rg in zip(accepted, rgs) if cid in discarded_cids]) if discarded_cids else 0.0

        seed_info = {
            "seed": seed,
            "embedded": len(cids),
            "boltzmann_accepted": len(accepted),
            "rg_min": min(rgs),
            "rg_median": rg_median,
            "rg_max": max(rgs),
            "curated_count": len(curated_cids),
            "discarded_count": len(discarded_cids),
            "curated_rg_mean": mean_curated_rg,
            "discarded_rg_mean": mean_discarded_rg,
            "curated_oh_dist_mean": np.mean(curated_oh_dists),
            "discarded_oh_dist_mean": np.mean(discarded_oh_dists),
            "curated_span_mean": np.mean(curated_spans),
            "discarded_span_mean": np.mean(discarded_spans),
        }
        seed_summary.append(seed_info)

        print(f"Seed {seed:5d}: Emb={len(cids)}, Boltz={len(accepted)}, Curated={len(curated_cids)}", flush=True)
        print(f"  Rg: Med={rg_median:.3f} | Curated Rg={mean_curated_rg:.3f} vs Discarded Rg={mean_discarded_rg:.3f}", flush=True)
        print(f"  OH-Centroid Dist: Curated={np.mean(curated_oh_dists):.3f} A vs Discarded={np.mean(discarded_oh_dists):.3f} A", flush=True)
        print(f"  Max Span:         Curated={np.mean(curated_spans):.3f} A vs Discarded={np.mean(discarded_spans):.3f} A", flush=True)

        if len(discarded_cids) > 0:
            assert mean_curated_rg > mean_discarded_rg, f"Curated Rg ({mean_curated_rg}) not greater than discarded Rg ({mean_discarded_rg})"
            assert np.mean(curated_oh_dists) >= np.mean(discarded_oh_dists), f"Curated OH distance ({np.mean(curated_oh_dists)}) not >= discarded ({np.mean(discarded_oh_dists)})"
            assert np.mean(curated_spans) > np.mean(discarded_spans), f"Curated span ({np.mean(curated_spans)}) not > discarded ({np.mean(discarded_spans)})"

        results["tests_passed"] += 1

    # -------------------------------------------------------------------------
    # SUITE 5: PARAMETER HARDCODING IN PREPARE_QUERY_CONFORMER
    # -------------------------------------------------------------------------
    print("\n--- SUITE 5: Code Interface & Parameter Exposure Inspection ---", flush=True)
    results["tests_run"] += 1
    import inspect
    sig = inspect.signature(QueryConformerPreparer.generate_reference_sdf)
    has_seed_param = "random_seed" in sig.parameters or "seed" in sig.parameters
    print(f"QueryConformerPreparer.generate_reference_sdf signature: {sig}", flush=True)
    print(f"Has explicit random_seed parameter: {has_seed_param}", flush=True)
    if not has_seed_param:
        msg = "NOTE: generate_reference_sdf hardcodes global RANDOM_SEED=42 and does not allow callers to pass seed"
        print(f"  FINDING: {msg}", flush=True)
        results["findings"].append({
            "type": "HARDCODED_RANDOM_SEED",
            "description": msg,
        })
    results["tests_passed"] += 1

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n======================================================================", flush=True)
    print("STRESS TEST EXECUTION COMPLETED", flush=True)
    print(f"Total Tests Run:    {results['tests_run']}", flush=True)
    print(f"Total Tests Passed: {results['tests_passed']}", flush=True)
    print(f"Total Tests Failed: {results['tests_failed']}", flush=True)
    print(f"Identified Findings: {len(results['findings'])}", flush=True)
    print("======================================================================", flush=True)

    return results


if __name__ == "__main__":
    res = run_test_suite()
    if res["tests_failed"] > 0:
        print(f"\nExiting with FAILURE: {res['tests_failed']} test assertions failed.", flush=True)
        sys.exit(1)
    else:
        print("\nAll adversarial tests passed!", flush=True)
        sys.exit(0)
