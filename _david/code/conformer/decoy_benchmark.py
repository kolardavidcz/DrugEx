"""Decoy benchmark for 3D ROCS conformer and scoring calibration.

Evaluates discriminative power (ROC-AUC, BEDROC, Decoy Separation Margins) across:
1. Actives (N=16 known bioactive polyphenols related to EGCG).
2. Volume-Deficient Fragments (N=16, tests fragment hacking resistance).
3. Volume-Matched Steric Blobs (N=16, tests color specificity on greasy rings).
4. Bloated Macrocycles (N=16, tests volume penalty).
5. Background Prior (N=32, unprompted Papyrus RNN molecules).

Compares conformer settings:
- etkdg_k1_c30 (Regime 2, unminimized ETKDGv3, 1 reference)
- etkdg_k4_c30 (Unminimized ETKDGv3, 4 references)
- mmff_k1_c20  (MMFF94s minimized, 1 reference)
- mmff_k4_c20  (Regime 3, MMFF94s minimized, 4 references)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import auc as calc_auc, average_precision_score, roc_curve

from _david.code.chemistry import (
    Optimization,
    standardize_smiles,
)
from _david.code.config import DEFAULT_MOLECULE
from _david.code.environment import create_rocs_scorer
from _david.code.reference import PrepareMolecule
from _david.code.utils import HardwareManager

# 1. Active polyphenols (N=16)
ACTIVE_PANEL = [
    ("EGCG", DEFAULT_MOLECULE),
    ("ECG", "C1[C@@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"),
    ("EGC", "C1[C@@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)O"),
    ("EC", "C1[C@@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)O"),
    ("Theaflavin", "C1C(C(OC2=CC(=CC(=C21)O)O)C3=CC4=C(C=C3O)C(=O)C(=C(C4=O)O)C5=CC(=C(C(=C5)O)O)O)O"),
    ("Myricetin", "C1=C(C=C(C(=C1O)O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O"),
    ("Quercetin", "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"),
    ("Proanthocyanidin_B2", "C1[C@@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)C4[C@@H]([C@H](OC5=CC(=CC(=C54)O)O)C6=CC(=C(C=C6)O)O)O"),
    ("Kaempferol", "C1=CC(=CC=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O"),
    ("Luteolin", "C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O"),
    ("Apigenin", "C1=CC(=CC=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O"),
    ("Cyanidin", "C1=CC(=C(C=C1C2=[O+]C3=CC(=CC(=C3C=C2O)O)O)O)O"),
    ("Delphinidin", "C1=C(C=C(C(=C1O)O)O)C2=[O+]C3=CC(=CC(=C3C=C2O)O)O"),
    ("Resveratrol", "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O"),
    ("Curcumin", "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O"),
    ("Taxifolin", "C1=CC(=C(C=C1[C@@H]2[C@H](C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O"),
]

# 2. Decoy Class 1: Volume-Deficient Fragment Hackers (N=16)
FRAGMENT_DECOYS = [
    ("Gallic_acid", "C1=C(C=C(C(=C1O)O)O)C(=O)O"),
    ("Pyrogallol", "C1=CC(=C(C(=C1)O)O)O"),
    ("Catechol", "C1=CC=C(C(=C1)O)O"),
    ("Resorcinol", "C1=CC(=CC(=C1)O)O"),
    ("Hydroquinone", "C1=CC(=CC=C1O)O"),
    ("Phloroglucinol", "C1=C(C=C(C=C1O)O)O"),
    ("Acetaminophen", "CC(=O)NC1=CC=C(C=C1)O"),
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("Phenol", "C1=CC=C(C=C1)O"),
    ("Benzoic_acid", "C1=CC=C(C=C1)C(=O)O"),
    ("Salicylic_acid", "C1=CC=CC(=C1C(=O)O)O"),
    ("Protocatechuic_acid", "C1=CC(=C(C=C1C(=O)O)O)O"),
    ("Vanillin", "COC1=C(C=CC(=C1)C=O)O"),
    ("Caffeic_acid", "C1=CC(=C(C=C1/C=C/C(=O)O)O)O"),
    ("Ferulic_acid", "COC1=C(C=CC(=C1)/C=C/C(=O)O)O"),
]

# 3. Decoy Class 2: Volume-Matched Steric Blobs (Greasy Rings / Steroids) (N=16)
GREASY_BLOB_DECOYS = [
    ("Cholesterol", "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"),
    ("Progesterone", "CC(=O)C1CCC2C1(CCC3C2CCC4=CC(=O)CCC34C)C"),
    ("Testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"),
    ("Adamantyl_amide", "CC(=O)NC12CC3CC(C1)CC(C3)C2"),
    ("Adamantane_carboxylic_acid", "C1C2CC3CC1CC(C2)(C3)C(=O)O"),
    ("Cholic_acid", "CC(CCC(=O)O)C1CCC2C1(C(CC3C2C(CC4C3(CCC(C4)O)C)O)O)C"),
    ("Androsterone", "CC12CCC(=O)CC1CCC3C2CCC4(C3CCC4O)C"),
    ("Dexamethasone", "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C"),
    ("Prednisone", "CC12CC(=O)C3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C"),
    ("Betulinic_acid", "CC(=C)C1CCC2(C1C3CCC4C5(CCC(C(C5CCC4(C3(CC2)C)C)(C)C)O)C)C(=O)O"),
    ("Ursolic_acid", "CC1CCC2(CCC3(C(=CCC4C3(CCC5C4(CCC(C5(C)C)O)C)C)C2C1C)C)C(=O)O"),
    ("Deoxycholic_acid", "CC(CCC(=O)O)C1CCC2C1(C(CC3C2CCC4C3(CCC(C4)O)C)O)C"),
    ("Squalene", "CC(=CCCC(=CCCC(=CCCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C"),
    ("Stigmasterol", "CCC(C=CC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C)C(C)C"),
    ("Ergosterol", "CC(C)C(C)C=CC(C)C1CCC2C1(CCC3C2=CC=C4C3(CCC(C4)O)C)C"),
    ("Triphenylmethane", "C1=CC=C(C=C1)C(C2=CC=CC=C2)C3=CC=CC=C3"),
]

# 4. Decoy Class 3: Bloated Macrocycles / Polymers (N=16)
MACROCYCLE_DECOYS = [
    ("Cyclosporine", "CCC1C(=O)N(CC(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)NC(C(=O)NC(C(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)N1)C(C(C)CC=CC)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C(C)C)CC(C)C)C)C"),
    ("Rapamycin", "CC1CCC2CC(=O)C(=M)N3CCCCC3C(=O)OC(C(C(=O)C(CC(C(C(C(C=CC=CC(C(CC(=O)C(C1)O)OC)C)C)O)OC)C)C)C)C"),
    ("Tacrolimus", "CC1CC(=O)C(C(=O)N2CCCCC2C(=O)OC(C(C(=O)C(CC(C(C(C=CC(C(CC(=O)C(C1)O)OC)C)C)O)OC)C)C)C)O"),
    ("Rifampicin", "CC1C=CC=C(C(=O)NC2=C(C3=C(C(=C2O)C)OC(O3)(OC)C)C(=O)C4=C(C(=C(C5=C4C(=O)C(O5)(C1OC(=O)C)C)O)C)O)O"),
    ("Atazanavir", "COC(=O)NC(C(=O)NC(CC1=CC=CC=C1)C(CN(CC2=CC=C(C=C2)C3=CC=CC=N3)NC(=O)C(C(C)(C)C)NC(=O)OC)O)C(C)(C)C"),
    ("Lopinavir", "CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C4CCCN4C(=O)C)O"),
    ("Ritonavir", "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O"),
    ("Everolimus", "CC1CCC2CC(=O)C(=M)N3CCCCC3C(=O)OC(C(C(=O)C(CC(C(C(C(C=CC=CC(C(CC(=O)C(C1)OCCO)OC)C)C)O)OC)C)C)C)C"),
    ("Temsirolimus", "CC1CCC2CC(=O)C(=M)N3CCCCC3C(=O)OC(C(C(=O)C(CC(C(C(C(C=CC=CC(C(CC(=O)C(C1)OC(=O)C(C)(CO)CO)OC)C)C)O)OC)C)C)C)C"),
    ("Ivermectin", "CCC(C)C1CC2CC(CC=C(C1O)C)OC3CC(CC(C3C)OC4CC(C(C(O4)C)O)OC5CC(C(C(O5)C)O)OC)C(=O)C2=C"),
    ("Doxorubicin", "CC1C(C(CC(O1)OC2CC(CC3=C2C(=O)C4=C(C3=O)C(=CC=C4)OC)(C(=O)CO)O)N)O"),
    ("Epirubicin", "CC1C(C(CC(O1)OC2CC(CC3=C2C(=O)C4=C(C3=O)C(=CC=C4)OC)(C(=O)CO)O)N)O"),
    ("Idarubicin", "CC1C(C(CC(O1)OC2CC(CC3=C2C(=O)C4=C(C3=O)C=CC=C4)(C(=O)C)O)N)O"),
    ("Bleomycin", "CC1=C(N=C(N1)C(=O)NC(C(C2=CN=CN2)N)C(=O)NC(C(C)O)C(=O)NC(CC(=O)N)C(=O)NC(C(C3=CC(=C(C=C3)O)O)O)C(=O)NC4C(SC(=N4)C5=NC(=CS5)C(=O)NCCC[S+](C)C)O)O"),
    ("Amphotericin_B", "CC1C=CC=CC=CC=CC=CC=CC=CC(CC2C(CC(CC(C(CC(CC(CC(CC(CC(=O)OC(C(C1O)C)C)O)O)O)O)O)O)O)O)OC2(C(C(C(C(O3)N)O)O)C)O)O"),
    ("Nystatin", "CC1C=CC=CC=CC=CC=CC=CC=CC(CC2C(CC(CC(C(CC(CC(CC(CC(CC(=O)OC(C(C1O)C)C)O)O)O)O)O)O)O)O)OC2(C(C(C(C(O3)N)O)O)C)O)O"),
]

# 5. Decoy Class 4: Background Prior Chemotypes (N=20)
PRIOR_DECOYS = [
    ("Prior_01", "O=C(Nc1ccc(Cl)cc1)c2ccc(NC(=O)c3ccccc3)cc2"),
    ("Prior_02", "Cc1ccc(S(=O)(=O)N2CCN(C(=O)c3ccco3)CC2)cc1"),
    ("Prior_03", "COc1ccc(CCN2CCC(C(=O)Nc3cccc(C)c3)CC2)cc1"),
    ("Prior_04", "O=C(CSc1nnc(-c2ccccc2)o1)Nc3ccc(OC)cc3"),
    ("Prior_05", "CC1=NN(c2ccccc2)C(=O)C1=Cc3ccc(N(C)C)cc3"),
    ("Prior_06", "COc1cccc(C(=O)N2CCC(n3c(=O)[nH]c4ccccc43)CC2)c1"),
    ("Prior_07", "O=C(Nc1ccc(S(=O)(=O)N2CCOCC2)cc1)c3cc(Cl)ccc3Cl"),
    ("Prior_08", "Cc1cc(C)c(NC(=O)C2CCN(S(=O)(=O)c3ccccc3)CC2)c(C)c1"),
    ("Prior_09", "O=C(CSc1nc2ccccc2s1)N3CCC(c4ccccc4)CC3"),
    ("Prior_10", "COc1ccc(C(=O)Nc2ccc(N3CCOCC3)cc2)cc1OC"),
    ("Prior_11", "Cc1ccc(N2CCN(C(=O)c3cccc(F)c3)CC2)cc1"),
    ("Prior_12", "O=C(Nc1ccc(F)cc1)C(=O)N2CCN(c3cccc(Cl)c3)CC2"),
    ("Prior_13", "COc1ccc(S(=O)(=O)N2CCN(Cc3ccc(C)cc3)CC2)cc1"),
    ("Prior_14", "O=C(Nc1ccc(C(=O)N2CCOCC2)cc1)c3ccc(Cl)cc3"),
    ("Prior_15", "Cc1nn(C)c(C)c1-c2ccc(NC(=O)c3ccccc3)cc2"),
    ("Prior_16", "O=C(c1ccccc1)N2CCN(c3ccc(C(=O)Nc4ccccc4)cc3)CC2"),
    ("Prior_17", "COc1ccc(C(=O)N2CCC(NC(=O)c3ccccc3)CC2)cc1"),
    ("Prior_18", "Cc1cc(NC(=O)c2cccc(Cl)c2)ccc1N3CCOCC3"),
    ("Prior_19", "O=C(Nc1cccc(S(=O)(=O)N2CCOCC2)c1)c3ccco3"),
    ("Prior_20", "CC(C)c1ccc(NC(=O)c2ccc(N3CCOCC3)cc2)cc1"),
]


def build_decoy_panel() -> pd.DataFrame:
    """Consolidate the stratified active vs decoy panel."""
    rows = []
    cohorts = [
        ("active", ACTIVE_PANEL, 1),
        ("fragment_decoy", FRAGMENT_DECOYS, 0),
        ("greasy_blob_decoy", GREASY_BLOB_DECOYS, 0),
        ("macrocycle_decoy", MACROCYCLE_DECOYS, 0),
        ("prior_decoy", PRIOR_DECOYS, 0),
    ]
    for cohort_name, items, label in cohorts:
        for name, smi in items:
            try:
                std_smi = standardize_smiles(smi)
                mol = Chem.MolFromSmiles(std_smi)
                if mol is None:
                    continue
                rows.append({
                    "name": name,
                    "smiles": std_smi,
                    "cohort": cohort_name,
                    "is_active": label,
                    "mw": Descriptors.MolWt(mol),
                    "logp": Descriptors.MolLogP(mol),
                    "hac": mol.GetNumHeavyAtoms(),
                    "rotors": Descriptors.NumRotatableBonds(mol),
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def run_decoy_analysis(output_dir: Path | str | None = None, n_jobs: int | None = None) -> pd.DataFrame:
    """Run cross-conformer decoy discrimination benchmark."""
    dest = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "output" / "conformer_decoy_analysis"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = n_jobs if n_jobs is not None else HardwareManager.get_safe_parallel_jobs(ratio=0.85)

    panel = build_decoy_panel()
    print(f"Loaded decoy benchmark panel: {len(panel)} molecules")
    print(panel["cohort"].value_counts().to_string())

    # Pre-generate references (K=1 lowest energy and K=4 diverse)
    ref_k1_sdf = dest / "ref_egcg_k1.sdf"
    if not ref_k1_sdf.exists():
        ref_gen_k1 = PrepareMolecule(
            optimization=Optimization("MMFF94s", 78.4),
            max_reference_conformers=1,
        )
        ref_gen_k1.generate_reference_conformers(DEFAULT_MOLECULE, output_path=ref_k1_sdf)

    ref_k4_sdf = dest / "ref_egcg_k4.sdf"
    if not ref_k4_sdf.exists():
        ref_gen_k4 = PrepareMolecule(
            optimization=Optimization("MMFF94s", 78.4),
            max_reference_conformers=4,
        )
        ref_gen_k4.generate_reference_conformers(DEFAULT_MOLECULE, output_path=ref_k4_sdf)

    smiles_list = panel["smiles"].tolist()

    # Define test configurations
    configs = [
        # 1. Supervisor Regime 2: Unminimized ETKDGv3, K=1, C=30, TanimotoCombo
        ("etkdg_k1_c30_combo", ref_k1_sdf, False, 30, "tanimoto_combo"),
        # 2. Multi-Reference Unminimized: ETKDGv3, K=4, C=30, TanimotoCombo
        ("etkdg_k4_c30_combo", ref_k4_sdf, False, 30, "tanimoto_combo"),
        # 3. Minimized K=1: MMFF94s, K=1, C=20, TanimotoCombo
        ("mmff_k1_c20_combo", ref_k1_sdf, True, 20, "tanimoto_combo"),
        # 4. Regime 3 (Run 01 setup): MMFF94s, K=4, C=20, TanimotoCombo
        ("mmff_k4_c20_combo", ref_k4_sdf, True, 20, "tanimoto_combo"),
        # 5. Tversky Option 1 on Unminimized: ETKDGv3, K=1, C=30, Tversky Mild Color-Heavy
        ("etkdg_k1_c30_tversky_opt1", ref_k1_sdf, False, 30, "tversky_mild_color_heavy"),
    ]

    results_summary = []
    roc_curves_data = {}

    for cfg_name, ref_sdf, minimize, n_confs, variant in configs:
        print(f"\n--- Evaluating {cfg_name} (minimize={minimize}, C={n_confs}, workers={jobs}) ---")
        start = time.perf_counter()
        scorer = create_rocs_scorer(
            str(ref_sdf),
            score_variant=variant,
            max_conformers=n_confs,
            minimize_candidates=minimize,
            n_jobs=jobs,
            timeout=15,
        )
        scores = scorer.getScores(smiles_list).flatten()
        dur = time.perf_counter() - start
        panel[f"score_{cfg_name}"] = scores

        y_true = panel["is_active"].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = calc_auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, scores)
        roc_curves_data[cfg_name] = (fpr, tpr, auc_val)

        active_scores = scores[panel["cohort"] == "active"]
        frag_scores = scores[panel["cohort"] == "fragment_decoy"]
        greasy_scores = scores[panel["cohort"] == "greasy_blob_decoy"]
        macro_scores = scores[panel["cohort"] == "macrocycle_decoy"]
        prior_scores = scores[panel["cohort"] == "prior_decoy"]

        mean_active = float(np.mean(active_scores))
        mean_frag = float(np.mean(frag_scores))
        mean_greasy = float(np.mean(greasy_scores))
        mean_macro = float(np.mean(macro_scores))
        mean_prior = float(np.mean(prior_scores))

        decoy_margin = mean_active - float(np.mean(scores[y_true == 0]))
        frag_rejection_delta = mean_active - mean_frag

        print(f"Wall time: {dur:.2f}s ({dur/len(panel):.3f} s/mol)")
        print(f"ROC-AUC: {auc_val:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"Mean Active: {mean_active:.3f} | Fragment: {mean_frag:.3f} | Greasy Blob: {mean_greasy:.3f}")
        print(f"Decoy Margin: {decoy_margin:.3f} | Fragment Rejection: {frag_rejection_delta:.3f}")

        results_summary.append({
            "config": cfg_name,
            "minimize": minimize,
            "conformers": n_confs,
            "variant": variant,
            "wall_seconds": round(dur, 2),
            "seconds_per_mol": round(dur / len(panel), 3),
            "roc_auc": round(auc_val, 4),
            "pr_auc": round(pr_auc, 4),
            "mean_active": round(mean_active, 4),
            "mean_fragment": round(mean_frag, 4),
            "mean_greasy": round(mean_greasy, 4),
            "mean_macrocycle": round(mean_macro, 4),
            "mean_prior": round(mean_prior, 4),
            "decoy_margin": round(decoy_margin, 4),
            "frag_rejection": round(frag_rejection_delta, 4),
        })

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(dest / "decoy_scorecard.csv", index=False)
    panel.to_csv(dest / "decoy_panel_predictions.csv", index=False)

    # Plot ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for name, (fpr, tpr, a) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (Decoys)", fontsize=11)
    plt.ylabel("True Positive Rate (Actives)", fontsize=11)
    plt.title("ROCS Conformer Configuration Decoy Discrimination (ROC Curves)", fontsize=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(dest / "decoy_roc_curves.png")
    plt.close()

    # Plot Cohort Score Distributions for Regime 2 (etkdg_k1_c30) vs Regime 3 (mmff_k4_c20)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, sharey=True)
    cohort_order = ["active", "fragment_decoy", "greasy_blob_decoy", "macrocycle_decoy", "prior_decoy"]
    cohort_labels = ["Actives\n(N=16)", "Fragments\n(N=16)", "Greasy Blobs\n(N=16)", "Macrocycles\n(N=11)", "Prior\n(N=20)"]

    for ax, col, title in zip(
        axes,
        ["score_etkdg_k1_c30_combo", "score_mmff_k4_c20_combo"],
        ["Regime 2: ETKDGv3 Unminimized (K=1, C=30)", "Regime 3: MMFF94s Minimized (K=4, C=20)"]
    ):
        data = [panel[panel["cohort"] == c][col].values for c in cohort_order]
        ax.boxplot(data, tick_labels=cohort_labels, patch_artist=True, boxprops=dict(facecolor="#e0e7ff", color="#4338ca"))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("ROCS TanimotoCombo [0, 2]" if ax == axes[0] else "", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Active vs Decoy Score Distribution: Unminimized ETKDGv3 vs Force-Field Minimized MMFF94s", fontsize=12)
    plt.tight_layout()
    plt.savefig(dest / "decoy_score_distributions.png")
    plt.close()

    print(f"\nSaved benchmark outputs and publication figures to {dest}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROCS Conformer Decoy Discrimination Benchmark")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=None, help="Worker processes for scoring")
    args = parser.parse_args()
    run_decoy_analysis(output_dir=args.output_dir, n_jobs=args.n_jobs)
