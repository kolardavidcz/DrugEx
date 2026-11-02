"""Script to reproduce supervisor Jozsef Fulop's exact EGCG ROCS RL run (epsilon=0.1).

Reference:
https://github.com/fulopjoz/DrugEx/blob/egcg-rocs-tutorial/tutorial/advanced/rocs/egcg/final_out/eps0.1/executed_eps0.1.ipynb
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import SequenceExplorer
from drugex.training.generators import SequenceRNN
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer


def fingerprint(model) -> str:
    h = hashlib.sha256()
    for k, t in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def summarise(df: pd.DataFrame, prefix: str) -> pd.Series:
    hac_vals = []
    for s in df["SMILES"]:
        m = Chem.MolFromSmiles(str(s))
        hac_vals.append(m.GetNumHeavyAtoms() if m else np.nan)
    df["HAC"] = hac_vals
    df["MW"] = [
        Descriptors.MolWt(Chem.MolFromSmiles(str(s)))
        if Chem.MolFromSmiles(str(s))
        else np.nan
        for s in df["SMILES"]
    ]

    res = {
        f"{prefix}_valid_ratio": float(df["VALID"].mean()),
        f"{prefix}_rocs_median": float(df["ROCS"].dropna().median())
        if "ROCS" in df.columns
        else np.nan,
        f"{prefix}_rocs_p90": float(np.nanpercentile(df["ROCS"], 90))
        if "ROCS" in df.columns
        else np.nan,
        f"{prefix}_sa_median": float(df["Raw_SA"].dropna().median())
        if "Raw_SA" in df.columns
        else np.nan,
        f"{prefix}_hac_median": float(df["HAC"].dropna().median()),
        f"{prefix}_mw_median": float(df["MW"].dropna().median()),
    }
    return pd.Series(res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce supervisor eps=0.1 EGCG ROCS RL run."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save outputs."
    )
    parser.add_argument(
        "--agent-path", type=str, required=True, help="Path to Papyrus pretrained .pkg."
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to Papyrus vocabulary .vocab.",
    )
    parser.add_argument(
        "--reference-sdf", type=str, required=True, help="Path to EGCG reference SDF."
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_data").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    print("=================================================================")
    print(f"REPRODUCING SUPERVISOR RUN (eps={args.epsilon}, epochs={args.epochs})")
    print(f"Output: {out_dir}")
    print(f"Reference SDF: {args.reference_sdf}")
    print(f"N_JOBS: {args.n_jobs}")
    print("=================================================================")

    # 1. Environment definition
    MAX_CONFORMERS, MAX_ISOMERS, MAX_HEAVY, MAX_ROTB = 30, 4, 45, 15
    RAW_SA_CUTOFF = 4.5

    def make_env(rocs_threshold: float) -> DrugExEnvironment:
        rocs = RDKitROCSScorer(
            conformer_generator=RDKitConformerGenerator(
                max_conformers=MAX_CONFORMERS,
                max_isomers=MAX_ISOMERS,
                max_heavy_atoms=MAX_HEAVY,
                max_rotatable_bonds=MAX_ROTB,
                num_threads=0,
                show_progress=False,
            ),
            references=str(args.reference_sdf),
            score_type="TanimotoCombo",
            use_colors=True,
            show_progress=False,
            n_jobs=args.n_jobs,
        )
        sa = Property("SA")
        sa.setModifier(ClippedScore(lower_x=5, upper_x=3))
        sa_thresh = float(
            ClippedScore(lower_x=5, upper_x=3)(np.array([RAW_SA_CUTOFF]))[0]
        )
        return DrugExEnvironment(
            scorers=[rocs, sa],
            thresholds=[rocs_threshold, sa_thresh],
            reward_scheme=ParetoCrowdingDistance(),
        )

    # 2. Vocabulary & Models
    voc = VocSmiles.fromFile(args.vocab_path, encode_frags=False)
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(args.agent_path)

    prior = SequenceRNN(voc, is_lstm=True)
    prior.loadStatesFromFile(args.agent_path)
    for q in prior.parameters():
        q.requires_grad_(False)
    prior.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    agent.to(device)
    prior.to(device)

    PRIOR_FP = fingerprint(prior)
    print(f"Vocab size: {voc.size} | Device: {device} | Prior FP: {PRIOR_FP[:16]}...")

    # 3. Calibration on Untrained Prior (500 molecules)
    print("\n--- Generating Calibration Sample (N=500) from Prior ---")
    calib_smiles = prior.generate(500)
    temp_env = make_env(rocs_threshold=1.0)
    rocs_key, sa_key = temp_env.getScorerKeys()[:2]
    calib_scores = temp_env.getScores(calib_smiles)

    rocs_vals = calib_scores[rocs_key].dropna().values
    percentile_thresh = float(np.percentile(rocs_vals, 85.0))
    print(f"Calibration ROCS 85th percentile threshold: {percentile_thresh:.4f}")

    # Build active environment
    env = make_env(rocs_threshold=percentile_thresh)

    # 4. Generate Pre-RL Baseline (1000 molecules)
    print("\n--- Generating Pre-RL Baseline (N=1000) ---")
    base_smiles = prior.generate(1000)
    base_scores = env.getScores(base_smiles)
    df_before = pd.DataFrame(
        {
            "SMILES": base_smiles,
            "VALID": [1 if s is not None and len(str(s)) > 0 else 0 for s in base_smiles],
            "ROCS": base_scores[rocs_key],
            "Raw_SA": [
                Descriptors.MolWt(Chem.MolFromSmiles(s))
                if Chem.MolFromSmiles(s)
                else np.nan
                for s in base_smiles
            ],  # placeholder overwritten below
        }
    )
    # Get exact raw SA
    raw_sa_scorer = Property("SA")
    df_before["Raw_SA"] = raw_sa_scorer.getScores(base_smiles)
    df_before.to_csv(out_dir / "raw_data/baseline_raw.csv.gz", index=False)

    # 5. Training Explorer
    print(f"\n--- Starting RL Training ({args.epochs} epochs x {args.n_samples} samples) ---")
    explorer = SequenceExplorer(
        agent=agent,
        env=env,
        mutate=prior,
        epsilon=args.epsilon,
        n_samples=args.n_samples,
    )
    monitor_prefix = str(out_dir / "models/SequenceRNN_reinforced")
    monitor = FileMonitor(monitor_prefix, save_smiles=True, reset_directory=True)

    t0 = time.time()
    explorer.fit(
        monitor=monitor,
        epochs=args.epochs,
        patience=args.epochs + 1,
        reload_interval=args.epochs + 1,
        min_epochs=args.epochs,
        criteria="desired_ratio",
    )
    monitor.close()
    train_min = (time.time() - t0) / 60.0
    print(f"RL Training completed in {train_min:.2f} min.")

    assert (
        fingerprint(prior) == PRIOR_FP
    ), "Safety check failed: frozen prior changed during RL!"

    # Save final model
    torch.save(
        {k: v.cpu() for k, v in agent.getModel().items()},
        str(out_dir / "models/SequenceRNN_reinforced_final_epoch.pkg"),
    )

    # 6. Post-RL Generation (1000 molecules)
    print("\n--- Generating Post-RL Library (N=1000) ---")
    after_smiles = agent.generate(1000)
    after_scores = env.getScores(after_smiles)
    df_after = pd.DataFrame(
        {
            "SMILES": after_smiles,
            "VALID": [1 if s is not None and len(str(s)) > 0 else 0 for s in after_smiles],
            "ROCS": after_scores[rocs_key],
            "Raw_SA": raw_sa_scorer.getScores(after_smiles),
        }
    )
    df_after.to_csv(out_dir / "raw_data/reinforced_raw.csv.gz", index=False)

    # Copy reference SDF to raw_data
    with open(args.reference_sdf, "rb") as src, open(
        out_dir / "raw_data/query_molecule_3d.sdf", "wb"
    ) as dst:
        dst.write(src.read())

    # 7. Winner's Curse Holdout Validation (60 conformers)
    print("\n--- Winner's Curse Holdout Validation (60 conformers) ---")
    holdout_scorer = RDKitROCSScorer(
        conformer_generator=RDKitConformerGenerator(
            max_conformers=60,
            max_isomers=MAX_ISOMERS,
            max_heavy_atoms=MAX_HEAVY,
            max_rotatable_bonds=MAX_ROTB,
            num_threads=0,
            show_progress=False,
        ),
        references=str(args.reference_sdf),
        score_type="TanimotoCombo",
        use_colors=True,
        show_progress=False,
        n_jobs=args.n_jobs,
    )

    df_valid = df_after[df_after["VALID"] == 1].copy()
    top20 = df_valid.sort_values(by="ROCS", ascending=False).head(20)
    rand20 = df_valid.sample(n=min(20, len(df_valid)), random_state=args.seed)

    holdout_check = pd.DataFrame(
        {
            "set": ["top"] * len(top20) + ["random"] * len(rand20),
            "original_rocs": list(top20["ROCS"]) + list(rand20["ROCS"]),
            "rescored_rocs": list(
                np.asarray(
                    holdout_scorer.getScores(list(top20["SMILES"]) + list(rand20["SMILES"])),
                    dtype=float,
                ).ravel()
            ),
        }
    )
    holdout_check["delta"] = (
        holdout_check["rescored_rocs"] - holdout_check["original_rocs"]
    )
    holdout_check.to_csv(out_dir / "holdout_rescore.csv", index=False)
    print(holdout_check.groupby("set")[["original_rocs", "rescored_rocs", "delta"]].median().round(4))

    # 8. Provenance Metadata
    prov = {
        "experiment": "supervisor_reproduction_eps0.1",
        "epsilon": args.epsilon,
        "epochs": args.epochs,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "rocs_threshold": percentile_thresh,
        "training_minutes": train_min,
        "prior_fingerprint": PRIOR_FP,
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    print("\nRun completed successfully!")


if __name__ == "__main__":
    main()
