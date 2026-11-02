#!/usr/bin/env python3
"""
Consultation with DeepSeek-v4.1-flash and GLM-5.3:
Analyzing empirical results of completed MetaCentrum runs (H1 vs H2) and in-flight H3/H4,
and deciding the next batch of production runs to execute on MetaCentrum using config.py defaults + CLI flags.
"""

import json
import subprocess
from pathlib import Path

API_URL = "https://llm.ai.e-infra.cz/v1/chat/completions"
API_KEY = "sk-50dc3f278a5545cebfba2ce0d4016597"
OUTPUT_DIR = Path("/home/kolar/DrugEx/_david/output")
OUTPUT_FILE = OUTPUT_DIR / "metacentrum_runs_consultation_deepseek_glm.md"

def query_model(model_name: str, messages: list, max_tokens: int = 4096, temperature: float = 0.5) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    cmd = [
        "curl", "-s", "-4", "-X", "POST", API_URL,
        "-H", f"Authorization: Bearer {API_KEY}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    return content if content else reasoning

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting Consultation: MetaCentrum Training Runs Strategy")

    prompt = """You are an elite Computational Medicinal Chemist and Reinforcement Learning Architect advising David Kolar on his Master's thesis (DrugEx SequenceRNN + 3D ROCS shape/color matching).

### Empirical Data from Completed MetaCentrum Runs (EGCG Seed, 200 epochs, 500 samples/epoch):
1. Sprint H1 (Pareto non-dominated sorting, ROCS + SA, no QED):
   - ROCS Top 10% Tail: 0.730
   - ROCS Median: 0.602
   - QED Median: 0.63
   - Raw SA: 2.91
   - Heavy Atom Count (HAC): 20.0 (Baseline Prior: 26.0) -> MW 330.9 Da
   - Grammar Validity: 71.8%

2. Sprint H2 (Harrington desirability aggregation, ROCS + SA, no QED) [JUST COMPLETED]:
   - ROCS Top 10% Tail: 0.742 (+0.012 higher than H1)
   - ROCS Median: 0.616 (+0.014 higher than H1)
   - QED Median: 0.63
   - Raw SA: 2.85 (more synthesizable than H1: 2.85 vs 2.91)
   - Heavy Atom Count (HAC): 22.0 (higher molecular complexity than H1) -> MW 351.4 Da
   - Grammar Validity: 74.7% (+2.9% higher than H1)

3. Sprints H3 & H4 (with QED included):
   - In-flight on MetaCentrum (Epoch ~80/200).

4. Supervisor Baseline & Ensemble Runs:
   - Job 25199074: Supervisor reproduction (eps=0.1, beta=0.1, K=1 conformer, 50 epochs).
   - Job 25199123: Run H6 (Multi-Conformer Reference Ensemble K=4, eps=0.1, 50 epochs).

### Architecture & Pipeline Configuration (`_david/code/config.py` defaults):
- `epochs: 200`, `min_epochs: 50`, `patience: 50`
- `n_samples: 500`, `batch_size: 64`, `n_eval_samples: 1000`
- `epsilon: 0.05`, `beta: 0.0`, `decay_type: "cosine"`, `use_epsilon_decay: False`
- `max_conformers: 20`, `minimize_candidates: True` (MMFF94s, dielectric=78.4)
- `score_variant: "tanimoto_combo"` (alternatives: `tversky_mild_color_heavy`, `tversky_scaffold_hop`, `selectivity_color_heavy`)
- `reward_scheme: "harrington"` (alternatives: `pareto_crowding`, `weighted_sum`)

### Available CLI Flags to override defaults:
- `--score-variant` (e.g. `tversky_mild_color_heavy`, `tversky_scaffold_hop`)
- `--reward-scheme` (`harrington`, `pareto_crowding`, `weighted_sum`)
- `--epsilon`, `--epsilon-end`, `--use-decay`, `--decay-epochs`, `--decay-type`
- `--beta` (policy baseline penalization, e.g. 0.1 vs 0.0)
- `--include-qed` / `--no-include-qed`
- `--conformers` (e.g. 20)
- `--reference-conformers` (e.g. 4)
- `--rocs-use-modifier` (`--rocs-clip-upper 1.8`)

### Questions for you to answer:
1. What does the comparison between H1 (Pareto) and H2 (Harrington) teach us about reward shaping in 3D molecular RL?
2. What are the 2-3 most impactful next training runs we should schedule on MetaCentrum right now?
   - Formulate exact CLI commands using `python -m _david.code.pipeline --output-dir ... [FLAGS]`.
   - Explain the precise scientific hypothesis behind each proposed run (e.g., Tversky asymmetric shape vs scaffold-hop color-heavy, dynamic epsilon decay, baseline penalization beta=0.1).
3. How will these runs answer the core scientific questions of David Kolar's thesis?"""

    print("[1/2] Querying DeepSeek-v4.1-flash...")
    deepseek_response = query_model("deepseek-v4.1-flash", [
        {"role": "system", "content": "You are DeepSeek-v4.1-flash, an elite computational chemist and reinforcement learning specialist. Provide structured, mathematically and pharmacologically rigorous recommendations."},
        {"role": "user", "content": prompt}
    ])
    print(f"DeepSeek response received ({len(deepseek_response)} chars).")

    print("[2/2] Querying GLM-5.3...")
    glm_prompt = f"""{prompt}

Here is DeepSeek-v4.1-flash's analysis:
=== DEEPSEEK ANALYSIS ===
{deepseek_response}
=== END DEEPSEEK ===

Review DeepSeek's analysis, provide your critique and adjustments, and finalize the recommended MetaCentrum runs."""

    glm_response = query_model("glm-5.3", [
        {"role": "system", "content": "You are GLM-5.3, an expert in cheminformatics and molecular AI. Provide dense, actionable, and mathematically justified conclusions."},
        {"role": "user", "content": glm_prompt}
    ])
    print(f"GLM-5.3 response received ({len(glm_response)} chars).")

    full_report = f"""# Expert Consultation on MetaCentrum Training Runs: DeepSeek-v4.1-flash & GLM-5.3

**Topic**: Production RL Training Matrix on MetaCentrum for DrugEx (3D ROCS Shape/Color Matching)  
**Evaluated Data**: H1 (Pareto) vs H2 (Harrington) empirical results, H3/H4 in-flight dynamics  
**Reference Target**: Epigallocatechin gallate (EGCG)  

---

## 1. DeepSeek-v4.1-flash Strategic Analysis

{deepseek_response}

---

## 2. GLM-5.3 Peer Review & Run Matrix Refinement

{glm_response}
"""

    with open(OUTPUT_FILE, "w") as f:
        f.write(full_report)
    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
