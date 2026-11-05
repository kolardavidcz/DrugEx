import json
import time
import httpx
from pathlib import Path

PROMPT = """You are a Principal Scientific Visualization Engineer and Senior Cheminformatics Peer Reviewer for David Kolar's Bachelor's Thesis at UCT Prague (VŠCHT Praha) on:
"RNN De Novo Drug Design (DrugEx) — 3D Hawkins Conformer Ensembles, ROCS Shape/Color Alignment & Multi-Objective RL".

CONTEXT & RECENT EMPIRICAL FINDINGS:
We just ran our initial visualization suite on the completed RL sprint arm H1 (`sprint_h1_without_qed_pareto`, 1000 generated molecules, stopped at Epoch 86).
The diagnostic revealed:
1. Significant size drift / shrinkage exploit: Median HAC dropped from 26 in baseline to 20 in reinforced agent (-23.1%), and MW dropped from 378.4 to 330.9 (-12.6%).
2. In Figure 11 (ΔHAC vs ROCS), 97.8% of high-ROCS candidates had ΔHAC < -3 (shrinkage exploit: agent learned smaller fragments match the 3D core easily while maintaining low SA and high QED).

USER FEEDBACK & CRITIQUE:
1. Figure 01 & 04: Figure 04 (unfavorable features/size drift) feels redundant as a standalone plot. Instead, extend Figure 01 into a unified, hierarchical diagnostic dashboard:
   - Critical diagnostics on top (ROCS 3D overlap, Synthetic Accessibility SA, QED).
   - Followed by size drift & complexity diagnostics (MW, HAC, Rotatable Bonds).
   - Followed by physicochemical integrity (LogP, TPSA, Fsp3 / Ring count).
   - Eliminate Figure 04 as a separate figure.
2. Figure 06 & 07: Currently visually crude ("extremely ugly").
   - Figure 06 (Relative Property Shifts Tornado Plot): Needs our clean publication color palette (slate/indigo/teal/amber/rose) and better grouping.
   - Figure 07 (Murcko Scaffold Diversity): The current top-k coverage metric is trivial and uninformative. What richer, more meaningful medicinal chemistry and information-theoretic metrics should we measure?
3. Figure 11 & Epoch-Wise Diagnostics:
   - A single static post-hoc scatter of ΔHAC vs ROCS is less informative than longitudinal tracking.
   - User requested: Per-epoch trajectory of statistics (mean, median, IQR / 10th-90th percentiles) for each property across training epochs. This shows exactly WHEN and HOW the agent evolved, drifted, or shrank across epochs.
4. Other Useful Metrics:
   - What additional high-value medicinal chemistry metrics can we compute and track from SMILES (e.g., Fsp3, ring complexity, charge states, distance to ChEMBL bioactives, Murcko scaffold entropy)?

CONSULTATION QUESTIONS:
1. Figure 01 Architecture: What is the optimal 3x3 or 3x4 grid organization of property distributions (Baseline Prior vs. Reinforced Agent KDE + 95% CI + Seed vertical reference) to present a complete, unassailable diagnostic story?
2. Figure 07 Scaffold & Diversity Upgrade: What specific mathematical and structural metrics best capture true scaffold diversity and scaffold-hopping without being trivial? (e.g., Shannon entropy, Gini-Simpson index, Bemis-Murcko framework distance to seed, cyclic system census).
3. Longitudinal Training Trajectory (Replacing/Upgrading Fig 11 to Per-Epoch Multi-Property Panel): How should the per-epoch multi-property dashboard be structured to clearly display the dynamics of ROCS, SA, QED, MW, HAC, Rotatable Bonds, and LogP across training epochs?
4. Additional High-Value Metrics: Recommend 2-3 specific medicinal chemistry descriptors that add genuine value for 3D shape-matching RL without adding noise.

Please provide a structured, mathematically rigorous, and concise recommendation. Do NOT attempt to read binary image files.
"""

headers = {
    "Authorization": "Bearer sk-50dc3f278a5545cebfba2ce0d4016597",
    "Content-Type": "application/json",
}

models_to_consult = ["deepseek-v4.1-flash", "qwen3.8-flash-next"]
results = {}

for model in models_to_consult:
    print(f"\n--- Consulting {model} on e-INFRA ---")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4096,
        "temperature": 0.2,
    }
    
    max_retries = 5
    backoff = 3.0
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    "http://127.0.0.1:34053/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    ans = msg.get("content") or msg.get("reasoning_content") or ""
                    if ans:
                        results[model] = ans
                        print(f"Success from {model} ({len(ans)} chars)")
                        break
                    else:
                        print(f"Empty answer from {model}, retrying...")
                elif resp.status_code == 429:
                    print(f"429 rate limit on {model}, sleeping {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 1.5
                else:
                    print(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(backoff)
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            time.sleep(backoff)

output_file = Path("_david/output/subagents_visualization_feedback.md")
output_file.parent.mkdir(parents=True, exist_ok=True)
md_lines = ["# Subagent Peer Review & Recommendations for Visualization Overhaul\n"]
for model, ans in results.items():
    md_lines.append(f"## Consultation with {model}\n\n{ans}\n\n---\n")

output_file.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\nSaved combined peer review to {output_file}")
