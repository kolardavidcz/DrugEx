import os
import time
import httpx
import json

PROMPT = """You are a Principal Scientific Visualization Engineer and Senior Cheminformatics Peer Reviewer for David Kolar's Bachelor's Thesis at UCT Prague (VŠCHT Praha) on:
"RNN De Novo Drug Design (DrugEx) — 3D Hawkins Conformer Ensembles, ROCS Shape/Color Alignment & Multi-Objective RL".

CONTEXT:
We are developing the publication-grade visualization suite for DrugEx RL (SequenceRNN de novo drug design guided by ROCS 3D molecular shape & color TanimotoCombo scoring).
A 4-hypothesis sprint is currently executing on MetaCentrum:
- H1: without QED, Pareto crowding (Completed at Epoch 86 via early stopping, best checkpoint at Epoch 35)
- H2: without QED, Harrington desirability (Running on cluster node elmu1-22)
- H3: with QED, Pareto crowding (Running on cluster node elmu1-24)
- H4: with QED, Harrington desirability (Running on cluster node elmu1-25)

USER PREFERENCES & SCIENTIFIC INTEGRITY BOUNDARIES (Commit 5293122):
- Strict reporting honesty: No misleading averages, no unphysical splines overshooting into unachievable Pareto space, straggler-aware reporting.
- Actionable medicinal chemistry insight: Anti-shrinkage diagnostics, genuine Pareto trade-offs, direct lead comparison.
- Rule: Do NOT open or read binary image files (.png, .jpg). Rely strictly on mathematical principles, cheminformatics data schemas (TSV/CSV/code), and scientific visualization rigor. If visual analysis is ever needed, instruct the subagent Gemini 3.8 to inspect it.

PROPOSED FIGURE IMPROVEMENTS & NEW ADDITIONS:

1. Figure 02 (`02_multiobjective_tradeoffs.png`):
   - Current: Simple scatter plot of ROCS vs SA.
   - Proposed: Trace empirical non-dominated frontier using step-wise dominance line (`drawstyle="steps-post"`), explicitly avoiding artificial splines.
   - Separate 2D pairwise envelope from true 3D Pareto-optimal points (mark true 3D Pareto leads with distinct star markers).
   - Overlay prior baseline 50% and 90% density contour envelopes to clearly show population shift.

2. Figure 03 (`03_training_trajectories.png`):
   - Current: Epoch-wise median and IQR.
   - Proposed: Mark Best Checkpoint (Epoch 35) and Early Stopping (Epoch 86) with vertical dashed lines and badge annotations.
   - Add a cumulative rolling p95 curve to show the upper ceiling of generated molecules without masking population variance.

3. Figure 05 (`05_lead_structures_gallery.png`):
   - Current: Static RDKit grid of top leads.
   - Proposed: Align 2D depictions to reference scaffold/Murcko core, with robust fallback to MCS (Maximum Common Substructure) when scaffold hopping occurs.
   - Render medicinal chemistry property badge pills (ROCS, SA, QED, MW) beneath each molecule card with color-coded thresholds.

4. NEW Figure 09 (Cross-Hypothesis 4-Arm Comparative Diagnostic):
   - Parallel Coordinates plot comparing H1, H2, H3, H4 across normalized metrics (ROCS, SA, QED, HAC preservation, Murcko diversity).
   - Overlay bootstrap 95% confidence intervals and paired dot-and-whisker distributions rather than spider/radar plots (to avoid polygon area distortion).

5. NEW Figure 10 (3D Lead-to-Seed Alignment Superposition):
   - 3D coordinate superposition of top prioritized leads aligned onto the query target shape conformer.
   - Recover aligned 3D poses via `rdShapeAlign.AlignMol` in-place coordinate update.
   - Export accompanying publication-grade `top_leads_aligned_to_seed.sdf` for PyMOL / Mol* interactive inspection.

6. NEW Figure 11 (Anti-Shrinkage Diagnostic: Delta HAC vs ROCS):
   - Address the known RL exploit where generators produce tiny inactive fragments (false ROCS positives).
   - Joint plot of Delta HAC (Heavy Atom Count difference relative to seed) vs ROCS score.
   - 4-quadrant diagnostic with shaded isosteric zone (|Delta HAC| <= 3).

REVIEW QUESTIONS FOR DEEPSEEK:
1. Mathematical & Cheminformatics Rigor: Evaluate the proposed step-wise Pareto front, 2D vs 3D non-dominated sorting, and MCS alignment fallback. Are there subtle mathematical pitfalls or edge cases?
2. Anti-Shrinkage & Diagnostic Validity: Does Figure 11 properly guard against reward gaming? What specific quadrant boundaries or annotations should be enforced?
3. Cross-Arm Synthesis: For Figure 09, how should parallel coordinates handle differing objective weights (e.g., H1/H2 without QED vs H3/H4 with QED)?
4. Final Verdict: Provide your critique, specific formulaic or implementation recommendations, and approval status.
"""

payload = {
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 4096,
    "temperature": 0.2
}

headers = {
    "Authorization": "Bearer sk-50dc3f278a5545cebfba2ce0d4016597",
    "Content-Type": "application/json"
}

print("Initiating consultation with DeepSeek v4.1 Flash on e-INFRA...")
max_retries = 10
backoff = 4.0

for attempt in range(max_retries):
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post("http://127.0.0.1:34053/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 429:
                print(f"[Attempt {attempt+1}/{max_retries}] Upstream 429 (concurrency limit), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 15.0)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if not content and "provider_specific_fields" in msg:
                content = (msg["provider_specific_fields"] or {}).get("reasoning", "")
            
            output_path = "/home/kolar/DrugEx/_david/output/deepseek_visualization_review.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"DeepSeek review completed successfully! Output saved to: {output_path}")
            print("\n=== SUMMARY OF DEEPSEEK VERDICT ===")
            print(content[:1500])
            break
    except Exception as e:
        print(f"[Attempt {attempt+1}/{max_retries}] Error: {e}, retrying in {backoff:.1f}s...")
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 15.0)
