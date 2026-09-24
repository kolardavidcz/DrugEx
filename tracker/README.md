# DrugEx Thesis Work & Time Tracking System

Continuous multi-pillar time tracking engine that audits, quantifies, and visualizes all engineering and research effort dedicated to the DrugEx bachelor's/master's thesis.

---

## 🏛️ The Three Pillars of Tracked Effort

1. **Tablet Study & Reading (Notewise)**:
   - Wire-format protobuf binary parser extracting 6,000+ stroke and annotation timestamps across all notebook pages.
   - Automatically credited with **+20% (`× 1.20`)** for pre-research, post-reading synthesis, and non-writing conceptual reflection.
2. **Dichtator / VSCHT VPN Sessions**:
   - Queries `Microsoft-Windows-NetworkProfile/Operational` and Cisco audit logs for connection sessions to `vpn.vscht.cz`.
   - IDE launch hook in `connect-dichtator.ps1` keeps connection logs up to date.
3. **LLM Workspaces & Pair-Programming**:
   - Deep forensic scanner over `~/.gemini/antigravity/brain/` transcripts matching regex `{.*}drug_?ex{.*}`, `dichtator`, `david.py`, `ba042aba`, etc.
   - Automatically credited with **+10% (`× 1.10`)** for prompt crafting, code reading, and review.

---

## 📊 Deployment & Output

- **Standalone Web Dashboard**: Generates [`../docs/stats.html`](../docs/stats.html), accessible on Vercel at `/stats`.
- **Database**: [`thesis_tracking_db.json`](thesis_tracking_db.json) persists all audited sessions and prevents duplicate counting.
- **Summary Report**: [`thesis_report.md`](thesis_report.md) provides a structured text summary for thesis documentation.

---

## 🚀 Usage

Run a full audit across all three pillars:
```bash
python thesis_tracker.py --sync-all
```

Or on Windows:
```powershell
python.exe tracker\thesis_tracker.py --sync-all
```
