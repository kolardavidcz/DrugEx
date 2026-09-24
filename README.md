# DrugEx Thesis Work & Time Tracking Platform

Standalone interactive telemetry dashboard and audit engine tracking all engineering, research, and literature study dedicated to the DrugEx thesis.

Deployed at: **`https://thesis-tracking-david.vercel.app/`** (or your custom Vercel domain)

---

## 🏛️ The Three Pillars of Effort

| Pillar | Measurement Method | Applied Buffer | Purpose |
| :--- | :--- | :--- | :--- |
| **Tablet Study & Reading** | Wire-format protobuf parser extracting 6,000+ strokes from `.notewise` | **+20% (`× 1.20`)** | Pre-reading exploration, post-reading synthesis, conceptual review |
| **Dichtator Remote Tunnel** | Windows Event Log (`vscht.cz`) + IDE launcher hooks | Direct | GPU execution, remote debugging, container runtime |
| **LLM Workspaces & Pair-Programming** | Turn clustering over 30+ Antigravity workspace trajectories | **+10% (`× 1.10`)** | Prompt engineering, reading documentation, code review |

---

## 📂 Repository Structure

```
.
├── index.html                   # Interactive dark-mode dashboard (served at / on Vercel)
├── vercel.json                  # Clean URLs & security caching headers
├── .gitignore                   # Ignore environment and local caches
├── tracker/
│   ├── thesis_tracker.py        # Core Python tracking & audit CLI
│   ├── thesis_tracking_db.json  # Persistent JSON database
│   ├── thesis_report.md         # Text-based summary report
│   ├── register_scheduler.ps1   # Windows Task Scheduler registration script
│   └── README.md                # Tracker component documentation
└── README.md                    # Project overview
```

---

## 🚀 Running the Audit

```powershell
python tracker/thesis_tracker.py --sync-all --push
```
