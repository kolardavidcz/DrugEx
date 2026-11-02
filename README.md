# DrugEx Bachelor Thesis Workload Proof & Time Tracking Platform

Official thesis workload proof calendar, cognitive supervision ledger, and empirical research telemetry for David Kolář's Bachelor Thesis at UCT Prague (VŠCHT Praha):
> **"RNN De Novo Drug Design (DrugEx) — 3D Hawkins Conformer Ensembles, ROCS Shape/Color Alignment & Multi-Objective RL"**

Deployed at: **[https://thesis-stats-david.vercel.app/](https://thesis-stats-david.vercel.app/)**

---

## 🏛️ Dual-Mode Defense Architecture

The platform addresses both academic examination auditing and the realistic cognitive workload of an AI-augmented computational chemistry thesis:

### Mode 1 · Strict Committee Floor (`195.3h`)
- **Target Audience:** Skeptical university thesis committee, accreditation boards, academic auditors.
- **Methodology:** Unassailable empirical floor. Bounded by timestamped inter-turn active cycles (`104.2h`), live Cisco AnyConnect VPN remote development on Dichtator GPU cluster (`81.5h`), and Notewise tablet pen logs (`12.9h`).
- **Quarantines:** PBS Pro cluster batch jobs (`14.0h`) and idle VPN standby (`99.2h`) are strictly quarantined at `0.0×` human credit.
- **Sleep Invariant:** Hard circadian rest window (`04:00–09:30 CET`) where zero human prompts occurred.

### Mode 2 · Comprehensive Immersion & Deliberative Reading (`385.2h`)
- **Target Audience:** Internal peace of mind, full research accounting, supervisory tracking.
- **Methodology:** Fully quantifies supervisory steering, prompt drafting under dysgraphia, code AST auditing, and cognitive deliberation over **1,098,796 words** of reasoning traces, literature derivations, and terminal logs.
- **Novice Assimilation:** Applies a calibrated $\Phi_{\text{novice}} = 1.30$ factor (+30% conceptual assimilation) reflecting the candidate self-teaching deep learning, PyTorch tensors, Sequence RNNs, and RDKit molecular mechanics from ground zero.

---

## 📂 Repository Structure (Branch: `thesis-tracking`)

```
.
├── index.html                   # Primary interactive 24h Workload Proof Calendar & Evidence Viewer
├── thesis_hourly_calendar.html  # Standalone calendar mirror
├── legacy.html                  # Historical timeline tracker (Oct 16 snapshot)
├── vercel.json                  # Clean URLs & caching headers for Vercel
├── .gitignore                   # Ignore environment and local caches
├── tracker/
│   ├── build_calendar.py        # HTML builder synthesizing WEEKS, evidence, and transcripts
│   ├── calendar_template.html   # Responsive dark-theme calendar template
│   ├── evidence.json            # 1,023 verified human steering messages across 82 chats
│   ├── sessions.json            # Categorized research session intervals
│   ├── message_ledger.csv       # Flattened message timestamps and token metrics
│   ├── cognitive_supervision_reading_framework.md # Formal mathematical reading velocity model
│   ├── THESIS_TRACKING_MASTER_DOSSIER.md          # 1,900-line master forensic report
│   ├── deleted_conversations_catalog/             # Markdown catalogs of 188 recovered sessions
│   ├── thesis_report.md         # Text-based summary report
│   └── thesis_tracker.py        # Telemetry ingestion CLI
└── README.md                    # Platform documentation
```

---

## 🔄 Rebuilding the Calendar

To re-aggregate all telemetry and re-render the calendar:

```bash
python3 tracker/build_calendar.py
```
