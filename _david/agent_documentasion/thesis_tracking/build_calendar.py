#!/usr/bin/env python3
"""Build the thesis workload proof calendar HTML.

Merges three data sources into one self-contained HTML page:

1. ``WEEKS`` — the curated dual-mode (strict / relaxed) event calendar,
   parsed out of the previous ``thesis_hourly_calendar.html`` (v1 backup).
2. ``evidence.json`` — every verified human steering message extracted from
   Antigravity transcript logs + T3 Code threads (see ``extract_evidence.py``).
3. A live "Week 8" shell so sessions on the generation day itself render.

The output is written to ``_david/thesis_hourly_calendar.html`` (gitignored).

Usage:
    python3 build_calendar.py
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(os.path.dirname(HERE), "thesis_hourly_calendar.html")
V1_BACKUP = os.path.join(HERE, "thesis_hourly_calendar_v1_backup.html")
TEMPLATE = os.path.join(HERE, "calendar_template.html")


def parse_weeks_v1() -> list[dict]:
    """Extract the WEEKS JSON blob from the v1 calendar HTML.

    The v1 live flag pointed at what is now Week 7; only the freshly inserted
    Week 8 shell is live, so the stale flag is cleared here.
    """
    html = open(V1_BACKUP, encoding="utf-8").read()
    i = html.find("var WEEKS = ")
    j = html.find(";\nvar currentWeekIdx", i)
    if i < 0 or j < 0:
        raise RuntimeError("could not locate WEEKS blob in v1 backup")
    weeks = json.loads(html[i + len("var WEEKS = "):j])
    for w in weeks:
        w["live"] = False
    return weeks


def week8() -> dict:
    """Live-tracking week shell (Mon Nov 2 – Sun Nov 8, 2026)."""
    monday = datetime(2026, 11, 2)
    days = []
    for k in range(7):
        dt = monday + timedelta(days=k)
        days.append({
            "name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k],
            "date": dt.strftime("%d %b"),
            "iso": dt.strftime("%Y-%m-%d"),
        })
    return {
        "k": "2026-11-02",
        "lbl": "Week 8: Nov 02 – Nov 08",
        "focus": "LIVE TRACKING — this proof calendar being built right now "
                 "(T3 Code thread + Antigravity steering sessions)",
        "days": days,
        "durs_strict": [0.0] * 7,
        "durs_relaxed": [0.0] * 7,
        "tot_strict": 0.0,
        "tot_relaxed": 0.0,
        "evs": [],
        "live": True,
    }


def main() -> None:
    weeks = parse_weeks_v1()
    weeks.insert(0, week8())  # newest-first ordering, matching v1

    ev = json.load(open(os.path.join(HERE, "evidence.json"), encoding="utf-8"))
    window_start = ev["window_start"]

    # Compact client-side message array (committee window only). All messages
    # ship to the client (they anchor session spans); `le` flags the
    # low-effort nudges that get filtered from proof counts and logs.
    msgs = []
    for m in ev["messages"]:
        if m["date"] < window_start:
            continue
        chat8 = m["chat"][:8]
        loc = datetime.fromisoformat(m["local"])
        msgs.append({
            "d": m["date"],
            "m": round((loc.hour * 60 + loc.minute + loc.second / 60.0), 2),
            "c": chat8,
            "u": (m["uuid"] or f"step-{m['step']}")[:8],
            "U": m["uuid"] or f"{m['chat']}:step{m['step']}",
            "w": m["words"],
            "x": m["excerpt"][:140],
            "o": m["out_tokens"],
            "n": m.get("turns", 0),
            "le": 1 if m.get("low_effort") else 0,
        })
    msgs.sort(key=lambda x: (x["d"], x["m"]))
    n_substantive = sum(1 for m in msgs if not m["le"])

    # Chat metadata (full ids for copy-to-clipboard proof).
    chats = {}
    for c in ev["chats"]:
        chats[c["chat"][:8]] = {
            "id": c["chat"],
            "msgs": c["msgs"],
            "turns": c["model_turns"],
            "tools": c["tool_calls"],
            "out_tok": c["out_tokens"],
            "harness": c["harness"],
        }

    meta = {
        "generated": ev["generated"],
        "gap_default": ev["gap_min"],
        "tz": ev["tz"],
    }

    def js(obj: object) -> str:
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = open(TEMPLATE, encoding="utf-8").read()
    html = (html
            .replace("{{WEEKS}}", js(weeks))
            .replace("{{MSGS}}", js(msgs))
            .replace("{{CHATS}}", js(chats))
            .replace("{{META}}", js(meta))
            .replace("{{NMSGS}}", f"{n_substantive:,}")
            .replace("{{NCHATS}}", str(len({m['c'] for m in msgs}))))
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)

    span = datetime.fromisoformat(ev["generated"]).astimezone(timezone.utc)
    print(f"wrote {OUT_PATH} ({len(html) // 1024} KB)")
    print(f"  weeks={len(weeks)} msgs={len(msgs)} "
          f"substantive={n_substantive} chats={len(chats)}")
    print(f"  generated={span.isoformat()}")


if __name__ == "__main__":
    main()
