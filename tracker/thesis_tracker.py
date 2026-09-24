#!/usr/bin/env python3
"""DrugEx Bachelor's/Master's Thesis Time & Work Tracker.

Tracks three core pillars of thesis effort:
1. Tablet Reading (Notewise .notewise exports) + 20% reading/research buffer
2. Dichtator / VSCHT VPN connection time (NetworkProfile event log + session hooks)
3. LLM Workspaces & Pair-Programming (~/.gemini/antigravity/brain/) + 10% review buffer

Generates persistent JSON database, Markdown report, and interactive Vercel-ready HTML dashboard.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DB_FILE = SCRIPT_DIR / "thesis_tracking_db.json"
REPORT_MD = SCRIPT_DIR / "thesis_report.md"
if (SCRIPT_DIR.parent / "index.html").exists():
    REPORT_HTML = SCRIPT_DIR.parent / "index.html"
elif (SCRIPT_DIR / "index.html").exists():
    REPORT_HTML = SCRIPT_DIR / "index.html"
else:
    REPORT_HTML = SCRIPT_DIR / "stats.html"
BRAIN_DIR = Path(os.environ.get("USERPROFILE", "C:/Users/kolar")) / ".gemini" / "antigravity" / "brain"

# Multipliers requested by user
NOTEWISE_BUFFER_MULTIPLIER = 1.20  # +20% for pre/post-research and non-writing reading
LLM_BUFFER_MULTIPLIER = 1.10       # +10% for reading, prompt writing, and review


# ==============================================================================
# 1. PROTOBUF HELPER (FOR NOTEWISE INGESTION)
# ==============================================================================

def decode_varint(data: bytes, offset: int) -> Tuple[int, int]:
    """Decodes a single protobuf varint."""
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, offset


class PbNode:
    """Represents a decoded protobuf tag/field node."""
    def __init__(self, field_number: int, wire_type: int, value: Any) -> None:
        self.field_number = field_number
        self.wire_type = wire_type
        self.value = value


def parse_protobuf(data: bytes) -> List[PbNode]:
    """Recursively parses raw protobuf wire format bytes."""
    nodes: List[PbNode] = []
    offset = 0
    while offset < len(data):
        try:
            tag, offset = decode_varint(data, offset)
        except Exception:
            break
        field_num = tag >> 3
        wire_type = tag & 0x07
        if field_num == 0 or field_num > 1000:
            break

        if wire_type == 0:  # Varint
            val, offset = decode_varint(data, offset)
            nodes.append(PbNode(field_num, wire_type, val))
        elif wire_type == 1:  # 64-bit
            val = data[offset:offset + 8]
            offset += 8
            nodes.append(PbNode(field_num, wire_type, val))
        elif wire_type == 2:  # Length-delimited
            length, offset = decode_varint(data, offset)
            val = data[offset:offset + length]
            offset += length
            try:
                sub = parse_protobuf(val)
                if sub:
                    nodes.append(PbNode(field_num, wire_type, sub))
                else:
                    nodes.append(PbNode(field_num, wire_type, val))
            except Exception:
                nodes.append(PbNode(field_num, wire_type, val))
        elif wire_type == 5:  # 32-bit
            val = data[offset:offset + 4]
            offset += 4
            nodes.append(PbNode(field_num, wire_type, val))
        else:
            break
    return nodes


# ==============================================================================
# 2. PILLAR 1: NOTEWISE INGESTION ENGINE
# ==============================================================================

class NotewiseIngester:
    """Ingests and analyzes .notewise export files."""

    def __init__(self, idle_threshold_minutes: int = 10) -> None:
        self.idle_threshold_ms = idle_threshold_minutes * 60 * 1000

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parses a .notewise zip file and calculates reading sessions."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        elements: List[Tuple[int, str, str]] = []  # (ts, page_name, elem_type)
        attachments: List[str] = []
        bookmarks: List[Dict[str, str]] = []

        with zipfile.ZipFile(file_path, "r") as zf:
            # 1. Parse note metadata
            if "note" in zf.namelist():
                cleaned = b"".join(zf.read("note").split())
                raw = base64.b64decode(cleaned)
                note_nodes = parse_protobuf(raw)
                for n in note_nodes:
                    if n.field_number == 6 and isinstance(n.value, list):
                        # PDF attachments
                        for sub in n.value:
                            if sub.field_number == 5 and isinstance(sub.value, bytes):
                                try:
                                    attachments.append(sub.value.decode("utf-8", errors="replace"))
                                except Exception:
                                    pass
                    elif n.field_number == 9 and isinstance(n.value, list):
                        # Bookmarks
                        bm_tag = ""
                        bm_page = ""
                        for sub in n.value:
                            if sub.field_number == 3 and isinstance(sub.value, bytes):
                                bm_page = sub.value.decode("utf-8", errors="replace")
                            elif sub.field_number == 4 and isinstance(sub.value, bytes):
                                bm_tag = sub.value.decode("utf-8", errors="replace")
                        if bm_tag:
                            bookmarks.append({"tag": bm_tag, "page": bm_page})

            # 2. Parse all page elements
            pages = [name for name in zf.namelist() if name.startswith("page/")]
            for page_name in pages:
                cleaned = b"".join(zf.read(page_name).split())
                raw = base64.b64decode(cleaned)
                page_nodes = parse_protobuf(raw)
                for pn in page_nodes:
                    if pn.field_number == 4:
                        sub = pn.value if isinstance(pn.value, list) else parse_protobuf(pn.value)
                        ts = None
                        elem_type = "stroke"
                        for en in sub:
                            if en.field_number == 2 and isinstance(en.value, int):
                                ts = en.value
                            elif en.field_number == 7:
                                elem_type = "image"
                            elif en.field_number == 8:
                                elem_type = "text"
                        if ts and ts > 1e11:  # Valid ms timestamp
                            elements.append((ts, page_name, elem_type))

        elements.sort(key=lambda x: x[0])
        sessions = self._cluster_sessions(elements)

        raw_duration_sec = sum(s["duration_sec"] for s in sessions)
        effective_duration_sec = raw_duration_sec * NOTEWISE_BUFFER_MULTIPLIER

        return {
            "source_file": str(file_path),
            "file_name": file_path.name,
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "total_elements": len(elements),
            "total_pages": len(pages),
            "attachments": attachments,
            "bookmarks": bookmarks,
            "sessions_count": len(sessions),
            "raw_duration_sec": raw_duration_sec,
            "raw_duration_hours": round(raw_duration_sec / 3600.0, 2),
            "buffer_multiplier": NOTEWISE_BUFFER_MULTIPLIER,
            "effective_duration_sec": effective_duration_sec,
            "effective_duration_hours": round(effective_duration_sec / 3600.0, 2),
            "sessions": sessions,
        }

    def _cluster_sessions(self, elements: List[Tuple[int, str, str]]) -> List[Dict[str, Any]]:
        """Clusters element timestamps into reading sessions based on idle gaps."""
        if not elements:
            return []

        sessions: List[List[Tuple[int, str, str]]] = []
        current: List[Tuple[int, str, str]] = []

        for elem in elements:
            if not current:
                current.append(elem)
            else:
                gap = elem[0] - current[-1][0]
                if gap > self.idle_threshold_ms:
                    sessions.append(current)
                    current = [elem]
                else:
                    current.append(elem)
        if current:
            sessions.append(current)

        result: List[Dict[str, Any]] = []
        for s in sessions:
            start_dt = datetime.fromtimestamp(s[0][0] / 1000.0)
            end_dt = datetime.fromtimestamp(s[-1][0] / 1000.0)
            dur = max((end_dt - start_dt).total_seconds(), 60.0)  # Min 1 min for quick notes
            pages = sorted(list(set(x[1] for x in s)))
            result.append({
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "duration_sec": dur,
                "duration_min": round(dur / 60.0, 1),
                "elements_count": len(s),
                "pages_touched": len(pages),
                "page_ids": pages[:3],
            })
        return result


# ==============================================================================
# 3. PILLAR 2: DICHTATOR / VPN TUNNEL TRACKER
# ==============================================================================

class DichtatorTracker:
    """Extracts and reconciles connection history to Dichtator and VSCHT VPN."""

    @staticmethod
    def query_network_profile_events() -> List[Dict[str, Any]]:
        """Queries Windows NetworkProfile log for vscht.cz and vscht connection sessions."""
        ps_cmd = """
        Get-WinEvent -LogName 'Microsoft-Windows-NetworkProfile/Operational' -ErrorAction SilentlyContinue | 
        ForEach-Object {
            [PSCustomObject]@{
                TimeCreated = $_.TimeCreated.ToString('o')
                Id = $_.Id
                Message = $_.Message
            }
        } | ConvertTo-Json -Compress
        """
        try:
            res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True)
            if not res.stdout.strip():
                return []
            events = json.loads(res.stdout)
            if isinstance(events, dict):
                events = [events]
        except Exception:
            return []

        vscht_events = []
        for ev in events:
            msg = ev.get("Message", "")
            if any(k in msg for k in ["vscht.cz", "vscht 2", "vscht"]):
                vscht_events.append(ev)

        vscht_events.reverse()  # Chronological order
        sessions: List[Dict[str, Any]] = []
        current_start: Optional[datetime] = None
        current_net: Optional[str] = None

        for ev in vscht_events:
            t = datetime.fromisoformat(ev["TimeCreated"])
            eid = ev.get("Id")
            msg = ev.get("Message", "")
            net = "vscht"
            for line in msg.splitlines():
                if "Name:" in line or "Název:" in line:
                    net = line.split(":", 1)[1].strip()

            if eid == 10000:  # Connected
                if current_start is None:
                    current_start = t
                    current_net = net
            elif eid == 10001:  # Disconnected
                if current_start is not None:
                    dur_sec = (t - current_start).total_seconds()
                    if dur_sec >= 10.0:  # Filter blips
                        sessions.append({
                            "start": current_start.isoformat(),
                            "end": t.isoformat(),
                            "duration_sec": dur_sec,
                            "duration_hours": round(dur_sec / 3600.0, 2),
                            "network": current_net,
                            "type": "NetworkProfile (VPN/Campus)",
                        })
                    current_start = None
                    current_net = None

        if current_start is not None:
            now = datetime.now(current_start.tzinfo)
            dur_sec = (now - current_start).total_seconds()
            sessions.append({
                "start": current_start.isoformat(),
                "end": now.isoformat(),
                "duration_sec": dur_sec,
                "duration_hours": round(dur_sec / 3600.0, 2),
                "network": current_net,
                "type": "Active Session",
            })

        return sessions


# ==============================================================================
# 4. PILLAR 3: ANTIGRAVITY WORKSPACE & LLM AUDITOR
# ==============================================================================

class AntigravityAuditor:
    """Audits conversation transcripts for DrugEx thesis workspaces and subagents."""

    SEARCH_PATTERNS = [
        re.compile(r".*drug_?ex.*", re.IGNORECASE),
        re.compile(r".*dichtator.*", re.IGNORECASE),
        re.compile(r".*david\.py.*", re.IGNORECASE),
        re.compile(r".*podcast_script.*", re.IGNORECASE),
    ]

    KNOWN_WORKSPACE_KEYWORDS = [
        "drugex", "drug_ex", "dichtator", "david.py",
        "home/kolar/drug_ex", "home/kolarv/drugex", "drug_ex podcast"
    ]

    def audit_all(self, brain_dir: Path = BRAIN_DIR) -> Dict[str, Any]:
        """Scans all Antigravity conversation transcripts and aggregates metrics."""
        if not brain_dir.exists():
            return {"conversations": [], "total_steps": 0, "net_hours": 0.0}

        matched_convs: List[Dict[str, Any]] = []

        for cid in os.listdir(brain_dir):
            cpath = brain_dir / cid
            if not cpath.is_dir() or cid == "tempmediaStorage":
                continue

            t_file = cpath / ".system_generated" / "logs" / "transcript.jsonl"
            if not t_file.exists():
                continue

            # Fast preliminary check: read first 8KB to check initial prompt, workspace, and config
            try:
                with open(t_file, "r", encoding="utf-8", errors="ignore") as f:
                    header = f.read(8192).lower()
                    if not any(k in header for k in self.KNOWN_WORKSPACE_KEYWORDS) and "drug" not in header and "dichtator" not in header:
                        # Quick check tail if large conversation
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        if size > 16384:
                            f.seek(max(0, size - 8192))
                            tail = f.read().lower()
                            if not any(k in tail for k in self.KNOWN_WORKSPACE_KEYWORDS) and "drug" not in tail and "dichtator" not in tail:
                                continue
                        else:
                            continue
            except Exception:
                continue

            is_match = False
            first_dt: Optional[datetime] = None
            last_dt: Optional[datetime] = None
            step_count = 0
            first_req = ""
            topics = set()
            step_timestamps: List[datetime] = []

            try:
                with open(t_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        step_count += 1
                        try:
                            s = json.loads(line)
                        except Exception:
                            continue
                        c_at = s.get("created_at")
                        if c_at:
                            dt = datetime.fromisoformat(c_at.replace("Z", "+00:00"))
                            step_timestamps.append(dt)
                            if not first_dt:
                                first_dt = dt
                            last_dt = dt

                        content = str(s.get("content", ""))
                        thinking = str(s.get("thinking", ""))
                        text_to_test = content + " " + thinking

                        for pat in self.SEARCH_PATTERNS:
                            if pat.search(text_to_test):
                                is_match = True
                                break

                        tool_calls = s.get("tool_calls", [])
                        for tc in tool_calls:
                            args = str(tc.get("arguments", ""))
                            if any(k in args.lower() for k in self.KNOWN_WORKSPACE_KEYWORDS):
                                is_match = True

                        if s.get("type") == "USER_INPUT" and not first_req:
                            first_req = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()[:100]

                        # Tagging topics
                        if "david.py" in text_to_test:
                            topics.add("david.py")
                        if "docstring" in text_to_test.lower() or "numpy" in text_to_test.lower():
                            topics.add("docstrings/types")
                        if "podcast" in text_to_test.lower():
                            topics.add("curriculum/podcast")
                        if "pipeline" in text_to_test.lower() or "schema" in text_to_test.lower():
                            topics.add("pipelines/schemas")
            except Exception:
                continue

            if is_match and step_timestamps:
                step_timestamps.sort()
                # Cluster into active interaction sessions (idle threshold: 15 min = 900s)
                conv_sessions: List[List[datetime]] = []
                curr_session: List[datetime] = []
                for st in step_timestamps:
                    if not curr_session:
                        curr_session.append(st)
                    else:
                        if (st - curr_session[-1]).total_seconds() > 900:
                            conv_sessions.append(curr_session)
                            curr_session = [st]
                        else:
                            curr_session.append(st)
                if curr_session:
                    conv_sessions.append(curr_session)

                # Sum active duration across sessions (min 1 min per session)
                active_sec = sum(max((cs[-1] - cs[0]).total_seconds(), 60.0) for cs in conv_sessions)

                matched_convs.append({
                    "cid": cid,
                    "start": step_timestamps[0].isoformat(),
                    "end": step_timestamps[-1].isoformat(),
                    "raw_duration_sec": active_sec,
                    "raw_duration_hours": round(active_sec / 3600.0, 2),
                    "steps": step_count,
                    "sessions_count": len(conv_sessions),
                    "prompt": first_req.replace("\n", " "),
                    "topics": list(topics),
                    "_intervals": [(cs[0], cs[-1]) for cs in conv_sessions],
                })

        matched_convs.sort(key=lambda x: x["start"])

        # Compute merged non-overlapping active intervals across all conversations
        all_intervals: List[Tuple[datetime, datetime]] = []
        for c in matched_convs:
            for s_dt, e_dt in c.pop("_intervals", []):
                all_intervals.append((s_dt, e_dt))

        all_intervals.sort(key=lambda x: x[0])
        merged: List[List[datetime]] = []
        for s_dt, e_dt in all_intervals:
            if not merged:
                merged.append([s_dt, e_dt])
            else:
                ps, pe = merged[-1]
                if s_dt <= pe:
                    merged[-1][1] = max(pe, e_dt)
                else:
                    merged.append([s_dt, e_dt])

        net_wall_clock_sec = sum((e - s for s, e in merged), timedelta()).total_seconds()
        effective_duration_sec = net_wall_clock_sec * LLM_BUFFER_MULTIPLIER

        return {
            "conversations_count": len(matched_convs),
            "total_steps": sum(c["steps"] for c in matched_convs),
            "net_wall_clock_sec": net_wall_clock_sec,
            "net_wall_clock_hours": round(net_wall_clock_sec / 3600.0, 2),
            "buffer_multiplier": LLM_BUFFER_MULTIPLIER,
            "effective_duration_sec": effective_duration_sec,
            "effective_duration_hours": round(effective_duration_sec / 3600.0, 2),
            "conversations": matched_convs,
        }


# ==============================================================================
# 5. CONSOLIDATED TRACKER CONTROLLER
# ==============================================================================

class ThesisTracker:
    """Main controller aggregating all tracking data and generating reports."""

    def __init__(self, db_path: Path = DB_FILE) -> None:
        self.db_path = db_path
        self.notewise_ingester = NotewiseIngester()
        self.dichtator_tracker = DichtatorTracker()
        self.antigravity_auditor = AntigravityAuditor()
        self.data: Dict[str, Any] = self._load_db()

    def _load_db(self) -> Dict[str, Any]:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "last_updated": None,
            "notewise": {},
            "dichtator": {},
            "antigravity": {},
            "totals": {},
        }

    def save_db(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def ingest_notewise(self, file_path: Path) -> None:
        """Parses and stores a .notewise file."""
        print(f"[Notewise] Ingesting: {file_path}")
        parsed = self.notewise_ingester.parse_file(file_path)
        self.data["notewise"][parsed["file_name"]] = parsed
        self._recalculate_totals()
        self.save_db()
        print(f"[Notewise] Successfully ingested {parsed['file_name']}: "
              f"{parsed['raw_duration_hours']}h raw -> {parsed['effective_duration_hours']}h effective (+20% buffer)")

    def auto_scan_notewise(self) -> None:
        """Scans standard mobile export directories for .notewise files."""
        candidates = [
            Path("C:/Users/kolar/Downloads/Mobile Devices/DrugEx.notewise"),
            Path("C:/Users/kolar/CrossDevice/Galaxy Tab S10 FE+/storage/Download/DrugEx.notewise"),
        ]
        found = False
        for c in candidates:
            if c.exists():
                self.ingest_notewise(c)
                found = True
        if not found:
            print("[Notewise] No candidate files found in default mobile download paths.")

    def sync_dichtator(self) -> None:
        """Syncs Dichtator VPN/campus connection sessions."""
        print("[Dichtator] Syncing connection sessions...")
        sessions = self.dichtator_tracker.query_network_profile_events()
        total_sec = sum(s["duration_sec"] for s in sessions)
        self.data["dichtator"] = {
            "sessions_count": len(sessions),
            "total_duration_sec": total_sec,
            "total_duration_hours": round(total_sec / 3600.0, 2),
            "sessions": sessions,
        }
        self._recalculate_totals()
        self.save_db()
        print(f"[Dichtator] Synced {len(sessions)} sessions ({round(total_sec/3600.0, 2)}h total).")

    def audit_workspaces(self) -> None:
        """Audits all Antigravity LLM workspaces and subagents."""
        print("[Antigravity] Auditing DrugEx workspaces and subagents...")
        audit = self.antigravity_auditor.audit_all()
        self.data["antigravity"] = audit
        self._recalculate_totals()
        self.save_db()
        print(f"[Antigravity] Audited {audit['conversations_count']} conversations, "
              f"{audit['total_steps']} steps: {audit['net_wall_clock_hours']}h net -> "
              f"{audit['effective_duration_hours']}h effective (+10% buffer).")

    def _recalculate_totals(self) -> None:
        """Computes grand totals across all pillars."""
        nw_effective_h = sum(
            item.get("effective_duration_hours", 0.0)
            for item in self.data.get("notewise", {}).values()
        )
        dich_h = self.data.get("dichtator", {}).get("total_duration_hours", 0.0)
        ag_effective_h = self.data.get("antigravity", {}).get("effective_duration_hours", 0.0)

        grand_total_h = round(nw_effective_h + dich_h + ag_effective_h, 2)

        self.data["totals"] = {
            "grand_total_hours": grand_total_h,
            "notewise_reading_effective_hours": nw_effective_h,
            "dichtator_tunnel_hours": dich_h,
            "antigravity_llm_effective_hours": ag_effective_h,
            "total_steps": self.data.get("antigravity", {}).get("total_steps", 0),
        }

    def generate_markdown_report(self) -> str:
        """Generates comprehensive thesis_report.md."""
        self._recalculate_totals()
        totals = self.data.get("totals", {})
        nw = self.data.get("notewise", {})
        dich = self.data.get("dichtator", {})
        ag = self.data.get("antigravity", {})

        md = f"""# DrugEx Thesis Work & Time Tracking Report

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

| Tracking Pillar | Raw Metrics | Multiplier | Effective Thesis Effort |
| :--- | :--- | :--- | :--- |
| **Tablet Reading ([Notewise](file:///C:/Users/kolar/Downloads/Mobile%20Devices/DrugEx.notewise))** | {sum(i.get('raw_duration_hours', 0.0) for i in nw.values())}h active writing | **+20%** (pre/post-research) | **{totals.get('notewise_reading_effective_hours', 0.0)} hours** |
| **Remote Compute & VPN (Dichtator)** | {dich.get('sessions_count', 0)} sessions | 1.0× direct | **{totals.get('dichtator_tunnel_hours', 0.0)} hours** |
| **LLM Workspaces & Pair-Programming** | {ag.get('net_wall_clock_hours', 0.0)}h wall-clock ({ag.get('total_steps', 0)} steps) | **+10%** (reading & review) | **{totals.get('antigravity_llm_effective_hours', 0.0)} hours** |
| **GRAND TOTAL THESIS EFFORT** | | | **{totals.get('grand_total_hours', 0.0)} hours** |

---

## 1. Tablet Reading Breakdown (Notewise)

*Note: All reading sessions carry a +20% buffer to credit non-writing contemplation, literature lookup, and post-reading synthesis.*

"""
        for fname, info in nw.items():
            md += f"### File: `{fname}`\n"
            md += f"- **Pages**: {info.get('total_pages', 0)} | **Annotations/Strokes**: {info.get('total_elements', 0)}\n"
            md += f"- **Raw Active Writing**: {info.get('raw_duration_hours', 0.0)}h | **Effective Study Time**: **{info.get('effective_duration_hours', 0.0)}h**\n"
            md += f"- **Attached References**: {', '.join(info.get('attachments', []))}\n"
            if info.get('bookmarks'):
                md += f"- **Key Bookmarks**: {', '.join(f'`{b['tag']}`' for b in info['bookmarks'])}\n"
            md += "\n"

        md += f"""---

## 2. Remote Compute & Dichtator Sessions

- **Total Connection Time Logged**: **{dich.get('total_duration_hours', 0.0)} hours** across **{dich.get('sessions_count', 0)} sessions**.
- **Coverage**: From initial connection setup (Sep 8/16) to current active session.

---

## 3. Antigravity Workspaces & Subagent Trajectories

- **Total Tracked Conversations**: **{ag.get('conversations_count', 0)}**
- **Total Trajectory Execution Steps**: **{ag.get('total_steps', 0)}**
- **Net Active Wall-Clock Time**: {ag.get('net_wall_clock_hours', 0.0)}h
- **Effective Development Time (+10% buffer)**: **{ag.get('effective_duration_hours', 0.0)} hours**

### Key Conversations Tracked:
"""
        for c in ag.get("conversations", [])[:15]:
            md += f"- **`{c['cid'][:8]}...`** ({round(c['raw_duration_sec']/3600.0, 1)}h, {c['steps']} steps): {c['prompt'][:80]}...\n"

        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md)
        return md

    def generate_html_dashboard(self) -> str:
        """Generates a modern, single-file, interactive HTML dashboard suitable for Vercel/GitHub Pages."""
        self._recalculate_totals()
        totals = self.data.get("totals", {})
        nw = self.data.get("notewise", {})
        dich = self.data.get("dichtator", {})
        ag = self.data.get("antigravity", {})

        # Prepare JSON data for client-side charts & tables
        nw_sessions = []
        for f_data in nw.values():
            for s in f_data.get("sessions", []):
                nw_sessions.append(s)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DrugEx Thesis Time Tracker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090a0f;
      --card-bg: rgba(22, 27, 34, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #38bdf8;
      --accent-purple: #a855f7;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, sans-serif;
      background: radial-gradient(circle at 50% 0%, #171d2d, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2.5rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 1.5rem;
    }}
    .title-group h1 {{
      font-size: 1.875rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, #fff 30%, var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .title-group p {{
      color: var(--text-muted);
      font-size: 0.875rem;
      margin-top: 0.25rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--accent);
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .grid-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.5rem;
      backdrop-filter: blur(12px);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.18);
    }}
    .kpi-label {{
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }}
    .kpi-val {{
      font-size: 2.25rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .kpi-sub {{
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 0.35rem;
    }}
    .progress-bar-container {{
      background: rgba(255,255,255,0.06);
      border-radius: 8px;
      height: 8px;
      overflow: hidden;
      margin-top: 0.75rem;
      display: flex;
    }}
    .progress-fill {{
      height: 100%;
      transition: width 0.5s ease;
    }}
    .section-title {{
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow-x: auto;
      margin-bottom: 2.5rem;
      backdrop-filter: blur(12px);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.85rem;
    }}
    th {{
      padding: 0.85rem 1.25rem;
      background: rgba(255,255,255,0.03);
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--card-border);
    }}
    td {{
      padding: 1rem 1.25rem;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      color: #cbd5e1;
    }}
    tr:hover td {{
      background: rgba(255,255,255,0.02);
    }}
    code {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      background: rgba(255,255,255,0.06);
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      color: var(--accent);
    }}
    footer {{
      text-align: center;
      color: var(--text-muted);
      font-size: 0.8rem;
      margin-top: 3rem;
      border-top: 1px solid var(--card-border);
      padding-top: 1.5rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>DrugEx Thesis Work & Time Tracker</h1>
        <p>Continuous automated audit across Tablet, Remote Compute, and LLM Engineering</p>
      </div>
      <div class="badge">Vercel Ready</div>
    </header>

    <div class="grid-kpis">
      <div class="card" style="border-top: 3px solid var(--accent);">
        <div class="kpi-label">Grand Total Thesis Effort</div>
        <div class="kpi-val" style="color: #fff;">{totals.get('grand_total_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">Equivalent to ~{(totals.get('grand_total_hours', 0.0)/40.0):.1f} full-time work weeks</div>
        <div class="progress-bar-container">
          <div class="progress-fill" style="width: {(totals.get('notewise_reading_effective_hours', 0.0)/max(totals.get('grand_total_hours', 1.0),1.0))*100}%; background: var(--accent-purple);" title="Notewise"></div>
          <div class="progress-fill" style="width: {(totals.get('dichtator_tunnel_hours', 0.0)/max(totals.get('grand_total_hours', 1.0),1.0))*100}%; background: var(--accent-emerald);" title="Dichtator"></div>
          <div class="progress-fill" style="width: {(totals.get('antigravity_llm_effective_hours', 0.0)/max(totals.get('grand_total_hours', 1.0),1.0))*100}%; background: var(--accent);" title="LLM Workspaces"></div>
        </div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent-purple);">
        <div class="kpi-label">Tablet Study & Reading</div>
        <div class="kpi-val" style="color: var(--accent-purple);">{totals.get('notewise_reading_effective_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">{sum(i.get('raw_duration_hours', 0.0) for i in nw.values())}h raw + 20% reading buffer</div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent-emerald);">
        <div class="kpi-label">Dichtator / VPN Tunnel</div>
        <div class="kpi-val" style="color: var(--accent-emerald);">{totals.get('dichtator_tunnel_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">{dich.get('sessions_count', 0)} sessions to vpn.vscht.cz</div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent);">
        <div class="kpi-label">LLM Engineering</div>
        <div class="kpi-val" style="color: var(--accent);">{totals.get('antigravity_llm_effective_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">{ag.get('total_steps', 0)} steps across {ag.get('conversations_count', 0)} workspaces (+10% buffer)</div>
      </div>
    </div>

    <div class="section-title">Antigravity Thesis Workspaces & Subagent Sweeps</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Workspace / Conversation</th>
            <th>Start (Local)</th>
            <th>Duration</th>
            <th>Steps</th>
            <th>Primary Topics</th>
            <th>Initial User Request</th>
          </tr>
        </thead>
        <tbody>
"""
        for c in ag.get("conversations", [])[::-1][:15]:
            start_str = c["start"][:19].replace("T", " ")
            dur_str = f"{round(c['raw_duration_sec']/3600.0, 1)}h"
            topics_html = " ".join(f"<code>{t}</code>" for t in c.get("topics", []))
            html += f"""          <tr>
            <td><code>{c['cid'][:8]}...</code></td>
            <td>{start_str}</td>
            <td>{dur_str}</td>
            <td><strong>{c['steps']}</strong></td>
            <td>{topics_html if topics_html else '<code>core</code>'}</td>
            <td style="max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{c['prompt']}</td>
          </tr>\n"""

        html += f"""        </tbody>
      </table>
    </div>

    <div class="section-title">Tablet Reading Sessions (Notewise Protobuf Milestones)</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Start Time</th>
            <th>Duration</th>
            <th>Strokes / Notes</th>
            <th>Pages Touched</th>
          </tr>
        </thead>
        <tbody>
"""
        for i, s in enumerate(nw_sessions[::-1][:15]):
            start_str = s["start"][:19].replace("T", " ")
            html += f"""          <tr>
            <td>Session {len(nw_sessions) - i:02d}</td>
            <td>{start_str}</td>
            <td>{s['duration_min']} min</td>
            <td><strong>{s['elements_count']}</strong> annotations</td>
            <td>{s['pages_touched']} pages</td>
          </tr>\n"""

        html += f"""        </tbody>
      </table>
    </div>

    <footer>
      DrugEx Thesis Automated Tracking System &bull; Single Source of Truth
    </footer>
  </div>
</body>
</html>
"""
        REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        return html


def git_commit_and_push(branch: str = "thesis-stats") -> None:
    """Commits and pushes updated database, report, and HTML dashboard to origin."""
    try:
        repo_dir = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tracker" else SCRIPT_DIR
        # Run git add
        subprocess.run(["git", "add", "index.html", "tracker/"], cwd=str(repo_dir), check=False)
        msg = f"chore(stats): automated weekly sync [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
        res = subprocess.run(["git", "commit", "-m", msg], cwd=str(repo_dir), capture_output=True, text=True)
        if "nothing to commit" not in res.stdout and "nothing to commit" not in res.stderr:
            print(f"[Git] Committed: {msg}")
            push_res = subprocess.run(["git", "push", "origin", branch], cwd=str(repo_dir), capture_output=True, text=True)
            if push_res.returncode == 0:
                print(f"[Git] Successfully pushed to origin/{branch}!")
            else:
                print(f"[Git] Push note: {push_res.stderr.strip() or push_res.stdout.strip()}")
        else:
            print("[Git] Working tree clean, no new metrics to commit.")
    except Exception as e:
        print(f"[Git] Note: Could not auto-push ({e})")


# ==============================================================================
# 6. CLI ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DrugEx Thesis Time Tracker")
    parser.add_argument("--sync-all", action="store_true", help="Run full audit across all 3 pillars")
    parser.add_argument("--ingest-notewise", type=str, help="Ingest a specific .notewise export file")
    parser.add_argument("--auto-notewise", action="store_true", help="Auto-discover and ingest candidate .notewise files")
    parser.add_argument("--sync-dichtator", action="store_true", help="Sync Dichtator VPN connection history")
    parser.add_argument("--audit-workspaces", action="store_true", help="Audit all Antigravity LLM workspaces")
    parser.add_argument("--report", action="store_true", help="Generate Markdown and HTML reports")
    parser.add_argument("--push", action="store_true", help="Auto-commit and push updated stats to GitHub origin/thesis-stats")

    args = parser.parse_args()
    tracker = ThesisTracker()

    if len(sys.argv) == 1 or args.sync_all:
        tracker.auto_scan_notewise()
        tracker.sync_dichtator()
        tracker.audit_workspaces()
        tracker.generate_markdown_report()
        tracker.generate_html_dashboard()
        print("\n[Done] Full sync complete! Reports generated:")
        print(f"  - Markdown: {REPORT_MD}")
        print(f"  - HTML:     {REPORT_HTML}")
        if args.push:
            git_commit_and_push()
        return

    if args.ingest_notewise:
        tracker.ingest_notewise(Path(args.ingest_notewise))
    if args.auto_notewise:
        tracker.auto_scan_notewise()
    if args.sync_dichtator:
        tracker.sync_dichtator()
    if args.audit_workspaces:
        tracker.audit_workspaces()
    if args.report:
        tracker.generate_markdown_report()
        tracker.generate_html_dashboard()
        print(f"[Done] Reports written to:\n  - {REPORT_MD}\n  - {REPORT_HTML}")
    if args.push:
        git_commit_and_push()


if __name__ == "__main__":
    main()
