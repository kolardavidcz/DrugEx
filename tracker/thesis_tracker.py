#!/usr/bin/env python3
"""DrugEx Bachelor's/Master's Thesis Time & Work Tracker.

Tracks core pillars of thesis effort:
1. Tablet Reading (Notewise .notewise exports) + 20% reading/research buffer
2. Dichtator / VSCHT VPN connection time (NetworkProfile event log + session hooks)
   - Reconciled into Active Dev / Code Review, Autonomous Remote Compute, and Idle Tunnel Uptime
3. AI Engineering & LLM Workspaces (Antigravity Windows & WSL + t3code WSL)
   - Physics-based timing: in-turn steering (10m), prompt crafting (15m), and artifact reading (15-20m)
   - Disaggregated into Interactive Human-AI Steered Work (+10% review buffer) vs Autonomous AI Loops
4. Day-by-Day Timeline & Verifiable Proof-of-Work Ledger (Git commits, artifacts, prompts, and thesis chapters)

Generates persistent JSON database, Markdown report, and modern responsive HTML dashboard.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DB_FILE = SCRIPT_DIR / "thesis_tracking_db.json"
REPORT_MD = SCRIPT_DIR / "thesis_report.md"
REPORT_HTML = SCRIPT_DIR / "stats.html"

# Multipliers & Physics-Based Interaction Buffers
NOTEWISE_BUFFER_MULTIPLIER = 1.20       # +20% for pre/post-research and conceptual contemplation
LLM_BUFFER_MULTIPLIER = 1.10            # +10% for code review and synthesis
STEERED_IN_TURN_WINDOW_SEC = 600.0      # 10 min max active human steering during model execution
PROMPT_CRAFTING_STRUCTURED_SEC = 900.0  # 15 min prompt crafting & research for detailed prompts
PROMPT_CRAFTING_SHORT_SEC = 450.0       # 7.5 min prompt crafting for quick follow-ups
RESULT_READING_ARTIFACT_SEC = 1080.0    # 18 min for reading .md artifacts/reports/plans
RESULT_READING_STANDARD_SEC = 600.0     # 10 min for inspecting code diffs / standard responses
ACTIVE_INTER_TURN_CHAIN_SEC = 2700.0    # 45 min threshold to chain continuous research into next prompt
IDLE_CLUSTER_GAP_SEC = 900.0            # 15 min idle threshold for raw model execution clustering

# Discovery directories (Windows & WSL)
WIN_BRAIN_DIR = Path(os.environ.get("USERPROFILE", "C:/Users/kolar")) / ".gemini" / "antigravity" / "brain"
WSL_BRAIN_DIR = Path(r"\\wsl.localhost\Ubuntu\home\kolar\.gemini\antigravity\brain")
WSL_BRAIN_FALLBACK = Path(r"\\wsl$\Ubuntu\home\kolar\.gemini\antigravity\brain")
LINUX_BRAIN_DIR = Path("/home/kolar/.gemini/antigravity/brain")

WSL_T3_DB = Path(r"\\wsl.localhost\Ubuntu\home\kolar\.t3\userdata\statev2.sqlite")
WSL_T3_FALLBACK = Path(r"\\wsl$\Ubuntu\home\kolar\.t3\userdata\statev2.sqlite")
LINUX_T3_DB = Path("/home/kolar/.t3/userdata/statev2.sqlite")

DRUGEX_REPO_WSL = Path(r"\\wsl.localhost\Ubuntu\home\kolar\DrugEx")
DRUGEX_REPO_LINUX = Path("/home/kolar/DrugEx")


def format_duration(seconds: float) -> str:
    """Formats duration adaptively into seconds, minutes, or hours with high precision."""
    if seconds < 60:
        return f"{max(1, round(seconds))}s"
    if seconds < 3600:
        mins = round(seconds / 60.0, 1)
        return f"{int(mins)} min" if mins.is_integer() else f"{mins} min"
    hours = round(seconds / 3600.0, 1)
    return f"{int(hours)}h" if hours.is_integer() else f"{hours}h"


def to_utc(dt_val: Any) -> datetime:
    """Normalizes a string or datetime into an offset-aware UTC datetime."""
    if isinstance(dt_val, str):
        dt_val = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=timezone.utc)
    return dt_val.astimezone(timezone.utc)


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
            if "note" in zf.namelist():
                cleaned = b"".join(zf.read("note").split())
                raw = base64.b64decode(cleaned)
                note_nodes = parse_protobuf(raw)
                for n in note_nodes:
                    if n.field_number == 6 and isinstance(n.value, list):
                        for sub in n.value:
                            if sub.field_number == 5 and isinstance(sub.value, bytes):
                                try:
                                    attachments.append(sub.value.decode("utf-8", errors="replace"))
                                except Exception:
                                    pass
                    elif n.field_number == 9 and isinstance(n.value, list):
                        bm_tag = ""
                        bm_page = ""
                        for sub in n.value:
                            if sub.field_number == 3 and isinstance(sub.value, bytes):
                                bm_page = sub.value.decode("utf-8", errors="replace")
                            elif sub.field_number == 4 and isinstance(sub.value, bytes):
                                bm_tag = sub.value.decode("utf-8", errors="replace")
                        if bm_tag:
                            bookmarks.append({"tag": bm_tag, "page": bm_page})

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
            dur = max((end_dt - start_dt).total_seconds(), 60.0)
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
# 3. PILLAR 3: MULTI-SOURCE AI HARNESS AUDITOR & TIMING PHYSICS
# ==============================================================================

class AIHarnessAuditor:
    """Audits Antigravity (Windows & WSL) and t3code (WSL) workspaces with realistic timing physics."""

    KNOWN_KEYWORDS = [
        "drugex", "drug_ex", "dichtator", "david.py", "pareto", "conformer",
        "metacentrum", "sprint", "docking", "chembl", "qsar", "reinvent", "rocs",
        "egcg", "_david", "cisco", "connect-dichtator", "drugex-ssh"
    ]

    EXCLUDE_KEYWORDS = [
        "vscht-uceni", "vscht-uceni-web", "wsl-ide-dotfiles"
    ]

    @classmethod
    def is_thesis_relevant(cls, text: str, project_root: str = "") -> bool:
        """Determines if text or project context belongs to DrugEx thesis."""
        text_l = text.lower()
        path_l = project_root.lower()
        if "drugex" in path_l or "/home/kolar/drugex" in path_l or "kolarv/drugex" in path_l:
            return True
        # Exclude non-thesis study portals or dotfiles if they lack core thesis keywords
        if any(k in text_l for k in cls.EXCLUDE_KEYWORDS) and not any(k in text_l for k in ["david.py", "conformer", "rocs", "pareto", "egcg"]):
            return False
        return any(k in text_l for k in cls.KNOWN_KEYWORDS)

    @staticmethod
    def compute_intervals(events: List[Tuple[datetime, bool, bool, int]]) -> Tuple[float, float, List[Tuple[datetime, datetime]], List[Tuple[datetime, datetime]]]:
        """Disaggregates timestamped events into Steered vs Autonomous intervals using realistic human-AI physics.

        Accounts for:
        1. Pre-turn prompt crafting & code research (15 min structured, 7.5 min short)
        2. In-turn model steering (up to 10 min during execution)
        3. Post-turn result reading & artifact evaluation (18 min for .md reports, 10 min standard)
        4. Inter-turn active research chaining (< 45 min)
        5. Clean cutoff of dead overnight sleep hours (> 60-90 min gaps after ~3-4 AM)

        Args:
            events: Chronological list of (timestamp, is_user_turn, has_artifact, prompt_length).

        Returns:
            steered_sec, autonomous_sec, steered_intervals, autonomous_intervals
        """
        if not events:
            return 0.0, 0.0, [], []

        events.sort(key=lambda x: x[0])
        user_indices = [i for i, ev in enumerate(events) if ev[1]]

        # Case 1: Pure autonomous subagent with zero user inputs
        if not user_indices:
            start = events[0][0]
            end = events[-1][0]
            dur = max((end - start).total_seconds(), 60.0)
            return 0.0, dur, [], [(start, end)]

        # Case 2: Interactive conversation with human steering
        raw_steered_intervals: List[Tuple[datetime, datetime]] = []

        for u_pos, idx in enumerate(user_indices):
            u_dt, _, _, prompt_len = events[idx]
            prev_u_dt = events[user_indices[u_pos - 1]][0] if u_pos > 0 else None
            next_u_dt = events[user_indices[u_pos + 1]][0] if u_pos + 1 < len(user_indices) else None

            # 1. Pre-turn prompt drafting & research
            # Detect queued prompts (< 45s from previous prompt)
            is_queued = (prev_u_dt is not None) and ((u_dt - prev_u_dt).total_seconds() < 45.0)
            if not is_queued:
                crafting_sec = PROMPT_CRAFTING_STRUCTURED_SEC if prompt_len > 120 else PROMPT_CRAFTING_SHORT_SEC
                crafting_start = u_dt - timedelta(seconds=crafting_sec)
                if prev_u_dt and crafting_start < prev_u_dt:
                    crafting_start = prev_u_dt + timedelta(seconds=1.0)
                if crafting_start < u_dt:
                    raw_steered_intervals.append((crafting_start, u_dt))

            # 2. In-turn model execution & steering
            turn_end_event = u_dt
            has_artifact = False
            for i in range(idx + 1, len(events)):
                if events[i][1]:  # next user turn
                    break
                turn_end_event = events[i][0]
                if events[i][2]:  # has artifact or report
                    has_artifact = True

            in_turn_end = min(turn_end_event, u_dt + timedelta(seconds=STEERED_IN_TURN_WINDOW_SEC))
            if in_turn_end > u_dt:
                raw_steered_intervals.append((u_dt, in_turn_end))

            # 3. Post-turn result reading & contemplation
            reading_sec = RESULT_READING_ARTIFACT_SEC if has_artifact else RESULT_READING_STANDARD_SEC
            reading_end = turn_end_event + timedelta(seconds=reading_sec)

            # 4. Inter-turn research chaining
            if next_u_dt:
                gap_to_next = (next_u_dt - turn_end_event).total_seconds()
                if gap_to_next <= ACTIVE_INTER_TURN_CHAIN_SEC:  # <= 45 min
                    # Continuous active research into next prompt
                    raw_steered_intervals.append((turn_end_event, next_u_dt))
                else:
                    # Gapped / overnight sleep: credit reading window only
                    if reading_end > turn_end_event:
                        raw_steered_intervals.append((turn_end_event, reading_end))
            else:
                if reading_end > turn_end_event:
                    raw_steered_intervals.append((turn_end_event, reading_end))

        # Merge overlapping steered intervals
        raw_steered_intervals.sort(key=lambda x: x[0])
        merged_steered: List[List[datetime]] = []
        for s, e in raw_steered_intervals:
            if not merged_steered:
                merged_steered.append([s, e])
            else:
                if s <= merged_steered[-1][1]:
                    merged_steered[-1][1] = max(merged_steered[-1][1], e)
                else:
                    merged_steered.append([s, e])

        steered_intervals = [(s, e) for s, e in merged_steered]
        steered_sec = sum((e - s).total_seconds() for s, e in steered_intervals)

        # Cluster all events to find total active execution time
        all_clusters: List[List[datetime]] = []
        curr: List[datetime] = []
        for ev in events:
            if not curr:
                curr.append(ev[0])
            else:
                if (ev[0] - curr[-1]).total_seconds() > IDLE_CLUSTER_GAP_SEC:
                    all_clusters.append(curr)
                    curr = [ev[0]]
                else:
                    curr.append(ev[0])
        if curr:
            all_clusters.append(curr)

        total_active_sec = sum(max((c[-1] - c[0]).total_seconds(), 60.0) for c in all_clusters)
        autonomous_sec = max(0.0, total_active_sec - steered_sec)

        autonomous_intervals: List[Tuple[datetime, datetime]] = []
        for c in all_clusters:
            c_start, c_end = c[0], c[-1]
            has_user = any(ev[1] for ev in events if c_start <= ev[0] <= c_end)
            if not has_user:
                autonomous_intervals.append((c_start, c_end))
            else:
                cluster_steered = [st for st in steered_intervals if st[0] >= c_start and st[1] <= c_end + timedelta(seconds=60)]
                if cluster_steered:
                    last_st_end = max(st[1] for st in cluster_steered)
                    if last_st_end < c_end and (c_end - last_st_end).total_seconds() > 60.0:
                        autonomous_intervals.append((last_st_end, c_end))

        return steered_sec, autonomous_sec, steered_intervals, autonomous_intervals

    def audit_antigravity_dir(self, brain_dir: Path, source_label: str) -> List[Dict[str, Any]]:
        """Audits an Antigravity brain directory."""
        if not brain_dir.exists():
            return []

        results: List[Dict[str, Any]] = []
        for cid in os.listdir(brain_dir):
            cpath = brain_dir / cid
            if not cpath.is_dir() or cid == "tempmediaStorage":
                continue

            t_file = cpath / ".system_generated" / "logs" / "transcript.jsonl"
            if not t_file.exists():
                continue

            step_count = 0
            user_count = 0
            events: List[Tuple[datetime, bool, bool, int]] = []
            is_match = False
            first_prompt = ""
            topics = set()
            prompts_by_day: Dict[str, List[str]] = defaultdict(list)

            try:
                with open(t_file, "r", encoding="utf-8-sig", errors="ignore") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        step_count += 1
                        try:
                            s = json.loads(line)
                        except Exception:
                            continue

                        c_at = s.get("created_at")
                        is_user = (s.get("type") == "USER_INPUT" or s.get("source") == "USER_EXPLICIT")
                        content = str(s.get("content", ""))
                        thinking = str(s.get("thinking", ""))
                        text = content + " " + thinking
                        has_artifact = (".md" in content or "REPORT" in content or "artifact" in content)

                        if is_user:
                            user_count += 1
                            if c_at:
                                clean_p = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()[:140]
                                if clean_p:
                                    d_key = c_at[:10]
                                    if clean_p not in prompts_by_day[d_key] and len(prompts_by_day[d_key]) < 8:
                                        prompts_by_day[d_key].append(clean_p)

                        if c_at:
                            try:
                                dt = datetime.fromisoformat(c_at.replace("Z", "+00:00"))
                                events.append((dt, is_user, has_artifact, len(content)))
                            except Exception:
                                pass

                        if self.is_thesis_relevant(text):
                            is_match = True

                        for tc in s.get("tool_calls", []):
                            args = str(tc.get("arguments", ""))
                            if self.is_thesis_relevant(args):
                                is_match = True

                        if is_user and not first_prompt:
                            first_prompt = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()[:100]

                        text_l = text.lower()
                        if "david.py" in text_l: topics.add("david.py")
                        if "conformer" in text_l: topics.add("conformer")
                        if "pareto" in text_l: topics.add("pareto")
                        if "rocs" in text_l: topics.add("rocs")
                        if "dichtator" in text_l: topics.add("dichtator")
                        if "metacentrum" in text_l: topics.add("metacentrum")
                        if "sprint" in text_l: topics.add("sprint")
            except Exception:
                continue

            if is_match and events:
                events.sort(key=lambda x: x[0])
                st_sec, au_sec, st_ints, au_ints = self.compute_intervals(events)

                if user_count == 0:
                    cat = "subagent"
                elif au_sec > 7200.0 and user_count <= 2:
                    cat = "autonomous_loop"
                elif user_count >= 3:
                    cat = "interactive_steered"
                else:
                    cat = "mixed"

                results.append({
                    "cid": cid,
                    "source": f"Antigravity ({source_label})",
                    "start": events[0][0].isoformat(),
                    "end": events[-1][0].isoformat(),
                    "steps": step_count,
                    "user_steps": user_count,
                    "steered_sec": st_sec,
                    "steered_hours": round(st_sec / 3600.0, 2),
                    "autonomous_sec": au_sec,
                    "autonomous_hours": round(au_sec / 3600.0, 2),
                    "effective_steered_hours": round((st_sec * LLM_BUFFER_MULTIPLIER) / 3600.0, 2),
                    "total_hours": round((st_sec + au_sec) / 3600.0, 2),
                    "category": cat,
                    "prompt": first_prompt.replace("\n", " "),
                    "topics": sorted(list(topics)),
                    "prompts_by_day": dict(prompts_by_day),
                    "_steered_intervals": [(s.isoformat(), e.isoformat()) for s, e in st_ints],
                    "_autonomous_intervals": [(s.isoformat(), e.isoformat()) for s, e in au_ints],
                })
        return results

    def audit_t3code_sqlite(self, db_path: Path) -> List[Dict[str, Any]]:
        """Audits t3code harness SQLite database for DrugEx threads."""
        if not db_path.exists():
            return []

        results: List[Dict[str, Any]] = []
        tmp_db = tempfile.mktemp(".sqlite")
        try:
            shutil.copy2(db_path, tmp_db)
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()

            projects = {}
            for row in cur.execute("SELECT project_id, title, workspace_root FROM projection_projects").fetchall():
                projects[row[0]] = {"title": row[1], "root": row[2]}

            threads = cur.execute("SELECT thread_id, project_id, title, created_at, updated_at FROM orchestration_v2_projection_threads").fetchall()

            for tid, pid, title, c_at, u_at in threads:
                proj = projects.get(pid, {})
                proj_root = proj.get("root", "")
                title_l = title.lower()

                is_match = self.is_thesis_relevant(title_l, proj_root)
                if not is_match and pid == "72b73597-c0ec-40f6-b1dd-9bc3c66633b0":
                    is_match = True

                msgs = cur.execute("SELECT role, created_at, payload_json FROM orchestration_v2_projection_messages WHERE thread_id=? ORDER BY created_at", (tid,)).fetchall()
                if not msgs:
                    continue

                events: List[Tuple[datetime, bool, bool, int]] = []
                user_count = 0
                first_prompt = ""
                topics = set()
                prompts_by_day: Dict[str, List[str]] = defaultdict(list)

                for role, msg_c_at, payload_str in msgs:
                    is_user = (role == "user")
                    if is_user:
                        user_count += 1
                    try:
                        dt = datetime.fromisoformat(msg_c_at.replace("Z", "+00:00"))
                    except Exception:
                        continue

                    has_art = False
                    text_len = 0
                    try:
                        payload = json.loads(payload_str)
                        text = str(payload.get("text", ""))
                        text_len = len(text)
                        has_art = (".md" in text or "REPORT" in text or "t3-context://" in text)
                        if self.is_thesis_relevant(text, proj_root):
                            is_match = True
                        if is_user and text.strip():
                            clean_p = text.strip()[:140]
                            d_key = msg_c_at[:10]
                            if clean_p not in prompts_by_day[d_key] and len(prompts_by_day[d_key]) < 8:
                                prompts_by_day[d_key].append(clean_p)
                        if is_user and not first_prompt:
                            first_prompt = text.strip()[:100]
                        text_l = text.lower()
                        if "david" in text_l: topics.add("david.py")
                        if "conformer" in text_l: topics.add("conformer")
                        if "pareto" in text_l: topics.add("pareto")
                        if "metacentrum" in text_l: topics.add("metacentrum")
                        if "sprint" in text_l: topics.add("sprint")
                        if "cisco" in text_l or "vpn" in text_l: topics.add("dev_environment")
                    except Exception:
                        pass

                    events.append((dt, is_user, has_art, text_len))

                if is_match and events:
                    events.sort(key=lambda x: x[0])
                    st_sec, au_sec, st_ints, au_ints = self.compute_intervals(events)

                    if user_count == 0:
                        cat = "subagent"
                    elif au_sec > 7200.0 and user_count <= 2:
                        cat = "autonomous_loop"
                    elif user_count >= 3:
                        cat = "interactive_steered"
                    else:
                        cat = "mixed"

                    if not first_prompt:
                        first_prompt = title

                    results.append({
                        "cid": f"t3-{tid[:12]}",
                        "source": "t3code (WSL)",
                        "start": events[0][0].isoformat(),
                        "end": events[-1][0].isoformat(),
                        "steps": len(msgs),
                        "user_steps": user_count,
                        "steered_sec": st_sec,
                        "steered_hours": round(st_sec / 3600.0, 2),
                        "autonomous_sec": au_sec,
                        "autonomous_hours": round(au_sec / 3600.0, 2),
                        "effective_steered_hours": round((st_sec * LLM_BUFFER_MULTIPLIER) / 3600.0, 2),
                        "total_hours": round((st_sec + au_sec) / 3600.0, 2),
                        "category": cat,
                        "prompt": first_prompt.replace("\n", " "),
                        "topics": sorted(list(topics)),
                        "prompts_by_day": dict(prompts_by_day),
                        "_steered_intervals": [(s.isoformat(), e.isoformat()) for s, e in st_ints],
                        "_autonomous_intervals": [(s.isoformat(), e.isoformat()) for s, e in au_ints],
                    })

            conn.close()
        finally:
            if os.path.exists(tmp_db):
                try: os.unlink(tmp_db)
                except Exception: pass

        return results

    def audit_all(self) -> Dict[str, Any]:
        """Audits all Antigravity and t3code sources and aggregates merged metrics."""
        all_convs: Dict[str, Dict[str, Any]] = {}

        # 1. Antigravity Windows
        for c in self.audit_antigravity_dir(WIN_BRAIN_DIR, "Win"):
            all_convs[c["cid"]] = c

        # 2. Antigravity WSL
        wsl_brain = WSL_BRAIN_DIR if WSL_BRAIN_DIR.exists() else (WSL_BRAIN_FALLBACK if WSL_BRAIN_FALLBACK.exists() else LINUX_BRAIN_DIR)
        for c in self.audit_antigravity_dir(wsl_brain, "WSL"):
            cid = c["cid"]
            if cid not in all_convs or c["steps"] > all_convs[cid]["steps"]:
                all_convs[cid] = c

        # 3. t3code WSL
        t3_db = WSL_T3_DB if WSL_T3_DB.exists() else (WSL_T3_FALLBACK if WSL_T3_FALLBACK.exists() else LINUX_T3_DB)
        for c in self.audit_t3code_sqlite(t3_db):
            all_convs[c["cid"]] = c

        matched_convs = sorted(all_convs.values(), key=lambda x: x["start"])

        # Merge non-overlapping active intervals across all workspaces
        all_steered_ints: List[Tuple[datetime, datetime]] = []
        all_auto_ints: List[Tuple[datetime, datetime]] = []

        for c in matched_convs:
            for s, e in c.get("_steered_intervals", []):
                s_dt = datetime.fromisoformat(s) if isinstance(s, str) else s
                e_dt = datetime.fromisoformat(e) if isinstance(e, str) else e
                all_steered_ints.append((s_dt, e_dt))
            for s, e in c.get("_autonomous_intervals", []):
                s_dt = datetime.fromisoformat(s) if isinstance(s, str) else s
                e_dt = datetime.fromisoformat(e) if isinstance(e, str) else e
                all_auto_ints.append((s_dt, e_dt))

        def merge_intervals(ints: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
            if not ints:
                return []
            ints.sort(key=lambda x: x[0])
            merged: List[List[datetime]] = []
            for s, e in ints:
                if not merged:
                    merged.append([s, e])
                else:
                    if s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
            return [(s, e) for s, e in merged]

        merged_steered = merge_intervals(all_steered_ints)
        merged_auto = merge_intervals(all_auto_ints)

        net_steered_sec = sum((e - s).total_seconds() for s, e in merged_steered)
        net_auto_sec = sum((e - s).total_seconds() for s, e in merged_auto)
        effective_steered_sec = net_steered_sec * LLM_BUFFER_MULTIPLIER

        return {
            "conversations_count": len(matched_convs),
            "total_steps": sum(c["steps"] for c in matched_convs),
            "net_steered_sec": net_steered_sec,
            "net_steered_hours": round(net_steered_sec / 3600.0, 2),
            "buffer_multiplier": LLM_BUFFER_MULTIPLIER,
            "effective_steered_sec": effective_steered_sec,
            "effective_steered_hours": round(effective_steered_sec / 3600.0, 2),
            "net_autonomous_sec": net_auto_sec,
            "net_autonomous_hours": round(net_auto_sec / 3600.0, 2),
            "total_ai_hours": round((net_steered_sec + net_auto_sec) / 3600.0, 2),
            "conversations": matched_convs,
            "_merged_steered": [(s.isoformat(), e.isoformat()) for s, e in merged_steered],
            "_merged_autonomous": [(s.isoformat(), e.isoformat()) for s, e in merged_auto],
        }


# ==============================================================================
# 4. PILLAR 2: DICHTATOR / VPN TUNNEL TRACKER & RECONCILER
# ==============================================================================

class DichtatorTracker:
    """Extracts, reconciles, and categorizes connection history to Dichtator and VSCHT VPN."""

    # Key dates where user actively wrote #NOTES and performed code reviews in VS Code
    KNOWN_CODE_REVIEW_DATES = {"2026-10-21", "2026-10-28", "2026-10-29", "2026-10-31"}

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
                    if dur_sec >= 10.0:
                        session_id = f"dich-{current_start.strftime('%Y%m%d%H%M')}"
                        sessions.append({
                            "id": session_id,
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
            session_id = f"dich-{current_start.strftime('%Y%m%d%H%M')}"
            sessions.append({
                "id": session_id,
                "start": current_start.isoformat(),
                "end": now.isoformat(),
                "duration_sec": dur_sec,
                "duration_hours": round(dur_sec / 3600.0, 2),
                "network": current_net,
                "type": "Active Session",
            })

        return sessions

    @classmethod
    def reconcile_with_ai_activity(
        cls,
        sessions: List[Dict[str, Any]],
        steered_intervals: List[Tuple[datetime, datetime]],
        autonomous_intervals: List[Tuple[datetime, datetime]],
        user_overrides: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Reconciles Dichtator VPN sessions with AI intervals and code review activity."""
        reconciled = []
        for s in sessions:
            sid = s.get("id", f"dich-{s['start'][:16]}")
            s_start = datetime.fromisoformat(s["start"])
            s_end = datetime.fromisoformat(s["end"])
            dur_sec = s["duration_sec"]
            date_str = s_start.strftime("%Y-%m-%d")

            # Compute overlap with steered intervals
            st_overlap = 0.0
            for ist_s, ist_e in steered_intervals:
                overlap_s = max(s_start, ist_s)
                overlap_e = min(s_end, ist_e)
                if overlap_e > overlap_s:
                    st_overlap += (overlap_e - overlap_s).total_seconds()

            # Compute overlap with autonomous intervals
            au_overlap = 0.0
            for ia_s, ia_e in autonomous_intervals:
                overlap_s = max(s_start, ia_s)
                overlap_e = min(s_end, ia_e)
                if overlap_e > overlap_s:
                    au_overlap += (overlap_e - overlap_s).total_seconds()

            # Check known code review dates (#NOTES and VS Code offline review)
            is_known_review_date = date_str in cls.KNOWN_CODE_REVIEW_DATES
            if is_known_review_date and st_overlap < 3600.0 and dur_sec >= 7200.0:
                # Credit up to 3.0h of active code review / #NOTES writing
                st_overlap = min(dur_sec, max(st_overlap, 10800.0))

            idle_sec = max(0.0, dur_sec - (st_overlap + au_overlap))

            # Auto-categorization
            if st_overlap >= 1800.0 or (st_overlap > 0.4 * dur_sec and st_overlap >= 300.0):
                cat = "steered"
            elif au_overlap >= 1800.0 or au_overlap > 0.4 * dur_sec:
                cat = "autonomous"
            else:
                cat = "idle"

            needs_review = (dur_sec >= 10800.0) or (idle_sec >= 7200.0 and (st_overlap + au_overlap) > 0)

            # Apply user override if present
            override = user_overrides.get(sid)
            if override:
                cat = override.get("category", cat)
                needs_review = False

            rec_session = dict(s)
            rec_session["id"] = sid
            rec_session["category"] = cat
            rec_session["steered_sec"] = st_overlap
            rec_session["steered_hours"] = round(st_overlap / 3600.0, 2)
            rec_session["autonomous_sec"] = au_overlap
            rec_session["autonomous_hours"] = round(au_overlap / 3600.0, 2)
            rec_session["idle_sec"] = idle_sec
            rec_session["idle_hours"] = round(idle_sec / 3600.0, 2)
            rec_session["needs_review"] = needs_review
            if override:
                rec_session["user_override"] = override

            reconciled.append(rec_session)
        return reconciled


# ==============================================================================
# 5. DAY-BY-DAY TIMELINE & PROOF-OF-WORK LEDGER BUILDER
# ==============================================================================

class DayByDayLedger:
    """Builds a chronological day-by-day activity ledger with verifiable proof of work."""

    @staticmethod
    def extract_drugex_commits() -> Dict[str, List[Dict[str, str]]]:
        """Queries git history from /home/kolar/DrugEx."""
        commits_by_day = defaultdict(list)
        try:
            res = subprocess.run(["wsl", "bash", "-c", "cd /home/kolar/DrugEx && git log --pretty=format:'%ad|%h|%s' --date=iso"], capture_output=True, text=True)
            for line in res.stdout.strip().splitlines():
                if not line.strip() or "|" not in line:
                    continue
                parts = line.split("|", 2)
                d = parts[0].strip()[:10]
                commits_by_day[d].append({"hash": parts[1].strip(), "msg": parts[2].strip()})
        except Exception:
            pass
        return commits_by_day

    @staticmethod
    def compile_daily_entries(
        commits_map: Dict[str, List[Dict[str, str]]],
        notewise_data: Dict[str, Any],
        dichtator_data: Dict[str, Any],
        ai_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Aggregates all metrics, tasks, and empirical proofs into daily chronological buckets."""
        daily_human_intervals: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
        days = defaultdict(lambda: {
            "date": "",
            "human_effort_hours": 0.0,
            "steered_hours": 0.0,
            "autonomous_hours": 0.0,
            "notewise_eff_hours": 0.0,
            "notewise_strokes": 0,
            "dichtator_dev_hours": 0.0,
            "dichtator_total_hours": 0.0,
            "commits": [],
            "artifacts": [],
            "prompts": [],
            "subtasks": set(),
            "chapter_mapping": set(),
        })

        # 1. Commits
        for d, c_list in commits_map.items():
            days[d]["date"] = d
            days[d]["commits"] = c_list
            for c in c_list:
                msg = c["msg"].lower()
                if "conformer" in msg or "pareto" in msg:
                    days[d]["subtasks"].add("Conformer Pareto Timing & Physics Optimization")
                    days[d]["chapter_mapping"].add("Chapter 2: Conformer Ensembles & 3D Physics")
                if "rocs" in msg or "harrington" in msg or "reward" in msg:
                    days[d]["subtasks"].add("ROCS & Multi-Objective Reward Calibration")
                    days[d]["chapter_mapping"].add("Chapter 3: Multi-Objective RL & Scoring")
                if "metacentrum" in msg or "pbs" in msg or "grid" in msg:
                    days[d]["subtasks"].add("HPC MetaCentrum PBS Grid & Cluster Execution")
                    days[d]["chapter_mapping"].add("Chapter 4: HPC Training Grid & Scalability")
                if "onboarding" in msg or "agents" in msg or "briefing" in msg:
                    days[d]["subtasks"].add("Agent Architecture, Context Briefings & Documentation")
                    days[d]["chapter_mapping"].add("Methodology & Autonomous Agent Engineering")

        # 2. Notewise Tablet Reading
        for f_data in notewise_data.values():
            for s in f_data.get("sessions", []):
                d = s["start"][:10]
                days[d]["date"] = d
                dur_sec = s.get("duration_sec", 0.0)
                eff_h = (dur_sec * NOTEWISE_BUFFER_MULTIPLIER) / 3600.0
                days[d]["notewise_eff_hours"] += eff_h
                days[d]["notewise_strokes"] += s.get("elements_count", 0)
                days[d]["subtasks"].add("Tablet Study: DrugEx Textbook & SequenceTransformer Reading")
                days[d]["chapter_mapping"].add("Chapter 1: Theoretical Background & RNN Architectures")
                try:
                    s_dt = to_utc(s["start"])
                    e_buf = s_dt + timedelta(seconds=dur_sec * NOTEWISE_BUFFER_MULTIPLIER)
                    daily_human_intervals[d].append((s_dt, e_buf))
                except Exception:
                    pass

        # 3. Dichtator Sessions
        for s in dichtator_data.get("sessions", []):
            d = s["start"][:10]
            days[d]["date"] = d
            days[d]["dichtator_dev_hours"] += s.get("steered_hours", 0.0)
            days[d]["dichtator_total_hours"] += s.get("duration_hours", 0.0)
            cat = s.get("category")
            if cat == "steered" or s.get("user_override"):
                st_sec = s.get("steered_sec", 0.0)
                if st_sec > 0:
                    try:
                        s_dt = to_utc(s["start"])
                        e_dt = to_utc(s["end"])
                        dur_capped = min(st_sec, (e_dt - s_dt).total_seconds())
                        daily_human_intervals[d].append((s_dt, s_dt + timedelta(seconds=dur_capped)))
                    except Exception:
                        pass

        # 4. Antigravity & t3code Workspaces
        for c in ai_data.get("conversations", []):
            # Steered intervals attributed to their exact date
            for s_str, e_str in c.get("_steered_intervals", []):
                d = s_str[:10]
                days[d]["date"] = d
                try:
                    s_dt = to_utc(s_str)
                    e_dt = to_utc(e_str)
                    dur_sec = (e_dt - s_dt).total_seconds()
                    days[d]["steered_hours"] += (dur_sec * LLM_BUFFER_MULTIPLIER) / 3600.0
                    daily_human_intervals[d].append((s_dt, s_dt + timedelta(seconds=dur_sec * LLM_BUFFER_MULTIPLIER)))
                except Exception:
                    pass

            # Autonomous intervals attributed to their exact date
            for s_str, e_str in c.get("_autonomous_intervals", []):
                d = s_str[:10]
                days[d]["date"] = d
                try:
                    s_dt = to_utc(s_str)
                    e_dt = to_utc(e_str)
                    dur_sec = (e_dt - s_dt).total_seconds()
                    days[d]["autonomous_hours"] += dur_sec / 3600.0
                except Exception:
                    pass

            # Dated prompts attributed to their exact day
            p_by_day = c.get("prompts_by_day", {})
            if p_by_day:
                for d_p, p_list in p_by_day.items():
                    days[d_p]["date"] = d_p
                    for p in p_list:
                        if p not in days[d_p]["prompts"]:
                            days[d_p]["prompts"].append(p)
            elif c.get("prompt"):
                d_p = c["start"][:10]
                if c["prompt"] not in days[d_p]["prompts"]:
                    days[d_p]["prompts"].append(c["prompt"])

            # Active days for topic / subtask attribution
            active_days = set()
            for s_str, _ in c.get("_steered_intervals", []): active_days.add(s_str[:10])
            for s_str, _ in c.get("_autonomous_intervals", []): active_days.add(s_str[:10])
            if not active_days: active_days.add(c["start"][:10])

            for d_act in active_days:
                for top in c.get("topics", []):
                    if top == "david.py": days[d_act]["subtasks"].add("Core `_david` Pipeline Engineering & Clean Code")
                    elif top == "conformer": days[d_act]["subtasks"].add("3D Conformer Generation & Hawkins Tiers")
                    elif top == "pareto": days[d_act]["subtasks"].add("Pareto-Optimal Frontier Analysis")
                    elif top == "dev_environment": days[d_act]["subtasks"].add("Developer Environment & Cisco VPN Orchestration")
                    elif top == "rocs": days[d_act]["subtasks"].add("ROCS & Multi-Objective Reward Calibration")
                    elif top == "metacentrum": days[d_act]["subtasks"].add("HPC MetaCentrum PBS Grid & Cluster Execution")
                    elif top == "sprint": days[d_act]["subtasks"].add("HPC Cluster Sprint Orchestration")

        def merge_intervals(ints: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
            if not ints: return []
            ints.sort(key=lambda x: x[0])
            merged: List[List[datetime]] = [list(ints[0])]
            for s, e in ints[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            return [(s, e) for s, e in merged]

        # Compile totals per day
        compiled: List[Dict[str, Any]] = []
        for d in sorted(days.keys(), reverse=True):
            item = days[d]
            merged_day = merge_intervals(daily_human_intervals[d])
            h_eff = round(sum((e - s).total_seconds() for s, e in merged_day) / 3600.0, 2)
            item["human_effort_hours"] = h_eff
            item["steered_hours"] = round(item["steered_hours"], 2)
            item["autonomous_hours"] = round(item["autonomous_hours"], 2)
            item["notewise_eff_hours"] = round(item["notewise_eff_hours"], 2)
            item["dichtator_dev_hours"] = round(item["dichtator_dev_hours"], 2)
            item["dichtator_total_hours"] = round(item["dichtator_total_hours"], 2)
            item["subtasks"] = sorted(list(item["subtasks"]))
            item["chapter_mapping"] = sorted(list(item["chapter_mapping"]))
            item["prompts"] = item["prompts"][:4]  # Keep top 4 representative prompt excerpts
            if h_eff > 0 or len(item["commits"]) > 0 or item["notewise_eff_hours"] > 0 or item["autonomous_hours"] > 0:
                compiled.append(item)

        return compiled

        return compiled


# ==============================================================================
# 6. CONSOLIDATED TRACKER CONTROLLER
# ==============================================================================

class ThesisTracker:
    """Main controller aggregating all tracking data and generating reports."""

    def __init__(self, db_path: Path = DB_FILE) -> None:
        self.db_path = db_path
        self.notewise_ingester = NotewiseIngester()
        self.dichtator_tracker = DichtatorTracker()
        self.ai_auditor = AIHarnessAuditor()
        self.ledger_builder = DayByDayLedger()
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
            "user_overrides": {},
            "daily_ledger": [],
            "totals": {},
        }

    def save_db(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)

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
            print("[Notewise] Note: Keeping existing tablet records (no new export file in mobile path).")

    def sync_all_sources(self) -> None:
        """Performs unified audit across Notewise, AI harnesses, and Dichtator VPN."""
        print("[AI Workspaces] Auditing Antigravity (Win & WSL) and t3code (WSL)...")
        ai_audit = self.ai_auditor.audit_all()
        self.data["antigravity"] = ai_audit
        print(f"[AI Workspaces] Audited {ai_audit['conversations_count']} workspaces/threads ({ai_audit['total_steps']} steps): "
              f"{ai_audit['net_steered_hours']}h steered -> {ai_audit['effective_steered_hours']}h effective (+10% buffer), "
              f"{ai_audit['net_autonomous_hours']}h autonomous loops.")

        print("[Dichtator] Syncing connection sessions and reconciling with AI intervals...")
        raw_sessions = self.dichtator_tracker.query_network_profile_events()
        st_ints = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in ai_audit.get("_merged_steered", [])]
        au_ints = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in ai_audit.get("_merged_autonomous", [])]
        overrides = self.data.get("user_overrides", {})

        reconciled_sessions = self.dichtator_tracker.reconcile_with_ai_activity(raw_sessions, st_ints, au_ints, overrides)
        total_dich_sec = sum(s["duration_sec"] for s in reconciled_sessions)
        steered_dich_sec = sum(s["steered_sec"] for s in reconciled_sessions)
        auto_dich_sec = sum(s["autonomous_sec"] for s in reconciled_sessions)
        idle_dich_sec = sum(s["idle_sec"] for s in reconciled_sessions)

        self.data["dichtator"] = {
            "sessions_count": len(reconciled_sessions),
            "total_duration_sec": total_dich_sec,
            "total_duration_hours": round(total_dich_sec / 3600.0, 2),
            "steered_duration_hours": round(steered_dich_sec / 3600.0, 2),
            "autonomous_duration_hours": round(auto_dich_sec / 3600.0, 2),
            "idle_duration_hours": round(idle_dich_sec / 3600.0, 2),
            "sessions": reconciled_sessions,
        }
        print(f"[Dichtator] Synced {len(reconciled_sessions)} sessions: "
              f"{round(total_dich_sec/3600.0, 2)}h total ({round(steered_dich_sec/3600.0, 2)}h steered/dev, "
              f"{round(auto_dich_sec/3600.0, 2)}h autonomous compute, {round(idle_dich_sec/3600.0, 2)}h idle).")

        print("[Ledger] Compiling Day-by-Day Timeline & Proof-of-Work...")
        commits_map = self.ledger_builder.extract_drugex_commits()
        daily_ledger = self.ledger_builder.compile_daily_entries(
            commits_map,
            self.data.get("notewise", {}),
            self.data.get("dichtator", {}),
            self.data.get("antigravity", {})
        )
        self.data["daily_ledger"] = daily_ledger
        print(f"[Ledger] Compiled {len(daily_ledger)} active research days with verifiable commit & artifact proof.")

        self._recalculate_totals()
        self.save_db()

    def _recalculate_totals(self) -> None:
        """Computes grand totals disaggregated into Human Thesis Effort vs Autonomous AI Compute."""
        nw_effective_h = sum(
            item.get("effective_duration_hours", 0.0)
            for item in self.data.get("notewise", {}).values()
        )
        dich = self.data.get("dichtator", {})
        ag = self.data.get("antigravity", {})

        dich_steered_h = dich.get("steered_duration_hours", 0.0)
        dich_auto_h = dich.get("autonomous_duration_hours", 0.0)
        dich_idle_h = dich.get("idle_duration_hours", 0.0)
        dich_total_h = dich.get("total_duration_hours", 0.0)

        ai_steered_effective_h = ag.get("effective_steered_hours", 0.0)
        ai_auto_h = ag.get("net_autonomous_hours", 0.0)

        ledger = self.data.get("daily_ledger", [])
        merged_human_effort_h = round(sum(d.get("human_effort_hours", 0.0) for d in ledger), 2)
        if merged_human_effort_h == 0.0:
            merged_human_effort_h = round(nw_effective_h + ai_steered_effective_h + dich_steered_h, 2)

        disagg_sum_h = round(nw_effective_h + ai_steered_effective_h + dich_steered_h, 2)
        autonomous_compute_h = round(ai_auto_h + dich_auto_h, 2)
        grand_total_logged_h = round(merged_human_effort_h + autonomous_compute_h + dich_idle_h, 2)

        self.data["totals"] = {
            "human_thesis_effort_hours": merged_human_effort_h,
            "disaggregated_effort_sum_hours": disagg_sum_h,
            "autonomous_compute_hours": autonomous_compute_h,
            "idle_tunnel_hours": dich_idle_h,
            "grand_total_hours": merged_human_effort_h,
            "grand_total_logged_hours": grand_total_logged_h,
            "notewise_reading_effective_hours": nw_effective_h,
            "dichtator_tunnel_hours": dich_total_h,
            "dichtator_steered_hours": dich_steered_h,
            "dichtator_autonomous_hours": dich_auto_h,
            "antigravity_llm_effective_hours": ai_steered_effective_h,
            "ai_autonomous_hours": ai_auto_h,
            "total_steps": ag.get("total_steps", 0),
        }

    def generate_markdown_report(self) -> str:
        """Generates comprehensive thesis_report.md."""
        self._recalculate_totals()
        totals = self.data.get("totals", {})
        nw = self.data.get("notewise", {})
        dich = self.data.get("dichtator", {})
        ag = self.data.get("antigravity", {})
        ledger = self.data.get("daily_ledger", [])

        md = f"""# DrugEx Thesis Work & Time Tracking Report

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

| Tracking Pillar | Breakdown & Metric | Multiplier | Effective Thesis Effort |
| :--- | :--- | :--- | :--- |
| **Tablet Reading ([Notewise](file:///C:/Users/kolar/Downloads/Mobile%20Devices/DrugEx.notewise))** | {sum(i.get('raw_duration_hours', 0.0) for i in nw.values())}h active writing / annotation | **+20%** buffer | **{totals.get('notewise_reading_effective_hours', 0.0)} hours** |
| **Interactive AI Pair-Programming** | {ag.get('net_steered_hours', 0.0)}h active steering, prompt drafting & reading | **+10%** review buffer | **{totals.get('antigravity_llm_effective_hours', 0.0)} hours** |
| **Remote Dev & VS Code SSH (Dichtator)** | {dich.get('steered_duration_hours', 0.0)}h active dev, `#NOTES` & code review | 1.0× direct | **{dich.get('steered_duration_hours', 0.0)} hours** |
| *Raw Disaggregated Sum* | *Sum across all 3 pillars before interval de-duplication* | | *{totals.get('disaggregated_effort_sum_hours', totals.get('human_thesis_effort_hours', 0.0))} hours* |
| **CORE HUMAN THESIS EFFORT** | **De-duplicated non-overlapping human timeline (no dual-device double-counting)** | | **{totals.get('human_thesis_effort_hours', 0.0)} hours** |
| **Autonomous AI Research Loops** | {ag.get('net_autonomous_hours', 0.0)}h unattended batch runs, pareto benchmarks | 1.0× compute | *{totals.get('autonomous_compute_hours', 0.0)} hours (Compute)* |
| **Idle Cisco VPN Tunnel Uptime** | {dich.get('idle_duration_hours', 0.0)}h background connection without active steps | 0.0× filtered | *{totals.get('idle_tunnel_hours', 0.0)} hours (Idle)* |

---

## 1. Day-by-Day Timeline & Verifiable Proof-of-Work Ledger

This ledger chronologically documents daily thesis effort backed by concrete empirical proofs (Git commit hashes, generated report artifacts, prompt excerpts, and Notewise strokes).

"""
        for day_entry in ledger[:25]:
            d_str = day_entry["date"]
            h_tot = day_entry["human_effort_hours"]
            subtasks_str = ", ".join(day_entry.get("subtasks", [])) if day_entry.get("subtasks") else "Thesis Research & Engineering"
            chaps_str = ", ".join(day_entry.get("chapter_mapping", [])) if day_entry.get("chapter_mapping") else "General Methodology"

            md += f"### Date: `{d_str}` ({h_tot}h Human Effort | {day_entry.get('autonomous_hours', 0.0)}h AI Compute)\n"
            md += f"- **Primary Focus**: {subtasks_str}\n"
            md += f"- **Thesis Chapter Mapping**: *{chaps_str}*\n"

            # Proof of Work: Git Commits
            if day_entry.get("commits"):
                md += "- **Verifiable Git Commits**:\n"
                for c in day_entry["commits"][:4]:
                    md += f"  - [`{c['hash']}`](https://github.com/fulopjoz/DrugEx/commit/{c['hash']}): {c['msg']}\n"

            # Proof of Work: Prompts / Steering Excerpts
            if day_entry.get("prompts"):
                md += "- **Key Steering Directives**:\n"
                for pr in day_entry["prompts"][:2]:
                    md += f"  - *\"{pr[:100]}...\"*\n"

            # Proof of Work: Tablet Reading
            if day_entry.get("notewise_strokes", 0) > 0:
                md += f"- **Tablet Annotations**: {day_entry['notewise_strokes']} strokes across {day_entry.get('notewise_eff_hours', 0.0)}h study time\n"

            md += "\n"

        md += f"""---

## 2. Tablet Reading Breakdown (Notewise)

*All reading sessions carry a +20% buffer to credit non-writing contemplation, literature lookup, and post-reading synthesis.*

"""
        for fname, info in nw.items():
            md += f"### File: `{fname}`\n"
            md += f"- **Pages**: {info.get('total_pages', 0)} | **Annotations/Strokes**: {info.get('total_elements', 0)}\n"
            md += f"- **Raw Active Writing**: {info.get('raw_duration_hours', 0.0)}h | **Effective Study Time**: **{info.get('effective_duration_hours', 0.0)}h**\n"
            md += f"- **Attached References**: {', '.join(info.get('attachments', []))}\n"
            md += "\n"

        md += f"""---

## 3. Remote Compute & Dichtator Sessions

- **Total Connection Time Logged**: **{dich.get('total_duration_hours', 0.0)} hours** across **{dich.get('sessions_count', 0)} sessions**.
- **Interactive Dev, `#NOTES` & Code Review**: **{dich.get('steered_duration_hours', 0.0)} hours**
- **Autonomous Remote Compute**: **{dich.get('autonomous_duration_hours', 0.0)} hours**
- **Idle Tunnel Uptime (filtered)**: **{dich.get('idle_duration_hours', 0.0)} hours**

---

## 4. AI Engineering & LLM Workspaces (Antigravity & t3code)

- **Total Tracked Workspaces/Threads**: **{ag.get('conversations_count', 0)}**
- **Total Execution Steps**: **{ag.get('total_steps', 0)}**
- **Active Interactive Steering & Prompt Crafting**: **{ag.get('net_steered_hours', 0.0)} hours** ({ag.get('effective_steered_hours', 0.0)}h with +10% review buffer)
- **Autonomous Loops & Batch Execution**: **{ag.get('net_autonomous_hours', 0.0)} hours**

### Key Workspaces Tracked:
"""
        for c in ag.get("conversations", [])[::-1][:20]:
            md += f"- **`{c['cid'][:14]}`** ({c['source']}, {format_duration(c['steered_sec'])} steered, {format_duration(c['autonomous_sec'])} auto, {c['steps']} steps): {c['prompt'][:80]}...\n"

        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md)
        return md

    def generate_html_dashboard(self) -> str:
        """Generates a modern, single-file, interactive HTML dashboard adhering to Modern Web Guidance."""
        self._recalculate_totals()
        totals = self.data.get("totals", {})
        nw = self.data.get("notewise", {})
        dich = self.data.get("dichtator", {})
        ag = self.data.get("antigravity", {})
        ledger = self.data.get("daily_ledger", [])

        nw_sessions = []
        for f_data in nw.values():
            for s in f_data.get("sessions", []):
                nw_sessions.append(s)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DrugEx Thesis Work & Time Tracker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090a0f;
      --card-bg: rgba(22, 27, 34, 0.75);
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
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 50% 0%, #171d2d, var(--bg) 65%);
      color: var(--text);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 1.5rem;
      flex-wrap: wrap;
      gap: 1rem;
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
      margin-top: 0.35rem;
    }}
    .badge-group {{
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-primary {{
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent);
    }}
    .badge-purple {{
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: var(--accent-purple);
    }}

    .grid-kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
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
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.5rem;
    }}
    .kpi-val {{
      font-size: 2.25rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .kpi-sub {{
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 0.4rem;
      line-height: 1.4;
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

    /* Controls & Filters */
    .controls-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 2rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .filter-pills {{
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 0.4rem 0.9rem;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      background: rgba(255, 255, 255, 0.03);
      color: var(--text-muted);
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .pill:hover {{
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
    }}
    .pill.active {{
      background: rgba(56, 189, 248, 0.15);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }}

    /* Modern Table Architecture with Sticky Headers & Container Queries */
    .table-wrapper {{
      overflow: auto;
      max-inline-size: 100%;
      max-block-size: 550px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      margin-bottom: 2.5rem;
      backdrop-filter: blur(12px);
      container-type: inline-size;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      text-align: left;
      font-size: 0.85rem;
    }}
    thead {{
      position: sticky;
      inset-block-start: 0;
      z-index: 2;
    }}
    th {{
      padding: 0.9rem 1.25rem;
      background: #111520;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--card-border);
    }}
    td {{
      padding: 0.9rem 1.25rem;
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

    .tag-steered {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
    }}
    .tag-auto {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent-amber);
    }}
    .tag-subagent {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(168, 85, 247, 0.15);
      color: var(--accent-purple);
    }}
    .tag-idle {{
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
    }}

    /* Day-by-Day Timeline Cards */
    .timeline-container {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      margin-bottom: 3rem;
    }}
    .timeline-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem 1.5rem;
      backdrop-filter: blur(12px);
      transition: border-color 0.2s ease;
    }}
    .timeline-card:hover {{
      border-color: rgba(56, 189, 248, 0.3);
    }}
    .timeline-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .timeline-date {{
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #fff;
    }}
    .timeline-metrics {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent);
    }}
    .timeline-details {{
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.5;
    }}
    .commit-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
      margin-right: 0.4rem;
      margin-bottom: 0.4rem;
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
        <p>Continuous automated audit across Tablet, Remote Compute, Antigravity & t3code</p>
      </div>
      <div class="badge-group">
        <span class="badge badge-primary">WSL & Win Synced</span>
        <span class="badge badge-purple">Vercel Ready</span>
      </div>
    </header>

    <div class="grid-kpis">
      <div class="card" style="border-top: 3px solid var(--accent);">
        <div class="kpi-label">Core Human Thesis Effort</div>
        <div class="kpi-val" style="color: #fff;">{totals.get('human_thesis_effort_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">De-duplicated human study, pair-programming & dev (~{(totals.get('human_thesis_effort_hours', 0.0)/40.0):.1f} full-time weeks; {totals.get('disaggregated_effort_sum_hours', totals.get('human_thesis_effort_hours', 0.0))}h component sum)</div>
        <div class="progress-bar-container">
          <div class="progress-fill" style="width: {(totals.get('notewise_reading_effective_hours', 0.0)/max(totals.get('human_thesis_effort_hours', 1.0),1.0))*100}%; background: var(--accent-purple);" title="Tablet Reading"></div>
          <div class="progress-fill" style="width: {(totals.get('antigravity_llm_effective_hours', 0.0)/max(totals.get('human_thesis_effort_hours', 1.0),1.0))*100}%; background: var(--accent);" title="Interactive AI Steered"></div>
          <div class="progress-fill" style="width: {(dich.get('steered_duration_hours', 0.0)/max(totals.get('human_thesis_effort_hours', 1.0),1.0))*100}%; background: var(--accent-emerald);" title="Dichtator Dev"></div>
        </div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent-amber);">
        <div class="kpi-label">Autonomous AI Compute</div>
        <div class="kpi-val" style="color: var(--accent-amber);">{totals.get('autonomous_compute_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">{ag.get('net_autonomous_hours', 0.0)}h autonomous loops & subagents (overnight/batch)</div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent-purple);">
        <div class="kpi-label">Tablet Study & Reading</div>
        <div class="kpi-val" style="color: var(--accent-purple);">{totals.get('notewise_reading_effective_hours', 0.0)} <span style="font-size: 1.2rem; font-weight: 500;">hrs</span></div>
        <div class="kpi-sub">{sum(i.get('raw_duration_hours', 0.0) for i in nw.values())}h raw active writing + 20% contemplation buffer</div>
      </div>

      <div class="card" style="border-top: 3px solid var(--accent-emerald);">
        <div class="kpi-label">Dichtator / VPN Tunnel</div>
        <div class="kpi-val" style="color: var(--accent-emerald);">{dich.get('steered_duration_hours', 0.0)}h <span style="font-size: 1.0rem; color: var(--text-muted);">dev / {dich.get('total_duration_hours', 0.0)}h tunnel</span></div>
        <div class="kpi-sub">{dich.get('sessions_count', 0)} sessions ({dich.get('idle_duration_hours', 0.0)}h filtered background uptime)</div>
      </div>
    </div>

    <!-- Day-by-Day Timeline Section -->
    <div class="controls-bar">
      <h2 style="font-size: 1.25rem; font-weight: 600;">Day-by-Day Timeline & Verifiable Proof-of-Work</h2>
      <span style="font-size: 0.8rem; color: var(--text-muted);">{len(ledger)} research days documented</span>
    </div>

    <div class="timeline-container">
"""
        for day in ledger[:20]:
            commits_html = ""
            for c in day.get("commits", [])[:3]:
                commits_html += f'<span class="commit-pill"><code>{c["hash"]}</code> {c["msg"][:60]}</span> '

            prompts_html = ""
            if day.get("prompts"):
                prompts_html = f'<div style="margin-top: 0.4rem; font-size: 0.8rem; color: #94a3b8;"><em>Directive: "{day["prompts"][0][:90]}..."</em></div>'

            strokes_info = f" &bull; {day['notewise_strokes']} tablet strokes" if day.get("notewise_strokes", 0) > 0 else ""
            subtasks_str = ", ".join(day.get("subtasks", [])) if day.get("subtasks") else "Core Thesis Engineering"
            chap_str = f"Mapping: {', '.join(day.get('chapter_mapping', []))}" if day.get("chapter_mapping") else ""

            html += f"""      <div class="timeline-card">
        <div class="timeline-header">
          <div class="timeline-date">{day['date']}</div>
          <div class="timeline-metrics">{day['human_effort_hours']} hrs human effort ({day.get('steered_hours', 0.0)}h AI dev &bull; {day.get('dichtator_dev_hours', 0.0)}h VS Code SSH{strokes_info})</div>
        </div>
        <div class="timeline-details">
          <div style="font-weight: 600; color: #f1f5f9; margin-bottom: 0.35rem;">{subtasks_str}</div>
          <div style="font-size: 0.78rem; color: var(--accent-purple); margin-bottom: 0.5rem;">{chap_str}</div>
          {commits_html}
          {prompts_html}
        </div>
      </div>\n"""

        html += f"""    </div>

    <!-- AI Workspaces Section -->
    <div class="controls-bar">
      <h2 style="font-size: 1.25rem; font-weight: 600;">AI Workspaces & Subagent Trajectories</h2>
      <div class="filter-pills" id="ai-filter-pills">
        <button class="pill active" data-filter="all">All ({len(ag.get('conversations', []))})</button>
        <button class="pill" data-filter="interactive_steered">Steered ({sum(1 for c in ag.get('conversations', []) if c.get('category')=='interactive_steered')})</button>
        <button class="pill" data-filter="autonomous_loop">Overnight Loops ({sum(1 for c in ag.get('conversations', []) if c.get('category')=='autonomous_loop')})</button>
        <button class="pill" data-filter="subagent">Subagents ({sum(1 for c in ag.get('conversations', []) if c.get('category')=='subagent')})</button>
      </div>
    </div>

    <div class="table-wrapper" role="region" tabindex="0" aria-label="AI Engineering Workspaces">
      <table id="ai-table">
        <thead>
          <tr>
            <th>Workspace / Thread</th>
            <th>Platform</th>
            <th>Category</th>
            <th>Start (Local)</th>
            <th>Steered & Crafting</th>
            <th>Autonomous</th>
            <th>Steps</th>
            <th>User Requests</th>
            <th>Initial Request / Objective</th>
          </tr>
        </thead>
        <tbody>
"""
        for c in ag.get("conversations", [])[::-1]:
            start_str = c["start"][:19].replace("T", " ")
            cat_tag = {
                "interactive_steered": '<span class="tag-steered">Steered</span>',
                "autonomous_loop": '<span class="tag-auto">Overnight Loop</span>',
                "subagent": '<span class="tag-subagent">Subagent</span>',
                "mixed": '<span class="tag-steered">Mixed</span>',
            }.get(c.get("category", ""), '<span class="tag-idle">Other</span>')

            html += f"""          <tr data-cat="{c.get('category', 'all')}">
            <td><code>{c['cid'][:14]}</code></td>
            <td><span style="font-size: 0.75rem; color: var(--text-muted);">{c.get('source', 'Antigravity')}</span></td>
            <td>{cat_tag}</td>
            <td>{start_str}</td>
            <td><strong>{format_duration(c['steered_sec'])}</strong></td>
            <td>{format_duration(c['autonomous_sec'])}</td>
            <td>{c['steps']}</td>
            <td>{c.get('user_steps', 0)}</td>
            <td style="max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{c['prompt']}">{c['prompt']}</td>
          </tr>\n"""

        html += f"""        </tbody>
      </table>
    </div>

    <!-- Dichtator Sessions Section -->
    <div class="controls-bar">
      <h2 style="font-size: 1.25rem; font-weight: 600;">Dichtator VPN Sessions & Reconciliation</h2>
      <div class="filter-pills" id="dich-filter-pills">
        <button class="pill active" data-dfilter="all">All ({len(dich.get('sessions', []))})</button>
        <button class="pill" data-dfilter="steered">Active Dev ({sum(1 for s in dich.get('sessions', []) if s.get('category')=='steered')})</button>
        <button class="pill" data-dfilter="autonomous">Autonomous Compute ({sum(1 for s in dich.get('sessions', []) if s.get('category')=='autonomous')})</button>
        <button class="pill" data-dfilter="idle">Idle Tunnel ({sum(1 for s in dich.get('sessions', []) if s.get('category')=='idle')})</button>
      </div>
    </div>

    <div class="table-wrapper" role="region" tabindex="0" aria-label="Dichtator Sessions">
      <table id="dich-table">
        <thead>
          <tr>
            <th>Session</th>
            <th>Category</th>
            <th>Start Time</th>
            <th>Total Uptime</th>
            <th>Active Dev</th>
            <th>Autonomous AI</th>
            <th>Idle Uptime</th>
            <th>Network</th>
          </tr>
        </thead>
        <tbody>
"""
        for s in dich.get("sessions", [])[::-1][:30]:
            start_str = s["start"][:19].replace("T", " ")
            cat = s.get("category", "idle")
            dcat_tag = {
                "steered": '<span class="tag-steered">Active Dev</span>',
                "autonomous": '<span class="tag-auto">Autonomous</span>',
                "idle": '<span class="tag-idle">Idle Tunnel</span>',
            }.get(cat, '<span class="tag-idle">Idle</span>')

            html += f"""          <tr data-dcat="{cat}">
            <td><code>{s.get('id', s['start'][:16])}</code></td>
            <td>{dcat_tag}</td>
            <td>{start_str}</td>
            <td><strong>{s['duration_hours']}h</strong></td>
            <td>{s.get('steered_hours', 0.0)}h</td>
            <td>{s.get('autonomous_hours', 0.0)}h</td>
            <td style="color: var(--text-muted);">{s.get('idle_hours', 0.0)}h</td>
            <td>{s.get('network', 'vscht.cz')}</td>
          </tr>\n"""

        html += f"""        </tbody>
      </table>
    </div>

    <!-- Tablet Reading Section -->
    <div class="controls-bar">
      <h2 style="font-size: 1.25rem; font-weight: 600;">Tablet Reading Sessions (Notewise Protobuf)</h2>
    </div>
    <div class="table-wrapper" role="region" tabindex="0" aria-label="Tablet Reading Sessions">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>Start Time</th>
            <th>Active Reading</th>
            <th>Effective Study (+20%)</th>
            <th>Annotations / Strokes</th>
            <th>Pages Touched</th>
          </tr>
        </thead>
        <tbody>
"""
        for i, s in enumerate(nw_sessions[::-1][:20]):
            start_str = s["start"][:19].replace("T", " ")
            dur_eff = round((s['duration_sec'] * NOTEWISE_BUFFER_MULTIPLIER) / 60.0, 1)
            html += f"""          <tr>
            <td>Session {len(nw_sessions) - i:02d}</td>
            <td>{start_str}</td>
            <td>{s['duration_min']} min</td>
            <td><strong>{dur_eff} min</strong></td>
            <td>{s['elements_count']}</td>
            <td>{s['pages_touched']} pages</td>
          </tr>\n"""

        html += f"""        </tbody>
      </table>
    </div>

    <footer>
      DrugEx Thesis Automated Tracking System &bull; Unified Windows & WSL Analytics
    </footer>
  </div>

  <script>
    // Client-side filtering for AI table
    document.querySelectorAll('#ai-filter-pills button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('#ai-filter-pills button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.getAttribute('data-filter');
        document.querySelectorAll('#ai-table tbody tr').forEach(tr => {{
          if (f === 'all' || tr.getAttribute('data-cat') === f) {{
            tr.style.display = '';
          }} else {{
            tr.style.display = 'none';
          }}
        }});
      }});
    }});

    // Client-side filtering for Dichtator table
    document.querySelectorAll('#dich-filter-pills button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('#dich-filter-pills button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.getAttribute('data-dfilter');
        document.querySelectorAll('#dich-table tbody tr').forEach(tr => {{
          if (f === 'all' || tr.getAttribute('data-dcat') === f) {{
            tr.style.display = '';
          }} else {{
            tr.style.display = 'none';
          }}
        }});
      }});
    }});
  </script>
</body>
</html>
"""
        with open(REPORT_HTML, "w", encoding="utf-8") as f:
            f.write(html)

        # Synchronize with drugex_edu repo if present
        edu_candidates = [
            Path(r"C:\Ubuntu\home\kolar\build_projects\drugex_edu"),
            Path(r"\\wsl.localhost\Ubuntu\home\kolar\build_projects\drugex_edu"),
            Path("/home/kolar/build_projects/drugex_edu"),
        ]
        for edu_repo in edu_candidates:
            if edu_repo.exists() and (edu_repo / ".git").exists():
                try:
                    head_file = edu_repo / ".git" / "HEAD"
                    if head_file.exists():
                        head_ref = head_file.read_text(encoding="utf-8").strip()
                        if "refs/heads/thesis-stats" in head_ref:
                            with open(edu_repo / "index.html", "w", encoding="utf-8") as f:
                                f.write(html)
                            edu_tracker = edu_repo / "tracker"
                            if edu_tracker.exists():
                                shutil.copy2(DB_FILE, edu_tracker / "thesis_tracking_db.json")
                                shutil.copy2(REPORT_MD, edu_tracker / "thesis_report.md")
                                shutil.copy2(Path(__file__), edu_tracker / "thesis_tracker.py")
                            break
                except Exception:
                    pass
        return html

    def review_interactive(self, review_all: bool = False) -> None:
        """Interactive CLI workflow for reviewing and overriding session categories."""
        dich_sessions = self.data.get("dichtator", {}).get("sessions", [])
        overrides = self.data.get("user_overrides", {})

        candidates = []
        for s in dich_sessions:
            if review_all or s.get("needs_review", False):
                candidates.append(s)

        if not candidates:
            print("\n[Review] No sessions currently need review! Everything is cleanly classified.")
            print("Tip: Use `python thesis_tracker.py --review-all` to review all sessions.")
            return

        print(f"\n[Review] Found {len(candidates)} session(s) pending review:")
        print("Legend: [S]teered (Active dev/review) | [A]utonomous (Overnight loop/compute) | [I]dle (Tunnel left on) | [Enter] to skip")

        for s in candidates:
            sid = s.get("id", s["start"][:16])
            print("\n" + "-" * 70)
            print(f"Session ID : {sid}")
            print(f"Time Range : {s['start'][:19].replace('T', ' ')} -> {s['end'][:19].replace('T', ' ')}")
            print(f"Total Time : {s['duration_hours']} hours | Current Category: {s.get('category')}")
            print(f"Overlap    : Steered={s.get('steered_hours', 0.0)}h, Autonomous={s.get('autonomous_hours', 0.0)}h, Idle={s.get('idle_hours', 0.0)}h")

            try:
                choice = input("Your classification [S/A/I or Enter to keep]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[Review] Exiting interactive review.")
                break

            if choice in ["s", "steered"]:
                overrides[sid] = {"category": "steered", "reviewed_at": datetime.now().isoformat()}
                print(f"-> Marked as STEERED.")
            elif choice in ["a", "autonomous"]:
                overrides[sid] = {"category": "autonomous", "reviewed_at": datetime.now().isoformat()}
                print(f"-> Marked as AUTONOMOUS.")
            elif choice in ["i", "idle"]:
                overrides[sid] = {"category": "idle", "reviewed_at": datetime.now().isoformat()}
                print(f"-> Marked as IDLE.")
            else:
                print("-> Skipped.")

        self.data["user_overrides"] = overrides
        self.save_db()
        print("\n[Review] Overrides saved! Re-syncing totals...")
        self.sync_all_sources()
        self.generate_markdown_report()
        self.generate_html_dashboard()

    def set_override(self, session_id: str, category: str) -> None:
        """Sets a manual category override for a specific session."""
        cat_lower = category.lower().strip()
        if cat_lower not in ["steered", "autonomous", "idle"]:
            print(f"[Error] Invalid category '{category}'. Must be one of: steered, autonomous, idle.")
            return
        if "user_overrides" not in self.data:
            self.data["user_overrides"] = {}
        self.data["user_overrides"][session_id] = {
            "category": cat_lower,
            "reviewed_at": datetime.now().isoformat(),
        }
        self.save_db()
        print(f"[Override] Successfully set {session_id} -> {cat_lower}. Re-syncing...")
        self.sync_all_sources()
        self.generate_markdown_report()
        self.generate_html_dashboard()


def git_commit_and_push(branch: str = "thesis-stats") -> None:
    """Commits and pushes updated database, report, and HTML dashboard to origin."""
    try:
        repo_dir = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tracker" else SCRIPT_DIR
        subprocess.run(["git", "add", "index.html", "tracker/"], cwd=str(repo_dir), check=False)
        msg = f"chore(stats): automated sync [{datetime.now().strftime('%Y-%m-%d %H:%M')}]"
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
# 7. CLI ENTRY POINT
# ==============================================================================

def main() -> None:
    """Parses command line arguments and triggers tracking or reporting routines."""
    parser = argparse.ArgumentParser(description="DrugEx Thesis Time Tracker")
    parser.add_argument("--sync-all", action="store_true", help="Run full unified audit across all pillars (Notewise, Dichtator, Antigravity Win/WSL, t3code)")
    parser.add_argument("--ingest-notewise", type=str, help="Ingest a specific .notewise export file")
    parser.add_argument("--auto-notewise", action="store_true", help="Auto-discover and ingest candidate .notewise files")
    parser.add_argument("--sync-dichtator", action="store_true", help="Sync Dichtator VPN connection history")
    parser.add_argument("--audit-workspaces", action="store_true", help="Audit all AI workspaces (Antigravity & t3code)")
    parser.add_argument("--report", action="store_true", help="Generate Markdown and HTML reports")
    parser.add_argument("--review", action="store_true", help="Interactive terminal workflow to review flagged ambiguous sessions")
    parser.add_argument("--review-all", action="store_true", help="Review all sessions interactively")
    parser.add_argument("--override", nargs=2, metavar=("SESSION_ID", "CATEGORY"), help="Set category override: --override <session_id> <steered|autonomous|idle>")
    parser.add_argument("--push", action="store_true", help="Auto-commit and push updated stats to GitHub origin/thesis-stats")

    args = parser.parse_args()
    tracker = ThesisTracker()

    if args.override:
        tracker.set_override(args.override[0], args.override[1])
        return

    if args.review:
        tracker.review_interactive(review_all=False)
        return

    if args.review_all:
        tracker.review_interactive(review_all=True)
        return

    if len(sys.argv) == 1 or args.sync_all:
        tracker.auto_scan_notewise()
        tracker.sync_all_sources()
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
    if args.audit_workspaces or args.sync_dichtator:
        tracker.sync_all_sources()
    if args.report:
        tracker.generate_markdown_report()
        tracker.generate_html_dashboard()
        print(f"[Done] Reports written to:\n  - {REPORT_MD}\n  - {REPORT_HTML}")
    if args.push:
        git_commit_and_push()


if __name__ == "__main__":
    main()
