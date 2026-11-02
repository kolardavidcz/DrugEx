#!/usr/bin/env python3
"""Forensic evidence extractor for the thesis hourly calendar.

Extracts every human-authored steering message from local Gemini Antigravity
transcript logs (``brain/*/.system_generated/logs/transcript.jsonl`` across
WSL and Windows stores), joins each message with its stable UUID from the
Antigravity conversation SQLite databases, merges T3 Code thread messages
(dumped via the T3 API into ``t3_messages.json``), deduplicates across
synced copies, and computes continuous *engagement sessions* with a
configurable inter-message gap threshold.

Outputs (written next to this script):
    evidence.json     -- all messages + per-chat metadata + daily aggregates
    sessions.json     -- engagement sessions per day (gap-merged)
    message_ledger.csv-- flat, human-filterable ledger of every message

All timestamps are UTC epochs in the raw data; every output carries both the
UTC ISO string and the Europe/Prague local rendering (validated against the
harness-injected "The current local time is: ..." ground truth).

Usage:
    python3 extract_evidence.py [--gap-min 35]
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

PRAGUE = ZoneInfo("Europe/Prague")
HERE = os.path.dirname(os.path.abspath(__file__))

# Calendar window of the committee-facing WEEKS model (inclusive start day).
WINDOW_START = "2026-09-14"

#: Reading rate for post-prompt assimilation of the triggered model response
#: (see cognitive_supervision_reading_framework.md: ~100 WPM ≈ 130 tokens/min).
READ_TOK_PER_MIN = 130.0

#: Nudge/acknowledgement vocabulary. A message whose every word (after
#: punctuation stripping) is in this set is a bare continuation with no
#: steering content. Single letters are deliberate nudges ("k", "y", "u").
_NUDGE_WORDS = frozenset(
    "continue ok okay okey k kk go on ahead yes yeah yep ya y sure nice good "
    "great done perfect awesome agreed thanks thank you thx ty np u please pls "
    "plz proceed keep going it got do where left off and so then now more we "
    "sould should hm hmm hmmm lol".split()
)

_SLASH_CMD_RE = re.compile(r"^/[a-z][\w-]*$", re.IGNORECASE)


def is_low_effort(text: str) -> bool:
    """True when a user message is a bare continuation/nudge with no content.

    Only explicit continuation/acknowledgement phrases are filtered: every
    word must come from the nudge vocabulary (<= 6 words). Anything with a
    digit, any real steering word, or an empty (attachment-only) prompt
    stays substantive.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return False  # empty = attachment-only prompt; keep as substantive
    if any(ch.isdigit() for ch in collapsed):
        return False  # numbers signal real content (times, ids, settings)
    if _SLASH_CMD_RE.match(collapsed):
        return True  # harness slash commands: /compact, /grill-me, ...
    words = re.sub(r"[^\w\s]", " ", collapsed.lower()).split()
    return len(words) <= 6 and all(w in _NUDGE_WORDS for w in words)

#: Brain transcript roots across WSL + Windows stores (label, path).
BRAIN_ROOTS: list[tuple[str, str]] = [
    ("antigravity", os.path.expanduser("~/.gemini/antigravity/brain")),
    ("antigravity-cli", os.path.expanduser("~/.gemini/antigravity-cli/brain")),
    ("antigravity-ide", os.path.expanduser("~/.gemini/antigravity-ide/brain")),
    ("antigravity", "/mnt/c/Users/kolar/.gemini/antigravity/brain"),
    ("antigravity-cli", "/mnt/c/Users/kolar/.gemini/antigravity-cli/brain"),
    ("antigravity-ide", "/mnt/c/Users/kolar/.gemini/antigravity-ide/brain"),
]

#: Conversation SQLite roots for message-UUID joins.
CONV_ROOTS: list[str] = [
    os.path.expanduser("~/.gemini/antigravity/conversations"),
    os.path.expanduser("~/.gemini/antigravity-cli/conversations"),
    os.path.expanduser("~/.gemini/antigravity-ide/conversations"),
    "/mnt/c/Users/kolar/.gemini/antigravity/conversations",
    "/mnt/c/Users/kolar/.gemini/antigravity-cli/conversations",
    "/mnt/c/Users/kolar/.gemini/antigravity-ide/conversations",
]


# ---------------------------------------------------------------------------
# Minimal protobuf walking (Antigravity steps.step_payload)
# ---------------------------------------------------------------------------

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _walk(buf: bytes, pos: int = 0, end: int | None = None) -> list[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) triples for a protobuf buffer."""
    if end is None:
        end = len(buf)
    out: list[tuple[int, int, object]] = []
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
            out.append((fn, wt, val))
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            out.append((fn, wt, buf[pos:pos + ln]))
            pos += ln
        elif wt == 5:
            out.append((fn, wt, buf[pos:pos + 4]))
            pos += 4
        elif wt == 1:
            out.append((fn, wt, buf[pos:pos + 8]))
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wt} at {pos}")
    return out


def load_message_uuids() -> dict[str, dict[int, str]]:
    """Map conversation id -> {step_idx: message UUID} from SQLite stores."""
    mapping: dict[str, dict[int, str]] = {}
    for root in CONV_ROOTS:
        for dbf in glob.glob(os.path.join(root, "*.db")):
            conv = os.path.basename(dbf)[:-3]
            best = mapping.get(conv, {})
            if len(best) >= 100000:
                continue
            try:
                db = sqlite3.connect(f"file:{dbf}?mode=ro", uri=True)
            except sqlite3.Error:
                continue
            try:
                cur = db.execute("SELECT idx, step_type, step_payload FROM steps")
            except sqlite3.OperationalError:
                db.close()
                continue
            local: dict[int, str] = {}
            for idx, stype, payload in cur.fetchall():
                if stype != 14 or not payload:
                    continue
                uuid: str | None = None
                for fn, _wt, val in _walk(payload):
                    if fn == 5 and isinstance(val, bytes):
                        for f2, _w2, v2 in _walk(val):
                            if f2 == 12 and isinstance(v2, bytes):
                                cand = v2.decode("utf-8", "replace")
                                if len(cand) == 36 and cand.count("-") == 4:
                                    uuid = cand
                if uuid:
                    local[idx] = uuid
            db.close()
            if len(local) > len(best):
                mapping[conv] = local
    return mapping


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

@dataclass
class RawMsg:
    chat: str
    step: int
    at_utc: str
    ts: float
    text: str
    harness: str
    uuid: str | None = None
    # Aggregate model-response volume this message triggered (filled later).
    out_tokens: int = 0
    think_words: int = 0
    tool_calls: int = 0
    response_words: int = 0
    turns: int = 0


def _clean_user_content(content: str) -> str:
    """Strip harness wrappers (<USER_REQUEST> etc.) from a user input blob."""
    txt = content
    if "<USER_REQUEST>" in txt:
        inner = txt.split("<USER_REQUEST>", 1)[1]
        inner = inner.split("</USER_REQUEST>", 1)[0]
        txt = inner
    for tag in ("<ADDITIONAL_METADATA>", "<USER_SETTINGS_CHANGE>", "<TOOL_RESULT>"):
        if tag in txt:
            txt = txt.split(tag, 1)[0]
    return txt.strip()


def _parse_transcript(path: str, harness: str, chat: str) -> tuple[list[RawMsg], dict]:
    """Parse one transcript.jsonl into user messages + conversation stats."""
    msgs: list[RawMsg] = []
    stats = {"user": 0, "model": 0, "tool_calls": 0, "out_tokens": 0}
    pending: RawMsg | None = None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = d.get("source")
            typ = d.get("type")
            at = d.get("created_at")
            if not at:
                continue
            if src == "USER_EXPLICIT" and typ == "USER_INPUT":
                if pending is not None:
                    msgs.append(pending)
                pending = RawMsg(
                    chat=chat,
                    step=d.get("step_index", -1),
                    at_utc=at,
                    ts=datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp(),
                    text=_clean_user_content(d.get("content", "")),
                    harness=harness,
                )
                stats["user"] += 1
            elif src == "MODEL" and pending is not None:
                stats["model"] += 1
                pending.turns += 1
                pending.out_tokens += int(d.get("output_tokens") or 0)
                thinking = d.get("thinking") or ""
                pending.think_words += len(thinking.split())
                calls = d.get("tool_calls") or []
                pending.tool_calls += len(calls)
                resp = d.get("content") or ""
                if resp and len(resp) < 200_000:
                    pending.response_words += len(resp.split())
                stats["tool_calls"] += len(calls)
                stats["out_tokens"] += int(d.get("output_tokens") or 0)
    if pending is not None:
        msgs.append(pending)
    return msgs, stats


def extract_all() -> tuple[list[RawMsg], dict[str, dict]]:
    """Extract + dedupe user messages across every store; return per-chat stats."""
    uuid_map = load_message_uuids()
    per_chat: dict[str, dict] = {}
    by_key: dict[tuple, RawMsg] = {}
    per_conv_copies: dict[str, list[tuple[str, int]]] = {}

    for harness, root in BRAIN_ROOTS:
        if not os.path.isdir(root):
            continue
        for conv_dir in sorted(glob.glob(os.path.join(root, "*"))):
            chat = os.path.basename(conv_dir)
            tpath = os.path.join(conv_dir, ".system_generated", "logs", "transcript.jsonl")
            if not os.path.isfile(tpath):
                continue
            try:
                msgs, stats = _parse_transcript(tpath, harness, chat)
            except (OSError, ValueError):
                continue
            per_conv_copies.setdefault(chat, []).append((tpath, len(msgs)))
            uuids = uuid_map.get(chat, {})
            for m in msgs:
                m.uuid = uuids.get(m.step)
                key = (m.ts, hashlib.sha1(m.text.encode()).hexdigest()[:12])
                prev = by_key.get(key)
                if prev is None or (m.uuid and not prev.uuid):
                    by_key[key] = m
            agg = per_chat.setdefault(chat, {
                "harness": harness, "msgs": 0, "model_turns": 0,
                "tool_calls": 0, "out_tokens": 0, "copies": [],
            })
            agg["msgs"] = max(agg["msgs"], stats["user"])
            agg["model_turns"] = max(agg["model_turns"], stats["model"])
            agg["tool_calls"] = max(agg["tool_calls"], stats["tool_calls"])
            agg["out_tokens"] = max(agg["out_tokens"], stats["out_tokens"])
            agg["copies"].append(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(tpath))))) + "/" + chat[:8])

    all_msgs = sorted(by_key.values(), key=lambda m: m.ts)
    return all_msgs, per_chat


# ---------------------------------------------------------------------------
# T3 Code threads
# ---------------------------------------------------------------------------

def load_t3_messages() -> list[RawMsg]:
    """Load user messages dumped from T3 Code threads via the T3 API."""
    path = os.path.join(HERE, "t3_messages.json")
    if not os.path.isfile(path):
        return []
    data = json.load(open(path))
    out: list[RawMsg] = []
    for thread_id, msgs in data.items():
        for m in msgs:
            at = m["at"]
            ts = datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
            out.append(RawMsg(
                chat=thread_id, step=-1, at_utc=at, ts=ts,
                text=m["txt"], harness="t3-code", uuid=m.get("id"),
            ))
    return out


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@dataclass
class Session:
    date: str
    start_min: float
    end_min: float
    #: substantive messages only (proof log rows, counts, ticks)
    msgs: list[dict] = field(default_factory=list)
    #: all anchored messages incl. continuations (span/presence)
    n_all: int = 0
    n_cont: int = 0
    words: int = 0
    chats: set = field(default_factory=set)
    out_tokens: int = 0
    tool_calls: int = 0
    #: out_tokens of the LAST anchored message — only its response is still
    #: unread when the session ends (mid-session responses were read during
    #: the gap-merged span, so they must not extend the tail again).
    last_out: int = 0
    read_min: float = 0.0
    act_min: float = 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "start_min": round(self.start_min, 2),
            "end_min": round(self.end_min, 2),
            "n_msgs": len(self.msgs),
            "n_cont": self.n_cont,
            "n_all": self.n_all,
            "words": self.words,
            "chats": sorted(self.chats),
            "out_tokens": self.out_tokens,
            "tool_calls": self.tool_calls,
            "act_min": round(self.act_min, 1),
            "read_min": round(self.read_min, 1),
            "msgs": self.msgs,
        }


def build_sessions(msgs: list[dict], gap_min: float, tail_min: float = 4.0) -> list[dict]:
    """Cluster messages into continuous engagement sessions.

    A session stays open while the gap to the next message is <= ``gap_min``
    minutes. *All* messages anchor presence (dropping continuations would
    split sessions and undercount), but only substantive messages enter the
    proof log/counts. Each session's end is extended past its last message
    by the *reading time* of the final response that message triggered:
    ``min(gap_min, max(tail_min, last_out / READ_TOK_PER_MIN))``. Responses
    to earlier prompts were already read inside the gap-merged span, and
    reading longer than one engagement window would have produced another
    steering prompt — hence the cap. The extension is clamped by the next
    session's start (same date) and by midnight so nothing is double-counted
    or rendered past the day column.
    """
    sessions: list[Session] = []
    cur: Session | None = None
    for m in msgs:
        dt = datetime.fromtimestamp(m["ts"], PRAGUE)
        date = dt.strftime("%Y-%m-%d")
        minute = dt.hour * 60 + dt.minute + dt.second / 60.0
        if cur is None or (date != cur.date) or (minute - cur.end_min) > gap_min:
            if cur is not None:
                sessions.append(cur)
            cur = Session(date=date, start_min=minute, end_min=minute)
        cur.end_min = max(cur.end_min, minute)
        cur.n_all += 1
        if m.get("le"):
            cur.n_cont += 1
        else:
            cur.msgs.append(m["ref"])
            cur.words += m["words"]
        cur.chats.add(m["chat"])
        cur.out_tokens += m["out_tokens"]
        cur.tool_calls += m["tool_calls"]
        cur.last_out = m["out_tokens"]
    if cur is not None:
        sessions.append(cur)

    # Trailing reading extension. end_min currently equals the last
    # message minute; extend by the final response's reading time (capped
    # at one engagement window), clamped by the next session's start on
    # the same date and by midnight.
    for i, s in enumerate(sessions):
        last_min = s.end_min
        s.act_min = last_min - s.start_min
        want = min(gap_min, max(tail_min, s.last_out / READ_TOK_PER_MIN))
        read = min(want, 1440.0 - last_min)
        nxt = sessions[i + 1] if i + 1 < len(sessions) else None
        if nxt is not None and nxt.date == s.date:
            read = min(read, max(0.0, nxt.start_min - last_min))
        s.read_min = read
        s.end_min = min(1440.0, last_min + read)
    return [s.to_dict() for s in sessions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-min", type=float, default=35.0,
                    help="Max inter-message gap (minutes) merged into one session")
    args = ap.parse_args()

    ag_msgs, per_chat = extract_all()
    t3_msgs = load_t3_messages()

    # Cross-source dedupe (some T3 antigravity-provider threads also mirror
    # into brain folders): key = (epoch-minute, text hash).
    seen: set[tuple] = set()
    merged: list[RawMsg] = []
    for m in sorted(ag_msgs + t3_msgs, key=lambda x: x.ts):
        key = (int(m.ts // 60), hashlib.sha1(m.text.encode()).hexdigest()[:12])
        if key in seen:
            continue
        seen.add(key)
        merged.append(m)

    records: list[dict] = []
    for m in merged:
        dt = datetime.fromtimestamp(m.ts, PRAGUE)
        rec = {
            "chat": m.chat, "uuid": m.uuid, "step": m.step,
            "at_utc": m.at_utc, "ts": m.ts,
            "local": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date": dt.strftime("%Y-%m-%d"),
            "hm": dt.strftime("%H:%M"),
            "words": len(m.text.split()),
            "excerpt": " ".join(m.text.split())[:180],
            "harness": m.harness,
            "low_effort": is_low_effort(m.text),
            "out_tokens": m.out_tokens, "think_words": m.think_words,
            "tool_calls": m.tool_calls, "response_words": m.response_words,
            "turns": m.turns,
        }
        rec["ref"] = None  # placeholder, filled for sessions
        records.append(rec)

    in_window = [r for r in records if r["date"] >= WINDOW_START]
    sessions = build_sessions(
        [{**r, "le": r["low_effort"], "ref": {
            "t": r["hm"], "c": r["chat"][:8], "u": (r["uuid"] or f"step{r['step']}")[:8],
            "w": r["words"], "x": r["excerpt"][:140],
            "o": r["out_tokens"], "tc": r["tool_calls"],
        }} for r in in_window],
        gap_min=args.gap_min,
    )

    hours = sum(s["end_min"] - s["start_min"] for s in sessions) / 60.0
    read_hours = sum(s["read_min"] for s in sessions) / 60.0
    n_sub = sum(1 for r in in_window if not r["low_effort"])
    evidence = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "gap_min": args.gap_min,
        "window_start": WINDOW_START,
        "tz": "Europe/Prague",
        "total_messages": len(records),
        "in_window_messages": len(in_window),
        "in_window_substantive": n_sub,
        "in_window_nudges": len(in_window) - n_sub,
        "total_words": sum(r["words"] for r in records),
        "in_window_words": sum(r["words"] for r in in_window),
        "n_chats": len({r["chat"] for r in records}),
        "engagement_hours": round(hours, 2),
        "reading_hours": round(read_hours, 2),
        "chats": [
            {"chat": c, **stats} for c, stats in sorted(
                per_chat.items(), key=lambda kv: -kv[1]["msgs"])
        ],
        "messages": [
            {k: v for k, v in r.items() if k != "ref"} for r in records
        ],
    }
    with open(os.path.join(HERE, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=1)
    with open(os.path.join(HERE, "sessions.json"), "w") as fh:
        json.dump({"gap_min": args.gap_min, "sessions": sessions}, fh, indent=1)

    with open(os.path.join(HERE, "message_ledger.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["local_time_prague", "utc", "date", "chat_id", "message_uuid",
                    "words", "low_effort", "out_tokens", "tool_calls", "harness",
                    "in_window", "excerpt"])
        for r in records:
            w.writerow([r["local"], r["at_utc"], r["date"], r["chat"], r["uuid"],
                        r["words"], r["low_effort"], r["out_tokens"],
                        r["tool_calls"], r["harness"],
                        r["date"] >= WINDOW_START, r["excerpt"]])

    print(f"messages total={len(records)} in-window={len(in_window)} "
          f"substantive={n_sub} nudges-filtered={len(in_window) - n_sub} "
          f"chats={evidence['n_chats']} words={evidence['total_words']:,}")
    print(f"sessions(gap<={args.gap_min:.0f}m)={len(sessions)} "
          f"engagement_hours={hours:.1f} (incl. {read_hours:.1f}h post-prompt "
          f"reading extension)")
    print("longest sessions:")
    for s in sorted(sessions, key=lambda s: s["end_min"] - s["start_min"], reverse=True)[:8]:
        span = (s["end_min"] - s["start_min"]) / 60
        print(f"  {s['date']} {int(s['start_min']//60):02d}:{int(s['start_min']%60):02d}"
              f"-{int(s['end_min']//60):02d}:{int(s['end_min']%60):02d} "
              f"{span:5.2f}h msgs={s['n_msgs']}(+{s['n_cont']} cont.) "
              f"words={s['words']:,} chats={len(s['chats'])} "
              f"read=+{s['read_min']:.0f}m")


if __name__ == "__main__":
    main()
