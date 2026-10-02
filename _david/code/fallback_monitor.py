"""Fallback monitoring and alerting system for DrugEx LBDD pipeline."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("DrugEx.FallbackMonitor")


@dataclass
class FallbackEvent:
    """Record of a single fallback mechanism invocation."""

    fallback_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fallback event to serializable dictionary."""
        return {
            "fallback_type": self.fallback_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context,
        }


class FallbackMonitor:
    """Centralized observable monitor and auditor for fallback mechanisms."""

    _events: list[FallbackEvent] = []
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def record_fallback(
        cls,
        fallback_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> FallbackEvent:
        """Record and alert an observable fallback mechanism invocation."""
        event = FallbackEvent(
            fallback_type=fallback_type,
            message=message,
            context=context or {},
        )
        with cls._lock:
            cls._events.append(event)

            alert_msg = (
                f"\n{'!' * 70}\n"
                f"[FALLBACK MONITOR ALERT] Category: {fallback_type}\n"
                f"Reason: {message}\n"
            )
            if context:
                ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
                alert_msg += f"Context: {ctx_str}\n"
            alert_msg += f"{'!' * 70}\n"

            print(alert_msg)
            logger.warning(
                "Fallback triggered: %s | %s | context=%s",
                fallback_type,
                message,
                context,
            )
        return event

    @classmethod
    def has_fallbacks(cls) -> bool:
        """Check if any fallback events have been recorded."""
        with cls._lock:
            return len(cls._events) > 0

    @classmethod
    def get_events(cls) -> list[FallbackEvent]:
        """Retrieve all recorded fallback events."""
        with cls._lock:
            return list(cls._events)

    @classmethod
    def get_counts(cls) -> dict[str, int]:
        """Aggregate fallback events by type."""
        with cls._lock:
            counts: dict[str, int] = {}
            for ev in cls._events:
                counts[ev.fallback_type] = counts.get(ev.fallback_type, 0) + 1
            return counts

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """Generate structured audit summary dictionary."""
        with cls._lock:
            return {
                "total_fallbacks": len(cls._events),
                "counts_by_type": cls.get_counts(),
                "events": [ev.to_dict() for ev in cls._events],
            }

    @classmethod
    def print_summary(cls) -> None:
        """Print a formatted post-run audit report to stdout."""
        with cls._lock:
            counts = cls.get_counts()
            print("\n" + "=" * 80)
            print(" DRUGEX LBDD FALLBACK & HEALTH AUDIT REPORT")
            print("=" * 80)
            if not cls._events:
                print(" [OK] Zero fallback mechanisms triggered.")
                print(" All calculations executed strictly on primary optimal algorithms.")
            else:
                print(f" [WARNING] Total fallback events recorded: {len(cls._events)}")
                print("-" * 80)
                for f_type, count in counts.items():
                    print(f"   • {f_type}: {count} event(s)")
                print("-" * 80)
                print(" Detailed Fallback Log:")
                for idx, ev in enumerate(cls._events, start=1):
                    ctx_detail = f" | context: {ev.context}" if ev.context else ""
                    print(f"   [{idx}] {ev.timestamp} | {ev.fallback_type}: {ev.message}{ctx_detail}")
            print("=" * 80 + "\n")

    @classmethod
    def save_audit_json(cls, output_path: Path) -> Path:
        """Persist structured fallback audit to JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = cls.get_summary()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return target

    @classmethod
    def reset(cls) -> None:
        """Clear all recorded fallback events for clean test runs."""
        with cls._lock:
            cls._events.clear()
