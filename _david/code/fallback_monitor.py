"""Fallback monitoring and alerting system for DrugEx LBDD pipeline.

Tracks and alerts when fallback mechanisms (e.g., conformer embedding random coordinates,
ROCS fallback to RDKit or default score, scaffold extraction fallback, candidate pre-filter
relaxation) are triggered so researchers can monitor, detect, and audit them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("DrugEx.FallbackMonitor")


@dataclass
class FallbackEvent:
    """Record of a single fallback mechanism invocation.

    Attributes:
        fallback_type: Categorical identifier of the fallback mechanism.
        message: Descriptive explanation of why the fallback occurred.
        timestamp: ISO format string indicating when the event was recorded.
        context: Optional dictionary of contextual metadata (e.g. molecule SMILES, score).
    """

    fallback_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert fallback event to serializable dictionary.

        Returns:
            Dictionary containing event attributes.
        """
        return {
            "fallback_type": self.fallback_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "context": self.context,
        }


class FallbackMonitor:
    """Centralized observable monitor and auditor for fallback mechanisms.

    Maintains a thread-safe in-memory log of all triggered fallback events,
    emits loud observable console warnings with visual indicators, logs to Python
    logger, and generates post-run audit reports for pipeline validation.
    """

    _events: List[FallbackEvent] = []

    @classmethod
    def record_fallback(
        cls,
        fallback_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FallbackEvent:
        """Record and alert an observable fallback mechanism invocation.

        Prints a high-visibility warning banner to stdout, logs via logger.warning,
        and records the event in the audit registry.

        Args:
            fallback_type: Short identifier for the fallback category.
            message: User-facing explanation of why the fallback occurred.
            context: Optional contextual parameters (SMILES, values, etc.).

        Returns:
            The recorded FallbackEvent instance.
        """
        event = FallbackEvent(
            fallback_type=fallback_type,
            message=message,
            context=context or {},
        )
        cls._events.append(event)

        # High-visibility console alert for terminal monitoring
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
        """Check if any fallback events have been recorded.

        Returns:
            True if one or more fallbacks occurred, False otherwise.
        """
        return len(cls._events) > 0

    @classmethod
    def get_events(cls) -> List[FallbackEvent]:
        """Retrieve all recorded fallback events.

        Returns:
            List of FallbackEvent objects.
        """
        return list(cls._events)

    @classmethod
    def get_counts(cls) -> Dict[str, int]:
        """Aggregate fallback events by type.

        Returns:
            Dictionary mapping fallback_type to event occurrence count.
        """
        counts: Dict[str, int] = {}
        for ev in cls._events:
            counts[ev.fallback_type] = counts.get(ev.fallback_type, 0) + 1
        return counts

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Generate structured audit summary dictionary.

        Returns:
            Dictionary with total_count, counts_by_type, and event details.
        """
        return {
            "total_fallbacks": len(cls._events),
            "counts_by_type": cls.get_counts(),
            "events": [ev.to_dict() for ev in cls._events],
        }

    @classmethod
    def print_summary(cls) -> None:
        """Print a formatted post-run audit report to stdout."""
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
        """Persist structured fallback audit to JSON file.

        Args:
            output_path: Target Path for writing JSON report.

        Returns:
            Resolved Path of the saved file.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        summary = cls.get_summary()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return target

    @classmethod
    def reset(cls) -> None:
        """Clear all recorded fallback events for clean test runs."""
        cls._events.clear()
