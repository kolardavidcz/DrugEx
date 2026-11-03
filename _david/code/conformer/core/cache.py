"""Content-addressed artifact cache.

Design
------
A cache key is ``sha256(algorithm_version, namespace, canonical_payload)``.
The same key is used for every representation of one artifact (JSON, SDF, NPZ),
so a caller can ask for the representation it needs and get a hit regardless of
which representation was written first.

Entries live under ``<root>/<namespace>/<first-2-hex>/<key>.<ext>`` with a
sibling ``<key>.meta.json``.  Metadata is written *after* the payload, so a
crash can leave an orphan payload but never a manifest that points at missing
data.

The cache is deliberately opt-in and can be disabled for debugging.  It never
mutates the objects it returns, and every read is validated by size before use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from _david.code.conformer.core import provenance, storage
from _david.code.conformer.core.contracts import ALGORITHM_VERSION, fingerprint, jsonable
from _david.code.conformer.core.errors import CacheError
from _david.code.conformer.core.paths import cache_root

CACHE_SCHEMA = "artifact-cache-v1"


@dataclass
class CacheStats:
    """Mutable hit/miss counters for one cache instance."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    disabled: int = 0

    @property
    def lookups(self) -> int:
        """Total lookups attempted."""
        return self.hits + self.misses + self.disabled

    @property
    def hit_rate(self) -> float:
        """Fraction of enabled lookups that were served from disk."""
        enabled = self.hits + self.misses
        return self.hits / enabled if enabled else 0.0

    def as_dict(self) -> dict[str, int | float]:
        """Serialize the counters for a run manifest."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "disabled": self.disabled,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 6),
        }


class ArtifactCache:
    """Filesystem-backed cache for expensive conformer artifacts."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        enabled: bool = True,
        algorithm_version: str = ALGORITHM_VERSION,
    ) -> None:
        self.root = Path(root) if root is not None else cache_root()
        self.enabled = bool(enabled)
        self.algorithm_version = algorithm_version
        self.stats = CacheStats()

    # -- keys ---------------------------------------------------------------
    def key(self, namespace: str, payload: Any) -> str:
        """Return the stable cache key for a namespace and payload."""
        return fingerprint(
            {
                "schema": CACHE_SCHEMA,
                "algorithm": self.algorithm_version,
                "namespace": namespace,
                "payload": jsonable(payload),
            }
        )

    def entry_dir(self, namespace: str, key: str) -> Path:
        """Return the sharded directory for one key."""
        return self.root / namespace / key[:2]

    def entry_path(self, namespace: str, key: str, extension: str) -> Path:
        """Return the payload path for one key and extension."""
        return self.entry_dir(namespace, key) / f"{key}.{extension}"

    def meta_path(self, namespace: str, key: str) -> Path:
        """Return the metadata path for one key."""
        return self.entry_dir(namespace, key) / f"{key}.meta.json"

    # -- generic ------------------------------------------------------------
    def has(self, namespace: str, key: str, extension: str) -> bool:
        """Return whether a non-empty payload exists for this key."""
        path = self.entry_path(namespace, key, extension)
        return path.is_file() and path.stat().st_size > 0

    def _record_hit(self) -> None:
        self.stats.hits += 1

    def _record_miss(self) -> None:
        if self.enabled:
            self.stats.misses += 1
        else:
            self.stats.disabled += 1

    def _write_meta(self, namespace: str, key: str, metadata: Mapping[str, Any] | None) -> None:
        payload = {
            "schema": CACHE_SCHEMA,
            "algorithm": self.algorithm_version,
            "namespace": namespace,
            "key": key,
            "created_utc": provenance.utc_now(),
            "metadata": jsonable(dict(metadata or {})),
        }
        storage.write_json(self.meta_path(namespace, key), payload)
        self.stats.writes += 1

    def read_meta(self, namespace: str, key: str) -> dict[str, Any]:
        """Return metadata for an entry, or an empty mapping when absent."""
        path = self.meta_path(namespace, key)
        if not path.is_file():
            return {}
        try:
            return storage.read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CacheError(f"Corrupt cache metadata at {path}: {exc}") from exc

    # -- JSON ---------------------------------------------------------------
    def get_json(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Load a JSON payload, or ``None`` on a miss."""
        if not self.enabled:
            self._record_miss()
            return None
        path = self.entry_path(namespace, key, "json")
        if not self.has(namespace, key, "json"):
            self._record_miss()
            return None
        try:
            payload = storage.read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CacheError(f"Corrupt cache JSON at {path}: {exc}") from exc
        self._record_hit()
        return payload

    def put_json(
        self,
        namespace: str,
        key: str,
        payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a JSON payload and its metadata.

        A disabled cache is a full no-op: the returned path is where the entry
        *would* live, so callers can still report a location without the cache
        ever touching the disk.
        """
        path = self.entry_path(namespace, key, "json")
        if not self.enabled:
            return path
        storage.write_json(path, payload)
        self._write_meta(namespace, key, metadata)
        return path

    # -- SDF ----------------------------------------------------------------
    def get_sdf(self, namespace: str, key: str) -> list[Any] | None:
        """Load a list of RDKit molecules, or ``None`` on a miss."""
        if not self.enabled:
            self._record_miss()
            return None
        path = self.entry_path(namespace, key, "sdf")
        if not self.has(namespace, key, "sdf"):
            self._record_miss()
            return None
        records = storage.read_sdf(path)
        if not records:
            raise CacheError(f"Empty cache SDF at {path}")
        self._record_hit()
        return records

    def put_sdf(
        self,
        namespace: str,
        key: str,
        records: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a list of RDKit molecules and its metadata."""
        path = self.entry_path(namespace, key, "sdf")
        if not self.enabled:
            return path
        storage.write_sdf(path, records)
        self._write_meta(namespace, key, metadata)
        return path

    # -- NPZ ----------------------------------------------------------------
    def get_npz(self, namespace: str, key: str) -> dict[str, np.ndarray] | None:
        """Load a NumPy archive, or ``None`` on a miss."""
        if not self.enabled:
            self._record_miss()
            return None
        path = self.entry_path(namespace, key, "npz")
        if not self.has(namespace, key, "npz"):
            self._record_miss()
            return None
        try:
            payload = storage.read_npz(path)
        except (OSError, ValueError) as exc:
            raise CacheError(f"Corrupt cache NPZ at {path}: {exc}") from exc
        self._record_hit()
        return payload

    def put_npz(
        self,
        namespace: str,
        key: str,
        payload: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write a NumPy archive and its metadata."""
        path = self.entry_path(namespace, key, "npz")
        if not self.enabled:
            return path
        storage.write_npz(path, payload)
        self._write_meta(namespace, key, metadata)
        return path

    # -- lifecycle ----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return a manifest-ready summary of the cache state."""
        return {
            "root": self.root.as_posix(),
            "enabled": self.enabled,
            "algorithm": self.algorithm_version,
            "stats": self.stats.as_dict(),
        }


__all__ = ["ArtifactCache", "CACHE_SCHEMA", "CacheStats"]
