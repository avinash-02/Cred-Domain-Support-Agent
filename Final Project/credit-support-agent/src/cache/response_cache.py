"""
Normalized response cache for grounded generation.

Caches AgentResponse objects keyed by a normalized form of the user query
so that repeated identical (or whitespace/punctuation-variant) queries bypass
the full crew kickoff and retrieval step entirely.
"""

import re
from typing import Any, Dict, Optional
import threading


def normalize_query(query: str) -> str:
    """
    Normalize a user query into a canonical cache key.

    Strips leading/trailing whitespace, lowercases, removes internal whitespace,
    and strips trailing punctuation.  "What is the loan policy?" and
    "what is the loan policy?   " therefore map to the same key.
    """
    if not query:
        return ""
    # Collapse all runs of whitespace into a single space
    normalized = re.sub(r'\s+', ' ', query).strip().lower()
    # Strip common sentence-ending punctuation
    normalized = normalized.rstrip('?:!.,;')
    return normalized


class QueryCache:
    """
    Thread-safe in-memory LRU-style response cache.

    Stores AgentResponse objects keyed by a normalized user query string.
    Tracks hit / miss statistics and provides a `clear` method to reset state
    between test runs.
    """

    _MAX_ENTRIES = 512

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.entries = 0

    def get(self, query: str) -> Optional[Any]:
        """Return cached response if present; increment hits counter."""
        key = normalize_query(query)
        with self._lock:
            cached = self._data.get(key)
            if cached is not None:
                self.hits += 1
                # LRU eviction: remove most-recently-used entry
                del self._data[key]
                self._data[key] = cached
                return cached
            self.misses += 1
            return None

    def set(self, query: str, response: Any) -> None:
        """Store a response under the normalized query key."""
        key = normalize_query(query)
        with self._lock:
            if len(self._data) >= self._MAX_ENTRIES:
                # Evict oldest entry (simple FIFO eviction)
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
            self._data[key] = response
            self.entries += 1

    def clear(self) -> None:
        """Remove all entries and reset all counters."""
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0
            self.entries = 0
