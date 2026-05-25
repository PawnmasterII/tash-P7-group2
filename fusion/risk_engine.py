from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

from tash.types import DetectionEvent, RiskTier, now

DEFAULT_WINDOW = timedelta(seconds=30)


@dataclass
class RiskEngine:
    """Fuses recent detection events into an overall risk tier. v1 takes
    the max tier across a sliding window; swap for a learned fuser once
    labelled scenarios exist."""

    window: timedelta = DEFAULT_WINDOW
    _events: deque[DetectionEvent] = field(default_factory=deque)

    def ingest(self, event: DetectionEvent) -> RiskTier:
        self._events.append(event)
        self._evict()
        return self.current_tier()

    def current_tier(self) -> RiskTier:
        self._evict()
        if not self._events:
            return RiskTier.NORMAL
        return max(e.risk_tier for e in self._events)

    def _evict(self) -> None:
        cutoff = now() - self.window
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()
