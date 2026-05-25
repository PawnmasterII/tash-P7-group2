from __future__ import annotations

from abc import ABC, abstractmethod

from tash.types import DetectionEvent, RiskTier


class Notifier(ABC):
    """Outbound channel to caregiver and/or dispatcher. Real implementations
    fan out to SMS, push, video bridge, or 911 PSAP integration."""

    @abstractmethod
    async def notify(
        self,
        tier: RiskTier,
        events: list[DetectionEvent],
        message: str,
    ) -> None: ...

    @abstractmethod
    async def open_video_feed(self) -> None: ...

    @abstractmethod
    async def dispatch_emergency(self) -> None: ...
