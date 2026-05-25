from __future__ import annotations

import logging

from tash.comms.base import Notifier
from tash.types import DetectionEvent, RiskTier

log = logging.getLogger(__name__)


class MockNotifier(Notifier):
    async def notify(
        self,
        tier: RiskTier,
        events: list[DetectionEvent],
        message: str,
    ) -> None:
        log.info("[notify] tier=%s msg=%s events=%d", tier.name, message, len(events))

    async def open_video_feed(self) -> None:
        log.info("[notify] opening video feed to caregiver")

    async def dispatch_emergency(self) -> None:
        log.info("[notify] dispatching emergency services")
