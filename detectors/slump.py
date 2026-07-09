from __future__ import annotations

import time

from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading

SLUMP_WATCH_DEG = 25.0
SLUMP_ALERT_DEG = 45.0

# How long to suppress slump alerts after the passenger is reassured.
_DISMISS_COOLDOWN_S: float = 60.0


class SlumpDetector(Detector):
    name = "slump"
    modality = Modality.POSTURE

    def __init__(self) -> None:
        self._dismissed_until: float = 0.0

    def dismiss(self, duration_s: float = _DISMISS_COOLDOWN_S) -> None:
        """Suppress all slump events for *duration_s* seconds.

        Called by the de-escalation path so that a passenger who says
        "okay" isn't immediately re-prompted while still physically
        slouched.
        """
        self._dismissed_until = time.monotonic() + duration_s

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        angle = float(reading.payload.get("slump_angle_deg", 0.0))
        dismissed = time.monotonic() < self._dismissed_until

        if angle >= SLUMP_ALERT_DEG:
            # During dismissal cooldown: downgrade severe slump to WATCH so
            # posture monitoring stays visible on the console but doesn't
            # re-trigger a voice check-in prompt.
            tier = RiskTier.WATCH if dismissed else RiskTier.CHECK_IN
        elif angle >= SLUMP_WATCH_DEG:
            tier = RiskTier.WATCH
        else:
            return None
        return DetectionEvent(
            detector=self.name,
            modality=self.modality,
            label="slump",
            confidence=min(1.0, angle / 90.0),
            risk_tier=tier,
            metadata={"angle_deg": angle},
        )
