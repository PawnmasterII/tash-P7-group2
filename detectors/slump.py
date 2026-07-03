from __future__ import annotations

from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading

SLUMP_WATCH_DEG = 25.0
SLUMP_ALERT_DEG = 45.0


class SlumpDetector(Detector):
    name = "slump"
    modality = Modality.POSTURE

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        angle = float(reading.payload.get("slump_angle_deg", 0.0))
        if angle >= SLUMP_ALERT_DEG:
            # Severe slump: gate on voice check-in before dispatching.
            # VoiceResponseDetector escalates to ELEVATED if no response arrives
            # within the response window (or on a distress cue).
            tier = RiskTier.CHECK_IN
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
