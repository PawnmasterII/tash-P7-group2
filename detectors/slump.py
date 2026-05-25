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
            tier = RiskTier.ELEVATED
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
