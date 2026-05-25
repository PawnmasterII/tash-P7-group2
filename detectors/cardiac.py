from __future__ import annotations

from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading

BRADY_BPM = 40.0
TACHY_BPM = 130.0


class CardiacAnomalyDetector(Detector):
    """Toy threshold detector. Replace with the MIMIC-trained model."""

    name = "cardiac_anomaly"
    modality = Modality.HEART_RATE

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        bpm = float(reading.payload.get("bpm", 0.0))
        if BRADY_BPM < bpm < TACHY_BPM:
            return None
        return DetectionEvent(
            detector=self.name,
            modality=self.modality,
            label="cardiac_anomaly",
            confidence=0.7,
            risk_tier=RiskTier.ELEVATED,
            metadata={"bpm": bpm},
        )
