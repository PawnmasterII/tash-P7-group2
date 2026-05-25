from __future__ import annotations

from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, SensorReading


class AgonalBreathingDetector(Detector):
    """Headline detector for the solo-passenger case. Replace the body
    with the UW smart-speaker agonal-breathing classifier or equivalent."""

    name = "agonal_breathing"
    modality = Modality.MICROPHONE

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        # TODO: window audio frames, run classifier, emit RiskTier.CRITICAL
        # with confidence on positive detection.
        return None
