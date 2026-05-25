from __future__ import annotations

from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, SensorReading


class VoiceResponseDetector(Detector):
    """Listens for a passenger reply after a check-in prompt. The response
    orchestrator arms it via the `on_check_in` hook; if no reply arrives
    inside the window, downstream logic should escalate to ELEVATED."""

    name = "voice_response"
    modality = Modality.MICROPHONE

    def __init__(self) -> None:
        self._awaiting = False

    async def arm(self) -> None:
        self._awaiting = True

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        if not self._awaiting:
            return None
        # TODO: VAD + keyword spotter ("yes"/"no"/distress phrases),
        # disarm on response, emit ELEVATED on window timeout.
        return None
