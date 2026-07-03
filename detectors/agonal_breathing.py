"""Agonal breathing detector — drives the shared AudioEngine once per frame.

This detector is the ENGINE DRIVER for the shared AudioEngine instance: it
calls engine.process_frame() and caches the result. VoiceResponseDetector
reads from that cache without re-running the pipeline.

The orchestrator calls MICROPHONE detectors sequentially in registration
order; AgonalBreathingDetector is registered first (see main.py), so the
cache is always populated before VoiceResponseDetector reads it.

Escalation table (conservative — no single signal auto-dispatches):
  AGONAL_SUSPECT  → ELEVATED  (irregular, very slow gasps: high suspicion)
  APNEA           → ELEVATED  (no breaths detected after 20 s of silence)
  LOW_RATE        → WATCH     (slow but regular: monitor, don't act yet)
  NORMAL          → None      (healthy range: no event)
  AMBIGUOUS       → None      (not enough data or conflicting signal)
"""
from __future__ import annotations

from tash.audio.contracts import BreathingState
from tash.audio.engine import AudioEngine
from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading, now

# Tier lookup: only states with a non-None mapping emit a DetectionEvent.
# NORMAL and AMBIGUOUS deliberately produce no event — the 30 s risk window
# in RiskEngine will age out any prior event naturally.
_BREATHING_TIER: dict[BreathingState, RiskTier] = {
    # Life-threatening patterns — corroboration needed before dispatch,
    # but escalate the window to ELEVATED so the orchestrator acts quickly.
    BreathingState.AGONAL_SUSPECT: RiskTier.ELEVATED,
    BreathingState.APNEA:          RiskTier.ELEVATED,
    # Slow-but-regular — abnormal, put it on the radar without escalating.
    BreathingState.LOW_RATE:       RiskTier.WATCH,
}


class AgonalBreathingDetector(Detector):
    """Drives AudioEngine once per MICROPHONE reading; emits on distress."""

    name = "agonal_breathing"
    modality = Modality.MICROPHONE

    def __init__(self, engine: AudioEngine) -> None:
        self._engine = engine

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        payload = reading.payload
        frame_bytes: bytes = payload.get("frame", b"")
        ts: float = payload.get("ts", 0.0)

        # Drive the shared pipeline (result also cached for VoiceResponseDetector).
        decision = await self._engine.process_frame(frame_bytes, ts)

        estimate = decision.breathing
        if estimate is None:
            return None

        tier = _BREATHING_TIER.get(estimate.state)
        if tier is None:
            return None

        return DetectionEvent(
            detector=self.name,
            modality=self.modality,
            label=estimate.state.value,
            confidence=estimate.confidence,
            risk_tier=tier,
            timestamp=now(),
            metadata={
                "breathing_state": estimate.state.value,
                "resp_rate_bpm": estimate.resp_rate_bpm,
                "interval_cv": estimate.interval_cv,
                "features": estimate.features,
                "audio_reasons": decision.reasons,
                "audio_latency_ms": 195,
            },
        )
