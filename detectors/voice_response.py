"""Voice response / distress cue detector.

Reads the CueWordEvent cached by AudioEngine after AgonalBreathingDetector
has driven the pipeline for this frame. No second pipeline run occurs.

Two distinct roles:
 1. Distress detection (always active): emits CHECK_IN whenever "help" is
    detected, regardless of armed state. This prompts the orchestrator to
    verify the passenger is okay before escalating further.
 2. Check-in response tracking (armed state): arm() is called by the
    ResponseOrchestrator after a VOICE_CHECK_IN action fires. While armed
    the detector waits up to _RESPONSE_WINDOW_S seconds then:
      - reassurance word ("fine"/"okay"/"ok") → de-escalate (return None)
      - distress cue ("help") → escalate to ELEVATED
      - no response within window → emit ELEVATED "no_response_timeout"

Safety invariant: "help" alone (unarmed) → CHECK_IN, NOT ELEVATED.
Corroboration (slump + no-response-to-prompt, or slump + "help") is what
drives ELEVATED — that logic runs through the armed window here.
"""
from __future__ import annotations

import logging
import time

from tash.audio.contracts import CueWordEvent
from tash.audio.engine import AudioEngine
from tash.detectors.base import Detector
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading, now

log = logging.getLogger(__name__)

# How long to wait for a passenger response after VOICE_CHECK_IN fires.
_RESPONSE_WINDOW_S: float = 30.0


class VoiceResponseDetector(Detector):
    """Reads engine cache; emits on distress cue word ("help")."""

    name = "voice_response"
    modality = Modality.MICROPHONE

    def __init__(self, engine: AudioEngine, response_window_s: float = _RESPONSE_WINDOW_S) -> None:
        self._engine = engine
        self._response_window_s = response_window_s
        self._awaiting = False
        self._arm_monotonic: float = 0.0
        self._last_cue_ts: float = -1e9

    async def arm(self) -> None:
        """Called by ResponseOrchestrator when a VOICE_CHECK_IN fires."""
        self._awaiting = True
        self._arm_monotonic = time.monotonic()

    async def observe(self, reading: SensorReading) -> DetectionEvent | None:
        # Timeout: armed + no response within window → escalate.
        if self._awaiting:
            elapsed = time.monotonic() - self._arm_monotonic
            if elapsed >= self._response_window_s:
                self._awaiting = False
                log.info("[voice_response] no response after %.0fs - escalating", elapsed)
                return DetectionEvent(
                    detector=self.name,
                    modality=self.modality,
                    label="no_response_timeout",
                    confidence=1.0,
                    risk_tier=RiskTier.ELEVATED,
                    timestamp=now(),
                    metadata={"elapsed_s": round(elapsed, 1)},
                )

        cue: CueWordEvent | None = self._engine.latest_cue

        if cue is None:
            return None

        # Deduplicate: only emit once per unique cue-word event (identified by ts).
        if cue.ts == self._last_cue_ts:
            return None
        self._last_cue_ts = cue.ts

        # Reassurance branch: de-escalate when armed, silently ignore when not.
        if cue.category == "reassurance":
            if self._awaiting:
                log.info("[voice_response] passenger reassured (%r) - de-escalating", cue.keyword)
                self._awaiting = False
            return None

        # Distress path ("help").
        # Armed: corroboration achieved (slump + distress cue) → ELEVATED.
        # Unarmed: single cue word — prompt passenger before escalating (safety invariant).
        if self._awaiting:
            self._awaiting = False
            return DetectionEvent(
                detector=self.name,
                modality=self.modality,
                label="help_while_awaiting",
                confidence=cue.confidence_proxy,
                risk_tier=RiskTier.ELEVATED,
                timestamp=now(),
                metadata={
                    "keyword": cue.keyword,
                    "vad_active": cue.vad_active,
                    "confidence_proxy": cue.confidence_proxy,
                    "audio_latency_ms": 195,
                },
            )

        return DetectionEvent(
            detector=self.name,
            modality=self.modality,
            label="help_distress_cue",
            confidence=cue.confidence_proxy,
            # CHECK_IN: single word is never enough to auto-dispatch
            # (TASHaudio safety invariant).
            risk_tier=RiskTier.CHECK_IN,
            timestamp=now(),
            metadata={
                "keyword": cue.keyword,
                "vad_active": cue.vad_active,
                "confidence_proxy": cue.confidence_proxy,
                "audio_latency_ms": 195,
            },
        )
