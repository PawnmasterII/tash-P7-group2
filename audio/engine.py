"""Shared AudioEngine — one Pipeline instance, used by both MICROPHONE detectors.

Design goals:
 - Process each sensor frame exactly ONCE (AgonalBreathingDetector drives it;
   VoiceResponseDetector reads from the cache).
 - Never block the asyncio event loop: all Pipeline calls go via
   asyncio.to_thread, serialized by an asyncio.Lock.
 - Load the Vosk model once at startup; expose a degraded flag if cue-word
   init fails (Pipeline already prints [DEGRADED] and continues without ASR).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from tash.audio.contracts import BreathingEstimate, CueWordEvent, EscalationLevel, FusionDecision
from tash.audio.pipeline import Pipeline

log = logging.getLogger(__name__)


def _empty_decision(ts: float) -> FusionDecision:
    return FusionDecision(ts=ts, level=EscalationLevel.NONE, reasons=[], cue_word=None, breathing=None)


class AudioEngine:
    """Thread-safe async wrapper around tash.audio.Pipeline.

    Usage (in build_runtime / _amain):
        engine = AudioEngine()
        await engine.start()          # loads Vosk model off the event loop
        # inject engine into both detectors and the Microphone sensor
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._lock = asyncio.Lock()
        self._latest: FusionDecision | None = None
        self._suppress_cues_until: float = 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _build(self) -> None:
        """Construct Pipeline (blocking — called via asyncio.to_thread)."""
        self._pipeline = Pipeline()
        if self._pipeline.degraded:
            log.warning("AudioEngine: cue-word stage disabled (Vosk model missing or vosk not installed)")

    async def start(self) -> None:
        """Initialize pipeline off the event loop (model load is ~1-2 s)."""
        await asyncio.to_thread(self._build)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()

    # -------------------------------------------------------------------------
    # Processing — called by AgonalBreathingDetector once per frame
    # -------------------------------------------------------------------------

    async def process_frame(
        self,
        frame_bytes: bytes,
        ts: float,
        passenger_responded: bool | None = None,
    ) -> FusionDecision:
        """Decode frame, run pipeline off-loop under lock, cache and return result.

        Empty frames (engine warm-up / stub sensor) return the last cached
        decision (or a NONE decision) without running the pipeline.
        """
        if not frame_bytes:
            return self._latest if self._latest is not None else _empty_decision(ts)

        pcm = np.frombuffer(frame_bytes, dtype=np.int16)

        async with self._lock:
            decision = await asyncio.to_thread(
                self._pipeline.process_chunk, pcm, ts, passenger_responded
            )

        self._latest = decision
        return decision

    # -------------------------------------------------------------------------
    # Cache accessors — called by VoiceResponseDetector (no pipeline re-run)
    # -------------------------------------------------------------------------

    @property
    def latest(self) -> FusionDecision | None:
        """Most recent FusionDecision; None until first real frame processed."""
        return self._latest

    @property
    def latest_cue(self) -> CueWordEvent | None:
        if self._latest is None:
            return None
        if time.monotonic() < self._suppress_cues_until:
            return None
        return self._latest.cue_word

    @property
    def latest_breathing(self) -> BreathingEstimate | None:
        return self._latest.breathing if self._latest is not None else None

    def suppress_cues(self, duration_s: float) -> None:
        """Suppress cue-word detection for the next *duration_s* seconds.

        Used by the live demo to prevent the microphone from picking up the
        spoken voice-check-in prompt (which itself contains "okay"/"fine")
        and falsely triggering de-escalation.
        """
        self._suppress_cues_until = time.monotonic() + duration_s
