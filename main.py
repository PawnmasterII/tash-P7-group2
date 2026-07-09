from __future__ import annotations

import asyncio
import logging
import os

from tash.audio.engine import AudioEngine
from tash.comms.mock import MockNotifier
from tash.core.orchestrator import TASHRuntime
from tash.detectors.agonal_breathing import AgonalBreathingDetector
from tash.detectors.cardiac import CardiacAnomalyDetector
from tash.detectors.slump import SlumpDetector
from tash.detectors.voice_response import VoiceResponseDetector
from tash.fusion.risk_engine import RiskEngine
from tash.response.state_machine import ResponseOrchestrator
from tash.sensors.heart_rate import HeartRateSensor
from tash.sensors.microphone import Microphone
from tash.sensors.posture import PostureSensor
from tash.sensors.respiratory import RespiratorySensor
from tash.types import TripContext
from tash.vehicle.mock import MockVehicle

# WAV replay directory for the mock demo. The default resolution in
# Microphone expects a sibling repo named "TASHaudio"; this tries both that
# name and the local "tash-audio-pipeline" layout so the demo runs regardless
# of how the sibling was cloned.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_WAV_CANDIDATES = [
    os.path.join(_REPO_ROOT, "..", "tash-audio-pipeline", "test_audio", "scenarios"),
    os.path.join(_REPO_ROOT, "..", "TASHaudio", "test_audio"),
]
_WAV_DIR = next((d for d in _WAV_CANDIDATES if os.path.isdir(d)), _WAV_CANDIDATES[0])


def build_runtime(engine: AudioEngine) -> TASHRuntime:
    trip = TripContext(
        trip_id="demo-trip",
        passenger_id="demo-passenger",
        nearest_hospital=(37.7749, -122.4194),
    )

    # Shared engine injected into both MICROPHONE detectors so the pipeline
    # runs exactly once per frame (AgonalBreathing drives it; VoiceResponse
    # reads the cache — see detectors/agonal_breathing.py for ordering note).
    agonal_detector = AgonalBreathingDetector(engine)
    voice_detector = VoiceResponseDetector(engine)

    sensors = [
        RespiratorySensor(),
        PostureSensor(),
        HeartRateSensor(),
        # WAV replay by default; pass mode="live" for a real microphone.
        Microphone(mode="wav", wav_dir=_WAV_DIR),
    ]
    detectors = [
        # AgonalBreathingDetector MUST be first among MICROPHONE detectors so it
        # populates the engine cache before VoiceResponseDetector reads it.
        agonal_detector,
        CardiacAnomalyDetector(),
        SlumpDetector(),
        voice_detector,
    ]
    response = ResponseOrchestrator(
        vehicle=MockVehicle(),
        notifier=MockNotifier(),
        trip=trip,
        on_check_in=voice_detector.arm,
    )
    return TASHRuntime(sensors, detectors, RiskEngine(), response)


async def _amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)

    log.info("Loading AudioEngine (Vosk model init) …")
    engine = AudioEngine()
    await engine.start()          # loads Vosk model off the event loop (~1-2 s)
    log.info("AudioEngine ready (degraded=%s)", engine._pipeline.degraded if engine._pipeline else "unknown")

    await build_runtime(engine).run()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
