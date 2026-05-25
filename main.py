from __future__ import annotations

import asyncio
import logging

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


def build_runtime() -> TASHRuntime:
    trip = TripContext(
        trip_id="demo-trip",
        passenger_id="demo-passenger",
        nearest_hospital=(37.7749, -122.4194),
    )
    sensors = [
        RespiratorySensor(),
        PostureSensor(),
        HeartRateSensor(),
        Microphone(),
    ]
    voice_detector = VoiceResponseDetector()
    detectors = [
        AgonalBreathingDetector(),
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
    await build_runtime().run()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
