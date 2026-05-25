from __future__ import annotations

import asyncio
import logging
from collections import deque

from tash.core.event_bus import EventBus
from tash.detectors.base import Detector
from tash.fusion.risk_engine import RiskEngine
from tash.response.state_machine import ResponseOrchestrator
from tash.sensors.base import Sensor
from tash.types import DetectionEvent, Modality, SensorReading

log = logging.getLogger(__name__)


class TASHRuntime:
    """Wires sensors -> detectors -> risk engine -> response. Swap in real
    sensors, vehicle controllers, and notifiers without touching this class."""

    def __init__(
        self,
        sensors: list[Sensor],
        detectors: list[Detector],
        risk_engine: RiskEngine,
        response: ResponseOrchestrator,
    ) -> None:
        self._sensors = sensors
        self._detectors_by_modality: dict[Modality, list[Detector]] = {}
        for d in detectors:
            self._detectors_by_modality.setdefault(d.modality, []).append(d)
        self._risk = risk_engine
        self._response = response
        self._reading_bus: EventBus[SensorReading] = EventBus()
        self._detection_bus: EventBus[DetectionEvent] = EventBus()
        self._reading_queue = self._reading_bus.subscribe()
        self._detection_queue = self._detection_bus.subscribe()

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self._run_sensor(s), name=f"sensor:{s.modality.value}")
            for s in self._sensors
        ]
        tasks.append(asyncio.create_task(self._run_detectors(), name="detectors"))
        tasks.append(asyncio.create_task(self._run_fusion(), name="fusion"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()

    async def _run_sensor(self, sensor: Sensor) -> None:
        await sensor.start()
        try:
            async for reading in sensor.stream():
                await self._reading_bus.publish(reading)
        finally:
            await sensor.stop()

    async def _run_detectors(self) -> None:
        while True:
            reading = await self._reading_queue.get()
            for detector in self._detectors_by_modality.get(reading.modality, ()):
                event = await detector.observe(reading)
                if event is not None:
                    await self._detection_bus.publish(event)

    async def _run_fusion(self) -> None:
        recent: deque[DetectionEvent] = deque(maxlen=32)
        while True:
            event = await self._detection_queue.get()
            recent.append(event)
            tier = self._risk.ingest(event)
            log.debug("event=%s tier=%s", event.label, tier.name)
            await self._response.handle(tier, list(recent))
