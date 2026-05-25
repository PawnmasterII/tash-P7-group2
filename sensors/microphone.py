from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading


class Microphone(Sensor):
    """Stub. Replace with the cabin mic-array driver; payload becomes a
    PCM frame plus sample rate so detectors can window on time."""

    modality = Modality.MICROPHONE

    def __init__(self, period_s: float = 0.5, sample_rate: int = 16000) -> None:
        self._period_s = period_s
        self._sample_rate = sample_rate

    async def stream(self) -> AsyncIterator[SensorReading]:
        while True:
            await asyncio.sleep(self._period_s)
            yield SensorReading(
                modality=self.modality,
                payload={"frame": b"", "sample_rate": self._sample_rate},
            )
