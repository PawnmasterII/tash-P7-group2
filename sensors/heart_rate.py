from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading


class HeartRateSensor(Sensor):
    """Stub. Replace with the contact (PPG/ECG) or radar HR driver."""

    modality = Modality.HEART_RATE

    def __init__(self, period_s: float = 1.0) -> None:
        self._period_s = period_s

    async def stream(self) -> AsyncIterator[SensorReading]:
        while True:
            await asyncio.sleep(self._period_s)
            yield SensorReading(
                modality=self.modality,
                payload={"bpm": random.uniform(60, 90)},
            )
