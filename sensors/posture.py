from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading


class PostureSensor(Sensor):
    """Stub. Replace with the cabin depth/ToF or vision pipeline."""

    modality = Modality.POSTURE

    def __init__(self, period_s: float = 0.5) -> None:
        self._period_s = period_s

    async def stream(self) -> AsyncIterator[SensorReading]:
        while True:
            await asyncio.sleep(self._period_s)
            yield SensorReading(
                modality=self.modality,
                payload={"slump_angle_deg": random.uniform(0, 15)},
            )
