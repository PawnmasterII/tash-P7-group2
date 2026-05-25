from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading


class RespiratorySensor(Sensor):
    """Stub. Replace with the chest-strap or mmWave-radar driver when available."""

    modality = Modality.RESPIRATORY

    def __init__(self, period_s: float = 1.0) -> None:
        self._period_s = period_s

    async def stream(self) -> AsyncIterator[SensorReading]:
        while True:
            await asyncio.sleep(self._period_s)
            yield SensorReading(
                modality=self.modality,
                payload={"breaths_per_min": random.uniform(12, 18)},
            )
