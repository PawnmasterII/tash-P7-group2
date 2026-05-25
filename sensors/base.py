from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from tash.types import Modality, SensorReading


class Sensor(ABC):
    """Hardware-agnostic sensor. Subclasses implement `stream` as an
    `async def` with `yield` statements; real-hardware variants replace
    the mocks one file at a time."""

    modality: Modality

    @abstractmethod
    def stream(self) -> AsyncIterator[SensorReading]: ...

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None
