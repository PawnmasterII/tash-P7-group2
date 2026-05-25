from __future__ import annotations

from abc import ABC, abstractmethod

from tash.types import DetectionEvent, Modality, SensorReading


class Detector(ABC):
    """Consumes readings from one modality and (optionally) emits a
    detection event. Stateful detectors (windowed analyses, debouncing)
    keep state on `self`."""

    name: str
    modality: Modality

    @abstractmethod
    async def observe(self, reading: SensorReading) -> DetectionEvent | None: ...
