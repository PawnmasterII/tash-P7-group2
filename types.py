from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any


class Modality(str, Enum):
    RESPIRATORY = "respiratory"
    POSTURE = "posture"
    HEART_RATE = "heart_rate"
    MICROPHONE = "microphone"


class RiskTier(IntEnum):
    NORMAL = 0
    WATCH = 1
    CHECK_IN = 2
    ELEVATED = 3
    CRITICAL = 4


def now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SensorReading:
    modality: Modality
    payload: Any
    timestamp: datetime = field(default_factory=now)


@dataclass(frozen=True)
class DetectionEvent:
    detector: str
    modality: Modality
    label: str
    confidence: float
    risk_tier: RiskTier
    timestamp: datetime = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TripContext:
    trip_id: str
    passenger_id: str
    destination: tuple[float, float] | None = None
    nearest_hospital: tuple[float, float] | None = None
    position: tuple[float, float] | None = None
