from __future__ import annotations

from abc import ABC, abstractmethod


class VehicleController(ABC):
    """Abstraction over the autonomous-vehicle command surface. Real
    integration replaces the mock with the Waymo/Cruise/CARLA driver."""

    @abstractmethod
    async def speak(self, text: str) -> None: ...

    @abstractmethod
    async def listen(self, timeout_s: float) -> str | None: ...

    @abstractmethod
    async def reroute_to(self, destination: tuple[float, float]) -> None: ...

    @abstractmethod
    async def pull_over(self) -> None: ...

    @abstractmethod
    async def unlock_doors(self) -> None: ...

    @abstractmethod
    async def hazard_lights(self, on: bool) -> None: ...
