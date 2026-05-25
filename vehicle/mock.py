from __future__ import annotations

import logging

from tash.vehicle.base import VehicleController

log = logging.getLogger(__name__)


class MockVehicle(VehicleController):
    async def speak(self, text: str) -> None:
        log.info("[vehicle] speak: %s", text)

    async def listen(self, timeout_s: float) -> str | None:
        log.info("[vehicle] listen (%.1fs)", timeout_s)
        return None

    async def reroute_to(self, destination: tuple[float, float]) -> None:
        log.info("[vehicle] reroute -> %s", destination)

    async def pull_over(self) -> None:
        log.info("[vehicle] pulling over")

    async def unlock_doors(self) -> None:
        log.info("[vehicle] doors unlocked")

    async def hazard_lights(self, on: bool) -> None:
        log.info("[vehicle] hazards %s", "ON" if on else "OFF")
