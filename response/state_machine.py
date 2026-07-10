from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tash.comms.base import Notifier
from tash.response.actions import Action
from tash.types import DetectionEvent, RiskTier, TripContext
from tash.vehicle.base import VehicleController

log = logging.getLogger(__name__)

# Action ladder per risk tier. Higher tiers are supersets: re-entering a
# tier is a no-op for actions already fired this trip (see _fired).
LADDER: dict[RiskTier, tuple[Action, ...]] = {
    RiskTier.NORMAL: (),
    RiskTier.WATCH: (),
    RiskTier.CHECK_IN: (Action.VOICE_CHECK_IN,),
    RiskTier.ELEVATED: (
        Action.VOICE_CHECK_IN,
        Action.NOTIFY_CAREGIVER,
        Action.OPEN_VIDEO_FEED,
        Action.REROUTE_HOSPITAL,
    ),
    RiskTier.CRITICAL: (
        Action.VOICE_CHECK_IN,
        Action.NOTIFY_CAREGIVER,
        Action.OPEN_VIDEO_FEED,
        Action.PULL_OVER,
        Action.UNLOCK_DOORS,
        Action.DISPATCH_911,
    ),
}


@dataclass
class ResponseOrchestrator:
    vehicle: VehicleController
    notifier: Notifier
    trip: TripContext
    on_check_in: Callable[[], Awaitable[None]] | None = None
    _fired: set[Action] = field(default_factory=set)
    _awaiting_checkin: bool = field(default=False, init=False)

    async def handle(self, tier: RiskTier, events: list[DetectionEvent]) -> None:
        # While a voice check-in is pending, hold off on further escalation
        # so the passenger has a chance to respond and de-escalate.
        # Exception: CRITICAL tier always proceeds; timeout/help events end the hold.
        if self._awaiting_checkin:
            # Check if the LATEST event resolves the check-in (timeout or distress).
            # Only inspect the last event to avoid stale labels from previous cycles.
            _RESOLVED_LABELS = ("no_response_timeout", "help_while_awaiting")
            latest = events[-1] if events else None
            if latest is not None and latest.label in _RESOLVED_LABELS:
                self._awaiting_checkin = False
            elif tier < RiskTier.CRITICAL:
                return
        for action in LADDER.get(tier, ()):
            if action in self._fired:
                continue
            await self._fire(action, tier, events)
            self._fired.add(action)
            # After firing a check-in, stop and wait for passenger response
            # before firing any further actions in the ladder.
            if action == Action.VOICE_CHECK_IN:
                break

    def reset(self) -> None:
        self._fired.clear()
        self._awaiting_checkin = False

    @property
    def fired_actions(self) -> frozenset[Action]:
        return frozenset(self._fired)

    async def _fire(
        self,
        action: Action,
        tier: RiskTier,
        events: list[DetectionEvent],
    ) -> None:
        log.info("[response] firing %s (tier=%s)", action.value, tier.name)
        match action:
            case Action.VOICE_CHECK_IN:
                self._awaiting_checkin = True
                await self.vehicle.speak(
                    "Are you okay? Say 'fine' or 'okay' if you're okay."
                )
                if self.on_check_in is not None:
                    await self.on_check_in()
            case Action.NOTIFY_CAREGIVER:
                await self.notifier.notify(tier, events, "Possible distress detected.")
            case Action.OPEN_VIDEO_FEED:
                await self.notifier.open_video_feed()
            case Action.REROUTE_HOSPITAL:
                if self.trip.nearest_hospital is not None:
                    await self.vehicle.reroute_to(self.trip.nearest_hospital)
            case Action.PULL_OVER:
                await self.vehicle.pull_over()
                await self.vehicle.hazard_lights(True)
            case Action.UNLOCK_DOORS:
                await self.vehicle.unlock_doors()
            case Action.DISPATCH_911:
                await self.notifier.dispatch_emergency()
