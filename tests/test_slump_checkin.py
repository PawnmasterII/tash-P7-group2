"""Tests for the slump → CHECK_IN → voice-response flow.

Verifies all three outcome paths:
  1. Passenger reassures ("fine"/"okay") → de-escalate, no further action.
  2. Passenger says "help" while armed → escalate to ELEVATED.
  3. No response within window → timeout escalates to ELEVATED.

Also covers:
  - SlumpDetector: severe angle maps to CHECK_IN not ELEVATED.
  - VoiceResponseDetector: unarmed "help" still yields CHECK_IN.
  - ResponseOrchestrator: voice prompt fires exactly once per CHECK_IN.

Run:  python -m pytest tests/test_slump_checkin.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tash.audio.contracts import CueWordEvent
from tash.detectors.slump import SLUMP_ALERT_DEG, SLUMP_WATCH_DEG, SlumpDetector
from tash.detectors.voice_response import VoiceResponseDetector, _RESPONSE_WINDOW_S
from tash.response.actions import Action
from tash.response.state_machine import ResponseOrchestrator
from tash.types import Modality, RiskTier, SensorReading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _slump_reading(angle_deg: float) -> SensorReading:
    return SensorReading(modality=Modality.POSTURE, payload={"slump_angle_deg": angle_deg})


def _mic_reading() -> SensorReading:
    return SensorReading(modality=Modality.MICROPHONE, payload={"frame": b"", "ts": 0.0})


def _cue(keyword: str, category: str, ts: float = 1.0) -> CueWordEvent:
    return CueWordEvent(
        ts=ts,
        keyword=keyword,
        vad_active=True,
        sensitivity=0.5,
        confidence_proxy=0.8,
        category=category,
    )


def _make_voice_detector(latest_cue: CueWordEvent | None = None) -> VoiceResponseDetector:
    engine = MagicMock()
    engine.latest_cue = latest_cue
    return VoiceResponseDetector(engine)


def _make_orchestrator(vehicle, notifier, trip, on_check_in=None) -> ResponseOrchestrator:
    return ResponseOrchestrator(
        vehicle=vehicle, notifier=notifier, trip=trip, on_check_in=on_check_in
    )


# ---------------------------------------------------------------------------
# SlumpDetector tier mapping
# ---------------------------------------------------------------------------

class TestSlumpDetector:
    def test_below_watch_returns_none(self):
        det = SlumpDetector()
        assert _run(det.observe(_slump_reading(10.0))) is None

    def test_watch_angle_emits_watch(self):
        det = SlumpDetector()
        event = _run(det.observe(_slump_reading(SLUMP_WATCH_DEG)))
        assert event is not None
        assert event.risk_tier == RiskTier.WATCH

    def test_alert_angle_emits_check_in_not_elevated(self):
        """Severe slump must go through check-in gate before dispatching."""
        det = SlumpDetector()
        event = _run(det.observe(_slump_reading(SLUMP_ALERT_DEG)))
        assert event is not None
        assert event.risk_tier == RiskTier.CHECK_IN, (
            f"Expected CHECK_IN, got {event.risk_tier} — slump must not auto-dispatch"
        )

    def test_severe_angle_emits_check_in(self):
        det = SlumpDetector()
        event = _run(det.observe(_slump_reading(80.0)))
        assert event is not None
        assert event.risk_tier == RiskTier.CHECK_IN


# ---------------------------------------------------------------------------
# VoiceResponseDetector: unarmed paths
# ---------------------------------------------------------------------------

class TestVoiceResponseDetectorUnarmed:
    def test_no_cue_returns_none(self):
        det = _make_voice_detector(latest_cue=None)
        assert _run(det.observe(_mic_reading())) is None

    def test_help_unarmed_yields_check_in(self):
        det = _make_voice_detector(latest_cue=_cue("help", "distress"))
        event = _run(det.observe(_mic_reading()))
        assert event is not None
        assert event.risk_tier == RiskTier.CHECK_IN
        assert event.label == "help_distress_cue"

    def test_reassurance_unarmed_returns_none(self):
        """Reassurance words while unarmed should be silently ignored."""
        det = _make_voice_detector(latest_cue=_cue("fine", "reassurance"))
        assert _run(det.observe(_mic_reading())) is None

    def test_deduplication(self):
        cue = _cue("help", "distress", ts=5.0)
        det = _make_voice_detector(latest_cue=cue)
        first = _run(det.observe(_mic_reading()))
        second = _run(det.observe(_mic_reading()))
        assert first is not None
        assert second is None, "Same-ts cue must not fire twice"


# ---------------------------------------------------------------------------
# VoiceResponseDetector: armed paths (the three outcome cases)
# ---------------------------------------------------------------------------

class TestVoiceResponseDetectorArmed:
    def test_reassurance_deescalates(self):
        """Path 1: passenger says 'okay' → disarm, no event."""
        async def _scenario():
            cue = _cue("okay", "reassurance", ts=2.0)
            det = _make_voice_detector(latest_cue=cue)
            await det.arm()
            assert det._awaiting is True
            event = await det.observe(_mic_reading())
            assert event is None, "Reassurance while armed must return None"
            assert det._awaiting is False, "Must disarm after reassurance"
        _run(_scenario())

    def test_fine_also_deescalates(self):
        async def _scenario():
            det = _make_voice_detector(latest_cue=_cue("fine", "reassurance", ts=3.0))
            await det.arm()
            event = await det.observe(_mic_reading())
            assert event is None
            assert det._awaiting is False
        _run(_scenario())

    def test_help_while_armed_escalates(self):
        """Path 2: passenger says 'help' while armed → ELEVATED."""
        async def _scenario():
            det = _make_voice_detector(latest_cue=_cue("help", "distress", ts=4.0))
            await det.arm()
            event = await det.observe(_mic_reading())
            assert event is not None
            assert event.risk_tier == RiskTier.ELEVATED
            assert event.label == "help_while_awaiting"
            assert det._awaiting is False
        _run(_scenario())

    def test_timeout_escalates(self):
        """Path 3: no response within window → ELEVATED 'no_response_timeout'."""
        async def _scenario():
            det = _make_voice_detector(latest_cue=None)
            await det.arm()
            arm_t = det._arm_monotonic
            with patch("tash.detectors.voice_response.time") as mock_time:
                mock_time.monotonic.return_value = arm_t + _RESPONSE_WINDOW_S + 1.0
                event = await det.observe(_mic_reading())
            assert event is not None
            assert event.risk_tier == RiskTier.ELEVATED
            assert event.label == "no_response_timeout"
            assert det._awaiting is False
        _run(_scenario())

    def test_no_timeout_before_window(self):
        """Armed but within window with no cue → no event yet."""
        async def _scenario():
            det = _make_voice_detector(latest_cue=None)
            await det.arm()
            arm_t = det._arm_monotonic
            with patch("tash.detectors.voice_response.time") as mock_time:
                mock_time.monotonic.return_value = arm_t + 5.0
                event = await det.observe(_mic_reading())
            assert event is None
            assert det._awaiting is True, "Must remain armed within window"
        _run(_scenario())


# ---------------------------------------------------------------------------
# ResponseOrchestrator: CHECK_IN fires voice prompt and arms detector
# ---------------------------------------------------------------------------

class TestResponseOrchestratorCheckIn:
    def test_check_in_tier_fires_voice_prompt_once(self):
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            on_check_in = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = None
            orchestrator = _make_orchestrator(vehicle, notifier, trip, on_check_in)
            await orchestrator.handle(RiskTier.CHECK_IN, [])
            vehicle.speak.assert_called_once()
            spoken = vehicle.speak.call_args[0][0]
            assert "okay" in spoken.lower() or "fine" in spoken.lower(), (
                f"Prompt should mention reassurance words, got: {spoken!r}"
            )
            on_check_in.assert_called_once()
        _run(_scenario())

    def test_check_in_fires_only_once_per_trip(self):
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = None
            orchestrator = _make_orchestrator(vehicle, notifier, trip)
            await orchestrator.handle(RiskTier.CHECK_IN, [])
            await orchestrator.handle(RiskTier.CHECK_IN, [])
            assert vehicle.speak.call_count == 1, "VOICE_CHECK_IN must not fire twice"
        _run(_scenario())

    def test_elevated_fires_downstream_when_check_in_already_done(self):
        """ELEVATED after CHECK_IN: VOICE_CHECK_IN is skipped (already fired),
        caregiver/video/reroute are new → must fire."""
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = (37.0, -122.0)
            orchestrator = _make_orchestrator(vehicle, notifier, trip)
            await orchestrator.handle(RiskTier.CHECK_IN, [])
            await orchestrator.handle(RiskTier.ELEVATED, [])
            assert Action.VOICE_CHECK_IN in orchestrator.fired_actions
            assert Action.NOTIFY_CAREGIVER in orchestrator.fired_actions
            assert Action.REROUTE_HOSPITAL in orchestrator.fired_actions
            assert vehicle.speak.call_count == 1  # once, not again at ELEVATED
        _run(_scenario())


# ---------------------------------------------------------------------------
# End-to-end synthetic scenarios
# ---------------------------------------------------------------------------

class TestEndToEndSlumpFlow:
    def test_slump_then_reassurance_no_escalation(self):
        """Slump → CHECK_IN → passenger says 'okay' → no caregiver dispatch."""
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = (37.0, -122.0)
            det = _make_voice_detector(latest_cue=None)
            orchestrator = _make_orchestrator(vehicle, notifier, trip, on_check_in=det.arm)

            slump_det = SlumpDetector()
            slump_event = await slump_det.observe(_slump_reading(50.0))
            assert slump_event is not None
            assert slump_event.risk_tier == RiskTier.CHECK_IN

            await orchestrator.handle(RiskTier.CHECK_IN, [slump_event])
            vehicle.speak.assert_called_once()
            assert det._awaiting is True

            det._engine.latest_cue = _cue("okay", "reassurance", ts=10.0)
            result = await det.observe(_mic_reading())
            assert result is None
            assert det._awaiting is False

            assert Action.NOTIFY_CAREGIVER not in orchestrator.fired_actions
            assert Action.REROUTE_HOSPITAL not in orchestrator.fired_actions
        _run(_scenario())

    def test_slump_then_help_escalates(self):
        """Slump → CHECK_IN → passenger says 'help' → ELEVATED actions fire."""
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = (37.0, -122.0)
            det = _make_voice_detector(latest_cue=None)
            orchestrator = _make_orchestrator(vehicle, notifier, trip, on_check_in=det.arm)

            await orchestrator.handle(RiskTier.CHECK_IN, [])
            assert det._awaiting is True

            det._engine.latest_cue = _cue("help", "distress", ts=15.0)
            event = await det.observe(_mic_reading())
            assert event is not None
            assert event.risk_tier == RiskTier.ELEVATED

            await orchestrator.handle(RiskTier.ELEVATED, [event])
            assert Action.NOTIFY_CAREGIVER in orchestrator.fired_actions
            notifier.notify.assert_called_once()
        _run(_scenario())

    def test_slump_then_timeout_escalates(self):
        """Slump → CHECK_IN → silence for >30 s → ELEVATED."""
        async def _scenario():
            vehicle = AsyncMock()
            notifier = AsyncMock()
            trip = MagicMock()
            trip.nearest_hospital = (37.0, -122.0)
            det = _make_voice_detector(latest_cue=None)
            orchestrator = _make_orchestrator(vehicle, notifier, trip, on_check_in=det.arm)

            await orchestrator.handle(RiskTier.CHECK_IN, [])
            assert det._awaiting is True
            arm_t = det._arm_monotonic

            with patch("tash.detectors.voice_response.time") as mock_time:
                mock_time.monotonic.return_value = arm_t + _RESPONSE_WINDOW_S + 1.0
                event = await det.observe(_mic_reading())

            assert event is not None
            assert event.risk_tier == RiskTier.ELEVATED
            assert event.label == "no_response_timeout"

            await orchestrator.handle(RiskTier.ELEVATED, [event])
            assert Action.NOTIFY_CAREGIVER in orchestrator.fired_actions
            assert Action.REROUTE_HOSPITAL in orchestrator.fired_actions
        _run(_scenario())
