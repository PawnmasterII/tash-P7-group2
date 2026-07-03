"""TASH full multi-modal demo — 4 scripted scenarios + optional audio.

Scenarios:
  1. Slump + no response   → CHECK_IN → 10 s silence → ELEVATED
  2. Slump + "help"        → CHECK_IN → distress cue while armed → ELEVATED
  3. Slump + "okay"        → CHECK_IN → passenger reassures → de-escalated
  4. Cardiac spike         → ELEVATED immediately (HR > 130 bpm)
  5. Audio WAV replay      → agonal breathing → ELEVATED  [if TASHaudio present]

Usage (from Desktop/code/ — NOT from inside the repo):
    py -3.12 -m tash.demo
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from tash.audio.contracts import CueWordEvent, EscalationLevel, FusionDecision
from tash.comms.base import Notifier
from tash.core.event_bus import EventBus
from tash.detectors.agonal_breathing import AgonalBreathingDetector
from tash.detectors.cardiac import CardiacAnomalyDetector
from tash.detectors.slump import SlumpDetector
from tash.detectors.voice_response import VoiceResponseDetector
from tash.fusion.risk_engine import RiskEngine
from tash.response.actions import Action
from tash.response.state_machine import ResponseOrchestrator
from tash.sensors.base import Sensor
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading, TripContext
from tash.vehicle.base import VehicleController

# ── Console colours (Windows 10+ supports ANSI natively) ─────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"

_t0: float = 0.0


def _ts() -> str:
    return f"{CYAN}[T+{time.monotonic() - _t0:5.1f}s]{RESET}"


def _header(title: str, desc: str) -> None:
    bar = "-" * 62
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"  {BOLD}{WHITE}{title}{RESET}")
    print(f"  {DIM}{desc}{RESET}")
    print(f"{BOLD}{bar}{RESET}")


def _action(msg: str) -> None:
    print(f"  {RED}>>{RESET} {_ts()} {BOLD}{msg}{RESET}")


def _event(msg: str) -> None:
    print(f"  {YELLOW}**{RESET} {_ts()} {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET} {_ts()} {msg}")


def _dim_print(msg: str) -> None:
    print(f"  {DIM}{_ts()} {msg}{RESET}")


# ── Mock AudioEngine (no real pipeline) ──────────────────────────────────────

class DemoAudioEngine:
    """Lightweight mock: no Vosk, no denoising.
    Cues can be injected externally to simulate voice detections."""

    def __init__(self) -> None:
        self._cue: CueWordEvent | None = None

    async def start(self) -> None:
        pass

    async def process_frame(
        self, frame_bytes: bytes, ts: float, passenger_responded: bool | None = None
    ) -> FusionDecision:
        return FusionDecision(
            ts=ts, level=EscalationLevel.NONE, reasons=[], cue_word=None, breathing=None
        )

    @property
    def latest_cue(self) -> CueWordEvent | None:
        return self._cue

    def inject_cue(self, keyword: str, category: str, ts: float | None = None) -> None:
        self._cue = CueWordEvent(
            ts=ts if ts is not None else time.monotonic(),
            keyword=keyword,
            vad_active=True,
            sensitivity=0.5,
            confidence_proxy=0.92,
            category=category,
        )

    def clear_cue(self) -> None:
        self._cue = None


# ── Scripted sensors ──────────────────────────────────────────────────────────

class ScriptedPostureSensor(Sensor):
    """Emits one slump at `at_s` seconds then stays normal."""
    modality = Modality.POSTURE

    def __init__(self, angle_deg: float = 50.0, at_s: float = 0.5) -> None:
        self._angle = angle_deg
        self._at_s = at_s

    async def stream(self) -> AsyncIterator[SensorReading]:
        await asyncio.sleep(self._at_s)
        _dim_print(f"PostureSensor → slump_angle_deg={self._angle}°")
        yield SensorReading(modality=self.modality, payload={"slump_angle_deg": self._angle})
        while True:
            await asyncio.sleep(5.0)
            yield SensorReading(modality=self.modality, payload={"slump_angle_deg": 5.0})


class ScriptedHeartRateSensor(Sensor):
    """Emits (delay_s, bpm) pairs then holds a normal rate."""
    modality = Modality.HEART_RATE

    def __init__(self, events: list[tuple[float, float]]) -> None:
        self._events = events

    async def stream(self) -> AsyncIterator[SensorReading]:
        for delay_s, bpm in self._events:
            await asyncio.sleep(delay_s)
            _dim_print(f"HeartRateSensor → bpm={bpm}")
            yield SensorReading(modality=self.modality, payload={"bpm": bpm})
        while True:
            await asyncio.sleep(2.0)
            yield SensorReading(modality=self.modality, payload={"bpm": 72.0})


class SilentMicrophone(Sensor):
    """Ticks the VoiceResponseDetector's timeout check every 100 ms.

    Sends empty frame_bytes so AudioEngine skips real processing but
    VoiceResponseDetector still gets called on each reading.
    """
    modality = Modality.MICROPHONE
    _TICK = 0.1

    async def stream(self) -> AsyncIterator[SensorReading]:
        ts = 0.0
        while True:
            await asyncio.sleep(self._TICK)
            yield SensorReading(
                modality=self.modality,
                payload={"frame": b"", "ts": ts},
            )
            ts += self._TICK


# ── Demo vehicle + notifier ───────────────────────────────────────────────────

class DemoVehicle(VehicleController):
    async def speak(self, text: str) -> None:
        _action(f'VEHICLE: "{text}"')

    async def listen(self, timeout_s: float) -> str | None:
        return None

    async def reroute_to(self, dest: tuple[float, float]) -> None:
        _action(f"REROUTE TO HOSPITAL {dest}")

    async def pull_over(self) -> None:
        _action("PULLING OVER — HAZARD LIGHTS ON")

    async def unlock_doors(self) -> None:
        _action("DOORS UNLOCKED")

    async def hazard_lights(self, on: bool) -> None:
        pass


class DemoNotifier(Notifier):
    async def notify(self, tier: RiskTier, events: list[DetectionEvent], msg: str) -> None:
        _action(f'CAREGIVER NOTIFIED  tier={tier.name}  "{msg}"')

    async def open_video_feed(self) -> None:
        _action("VIDEO FEED OPENED TO CAREGIVER")

    async def dispatch_emergency(self) -> None:
        _action("911 DISPATCHED")


# ── Demo orchestrator — signals done when a stop action fires ─────────────────

class DemoOrchestrator(ResponseOrchestrator):
    def __init__(
        self,
        *args,
        done_event: asyncio.Event,
        stop_on: set[Action],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._done = done_event
        self._stop_on = stop_on

    async def _fire(self, action: Action, tier: RiskTier, events: list[DetectionEvent]) -> None:
        await super()._fire(action, tier, events)
        if action in self._stop_on:
            self._done.set()


# ── Scenario runner ───────────────────────────────────────────────────────────

async def run_scenario(
    *,
    title: str,
    description: str,
    sensors: list[Sensor],
    detectors: list,
    engine: DemoAudioEngine,
    stop_on: set[Action],
    timeout_s: float,
    cue_injections: list[tuple[float, str, str]] | None = None,
) -> None:
    """Run one scenario to completion (stop action fires or timeout)."""
    global _t0
    _t0 = time.monotonic()
    _header(title, description)

    done = asyncio.Event()
    trip = TripContext(
        trip_id="demo",
        passenger_id="passenger-demo",
        nearest_hospital=(37.7749, -122.4194),
    )

    voice_det = next((d for d in detectors if isinstance(d, VoiceResponseDetector)), None)

    orchestrator = DemoOrchestrator(
        vehicle=DemoVehicle(),
        notifier=DemoNotifier(),
        trip=trip,
        on_check_in=voice_det.arm if voice_det else None,
        done_event=done,
        stop_on=stop_on,
    )
    risk_engine = RiskEngine()

    reading_bus: EventBus[SensorReading] = EventBus()
    detection_bus: EventBus[DetectionEvent] = EventBus()
    reading_q = reading_bus.subscribe()
    detection_q = detection_bus.subscribe()

    det_map: dict[Modality, list] = {}
    for d in detectors:
        det_map.setdefault(d.modality, []).append(d)

    async def _sensor_loop(s: Sensor) -> None:
        async for reading in s.stream():
            await reading_bus.publish(reading)

    async def _detector_loop() -> None:
        while True:
            reading = await reading_q.get()
            for det in det_map.get(reading.modality, ()):
                ev = await det.observe(reading)
                if ev is not None:
                    _event(
                        f"{det.name}: label={ev.label!r}  "
                        f"tier={ev.risk_tier.name}"
                    )
                    await detection_bus.publish(ev)

    async def _fusion_loop() -> None:
        recent: deque[DetectionEvent] = deque(maxlen=32)
        while True:
            ev = await detection_q.get()
            recent.append(ev)
            tier = risk_engine.ingest(ev)
            await orchestrator.handle(tier, list(recent))

    async def _cue_task() -> None:
        if not cue_injections:
            return
        t_start = time.monotonic()
        for at_s, keyword, category in sorted(cue_injections):
            wait = at_s - (time.monotonic() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)
            _dim_print(f"[inject cue] keyword={keyword!r}  category={category}")
            engine.inject_cue(keyword, category)

    async def _watchdog() -> None:
        await asyncio.sleep(timeout_s)
        done.set()

    tasks = [
        asyncio.create_task(_sensor_loop(s)) for s in sensors
    ] + [
        asyncio.create_task(_detector_loop()),
        asyncio.create_task(_fusion_loop()),
        asyncio.create_task(_cue_task()),
        asyncio.create_task(_watchdog()),
    ]

    await done.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - _t0
    _ok(f"Scenario finished in {elapsed:.1f}s")


# ── Optional: audio WAV scenario ─────────────────────────────────────────────

async def run_audio_scenario() -> None:
    """Full audio pipeline on agonal_gasps.wav.  Requires TASHaudio sibling repo."""
    wav_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "TASHaudio", "test_audio")
    )
    if not os.path.isdir(wav_dir) or not any(
        f.endswith(".wav") for f in os.listdir(wav_dir)
    ):
        print(f"\n  {DIM}Skipping audio scenario — "
              f"no WAV files found at {wav_dir}{RESET}")
        return

    from tash.audio.engine import AudioEngine
    from tash.sensors.microphone import Microphone

    _header(
        "Scenario 5 — Audio: Agonal Breathing (WAV replay)",
        "Replays agonal_gasps.wav through the full Vosk/denoise/breathing pipeline.",
    )

    print(f"  {DIM}Loading Vosk model … (first run may take a moment){RESET}")
    engine = AudioEngine()
    await engine.start()
    print(f"  {DIM}AudioEngine ready  degraded={engine._pipeline.degraded}{RESET}")

    global _t0
    _t0 = time.monotonic()

    agonal_det = AgonalBreathingDetector(engine)
    voice_det  = VoiceResponseDetector(engine, response_window_s=30.0)

    done = asyncio.Event()
    trip = TripContext(
        trip_id="demo-audio",
        passenger_id="passenger-demo",
        nearest_hospital=(37.7749, -122.4194),
    )
    orchestrator = DemoOrchestrator(
        vehicle=DemoVehicle(),
        notifier=DemoNotifier(),
        trip=trip,
        on_check_in=voice_det.arm,
        done_event=done,
        stop_on={Action.REROUTE_HOSPITAL},
    )
    risk_engine = RiskEngine()

    reading_bus: EventBus[SensorReading] = EventBus()
    detection_bus: EventBus[DetectionEvent] = EventBus()
    reading_q = reading_bus.subscribe()
    detection_q = detection_bus.subscribe()

    detectors = [agonal_det, voice_det]
    det_map = {Modality.MICROPHONE: detectors}

    mic = Microphone(mode="wav", wav_dir=wav_dir, loop=False)

    async def _sensor_loop() -> None:
        async for r in mic.stream():
            await reading_bus.publish(r)
        done.set()  # WAV finished

    async def _detector_loop() -> None:
        while True:
            reading = await reading_q.get()
            for det in det_map.get(reading.modality, ()):
                ev = await det.observe(reading)
                if ev is not None:
                    _event(f"{det.name}: label={ev.label!r}  tier={ev.risk_tier.name}")
                    await detection_bus.publish(ev)

    async def _fusion_loop() -> None:
        recent: deque[DetectionEvent] = deque(maxlen=32)
        while True:
            ev = await detection_q.get()
            recent.append(ev)
            tier = risk_engine.ingest(ev)
            await orchestrator.handle(tier, list(recent))

    async def _watchdog() -> None:
        await asyncio.sleep(180.0)  # 3 min hard cap
        done.set()

    tasks = [
        asyncio.create_task(_sensor_loop()),
        asyncio.create_task(_detector_loop()),
        asyncio.create_task(_fusion_loop()),
        asyncio.create_task(_watchdog()),
    ]
    await done.wait()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - _t0
    _ok(f"Audio scenario finished in {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

DEMO_WINDOW_S = 10.0   # Voice response window for scenarios 1–3 (production = 30 s)


async def _amain() -> None:
    import logging
    # Ensure Unicode output works on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.CRITICAL)  # silence library noise during demo

    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"  {BOLD}{WHITE}TASH -- In-Cabin Safety Hub  |  Multi-Modal Demo{RESET}")
    print(f"  {DIM}Python {sys.version.split()[0]}  |  "
          f"4 scripted scenarios  |  ~90 s total{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}\n")
    print(f"  Response window for demo scenarios: {DEMO_WINDOW_S}s  "
          f"{DIM}(production: 30 s){RESET}")

    # ── Scenario 1: Slump + no response ──────────────────────────────────────
    engine1 = DemoAudioEngine()
    slump_det1 = SlumpDetector()
    voice_det1 = VoiceResponseDetector(engine1, response_window_s=DEMO_WINDOW_S)

    await run_scenario(
        title="Scenario 1 — Slump + No Response",
        description=(
            "Passenger slumps (50°).  Vehicle asks 'Are you okay?'  "
            f"No reply in {DEMO_WINDOW_S:.0f}s → caregiver notified, reroute."
        ),
        sensors=[ScriptedPostureSensor(angle_deg=50.0, at_s=0.5), SilentMicrophone()],
        detectors=[voice_det1, slump_det1],
        engine=engine1,
        stop_on={Action.REROUTE_HOSPITAL},
        timeout_s=DEMO_WINDOW_S + 5,
    )

    await asyncio.sleep(1.5)

    # ── Scenario 2: Slump + distress cue while armed ──────────────────────────
    engine2 = DemoAudioEngine()
    slump_det2 = SlumpDetector()
    voice_det2 = VoiceResponseDetector(engine2, response_window_s=DEMO_WINDOW_S)

    await run_scenario(
        title="Scenario 2 — Slump + Distress Cue",
        description=(
            "Passenger slumps → voice prompt fires → passenger says 'help' "
            "while armed → immediate escalation."
        ),
        sensors=[ScriptedPostureSensor(angle_deg=50.0, at_s=0.5), SilentMicrophone()],
        detectors=[voice_det2, slump_det2],
        engine=engine2,
        stop_on={Action.REROUTE_HOSPITAL},
        timeout_s=12.0,
        cue_injections=[(3.0, "help", "distress")],
    )

    await asyncio.sleep(1.5)

    # ── Scenario 3: Slump + reassurance ──────────────────────────────────────
    engine3 = DemoAudioEngine()
    slump_det3 = SlumpDetector()
    voice_det3 = VoiceResponseDetector(engine3, response_window_s=DEMO_WINDOW_S)

    print(f"\n  {DIM}(In this scenario no ELEVATED actions should fire.){RESET}")
    await run_scenario(
        title="Scenario 3 — Slump + Reassurance",
        description=(
            "Passenger slumps → voice prompt fires → passenger says 'okay' "
            "→ de-escalated, no caregiver notification."
        ),
        sensors=[ScriptedPostureSensor(angle_deg=50.0, at_s=0.5), SilentMicrophone()],
        detectors=[voice_det3, slump_det3],
        engine=engine3,
        stop_on=set(),  # no terminal action expected — run to timeout
        timeout_s=7.0,
        cue_injections=[(3.0, "okay", "reassurance")],
    )

    await asyncio.sleep(1.5)

    # ── Scenario 4: Cardiac anomaly ───────────────────────────────────────────
    await run_scenario(
        title="Scenario 4 — Cardiac Anomaly",
        description=(
            "Heart rate spikes to 155 bpm (tachycardia threshold: 130 bpm) "
            "→ ELEVATED immediately, no voice gate required."
        ),
        sensors=[ScriptedHeartRateSensor(events=[(1.5, 155.0)])],
        detectors=[CardiacAnomalyDetector()],
        engine=DemoAudioEngine(),
        stop_on={Action.REROUTE_HOSPITAL},
        timeout_s=8.0,
    )

    await asyncio.sleep(1.5)

    # ── Scenario 5 (optional): Audio WAV ─────────────────────────────────────
    await run_audio_scenario()

    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"  {GREEN}{BOLD}All scenarios complete.{RESET}")
    print(f"  {DIM}Run  py -3.12 -m tash.main  for the live audio WAV demo.{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}\n")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
