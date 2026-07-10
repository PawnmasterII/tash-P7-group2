"""TASH live demo — real webcam + real microphone.

Pipeline:
  Webcam  → WebcamPostureSensor (MediaPipe Pose) → SlumpDetector
  Mic     → AgonalBreathingDetector (Vosk/AudioEngine)
          → VoiceResponseDetector (armed by ResponseOrchestrator)
  → RiskEngine (30 s window) → LiveOrchestrator → terminal output

Threading model (required for OpenCV GUI on Windows):
  Main thread  : OpenCV display loop (cv2.imshow must be called here on Windows)
  Daemon thread: asyncio event loop — sensors, detectors, fusion, response

Install prerequisites (run from Desktop\\code\\ — the "./" prefix matters,
otherwise pip parses "tash-P7-group2[live]" as a package name, not a path):
  py -3.12 -m pip install -e "./tash-P7-group2[live]"   # mediapipe, opencv, pvrecorder, pyttsx3
  py -3.12 -m tash.audio.download_model                  # Vosk ~40 MB, once

Run (from Desktop\\code\\, NOT from inside the repo):
  py -3.12 -m tash.live

Voice check-in prompts are spoken aloud via pyttsx3 (offline TTS, SAPI5 on
Windows) if installed; otherwise they fall back to text-only (printed/logged).
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from tash.audio.engine import AudioEngine
from tash.comms.base import Notifier
from tash.core.event_bus import EventBus
from tash.detectors.agonal_breathing import AgonalBreathingDetector
from tash.detectors.slump import SlumpDetector
from tash.detectors.voice_response import VoiceResponseDetector
from tash.fusion.risk_engine import RiskEngine
from tash.response.actions import Action
from tash.response.state_machine import ResponseOrchestrator
from tash.sensors.microphone import Microphone
from tash.sensors.webcam import WebcamPostureSensor
from tash.types import DetectionEvent, Modality, RiskTier, SensorReading, TripContext
from tash.vehicle.base import VehicleController

# ── ANSI colours ─────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
MAGENTA = "\033[35m"
WHITE  = "\033[97m"

_t0: float = 0.0


def _ts() -> str:
    return f"{CYAN}[T+{time.monotonic() - _t0:6.1f}s]{RESET}"


def _action(msg: str) -> None:
    print(f"  {RED}>>{RESET} {_ts()} {BOLD}{msg}{RESET}", flush=True)


def _event(msg: str) -> None:
    print(f"  {YELLOW}**{RESET} {_ts()} {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"  {CYAN}  {RESET} {_ts()} {msg}", flush=True)


# ── Shared state (written by asyncio thread, read by display thread) ─────────

@dataclass
class LiveState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    tier: RiskTier = RiskTier.NORMAL
    angle: float = 0.0
    last_action: str = ""
    last_event: str = ""
    playing_wav: str = ""
    frame: Any = None        # numpy ndarray, latest annotated BGR frame
    webcam_active: bool = False
    quit: bool = False


# ── Text-to-speech (dedicated worker thread) ──────────────────────────────────
#
# pyttsx3's SAPI5 driver binds to a COM apartment on whatever thread creates
# the engine. asyncio.to_thread() hands out threads from a rotating pool, so
# calling pyttsx3 that way silently breaks after the first utterance. A single
# long-lived thread that owns the engine for the process lifetime avoids that.

class SpeechEngine:
    """Speaks text aloud on a dedicated background thread. No-op if pyttsx3
    is not installed (falls back to text-only check-in prompts)."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._available = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tash-tts")
        self._thread.start()

    def _run(self) -> None:
        log = logging.getLogger(__name__)
        try:
            import pyttsx3
        except ImportError:
            log.warning(
                "pyttsx3 not installed — voice prompts will be text-only. "
                "Install with: pip install pyttsx3"
            )
            while self._queue.get() is not None:
                pass
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception as exc:
            log.warning("TTS engine failed to initialise (%s) — text-only prompts.", exc)
            while self._queue.get() is not None:
                pass
            return

        self._available.set()
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                log.warning("TTS playback failed: %s", exc)

    def speak(self, text: str) -> None:
        self._queue.put(text)

    def stop(self) -> None:
        self._queue.put(None)


# ── Live vehicle + notifier (terminal output + state update) ─────────────────

class LiveVehicle(VehicleController):
    def __init__(self, state: LiveState, speech: SpeechEngine, engine: AudioEngine) -> None:
        self._state = state
        self._speech = speech
        self._engine = engine

    async def speak(self, text: str) -> None:
        msg = f'VEHICLE: "{text}"'
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg
        # Suppress cue detection while TTS plays to stop the mic from
        # picking up the spoken prompt ("okay"/"fine") and falsely
        # de-escalating before the passenger has a chance to respond.
        est_duration = max(2.0, len(text) / 14.0)
        self._engine.suppress_cues(est_duration)
        self._speech.speak(text)

    async def listen(self, timeout_s: float) -> str | None:
        return None

    async def reroute_to(self, dest: tuple[float, float]) -> None:
        msg = f"REROUTE TO HOSPITAL {dest}"
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg

    async def pull_over(self) -> None:
        msg = "PULLING OVER -- HAZARD LIGHTS ON"
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg

    async def unlock_doors(self) -> None:
        msg = "DOORS UNLOCKED"
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg

    async def hazard_lights(self, on: bool) -> None:
        pass


class LiveNotifier(Notifier):
    def __init__(self, state: LiveState) -> None:
        self._state = state

    async def notify(self, tier: RiskTier, events: list[DetectionEvent], msg: str) -> None:
        text = f'CAREGIVER NOTIFIED  tier={tier.name}  "{msg}"'
        _action(text)
        with self._state.lock:
            self._state.last_action = text

    async def open_video_feed(self) -> None:
        msg = "VIDEO FEED OPENED TO CAREGIVER"
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg

    async def dispatch_emergency(self) -> None:
        msg = "911 DISPATCHED"
        _action(msg)
        with self._state.lock:
            self._state.last_action = msg


# ── Live orchestrator — keeps LiveState.tier current on every fusion tick ────

class LiveOrchestrator(ResponseOrchestrator):
    def __init__(self, *args, state: LiveState, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state = state

    async def handle(self, tier: RiskTier, events: list[DetectionEvent]) -> None:
        with self._state.lock:
            self._state.tier = tier
        await super().handle(tier, events)


# ── Test audio playback (speaker loopback for mic testing) ────────────────────

def _resolve_scenario_dir() -> str | None:
    """Find the sibling test-audio directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "tash-audio-pipeline", "test_audio", "scenarios"),
        os.path.join(here, "..", "..", "TASHaudio", "test_audio"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isdir(c):
            return c
    return None


def _play_wav(path: str, state: LiveState) -> None:
    """Play a WAV through the speakers in a daemon thread (non-blocking)."""

    def _worker() -> None:
        log = logging.getLogger(__name__)
        try:
            import sounddevice as sd
            import soundfile as sf

            data, fs = sf.read(path, dtype="float32")

            with state.lock:
                state.playing_wav = os.path.basename(path)
            log.info("Playing test audio: %s", path)
            sd.play(data, fs)
            sd.wait()
        except Exception as exc:
            log.warning("Failed to play %s: %s", path, exc)
        finally:
            with state.lock:
                state.playing_wav = ""

    threading.Thread(target=_worker, daemon=True, name="tash-wav").start()


# ── OpenCV display (runs in main thread) ──────────────────────────────────────

# Tier → BGR colour for the badge
_TIER_BGR: dict[RiskTier, tuple[int, int, int]] = {
    RiskTier.NORMAL:   (0, 200, 0),
    RiskTier.WATCH:    (0, 220, 220),
    RiskTier.CHECK_IN: (0, 140, 255),
    RiskTier.ELEVATED: (0, 0, 220),
    RiskTier.CRITICAL: (0, 0, 120),
}


def _draw_overlay(frame: Any, state: LiveState, cv2: Any) -> Any:
    """Draw risk tier, slump angle, and last action onto the frame."""
    h, w = frame.shape[:2]

    with state.lock:
        tier = state.tier
        angle = state.angle
        last_action = state.last_action
        playing = state.playing_wav

    color = _TIER_BGR.get(tier, (128, 128, 128))

    # Tier badge (top-left)
    tier_text = f"TIER: {tier.name}"
    cv2.rectangle(frame, (8, 8), (260, 50), color, -1)
    cv2.putText(frame, tier_text, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Slump angle (below badge)
    angle_color = (0, 0, 220) if angle >= 45 else (0, 165, 255) if angle >= 25 else (0, 200, 0)
    cv2.putText(frame, f"Slump: {angle:.1f} deg", (14, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, angle_color, 2, cv2.LINE_AA)

    # Last action (bottom strip)
    if last_action:
        strip_y = h - 40
        cv2.rectangle(frame, (0, strip_y), (w, h), (30, 30, 30), -1)
        display = last_action if len(last_action) <= 60 else last_action[:57] + "..."
        cv2.putText(frame, display, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Test audio indicator (below slump angle)
    if playing:
        cv2.rectangle(frame, (8, 85), (380, 112), (0, 100, 180), -1)
        cv2.putText(frame, f">> Playing: {playing}", (14, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Quit hint (top-right)
    cv2.putText(frame, "q = quit", (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def _make_placeholder(cv2: Any, w: int = 800, h: int = 600) -> Any:
    """Black frame shown when no webcam is connected."""
    import numpy as np
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, "No webcam detected", (w // 2 - 160, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, "Audio pipeline active  |  q = quit", (w // 2 - 195, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)
    return frame


def display_loop(state: LiveState) -> None:
    """Blocks the main thread.  Shows annotated webcam frames + overlay."""
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed — display skipped.", flush=True)
        while not state.quit:
            time.sleep(0.1)
        return

    cv2.namedWindow("TASH Live Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TASH Live Demo", 800, 600)
    placeholder = _make_placeholder(cv2)

    while True:
        with state.lock:
            frame = state.frame
            should_quit = state.quit

        if should_quit:
            break

        if frame is not None:
            display = _draw_overlay(frame.copy(), state, cv2)
        else:
            # No webcam frame yet — show placeholder with tier overlay
            display = _draw_overlay(placeholder.copy(), state, cv2)

        cv2.imshow("TASH Live Demo", display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # q, Q, or Esc
            with state.lock:
                state.quit = True
            break

        # Test audio playback: press '1' to play a random WAV from the test data
        if chr(key) == "1":
            import random

            wav_dir = _resolve_scenario_dir()
            if wav_dir is not None:
                wavs = sorted(
                    f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")
                )
                if wavs:
                    picked = random.choice(wavs)
                    _play_wav(os.path.join(wav_dir, picked), state)
            else:
                print(f"  {YELLOW}[test-audio]{RESET} "
                      "Scenario directory not found.", flush=True)

    cv2.destroyAllWindows()


# ── Async runtime (runs in daemon thread) ─────────────────────────────────────

async def async_main(state: LiveState) -> None:
    global _t0
    _t0 = time.monotonic()

    log = logging.getLogger(__name__)
    log.info("Loading AudioEngine (Vosk model) ...")
    engine = AudioEngine()
    await engine.start()
    log.info("AudioEngine ready  degraded=%s", engine._pipeline.degraded if engine._pipeline else "?")

    agonal_det = AgonalBreathingDetector(engine)

    # De-escalation callback: fired when passenger says "okay"/"fine" while
    # armed. Suppresses the slump detector (cooldown), flushes the risk
    # window, and resets the orchestrator so a *future* slump can re-engage.
    async def _on_deescalate() -> None:
        slump_det.dismiss()
        risk_engine.clear()
        orchestrator.reset()
        engine.reset_breathing()
        with state.lock:
            state.tier = RiskTier.NORMAL
            state.last_action = "Passenger reassured — de-escalated"
        _event(f"{GREEN}DE-ESCALATED — passenger reassured{RESET}")

    voice_det  = VoiceResponseDetector(engine, response_window_s=30.0, on_deescalate=_on_deescalate)
    slump_det  = SlumpDetector()

    speech = SpeechEngine()

    trip = TripContext(
        trip_id="live-demo",
        passenger_id="live-passenger",
        nearest_hospital=(37.7749, -122.4194),
    )

    orchestrator = LiveOrchestrator(
        vehicle=LiveVehicle(state, speech, engine),
        notifier=LiveNotifier(state),
        trip=trip,
        on_check_in=voice_det.arm,
        state=state,
    )
    risk_engine = RiskEngine()

    # Sensor setup — webcam is optional; fall back to audio-only if unavailable
    mic_sensor = Microphone(mode="live")
    sensors = [mic_sensor]
    detectors = [agonal_det, voice_det]

    try:
        webcam_sensor = WebcamPostureSensor(state, camera_index=0)
        await webcam_sensor.start()
        sensors.insert(0, webcam_sensor)
        detectors.append(slump_det)
        log.info("Webcam opened — posture detection active.")
        with state.lock:
            state.webcam_active = True
    except Exception as exc:
        log.warning("Webcam unavailable (%s) — running audio-only.", exc)
        print(
            f"  {YELLOW}[webcam]{RESET} No camera found — "
            "posture detection disabled, audio pipeline active.",
            flush=True,
        )

    det_map: dict[Modality, list] = {}
    for d in detectors:
        det_map.setdefault(d.modality, []).append(d)

    reading_bus: EventBus[SensorReading] = EventBus()
    detection_bus: EventBus[DetectionEvent] = EventBus()
    reading_q  = reading_bus.subscribe()
    detection_q = detection_bus.subscribe()

    # Start remaining sensors (webcam already started above if available)
    for s in sensors:
        if not isinstance(s, WebcamPostureSensor):
            await s.start()

    async def _sensor_loop(s):
        try:
            async for reading in s.stream():
                if state.quit:
                    break
                await reading_bus.publish(reading)
        finally:
            await s.stop()

    async def _detector_loop():
        while not state.quit:
            reading = await reading_q.get()
            for det in det_map.get(reading.modality, ()):
                ev = await det.observe(reading)
                if ev is not None:
                    _event(
                        f"{det.name}: label={ev.label!r}  tier={ev.risk_tier.name}"
                    )
                    with state.lock:
                        state.last_event = f"{det.name}: {ev.label}"
                    await detection_bus.publish(ev)

    async def _fusion_loop():
        recent: deque[DetectionEvent] = deque(maxlen=32)
        while not state.quit:
            ev = await detection_q.get()
            recent.append(ev)
            tier = risk_engine.ingest(ev)
            await orchestrator.handle(tier, list(recent))

    async def _quit_watcher():
        while not state.quit:
            await asyncio.sleep(0.1)

    tasks = [
        asyncio.create_task(_sensor_loop(s)) for s in sensors
    ] + [
        asyncio.create_task(_detector_loop()),
        asyncio.create_task(_fusion_loop()),
        asyncio.create_task(_quit_watcher()),
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    speech.stop()
    engine.close()
    log.info("Live demo stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)

    print(f"\n{BOLD}{'=' * 62}{RESET}")
    print(f"  {BOLD}{WHITE}TASH -- Live Demo  |  Webcam + Microphone{RESET}")
    print(f"  {DIM}Webcam: lean forward to trigger posture detection (optional).{RESET}")
    print(f"  {DIM}Mic:    say 'help' for distress cue, 'okay'/'fine' to de-escalate.{RESET}")
    print(f"  {DIM}        Key 1 in video window: play agonal test audio through speakers.{RESET}")
    print(f"  {DIM}Press 'q' in the video window (or Ctrl+C) to quit.{RESET}")
    print(f"{BOLD}{'=' * 62}{RESET}\n")

    state = LiveState()

    # Run asyncio in a daemon thread so the main thread is free for OpenCV GUI.
    loop = asyncio.new_event_loop()

    def _run_async() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_main(state))
        except Exception as exc:
            log.error("async_main error: %s", exc, exc_info=True)
            with state.lock:
                state.quit = True

    async_thread = threading.Thread(target=_run_async, daemon=True, name="tash-async")
    async_thread.start()

    try:
        display_loop(state)       # blocks until user presses q or window closes
    except KeyboardInterrupt:
        pass
    finally:
        with state.lock:
            state.quit = True
        log.info("Waiting for async thread to stop ...")
        async_thread.join(timeout=3.0)
        print(f"\n{GREEN}Demo stopped.{RESET}")


if __name__ == "__main__":
    main()
