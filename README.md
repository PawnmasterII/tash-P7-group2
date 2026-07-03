# TASH — In-Cabin Safety Hub

**TASH** is an in-cabin safety hub for autonomous vehicles. It fuses signals
from multiple sensors to detect medical distress — agonal breathing, cardiac
anomalies, postural slump, voice unresponsiveness — and escalates a graded
response: from a spoken voice check-in, through notifying a caregiver and
rerouting to a hospital, up to pulling over, unlocking doors, and dispatching
emergency services.

> **Status: prototype.** The audio pipeline (agonal-breathing ML model +
> cue-word detection) and the vision pipeline (webcam slump detection) are
> real and working. Heart rate, respiratory rate, the vehicle controller, and
> the caregiver/dispatch notifier are still mocked/stubbed. See
> [Status & known limitations](#status--known-limitations).

## Table of contents

- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running it](#running-it)
  - [Mock demo (no hardware)](#mock-demo-no-hardware)
  - [Live demo (real webcam + mic + voice)](#live-demo-real-webcam--mic--voice)
- [Risk tiers and the response ladder](#risk-tiers-and-the-response-ladder)
- [Testing & validation](#testing--validation)
- [Training / retraining the agonal-breathing model](#training--retraining-the-agonal-breathing-model)
- [Pipeline integration (for contributors adding a sensor/detector)](#pipeline-integration-for-contributors-adding-a-sensordetector)
- [Status & known limitations](#status--known-limitations)

## Architecture

```
sensors  ->  detectors  ->  risk engine  ->  response orchestrator  ->  vehicle / notifier
```

- **`sensors/`** — respiratory, posture, heart-rate, and microphone input
  streams (real or mocked).
- **`detectors/`** — per-modality classifiers that consume `SensorReading`s
  and emit `DetectionEvent`s carrying a `RiskTier`.
- **`fusion/risk_engine.py`** — sliding 30 s window fusion across recent
  events (v1 takes the max tier across the window).
- **`response/state_machine.py`** — risk-tier → action ladder; each action
  fires at most once per trip.
- **`core/orchestrator.py`** — async wiring of the pipeline (`TASHRuntime`).
- **`audio/`** — the vendored TASHaudio signal pipeline: denoise → Vosk
  ASR (cue words) → breathing detector (ML model) → fusion. Wrapped by
  `AudioEngine` and exposed to the runtime via
  `detectors/agonal_breathing.py` and `detectors/voice_response.py`.
- **`vehicle/`, `comms/`** — pluggable controllers/notifiers. Today these are
  `MockVehicle`/`MockNotifier` (demo) or `LiveVehicle`/`LiveNotifier` (live
  demo, terminal + spoken output); real vehicle/dispatch integrations slot in
  without changing the runtime.

Full data contract (schemas, per-pipeline tier mappings, timing, fusion
semantics): see [FUSION_CONTRACT.md](./FUSION_CONTRACT.md).

## Project layout

```
tash-P7-group2/
  audio/            # vendored audio pipeline (denoise, Vosk ASR, breathing ML model, fusion)
    models/         # trained model artifacts (gitignored — not committed)
  comms/            # notifier interface + mock
  core/             # event bus + async runtime (orchestrator.py)
  demo/             # `python -m tash.demo` entry point
  detectors/        # per-modality detectors (slump, cardiac, agonal_breathing, voice_response)
  fusion/           # risk_engine.py — sliding-window tier fusion
  live/             # `python -m tash.live` — real webcam + mic + TTS demo
  response/         # action ladder + state machine
  scripts/          # data-acquisition helpers (download_data.py)
  sensors/          # modality-specific input streams (real + stub)
  tests/            # pytest suite + standalone benchmark/eval scripts
  vehicle/          # vehicle controller interface + mock
  types.py          # shared dataclasses/enums (the cross-pipeline contract)
  main.py           # `python -m tash.main` — mock-sensor demo entry point
  setup.py          # pip install -e . wiring
  requirements.txt  # pinned dependency list
```

## Requirements

- **Python 3.12** (required — `webrtcvad-wheels` has no wheels for 3.13+; the
  pipeline falls back to a coarser energy-VAD without it, but 3.12 is
  recommended).
- Windows, macOS, or Linux. The live demo's OpenCV window and SAPI5
  text-to-speech are exercised primarily on Windows.
- A webcam and microphone are only needed for the **live demo** — the mock
  demo (`python -m tash.main`) needs neither.

## Installation

Clone (or place) the repo so it's importable as the package name `tash` —
`setup.py` maps `package_dir={"tash": "."}`, so you install and run it from
**the parent directory**, not from inside the repo:

```powershell
cd C:\path\to\parent-dir          # e.g. Desktop\code\, NOT inside tash-P7-group2
py -3.12 -m pip install -e "./tash-P7-group2"
```

> The `./` prefix matters — `pip install -e "tash-P7-group2"` (no path
> separator) is parsed as a package *name*, not a local path, and fails with
> `is not a valid editable requirement`.

Install extras as needed:

```powershell
py -3.12 -m pip install -e "./tash-P7-group2[vad]"    # + webrtcvad-wheels (production-grade VAD)
py -3.12 -m pip install -e "./tash-P7-group2[live]"   # + mediapipe, opencv-python, pvrecorder, pyttsx3
```

| Extra  | Adds                                             | Needed for |
|--------|---------------------------------------------------|------------|
| (none) | numpy, scipy, vosk, librosa, soundfile, noisereduce, scikit-learn, joblib | mock demo, WAV replay, unit tests |
| `vad`  | `webrtcvad-wheels`                                 | production-grade voice-activity detection (energy-VAD fallback works without it) |
| `live` | `mediapipe`, `opencv-python`, `pvrecorder`, `pyttsx3` | real webcam posture detection, real microphone capture, spoken voice check-ins |

Download the offline Vosk speech model (~40 MB, one-time; without it the
pipeline runs in degraded mode — no cue-word detection, breathing detection
still works):

```powershell
py -3.12 -m tash.audio.download_model
```

## Running it

### Mock demo (no hardware)

Boots the full runtime — sensors, detectors, fusion, response — with mock
sensors and a mock vehicle/notifier. Detection events and actions are logged
to stdout; nothing touches real hardware.

```powershell
python -m tash.main
```

By default the microphone sensor replays WAV files from
`../../TASHaudio/test_audio/*.wav` (a sibling repo). To point it elsewhere,
edit `Microphone(mode="wav", wav_dir="/path/to/wavs")` in `main.py`, or set
`mode="live"` to use a real microphone (requires `pvrecorder`, see the `live`
extra above).

### Live demo (real webcam + mic + voice)

Runs the real pipeline end-to-end: webcam → MediaPipe pose → slump detector;
microphone → Vosk ASR + ML breathing model → agonal-breathing / voice-response
detectors; fused through the same 30 s risk engine and response ladder as the
mock demo. Voice check-ins ("Are you okay? Say 'fine' or 'okay' if you're
okay.") are spoken aloud via offline TTS (`pyttsx3`, SAPI5 on Windows), not
just logged.

```powershell
py -3.12 -m pip install -e "./tash-P7-group2[live]"
py -3.12 -m tash.audio.download_model
py -3.12 -m tash.live
```

Run from the parent directory (e.g. `Desktop\code\`), not from inside the
repo — same import-path requirement as installation. An OpenCV window shows
the annotated webcam feed with the current risk tier, slump angle, and last
action overlaid; press `q` (or Ctrl+C) to quit. If no webcam is found, it
falls back to audio-only automatically. If `pyttsx3` isn't installed or TTS
fails to initialize, prompts fall back to text-only (logged) with a warning.

## Risk tiers and the response ladder

| Tier      | Actions (cumulative; each fires at most once per trip)                 |
|-----------|--------------------------------------------------------------------------|
| `NORMAL`  | —                                                                         |
| `WATCH`   | —                                                                         |
| `CHECK_IN`| voice check-in ("Are you okay? Say 'fine' or 'okay' if you're okay.")     |
| `ELEVATED`| + notify caregiver, open video feed, reroute to nearest hospital         |
| `CRITICAL`| + pull over, unlock doors, dispatch 911                                 |

**Safety invariant: no single sensor signal auto-dispatches.** A severe slump
(≥ 45°) or a spontaneous "help" enters at `CHECK_IN`, not `ELEVATED` — the
system always asks the passenger first. `VoiceResponseDetector` then drives
escalation from there: reassurance ("fine"/"okay"/"ok") de-escalates and
disarms; "help" while awaiting a response, or 30 s of silence, escalates to
`ELEVATED`. Reaching `CRITICAL` requires a detector that independently
warrants it (or a future corroboration rule — see
[FUSION_CONTRACT.md §5](./FUSION_CONTRACT.md#5-fusion-semantics)).

## Testing & validation

```powershell
# Full unit/integration suite (slump → voice check-in flow: reassurance,
# "help", timeout, tier mapping, orchestrator idempotency — no real audio needed)
py -3.12 -m pytest tests/test_slump_checkin.py -v

# Audio pipeline benchmark: per-stage latency, denoiser suppression (dB),
# cue-word false-positive rate, synthetic breathing classification accuracy
py -3.12 tests/bench_audio_pipeline.py [--quick] [--wav-dir PATH]

# ML agonal-breathing model — out-of-distribution generalization
# (fresh synthetic gasps at multiple SNRs + hard negatives, unseen seed)
py -3.12 tests/test_generalization.py

# ML agonal-breathing model — evaluation against real audio recordings
# (real agonal clips, ESC-50 environmental sounds, ICBHI stethoscope recordings)
py -3.12 tests/test_real_data.py

# Quick smoke test: load the trained model bundle and score a handful of clips
py -3.12 tests/smoke_test_model.py
```

> **Reading the ML model results:** `test_real_data.py`'s positive
> (agonal-breathing) files were all part of the model's training set — high
> scores there confirm memorization, not generalization. The genuinely
> held-out check is `test_generalization.py`'s synthetic-gasp section
> (different RNG seed than training) plus `test_real_data.py`'s ESC-50
> section (~85% of clips unseen during training). There is currently no
> held-out test using *real, unseen* agonal-breathing recordings.

Vision/HR contributors: validate a `Detector` in isolation by feeding it
`SensorReading`s and asserting the emitted `DetectionEvent` matches
[FUSION_CONTRACT.md §3](./FUSION_CONTRACT.md#3-per-pipeline-integration).

## Training / retraining the agonal-breathing model

```powershell
# Optional: pull in more real negative/positive training data
py -3.12 scripts/download_data.py

# Retrain
py -3.12 -m audio.train_agonal_detector \
    --positive-dir data/agonal_real \
    --negative-dir data/icbhi_full \
    --esc50-dir    data/ESC-50-master
```

The trained model bundle (`audio/models/agonal_detector.joblib`) is
gitignored — it's a binary artifact, not source. Regenerate it locally or
pull it from wherever your team distributes trained models.

## Pipeline integration (for contributors adding a sensor/detector)

Three pipelines (audio, vision, HR) plug into this runtime independently
through the shared contract in `types.py`. If you're integrating a new
sensor/detector:

1. Read [FUSION_CONTRACT.md](./FUSION_CONTRACT.md) — the `DetectionEvent` /
   `SensorReading` schema, per-pipeline tier mappings, timing, and fusion
   semantics.
2. Follow [INTEGRATION_CHECKLIST.md](./INTEGRATION_CHECKLIST.md) — phased
   checklist from solo development through full-runtime integration testing.
3. Reference implementations to copy: `detectors/slump.py` (POSTURE) and
   `detectors/cardiac.py` (HEART_RATE) are complete, working examples.
   `detectors/agonal_breathing.py` and `detectors/voice_response.py` show the
   audio pipeline's adapter pattern — see
   [INTEGRATION_NOTES.md](./INTEGRATION_NOTES.md) for how they wrap the
   vendored `audio/` pipeline.

You add a `Sensor` and/or `Detector` and register it in `main.py` — you do
**not** touch the orchestrator, risk engine, or response ladder.

## Status & known limitations

- **Real and working:** audio pipeline (denoise → Vosk ASR cue-word spotting
  → ML agonal-breathing classifier → fusion), webcam-based slump detection
  (`live/` only), the risk engine, the response state machine, and the
  live demo's spoken voice check-ins.
- **Still stubbed/mocked:**
  - `sensors/heart_rate.py` — returns `random.uniform(60, 90)`, no real
    hardware driver.
  - `sensors/respiratory.py` — returns `random.uniform(12, 18)`, no real
    hardware driver.
  - `sensors/posture.py` (used by the mock demo) — random stub; the real
    implementation is `sensors/webcam.py`, wired only into `tash.live`.
  - `detectors/cardiac.py` — fixed brady/tachy thresholds, not the planned
    MIMIC-trained model.
  - `vehicle/mock.py`, `comms/mock.py` — log actions to stdout; no real
    vehicle or dispatch/caregiver integration.
- **ML model caveats:** the agonal-breathing classifier is validated on
  synthetic audio and a real-but-trained-on positive set; see the
  [Testing & validation](#testing--validation) note above. Real, held-out
  cabin-noise and clinical agonal-breathing recordings are needed before
  production use.
- **Open design items (tracked as v1.2 in FUSION_CONTRACT.md):** whether two
  independent `ELEVATED` signals should escalate to `CRITICAL` (currently
  they don't — `max()` fusion), and a staleness watchdog for "all sensors
  silent" (currently indistinguishable from "all healthy").
