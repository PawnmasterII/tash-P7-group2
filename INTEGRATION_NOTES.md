# Audio Pipeline Integration Notes

Integrates the TASHaudio signal-processing pipeline into the TASH multi-modal safety hub as the `tash.audio` sub-package.

## Architecture

### Adapter design

```
Microphone sensor (WAV replay or live)
    │  payload: {frame: bytes, sample_rate: 16000, ts: float}
    ▼
TASHRuntime._run_detectors()  [sequential per reading, insertion order]
    │
    ├─► AgonalBreathingDetector.observe()
    │       └─► AudioEngine.process_frame(frame, ts)      ← pipeline runs HERE, once
    │               └─► Pipeline.process_chunk(pcm, ts)   ← blocking, off-loop via to_thread
    │                       Stage1 denoise → Stage2 Vosk ASR → Stage3 breathing → FusionEngine
    │                   result cached as engine._latest
    │           reads engine.latest.breathing → DetectionEvent (breathing tier)
    │
    └─► VoiceResponseDetector.observe()
            reads engine.latest_cue (NO second pipeline run)
            → DetectionEvent (CHECK_IN on "help")
```

The pipeline runs **exactly once per sensor frame**. `AgonalBreathingDetector` is registered first in `build_runtime()` to guarantee the cache is populated before `VoiceResponseDetector` reads it. The orchestrator calls detectors sequentially per reading.

### Blocking isolation

`Pipeline.process_chunk()` is synchronous and CPU-bound (numpy/scipy/Vosk). It is called via `asyncio.to_thread()` inside an `asyncio.Lock`, so:
- It never blocks the event loop.
- Concurrent calls (impossible by the sequential detector design, but guarded anyway) are serialized.

### Shared engine lifecycle

```python
# main.py / _amain()
engine = AudioEngine()
await engine.start()   # loads Vosk model (~1-2 s) via asyncio.to_thread
# inject engine into both detectors + sensor wiring
```

If the Vosk model is missing or `vosk` is not installed, `Pipeline` initialises in degraded mode (cue-word stage disabled, breathing still works). `AudioEngine` logs a warning and continues.

---

## Escalation mapping

| Breathing state | Risk tier | Rationale |
|---|---|---|
| `AGONAL_SUSPECT` | `ELEVATED` | Irregular gasps < 6 bpm + high rhythm variability — high suspicion of distress |
| `APNEA` | `ELEVATED` | No breaths detected in 20 s of non-speech audio |
| `LOW_RATE` | `WATCH` | Slow but regular — monitor, do not act yet |
| `NORMAL` | *(no event)* | Healthy range |
| `AMBIGUOUS` | *(no event)* | Insufficient history or conflicting signal |

| Voice signal | Label | Risk tier | Rationale |
|---|---|---|---|
| Distress cue ("help"), unarmed | `"help_distress_cue"` | `CHECK_IN` | Single word is never enough to auto-dispatch; orchestrator prompts passenger first |
| Distress cue ("help") while armed | `"help_while_awaiting"` | `ELEVATED` | Corroboration: slump + active distress cue → escalate |
| No response within 30 s window | `"no_response_timeout"` | `ELEVATED` | Corroboration: slump + no response → escalate |
| Reassurance word ("fine"/"okay"/"ok") while armed | — | `None` | De-escalate; disarm the response window |
| Reassurance word while unarmed | — | `None` | Silently ignored |

**Safety invariant preserved:** No single audio signal reaches `CRITICAL`. Corroboration (cue + agonal, slump + no-response, or slump + active distress cue) is required to reach `ELEVATED` or higher — that logic lives in `VoiceResponseDetector`'s armed window and the 30 s `RiskEngine` eviction.

---

## How to run

### 1. Prerequisites

```
Python 3.12   (webrtcvad-wheels unavailable on 3.13+; energy-VAD fallback otherwise)
```

### 2. Install the package

```powershell
# From C:\Users\dongt\Desktop\code\tash-P7-group2\
pip install -e ".[vad]"        # includes webrtcvad-wheels for production VAD
# or minimal:
pip install -e .               # energy-VAD fallback (coarser but functional)
```

The `package_dir={"tash": "."}` mapping in `setup.py` lets `python -m tash.main` resolve even though the directory is named `tash-P7-group2`.

### 3. Download the Vosk model (~40 MB, once)

```powershell
python -m tash.audio.download_model
```

This places the model at `tash-P7-group2/audio/models/vosk-model-small-en-us-0.15/`. Without it the pipeline runs in degraded mode (no cue-word detection; breathing analysis still works).

### 4. Run in WAV-replay mode (no microphone needed)

```powershell
python -m tash.main
```

The `Microphone` sensor defaults to `mode="wav"` and looks for WAV files at `../../TASHaudio/test_audio/*.wav` (the sibling TASHaudio repo). You will see log output from all sensors, detectors, risk engine, and response orchestrator.

To point it at a different directory:

```python
# in main.py, change:
Microphone(mode="wav", wav_dir="/path/to/wavs")
```

### 5. Run with a real microphone

```powershell
pip install pvrecorder
# in main.py, change Microphone(mode="wav") → Microphone(mode="live")
python -m tash.main
```

---

## Files changed / added

| File | Change |
|---|---|
| `audio/__init__.py` | New — package root |
| `audio/config.py` | Vendored; `VOSK_MODEL_PATH` uses `__file__` (auto-correct on move) |
| `audio/contracts.py` | Vendored; no logic changes |
| `audio/stage1_denoise.py` | Vendored; imports → `from tash.audio import config` etc. |
| `audio/stage2_cueword.py` | Vendored; imports fixed |
| `audio/stage3_breathing.py` | Vendored; imports fixed |
| `audio/fusion.py` | Vendored; imports fixed |
| `audio/pipeline.py` | Vendored; imports fixed |
| `audio/engine.py` | **New** — AsyncEngine wrapper (to_thread + Lock + cache) |
| `audio/download_model.py` | New — idempotent Vosk model downloader |
| `sensors/microphone.py` | **Replaced** — WAV replay + pvrecorder live modes |
| `detectors/agonal_breathing.py` | **Filled** — drives engine, maps BreathingState → RiskTier |
| `detectors/voice_response.py` | **Updated** — armed response window (30 s timeout → ELEVATED), reassurance de-escalation, distress-while-armed → ELEVATED |
| `main.py` | Updated — constructs AudioEngine, injects into detectors |
| `requirements.txt` | **New** — audio pipeline deps |
| `setup.py` | **New** — `pip install -e .` wiring (`package_dir={"tash": "."}`) |
| `.gitignore` | Updated — excludes `audio/models/`, `__pycache__/`, `.venv*` |

---

## Verified output (WAV-replay demo, Python 3.12)

Running `python -m tash.main` against TASHaudio's synthetic corpus produces:

**agonal_gasps.wav** (~25 s, sparse irregular gasps):
- Pipeline produces `CONFIRM` internally after 3 consecutive agonal windows
- `AgonalBreathingDetector` emits `DetectionEvent(label="agonal_suspect", risk_tier=ELEVATED)`
- TASH `RiskEngine` aggregates to `ELEVATED` → ResponseOrchestrator fires:
  ```
  [response] firing voice_check_in (tier=ELEVATED)
  [vehicle]  speak: Are you okay? Say 'fine' or 'okay' if you're okay.
  [response] firing notify_caregiver (tier=ELEVATED)
  [response] firing open_video_feed (tier=ELEVATED)
  [response] firing reroute_hospital (tier=ELEVATED)
  ```

**say_help.wav** (synthetic "help" utterance):
- 2 `CueWordEvent` detections with confidence_proxy > 0.5, `category="distress"`
- `VoiceResponseDetector` emits `DetectionEvent(label="help_distress_cue", risk_tier=CHECK_IN)`
- TASH `RiskEngine` → `CHECK_IN` tier → `VOICE_CHECK_IN` action fires, detector armed
- If passenger says "fine"/"okay": de-escalated (no further actions)
- If no response within 30 s: `no_response_timeout` event → `ELEVATED` → caregiver/reroute fire

**clean_silence.wav, road_noise.wav**: no events (below threshold / insufficient history)

**normal_breathing.wav**: `BreathingState.NORMAL` → no DetectionEvent emitted (by design)

---

## Known limitations

- **Synthetic corpus only.** Breathing detection is validated on synthetic WAV files (`make_test_audio.py`). Real cabin noise and clinical agonal-breathing recordings are needed before production use.
- **20 s warm-up.** Stage 3 requires a 20 s rolling window before emitting any BreathingEstimate. No breathing events appear in the first ~20 s of a session.
- **Vosk degrades gracefully.** If the model is absent or `vosk` is not installed, cue-word detection is disabled. Breathing detection and the rest of the pipeline continue normally.
- **WAV replay pacing.** The sensor sleeps `_CHUNK_DURATION_S` (32 ms) between chunks to avoid flooding the event loop, but real-time is not guaranteed under CPU load.
