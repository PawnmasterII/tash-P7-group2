# Fusion Contract: Multi-Modal Pipeline Integration Specification

This document defines the data schema, timing, and orchestration logic that
connects the independent pipelines feeding the TASH in-cabin safety hub:
- **Audio Pipeline** — agonal breathing + distress cue words + voice responsiveness
- **Vision Pipeline** — postural slump / fall detection, pose landmarks
- **HR Sensor Pipeline** — heart rate + anomalies

All pipelines **MUST** plug into this contract by **June 15, 2026**.

- **Version:** 1.1
- **Last Updated:** 2026-06-09
- **Maintained by:** Audio Lead (dongtinzin@gmail.com)

> **This contract documents the integration model already implemented in this
> repo** (`tash/types.py`, the `Detector`/`Sensor` ABCs, `fusion/risk_engine.py`,
> `response/state_machine.py`). It is descriptive, not aspirational: the shared
> dataclasses and the detector interface below ARE the contract. New pipelines
> conform to it by subclassing the ABCs — they do not invent their own schema.

---

## 1. The integration model

```
sensors  ->  detectors  ->  risk engine  ->  response orchestrator  ->  vehicle / notifier
```

Every pipeline integrates at one or both of two seams, both defined in code:

1. **`Sensor`** (`tash/sensors/base.py`) — a hardware-agnostic source that
   `async`-streams `SensorReading`s for one `Modality`.
2. **`Detector`** (`tash/detectors/base.py`) — consumes `SensorReading`s of its
   modality and emits a `DetectionEvent` (or `None`) carrying a `RiskTier`.

The runtime (`tash/core/orchestrator.py`) wires them: sensor readings fan out to
the detectors registered for that modality; emitted events go to the
`RiskEngine`, whose current tier drives the `ResponseOrchestrator`. **You do not
touch the orchestrator** — you add a `Sensor` and/or `Detector` and register them
in `main.py`.

---

## 2. Core schema (the contract — `tash/types.py`)

These are the exact, frozen dataclasses every pipeline produces/consumes. Do not
fork them.

```python
class Modality(str, Enum):
    RESPIRATORY = "respiratory"
    POSTURE     = "posture"
    HEART_RATE  = "heart_rate"
    MICROPHONE  = "microphone"

class RiskTier(IntEnum):          # ORDINAL — fusion compares with max()
    NORMAL   = 0
    WATCH    = 1
    CHECK_IN = 2
    ELEVATED = 3
    CRITICAL = 4

def now() -> datetime:            # tz-aware UTC; the canonical clock
    return datetime.now(timezone.utc)

@dataclass(frozen=True)
class SensorReading:
    modality: Modality
    payload: Any                  # modality-specific dict (see §3)
    timestamp: datetime = field(default_factory=now)

@dataclass(frozen=True)
class DetectionEvent:
    detector: str                 # detector.name, e.g. "agonal_breathing"
    modality: Modality
    label: str                    # short outcome tag, e.g. "slump", "cue_word"
    confidence: float             # 0.0–1.0
    risk_tier: RiskTier
    timestamp: datetime = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)   # everything else

@dataclass(frozen=True)
class TripContext:
    trip_id: str
    passenger_id: str
    destination: tuple[float, float] | None = None
    nearest_hospital: tuple[float, float] | None = None
    position: tuple[float, float] | None = None
```

### Rules
- **Timestamps are tz-aware UTC `datetime`** from `types.now()` — *not* epoch
  milliseconds. Leave the field at its default factory unless you are replaying
  recorded data, in which case set it explicitly.
- **`risk_tier` is ordinal.** Fusion takes the max across a window (§5), so choose
  a tier honestly: reserve `CRITICAL` for "dispatch 911 is warranted on this
  signal alone."
- **Put everything modality-specific in `metadata`** (bpm, resp-rate, landmarks,
  latency, raw confidences). The top-level fields are the only cross-modal API.
- **A detector returns `None` when there is nothing to report.** No event = no
  contribution to the risk tier.

---

## 3. Per-pipeline integration

Each pipeline implements a `Detector` (the canonical examples already in the repo
are `detectors/slump.py` and `detectors/cardiac.py` — copy their shape) and, for
real hardware, a `Sensor` that streams the `payload` its detector expects.

### 3a. Audio Pipeline → two detectors on `Modality.MICROPHONE`

The audio pipeline (standalone `TASHaudio` repo) already produces a
`FusionDecision` per chunk via `Pipeline.process_chunk(...)`. Wrap it in detector
adapters that emit `DetectionEvent`s:

- **`AgonalBreathingDetector`** (`detectors/agonal_breathing.py`, stub today) —
  driven by the pipeline's Stage 3 `BreathingEstimate`.

  | Audio `BreathingState` | `label` | `risk_tier` | notes |
  |---|---|---|---|
  | `APNEA` | `"apnea"` | `ELEVATED` ⚠️ | advisory; corroborate before dispatch |
  | `AGONAL_SUSPECT` | `"agonal"` | `ELEVATED` ⚠️ | advisory; corroborate before dispatch |
  | `LOW_RATE` | `"low_rate"` | `WATCH` | |
  | `NORMAL` / `AMBIGUOUS` | — | — | return `None` |

  `metadata = {"resp_rate_bpm", "interval_cv", "confidence", "audio_latency_ms"}`.

  > ⚠️ **Safety flag (needs a team decision).** The current
  > `AgonalBreathingDetector` docstring says to "emit `RiskTier.CRITICAL`". The
  > audio pipeline's own code is emphatic that breathing is **advisory only and
  > must never auto-dispatch alone** (`stage3_breathing.py`, `contracts.py`), and
  > it is validated on *synthetic data only*. Because `RiskEngine` takes the max
  > tier, mapping agonal → `CRITICAL` would let one heuristic suspicion trigger
  > pull-over + 911. **Recommended:** map agonal → `ELEVATED` (notify caregiver +
  > open video feed) until clinically validated; reserve `CRITICAL` for
  > corroborated signals. Resolve in v1.1.

- **`VoiceResponseDetector`** (`detectors/voice_response.py`) — driven by
  Stage 2 cue-word detection and the check-in response window.

  | Situation | `label` | `risk_tier` |
  |---|---|---|
  | Spontaneous distress cue ("help"), unarmed | `"help_distress_cue"` | `CHECK_IN` |
  | Armed for check-in; distress cue ("help") arrives | `"help_while_awaiting"` | `ELEVATED` |
  | Armed for check-in; no reply within 30 s window | `"no_response_timeout"` | `ELEVATED` |
  | Armed for check-in; reassurance ("fine"/"okay"/"ok") | — | `None` (de-escalate, disarm) |
  | Reassurance word heard while unarmed | — | `None` (silently ignored) |

  `metadata = {"keyword", "confidence_proxy", "vad_active", "audio_latency_ms"}` for
  cue-word events; `{"elapsed_s"}` for timeout events. The orchestrator arms this
  detector via the `on_check_in` hook (wired in `main.py`).

  The response window is 30 s (`_RESPONSE_WINDOW_S` in `voice_response.py`), checked
  on each incoming MICROPHONE frame. Vosk is configured (in `audio/pipeline.py`) to
  recognise both distress and reassurance words; `CueWordEvent.category` carries
  `"distress"` or `"reassurance"` so the detector can branch without re-importing
  the word lists.

> The audio pipeline's internal `EscalationLevel`/`FusionEngine` is **redundant**
> with the monorepo's central `RiskEngine` once integrated. Use it standalone for
> the audio repo's own tests; at integration, the detector adapters above are the
> contract surface, and central fusion does the cross-modal combining.

### 3b. Vision Pipeline → `SlumpDetector` on `Modality.POSTURE`

`detectors/slump.py` is the working reference. The vision team replaces its body
with the real model and feeds it via a posture `Sensor` whose `payload` provides
pose data.

- `label = "slump"`; `risk_tier`:
  - `WATCH` — lean ≥ 25° (monitor; no actions fire yet).
  - `CHECK_IN` — slump ≥ 45° (severe slump gates on a voice check-in before
    dispatching; `VoiceResponseDetector` drives escalation to `ELEVATED` if the
    passenger does not respond within the 30 s window).
- Below 25° → return `None`.
- `metadata = {"angle_deg", "pose_quality", "motion", "pose_landmarks", "vision_latency_ms"}`.
  `pose_landmarks` may be omitted when quality is low.
- Expected `SensorReading.payload`: `{"slump_angle_deg": float, ...}` (see the
  reference detector).

> **Design note:** mapping severe slump to `CHECK_IN` (not `ELEVATED`) prevents
> the caregiver/reroute actions from firing on a slump alone — the safety invariant
> requires corroboration (slump + no response, or slump + active distress cue)
> before dispatching. `VoiceResponseDetector` provides that gate.

### 3c. HR Sensor Pipeline → `CardiacAnomalyDetector` on `Modality.HEART_RATE`

`detectors/cardiac.py` is the working reference (currently fixed brady/tachy
thresholds). The HR team feeds real readings via `sensors/heart_rate.py` and
replaces the thresholds with **baseline-relative** anomaly detection.

- `label`: `"spike"` | `"drop"` | `"erratic"` (or `"cardiac_anomaly"`); `risk_tier`
  `WATCH`/`ELEVATED` by severity.
- `bpm == 0` (sensor failure) → return `None` (no signal ≠ a healthy reading).
  Optionally emit `label="sensor_fault", risk_tier=NORMAL` so the fault is logged.
- In-range → return `None`.
- `metadata = {"bpm", "baseline_bpm", "signal_quality", "hr_latency_ms"}`.
  `baseline_bpm` is a running average over ~30 s.
- Expected `SensorReading.payload`: `{"bpm": float, ...}`.

---

## 4. Timing & synchronization

### Clock
All events carry a tz-aware UTC `datetime` from `types.now()`. There is **no**
manual epoch-ms stamping — alignment is by `datetime` comparison inside the risk
window. Replays set `timestamp` explicitly to preserve ordering.

### Cadence & latency (advisory, carried in `metadata`)
| Pipeline | Emit cadence | Latency budget | Source |
|----------|--------------|----------------|--------|
| Audio | ~32 ms/chunk (512 samples @ 16 kHz) | ~195 ms worst case (denoise 10 + Vosk-final 150 + breathing 20 + fusion 15, sequential per chunk) | `audio-pipeline/config.py:AUDIO_LATENCY_MS` |
| Vision | ~33 ms (30 Hz) | ~33 ms (one frame) | vision repo |
| HR | ~1000 ms (1 Hz) | ~1000 ms (one reading) | hr repo |

> Note: 512 samples @ 16 kHz = **32 ms** (not 20 ms). The audio latency is an
> engineering budget, **not measured** — see the warning in
> `audio-pipeline/config.py`.

### Windowed fusion handles cadence skew for you
You do **not** hand-align buffers. `RiskEngine` keeps a **30 s sliding window**
(`fusion/risk_engine.py:DEFAULT_WINDOW`) and evicts events older than the window
by their `timestamp`. The slow HR cadence (~1 Hz) and the fast audio cadence
(~30 Hz) coexist naturally: a recent HR `ELEVATED` stays "live" for 30 s and can
corroborate an audio event that arrives later.

---

## 5. Fusion semantics

`RiskEngine.current_tier()` returns **the maximum `risk_tier` across all events in
the window** (`NORMAL` if empty). That tier drives the action ladder in
`response/state_machine.py`:

| Tier | Actions (cumulative; each fires at most once per trip) |
|------|--------------------------------------------------------|
| `NORMAL` / `WATCH` | — |
| `CHECK_IN` | voice check-in ("Are you okay?") |
| `ELEVATED` | + notify caregiver, open video feed, reroute to nearest hospital |
| `CRITICAL` | + pull over, unlock doors, dispatch 911 |

**Corroboration is emergent, not a hard-coded tree.** v1 takes the max tier, so:
- A single `CHECK_IN` event (a slump ≥ 45°, a spontaneous "help" cue, or any
  other `CHECK_IN`-tier detector) prompts the passenger via `VOICE_CHECK_IN`.
- The check-in escalation path (slump → prompt → no response → `ELEVATED`) is
  implemented entirely in `VoiceResponseDetector`: it emits `ELEVATED` on a
  30 s timeout or on a distress cue received while armed (§3a). The `RiskEngine`
  simply takes the max of whatever events are present — it has no check-in state.
- Two independent `ELEVATED` signals (e.g. agonal breathing + HR anomaly) still
  yield `ELEVATED` under `max()` — they don't automatically sum to `CRITICAL`. If
  product wants "two ELEVATED modalities → CRITICAL," that is a **learned/dual-
  signal fuser** change (the `RiskEngine` docstring already anticipates this).
  Track as v1.2.

---

## 6. Robustness & failure modes

The windowed, additive design degrades gracefully without special-casing:

- **A pipeline crashes / stops emitting** → its events simply age out of the 30 s
  window; the tier it contributed decays to `NORMAL`. No exception, no false
  escalation. Other modalities keep working.
- **A sensor disconnects** → no `SensorReading`s of that modality → its detectors
  never fire. Same graceful decay.
- **A detector errors on one reading** → it should catch internally and return
  `None`; one bad frame must not crash the run loop.
- **All pipelines silent** → tier is `NORMAL`. ⚠️ Note this is indistinguishable
  from "all healthy." If you need an explicit *"monitor offline"* alert, add a
  **staleness watchdog** (e.g. a sensor heartbeat; if no readings for N seconds,
  emit a `WATCH`/notify event). Not in v1 — track as v1.1.
- **Stale/replayed timestamps** → events older than the window are evicted on
  `ingest`/`current_tier`; out-of-window data cannot influence the decision.

---

## 7. Conformance checklist (per pipeline)

- [ ] My detector subclasses `tash.detectors.base.Detector` with `name` + `modality`.
- [ ] `observe()` returns a `DetectionEvent` built from `tash.types` (or `None`).
- [ ] `risk_tier` chosen per §3 (and the agonal-breathing safety flag respected).
- [ ] Modality-specific data lives in `metadata`, not new top-level fields.
- [ ] Timestamps left to `types.now()` (or set explicitly only for replays).
- [ ] Registered in `main.py` (`detectors=[...]`, and a `Sensor` if real hardware).
- [ ] `python -m tash.main` runs with my detector wired (mocks OK).

---

## 8. Versioning

**Version 1.0 (2026-06-04)** — initial `DetectionEvent` / `Detector` /
`RiskEngine` integration model; per-pipeline mapping for audio, vision, HR.

**Version 1.1 (2026-06-09)** — slump/voice check-in flow implemented:
- `SlumpDetector`: severe slump (≥ 45°) now emits `CHECK_IN` instead of `ELEVATED`
  so the voice gate runs before caregiver/reroute actions fire.
- `VoiceResponseDetector`: armed response window (30 s); emits `ELEVATED` on
  timeout or active "help"; de-escalates silently on reassurance ("fine"/"okay"/"ok").
- `CueWordEvent` gains a `category` field (`"distress"` | `"reassurance"`).
- `CueWordDetector` gains a `cue_words` constructor param (default = `["help"]`);
  `Pipeline` passes both distress + reassurance words so Vosk recognises all.
- Voice prompt updated to `"Are you okay? Say 'fine' or 'okay' if you're okay."`.
- `tests/test_slump_checkin.py` (19 tests) covers all three flow outcomes.

**Open items for 1.2:**
- Agonal breathing tier: `ELEVATED` (recommended) vs `CRITICAL` (current stub) — safety decision.
- Dual-signal corroboration (two `ELEVATED` → `CRITICAL`?) — needs a learned/rule fuser.
- Staleness watchdog for "monitor offline."
