# Integration Checklist: Connecting Your Pipeline to TASH

Use this when your pipeline is ready to plug into the runtime. Complete the
phases **in order**. The schema and tier mappings you must match are in
[FUSION_CONTRACT.md](./FUSION_CONTRACT.md) — read it first. Your integration
surface is a `Detector` subclass (and, for real hardware, a `Sensor`); you do
**not** modify the orchestrator, risk engine, or response ladder.

---

## Phase 1: Pre-Integration (solo development)

- [ ] My pipeline runs independently (no crashes) on synthetic or real input.
- [ ] I have read [FUSION_CONTRACT.md](./FUSION_CONTRACT.md) §2 (schema) and §3
      (my pipeline's detector + tier mapping).
- [ ] My detector subclasses `tash.detectors.base.Detector` with class attrs
      `name: str` and `modality: Modality`.
- [ ] `async def observe(self, reading) -> DetectionEvent | None` returns a
      `DetectionEvent` built from `tash.types` — or `None` when there's nothing
      to report (no event = no risk contribution).
- [ ] `risk_tier` follows §3 of the contract. **Audio team:** honor the
      agonal-breathing safety flag (map to `ELEVATED`, not `CRITICAL`, pending
      clinical validation).
- [ ] Modality-specific data (bpm, resp-rate, landmarks, latency, raw confidence)
      goes in `metadata`, not new top-level fields.
- [ ] I left `timestamp` to its default (`types.now()`), or set it explicitly
      only for recorded-data replays.
- [ ] I unit-tested the detector by feeding it `SensorReading`s and asserting the
      emitted `DetectionEvent` (label / tier / metadata) is correct.
- [ ] I documented any deviation from the contract in my repo's `NOTES.md`.

> **Reference implementations to copy:** `detectors/slump.py` (POSTURE) and
> `detectors/cardiac.py` (HEART_RATE) are complete, working detectors that show
> the exact shape. The audio detectors `detectors/agonal_breathing.py` and
> `detectors/voice_response.py` are stubs awaiting your model.

---

## Phase 2: Wire into the runtime

1. **Get the repo and confirm the demo runs as-is:**
   ```bash
   git clone https://github.com/PawnmasterII/tash-P7-group2.git
   python -m tash.main          # mock sensors -> detectors -> fusion -> response
   ```
   ⚠️ `python -m tash.main` imports the package as `tash`, so the repo must be
   importable under that name (run from a parent dir that has the repo on
   `PYTHONPATH`, or check it out as a `tash/` directory). If you get
   `ModuleNotFoundError: tash`, that's the cause.

2. **Drop your detector in** `detectors/your_detector.py` (or replace the stub
   body), keeping the `Detector` interface.

3. **Provide input:** for real hardware, add/replace the matching `Sensor` in
   `sensors/` so its `stream()` yields `SensorReading(modality, payload=...)`
   with the `payload` keys your detector reads (see contract §3). For now the
   `Mock*` sensors are fine.

4. **Register both in `main.py`:**
   - [ ] add your detector to the `detectors=[...]` list in `build_runtime()`
   - [ ] add your sensor to the `sensors=[...]` list (if you added one)

5. **Record your latency/constraints** (advisory metadata): mirror
   `audio-pipeline/config.py:AUDIO_LATENCY_MS` in your repo, and surface the
   number in your event's `metadata["..._latency_ms"]`.

6. **Run it:**
   ```bash
   python -m tash.main
   ```
   - **Expected:** your `DetectionEvent`s appear in the log and move the risk
     tier; the response ladder fires the matching actions.

---

## Phase 3: Integration testing (full runtime)

- [ ] `python -m tash.main` runs with my detector wired and logs my events.
- [ ] Drive a positive case and **verify** the expected tier and actions:
  - Slump-angle payload ≥ 45° → `CHECK_IN` → voice check-in fires, detector armed.
    Then: reassurance word → de-escalate (no further actions); OR "help" /
    30 s silence → `ELEVATED` → notify caregiver, open video, reroute.
  - Out-of-range bpm → `ELEVATED` → notify/video/reroute.
  - Spontaneous "help" cue (unarmed) → `CHECK_IN` → voice check-in fires.
  - `CRITICAL` detector event → pull over/unlock/911.
- [ ] Drive a normal case and **verify** no event / `NORMAL` tier.
- [ ] Stop my pipeline mid-run and **verify** the runtime keeps going and my
      tier contribution **decays to `NORMAL`** as events age out of the 30 s
      window (no crash, no false escalation).
- [ ] Confirm cross-modal behavior with another detector active (e.g. slump +
      cardiac) matches the max-tier fusion semantics (contract §5).
- [ ] Document results in `docs/INTEGRATION_REPORT.md`.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tash'`**
  - The package must be importable as `tash` — see Phase 2, step 1.
- **My events never change the risk tier**
  - Check `detector.modality` matches the `SensorReading.modality` (the
    orchestrator routes by modality), and that `observe()` returns a real
    `DetectionEvent` (not `None`) with a tier above `NORMAL`.
- **`TypeError`/`KeyError` building a `DetectionEvent`**
  - You added or renamed a top-level field. The schema is fixed (contract §2);
    put extras in `metadata`.
- **My action fires only once**
  - By design — `response/state_machine.py` fires each action at most once per
    trip (`_fired`). Call `ResponseOrchestrator.reset()` between test scenarios.
- **Timestamps/ordering look wrong**
  - Don't hand-stamp epoch millis. Use the default `types.now()` (tz-aware UTC);
    only set `timestamp` explicitly when replaying recorded data.
- **No audio/video/sensor device**
  - Use the `Mock*` sensors (or your own synthetic `Sensor`); device access isn't
    required to validate the contract.
