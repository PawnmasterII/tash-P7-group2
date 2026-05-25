# tash-P7-group2

**TASH** — an in-cabin safety hub for autonomous vehicles. It fuses signals
from multiple sensors to detect medical distress (agonal breathing, cardiac
anomalies, postural slump, voice unresponsiveness) and escalates a graded
response: from a voice check-in, through notifying a caregiver and rerouting
to a hospital, up to pulling over, unlocking doors, and dispatching emergency
services.

## Architecture

```
sensors  ->  detectors  ->  risk engine  ->  response orchestrator  ->  vehicle / notifier
```

- `sensors/` — respiratory, posture, heart-rate, microphone streams.
- `detectors/` — per-modality classifiers that emit `DetectionEvent`s with a
  `RiskTier`.
- `fusion/risk_engine.py` — sliding-window fusion across recent events
  (v1 takes the max tier; swap for a learned fuser once labelled data exists).
- `response/state_machine.py` — risk-tier to action ladder; each action fires
  at most once per trip.
- `core/orchestrator.py` — async wiring of the pipeline.
- `vehicle/`, `comms/` — pluggable controllers and notifiers. Today these are
  `MockVehicle` and `MockNotifier`; real implementations slot in without
  changes to the runtime.

## Risk tiers and actions

| Tier      | Actions                                                                 |
|-----------|-------------------------------------------------------------------------|
| NORMAL    | -                                                                       |
| WATCH     | -                                                                       |
| CHECK_IN  | voice check-in                                                          |
| ELEVATED  | + notify caregiver, open video feed, reroute to nearest hospital        |
| CRITICAL  | + pull over, unlock doors, dispatch 911                                 |

## Running the demo

Requires Python 3.10+ (the codebase uses `match` and PEP 604 union syntax).

```bash
python -m tash.main
```

This boots the runtime with mock sensors and a mock vehicle/notifier, so
detection events and actions are logged to stdout rather than affecting a
real vehicle.

## Project layout

```
tash/
  core/        # event bus + async runtime
  sensors/     # modality-specific input streams
  detectors/   # per-modality detectors
  fusion/      # risk fusion
  response/    # action ladder + state machine
  vehicle/     # vehicle controller interface (+ mock)
  comms/       # notifier interface (+ mock)
  types.py     # shared dataclasses and enums
  main.py      # demo wiring
```

## Status

Prototype. Sensors, vehicle, and notifier are mocks; detectors are
heuristic placeholders intended to be replaced with trained models.
