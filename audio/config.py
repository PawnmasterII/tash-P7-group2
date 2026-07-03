"""Central, immutable configuration for the audio pipeline.

Every constant the stages depend on lives here so tuning happens in ONE place
and the canonical signal contract can never drift between stages.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Canonical signal contract -------------------------------------------
# The Vosk small EN model and webrtcvad both expect 16 kHz; librosa/scipy are
# rate-agnostic. Locking everything here removes a whole class of bugs.
SAMPLE_RATE = 16_000        # Hz, mono
SAMPLE_WIDTH = 2            # bytes (int16)
CHANNELS = 1

# --- Stage 1: denoise ------------------------------------------------------
BANDPASS_LOW_HZ = 80.0
BANDPASS_HIGH_HZ = 4_000.0
BANDPASS_ORDER = 4

DENOISE_N_FFT = 512
DENOISE_HOP = 128
DENOISE_TIME_CONSTANT_S = 2.0

# Keyword path: aggressive. Breathing path: gentle (faint gasps must survive).
KEYWORD_PROP_DECREASE = 0.8
BREATHING_PROP_DECREASE = 0.4

NOISE_PROFILE_MAX_AGE_S = 30.0     # force-refresh stale baselines
NOISE_BASELINE_SECONDS = 1.5       # engine-start capture length

# --- Stage 2: cue word (Vosk offline ASR — no key, pre-trained) ------------
# Vosk runs fully offline (no key/signup) and recognizes arbitrary words, so
# "help" works out of the box with no custom model or training.
import os as _os

VAD_AGGRESSIVENESS = 2             # 0..3 (only used if real webrtcvad present)
VAD_FRAME_MS = 30                  # webrtcvad accepts 10/20/30 ms only
VAD_HANGOVER_MS = 300              # keep "speech active" after last positive

# Path to an unpacked Vosk model directory. Download via:
#   python -m tash.audio.download_model
# (places model at tash/audio/models/vosk-model-small-en-us-0.15)
VOSK_MODEL_PATH = _os.path.join(
    _os.path.dirname(__file__), "models", "vosk-model-small-en-us-0.15"
)

# Distress cue words to spot. Matched as whole words in Vosk's full-vocabulary
# transcript (NOT a restricted grammar — that force-decodes noise into "help").
CUE_WORDS: list[str] = ["help"]
# Reassurance words: passenger response to a VOICE_CHECK_IN prompt. Recognized
# during the armed response window; de-escalate if heard, escalate on silence.
REASSURANCE_WORDS: list[str] = ["fine", "okay", "ok"]
CUE_WORD_DEBOUNCE_MS = 1_500
CUE_WORD_MIN_CONFIDENCE = 0.5      # reject low-confidence final-result matches

# Fallback gate when webrtcvad is unavailable (e.g. no wheel for this Python).
# Normalized RMS above this counts as speech. Coarser than webrtcvad — for
# bring-up only; install real webrtcvad for production.
ENERGY_VAD_RMS_THRESHOLD = 0.01

# --- Stage 3: breathing (advisory heuristic) -------------------------------
BREATH_BAND_LOW_HZ = 100.0
BREATH_BAND_HIGH_HZ = 2_000.0
RESP_WINDOW_S = 20.0               # respiration is slow; need a long window
RESP_HOP_S = 1.0                   # but slide it for ~1 s update latency
RESP_RMS_FRAME = 1024
RESP_RMS_HOP = 512

# Rule thresholds (breaths per minute). Tune/validate on real recordings.
RESP_RATE_NORMAL_MIN = 10.0
RESP_RATE_NORMAL_MAX = 22.0
RESP_RATE_LOW = 6.0                # below this with no speech => distress candidate
RESP_PERSISTENCE_WINDOWS = 3       # consecutive windows before raising state


@dataclass(frozen=True)
class DerivedSizes:
    """Frame sizes derived from the contract — computed, never hand-typed."""

    vad_frame_samples: int
    vad_frame_bytes: int
    vad_hangover_frames: int

    @classmethod
    def build(cls) -> "DerivedSizes":
        vad_samples = SAMPLE_RATE * VAD_FRAME_MS // 1000           # 480 @16k/30ms
        return cls(
            vad_frame_samples=vad_samples,
            vad_frame_bytes=vad_samples * SAMPLE_WIDTH,
            vad_hangover_frames=max(1, VAD_HANGOVER_MS // VAD_FRAME_MS),
        )


SIZES = DerivedSizes.build()


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY BUDGET (for fusion synchronization with HR / vision sensors)
# ─────────────────────────────────────────────────────────────────────────────
# WHY this exists: the fusion layer corroborates this audio stream against other
# modalities (heart-rate, in-cabin vision). To align events on a shared timeline
# those consumers need to know how stale an audio event is by the time they see
# it. These are PER-STAGE ENGINEERING BUDGET TARGETS, not measured values — they
# are the ceilings each stage is expected to stay under, used for sync planning.
#
# ⚠️  NOT MEASURED on this hardware. To measure on yours, wrap each stage's
#     `.process(...)` call in `time.perf_counter()` over a representative chunk
#     stream and replace these numbers. Vosk final-result latency in particular
#     is data-dependent (it waits for a speech→silence edge) and dominates.
# Surfaced to the monorepo fusion layer as DetectionEvent metadata
# (metadata["audio_latency_ms"] = TOTAL_LATENCY_MS). See
# tash-P7-group2/FUSION_CONTRACT.md for how the audio detectors emit
# DetectionEvent — this budget is advisory metadata, not part of the schema.
AUDIO_LATENCY_MS = {
    "stage1_denoise": 10,        # noisereduce spectral gate + scipy band-pass, per chunk
    "stage2_vosk_partial": 50,   # Vosk partial hypothesis (may emit mid-word; unused — we gate on finals)
    "stage2_vosk_final": 150,    # Vosk final result — fires only on a speech→silence edge (the real cost)
    "stage3_breathing": 20,      # librosa RMS envelope + scipy find_peaks over the rolling window
    "fusion_logic": 15,          # the audio pipeline's OWN escalation truth table (fusion.py)
    "total_pipeline_ms": 195,    # worst case, surfaced as DetectionEvent metadata (see assert below)
}

# Worst-case end-to-end is SEQUENTIAL: pipeline.process_chunk() runs stage1 →
# stage2 → stage3 → fusion in one thread per chunk, so on a chunk that produces
# a Vosk FINAL result the wall time is denoise + vosk_final + breathing + fusion.
# (partial and final are mutually exclusive per chunk, so partial is NOT summed.)
_DERIVED_TOTAL_MS = (
    AUDIO_LATENCY_MS["stage1_denoise"]
    + AUDIO_LATENCY_MS["stage2_vosk_final"]
    + AUDIO_LATENCY_MS["stage3_breathing"]
    + AUDIO_LATENCY_MS["fusion_logic"]
)
assert _DERIVED_TOTAL_MS == AUDIO_LATENCY_MS["total_pipeline_ms"], (
    "AUDIO_LATENCY_MS components must sum to total_pipeline_ms "
    f"({_DERIVED_TOTAL_MS} != {AUDIO_LATENCY_MS['total_pipeline_ms']}) — "
    "update both this dict and FUSION_CONTRACT.md together."
)
# Single value the detector adapter puts in DetectionEvent.metadata["audio_latency_ms"].
TOTAL_LATENCY_MS = AUDIO_LATENCY_MS["total_pipeline_ms"]


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE TARGETS (acceptance goals — what "good enough" looks like)
# ─────────────────────────────────────────────────────────────────────────────
# WHY this exists: gives the team and any later eval harness a single, explicit
# definition of done. ⚠️  Targets, not guarantees — current numbers are proven
# only on the SYNTHETIC corpus (see CONSTRAINTS["synthetic_testing_only"]).
PERFORMANCE_TARGETS = {
    "vosk_accuracy_on_help": 0.95,           # ≥95% true-positive rate on a spoken "help"
    "false_positive_rate": 0.05,             # <5% false alarms on cabin noise / non-cue speech
    "breathing_detection_sensitivity": None, # advisory only; TBD — needs real agonal recordings
    "end_to_end_latency_ms": 200,            # audio-only ceiling; total_pipeline_ms (195) must stay under this
}


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN CONSTRAINTS (the hard edges — violating any of these breaks the pipeline)
# ─────────────────────────────────────────────────────────────────────────────
# WHY this exists: makes the non-negotiable assumptions explicit so a teammate
# doesn't, say, swap in a 44.1 kHz mic or a Python 3.13 venv and burn an hour.
CONSTRAINTS = {
    "python_version": "3.12 required for the real webrtcvad-wheels + librosa stack (3.14 runs degraded on the energy-VAD fallback)",
    "audio_sample_rate": "16000 Hz mono — fixed by the Vosk model and webrtcvad; see SAMPLE_RATE",
    # NOTE: the live-mic chunk is 512 samples = 32 ms @ 16 kHz (NOT 20 ms — that
    # would be 320 samples). The VAD frame is a separate 480 samples (30 ms);
    # Stage 2 re-blocks the 512-sample chunks into 480-sample VAD frames.
    "frame_size": "live-mic chunk = 512 samples (32 ms @ 16 kHz); VAD frame = 480 samples (30 ms), re-blocked internally",
    "vosk_model_size": "~40 MB, downloaded once into audio/models/ via python -m tash.audio.download_model",
    "synthetic_testing_only": "metrics validated on synthetic audio only — real cabin noise + clinical agonal-breathing recordings required before production",
    # The monorepo aligns modalities by DetectionEvent.timestamp — a tz-aware UTC
    # datetime from tash.types.now(). Stage ts here is stream-relative seconds
    # (frames*512/SAMPLE_RATE); the detector adapter stamps each emitted
    # DetectionEvent with tash.types.now() instead.
    "realtime_sync_clock": "monorepo uses tz-aware UTC datetime (tash.types.now()) on DetectionEvent; pipeline ts is stream-relative until the detector adapter wraps it",
}
