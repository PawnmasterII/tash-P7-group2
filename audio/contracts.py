"""Data contracts exchanged between pipeline stages.

These dataclasses ARE the integration spec from the architecture doc (§6 of
each stage). Stages must produce/consume exactly these — nothing implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # only needed for annotations; keeps Layer-1 tests stdlib-only
    import numpy as np


@dataclass(frozen=True)
class DenoisedChunk:
    """Stage 1 output. Two denoise strengths on a shared timeline."""

    ts: float                       # seconds, monotonic, chunk start
    int16: bytes                    # full-denoise PCM (VAD / wake-word engines)
    float32: np.ndarray             # full-denoise float view, [-1, 1]
    mild_float32: np.ndarray        # gentle denoise for breathing (gasps kept)
    noise_profile_age_s: float


@dataclass(frozen=True)
class CueWordEvent:
    """Stage 2 output (event). Emitted only on a debounced detection."""

    ts: float
    keyword: str
    vad_active: bool
    sensitivity: float
    confidence_proxy: float         # 0..1, NOT a calibrated probability
    category: str = "distress"      # "distress" or "reassurance"


class BreathingState(str, Enum):
    NORMAL = "normal"
    LOW_RATE = "low_rate"
    AGONAL_SUSPECT = "agonal_suspect"
    APNEA = "apnea"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class BreathingEstimate:
    """Stage 3 output. ADVISORY ONLY — never an auto-dispatch trigger."""

    ts: float
    state: BreathingState
    resp_rate_bpm: Optional[float]
    interval_cv: Optional[float]    # coefficient of variation of intervals
    confidence: float               # 0..1, heuristic confidence
    features: dict = field(default_factory=dict)


class EscalationLevel(str, Enum):
    NONE = "none"
    MONITOR = "monitor"
    CONFIRM = "confirm"             # prompt passenger "Are you okay?"
    ESCALATE = "escalate"           # human/operator dispatch path


@dataclass(frozen=True)
class FusionDecision:
    ts: float
    level: EscalationLevel
    reasons: list[str]
    cue_word: Optional[CueWordEvent]
    breathing: Optional[BreathingEstimate]
