"""Stage 3 — Breathing detection (ADVISORY heuristic, not a medical classifier).

No validated pre-trained agonal-breathing model exists, and training is
forbidden. This is interpretable signal heuristics: envelope -> breath onsets
-> respiration rate / rhythm -> rule-based state. Output feeds the fusion
layer with confidence; it must NEVER auto-dispatch on its own.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal

from tash.audio import config
from tash.audio.contracts import BreathingEstimate, BreathingState

try:
    import librosa
except ImportError:
    librosa = None


class BreathingDetector:
    def __init__(self) -> None:
        self._sos = signal.butter(
            config.BANDPASS_ORDER,
            [config.BREATH_BAND_LOW_HZ, config.BREATH_BAND_HIGH_HZ],
            btype="band",
            fs=config.SAMPLE_RATE,
            output="sos",
        )
        self._window = deque(maxlen=int(config.RESP_WINDOW_S * config.SAMPLE_RATE))
        self._recent_states: deque[BreathingState] = deque(
            maxlen=config.RESP_PERSISTENCE_WINDOWS
        )

    def _rms_envelope(self, x: np.ndarray) -> np.ndarray:
        if librosa is not None:
            return librosa.feature.rms(
                y=x, frame_length=config.RESP_RMS_FRAME, hop_length=config.RESP_RMS_HOP
            )[0]
        # Fallback envelope if librosa absent.
        hop, n = config.RESP_RMS_HOP, config.RESP_RMS_FRAME
        out = [np.sqrt(np.mean(x[i : i + n] ** 2)) for i in range(0, len(x) - n, hop)]
        return np.asarray(out, dtype=np.float32)

    def _detect_breaths(self, env: np.ndarray) -> np.ndarray:
        """Peak times (s) in the envelope; guarded against empty/flat input."""
        if env.size < 3 or not np.any(env > 0):
            return np.empty(0)
        thresh = env.mean() + 0.5 * env.std()
        # Min spacing ~1.5 s between breaths (caps absurd rates).
        hop_s = config.RESP_RMS_HOP / config.SAMPLE_RATE
        distance = max(1, int(1.5 / hop_s))
        peaks, _ = signal.find_peaks(env, height=thresh, distance=distance)
        return peaks * hop_s

    def _classify(self, rate: float | None, cv: float | None,
                  speech_present: bool) -> tuple[BreathingState, float]:
        if rate is None:
            # No breaths AND no speech is genuinely ambiguous (silence vs apnea).
            return (BreathingState.APNEA if not speech_present
                    else BreathingState.AMBIGUOUS), 0.3
        if rate < config.RESP_RATE_LOW:
            # Slow + irregular gasps = agonal suspicion.
            if cv is not None and cv > 0.5:
                return BreathingState.AGONAL_SUSPECT, 0.6
            return BreathingState.LOW_RATE, 0.5
        if config.RESP_RATE_NORMAL_MIN <= rate <= config.RESP_RATE_NORMAL_MAX:
            return BreathingState.NORMAL, 0.7
        return BreathingState.AMBIGUOUS, 0.4

    def _persist(self, state: BreathingState) -> BreathingState:
        """Require N consecutive windows before honoring a non-normal state."""
        self._recent_states.append(state)
        if len(self._recent_states) < self._recent_states.maxlen:
            return BreathingState.AMBIGUOUS
        if all(s == state for s in self._recent_states):
            return state
        return BreathingState.AMBIGUOUS

    def process(self, mild_float32: np.ndarray, ts: float,
                speech_present: bool) -> BreathingEstimate | None:
        """Gentle-denoise float chunk -> BreathingEstimate. Skips speech."""
        if speech_present:
            return None  # respiration analysis only on non-speech segments

        self._window.extend(np.asarray(mild_float32, dtype=np.float32))
        if len(self._window) < self._window.maxlen:
            return None  # not enough history to estimate a slow rate yet

        x = self._bandpass(np.asarray(self._window, dtype=np.float32))
        env = self._rms_envelope(x)
        breaths = self._detect_breaths(env)

        rate = cv = None
        if breaths.size >= 2:
            intervals = np.diff(breaths)
            rate = 60.0 / intervals.mean() if intervals.mean() > 0 else None
            cv = float(intervals.std() / intervals.mean()) if intervals.mean() > 0 else None

        raw_state, conf = self._classify(rate, cv, speech_present)
        state = self._persist(raw_state)

        return BreathingEstimate(
            ts=ts,
            state=state,
            resp_rate_bpm=rate,
            interval_cv=cv,
            confidence=conf,
            features={"n_breaths": int(breaths.size), "raw_state": raw_state.value},
        )

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        return signal.sosfilt(self._sos, x).astype(np.float32)
