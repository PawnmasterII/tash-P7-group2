"""Stage 3 (ML variant) -- 2-stage agonal breathing detector.

Drop-in replacement for stage3_breathing.BreathingDetector.
Same BreathingEstimate output contract.

Architecture (2-stage, following UW 2019 npj Digital Medicine)
--------------------------------------------------------------
Stage A -- Gasp Detector (ML, per 2.5 s window):
    Scores each 2.5 s sliding window: P(contains agonal gasp)
    Model: LinearSVC (Platt calibrated), log-mel + MFCC features

Stage B -- Temporal Rate Filter (rule-based, per 30 s buffer):
    Collects Stage-A high-confidence events over a rolling 30 s window.
    Rate 3-6 events/min + irregular spacing -> AGONAL_SUSPECT
    Rate = 0 (no events in 30 s of non-speech audio) -> APNEA candidate
    Regular events at 12-20 BPM -> NORMAL
    Anything else -> AMBIGUOUS

Why 2 stages?
    Agonal breathing is 3-6 BPM = one gasp every 10-20 s.  A single
    2.5 s window mostly catches SILENCE.  Stage A detects individual
    gasps; Stage B checks whether gasps are occurring at agonal rate.

Performance
-----------
    Stage A (per 2.5 s update): ~4-6 ms  [target: 20 ms]
    Stage B (temporal logic):   < 0.1 ms
    Total update every 0.5 s:  ~5 ms well within budget

Usage in pipeline.py
--------------------
    from tash.audio.stage3_ml_breathing import MLBreathingDetector as BreathingDetector
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tash.audio.contracts import BreathingEstimate, BreathingState
from tash.audio import config

_librosa = None
_joblib  = None

def _lazy_imports() -> None:
    global _librosa, _joblib
    if _librosa is None:
        import librosa as _lib
        _librosa = _lib
    if _joblib is None:
        import joblib as _jl
        _joblib = _jl


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "agonal_detector.joblib")

SR       = config.SAMPLE_RATE
CHUNK    = 512

# Stage A: 2.5 s sliding window, hop 0.5 s
WINDOW_S = 2.5
WINDOW_N = int(WINDOW_S * SR)    # 40 000 samples
HOP_A_S  = 0.5
HOP_A_N  = int(HOP_A_S * SR)    # 8 000 samples

# Stage B: temporal rate filter over a 30 s rolling event list
RATE_WINDOW_S  = 30.0            # collect events over this many seconds
AGONAL_BPM_MIN = 3.0             # agonal: 3-6 events/min
AGONAL_BPM_MAX = 6.0
NORMAL_BPM_MIN = 10.0
NORMAL_BPM_MAX = 25.0

# Stage A probability threshold to count an event
GASP_THRESH      = 0.55
HIGH_GASP_THRESH = 0.75          # immediate AGONAL_SUSPECT on very high confidence

# Persistence: require N consecutive positive Stage-B windows before escalating
PERSIST_N = 2


class MLBreathingDetector:
    """2-stage ML agonal breathing detector.  Matches BreathingDetector interface."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        _lazy_imports()
        try:
            bundle = _joblib.load(model_path)
            self._pipe  = bundle["pipeline"]
            self._win_n = bundle["window_n"]
            self._n_mels = bundle["n_mels"]
            self._n_mfcc = bundle["n_mfcc"]
            self._n_fft  = bundle["n_fft"]
            self._hop    = bundle["hop_len"]
            self._ready  = True
        except FileNotFoundError:
            print(f"[Stage3-ML] Model not found at {model_path}. "
                  f"Run: py -3.12 -m audio.train_agonal_detector")
            self._ready = False

        # Stage A: rolling audio buffer (2.5 s)
        self._buf: deque[float] = deque(maxlen=WINDOW_N)
        self._samples_since_a = 0        # since last Stage-A update

        # Stage B: timestamped event list (ts when gasp was detected)
        self._events: list[float] = []   # stream-relative seconds

        # Persistence filter (Stage B output)
        self._recent_b: deque[BreathingState] = deque(maxlen=PERSIST_N)
        self._last_prob = 0.0
        self._last_ts   = 0.0

        # Exponential moving average of chunk RMS — used to tell
        # silence (APNEA) from active audio (NORMAL) when no gasps detected
        self._rms_ema: float = 0.0
        self._rms_alpha: float = 0.05   # ~20 chunk time constant

    # ------------------------------------------------------------------
    # Feature extraction (must match train_agonal_detector.extract_features)
    # ------------------------------------------------------------------

    def _extract(self, win: np.ndarray) -> np.ndarray:
        """181-dim feature vector — must stay in sync with train_agonal_detector.extract_features."""
        y = win.astype(np.float32)
        if len(y) < self._win_n:
            y = np.pad(y, (0, self._win_n - len(y)))
        else:
            y = y[:self._win_n]

        mel = _librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=self._n_fft,
            hop_length=self._hop, n_mels=self._n_mels)
        log_mel = _librosa.power_to_db(mel, ref=np.max)
        mel_mean = log_mel.mean(axis=1)
        mel_std  = log_mel.std(axis=1)

        mfcc = _librosa.feature.mfcc(
            y=y, sr=SR, n_mfcc=self._n_mfcc,
            n_fft=self._n_fft, hop_length=self._hop)
        delta = _librosa.feature.delta(mfcc)
        mfcc_mean  = mfcc.mean(axis=1)
        mfcc_std   = mfcc.std(axis=1)
        delta_mean = delta.mean(axis=1)
        delta_std  = delta.std(axis=1)

        rms = _librosa.feature.rms(y=y, frame_length=1024, hop_length=self._hop)[0]
        sil_frac = float((rms < 0.008).mean())
        rms_feats = np.array([rms.mean(), rms.std(), rms.max(), sil_frac], dtype=np.float32)

        rms_fft = np.abs(np.fft.rfft(rms - rms.mean(), n=256))
        period_feats = rms_fft[:8].astype(np.float32)

        zcr     = _librosa.feature.zero_crossing_rate(y, hop_length=self._hop)[0]
        rolloff = _librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=self._hop)[0]
        misc_feats = np.array([zcr.mean(), zcr.std(), rolloff.mean(), rolloff.std()],
                               dtype=np.float32)

        # Spectral contrast (14 dims = 7 subbands × mean+std)
        contrast = _librosa.feature.spectral_contrast(
            y=y, sr=SR, n_fft=self._n_fft, hop_length=self._hop)
        contrast_mean = contrast.mean(axis=1).astype(np.float32)
        contrast_std  = contrast.std(axis=1).astype(np.float32)

        # F0 via YIN (80–400 Hz) — 3 dims: mean, std, voiced-frame fraction
        f0 = _librosa.yin(y, fmin=80.0, fmax=400.0, sr=SR, hop_length=self._hop)
        voiced_mask = f0 > 0
        f0_mean     = float(f0[voiced_mask].mean()) if voiced_mask.any() else 0.0
        f0_std      = float(f0[voiced_mask].std())  if voiced_mask.sum() > 1 else 0.0
        voiced_frac = float(voiced_mask.mean())
        f0_feats    = np.array([f0_mean, f0_std, voiced_frac], dtype=np.float32)

        # HPSS percussiveness — 4 dims: h_rms, p_rms, percussive ratio, p-dominant fraction
        H, P   = _librosa.effects.hpss(y)
        h_rms  = float(np.sqrt(np.mean(H ** 2))) + 1e-9
        p_rms  = float(np.sqrt(np.mean(P ** 2))) + 1e-9
        p_ratio = p_rms / (h_rms + p_rms)
        p_dom   = float(np.mean(P ** 2 > H ** 2))
        hpss_feats = np.array([h_rms, p_rms, p_ratio, p_dom], dtype=np.float32)

        return np.concatenate([
            mel_mean, mel_std,
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            rms_feats, period_feats, misc_feats,
            contrast_mean, contrast_std,
            f0_feats,
            hpss_feats,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage B: temporal rate classification
    # ------------------------------------------------------------------

    def _stage_b(self, ts: float) -> tuple[BreathingState, float]:
        """Classify based on recent gasp event rate."""
        # Prune old events outside the rate window
        cutoff = ts - RATE_WINDOW_S
        self._events = [t for t in self._events if t >= cutoff]

        n_events = len(self._events)
        elapsed  = min(ts, RATE_WINDOW_S)

        if elapsed < 10.0:
            return BreathingState.AMBIGUOUS, 0.3

        if n_events == 0:
            # Distinguish silence (true apnea) from active audio where no
            # gasps were detected (i.e. normal breathing — ML correctly
            # refused to label it as a gasp).
            if self._rms_ema > 0.01:
                return BreathingState.NORMAL, 0.60
            return BreathingState.APNEA, 0.4

        rate_bpm = (n_events / elapsed) * 60.0

        if AGONAL_BPM_MIN <= rate_bpm <= AGONAL_BPM_MAX:
            # Compute irregularity if we have multiple events
            if n_events >= 2:
                intervals = np.diff(sorted(self._events))
                cv = float(intervals.std() / (intervals.mean() + 1e-9))
                if cv > 0.35:          # irregular (agonal) vs regular (sleep apnea)
                    return BreathingState.AGONAL_SUSPECT, 0.70
            return BreathingState.LOW_RATE, 0.55

        if NORMAL_BPM_MIN <= rate_bpm <= NORMAL_BPM_MAX:
            return BreathingState.NORMAL, 0.70

        return BreathingState.AMBIGUOUS, 0.35

    # ------------------------------------------------------------------
    # Persistence filter
    # ------------------------------------------------------------------

    def _persist(self, state: BreathingState) -> BreathingState:
        self._recent_b.append(state)
        if len(self._recent_b) < self._recent_b.maxlen:
            return BreathingState.AMBIGUOUS
        if all(s == state for s in self._recent_b):
            return state
        return BreathingState.AMBIGUOUS

    # ------------------------------------------------------------------
    # Public interface (matches BreathingDetector.process)
    # ------------------------------------------------------------------

    def process(self,
                mild_float32: np.ndarray,
                ts: float,
                speech_present: bool) -> BreathingEstimate | None:
        """Gentle-denoised float32 chunk -> BreathingEstimate or None."""
        if speech_present:
            return None

        self._last_ts = ts
        chunk = mild_float32.ravel().astype(np.float64)
        self._buf.extend(chunk)
        self._samples_since_a += len(chunk)
        chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
        self._rms_ema = self._rms_alpha * chunk_rms + (1 - self._rms_alpha) * self._rms_ema

        if len(self._buf) < WINDOW_N:
            return None

        # Run Stage A every HOP_A_N samples (0.5 s)
        if self._samples_since_a < HOP_A_N:
            return None
        self._samples_since_a = 0

        if not self._ready:
            return BreathingEstimate(
                ts=ts, state=BreathingState.AMBIGUOUS,
                resp_rate_bpm=None, interval_cv=None, confidence=0.1,
                features={"ml_ready": False})

        win = np.array(self._buf, dtype=np.float32)
        feat = self._extract(win).reshape(1, -1)
        prob = float(self._pipe.predict_proba(feat)[0, 1])
        self._last_prob = prob

        # Register gasp event if confident enough
        if prob >= GASP_THRESH:
            # Debounce: ignore if we just registered an event < 2 s ago
            if not self._events or (ts - self._events[-1]) >= 2.0:
                self._events.append(ts)

        # Immediate high-confidence escalation (bypass temporal filter)
        if prob >= HIGH_GASP_THRESH:
            state = BreathingState.AGONAL_SUSPECT
            conf  = prob
            state = self._persist(state)
        else:
            raw_state, conf = self._stage_b(ts)
            state = self._persist(raw_state)

        return BreathingEstimate(
            ts=ts,
            state=state,
            resp_rate_bpm=None,
            interval_cv=None,
            confidence=conf,
            features={
                "gasp_prob": round(prob, 4),
                "n_events_30s": len(self._events),
                "ml_model": "gasp_detector_gbm",
            },
        )
