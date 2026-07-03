"""Stage 1 — Denoising.

Suppress car noise while preserving voice and (faintly) breath sounds.
Emits two strengths: aggressive for the keyword path, gentle for breathing.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from tash.audio import config
from tash.audio.contracts import DenoisedChunk

try:
    import noisereduce as nr
except ImportError:  # keep the module import-safe without the dep installed
    nr = None


def _int16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return (pcm.astype(np.float32)) / 32768.0


def _float32_to_int16(x: np.ndarray) -> np.ndarray:
    # np.clip leaves NaN/inf untouched; scrub them or the int16 cast yields
    # garbage that corrupts downstream ASR. (NaNs arise from noisereduce's
    # divide-by-~0 on near-silent frames.)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


class Denoiser:
    """Stateful so it can carry a rolling noise profile and band-pass state."""

    def __init__(self) -> None:
        self._sos = signal.butter(
            config.BANDPASS_ORDER,
            [config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ],
            btype="band",
            fs=config.SAMPLE_RATE,
            output="sos",
        )
        self._noise_clip: np.ndarray | None = None
        self._noise_clip_ts: float = 0.0

    # -- noise baseline -----------------------------------------------------
    def set_noise_baseline(self, pcm_int16: np.ndarray, ts: float) -> None:
        """Capture/refresh the noise profile (engine-start, or VAD-silence)."""
        self._noise_clip = _int16_to_float32(np.asarray(pcm_int16))
        self._noise_clip_ts = ts

    def _profile_age(self, now: float) -> float:
        if self._noise_clip is None:
            return float("inf")
        return now - self._noise_clip_ts

    # -- core ---------------------------------------------------------------
    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        return signal.sosfilt(self._sos, x).astype(np.float32)

    def _reduce(self, x: np.ndarray, prop_decrease: float) -> np.ndarray:
        if nr is None:
            # Dependency absent: band-pass-only fallback keeps the pipeline live.
            return x
        kwargs = dict(
            y=x,
            sr=config.SAMPLE_RATE,
            stationary=False,
            prop_decrease=prop_decrease,
            n_fft=config.DENOISE_N_FFT,
            hop_length=config.DENOISE_HOP,
            time_constant_s=config.DENOISE_TIME_CONSTANT_S,
        )
        if self._noise_clip is not None:
            kwargs["y_noise"] = self._noise_clip
        out = nr.reduce_noise(**kwargs).astype(np.float32)
        # Near-silent frames make the spectral gate divide by ~0 -> NaN/inf.
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def process(self, pcm_int16: np.ndarray, ts: float) -> DenoisedChunk:
        """int16 chunk @16kHz -> DenoisedChunk (two strengths, shared ts)."""
        pcm_int16 = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        assert pcm_int16.size > 0, "empty audio chunk"

        x = self._bandpass(_int16_to_float32(pcm_int16))

        keyword = self._reduce(x, config.KEYWORD_PROP_DECREASE)
        breathing = self._reduce(x, config.BREATHING_PROP_DECREASE)

        return DenoisedChunk(
            ts=ts,
            int16=_float32_to_int16(keyword).tobytes(),
            float32=keyword,
            mild_float32=breathing,
            noise_profile_age_s=self._profile_age(ts),
        )
