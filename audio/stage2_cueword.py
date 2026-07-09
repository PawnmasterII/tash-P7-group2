"""Stage 2 — Cue word detection (Vosk offline ASR + energy/VAD speech gate).

Vosk runs fully offline (no key/signup) and recognizes arbitrary words — so the
distress word "help" works with no custom model and no training. We use the
model's FULL vocabulary and match cue words as whole tokens in the transcript;
a restricted grammar would force-decode noise into "help" (see __init__).

The VAD gate both restricts what reaches the recognizer and publishes
`speech_present` for Stage 3 (breathing analysis must skip speech).
"""
from __future__ import annotations

import json
import os

import numpy as np

from tash.audio import config
from tash.audio.contracts import CueWordEvent

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import vosk
    vosk.SetLogLevel(-1)            # silence Kaldi's verbose stderr
except ImportError:
    vosk = None


class CueWordError(RuntimeError):
    """Raised loud on init/runtime failure so fusion can flag degraded mode."""


class _EnergyVad:
    """Coarse RMS gate used only when webrtcvad can't be installed.

    Same interface as webrtcvad.Vad.is_speech(frame_bytes, sample_rate) so the
    rest of Stage 2 is unchanged. Bring-up aid, NOT a production VAD.
    """

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        x = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if x.size == 0:
            return False
        rms = float(np.sqrt(np.mean(x ** 2)))
        return rms > config.ENERGY_VAD_RMS_THRESHOLD


class _Reblocker:
    """Re-block an int16 stream into fixed-size frames (sample-accurate)."""

    def __init__(self, frame_samples: int) -> None:
        self._n = frame_samples
        self._buf = np.empty(0, dtype=np.int16)

    def push(self, pcm: np.ndarray):
        self._buf = np.concatenate([self._buf, pcm.astype(np.int16)])
        while self._buf.size >= self._n:
            frame, self._buf = self._buf[: self._n], self._buf[self._n :]
            assert frame.size == self._n
            yield frame


class CueWordDetector:
    """Vosk-backed cue-word spotter. No access key required."""

    def __init__(self, cue_words: list[str] | None = None) -> None:
        if vosk is None:
            raise CueWordError("vosk not installed (required for Stage 2)")
        if not os.path.isdir(config.VOSK_MODEL_PATH):
            raise CueWordError(
                f"Vosk model not found at {config.VOSK_MODEL_PATH} — "
                "download from https://alphacephei.com/vosk/models and unpack there"
            )

        try:
            self._model = vosk.Model(config.VOSK_MODEL_PATH)
            # FULL vocabulary (no restricted grammar): a 2-word grammar force-
            # decodes noise into the cue word and makes confidence meaningless
            # (always 1.0). With full vocab, noise transcribes to other words
            # ("huh"), so a cue match is genuine and its confidence is real.
            self._rec = vosk.KaldiRecognizer(self._model, config.SAMPLE_RATE)
            self._rec.SetWords(True)         # per-word confidences in final results
        except Exception as e:
            raise CueWordError(f"Vosk init failed: {e}") from e

        if webrtcvad is not None:
            self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        else:
            print("[Stage2] webrtcvad unavailable -> coarse energy VAD fallback")
            self._vad = _EnergyVad()

        self._vad_reblock = _Reblocker(config.SIZES.vad_frame_samples)
        self._hangover = 0
        self._speech_present = False
        self._last_event_ts = -1e9
        self._cue_set = {w.lower() for w in (cue_words if cue_words is not None else config.CUE_WORDS)}
        self._reassurance_set = {w.lower() for w in config.REASSURANCE_WORDS}

    @property
    def speech_present(self) -> bool:
        return self._speech_present

    def _update_vad(self, pcm_int16: np.ndarray) -> None:
        for frame in self._vad_reblock.push(pcm_int16):
            if self._vad.is_speech(frame.tobytes(), config.SAMPLE_RATE):
                self._hangover = config.SIZES.vad_hangover_frames
            elif self._hangover > 0:
                self._hangover -= 1
        self._speech_present = self._hangover > 0

    def _debounced(self, ts: float) -> bool:
        return (ts - self._last_event_ts) * 1000 >= config.CUE_WORD_DEBOUNCE_MS

    def _match_final(self, result_json: str) -> tuple[str, float] | None:
        """Whole-utterance result: match a cue token with its confidence."""
        res = json.loads(result_json or "{}")
        words = res.get("result", [])
        for w in words:
            token = w.get("word", "").lower()
            conf = float(w.get("conf", 0.0))
            if token in self._cue_set and conf >= config.CUE_WORD_MIN_CONFIDENCE:
                return token, conf
        return None

    def process(self, pcm_int16, ts: float) -> CueWordEvent | None:
        """int16 chunk @16kHz -> CueWordEvent (or None). Updates speech_present."""
        pcm_int16 = np.asarray(pcm_int16, dtype=np.int16).reshape(-1)
        was_speech = self._speech_present
        self._update_vad(pcm_int16)

        # Emit only on confident FINAL results, not eager partials. Partials
        # flicker to "help" on noise; finals carry per-word confidence we can
        # threshold. Trade-off: detection fires at the word's end, not mid-word.
        hit: tuple[str, float] | None = None
        # Always send audio to Vosk — the VAD gate was too aggressive for live
        # mics and blocked short words ("fine", "okay") from ever reaching the
        # recognizer. The full-vocabulary model + CUE_WORD_MIN_CONFIDENCE
        # threshold handles false positives from silence/noise (the gate was
        # originally for a restricted grammar that force-decoded noise to
        # "help", but this pipeline uses the full vocabulary).
        if self._rec.AcceptWaveform(pcm_int16.tobytes()):   # utterance end
            hit = self._match_final(self._rec.Result())
        elif was_speech and not self._speech_present:
            # Falling edge: flush the buffered utterance BEFORE it's discarded
            # (FinalResult also resets the recognizer for the next utterance).
            hit = self._match_final(self._rec.FinalResult())

        if hit is not None and self._debounced(ts):
            self._last_event_ts = ts
            keyword, conf = hit
            category = "reassurance" if keyword in self._reassurance_set else "distress"
            return CueWordEvent(
                ts=ts,
                keyword=keyword,
                vad_active=self._speech_present,
                sensitivity=config.CUE_WORD_MIN_CONFIDENCE,
                confidence_proxy=conf,
                category=category,
            )
        return None

    def flush(self, ts: float) -> CueWordEvent | None:
        """Force-decode any buffered audio. Call at end-of-stream / shutdown:
        a finite recording may end mid-utterance (no speech->silence edge), so
        the last words would otherwise never be read out of the recognizer.
        In live use the falling edge normally handles this; flush is the
        finite-stream safety net."""
        hit = self._match_final(self._rec.FinalResult())
        if hit is not None and self._debounced(ts):
            self._last_event_ts = ts
            keyword, conf = hit
            category = "reassurance" if keyword in self._reassurance_set else "distress"
            return CueWordEvent(ts=ts, keyword=keyword, vad_active=False,
                                sensitivity=config.CUE_WORD_MIN_CONFIDENCE,
                                confidence_proxy=conf, category=category)
        return None

    def close(self) -> None:
        self._rec = None
        self._model = None
