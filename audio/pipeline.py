"""Orchestration — wires the three stages + fusion into one stream processor.

Note the deliberate one-frame lag: Stage 1's noise re-estimation consumes the
PREVIOUS chunk's `speech_present` (from Stage 2), breaking the apparent cycle
between S1 and S2 (architecture §7).
"""
from __future__ import annotations

import numpy as np

from tash.audio import config
from tash.audio.contracts import FusionDecision
from tash.audio.fusion import FusionEngine
from tash.audio.stage1_denoise import Denoiser
from tash.audio.stage2_cueword import CueWordDetector, CueWordError
from tash.audio.stage3_ml_breathing import MLBreathingDetector as BreathingDetector


class Pipeline:
    def __init__(self) -> None:
        self.denoiser = Denoiser()
        self.breathing = BreathingDetector()
        self.fusion = FusionEngine()
        self.degraded = False
        try:
            # Pass distress + reassurance words so the pipeline can recognise both;
            # VoiceResponseDetector uses cue.category to branch on armed state.
            self.cueword: CueWordDetector | None = CueWordDetector(
                cue_words=config.CUE_WORDS + config.REASSURANCE_WORDS
            )
        except CueWordError as e:  # fail loud, run degraded (no keyword path)
            print(f"[DEGRADED] cue-word stage disabled: {e}")
            self.cueword = None
            self.degraded = True
        self._prev_speech_present = False

    def prime_noise_baseline(self, pcm_int16: np.ndarray, ts: float = 0.0) -> None:
        self.denoiser.set_noise_baseline(pcm_int16, ts)

    def process_chunk(self, pcm_int16: np.ndarray, ts: float,
                      passenger_responded: bool | None = None) -> FusionDecision:
        # Stage 1 — uses previous chunk's speech flag for noise refresh timing.
        if (not self._prev_speech_present
                and self.denoiser._profile_age(ts) > config.NOISE_PROFILE_MAX_AGE_S):
            self.denoiser.set_noise_baseline(pcm_int16, ts)
        chunk = self.denoiser.process(pcm_int16, ts)

        # Stage 2 — updates speech_present as a side effect.
        # NOTE: Vosk is a noise-robust ASR trained on noisy speech; aggressive
        # spectral-gating denoise gives it no benefit and slightly degrades
        # recognition at larger chunk sizes (measured). So feed Stage 2 the RAW
        # signal. (Stage 1 still serves Stage 3 and any future wake-word engine
        # that *does* prefer denoise.)
        cue_event = None
        speech_present = False
        if self.cueword is not None:
            cue_event = self.cueword.process(pcm_int16, ts)
            speech_present = self.cueword.speech_present
        self._prev_speech_present = speech_present

        # Stage 3 — only on non-speech, gentle-denoise stream.
        breath_est = self.breathing.process(chunk.mild_float32, ts, speech_present)

        return self.fusion.decide(ts, cue_event, breath_est, passenger_responded)

    def flush(self, ts: float) -> FusionDecision:
        """End-of-stream: drain Stage 2's buffered utterance through fusion."""
        cue_event = self.cueword.flush(ts) if self.cueword is not None else None
        return self.fusion.decide(ts, cue_event, None)

    def close(self) -> None:
        if self.cueword is not None:
            self.cueword.close()
