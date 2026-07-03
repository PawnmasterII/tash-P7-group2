"""Microphone sensor: WAV-replay (default) or live pvrecorder capture.

WAV replay feeds *.wav files from `wav_dir` in sorted order, 512-sample
chunks at 16 kHz, paced to real time. Live capture uses pvrecorder. The
default is WAV replay so the demo runs without a physical microphone.

Payload schema (all modes):
    {
        "frame":       <bytes>  # int16 PCM, 512 samples @ 16 kHz
        "sample_rate": 16000
        "ts":          <float>  # stream-relative seconds (chunk start)
    }

WAV replay also adds:
    {
        "file":        <str>   # basename of current WAV file
        "is_first":    <bool>  # True on the first chunk of each file
    }
"""
from __future__ import annotations

import asyncio
import glob
import os
from collections.abc import AsyncIterator

import numpy as np

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pvrecorder as _pvr
    _PVRECORDER_AVAILABLE = True
except ImportError:
    _pvr = None
    _PVRECORDER_AVAILABLE = False

_SAMPLE_RATE = 16_000
_CHUNK_SAMPLES = 512
_CHUNK_DURATION_S = _CHUNK_SAMPLES / _SAMPLE_RATE   # 0.032 s


class Microphone(Sensor):
    """16 kHz mono int16 PCM source.

    Parameters
    ----------
    mode : "wav" | "live"
        "wav"  — replay WAV files from *wav_dir* (default, no mic required)
        "live" — capture from the default audio device via pvrecorder
    wav_dir : str | None
        Directory containing *.wav files for replay mode. If None, the sensor
        looks for a ``test_audio/`` directory adjacent to the TASHaudio repo
        (sibling to this package at ``../../TASHaudio/test_audio``).
    device_index : int
        pvrecorder device index (-1 = system default). Only used in live mode.
    loop : bool
        WAV mode only. If True, replay files in a loop indefinitely.
        Default False (replay once then stop).
    """

    modality = Modality.MICROPHONE

    def __init__(
        self,
        mode: str = "wav",
        wav_dir: str | None = None,
        device_index: int = -1,
        loop: bool = False,
    ) -> None:
        if mode not in ("wav", "live"):
            raise ValueError(f"mode must be 'wav' or 'live', got {mode!r}")
        self._mode = mode
        self._wav_dir = _resolve_wav_dir(wav_dir)
        self._device_index = device_index
        self._loop = loop

    def stream(self) -> AsyncIterator[SensorReading]:  # type: ignore[override]
        if self._mode == "live":
            return self._live_stream()
        return self._wav_stream()

    # -------------------------------------------------------------------------
    # WAV replay
    # -------------------------------------------------------------------------

    async def _wav_stream(self) -> AsyncIterator[SensorReading]:
        if sf is None:
            raise RuntimeError(
                "soundfile not installed — cannot replay WAV files. "
                "Install it with: pip install soundfile"
            )
        files = sorted(glob.glob(os.path.join(self._wav_dir, "*.wav")))
        if not files:
            raise RuntimeError(
                f"No *.wav files found in {self._wav_dir}. "
                "Copy or symlink TASHaudio/test_audio/ there, or pass wav_dir= explicitly."
            )

        first_pass = True
        while first_pass or self._loop:
            first_pass = False
            for path in files:
                pcm, sr = sf.read(path, dtype="int16", always_2d=True)
                if sr != _SAMPLE_RATE:
                    raise RuntimeError(
                        f"{path}: sample_rate={sr}, expected {_SAMPLE_RATE} Hz"
                    )
                pcm = pcm[:, 0]  # ensure mono

                stream_ts = 0.0
                is_first = True
                n = len(pcm)
                for i in range(0, n - _CHUNK_SAMPLES + 1, _CHUNK_SAMPLES):
                    chunk = pcm[i : i + _CHUNK_SAMPLES]
                    yield SensorReading(
                        modality=self.modality,
                        payload={
                            "frame": chunk.tobytes(),
                            "sample_rate": _SAMPLE_RATE,
                            "ts": stream_ts,
                            "file": os.path.basename(path),
                            "is_first": is_first,
                        },
                    )
                    is_first = False
                    stream_ts += _CHUNK_DURATION_S
                    # Pace to real time so the asyncio event loop stays responsive
                    # and the 30s risk window advances at the correct rate.
                    await asyncio.sleep(_CHUNK_DURATION_S)

    # -------------------------------------------------------------------------
    # Live capture
    # -------------------------------------------------------------------------

    async def _live_stream(self) -> AsyncIterator[SensorReading]:
        if not _PVRECORDER_AVAILABLE:
            raise RuntimeError(
                "pvrecorder not installed — use mode='wav' or: pip install pvrecorder"
            )
        recorder = _pvr.PvRecorder(
            device_index=self._device_index, frame_length=_CHUNK_SAMPLES
        )
        recorder.start()
        stream_ts = 0.0
        try:
            while True:
                frame = recorder.read()         # blocking; returns list[int]
                pcm = np.array(frame, dtype=np.int16)
                yield SensorReading(
                    modality=self.modality,
                    payload={
                        "frame": pcm.tobytes(),
                        "sample_rate": _SAMPLE_RATE,
                        "ts": stream_ts,
                    },
                )
                stream_ts += _CHUNK_DURATION_S
                await asyncio.sleep(0)          # yield to event loop between reads
        finally:
            recorder.stop()
            recorder.delete()


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------

def _resolve_wav_dir(wav_dir: str | None) -> str:
    if wav_dir is not None:
        return wav_dir
    # Default: look next to the TASHaudio repo (sibling to tash-P7-group2).
    # Layout expected:
    #   Desktop/code/
    #     TASHaudio/test_audio/   ← WAV files live here
    #     tash-P7-group2/         ← this package
    here = os.path.dirname(os.path.abspath(__file__))   # .../tash-P7-group2/sensors/
    candidate = os.path.normpath(os.path.join(here, "..", "..", "TASHaudio", "test_audio"))
    return candidate
