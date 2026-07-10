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
from tash.audio.test_audio import resolve_test_audio_dir

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import sounddevice as _sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _sd = None
    _SOUNDDEVICE_AVAILABLE = False

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
        Directory containing *.wav files for replay mode. If None, uses
        ``audio/test_audio/`` in this repo, then sibling test-audio dirs.
    device_index : int
        pvrecorder device index (-1 = system default). Only used in live mode.
    loop : bool
        WAV mode only. If True, replay files in a loop indefinitely.
        Default False (replay once then stop).
    include : list[str] | None
        WAV mode only. If set, replay only these basenames (e.g.
        ``["demo_agonal_real.wav"]``).
    """

    modality = Modality.MICROPHONE

    def __init__(
        self,
        mode: str = "wav",
        wav_dir: str | None = None,
        device_index: int = -1,
        loop: bool = False,
        include: list[str] | None = None,
    ) -> None:
        if mode not in ("wav", "live"):
            raise ValueError(f"mode must be 'wav' or 'live', got {mode!r}")
        self._mode = mode
        self._wav_dir = _resolve_wav_dir(wav_dir)
        self._device_index = device_index
        self._loop = loop
        self._include = set(include) if include else None

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
        if self._include is not None:
            files = [f for f in files if os.path.basename(f) in self._include]
        if not files:
            raise RuntimeError(
                f"No *.wav files found in {self._wav_dir}. "
                "Run: py -3.12 scripts/sync_demo_audio.py"
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
        if _SOUNDDEVICE_AVAILABLE:
            async for reading in self._live_stream_sounddevice():
                yield reading
        elif _PVRECORDER_AVAILABLE:
            async for reading in self._live_stream_pvrecorder():
                yield reading
        else:
            raise RuntimeError(
                "Neither sounddevice nor pvrecorder installed — "
                "use mode='wav' or: pip install sounddevice"
            )

    async def _live_stream_sounddevice(self) -> AsyncIterator[SensorReading]:
        import queue as _queue

        audio_q: _queue.Queue[np.ndarray] = _queue.Queue(maxsize=64)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            try:
                audio_q.put_nowait(indata[:, 0].copy())
            except _queue.Full:
                pass

        stream = _sd.InputStream(
            samplerate=_SAMPLE_RATE,
            blocksize=_CHUNK_SAMPLES,
            channels=1,
            dtype="int16",
            callback=_callback,
        )
        stream.start()
        stream_ts = 0.0
        try:
            while True:
                pcm = await asyncio.to_thread(audio_q.get)
                yield SensorReading(
                    modality=self.modality,
                    payload={
                        "frame": pcm.tobytes(),
                        "sample_rate": _SAMPLE_RATE,
                        "ts": stream_ts,
                    },
                )
                stream_ts += _CHUNK_DURATION_S
        finally:
            stream.stop()
            stream.close()

    async def _live_stream_pvrecorder(self) -> AsyncIterator[SensorReading]:
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
    resolved = resolve_test_audio_dir()
    if resolved is not None:
        return resolved
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "audio", "test_audio"))
