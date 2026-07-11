"""Demo / replay WAV discovery and safe speaker playback.

In-repo demo clips live in ``audio/test_audio/demo_*.wav`` (git-tracked).
They are copied from ``data/agonal_real/`` via ``scripts/sync_demo_audio.py``.

Fallback order when resolving a replay directory:
  1. ``audio/test_audio/`` inside this repo
  2. sibling ``tash-audio-pipeline/test_audio/scenarios``
  3. sibling ``TASHaudio/test_audio``
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable

log = logging.getLogger(__name__)

_AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_TEST_AUDIO_DIR = os.path.join(_AUDIO_DIR, "test_audio")

# Prefer demo_*.wav clips; fall back to any *.wav in the resolved directory.
_DEMO_GLOB = "demo_*.wav"
_WAV_GLOB = "*.wav"
PRIMARY_DEMO_WAV = "demo_agonal_real.wav"
DEMO_PLAYBACK_LAST_SECONDS = 10.0


def _repo_root() -> str:
    return os.path.dirname(_AUDIO_DIR)


def test_audio_candidates() -> list[str]:
    """Ordered list of directories that may contain replay/demo WAV files."""
    root = _repo_root()
    return [
        REPO_TEST_AUDIO_DIR,
        os.path.normpath(os.path.join(root, "..", "tash-audio-pipeline", "test_audio", "scenarios")),
        os.path.normpath(os.path.join(root, "..", "TASHaudio", "test_audio")),
    ]


def resolve_test_audio_dir() -> str | None:
    """Return the first existing test-audio directory that contains WAV files."""
    import glob

    for directory in test_audio_candidates():
        if not os.path.isdir(directory):
            continue
        if glob.glob(os.path.join(directory, _WAV_GLOB)):
            return directory
    return None


def primary_demo_wav(directory: str | None = None) -> str | None:
    """Basename of the main demo clip, if present in the resolved directory."""
    directory = directory or resolve_test_audio_dir()
    if directory is None:
        return None
    path = os.path.join(directory, PRIMARY_DEMO_WAV)
    return PRIMARY_DEMO_WAV if os.path.isfile(path) else None


def list_demo_wavs(directory: str | None = None) -> list[str]:
    """Sorted basenames of demo WAV files (prefers ``demo_*.wav``)."""
    import glob

    directory = directory or resolve_test_audio_dir()
    if directory is None:
        return []
    demo = sorted(os.path.basename(p) for p in glob.glob(os.path.join(directory, _DEMO_GLOB)))
    if demo:
        return demo
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(directory, _WAV_GLOB)))


def load_for_playback(path: str, *, last_seconds: float | None = None) -> tuple["object", int]:
    """Load a WAV as mono float32 for speaker playback.

    If *last_seconds* is set, only the final N seconds are returned.
    """
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile not installed — pip install soundfile") from exc

    data, fs = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if last_seconds is not None and last_seconds > 0 and len(data) > 0:
        max_samples = int(last_seconds * fs)
        if len(data) > max_samples:
            data = data[-max_samples:]
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 1.0:
        data = data / peak
    return data, int(fs)


class WavPlayer:
    """Single-threaded speaker playback — prevents overlapping sd.play() crashes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop.clear()

    def play(
        self,
        path: str,
        *,
        last_seconds: float | None = DEMO_PLAYBACK_LAST_SECONDS,
        on_start: Callable[[str], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> bool:
        """Play *path* in a background thread. Returns False if already playing."""
        if not os.path.isfile(path):
            log.warning("WAV not found: %s", path)
            return False

        with self._lock:
            if self.is_playing:
                self.stop()
            self._thread = threading.Thread(
                target=self._worker,
                args=(path, last_seconds, on_start, on_end),
                daemon=True,
                name="tash-wav-player",
            )
            self._thread.start()
            return True

    def _worker(
        self,
        path: str,
        last_seconds: float | None,
        on_start: Callable[[str], None] | None,
        on_end: Callable[[], None] | None,
    ) -> None:
        basename = os.path.basename(path)
        try:
            import sounddevice as sd

            data, fs = load_for_playback(path, last_seconds=last_seconds)
            if on_start is not None:
                on_start(basename)
            log.info(
                "Playing test audio: %s (last %.1fs)",
                path,
                last_seconds if last_seconds is not None else len(data) / fs,
            )

            block = max(1024, int(fs * 0.05))
            with sd.OutputStream(samplerate=fs, channels=1, dtype="float32") as stream:
                for i in range(0, len(data), block):
                    if self._stop.is_set():
                        break
                    stream.write(data[i : i + block])
        except Exception as exc:
            log.warning("Failed to play %s: %s", path, exc)
        finally:
            if on_end is not None:
                on_end()
