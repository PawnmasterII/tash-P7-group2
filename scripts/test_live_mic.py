"""Interactive live-mic cue-word test.

Run from the parent directory (Desktop\\code\\):
    python -m tash.scripts.test_live_mic

Shows real-time audio levels, VAD state, and every word Vosk recognizes,
so you can verify the mic is picking up speech and the cue-word detector
fires on "help", "fine", "okay", and "ok".

Press Ctrl+C to stop.
"""
from __future__ import annotations

import json
import queue
import sys
import time

import numpy as np

from tash.audio import config
from tash.audio.stage2_cueword import CueWordDetector


def main() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed — run: pip install sounddevice")
        sys.exit(1)

    cue_words = config.CUE_WORDS + config.REASSURANCE_WORDS
    det = CueWordDetector(cue_words=cue_words)
    print(f"Cue words: {cue_words}")
    print(f"Min confidence: {config.CUE_WORD_MIN_CONFIDENCE}")
    print()

    audio_q: queue.Queue = queue.Queue(maxsize=200)

    def _callback(indata, frames, time_info, status) -> None:
        try:
            audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    chunk_samples = config.SIZES.vad_frame_samples
    stream = sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        blocksize=512,
        channels=1,
        dtype="int16",
        callback=_callback,
    )

    stream.start()
    print("Listening... say 'help', 'fine', 'okay', or 'ok'. Ctrl+C to stop.")
    print()

    ts = 0.0
    chunk_dur = 512 / config.SAMPLE_RATE
    last_print = 0.0

    try:
        while True:
            pcm = audio_q.get(timeout=5.0)
            ev = det.process(pcm, ts)

            rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))

            # Print level bar every 0.5s
            if ts - last_print >= 0.5:
                bar = "#" * min(int(rms * 300), 40)
                vad = "SPEECH" if det.speech_present else "      "
                print(f"  [{ts:5.1f}s] {vad} |{bar:<40s}| rms={rms:.4f}")
                last_print = ts

            if ev is not None:
                cat_tag = "DE-ESCALATE" if ev.category == "reassurance" else "DISTRESS    "
                print(
                    f"  ** {cat_tag} **  word={ev.keyword!r}  "
                    f"category={ev.category}  conf={ev.confidence_proxy:.3f}"
                )

            ts += chunk_dur
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
