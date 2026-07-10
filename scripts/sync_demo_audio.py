"""Copy a short real agonal clip from data/agonal_real into audio/test_audio/.

Builds ``demo_agonal_real.wav`` (~20 s) by trimming the start of the shortest
full recording in data/agonal_real/ (``agonal_aug_faster.wav``, ~43 s).

These ``demo_*.wav`` files are git-tracked and used by the live demo (key 1),
``tash.demo`` scenario 5, and ``tash.main`` WAV replay — no sibling repo needed.

Usage:
    py -3.12 scripts/sync_demo_audio.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO, "data", "agonal_real")
OUT_DIR = os.path.join(REPO, "audio", "test_audio")

# Shortest full clip in data/agonal_real/; first 20 s has the most gasp activity.
DEMO_SOURCE = "agonal_aug_faster.wav"
DEMO_DEST = "demo_agonal_real.wav"
DEMO_DURATION_S = 20.0
SAMPLE_RATE = 16_000


def main() -> int:
    if not os.path.isdir(SRC_DIR):
        print(f"Source not found: {SRC_DIR}")
        print("Run scripts/download_data.py first, or add clips to data/agonal_real/.")
        return 1

    src = os.path.join(SRC_DIR, DEMO_SOURCE)
    if not os.path.isfile(src):
        print(f"Source clip not found: {src}")
        return 1

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed — pip install soundfile")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, DEMO_DEST)

    audio, sr = sf.read(src, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        print(f"Expected {SAMPLE_RATE} Hz, got {sr} in {DEMO_SOURCE}")
        return 1
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    n = min(len(audio), int(DEMO_DURATION_S * sr))
    clip = audio[:n]
    sf.write(dest, clip, sr, subtype="PCM_16")

    dur_s = len(clip) / sr
    size_kb = os.path.getsize(dest) // 1024
    print(f"  {DEMO_DEST}  ({dur_s:.1f}s, {size_kb} KB)  <-  {DEMO_SOURCE} [0:{DEMO_DURATION_S:.0f}s]")

    stale = os.path.join(OUT_DIR, "demo_agonal_clean.wav")
    if os.path.isfile(stale):
        os.remove(stale)
        print(f"  removed stale {os.path.basename(stale)}")

    print(f"\nDone — demo clip in audio/test_audio/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
