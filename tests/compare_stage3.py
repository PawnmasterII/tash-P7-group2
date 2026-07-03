"""Compare heuristic Stage 3 vs ML Stage 3 on identical synthetic scenarios.

Run:  py -3.12 tests/compare_stage3.py
"""
from __future__ import annotations
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from scipy import signal as sp_signal
from tash.audio.config import SAMPLE_RATE as SR
from tash.audio.contracts import BreathingState
from tash.audio.stage3_breathing import BreathingDetector
from tash.audio.stage3_ml_breathing import MLBreathingDetector

CHUNK = 512


def _bandpass(x, lo, hi):
    sos = sp_signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sp_signal.sosfilt(sos, x).astype(np.float32)


def _gen_agonal(duration_s, rng):
    n = int(duration_s * SR)
    audio = np.zeros(n, dtype=np.float32)
    t = 0.0
    while t < duration_s:
        gap_s = rng.uniform(6.0, 18.0)
        t += gap_s
        if t >= duration_s:
            break
        gasp_s = rng.uniform(0.3, 1.5)
        gasp_n = int(gasp_s * SR)
        start = int(t * SR)
        end = min(start + gasp_n, n)
        carrier = _bandpass((np.random.randn(gasp_n) * 0.15).astype(np.float32), 100, 2000)
        attack = max(1, int(0.05 * gasp_n))
        decay = gasp_n - attack
        env = np.concatenate([np.linspace(0, 1, attack),
                               np.exp(-3.0 * np.linspace(0, 1, decay))]).astype(np.float32)
        audio[start:end] += carrier[:end-start] * env[:end-start] * rng.uniform(0.4, 1.0)
        t += gasp_s
    audio += (np.random.randn(n) * 0.003).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _gen_normal(duration_s, rng, bpm=16.0):
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n)
    env = np.clip(np.sin(2 * np.pi * (bpm / 60.0) * t), 0, 1).astype(np.float32)
    carrier = _bandpass((np.random.randn(n) * 0.06).astype(np.float32), 80, 1800)
    return np.clip(carrier * env, -1, 1)


def run_detector(det, audio, label=""):
    """Feed audio as 32ms chunks; return final BreathingState and p95 latency."""
    chunks = [audio[i:i+CHUNK].astype(np.float32) for i in range(0, len(audio)-CHUNK, CHUNK)]
    last_state = None
    lats = []
    for i, chunk in enumerate(chunks):
        ts = i * CHUNK / SR
        mild = chunk / 32768.0 if chunk.max() > 1.5 else chunk
        t0 = time.perf_counter()
        est = det.process(mild, ts, speech_present=False)
        lats.append((time.perf_counter() - t0) * 1000)
        if est is not None:
            last_state = est.state
    lat95 = float(np.percentile(lats, 95))
    return last_state, lat95


scenarios = [
    ("30 BPM tachypnea",  lambda rng: _gen_normal(70, rng, bpm=30), [BreathingState.AMBIGUOUS]),
    ("16 BPM normal",     lambda rng: _gen_normal(70, rng, bpm=16), [BreathingState.NORMAL]),
    ("12 BPM normal",     lambda rng: _gen_normal(70, rng, bpm=12), [BreathingState.NORMAL]),
    ("6 BPM agonal",      lambda rng: _gen_normal(70, rng, bpm=6),  [BreathingState.LOW_RATE, BreathingState.AGONAL_SUSPECT]),
    ("agonal (realistic)",lambda rng: _gen_agonal(70, rng),         [BreathingState.AGONAL_SUSPECT, BreathingState.APNEA]),
    ("silence",           lambda rng: np.zeros(int(70 * SR), dtype=np.float32), [BreathingState.AMBIGUOUS, BreathingState.APNEA]),
]

print("Stage 3 Comparison: Heuristic vs ML")
print(f"{'Scenario':<24} {'Heuristic':<17} {'ML':<17} {'Expected'}")
print("-" * 80)

heur_pass = ml_pass = 0
rng = np.random.default_rng(99)

for label, audio_fn, expected in scenarios:
    audio = audio_fn(rng)
    audio_int16 = (audio * 32767).astype(np.int16) if audio.max() <= 1.0 else audio.astype(np.int16)

    h_det = BreathingDetector()
    m_det = MLBreathingDetector()

    h_state, h_lat = run_detector(h_det, audio)
    m_state, m_lat = run_detector(m_det, audio)

    h_ok = h_state in expected
    m_ok = m_state in expected
    heur_pass += int(h_ok)
    ml_pass   += int(m_ok)

    h_str = f"{h_state.value if h_state else 'None'} {'[OK]' if h_ok else '[!!]'}"
    m_str = f"{m_state.value if m_state else 'None'} {'[OK]' if m_ok else '[!!]'}"
    exp_str = [s.value for s in expected]
    print(f"  {label:<22} {h_str:<17} {m_str:<17} {exp_str}")

n = len(scenarios)
print("-" * 80)
print(f"  TOTAL                  heuristic {heur_pass}/{n}          ML {ml_pass}/{n}")
