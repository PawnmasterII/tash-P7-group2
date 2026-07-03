"""Audio pipeline benchmark -- measures real latency and detection accuracy.

Run from the repo root with Python 3.12:
    py -3.12 tests/bench_audio_pipeline.py [--quick] [--wav-dir PATH]

What it tests
-------------
Stage 1  Denoiser      latency per chunk, noise suppression (dB)
Stage 2  CueWord       false-positive rate on silence/white/pink noise
                       true-positive rate (optional --wav-dir with speech WAVs)
Stage 3  Breathing     accuracy on synthetic sine-wave breathing (6/12/16/22/30 BPM)
Pipeline               end-to-end wall time per 512-sample chunk
Summary                measured values vs design targets in audio/config.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from scipy import signal as sp_signal

from tash.audio import config
from tash.audio.contracts import BreathingState, EscalationLevel
from tash.audio.stage1_denoise import Denoiser
from tash.audio.stage2_cueword import CueWordDetector, CueWordError
from tash.audio.stage3_breathing import BreathingDetector
from tash.audio.fusion import FusionEngine

SR = config.SAMPLE_RATE   # 16000
CHUNK = 512               # samples per live-mic frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_noise(n_samples: int, rms: float = 0.05) -> np.ndarray:
    x = np.random.randn(n_samples).astype(np.float32) * rms
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)


def _pink_noise(n_samples: int, rms: float = 0.05) -> np.ndarray:
    """1/f-ish noise via cumsum -- approximates road/engine rumble."""
    white = np.random.randn(n_samples).astype(np.float64)
    pink = np.cumsum(white)
    pink -= pink.mean()
    pink /= (np.abs(pink).max() + 1e-9)
    pink = (pink * rms).astype(np.float32)
    return (np.clip(pink, -1, 1) * 32767).astype(np.int16)


def _breathing_audio(bpm: float, duration_s: float, rms: float = 0.03) -> np.ndarray:
    """Amplitude-modulated white noise at the given breath rate (BPM)."""
    n = int(duration_s * SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    freq_hz = bpm / 60.0
    envelope = np.clip(np.sin(2 * np.pi * freq_hz * t), 0, 1)
    carrier = np.random.randn(n).astype(np.float64)
    sos = sp_signal.butter(4, [100, 2000], btype="band", fs=SR, output="sos")
    carrier = sp_signal.sosfilt(sos, carrier)
    carrier /= (np.abs(carrier).max() + 1e-9)
    audio = (carrier * envelope * rms).astype(np.float32)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.int16)


def _rms_float(pcm_int16: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm_int16.astype(np.float64) ** 2))) / 32768.0


def _sep(char: str = "-", width: int = 60) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Stage 1 -- Denoiser
# ---------------------------------------------------------------------------

def bench_stage1(n_chunks: int = 100) -> dict:
    print("\n" + _sep("="))
    print("STAGE 1: Denoiser")
    print(_sep("="))
    denoiser = Denoiser()
    baseline = _white_noise(int(1.5 * SR), rms=0.02)
    denoiser.set_noise_baseline(baseline, 0.0)

    durations_ms: list[float] = []
    input_rms: list[float] = []
    output_rms_kw: list[float] = []
    output_rms_br: list[float] = []

    for i in range(n_chunks):
        chunk = _white_noise(CHUNK, rms=0.05)
        ts = i * CHUNK / SR
        t0 = time.perf_counter()
        result = denoiser.process(chunk, ts)
        elapsed = (time.perf_counter() - t0) * 1000
        durations_ms.append(elapsed)
        input_rms.append(_rms_float(chunk))
        output_rms_kw.append(float(np.sqrt(np.mean(result.float32 ** 2))))
        output_rms_br.append(float(np.sqrt(np.mean(result.mild_float32 ** 2))))

    # Pink (car) noise test
    pink_rms_out = []
    denoiser2 = Denoiser()
    denoiser2.set_noise_baseline(_pink_noise(int(1.5 * SR), rms=0.04), 0.0)
    for i in range(50):
        chunk = _pink_noise(CHUNK, rms=0.06)
        r = denoiser2.process(chunk, i * CHUNK / SR)
        pink_rms_out.append(float(np.sqrt(np.mean(r.float32 ** 2))))

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    p99 = float(np.percentile(durations_ms, 99))
    target_ms = config.AUDIO_LATENCY_MS["stage1_denoise"]
    avg_in = float(np.mean(input_rms))
    avg_kw = float(np.mean(output_rms_kw))
    avg_br = float(np.mean(output_rms_br))
    supp_kw = 20 * np.log10(avg_in / (avg_kw + 1e-9))
    supp_br = 20 * np.log10(avg_in / (avg_br + 1e-9))
    pink_avg_in = float(np.mean([_rms_float(_pink_noise(CHUNK, 0.06)) for _ in range(20)]))
    supp_pink = 20 * np.log10(pink_avg_in / (float(np.mean(pink_rms_out)) + 1e-9))

    lat_ok = "[PASS]" if p99 <= target_ms else "[FAIL]"
    print(f"  Latency  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  "
          f"target<={target_ms}ms  {lat_ok}")
    print(f"  White-noise suppression:  aggressive={supp_kw:.1f}dB  gentle={supp_br:.1f}dB")
    print(f"  Pink-noise suppression:   aggressive={supp_pink:.1f}dB")

    return dict(
        latency_p50_ms=p50, latency_p95_ms=p95, latency_p99_ms=p99,
        target_ms=target_ms, pass_latency=p99 <= target_ms,
        suppression_white_kw_db=supp_kw,
        suppression_white_br_db=supp_br,
        suppression_pink_db=supp_pink,
    )


# ---------------------------------------------------------------------------
# Stage 2 -- CueWord false-positive rate
# ---------------------------------------------------------------------------

def bench_stage2_fp(cue: CueWordDetector, n_seconds: int = 30) -> dict:
    print("\n" + _sep("="))
    print("STAGE 2: CueWord -- False Positive Rate")
    print(_sep("="))
    scenarios = {
        "silence":     _silence(n_seconds * SR),
        "white_noise": _white_noise(n_seconds * SR, rms=0.05),
        "pink_noise":  _pink_noise(n_seconds * SR, rms=0.05),
    }
    results = {}
    for name, audio in scenarios.items():
        fp_count = 0
        chunks = [audio[i: i + CHUNK] for i in range(0, len(audio) - CHUNK, CHUNK)]
        for i, chunk in enumerate(chunks):
            ts = i * CHUNK / SR
            event = cue.process(chunk.astype(np.int16), ts)
            if event is not None:
                fp_count += 1
        fp_rate = fp_count / max(len(chunks), 1)
        target = config.PERFORMANCE_TARGETS["false_positive_rate"]
        ok = "[PASS]" if fp_rate <= target else "[FAIL]"
        print(f"  {name:<14}  FP_events={fp_count:3d}  "
              f"FP_rate={fp_rate:.4f}  target<={target:.2f}  {ok}")
        results[name] = dict(fp_count=fp_count, fp_rate=fp_rate, pass_fp=fp_rate <= target)
    return results


def bench_stage2_latency(cue: CueWordDetector, n_chunks: int = 200) -> dict:
    print("\n" + _sep("-"))
    print("Stage 2 -- Per-Chunk Latency (below energy-VAD threshold)")
    print(_sep("-"))
    durations_ms = []
    for i in range(n_chunks):
        chunk = _white_noise(CHUNK, rms=0.005)  # quiet -> VAD inactive
        t0 = time.perf_counter()
        cue.process(chunk, i * CHUNK / SR)
        durations_ms.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    target = config.AUDIO_LATENCY_MS["stage2_vosk_partial"]
    ok = "[PASS]" if p95 <= target else "[FAIL]"
    print(f"  p50={p50:.1f}ms  p95={p95:.1f}ms  target<={target}ms  {ok}")
    print(f"  (Final-result target={config.AUDIO_LATENCY_MS['stage2_vosk_final']}ms "
          f"-- fires only at speech->silence edge, data-dependent)")
    return dict(latency_p50_ms=p50, latency_p95_ms=p95,
                target_ms=target, pass_latency=p95 <= target)


def bench_stage2_wav(cue: CueWordDetector, wav_dir: str) -> dict | None:
    """TP test using real WAV files. Layout: wav_dir/positive/*.wav, negative/*.wav"""
    import glob
    pos = glob.glob(os.path.join(wav_dir, "positive", "*.wav"))
    neg = glob.glob(os.path.join(wav_dir, "negative", "*.wav"))
    if not pos and not neg:
        print(f"  No WAV files found in {wav_dir}/positive/ or /negative/")
        return None
    try:
        import soundfile as sf
    except ImportError:
        print("  soundfile not installed -- skipping WAV TP test")
        return None

    print("\n" + _sep("-"))
    print("Stage 2 -- WAV True-Positive Test")
    print(_sep("-"))

    def _run_wav(path: str):
        audio, file_sr = sf.read(path, dtype="int16", always_2d=False)
        if file_sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(SR, file_sr)
            audio = resample_poly(audio, SR // g, file_sr // g).astype(np.int16)
        dets = []
        for i in range(0, len(audio) - CHUNK, CHUNK):
            event = cue.process(audio[i: i + CHUNK].astype(np.int16), i / SR)
            if event is not None:
                dets.append(event.keyword)
        return dets

    tp = fn = fp_neg = 0
    for p in pos:
        dets = _run_wav(p)
        if any(d == "help" for d in dets):
            tp += 1
        else:
            fn += 1
    for n_file in neg:
        fp_neg += len(_run_wav(n_file))

    total_pos = len(pos)
    tp_rate = tp / total_pos if total_pos else float("nan")
    target = config.PERFORMANCE_TARGETS["vosk_accuracy_on_help"]
    ok = "[PASS]" if tp_rate >= target else ("[N/A]" if total_pos == 0 else "[FAIL]")
    print(f"  WAV TP  {tp}/{total_pos} = {tp_rate:.2f}  target>={target:.2f}  {ok}")
    print(f"  WAV FP  {fp_neg} triggers on {len(neg)} negative files")
    return dict(tp=tp, fn=fn, fp_neg=fp_neg, tp_rate=tp_rate, pass_tp=tp_rate >= target)


# ---------------------------------------------------------------------------
# Stage 3 -- Breathing detection on synthetic patterns
# ---------------------------------------------------------------------------

def bench_stage3(n_runs: int = 3) -> dict:
    """Feed 70 s of synthetic breathing at various BPMs and check final state."""
    print("\n" + _sep("="))
    print("STAGE 3: Breathing Detector (synthetic audio)")
    print(_sep("="))
    target_ms = config.AUDIO_LATENCY_MS["stage3_breathing"]

    # (bpm, expected_acceptable_states, label)
    scenarios = [
        (30.0, [BreathingState.AMBIGUOUS],                               "30 BPM tachypnea"),
        (16.0, [BreathingState.NORMAL],                                  "16 BPM normal"),
        (12.0, [BreathingState.NORMAL],                                  "12 BPM normal"),
        (6.0,  [BreathingState.LOW_RATE, BreathingState.AGONAL_SUSPECT], "6 BPM agonal"),
        (3.0,  [BreathingState.APNEA, BreathingState.AGONAL_SUSPECT,
                BreathingState.LOW_RATE],                                "3 BPM apnea-like"),
    ]

    results = {}
    for bpm, expected, label in scenarios:
        finals: list[BreathingState | None] = []
        lats: list[float] = []

        for _ in range(n_runs):
            det = BreathingDetector()
            audio = _breathing_audio(bpm, duration_s=70.0)
            last_state: BreathingState | None = None

            for i in range(0, len(audio) - CHUNK, CHUNK):
                ts = i / SR
                mild = audio[i: i + CHUNK].astype(np.float32) / 32768.0
                t0 = time.perf_counter()
                est = det.process(mild, ts, speech_present=False)
                lats.append((time.perf_counter() - t0) * 1000)
                if est is not None:
                    last_state = est.state
            finals.append(last_state)

        dominant = max(set(finals), key=finals.count)
        hit = dominant in expected
        lat95 = float(np.percentile(lats, 95))
        cls_ok  = "[PASS]" if hit else "[WARN]"
        lat_ok  = "[PASS]" if lat95 <= target_ms else "[FAIL]"

        print(f"  {label:<22}  final={str(dominant.value if dominant else 'None'):<16}"
              f"  expected={[s.value for s in expected]}  {cls_ok}"
              f"  lat_p95={lat95:.1f}ms {lat_ok}")
        results[label] = dict(
            bpm=bpm, dominant=dominant, expected=expected,
            pass_cls=hit, lat_p95_ms=lat95, pass_lat=lat95 <= target_ms,
        )

    return results


# ---------------------------------------------------------------------------
# Full pipeline timing
# ---------------------------------------------------------------------------

def bench_pipeline(cue: CueWordDetector, n_chunks: int = 300) -> dict:
    """Time Stage1+Stage2+Stage3+Fusion together on noise (no Vosk final result)."""
    print("\n" + _sep("="))
    print("FULL PIPELINE: End-to-End Latency")
    print(_sep("="))
    denoiser = Denoiser()
    breathing = BreathingDetector()
    fusion = FusionEngine()
    denoiser.set_noise_baseline(_white_noise(int(1.5 * SR), rms=0.02), 0.0)

    durations_ms = []
    for i in range(n_chunks):
        chunk = _white_noise(CHUNK, rms=0.03)
        ts = i * CHUNK / SR
        t0 = time.perf_counter()
        dc = denoiser.process(chunk, ts)
        ev = cue.process(chunk, ts)
        sp = cue.speech_present
        be = breathing.process(dc.mild_float32, ts, sp)
        fusion.decide(ts, ev, be)
        durations_ms.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(durations_ms, 50))
    p95 = float(np.percentile(durations_ms, 95))
    p99 = float(np.percentile(durations_ms, 99))
    target = config.AUDIO_LATENCY_MS["total_pipeline_ms"]

    print(f"  No-Vosk-final chunks (noise):  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    print(f"  Design worst-case (with Vosk final result): {target}ms")
    print(f"  Real-time budget per chunk: {CHUNK/SR*1000:.0f}ms (512 samples @ {SR}Hz)")
    print(f"\n  Latency budget breakdown (from audio/config.py):")
    for k, v in config.AUDIO_LATENCY_MS.items():
        print(f"    {k:<32} {v:>4} ms")

    return dict(latency_p50_ms=p50, latency_p95_ms=p95,
                latency_p99_ms=p99, target_ms=target)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(s1, s2_fp, s2_lat, s3, pl, s2_wav) -> None:
    print("\n" + _sep("=", 65))
    print("AUDIO PIPELINE BENCHMARK -- SUMMARY")
    print(_sep("=", 65))

    # Stage 1
    ok = "OK" if s1["pass_latency"] else "!!"
    print(f"\nStage 1 -- Denoiser")
    print(f"  Latency p99: {s1['latency_p99_ms']:.1f}ms  (target <={s1['target_ms']}ms) [{ok}]")
    print(f"  Suppression: white={s1['suppression_white_kw_db']:.1f}dB aggressive / "
          f"{s1['suppression_white_br_db']:.1f}dB gentle | pink={s1['suppression_pink_db']:.1f}dB")

    # Stage 2
    print(f"\nStage 2 -- Cue Word  (Vosk vosk-model-small-en-us-0.15)")
    print(f"  Published WER (Librispeech test-clean, vendor):  9.85%")
    print(f"  Real-world WER (accent study, 2025):             41.8% mean  +/-13.5%")
    print(f"  Vosk 0.22 model (1.8GB) improves WER by ~20% over 0.15")
    ok = "OK" if s2_lat.get("pass_latency") else "!!"
    print(f"  Per-chunk latency p95: {s2_lat['latency_p95_ms']:.1f}ms  "
          f"(target <={s2_lat['target_ms']}ms) [{ok}]")
    for name, r in s2_fp.items():
        ok = "OK" if r["pass_fp"] else "!!"
        print(f"  FP rate ({name:<12}) {r['fp_rate']:.4f}  "
              f"(target <={config.PERFORMANCE_TARGETS['false_positive_rate']:.2f}) [{ok}]")
    tp_tgt = config.PERFORMANCE_TARGETS["vosk_accuracy_on_help"]
    if s2_wav:
        ok = "OK" if s2_wav["pass_tp"] else "!!"
        print(f"  TP rate (WAV 'help'): {s2_wav['tp_rate']:.2f}  (target >={tp_tgt:.2f}) [{ok}]")
    else:
        print(f"  TP rate ('help'): NOT MEASURED -- provide --wav-dir with real speech WAVs")
        print(f"    Target: >={tp_tgt:.2f} TP rate")
        print(f"    webrtcvad VAD at 5% FPR achieves ~50% TPR (literature baseline)")
        print(f"    Silero VAD: 87.7% TPR  |  Cobra VAD: 98.9% TPR at same 5% FPR")

    # Stage 3
    print(f"\nStage 3 -- Breathing Detector  (heuristic, no ML model)")
    print(f"  No published benchmark exists for agonal-breath detection in-car audio.")
    s3_pass = sum(1 for v in s3.values() if v["pass_cls"])
    print(f"  Synthetic accuracy: {s3_pass}/{len(s3)} scenarios classified correctly")
    for label, r in s3.items():
        ok  = "OK" if r["pass_cls"] else "!!"
        lok = "OK" if r["pass_lat"] else "!!"
        print(f"    {label:<22} got={str(r['dominant'].value if r['dominant'] else 'None'):<16} "
              f"cls[{ok}] lat[{lok}] p95={r['lat_p95_ms']:.1f}ms")
    print(f"  NOTE: synthetic sine-wave != real agonal breathing.")
    print(f"  Real agonal recordings required before production validation.")

    # Pipeline
    print(f"\nFull Pipeline")
    print(f"  p50={pl['latency_p50_ms']:.1f}ms  "
          f"p95={pl['latency_p95_ms']:.1f}ms  "
          f"p99={pl['latency_p99_ms']:.1f}ms")
    print(f"  Worst-case design target (with Vosk final): {pl['target_ms']}ms")

    # Constraints
    print(f"\nKnown Constraints (from audio/config.py CONSTRAINTS):")
    for k, v in config.CONSTRAINTS.items():
        print(f"  [{k}]")
        print(f"    {v}")

    # Comparison table
    print(f"\nComponent comparison vs alternatives (literature):")
    rows = [
        ("Component",              "This pipeline",    "Better alternative"),
        (_sep("-", 28),            _sep("-", 14),      _sep("-", 22)),
        ("ASR model",              "Vosk 0.15 (40MB)", "Vosk 0.22 (1.8GB)"),
        ("ASR WER clean speech",   "9.85%",            "5.69%"),
        ("ASR WER real world",     "~41.8%",           "~32.9%"),
        ("VAD",                    "webrtcvad",        "Silero / Cobra VAD"),
        ("VAD TPR at 5% FPR",      "~50%",             "88% / 99%"),
        ("Noise reduction",        "noisereduce",      "Denoiser (33M params)"),
        ("Breathing detection",    "Heuristic only",   "No validated model exists"),
    ]
    col_w = [30, 18, 24]
    for row in rows:
        print("  " + "  ".join(str(c).ljust(col_w[i]) for i, c in enumerate(row)))

    print(f"\nSources:")
    print(f"  alphacephei.com/vosk/models (WER figures)")
    print(f"  picovoice.ai/blog/best-voice-activity-detection-vad (VAD comparison)")
    print(f"  nature.com/articles/s41598-025-13108-x (noisereduce 2025 paper)")
    print(f"  nhsjs.com/2025 (Vosk accent WER study)")
    print(_sep("=", 65))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Audio pipeline benchmark")
    parser.add_argument("--wav-dir", default=None,
                        help="Dir with positive/ and negative/ WAV subdirs")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer iterations (faster but less stable numbers)")
    args = parser.parse_args()

    n_s1 = 50  if args.quick else 150
    n_s2 = 15  if args.quick else 30
    n_s3 = 1   if args.quick else 3
    n_pl = 100 if args.quick else 300

    print("Audio Pipeline Benchmark")
    print(f"Python {sys.version}")
    print(f"SR={SR}Hz  chunk={CHUNK} samples ({CHUNK/SR*1000:.0f}ms)")

    s1 = bench_stage1(n_chunks=n_s1)

    print("\n" + _sep("="))
    print("STAGE 2: CueWord -- Loading Vosk model...")
    print(_sep("="))
    try:
        cue = CueWordDetector(cue_words=config.CUE_WORDS + config.REASSURANCE_WORDS)
        print("  Vosk model loaded OK")
        vosk_ok = True
    except CueWordError as e:
        print(f"  [DEGRADED] {e}")
        cue = None
        vosk_ok = False

    if vosk_ok:
        s2_fp  = bench_stage2_fp(cue, n_seconds=n_s2)
        s2_lat = bench_stage2_latency(cue, n_chunks=200)
        s2_wav = bench_stage2_wav(cue, args.wav_dir) if args.wav_dir else None
    else:
        s2_fp  = {}
        s2_lat = {"latency_p95_ms": float("nan"), "target_ms": 50, "pass_latency": False}
        s2_wav = None

    s3 = bench_stage3(n_runs=n_s3)

    if vosk_ok and cue is not None:
        pl = bench_pipeline(cue, n_chunks=n_pl)
    else:
        pl = {"latency_p50_ms": float("nan"), "latency_p95_ms": float("nan"),
              "latency_p99_ms": float("nan"),
              "target_ms": config.AUDIO_LATENCY_MS["total_pipeline_ms"]}

    print_summary(s1, s2_fp, s2_lat, s3, pl, s2_wav)


if __name__ == "__main__":
    main()
