"""Out-of-distribution generalization test.

Generates brand-new clips with seed 99 (training used seed 42 — zero overlap)
and evaluates the model at every SNR level independently.

Sections
--------
1. Fresh synthetic gasps at all 6 SNR levels
2. Hard negatives: snoring, HVAC, radio, car, slow-breathing, speech
3. Borderline cases: very faint gasps (RMS < 0.1), very long gasps, double gasps
4. Per-SNR performance table showing where the model starts to degrade
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
from collections import defaultdict

from audio.train_agonal_detector import (
    extract_features, gen_gasp,
    gen_normal_breath_window, gen_slow_breath_window,
    gen_snore_window, gen_car_window, gen_speech_window,
    gen_hvac_window, gen_radio_window, gen_silence_window,
    _mix_at_snr, WINDOW_N, SNR_LEVELS_DB,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "audio", "models", "agonal_detector.joblib")
THRESH = 0.55   # matches stage3_ml_breathing.GASP_THRESH

RNG = np.random.default_rng(99)   # completely different from training seed (42)
N_PER_CLASS = 60                  # samples per test bucket


def predict(pipe, win: np.ndarray) -> tuple[float, bool]:
    feat = extract_features(win).reshape(1, -1)
    prob = float(pipe.predict_proba(feat)[0, 1])
    return prob, prob >= THRESH


def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def report(name: str, results: list[tuple[float, bool, int]]) -> None:
    """results = list of (prob, predicted_positive, true_label)"""
    tp = sum(1 for _, pred, lbl in results if pred and lbl == 1)
    tn = sum(1 for _, pred, lbl in results if not pred and lbl == 0)
    fp = sum(1 for _, pred, lbl in results if pred and lbl == 0)
    fn = sum(1 for _, pred, lbl in results if not pred and lbl == 1)
    n  = len(results)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    mean_p = np.mean([p for p, _, _ in results])
    print(f"  {name:<40}  n={n:3d}  mean_P={mean_p:.3f}  F1={f1:.3f}  "
          f"TP={tp} TN={tn} FP={fp} FN={fn}")


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipe   = bundle["pipeline"]
    print(f"Loaded: {type(pipe.named_steps['clf']).__name__}  "
          f"({bundle['feature_dim']}-dim features)")

    # =========================================================
    # 1. Fresh gasps at each SNR level (pure generalization test)
    # =========================================================
    section("1. Fresh synthetic gasps vs noise — per SNR level")
    print(f"  (seed=99, N={N_PER_CLASS} gasps per SNR, never seen during training)\n")
    snr_results: dict[float, list] = defaultdict(list)
    noise_fns = [gen_car_window, gen_hvac_window, gen_radio_window]

    for snr_db in SNR_LEVELS_DB:
        for _ in range(N_PER_CLASS):
            clean_gasp = gen_gasp(rng=RNG)
            noise_fn   = noise_fns[int(RNG.integers(len(noise_fns)))]
            noisy_gasp = _mix_at_snr(clean_gasp, noise_fn(RNG), snr_db)
            prob, pred = predict(pipe, noisy_gasp)
            snr_results[snr_db].append((prob, pred, 1))

    print(f"  {'SNR (dB)':<12}  {'n':>4}  {'mean P(gasp)':>14}  {'Recall':>8}  {'Misses':>7}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*14}  {'-'*8}  {'-'*7}")
    for snr_db in sorted(snr_results):
        res = snr_results[snr_db]
        mean_p = np.mean([p for p, _, _ in res])
        recall = sum(1 for _, pred, _ in res if pred) / len(res)
        misses = sum(1 for _, pred, _ in res if not pred)
        bar = "#" * int(recall * 20)
        print(f"  {snr_db:>+6.0f} dB       {len(res):>4}  {mean_p:>14.3f}  "
              f"{recall:>7.1%}  {misses:>6}  |{bar:<20}|")

    # =========================================================
    # 2. Hard negatives — should all score BELOW threshold
    # =========================================================
    section("2. Hard negatives — expect P(gasp) < 0.55 (FP = bad)")
    print(f"  (seed=99, N={N_PER_CLASS} per class)\n")

    neg_classes = {
        "Normal breathing (12-20 BPM)": gen_normal_breath_window,
        "Slow breathing  (7-10 BPM)  ": gen_slow_breath_window,
        "Snoring (similar shape!)     ": gen_snore_window,
        "Car engine + road noise      ": gen_car_window,
        "Speech (AM modulated)        ": gen_speech_window,
        "HVAC / fan hiss              ": gen_hvac_window,
        "Radio / music (in-cabin)     ": gen_radio_window,
        "Silence / near-silence       ": lambda _rng: gen_silence_window(),
    }

    for name, fn in neg_classes.items():
        res = []
        for _ in range(N_PER_CLASS):
            win = fn(RNG)
            prob, pred = predict(pipe, win)
            res.append((prob, pred, 0))
        report(name, res)

    # =========================================================
    # 3. Borderline / stress cases
    # =========================================================
    section("3. Borderline and stress cases")

    # 3a. Very faint gasps (RMS ≈ 0.03 — barely above silence threshold)
    print("\n  3a. Faint gasps (amplitude scaled to 5% of normal):")
    faint_res = []
    for _ in range(N_PER_CLASS):
        gasp = gen_gasp(rng=RNG) * 0.05
        prob, pred = predict(pipe, gasp)
        faint_res.append((prob, pred, 1))
    report("Faint gasp (0.05 amp scale)", faint_res)

    # 3b. Gasp buried in very loud car noise (SNR = -10 dB, worse than training)
    print("\n  3b. Gasp in extreme noise (SNR = -10 dB, outside training range):")
    extreme_res = []
    for _ in range(N_PER_CLASS):
        gasp  = gen_gasp(rng=RNG)
        noisy = _mix_at_snr(gasp, gen_car_window(RNG), -10.0)
        prob, pred = predict(pipe, noisy)
        extreme_res.append((prob, pred, 1))
    report("Gasp at -10 dB SNR (OOD)", extreme_res)

    # 3c. Gasp + snoring overlay (realistic confusion: agonal vs snoring)
    print("\n  3c. Gasp + snoring mixed (hardest real-world case):")
    gasp_snore_res = []
    for _ in range(N_PER_CLASS):
        gasp  = gen_gasp(rng=RNG)
        snore = gen_snore_window(RNG) * 0.5
        mixed = np.clip(gasp + snore, -1.0, 1.0).astype(np.float32)
        prob, pred = predict(pipe, mixed)
        gasp_snore_res.append((prob, pred, 1))
    report("Gasp + snoring overlay", gasp_snore_res)

    # 3d. Snoring alone at different volumes (false alarm risk)
    print("\n  3d. Snoring at high volume (FP risk):")
    loud_snore_res = []
    for _ in range(N_PER_CLASS):
        snore = gen_snore_window(RNG) * 2.0
        snore = np.clip(snore, -1.0, 1.0).astype(np.float32)
        prob, pred = predict(pipe, snore)
        loud_snore_res.append((prob, pred, 0))
    report("Loud snoring (label=negative)", loud_snore_res)

    # 3e. Window of pure speech — common in-car distractor
    print("\n  3e. Heavy speech (common while driving):")
    speech_res = []
    for _ in range(N_PER_CLASS):
        win = gen_speech_window(RNG)
        prob, pred = predict(pipe, win)
        speech_res.append((prob, pred, 0))
    report("Speech window (label=negative)", speech_res)

    # =========================================================
    # 4. Overall summary
    # =========================================================
    section("4. Overall OOD summary")
    all_pos = [(p, pred, lbl) for snr in snr_results.values() for p, pred, lbl in snr]
    all_neg = []
    for name, fn in neg_classes.items():
        for _ in range(20):
            win = fn(RNG)
            prob, pred = predict(pipe, win)
            all_neg.append((prob, pred, 0))

    report("All fresh gasps (across SNRs)", all_pos)
    report("All hard negatives           ", all_neg)

    all_combined = all_pos + all_neg
    tp = sum(1 for _, pred, lbl in all_combined if pred and lbl == 1)
    tn = sum(1 for _, pred, lbl in all_combined if not pred and lbl == 0)
    fp = sum(1 for _, pred, lbl in all_combined if pred and lbl == 0)
    fn = sum(1 for _, pred, lbl in all_combined if not pred and lbl == 1)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"\n  {'-'*50}")
    print(f"  COMBINED OOD  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}  Total={len(all_combined)}")
    print(f"  {'-'*50}\n")


if __name__ == "__main__":
    main()
