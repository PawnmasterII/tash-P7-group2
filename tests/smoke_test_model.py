"""Quick smoke test: load the trained model and score real clips."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

from audio.train_agonal_detector import extract_features, _active_windows

SR = 16_000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "audio", "models", "agonal_detector.joblib")


def load_wav(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            g = gcd(SR, sr)
            audio = resample_poly(audio, SR // g, sr // g).astype("float32")
        return audio
    except Exception as e:
        print(f"    Could not load {os.path.basename(path)}: {e}")
        return None


def score_clip(pipe, path: str):
    audio = load_wav(path)
    if audio is None:
        return None
    wins = _active_windows(audio, hop_frac=0.5, rms_threshold=0.0)
    if not wins:
        return None
    probs = [float(pipe.predict_proba(extract_features(w).reshape(1, -1))[0, 1])
             for w in wins[:8]]
    return float(np.mean(probs)), float(np.max(probs))


def main():
    # ---------- Load model ----------
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    clf_type = type(pipe.named_steps["clf"]).__name__

    print("=" * 60)
    print("Model bundle")
    print("=" * 60)
    print(f"  classifier  : {clf_type}")
    print(f"  feature_dim : {bundle['feature_dim']}")
    print(f"  sample_rate : {bundle['sample_rate']} Hz")
    print(f"  window_s    : {bundle['window_s']} s")
    print(f"  model_type  : {bundle['model_type']}")

    repo = os.path.join(os.path.dirname(__file__), "..")
    agonal_dir = os.path.join(repo, "data", "agonal_real")
    neg_dir    = os.path.join(repo, "data", "negatives_real")

    # ---------- Score agonal clips (should be HIGH) ----------
    print()
    print("=" * 60)
    print("Agonal clips  (expect P(gasp) > 0.50)")
    print("=" * 60)
    tp = fp = 0
    for fname in sorted(os.listdir(agonal_dir))[:10]:
        path = os.path.join(agonal_dir, fname)
        if not fname.endswith((".wav", ".webm")):
            continue
        r = score_clip(pipe, path)
        if r is None:
            continue
        mean_p, max_p = r
        correct = mean_p > 0.5
        tp += int(correct)
        fp += int(not correct)
        mark = "OK" if correct else "MISS"
        print(f"  [{mark}]  {fname[:44]:<44}  mean={mean_p:.3f}  max={max_p:.3f}")

    # ---------- Score negative clips (should be LOW) ----------
    print()
    print("=" * 60)
    print("Negative clips  (expect P(gasp) < 0.50)")
    print("=" * 60)
    tn = fn = 0
    for fname in sorted(os.listdir(neg_dir))[:10]:
        path = os.path.join(neg_dir, fname)
        if not fname.endswith(".wav"):
            continue
        r = score_clip(pipe, path)
        if r is None:
            continue
        mean_p, max_p = r
        correct = mean_p < 0.5
        tn += int(correct)
        fn += int(not correct)
        mark = "OK" if correct else "FP"
        print(f"  [{mark}]  {fname[:44]:<44}  mean={mean_p:.3f}  max={max_p:.3f}")

    # ---------- Summary ----------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Agonal clips : {tp} correct / {tp+fp} tested  (TP rate = {tp/(tp+fp+1e-9):.0%})")
    print(f"  Neg clips    : {tn} correct / {tn+fn} tested  (TN rate = {tn/(tn+fn+1e-9):.0%})")
    print()
    if tp + fp > 0 and tn + fn > 0:
        prec = tp / (tp + fn + 1e-9)
        rec  = tp / (tp + fp + 1e-9)
        print(f"  Precision = {prec:.2f}  Recall = {rec:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
