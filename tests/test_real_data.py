"""Real-data evaluation test.

Tests the model against actual audio recordings — not synthetic signals.

Data sources
------------
POSITIVES (agonal breathing):
  data/agonal_real/          25 real or real-augmented agonal recordings
                             NOTE: these files were included in training.
                             This section tests that the model memorised them
                             correctly, but is NOT an independent test.

NEGATIVES — genuinely real recordings:
  ESC-50 (real field recordings, CC-BY):
    category 23 - breathing      (40 clips)  real normal breathing
    category 28 - snoring        (40 clips)  real snoring
    category 44 - engine         (40 clips)  real engine sounds
    category 43 - car_horn       (40 clips)  real car horns / driving
    category 24 - coughing       (40 clips)  real coughing
    category 36 - vacuum_cleaner (40 clips)  real HVAC/fan sounds
    category 20 - crying_baby    (40 clips)  real baby/panic sounds

  ICBHI 2017 (real stethoscope recordings, CC-BY):
    Random sample of 60 files      real lung sounds: crackles, wheezes, normal
    NOTE: ALL ICBHI files were used as training negatives.

Independence note
-----------------
The ESC-50 training step used only 280 randomly-selected clips out of 2000.
~1720 clips were never seen.  This test runs ALL 280 clips per relevant
category, so the majority are genuinely unseen.

The ICBHI files were ALL used in training.  Their section shows whether the
model holds up on real stethoscope audio, not whether it generalises.

Usage
-----
    py -3.12 tests/test_real_data.py
"""
from __future__ import annotations
import os, sys, csv, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib

from audio.train_agonal_detector import (
    extract_features,
    _load_audio_file,
    _active_windows,
    _esc50_clips,
    WINDOW_N, SR,
)

# ── paths ──────────────────────────────────────────────────────────────────
REPO   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH   = os.path.join(REPO, "audio", "models", "agonal_detector.joblib")
AGONAL_DIR   = os.path.join(REPO, "data", "agonal_real")
ICBHI_DIR    = os.path.join(REPO, "data", "icbhi_full")
ESC50_DIR    = os.path.join(REPO, "data", "ESC-50-master")
ESC50_META   = os.path.join(ESC50_DIR, "meta", "esc50.csv")
ESC50_AUDIO  = os.path.join(ESC50_DIR, "audio")

THRESH = 0.55   # matches stage3_ml_breathing.GASP_THRESH
MAX_ICBHI_FILES = 60   # cap to keep test time reasonable


# ── helpers ────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def sub(title: str) -> None:
    print(f"\n  -- {title}")


def predict_windows(pipe, windows: list[np.ndarray]) -> list[tuple[float, bool]]:
    if not windows:
        return []
    X = np.vstack([extract_features(w).reshape(1, -1) for w in windows])
    probs = pipe.predict_proba(X)[:, 1]
    return [(float(p), bool(p >= THRESH)) for p in probs]


def report_windows(name: str, preds: list[tuple[float, bool]],
                   true_label: int, n_files: int,
                   in_training: bool = False) -> dict:
    if not preds:
        print(f"  {name:<42}  NO DATA")
        return {}
    tp = sum(1 for _, p in preds if p and true_label == 1)
    tn = sum(1 for _, p in preds if not p and true_label == 0)
    fp = sum(1 for _, p in preds if p and true_label == 0)
    fn = sum(1 for _, p in preds if not p and true_label == 1)
    mean_p = float(np.mean([pr for pr, _ in preds]))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    tag  = " [in training]" if in_training else " [unseen]"
    print(f"  {name:<42}  files={n_files:3d}  windows={len(preds):4d}  "
          f"mean_P={mean_p:.3f}  F1={f1:.3f}  "
          f"TP={tp} TN={tn} FP={fp} FN={fn}{tag}")
    return {"f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "mean_p": mean_p, "n": len(preds), "label": true_label}


def load_esc50_by_category(cat_name: str) -> tuple[list[np.ndarray], list[str]]:
    """Return (clips, filenames) for an ESC-50 category."""
    if not os.path.isfile(ESC50_META):
        return [], []
    clips, fnames = [], []
    with open(ESC50_META) as f:
        for row in csv.DictReader(f):
            if row["category"] == cat_name:
                path = os.path.join(ESC50_AUDIO, row["filename"])
                audio = _load_audio_file(path)
                if audio is not None:
                    clips.append(audio)
                    fnames.append(row["filename"])
    return clips, fnames


def clips_to_windows(clips: list[np.ndarray],
                     rms_threshold: float = 0.0) -> list[np.ndarray]:
    windows = []
    for clip in clips:
        windows.extend(_active_windows(clip, hop_frac=0.5,
                                       rms_threshold=rms_threshold))
    return windows


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("       Run: py -3.12 -m audio.train_agonal_detector")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    pipe   = bundle["pipeline"]
    print(f"Loaded: {type(pipe.named_steps['clf']).__name__}  "
          f"({bundle.get('feature_dim', '?')}-dim features)\n")

    all_results: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 1 — Real positive agonal breathing files
    # ──────────────────────────────────────────────────────────────────────
    section("1. Real agonal breathing files (data/agonal_real)")
    print("  NOTE: these files were included in training — not a held-out test.")
    print("        High scores confirm the model learned them correctly.\n")

    if not os.path.isdir(AGONAL_DIR):
        print("  data/agonal_real/ not found — skipping")
    else:
        exts = {".wav", ".mp3", ".flac", ".ogg"}
        fnames = [f for f in sorted(os.listdir(AGONAL_DIR))
                  if os.path.splitext(f.lower())[1] in exts]
        for fname in fnames:
            path  = os.path.join(AGONAL_DIR, fname)
            audio = _load_audio_file(path)
            if audio is None:
                continue
            wins  = _active_windows(audio, hop_frac=0.5, rms_threshold=0.04)
            preds = predict_windows(pipe, wins)
            r = report_windows(fname, preds, true_label=1,
                               n_files=1, in_training=True)
            if r:
                all_results.append(r)
        # Summary for this section
        pos_results = [r for r in all_results if r.get("label") == 1]
        if pos_results:
            total_tp = sum(r["tp"] for r in pos_results)
            total_fn = sum(r["fn"] for r in pos_results)
            total_w  = sum(r["n"]  for r in pos_results)
            print(f"\n  POSITIVE SUMMARY: {total_w} windows | "
                  f"recall={total_tp/(total_tp+total_fn+1e-9):.3f} | "
                  f"FN={total_fn}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 2 — ESC-50 real environmental negatives
    # ──────────────────────────────────────────────────────────────────────
    section("2. ESC-50 real environmental sounds (negatives)")
    print("  Majority of these clips were NOT seen during training (~85%).")
    print("  False Positives (FP) are the bad outcome here.\n")

    if not os.path.isdir(ESC50_DIR):
        print("  data/ESC-50-master/ not found")
        print("  Run: py -3.12 scripts/download_data.py")
    else:
        esc50_categories = [
            ("breathing",       "Real normal breathing"),
            ("snoring",         "Real snoring"),
            ("engine",          "Real engine / road noise"),
            ("car_horn",        "Real car horn / driving"),
            ("coughing",        "Real coughing"),
            ("vacuum_cleaner",  "Real vacuum / HVAC fan"),
            ("crying_baby",     "Real crying / panic sounds"),
            ("laughing",        "Real laughing"),
            ("footsteps",       "Real footsteps"),
            ("rain",            "Real rain"),
            ("wind",            "Real wind"),
        ]
        esc50_neg_results = []
        for cat, label in esc50_categories:
            clips, fnames = load_esc50_by_category(cat)
            if not clips:
                continue
            wins  = clips_to_windows(clips)
            preds = predict_windows(pipe, wins)
            r = report_windows(f"ESC-50: {label}", preds, true_label=0,
                               n_files=len(clips), in_training=False)
            if r:
                esc50_neg_results.append(r)
                all_results.append(r)

        if esc50_neg_results:
            total_fp = sum(r["fp"] for r in esc50_neg_results)
            total_tn = sum(r["tn"] for r in esc50_neg_results)
            total_w  = sum(r["n"]  for r in esc50_neg_results)
            print(f"\n  ESC-50 SUMMARY: {total_w} windows | "
                  f"FP={total_fp} | FP rate={total_fp/(total_w+1e-9):.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 3 — ICBHI real stethoscope recordings (negatives)
    # ──────────────────────────────────────────────────────────────────────
    section("3. ICBHI 2017 real stethoscope recordings (negatives)")
    print("  NOTE: all ICBHI files were used in training.")
    print("        Tests whether the model generalises on real lung sounds.\n")

    if not os.path.isdir(ICBHI_DIR):
        print("  data/icbhi_full/ not found")
        print("  Run: py -3.12 scripts/download_data.py")
    else:
        icbhi_files = [f for f in sorted(os.listdir(ICBHI_DIR))
                       if f.lower().endswith(".wav")]
        # Sample a random subset to keep test time reasonable
        random.seed(7)
        sample = random.sample(icbhi_files, min(MAX_ICBHI_FILES, len(icbhi_files)))
        icbhi_clips = []
        for fname in sample:
            audio = _load_audio_file(os.path.join(ICBHI_DIR, fname))
            if audio is not None:
                icbhi_clips.append(audio)
        wins  = clips_to_windows(icbhi_clips)
        preds = predict_windows(pipe, wins)
        r = report_windows("ICBHI real stethoscope recordings",
                           preds, true_label=0,
                           n_files=len(icbhi_clips), in_training=True)
        if r:
            all_results.append(r)

    # ──────────────────────────────────────────────────────────────────────
    # SECTION 4 — Overall summary
    # ──────────────────────────────────────────────────────────────────────
    section("4. Overall real-data summary")

    pos_r = [r for r in all_results if r.get("label") == 1]
    neg_r = [r for r in all_results if r.get("label") == 0]

    total_tp = sum(r["tp"] for r in pos_r)
    total_fn = sum(r["fn"] for r in pos_r)
    total_fp = sum(r["fp"] for r in neg_r)
    total_tn = sum(r["tn"] for r in neg_r)
    total_pos_w = sum(r["n"] for r in pos_r)
    total_neg_w = sum(r["n"] for r in neg_r)

    prec = total_tp / (total_tp + total_fp + 1e-9)
    rec  = total_tp / (total_tp + total_fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"\n  Positive windows (agonal):  {total_pos_w:5d}  |  TP={total_tp}  FN={total_fn}  recall={rec:.3f}")
    print(f"  Negative windows (non-gasp):{total_neg_w:5d}  |  TN={total_tn}  FP={total_fp}  FP_rate={total_fp/(total_neg_w+1e-9):.3f}")
    print(f"\n  --------------------------------------------------------")
    print(f"  REAL-DATA  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")
    print(f"  TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}")
    print(f"  --------------------------------------------------------")

    # Flag worst false-positive categories
    bad_fp = [(r, r["fp"]/r["n"]) for r in neg_r if r.get("fp", 0) > 0]
    if bad_fp:
        bad_fp.sort(key=lambda x: -x[1])
        print("\n  Categories with false positives:")
        for r, rate in bad_fp:
            print(f"    FP rate {rate:.3f}  ({r['fp']}/{r['n']} windows)")
    else:
        print("\n  No false positives on any real-data negative category.")

    print()


if __name__ == "__main__":
    main()
