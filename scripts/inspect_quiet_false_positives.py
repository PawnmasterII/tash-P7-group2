"""Trace the agonal detector's false positives on the "quiet" held-out subset
back to their exact source file/patient, and save the offending audio for
manual listening.

Rebuilds the dataset with the SAME seed/params used to produce the currently
saved model (audio/models/agonal_detector.joblib), so X/y/noise_labels/groups
and the GroupShuffleSplit(random_state=42) reproduce bit-for-bit -- the saved
pipe's predictions on this rebuild exactly match what train() saw.

The "quiet" subset (no SNR/snore-mix noise overlay) is the closest proxy to
real deployment: real agonal recordings + clean synthetic gasps vs. genuinely
unseen ICBHI/ESC-50 negatives. Its false positives are the ones worth
understanding.

Usage:
    py -3.12 scripts/inspect_quiet_false_positives.py
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from audio.train_agonal_detector import (  # noqa: E402
    MODEL_PATH, WINDOW_N, _NL_CLEAN_REAL, _NL_ESC50, _NL_ICBHI, _NL_SYNTHETIC,
    _active_windows, _icbhi_patient_id, _load_audio_file, build_dataset,
)

DATA_DIR     = os.path.join(_REPO_ROOT, "data")
ICBHI_DIR    = os.path.join(DATA_DIR, "icbhi_full")
ESC50_DIR    = os.path.join(DATA_DIR, "ESC-50-master")
ESC50_META   = os.path.join(ESC50_DIR, "meta", "esc50.csv")
ESC50_AUDIO  = os.path.join(ESC50_DIR, "audio")
AGONAL_DIR   = os.path.join(DATA_DIR, "agonal_real")

GASP_THRESH = 0.55  # matches audio/stage3_ml_breathing.GASP_THRESH


def esc50_category_map() -> dict[str, str]:
    cats = {}
    with open(ESC50_META) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                cats[parts[0]] = parts[3]
    return cats


def icbhi_windows_metadata() -> list[tuple[str, int]]:
    """(filename, window_index_within_file) for every ICBHI window, in the
    EXACT order build_dataset() produces them (sorted filenames, hop_frac=0.5,
    rms_threshold=0.0 -> no filtering)."""
    meta = []
    for fname in sorted(os.listdir(ICBHI_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        audio = _load_audio_file(os.path.join(ICBHI_DIR, fname))
        if audio is None:
            continue
        wins = _active_windows(audio, hop_frac=0.5, rms_threshold=0.0)
        for i in range(len(wins)):
            meta.append((fname, i))
    return meta


def esc50_windows_metadata(categories: list[str]) -> list[tuple[str, int]]:
    """(filename, window_index_within_file) for every ESC-50 window used in
    training, in the same order _esc50_clips()+build_dataset()'s ESC-50 loop
    produces them."""
    cat_of = esc50_category_map()
    meta = []
    with open(ESC50_META) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            fname, _, _, category = parts[0], parts[1], parts[2], parts[3]
            path = os.path.join(ESC50_AUDIO, fname)
            if category in categories and os.path.isfile(path):
                audio = _load_audio_file(path)
                if audio is None:
                    continue
                n_win = len(range(0, len(audio) - WINDOW_N, WINDOW_N // 2))
                for i in range(n_win):
                    meta.append((fname, i))
    return meta


def main() -> None:
    print("Rebuilding dataset with the exact params used for the saved model "
          "(this repeats the ~14 min feature-extraction pass)...")
    X, y, noise_labels, groups = build_dataset(
        n_gasps=400, n_neg=800,
        positive_dir=AGONAL_DIR, negative_dir=ICBHI_DIR, esc50_dir=ESC50_DIR,
        n_jobs=4,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    print(f"Reproduced split: {len(train_idx)} train / {len(test_idx)} test "
          f"(must match the training run's 23543 / 5969)")

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    quiet_labels = {_NL_SYNTHETIC, _NL_CLEAN_REAL, _NL_ICBHI, _NL_ESC50}
    quiet_neg_mask_full = (np.isin(noise_labels, list(quiet_labels))) & (y == 0)
    # restrict to the held-out test rows only
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    fp_candidate_mask = quiet_neg_mask_full & test_mask
    idx_candidates = np.where(fp_candidate_mask)[0]
    print(f"\nQuiet, held-out, true-negative windows: {len(idx_candidates)}")

    probs = pipe.predict_proba(X[idx_candidates])[:, 1]

    for label, thresh in [("default 0.50", 0.50), ("production 0.55", GASP_THRESH)]:
        fp_mask = probs >= thresh
        print(f"\nFalse positives @ threshold {label}: {fp_mask.sum()} / {len(idx_candidates)} "
              f"({fp_mask.sum()/len(idx_candidates):.2%})")

    # Use the production threshold for the detailed trace.
    fp_mask = probs >= GASP_THRESH
    fp_rows = idx_candidates[fp_mask]
    fp_probs = probs[fp_mask]
    fp_noise_labels = noise_labels[fp_rows]
    fp_groups = groups[fp_rows]

    print(f"\n=== {len(fp_rows)} false positives @ GASP_THRESH={GASP_THRESH} ===")
    for nl_val, nl_name in [(_NL_ICBHI, "ICBHI"), (_NL_ESC50, "ESC-50"),
                             (_NL_SYNTHETIC, "synthetic-negative")]:
        n = int((fp_noise_labels == nl_val).sum())
        if n:
            print(f"  {nl_name}: {n}")

    icbhi_fp_details: list[tuple[str, str, int, float]] = []
    esc50_fp_details: list[tuple[str, str, int, float]] = []

    # ---- Trace ICBHI false positives to (filename, window_index) ----------
    icbhi_sel = fp_noise_labels == _NL_ICBHI
    if icbhi_sel.any():
        print("\nBuilding ICBHI (filename, window_index) map for exact tracing "
              "(audio decode only, no feature extraction, should be quick)...")
        icbhi_meta = icbhi_windows_metadata()
        icbhi_row_indices = np.where(noise_labels == _NL_ICBHI)[0]
        assert len(icbhi_row_indices) == len(icbhi_meta), (
            f"ICBHI window count mismatch: dataset has {len(icbhi_row_indices)}, "
            f"metadata pass produced {len(icbhi_meta)}")
        row_to_pos = {row: pos for pos, row in enumerate(icbhi_row_indices)}

        for row, prob in zip(fp_rows[icbhi_sel], fp_probs[icbhi_sel]):
            fname, win_i = icbhi_meta[row_to_pos[row]]
            icbhi_fp_details.append((_icbhi_patient_id(fname), fname, win_i, float(prob)))
        icbhi_fp_details.sort(key=lambda t: -t[3])
        print("\n  ICBHI false positives (patient, file, window#, P(gasp)):")
        for pid, fname, win_i, prob in icbhi_fp_details:
            print(f"    patient={pid:<5} file={fname:<32} window#{win_i:<3} P(gasp)={prob:.3f}")

    # ---- Trace ESC-50 false positives to (filename, category, window_index)
    esc50_sel = fp_noise_labels == _NL_ESC50
    if esc50_sel.any():
        esc_cats = ["breathing", "snoring", "engine", "car_horn",
                    "crying_baby", "footsteps", "laughing",
                    "rain", "wind", "vacuum_cleaner", "washing_machine",
                    "thunderstorm", "water_drops", "pouring_water"]
        print("\nBuilding ESC-50 (filename, window_index) map for exact tracing...")
        esc50_meta = esc50_windows_metadata(esc_cats)
        esc50_row_indices = np.where(noise_labels == _NL_ESC50)[0]
        assert len(esc50_row_indices) == len(esc50_meta), (
            f"ESC-50 window count mismatch: dataset has {len(esc50_row_indices)}, "
            f"metadata pass produced {len(esc50_meta)}")
        row_to_pos = {row: pos for pos, row in enumerate(esc50_row_indices)}
        cat_of = esc50_category_map()

        for row, prob in zip(fp_rows[esc50_sel], fp_probs[esc50_sel]):
            fname, win_i = esc50_meta[row_to_pos[row]]
            esc50_fp_details.append((cat_of.get(fname, "?"), fname, win_i, float(prob)))
        esc50_fp_details.sort(key=lambda t: -t[3])
        print("\n  ESC-50 false positives (category, file, window#, P(gasp)):")
        for cat, fname, win_i, prob in esc50_fp_details:
            print(f"    category={cat:<16} file={fname:<20} window#{win_i:<3} P(gasp)={prob:.3f}")

    # ---- Save the actual offending audio windows for manual listening -----
    scratch_out = os.environ.get("FP_AUDIO_OUT")
    if scratch_out:
        os.makedirs(scratch_out, exist_ok=True)
        n_saved = 0
        icbhi_cache: dict[str, list[np.ndarray]] = {}
        for pid, fname, win_i, prob in icbhi_fp_details:
            if fname not in icbhi_cache:
                audio = _load_audio_file(os.path.join(ICBHI_DIR, fname))
                icbhi_cache[fname] = _active_windows(audio, hop_frac=0.5, rms_threshold=0.0)
            win = icbhi_cache[fname][win_i]
            out_path = os.path.join(scratch_out, f"icbhi_p{pid}_{fname[:-4]}_w{win_i}_p{prob:.2f}.wav")
            sf.write(out_path, win, 16000)
            n_saved += 1
        esc_cache: dict[str, list[np.ndarray]] = {}
        for cat, fname, win_i, prob in esc50_fp_details:
            if fname not in esc_cache:
                audio = _load_audio_file(os.path.join(ESC50_AUDIO, fname))
                starts = list(range(0, len(audio) - WINDOW_N, WINDOW_N // 2))
                esc_cache[fname] = [audio[s:s + WINDOW_N] for s in starts]
            win = esc_cache[fname][win_i]
            out_path = os.path.join(scratch_out, f"esc50_{cat}_{fname[:-4]}_w{win_i}_p{prob:.2f}.wav")
            sf.write(out_path, win, 16000)
            n_saved += 1
        print(f"\nSaved {n_saved} false-positive audio clips to {scratch_out}")

    print("\nDONE")


if __name__ == "__main__":
    main()
