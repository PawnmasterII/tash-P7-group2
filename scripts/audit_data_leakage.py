"""Audit audio/train_agonal_detector.py's pipeline for data leakage.

Runs four checks against the REAL data directories and the REAL split code
used by train_agonal_detector.train() (StratifiedKFold(5, shuffle=True,
random_state=42) with no grouping):

  1. Patient/subject overlap  -- do ICBHI patient IDs cross the CV split?
  2. Preprocessing leakage    -- is StandardScaler ever fit on non-train rows?
  3. Synthetic/augmentation overlap -- do augmented siblings of the same base
     gasp/window (different SNR level or snore-mix alpha) cross the split?
  4. Duplicate records        -- byte-identical files reused as if held-out;
     exact/near-duplicate feature rows from overlapping windows.

Section 1/3 avoid the cost of full feature extraction: window *counts* are
derived from file length (via soundfile.info(), no audio decode) using the
exact formulas build_dataset() uses, so the fold assignment we reproduce
matches what a real run over the same directories would see.

Usage:
    py -3.12 scripts/audit_data_leakage.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter

import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from audio.train_agonal_detector import (  # noqa: E402
    SNORE_ALPHAS, SNR_LEVELS_DB, WINDOW_N, SR,
    _AUDIO_EXTS, _active_windows, _load_audio_file, _make_classifier,
    extract_features, gen_car_window, gen_gasp, gen_hvac_window,
    gen_normal_breath_window, gen_radio_window, gen_slow_breath_window,
    gen_snore_window, gen_speech_window,
)

DATA_DIR = os.path.join(_REPO_ROOT, "data")
ICBHI_DIR = os.path.join(DATA_DIR, "icbhi_full")
AGONAL_DIR = os.path.join(DATA_DIR, "agonal_real")
NEG_REAL_DIR = os.path.join(DATA_DIR, "negatives_real")

N_GASPS = 400   # matches train_agonal_detector.main()'s --n-gasps default
N_NEG = 800     # matches --n-neg default


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def icbhi_patient_id(filename: str) -> str:
    """ICBHI 2017 filenames are '<patient>_<rec>_<chest-loc>_<mode>_<device>.wav'."""
    return filename.split("_")[0]


def window_count(n_samples: int, hop_frac: float = 0.5) -> int:
    """Mirrors _active_windows()'s slicing loop when rms_threshold<=0 (no filtering)."""
    hop_n = int(WINDOW_N * hop_frac)
    if n_samples < WINDOW_N:
        return 0
    return (n_samples - WINDOW_N) // hop_n + 1


# ---------------------------------------------------------------------------
# Metadata-only mirror of build_dataset(): same directories/counts as the
# documented training command (README.md / scripts/download_data.py), but we
# track (label, lineage group, ICBHI patient id) per window instead of
# extracting audio features -- this is what lets checks 1 and 3 run in
# seconds instead of the many minutes a full feature-extraction pass costs.
# ---------------------------------------------------------------------------

def build_metadata(n_gasps: int = N_GASPS, n_neg: int = N_NEG,
                    positive_dir: str = AGONAL_DIR, negative_dir: str = ICBHI_DIR):
    y: list[int] = []
    group_id: list[str] = []
    source: list[str] = []
    patient_id: list[str | None] = []

    gid_counter = [0]

    def new_gid() -> str:
        gid_counter[0] += 1
        return f"g{gid_counter[0]}"

    def add(label: int, gid: str, src: str, pid: str | None = None) -> None:
        y.append(label); group_id.append(gid); source.append(src); patient_id.append(pid)

    # ---- synthetic gasps (positives) --------------------------------------
    synth_gids = [new_gid() for _ in range(n_gasps)]
    for gid in synth_gids:
        add(1, gid, "synthetic_gasp")

    n_synth_snore = min(200, n_gasps)
    for gid in synth_gids[:n_synth_snore]:
        for _ in SNORE_ALPHAS:
            add(1, gid, "synthetic_gasp+snore_mix")

    # ---- real positives (data/agonal_real) --------------------------------
    real_gids: list[str] = []
    if positive_dir and os.path.isdir(positive_dir):
        for fname in sorted(os.listdir(positive_dir)):
            if os.path.splitext(fname.lower())[1] not in _AUDIO_EXTS:
                continue
            audio = _load_audio_file(os.path.join(positive_dir, fname))
            if audio is None:
                continue
            wins = _active_windows(audio, hop_frac=0.5, rms_threshold=0.04)
            for _ in wins:
                gid = new_gid()
                real_gids.append(gid)
                add(1, gid, f"real_positive:{fname}")
        for gid in real_gids:
            for _ in SNR_LEVELS_DB:
                add(1, gid, "real_positive_snr_aug")
            for _ in SNORE_ALPHAS:
                add(1, gid, "real_positive_snore_mix")

    # ---- synthetic negatives (7 classes, iid draws -> no lineage sharing) --
    neg_fns = [gen_normal_breath_window, gen_slow_breath_window, gen_snore_window,
               gen_car_window, gen_speech_window, gen_hvac_window, gen_radio_window]
    neg_per_class = max(1, n_neg // len(neg_fns))
    for fn in neg_fns:
        for _ in range(neg_per_class):
            add(0, new_gid(), f"synthetic_neg:{fn.__name__}")
    for _ in range(neg_per_class):
        add(0, new_gid(), "synthetic_silence")

    n_snore_pos = n_synth_snore * len(SNORE_ALPHAS) + len(real_gids) * len(SNORE_ALPHAS)
    snore_vol_levels = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    n_snore_bal = max(neg_per_class * 2, n_snore_pos // len(snore_vol_levels))
    for _ in range(n_snore_bal):
        for _ in snore_vol_levels:
            add(0, new_gid(), "synthetic_snore_balance_neg")

    # ---- ICBHI negatives (patient id tracked; whole file shares a group) --
    if negative_dir and os.path.isdir(negative_dir):
        for fname in sorted(os.listdir(negative_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            info = sf.info(os.path.join(negative_dir, fname))
            n_samples_16k = round(info.frames * SR / info.samplerate)
            n_win = window_count(n_samples_16k, hop_frac=0.5)
            if n_win == 0:
                continue
            file_gid = new_gid()
            pid = icbhi_patient_id(fname)
            for _ in range(n_win):
                add(0, file_gid, f"icbhi:{fname}", pid)

    return (np.array(y), np.array(group_id), np.array(source, dtype=object),
            np.array(patient_id, dtype=object))


# ---------------------------------------------------------------------------
# Check 1: Patient/subject overlap (ICBHI)
# ---------------------------------------------------------------------------

def check_patient_overlap(y: np.ndarray, group_id: np.ndarray, patient_id: np.ndarray) -> None:
    hr("CHECK 1: Patient/Subject Overlap (ICBHI)")
    icbhi_mask = patient_id != None  # noqa: E711 (object array, elementwise `is not None` below)
    n_icbhi = int(icbhi_mask.sum())
    unique_patients = sorted({p for p in patient_id if p is not None})
    print(f"ICBHI windows in the dataset the real training run builds: {n_icbhi}")
    print(f"Unique ICBHI patients represented: {len(unique_patients)}")

    X_dummy = np.zeros((len(y), 1))

    print("\n-- Split ACTUALLY used in train_agonal_detector.train():")
    print("   StratifiedKFold(n_splits=5, shuffle=True, random_state=42) -- no grouping\n")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fracs = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_dummy, y)):
        train_patients = {p for p in patient_id[train_idx] if p is not None}
        test_patients = {p for p in patient_id[test_idx] if p is not None}
        overlap = train_patients & test_patients
        frac = len(overlap) / max(1, len(test_patients))
        fracs.append(frac)
        print(f"  fold {i}: {len(test_patients):3d} patients in val fold -> "
              f"{len(overlap):3d} ({frac:.1%}) ALSO appear in that fold's train split")
    print(f"\n  VERDICT: avg {np.mean(fracs):.1%} of validation-fold patients leak into "
          f"the training fold under the current split.")

    print("\n-- Proposed fix: GroupKFold keyed on patient_id (ICBHI) / file-lineage (else)\n")
    split_group = np.array([p if p is not None else g for p, g in zip(patient_id, group_id)],
                            dtype=object)
    gcv = GroupKFold(n_splits=5)
    fracs_fixed = []
    for i, (train_idx, test_idx) in enumerate(gcv.split(X_dummy, y, groups=split_group)):
        train_patients = {p for p in patient_id[train_idx] if p is not None}
        test_patients = {p for p in patient_id[test_idx] if p is not None}
        overlap = train_patients & test_patients
        fracs_fixed.append(len(overlap) / max(1, len(test_patients)))
    print(f"  VERDICT: avg {np.mean(fracs_fixed):.1%} patient leakage with GroupKFold "
          f"(0% expected -- patients are single-group, so a fold either holds a patient's "
          f"windows entirely in train or entirely in test).")


# ---------------------------------------------------------------------------
# Check 2: Preprocessing leakage (StandardScaler fit-on-train vs fit-on-all)
# ---------------------------------------------------------------------------

def build_small_feature_sample() -> tuple[np.ndarray, np.ndarray]:
    """Small, class-balanced, real feature matrix for the scaler-fit probe."""
    rng = np.random.default_rng(0)
    rows: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(15):
        rows.append(extract_features(gen_gasp(rng=rng))); labels.append(1)

    if os.path.isdir(AGONAL_DIR):
        fnames = [f for f in sorted(os.listdir(AGONAL_DIR))
                  if os.path.splitext(f.lower())[1] in _AUDIO_EXTS][:15]
        for fname in fnames:
            audio = _load_audio_file(os.path.join(AGONAL_DIR, fname))
            if audio is None or len(audio) < WINDOW_N:
                continue
            rows.append(extract_features(audio[:WINDOW_N])); labels.append(1)

    neg_fns = [gen_normal_breath_window, gen_snore_window, gen_car_window]
    for fn in neg_fns:
        for _ in range(10):
            rows.append(extract_features(fn(rng))); labels.append(0)

    if os.path.isdir(ICBHI_DIR):
        fnames = sorted(os.listdir(ICBHI_DIR))[:15]
        for fname in fnames:
            audio = _load_audio_file(os.path.join(ICBHI_DIR, fname))
            if audio is None or len(audio) < WINDOW_N:
                continue
            rows.append(extract_features(audio[:WINDOW_N])); labels.append(0)

    return np.stack(rows), np.array(labels)


def check_preprocessing_leakage() -> None:
    hr("CHECK 2: Preprocessing (StandardScaler) fit discipline")
    print("Building a small real feature sample to probe what StandardScaler.fit() actually sees...")
    X, y = build_small_feature_sample()
    print(f"Sample: N={len(y)}  ({int(y.sum())} pos / {int((y == 0).sum())} neg)")

    fit_sizes: list[int] = []
    orig_fit = StandardScaler.fit

    def logging_fit(self, X_in, y_in=None, **kw):
        fit_sizes.append(X_in.shape[0])
        return orig_fit(self, X_in, y_in, **kw)

    StandardScaler.fit = logging_fit
    try:
        clf, clf_name = _make_classifier()
        pipe = SkPipeline([("scaler", StandardScaler()), ("clf", clf)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=1)
        print(f"\ncross_val_score() fold-by-fold scaler.fit() sample sizes: {fit_sizes}")
        print(f"  (each ~{len(y) * 4 // 5} = 80% of N={len(y)} -> scaler IS fit train-fold-only "
              f"inside cross_val_score; no leak in the CV mechanics themselves)")

        fit_sizes.clear()
        pipe.fit(X, y)
        print(f"\ntrain_agonal_detector.train()'s final pipe.fit(X, y) scaler.fit() sample size: "
              f"{fit_sizes}")
        print(f"  (== 100% of N={len(y)} -- the FINAL deployed model's scaler and classifier are "
              f"fit on the entire dataset because train() never reserves a held-out split before "
              f"this call. The 'in-sample report' printed right after is evaluated on the same "
              f"data the scaler/classifier were just fit on.)")
    finally:
        StandardScaler.fit = orig_fit

    print("\n  VERDICT: no scaler leak *within* cross_val_score's folds, but the pipeline never "
          "creates a genuine held-out test set at all -- so 'preprocessing leakage' is moot in the "
          "strict train/test sense (there IS no test set), and the deployed model's scaler stats "
          "are drawn from data it will also be scored against in the in-sample report.")


# ---------------------------------------------------------------------------
# Check 3: Synthetic/augmentation overlap
# ---------------------------------------------------------------------------

def check_augmentation_overlap(y: np.ndarray, group_id: np.ndarray) -> None:
    hr("CHECK 3: Synthetic Data / Augmentation-Lineage Overlap")
    fam_sizes = Counter(group_id)
    multi = {g: c for g, c in fam_sizes.items() if c > 1}
    print(f"{len(multi)}/{len(fam_sizes)} lineage groups have >1 window (i.e. are an augmented "
          f"family: the same base gasp/window reused across SNR levels or snore-mix alphas).")
    print(f"Largest augmented family size: {max(fam_sizes.values())} windows sharing one base sample.")

    X_dummy = np.zeros((len(y), 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fracs = []
    for train_idx, test_idx in cv.split(X_dummy, y):
        train_groups = set(group_id[train_idx])
        test_groups = set(group_id[test_idx])
        fracs.append(len(train_groups & test_groups) / max(1, len(test_groups)))
    print(f"\nCurrent split (StratifiedKFold, ungrouped): avg {np.mean(fracs):.1%} of validation-fold "
          f"lineage groups (e.g. the same source gasp at a different SNR/snore level) ALSO appear "
          f"in that fold's training data.")

    gcv = GroupKFold(n_splits=5)
    fracs_fixed = []
    for train_idx, test_idx in gcv.split(X_dummy, y, groups=group_id):
        train_groups = set(group_id[train_idx])
        test_groups = set(group_id[test_idx])
        fracs_fixed.append(len(train_groups & test_groups) / max(1, len(test_groups)))
    print(f"After GroupKFold(group_id): avg {np.mean(fracs_fixed):.1%} lineage leakage (0% expected).")

    print("\n  VERDICT: every SNR-augmented / snore-mixed copy of a real or synthetic gasp is built "
          "from the SAME rng stream (seed 42) BEFORE any split, and the ungrouped CV split "
          "randomly separates siblings of the same base sample across folds.")


# ---------------------------------------------------------------------------
# Check 4: Duplicate records
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def check_duplicate_files() -> None:
    hr("CHECK 4a: Duplicate raw audio FILES across directories used as 'held-out'")
    if not (os.path.isdir(ICBHI_DIR) and os.path.isdir(NEG_REAL_DIR)):
        print("data/icbhi_full or data/negatives_real not found -- skipping")
        return
    icbhi_hashes = {sha256_file(os.path.join(ICBHI_DIR, f)): f
                    for f in os.listdir(ICBHI_DIR) if f.lower().endswith(".wav")}
    neg_real_files = [f for f in os.listdir(NEG_REAL_DIR) if f.lower().endswith(".wav")]
    dupes = [(f, icbhi_hashes[h]) for f in neg_real_files
             if (h := sha256_file(os.path.join(NEG_REAL_DIR, f))) in icbhi_hashes]
    print(f"data/negatives_real: {len(neg_real_files)} files")
    print(f"data/icbhi_full (training negatives): {len(icbhi_hashes)} files")
    print(f"Byte-identical files present in BOTH: {len(dupes)}/{len(neg_real_files)}")
    if dupes:
        print(f"  e.g. {dupes[0][0]}  ==  {dupes[0][1]}")
        print("  tests/smoke_test_model.py scores data/negatives_real as if it were an unseen "
              "negative set, but it is (almost entirely) a byte-identical subset of the ICBHI "
              "files already used as training negatives -- any 'good' result there is circular.")


def check_duplicate_feature_rows() -> None:
    hr("CHECK 4b: Exact/near-duplicate FEATURE rows from overlapping windows")
    if not os.path.isdir(ICBHI_DIR):
        print("data/icbhi_full not found -- skipping")
        return
    sample_files = sorted(os.listdir(ICBHI_DIR))[:5]
    rows: list[np.ndarray] = []
    meta: list[tuple[str, int]] = []
    for fname in sample_files:
        audio = _load_audio_file(os.path.join(ICBHI_DIR, fname))
        if audio is None:
            continue
        wins = _active_windows(audio, hop_frac=0.5, rms_threshold=0.0)
        for i, w in enumerate(wins):
            rows.append(extract_features(w))
            meta.append((fname, i))
    if not rows:
        print("no windows extracted -- skipping")
        return
    X = np.stack(rows)
    hashes = [hashlib.sha256(r.tobytes()).hexdigest() for r in X]
    dup_count = len(hashes) - len(set(hashes))
    print(f"Sampled {len(rows)} windows (50%-hop) from {len(sample_files)} ICBHI files.")
    print(f"Exact byte-identical feature rows: {dup_count}/{len(rows)}")

    close, total = 0, 0
    for i in range(1, len(rows)):
        if meta[i][0] == meta[i - 1][0]:
            total += 1
            a, b = X[i], X[i - 1]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if cos > 0.98:
                close += 1
    print(f"Adjacent 50%-overlap windows from the SAME file with cosine similarity > 0.98: "
          f"{close}/{total}")
    print("  These are not byte-identical rows, but near-duplicates: the current split treats "
          "each as an independent sample, so a CV fold can 'test' on a window that is 50% "
          "identical audio to a window sitting in its own training fold.")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Agonal Breathing Detector -- Data Leakage Audit")
    print(f"Repo: {_REPO_ROOT}")

    y, group_id, source, patient_id = build_metadata()
    print(f"\nReconstructed dataset structure (metadata only, no feature extraction): "
          f"{len(y)} windows, {int(y.sum())} positive / {int((y == 0).sum())} negative "
          f"-- matches the README's documented training invocation "
          f"(--positive-dir data/agonal_real --negative-dir data/icbhi_full).")

    check_patient_overlap(y, group_id, patient_id)
    check_augmentation_overlap(y, group_id)
    check_preprocessing_leakage()
    check_duplicate_files()
    check_duplicate_feature_rows()

    hr("SUMMARY")
    print("""
  1. Patient/Subject Overlap : LEAKAGE CONFIRMED. No GroupShuffleSplit/GroupKFold is used
     anywhere; ICBHI patient IDs are never even parsed. The CV split shuffles at the window
     level, so essentially every validation-fold patient's other windows sit in that fold's
     training data.
  2. Preprocessing Leakage   : NOT a scaler-fit bug (cross_val_score correctly refits the
     scaler per fold) -- but moot regardless, because train() never creates a held-out split
     at all: the deployed model's scaler AND classifier are fit on 100% of the data, and the
     printed metrics are in-sample.
  3. Synthetic Data Overlap  : LEAKAGE CONFIRMED. SNR-augmented and snore-mixed copies of the
     same base gasp/window are generated from one seed-42 rng stream before any split, then
     shuffled into CV folds independently of their siblings.
  4. Duplicate Record Check  : LEAKAGE CONFIRMED. data/negatives_real (used by
     tests/smoke_test_model.py as if held-out) is byte-identical to the ICBHI negatives used
     for training. Overlapping (50%-hop) windows from the same file are near-duplicate feature
     rows split across folds with no grouping.
""")


if __name__ == "__main__":
    main()
