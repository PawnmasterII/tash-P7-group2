"""Train the ML agonal breathing detector and save it to audio/models/.

Architecture (follows UW 2019 npj Digital Medicine, Donahue et al.)
--------------------------------------------------------------------
Stage A -- Gasp Detector:
    2.5 s audio window -> log-mel + MFCC features -> LinearSVC
    Binary: "contains a gasp event" (1) vs "background/normal" (0)

Stage B -- Temporal Rate Filter (done at inference time, not trained):
    Counts gasp-positive events over a rolling 30 s window.
    Rate 3-6 events/min AND irregular spacing -> AGONAL_SUSPECT
    No gasps for > 12 s -> APNEA candidate

Why 2 stages?
    Agonal breathing is 3-6 BPM = one gasp every 10-20 s.  A single
    2.5 s window almost always contains SILENCE between gasps.  Training
    a window classifier on "is this whole clip agonal?" creates a model
    that just detects silence.  The UW paper trained on 2.5 s segments
    CENTERED ON actual gasps, then applied a temporal frequency filter.

Training data
-------------
    Positives: 2.5 s windows sliced FROM gasp events in synthetic agonal
               clips (known onset times) + any audio files in --positive-dir.
    Negatives: 2.5 s windows from silence, normal breathing, slow regular
               breathing, snoring, car noise, speech-like AM noise.
               + ESC-50 clips if --esc50-dir provided.

Real audio from the web
-----------------------
    theSimTech agonal breathing MP3 (free, medical simulation):
        https://www.thesimtech.org/av-stimuli/audio/
        Download, place in e.g. data/agonal_real/, then:
        py -3.12 -m audio.train_agonal_detector --positive-dir data/agonal_real

    ICBHI 2017 Respiratory Sound Database (free, 920 WAV files):
        https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
        Good hard negatives (wheezing, crackles, normal breathing):
        py -3.12 -m audio.train_agonal_detector --negative-dir icbhi2017/

    ESC-50 environmental sounds (free, CC-BY):
        https://github.com/karolpiczak/ESC-50
        py -3.12 -m audio.train_agonal_detector --esc50-dir ESC-50-master/

    Combining all:
        py -3.12 -m audio.train_agonal_detector \\
            --positive-dir data/agonal_real \\
            --negative-dir data/icbhi \\
            --esc50-dir    data/ESC-50-master \\
            --n-gasps 0    # skip synthetic if you have enough real clips

    Supported formats: .wav .mp3 .flac .ogg .mp4 .m4a

Usage
-----
    py -3.12 -m audio.train_agonal_detector
    py -3.12 -m audio.train_agonal_detector --positive-dir path/agonal_audio
    py -3.12 -m audio.train_agonal_detector --n-gasps 400 --n-neg 800

Output: audio/models/agonal_detector.joblib
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import joblib
import librosa
import numpy as np
from scipy import signal as sp_signal
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                      cross_val_score)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from tash.audio.config import SAMPLE_RATE as SR

# ---- Segment parameters (Stage A: gasp classifier) ----------------------
WINDOW_S = 2.5
WINDOW_N = int(WINDOW_S * SR)   # 40 000 samples (matches UW paper)

# ---- Feature parameters --------------------------------------------------
N_MELS  = 32
N_MFCC  = 20
N_FFT   = 512
HOP_LEN = 256    # ~16 ms hop inside each 2.5 s window

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "agonal_detector.joblib")

# SNR levels used for noise augmentation of real positive clips
SNR_LEVELS_DB: list[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

# Noise-level tags for stratified evaluation
_NL_SYNTHETIC  = 0   # synthetic gasps / negatives (clean by design)
_NL_CLEAN_REAL = 1   # real clips loaded from --positive-dir (unmodified)
_NL_SNR_AUG    = 2   # SNR-augmented copies of real positives (noisy)
_NL_ICBHI      = 3   # ICBHI real negatives
_NL_ESC50      = 4   # ESC-50 real negatives
_NL_SNORE_MIX  = 5   # gasp overlaid with snoring (targets the gasp+snore weakness)

# Snoring amplitude levels used when mixing with gasps for augmentation
SNORE_ALPHAS: list[float] = [0.3, 0.5, 0.7, 1.0]


# ---------------------------------------------------------------------------
# Synthetic audio generators
# ---------------------------------------------------------------------------

def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = sp_signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sp_signal.sosfilt(sos, x).astype(np.float32)


def gen_gasp(duration_s: float | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """One synthetic agonal gasp: explosive burst 100-2000 Hz, rapid attack.

    Duration: 0.3 - 1.5 s (random if not specified).
    Amplitude: 0.4 - 1.0 (random).
    Result is padded/trimmed to WINDOW_N and placed at a random offset
    within the window so the classifier learns position invariance.
    """
    if rng is None:
        rng = np.random.default_rng()
    gasp_s = duration_s if duration_s else rng.uniform(0.3, 1.5)
    gasp_n = int(gasp_s * SR)

    carrier = _bandpass(
        (np.random.randn(gasp_n) * 0.15).astype(np.float32), 100.0, 2000.0)
    attack = max(1, int(0.05 * gasp_n))
    decay  = gasp_n - attack
    env = np.concatenate([
        np.linspace(0, 1, attack),
        np.exp(-3.0 * np.linspace(0, 1, decay)),
    ]).astype(np.float32)

    # Optional pitch modulation (glottal-like irregularity)
    t = np.linspace(0, gasp_s, gasp_n)
    pitch = 1.0 + 0.12 * np.sin(2 * np.pi * rng.uniform(2, 8) * t).astype(np.float32)
    gasp = carrier * env * pitch * rng.uniform(0.4, 1.0)

    # Place the gasp at a random position within WINDOW_N
    window = (np.random.randn(WINDOW_N) * 0.003).astype(np.float32)   # noise floor
    max_start = max(0, WINDOW_N - gasp_n - 1)
    start = rng.integers(0, max_start + 1) if max_start > 0 else 0
    end   = min(start + gasp_n, WINDOW_N)
    window[start:end] += gasp[:end - start]
    return np.clip(window, -1.0, 1.0)


def gen_normal_breath_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """One 2.5 s window of normal breathing (hard negative)."""
    if rng is None:
        rng = np.random.default_rng()
    bpm   = rng.uniform(12.0, 20.0)
    n     = WINDOW_N
    t     = np.linspace(0, WINDOW_S, n)
    env   = np.clip(np.sin(2 * np.pi * (bpm / 60.0) * t), 0, 1).astype(np.float32)
    carrier = _bandpass((np.random.randn(n) * 0.06).astype(np.float32), 80.0, 1800.0)
    return np.clip(carrier * env + (np.random.randn(n) * 0.004).astype(np.float32), -1, 1)


def gen_slow_breath_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """Slow regular breathing (7-10 BPM) -- hard negative (rate is low but steady)."""
    if rng is None:
        rng = np.random.default_rng()
    bpm   = rng.uniform(7.0, 10.0)
    n     = WINDOW_N
    t     = np.linspace(0, WINDOW_S, n)
    env   = np.clip(np.sin(2 * np.pi * (bpm / 60.0) * t), 0, 1).astype(np.float32)
    carrier = _bandpass((np.random.randn(n) * 0.07).astype(np.float32), 80.0, 2000.0)
    return np.clip(carrier * env, -1, 1)


def gen_snore_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """Realistic snoring: drifting fundamental 50–250 Hz, random pauses, variable harmonics.

    Much harder negative than the old fixed-4-harmonic version because:
      - Fundamental varies over [50, 250] Hz — directly overlapping the agonal range
      - Pitch drifts within the window (real snoring is not perfectly periodic)
      - 3–7 harmonics with random brightness (some snores are brighter than others)
      - Random amplitude envelope at slow breathing rate (~0.2–0.5 Hz)
      - 30% chance of a silence gap (snorer briefly stops then restarts)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = WINDOW_N
    t = np.linspace(0, WINDOW_S, n)

    # Drifting fundamental frequency
    f0    = rng.uniform(50.0, 250.0)
    drift = f0 + rng.uniform(-20.0, 20.0) * np.linspace(0, 1, n)
    phase = np.cumsum(2 * np.pi * drift / SR).astype(np.float32)

    # Variable number of harmonics, amplitude 1/k roll-off with jitter
    n_harmonics = int(rng.integers(3, 8))
    harmonics   = np.zeros(n, dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        harmonics += (rng.uniform(0.02, 0.25) / k * np.sin(k * phase)).astype(np.float32)

    # Slow amplitude envelope (one or two snore cycles in 2.5 s)
    snore_rate = rng.uniform(0.2, 0.5)
    env = np.clip(np.sin(2 * np.pi * snore_rate * t), 0, 1).astype(np.float32)

    # Occasional silence gap (snorer pauses mid-window)
    if rng.random() < 0.3:
        gap_start = int(rng.uniform(0.2, 0.6) * n)
        gap_len   = int(rng.uniform(0.1, 0.4) * n)
        env[gap_start: gap_start + gap_len] = 0.0

    noise = (np.random.randn(n) * 0.015).astype(np.float32)
    return np.clip(harmonics * env + noise, -1.0, 1.0)


def gen_car_window(rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = WINDOW_N
    t = np.linspace(0, WINDOW_S, n)
    engine_hz = rng.uniform(80, 150)
    engine = np.zeros(n, dtype=np.float32)
    for h in range(1, 5):
        engine += (rng.uniform(0.03, 0.12) / h *
                   np.sin(2 * np.pi * engine_hz * h * t)).astype(np.float32)
    road = np.cumsum(np.random.randn(n).astype(np.float64)).astype(np.float32)
    road -= road.mean()
    road /= (np.abs(road).max() + 1e-9)
    return np.clip(engine + road * 0.06, -1, 1)


def gen_silence_window(**_) -> np.ndarray:
    return (np.random.randn(WINDOW_N) * 0.004).astype(np.float32)


def gen_speech_window(rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = WINDOW_N
    t = np.linspace(0, WINDOW_S, n)
    mod = np.clip(np.sin(2 * np.pi * rng.uniform(3, 6) * t), 0, 1).astype(np.float32)
    carrier = _bandpass((np.random.randn(n) * 0.08).astype(np.float32), 200.0, 3500.0)
    return np.clip(carrier * mod, -1, 1)


def gen_hvac_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """HVAC/fan noise: broadband mid-frequency hiss with blade-rate periodicity."""
    if rng is None:
        rng = np.random.default_rng()
    n = WINDOW_N
    t = np.linspace(0, WINDOW_S, n)
    hiss = _bandpass((np.random.randn(n) * 0.07).astype(np.float32), 200.0, 3000.0)
    blade_freq = rng.uniform(15.0, 30.0)
    mod = (1.0 + 0.15 * np.sin(2 * np.pi * blade_freq * t)).astype(np.float32)
    return np.clip(hiss * mod, -1.0, 1.0)


def gen_radio_window(rng: np.random.Generator | None = None) -> np.ndarray:
    """In-cabin radio / music noise: multiple AM-modulated bands."""
    if rng is None:
        rng = np.random.default_rng()
    n = WINDOW_N
    t = np.linspace(0, WINDOW_S, n)
    sig = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(2, 5))):
        mod_freq  = rng.uniform(1.0, 8.0)
        lo = rng.uniform(200.0, 1000.0)
        hi = float(min(lo + rng.uniform(200.0, 800.0), 3500.0))
        carrier = _bandpass((np.random.randn(n) * 0.05).astype(np.float32), lo, hi)
        mod = (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * mod_freq * t))).astype(np.float32)
        sig += carrier * mod
    return np.clip(sig, -1.0, 1.0)


# ---------------------------------------------------------------------------
# SNR mixing helper + augmentation
# ---------------------------------------------------------------------------

def _mix_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix *sig* and *noise* so that the signal-to-noise ratio equals snr_db."""
    sig_f   = sig.astype(np.float64)
    noise_f = noise.astype(np.float64)
    sig_rms   = np.sqrt(np.mean(sig_f ** 2))   + 1e-9
    noise_rms = np.sqrt(np.mean(noise_f ** 2)) + 1e-9
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    mixed = sig_f + noise_f * (target_noise_rms / noise_rms)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def snr_augment_windows(windows: list[np.ndarray],
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Return one noisy copy of each window per SNR level in SNR_LEVELS_DB.

    Noise source is randomly chosen from car/HVAC/radio generators so the model
    is exposed to multiple noise types at each SNR.
    """
    noise_fns = [gen_car_window, gen_hvac_window, gen_radio_window]
    augmented: list[np.ndarray] = []
    for win in windows:
        for snr_db in SNR_LEVELS_DB:
            noise_fn = noise_fns[int(rng.integers(len(noise_fns)))]
            augmented.append(_mix_at_snr(win, noise_fn(rng), snr_db))
    return augmented


# ---------------------------------------------------------------------------
# Feature extraction (per 2.5 s window)
# ---------------------------------------------------------------------------

def extract_features(win: np.ndarray) -> np.ndarray:
    """181-dim feature vector from a 2.5 s window.

    Groups:
      log-mel stats        64  (32 mel bands × mean+std)
      MFCC + delta stats   80  (20 coefficients × 2 stats × 2 sets)
      RMS envelope          4  (mean, std, max, silence fraction)
      envelope periodicity  8  (first 8 FFT bins of RMS envelope)
      ZCR / rolloff         4  (mean+std each)
      spectral contrast    14  (7 subbands × mean+std)
      F0 / voiced stats     3  (F0 mean, F0 std, voiced-frame fraction)
      HPSS percussiveness   4  (harmonic RMS, percussive RMS,
                                percussive ratio, percussive-dominant fraction)
    """
    y = win.astype(np.float32)
    if len(y) < WINDOW_N:
        y = np.pad(y, (0, WINDOW_N - len(y)))
    else:
        y = y[:WINDOW_N]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mel_mean = log_mel.mean(axis=1)
    mel_std  = log_mel.std(axis=1)

    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
    delta = librosa.feature.delta(mfcc)
    mfcc_mean  = mfcc.mean(axis=1)
    mfcc_std   = mfcc.std(axis=1)
    delta_mean = delta.mean(axis=1)
    delta_std  = delta.std(axis=1)

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP_LEN)[0]
    sil_frac = float((rms < 0.008).mean())
    rms_feats = np.array([rms.mean(), rms.std(), rms.max(), sil_frac], dtype=np.float32)

    rms_fft = np.abs(np.fft.rfft(rms - rms.mean(), n=256))
    period_feats = rms_fft[:8].astype(np.float32)

    zcr     = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LEN)[0]
    misc_feats = np.array([zcr.mean(), zcr.std(), rolloff.mean(), rolloff.std()],
                           dtype=np.float32)

    # Spectral contrast: distinguishes the explosive gasp burst from smooth breathing.
    # 7 subbands (6 bands + top) × mean + std = 14 dims.
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN)
    contrast_mean = contrast.mean(axis=1).astype(np.float32)
    contrast_std  = contrast.std(axis=1).astype(np.float32)

    # Fundamental frequency via YIN (80–400 Hz covers agonal-gasp pitch range).
    # voiced_frac captures how much of the window has periodic content — gasps
    # are partly voiced; background noise and silence are mostly unvoiced.
    f0 = librosa.yin(y, fmin=80.0, fmax=400.0, sr=SR, hop_length=HOP_LEN)
    voiced_mask = f0 > 0
    f0_mean     = float(f0[voiced_mask].mean()) if voiced_mask.any() else 0.0
    f0_std      = float(f0[voiced_mask].std())  if voiced_mask.sum() > 1 else 0.0
    voiced_frac = float(voiced_mask.mean())
    f0_feats    = np.array([f0_mean, f0_std, voiced_frac], dtype=np.float32)

    # Harmonic-Percussive Source Separation (HPSS).
    # Agonal gasps are transient/percussive bursts; snoring is sustained/harmonic.
    # The percussive ratio gives the model a direct "see-through-snoring" signal.
    H, P   = librosa.effects.hpss(y)
    h_rms  = float(np.sqrt(np.mean(H ** 2))) + 1e-9
    p_rms  = float(np.sqrt(np.mean(P ** 2))) + 1e-9
    p_ratio = p_rms / (h_rms + p_rms)           # high for gasps, low for snoring
    p_dom   = float(np.mean(P ** 2 > H ** 2))   # fraction where percussive > harmonic
    hpss_feats = np.array([h_rms, p_rms, p_ratio, p_dom], dtype=np.float32)

    return np.concatenate([
        mel_mean, mel_std,
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        rms_feats, period_feats, misc_feats,
        contrast_mean, contrast_std,
        f0_feats,
        hpss_feats,
    ]).astype(np.float32)


def _featurise_one(win_label: tuple[np.ndarray, int]) -> tuple[np.ndarray, int]:
    win, label = win_label
    return (extract_features(win), label)


# ---------------------------------------------------------------------------
# Audio file helpers (WAV, MP3, FLAC, OGG, …)
# ---------------------------------------------------------------------------

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".mp4", ".m4a", ".aac"}


def _icbhi_patient_id(filename: str) -> str:
    """ICBHI 2017 filenames are '<patient>_<rec>_<chest-loc>_<mode>_<device>.wav'.

    The leading number identifies the patient; a patient has multiple
    recordings, so grouping by this ID (not by filename) is required to
    keep a patient's audio entirely in train OR entirely in test.
    """
    return filename.split("_")[0]


def _load_audio_file(path: str) -> np.ndarray | None:
    """Load any audio file to a mono float32 array resampled to SR.

    Tries soundfile first (fast, no extra deps), falls back to librosa
    (handles MP3/M4A via audioread/ffmpeg).
    """
    from math import gcd
    try:
        import soundfile as sf
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            from scipy.signal import resample_poly
            g = gcd(SR, file_sr)
            audio = resample_poly(audio, SR // g, file_sr // g).astype(np.float32)
        return audio
    except Exception:
        pass
    # soundfile failed (e.g. MP3 without libsndfile MPEG support) — try librosa
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"    Warning: could not load {os.path.basename(path)}: {e}")
        return None


def _load_audio_dir(directory: str) -> list[np.ndarray]:
    """Load all supported audio files from a directory."""
    return [audio for _, audio in _load_audio_dir_with_names(directory)]


def _load_audio_dir_with_names(directory: str) -> list[tuple[str, np.ndarray]]:
    """Like _load_audio_dir but keeps the filename, needed to group windows
    from the same source recording for a leakage-safe train/test split."""
    clips = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname.lower())[1] not in _AUDIO_EXTS:
            continue
        audio = _load_audio_file(os.path.join(directory, fname))
        if audio is not None:
            clips.append((fname, audio))
    return clips


def _active_windows(clip: np.ndarray,
                    hop_frac: float = 0.5,
                    rms_threshold: float = 0.04) -> list[np.ndarray]:
    """Slice a clip into WINDOW_N windows, returning only those with enough energy.

    For *positive* agonal clips the silence gaps between gasps must not be
    labeled as positives.  Only windows whose peak RMS frame exceeds
    rms_threshold are kept.  For negative clips pass rms_threshold=0 to
    keep all windows.
    """
    hop_n = int(WINDOW_N * hop_frac)
    windows = []
    for start in range(0, len(clip) - WINDOW_N + 1, hop_n):
        win = clip[start: start + WINDOW_N]
        if rms_threshold > 0:
            rms = librosa.feature.rms(y=win, frame_length=1024, hop_length=256)[0]
            if rms.max() < rms_threshold:
                continue
        windows.append(win)
    return windows


def _esc50_clips(esc50_dir: str, categories: list[str]) -> list[tuple[str, np.ndarray]]:
    """Returns (filename, audio) pairs -- the filename is needed to group a
    clip's windows for a leakage-safe train/test split."""
    meta = os.path.join(esc50_dir, "meta", "esc50.csv")
    if not os.path.isfile(meta):
        print(f"    Warning: no esc50.csv found in {esc50_dir}/meta/")
        return []
    audio_dir = os.path.join(esc50_dir, "audio")
    clips = []
    with open(meta) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            fname, _, _, category = parts[0], parts[1], parts[2], parts[3]
            path = os.path.join(audio_dir, fname)
            if category in categories and os.path.isfile(path):
                audio = _load_audio_file(path)
                if audio is not None:
                    clips.append((fname, audio))
    return clips


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(n_gasps: int = 400,
                   n_neg:   int = 800,
                   positive_dir: str | None = None,
                   negative_dir: str | None = None,
                   esc50_dir:    str | None = None,
                   n_jobs: int = 4,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, noise_labels, groups) where positives are 2.5 s windows
    containing a gasp.

    noise_labels encodes each sample's origin for stratified evaluation:
      0 (_NL_SYNTHETIC)  — synthetic gasps / negatives (clean)
      1 (_NL_CLEAN_REAL) — real clips from --positive-dir (unmodified)
      2 (_NL_SNR_AUG)    — SNR-augmented copies of real positives (noisy)
      3 (_NL_ICBHI)      — ICBHI real negatives
      4 (_NL_ESC50)      — ESC-50 real negatives
      5 (_NL_SNORE_MIX)  — gasp + snoring overlay (targets gasp+snore weakness)

    groups identifies each sample's independence unit for a leakage-safe
    split (see GroupShuffleSplit/GroupKFold in train()):
      - ICBHI windows                     -> the ICBHI patient ID (a patient
                                              has multiple recordings; all of
                                              them must land on the same side
                                              of the split)
      - --positive-dir windows and every SNR/snore-mix copy derived from them
                                           -> the source filename (same voice
                                              / recording session)
      - synthetic gasps and every snore-mix copy derived from them
                                           -> a per-base-gasp id (same
                                              underlying waveform, only the
                                              added noise differs)
      - ESC-50 clips                      -> the source filename
      - other synthetic negatives (independent draws, no shared lineage)
                                           -> a unique id each
    Splitting on raw row index (a plain (Stratified)KFold/train_test_split)
    lets augmented siblings or a patient's own recordings appear on both
    sides of the split -- always split on `groups` instead.
    """
    rng = np.random.default_rng(42)
    pairs:        list[tuple[np.ndarray, int]] = []
    noise_labels: list[int] = []
    groups:       list[str] = []
    _gid = [0]

    def new_gid(prefix: str) -> str:
        _gid[0] += 1
        return f"{prefix}:{_gid[0]}"

    # ---- POSITIVES: windows with a gasp -----------------------------------
    print(f"Generating {n_gasps} synthetic gasp windows (positives)...")
    synth_gasp_list: list[np.ndarray] = []
    synth_gasp_gids: list[str] = []
    for _ in range(n_gasps):
        g = gen_gasp(rng=rng)
        gid = new_gid("synth_gasp")
        pairs.append((g, 1))
        noise_labels.append(_NL_SYNTHETIC)
        groups.append(gid)
        synth_gasp_list.append(g)
        synth_gasp_gids.append(gid)

    # Synthetic gasp + snoring mixtures (covers the snoring-overlap case even
    # when no real positive dir is provided). Shares its parent gasp's group
    # id: it's the same base waveform, just mixed with extra noise.
    n_synth_snore = min(200, n_gasps)
    for win, gid in zip(synth_gasp_list[:n_synth_snore], synth_gasp_gids[:n_synth_snore]):
        for alpha in SNORE_ALPHAS:
            snore = gen_snore_window(rng)
            mixed = np.clip(win + snore * alpha, -1.0, 1.0).astype(np.float32)
            pairs.append((mixed, 1))
            noise_labels.append(_NL_SNORE_MIX)
            groups.append(gid)
    print(f"    => {n_synth_snore * len(SNORE_ALPHAS)} synthetic gasp+snore windows added")

    if positive_dir and os.path.isdir(positive_dir):
        real = _load_audio_dir_with_names(positive_dir)
        print(f"  + {len(real)} real positive audio files")
        real_win_list: list[np.ndarray] = []
        real_win_gids: list[str] = []
        real_wins = 0
        for fname, clip in real:
            gid = f"real_pos:{fname}"
            wins = _active_windows(clip, hop_frac=0.5, rms_threshold=0.04)
            for w in wins:
                pairs.append((w, 1))
                noise_labels.append(_NL_CLEAN_REAL)
                groups.append(gid)
                real_win_list.append(w)
                real_win_gids.append(gid)
            real_wins += len(wins)
        print(f"    => {real_wins} active windows extracted "
              f"(silent gaps skipped to avoid mislabeling)")

        # SNR augmentation: each real positive × 6 SNR levels × random noise type
        if real_win_list:
            aug_wins = snr_augment_windows(real_win_list, rng)
            aug_gids = [gid for gid in real_win_gids for _ in SNR_LEVELS_DB]
            for aug, gid in zip(aug_wins, aug_gids):
                pairs.append((aug, 1))
                noise_labels.append(_NL_SNR_AUG)
                groups.append(gid)
            print(f"    => {len(aug_wins)} SNR-augmented windows added "
                  f"({len(SNR_LEVELS_DB)} levels × {len(real_win_list)} windows, "
                  f"car/HVAC/radio noise)")

            # Snore-mix augmentation: real positive × 4 snore alpha levels
            # Directly trains the model to detect gasps through snoring overlap.
            n_snore_aug = 0
            for win, gid in zip(real_win_list, real_win_gids):
                for alpha in SNORE_ALPHAS:
                    snore = gen_snore_window(rng)
                    mixed = np.clip(win + snore * alpha, -1.0, 1.0).astype(np.float32)
                    pairs.append((mixed, 1))
                    noise_labels.append(_NL_SNORE_MIX)
                    groups.append(gid)
                    n_snore_aug += 1
            print(f"    => {n_snore_aug} gasp+snore mixed windows added "
                  f"({len(SNORE_ALPHAS)} alpha levels × {len(real_win_list)} windows)")

    # ---- NEGATIVES: windows WITHOUT a gasp --------------------------------
    neg_fns = [gen_normal_breath_window, gen_slow_breath_window,
               gen_snore_window, gen_car_window, gen_speech_window,
               gen_hvac_window, gen_radio_window]
    neg_per_class = max(1, n_neg // len(neg_fns))
    print(f"Generating ~{n_neg} negative windows ({neg_per_class} per class, "
          f"{len(neg_fns)} classes)...")
    for fn in neg_fns:
        for _ in range(neg_per_class):
            pairs.append((fn(rng), 0))
            noise_labels.append(_NL_SYNTHETIC)
            groups.append(new_gid("synth_neg"))

    # Pure silence: avoids false alarms on quiet car cabins
    for _ in range(neg_per_class):
        pairs.append((gen_silence_window(), 0))
        noise_labels.append(_NL_SYNTHETIC)
        groups.append(new_gid("synth_neg"))

    # Snore-balance negatives: plain snoring at multiple amplitudes.
    # Required because gasp+snore augmentation (positives) teaches the model that
    # "audio containing snoring" can be a gasp — we must also teach it that snoring
    # alone, at any volume up to 2x normal, is NOT a gasp.
    # Target: roughly match the total gasp+snore positive count so the model sees
    # balanced evidence on both sides of the snore boundary.
    n_snore_pos = n_synth_snore * len(SNORE_ALPHAS)
    if positive_dir and os.path.isdir(positive_dir):
        # account for real gasp+snore augmentation already added
        n_snore_pos += len(real_win_list) * len(SNORE_ALPHAS)  # type: ignore[name-defined]
    snore_vol_levels = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    n_snore_bal = max(neg_per_class * 2,
                      n_snore_pos // len(snore_vol_levels))
    print(f"    + {n_snore_bal * len(snore_vol_levels)} snore-balance negatives "
          f"({n_snore_bal} × {len(snore_vol_levels)} amplitude levels) "
          f"to match {n_snore_pos} gasp+snore positives")
    for _ in range(n_snore_bal):
        for vol in snore_vol_levels:
            snore = gen_snore_window(rng)
            win   = np.clip(snore * vol, -1.0, 1.0).astype(np.float32)
            pairs.append((win, 0))
            noise_labels.append(_NL_SYNTHETIC)
            groups.append(new_gid("synth_neg"))

    if negative_dir and os.path.isdir(negative_dir):
        real = _load_audio_dir_with_names(negative_dir)
        print(f"  + {len(real)} real negative audio files (ICBHI)")
        neg_wins = 0
        for fname, clip in real:
            gid = f"icbhi_patient:{_icbhi_patient_id(fname)}"
            wins = _active_windows(clip, hop_frac=0.5, rms_threshold=0.0)
            for w in wins:
                pairs.append((w, 0))
                noise_labels.append(_NL_ICBHI)
                groups.append(gid)
            neg_wins += len(wins)
        print(f"    => {neg_wins} windows extracted")

    if esc50_dir and os.path.isdir(esc50_dir):
        esc_cats = ["breathing", "snoring", "engine", "car_horn",
                    "crying_baby", "footsteps", "laughing",
                    "rain", "wind", "vacuum_cleaner", "washing_machine",
                    "thunderstorm", "water_drops", "pouring_water"]
        esc_clips = _esc50_clips(esc50_dir, esc_cats)
        print(f"  + {len(esc_clips)} ESC-50 clips as negatives")
        for fname, clip in esc_clips:
            gid = f"esc50:{fname}"
            for start in range(0, len(clip) - WINDOW_N, WINDOW_N // 2):
                pairs.append((clip[start: start + WINDOW_N], 0))
                noise_labels.append(_NL_ESC50)
                groups.append(gid)

    # ---- Parallel feature extraction --------------------------------------
    print(f"Extracting features from {len(pairs)} windows (n_jobs={n_jobs})...")
    t0 = time.perf_counter()
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(_featurise_one)(p) for p in pairs)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    X = np.stack([r[0] for r in results])
    y = np.array([r[1] for r in results], dtype=np.int32)
    nl = np.array(noise_labels, dtype=np.int32)
    grp = np.array(groups)
    print(f"  {X.shape[0]} windows, {y.sum()} pos / {(y==0).sum()} neg, "
          f"{X.shape[1]} features, {len(set(groups))} independent groups "
          f"(patients / source files / base-sample lineages)")
    return X, y, nl, grp


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_classifier() -> tuple[object, str]:
    """Return (classifier, name).  Prefers LightGBM; falls back to sklearn HGBC."""
    try:
        import lightgbm as lgb  # type: ignore[import]
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        )
        return clf, "LightGBM"
    except ImportError:
        pass

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42,
    )
    return clf, "HistGradientBoosting"


def train(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
          test_size: float = 0.2,
          ) -> tuple[SkPipeline, np.ndarray, np.ndarray, dict]:
    """Fit the gasp classifier with a leakage-safe, group-held-out split.

    Returns (pipe, train_idx, test_idx, held_out_metrics). `pipe` is fit ONLY
    on the train split -- it is what gets saved -- so held_out_metrics (scored
    on windows whose entire patient/source-file/base-sample group was excluded
    from fitting) is a trustworthy generalization estimate, unlike the old
    in-sample pipe.fit(X, y); pipe.predict(X) report.
    """
    base, clf_name = _make_classifier()
    print(f"\nTraining {clf_name}...")
    t0 = time.perf_counter()

    # Group-held-out split: an entire ICBHI patient, source recording, or
    # synthetic-gasp lineage lands ENTIRELY in train or ENTIRELY in test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test,  y_test               = X[test_idx],  y[test_idx]
    assert not set(groups_train) & set(groups[test_idx]), "group leaked across the split"
    print(f"  Held-out split: {len(train_idx)} train / {len(test_idx)} test windows "
          f"({len(set(groups_train))} / {len(set(groups[test_idx]))} groups, "
          f"0 groups shared between them)")

    # Tree-based models are scale-invariant, but keeping StandardScaler in the
    # pipeline preserves a consistent bundle schema for future clf swaps.
    pipe = SkPipeline([("scaler", StandardScaler()), ("clf", base)])

    # Group-aware CV on the TRAIN split only -- never touches the held-out test
    # rows, so it can't leak into the final held-out numbers below.
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv.split(X_train, y_train, groups_train),
                                 scoring="f1", n_jobs=-1)
    print(f"  5-fold group-aware CV F1 (train split only): "
          f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Per-fold: {' '.join(f'{s:.3f}' for s in cv_scores)}")

    pipe.fit(X_train, y_train)   # scaler + classifier fit on TRAIN ONLY

    # TRAIN-split report: same fitted pipe, scored on the data it was fit on.
    # Printed purely so it can be compared side-by-side against the held-out
    # report below -- a big gap would indicate overfitting; a small gap with
    # both scores high means the model has genuinely learned the task.
    y_train_pred = pipe.predict(X_train)
    train_metrics = {
        "accuracy":  accuracy_score(y_train, y_train_pred),
        "f1":        f1_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred),
        "recall":    recall_score(y_train, y_train_pred),
        "n_train":   int(len(train_idx)),
    }
    print(f"\n  TRAIN-split report (fit on this data -- compare against held-out "
          f"below to check for overfitting):")
    print(f"    accuracy={train_metrics['accuracy']:.3f}  f1={train_metrics['f1']:.3f}  "
          f"precision={train_metrics['precision']:.3f}  recall={train_metrics['recall']:.3f}")

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    print("\n  HELD-OUT TEST report (no shared patient/source/lineage with train):")
    print(classification_report(y_test, y_pred,
                                  target_names=["no-gasp", "gasp"], digits=3))
    held_out = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_prob),
        "n_test":    int(len(test_idx)),
        "train_metrics": train_metrics,
    }
    print(f"  Held-out AUC: {held_out['auc']:.4f}")
    print(f"  Train vs. held-out gap: accuracy {train_metrics['accuracy']-held_out['accuracy']:+.3f}  "
          f"f1 {train_metrics['f1']-held_out['f1']:+.3f}")
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")
    return pipe, train_idx, test_idx, held_out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model(pipe: SkPipeline, path: str = MODEL_PATH,
               held_out_metrics: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scaler = pipe.named_steps.get("scaler")
    feature_dim = int(scaler.n_features_in_) if scaler is not None else None
    bundle = {
        "pipeline":    pipe,
        "window_n":    WINDOW_N,
        "window_s":    WINDOW_S,
        "sample_rate": SR,
        "n_mels":      N_MELS,
        "n_mfcc":      N_MFCC,
        "n_fft":       N_FFT,
        "hop_len":     HOP_LEN,
        "model_type":  "gasp_detector",
        "feature_dim": feature_dim,
        # Metrics from train()'s group-held-out split (no shared ICBHI
        # patient / source recording / synthetic lineage with the training
        # data) -- the trustworthy generalization estimate for this model.
        "held_out_metrics": held_out_metrics,
    }
    joblib.dump(bundle, path, compress=3)
    size_kb = os.path.getsize(path) / 1024
    print(f"\nModel saved: {path}  ({size_kb:.0f} KB, {feature_dim} features)")


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Agonal model not found: {path}\n"
            "Run: py -3.12 -m audio.train_agonal_detector")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Stratified evaluation (quiet vs noisy environments)
# ---------------------------------------------------------------------------

_NL_NAMES = {
    _NL_SYNTHETIC:  "synthetic (quiet)",
    _NL_CLEAN_REAL: "clean real (quiet)",
    _NL_SNR_AUG:    "SNR-augmented (noisy)",
    _NL_ICBHI:      "ICBHI negatives",
    _NL_ESC50:      "ESC-50 negatives",
    _NL_SNORE_MIX:  "gasp+snore mixed",
}

_QUIET_LABELS = {_NL_SYNTHETIC, _NL_CLEAN_REAL, _NL_ICBHI, _NL_ESC50}
_NOISY_LABELS = {_NL_SNR_AUG}


def stratified_eval(pipe: SkPipeline,
                    X: np.ndarray,
                    y: np.ndarray,
                    noise_labels: np.ndarray) -> None:
    """Report F1 / precision / recall on quiet and noisy subsets separately.

    Quiet  = synthetic + clean real recordings (no extra noise overlay).
    Noisy  = SNR-augmented windows (car/HVAC/radio noise at -5 to +20 dB SNR).

    Use this to spot if the model degrades under noise or over-fits to clean data.
    """
    print("\n" + "=" * 60)
    print("Stratified Evaluation — Quiet vs Noisy Environments")
    print("=" * 60)

    subsets = {
        "quiet (no noise overlay)":   np.isin(noise_labels, list(_QUIET_LABELS)),
        "noisy (SNR-augmented)":       np.isin(noise_labels, list(_NOISY_LABELS)),
        "all samples":                 np.ones(len(y), dtype=bool),
    }

    for name, mask in subsets.items():
        n = int(mask.sum())
        if n < 2:
            print(f"  [{name}]  too few samples ({n}), skipping")
            continue
        X_s, y_s = X[mask], y[mask]
        y_pred = pipe.predict(X_s)
        y_prob = pipe.predict_proba(X_s)[:, 1]
        f1   = f1_score(y_s, y_pred, zero_division=0)
        prec = precision_score(y_s, y_pred, zero_division=0)
        rec  = recall_score(y_s, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_s, y_prob)
        except ValueError:
            auc = float("nan")
        n_pos = int(y_s.sum())
        n_neg = n - n_pos
        print(f"\n  [{name}]  n={n}  ({n_pos} pos / {n_neg} neg)")
        print(f"    F1={f1:.3f}   Precision={prec:.3f}   Recall={rec:.3f}   AUC={auc:.4f}")

    # Per-source breakdown
    print("\n  Per-source breakdown:")
    for label, label_name in sorted(_NL_NAMES.items()):
        mask = noise_labels == label
        if mask.sum() < 2:
            continue
        y_s    = y[mask]
        y_pred = pipe.predict(X[mask])
        f1 = f1_score(y_s, y_pred, zero_division=0)
        print(f"    {label_name:<30}  n={mask.sum():5d}  F1={f1:.3f}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", default=None)
    parser.add_argument("--negative-dir", default=None)
    parser.add_argument("--esc50-dir",    default=None)
    parser.add_argument("--n-gasps", type=int, default=400,
                        help="Synthetic gasp windows for positives")
    parser.add_argument("--n-neg",   type=int, default=800,
                        help="Negative windows (split across 5+1 classes)")
    parser.add_argument("--n-jobs",  type=int, default=4)
    args = parser.parse_args()

    print("Agonal Gasp Detector -- Training")
    print(f"  Window {WINDOW_S}s  SR {SR}Hz  N_MELS {N_MELS}  N_MFCC {N_MFCC}")

    X, y, noise_labels, groups = build_dataset(
        n_gasps=args.n_gasps, n_neg=args.n_neg,
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        esc50_dir=args.esc50_dir,
        n_jobs=args.n_jobs,
    )
    pipe, train_idx, test_idx, held_out_metrics = train(X, y, groups)
    # Stratify on the held-out split only -- these are the windows the model
    # never saw, from patients/recordings/lineages excluded from training.
    stratified_eval(pipe, X[test_idx], y[test_idx], noise_labels[test_idx])
    save_model(pipe, held_out_metrics=held_out_metrics)

    if args.positive_dir:
        print("\nTrained with real audio from:", args.positive_dir)
        print("Rerun anytime to add more clips to the same directory.")
    else:
        print("\nNOTE: trained on synthetic gasps only.")
        print("For better accuracy, add real agonal recordings:")
        print("  theSimTech (free MP3): https://www.thesimtech.org/av-stimuli/audio/")
        print("  ICBHI negatives:       https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge")
        print("  UW dataset (request):  https://cardiacalert.cs.washington.edu/")
        print()
        print("Once downloaded, run:")
        print("  py -3.12 -m audio.train_agonal_detector --positive-dir data/agonal_real")


if __name__ == "__main__":
    main()
