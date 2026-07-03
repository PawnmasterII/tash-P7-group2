"""Download and extract additional training data.

Sources
-------
1. ICBHI 2017 (already in repo as ZIP) — extract all 920 WAV files to data/icbhi_full/
   Currently only 81 files are used.  This gives 11x more real respiratory negatives.

2. ESC-50 Environmental Sound Classification (CC-BY, GitHub)
   2000 labelled clips: breathing, snoring, engine, car_horn, footsteps, etc.
   Useful as both hard negatives and environment coverage.

3. UrbanSound8K (free research, Zenodo)
   8732 clips of 10 urban sound classes (car_horn, engine, street_music, etc.)
   Good in-cabin noise negatives.

Usage
-----
    py -3.12 scripts/download_data.py

Then retrain with:
    py -3.12 -m audio.train_agonal_detector \\
        --positive-dir data/agonal_real \\
        --negative-dir data/icbhi_full \\
        --esc50-dir    data/ESC-50-master
"""
from __future__ import annotations
import os
import sys
import zipfile
import urllib.request
import shutil
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, "data")


def _bar(done: int, total: int, width: int = 40) -> str:
    pct  = done / max(total, 1)
    fill = int(pct * width)
    return f"[{'#'*fill}{' '*(width-fill)}] {pct:5.1%}  {done//1024//1024:4d}/{total//1024//1024:4d} MB"


def _download(url: str, dest: str, label: str) -> bool:
    """Download url → dest with a progress bar. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            t0    = time.time()
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 17)   # 128 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    elapsed = time.time() - t0
                    speed   = done / max(elapsed, 0.1) / 1024 / 1024
                    bar     = _bar(done, total)
                    print(f"\r  {label}: {bar}  {speed:.1f} MB/s", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  Warning: could not download {label}: {e}")
        if os.path.isfile(dest):
            os.remove(dest)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 1. ICBHI full database (ZIP already in repo)
# ──────────────────────────────────────────────────────────────────────────────

def extract_icbhi() -> str:
    """Extract all 920 WAV files from the ICBHI ZIP to data/icbhi_full/."""
    zip_path = os.path.join(DATA_DIR, "ICBHI_final_database.zip")
    out_dir  = os.path.join(DATA_DIR, "icbhi_full")

    if not os.path.isfile(zip_path):
        print("  ICBHI ZIP not found — skipping")
        return ""

    existing_wavs = sum(1 for f in os.listdir(out_dir) if f.lower().endswith(".wav")) \
        if os.path.isdir(out_dir) else 0

    with zipfile.ZipFile(zip_path) as z:
        all_wavs = [n for n in z.namelist() if n.lower().endswith(".wav")]

    if existing_wavs >= len(all_wavs):
        print(f"  ICBHI already extracted: {existing_wavs} WAV files in {out_dir}")
        return out_dir

    print(f"  Extracting {len(all_wavs)} WAV files from ICBHI ZIP...")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for i, name in enumerate(all_wavs, 1):
            dest_name = os.path.basename(name)
            dest_path = os.path.join(out_dir, dest_name)
            if not os.path.isfile(dest_path):
                with z.open(name) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if i % 100 == 0 or i == len(all_wavs):
                print(f"    {i}/{len(all_wavs)} files extracted", end="\r")
    print(f"\n  Done. {len(all_wavs)} WAV files in {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# 2. ESC-50  (CC-BY, GitHub)
# ──────────────────────────────────────────────────────────────────────────────

ESC50_URL  = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_FALLBACK = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

def download_esc50() -> str:
    out_dir  = os.path.join(DATA_DIR, "ESC-50-master")
    zip_dest = os.path.join(DATA_DIR, "_esc50.zip")

    # Already extracted?
    if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "meta", "esc50.csv")):
        n_audio = sum(1 for f in os.listdir(os.path.join(out_dir, "audio"))
                      if f.endswith(".wav"))
        print(f"  ESC-50 already present: {n_audio} clips in {out_dir}")
        return out_dir

    print("  Downloading ESC-50...")
    ok = _download(ESC50_URL, zip_dest, "ESC-50")
    if not ok:
        ok = _download(ESC50_FALLBACK, zip_dest, "ESC-50 (fallback)")
    if not ok:
        return ""

    print("  Extracting ESC-50...")
    with zipfile.ZipFile(zip_dest) as z:
        # The archive root might be ESC-50-master/ or similar
        members = z.namelist()
        audio_files = [m for m in members if "/audio/" in m and m.endswith(".wav")]
        meta_files  = [m for m in members if "/meta/" in m]
        print(f"    {len(audio_files)} audio clips, {len(meta_files)} meta files")
        z.extractall(DATA_DIR)

    # Rename extracted root folder to ESC-50-master if needed
    for name in os.listdir(DATA_DIR):
        full = os.path.join(DATA_DIR, name)
        if os.path.isdir(full) and name.startswith("ESC-50") and name != "ESC-50-master":
            os.rename(full, out_dir)
            break

    os.remove(zip_dest)

    if os.path.isdir(out_dir):
        n_audio = sum(1 for f in os.listdir(os.path.join(out_dir, "audio"))
                      if f.endswith(".wav"))
        print(f"  Done. {n_audio} clips in {out_dir}")
        return out_dir
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 3. UrbanSound8K (free for research, Zenodo)
# ──────────────────────────────────────────────────────────────────────────────

URBANSOUND_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"

def download_urbansound() -> str:
    """Download UrbanSound8K if not already present. Returns path or ''."""
    out_dir  = os.path.join(DATA_DIR, "UrbanSound8K")
    tar_dest = os.path.join(DATA_DIR, "_urbansound.tar.gz")

    if os.path.isdir(out_dir):
        n = sum(len(files) for _, _, files in os.walk(out_dir)
                if any(f.endswith(".wav") for f in _))
        print(f"  UrbanSound8K already present in {out_dir}")
        return out_dir

    print("  Downloading UrbanSound8K (~6 GB) — this takes a while...")
    ok = _download(URBANSOUND_URL, tar_dest, "UrbanSound8K")
    if not ok:
        print("  Skipping UrbanSound8K (download failed — try manually from https://urbansounddataset.weebly.com/)")
        return ""

    print("  Extracting UrbanSound8K...")
    import tarfile
    with tarfile.open(tar_dest, "r:gz") as t:
        t.extractall(DATA_DIR)
    os.remove(tar_dest)
    print(f"  Done. UrbanSound8K in {out_dir}")
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Data acquisition for agonal breathing detector")
    print("=" * 60)

    results: dict[str, str] = {}

    # 1. ICBHI full
    print("\n[1/3] ICBHI 2017 Respiratory Sound Database (full)")
    results["icbhi"] = extract_icbhi()

    # 2. ESC-50
    print("\n[2/3] ESC-50 Environmental Sound Classification")
    results["esc50"] = download_esc50()

    # 3. UrbanSound8K (large — skip if user doesn't want to wait)
    print("\n[3/3] UrbanSound8K (optional, ~6 GB)")
    ans = input("  Download UrbanSound8K? (y/N): ").strip().lower()
    if ans == "y":
        results["urbansound"] = download_urbansound()
    else:
        print("  Skipped.")
        results["urbansound"] = ""

    # Summary + retrain command
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, path in results.items():
        status = path if path else "(not downloaded)"
        print(f"  {name:<14}  {status}")

    print("\nRecommended training command:")
    neg_dir  = results.get("icbhi")  or "data/negatives_real"
    esc_arg  = f"\n    --esc50-dir    {results['esc50']}" if results.get("esc50") else ""
    print(f"""
    py -3.12 -m audio.train_agonal_detector \\
        --positive-dir data/agonal_real \\
        --negative-dir {neg_dir}{esc_arg}
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
