"""Download the Vosk small-EN model into tash/audio/models/.

Usage:
    python -m tash.audio.download_model

Idempotent: skips the download if the model directory already exists.
The model is ~40 MB and is excluded from git (see .gitignore).
"""
from __future__ import annotations

import os
import urllib.request
import zipfile

from tash.audio.config import VOSK_MODEL_PATH

_MODEL_URL = (
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
)
_MODELS_DIR = os.path.dirname(VOSK_MODEL_PATH)
_ZIP_PATH = os.path.join(_MODELS_DIR, "vosk-model-small-en-us-0.15.zip")


def download() -> None:
    if os.path.isdir(VOSK_MODEL_PATH):
        print(f"Model already present at {VOSK_MODEL_PATH} — nothing to do.")
        return

    os.makedirs(_MODELS_DIR, exist_ok=True)
    print(f"Downloading Vosk model to {_ZIP_PATH} …")
    urllib.request.urlretrieve(_MODEL_URL, _ZIP_PATH)

    print("Unpacking …")
    with zipfile.ZipFile(_ZIP_PATH) as z:
        z.extractall(_MODELS_DIR)

    os.remove(_ZIP_PATH)
    print(f"Done. Model at {VOSK_MODEL_PATH}")


if __name__ == "__main__":
    download()
