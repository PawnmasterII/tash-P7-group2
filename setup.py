"""Install the tash package so 'python -m tash.main' resolves correctly.

The repo root IS the tash package (package_dir={"tash": "."}).
All sub-directories with __init__.py become tash.<subdir>.

Usage:
    pip install -e .        # editable install from repo root
    python -m tash.main     # run from any directory after install
"""
from setuptools import setup

PACKAGES = [
    "tash",
    "tash.audio",
    "tash.comms",
    "tash.core",
    "tash.demo",
    "tash.detectors",
    "tash.fusion",
    "tash.live",
    "tash.response",
    "tash.sensors",
    "tash.vehicle",
]

setup(
    name="tash",
    version="0.1.0",
    python_requires=">=3.12",
    packages=PACKAGES,
    package_dir={"tash": "."},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "noisereduce>=3.0.0",
        "vosk>=0.3.45",
        "librosa>=0.10",
        "soundfile>=0.12",
    ],
    extras_require={
        "vad":  ["webrtcvad-wheels>=2.0.10"],
        "live": [
            "pvrecorder>=1.2",
            "mediapipe>=0.10.0",
            "opencv-python>=4.8.0",
            "pyttsx3>=2.90",
        ],
    },
)
