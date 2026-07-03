"""Webcam posture sensor — MediaPipe Pose → slump_angle_deg.

Uses OpenCV to capture frames and MediaPipe Pose to estimate the angle of
the passenger's head/torso relative to vertical.  The annotated frame is
written to a shared ``LiveState`` object so the display thread can show it
without re-running inference.

Slump angle definition (front-facing camera):
  shoulder_mid = average of LEFT_SHOULDER + RIGHT_SHOULDER (normalised coords)
  nose         = NOSE landmark
  dx, dy       = pixel offset of nose from shoulder_mid
                 (dy > 0 means nose is *above* shoulders — upright)
  angle        = atan2(|dx|, dy)  in degrees
                 → 0° when perfectly upright, grows as head droops forward.

Thresholds in SlumpDetector:  WATCH ≥ 25°,  CHECK_IN ≥ 45°.

Import guard: cv2 and mediapipe are optional dependencies.  The sensor raises
a clear RuntimeError at construction time if they are not installed so that
the rest of the package can still be imported without them.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from tash.sensors.base import Sensor
from tash.types import Modality, SensorReading

if TYPE_CHECKING:
    import numpy as np

_FRAME_PERIOD_S = 1 / 15  # target ~15 fps for pose inference


def _check_deps() -> tuple[Any, Any, Any]:
    """Return (cv2, mp, mp_pose) or raise RuntimeError."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python not installed.\n"
            "  pip install opencv-python"
        )
    try:
        import mediapipe as mp
    except ImportError:
        raise RuntimeError(
            "mediapipe not installed.\n"
            "  pip install mediapipe"
        )
    return cv2, mp, mp.solutions.pose


def _calc_slump_angle(landmarks: Any, width: int, height: int) -> float:
    """Return slump angle in degrees from MediaPipe Pose landmarks."""
    import mediapipe as mp
    lm = landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    ls = lm[PL.LEFT_SHOULDER]
    rs = lm[PL.RIGHT_SHOULDER]
    nose = lm[PL.NOSE]

    smx = (ls.x + rs.x) / 2
    smy = (ls.y + rs.y) / 2

    dx = (nose.x - smx) * width
    dy = (smy - nose.y) * height  # positive = nose above shoulder_mid

    if dy < 5:
        # Nose at or below shoulders — extreme slump or body not fully visible
        return 90.0

    return math.degrees(math.atan2(abs(dx), dy))


class WebcamPostureSensor(Sensor):
    """Captures webcam frames, runs MediaPipe Pose, yields slump readings.

    Parameters
    ----------
    state :
        Shared ``LiveState`` object.  The annotated frame, angle, and
        landmark visibility are written here for the display thread.
    camera_index :
        OpenCV device index (0 = default webcam).
    """

    modality = Modality.POSTURE

    def __init__(self, state: Any, camera_index: int = 0) -> None:
        self._state = state
        self._camera_index = camera_index
        self._cv2, self._mp, self._mp_pose = _check_deps()
        self._cap: Any = None
        self._pose: Any = None
        self._drawing: Any = None

    async def start(self) -> None:
        cv2 = self._cv2
        mp = self._mp
        mp_pose = self._mp_pose

        def _open() -> None:
            self._cap = cv2.VideoCapture(self._camera_index)
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,          # fastest model
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._drawing = mp.solutions.drawing_utils
            self._drawing_styles = mp.solutions.drawing_styles

        await asyncio.to_thread(_open)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self._camera_index}. "
                "Check that a camera is connected and not in use by another app."
            )

    async def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._pose is not None:
            self._pose.close()

    async def stream(self) -> AsyncIterator[SensorReading]:  # type: ignore[override]
        cv2 = self._cv2
        mp_pose = self._mp_pose

        def _read_and_process() -> tuple[float, Any]:
            """Blocking: capture frame + run MediaPipe. Returns (angle, annotated_bgr)."""
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return 0.0, None

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._pose.process(rgb)
            rgb.flags.writeable = True

            annotated = frame.copy()
            angle = 0.0

            if results.pose_landmarks:
                angle = _calc_slump_angle(results.pose_landmarks, w, h)
                self._drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=self._drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2
                    ),
                )

            return angle, annotated

        while True:
            angle, annotated = await asyncio.to_thread(_read_and_process)

            if annotated is not None:
                import threading
                with self._state.lock:
                    self._state.frame = annotated
                    self._state.angle = angle

                yield SensorReading(
                    modality=self.modality,
                    payload={
                        "slump_angle_deg": angle,
                        "pose_quality": 1.0 if angle < 89.0 else 0.0,
                        "vision_latency_ms": int(_FRAME_PERIOD_S * 1000),
                    },
                )

            await asyncio.sleep(0)  # yield to event loop between frames
