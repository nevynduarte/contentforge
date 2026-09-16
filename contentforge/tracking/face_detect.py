"""Face detection + landmarks on the MediaPipe Tasks API (works on mediapipe 0.10.x and 1.x).

Falls back to OpenCV Haar cascades when mediapipe is not installed.
Mouth openness is exposed so speaker_focus can estimate who is talking from lip motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..utils.paths import model_path


@dataclass
class Face:
    x: float        # normalized [0,1] box
    y: float
    w: float
    h: float
    score: float = 1.0
    mouth_open: float = 0.0     # lip gap / face height, 0 when unknown

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


_detector = None
_landmarker = None
_cascade = None

# FaceLandmarker indices: upper inner lip 13, lower inner lip 14, forehead 10, chin 152
_UP, _LOW, _TOP, _CHIN = 13, 14, 10, 152


def _mp():
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mpp  # type: ignore
    from mediapipe.tasks.python import vision  # type: ignore
    return mp, mpp, vision


def _get_detector():
    global _detector
    if _detector is None:
        mp, mpp, vision = _mp()
        opts = vision.FaceDetectorOptions(base_options=mpp.BaseOptions(model_asset_path=str(model_path("blaze_face_short_range.tflite"))),
                                          min_detection_confidence=0.5)
        _detector = vision.FaceDetector.create_from_options(opts)
    return _detector


def _get_landmarker(max_faces: int = 2):
    global _landmarker
    if _landmarker is None:
        mp, mpp, vision = _mp()
        opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(model_path("face_landmarker.task"))),
                                            num_faces=max_faces, min_face_detection_confidence=0.5)
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def detect_faces(img: Image.Image, landmarks: bool = True, max_faces: int = 2) -> list[Face]:
    """Detect faces; with landmarks=True also returns mouth openness (needed for speaker detection)."""
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    h, w = arr.shape[:2]
    try:
        mp, _, _ = _mp()
    except ImportError:
        return _haar(arr)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    if landmarks:
        res = _get_landmarker(max_faces).detect(mp_img)
        out = []
        for lm in res.face_landmarks:
            xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            face_h = max(1e-6, lm[_CHIN].y - lm[_TOP].y)
            gap = abs(lm[_LOW].y - lm[_UP].y) / face_h
            out.append(Face(float(x0), float(y0), float(x1 - x0), float(y1 - y0), 1.0, float(gap)))
        if out:
            return out
    res = _get_detector().detect(mp_img)
    out = []
    for d in res.detections:
        bb = d.bounding_box
        out.append(Face(bb.origin_x / w, bb.origin_y / h, bb.width / w, bb.height / h, float(d.categories[0].score)))
    return out


def _haar(arr: np.ndarray) -> list[Face]:
    global _cascade
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    return [Face(x / w, y / h, fw / w, fh / h, 0.8) for (x, y, fw, fh) in _cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))]
