"""Face detection. Backends: MediaPipe (preferred, CPU-fast) -> OpenCV Haar cascade -> none.

Install: pip install contentforge[tracking]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Face:
    x: float        # normalized [0,1] box
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


_mp_detector = None
_cv_cascade = None


def detect_faces(img: Image.Image, min_score: float = 0.5) -> list[Face]:
    global _mp_detector, _cv_cascade
    arr = np.asarray(img.convert("RGB"))
    try:
        import mediapipe as mp  # type: ignore
        if _mp_detector is None:
            _mp_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_score)
        res = _mp_detector.process(arr)
        out = []
        for d in res.detections or []:
            bb = d.location_data.relative_bounding_box
            out.append(Face(bb.xmin, bb.ymin, bb.width, bb.height, float(d.score[0])))
        return out
    except ImportError:
        pass
    try:
        import cv2  # type: ignore
        if _cv_cascade is None:
            _cv_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        return [Face(x / w, y / h, fw / w, fh / h, 0.8) for (x, y, fw, fh) in _cv_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))]
    except ImportError:
        return []
