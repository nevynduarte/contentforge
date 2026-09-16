"""Track the two seated speakers through a clip.

analyze() samples frames at `every` seconds, detects faces (InsightFace SCRFD when
available, otherwise MediaPipe), and assigns each detection to a seat ("L"/"R") by
x position, which is far more robust for a static two-person podcast than generic
multi-object tracking. Lip activity is measured as pixel motion inside the mouth
patch between consecutive samples, which works on three-quarter and profile views
where landmark-based mouth openness fails.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ..utils import ffmpeg

SEATS = ("L", "R")
MOUTH_PATCH = (48, 24)


@dataclass
class Det:
    x: float; y: float; w: float; h: float          # normalized box
    mouth: Optional[tuple[float, float, float, float]] = None  # normalized mouth patch box
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


@dataclass
class Tracks:
    duration: float
    width: int
    height: int
    every: float
    times: list[float] = field(default_factory=list)
    boxes: dict[str, list[Optional[tuple[float, float, float, float]]]] = field(default_factory=lambda: {"L": [], "R": []})
    mouth: dict[str, list[float]] = field(default_factory=lambda: {"L": [], "R": []})   # lip motion energy per sample

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps({"duration": self.duration, "width": self.width, "height": self.height, "every": self.every,
                                          "times": self.times, "boxes": self.boxes, "mouth": self.mouth}), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "Tracks":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls(d["duration"], d["width"], d["height"], d["every"], d["times"])
        t.boxes = {k: [tuple(b) if b else None for b in v] for k, v in d["boxes"].items()}
        t.mouth = d["mouth"]
        return t

    def seat_box(self, seat: str, t: float) -> Optional[tuple[float, float, float, float]]:
        """Nearest valid box (pixels: x, y, w, h) for a seat at time t."""
        if not self.times:
            return None
        i = int(np.clip(round(t / self.every), 0, len(self.times) - 1))
        for j in list(range(i, -1, -1)) + list(range(i + 1, len(self.times))):
            b = self.boxes[seat][j]
            if b:
                return (b[0] * self.width, b[1] * self.height, b[2] * self.width, b[3] * self.height)
        return None

    def median_box(self, seat: str) -> Optional[tuple[float, float, float, float]]:
        bs = [b for b in self.boxes[seat] if b]
        if not bs:
            return None
        arr = np.median(np.array(bs), axis=0)
        return (arr[0] * self.width, arr[1] * self.height, arr[2] * self.width, arr[3] * self.height)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
_app = None


def _insightface():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis  # type: ignore
        from ..utils.paths import MODELS_DIR
        _app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"],
                            root=str(MODELS_DIR.parent / "insightface"))
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def detect(rgb: np.ndarray) -> list[Det]:
    """Faces in an RGB frame, largest first. InsightFace -> MediaPipe fallback."""
    h, w = rgb.shape[:2]
    try:
        app = _insightface()
        out = []
        for f in app.get(np.ascontiguousarray(rgb[:, :, ::-1])):
            x0, y0, x1, y1 = f.bbox
            k = f.kps
            mx0, mx1 = min(k[3][0], k[4][0]), max(k[3][0], k[4][0])
            my = (k[3][1] + k[4][1]) / 2
            mw = max(24.0, (mx1 - mx0) * 1.6)
            mh = mw * 0.55
            mouth = ((mx0 + mx1) / 2 - mw / 2, my - mh / 2, mw, mh)
            out.append(Det(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h,
                           (mouth[0] / w, mouth[1] / h, mouth[2] / w, mouth[3] / h), float(f.det_score)))
        return sorted(out, key=lambda d: -d.w * d.h)
    except ImportError:
        from .face_detect import detect_faces
        faces = detect_faces(Image.fromarray(rgb), landmarks=False)
        return sorted([Det(f.x, f.y, f.w, f.h, None, f.score) for f in faces], key=lambda d: -d.w * d.h)


def _mouth_patch(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    x0, y0 = int(max(0, x * w)), int(max(0, y * h))
    x1, y1 = int(min(w, (x + bw) * w)), int(min(h, (y + bh) * h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros(MOUTH_PATCH[::-1], np.float32)
    g = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, MOUTH_PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-3)


def _iter_frames(src: Path, every: float, width: int):
    info = ffmpeg.probe(src)
    h = int(round(info.height * width / info.width / 2)) * 2
    cmd = [ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", f"fps=1/{every},scale={width}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width * h * 3 * 4)
    n = 0
    try:
        while True:
            raw = proc.stdout.read(width * h * 3)
            if len(raw) < width * h * 3:
                break
            yield n * every, np.frombuffer(raw, np.uint8).reshape(h, width, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()


def analyze(src: str | Path, every: float = 0.2, sample_width: int = 1280) -> Tracks:
    """Detect both speakers over the whole clip and measure lip motion per seat."""
    src = Path(src)
    info = ffmpeg.probe(src)
    tr = Tracks(info.duration, info.width, info.height, every)
    prev_patch: dict[str, Optional[np.ndarray]] = {"L": None, "R": None}
    split = 0.5
    for t, frame in _iter_frames(src, every, sample_width):
        dets = detect(frame)[:2]
        slot: dict[str, Optional[Det]] = {"L": None, "R": None}
        if len(dets) == 2:
            a, b = sorted(dets, key=lambda d: d.cx)
            slot["L"], slot["R"] = a, b
            split = (a.cx + b.cx) / 2
        elif len(dets) == 1:
            slot["L" if dets[0].cx < split else "R"] = dets[0]
        tr.times.append(t)
        for s in SEATS:
            d = slot[s]
            tr.boxes[s].append((d.x, d.y, d.w, d.h) if d else None)
            if d and d.mouth:
                patch = _mouth_patch(frame, d.mouth)
                energy = float(np.mean(np.abs(patch - prev_patch[s]))) if prev_patch[s] is not None else 0.0
                prev_patch[s] = patch
                tr.mouth[s].append(energy)
            else:
                tr.mouth[s].append(float("nan"))
    return tr
