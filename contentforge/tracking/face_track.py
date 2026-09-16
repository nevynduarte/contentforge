"""Multi-face tracking across sampled frames (greedy nearest-centre association)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..utils import ffmpeg
from .face_detect import Face, detect_faces


@dataclass
class Track:
    id: int
    samples: list[tuple[float, Face]] = field(default_factory=list)

    @property
    def last(self) -> Face:
        return self.samples[-1][1]

    def mean_cx(self) -> float:
        return sum(f.cx for _, f in self.samples) / len(self.samples)


def track_faces(src: str | Path, every: float = 0.5, max_jump: float = 0.15, sample_width: int = 640) -> list[Track]:
    """Detect faces every `every` seconds and link them into tracks by centre proximity."""
    info = ffmpeg.probe(src)
    tracks: list[Track] = []
    next_id = 0
    t = 0.0
    with tempfile.TemporaryDirectory() as td:
        while t < info.duration:
            f = ffmpeg.extract_frame(src, t, Path(td) / "s.jpg", width=sample_width)
            faces = detect_faces(Image.open(f))
            unmatched = list(faces)
            for tr in tracks:
                if not unmatched or t - tr.samples[-1][0] > every * 4:
                    continue
                best = min(unmatched, key=lambda fc: abs(fc.cx - tr.last.cx) + abs(fc.cy - tr.last.cy))
                if abs(best.cx - tr.last.cx) + abs(best.cy - tr.last.cy) <= max_jump:
                    tr.samples.append((t, best))
                    unmatched.remove(best)
            for fc in unmatched:
                tracks.append(Track(next_id, [(t, fc)]))
                next_id += 1
            t += every
    return [tr for tr in tracks if len(tr.samples) >= 3]
