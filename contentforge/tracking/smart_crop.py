"""Smart crop trajectory: face tracks -> smoothed centre-x path for vertical reframing.

Smoothing is a critically-damped exponential follower with a dead-zone, which
avoids the jitter of raw detections and the overshoot of naive Kalman tuning.
"""
from __future__ import annotations

from pathlib import Path

from ..utils import ffmpeg
from .face_track import Track, track_faces


def smooth(path: list[tuple[float, float]], alpha: float = 0.12, dead_zone: float = 40.0) -> list[tuple[float, float]]:
    if not path:
        return []
    out = [path[0]]
    cur = path[0][1]
    for t, x in path[1:]:
        if abs(x - cur) > dead_zone:
            cur += alpha * (x - cur)
        out.append((t, cur))
    return out


def raw_centres(tracks: list[Track], duration: float, frame_w: int, step: float = 0.25) -> list[tuple[float, float]]:
    """Centre on the union of visible faces at each t (keeps both speakers when possible)."""
    out, t = [], 0.0
    while t <= duration:
        xs = []
        for tr in tracks:
            near = [f for (ts, f) in tr.samples if abs(ts - t) <= step]
            if near:
                xs.append(near[0].cx * frame_w)
        out.append((t, (min(xs) + max(xs)) / 2 if xs else frame_w / 2))
        t += step
    return out


def trajectory(src: str | Path, every: float = 0.5) -> list[tuple[float, float]]:
    info = ffmpeg.probe(src)
    tracks = track_faces(src, every)
    return smooth(raw_centres(tracks, info.duration, info.width, every))
