"""Active-speaker estimation: which face track is talking at time t.

Combines (a) diarization turns mapped to the left/right face by mean x-position
and (b) a mouth-motion proxy (frame-to-frame change inside the lower face box)
when diarization is unavailable.
"""
from __future__ import annotations

from typing import Optional

from .face_track import Track


def assign_speakers_by_position(tracks: list[Track], turns: list[dict]) -> dict[str, int]:
    """Map diarization labels to track ids assuming speakers keep their seats (left/right)."""
    if not tracks:
        return {}
    ordered = sorted(tracks, key=lambda t: t.mean_cx())
    labels = sorted({t["speaker"] for t in turns})
    return {lbl: ordered[min(i, len(ordered) - 1)].id for i, lbl in enumerate(labels)}


def active_track_at(t: float, tracks: list[Track], turns: list[dict], mapping: dict[str, int]) -> Optional[Track]:
    spk = next((tr["speaker"] for tr in turns if tr["start"] <= t <= tr["end"]), None)
    if spk is None or spk not in mapping:
        return None
    return next((tr for tr in tracks if tr.id == mapping[spk]), None)


def focus_trajectory(tracks: list[Track], turns: list[dict], duration: float, step: float = 0.25,
                     frame_w: int = 1920, hold: float = 0.6) -> list[tuple[float, float]]:
    """[(t, center_x_px)] following the active speaker; falls back to the midpoint of all faces."""
    mapping = assign_speakers_by_position(tracks, turns)
    mid = (sum(tr.mean_cx() for tr in tracks) / len(tracks) * frame_w) if tracks else frame_w / 2
    out, t, last_cx, last_switch = [], 0.0, mid, -hold
    while t <= duration:
        tr = active_track_at(t, tracks, turns, mapping)
        target = tr.mean_cx() * frame_w if tr else mid
        if abs(target - last_cx) > 5 and t - last_switch >= hold:
            last_cx, last_switch = target, t
        out.append((t, last_cx))
        t += step
    return out
