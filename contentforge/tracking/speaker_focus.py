"""Active speaker detection: which seat is talking at time t.

Signal 1 (always available): lip-motion energy per seat from face_track mouth openness,
gated by audio RMS so silence never flips the crop.
Signal 2 (optional): pyannote diarization turns, mapped to seats by agreement with
signal 1. Used when HF_TOKEN is set and pyannote.audio is installed.

Output is a list of turns [{"start","end","seat"}] plus per-word seat labels.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..pipeline.audio import decode_pcm
from .face_track import SEATS, Tracks


def _audio_activity(media: str, times: list[float], every: float, sr: int = 16000) -> np.ndarray:
    pcm = decode_pcm(media, sr)
    rms = np.array([np.sqrt(np.mean(pcm[int(t * sr):int((t + every) * sr)] ** 2)) if int(t * sr) < len(pcm) else 0.0 for t in times])
    thr = max(1e-4, np.percentile(rms[rms > 0], 25) if np.any(rms > 0) else 0.0)
    return rms > thr


def lip_scores(tr: Tracks, win: int = 4) -> dict[str, np.ndarray]:
    """Rolling mean of lip-motion energy per seat (higher = talking), normalized per seat. NaN-safe."""
    out = {}
    for s in SEATS:
        m = np.array(tr.mouth[s], dtype=float)
        valid = ~np.isnan(m)
        m = np.where(valid, m, 0.0)
        k = np.ones(win) / win
        sm = np.convolve(m, k, mode="same")
        base = np.percentile(sm[valid], 20) if valid.any() else 0.0
        out[s] = np.clip(sm - base, 0, None)
    return out


def seat_timeline(tr: Tracks, media: str, hold: float = 0.8, margin: float = 1.15) -> list[dict]:
    """Turns from lip motion. A switch needs the other seat to lead by `margin` for `hold` seconds."""
    scores = lip_scores(tr)
    active = _audio_activity(media, tr.times, tr.every)
    cur: Optional[str] = None
    cand, cand_since = None, 0.0
    turns: list[dict] = []
    for i, t in enumerate(tr.times):
        l, r = scores["L"][i], scores["R"][i]
        if not active[i] or (l == 0 and r == 0):
            want = cur
        else:
            want = "L" if l >= r * margin else ("R" if r >= l * margin else cur)
        if cur is None and want:
            cur, cand = want, None
            turns.append({"start": t, "end": t, "seat": cur})
        elif want and want != cur:
            if cand != want:
                cand, cand_since = want, t
            elif t - cand_since >= hold:
                turns[-1]["end"] = cand_since
                cur, cand = want, None
                turns.append({"start": cand_since, "end": t, "seat": cur})
        else:
            cand = None
        if turns:
            turns[-1]["end"] = t + tr.every
    if not turns:
        turns = [{"start": 0.0, "end": tr.duration, "seat": "L"}]
    turns[-1]["end"] = tr.duration
    return turns


def diarize_turns(media: str, num_speakers: int = 2) -> Optional[list[dict]]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    try:
        from ..pipeline.diarize import diarize
        return diarize(media, num_speakers, token)
    except Exception:
        return None


def fuse(lip_turns: list[dict], dia: Optional[list[dict]], tr: Tracks) -> list[dict]:
    """Map diarization labels to seats by overlap with lip turns; prefer diarization boundaries when available."""
    if not dia:
        return lip_turns
    labels = sorted({d["speaker"] for d in dia})
    overlap = {lbl: {s: 0.0 for s in SEATS} for lbl in labels}
    for d in dia:
        for lt in lip_turns:
            o = max(0.0, min(d["end"], lt["end"]) - max(d["start"], lt["start"]))
            overlap[d["speaker"]][lt["seat"]] += o
    mapping = {lbl: max(SEATS, key=lambda s: overlap[lbl][s]) for lbl in labels}
    if len(labels) >= 2 and len(set(mapping.values())) == 1:  # both mapped to same seat: force the weaker onto the other
        weakest = min(labels, key=lambda l: max(overlap[l].values()))
        mapping[weakest] = "R" if mapping[weakest] == "L" else "L"
    out = [{"start": d["start"], "end": d["end"], "seat": mapping[d["speaker"]]} for d in dia]
    merged: list[dict] = []
    for tturn in sorted(out, key=lambda x: x["start"]):
        if merged and merged[-1]["seat"] == tturn["seat"] and tturn["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], tturn["end"])
        else:
            merged.append(dict(tturn))
    return merged


def seat_at(turns: list[dict], t: float) -> str:
    for tr in turns:
        if tr["start"] <= t < tr["end"]:
            return tr["seat"]
    return min(turns, key=lambda x: abs(x["start"] - t))["seat"] if turns else "L"


def label_words(words: list[dict], turns: list[dict]) -> list[dict]:
    return [{**w, "seat": seat_at(turns, (w["start"] + w["end"]) / 2)} for w in words]


def detect(media: str, tracks: Tracks, use_diarization: bool = True) -> list[dict]:
    lips = seat_timeline(tracks, media)
    return fuse(lips, diarize_turns(media) if use_diarization else None, tracks)
