"""Speaker diarization (who spoke when).

Primary backend: pyannote.audio 3.x (needs `pip install contentforge[diarize]`
and a Hugging Face token with access to pyannote/speaker-diarization-3.1).
Fallback: a simple energy/turn-taking heuristic that alternates speakers on
long pauses, good enough for 2-person podcasts when pyannote is unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def diarize(media: str | Path, num_speakers: Optional[int] = 2, hf_token: Optional[str] = None) -> list[dict]:
    """Return [{"start", "end", "speaker"}] turns."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        return heuristic_turns(media, num_speakers or 2)
    token = hf_token or os.environ.get("HF_TOKEN")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception:
        pass
    ann = pipe(str(media), num_speakers=num_speakers)
    return [{"start": float(s.start), "end": float(s.end), "speaker": lbl} for s, _, lbl in ann.itertracks(yield_label=True)]


def heuristic_turns(media: str | Path, num_speakers: int = 2, pause: float = 0.9) -> list[dict]:
    from .transcribe import transcribe
    words = transcribe(media, model="base")["words"]
    turns, cur, spk = [], None, 0
    for w in words:
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "speaker": f"SPEAKER_{spk:02d}"}
        elif w["start"] - cur["end"] > pause:
            turns.append(cur)
            spk = (spk + 1) % num_speakers
            cur = {"start": w["start"], "end": w["end"], "speaker": f"SPEAKER_{spk:02d}"}
        else:
            cur["end"] = w["end"]
    if cur:
        turns.append(cur)
    return turns


def label_words(words: list[dict], turns: list[dict]) -> list[dict]:
    """Attach a `speaker` key to each word by overlap with diarization turns."""
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        spk = next((t["speaker"] for t in turns if t["start"] <= mid <= t["end"]), None)
        out.append({**w, "speaker": spk})
    return out
