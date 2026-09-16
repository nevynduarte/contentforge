"""Kept-range detection: remove dead air and (optionally) chosen fragments before shot rendering.

Three sources, all returning [(start, end)] in source seconds:
  auto_editor()   - runs auto-editor (audio loudness) and reads its v3 timeline export
  from_words()    - gaps in the word-level transcript longer than `gap`
  drop_phrases()  - remove specific phrases (fillers, false starts) found in the transcript
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

Range = tuple[float, float]


def auto_editor(media: str | Path, threshold: str = "4%", margin: str = "0.2s", start: float = 0.0,
                end: Optional[float] = None) -> list[Range]:
    exe = shutil.which("auto-editor") or str(Path(__import__("sys").executable).parent / "auto-editor.exe")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tl.v3"
        subprocess.run([exe, str(media), "--edit", f"audio:threshold={threshold}", "--margin", margin, "--export", "v3",
                        "-o", str(out), "--no-open"], check=True, capture_output=True)
        tl = json.loads(out.read_text(encoding="utf-8"))
    num, den = (int(x) for x in tl["timebase"].split("/"))
    tb = num / den
    ranges = []
    for clip in tl["v"][0]:
        a, b = clip["offset"] / tb, (clip["offset"] + clip["dur"]) / tb
        if end is not None:
            b = min(b, end)
        a = max(a, start)
        if b - a > 0.05:
            ranges.append((round(a, 3), round(b, 3)))
    return merge(ranges)


def from_words(words: list[dict], gap: float = 0.7, start: float = 0.0, end: Optional[float] = None, pad: float = 0.15) -> list[Range]:
    ws = [w for w in words if w["end"] > start and (end is None or w["start"] < end)]
    if not ws:
        return [(start, end or 0.0)]
    ranges, a, last = [], max(start, ws[0]["start"] - pad), ws[0]["end"]
    for w in ws[1:]:
        if w["start"] - last > gap:
            ranges.append((a, last + pad))
            a = w["start"] - pad
        last = w["end"]
    ranges.append((a, min(last + pad, end) if end else last + pad))
    return merge(ranges)


def drop_phrases(words: list[dict], ranges: list[Range], phrases: Iterable[str], pad: float = 0.05) -> list[Range]:
    """Cut every occurrence of each phrase out of the kept ranges."""
    norm = lambda s: [re.sub(r"[^a-z0-9']", "", t.lower()) for t in s.split()]  # noqa: E731
    toks = [(norm(w["word"]) or [""])[0] for w in words]
    holes: list[Range] = []
    for ph in phrases:
        tgt = norm(ph)
        n = len(tgt)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == tgt:
                holes.append((words[i]["start"] - pad, words[i + n - 1]["end"] + pad))
    return subtract(ranges, holes)


def merge(ranges: list[Range], min_gap: float = 0.15) -> list[Range]:
    out: list[Range] = []
    for a, b in sorted(ranges):
        if out and a - out[-1][1] <= min_gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(ranges: list[Range], holes: list[Range]) -> list[Range]:
    out = list(ranges)
    for ha, hb in holes:
        nxt = []
        for a, b in out:
            if hb <= a or ha >= b:
                nxt.append((a, b))
            else:
                if ha > a:
                    nxt.append((a, ha))
                if hb < b:
                    nxt.append((hb, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 0.2]


def total(ranges: list[Range]) -> float:
    return sum(b - a for a, b in ranges)


def fmt(ranges: list[Range]) -> str:
    return ",".join(f"{a:.2f}-{b:.2f}" for a, b in ranges)
