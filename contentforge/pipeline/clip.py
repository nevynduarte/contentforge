"""Best-moment detection: propose clip boundaries from a transcript and score them.

Boundaries come from natural pauses (word gaps) and sentence ends; candidates
are windows of 20-90 s that start on a sentence boundary. Scoring is heuristic
by default, optionally refined by an LLM (see ai.content_scorer).
"""
from __future__ import annotations

from typing import Optional

from ..ai.content_scorer import ClipScore, heuristic_score, llm_score


def sentence_starts(words: list[dict], min_gap: float = 0.5) -> list[int]:
    idx = [0]
    for i in range(1, len(words)):
        prev = words[i - 1]
        gap = words[i]["start"] - prev["end"]
        if gap >= min_gap or prev["word"].strip().endswith((".", "?", "!")):
            idx.append(i)
    return idx


def propose(words: list[dict], min_len: float = 20.0, max_len: float = 90.0, step: int = 1,
            target_lens: tuple[float, ...] = (30.0, 45.0, 60.0, 85.0)) -> list[ClipScore]:
    starts = sentence_starts(words)
    out: list[ClipScore] = []
    for si in starts[::step]:
        t0 = words[si]["start"]
        for tl in target_lens:
            # end on the last sentence boundary that keeps duration within [min_len, max_len] near tl
            best_end = None
            for ej in starts:
                if ej <= si:
                    continue
                t1 = words[ej - 1]["end"]
                if t1 - t0 < min_len:
                    continue
                if t1 - t0 > max_len:
                    break
                if best_end is None or abs((t1 - t0) - tl) < abs((best_end - t0) - tl):
                    best_end = t1
            if best_end:
                out.append(heuristic_score(words, t0, best_end))
    return dedupe(sorted(out, key=lambda s: -s.total))


def dedupe(scores: list[ClipScore], overlap: float = 0.5) -> list[ClipScore]:
    kept: list[ClipScore] = []
    for s in scores:
        if all(_overlap(s, k) < overlap for k in kept):
            kept.append(s)
    return kept


def _overlap(a: ClipScore, b: ClipScore) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def best_clips(words: list[dict], n: int = 5, platform: str = "Instagram Reel", use_llm: bool = False,
               llm_top_k: int = 15, **kw) -> list[ClipScore]:
    cands = propose(words, **kw)
    if use_llm:
        for c in cands[:llm_top_k]:
            r: Optional[dict] = llm_score(c.text, platform)
            if r:
                c.total = round(0.5 * c.total + 0.05 * float(r.get("virality", 0)), 3)
                c.reasons.append(f"llm: {r.get('why', '')}")
        cands.sort(key=lambda s: -s.total)
    return cands[:n]
