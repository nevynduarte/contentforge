"""Topics, keywords, and quotable lines from a transcript.

Uses KeyBERT when installed (`pip install contentforge[nlp]`), otherwise a
stopword-filtered frequency model. Quotes are sentences that are short,
complete, and contain hook cues.
"""
from __future__ import annotations

import re
from collections import Counter

STOP = set("""a an the and or but if then so of to in on at for with from by as is are was were be been being it its this
that these those i you he she we they me him her us them my your his our their what which who whom when where why how
not no yes do does did done have has had having can could would should will shall may might must just like very really
about into over under again also there here up down out off than too more most some any all each every both few own
same other such only same um uh yeah okay right know think mean going get got go one two""".split())

HOOK_CUES = ("never", "nobody", "secret", "mistake", "biggest", "honestly", "truth", "?", "always", "wrong", "changed")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def keywords(text: str, n: int = 8) -> list[tuple[str, float]]:
    try:
        from keybert import KeyBERT  # type: ignore
        return KeyBERT().extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=n)
    except ImportError:
        pass
    toks = [t for t in re.findall(r"[a-z][a-z'-]+", text.lower()) if t not in STOP and len(t) > 2]
    uni = Counter(toks)
    bi = Counter(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    scored = [(k, v * 1.0) for k, v in uni.most_common(n * 2)] + [(k, v * 1.8) for k, v in bi.most_common(n) if v > 1]
    scored.sort(key=lambda kv: -kv[1])
    top = scored[:n]
    mx = top[0][1] if top else 1
    return [(k, round(v / mx, 3)) for k, v in top]


def quotes(text: str, n: int = 5, min_words: int = 6, max_words: int = 28) -> list[str]:
    out = []
    for s in sentences(text):
        w = len(s.split())
        if min_words <= w <= max_words:
            score = sum(c in s.lower() for c in HOOK_CUES) + (0.5 if w <= 16 else 0)
            out.append((score, s))
    out.sort(key=lambda x: -x[0])
    return [s for _, s in out[:n]]


def summarize(text: str) -> dict:
    return {"keywords": keywords(text), "quotes": quotes(text), "sentences": len(sentences(text)),
            "words": len(text.split())}
