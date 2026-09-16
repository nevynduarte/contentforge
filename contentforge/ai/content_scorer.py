"""Score candidate clips for engagement potential.

Two layers:
  heuristic_score()  - fast, offline: hook strength, completeness, pace, emotion cues
  llm_score()        - Claude API (or Ollama) judgement with structured JSON output
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

HOOK_PATTERNS = [
    r"\b(nobody|no one) (tells|talks about)", r"\bthe (biggest|worst|best|hardest)\b", r"\bhere'?s (the thing|why|what)",
    r"\bI (sold|lost|made|built|quit|failed)\b", r"\b(secret|mistake|truth|lie|myth)\b", r"\?\s*$",
    r"\b(never|always|every|nothing|everything)\b", r"\bwhat (if|happens)\b", r"\b(crazy|insane|wild|unbelievable)\b",
    r"\b\d+ ?(%|percent|million|billion|x)\b",
]
EMOTION_WORDS = {"love", "hate", "scared", "terrified", "excited", "amazing", "shocked", "angry", "happy", "worst",
                 "best", "afraid", "fear", "proud", "laugh", "cry", "honestly", "literally"}
FILLERS = {"um", "uh", "like", "you know", "sort of", "kind of"}


@dataclass
class ClipScore:
    start: float
    end: float
    text: str
    hook: float = 0.0
    completeness: float = 0.0
    pace: float = 0.0
    emotion: float = 0.0
    clarity: float = 0.0
    total: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def heuristic_score(words: list[dict], start: float, end: float) -> ClipScore:
    seg = [w for w in words if start <= w["start"] < end]
    text = "".join(w["word"] for w in seg).strip()
    sc = ClipScore(start, end, text)
    if not seg:
        return sc
    first = " ".join(w["word"].strip() for w in seg[:12]).lower()
    sc.hook = min(1.0, 0.25 + 0.35 * sum(bool(re.search(p, first)) for p in HOOK_PATTERNS))
    sc.reasons.append(f"hook: '{first[:60]}'")
    ends_clean = bool(re.search(r"[.!?]\s*$", seg[-1]["word"])) and seg[0]["word"].strip()[:1].isupper()
    sc.completeness = 1.0 if ends_clean else 0.5
    wpm = len(seg) / max(1e-3, (end - start)) * 60
    sc.pace = max(0.0, 1.0 - abs(wpm - 165) / 120)
    lw = [w["word"].strip().lower().strip(".,!?") for w in seg]
    sc.emotion = min(1.0, sum(w in EMOTION_WORDS for w in lw) / 3)
    sc.clarity = max(0.0, 1.0 - sum(w in FILLERS for w in lw) / max(1, len(lw)) * 6)
    dur_pen = 1.0 if 20 <= sc.duration <= 90 else 0.7
    sc.total = round(dur_pen * (0.35 * sc.hook + 0.25 * sc.completeness + 0.15 * sc.pace + 0.15 * sc.emotion + 0.10 * sc.clarity), 3)
    return sc


LLM_PROMPT = """You are a short-form video editor. Rate this transcript excerpt for a {platform} clip.
Return JSON only: {{"virality": 0-10, "hook": 0-10, "complete_thought": true/false,
"suggested_start_phrase": "...", "suggested_end_phrase": "...", "why": "one sentence"}}

Transcript:
\"\"\"{text}\"\"\""""


def llm_score(text: str, platform: str = "Instagram Reel", model: str = "claude-sonnet-5", provider: str = "auto") -> dict | None:
    """Ask an LLM to rate the excerpt. Returns parsed JSON or None if no provider is configured."""
    prompt = LLM_PROMPT.format(platform=platform, text=text[:6000])
    raw = complete(prompt, model=model, provider=provider)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def complete(prompt: str, model: str = "claude-sonnet-5", provider: str = "auto", max_tokens: int = 800) -> str | None:
    """Minimal LLM completion: Anthropic API if ANTHROPIC_API_KEY is set, else local Ollama, else None."""
    if provider in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic()
            msg = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            return "".join(getattr(b, "text", "") for b in msg.content)
        except Exception as e:  # pragma: no cover
            if provider == "anthropic":
                raise
            print(f"[content_scorer] anthropic failed: {e}")
    if provider in ("auto", "ollama"):
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                         data=json.dumps({"model": os.environ.get("OLLAMA_MODEL", "llama3.1"), "prompt": prompt, "stream": False}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["response"]
        except Exception:
            return None
    return None
