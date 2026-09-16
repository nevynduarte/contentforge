"""Generate platform copy: titles, captions, hashtags, alt text, A/B variants."""
from __future__ import annotations

import json
import re
from typing import Optional

from ..config import Brand
from .content_scorer import complete
from .topic_extract import keywords, quotes

PLATFORM_RULES = {
    "instagram": "Reel caption: 1-2 punchy lines + 3-5 hashtags. Hook first. No links.",
    "youtube": "Shorts title (<=60 chars, curiosity gap) + 2-line description ending with a subscribe CTA and #Shorts.",
    "tiktok": "Hook in the first words, casual tone, 3-5 hashtags, <=150 chars.",
    "linkedin": "Professional tone, 3-5 short lines, an insight and a question to invite comments, 3 hashtags.",
}

PROMPT = """You write social copy for "{show}" by {company}. Episode: {episode}.
Platform rules: {rules}
Brand hashtags to include where natural: {hashtags}
Transcript of the clip:
\"\"\"{text}\"\"\"
Return JSON only:
{{"title": "...", "caption": "...", "hashtags": ["#..."], "alt_text": "...", "hook_variants": ["...","...","..."]}}"""


def generate(text: str, brand: Brand, platform: str = "instagram", episode: str = "", use_llm: bool = True,
             model: str = "claude-sonnet-5") -> dict:
    rules = PLATFORM_RULES.get(platform, PLATFORM_RULES["instagram"])
    if use_llm:
        raw = complete(PROMPT.format(show=brand.show or "the show", company=brand.company or "us", episode=episode or "-",
                                     rules=rules, hashtags=" ".join(brand.hashtags), text=text[:6000]), model=model)
        if raw:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    d = json.loads(m.group(0))
                    d["hashtags"] = _merge_tags(d.get("hashtags", []), brand.hashtags, 5)
                    d["source"] = "llm"
                    return d
                except json.JSONDecodeError:
                    pass
    return fallback(text, brand, platform)


def fallback(text: str, brand: Brand, platform: str) -> dict:
    q = quotes(text, 3) or [text[:80]]
    kw = [k for k, _ in keywords(text, 3)]
    title = q[0].rstrip(".").strip()
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "…"
    tags = _merge_tags([f"#{re.sub(r'[^a-z0-9]', '', k.lower())}" for k in kw], brand.hashtags, 5)
    cta = " Subscribe for more. #Shorts" if platform == "youtube" else ""
    return {"title": title, "caption": f"{q[0]}{cta}\n\n{' '.join(tags)}", "hashtags": tags,
            "alt_text": f"Podcast clip from {brand.show or 'the show'}: {title}",
            "hook_variants": q[:3], "source": "heuristic"}


def _merge_tags(a: list[str], b: list[str], n: int) -> list[str]:
    out: list[str] = []
    for t in list(b) + list(a):
        t = t if t.startswith("#") else f"#{t}"
        if t.lower() not in {x.lower() for x in out} and len(t) > 1:
            out.append(t)
    return out[:n]
