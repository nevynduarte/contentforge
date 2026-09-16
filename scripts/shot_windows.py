"""Render short (30-60 s) shot variants for hand-picked moments.

Each window is defined by transcript anchors (first words / last words) so timings
come from the word JSON, not guesswork. Usage:
    python scripts/shot_windows.py [variants...] [--only ID]
variants default: speaker stacked both
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project  # noqa: E402
from contentforge.pipeline.batch import find_studio, words_path  # noqa: E402
from contentforge.pipeline.shots import Segment, ShotPlan, analysis_for, render, render_landscape  # noqa: E402
from contentforge.pipeline.transcribe import load_words  # noqa: E402

# id, clip, question, start anchor, end anchor
WINDOWS = [
    ("zillow", "clip_02", "How did you sell your parents' house on Zillow in eighth grade?",
     "sold my house on zillow", "posed as my mom"),
    ("bears", "clip_03", "Where did the idea for the safety app come from?",
     "so i facetimed my friend chris", "does have that integration"),
    ("wrapper", "clip_04", "Is your product just a ChatGPT wrapper?",
     "people in the comments", "unless you execute"),
    ("patents", "clip_04", "Do you need a patent to start a company?",
     "a lot of especially angel investors", "legal fees to enforce this"),
    ("snapchat", "clip_05", "How did you fund the first version?",
     "reached out to my two friends", "friend slash investor"),
    ("misread", "clip_06", "What happens when AI gets a safety call wrong?",
     "the way they prompted their ai", "before putting out that type of thing"),
    ("extra_zero", "clip_01", "Why keep a human in the loop?",
     "what if ai misread one of the quotes", "human in the loop"),
    ("car_leases", "clip_07", "What were you hustling on before AI?",
     "went into like consulting", "stuff like that growing up"),
    ("outage", "clip_09", "What did the big cloud outage teach you?",
     "was it microsoft or aws", "went up instead of down"),
    ("backend", "clip_10", "Where should AI never touch your product?",
     "there's a reason you said", "leave the human in the loop"),
    ("smb", "clip_10", "How do you bring AI into a 40-year-old business?",
     "what we really want to do is come in", "done by humans still"),
]


# reference images (relative to the project root) for the image-based layouts
IMAGES = {
    "bears": "edit/refs/bears.png", "wrapper": "edit/refs/wrapper.png", "outage": "edit/refs/outage.png",
    "backend": "edit/refs/backend.png", "smb": "edit/refs/smb.png",
}


def _norm(s: str) -> list[str]:
    return [re.sub(r"[^a-z0-9']", "", w.lower()) for w in s.split()]


def find_phrase(words: list[dict], phrase: str, after: float = 0.0) -> tuple[int, int]:
    """Index range of the first occurrence of `phrase` (token match, punctuation-insensitive)."""
    toks = [_norm(w["word"])[0] if _norm(w["word"]) else "" for w in words]
    target = _norm(phrase)
    n = len(target)
    for i in range(len(words) - n + 1):
        if words[i]["start"] < after:
            continue
        if toks[i:i + n] == target:
            return i, i + n - 1
    raise ValueError(f"phrase not found: {phrase!r}")


def resolve(words: list[dict], start_anchor: str, end_anchor: str, pad: float = 0.25) -> tuple[float, float]:
    s0, _ = find_phrase(words, start_anchor)
    _, e1 = find_phrase(words, end_anchor, after=words[s0]["start"])
    return max(0.0, words[s0]["start"] - pad), words[e1]["end"] + 0.45


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    variants = args or ["speaker", "stacked", "both"]
    project = Project.load("bridges_ai_3rdi")
    out_dir = project.root / "edit" / "shorts"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for wid, cid, question, a, b in WINDOWS:
        if only and wid != only:
            continue
        studio, words = find_studio(project, cid), load_words(words_path(project, cid))
        try:
            s, e = resolve(words, a, b)
        except ValueError as ex:
            print(f"{wid}: {ex}", flush=True)
            continue
        print(f"{wid}: {cid} {s:.1f}-{e:.1f} ({e - s:.0f}s)", flush=True)
        tracks, turns = analysis_for(studio)
        for v in variants:
            dst = out_dir / f"{wid}_{v}.mp4"
            if dst.exists():
                print(f"  {v}: exists", flush=True)
                continue
            base = "both" if v in ("both", "landscape") else ("stacked" if v == "sidebyside" else v)
            image = None
            if v in ("speaker_image", "image"):
                if wid not in IMAGES:
                    print(f"  {v}: no reference image for {wid}, skipping", flush=True)
                    continue
                image = str(project.root / IMAGES[wid])
            plan = ShotPlan([Segment(s, e, base, "auto", image)], question, upscale="none" if v in ("both", "landscape", "image") else "fast")
            plan.save(out_dir / f"{wid}_{v}.plan.json")
            t0 = time.time()
            if v in ("landscape", "sidebyside"):
                render_landscape(studio, plan, dst, words, project.brand, mode=v, tracks=tracks)
            else:
                render(studio, plan, dst, words, project.brand, tracks, turns)
            print(f"  {v}: {dst.name} in {time.time() - t0:.0f}s", flush=True)
        manifest.append({"id": wid, "clip": cid, "question": question, "start": s, "end": e, "variants": variants})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("SHORTS_DONE", flush=True)


if __name__ == "__main__":
    main()
