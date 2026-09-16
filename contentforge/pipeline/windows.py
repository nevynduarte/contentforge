"""Short-form windows: transcript-anchored 30-60 s moments rendered in every shot layout.

A project's `windows.yaml` lists windows (id, clip, question, start/end anchor phrases,
optional reference image) and the Mermaid sources for those images. Anchors are matched
against the clip's word JSON, so timings are exact and survive re-transcription.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console

from ..config import Project
from .batch import find_studio, words_path
from .shots import Segment, ShotPlan, analysis_for, render, render_landscape
from .transcribe import load_words

console = Console()

VARIANTS = ("speaker", "stacked", "both", "landscape", "sidebyside", "speaker_image", "image")
DEFAULT_VARIANTS = ("speaker", "stacked", "both", "landscape", "sidebyside")


@dataclass
class Window:
    id: str
    clip: str
    question: str
    start: str            # first words of the window
    end: str              # last words of the window
    image: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Window:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WindowSet:
    windows: list[Window]
    diagrams: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, project: Project) -> WindowSet:
        p = _windows_file(project)
        if not p.exists():
            raise FileNotFoundError(f"no windows.yaml for project {project.name} (looked at {p})")
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls([Window.from_dict(w) for w in d.get("windows", [])], d.get("diagrams") or {})


def _windows_file(project: Project) -> Path:
    from ..config import PROJECTS_DIR
    return PROJECTS_DIR / project.name / "windows.yaml"


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def _norm(s: str) -> list[str]:
    return [t for t in (re.sub(r"[^a-z0-9']", "", w.lower()) for w in s.split()) if t]


def find_phrase(words: list[dict], phrase: str, after: float = 0.0) -> tuple[int, int]:
    """Index range of the first occurrence of `phrase` at or after `after` seconds (punctuation-insensitive)."""
    toks = [(_norm(w["word"]) or [""])[0] for w in words]
    target = _norm(phrase)
    n = len(target)
    for i in range(len(words) - n + 1):
        if words[i]["start"] < after:
            continue
        if toks[i:i + n] == target:
            return i, i + n - 1
    raise ValueError(f"phrase not found: {phrase!r}")


def resolve(words: list[dict], start_anchor: str, end_anchor: str, pad: float = 0.25, tail: float = 0.45) -> tuple[float, float]:
    s0, _ = find_phrase(words, start_anchor)
    _, e1 = find_phrase(words, end_anchor, after=words[s0]["start"])
    return max(0.0, words[s0]["start"] - pad), words[e1]["end"] + tail


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_window(project: Project, w: Window, variant: str, out_dir: Path, force: bool = False) -> Path | None:
    """Render one window in one layout. Returns the output path, or None if skipped."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    dst = out_dir / f"{w.id}_{variant}.mp4"
    if dst.exists() and not force:
        return dst
    studio, words = find_studio(project, w.clip), load_words(words_path(project, w.clip))
    if studio is None:
        raise FileNotFoundError(f"no studio clip for {w.clip}")
    s, e = resolve(words, w.start, w.end)
    image = None
    if variant in ("speaker_image", "image"):
        if not w.image:
            console.print(f"[yellow]{w.id}: no reference image, skipping {variant}[/yellow]")
            return None
        image = str((project.root / w.image).resolve())
    base = {"landscape": "both", "sidebyside": "stacked"}.get(variant, variant)
    plan = ShotPlan([Segment(s, e, base, "auto", image)], w.question,
                    upscale="none" if variant in ("both", "landscape", "image") else "fast")
    plan.save(out_dir / f"{w.id}_{variant}.plan.json")
    tracks, turns = analysis_for(studio)
    if variant in ("landscape", "sidebyside"):
        render_landscape(studio, plan, dst, words, project.brand, mode=variant, tracks=tracks)
    else:
        render(studio, plan, dst, words, project.brand, tracks, turns)
    return dst


def render_windows(project: Project, variants: Iterable[str] = DEFAULT_VARIANTS, ids: Iterable[str] | None = None,
                   out_dir: Path | None = None, force: bool = False) -> list[dict]:
    ws = WindowSet.load(project)
    out_dir = out_dir or project.root / "edit" / "shorts"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(ids) if ids else None
    manifest = []
    for w in ws.windows:
        if wanted and w.id not in wanted:
            continue
        entry = {"id": w.id, "clip": w.clip, "question": w.question, "outputs": {}}
        for v in variants:
            t0 = time.time()
            try:
                out = render_window(project, w, v, out_dir, force)
            except ValueError as ex:
                console.print(f"[red]{w.id}/{v}: {ex}[/red]")
                continue
            if out:
                entry["outputs"][v] = str(out)
                console.print(f"{w.id}/{v}: {out.name} ({time.time() - t0:.0f}s)")
        manifest.append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def make_diagrams(project: Project, out_dir: Path | None = None, width: int = 1200) -> list[Path]:
    """Render the windows.yaml Mermaid diagrams to edit/refs/<id>.png in brand colours."""
    from ..ai.diagram import mermaid_to_png
    ws = WindowSet.load(project)
    out_dir = out_dir or project.root / "edit" / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return [mermaid_to_png(src, out_dir / f"{k}.png", project.brand, width=width) for k, src in ws.diagrams.items()]
