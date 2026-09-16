"""Batch orchestration for a project: grade -> transcribe -> render presets -> manifest."""
from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console

from ..config import ClipDef, Project
from ..utils import ffmpeg
from ..utils.formats import load_preset
from . import grade as grademod
from . import render as rendermod
from . import transcribe as trmod

console = Console()


def studio_path(project: Project, c: ClipDef) -> Path:
    src_stem = Path(project.sources.get(c.source, c.source)).stem
    return project.studio_dir / f"{c.id}_{src_stem}_{int(c.start)}s_studio.mp4"


def find_studio(project: Project, clip_id: str) -> Path | None:
    hits = sorted(glob.glob(str(project.studio_dir / f"{clip_id}_*_studio.mp4")))
    return Path(hits[0]) if hits else None


def words_path(project: Project, clip_id: str) -> Path:
    return project.captions_dir / f"{clip_id}_words.json"


def ensure_graded(project: Project, c: ClipDef, force: bool = False, gpu: bool = False) -> Path:
    out = find_studio(project, c.id) or studio_path(project, c)
    if out.exists() and not force and ffmpeg.is_healthy(out):
        return out
    raw = project.clips_dir / f"{c.id}_{Path(project.sources.get(c.source, c.source)).stem}_{int(c.start)}s.mp4"
    if raw.exists() and ffmpeg.is_healthy(raw):
        console.print(f"[dim]{c.id}: grading from raw cut {raw.name}[/dim]")
        return grademod.grade(raw, out, gpu=gpu)
    console.print(f"[dim]{c.id}: grading from source {c.source} @ {c.start}s[/dim]")
    return grademod.grade(project.source_path(c.source), out, start=c.start, duration=c.duration, gpu=gpu)


def ensure_words(project: Project, c: ClipDef, studio: Path, force: bool = False) -> Path:
    wp = words_path(project, c.id)
    if wp.exists() and not force:
        return wp
    trmod.transcribe(studio, wp, project.captions_dir / f"{c.id}.srt", model=project.whisper_model)
    return wp


def run(project: Project, clip_ids: Iterable[str] | None = None, presets: list[str] | None = None,
        force_grade: bool = False, force_words: bool = False, force_render: bool = False, gpu: bool = False,
        show_logo: bool = True) -> list[dict]:
    presets = presets or project.presets
    wanted = set(clip_ids) if clip_ids else None
    manifest: list[dict] = []
    for c in project.clips:
        if wanted and c.id not in wanted:
            continue
        console.rule(f"[bold]{c.id}[/bold] {c.title}")
        studio = ensure_graded(project, c, force_grade, gpu)
        words = ensure_words(project, c, studio, force_words)
        outputs: dict[str, Path] = {}
        for pname in presets:
            p = load_preset(pname)
            dst = project.social_dir / pname / f"{c.id}_{pname}.mp4"
            if dst.exists() and not force_render and ffmpeg.is_healthy(dst):
                console.print(f"[dim]{dst.name} exists, skipping[/dim]")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                lt = None
                if c.speakers and project.brand.speakers.get(c.speakers[0]):
                    s = project.brand.speakers[c.speakers[0]]
                    lt = {"name": s.get("name", c.speakers[0]), "title": s.get("title", "")}
                rendermod.render_preset(studio, dst, p, words, project.brand, gpu=gpu, lower_third=lt, show_logo=show_logo)
            outputs[pname] = dst
        manifest.append(rendermod.manifest_entry(c.id, {"studio": studio, "words": words, **outputs},
                                                 {"title": c.title, "source": c.source, "start": c.start}))
    rendermod.write_manifest(manifest, project.social_dir / "manifest.json")
    return manifest
