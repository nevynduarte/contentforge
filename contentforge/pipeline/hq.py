"""High-quality tier: SeedVR2-restored speaker shorts.

plain speaker render (no overlays, CRF 14) -> SeedVR2 restore -> banner + captions + lockup -> outro.
Slow (minutes per clip on an RTX 3090) but gives natural skin and hair detail on the cropped speaker.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from rich.console import Console

from ..config import CaptionStyle, Project
from ..utils.paths import WORK_DIR
from .batch import find_studio, words_path
from .bumper import append_outro
from .captions import render_captioned_video
from .shots import Banner, H, Segment, ShotPlan, W, analysis_for, render
from .transcribe import load_words
from .upscale_seedvr import restore
from .windows import WindowSet, resolve

console = Console()


def hq_speaker(project: Project, ids: Iterable[str] | None = None, out_dir: Path | None = None,
               model: str = "seedvr2_ema_3b_fp16.safetensors", batch_size: int = 5, force: bool = False,
               outro: Path | None = None) -> list[Path]:
    ws = WindowSet.load(project)
    out_dir = out_dir or project.root / "edit" / "shorts_hq"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = WORK_DIR / "seedvr"
    work.mkdir(parents=True, exist_ok=True)
    outro = outro or project.root / "edit" / "bumpers" / "outro_portrait.mp4"
    wanted = set(ids) if ids else None
    outputs: list[Path] = []
    for w in ws.windows:
        if wanted and w.id not in wanted:
            continue
        dst = out_dir / f"{w.id}_speaker_hq.mp4"
        if dst.exists() and not force:
            outputs.append(dst)
            continue
        studio, words = find_studio(project, w.clip), load_words(words_path(project, w.clip))
        s, e = resolve(words, w.start, w.end)
        tracks, turns = analysis_for(studio)
        t0 = time.time()
        plain = work / f"{w.id}_plain.mp4"
        render(studio, ShotPlan([Segment(s, e, "speaker", "auto")], "", captions=False, logo=False, upscale="none"),
               plain, None, project.brand, tracks, turns, crf=14)
        t1 = time.time()
        restored = work / f"{w.id}_restored.mp4"
        restore(plain, restored, resolution=1080, batch_size=batch_size, model=model)
        t2 = time.time()
        out_words = [{**x, "start": x["start"] - s, "end": min(x["end"], e) - s} for x in words if s <= x["start"] < e]
        banner = Banner(w.question, project.brand, True)
        style = CaptionStyle(**{**asdict(project.brand.captions), "y": 1560})
        render_captioned_video(restored, dst, out_words, style, W, H, pre_filter="", fps=30, crf=18, gpu=True,
                               fonts_dir=project.brand.fonts_dir, frame_hook=lambda img, t, b=banner: (b.draw(img, "speaker"), img)[1],
                               audio_src=plain)
        t3 = time.time()
        console.print(f"{w.id}: plain {t1 - t0:.0f}s, seedvr2 {t2 - t1:.0f}s, overlay {t3 - t2:.0f}s -> {dst.name}")
        if outro and Path(outro).exists():
            final = project.root / "edit" / "final" / dst.name
            final.parent.mkdir(parents=True, exist_ok=True)
            append_outro(dst, outro, final)
        if os.environ.get("CONTENTFORGE_KEEP_WORK") != "1":
            plain.unlink(missing_ok=True)
            restored.unlink(missing_ok=True)
        outputs.append(dst)
    return outputs
