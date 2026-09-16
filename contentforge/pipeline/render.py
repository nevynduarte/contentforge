"""Final rendering: apply a platform preset (layout + captions + brand) and encode.

Also writes sidecar SRTs and a per-clip manifest entry.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..config import Brand
from ..utils import ffmpeg
from ..utils.formats import Preset, load_preset
from . import brand as brandmod
from . import reframe
from .captions import render_captioned_video, style_for_preset
from .transcribe import load_words, write_srt


def render_preset(
    src: str | Path,
    dst: str | Path,
    preset: str | Preset,
    words_json: Optional[str | Path] = None,
    brand: Optional[Brand] = None,
    crop_mode: str = "center",
    gpu: bool = False,
    lower_third: Optional[dict] = None,
    show_logo: bool = True,
    max_duration: Optional[float] = None,
) -> Path:
    """Render `src` (a graded landscape clip) into a platform-specific output."""
    p = load_preset(preset) if isinstance(preset, str) else preset
    src, dst = Path(src), Path(dst)
    info = ffmpeg.probe(src)
    bg = brand.colors.get("background", "#111111") if brand else "#111111"
    vf = reframe.filter_for_layout(info, p.layout, p.width, p.height, p.video_y, bg, crop_mode)
    # max_duration=None -> preset cap; max_duration=0 -> no cap (full-length clip)
    limit = p.max_duration if max_duration is None else (max_duration or None)
    if limit and info.duration > limit:
        vf = f"trim=duration={limit},setpts=PTS-STARTPTS,{vf}"

    words = load_words(words_json) if words_json and Path(words_json).exists() else []
    style = style_for_preset(brand.captions if brand else Brand().captions, p.caption_y)
    hook = brandmod.frame_hook(brand, p.width, p.height, lower_third=lower_third, show_logo=show_logo) if brand else None

    if p.caption_style == "none" or not words:
        args = ["-i", str(src)]
        if limit and info.duration > limit:
            args += ["-t", str(limit)]
        args += ["-vf", vf, *ffmpeg.video_codec_args("libx264", p.crf, "medium", gpu), "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", p.audio_bitrate, "-movflags", "+faststart", str(dst)]
        ffmpeg.run(args, duration=min(info.duration, limit or info.duration), description=f"render {dst.name}")
    else:
        render_captioned_video(src, dst, words, style, p.width, p.height, pre_filter=vf,
                               mode=p.caption_style, fps=min(info.fps, p.fps), crf=p.crf, gpu=gpu,
                               audio_bitrate=p.audio_bitrate, fonts_dir=brand.fonts_dir if brand else None,
                               frame_hook=hook)
    if p.srt_sidecar and words:
        from .captions import chunk_words
        segs = [{"start": c[0]["start"], "end": c[-1]["end"], "text": " ".join(w["word"].strip() for w in c)}
                for c in chunk_words(words, 12, 80)]
        write_srt(segs, dst.with_suffix(".srt"))
    return dst


def write_manifest(entries: list[dict], path: str | Path) -> Path:
    Path(path).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return Path(path)


def manifest_entry(clip_id: str, outputs: dict[str, Path], meta: Optional[dict] = None, preset: Optional[Preset] = None) -> dict:
    return {"clip": clip_id, "outputs": {k: str(v) for k, v in outputs.items()},
            "preset": asdict(preset) if preset else None, "meta": meta or {}}
