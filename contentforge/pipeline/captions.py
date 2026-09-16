"""Animated caption rendering with Pillow (no libass / drawtext needed).

Frames are decoded by ffmpeg to raw RGB, captions are composited in Python,
and frames are piped back to ffmpeg for encoding.  Caption overlays are
pre-rendered once per (chunk, active-word) state and cached, so per-frame work
is a single alpha-composite of a small strip rather than full text layout.

Styles:
  karaoke  - phrase on a pill, active word highlighted (default)
  popup    - one word at a time, large
  bar      - plain subtitle bar, phrase at a time, no highlight
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ..config import CaptionStyle
from ..utils import ffmpeg
from ..utils.colors import hex_to_rgb, hex_to_rgba

_FONT_DIRS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/msttcorefonts"),
]
_FONT_FALLBACKS = ["arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"]


def load_font(name: str, size: int, extra_dirs: list[Path] | None = None) -> ImageFont.FreeTypeFont:
    candidates = [Path(name)] if Path(name).is_absolute() else []
    for d in (extra_dirs or []) + _FONT_DIRS:
        candidates.append(d / name)
    for fb in _FONT_FALLBACKS:
        for d in (extra_dirs or []) + _FONT_DIRS:
            candidates.append(d / fb)
    for c in candidates:
        if c.exists():
            return ImageFont.truetype(str(c), size)
    return ImageFont.load_default(size)  # type: ignore[return-value]


def chunk_words(words: list[dict], max_words: int = 5, max_chars: int = 35, max_gap: float = 1.2) -> list[list[dict]]:
    """Group words into caption phrases; break on count, length, or long pauses."""
    chunks: list[list[dict]] = []
    chunk: list[dict] = []
    for w in words:
        if chunk and (w["start"] - chunk[-1]["end"]) > max_gap:
            chunks.append(chunk)
            chunk = []
        chunk.append(w)
        text = " ".join(x["word"].strip() for x in chunk)
        if len(chunk) >= max_words or len(text) > max_chars:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


@dataclass
class Overlay:
    image: Image.Image   # RGBA strip
    y: int               # where its top goes in the frame


class CaptionRenderer:
    """Pre-renders caption strips for a given style, frame width and word list."""

    def __init__(self, words: list[dict], style: CaptionStyle, frame_w: int, frame_h: int,
                 mode: str = "karaoke", fonts_dir: Optional[Path] = None):
        self.words, self.style, self.w, self.h, self.mode = words, style, frame_w, frame_h, mode
        self.font = load_font(style.font, style.size, [fonts_dir] if fonts_dir else None)
        if mode == "popup":
            self.chunks = [[w] for w in words]
        else:
            self.chunks = chunk_words(words, style.max_words, style.max_chars)
        self.index = [(c[0]["start"], c[-1]["end"] + style.hold, i) for i, c in enumerate(self.chunks)]
        self._cache: dict[tuple[int, int], Overlay] = {}
        self._probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        self._line_h = self._probe.textbbox((0, 0), "Hg", font=self.font)[3]

    # -- lookup -----------------------------------------------------------
    def state_at(self, t: float) -> Optional[tuple[int, int]]:
        """(chunk_idx, active_word_idx or -1) for time t, or None if no caption."""
        for cs, ce, i in self.index:
            if cs <= t <= ce:
                active = -1
                for j, w in enumerate(self.chunks[i]):
                    if w["start"] <= t <= w["end"]:
                        active = j
                        break
                    if w["start"] > t:
                        break
                    active = j  # hold last spoken word highlighted until the next starts
                return i, active
        return None

    # -- rendering --------------------------------------------------------
    def overlay(self, chunk_idx: int, active: int) -> Overlay:
        key = (chunk_idx, active if self.mode == "karaoke" else -1)
        ov = self._cache.get(key)
        if ov is None:
            ov = self._render(self.chunks[chunk_idx], key[1])
            self._cache[key] = ov
        return ov

    def _render(self, chunk: list[dict], active: int) -> Overlay:
        st = self.style
        tokens = [w["word"].strip() for w in chunk]
        widths = [self._probe.textlength(tok + " ", font=self.font) for tok in tokens]
        text_w = int(sum(widths) - self._probe.textlength(" ", font=self.font))
        pad, sw = st.pill_pad, st.stroke_width
        strip_w = min(self.w, text_w + 2 * pad + 2 * sw)
        strip_h = self._line_h + 2 * pad + 2 * sw
        img = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if st.pill_alpha > 0:
            d.rounded_rectangle([sw, sw, strip_w - sw - 1, strip_h - sw - 1], radius=st.pill_radius,
                                fill=hex_to_rgba(st.pill_color, st.pill_alpha))
        x = (strip_w - text_w) // 2
        y = pad + sw - self._probe.textbbox((0, 0), "Hg", font=self.font)[1]
        hi, base = hex_to_rgba(st.highlight), hex_to_rgba(st.color)
        for i, (tok, tw) in enumerate(zip(tokens, widths)):
            color = hi if (i == active and self.mode == "karaoke") else base
            d.text((x, y), tok, font=self.font, fill=color, stroke_width=sw, stroke_fill=hex_to_rgba(st.stroke, 220))
            x += int(tw)
        return Overlay(img, st.y - pad - sw)

    def composite(self, frame: Image.Image, t: float) -> Image.Image:
        state = self.state_at(t)
        if state is None:
            return frame
        ov = self.overlay(*state)
        x = (self.w - ov.image.width) // 2
        frame.paste(ov.image, (x, ov.y), ov.image)
        return frame


def render_captioned_video(
    src: str | Path,
    dst: str | Path,
    words: list[dict],
    style: CaptionStyle,
    out_w: int = 1080,
    out_h: int = 1920,
    pre_filter: str = "",
    mode: str = "karaoke",
    fps: Optional[float] = None,
    crf: int = 20,
    gpu: bool = False,
    audio_bitrate: str = "192k",
    fonts_dir: Optional[Path] = None,
    frame_hook: Optional[Callable[[Image.Image, float], Image.Image]] = None,
    audio_src: Optional[str | Path] = None,
) -> Path:
    """Decode -> composite captions (and optional per-frame hook) -> encode.

    pre_filter: ffmpeg -vf chain producing out_w x out_h frames (e.g. letterbox/crop).
    frame_hook: extra Pillow compositing (logo, lower third, progress bar...).
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = ffmpeg.probe(src)
    fps = fps or min(info.fps, 30.0)
    total_frames = int(info.duration * fps)
    renderer = CaptionRenderer(words, style, out_w, out_h, mode, fonts_dir)
    frame_bytes = out_w * out_h * 3
    vf = (pre_filter + "," if pre_filter else "") + f"scale={out_w}:{out_h}:flags=lanczos"

    dec = subprocess.Popen(
        [ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", vf,
         "-r", f"{fps:.6f}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    tmp = dst.with_suffix(".tmp.mp4")
    enc_args = [ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", f"{fps:.6f}", "-i", "pipe:0",
                "-i", str(audio_src or src), "-map", "0:v", "-map", "1:a?",
                *ffmpeg.video_codec_args("libx264", crf, "medium", gpu),
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart",
                "-shortest", str(tmp)]
    enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
    assert dec.stdout and enc.stdin

    progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(),
                        TextColumn("{task.completed}/{task.total} frames"), TimeRemainingColumn())
    n = 0
    try:
        with progress:
            task = progress.add_task(f"captions {dst.name}", total=total_frames)
            while True:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                t = n / fps
                state = renderer.state_at(t)
                if state is None and frame_hook is None:
                    enc.stdin.write(raw)  # fast path: no compositing needed
                else:
                    img = Image.frombuffer("RGB", (out_w, out_h), raw, "raw", "RGB", 0, 1).copy()
                    if state is not None:
                        img = renderer.composite(img, t)
                    if frame_hook is not None:
                        img = frame_hook(img, t)
                    enc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
                n += 1
                if n % 30 == 0:
                    progress.update(task, completed=n)
            progress.update(task, completed=total_frames)
    finally:
        dec.stdout.close()
        enc.stdin.close()
        dec.wait()
        enc.wait()
    if enc.returncode != 0 or not tmp.exists():
        err = enc.stderr.read().decode(errors="replace") if enc.stderr else ""
        raise ffmpeg.FFmpegError(f"caption encode failed ({enc.returncode}): {err[-1500:]}")
    tmp.replace(dst)
    return dst


def style_for_preset(base: CaptionStyle, preset_caption_y: Optional[int]) -> CaptionStyle:
    return replace(base, y=preset_caption_y) if preset_caption_y is not None else base
