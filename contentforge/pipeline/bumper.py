"""Branded outro / intro bumpers rendered with Pillow (no After Effects, no templates).

Outro: ink background, lockup fades and settles in, serif episode line, bronze rule draws,
sans CTA and URL. Silent audio track included so it concatenates cleanly with clips.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import Brand
from ..utils import ffmpeg
from ..utils.colors import hex_to_rgb
from .captions import load_font


def _ease(t: float) -> float:
    return 0.0 if t <= 0 else 1.0 if t >= 1 else 1 - math.pow(1 - t, 3)


def outro(brand: Brand, dst: str | Path, width: int = 1080, height: int = 1920, duration: float = 5.0,
          fps: float = 30.0, episode: str = "", cta: str = "Follow for more", url: str = "", gpu: bool = True) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fonts = brand.raw.get("fonts", {})
    fdir = [brand.fonts_dir] if brand.fonts_dir else None
    portrait = height > width
    ink = hex_to_rgb(brand.colors.get("ink", brand.colors["primary"]))
    paper = hex_to_rgb(brand.colors["text"])
    bronze = hex_to_rgb(brand.colors.get("accent_light", brand.colors["accent"]))
    muted = hex_to_rgb(brand.colors.get("muted", "#8A94A6"))
    serif = load_font(fonts.get("display", brand.captions.font), 64 if portrait else 56, fdir)
    sans = load_font(fonts.get("sans_semibold", brand.captions.font), 34 if portrait else 30, fdir)
    small = load_font(fonts.get("sans_medium", brand.captions.font), 28 if portrait else 26, fdir)
    logo = Image.open(brand.logo).convert("RGBA") if brand.logo and Path(brand.logo).exists() else None
    lw = int(width * (0.62 if portrait else 0.36))
    if logo is not None:
        logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    cy = height // 2 - (120 if portrait else 60)

    n = int(duration * fps)
    tmp = dst.with_suffix(".video.tmp.mp4")
    enc = subprocess.Popen([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "pipe:0",
                            *ffmpeg.video_codec_args("libx264", 18, "medium", gpu), "-pix_fmt", "yuv420p", str(tmp)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    try:
        for i in range(n):
            t = i / fps
            img = Image.new("RGB", (width, height), ink)
            d = ImageDraw.Draw(img)
            # 1. lockup: fade + settle from slightly larger
            a = _ease((t - 0.2) / 0.9)
            if logo is not None and a > 0:
                s = 1.08 - 0.08 * a
                lg = logo.resize((int(logo.width * s), int(logo.height * s)), Image.LANCZOS)
                alpha = lg.split()[3].point(lambda v, a=a: int(v * a))
                lg.putalpha(alpha)
                img.paste(lg, ((width - lg.width) // 2, cy - lg.height // 2 - 40), lg)
            # 2. bronze rule draws left-to-right
            r = _ease((t - 0.9) / 0.6)
            if r > 0:
                rw = int(120 * r)
                d.rectangle([width // 2 - rw // 2, cy + 80, width // 2 + rw // 2, cy + 81], fill=bronze)
            # 3. episode line (serif)
            e = _ease((t - 1.2) / 0.7)
            if episode and e > 0:
                col = tuple(int(ink[k] + (paper[k] - ink[k]) * e) for k in range(3))
                d.text(((width - probe.textlength(episode, font=serif)) // 2, cy + 110 + int(16 * (1 - e))), episode, font=serif, fill=col)
            # 4. CTA + URL (sans)
            c = _ease((t - 1.8) / 0.7)
            if c > 0:
                col = tuple(int(ink[k] + (bronze[k] - ink[k]) * c) for k in range(3))
                cta_u = "  ".join(cta.upper())
                d.text(((width - probe.textlength(cta_u, font=sans)) // 2, cy + 215), cta_u, font=sans, fill=col)
                if url:
                    col2 = tuple(int(ink[k] + (muted[k] - ink[k]) * c) for k in range(3))
                    d.text(((width - probe.textlength(url, font=small)) // 2, cy + 275), url, font=small, fill=col2)
            # 5. fade out over the last 0.5 s
            f = _ease((duration - t) / 0.5)
            if f < 1:
                arr = np.asarray(img).astype(np.float32)
                arr = arr * f + np.array(ink, np.float32) * (1 - f)
                img = Image.fromarray(arr.astype(np.uint8))
            enc.stdin.write(np.asarray(img, np.uint8).tobytes())
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise ffmpeg.FFmpegError(enc.stderr.read().decode(errors="replace")[-1000:])
    ffmpeg.run(["-i", str(tmp), "-f", "lavfi", "-t", str(duration), "-i", "anullsrc=r=48000:cl=stereo",
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
                "-movflags", "+faststart", str(dst)], show_progress=False)
    tmp.unlink(missing_ok=True)
    return dst


def append_outro(clip: str | Path, outro_path: str | Path, dst: str | Path, fade: float = 0.4, gpu: bool = True) -> Path:
    """clip + outro with a short crossfade; output matches the clip's resolution."""
    from .brand import concat_with_bumpers
    return concat_with_bumpers(Path(clip), Path(dst), None, Path(outro_path), fade=fade, gpu=gpu)


def make_outros(brand: Brand, out_dir: str | Path, episode: str = "", cta: str = "Follow for more", url: str = "",
                duration: float = 5.0) -> dict[str, Path]:
    """Portrait + landscape outro bumpers for a project."""
    out_dir = Path(out_dir)
    return {"portrait": outro(brand, out_dir / "outro_portrait.mp4", 1080, 1920, duration, episode=episode, cta=cta, url=url),
            "landscape": outro(brand, out_dir / "outro_landscape.mp4", 1920, 1080, duration, episode=episode, cta=cta, url=url)}


def outro_for(video: str | Path, bumpers_dir: str | Path) -> Path:
    """Pick the orientation-matched outro for a clip."""
    info = ffmpeg.probe(video)
    return Path(bumpers_dir) / ("outro_portrait.mp4" if info.height > info.width else "outro_landscape.mp4")


def append_outros(src_dir: str | Path, out_dir: str | Path, bumpers_dir: str | Path, force: bool = False,
                  pattern: str = "*.mp4") -> list[Path]:
    """Append the matching outro to every clip in src_dir, writing to out_dir (existing outputs are kept)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for f in sorted(src_dir.glob(pattern)):
        if ".tmp" in f.name:
            continue
        dst = out_dir / f.name
        if dst.exists() and not force and ffmpeg.is_healthy(dst):
            done.append(dst)
            continue
        append_outro(f, outro_for(f, bumpers_dir), dst)
        done.append(dst)
    return done
