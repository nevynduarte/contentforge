"""Branding: logo overlay, lower thirds, progress bar, intro/outro concat.

Overlays are Pillow-based and plug into the caption renderer's frame_hook so a
single decode/encode pass produces captions + branding together.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw

from ..config import Brand
from ..utils import ffmpeg
from ..utils.colors import hex_to_rgb, hex_to_rgba
from .captions import load_font


def _logo_image(brand: Brand, width: int) -> Optional[Image.Image]:
    if not brand.logo or not Path(brand.logo).exists():
        return None
    img = Image.open(brand.logo).convert("RGBA")
    scale = width / img.width
    return img.resize((width, max(1, int(img.height * scale))), Image.LANCZOS)


def lower_third_image(brand: Brand, name: str, title: str = "", width: int = 1080, fonts_dir: Optional[Path] = None) -> Image.Image:
    """Clean minimal lower third: accent bar + name (bold) + title (muted)."""
    fonts_dir = fonts_dir or brand.fonts_dir
    name_font = load_font(brand.captions.font, 44, [fonts_dir] if fonts_dir else None)
    title_font = load_font("arial.ttf", 30, [fonts_dir] if fonts_dir else None)
    pad, bar_w = 22, 10
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    nw = probe.textlength(name, font=name_font)
    tw = probe.textlength(title, font=title_font) if title else 0
    w = int(max(nw, tw) + pad * 2 + bar_w + 14)
    h = 44 + (38 if title else 0) + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=10, fill=hex_to_rgba(brand.colors["primary"], 215))
    d.rectangle([0, 0, bar_w, h], fill=hex_to_rgb(brand.colors["accent"]))
    x = bar_w + 14 + pad // 2
    d.text((x, pad - 6), name, font=name_font, fill=hex_to_rgb(brand.colors["text"]))
    if title:
        d.text((x, pad + 44), title, font=title_font, fill=hex_to_rgb(brand.colors["muted"]))
    return img


def frame_hook(
    brand: Brand,
    out_w: int,
    out_h: int,
    lower_third: Optional[dict] = None,   # {"name":..., "title":..., "start": t, "end": t, "y": px}
    show_logo: bool = True,
    logo_width: int = 140,
    logo_margin: int = 40,
    progress_bar: bool = False,
    duration: Optional[float] = None,
) -> Callable[[Image.Image, float], Image.Image]:
    """Return a per-frame compositor for the caption renderer."""
    # letterboxed social clips leave an empty band under the captions: centre a large lockup there
    logo = _logo_image(brand, 420 if out_h > out_w else logo_width) if show_logo else None
    logo_pos = ((out_w - logo.width) // 2, out_h - logo.height - 70) if (logo is not None and out_h > out_w) else \
               ((out_w - logo.width - logo_margin, logo_margin) if logo is not None else (0, 0))
    lt_img = None
    if lower_third:
        lt_img = lower_third_image(brand, lower_third.get("name", ""), lower_third.get("title", ""), out_w)
    lt_start, lt_end = (lower_third or {}).get("start", 0.5), (lower_third or {}).get("end", 6.0)
    lt_y = (lower_third or {}).get("y", out_h - 420)
    accent = hex_to_rgb(brand.colors["accent"])

    def hook(img: Image.Image, t: float) -> Image.Image:
        if logo is not None:
            img.paste(logo, logo_pos, logo)
        if lt_img is not None and lt_start <= t <= lt_end:
            fade = min(1.0, (t - lt_start) / 0.35, (lt_end - t) / 0.35)
            if fade < 1.0:
                a = lt_img.split()[3].point(lambda v: int(v * fade))
                tmp = lt_img.copy()
                tmp.putalpha(a)
                img.paste(tmp, (40, lt_y), tmp)
            else:
                img.paste(lt_img, (40, lt_y), lt_img)
        if progress_bar and duration:
            d = ImageDraw.Draw(img)
            d.rectangle([0, out_h - 8, int(out_w * min(1.0, t / duration)), out_h], fill=accent)
        return img

    return hook


def title_card(brand: Brand, title: str, subtitle: str = "", out_w: int = 1080, out_h: int = 1920, dst: Optional[Path] = None) -> Image.Image:
    """Static branded title card (used for intro stills and thumbnails)."""
    img = Image.new("RGB", (out_w, out_h), hex_to_rgb(brand.colors["primary"]))
    d = ImageDraw.Draw(img)
    f1 = load_font(brand.captions.font, 84, [brand.fonts_dir] if brand.fonts_dir else None)
    f2 = load_font("arial.ttf", 44, [brand.fonts_dir] if brand.fonts_dir else None)
    lines, line = [], ""
    for w in title.split():
        if d.textlength((line + " " + w).strip(), font=f1) > out_w - 160:
            lines.append(line.strip())
            line = w
        else:
            line += " " + w
    lines.append(line.strip())
    y = out_h // 2 - (len(lines) * 100) // 2
    for ln in lines:
        d.text(((out_w - d.textlength(ln, font=f1)) // 2, y), ln, font=f1, fill=hex_to_rgb(brand.colors["text"]))
        y += 100
    if subtitle:
        d.text(((out_w - d.textlength(subtitle, font=f2)) // 2, y + 20), subtitle, font=f2, fill=hex_to_rgb(brand.colors["accent"]))
    d.rectangle([out_w // 2 - 60, y + 100, out_w // 2 + 60, y + 108], fill=hex_to_rgb(brand.colors["accent"]))
    logo = _logo_image(brand, 260 if out_h >= 1200 else 120)
    if logo is not None and out_h // 2 - (len(lines) * 100) // 2 - 40 > logo.height + 40:
        img.paste(logo, ((out_w - logo.width) // 2, 40 if out_h < 1200 else 160), logo)
    if dst:
        img.save(dst)
    return img


def concat_with_bumpers(main: Path, dst: Path, intro: Optional[Path] = None, outro: Optional[Path] = None,
                        fade: float = 0.5, gpu: bool = False) -> Path:
    """Concatenate intro + main + outro with crossfades (re-encodes; all inputs must share resolution)."""
    parts = [p for p in (intro, main, outro) if p and Path(p).exists()]
    if len(parts) == 1:
        subprocess.run([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(main), "-c", "copy", str(dst)], check=True)
        return dst
    infos = [ffmpeg.probe(p) for p in parts]
    args: list[str] = []
    for p in parts:
        args += ["-i", str(p)]
    fc, prev_v, prev_a, offset = "", "[0:v]", "[0:a]", 0.0
    for i in range(1, len(parts)):
        offset += infos[i - 1].duration - fade
        fc += f"{prev_v}[{i}:v]xfade=transition=fade:duration={fade}:offset={offset:.3f}[v{i}];"
        fc += f"{prev_a}[{i}:a]acrossfade=d={fade}[a{i}];"
        prev_v, prev_a = f"[v{i}]", f"[a{i}]"
    args += ["-filter_complex", fc.rstrip(";"), "-map", prev_v, "-map", prev_a,
             *ffmpeg.video_codec_args("libx264", 20, "medium", gpu), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", str(dst)]
    ffmpeg.run(args, duration=sum(i.duration for i in infos), description=f"bumpers {dst.name}")
    return dst
