"""Thumbnail candidates: pick sharp, well-exposed frames (ideally with a face) and overlay a title."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import Brand
from ..pipeline.captions import load_font
from ..utils import ffmpeg
from ..utils.colors import hex_to_rgb


def frame_quality(img: Image.Image) -> float:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = np.abs(np.diff(g, 2, axis=0)).mean() + np.abs(np.diff(g, 2, axis=1)).mean()  # sharpness proxy
    exposure = 1.0 - abs(g.mean() - 118) / 118                                          # mid-grey preference
    return float(lap * max(0.2, exposure))


def best_frame_time(src: str | Path, samples: int = 12, skip_head: float = 1.0) -> float:
    info = ffmpeg.probe(src)
    best, best_t = -1.0, skip_head
    with tempfile.TemporaryDirectory() as td:
        for i in range(samples):
            t = skip_head + (info.duration - 2 * skip_head) * (i + 0.5) / samples
            f = ffmpeg.extract_frame(src, t, Path(td) / f"f{i}.png", width=480)
            q = frame_quality(Image.open(f))
            try:
                from ..tracking.face_detect import detect_faces
                if detect_faces(Image.open(f)):
                    q *= 1.5
            except Exception:
                pass
            if q > best:
                best, best_t = q, t
    return best_t


def make_thumbnail(src: str | Path, dst: str | Path, text: str = "", brand: Brand | None = None,
                   t: float | None = None, width: int = 1080, height: int = 1920) -> Path:
    brand = brand or Brand()
    t = best_frame_time(src) if t is None else t
    with tempfile.TemporaryDirectory() as td:
        frame = Image.open(ffmpeg.extract_frame(src, t, Path(td) / "f.png")).convert("RGB")
    # cover-fit the frame
    scale = max(width / frame.width, height / frame.height)
    frame = frame.resize((int(frame.width * scale), int(frame.height * scale)), Image.LANCZOS)
    x, y = (frame.width - width) // 2, (frame.height - height) // 2
    canvas = frame.crop((x, y, x + width, y + height))
    if text:
        # dark gradient band + wrapped bold text
        band = Image.new("RGBA", (width, height // 3), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        for i in range(band.height):
            bd.line([(0, i), (width, i)], fill=(0, 0, 0, int(200 * i / band.height)))
        canvas.paste(band, (0, height - band.height), band)
        font = load_font(brand.captions.font, 84, [brand.fonts_dir] if brand.fonts_dir else None)
        d = ImageDraw.Draw(canvas)
        lines, line = [], ""
        for w in text.split():
            if d.textlength((line + " " + w).strip(), font=font) > width - 120:
                lines.append(line.strip())
                line = w
            else:
                line += " " + w
        lines.append(line.strip())
        yy = height - 120 - 96 * len(lines)
        for ln in lines:
            d.text((60, yy), ln, font=font, fill=hex_to_rgb(brand.colors["highlight"]), stroke_width=4, stroke_fill=(0, 0, 0))
            yy += 96
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, quality=92)
    return Path(dst)
