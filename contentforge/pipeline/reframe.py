"""Reframing landscape footage to vertical / square layouts.

Layouts (as ffmpeg -vf chains that produce exactly out_w x out_h):
  letterbox : full frame scaled to width, placed at video_y on a dark background
              (the "podcast social" look: video on top, captions below)
  crop      : static 9:16 crop at a chosen x (center / left / right speaker)
  track     : crop that follows a smoothed x-trajectory from face tracking
              (see contentforge.tracking.smart_crop), rendered via sendcmd
  fit       : scale to fit inside frame with blurred background fill
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..utils.colors import ffmpeg_color
from ..utils.ffmpeg import MediaInfo


def letterbox(info: MediaInfo, out_w: int = 1080, out_h: int = 1920, video_y: int | None = 80,
              bg: str = "#111111") -> str:
    vh = round(out_w * info.height / info.width)
    vh -= vh % 2
    y = video_y if video_y is not None else (out_h - vh) // 2
    return f"scale={out_w}:{vh}:flags=lanczos,pad={out_w}:{out_h}:0:{y}:color={ffmpeg_color(bg)}"


def crop_x_for(info: MediaInfo, out_w: int, out_h: int, mode: str = "center") -> tuple[int, int, int]:
    """Return (crop_w, crop_h, crop_x) for a static crop with the output aspect."""
    crop_h = info.height
    crop_w = int(crop_h * out_w / out_h)
    crop_w -= crop_w % 2
    if crop_w > info.width:
        crop_w = info.width
        crop_h = int(crop_w * out_h / out_w)
    if mode == "left":
        x = info.width // 4 - crop_w // 2
    elif mode == "right":
        x = 3 * info.width // 4 - crop_w // 2
    elif isinstance(mode, (int, float)):
        x = int(mode) - crop_w // 2
    else:
        x = (info.width - crop_w) // 2
    return crop_w, crop_h, max(0, min(x, info.width - crop_w))


def static_crop(info: MediaInfo, out_w: int = 1080, out_h: int = 1920, mode: str = "center") -> str:
    cw, ch, cx = crop_x_for(info, out_w, out_h, mode)
    return f"crop={cw}:{ch}:{cx}:0,scale={out_w}:{out_h}:flags=lanczos"


def fit_blur(info: MediaInfo, out_w: int = 1080, out_h: int = 1920, blur: int = 40) -> str:
    """Scaled-to-fit video over a blurred, zoomed copy of itself."""
    return (f"split[a][b];[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur={blur}:5[bg];"
            f"[b]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2")


def tracked_crop(info: MediaInfo, trajectory: Sequence[tuple[float, float]], out_w: int, out_h: int,
                 cmd_file: str | Path) -> str:
    """Write a sendcmd file that animates crop x along `trajectory` [(t, center_x), ...]."""
    cw, ch, _ = crop_x_for(info, out_w, out_h)
    lines = []
    for t, cx in trajectory:
        x = max(0, min(int(cx - cw / 2), info.width - cw))
        lines.append(f"{t:.3f} crop x {x};")
    Path(cmd_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd_path = str(Path(cmd_file)).replace("\\", "/").replace(":", "\\:")
    return f"sendcmd=f='{cmd_path}',crop={cw}:{ch}:0:0,scale={out_w}:{out_h}:flags=lanczos"


def filter_for_layout(info: MediaInfo, layout: str, out_w: int, out_h: int, video_y: int | None = None,
                      bg: str = "#111111", crop_mode: str = "center") -> str:
    if layout == "letterbox":
        return letterbox(info, out_w, out_h, video_y, bg)
    if layout == "crop":
        return static_crop(info, out_w, out_h, crop_mode)
    if layout == "fit":
        return fit_blur(info, out_w, out_h)
    if layout == "none":
        return f"scale={out_w}:{out_h}:flags=lanczos"
    raise ValueError(f"unknown layout {layout!r}")
