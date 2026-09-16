"""Studio color / lighting grade.

The STUDIO_V2 chain is the exact grade used for the Bridges AI x 3rd i Podcast
(aggressive shadow lift, face gamma, orange cast correction, mild sharpen).
Filters run on CPU; encoding can optionally go through NVENC.
"""
from __future__ import annotations

from pathlib import Path

from ..utils import ffmpeg
from .audio import PODCAST_VOICE_V2

STUDIO_V2 = (
    "curves="
    "master='0/0 0.04/0.06 0.12/0.18 0.25/0.34 0.4/0.48 0.55/0.6 0.7/0.74 0.85/0.87 1/1':"
    "red='0/0 0.2/0.18 0.5/0.47 0.8/0.77 1/0.95':"
    "blue='0/0.03 0.2/0.24 0.5/0.53 0.8/0.81 1/1',"
    "eq=brightness=0.12:contrast=1.15:saturation=1.1:gamma=1.35,"
    "colorbalance=rs=-0.08:gs=-0.02:bs=0.06:rm=-0.06:gm=0.0:bm=0.05:rh=-0.04:gh=0.0:bh=0.03,"
    "unsharp=5:5:0.7:5:5:0.0"
)

GRADES = {"studio_v2": STUDIO_V2, "none": "null"}


def grade(
    src: str | Path,
    dst: str | Path,
    start: float | None = None,
    duration: float | None = None,
    video_filters: str = STUDIO_V2,
    audio_filters: str = PODCAST_VOICE_V2,
    crf: int = 18,
    preset: str = "medium",
    gpu: bool = False,
    audio_bitrate: str = "192k",
) -> Path:
    """Apply grade + voice cleanup and encode (libx264 medium CRF18 by default, faststart)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = ffmpeg.probe(src)
    args: list[str] = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(src)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vf", video_filters, "-af", audio_filters]
    args += ffmpeg.video_codec_args("libx264", crf, preset, gpu)
    args += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart"]
    tmp = dst.with_suffix(".tmp.mp4")
    args.append(str(tmp))
    total = duration if duration is not None else max(0.0, info.duration - (start or 0))
    ffmpeg.run(args, duration=total, description=f"grade {dst.name}")
    tmp.replace(dst)
    return dst
