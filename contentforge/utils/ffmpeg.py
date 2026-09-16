"""FFmpeg / ffprobe wrapper with progress reporting."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class FFmpegError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    vcodec: str
    pix_fmt: str
    acodec: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    bit_rate: Optional[int]

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


def which(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FFmpegError(f"{name} not found on PATH")
    return p


def probe(path: str | Path) -> MediaInfo:
    """Return stream/format metadata. Raises FFmpegError on unreadable files (e.g. missing moov atom)."""
    cmd = [which("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {r.stderr.strip()}")
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise FFmpegError(f"no video stream in {path}")
    num, den = (int(x) for x in v.get("r_frame_rate", "30/1").split("/"))
    fmt = info.get("format", {})
    return MediaInfo(
        path=Path(path),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=num / den if den else 30.0,
        duration=float(fmt.get("duration") or v.get("duration") or 0),
        vcodec=v.get("codec_name", ""),
        pix_fmt=v.get("pix_fmt", ""),
        acodec=a.get("codec_name") if a else None,
        sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        channels=int(a["channels"]) if a and a.get("channels") else None,
        bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
    )


def is_healthy(path: str | Path) -> bool:
    """True if ffprobe can read a duration (catches truncated / moov-less MP4s)."""
    try:
        return probe(path).duration > 0
    except FFmpegError:
        return False


@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    r = subprocess.run([which("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True)
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if line.startswith(" V") and len(parts) > 1:
            out.add(parts[1])
    return out


def has_nvenc() -> bool:
    return "h264_nvenc" in available_encoders()


def video_codec_args(codec: str = "libx264", crf: int = 18, preset: str = "medium", gpu: bool = False) -> list[str]:
    """Encoder args. gpu=True swaps in h264_nvenc with an equivalent constant-quality target."""
    if gpu and has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0", "-profile:v", "high"]
    return ["-c:v", codec, "-preset", preset, "-crf", str(crf)]


def run(
    args: Iterable[str],
    duration: Optional[float] = None,
    description: str = "ffmpeg",
    on_progress: Optional[Callable[[float], None]] = None,
    show_progress: bool = True,
) -> None:
    """Run ffmpeg with a Rich progress bar driven by -progress output."""
    cmd = [which("ffmpeg"), "-hide_banner", "-nostats", "-loglevel", "error",
           "-progress", "pipe:1", "-y", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(),
                        TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn(),
                        disable=not (show_progress and duration))
    with progress:
        task = progress.add_task(description, total=duration or 1.0)
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    t = int(line.split("=")[1]) / 1_000_000
                except ValueError:
                    continue
                if duration:
                    progress.update(task, completed=min(t, duration))
                if on_progress:
                    on_progress(t)
        proc.wait()
        if duration:
            progress.update(task, completed=duration)
    if proc.returncode != 0:
        err = proc.stderr.read() if proc.stderr else ""
        raise FFmpegError(f"ffmpeg exited {proc.returncode}: {err[-2000:]}")


def extract_frame(path: str | Path, t: float, out: str | Path, width: Optional[int] = None) -> Path:
    vf = f"scale={width}:-2" if width else "null"
    subprocess.run([which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
                    "-i", str(path), "-frames:v", "1", "-vf", vf, str(out)], check=True)
    return Path(out)
