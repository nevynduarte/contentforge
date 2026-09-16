"""Ingest: discover raw footage, probe metadata, flag broken files, cut raw clips."""
from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from rich.table import Table

from ..utils import ffmpeg

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mts", ".avi"}
AUDIO_EXT = {".m4a", ".wav", ".mp3", ".aac", ".flac"}


def scan(root: str | Path, recursive: bool = False) -> list[dict]:
    root = Path(root)
    it: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    rows = []
    for p in sorted(it):
        if p.suffix.lower() in VIDEO_EXT:
            try:
                rows.append({"kind": "video", "ok": True, **asdict(ffmpeg.probe(p))})
            except ffmpeg.FFmpegError as e:
                rows.append({"kind": "video", "ok": False, "path": p, "error": str(e)[:120]})
        elif p.suffix.lower() in AUDIO_EXT:
            rows.append({"kind": "audio", "ok": True, "path": p, "size": p.stat().st_size})
    return rows


def table(rows: list[dict]) -> Table:
    t = Table(title="Media")
    for col in ("file", "kind", "res", "fps", "duration", "codec", "status"):
        t.add_column(col)
    for r in rows:
        p = Path(r["path"])
        if r["kind"] == "video" and r["ok"]:
            t.add_row(p.name, "video", f"{r['width']}x{r['height']}", f"{r['fps']:.2f}", f"{r['duration']:.1f}s",
                      r["vcodec"], "[green]ok[/green]")
        elif r["kind"] == "video":
            t.add_row(p.name, "video", "-", "-", "-", "-", f"[red]BROKEN[/red] {r['error']}")
        else:
            t.add_row(p.name, "audio", "-", "-", "-", "-", "ok")
    return t


def cut_raw(src: str | Path, dst: str | Path, start: float, duration: float) -> Path:
    """Stream-copy a segment (fast, keyframe-aligned; use grade() for frame-accurate cuts)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-i", str(src), "-t", str(duration), "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    return Path(dst)
