"""Audio cleanup, loudness measurement, sync offset, and music ducking."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils import ffmpeg

# Exact podcast voice chain used for Bridges AI x 3rd i (highpass -> loudnorm -16 LUFS).
PODCAST_VOICE_V2 = (
    "highpass=f=80,lowpass=f=14000,"
    "afftdn=nf=-25:nr=15:nt=w,"
    "agate=threshold=0.012:ratio=4:attack=5:release=50,"
    "acompressor=threshold=-22dB:ratio=3.5:attack=5:release=100:makeup=5dB,"
    "equalizer=f=180:t=q:w=1.5:g=-4,"
    "equalizer=f=2500:t=q:w=1.5:g=3,"
    "equalizer=f=3500:t=q:w=1.5:g=4,"
    "equalizer=f=5500:t=q:w=2:g=2.5,"
    "equalizer=f=8000:t=q:w=2:g=1.5,"
    "loudnorm=I=-16:LRA=11:TP=-1.5"
)

CHAINS = {"podcast_voice_v2": PODCAST_VOICE_V2, "none": "anull"}


def decode_pcm(path: str | Path, sr: int = 16000, start: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Decode mono float32 PCM via ffmpeg (used by sync + speech-activity code)."""
    cmd = [ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def measure_loudness(path: str | Path) -> dict:
    """Integrated loudness / true peak / LRA via ffmpeg loudnorm analysis pass."""
    cmd = [ffmpeg.which("ffmpeg"), "-hide_banner", "-i", str(path), "-vn",
           "-af", "loudnorm=I=-16:LRA=11:TP=-1.5:print_format=json", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\}", r.stderr, re.S)
    if not m:
        raise ffmpeg.FFmpegError("loudnorm did not return stats")
    return {k: float(v) if _isnum(v) else v for k, v in json.loads(m.group(0)).items()}


def _isnum(v: object) -> bool:
    try:
        float(v)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def find_offset(reference: str | Path, other: str | Path, window: float = 120.0, sr: int = 8000) -> float:
    """Seconds to add to `other` so it lines up with `reference` (cross-correlation of envelopes).

    Useful for aligning a separate mic recording (e.g. Regus.m4a) with camera audio.
    """
    a = decode_pcm(reference, sr, 0, window)
    b = decode_pcm(other, sr, 0, window)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    env = lambda x: np.abs(x) - np.mean(np.abs(x))  # noqa: E731
    fa, fb = np.fft.rfft(env(a), 2 * n), np.fft.rfft(env(b), 2 * n)
    corr = np.fft.irfft(fa * np.conj(fb))
    lag = int(np.argmax(corr))
    if lag > n:
        lag -= 2 * n
    return lag / sr


def clean(src: str | Path, dst: str | Path, chain: str = PODCAST_VOICE_V2, bitrate: str = "192k") -> Path:
    """Audio-only cleanup to AAC/M4A."""
    dst = Path(dst)
    info = ffmpeg.probe(src) if Path(src).suffix.lower() in {".mp4", ".mov", ".mkv"} else None
    ffmpeg.run(["-i", str(src), "-vn", "-af", chain, "-c:a", "aac", "-b:a", bitrate, str(dst)],
               duration=info.duration if info else None, description=f"audio {dst.name}")
    return dst


def duck_music_filter(voice_label: str = "[0:a]", music_label: str = "[1:a]", music_gain_db: float = -18.0,
                      threshold: float = 0.02, ratio: float = 8.0, out_label: str = "[aout]") -> str:
    """filter_complex snippet: sidechain-compress music under the voice, then mix."""
    return (f"{music_label}volume={music_gain_db}dB[m];"
            f"{voice_label}asplit=2[v1][v2];"
            f"[m][v2]sidechaincompress=threshold={threshold}:ratio={ratio}:attack=20:release=400[md];"
            f"[v1][md]amix=inputs=2:duration=first:dropout_transition=2{out_label}")


def replace_audio(video: str | Path, audio: str | Path, dst: str | Path, offset: float = 0.0, chain: Optional[str] = None) -> Path:
    """Mux a better audio recording onto a video (offset in seconds, positive delays audio)."""
    info = ffmpeg.probe(video)
    args = ["-i", str(video), "-itsoffset", f"{offset:.3f}", "-i", str(audio), "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    if chain:
        args += ["-af", chain]
    args += ["-movflags", "+faststart", str(dst)]
    ffmpeg.run(args, duration=info.duration, description=f"mux {Path(dst).name}")
    return Path(dst)
