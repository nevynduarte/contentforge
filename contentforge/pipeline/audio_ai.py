"""Neural speech cleanup (DeepFilterNet) and Python-side dynamics / ducking (pedalboard)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..utils import ffmpeg

SR = 48000


def _read(path: str | Path, sr: int = SR) -> np.ndarray:
    r = subprocess.run([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def _write(path: str | Path, audio: np.ndarray, sr: int = SR, bitrate: str = "192k") -> Path:
    subprocess.run([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                    "-i", "pipe:0", "-c:a", "aac", "-b:a", bitrate, str(path)], input=audio.astype(np.float32).tobytes(), check=True)
    return Path(path)


def denoise(src: str | Path, dst: str | Path) -> Path:
    """DeepFilterNet3 speech enhancement -> mono AAC at 48 kHz."""
    import torch  # type: ignore
    from df.enhance import enhance, init_df  # type: ignore
    model, df_state, _ = init_df()
    audio = _read(src, df_state.sr())
    out = enhance(model, df_state, torch.from_numpy(audio)[None, :])
    return _write(dst, out.squeeze(0).numpy(), df_state.sr())


def voice_chain(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Podcast voice dynamics in pedalboard (mirrors the ffmpeg chain, but usable on numpy buffers)."""
    from pedalboard import Compressor, HighpassFilter, Limiter, LowpassFilter, NoiseGate, PeakFilter, Pedalboard  # type: ignore
    board = Pedalboard([
        HighpassFilter(80), LowpassFilter(14000),
        NoiseGate(threshold_db=-38, ratio=4, attack_ms=5, release_ms=50),
        Compressor(threshold_db=-22, ratio=3.5, attack_ms=5, release_ms=100),
        PeakFilter(180, -4, 1.5), PeakFilter(2500, 3, 1.5), PeakFilter(3500, 4, 1.5), PeakFilter(5500, 2.5, 2), PeakFilter(8000, 1.5, 2),
        Limiter(threshold_db=-1.5),
    ])
    return board(audio, sr)


def duck(voice: np.ndarray, music: np.ndarray, sr: int = SR, music_db: float = -18.0, duck_db: float = -12.0,
         attack: float = 0.05, release: float = 0.6) -> np.ndarray:
    """Mix music under voice, pulling music down by duck_db whenever voice is present."""
    n = max(len(voice), len(music))
    v = np.pad(voice, (0, n - len(voice)))
    m = np.pad(music, (0, n - len(music)))[:n] if len(music) >= n else np.resize(music, n)
    win = int(sr * 0.02)
    env = np.sqrt(np.convolve(v ** 2, np.ones(win) / win, mode="same"))
    gate = (env > 0.01).astype(np.float32)
    a, r = np.exp(-1 / (sr * attack)), np.exp(-1 / (sr * release))
    g, out = 0.0, np.empty(n, np.float32)
    for i in range(n):  # one-pole follower; fine for clip lengths
        g = a * g + (1 - a) * gate[i] if gate[i] > g else r * g + (1 - r) * gate[i]
        out[i] = g
    gain = 10 ** (music_db / 20) * 10 ** (duck_db * out / 20)
    return np.clip(v + m * gain, -1, 1)


def loudness_normalize(src: str | Path, dst: str | Path, target: float = -16.0) -> Path:
    """Two-pass ffmpeg loudnorm to an exact integrated loudness."""
    from .audio import measure_loudness
    s = measure_loudness(src)
    af = (f"loudnorm=I={target}:LRA=11:TP=-1.5:measured_I={s['input_i']}:measured_LRA={s['input_lra']}:"
          f"measured_TP={s['input_tp']}:measured_thresh={s['input_thresh']}:offset={s['target_offset']}:linear=true")
    ffmpeg.run(["-i", str(src), "-af", af, "-c:a", "aac", "-b:a", "192k", str(dst)], show_progress=False)
    return Path(dst)
