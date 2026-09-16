"""SeedVR2 video restoration / upscaling (ByteDance, Apache-2.0) through the numz CLI.

Two-pass "clean speaker" flow:
  1. render the speaker layout with no banner / captions / logo (plain video)
  2. SeedVR2 restores it (temporally consistent, real skin detail)
  3. overlay banner + captions + logo on the restored video

The CLI lives in a git checkout (CONTENTFORGE_SEEDVR2 or D:/contentforge-cache/seedvr2/repo)
and downloads weights on first run. 3B fp16 fits a 24 GB card without offloading.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..utils.paths import MODELS_DIR

_CANDIDATES = [os.environ.get("CONTENTFORGE_SEEDVR2"), str(MODELS_DIR.parent / "seedvr2" / "repo")]


def repo() -> Path | None:
    for c in _CANDIDATES:
        if c and (Path(c) / "inference_cli.py").exists():
            return Path(c)
    return None


def available() -> bool:
    return repo() is not None


# Memory-safe defaults for a 24 GB card at 1080x1920: tiled VAE and offloading models between phases.
MEMORY_SAFE = ["--vae_encode_tiled", "--vae_decode_tiled", "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
               "--chunk_size", "65", "--cache_dit", "--cache_vae"]   # stream 65-frame chunks instead of loading the whole clip into RAM


def restore(src: str | Path, dst: str | Path, resolution: int = 1080, batch_size: int = 5, model: str | None = None,
            extra: list[str] | None = None, memory_safe: bool = True) -> Path:
    """Run SeedVR2 on a video. `resolution` is the target short side; batch_size must be 4n+1."""
    r = repo()
    if r is None:
        raise RuntimeError("SeedVR2 CLI not found; clone numz/ComfyUI-SeedVR2_VideoUpscaler and set CONTENTFORGE_SEEDVR2")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "inference_cli.py", str(Path(src).resolve()), "--resolution", str(resolution),
           "--batch_size", str(batch_size), "--output", str(dst.resolve())]
    if model:
        cmd += ["--dit_model", model]
    if memory_safe:
        cmd += MEMORY_SAFE
    cmd += extra or []
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "HF_HOME": os.environ.get("HF_HOME", str(MODELS_DIR.parent / "hf"))}
    proc = subprocess.run(cmd, cwd=str(r), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"SeedVR2 failed ({proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    return dst
