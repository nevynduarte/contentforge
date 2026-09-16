"""Voiceover for intros/outros with Kokoro-82M (Apache-2.0 weights, runs on CPU or GPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_pipe = None


def synthesize(text: str, dst: str | Path, voice: str = "af_heart", speed: float = 1.0, lang: str = "a") -> Path:
    """Text -> WAV (24 kHz). voice examples: af_heart, af_bella, am_adam, bm_george; lang 'a' = American English."""
    global _pipe
    import soundfile as sf  # type: ignore
    from kokoro import KPipeline  # type: ignore
    if _pipe is None:
        _pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")
    chunks = [audio for _, _, audio in _pipe(text, voice=voice, speed=speed)]
    audio = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(2400, np.float32)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, 24000)
    return Path(dst)
