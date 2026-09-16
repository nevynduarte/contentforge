"""Speech-to-text with word-level timestamps (faster-whisper, CUDA when available).

Outputs the same word JSON shape the caption renderer consumes:
    [{"word": " these", "start": 0.0, "end": 0.16}, ...]
plus a segment-level SRT.
"""
from __future__ import annotations

import glob
import json
import os
import site
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console

console = Console()


def _enable_cuda_dlls() -> None:
    """Expose pip-installed NVIDIA cuBLAS/cuDNN DLLs to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def load_model(model: str = "large-v3", device: str = "auto", compute_type: Optional[str] = None):
    _enable_cuda_dlls()
    from faster_whisper import WhisperModel  # heavy import, keep lazy

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    key = (model, device, compute_type)
    if key not in _MODEL_CACHE:
        console.print(f"[dim]loading whisper {model} on {device} ({compute_type})[/dim]")
        _MODEL_CACHE[key] = WhisperModel(model, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: Iterable[dict], path: str | Path) -> Path:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}", seg["text"].strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def transcribe(
    media: str | Path,
    words_json: Optional[str | Path] = None,
    srt: Optional[str | Path] = None,
    model: str = "large-v3",
    language: Optional[str] = "en",
    device: str = "auto",
    beam_size: int = 5,
    initial_prompt: Optional[str] = None,
    vad: Optional[bool] = None,
) -> dict:
    """Transcribe with word timestamps. Returns {"words": [...], "segments": [...], "language": str}.

    vad=None picks a default: on. Except on Windows with CUDA, where importing onnxruntime
    (Silero VAD) after the CUDA model is loaded crashes the process (0xc000070a); there it is off.
    Set CONTENTFORGE_VAD=1 to force it on.
    """
    wm = load_model(model, device)
    if vad is None:
        cuda_win = os.name == "nt" and getattr(wm, "model", None) is not None and wm.model.device == "cuda"
        vad = os.environ.get("CONTENTFORGE_VAD") == "1" or not cuda_win
    seg_iter, info = wm.transcribe(
        str(media), language=language, beam_size=beam_size, word_timestamps=True,
        vad_filter=vad, vad_parameters={"min_silence_duration_ms": 300}, initial_prompt=initial_prompt,
        condition_on_previous_text=False,
    )
    words: list[dict] = []
    segments: list[dict] = []
    for seg in seg_iter:
        segments.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": seg.text})
        for w in seg.words or []:
            words.append({"word": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                          "probability": round(float(w.probability), 3)})
    if words_json:
        Path(words_json).parent.mkdir(parents=True, exist_ok=True)
        Path(words_json).write_text(json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")
    if srt:
        write_srt(segments, srt)
    return {"words": words, "segments": segments, "language": info.language}


def load_words(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def words_to_text(words: list[dict]) -> str:
    return "".join(w["word"] for w in words).strip()
