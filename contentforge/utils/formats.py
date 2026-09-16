"""Output format presets (loaded from presets/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"


@dataclass
class Preset:
    name: str
    width: int
    height: int
    max_duration: Optional[float] = None
    fps: float = 30.0
    crf: int = 20
    audio_bitrate: str = "192k"
    layout: str = "letterbox"          # letterbox | crop | fit
    caption_style: str = "karaoke"     # karaoke | popup | bar | none
    caption_y: Optional[int] = None
    video_y: Optional[int] = None
    hashtags_max: int = 5
    srt_sidecar: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def aspect(self) -> float:
        return self.width / self.height


def load_preset(name_or_path: str | Path) -> Preset:
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"preset not found: {name_or_path} (looked in {PRESETS_DIR})")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    known = {k: data.pop(k) for k in list(data) if k in Preset.__dataclass_fields__}
    known.setdefault("name", p.stem)
    return Preset(**known, extra=data)


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
