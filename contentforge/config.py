"""Project + brand configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
PROJECTS_DIR = ROOT / "projects"


@dataclass
class CaptionStyle:
    font: str = "arialbd.ttf"
    size: int = 52
    color: str = "#FFFFFF"
    highlight: str = "#FFDC32"
    stroke: str = "#000000"
    stroke_width: int = 2
    pill_color: str = "#000000"
    pill_alpha: int = 180
    pill_radius: int = 12
    pill_pad: int = 16
    max_words: int = 5
    max_chars: int = 35
    hold: float = 0.4
    y: int = 780


@dataclass
class Brand:
    name: str = "default"
    company: str = ""
    show: str = ""
    colors: dict[str, str] = field(default_factory=lambda: {
        "primary": "#0B1F3A", "accent": "#00C2A8", "highlight": "#FFDC32",
        "background": "#111111", "text": "#FFFFFF", "muted": "#8A94A6"})
    hashtags: list[str] = field(default_factory=list)
    logo: Path | None = None
    lower_third: Path | None = None
    intro: Path | None = None
    outro: Path | None = None
    fonts_dir: Path | None = None
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    speakers: dict[str, dict[str, str]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str | Path) -> Brand:
        p = Path(name_or_path)
        if p.is_dir():
            p = p / "config.yaml"
        if not p.exists():
            p = TEMPLATES_DIR / str(name_or_path) / "config.yaml"
        if not p.exists():
            raise FileNotFoundError(f"brand template not found: {name_or_path}")
        base = p.parent
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cap = CaptionStyle(**{k: v for k, v in (d.get("captions") or {}).items() if k in CaptionStyle.__dataclass_fields__})

        def asset(key: str) -> Path | None:
            v = d.get("assets", {}).get(key)
            if not v:
                return None
            ap = base / v
            return ap if ap.exists() else None

        return cls(
            name=d.get("name", base.name), company=d.get("company", ""), show=d.get("show", ""),
            colors={**cls().colors, **(d.get("colors") or {})}, hashtags=d.get("hashtags") or [],
            logo=asset("logo"), lower_third=asset("lower_third"), intro=asset("intro"), outro=asset("outro"),
            fonts_dir=(base / "fonts") if (base / "fonts").exists() else None,
            captions=cap, speakers=d.get("speakers") or {}, raw=d,
        )


@dataclass
class ClipDef:
    id: str
    source: str
    start: float
    duration: float
    title: str = ""
    speakers: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Project:
    name: str
    root: Path                 # where the media lives
    brand: Brand
    episode: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    audio_sources: dict[str, str] = field(default_factory=dict)
    clips: list[ClipDef] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    presets: list[str] = field(default_factory=lambda: ["instagram_reel"])
    whisper_model: str = "large-v3"
    raw: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str, default: str) -> Path:
        return (self.root / self.outputs.get(key, default)).resolve()

    @property
    def studio_dir(self) -> Path:
        return self.path("studio", "edit/studio")

    @property
    def social_dir(self) -> Path:
        return self.path("social", "edit/social")

    @property
    def captions_dir(self) -> Path:
        return self.path("captions", "captions")

    @property
    def clips_dir(self) -> Path:
        return self.path("clips", "clips")

    def source_path(self, key: str) -> Path:
        return (self.root / self.sources.get(key, key)).resolve()

    @classmethod
    def load(cls, name_or_path: str | Path) -> Project:
        p = Path(name_or_path)
        if p.is_dir():
            p = p / "project.yaml"
        if not p.exists():
            p = PROJECTS_DIR / str(name_or_path) / "project.yaml"
        if not p.exists():
            raise FileNotFoundError(f"project not found: {name_or_path}")
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        root = Path(d.get("root") or p.parent).expanduser()
        clips: list[ClipDef] = []
        clip_src = d.get("clips")
        if isinstance(clip_src, str):
            cp = (p.parent / clip_src)
            files = sorted(cp.glob("*.yaml")) if cp.is_dir() else [cp]
            for f in files:
                cd = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                items = cd if isinstance(cd, list) else cd.get("clips", [cd])
                clips.extend(ClipDef(**{k: v for k, v in c.items() if k in ClipDef.__dataclass_fields__}) for c in items)
        elif isinstance(clip_src, list):
            clips = [ClipDef(**{k: v for k, v in c.items() if k in ClipDef.__dataclass_fields__}) for c in clip_src]
        return cls(
            name=d.get("name", p.parent.name), root=root, brand=Brand.load(d.get("brand", "default")),
            episode=d.get("episode", ""), sources=d.get("sources") or {}, audio_sources=d.get("audio_sources") or {},
            clips=clips, outputs=d.get("outputs") or {}, presets=d.get("presets") or ["instagram_reel"],
            whisper_model=d.get("whisper_model", "large-v3"), raw=d,
        )
