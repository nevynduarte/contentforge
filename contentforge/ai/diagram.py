"""Reference-image generation from text: Mermaid diagrams (via mermaid-cli) and simple branded cards.

mermaid-cli lives in a node project; set CONTENTFORGE_MMDC to its mmdc path or let
this module look in the default cache locations.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..config import Brand
from ..utils.paths import MODELS_DIR

_MMDC_CANDIDATES = [os.environ.get("CONTENTFORGE_MMDC"), str(MODELS_DIR.parent / "npm/node_modules/.bin/mmdc.cmd"),
                    str(MODELS_DIR.parent / "npm/node_modules/.bin/mmdc"), shutil.which("mmdc")]


def mmdc_path() -> Optional[str]:
    return next((p for p in _MMDC_CANDIDATES if p and Path(p).exists()), None)


def mermaid_to_png(source: str, dst: str | Path, brand: Optional[Brand] = None, width: int = 1080, scale: int = 2,
                   theme: str = "base") -> Path:
    """Render Mermaid text to PNG with brand colours, transparent background."""
    mmdc = mmdc_path()
    if not mmdc:
        raise RuntimeError("mermaid-cli not found; npm install @mermaid-js/mermaid-cli and set CONTENTFORGE_MMDC")
    brand = brand or Brand()
    cfg = {"theme": theme, "themeVariables": {
        "primaryColor": brand.colors["primary"], "primaryTextColor": brand.colors["text"], "primaryBorderColor": brand.colors["accent"],
        "lineColor": brand.colors["accent"], "secondaryColor": brand.colors["accent"], "tertiaryColor": brand.colors["background"],
        "fontFamily": "Arial", "fontSize": "22px"}}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "d.mmd"
        src.write_text(source, encoding="utf-8")
        conf = Path(td) / "c.json"
        import json
        conf.write_text(json.dumps(cfg), encoding="utf-8")
        env = {**os.environ, "PUPPETEER_CACHE_DIR": os.environ.get("PUPPETEER_CACHE_DIR", str(MODELS_DIR.parent / "puppeteer"))}
        subprocess.run([mmdc, "-i", str(src), "-o", str(dst), "-c", str(conf), "-b", "transparent", "-w", str(width), "-s", str(scale)],
                       check=True, env=env, capture_output=True)
    return Path(dst)


def card(title: str, subtitle: str, dst: str | Path, brand: Optional[Brand] = None, w: int = 1080, h: int = 760) -> Path:
    from ..pipeline.brand import title_card
    title_card(brand or Brand(), title, subtitle, w, h, dst=Path(dst))
    return Path(dst)
