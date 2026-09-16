"""Generate placeholder brand assets (logo.png, lower_third.png) for a template.

Usage: python scripts/make_brand_assets.py bridges_ai
Replace the generated files with real artwork whenever it is ready.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from contentforge.config import TEMPLATES_DIR, Brand
from contentforge.pipeline.brand import lower_third_image
from contentforge.pipeline.captions import load_font
from contentforge.utils.colors import hex_to_rgb


def make_logo(brand: Brand, dst: Path, size: int = 512) -> None:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    navy, teal, amber = (hex_to_rgb(brand.colors[k]) for k in ("primary", "accent", "highlight"))
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 6, fill=navy)
    # stylised bridge arc + AI node
    d.arc([size * 0.12, size * 0.30, size * 0.88, size * 1.05], start=200, end=340, fill=teal, width=size // 14)
    d.ellipse([size * 0.42, size * 0.20, size * 0.58, size * 0.36], fill=amber)
    d.line([size * 0.5, size * 0.36, size * 0.5, size * 0.56], fill=amber, width=size // 40)
    f = load_font(brand.captions.font, size // 6)
    text = "".join(w[0] for w in (brand.company or "CF").split())[:3].upper()
    tw = d.textlength(text, font=f)
    d.text(((size - tw) / 2, size * 0.62), text, font=f, fill=(255, 255, 255))
    img.save(dst)


def main(name: str = "bridges_ai") -> None:
    tdir = TEMPLATES_DIR / name
    brand = Brand.load(name)
    make_logo(brand, tdir / "logo.png")
    lower_third_image(brand, brand.speakers.get("host", {}).get("name", "Speaker"),
                      brand.speakers.get("host", {}).get("title", brand.show)).save(tdir / "lower_third.png")
    print("wrote", tdir / "logo.png", tdir / "lower_third.png")


if __name__ == "__main__":
    main(*sys.argv[1:2])
