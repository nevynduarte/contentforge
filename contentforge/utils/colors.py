"""Color helpers: hex parsing, contrast, ffmpeg color strings."""
from __future__ import annotations

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]


def hex_to_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_rgba(h: str, alpha: int = 255) -> RGBA:
    r, g, b = hex_to_rgb(h)
    return r, g, b, alpha


def rgb_to_hex(rgb: RGB) -> str:
    return "#%02X%02X%02X" % rgb


def ffmpeg_color(h: str, alpha: float | None = None) -> str:
    """ffmpeg accepts 0xRRGGBB or 0xRRGGBB@alpha."""
    base = "0x" + h.lstrip("#").upper()
    return base + ("@%s" % alpha if alpha is not None else "")


def relative_luminance(rgb: RGB) -> float:
    def ch(c: int) -> float:
        c2 = c / 255
        return c2 / 12.92 if c2 <= 0.03928 else ((c2 + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a: RGB, b: RGB) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def best_text_color(bg: RGB) -> RGB:
    white, dark = (255, 255, 255), (17, 17, 17)
    return white if contrast_ratio(bg, white) >= contrast_ratio(bg, dark) else dark
