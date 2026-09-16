"""GPU upscaling of cropped regions with spandrel-loaded models (Real-ESRGAN by default).

Used inside the shot renderer: a 600x1080 speaker crop is upscaled before being
resized onto the 1080x1920 canvas, so the final image has real detail rather than
bilinear blur. SeedVR2 (video-native, higher quality) can be plugged in through
ComfyUI later; this module is the fast per-frame path.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.paths import model_path

_models: dict[str, object] = {}


def _load(name: str):
    if name in _models:
        return _models[name]
    import torch  # type: ignore
    import spandrel  # type: ignore
    m = spandrel.ModelLoader().load_from_file(str(model_path(name)))
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda().half()
    _models[name] = m
    return m


class Upscaler:
    """Callable: RGB uint8 array -> upscaled RGB uint8 array (model scale, usually 4x)."""

    def __init__(self, model: str = "realesr-general-x4v3.pth", denoise_mix: float = 0.0):
        self.model_name = model
        self.denoise_mix = denoise_mix   # 0..1 blend with the -wdn (denoising) variant, like realesrgan -dn
        self.m = _load(model)
        self.m_dn = _load("realesr-general-wdn-x4v3.pth") if denoise_mix > 0 else None
        import torch  # type: ignore
        self.torch = torch
        self.scale = getattr(self.m, "scale", 4)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if torch.cuda.is_available():
            x = x.cuda().half()
        with torch.no_grad():
            y = self.m(x)
            if self.m_dn is not None:
                y = (1 - self.denoise_mix) * y + self.denoise_mix * self.m_dn(x)
        y = (y.clamp(0, 1) * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return y


def available() -> bool:
    try:
        import spandrel  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get(quality: str = "fast") -> Optional[Upscaler]:
    if quality == "none" or not available():
        return None
    return Upscaler(denoise_mix=0.3 if quality == "clean" else 0.0)
