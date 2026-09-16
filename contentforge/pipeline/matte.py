"""Person matting with Robust Video Matting (MobileNetV3), for the speaker-over-image shot.

RVM is recurrent: call `Matter.step(frame)` in order for temporally stable alpha.
"""
from __future__ import annotations

import numpy as np

from ..utils.paths import model_path


class Matter:
    def __init__(self, downsample_ratio: float = 0.375):
        import torch  # type: ignore
        self.torch = torch
        # torch.hub pulls the model definition from the RVM repo; weights come from our models dir.
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", pretrained=False, trust_repo=True)
        self.model.load_state_dict(torch.load(str(model_path("rvm_mobilenetv3.pth")), map_location="cpu"))
        self.model.eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.dev)
        if self.dev == "cuda":
            self.model = self.model.half()
        self.rec = [None] * 4
        self.ds = downsample_ratio

    def reset(self) -> None:
        self.rec = [None] * 4

    def step(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> alpha float32 HxW in [0,1]."""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.dev)
        if self.dev == "cuda":
            x = x.half()
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(x, *self.rec, self.ds)
        return pha[0, 0].float().cpu().numpy()


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
