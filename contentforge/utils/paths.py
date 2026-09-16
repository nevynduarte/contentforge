"""Model / cache locations. Heavy assets live off the system drive when possible."""
from __future__ import annotations

import os
from pathlib import Path

_CANDIDATES = [os.environ.get("CONTENTFORGE_MODELS"), "D:/contentforge-cache/models", str(Path.home() / ".cache/contentforge/models")]
MODELS_DIR = next(Path(p) for p in _CANDIDATES if p and Path(p).exists()) if any(p and Path(p).exists() for p in _CANDIDATES) \
    else Path.home() / ".cache/contentforge/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WORK_DIR = Path(os.environ.get("CONTENTFORGE_WORK") or ("D:/contentforge-cache/work" if Path("D:/").exists() else Path.home() / ".cache/contentforge/work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Keep HF / torch hub downloads next to the models unless the user already configured them.
os.environ.setdefault("HF_HOME", str(MODELS_DIR.parent / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR.parent / "torch"))

MODEL_URLS = {
    "realesr-general-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "rvm_mobilenetv3.pth": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
}


def model_path(name: str) -> Path:
    """Return the local path of a model file, downloading it on first use."""
    p = MODELS_DIR / name
    if not p.exists():
        url = MODEL_URLS.get(name)
        if not url:
            raise FileNotFoundError(f"model {name} not found in {MODELS_DIR} and no download URL known")
        import urllib.request
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(p)
    return p
