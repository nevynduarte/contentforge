"""Render full-length social clips (1080x1920 letterbox + karaoke captions) for a project.

Usage: python scripts/render_social_clips.py [project] [clip ids...]
Writes <social_dir>/clip_XX_social.mp4. Skips healthy existing outputs unless FORCE=1.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project  # noqa: E402
from contentforge.pipeline.batch import find_studio, words_path  # noqa: E402
from contentforge.pipeline.render import render_preset  # noqa: E402
from contentforge.utils import ffmpeg  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    project = Project.load(args[0] if args else "bridges_ai_3rdi")
    ids = args[1:] or [c.id for c in project.clips]
    gpu = os.environ.get("GPU", "0") == "1"
    for cid in ids:
        studio, words = find_studio(project, cid), words_path(project, cid)
        dst = project.social_dir / f"{cid}_social.mp4"
        if not studio or not words.exists():
            print(f"{cid}: missing studio ({studio}) or words ({words}); skipping", flush=True)
            continue
        if dst.exists() and ffmpeg.is_healthy(dst) and os.environ.get("FORCE") != "1":
            print(f"{cid}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        render_preset(studio, dst, "instagram_reel", words, project.brand, gpu=gpu, show_logo=False, max_duration=0)
        print(f"{cid}: {dst.name} {dst.stat().st_size/1e6:.0f}MB in {time.time()-t0:.0f}s", flush=True)
    print("SOCIAL_DONE", flush=True)


if __name__ == "__main__":
    main()
