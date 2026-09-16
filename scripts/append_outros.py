"""Append the brand outro to every short: edit/shorts/*.mp4 -> edit/final/*.mp4 (orientation-matched)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project  # noqa: E402
from contentforge.pipeline.bumper import append_outro  # noqa: E402
from contentforge.utils import ffmpeg  # noqa: E402


def main() -> None:
    project = Project.load(sys.argv[1] if len(sys.argv) > 1 else "bridges_ai_3rdi")
    src_dir, out_dir = project.root / "edit" / "shorts", project.root / "edit" / "final"
    bumpers = project.root / "edit" / "bumpers"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.mp4")):
        if ".tmp" in f.name:
            continue
        dst = out_dir / f.name
        if dst.exists() and ffmpeg.is_healthy(dst) and os.environ.get("FORCE") != "1":
            print(f"{f.name}: exists", flush=True)
            continue
        info = ffmpeg.probe(f)
        outro = bumpers / ("outro_portrait.mp4" if info.height > info.width else "outro_landscape.mp4")
        t0 = time.time()
        append_outro(f, outro, dst)
        print(f"{f.name}: +outro in {time.time() - t0:.0f}s", flush=True)
    print("OUTROS_DONE", flush=True)


if __name__ == "__main__":
    main()
