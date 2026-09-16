"""High-quality speaker shorts: plain speaker render -> SeedVR2 restore -> banner + captions + logo overlay.

Usage: python scripts/seedvr_speaker.py [window ids...]   (default: all windows in shot_windows.WINDOWS)
Outputs edit/shorts_hq/<id>_speaker_hq.mp4 (and the outro-appended edit/final version if the outro exists).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shot_windows import WINDOWS, resolve  # noqa: E402

from contentforge.config import CaptionStyle, Project  # noqa: E402
from contentforge.pipeline.batch import find_studio, words_path  # noqa: E402
from contentforge.pipeline.captions import render_captioned_video  # noqa: E402
from contentforge.pipeline.shots import Banner, Segment, ShotPlan, W, H, analysis_for, render  # noqa: E402
from contentforge.pipeline.transcribe import load_words  # noqa: E402
from contentforge.pipeline.upscale_seedvr import restore  # noqa: E402
from contentforge.pipeline.bumper import append_outro  # noqa: E402
from contentforge.utils.paths import WORK_DIR  # noqa: E402


def main() -> None:
    ids = [a for a in sys.argv[1:] if not a.startswith("--")]
    project = Project.load("bridges_ai_3rdi")
    out_dir = project.root / "edit" / "shorts_hq"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = WORK_DIR / "seedvr"
    work.mkdir(parents=True, exist_ok=True)
    outro = project.root / "edit" / "bumpers" / "outro_portrait.mp4"
    for wid, cid, question, a, b in WINDOWS:
        if ids and wid not in ids:
            continue
        dst = out_dir / f"{wid}_speaker_hq.mp4"
        if dst.exists() and os.environ.get("FORCE") != "1":
            print(f"{wid}: exists", flush=True)
            continue
        studio, words = find_studio(project, cid), load_words(words_path(project, cid))
        s, e = resolve(words, a, b)
        tracks, turns = analysis_for(studio)
        t0 = time.time()
        # 1. plain speaker video: no banner, captions or logo; bilinear only (SeedVR2 does the detail)
        plain = work / f"{wid}_plain.mp4"
        plan = ShotPlan([Segment(s, e, "speaker", "auto")], "", captions=False, logo=False, upscale="none")
        render(studio, plan, plain, None, project.brand, tracks, turns, crf=14)
        t1 = time.time()
        # 2. SeedVR2 restoration at 1080 short side (input is already 1080x1920, so this is a restore pass)
        restored = work / f"{wid}_restored.mp4"
        restore(plain, restored, resolution=1080, batch_size=int(os.environ.get("SEEDVR_BATCH", "9")),
                model=os.environ.get("SEEDVR_MODEL", "seedvr2_ema_3b_fp16.safetensors"))
        t2 = time.time()
        # 3. overlay banner + captions + logo, audio from the plain render
        out_words = [{**w, "start": w["start"] - s, "end": min(w["end"], e) - s} for w in words if s <= w["start"] < e]
        banner = Banner(question, project.brand, True)
        style = CaptionStyle(**{**asdict(project.brand.captions), "y": 1560})
        render_captioned_video(restored, dst, out_words, style, W, H, pre_filter="", fps=30, crf=18, gpu=True,
                               fonts_dir=project.brand.fonts_dir, frame_hook=lambda img, t: (banner.draw(img, "speaker"), img)[1],
                               audio_src=plain)
        t3 = time.time()
        print(f"{wid}: plain {t1 - t0:.0f}s, seedvr2 {t2 - t1:.0f}s, overlay {t3 - t2:.0f}s -> {dst.name}", flush=True)
        if outro.exists():
            final = project.root / "edit" / "final" / f"{wid}_speaker_hq.mp4"
            append_outro(dst, outro, final)
    print("SEEDVR_SPEAKER_DONE", flush=True)


if __name__ == "__main__":
    main()
