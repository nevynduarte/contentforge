"""The five shot layouts, cut lists, and the question banner.

A ShotPlan is a list of Segments (source time ranges from one graded clip). Each
segment renders with one layout:

  speaker        tight portrait crop following the active seat (S1)
  both           full 16:9 frame letterboxed (S2)
  speaker_image  reference image on top, speaker crop below (S3)
  stacked        head-and-torso crops of both seats, one above the other (S4)
  image          reference image only over the interview audio (S5)
  auto           speaker when the active seat is known, otherwise both

Frames are composed in numpy/OpenCV, captions + banner in Pillow, then piped to
ffmpeg. Audio is cut with the same segment list and muxed at the end.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ..config import Brand, CaptionStyle
from ..tracking.face_track import Tracks, analyze
from ..tracking.speaker_focus import detect, seat_at
from ..utils import ffmpeg
from ..utils.colors import hex_to_rgb, hex_to_rgba
from ..utils.paths import WORK_DIR
from .captions import CaptionRenderer, load_font

W, H = 1080, 1920
BANNER_TOP, BANNER_MAX_H = 60, 200          # question banner region
CONTENT_TOP = 270                            # first pixel below the banner
SHOTS = ("speaker", "both", "speaker_image", "stacked", "image", "auto")


@dataclass
class Segment:
    start: float
    end: float
    shot: str = "auto"
    seat: str = "auto"           # L | R | auto
    image: Optional[str] = None  # reference image path for speaker_image / image

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ShotPlan:
    segments: list[Segment]
    question: str = ""
    captions: bool = True
    logo: bool = True
    upscale: str = "fast"        # none | fast | clean

    def save(self, path: str | Path) -> Path:
        Path(path).write_text(json.dumps({"question": self.question, "captions": self.captions, "logo": self.logo,
                                          "upscale": self.upscale, "segments": [asdict(s) for s in self.segments]}, indent=2), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "ShotPlan":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Segment(**s) for s in d["segments"]], d.get("question", ""), d.get("captions", True), d.get("logo", True), d.get("upscale", "fast"))

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)


# ---------------------------------------------------------------------------
# analysis cache
# ---------------------------------------------------------------------------
def analysis_for(clip: Path, every: float = 0.2, force: bool = False) -> tuple[Tracks, list[dict]]:
    """Face tracks + speaker turns for a clip, cached next to the work dir."""
    cache = WORK_DIR / "analysis"
    cache.mkdir(parents=True, exist_ok=True)
    tp, sp = cache / f"{clip.stem}.tracks.json", cache / f"{clip.stem}.turns.json"
    if tp.exists() and sp.exists() and not force:
        return Tracks.load(tp), json.loads(sp.read_text(encoding="utf-8"))
    tr = analyze(clip, every)
    tr.save(tp)
    turns = detect(str(clip), tr)
    sp.write_text(json.dumps(turns), encoding="utf-8")
    return tr, turns


# ---------------------------------------------------------------------------
# automatic plans
# ---------------------------------------------------------------------------
def auto_plan(clip: Path, keep: list[tuple[float, float]], question: str = "", image: Optional[str] = None,
              prefer: str = "speaker", turns: Optional[list[dict]] = None) -> ShotPlan:
    """Build a plan from kept ranges. At every join where a fragment was removed, the shot changes
    (speaker -> both / image, or the other speaker) so the cut reads as an edit, not a glitch."""
    if turns is None:
        _, turns = analysis_for(clip)
    segs: list[Segment] = []
    prev_shot, prev_seat, prev_end = None, None, None
    for a, b in keep:
        seat = seat_at(turns, (a + b) / 2)
        shot = prefer
        removed_before = prev_end is not None and a - prev_end > 0.05
        if removed_before and prev_shot == shot and prev_seat == seat and shot == "speaker":
            shot = "speaker_image" if image else "both"
        segs.append(Segment(a, b, shot, seat, image if shot in ("speaker_image", "image") else None))
        prev_shot, prev_seat, prev_end = shot, seat, b
    return ShotPlan(segs, question, upscale="fast")


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _portrait_crop(box: tuple[float, float, float, float], aspect: float, src_w: int, src_h: int,
                   head_scale: float = 5.4, y_bias: float = 0.20) -> tuple[int, int, int, int]:
    """Crop rect (x, y, w, h) around a face box: height = head_scale * face_h, given output aspect (w/h)."""
    fx, fy, fw, fh = box
    ch = min(src_h, fh * head_scale)
    cw = ch * aspect
    if cw > src_w:
        cw = src_w
        ch = cw / aspect
    cx, cy = fx + fw / 2, fy + fh / 2 + y_bias * ch
    x = int(np.clip(cx - cw / 2, 0, src_w - cw))
    y = int(np.clip(cy - ch / 2, 0, src_h - ch))
    return x, y, int(cw), int(ch)


def _resize(img: np.ndarray, w: int, h: int, up: Optional[object]) -> np.ndarray:
    """Resize with optional model upscale when enlarging by more than ~1.2x."""
    sh, sw = img.shape[:2]
    if up is not None and w / sw > 1.6:
        return up(img, (w, h))          # model upscale + antialiased resize on the GPU
    interp = cv2.INTER_AREA if w < sw else cv2.INTER_LANCZOS4
    return cv2.resize(img, (w, h), interpolation=interp)


def _fit_image(path: str, w: int, h: int, bg: tuple[int, int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(canvas)


class _Smoother:
    """Exponential follower with dead zone, per seat, for crop centres."""

    def __init__(self, alpha: float = 0.12, dead: float = 26.0):
        self.alpha, self.dead, self.cur = alpha, dead, {}

    def __call__(self, seat: str, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        fx, fy, fw, fh = box
        cx, cy, fs = fx + fw / 2, fy + fh / 2, (fw + fh) / 2
        c = self.cur.get(seat)
        if c is None:
            c = [cx, cy, fs]
        else:
            for i, v in enumerate((cx, cy)):
                if abs(v - c[i]) > self.dead:
                    c[i] += self.alpha * (v - c[i])
            c[2] += 0.05 * (fs - c[2])
        self.cur[seat] = c
        return c[0] - c[2] / 2, c[1] - c[2] / 2, c[2], c[2]

    def reset(self) -> None:
        self.cur = {}


# ---------------------------------------------------------------------------
# banner + brand overlays
# ---------------------------------------------------------------------------
class Banner:
    """Question banner in the brand's editorial style (serif headline, sans eyebrow, ink band, bronze rule)
    plus the logo lockup, placed per layout: bottom-left over full-bleed video, centred in the empty band otherwise."""

    def __init__(self, question: str, brand: Brand, logo: bool = True, width: int = W):
        self.width = width
        self.img = None
        fonts = brand.raw.get("fonts", {})
        fdir = [brand.fonts_dir] if brand.fonts_dir else None
        ink = brand.colors.get("ink", brand.colors["primary"])
        accent_light = brand.colors.get("accent_light", brand.colors["accent"])
        if question:
            font = load_font(fonts.get("display", brand.captions.font), 50, fdir)
            tag_font = load_font(fonts.get("sans_semibold", brand.captions.font), 22, fdir)
            probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
            lines, line = [], ""
            for word in question.split():
                if probe.textlength((line + " " + word).strip(), font=font) > width - 176:
                    lines.append(line.strip())
                    line = word
                else:
                    line += " " + word
            lines.append(line.strip())
            lines = lines[:3]
            lh = 60
            h = 36 + 30 + lh * len(lines) + 30
            img = Image.new("RGBA", (width - 80, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, img.width - 1, h - 1], radius=4, fill=hex_to_rgba(ink, 235))
            # eyebrow: Inter 600, uppercase, 0.2em tracking, bronze
            x = 40
            for ch in "QUESTION":
                d.text((x, 30), ch, font=tag_font, fill=hex_to_rgb(accent_light))
                x += d.textlength(ch, font=tag_font) + 22 * 0.2
            y = 64
            for ln in lines:
                d.text((40, y), ln, font=font, fill=hex_to_rgb(brand.colors["text"]))
                y += lh
            d.rectangle([40, y + 16, 40 + 48, y + 17], fill=hex_to_rgb(accent_light))   # .rule-accent
            self.img = img
        self.logo = None
        self.logo_big = None
        if logo and brand.logo and Path(brand.logo).exists():
            lg = Image.open(brand.logo).convert("RGBA")
            self.logo = lg.resize((300, int(lg.height * 300 / lg.width)), Image.LANCZOS)
            self.logo_big = lg.resize((420, int(lg.height * 420 / lg.width)), Image.LANCZOS)

    def draw(self, frame: Image.Image, shot: str = "speaker") -> None:
        if shot in ("landscape", "sidebyside"):
            if self.img is not None:
                frame.paste(self.img, (40, 40), self.img)
            if self.logo is not None:
                lg = self.logo
                x, y = frame.width - lg.width - 48, frame.height - lg.height - 44
                shadow = Image.new("RGBA", lg.size, (0, 0, 0, 0))
                shadow.paste((0, 0, 0, 140), (0, 0, *lg.size), lg)
                frame.paste(shadow, (x + 2, y + 3), shadow)
                frame.paste(lg, (x, y), lg)
            return
        if self.img is not None:
            frame.paste(self.img, (40, BANNER_TOP), self.img)
        if self.logo is None:
            return
        if shot in ("speaker", "speaker_image"):
            # full-bleed video at the bottom: small lockup bottom-left, above the safe zone, with a soft shadow
            lg = self.logo
            x, y = 40, H - lg.height - (60 if shot == "speaker_image" else 120)
            shadow = Image.new("RGBA", lg.size, (0, 0, 0, 0))
            shadow.paste((0, 0, 0, 140), (0, 0, *lg.size), lg)
            frame.paste(shadow, (x + 2, y + 3), shadow)
            frame.paste(lg, (x, y), lg)
        else:
            # letterboxed layouts leave an empty band at the bottom: centre the lockup there
            lg = self.logo_big
            frame.paste(lg, ((W - lg.width) // 2, H - lg.height - 70), lg)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------
def render(clip: str | Path, plan: ShotPlan, dst: str | Path, words: Optional[list[dict]] = None,
           brand: Optional[Brand] = None, tracks: Optional[Tracks] = None, turns: Optional[list[dict]] = None,
           fps: float = 30.0, crf: int = 19, gpu: bool = True) -> Path:
    clip, dst = Path(clip), Path(dst)
    brand = brand or Brand()
    info = ffmpeg.probe(clip)
    fps = min(fps, info.fps)
    if tracks is None or turns is None:
        tracks, turns = analysis_for(clip)
    bg = hex_to_rgb(brand.colors["background"])
    banner = Banner(plan.question, brand, plan.logo)
    up = None
    if plan.upscale != "none":
        from . import upscale as upmod
        up = upmod.get(plan.upscale)
    needs_matte = any(s.shot == "speaker_image" and s.image and _wants_matte(s) for s in plan.segments)
    matter = None
    if needs_matte:
        from .matte import Matter
        matter = Matter()

    # captions on the output timeline
    out_words: list[dict] = []
    t_out = 0.0
    for s in plan.segments:
        for w in (words or []):
            if s.start <= w["start"] < s.end:
                out_words.append({**w, "start": w["start"] - s.start + t_out, "end": min(w["end"], s.end) - s.start + t_out})
        t_out += s.duration
    cap_y = {"both": 960, "speaker": 1560, "stacked": 1715, "speaker_image": 1630, "image": 1610}

    total_frames = int(plan.duration * fps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = dst.with_suffix(".video.tmp.mp4")
    enc = subprocess.Popen([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:.5f}", "-i", "pipe:0",
                            *ffmpeg.video_codec_args("libx264", crf, "medium", gpu), "-pix_fmt", "yuv420p", str(tmp_v)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=W * H * 3 * 4)
    smoother = _Smoother()
    renderers: dict[str, CaptionRenderer] = {}
    progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn())
    n_out = 0
    try:
        with progress:
            task = progress.add_task(f"shots {dst.name}", total=total_frames)
            for seg in plan.segments:
                shot = seg.shot
                if shot == "auto":
                    shot = "speaker" if tracks.median_box("L") or tracks.median_box("R") else "both"
                ref_img = _fit_image(seg.image, W, 800, bg) if seg.image and shot == "speaker_image" else \
                          _fit_image(seg.image, W, 1300, bg) if seg.image and shot == "image" else None
                if matter is not None:
                    matter.reset()
                smoother.reset()
                for t_src, frame in _decode(clip, seg.start, seg.duration, fps):
                    t_out = n_out / fps
                    seat = seg.seat if seg.seat in ("L", "R") else seat_at(turns, t_src)
                    canvas = np.empty((H, W, 3), np.uint8)
                    canvas[:] = bg
                    if shot == "both":
                        vh = int(round(W * info.height / info.width / 2)) * 2
                        canvas[CONTENT_TOP:CONTENT_TOP + vh] = _resize(frame, W, vh, None)
                    elif shot == "speaker":
                        _paste_speaker(canvas, frame, tracks, seat, t_src, smoother, up, 0, H, info, head_scale=5.4)
                    elif shot == "stacked":
                        top_h = (1700 - CONTENT_TOP) // 2
                        _paste_speaker(canvas, frame, tracks, "L", t_src, smoother, up, CONTENT_TOP, top_h, info, head_scale=3.4)
                        _paste_speaker(canvas, frame, tracks, "R", t_src, smoother, up, CONTENT_TOP + top_h, top_h, info, head_scale=3.4)
                        cv2.line(canvas, (0, CONTENT_TOP + top_h), (W, CONTENT_TOP + top_h), hex_to_rgb(brand.colors["accent"]), 4)
                    elif shot == "speaker_image":
                        if ref_img is not None:
                            canvas[CONTENT_TOP:CONTENT_TOP + 800] = ref_img
                        _paste_speaker(canvas, frame, tracks, seat, t_src, smoother, up, CONTENT_TOP + 800, H - CONTENT_TOP - 800, info,
                                       head_scale=3.6, matter=matter)
                    elif shot == "image":
                        if ref_img is not None:
                            canvas[CONTENT_TOP:CONTENT_TOP + 1300] = ref_img
                    pil = Image.fromarray(canvas)
                    banner.draw(pil, shot)
                    if plan.captions and out_words:
                        r = renderers.get(shot)
                        if r is None:
                            style = CaptionStyle(**{**asdict(brand.captions), "y": cap_y.get(shot, 1560)})
                            r = renderers[shot] = CaptionRenderer(out_words, style, W, H, "karaoke", brand.fonts_dir)
                        r.composite(pil, t_out)
                    enc.stdin.write(np.asarray(pil, np.uint8).tobytes())
                    n_out += 1
                    if n_out % 15 == 0:
                        progress.update(task, completed=n_out)
            progress.update(task, completed=total_frames)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise ffmpeg.FFmpegError(enc.stderr.read().decode(errors="replace")[-1500:])

    # audio: cut the same ranges, concat, mux
    parts = "".join(f"[0:a]atrim={s.start:.3f}:{s.end:.3f},asetpts=PTS-STARTPTS[a{i}];" for i, s in enumerate(plan.segments))
    fc = parts + "".join(f"[a{i}]" for i in range(len(plan.segments))) + f"concat=n={len(plan.segments)}:v=0:a=1[aout]"
    ffmpeg.run(["-i", str(clip), "-i", str(tmp_v), "-filter_complex", fc, "-map", "1:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(dst)],
               duration=plan.duration, description=f"mux {dst.name}", show_progress=False)
    tmp_v.unlink(missing_ok=True)
    return dst


def _wants_matte(seg: Segment) -> bool:
    return False  # matting over the image is opt-in per segment (future: seg.matte flag); default keeps the split layout


def _paste_speaker(canvas: np.ndarray, frame: np.ndarray, tracks: Tracks, seat: str, t: float, smoother: _Smoother,
                   up, y0: int, h: int, info, head_scale: float = 4.2, matter=None) -> None:
    box = tracks.seat_box(seat, t) or tracks.median_box(seat) or tracks.median_box("L" if seat == "R" else "R")
    if box is None:
        # no faces found at all: centre crop
        cw = int(info.height * W / h) if h else info.width
        box = (info.width / 2 - 100, info.height / 2 - 100, 200, 200)
    sb = smoother(seat, box)
    x, y, cw, ch = _portrait_crop(sb, W / h, info.width, info.height, head_scale)
    crop = frame[y:y + ch, x:x + cw]
    region = _resize(crop, W, h, up)
    if matter is not None:
        alpha = matter.step(region)[..., None]
        canvas[y0:y0 + h] = (region * alpha + canvas[y0:y0 + h] * (1 - alpha)).astype(np.uint8)
    else:
        canvas[y0:y0 + h] = region


def _decode(clip: Path, start: float, duration: float, fps: float):
    info = ffmpeg.probe(clip)
    fb = info.width * info.height * 3
    proc = subprocess.Popen([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(clip),
                             "-t", f"{duration:.3f}", "-vf", f"fps={fps:.5f}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=fb * 4)
    n = 0
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb:
                break
            yield start + n / fps, np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# landscape (1920x1080)
#   landscape : full frame, banner top-left, captions bottom, lockup bottom-right
#   sidebyside: each seat cropped head-and-torso into its own 960x1080 half
# ---------------------------------------------------------------------------
def render_landscape(clip: str | Path, plan: ShotPlan, dst: str | Path, words: Optional[list[dict]] = None,
                     brand: Optional[Brand] = None, mode: str = "landscape", tracks: Optional[Tracks] = None,
                     fps: float = 30.0, crf: int = 19, gpu: bool = True) -> Path:
    clip, dst = Path(clip), Path(dst)
    brand = brand or Brand()
    info = ffmpeg.probe(clip)
    fps = min(fps, info.fps)
    LW, LH = 1920, 1080
    if mode == "sidebyside" and tracks is None:
        tracks, _ = analysis_for(clip)
    up = None
    if mode == "sidebyside" and plan.upscale != "none":
        from . import upscale as upmod
        up = upmod.get(plan.upscale)
    banner = Banner(plan.question, brand, plan.logo, width=1180)
    out_words, t_out = [], 0.0
    for s in plan.segments:
        for w in (words or []):
            if s.start <= w["start"] < s.end:
                out_words.append({**w, "start": w["start"] - s.start + t_out, "end": min(w["end"], s.end) - s.start + t_out})
        t_out += s.duration
    style = CaptionStyle(**{**asdict(brand.captions), "y": 940, "size": 46})
    renderer = CaptionRenderer(out_words, style, LW, LH, "karaoke", brand.fonts_dir) if (plan.captions and out_words) else None
    total_frames = int(plan.duration * fps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_v = dst.with_suffix(".video.tmp.mp4")
    enc = subprocess.Popen([ffmpeg.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{LW}x{LH}", "-r", f"{fps:.5f}", "-i", "pipe:0",
                            *ffmpeg.video_codec_args("libx264", crf, "medium", gpu), "-pix_fmt", "yuv420p", str(tmp_v)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=LW * LH * 3 * 4)
    progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn())
    smoother = _Smoother()
    accent = hex_to_rgb(brand.colors.get("accent_light", brand.colors["accent"]))
    n_out = 0
    try:
        with progress:
            task = progress.add_task(f"{mode} {dst.name}", total=total_frames)
            for seg in plan.segments:
                smoother.reset()
                for t_src, frame in _decode(clip, seg.start, seg.duration, fps):
                    t_out = n_out / fps
                    if mode == "sidebyside":
                        canvas = np.empty((LH, LW, 3), np.uint8)
                        canvas[:] = hex_to_rgb(brand.colors["background"])
                        half = LW // 2
                        for i, seat in enumerate(("L", "R")):
                            box = tracks.seat_box(seat, t_src) or tracks.median_box(seat)
                            if box is None:
                                continue
                            sb = smoother(seat, box)
                            x, y, cw, ch = _portrait_crop(sb, half / LH, info.width, info.height, head_scale=4.6, y_bias=0.16)
                            canvas[:, i * half:(i + 1) * half] = _resize(frame[y:y + ch, x:x + cw], half, LH, up)
                        cv2.line(canvas, (half, 0), (half, LH), accent, 4)
                        pil = Image.fromarray(canvas)
                    else:
                        if frame.shape[1] != LW or frame.shape[0] != LH:
                            frame = cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA)
                        pil = Image.fromarray(np.ascontiguousarray(frame))
                    banner.draw(pil, mode)
                    if renderer is not None:
                        renderer.composite(pil, t_out)
                    enc.stdin.write(np.asarray(pil, np.uint8).tobytes())
                    n_out += 1
                    if n_out % 15 == 0:
                        progress.update(task, completed=n_out)
            progress.update(task, completed=total_frames)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode != 0:
        raise ffmpeg.FFmpegError(enc.stderr.read().decode(errors="replace")[-1500:])
    parts = "".join(f"[0:a]atrim={s.start:.3f}:{s.end:.3f},asetpts=PTS-STARTPTS[a{i}];" for i, s in enumerate(plan.segments))
    fc = parts + "".join(f"[a{i}]" for i in range(len(plan.segments))) + f"concat=n={len(plan.segments)}:v=0:a=1[aout]"
    ffmpeg.run(["-i", str(clip), "-i", str(tmp_v), "-filter_complex", fc, "-map", "1:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(dst)],
               duration=plan.duration, description=f"mux {dst.name}", show_progress=False)
    tmp_v.unlink(missing_ok=True)
    return dst
