"""ContentForge CLI (Typer)."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Brand, Project

app = typer.Typer(help="ContentForge: raw podcast footage -> polished social clips.", no_args_is_help=True)
console = Console()


@app.callback()
def _main(version: bool = typer.Option(False, "--version", is_eager=True)):
    if version:
        console.print(f"contentforge {__version__}")
        raise typer.Exit()


@app.command()
def probe(path: Path, recursive: bool = False):
    """Probe media file(s); flags broken (moov-less) MP4s."""
    from .pipeline import ingest
    if path.is_dir():
        console.print(ingest.table(ingest.scan(path, recursive)))
    else:
        from .utils.ffmpeg import probe as p
        console.print(p(path))


@app.command()
def grade(src: Path, dst: Path, start: float | None = None, duration: float | None = None,
          crf: int = 18, gpu: bool = False):
    """Apply the studio_v2 grade + podcast voice chain and encode."""
    from .pipeline.grade import grade as g
    console.print(f"[green]done[/green] {g(src, dst, start, duration, crf=crf, gpu=gpu)}")


@app.command()
def transcribe(media: Path, out_dir: Path | None = None, model: str = "large-v3", language: str = "en",
               device: str = "auto"):
    """Word-level transcription (faster-whisper); writes <stem>_words.json + <stem>.srt."""
    from .pipeline.transcribe import transcribe as t
    out_dir = out_dir or media.parent
    stem = media.stem.replace("_studio", "")
    r = t(media, out_dir / f"{stem}_words.json", out_dir / f"{stem}.srt", model=model, language=language, device=device)
    console.print(f"[green]{len(r['words'])} words[/green] -> {out_dir / (stem + '_words.json')}")


@app.command()
def social(src: Path, words: Path, dst: Path, preset: str = "instagram_reel", brand: str = "default",
           gpu: bool = False, no_logo: bool = False, crop: str = "center",
           full: bool = typer.Option(False, help="Ignore the preset's duration cap (render the whole clip).")):
    """Render one graded clip into a platform preset with karaoke captions."""
    from .pipeline.render import render_preset
    b = Brand.load(brand)
    out = render_preset(src, dst, preset, words, b, crop_mode=crop, gpu=gpu, show_logo=not no_logo,
                        max_duration=0 if full else None)
    console.print(f"[green]done[/green] {out}")


@app.command()
def batch(project: str, clips: list[str] | None = typer.Argument(None), preset: list[str] | None = typer.Option(None),
          force_grade: bool = False, force_words: bool = False, force_render: bool = False, gpu: bool = False,
          no_logo: bool = False):
    """Run the whole pipeline for a project (grade -> words -> presets -> manifest)."""
    from .pipeline.batch import run
    p = Project.load(project)
    m = run(p, clips or None, preset or None, force_grade, force_words, force_render, gpu, show_logo=not no_logo)
    console.print(f"[green]{len(m)} clips rendered[/green]; manifest at {p.social_dir / 'manifest.json'}")


@app.command()
def suggest(words: Path, n: int = 5, platform: str = "Instagram Reel", llm: bool = False,
            min_len: float = 20, max_len: float = 90):
    """Suggest best clip windows from a word-level transcript."""
    from .pipeline.clip import best_clips
    from .pipeline.transcribe import load_words
    res = best_clips(load_words(words), n, platform, use_llm=llm, min_len=min_len, max_len=max_len)
    t = Table(title="Suggested clips")
    for c in ("score", "start", "end", "len", "text"):
        t.add_column(c)
    for s in res:
        t.add_row(f"{s.total:.2f}", f"{s.start:.1f}", f"{s.end:.1f}", f"{s.duration:.0f}s", s.text[:90] + "…")
    console.print(t)


@app.command()
def copy(words: Path, brand: str = "default", platform: str = "instagram", llm: bool = True, out: Path | None = None):
    """Generate title / caption / hashtags for a clip transcript."""
    from .ai.social_copy import generate
    from .pipeline.transcribe import load_words, words_to_text
    r = generate(words_to_text(load_words(words)), Brand.load(brand), platform, use_llm=llm)
    console.print_json(json.dumps(r))
    if out:
        out.write_text(json.dumps(r, indent=2), encoding="utf-8")


@app.command()
def thumbnail(src: Path, dst: Path, text: str = "", brand: str = "default", t: float | None = None,
              width: int = 1080, height: int = 1920):
    """Pick a sharp, well-lit frame and overlay title text."""
    from .ai.thumbnail import make_thumbnail
    console.print(make_thumbnail(src, dst, text, Brand.load(brand), t, width, height))


@app.command()
def track(src: Path, out: Path, every: float = 0.5):
    """Detect and track faces; write a smoothed crop trajectory JSON (for reframe --layout track)."""
    from .tracking.smart_crop import trajectory
    traj = trajectory(src, every)
    out.write_text(json.dumps(traj), encoding="utf-8")
    console.print(f"{len(traj)} samples -> {out}")


@app.command()
def analyze(clip: Path, every: float = 0.2, force: bool = False):
    """Track both speakers' faces and detect who is talking; caches results for `shots`."""
    from .pipeline.shots import analysis_for
    tr, turns = analysis_for(clip, every, force)
    found = {s: sum(1 for b in tr.boxes[s] if b) for s in ("L", "R")}
    console.print(f"samples: {len(tr.times)}  faces found L/R: {found['L']}/{found['R']}")
    t = Table(title="Speaker turns")
    for c in ("start", "end", "seat"):
        t.add_column(c)
    for tr_ in turns:
        t.add_row(f"{tr_['start']:.1f}", f"{tr_['end']:.1f}", tr_["seat"])
    console.print(t)


@app.command()
def shots(clip: Path, dst: Path, words: Path | None = None, plan: Path | None = None,
          shot: str = typer.Option("speaker", help="speaker | both | speaker_image | stacked | image | landscape | sidebyside | auto"),
          start: float = 0.0, end: float | None = None, question: str = "", image: Path | None = None,
          brand: str = "bridges_ai", upscale: str = typer.Option("fast", help="none | fast | clean"),
          keep: str | None = typer.Option(None, help="Kept ranges 'a-b,c-d' in seconds; shots change at each cut"),
          logo: bool = typer.Option(True, "--logo/--no-logo"), no_captions: bool = False, gpu: bool = True):
    """Render one of the five shot layouts (or a multi-segment plan) from a graded clip."""
    from .pipeline.shots import Segment, ShotPlan, auto_plan, render
    from .pipeline.transcribe import load_words
    from .utils.ffmpeg import probe
    b = Brand.load(brand)
    if plan:
        p = ShotPlan.load(plan)
    elif keep:
        ranges = [tuple(float(x) for x in r.split("-")) for r in keep.split(",")]
        p = auto_plan(clip, ranges, question, str(image) if image else None, prefer=shot)
    else:
        e = end if end is not None else probe(clip).duration
        p = ShotPlan([Segment(start, e, shot, "auto", str(image) if image else None)], question)
    p.upscale, p.logo, p.captions = upscale, logo, not no_captions
    ws = load_words(words) if words else None
    if shot in ("landscape", "sidebyside"):
        from .pipeline.shots import render_landscape
        out = render_landscape(clip, p, dst, ws, b, mode=shot, gpu=gpu)
    else:
        out = render(clip, p, dst, ws, b, gpu=gpu)
    console.print(f"[green]done[/green] {out}  ({p.duration:.1f}s, {len(p.segments)} segment(s))")


@app.command()
def cuts(clip: Path, words: Path | None = None, start: float = 0.0, end: float | None = None,
         method: str = typer.Option("auto-editor", help="auto-editor | words"), gap: float = 0.7,
         drop: list[str] | None = typer.Option(None, help="Phrases to remove (repeatable)")):
    """Print kept ranges (dead air removed) in the format `shots --keep` accepts."""
    from .pipeline import cuts as cm
    from .pipeline.transcribe import load_words
    ws = load_words(words) if words else None
    r = cm.auto_editor(clip, start=start, end=end) if method == "auto-editor" else cm.from_words(ws or [], gap, start, end)
    if drop and ws:
        r = cm.drop_phrases(ws, r, drop)
    console.print(f"{len(r)} ranges, {cm.total(r):.1f}s kept")
    console.print(cm.fmt(r))


@app.command("transcribe-project")
def transcribe_project_cmd(project: str, sessions: bool = typer.Option(False, help="Full session encodes instead of clips"),
                           model: str | None = None, force: bool = False):
    """Word-level transcripts for every clip (or session) of a project, plus a combined Markdown transcript."""
    from .pipeline.transcribe import transcribe_project
    out = transcribe_project(Project.load(project), sessions, model, force)
    console.print(f"[green]{len(out) - 1} transcripts[/green]; combined -> {out['combined']}")


@app.command()
def windows(project: str, ids: list[str] | None = typer.Argument(None),
            variant: list[str] | None = typer.Option(None, "--variant", "-v", help="speaker | stacked | both | landscape | sidebyside | speaker_image | image"),
            force: bool = False):
    """Render the project's transcript-anchored short windows (projects/<name>/windows.yaml) in each layout."""
    from .pipeline.windows import DEFAULT_VARIANTS, render_windows
    p = Project.load(project)
    m = render_windows(p, variant or DEFAULT_VARIANTS, ids or None, force=force)
    console.print(f"[green]{len(m)} windows[/green] -> {p.root / 'edit' / 'shorts'}")


@app.command()
def diagrams(project: str):
    """Render the Mermaid diagrams in windows.yaml to brand-coloured reference images (edit/refs/)."""
    from .pipeline.windows import make_diagrams
    for pth in make_diagrams(Project.load(project)):
        console.print(pth)


@app.command()
def outro(project: str, episode: str | None = None, cta: str = "Follow for more", url: str = "", duration: float = 5.0):
    """Build the branded portrait + landscape outro bumpers (edit/bumpers/)."""
    from .pipeline.bumper import make_outros
    p = Project.load(project)
    out = make_outros(p.brand, p.root / "edit" / "bumpers", episode if episode is not None else p.episode, cta, url, duration)
    for k, v in out.items():
        console.print(f"{k}: {v}")


@app.command("append-outros")
def append_outros_cmd(project: str, src: str = "edit/shorts", dst: str = "edit/final", force: bool = False):
    """Append the orientation-matched outro to every clip in src, writing to dst."""
    from .pipeline.bumper import append_outros
    p = Project.load(project)
    done = append_outros(p.root / src, p.root / dst, p.root / "edit" / "bumpers", force)
    console.print(f"[green]{len(done)} clips[/green] -> {p.root / dst}")


@app.command()
def hq(project: str, ids: list[str] | None = typer.Argument(None), model: str = "seedvr2_ema_3b_fp16.safetensors",
       batch_size: int = 5, force: bool = False):
    """High-quality speaker shorts: plain render -> SeedVR2 restore -> overlays -> outro (slow, GPU)."""
    from .pipeline.hq import hq_speaker
    outs = hq_speaker(Project.load(project), ids or None, model=model, batch_size=batch_size, force=force)
    console.print(f"[green]{len(outs)} HQ clips[/green]")


@app.command()
def presets():
    """List available output presets."""
    from .utils.formats import list_presets, load_preset
    t = Table()
    for c in ("name", "size", "max", "layout", "captions"):
        t.add_column(c)
    for n in list_presets():
        p = load_preset(n)
        t.add_row(n, f"{p.width}x{p.height}", f"{p.max_duration or '-'}", p.layout, p.caption_style)
    console.print(t)


@app.command()
def doctor():
    """Check ffmpeg, NVENC, CUDA for whisper, fonts."""
    from .pipeline.captions import load_font
    from .pipeline.transcribe import _enable_cuda_dlls
    from .utils import ffmpeg as f
    ok = lambda b: "[green]ok[/green]" if b else "[red]missing[/red]"
    try:
        f.which("ffmpeg"); f.which("ffprobe"); ff = True
    except f.FFmpegError:
        ff = False
    console.print(f"ffmpeg/ffprobe: {ok(ff)}")
    console.print(f"h264_nvenc: {ok(ff and f.has_nvenc())}")
    _enable_cuda_dlls()
    try:
        import ctranslate2
        console.print(f"whisper CUDA devices: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:
        console.print(f"ctranslate2: [red]{e}[/red]")
    console.print(f"bold font: {load_font('arialbd.ttf', 20).path if hasattr(load_font('arialbd.ttf', 20), 'path') else 'fallback'}")
    for mod in ("mediapipe", "cv2", "pyannote.audio", "anthropic", "keybert"):
        try:
            __import__(mod)
            console.print(f"{mod}: {ok(True)}")
        except ImportError:
            console.print(f"{mod}: [yellow]optional, not installed[/yellow]")


if __name__ == "__main__":
    app()
