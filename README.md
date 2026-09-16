# ContentForge

Turn raw podcast / interview / talk footage into polished, captioned, branded
short-form clips for Instagram Reels, YouTube Shorts, TikTok, Instagram Feed,
LinkedIn and YouTube, from one command.

Built for the **Bridges AI x 3rd i Podcast** ("AI in Safety Tech") and designed
to be reused for any future recording.

```
raw camera files ──▶ ingest ──▶ grade + voice cleanup ──▶ transcribe (word timestamps)
        ──▶ [diarize] ──▶ [AI clip suggestion] ──▶ reframe ──▶ karaoke captions + brand ──▶ per-platform renders + manifest
```

## What works today

| Stage | Module | Status |
|---|---|---|
| Probe media, flag broken (moov-less) MP4s | `pipeline/ingest.py` | ✅ |
| Studio grade v2 + podcast voice chain (exact filters preserved) | `pipeline/grade.py`, `pipeline/audio.py` | ✅ |
| Word-level transcription, faster-whisper on CUDA | `pipeline/transcribe.py` | ✅ |
| Karaoke / popup / bar captions rendered with Pillow (no libass) | `pipeline/captions.py` | ✅ |
| Letterbox / crop / blur-fit / tracked-crop reframing | `pipeline/reframe.py` | ✅ |
| Logo, lower third, progress bar, title card, intro/outro concat | `pipeline/brand.py` | ✅ |
| Platform presets + manifest | `pipeline/render.py`, `presets/` | ✅ |
| Batch orchestrator per project | `pipeline/batch.py` | ✅ |
| Heuristic clip scoring + LLM refinement (Claude / Ollama) | `ai/content_scorer.py`, `pipeline/clip.py` | ✅ heuristic, LLM optional |
| Keywords, quotes, social copy, thumbnails | `ai/` | ✅ heuristic, LLM optional |
| Face detection / tracking / active speaker / smart crop | `tracking/` | ⚠️ implemented, needs `[tracking]` extra and real-footage tuning |
| Speaker diarization (pyannote) | `pipeline/diarize.py` | ⚠️ needs `[diarize]` extra + HF token; heuristic fallback |
| Audio sync of external mic recording | `pipeline/audio.py::find_offset` | ✅ |

## Install (Windows, Python 3.12)

```powershell
git clone https://github.com/nevynduarte/contentforge
cd contentforge
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -e ".[cuda]"          # add ,tracking,nlp,diarize as needed
contentforge doctor                  # checks ffmpeg, NVENC, CUDA for whisper, fonts
```

Requirements: FFmpeg (full build) on PATH. An NVIDIA GPU is optional but makes
transcription ~20x faster and enables `--gpu` (NVENC) encoding.

## Quick start

```bash
contentforge probe C:/footage                                   # inventory + broken-file check
contentforge grade MVI_8429.MP4 clip.mp4 --start 72 --duration 124
contentforge transcribe clip.mp4 --out-dir captions             # -> clip_words.json + clip.srt
contentforge social clip.mp4 captions/clip_words.json out.mp4 --preset instagram_reel --brand bridges_ai
contentforge batch bridges_ai_3rdi --preset instagram_reel --preset youtube_short
contentforge suggest captions/session_words.json --n 8          # best-moment candidates
contentforge copy captions/clip_02_words.json --brand bridges_ai --platform instagram
contentforge thumbnail clip.mp4 thumb.jpg --text "Sold my house in 8th grade" --brand bridges_ai
```

## Project layout

```
contentforge/         package (pipeline/, ai/, tracking/, utils/, cli.py, config.py)
presets/              platform output presets (size, duration cap, layout, caption style)
templates/<brand>/    config.yaml, logo.png, lower_third.png, fonts/, intro.mp4, outro.mp4
projects/<name>/      project.yaml (sources, output dirs, presets) + clips/*.yaml
```

A project points `root:` at the folder holding the footage; nothing under
`projects/` needs to contain media.

## The social clip look

1080x1920 canvas, full 16:9 frame scaled to 1080x607 and placed at y=80 on
`#111111`; captions at y=780: 52 pt bold, white with the active word in
`#FFDC32`, 2 px black stroke, on a rounded semi-transparent black pill.
Frames are decoded by ffmpeg to raw RGB, composited in Pillow (caption strips
are pre-rendered and cached per phrase/word state), and piped back to ffmpeg.

## Studio grade v2 (preserved exactly)

See `pipeline/grade.py::STUDIO_V2` and `pipeline/audio.py::PODCAST_VOICE_V2`.
Encoding: libx264 medium CRF 18, AAC 192k, `+faststart` (or `--gpu` for NVENC
at an equivalent `-cq`).

## Roadmap

- [ ] Tune face tracking + active-speaker crop on real two-person footage
- [ ] LLM-driven auto-clipping end-to-end (`suggest` -> clip defs -> batch)
- [ ] Animated intro/outro generator from brand config
- [ ] Emoji / reaction overlays at key moments
- [ ] Demucs voice isolation stage
- [ ] Multi-format render in a single decode pass

## License

MIT
