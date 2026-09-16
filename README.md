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

## The five shots

```bash
contentforge analyze clip.mp4                       # face tracks + who-is-talking, cached
contentforge cuts clip.mp4 --words clip_words.json  # kept ranges with dead air removed
contentforge shots clip.mp4 out.mp4 --words clip_words.json --shot speaker \
    --question "How did you fund the first version?" --keep "0-12,20-32,60-72"
```

| Shot | `--shot` | Size | What it does |
|---|---|---|---|
| S1 speaker only | `speaker` | 1080x1920 | Tight portrait crop following the active speaker; Real-ESRGAN upscale of the crop |
| S2 both | `both` | 1080x1920 | Full 16:9 frame letterboxed |
| S3 speaker + image | `speaker_image --image ref.png` | 1080x1920 | Reference image on top, speaker crop below |
| S4 stacked | `stacked` | 1080x1920 | Head-and-torso of both seats, one above the other |
| S5 image only | `image --image ref.png` | 1080x1920 | Reference image over the interview audio |
| Landscape | `landscape` | 1920x1080 | Full frame, banner top-left, lockup bottom-right |
| Side by side | `sidebyside` | 1920x1080 | Each seat's head-and-torso fills half the frame |

The question is drawn as a banner at the top, so the clip can skip the host asking it.
With `--keep`, the shot changes at every join where a fragment was removed, so cuts
read as edits rather than glitches. Speaker detection is lip-motion energy from
InsightFace mouth keypoints gated by audio, with pyannote diarization fused in when
`HF_TOKEN` is set.

### Windows, bumpers, and the HQ tier

A project can define transcript-anchored moments in `projects/<name>/windows.yaml`
(id, clip, question, first/last words, optional reference image) plus the Mermaid
sources for those images. Then:

```bash
contentforge diagrams bridges_ai_3rdi                  # Mermaid -> edit/refs/*.png in brand colours
contentforge windows bridges_ai_3rdi                   # every window in speaker/stacked/both/landscape/sidebyside
contentforge windows bridges_ai_3rdi -v speaker_image -v image   # the image layouts (windows with an image)
contentforge outro bridges_ai_3rdi --url bridgesai.consulting    # portrait + landscape outro bumpers
contentforge append-outros bridges_ai_3rdi             # edit/shorts -> edit/final with a crossfaded outro
contentforge hq bridges_ai_3rdi zillow                 # SeedVR2-restored speaker short (slow, best quality)
```

The HQ tier renders the speaker crop without overlays, restores it with SeedVR2 3B
(Apache-2.0, runs on a 24 GB card) through the numz CLI checkout, then overlays banner,
captions and lockup and appends the outro. Expect minutes per clip.

Heavy assets (torch, model weights, SeedVR2 checkout, caches) live under
`D:\contentforge-cache` on the P620; override with `CONTENTFORGE_MODELS`,
`CONTENTFORGE_WORK`, `CONTENTFORGE_SEEDVR2` and `CONTENTFORGE_MMDC`.

### Brand template

`templates/bridges_ai/config.yaml` carries the design-system tokens (paper, navy ink,
bronze accent), the fonts (Inter for UI and captions, Source Serif 4 for the question),
and the official lockup. Question banners use the serif with an Inter eyebrow; captions
are Inter Bold with a warm-gold active word on a squared ink pill. The lockup sits
bottom-right on landscape, bottom-left over full-bleed portrait video, and centred in
the empty band on letterboxed portrait layouts.

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
