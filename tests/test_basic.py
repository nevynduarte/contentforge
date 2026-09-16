from PIL import Image

from contentforge.config import Brand, CaptionStyle, Project
from contentforge.pipeline.audio import PODCAST_VOICE_V2
from contentforge.pipeline.captions import CaptionRenderer, chunk_words
from contentforge.pipeline.grade import STUDIO_V2
from contentforge.utils.colors import contrast_ratio, hex_to_rgb
from contentforge.utils.formats import list_presets, load_preset

WORDS = [{"word": " these", "start": 0.0, "end": 0.16}, {"word": " founders", "start": 0.16, "end": 0.46},
         {"word": " and", "start": 0.46, "end": 0.64}, {"word": " some", "start": 0.64, "end": 0.74},
         {"word": " of", "start": 0.74, "end": 0.82}, {"word": " these", "start": 0.82, "end": 0.92},
         {"word": " people.", "start": 3.0, "end": 3.3}]


def test_filter_chains_preserved():
    assert "gamma=1.35" in STUDIO_V2 and "unsharp=5:5:0.7:5:5:0.0" in STUDIO_V2
    assert PODCAST_VOICE_V2.endswith("loudnorm=I=-16:LRA=11:TP=-1.5")


def test_chunking_breaks_on_pause():
    chunks = chunk_words(WORDS, max_words=5, max_chars=35)
    assert [len(c) for c in chunks] == [5, 1, 1]


def test_renderer_states_and_overlay():
    r = CaptionRenderer(WORDS, CaptionStyle(), 1080, 1920)
    assert r.state_at(0.2) == (0, 1)
    assert r.state_at(10.0) is None
    frame = Image.new("RGB", (1080, 1920), (17, 17, 17))
    out = r.composite(frame, 0.2)
    assert out.size == (1080, 1920)
    assert len(r._cache) == 1


def test_presets_and_brands_load():
    assert {"instagram_reel", "youtube_short", "tiktok", "instagram_feed", "youtube_landscape", "linkedin"} <= set(list_presets())
    p = load_preset("instagram_reel")
    assert (p.width, p.height, p.max_duration) == (1080, 1920, 90)
    b = Brand.load("bridges_ai")
    assert "#BridgesAI" in b.hashtags
    assert contrast_ratio(hex_to_rgb(b.colors["text"]), hex_to_rgb(b.colors["primary"])) > 7


def test_project_loads_clips():
    p = Project.load("bridges_ai_3rdi")
    assert len(p.clips) == 10 and p.clips[1].start == 72
