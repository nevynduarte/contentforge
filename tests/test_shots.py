import pytest
from PIL import Image

from contentforge.config import Brand, Project
from contentforge.pipeline import cuts
from contentforge.pipeline.shots import Banner, Segment, ShotPlan, _portrait_crop
from contentforge.pipeline.windows import WindowSet, find_phrase, resolve

WORDS = [
    {"word": " Um,", "start": 0.0, "end": 0.2}, {"word": " sold", "start": 0.3, "end": 0.6}, {"word": " my", "start": 0.6, "end": 0.7},
    {"word": " house", "start": 0.7, "end": 1.0}, {"word": " on", "start": 1.0, "end": 1.1}, {"word": " Zillow", "start": 1.1, "end": 1.5},
    {"word": " when", "start": 2.6, "end": 2.8}, {"word": " I", "start": 2.8, "end": 2.9}, {"word": " was", "start": 2.9, "end": 3.1},
    {"word": " 9", "start": 3.2, "end": 3.3}, {"word": "-1", "start": 3.3, "end": 3.4}, {"word": "-1.", "start": 3.4, "end": 3.6},
]


def test_find_phrase_is_punctuation_insensitive():
    assert find_phrase(WORDS, "sold my house on zillow") == (1, 5)
    assert find_phrase(WORDS, "Um sold") == (0, 1)
    with pytest.raises(ValueError):
        find_phrase(WORDS, "integration with 9-1-1")


def test_resolve_window_bounds():
    s, e = resolve(WORDS, "sold my house", "when i was")
    assert s == pytest.approx(0.05) and e == pytest.approx(3.1 + 0.45)


def test_cuts_from_words_and_phrase_drop():
    r = cuts.from_words(WORDS, gap=0.7)
    assert len(r) == 2 and r[0][0] == 0.0
    r2 = cuts.drop_phrases(WORDS, r, ["on zillow"])
    assert cuts.total(r2) < cuts.total(r)
    assert cuts.merge([(0, 1), (1.05, 2), (3, 4)]) == [(0, 2), (3, 4)]


def test_portrait_crop_stays_inside_source():
    x, y, w, h = _portrait_crop((1700, 400, 150, 180), 1080 / 1920, 1920, 1080)
    assert x >= 0 and x + w <= 1920 and y >= 0 and y + h <= 1080
    assert abs(w / h - 1080 / 1920) < 0.02


def test_shot_plan_roundtrip(tmp_path):
    p = ShotPlan([Segment(1.0, 5.0, "speaker", "L"), Segment(9.0, 12.0, "image", "auto", "ref.png")], "Why?", logo=False)
    f = p.save(tmp_path / "plan.json")
    q = ShotPlan.load(f)
    assert q.duration == 7.0 and q.segments[1].image == "ref.png" and q.logo is False


def test_banner_draws_question_and_logo():
    b = Brand.load("bridges_ai")
    banner = Banner("How do you bring AI into a 40-year-old business?", b, logo=True)
    assert banner.img is not None and banner.img.width == 1000
    frame = Image.new("RGB", (1080, 1920), (10, 10, 10))
    banner.draw(frame, "speaker")
    banner.draw(frame, "stacked")
    banner.draw(frame, "landscape")
    assert frame.getpixel((60, 100)) != (10, 10, 10)


def test_project_windows_load():
    ws = WindowSet.load(Project.load("bridges_ai_3rdi"))
    assert len(ws.windows) == 11
    assert {w.id for w in ws.windows if w.image} == set(ws.diagrams)
