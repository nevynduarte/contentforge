"""Transcribe full session encodes for a project -> captions/<session>_words.json + .srt + combined transcript."""
import sys, time, json, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contentforge.pipeline.transcribe import transcribe
root = Path(r"C:\Users\User\Documents\3rdiPodcast")
files = sorted(glob.glob(str(root / "edit/studio/sessions/session*_studio.mp4")))
for f in files:
    stem = Path(f).stem.replace("_studio", "")
    t0 = time.time()
    r = transcribe(f, root / "captions" / f"{stem}_words.json", root / "captions" / f"{stem}.srt", model="large-v3")
    print(f"{stem}: {len(r['words'])} words in {time.time()-t0:.0f}s", flush=True)
out = ["# Bridges AI x 3rd i Podcast - AI in Safety Tech - full transcript\n"]
for f in sorted(glob.glob(str(root / "captions/session*_words.json"))):
    words = json.load(open(f, encoding="utf-8"))
    paras, cur, last_end = [], [], 0.0
    for w in words:
        if cur and (w["start"] - last_end > 1.5 or len(cur) > 120) and cur[-1]["word"].strip().endswith((".", "?", "!")):
            paras.append(cur); cur = []
        cur.append(w); last_end = w["end"]
    if cur: paras.append(cur)
    out.append(f"\n## {Path(f).stem.replace('_words','')}\n")
    for p in paras:
        m, s = divmod(int(p[0]["start"]), 60)
        out.append(f"\n[{m:02d}:{s:02d}] " + re.sub(r"\s+", " ", "".join(w["word"] for w in p).strip()))
(root / "captions" / "full_transcript.md").write_text("\n".join(out), encoding="utf-8")
print("SESSIONS_TRANSCRIBE_DONE", flush=True)
