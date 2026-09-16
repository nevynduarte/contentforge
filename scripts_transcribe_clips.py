import sys, glob, os, time
sys.path.insert(0, r"C:\Users\User\Documents\contentforge")
from contentforge.pipeline.transcribe import transcribe
root = r"C:\Users\User\Documents\3rdiPodcast"
nums = sys.argv[1:] or [f"{i:02d}" for i in range(2, 11)]
for n in nums:
    files = glob.glob(os.path.join(root, "edit", "studio", f"clip_{n}_*_studio.mp4"))
    if not files:
        print("no studio file for", n); continue
    t0 = time.time()
    out = transcribe(files[0], os.path.join(root, "captions", f"clip_{n}_words.json"),
                     os.path.join(root, "captions", f"clip_{n}.srt"), model="large-v3")
    print(f"clip_{n}: {len(out['words'])} words, {len(out['segments'])} segments in {time.time()-t0:.0f}s", flush=True)
print("TRANSCRIBE_DONE")
