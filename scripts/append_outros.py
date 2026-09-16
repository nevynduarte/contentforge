"""Thin wrapper over `contentforge append-outros`: edit/shorts/*.mp4 -> edit/final/*.mp4 with the matching outro.
Usage: python scripts/append_outros.py [project]   (FORCE=1 re-renders existing outputs)"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project
from contentforge.pipeline.bumper import append_outros

if __name__ == "__main__":
    p = Project.load(sys.argv[1] if len(sys.argv) > 1 else "bridges_ai_3rdi")
    done = append_outros(p.root / "edit" / "shorts", p.root / "edit" / "final", p.root / "edit" / "bumpers", os.environ.get("FORCE") == "1")
    print(f"{len(done)} clips -> {p.root / 'edit' / 'final'}")
    print("OUTROS_DONE", flush=True)
