"""Thin wrapper: word-level transcripts for a project's studio clips.  Usage: python scripts/transcribe_clips.py [project]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project
from contentforge.pipeline.transcribe import transcribe_project

if __name__ == "__main__":
    out = transcribe_project(Project.load(sys.argv[1] if len(sys.argv) > 1 else "bridges_ai_3rdi"), sessions=False)
    print("\n".join(f"{k}: {v}" for k, v in out.items()))
    print("TRANSCRIBE_DONE")
