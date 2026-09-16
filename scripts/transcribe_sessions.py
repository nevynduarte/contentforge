"""Thin wrapper: transcribe a project's full session encodes and build captions/full_transcript.md.
Usage: python scripts/transcribe_sessions.py [project]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project
from contentforge.pipeline.transcribe import transcribe_project

if __name__ == "__main__":
    out = transcribe_project(Project.load(sys.argv[1] if len(sys.argv) > 1 else "bridges_ai_3rdi"), sessions=True)
    print("\n".join(f"{k}: {v}" for k, v in out.items()))
    print("SESSIONS_TRANSCRIBE_DONE")
