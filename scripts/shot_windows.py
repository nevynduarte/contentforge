"""Thin wrapper over `contentforge windows`: render a project's transcript-anchored windows.

Usage: python scripts/shot_windows.py [variants...] [--only ID] [--project NAME] [--force]
variants default: speaker stacked both landscape sidebyside
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contentforge.config import Project
from contentforge.pipeline.windows import DEFAULT_VARIANTS, render_windows


def main() -> None:
    args = sys.argv[1:]
    only = args[args.index("--only") + 1] if "--only" in args else None
    project = args[args.index("--project") + 1] if "--project" in args else "bridges_ai_3rdi"
    variants = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--only", "--project"))]
    render_windows(Project.load(project), variants or DEFAULT_VARIANTS, [only] if only else None, force="--force" in args)
    print("SHORTS_DONE", flush=True)


if __name__ == "__main__":
    main()
