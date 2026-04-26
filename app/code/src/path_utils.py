from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_path() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[3],  # repo root when app/code/src/path_utils.py lives inside this repo
        current.parents[2],  # /app when app/ contents are copied into the container root
    ]
    for candidate in candidates:
        if (candidate / "src").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("Could not locate project root containing src/.")
