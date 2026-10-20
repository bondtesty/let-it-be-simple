"""Direct-script import helpers.

Scripts run as `python scripts/name.py` start with `scripts/` on sys.path, not
the repository root. Add the root before importing `scripts.lib` or `openpi`.
"""

from __future__ import annotations

from pathlib import Path
import sys


def add_repo_root(file: str | Path, *, include_src: bool = False) -> Path:
    root = Path(file).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if include_src:
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    return root
