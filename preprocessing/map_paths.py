"""Normalize an existing expansion-folder layout without modifying raw data."""
import hashlib
from pathlib import Path

from .common import PROJECT_ROOT


def map_root_view(root):
    root = Path(root).resolve()
    if (root/"maps/expansion").is_dir():
        return root
    if not (root/"expansion").is_dir():
        raise FileNotFoundError(f"No nuScenes expansion maps under {root}")
    key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    view = PROJECT_ROOT/"outputs/stage2/map_views/cache"/key
    (view/"maps").mkdir(parents=True, exist_ok=True)
    link = view/"maps/expansion"
    if not link.exists():
        link.symlink_to(root/"expansion", target_is_directory=True)
    for source in (root/"maps").glob("*.png"):
        target = view/"maps"/source.name
        if not target.exists():
            target.symlink_to(source)
    return view
