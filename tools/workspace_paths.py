"""Resolve developer tools from source, snapshots or packaged checkout paths."""
from pathlib import Path
import shutil

APP_ROOT = Path(__file__).resolve().parents[1]


def workspace_root():
    for root in (APP_ROOT, *APP_ROOT.parents):
        if (root / 'test-materials').is_dir() and (root / '.work').is_dir():
            return root
    return APP_ROOT.parent


def ffmpeg_binary(name):
    root = workspace_root()
    path = root / '.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin' / name
    return path if path.is_file() else Path(shutil.which(name) or name)
