from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = APP_ROOT.parent
DEFAULT_MATERIAL_ROOT = WORKSPACE_ROOT / "test-materials" / "music"
DEFAULT_WORK_ROOT = WORKSPACE_ROOT / ".work" / "music-v4"
DEFAULT_FFMPEG = WORKSPACE_ROOT / ".work" / "ffmpeg-7.1.1-extract" / "ffmpeg-7.1.1-essentials_build" / "bin" / "ffmpeg.exe"
DEFAULT_BASELINE = APP_ROOT / "tests" / "fixtures" / "music_visual_baseline_v4.npz"
MANIFEST_PATH = APP_ROOT / "provenance" / "music-dev-tools.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing development tool manifest: {MANIFEST_PATH}")
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("music-dev-tools.json must contain an object")
    return value


def verify_ffmpeg(path: Path) -> None:
    manifest = load_manifest()
    tool = manifest.get("ffmpeg", {})
    if tool.get("version") != "7.1.1":
        raise RuntimeError("The development manifest must pin FFmpeg 7.1.1")
    expected = str(tool.get("executable_sha256", "")).upper()
    if not expected:
        raise RuntimeError("FFmpeg SHA256 is not recorded; fill music-dev-tools.json before extraction")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"FFmpeg SHA256 mismatch: expected {expected}, got {actual}")


def extract_frames(ffmpeg: Path, material_root: Path, work_root: Path, fps: float) -> list[Path]:
    work_root.mkdir(parents=True, exist_ok=True)
    frame_roots: list[Path] = []
    for source in sorted(material_root.glob("*.mp4")):
        target = work_root / source.stem
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                f"fps={fps},scale=1280:720:flags=area",
                "-vsync",
                "vfr",
                str(target / "frame-%06d.ppm"),
            ],
            check=True,
        )
        frame_roots.append(target)
    return frame_roots


def build_fixture(frame_roots: list[Path], output: Path) -> None:
    import numpy as np

    from agent.music.vision import bgr_to_hsv

    def read_ppm(path: Path) -> Any:
        with path.open("rb") as stream:
            if stream.readline().strip() != b"P6":
                raise ValueError(f"Unsupported PPM header: {path}")
            dimensions = stream.readline().strip()
            while dimensions.startswith(b"#"):
                dimensions = stream.readline().strip()
            width, height = (int(value) for value in dimensions.split())
            if int(stream.readline().strip()) != 255:
                raise ValueError(f"Unsupported PPM color depth: {path}")
            pixels = np.frombuffer(stream.read(), dtype=np.uint8)
        expected = width * height * 3
        if pixels.size != expected:
            raise ValueError(f"Truncated PPM frame: {path}")
        return pixels.reshape((height, width, 3))
    hue_histogram = np.zeros(180, dtype=np.int64)
    saturation_histogram = np.zeros(256, dtype=np.int64)
    value_histogram = np.zeros(256, dtype=np.int64)
    source_hashes: list[str] = []
    sampled_frames = 0
    for root in frame_roots:
        for path in sorted(root.glob("*.ppm")):
            rgb = read_ppm(path)
            bgr = rgb[..., ::-1]
            hsv = bgr_to_hsv(bgr)
            chromatic = (hsv[..., 1] >= 35) & (hsv[..., 2] >= 100)
            hue_histogram += np.bincount(hsv[..., 0][chromatic], minlength=180)
            saturation_histogram += np.bincount(hsv[..., 1][chromatic], minlength=256)
            value_histogram += np.bincount(hsv[..., 2][chromatic], minlength=256)
            source_hashes.append(sha256(path))
            sampled_frames += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    # The fixture deliberately excludes frame time, frame order, song name and lane actions.
    np.savez_compressed(
        output,
        schema_version=np.asarray([1], dtype=np.uint8),
        baseline_version=np.asarray(["maes-music-v4-2026-07"]),
        hue_histogram=hue_histogram,
        saturation_histogram=saturation_histogram,
        value_histogram=value_histogram,
        sampled_frame_count=np.asarray([sampled_frames], dtype=np.int32),
        unordered_frame_hashes=np.asarray(sorted(source_hashes)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a song-agnostic MAES V4 visual baseline")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIAL_ROOT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"FFmpeg 7.1.1 essentials was not found at {args.ffmpeg}")
    verify_ffmpeg(args.ffmpeg)
    roots = extract_frames(args.ffmpeg, args.materials, args.work, args.fps)
    sys.path.insert(0, str(APP_ROOT))
    build_fixture(roots, args.output)
    print(f"Wrote song-agnostic V4 baseline: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
